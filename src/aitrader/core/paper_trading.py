from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date
from math import isfinite

from aitrader.ai.gate import AITradingGate
from aitrader.ai.models import AIAnalysisResult
from aitrader.execution.engine import OrderEngine
from aitrader.execution.models import (
    OrderRecord,
    OrderRequest,
    OrderStatus,
)
from aitrader.notifications import TelegramNotifier
from aitrader.portfolio.manager import PositionManager
from aitrader.portfolio.models import Position, PositionStatus
from aitrader.risk.engine import (
    MarketType,
    OrderSide,
    RiskDecision,
)
from aitrader.risk.manager import AccountRiskState
from aitrader.strategy.models import MarketBias, StrategySignal
from aitrader.strategy.multi_timeframe import MultiTimeframeResult


@dataclass(frozen=True, slots=True)
class PaperTradingResult:
    decision: RiskDecision
    order: OrderRecord | None
    position: Position | None
    notification_errors: tuple[str, ...] = ()


class PaperTradingService:
    """
    Coordinates AI approval, paper execution, portfolio updates,
    and optional Telegram notifications.

    Telegram failures never roll back an already completed
    paper execution or portfolio update.
    """

    def __init__(
        self,
        ai_gate: AITradingGate,
        order_engine: OrderEngine,
        position_manager: PositionManager,
        notifier: TelegramNotifier | None = None,
    ):
        self.ai_gate = ai_gate
        self.order_engine = order_engine
        self.position_manager = position_manager
        self.notifier = notifier

    def _notify(
        self,
        errors: list[str],
        method_name: str,
        *args,
    ):
        if self.notifier is None:
            return

        try:
            method = getattr(self.notifier, method_name)
            method(*args)
        except Exception as exc:
            # Do not expose tokens or sensitive request details.
            errors.append(
                f"{method_name}: {type(exc).__name__}"
            )

    def enter(
        self,
        *,
        analysis: AIAnalysisResult,
        consensus: MultiTimeframeResult,
        execution_signal: StrategySignal,
        entry_price: float,
        account: AccountRiskState,
        market_type: MarketType | str,
        client_order_id: str | None = None,
    ) -> PaperTradingResult:

        errors = []

        decision = self.ai_gate.evaluate(
            analysis=analysis,
            consensus=consensus,
            execution_signal=execution_signal,
            entry_price=entry_price,
            account=account,
            market_type=market_type,
        )

        # Prefer the actual execution signal for order direction.
        bias = execution_signal.bias

        if bias not in (MarketBias.BULLISH, MarketBias.BEARISH):
            bias = consensus.bias

        if bias not in (MarketBias.BULLISH, MarketBias.BEARISH):
            bias = analysis.bias

        # No directional signal means no order object is created.
        if bias not in (MarketBias.BULLISH, MarketBias.BEARISH):
            self._notify(
                errors,
                "send_message",
                (
                    "⛔ AITrader 신규 진입 거부\n"
                    f"종목: {analysis.symbol}\n"
                    f"사유: {decision.rejection_reason or 'No directional signal'}"
                ),
            )

            return PaperTradingResult(
                decision=decision,
                order=None,
                position=None,
                notification_errors=tuple(errors),
            )

        side = (
            OrderSide.BUY
            if bias == MarketBias.BULLISH
            else OrderSide.SELL
        )

        request = OrderRequest(
            exchange=execution_signal.exchange,
            symbol=execution_signal.symbol,
            side=side,
            reference_price=entry_price,
            client_order_id=client_order_id,
            reduce_only=False,
        )

        order = self.order_engine.submit_paper_order(
            request,
            decision,
        )

        self._notify(
            errors,
            "send_order_notification",
            order,
        )

        if order.status != OrderStatus.FILLED:
            return PaperTradingResult(
                decision=decision,
                order=order,
                position=None,
                notification_errors=tuple(errors),
            )

        try:
            position = self.position_manager.open_from_order(
                order,
                market_type,
                leverage=account.leverage,
            )
        except Exception as exc:
            self._notify(
                errors,
                "send_system_error",
                "PaperTradingService",
                f"Order filled but position creation failed: {type(exc).__name__}",
            )
            raise

        self._notify(
            errors,
            "send_position_opened",
            position,
        )

        return PaperTradingResult(
            decision=decision,
            order=order,
            position=position,
            notification_errors=tuple(errors),
        )

    def close_position(
        self,
        position_id: str,
        mark_price: float,
        quantity: float | None = None,
        client_order_id: str | None = None,
    ) -> PaperTradingResult:

        errors = []

        position = self.position_manager.get_position(position_id)

        if position is None:
            raise ValueError("Position not found")

        if position.status != PositionStatus.OPEN:
            raise ValueError("Position is already closed")

        if (
            isinstance(mark_price, bool)
            or not isinstance(mark_price, (int, float))
            or not isfinite(mark_price)
            or mark_price <= 0
        ):
            raise ValueError("Invalid exit price")

        if quantity is None:
            quantity = position.remaining_quantity

        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, (int, float))
            or not isfinite(quantity)
            or quantity <= 0
        ):
            raise ValueError("Invalid exit quantity")

        if quantity > position.remaining_quantity + 1e-12:
            raise ValueError("Exit quantity exceeds remaining position")

        side = (
            OrderSide.SELL
            if position.side.value == "long"
            else OrderSide.BUY
        )

        request = OrderRequest(
            exchange=position.exchange,
            symbol=position.symbol,
            side=side,
            reference_price=mark_price,
            client_order_id=client_order_id,
            reduce_only=True,
        )

        # Closing an existing position is risk-reducing.
        # PositionManager still verifies side and quantity.
        decision = RiskDecision(
            approved=True,
            quantity=quantity,
            stop_price=None,
        )

        order = self.order_engine.submit_paper_order(
            request,
            decision,
        )

        self._notify(
            errors,
            "send_order_notification",
            order,
        )

        if order.status != OrderStatus.FILLED:
            return PaperTradingResult(
                decision=decision,
                order=order,
                position=position,
                notification_errors=tuple(errors),
            )

        updated = self.position_manager.apply_exit_order(
            position_id,
            order,
        )

        if updated.status == PositionStatus.CLOSED:
            self._notify(
                errors,
                "send_position_closed",
                updated,
            )
        else:
            self._notify(
                errors,
                "send_message",
                (
                    "🔄 AITrader 부분 청산\n"
                    f"종목: {updated.symbol}\n"
                    f"잔여 수량: {updated.remaining_quantity:.8f}\n"
                    f"누적 실현손익: {updated.realized_pnl:.4f}"
                ),
            )

        return PaperTradingResult(
            decision=decision,
            order=order,
            position=updated,
            notification_errors=tuple(errors),
        )

    def send_daily_summary(
        self,
        utc_date: str,
        equity: float,
        mark_prices: Mapping[tuple[str, str], float] | None = None,
    ) -> tuple[str, ...]:

        target_date = date.fromisoformat(utc_date)

        positions = self.position_manager.list_positions()

        closed_today = [
            position
            for position in positions
            if (
                position.status == PositionStatus.CLOSED
                and position.closed_at is not None
                and position.closed_at.astimezone(UTC).date() == target_date
            )
        ]

        wins = sum(
            position.realized_pnl > 0
            for position in closed_today
        )

        losses = sum(
            position.realized_pnl < 0
            for position in closed_today
        )

        realized_pnl = sum(
            position.realized_pnl
            for position in closed_today
        )

        unrealized_pnl = 0.0
        mark_prices = mark_prices or {}

        for position in positions:
            if position.status != PositionStatus.OPEN:
                continue

            key = (position.exchange, position.symbol)

            if key not in mark_prices:
                raise ValueError(
                    f"Missing mark price for {position.exchange} {position.symbol}"
                )

            unrealized_pnl += position.unrealized_pnl(
                mark_prices[key]
            )

        errors = []

        self._notify(
            errors,
            "send_daily_summary",
            utc_date,
            len(closed_today),
            wins,
            losses,
            realized_pnl,
            equity,
            unrealized_pnl,
        )

        return tuple(errors)

    def notify_system_error(
        self,
        component: str,
        message: str,
    ) -> tuple[str, ...]:

        errors = []

        self._notify(
            errors,
            "send_system_error",
            component,
            message,
        )

        return tuple(errors)
