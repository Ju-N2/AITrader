from dataclasses import dataclass
from math import isfinite

from aitrader.risk.engine import (
    MarketType,
    OrderSide,
    RiskDecision,
    RiskEngine,
    RiskRequest,
)
from aitrader.strategy.models import MarketBias, StrategySignal
from aitrader.strategy.multi_timeframe import MultiTimeframeResult


@dataclass(frozen=True, slots=True)
class AccountRiskState:
    account_equity: float
    available_balance: float
    daily_start_equity: float

    daily_realized_pnl: float = 0.0
    current_exposure: float = 0.0
    open_positions: int = 0
    leverage: float = 1.0


class RiskManager:
    """
    Connects strategy signals and multi-timeframe consensus
    to the independent RiskEngine.

    This manager evaluates new entries only.
    It does not submit exchange orders.
    """

    def __init__(
        self,
        risk_engine: RiskEngine,
        atr_stop_multiplier: float = 3.0,
    ):
        if (
            isinstance(atr_stop_multiplier, bool)
            or not isinstance(atr_stop_multiplier, (int, float))
            or not isfinite(atr_stop_multiplier)
            or atr_stop_multiplier <= 0
        ):
            raise ValueError("Invalid ATR stop multiplier")

        self.risk_engine = risk_engine
        self.atr_stop_multiplier = float(atr_stop_multiplier)

    @staticmethod
    def _reject(reason: str) -> RiskDecision:
        return RiskDecision(
            approved=False,
            rejection_reason=reason,
        )

    def calculate_stop_price(
        self,
        bias: MarketBias,
        entry_price: float,
        atr: float,
    ) -> float:
        if bias not in (MarketBias.BULLISH, MarketBias.BEARISH):
            raise ValueError("Cannot calculate stop for neutral signal")

        if (
            isinstance(entry_price, bool)
            or not isinstance(entry_price, (int, float))
            or not isfinite(entry_price)
            or entry_price <= 0
        ):
            raise ValueError("Invalid entry price")

        if (
            isinstance(atr, bool)
            or not isinstance(atr, (int, float))
            or not isfinite(atr)
            or atr <= 0
        ):
            raise ValueError("Invalid ATR")

        stop_distance = atr * self.atr_stop_multiplier

        if bias == MarketBias.BULLISH:
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        if not isfinite(stop_price) or stop_price <= 0:
            raise ValueError("Calculated stop price is invalid")

        return stop_price

    def evaluate(
        self,
        consensus: MultiTimeframeResult,
        execution_signal: StrategySignal,
        entry_price: float,
        account: AccountRiskState,
        market_type: MarketType | str,
    ) -> RiskDecision:

        # 1. Consensus must provide a directional entry.
        if consensus.bias == MarketBias.NEUTRAL:
            return self._reject("Neutral market consensus")

        if consensus.bias not in (
            MarketBias.BULLISH,
            MarketBias.BEARISH,
        ):
            return self._reject("Unsupported market consensus")

        # 2. Strategy signal must belong to the same market.
        if execution_signal.exchange != consensus.exchange:
            return self._reject("Exchange mismatch")

        if execution_signal.symbol != consensus.symbol:
            return self._reject("Symbol mismatch")

        # 3. Execution signal must agree with MTF consensus.
        if execution_signal.bias != consensus.bias:
            return self._reject("Execution signal conflicts with consensus")

        # 4. ATR must be available.
        if execution_signal.atr is None:
            return self._reject("ATR is unavailable")

        side = (
            OrderSide.BUY
            if consensus.bias == MarketBias.BULLISH
            else OrderSide.SELL
        )

        # 5. Calculate ATR-based stop.
        try:
            stop_price = self.calculate_stop_price(
                bias=consensus.bias,
                entry_price=entry_price,
                atr=execution_signal.atr,
            )
        except ValueError as exc:
            return self._reject(str(exc))

        # 6. Build a risk request.
        request = RiskRequest(
            account_equity=account.account_equity,
            available_balance=account.available_balance,
            entry_price=entry_price,
            stop_price=stop_price,
            daily_start_equity=account.daily_start_equity,
            market_type=market_type,
            side=side,
            daily_realized_pnl=account.daily_realized_pnl,
            current_exposure=account.current_exposure,
            open_positions=account.open_positions,
            leverage=account.leverage,
        )

        return self.risk_engine.evaluate(request)
