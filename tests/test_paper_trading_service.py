from datetime import UTC, datetime

import pytest

from aitrader.ai import AIAnalysisEngine, AITradingGate
from aitrader.core.paper_trading import PaperTradingService
from aitrader.data.models import Candle
from aitrader.execution import OrderEngine
from aitrader.execution.models import OrderStatus
from aitrader.indicators.models import IndicatorSnapshot
from aitrader.portfolio import PositionManager, PositionStatus
from aitrader.risk import (
    AccountRiskState,
    MarketType,
    RiskConfig,
    RiskEngine,
    RiskManager,
)
from aitrader.strategy.models import MarketBias, StrategySignal
from aitrader.strategy.multi_timeframe import MultiTimeframeResult


class FakeNotifier:
    def __init__(self, fail=False):
        self.events = []
        self.fail = fail

    def _record(self, event, *args):
        if self.fail:
            raise RuntimeError("Telegram unavailable")

        self.events.append((event, args))

    def send_order_notification(self, order):
        self._record("order", order)

    def send_position_opened(self, position):
        self._record("opened", position)

    def send_position_closed(self, position):
        self._record("closed", position)

    def send_message(self, text):
        self._record("message", text)

    def send_daily_summary(self, *args):
        self._record("daily", *args)

    def send_system_error(self, *args):
        self._record("error", *args)


def build_service(tmp_path, atr=1.0, notifier=None):
    timestamp = 1_700_000_000_000

    candle = Candle(
        exchange="binance_spot",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=timestamp,
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
        volume=10.0,
    )

    indicators = IndicatorSnapshot(
        timestamp_ms=timestamp,
        sma_20=98.0,
        sma_50=95.0,
        ema_12=99.0,
        ema_26=97.0,
        rsi_14=60.0,
        macd=2.0,
        macd_signal=1.0,
        macd_histogram=1.0,
        bollinger_upper=105.0,
        bollinger_middle=100.0,
        bollinger_lower=95.0,
        atr_14=atr,
    )

    signal = StrategySignal(
        exchange="binance_spot",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=timestamp,
        bias=MarketBias.BULLISH,
        score=4,
        max_score=5,
        reasons=("Bullish signal",),
        atr=atr,
    )

    consensus = MultiTimeframeResult(
        exchange="binance_spot",
        symbol="BTC/USDT",
        bias=MarketBias.BULLISH,
        weighted_score=2.5,
        available_timeframes=("1m", "5m", "15m", "1h", "4h"),
        missing_timeframes=(),
        reasons=("Bullish consensus",),
    )

    account = AccountRiskState(
        account_equity=10_000.0,
        available_balance=10_000.0,
        daily_start_equity=10_000.0,
    )

    config = RiskConfig(
        risk_per_trade_fraction=0.01,
        max_position_fraction=0.25,
        max_total_exposure_fraction=0.50,
        max_daily_loss_fraction=0.02,
        max_leverage=3.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
        min_notional=10.0,
        max_open_positions=3,
    )

    gate = AITradingGate(
        RiskManager(
            RiskEngine(config),
            atr_stop_multiplier=3.0,
        )
    )

    analysis = AIAnalysisEngine().analyze(
        candle,
        indicators,
    )

    service = PaperTradingService(
        ai_gate=gate,
        order_engine=OrderEngine(),
        position_manager=PositionManager(
            tmp_path / "paper_trading.db"
        ),
        notifier=notifier,
    )

    return service, analysis, consensus, signal, account


def enter_position(service, analysis, consensus, signal, account):
    return service.enter(
        analysis=analysis,
        consensus=consensus,
        execution_signal=signal,
        entry_price=100.0,
        account=account,
        market_type=MarketType.SPOT,
        client_order_id="service-entry-001",
    )


def test_entry_sends_order_and_position_notifications(tmp_path):
    notifier = FakeNotifier()

    service, analysis, consensus, signal, account = build_service(
        tmp_path,
        notifier=notifier,
    )

    result = enter_position(
        service,
        analysis,
        consensus,
        signal,
        account,
    )

    assert result.decision.approved
    assert result.order.status == OrderStatus.FILLED
    assert result.position.status == PositionStatus.OPEN

    assert [event[0] for event in notifier.events] == [
        "order",
        "opened",
    ]


def test_rejected_entry_sends_order_notification(tmp_path):
    notifier = FakeNotifier()

    service, analysis, consensus, signal, account = build_service(
        tmp_path,
        atr=2.0,
        notifier=notifier,
    )

    result = enter_position(
        service,
        analysis,
        consensus,
        signal,
        account,
    )

    assert not result.decision.approved
    assert result.order.status == OrderStatus.REJECTED
    assert result.position is None

    assert [event[0] for event in notifier.events] == ["order"]


def test_telegram_failure_does_not_undo_position(tmp_path):
    notifier = FakeNotifier(fail=True)

    service, analysis, consensus, signal, account = build_service(
        tmp_path,
        notifier=notifier,
    )

    result = enter_position(
        service,
        analysis,
        consensus,
        signal,
        account,
    )

    assert result.order.status == OrderStatus.FILLED
    assert result.position.status == PositionStatus.OPEN
    assert len(result.notification_errors) == 2

    restored = service.position_manager.get_position(
        result.position.position_id
    )

    assert restored.status == PositionStatus.OPEN


def test_partial_and_full_exit_send_notifications(tmp_path):
    notifier = FakeNotifier()

    service, analysis, consensus, signal, account = build_service(
        tmp_path,
        notifier=notifier,
    )

    entry = enter_position(
        service,
        analysis,
        consensus,
        signal,
        account,
    )

    position = entry.position
    half = position.remaining_quantity / 2

    partial = service.close_position(
        position.position_id,
        mark_price=110.0,
        quantity=half,
        client_order_id="service-exit-001",
    )

    assert partial.order.reduce_only
    assert partial.position.status == PositionStatus.OPEN
    assert partial.position.remaining_quantity == pytest.approx(half)

    assert notifier.events[-1][0] == "message"

    closed = service.close_position(
        position.position_id,
        mark_price=105.0,
        client_order_id="service-exit-002",
    )

    assert closed.position.status == PositionStatus.CLOSED
    assert closed.position.remaining_quantity == 0

    assert notifier.events[-1][0] == "closed"


def test_exit_cannot_exceed_remaining_quantity(tmp_path):
    service, analysis, consensus, signal, account = build_service(
        tmp_path,
    )

    entry = enter_position(
        service,
        analysis,
        consensus,
        signal,
        account,
    )

    with pytest.raises(ValueError, match="exceeds"):
        service.close_position(
            entry.position.position_id,
            mark_price=110.0,
            quantity=entry.position.remaining_quantity + 1,
        )


def test_daily_summary_uses_closed_positions(tmp_path):
    notifier = FakeNotifier()

    service, analysis, consensus, signal, account = build_service(
        tmp_path,
        notifier=notifier,
    )

    entry = enter_position(
        service,
        analysis,
        consensus,
        signal,
        account,
    )

    service.close_position(
        entry.position.position_id,
        mark_price=110.0,
    )

    today = datetime.now(UTC).date().isoformat()

    errors = service.send_daily_summary(
        utc_date=today,
        equity=10_100.0,
    )

    assert errors == ()
    assert notifier.events[-1][0] == "daily"

    summary = notifier.events[-1][1]

    assert summary[1] == 1
    assert summary[2] == 1
    assert summary[3] == 0


def test_system_error_notification(tmp_path):
    notifier = FakeNotifier()

    service, *_ = build_service(
        tmp_path,
        notifier=notifier,
    )

    errors = service.notify_system_error(
        "RiskEngine",
        "Test failure",
    )

    assert errors == ()
    assert notifier.events[-1][0] == "error"


def test_missing_mark_price_blocks_incomplete_daily_summary(tmp_path):
    service, analysis, consensus, signal, account = build_service(
        tmp_path,
    )

    entry = enter_position(
        service,
        analysis,
        consensus,
        signal,
        account,
    )

    with pytest.raises(ValueError, match="Missing mark price"):
        service.send_daily_summary(
            utc_date=datetime.now(UTC).date().isoformat(),
            equity=10_000.0,
        )
