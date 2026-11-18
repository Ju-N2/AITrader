import pytest

from aitrader.risk import (
    AccountRiskState,
    MarketType,
    RiskConfig,
    RiskEngine,
    RiskManager,
)
from aitrader.strategy.models import (
    MarketBias,
    StrategySignal,
)
from aitrader.strategy.multi_timeframe import MultiTimeframeResult


@pytest.fixture
def manager():
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

    return RiskManager(
        RiskEngine(config),
        atr_stop_multiplier=3.0,
    )


@pytest.fixture
def account():
    return AccountRiskState(
        account_equity=10_000.0,
        available_balance=10_000.0,
        daily_start_equity=10_000.0,
    )


def make_consensus(bias=MarketBias.BULLISH):
    return MultiTimeframeResult(
        exchange="binance_spot",
        symbol="BTC/USDT",
        bias=bias,
        weighted_score=2.5,
        available_timeframes=("1m", "5m", "15m", "1h", "4h"),
        missing_timeframes=(),
        reasons=("Bullish consensus",),
    )


def make_signal(bias=MarketBias.BULLISH, atr=2.0):
    return StrategySignal(
        exchange="binance_spot",
        symbol="BTC/USDT",
        timeframe="1m",
        timestamp_ms=1_700_000_000_000,
        bias=bias,
        score=4 if bias == MarketBias.BULLISH else -4,
        max_score=5,
        reasons=("Test signal",),
        atr=atr,
    )


def test_bullish_signal_is_approved(manager, account):
    decision = manager.evaluate(
        consensus=make_consensus(),
        execution_signal=make_signal(),
        entry_price=100.0,
        account=account,
        market_type=MarketType.SPOT,
    )

    assert decision.approved is True
    assert decision.quantity > 0
    assert decision.stop_price == pytest.approx(94.0)
    assert decision.estimated_max_loss <= decision.risk_budget


def test_bearish_signal_calculates_short_stop(manager, account):
    decision = manager.evaluate(
        consensus=make_consensus(MarketBias.BEARISH),
        execution_signal=make_signal(MarketBias.BEARISH),
        entry_price=100.0,
        account=AccountRiskState(
            account_equity=10_000.0,
            available_balance=10_000.0,
            daily_start_equity=10_000.0,
            leverage=2.0,
        ),
        market_type=MarketType.FUTURES,
    )

    assert decision.approved is True
    assert decision.stop_price == pytest.approx(106.0)
    assert decision.effective_leverage == 2.0


def test_neutral_consensus_is_rejected(manager, account):
    decision = manager.evaluate(
        make_consensus(MarketBias.NEUTRAL),
        make_signal(),
        100.0,
        account,
        MarketType.SPOT,
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Neutral market consensus"


def test_signal_conflicting_with_consensus_is_rejected(manager, account):
    decision = manager.evaluate(
        make_consensus(MarketBias.BULLISH),
        make_signal(MarketBias.BEARISH),
        100.0,
        account,
        MarketType.FUTURES,
    )

    assert decision.approved is False
    assert "conflicts" in decision.rejection_reason


def test_missing_atr_is_rejected(manager, account):
    decision = manager.evaluate(
        make_consensus(),
        make_signal(atr=None),
        100.0,
        account,
        MarketType.SPOT,
    )

    assert decision.approved is False
    assert decision.rejection_reason == "ATR is unavailable"


def test_invalid_atr_is_rejected(manager, account):
    decision = manager.evaluate(
        make_consensus(),
        make_signal(atr=0),
        100.0,
        account,
        MarketType.SPOT,
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Invalid ATR"


def test_market_mismatch_is_rejected(manager, account):
    signal = StrategySignal(
        exchange="upbit",
        symbol="BTC/USDT",
        timeframe="1m",
        timestamp_ms=1_700_000_000_000,
        bias=MarketBias.BULLISH,
        score=4,
        max_score=5,
        reasons=("Test signal",),
        atr=2.0,
    )

    decision = manager.evaluate(
        make_consensus(),
        signal,
        100.0,
        account,
        MarketType.SPOT,
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Exchange mismatch"


def test_spot_does_not_open_short_positions(manager, account):
    decision = manager.evaluate(
        make_consensus(MarketBias.BEARISH),
        make_signal(MarketBias.BEARISH),
        100.0,
        account,
        MarketType.SPOT,
    )

    assert decision.approved is False
    assert decision.rejection_reason == (
        "Spot short selling is not supported"
    )


def test_daily_loss_limit_is_enforced_through_manager(manager):
    account = AccountRiskState(
        account_equity=10_000.0,
        available_balance=10_000.0,
        daily_start_equity=10_000.0,
        daily_realized_pnl=-200.0,
    )

    decision = manager.evaluate(
        make_consensus(),
        make_signal(),
        100.0,
        account,
        MarketType.SPOT,
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Daily loss limit reached"


def test_stop_price_must_be_positive(manager, account):
    decision = manager.evaluate(
        make_consensus(),
        make_signal(atr=40.0),
        100.0,
        account,
        MarketType.SPOT,
    )

    assert decision.approved is False
    assert decision.rejection_reason == (
        "Calculated stop price is invalid"
    )


def test_invalid_stop_multiplier_is_rejected():
    with pytest.raises(ValueError):
        RiskManager(
            RiskEngine(
                RiskConfig(
                    risk_per_trade_fraction=0.01,
                    max_position_fraction=0.25,
                    max_total_exposure_fraction=0.50,
                    max_daily_loss_fraction=0.02,
                    max_leverage=3.0,
                )
            ),
            atr_stop_multiplier=0,
        )
