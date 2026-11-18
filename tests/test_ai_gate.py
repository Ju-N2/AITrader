from dataclasses import replace

import pytest

from aitrader.ai import AIAnalysisEngine, AITradingGate
from aitrader.ai.models import MarketRegime, VolatilityLevel
from aitrader.data.models import Candle
from aitrader.indicators.models import IndicatorSnapshot
from aitrader.risk import (
    AccountRiskState,
    MarketType,
    RiskConfig,
    RiskEngine,
    RiskManager,
)
from aitrader.strategy.models import MarketBias, StrategySignal
from aitrader.strategy.multi_timeframe import MultiTimeframeResult


@pytest.fixture
def market():
    candle = Candle(
        exchange="binance_spot",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1_700_000_000_000,
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
        volume=10.0,
    )

    indicators = IndicatorSnapshot(
        timestamp_ms=candle.timestamp_ms,
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
        atr_14=1.0,
    )

    signal = StrategySignal(
        exchange="binance_spot",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=candle.timestamp_ms,
        bias=MarketBias.BULLISH,
        score=4,
        max_score=5,
        reasons=("Bullish signal",),
        atr=1.0,
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

    analysis = AIAnalysisEngine().analyze(
        candle,
        indicators,
    )

    return candle, indicators, signal, consensus, analysis


@pytest.fixture
def gate():
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

    risk_manager = RiskManager(
        RiskEngine(config),
        atr_stop_multiplier=3.0,
    )

    return AITradingGate(risk_manager)


@pytest.fixture
def account():
    return AccountRiskState(
        account_equity=10_000.0,
        available_balance=10_000.0,
        daily_start_equity=10_000.0,
    )


def evaluate(gate, market, account, analysis=None, consensus=None, signal=None):
    _, _, default_signal, default_consensus, default_analysis = market

    return gate.evaluate(
        analysis=analysis or default_analysis,
        consensus=consensus or default_consensus,
        execution_signal=signal or default_signal,
        entry_price=100.0,
        account=account,
        market_type=MarketType.SPOT,
    )


def test_valid_ai_analysis_passes_to_risk_manager(gate, market, account):
    decision = evaluate(gate, market, account)

    assert decision.approved
    assert decision.quantity > 0
    assert decision.stop_price == pytest.approx(97.0)
    assert any("AI confidence" in item for item in decision.warnings)


def test_neutral_ai_analysis_is_rejected(gate, market, account):
    analysis = replace(
        market[4],
        bias=MarketBias.NEUTRAL,
        score=0.0,
        confidence=0.0,
        agreement=0.0,
    )

    decision = evaluate(gate, market, account, analysis=analysis)

    assert not decision.approved
    assert decision.rejection_reason == "AI analysis is not directional"


def test_ai_consensus_conflict_is_rejected(gate, market, account):
    analysis = replace(
        market[4],
        bias=MarketBias.BEARISH,
        score=-1.0,
    )

    decision = evaluate(gate, market, account, analysis=analysis)

    assert not decision.approved
    assert decision.rejection_reason == (
        "AI analysis conflicts with consensus"
    )


def test_low_confidence_is_rejected(gate, market, account):
    analysis = replace(market[4], confidence=0.50)

    decision = evaluate(gate, market, account, analysis=analysis)

    assert not decision.approved
    assert decision.rejection_reason == (
        "AI confidence below minimum threshold"
    )


def test_low_agreement_is_rejected(gate, market, account):
    analysis = replace(market[4], agreement=0.40)

    decision = evaluate(gate, market, account, analysis=analysis)

    assert not decision.approved
    assert decision.rejection_reason == (
        "AI agreement below minimum threshold"
    )


def test_high_volatility_is_rejected(gate, market, account):
    analysis = replace(
        market[4],
        volatility=VolatilityLevel.HIGH,
    )

    decision = evaluate(gate, market, account, analysis=analysis)

    assert not decision.approved
    assert decision.rejection_reason == (
        "High volatility entry blocked by AI gate"
    )


def test_unknown_regime_is_rejected(gate, market, account):
    analysis = replace(
        market[4],
        regime=MarketRegime.UNKNOWN,
    )

    decision = evaluate(gate, market, account, analysis=analysis)

    assert not decision.approved
    assert decision.rejection_reason == (
        "AI market regime unavailable"
    )


def test_wrong_timestamp_is_rejected(gate, market, account):
    analysis = replace(
        market[4],
        timestamp_ms=market[4].timestamp_ms + 60_000,
    )

    decision = evaluate(gate, market, account, analysis=analysis)

    assert not decision.approved
    assert decision.rejection_reason == (
        "AI analysis timestamp mismatch"
    )


def test_wrong_timeframe_is_rejected(gate, market, account):
    analysis = replace(
        market[4],
        timeframe="15m",
    )

    decision = evaluate(gate, market, account, analysis=analysis)

    assert not decision.approved
    assert decision.rejection_reason == (
        "AI analysis timeframe mismatch"
    )


def test_risk_manager_remains_final_authority(gate, market):
    account_with_daily_loss = AccountRiskState(
        account_equity=10_000.0,
        available_balance=10_000.0,
        daily_start_equity=10_000.0,
        daily_realized_pnl=-200.0,
    )

    decision = evaluate(
        gate,
        market,
        account_with_daily_loss,
    )

    assert not decision.approved
    assert decision.rejection_reason == "Daily loss limit reached"


def test_missing_momentum_data_reduces_confidence(gate, market, account):
    from dataclasses import replace as dc_replace

    candle, indicators, _, _, _ = market

    indicators = dc_replace(
        indicators,
        macd_histogram=None,
        rsi_14=None,
    )

    analysis = AIAnalysisEngine().analyze(
        candle,
        indicators,
    )

    assert analysis.bias == MarketBias.BULLISH
    assert analysis.confidence < 0.70

    decision = evaluate(
        gate,
        market,
        account,
        analysis=analysis,
    )

    assert not decision.approved
    assert decision.rejection_reason == (
        "AI confidence below minimum threshold"
    )


def test_market_mismatch_is_rejected(gate, market, account):
    signal = replace(
        market[2],
        exchange="upbit",
    )

    decision = evaluate(
        gate,
        market,
        account,
        signal=signal,
    )

    assert not decision.approved
    assert decision.rejection_reason == (
        "AI analysis exchange mismatch"
    )


def test_invalid_gate_threshold_is_rejected():
    with pytest.raises(ValueError):
        AITradingGate(
            risk_manager=None,
            min_confidence=1.5,
        )
