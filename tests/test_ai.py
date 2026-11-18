from dataclasses import replace

import pytest

from aitrader.ai import (
    AIAnalysisEngine,
    AnalysisAggregator,
    MarketRegime,
    VolatilityLevel,
)
from aitrader.ai.agents import (
    TrendAgent,
    MomentumAgent,
    VolatilityAgent,
    MarketRegimeAgent,
)
from aitrader.ai.models import AgentReport
from aitrader.data.models import Candle
from aitrader.indicators.models import IndicatorSnapshot
from aitrader.strategy.models import MarketBias


@pytest.fixture
def candle():
    return Candle(
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


@pytest.fixture
def indicators(candle):
    return IndicatorSnapshot(
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


def test_trend_agent_detects_bullish(candle, indicators):
    report = TrendAgent().analyze(candle, indicators)

    assert report.bias == MarketBias.BULLISH
    assert report.score == 1.0
    assert report.confidence == 1.0


def test_trend_agent_detects_bearish(candle, indicators):
    bearish = replace(
        indicators,
        sma_20=102.0,
        sma_50=105.0,
        ema_12=94.0,
        ema_26=96.0,
    )

    report = TrendAgent().analyze(candle, bearish)

    assert report.bias == MarketBias.BEARISH
    assert report.score == -1.0


def test_trend_agent_handles_missing_indicators(candle, indicators):
    missing = replace(indicators, sma_20=None)

    report = TrendAgent().analyze(candle, missing)

    assert report.bias == MarketBias.NEUTRAL
    assert report.confidence == 0.0


def test_momentum_agent_detects_bullish(candle, indicators):
    report = MomentumAgent().analyze(candle, indicators)

    assert report.bias == MarketBias.BULLISH
    assert report.score == 1.0


def test_momentum_agent_detects_conflicting_signals(candle, indicators):
    mixed = replace(
        indicators,
        rsi_14=60.0,
        macd_histogram=-1.0,
    )

    report = MomentumAgent().analyze(candle, mixed)

    assert report.bias == MarketBias.NEUTRAL
    assert report.score == 0.0


def test_momentum_agent_handles_missing_data(candle, indicators):
    missing = replace(indicators, rsi_14=None)

    report = MomentumAgent().analyze(candle, missing)

    assert report.confidence == 0.0


def test_volatility_agent_detects_high_volatility(candle, indicators):
    volatile = replace(indicators, atr_14=2.0)

    report = VolatilityAgent().analyze(candle, volatile)

    assert report.volatility == VolatilityLevel.HIGH
    assert report.bias == MarketBias.NEUTRAL
    assert report.metrics[0][0] == "atr_pct"
    assert report.metrics[0][1] == pytest.approx(2.0)


def test_volatility_agent_detects_low_volatility(candle, indicators):
    quiet = replace(indicators, atr_14=0.2)

    report = VolatilityAgent().analyze(candle, quiet)

    assert report.volatility == VolatilityLevel.LOW


def test_regime_agent_detects_bull_trend(candle, indicators):
    volatility = VolatilityAgent().analyze(candle, indicators)

    report = MarketRegimeAgent().analyze(
        candle,
        indicators,
        volatility,
    )

    assert report.regime == MarketRegime.BULL_TREND
    assert report.bias == MarketBias.BULLISH


def test_regime_agent_detects_high_volatility(candle, indicators):
    volatile = replace(indicators, atr_14=2.0)

    volatility = VolatilityAgent().analyze(candle, volatile)

    report = MarketRegimeAgent().analyze(
        candle,
        volatile,
        volatility,
    )

    assert report.regime == MarketRegime.HIGH_VOLATILITY


def test_aggregator_combines_directional_agents(candle):
    trend = AgentReport(
        agent_name="trend",
        bias=MarketBias.BULLISH,
        score=1.0,
        confidence=1.0,
        reasons=("Bullish trend",),
    )

    momentum = AgentReport(
        agent_name="momentum",
        bias=MarketBias.BEARISH,
        score=-1.0,
        confidence=1.0,
        reasons=("Bearish momentum",),
    )

    volatility = AgentReport(
        agent_name="volatility",
        bias=MarketBias.NEUTRAL,
        score=0.0,
        confidence=1.0,
        reasons=("Normal volatility",),
        volatility=VolatilityLevel.NORMAL,
    )

    result = AnalysisAggregator().aggregate(
        candle,
        (trend, momentum, volatility),
    )

    assert result.score == pytest.approx(0.2)
    assert result.bias == MarketBias.NEUTRAL
    assert result.volatility == VolatilityLevel.NORMAL


def test_engine_returns_complete_bullish_analysis(candle, indicators):
    result = AIAnalysisEngine().analyze(candle, indicators)

    assert result.bias == MarketBias.BULLISH
    assert result.regime == MarketRegime.BULL_TREND
    assert result.volatility == VolatilityLevel.NORMAL
    assert len(result.reports) == 4
    assert result.confidence > 0


def test_engine_reduces_confidence_in_high_volatility(candle, indicators):
    normal = AIAnalysisEngine().analyze(candle, indicators)

    volatile_indicators = replace(indicators, atr_14=2.0)

    volatile = AIAnalysisEngine().analyze(
        candle,
        volatile_indicators,
    )

    assert volatile.volatility == VolatilityLevel.HIGH
    assert volatile.confidence < normal.confidence
    assert volatile.bias == normal.bias


def test_engine_rejects_timestamp_mismatch(candle, indicators):
    wrong = replace(
        indicators,
        timestamp_ms=indicators.timestamp_ms + 60_000,
    )

    with pytest.raises(ValueError):
        AIAnalysisEngine().analyze(candle, wrong)


def test_engine_handles_indicator_warmup(candle, indicators):
    unavailable = replace(
        indicators,
        sma_20=None,
        sma_50=None,
        ema_12=None,
        ema_26=None,
        rsi_14=None,
        macd_histogram=None,
        bollinger_upper=None,
        bollinger_middle=None,
        bollinger_lower=None,
        atr_14=None,
    )

    result = AIAnalysisEngine().analyze(candle, unavailable)

    assert result.bias == MarketBias.NEUTRAL
    assert result.confidence == 0.0
    assert result.regime == MarketRegime.UNKNOWN
    assert result.volatility == VolatilityLevel.UNKNOWN
