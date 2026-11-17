import pytest

from aitrader.core.exceptions import StrategyError
from aitrader.data.models import Candle
from aitrader.indicators.models import IndicatorSnapshot
from aitrader.strategy.models import MarketBias
from aitrader.strategy.technical import TechnicalTrendStrategy


def create_candle(close=120, timestamp=1000):
    return Candle(
        exchange="test",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=timestamp,
        open=close,
        high=close + 2,
        low=close - 2,
        close=close,
        volume=10,
    )


def create_indicators(
    timestamp=1000,
    sma20=110,
    sma50=100,
    ema12=120,
    ema26=110,
    rsi14=60,
    macd_histogram=2,
):
    return IndicatorSnapshot(
        timestamp_ms=timestamp,
        sma_20=sma20,
        sma_50=sma50,
        ema_12=ema12,
        ema_26=ema26,
        rsi_14=rsi14,
        macd=1,
        macd_signal=0,
        macd_histogram=macd_histogram,
        bollinger_upper=120,
        bollinger_middle=110,
        bollinger_lower=100,
        atr_14=2,
    )


def test_bullish_signal():
    strategy = TechnicalTrendStrategy()

    result = strategy.evaluate(
        create_candle(),
        create_indicators(),
    )

    assert result.bias == MarketBias.BULLISH
    assert result.score == 5


def test_bearish_signal():
    strategy = TechnicalTrendStrategy()

    candle = create_candle(close=80)

    indicators = create_indicators(
        sma20=90,
        sma50=100,
        ema12=80,
        ema26=90,
        rsi14=40,
        macd_histogram=-2,
    )

    result = strategy.evaluate(candle, indicators)

    assert result.bias == MarketBias.BEARISH
    assert result.score == -5


def test_neutral_signal():
    strategy = TechnicalTrendStrategy()

    candle = create_candle(close=105)

    indicators = create_indicators(
        sma20=100,
        sma50=105,
        ema12=100,
        ema26=101,
        rsi14=50,
        macd_histogram=1,
    )

    result = strategy.evaluate(candle, indicators)

    assert result.bias == MarketBias.NEUTRAL
    assert result.score == 0


def test_timestamp_mismatch():
    strategy = TechnicalTrendStrategy()

    with pytest.raises(StrategyError):
        strategy.evaluate(
            create_candle(timestamp=1000),
            create_indicators(timestamp=2000),
        )


def test_missing_indicator():
    strategy = TechnicalTrendStrategy()

    indicators = create_indicators()
    indicators = IndicatorSnapshot(
        **{
            **indicators.__dict__,
            "rsi_14": None,
        }
    )

    with pytest.raises(StrategyError):
        strategy.evaluate(create_candle(), indicators)


def test_invalid_threshold():
    with pytest.raises(ValueError):
        TechnicalTrendStrategy(bullish_threshold=6)
