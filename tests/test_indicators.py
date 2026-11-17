import pytest

from aitrader.data.models import Candle
from aitrader.indicators.calculations import (
    sma,
    ema,
    rsi,
    macd,
    bollinger_bands,
    atr,
)
from aitrader.indicators.engine import IndicatorEngine
from aitrader.core.exceptions import DataError


def create_candles(count=60):
    candles = []

    for i in range(count):
        close = 100 + i

        candles.append(
            Candle(
                exchange="test",
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp_ms=1000 + i * 3600000,
                open=close,
                high=close + 2,
                low=close - 2,
                close=close,
                volume=10,
            )
        )

    return candles


def test_sma():
    result = sma([1, 2, 3, 4, 5], 3)

    assert result[:2] == [None, None]
    assert result[2:] == [2.0, 3.0, 4.0]


def test_ema():
    result = ema([1, 2, 3, 4, 5], 3)

    assert result[0] is None
    assert result[1] is None
    assert result[2] == 2.0
    assert result[3] == 3.0
    assert result[4] == 4.0


def test_rsi():
    result = rsi([1, 2, 3, 4, 5], 3)

    assert result[2] is None
    assert result[3] == 100.0


def test_rsi_flat_market():
    result = rsi([10, 10, 10, 10, 10], 3)

    assert result[3] == 50.0


def test_bollinger_bands():
    upper, middle, lower = bollinger_bands(
        [1, 2, 3],
        period=3,
        stddev=2,
    )

    assert middle[2] == 2.0
    assert upper[2] > middle[2]
    assert lower[2] < middle[2]


def test_macd():
    line, signal, histogram = macd(
        list(range(1, 41))
    )

    assert line[24] is None
    assert line[25] is not None
    assert signal[32] is None
    assert signal[33] is not None
    assert histogram[33] is not None


def test_atr():
    result = atr(
        highs=[11, 11, 11, 11],
        lows=[9, 9, 9, 9],
        closes=[10, 10, 10, 10],
        period=3,
    )

    assert result[1] is None
    assert result[2] == 2.0
    assert result[3] == 2.0


def test_invalid_period():
    with pytest.raises(ValueError):
        sma([1, 2, 3], 0)


def test_indicator_engine():
    candles = create_candles(60)

    engine = IndicatorEngine()
    result = engine.calculate(candles)

    assert len(result) == 60
    assert result[-1].sma_20 is not None
    assert result[-1].sma_50 is not None
    assert result[-1].rsi_14 is not None
    assert result[-1].macd is not None
    assert result[-1].atr_14 is not None


def test_empty_candles():
    with pytest.raises(DataError):
        IndicatorEngine().calculate([])


def test_unsorted_candles():
    candles = create_candles(5)
    candles.reverse()

    with pytest.raises(DataError):
        IndicatorEngine().calculate(candles)
