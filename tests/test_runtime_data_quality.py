import pytest

from aitrader.core.runtime import PaperTradingRuntime
from aitrader.data.models import Candle


INTERVAL = 3_600_000


def make_candle(timestamp):
    return Candle(
        exchange="binance_spot",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=timestamp,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=10.0,
    )


def test_continuous_hourly_candles_are_accepted():
    candles = [
        make_candle(1_700_000_000_000 + i * INTERVAL)
        for i in range(5)
    ]

    assert PaperTradingRuntime._validate_candle_continuity(
        candles,
        "1h",
    ) is True


def test_missing_hourly_candle_is_rejected():
    candles = [
        make_candle(1_700_000_000_000),
        make_candle(1_700_000_000_000 + INTERVAL),
        make_candle(1_700_000_000_000 + INTERVAL * 3),
    ]

    with pytest.raises(RuntimeError, match="Candle continuity error"):
        PaperTradingRuntime._validate_candle_continuity(
            candles,
            "1h",
        )


def test_irregular_candle_interval_is_rejected():
    candles = [
        make_candle(1_700_000_000_000),
        make_candle(1_700_000_000_000 + INTERVAL),
        make_candle(1_700_000_000_000 + INTERVAL + 1000),
    ]

    with pytest.raises(RuntimeError, match="Candle continuity error"):
        PaperTradingRuntime._validate_candle_continuity(
            candles,
            "1h",
        )
