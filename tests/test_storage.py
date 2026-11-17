import pytest

from aitrader.data.models import Candle
from aitrader.data.storage import SQLiteCandleRepository


@pytest.fixture
def repository(tmp_path):
    db_path = tmp_path / "test.db"
    return SQLiteCandleRepository(db_path)


def create_candle(timestamp, close=105):
    return Candle(
        exchange="upbit",
        symbol="BTC/KRW",
        timeframe="1h",
        timestamp_ms=timestamp,
        open=100,
        high=110,
        low=90,
        close=close,
        volume=10,
    )


def test_save_and_read_candles(repository):
    candles = [
        create_candle(1000),
        create_candle(2000),
    ]

    assert repository.save_candles(candles) == 2

    result = repository.get_candles(
        "upbit", "BTC/KRW", "1h"
    )

    assert len(result) == 2
    assert result[0].timestamp_ms == 1000
    assert result[1].timestamp_ms == 2000


def test_duplicate_candle_updates(repository):
    repository.save_candles([create_candle(1000, close=105)])
    repository.save_candles([create_candle(1000, close=120)])

    result = repository.get_candles(
        "upbit", "BTC/KRW", "1h"
    )

    assert len(result) == 1
    assert result[0].close == 120


def test_count_candles(repository):
    repository.save_candles([
        create_candle(1000),
        create_candle(2000),
    ])

    assert repository.count_candles(
        "upbit", "BTC/KRW", "1h"
    ) == 2


def test_empty_save(repository):
    assert repository.save_candles([]) == 0


def test_limit(repository):
    repository.save_candles([
        create_candle(1000),
        create_candle(2000),
        create_candle(3000),
    ])

    result = repository.get_candles(
        "upbit", "BTC/KRW", "1h", limit=2
    )

    assert len(result) == 2
    assert result[0].timestamp_ms == 2000
    assert result[1].timestamp_ms == 3000


def test_invalid_limit(repository):
    with pytest.raises(ValueError):
        repository.get_candles(
            "upbit", "BTC/KRW", "1h", limit=0
        )
