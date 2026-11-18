import pytest

from aitrader.core.exceptions import DataError, ExchangeError
from aitrader.data.collector import MarketDataCollector


class FakeExchange:
    def __init__(self, rows=None, error=None):
        self.rows = rows or []
        self.error = error

    def fetch_ohlcv(self, symbol, timeframe, limit):
        if self.error:
            raise self.error
        return self.rows


def test_fetch_ohlcv_success():
    exchange = FakeExchange([
        [1000, 100, 110, 90, 105, 10],
        [2000, 105, 120, 100, 115, 20],
    ])

    collector = MarketDataCollector(exchange, "test")

    candles = collector.fetch_ohlcv("BTC/USDT")

    assert len(candles) == 2
    assert candles[0].close == 105
    assert candles[1].close == 115
    assert candles[0].exchange == "test"


def test_empty_data():
    collector = MarketDataCollector(FakeExchange(), "test")

    with pytest.raises(DataError):
        collector.fetch_ohlcv("BTC/USDT")


def test_invalid_ohlc():
    exchange = FakeExchange([
        [1000, 100, 90, 95, 105, 10],
    ])

    collector = MarketDataCollector(exchange, "test")

    with pytest.raises(DataError):
        collector.fetch_ohlcv("BTC/USDT")


def test_exchange_error():
    exchange = FakeExchange(error=ConnectionError("Connection failed"))

    collector = MarketDataCollector(exchange, "test")

    with pytest.raises(ExchangeError):
        collector.fetch_ohlcv("BTC/USDT")


def test_invalid_limit():
    collector = MarketDataCollector(FakeExchange(), "test")

    with pytest.raises(DataError):
        collector.fetch_ohlcv("BTC/USDT", limit=0)


def test_empty_incremental_response_returns_empty_list():
    class EmptyExchange:
        def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
            return []

    collector = MarketDataCollector(EmptyExchange(), "binance")

    candles = collector.fetch_ohlcv(
        "BTC/USDT",
        timeframe="1m",
        since=123456789,
    )

    assert candles == []
