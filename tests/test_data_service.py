from aitrader.data.service import MarketDataService
from aitrader.data.storage import SQLiteCandleRepository


class FakeExchange:
    def fetch_ohlcv(self, symbol, timeframe, limit):
        return [
            [1000, 100, 110, 90, 105, 10],
            [2000, 105, 120, 100, 115, 20],
        ]


def test_collect_and_store(tmp_path):
    repository = SQLiteCandleRepository(tmp_path / "test.db")

    service = MarketDataService(
        client=FakeExchange(),
        exchange_name="test",
        repository=repository,
    )

    count = service.collect_and_store(
        symbol="BTC/USDT",
        timeframe="1h",
        limit=2,
    )

    assert count == 2

    candles = repository.get_candles(
        "test",
        "BTC/USDT",
        "1h",
    )

    assert len(candles) == 2
    assert candles[0].close == 105
    assert candles[1].close == 115
