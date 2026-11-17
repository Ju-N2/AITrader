from aitrader.core.logger import setup_logger
from aitrader.data.collector import MarketDataCollector
from aitrader.data.storage import SQLiteCandleRepository


class MarketDataService:
    """Coordinate market data collection and storage."""

    def __init__(
        self,
        client,
        exchange_name: str,
        repository: SQLiteCandleRepository | None = None,
    ):
        self.collector = MarketDataCollector(client, exchange_name)
        self.repository = repository or SQLiteCandleRepository()
        self.logger = setup_logger("aitrader.data")

    def collect_and_store(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> int:
        candles = self.collector.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )

        saved_count = self.repository.save_candles(candles)

        self.logger.info(
            "Market data collected | exchange=%s symbol=%s "
            "timeframe=%s count=%d",
            self.collector.exchange_name,
            symbol,
            timeframe,
            saved_count,
        )

        return saved_count
