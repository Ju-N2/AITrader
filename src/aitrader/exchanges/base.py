from abc import ABC, abstractmethod
from typing import Any


class ExchangeClient(ABC):
    """Common interface for cryptocurrency exchanges."""

    def __init__(self, exchange: Any):
        self.exchange = exchange

    def fetch_ticker(self, symbol: str) -> dict:
        return self.exchange.fetch_ticker(symbol)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        since: int | None = None,
    ) -> list:
        if since is None:
            return self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
            )

        return self.exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
        )

    def load_markets(self) -> dict:
        return self.exchange.load_markets()

    def close(self) -> None:
        close = getattr(self.exchange, "close", None)
        if callable(close):
            close()

    @abstractmethod
    def get_exchange_name(self) -> str:
        pass
