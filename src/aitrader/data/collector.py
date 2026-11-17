from math import isfinite

from aitrader.core.exceptions import DataError, ExchangeError
from aitrader.data.models import Candle


class MarketDataCollector:
    """Collect and normalize OHLCV market data."""

    def __init__(self, client, exchange_name: str):
        self.client = client
        self.exchange_name = exchange_name

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        since: int | None = None,
    ) -> list[Candle]:

        if not symbol:
            raise DataError("Symbol cannot be empty")

        if not 1 <= limit <= 1000:
            raise DataError("Limit must be between 1 and 1000")

        if since is not None and since < 0:
            raise DataError("Since must be non-negative")

        try:
            kwargs = {
                "timeframe": timeframe,
                "limit": limit,
            }

            if since is not None:
                kwargs["since"] = since

            rows = self.client.fetch_ohlcv(
                symbol,
                **kwargs,
            )

        except Exception as error:
            raise ExchangeError(
                f"{self.exchange_name} OHLCV request failed"
            ) from error

        if not rows:
            raise DataError("No OHLCV data received")

        candles = []
        previous_timestamp = 0

        for index, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) != 6:
                raise DataError(f"Invalid OHLCV row at index {index}")

            try:
                timestamp = int(row[0])
                open_price, high, low, close, volume = map(
                    float, row[1:]
                )
            except (TypeError, ValueError, OverflowError) as error:
                raise DataError(
                    f"Invalid numeric data at index {index}"
                ) from error

            values = [open_price, high, low, close, volume]

            if not all(isfinite(value) for value in values):
                raise DataError(f"Non-finite value at index {index}")

            if timestamp <= previous_timestamp or timestamp <= 0:
                raise DataError("Invalid candle timestamp ordering")

            if min(open_price, high, low, close) <= 0 or volume < 0:
                raise DataError(f"Invalid price or volume at index {index}")

            if (
                high < low
                or high < max(open_price, close)
                or low > min(open_price, close)
            ):
                raise DataError(f"Invalid OHLC relationship at index {index}")

            candles.append(
                Candle(
                    exchange=self.exchange_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp_ms=timestamp,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
            )

            previous_timestamp = timestamp

        return candles
