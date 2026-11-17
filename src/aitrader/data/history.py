from datetime import datetime, timezone
import time
import ccxt
from aitrader.core.exceptions import DataError
from aitrader.data.collector import MarketDataCollector
from aitrader.data.storage import SQLiteCandleRepository


DAY_MS = 86_400_000

TIMEFRAME_CONFIG = {
    "1m": {"interval": 60_000, "days": 30},
    "5m": {"interval": 300_000, "days": 30},
    "15m": {"interval": 900_000, "days": 30},
    "1h": {"interval": 3_600_000, "days": 90},
    "4h": {"interval": 14_400_000, "days": 180},
}

PAGE_LIMITS = {
    "upbit": 200,
    "binance": 1000,
}

MAX_PAGES = 500


class HistoricalDataDownloader:

    def _fetch_page(
        self,
        symbol: str,
        timeframe: str,
        cursor: int,
        limit: int,
    ):
        max_retries = 4

        for attempt in range(max_retries):
            try:
                return self.collector.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    since=cursor,
                )

            except Exception as error:
                cause = error.__cause__

                if not isinstance(cause, ccxt.RateLimitExceeded):
                    raise

                if attempt == max_retries - 1:
                    raise

                delay = 2 ** (attempt + 1)

                print(
                    f"[RATE LIMIT] Waiting {delay}s "
                    f"before retry ({attempt + 1}/{max_retries - 1})"
                )

                time.sleep(delay)

        raise DataError("Historical download retry limit exceeded")

    def __init__(
        self,
        client,
        exchange_name: str,
        repository: SQLiteCandleRepository,
    ):
        self.client = client
        self.exchange_name = exchange_name
        self.repository = repository

        self.collector = MarketDataCollector(
            client,
            exchange_name,
        )

    def download(
        self,
        symbol: str,
        timeframe: str,
        days_override: int | None = None,
    ) -> int:

        if timeframe not in TIMEFRAME_CONFIG:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        config = TIMEFRAME_CONFIG[timeframe]

        interval_ms = config["interval"]
        days = config["days"] if days_override is None else days_override

        if (
            not isinstance(days, int)
            or isinstance(days, bool)
            or days <= 0
        ):
            raise ValueError("Invalid historical download period")

        now_ms = int(
            datetime.now(timezone.utc).timestamp() * 1000
        )

        # 진행 중인 캔들을 제외
        end_ms = (now_ms // interval_ms) * interval_ms

        start_ms = (
            (end_ms - days * DAY_MS) // interval_ms
        ) * interval_ms

        exchange_id = self.client.exchange.id

        page_limit = PAGE_LIMITS.get(exchange_id, 500)

        cursor = start_ms
        processed = 0
        pages = 0

        while cursor < end_ms:

            if pages >= MAX_PAGES:
                raise DataError("Historical download page limit reached")

            candles = self._fetch_page(
                symbol=symbol,
                timeframe=timeframe,
                cursor=cursor,
                limit=page_limit,
            )

            pages += 1

            if not candles:
                break

            valid_candles = [
                candle
                for candle in candles
                if cursor <= candle.timestamp_ms < end_ms
            ]

            if not valid_candles:
                if any(c.timestamp_ms >= end_ms for c in candles):
                    break

                raise DataError("Historical pagination did not advance")

            processed += self.repository.save_candles(valid_candles)

            if exchange_id == "upbit":
                next_cursor = cursor + interval_ms * page_limit
            else:
                last_timestamp = valid_candles[-1].timestamp_ms
                next_cursor = last_timestamp + interval_ms

            if next_cursor <= cursor:
                raise DataError("Historical pagination stalled")

            print(
                f"[{exchange_id} {timeframe}] "
                f"Page {pages} | "
                f"Fetched: {len(valid_candles)} | "
                f"Processed: {processed}"
            )

            cursor = next_cursor

        return processed
