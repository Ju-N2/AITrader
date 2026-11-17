import sqlite3
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Iterable

from aitrader.core.exceptions import StorageError
from aitrader.data.models import Candle


class SQLiteCandleRepository:
    """SQLite repository for cryptocurrency candle data."""

    def __init__(self, db_path: str | Path = "data/aitrader.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.db_path, timeout=10)

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        try:
            with self._connection() as connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS candles (
                        exchange TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp_ms INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,

                        PRIMARY KEY (
                            exchange,
                            symbol,
                            timeframe,
                            timestamp_ms
                        )
                    )
                """)

                connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_candles_lookup
                    ON candles (symbol, timeframe, timestamp_ms)
                """)

        except sqlite3.Error as error:
            raise StorageError("Failed to initialize database") from error

    def save_candles(self, candles: Iterable[Candle]) -> int:
        rows = [
            (
                candle.exchange,
                candle.symbol,
                candle.timeframe,
                candle.timestamp_ms,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            )
            for candle in candles
        ]

        if not rows:
            return 0

        try:
            with self._connection() as connection:
                connection.executemany("""
                    INSERT INTO candles (
                        exchange,
                        symbol,
                        timeframe,
                        timestamp_ms,
                        open,
                        high,
                        low,
                        close,
                        volume
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

                    ON CONFLICT (
                        exchange,
                        symbol,
                        timeframe,
                        timestamp_ms
                    )
                    DO UPDATE SET
                        open = excluded.open,
                        high = excluded.high,
                        low = excluded.low,
                        close = excluded.close,
                        volume = excluded.volume
                """, rows)

            return len(rows)

        except sqlite3.Error as error:
            raise StorageError("Failed to save candle data") from error

    def get_candles(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[Candle]:
        if limit < 1:
            raise ValueError("Limit must be greater than zero")

        try:
            with self._connection() as connection:
                connection.row_factory = sqlite3.Row

                rows = connection.execute("""
                    SELECT *
                    FROM candles
                    WHERE exchange = ?
                      AND symbol = ?
                      AND timeframe = ?
                    ORDER BY timestamp_ms DESC
                    LIMIT ?
                """, (exchange, symbol, timeframe, limit)).fetchall()

            rows.reverse()

            return [
                Candle(
                    exchange=row["exchange"],
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    timestamp_ms=row["timestamp_ms"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )
                for row in rows
            ]

        except sqlite3.Error as error:
            raise StorageError("Failed to read candle data") from error

    def count_candles(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
    ) -> int:
        try:
            with self._connection() as connection:
                result = connection.execute("""
                    SELECT COUNT(*)
                    FROM candles
                    WHERE exchange = ?
                      AND symbol = ?
                      AND timeframe = ?
                """, (exchange, symbol, timeframe)).fetchone()

            return result[0]

        except sqlite3.Error as error:
            raise StorageError("Failed to count candle data") from error
