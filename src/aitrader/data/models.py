from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Candle:
    exchange: str
    symbol: str
    timeframe: str

    timestamp_ms: int

    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def datetime_utc(self) -> datetime:
        return datetime.fromtimestamp(
            self.timestamp_ms / 1000,
            tz=timezone.utc,
        )
