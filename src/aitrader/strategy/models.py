from dataclasses import dataclass
from enum import Enum


class MarketBias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class StrategySignal:
    exchange: str
    symbol: str
    timeframe: str

    timestamp_ms: int

    bias: MarketBias
    score: int
    max_score: int

    reasons: tuple[str, ...]

    atr: float | None = None