from dataclasses import dataclass
from enum import Enum
from math import isfinite

from aitrader.strategy.models import MarketBias


class VolatilityLevel(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"


class MarketRegime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RANGE_OR_TRANSITION = "range_or_transition"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AgentReport:
    agent_name: str
    bias: MarketBias
    score: float
    confidence: float
    reasons: tuple[str, ...]

    volatility: VolatilityLevel = VolatilityLevel.UNKNOWN
    regime: MarketRegime = MarketRegime.UNKNOWN

    metrics: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        if not self.agent_name:
            raise ValueError("Agent name is required")

        if not isfinite(self.score) or not -1 <= self.score <= 1:
            raise ValueError("Agent score must be between -1 and 1")

        if (
            not isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Confidence must be between 0 and 1")

        if not isinstance(self.bias, MarketBias):
            raise ValueError("Invalid market bias")

        if not all(isinstance(reason, str) for reason in self.reasons):
            raise ValueError("Invalid agent reasons")

        for name, value in self.metrics:
            if not name or not isfinite(value):
                raise ValueError("Invalid agent metric")


@dataclass(frozen=True, slots=True)
class AIAnalysisResult:
    exchange: str
    symbol: str
    timeframe: str
    timestamp_ms: int

    bias: MarketBias
    score: float
    confidence: float
    agreement: float

    volatility: VolatilityLevel
    regime: MarketRegime

    reports: tuple[AgentReport, ...]
    reasons: tuple[str, ...]

    def __post_init__(self):
        if not self.exchange or not self.symbol or not self.timeframe:
            raise ValueError("Invalid analysis market information")

        if self.timestamp_ms < 0:
            raise ValueError("Invalid analysis timestamp")

        if not isfinite(self.score) or not -1 <= self.score <= 1:
            raise ValueError("Invalid analysis score")

        if not 0 <= self.confidence <= 1:
            raise ValueError("Invalid analysis confidence")

        if not 0 <= self.agreement <= 1:
            raise ValueError("Invalid analysis agreement")
