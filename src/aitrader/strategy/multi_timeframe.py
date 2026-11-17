from dataclasses import dataclass
from collections.abc import Mapping

from aitrader.core.exceptions import StrategyError
from aitrader.strategy.models import MarketBias, StrategySignal


@dataclass(frozen=True)
class MultiTimeframeResult:
    exchange: str
    symbol: str

    bias: MarketBias
    weighted_score: float

    available_timeframes: tuple[str, ...]
    missing_timeframes: tuple[str, ...]

    reasons: tuple[str, ...]


class MultiTimeframeEngine:

    WEIGHTS = {
        "1m": 1,
        "5m": 2,
        "15m": 3,
        "1h": 4,
        "4h": 5,
    }

    CORE_TIMEFRAMES = {"15m", "1h", "4h"}

    def __init__(
        self,
        bullish_threshold: float = 1.5,
        bearish_threshold: float = -1.5,
        min_timeframes: int = 4,
    ):
        if bullish_threshold <= 0 or bearish_threshold >= 0:
            raise ValueError("Invalid score thresholds")

        if bullish_threshold <= bearish_threshold:
            raise ValueError("Invalid threshold configuration")

        if not 1 <= min_timeframes <= len(self.WEIGHTS):
            raise ValueError("Invalid minimum timeframe count")

        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.min_timeframes = min_timeframes

    def evaluate(
        self,
        signals: Mapping[str, StrategySignal],
    ) -> MultiTimeframeResult:

        unknown = set(signals) - set(self.WEIGHTS)

        if unknown:
            raise StrategyError(
                f"Unsupported timeframes: {sorted(unknown)}"
            )

        if not signals:
            raise StrategyError("No timeframe signals provided")

        first = next(iter(signals.values()))

        total_score = 0.0
        total_weight = 0

        reasons = []

        for timeframe, signal in signals.items():

            if signal.timeframe != timeframe:
                raise StrategyError(
                    f"Timeframe mismatch: {timeframe}"
                )

            if signal.exchange != first.exchange:
                raise StrategyError("Exchange mismatch")

            if signal.symbol != first.symbol:
                raise StrategyError("Symbol mismatch")

            if signal.max_score <= 0:
                raise StrategyError("Invalid maximum score")

            if not -signal.max_score <= signal.score <= signal.max_score:
                raise StrategyError("Strategy score out of range")

            weight = self.WEIGHTS[timeframe]

            normalized_score = (
                signal.score / signal.max_score
            ) * 5

            total_score += normalized_score * weight
            total_weight += weight

            reasons.append(
                f"{timeframe}: {signal.score:+d}/"
                f"{signal.max_score} (weight {weight})"
            )

        weighted_score = total_score / total_weight

        available = tuple(
            timeframe
            for timeframe in self.WEIGHTS
            if timeframe in signals
        )

        missing = tuple(
            timeframe
            for timeframe in self.WEIGHTS
            if timeframe not in signals
        )

        coverage_ok = (
            len(available) >= self.min_timeframes
            and self.CORE_TIMEFRAMES.issubset(available)
        )

        if not coverage_ok:
            bias = MarketBias.NEUTRAL
            reasons.append("Insufficient timeframe coverage")

        elif (
            signals["4h"].bias == MarketBias.BEARISH
            or signals["1h"].bias == MarketBias.BEARISH
            or weighted_score <= self.bearish_threshold
        ):
            bias = MarketBias.BEARISH
            reasons.append("Higher timeframe bearish filter")

        elif weighted_score >= self.bullish_threshold:

            if signals["4h"].bias != MarketBias.BULLISH:
                bias = MarketBias.NEUTRAL
                reasons.append("4h trend filter blocked long entry")

            elif signals["15m"].bias != MarketBias.BULLISH:
                bias = MarketBias.NEUTRAL
                reasons.append("15m confirmation missing")

            elif (
                "1m" in signals
                and signals["1m"].bias != MarketBias.BULLISH
            ):
                bias = MarketBias.NEUTRAL
                reasons.append("1m entry confirmation missing")

            else:
                bias = MarketBias.BULLISH
                reasons.append("Bullish multi-timeframe consensus")

        else:
            bias = MarketBias.NEUTRAL
            reasons.append("Mixed market signals")

        return MultiTimeframeResult(
            exchange=first.exchange,
            symbol=first.symbol,
            bias=bias,
            weighted_score=weighted_score,
            available_timeframes=available,
            missing_timeframes=missing,
            reasons=tuple(reasons),
        )
