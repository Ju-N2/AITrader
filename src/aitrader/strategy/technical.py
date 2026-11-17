from aitrader.core.exceptions import StrategyError
from aitrader.data.models import Candle
from aitrader.indicators.models import IndicatorSnapshot

from aitrader.strategy.models import (
    MarketBias,
    StrategySignal,
)


class TechnicalTrendStrategy:
    """
    Rule-based technical trend strategy.

    Maximum score: +5
    Minimum score: -5
    """

    def __init__(
        self,
        bullish_threshold: int = 3,
        bearish_threshold: int = -3,
    ):
        if not 1 <= bullish_threshold <= 5:
            raise ValueError("Invalid bullish threshold")

        if not -5 <= bearish_threshold <= -1:
            raise ValueError("Invalid bearish threshold")

        if bearish_threshold >= bullish_threshold:
            raise ValueError("Invalid threshold configuration")

        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold

    def evaluate(
        self,
        candle: Candle,
        indicators: IndicatorSnapshot,
    ) -> StrategySignal:

        if candle.timestamp_ms != indicators.timestamp_ms:
            raise StrategyError("Candle and indicator timestamps do not match")

        required = [
            indicators.sma_20,
            indicators.sma_50,
            indicators.ema_12,
            indicators.ema_26,
            indicators.rsi_14,
            indicators.macd_histogram,
        ]

        if any(value is None for value in required):
            raise StrategyError("Required indicators are not available")

        if not 0 <= indicators.rsi_14 <= 100:
            raise StrategyError("Invalid RSI value")

        score = 0
        reasons = []

        # 1. Main trend: maximum +/- 2
        if candle.close > indicators.sma_20 > indicators.sma_50:
            score += 2
            reasons.append("Bullish SMA alignment")

        elif candle.close < indicators.sma_20 < indicators.sma_50:
            score -= 2
            reasons.append("Bearish SMA alignment")

        else:
            reasons.append("Mixed SMA alignment")

        # 2. EMA trend: maximum +/- 1
        if indicators.ema_12 > indicators.ema_26:
            score += 1
            reasons.append("EMA bullish")

        elif indicators.ema_12 < indicators.ema_26:
            score -= 1
            reasons.append("EMA bearish")

        # 3. MACD momentum: maximum +/- 1
        if indicators.macd_histogram > 0:
            score += 1
            reasons.append("MACD bullish momentum")

        elif indicators.macd_histogram < 0:
            score -= 1
            reasons.append("MACD bearish momentum")

        # 4. RSI momentum: maximum +/- 1
        if indicators.rsi_14 >= 55:
            score += 1
            reasons.append("RSI bullish momentum")

        elif indicators.rsi_14 <= 45:
            score -= 1
            reasons.append("RSI bearish momentum")

        else:
            reasons.append("RSI neutral")

        # Final market bias
        if score >= self.bullish_threshold:
            bias = MarketBias.BULLISH

        elif score <= self.bearish_threshold:
            bias = MarketBias.BEARISH

        else:
            bias = MarketBias.NEUTRAL

        return StrategySignal(
            exchange=candle.exchange,
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            timestamp_ms=candle.timestamp_ms,
            bias=bias,
            score=score,
            max_score=5,
            reasons=tuple(reasons),
            atr=indicators.atr_14,
        )
