from aitrader.ai.common import numeric
from aitrader.ai.models import AgentReport
from aitrader.data.models import Candle
from aitrader.indicators.models import IndicatorSnapshot
from aitrader.strategy.models import MarketBias


class MomentumAgent:
    """
    Analyzes RSI and MACD momentum.
    """

    def analyze(
        self,
        candle: Candle,
        indicators: IndicatorSnapshot,
    ) -> AgentReport:

        rsi_value = numeric(indicators.rsi_14)
        macd_histogram = numeric(indicators.macd_histogram)

        if rsi_value is None or macd_histogram is None:
            return AgentReport(
                agent_name="momentum",
                bias=MarketBias.NEUTRAL,
                score=0.0,
                confidence=0.0,
                reasons=("Momentum indicators unavailable",),
            )

        if not 0 <= rsi_value <= 100:
            return AgentReport(
                agent_name="momentum",
                bias=MarketBias.NEUTRAL,
                score=0.0,
                confidence=0.0,
                reasons=("Invalid RSI value",),
            )

        if rsi_value >= 55:
            rsi_direction = 1
            rsi_reason = "RSI bullish"
        elif rsi_value <= 45:
            rsi_direction = -1
            rsi_reason = "RSI bearish"
        else:
            rsi_direction = 0
            rsi_reason = "RSI neutral"

        if macd_histogram > 0:
            macd_direction = 1
            macd_reason = "MACD bullish momentum"
        elif macd_histogram < 0:
            macd_direction = -1
            macd_reason = "MACD bearish momentum"
        else:
            macd_direction = 0
            macd_reason = "MACD neutral"

        score = (rsi_direction + macd_direction) / 2

        if score >= 0.5:
            bias = MarketBias.BULLISH
        elif score <= -0.5:
            bias = MarketBias.BEARISH
        else:
            bias = MarketBias.NEUTRAL

        return AgentReport(
            agent_name="momentum",
            bias=bias,
            score=score,
            confidence=1.0,
            reasons=(rsi_reason, macd_reason),
            metrics=(
                ("rsi_14", rsi_value),
                ("macd_histogram", macd_histogram),
            ),
        )
