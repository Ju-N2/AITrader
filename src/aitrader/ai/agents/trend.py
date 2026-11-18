from aitrader.ai.common import direction, numeric
from aitrader.ai.models import AgentReport
from aitrader.data.models import Candle
from aitrader.indicators.models import IndicatorSnapshot
from aitrader.strategy.models import MarketBias


class TrendAgent:
    """
    Analyzes price and moving-average alignment.
    """

    def analyze(
        self,
        candle: Candle,
        indicators: IndicatorSnapshot,
    ) -> AgentReport:

        values = {
            "close": numeric(candle.close),
            "sma_20": numeric(indicators.sma_20),
            "sma_50": numeric(indicators.sma_50),
            "ema_12": numeric(indicators.ema_12),
            "ema_26": numeric(indicators.ema_26),
        }

        if any(value is None for value in values.values()):
            return AgentReport(
                agent_name="trend",
                bias=MarketBias.NEUTRAL,
                score=0.0,
                confidence=0.0,
                reasons=("Trend indicators unavailable",),
            )

        if values["close"] <= 0:
            return AgentReport(
                agent_name="trend",
                bias=MarketBias.NEUTRAL,
                score=0.0,
                confidence=0.0,
                reasons=("Invalid closing price",),
            )

        factors = (
            direction(values["close"], values["sma_20"]),
            direction(values["sma_20"], values["sma_50"]),
            direction(values["ema_12"], values["ema_26"]),
        )

        score = sum(factors) / len(factors)

        if score >= 0.5:
            bias = MarketBias.BULLISH
        elif score <= -0.5:
            bias = MarketBias.BEARISH
        else:
            bias = MarketBias.NEUTRAL

        labels = (
            "Price versus SMA20",
            "SMA20 versus SMA50",
            "EMA12 versus EMA26",
        )

        reasons = tuple(
            f"{label}: "
            f"{'bullish' if factor > 0 else 'bearish' if factor < 0 else 'neutral'}"
            for label, factor in zip(labels, factors)
        )

        return AgentReport(
            agent_name="trend",
            bias=bias,
            score=score,
            confidence=1.0,
            reasons=reasons,
            metrics=(("trend_score", score),),
        )
