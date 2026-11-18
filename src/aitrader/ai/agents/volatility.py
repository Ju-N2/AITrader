from aitrader.ai.common import numeric
from aitrader.ai.models import AgentReport, VolatilityLevel
from aitrader.data.models import Candle
from aitrader.indicators.models import IndicatorSnapshot
from aitrader.strategy.models import MarketBias


class VolatilityAgent:
    """
    Measures volatility using ATR and Bollinger Band width.

    Thresholds are configurable because volatility differs
    across assets and timeframes.
    """

    def __init__(
        self,
        elevated_atr_pct: float = 0.75,
        high_atr_pct: float = 1.50,
    ):
        if (
            elevated_atr_pct <= 0
            or high_atr_pct <= elevated_atr_pct
        ):
            raise ValueError("Invalid volatility thresholds")

        self.elevated_atr_pct = elevated_atr_pct
        self.high_atr_pct = high_atr_pct

    def analyze(
        self,
        candle: Candle,
        indicators: IndicatorSnapshot,
    ) -> AgentReport:

        close = numeric(candle.close)
        atr = numeric(indicators.atr_14)
        upper = numeric(indicators.bollinger_upper)
        middle = numeric(indicators.bollinger_middle)
        lower = numeric(indicators.bollinger_lower)

        values = (close, atr, upper, middle, lower)

        if any(value is None for value in values):
            return AgentReport(
                agent_name="volatility",
                bias=MarketBias.NEUTRAL,
                score=0.0,
                confidence=0.0,
                reasons=("Volatility indicators unavailable",),
            )

        if close <= 0 or middle <= 0 or atr < 0 or upper < lower:
            return AgentReport(
                agent_name="volatility",
                bias=MarketBias.NEUTRAL,
                score=0.0,
                confidence=0.0,
                reasons=("Invalid volatility data",),
            )

        atr_pct = atr / close * 100
        band_width_pct = (upper - lower) / middle * 100

        if atr_pct >= self.high_atr_pct:
            level = VolatilityLevel.HIGH
            reason = "High volatility"
        elif atr_pct >= self.elevated_atr_pct:
            level = VolatilityLevel.NORMAL
            reason = "Elevated volatility"
        else:
            level = VolatilityLevel.LOW
            reason = "Low volatility"

        return AgentReport(
            agent_name="volatility",
            bias=MarketBias.NEUTRAL,
            score=0.0,
            confidence=1.0,
            reasons=(reason,),
            volatility=level,
            metrics=(
                ("atr_pct", atr_pct),
                ("bollinger_width_pct", band_width_pct),
            ),
        )
