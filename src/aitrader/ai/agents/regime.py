from aitrader.ai.common import numeric
from aitrader.ai.models import (
    AgentReport,
    MarketRegime,
    VolatilityLevel,
)
from aitrader.data.models import Candle
from aitrader.indicators.models import IndicatorSnapshot
from aitrader.strategy.models import MarketBias


class MarketRegimeAgent:
    """
    Classifies the current market regime.
    """

    def analyze(
        self,
        candle: Candle,
        indicators: IndicatorSnapshot,
        volatility_report: AgentReport,
    ) -> AgentReport:

        if volatility_report.confidence > 0:
            if volatility_report.volatility == VolatilityLevel.HIGH:
                return AgentReport(
                    agent_name="regime",
                    bias=MarketBias.NEUTRAL,
                    score=0.0,
                    confidence=1.0,
                    reasons=("Market classified as high volatility",),
                    volatility=volatility_report.volatility,
                    regime=MarketRegime.HIGH_VOLATILITY,
                )

        values = (
            numeric(candle.close),
            numeric(indicators.sma_20),
            numeric(indicators.sma_50),
            numeric(indicators.ema_12),
            numeric(indicators.ema_26),
        )

        if any(value is None for value in values):
            return AgentReport(
                agent_name="regime",
                bias=MarketBias.NEUTRAL,
                score=0.0,
                confidence=0.0,
                reasons=("Market regime data unavailable",),
                regime=MarketRegime.UNKNOWN,
            )

        close, sma20, sma50, ema12, ema26 = values

        if close > sma20 > sma50 and ema12 > ema26:
            regime = MarketRegime.BULL_TREND
            bias = MarketBias.BULLISH
            reason = "Bullish trending market"

        elif close < sma20 < sma50 and ema12 < ema26:
            regime = MarketRegime.BEAR_TREND
            bias = MarketBias.BEARISH
            reason = "Bearish trending market"

        else:
            regime = MarketRegime.RANGE_OR_TRANSITION
            bias = MarketBias.NEUTRAL
            reason = "Range or transition market"

        score = (
            1.0 if bias == MarketBias.BULLISH
            else -1.0 if bias == MarketBias.BEARISH
            else 0.0
        )

        return AgentReport(
            agent_name="regime",
            bias=bias,
            score=score,
            confidence=1.0,
            reasons=(reason,),
            volatility=volatility_report.volatility,
            regime=regime,
        )
