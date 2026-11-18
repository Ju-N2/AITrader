from math import isfinite

from aitrader.ai.agents import (
    TrendAgent,
    MomentumAgent,
    VolatilityAgent,
    MarketRegimeAgent,
)
from aitrader.ai.aggregator import AnalysisAggregator
from aitrader.ai.models import AIAnalysisResult
from aitrader.data.models import Candle
from aitrader.indicators.models import IndicatorSnapshot


class AIAnalysisEngine:
    """
    Local multi-agent technical analysis engine.

    Input must contain a completed candle and its matching
    indicator snapshot.

    This engine is advisory only and does not submit orders.
    """

    def __init__(
        self,
        trend_agent=None,
        momentum_agent=None,
        volatility_agent=None,
        regime_agent=None,
        aggregator=None,
    ):
        self.trend_agent = trend_agent or TrendAgent()
        self.momentum_agent = momentum_agent or MomentumAgent()
        self.volatility_agent = volatility_agent or VolatilityAgent()
        self.regime_agent = regime_agent or MarketRegimeAgent()
        self.aggregator = aggregator or AnalysisAggregator()

    def analyze(
        self,
        candle: Candle,
        indicators: IndicatorSnapshot,
    ) -> AIAnalysisResult:

        if candle.timestamp_ms != indicators.timestamp_ms:
            raise ValueError(
                "Candle and indicator timestamps do not match"
            )

        if (
            isinstance(candle.close, bool)
            or not isinstance(candle.close, (int, float))
            or not isfinite(candle.close)
            or candle.close <= 0
        ):
            raise ValueError("Invalid candle closing price")

        trend_report = self.trend_agent.analyze(
            candle,
            indicators,
        )

        momentum_report = self.momentum_agent.analyze(
            candle,
            indicators,
        )

        volatility_report = self.volatility_agent.analyze(
            candle,
            indicators,
        )

        regime_report = self.regime_agent.analyze(
            candle,
            indicators,
            volatility_report,
        )

        reports = (
            trend_report,
            momentum_report,
            volatility_report,
            regime_report,
        )

        return self.aggregator.aggregate(
            candle,
            reports,
        )
