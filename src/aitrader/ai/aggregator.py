from aitrader.ai.models import (
    AgentReport,
    AIAnalysisResult,
    MarketRegime,
    VolatilityLevel,
)
from aitrader.data.models import Candle
from aitrader.strategy.models import MarketBias


class AnalysisAggregator:
    """
    Combines directional reports.

    Volatility and regime reports provide context but do not
    directly vote on directional bias.
    """

    WEIGHTS = {
        "trend": 0.60,
        "momentum": 0.40,
    }

    BIAS_THRESHOLD = 0.35

    def aggregate(
        self,
        candle: Candle,
        reports: tuple[AgentReport, ...],
    ) -> AIAnalysisResult:

        report_map = {
            report.agent_name: report
            for report in reports
        }

        if len(report_map) != len(reports):
            raise ValueError("Duplicate agent reports")

        weighted_score = 0.0
        total_weight = 0.0

        active_reports = []

        for name, weight in self.WEIGHTS.items():
            report = report_map.get(name)

            if report is None or report.confidence <= 0:
                continue

            adjusted_weight = weight * report.confidence

            weighted_score += report.score * adjusted_weight
            total_weight += adjusted_weight

            active_reports.append((report, adjusted_weight))

        if total_weight > 0:
            score = weighted_score / total_weight
        else:
            score = 0.0

        if score >= self.BIAS_THRESHOLD:
            bias = MarketBias.BULLISH
        elif score <= -self.BIAS_THRESHOLD:
            bias = MarketBias.BEARISH
        else:
            bias = MarketBias.NEUTRAL

        if bias == MarketBias.NEUTRAL or total_weight == 0:
            agreement = 0.0
            confidence = 0.0
        else:
            support = 0.0

            for report, weight in active_reports:
                if report.bias == bias:
                    support += weight
                elif report.bias == MarketBias.NEUTRAL:
                    support += weight * 0.5

            agreement = support / total_weight

            # Missing directional agents reduce confidence.
            total_expected_weight = sum(self.WEIGHTS.values())
            coverage = total_weight / total_expected_weight

            confidence = min(
                1.0,
                abs(score) * agreement * coverage,
            )

        volatility_report = report_map.get("volatility")
        regime_report = report_map.get("regime")

        volatility = (
            volatility_report.volatility
            if volatility_report
            else VolatilityLevel.UNKNOWN
        )

        regime = (
            regime_report.regime
            if regime_report
            else MarketRegime.UNKNOWN
        )

        reasons = []

        if total_weight == 0:
            reasons.append("No valid directional analysis available")
        else:
            reasons.append(
                f"Weighted directional score: {score:+.3f}"
            )

        reasons.append(
            f"Directional agreement: {agreement:.1%}"
        )

        if volatility == VolatilityLevel.HIGH:
            confidence *= 0.75
            reasons.append(
                "Confidence reduced due to high volatility"
            )

        return AIAnalysisResult(
            exchange=candle.exchange,
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            timestamp_ms=candle.timestamp_ms,
            bias=bias,
            score=score,
            confidence=confidence,
            agreement=agreement,
            volatility=volatility,
            regime=regime,
            reports=reports,
            reasons=tuple(reasons),
        )
