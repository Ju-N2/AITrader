from dataclasses import replace
from math import isfinite

from aitrader.ai.models import (
    AIAnalysisResult,
    MarketRegime,
    VolatilityLevel,
)
from aitrader.risk.engine import (
    MarketType,
    RiskDecision,
)
from aitrader.risk.manager import (
    AccountRiskState,
    RiskManager,
)
from aitrader.strategy.models import (
    MarketBias,
    StrategySignal,
)
from aitrader.strategy.multi_timeframe import (
    MultiTimeframeResult,
)


class AITradingGate:
    """
    Validates AI analysis before forwarding an entry to RiskManager.

    AI analysis is advisory.
    RiskManager remains the final risk authority.
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        min_confidence: float = 0.70,
        min_agreement: float = 0.60,
        allow_high_volatility: bool = False,
    ):
        for name, value in (
            ("min_confidence", min_confidence),
            ("min_agreement", min_agreement),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"Invalid {name}")

        self.risk_manager = risk_manager
        self.min_confidence = float(min_confidence)
        self.min_agreement = float(min_agreement)
        self.allow_high_volatility = allow_high_volatility

    @staticmethod
    def _reject(reason: str) -> RiskDecision:
        return RiskDecision(
            approved=False,
            rejection_reason=reason,
        )

    def evaluate(
        self,
        analysis: AIAnalysisResult,
        consensus: MultiTimeframeResult,
        execution_signal: StrategySignal,
        entry_price: float,
        account: AccountRiskState,
        market_type: MarketType | str,
    ) -> RiskDecision:

        if not isinstance(analysis, AIAnalysisResult):
            return self._reject("Invalid AI analysis result")

        if not isinstance(consensus, MultiTimeframeResult):
            return self._reject("Invalid market consensus")

        if not isinstance(execution_signal, StrategySignal):
            return self._reject("Invalid execution signal")

        # 1. All inputs must describe the same market.
        if not (
            analysis.exchange
            == consensus.exchange
            == execution_signal.exchange
        ):
            return self._reject("AI analysis exchange mismatch")

        if not (
            analysis.symbol
            == consensus.symbol
            == execution_signal.symbol
        ):
            return self._reject("AI analysis symbol mismatch")

        # 2. AI result must match the execution candle.
        if analysis.timeframe != execution_signal.timeframe:
            return self._reject("AI analysis timeframe mismatch")

        if analysis.timestamp_ms != execution_signal.timestamp_ms:
            return self._reject("AI analysis timestamp mismatch")

        # 3. Only directional AI decisions are eligible.
        if analysis.bias not in (
            MarketBias.BULLISH,
            MarketBias.BEARISH,
        ):
            return self._reject("AI analysis is not directional")

        # 4. AI, MTF and execution signal must agree.
        if analysis.bias != consensus.bias:
            return self._reject("AI analysis conflicts with consensus")

        if analysis.bias != execution_signal.bias:
            return self._reject("AI analysis conflicts with execution signal")

        # 5. Require sufficient confidence and agreement.
        if analysis.confidence < self.min_confidence:
            return self._reject("AI confidence below minimum threshold")

        if analysis.agreement < self.min_agreement:
            return self._reject("AI agreement below minimum threshold")

        # 6. Unknown context is rejected.
        if analysis.volatility == VolatilityLevel.UNKNOWN:
            return self._reject("AI volatility data unavailable")

        if analysis.regime == MarketRegime.UNKNOWN:
            return self._reject("AI market regime unavailable")

        # 7. High volatility protection.
        if (
            analysis.volatility == VolatilityLevel.HIGH
            and not self.allow_high_volatility
        ):
            return self._reject("High volatility entry blocked by AI gate")

        # 8. Contradictory market regime protection.
        if (
            analysis.bias == MarketBias.BULLISH
            and analysis.regime == MarketRegime.BEAR_TREND
        ):
            return self._reject("Bullish signal conflicts with bearish regime")

        if (
            analysis.bias == MarketBias.BEARISH
            and analysis.regime == MarketRegime.BULL_TREND
        ):
            return self._reject("Bearish signal conflicts with bullish regime")

        # 9. Delegate final approval to RiskManager.
        decision = self.risk_manager.evaluate(
            consensus=consensus,
            execution_signal=execution_signal,
            entry_price=entry_price,
            account=account,
            market_type=market_type,
        )

        # Preserve AI evaluation context for diagnostics.
        warnings = decision.warnings + (
            f"AI score: {analysis.score:+.3f}",
            f"AI confidence: {analysis.confidence:.1%}",
            f"AI agreement: {analysis.agreement:.1%}",
            f"AI regime: {analysis.regime.value}",
        )

        return replace(
            decision,
            warnings=warnings,
        )
