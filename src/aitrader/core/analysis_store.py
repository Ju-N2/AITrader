import json
import os
import tempfile

from datetime import UTC, datetime
from pathlib import Path


class AIAnalysisStore:
    """Persist the latest AI analysis as an atomic JSON snapshot."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _serialize_report(report):
        return {
            "agent_name": report.agent_name,
            "bias": report.bias.value,
            "score": report.score,
            "confidence": report.confidence,
            "reasons": list(report.reasons),
            "volatility": report.volatility.value,
            "regime": report.regime.value,
            "metrics": {
                name: value
                for name, value in report.metrics
            },
        }

    def save(self, analysis):
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "exchange": analysis.exchange,
            "symbol": analysis.symbol,
            "timeframe": analysis.timeframe,
            "timestamp_ms": analysis.timestamp_ms,
            "bias": analysis.bias.value,
            "score": analysis.score,
            "confidence": analysis.confidence,
            "agreement": analysis.agreement,
            "volatility": analysis.volatility.value,
            "regime": analysis.regime.value,
            "reasons": list(analysis.reasons),
            "reports": [
                self._serialize_report(report)
                for report in analysis.reports
            ],
        }

        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=".latest_ai_analysis_",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)

                json.dump(
                    payload,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )

                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary_path, self.path)

        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
