import json

from aitrader.ai import AIAnalysisEngine
from aitrader.core.analysis_store import AIAnalysisStore
from aitrader.data.models import Candle
from aitrader.indicators.models import IndicatorSnapshot


def test_analysis_store_saves_complete_result(tmp_path):
    candle = Candle(
        exchange="binance_spot",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1700000000000,
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
        volume=10.0,
    )

    indicators = IndicatorSnapshot(
        timestamp_ms=candle.timestamp_ms,
        sma_20=98.0,
        sma_50=95.0,
        ema_12=99.0,
        ema_26=97.0,
        rsi_14=60.0,
        macd=2.0,
        macd_signal=1.0,
        macd_histogram=1.0,
        bollinger_upper=105.0,
        bollinger_middle=100.0,
        bollinger_lower=95.0,
        atr_14=1.0,
    )

    analysis = AIAnalysisEngine().analyze(candle, indicators)

    path = tmp_path / "latest_ai_analysis.json"
    store = AIAnalysisStore(path)
    store.save(analysis)

    result = json.loads(path.read_text(encoding="utf-8"))

    assert result["symbol"] == "BTC/USDT"
    assert result["exchange"] == "binance_spot"
    assert result["bias"] == analysis.bias.value
    assert result["score"] == analysis.score
    assert result["confidence"] == analysis.confidence
    assert result["agreement"] == analysis.agreement
    assert len(result["reports"]) == 4
    assert result["generated_at"]
