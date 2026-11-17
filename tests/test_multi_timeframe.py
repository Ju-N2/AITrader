import pytest

from aitrader.core.exceptions import StrategyError
from aitrader.strategy.models import MarketBias, StrategySignal
from aitrader.strategy.multi_timeframe import MultiTimeframeEngine


def create_signal(timeframe, score, exchange="binance_spot"):
    return StrategySignal(
        exchange=exchange,
        symbol="BTC/USDT",
        timeframe=timeframe,
        timestamp_ms=1000,
        bias=(
            MarketBias.BULLISH if score > 0
            else MarketBias.BEARISH if score < 0
            else MarketBias.NEUTRAL
        ),
        score=score,
        max_score=5,
        reasons=("test",),
    )


def bullish_signals():
    return {
        timeframe: create_signal(timeframe, 5)
        for timeframe in ["1m", "5m", "15m", "1h", "4h"]
    }


def test_all_bullish():
    engine = MultiTimeframeEngine()

    result = engine.evaluate(bullish_signals())

    assert result.bias == MarketBias.BULLISH
    assert result.weighted_score == 5.0


def test_all_bearish():
    engine = MultiTimeframeEngine()

    signals = {
        timeframe: create_signal(timeframe, -5)
        for timeframe in ["1m", "5m", "15m", "1h", "4h"]
    }

    result = engine.evaluate(signals)

    assert result.bias == MarketBias.BEARISH
    assert result.weighted_score == -5.0


def test_mixed_signals():
    engine = MultiTimeframeEngine()

    signals = {
        "1m": create_signal("1m", -5),
        "5m": create_signal("5m", -5),
        "15m": create_signal("15m", 1),
        "1h": create_signal("1h", 2),
        "4h": create_signal("4h", 3),
    }

    result = engine.evaluate(signals)

    assert result.bias == MarketBias.NEUTRAL


def test_missing_lower_timeframe():
    engine = MultiTimeframeEngine()

    signals = {
        "5m": create_signal("5m", 5),
        "15m": create_signal("15m", 5),
        "1h": create_signal("1h", 5),
        "4h": create_signal("4h", 5),
    }

    result = engine.evaluate(signals)

    assert result.bias == MarketBias.BULLISH
    assert "1m" in result.missing_timeframes


def test_missing_core_timeframe():
    engine = MultiTimeframeEngine()

    signals = {
        "1m": create_signal("1m", 5),
        "5m": create_signal("5m", 5),
        "15m": create_signal("15m", 5),
        "1h": create_signal("1h", 5),
    }

    result = engine.evaluate(signals)

    assert result.bias == MarketBias.NEUTRAL
    assert "4h" in result.missing_timeframes


def test_empty_signals():
    engine = MultiTimeframeEngine()

    with pytest.raises(StrategyError):
        engine.evaluate({})


def test_exchange_mismatch():
    engine = MultiTimeframeEngine()

    signals = bullish_signals()
    signals["4h"] = create_signal(
        "4h", 5, exchange="upbit"
    )

    with pytest.raises(StrategyError):
        engine.evaluate(signals)


def test_invalid_score():
    engine = MultiTimeframeEngine()

    signals = bullish_signals()
    signals["1m"] = create_signal("1m", 10)

    with pytest.raises(StrategyError):
        engine.evaluate(signals)


def test_invalid_threshold():
    with pytest.raises(ValueError):
        MultiTimeframeEngine(bullish_threshold=-1)

def test_4h_neutral_blocks_bullish_consensus():
    engine = MultiTimeframeEngine()

    signals = bullish_signals()
    signals["4h"] = create_signal("4h", 0)

    result = engine.evaluate(signals)

    assert result.bias == MarketBias.NEUTRAL
    assert "4h trend filter blocked long entry" in result.reasons


def test_1h_bearish_forces_bearish():
    engine = MultiTimeframeEngine()

    signals = bullish_signals()
    signals["1h"] = create_signal("1h", -5)

    result = engine.evaluate(signals)

    assert result.bias == MarketBias.BEARISH


def test_15m_confirmation_required():
    engine = MultiTimeframeEngine()

    signals = bullish_signals()
    signals["15m"] = create_signal("15m", 0)

    result = engine.evaluate(signals)

    assert result.bias == MarketBias.NEUTRAL
    assert "15m confirmation missing" in result.reasons

def test_1m_bearish_blocks_bullish_consensus():
    engine = MultiTimeframeEngine()

    signals = bullish_signals()
    signals["1m"] = create_signal("1m", -5)

    result = engine.evaluate(signals)

    assert result.bias == MarketBias.NEUTRAL
    assert "1m entry confirmation missing" in result.reasons


def test_1m_neutral_blocks_bullish_consensus():
    engine = MultiTimeframeEngine()

    signals = bullish_signals()
    signals["1m"] = create_signal("1m", 0)

    result = engine.evaluate(signals)

    assert result.bias == MarketBias.NEUTRAL
    assert "1m entry confirmation missing" in result.reasons
