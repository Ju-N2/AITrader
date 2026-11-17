import pytest

from aitrader.data.models import Candle
from aitrader.strategy.models import StrategySignal, MarketBias

from aitrader.backtesting.models import BacktestConfig
from aitrader.backtesting.engine import BacktestEngine


def make_candle(timestamp, open_price, close_price):
    return Candle(
        exchange="binance_spot",
        symbol="BTC/USDT",
        timeframe="1m",
        timestamp_ms=timestamp,
        open=open_price,
        high=max(open_price, close_price) + 1,
        low=min(open_price, close_price) - 1,
        close=close_price,
        volume=10,
    )


def make_signal(timestamp, bias):
    return StrategySignal(
        exchange="binance_spot",
        symbol="BTC/USDT",
        timeframe="1m",
        timestamp_ms=timestamp,
        bias=bias,
        score=5 if bias == MarketBias.BULLISH else -5,
        max_score=5,
        reasons=("test",),
    )


def test_profitable_long_trade():

    candles = [
        make_candle(1000, 100, 100),
        make_candle(2000, 100, 110),
        make_candle(3000, 110, 115),
        make_candle(4000, 120, 120),
    ]

    signals = [
        make_signal(1000, MarketBias.BULLISH),
        make_signal(2000, MarketBias.NEUTRAL),
        make_signal(3000, MarketBias.BEARISH),
        make_signal(4000, MarketBias.NEUTRAL),
    ]

    config = BacktestConfig(
        initial_capital=1000,
        fee_rate=0,
        slippage_rate=0,
        position_fraction=1,
    )

    result = BacktestEngine().run(candles, signals, config)

    assert result.total_trades == 1
    assert result.final_equity == 1200
    assert result.total_return_pct == pytest.approx(20.0)
    assert result.win_rate_pct == 100


def test_fees_and_slippage_reduce_profit():

    candles = [
        make_candle(1000, 100, 100),
        make_candle(2000, 100, 110),
        make_candle(3000, 110, 115),
        make_candle(4000, 120, 120),
    ]

    signals = [
        make_signal(1000, MarketBias.BULLISH),
        make_signal(2000, MarketBias.NEUTRAL),
        make_signal(3000, MarketBias.BEARISH),
        make_signal(4000, MarketBias.NEUTRAL),
    ]

    config = BacktestConfig(
        initial_capital=1000,
        fee_rate=0.001,
        slippage_rate=0.001,
        position_fraction=1,
    )

    result = BacktestEngine().run(candles, signals, config)

    assert result.total_fees > 0
    assert result.final_equity < 1200


def test_neutral_market():

    candles = [
        make_candle(1000, 100, 100),
        make_candle(2000, 110, 110),
    ]

    signals = [
        make_signal(1000, MarketBias.NEUTRAL),
        make_signal(2000, MarketBias.NEUTRAL),
    ]

    result = BacktestEngine().run(candles, signals)

    assert result.total_trades == 0
    assert result.final_equity == result.initial_capital


def test_close_position_at_end():

    candles = [
        make_candle(1000, 100, 100),
        make_candle(2000, 100, 110),
    ]

    signals = [
        make_signal(1000, MarketBias.BULLISH),
        make_signal(2000, MarketBias.NEUTRAL),
    ]

    config = BacktestConfig(
        initial_capital=1000,
        fee_rate=0,
        slippage_rate=0,
        position_fraction=1,
    )

    result = BacktestEngine().run(candles, signals, config)

    assert result.total_trades == 1
    assert result.final_equity == 1100


def test_signal_count_mismatch():

    candles = [
        make_candle(1000, 100, 100),
        make_candle(2000, 100, 100),
    ]

    signals = [
        make_signal(1000, MarketBias.NEUTRAL),
    ]

    with pytest.raises(Exception):
        BacktestEngine().run(candles, signals)


def test_invalid_config():

    with pytest.raises(ValueError):
        BacktestConfig(fee_rate=2)


def test_atr_cooldown_blocks_reentry():
    candles = []

    for i in range(20):
        low = 97 if i == 1 else 99

        candles.append(
            Candle(
                exchange="binance_spot",
                symbol="BTC/USDT",
                timeframe="1m",
                timestamp_ms=1000 + i * 60000,
                open=100,
                high=101,
                low=low,
                close=100,
                volume=10,
            )
        )

    signals = [
        StrategySignal(
            exchange="binance_spot",
            symbol="BTC/USDT",
            timeframe="1m",
            timestamp_ms=candle.timestamp_ms,
            bias=MarketBias.BULLISH,
            score=5,
            max_score=5,
            reasons=("test",),
            atr=1.0,
        )
        for candle in candles
    ]

    config = BacktestConfig(
        initial_capital=1000,
        fee_rate=0,
        slippage_rate=0,
        position_fraction=0.25,
        atr_stop_multiplier=2.0,
        atr_cooldown_bars=15,
    )

    result = BacktestEngine().run(
        candles,
        signals,
        config,
    )

    assert result.stop_loss_count == 1
    assert result.cooldown_blocked_entries == 15
    assert result.total_trades == 2

    # 두 번째 거래는 손절 후 15개 캔들이 지난 다음 시작
    assert result.trades[1].entry_timestamp_ms == candles[17].timestamp_ms
