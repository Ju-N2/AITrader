from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone

from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine
from aitrader.strategy.technical import TechnicalTrendStrategy
from aitrader.strategy.multi_timeframe import MultiTimeframeEngine
from aitrader.strategy.models import StrategySignal, MarketBias
from aitrader.backtesting.engine import BacktestEngine
from aitrader.backtesting.models import BacktestConfig


EXCHANGE = "binance_spot"
SYMBOL = "BTC/USDT"
DATABASE = "data/aitrader.db"

TIMEFRAMES = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}

INITIAL_CAPITAL = 10_000.0
HORIZONS = {
    "15m": 15,
    "1h": 60,
    "4h": 240,
}


def is_ready(snapshot):
    return all(value is not None for value in [
        snapshot.sma_20,
        snapshot.sma_50,
        snapshot.ema_12,
        snapshot.ema_26,
        snapshot.rsi_14,
        snapshot.macd_histogram,
        snapshot.atr_14,
    ])


def build_timeframe_data(repository, timeframe, interval_ms, now_ms):
    candles = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        timeframe,
        limit=200000,
    )

    candles = [
        candle for candle in candles
        if candle.timestamp_ms + interval_ms <= now_ms
    ]

    if len(candles) < 50:
        raise RuntimeError(
            f"Insufficient data: {timeframe} ({len(candles)})"
        )

    snapshots = IndicatorEngine().calculate(candles)
    strategy = TechnicalTrendStrategy()

    close_times = []
    signals = []
    valid_snapshots = []

    for candle, snapshot in zip(candles, snapshots):
        if not is_ready(snapshot):
            continue

        signal = strategy.evaluate(candle, snapshot)

        close_times.append(candle.timestamp_ms + interval_ms)
        signals.append(signal)
        valid_snapshots.append(snapshot)

    return {
        "close_times": close_times,
        "signals": signals,
        "snapshots": valid_snapshots,
    }


def latest_index(data, timestamp_ms):
    index = bisect_right(
        data["close_times"],
        timestamp_ms,
    ) - 1

    if index < 0:
        return None

    return index


def build_mtf_inputs(repository, now_ms):
    timeframe_data = {}

    print("\n========== BUILDING TIMEFRAME DATA ==========")

    for timeframe, interval_ms in TIMEFRAMES.items():
        data = build_timeframe_data(
            repository,
            timeframe,
            interval_ms,
            now_ms,
        )

        timeframe_data[timeframe] = data

        print(
            f"{timeframe}: "
            f"{len(data['signals']):,} completed signals"
        )

    candles_1m = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        "1m",
        limit=200000,
    )

    candles_1m = [
        candle for candle in candles_1m
        if candle.timestamp_ms + 60_000 <= now_ms
    ]

    multi_engine = MultiTimeframeEngine(min_timeframes=5)

    aligned_candles = []
    aligned_signals = []
    unavailable = 0

    for candle in candles_1m:
        decision_time = candle.timestamp_ms + 60_000
        current_signals = {}

        for timeframe, interval_ms in TIMEFRAMES.items():
            data = timeframe_data[timeframe]
            index = latest_index(data, decision_time)

            if index is None:
                break

            signal_time = data["close_times"][index]

            if decision_time - signal_time > interval_ms * 3:
                break

            current_signals[timeframe] = data["signals"][index]

        if len(current_signals) != len(TIMEFRAMES):
            unavailable += 1

            combined = StrategySignal(
                exchange=EXCHANGE,
                symbol=SYMBOL,
                timeframe="1m",
                timestamp_ms=candle.timestamp_ms,
                bias=MarketBias.NEUTRAL,
                score=0,
                max_score=5,
                reasons=("Insufficient timeframe data",),
                atr=(
                    current_signals["15m"].atr
                    if "15m" in current_signals else None
                ),
            )
        else:
            result = multi_engine.evaluate(current_signals)

            combined = StrategySignal(
                exchange=EXCHANGE,
                symbol=SYMBOL,
                timeframe="1m",
                timestamp_ms=candle.timestamp_ms,
                bias=result.bias,
                score=max(-5, min(5, round(result.weighted_score))),
                max_score=5,
                reasons=result.reasons,
                atr=current_signals["15m"].atr,
            )

        aligned_candles.append(candle)
        aligned_signals.append(combined)

    print(f"Aligned candles: {len(aligned_candles):,}")
    print(f"Unavailable decisions: {unavailable}")

    return (
        timeframe_data,
        aligned_candles,
        aligned_signals,
    )


def fmt(value, digits=3):
    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


def analyze_future(candles, timestamps, exit_index, exit_price):
    results = {}

    for label, minutes in HORIZONS.items():
        target_time = timestamps[exit_index] + minutes * 60_000

        start = bisect_right(
            timestamps,
            timestamps[exit_index],
        )

        end = bisect_right(timestamps, target_time)

        future = candles[start:end]

        if not future:
            results[label] = None
            continue

        target_candle_index = bisect_right(
            timestamps,
            target_time,
        ) - 1

        if target_candle_index <= exit_index:
            results[label] = None
            continue

        target_close = candles[target_candle_index].close

        results[label] = {
            "close_return": (target_close / exit_price - 1) * 100,
            "max_up": (max(c.high for c in future) / exit_price - 1) * 100,
            "max_down": (min(c.low for c in future) / exit_price - 1) * 100,
        }

    return results


def main():
    repository = SQLiteCandleRepository(DATABASE)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    timeframe_data, candles, signals = build_mtf_inputs(
        repository,
        now_ms,
    )

    config = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        fee_rate=0.001,
        slippage_rate=0.0005,
        position_fraction=0.25,
        atr_stop_multiplier=2.0,
        atr_cooldown_bars=15,
    )

    result = BacktestEngine().run(
        candles,
        signals,
        config,
    )

    timestamps = [c.timestamp_ms for c in candles]

    stops = [
        trade for trade in result.trades
        if trade.exit_reason == "atr_stop"
    ]

    print("\n========== REPRODUCED BACKTEST ==========")
    print(f"Candles: {len(candles):,}")
    print(f"Total trades: {result.total_trades}")
    print(f"ATR stops: {len(stops)}")
    print(f"Final equity: {result.final_equity:.2f} USDT")
    print(f"Return: {result.total_return_pct:+.2f}%")

    if not stops:
        print("[INFO] No ATR stop trades found.")
        return

    bias_counts = Counter()

    post_returns = {
        label: []
        for label in HORIZONS
    }

    for number, trade in enumerate(stops, 1):
        entry_signal_time = trade.entry_timestamp_ms - 60_000

        entry_index = bisect_right(
            timestamps,
            entry_signal_time,
        ) - 1

        exit_index = bisect_right(
            timestamps,
            trade.exit_timestamp_ms,
        ) - 1

        if entry_index < 0 or exit_index < 0:
            continue

        entry_time = datetime.fromtimestamp(
            trade.entry_timestamp_ms / 1000,
            timezone.utc,
        )

        exit_time = datetime.fromtimestamp(
            trade.exit_timestamp_ms / 1000,
            timezone.utc,
        )

        holding = (
            trade.exit_timestamp_ms - trade.entry_timestamp_ms
        ) / 60_000

        entry_signal = signals[entry_index]
        bias_counts[entry_signal.bias.value] += 1

        print(f"\n{'=' * 55}")
        print(f"ATR STOP #{number}")
        print(f"Entry: {entry_time}")
        print(f"Exit:  {exit_time}")
        print(f"Holding: {holding:.1f} minutes")
        print(f"Entry price: {trade.entry_price:.2f}")
        print(f"Exit price:  {trade.exit_price:.2f}")
        print(f"PnL: {trade.pnl:+.2f} USDT")
        print(f"Trade return: {trade.return_pct:+.3f}%")

        print("\n--- Entry MTF Decision ---")
        print(f"Bias: {entry_signal.bias.value}")
        print(f"Score: {entry_signal.score:+d}/{entry_signal.max_score}")

        for reason in entry_signal.reasons:
            print(f"- {reason}")

        print("\n--- Entry Indicators ---")

        for timeframe in TIMEFRAMES:
            data = timeframe_data[timeframe]
            index = latest_index(data, trade.entry_timestamp_ms)

            if index is None:
                print(f"{timeframe}: unavailable")
                continue

            snapshot = data["snapshots"][index]
            signal = data["signals"][index]

            print(
                f"{timeframe}: "
                f"Bias={signal.bias.value}, "
                f"Score={signal.score:+d}, "
                f"RSI={fmt(snapshot.rsi_14)}, "
                f"MACD_H={fmt(snapshot.macd_histogram)}, "
                f"ATR={fmt(snapshot.atr_14)}"
            )

        print("\n--- Price After Stop ---")

        future = analyze_future(
            candles,
            timestamps,
            exit_index,
            trade.exit_price,
        )

        for label, values in future.items():
            if values is None:
                print(f"{label}: insufficient future data")
                continue

            post_returns[label].append(values["close_return"])

            print(
                f"{label}: "
                f"Close={values['close_return']:+.3f}%, "
                f"Max Up={values['max_up']:+.3f}%, "
                f"Max Down={values['max_down']:+.3f}%"
            )

    print("\n========== STOP ANALYSIS SUMMARY ==========")
    print(f"Reproduced ATR stops: {len(stops)}")
    print(f"Entry bias distribution: {dict(bias_counts)}")

    for label, values in post_returns.items():
        if not values:
            continue

        positive = sum(value > 0 for value in values)

        print(
            f"{label}: "
            f"Positive closes={positive}/{len(values)}, "
            f"Average return={sum(values) / len(values):+.3f}%"
        )

    print("\n========== ANALYSIS COMPLETED ==========")


if __name__ == "__main__":
    main()
