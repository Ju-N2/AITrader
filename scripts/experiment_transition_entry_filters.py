import csv
from bisect import bisect_right
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from analyze_market_regimes_warmup_180d import (
    DAY_MS,
    DATA_LIMIT,
    EXCHANGE,
    SYMBOL,
    TIMEFRAMES,
    BULLISH_TRANSITION,
    build_aligned_data,
    build_market_regimes,
    create_config,
    filter_entries_by_regime,
    get_regime_at,
    indicator_ready,
)

from aitrader.backtesting.engine import BacktestEngine
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine
from aitrader.strategy.models import MarketBias, StrategySignal


EVALUATION_DAYS = 120
TRAIN_END_OFFSET_DAYS = 180
VALIDATION_END_OFFSET_DAYS = 60

QUANTILE = 0.75
INITIAL_CAPITAL = 10_000.0

TIMEFRAME_FILTERS = ("15m", "1h")
FEATURES = ("ema", "macd", "atr")

OUTPUT_FILE = Path(
    "data/transition_entry_filter_experiment.csv"
)


def percentile(values, quantile):
    values = sorted(values)

    if not values:
        raise RuntimeError("Cannot calculate percentile from empty data")

    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def load_indicator_series(repository, timeframe, now_ms):
    interval = TIMEFRAMES[timeframe]

    candles = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        timeframe,
        limit=DATA_LIMIT,
    )

    candles = [
        candle
        for candle in candles
        if candle.timestamp_ms + interval <= now_ms
    ]

    snapshots = IndicatorEngine().calculate(candles)

    close_times = []
    valid_snapshots = []

    for candle, snapshot in zip(candles, snapshots):
        if not indicator_ready(snapshot):
            continue

        close_times.append(
            candle.timestamp_ms + interval
        )

        valid_snapshots.append(snapshot)

    print(
        f"{timeframe}: {len(valid_snapshots):,} "
        "completed indicator snapshots"
    )

    return close_times, valid_snapshots


def get_snapshot(series, timeframe, decision_time):
    close_times, snapshots = series[timeframe]

    index = bisect_right(close_times, decision_time) - 1

    if index < 0:
        return None

    signal_time = close_times[index]

    if decision_time - signal_time > TIMEFRAMES[timeframe] * 3:
        return None

    return snapshots[index]


def build_features(candles, signals, regimes, series):
    features = []

    for candle, signal, regime in zip(
        candles,
        signals,
        regimes,
    ):
        row = {}

        if (
            signal.bias == MarketBias.BULLISH
            and regime == BULLISH_TRANSITION
        ):
            decision_time = candle.timestamp_ms + TIMEFRAMES["1m"]
            reference_price = candle.close

            if reference_price <= 0:
                features.append(None)
                continue

            for timeframe in TIMEFRAME_FILTERS:
                snapshot = get_snapshot(
                    series,
                    timeframe,
                    decision_time,
                )

                if snapshot is None:
                    row = {}
                    break

                row[f"{timeframe}_ema"] = (
                    (snapshot.ema_12 - snapshot.ema_26)
                    / reference_price
                    * 100
                )

                row[f"{timeframe}_macd"] = (
                    snapshot.macd_histogram
                    / reference_price
                    * 100
                )

                row[f"{timeframe}_atr"] = (
                    snapshot.atr_14
                    / reference_price
                    * 100
                )

        features.append(row if row else None)

    return features


def build_thresholds(features):
    thresholds = {}

    for timeframe in TIMEFRAME_FILTERS:
        for feature in FEATURES:
            key = f"{timeframe}_{feature}"

            values = [
                row[key]
                for row in features
                if row is not None and key in row
            ]

            thresholds[key] = percentile(
                values,
                QUANTILE,
            )

    return thresholds


def blocked_by_filter(row, filter_name, thresholds):
    if row is None:
        return False

    active_features = {
        "A": ("ema",),
        "B": ("macd",),
        "C": ("atr",),
        "AB": ("ema", "macd"),
        "AC": ("ema", "atr"),
        "BC": ("macd", "atr"),
        "ABC": ("ema", "macd", "atr"),
    }

    for feature in active_features[filter_name]:
        for timeframe in TIMEFRAME_FILTERS:
            key = f"{timeframe}_{feature}"

            if row[key] > thresholds[key]:
                return True

    return False


def apply_filters(signals, features, thresholds, filter_name):
    result = []

    for signal, row in zip(signals, features):
        blocked = (
            signal.bias == MarketBias.BULLISH
            and blocked_by_filter(
                row,
                filter_name,
                thresholds,
            )
        )

        if blocked:
            result.append(
                StrategySignal(
                    exchange=signal.exchange,
                    symbol=signal.symbol,
                    timeframe=signal.timeframe,
                    timestamp_ms=signal.timestamp_ms,
                    bias=MarketBias.NEUTRAL,
                    score=0,
                    max_score=signal.max_score,
                    reasons=signal.reasons + (
                        f"Experiment filter blocked: {filter_name}",
                    ),
                    atr=signal.atr,
                )
            )
        else:
            result.append(signal)

    return result


def diagnose_blocked_entries(candles, signals, features, thresholds):
    """
    BASE 백테스트의 포지션 상태를 재현하여
    A 필터가 차단한 신호의 실제 진입 가능 여부를 분류한다.
    """
    config = create_config()

    counts = {
        "POSITION_OPEN": 0,
        "EXIT_STEP": 0,
        "COOLDOWN": 0,
        "INVALID_ATR": 0,
        "INVALID_STOP": 0,
        "ENTRY_ELIGIBLE": 0,
    }

    position = None
    cooldown_until = -1
    replica_entry_times = set()

    for i in range(len(candles) - 1):
        signal = signals[i]
        next_candle = candles[i + 1]

        exited_this_step = False
        exit_reason = None

        # BASE 엔진의 기존 포지션 청산 로직 재현
        if position is not None:
            if signal.bias == MarketBias.BEARISH:
                exit_reason = "strategy"

            elif position["stop_price"] is not None:
                stop_price = position["stop_price"]

                if next_candle.open <= stop_price:
                    exit_reason = "atr_stop"
                elif next_candle.low <= stop_price:
                    exit_reason = "atr_stop"

            if exit_reason is not None:
                position = None
                exited_this_step = True

                if exit_reason == "atr_stop":
                    cooldown_until = (
                        i + 1 + config.atr_cooldown_bars
                    )

        # 현재 BASE 상태에서 A 필터가 차단하는 신호 분류
        row = features[i]

        blocked = (
            signal.bias == MarketBias.BULLISH
            and blocked_by_filter(row, "A", thresholds)
        )

        if blocked:
            if position is not None:
                category = "POSITION_OPEN"

            elif exited_this_step:
                category = "EXIT_STEP"

            elif i + 1 <= cooldown_until:
                category = "COOLDOWN"

            else:
                atr_value = signal.atr

                valid_atr = (
                    atr_value is not None
                    and atr_value > 0
                    and abs(atr_value) != float("inf")
                )

                if not valid_atr:
                    category = "INVALID_ATR"

                else:
                    entry_price = (
                        next_candle.open
                        * (1 + config.slippage_rate)
                    )

                    stop_price = (
                        entry_price
                        - atr_value * config.atr_stop_multiplier
                    )

                    category = (
                        "ENTRY_ELIGIBLE"
                        if stop_price > 0
                        else "INVALID_STOP"
                    )

            counts[category] += 1

        # BASE 전략의 신규 진입 로직 재현
        if (
            position is None
            and not exited_this_step
            and signal.bias == MarketBias.BULLISH
            and i + 1 > cooldown_until
        ):
            atr_value = signal.atr

            valid_atr = (
                atr_value is not None
                and atr_value > 0
                and abs(atr_value) != float("inf")
            )

            if valid_atr:
                entry_price = (
                    next_candle.open
                    * (1 + config.slippage_rate)
                )

                stop_price = (
                    entry_price
                    - atr_value * config.atr_stop_multiplier
                )

                if stop_price > 0:
                    position = {
                        "stop_price": stop_price,
                    }

                    replica_entry_times.add(
                        next_candle.timestamp_ms
                    )

                    # 진입한 캔들 안에서 즉시 손절되는 경우
                    if next_candle.low <= stop_price:
                        position = None
                        cooldown_until = (
                            i + 1 + config.atr_cooldown_bars
                        )

    print("\\n========== A FILTER BLOCKED ENTRY DIAGNOSIS ==========")

    total = sum(counts.values())

    for category, count in counts.items():
        percentage = count / total * 100 if total else 0

        print(
            f"{category:<18}: {count:>3} / {total:>3} "
            f"({percentage:>6.2f}%)"
        )

    print(f"{'TOTAL':<18}: {total:>3}")

    return replica_entry_times


def calculate_metrics(result):
    gross_profit = sum(
        trade.pnl
        for trade in result.trades
        if trade.pnl > 0
    )

    gross_loss = abs(sum(
        trade.pnl
        for trade in result.trades
        if trade.pnl < 0
    ))

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else None
    )

    costs = (
        result.total_fees
        + result.total_slippage_cost
    )

    return {
        "return_pct": result.total_return_pct,
        "mdd_pct": result.max_drawdown_pct,
        "trades": result.total_trades,
        "win_rate_pct": result.win_rate_pct,
        "profit_factor": pf,
        "atr_stops": result.stop_loss_count,
        "costs": costs,
        "pnl": sum(trade.pnl for trade in result.trades),
    }


def run_experiments(label, candles, signals, features, thresholds):
    names = ("BASE", "A", "B", "C", "AB", "AC", "BC", "ABC")
    rows = []
    entry_times = {}

    print(f"\n========== {label} ==========")

    for name in names:
        if name == "BASE":
            test_signals = signals
        else:
            test_signals = apply_filters(
                signals,
                features,
                thresholds,
                name,
            )

        result = BacktestEngine().run(
            candles,
            test_signals,
            create_config(),
        )

        metrics = calculate_metrics(result)

        entry_times[name] = {
            trade.entry_timestamp_ms
            for trade in result.trades
        }

        row = {
            "period": label,
            "experiment": name,
            **metrics,
        }

        rows.append(row)

        pf = (
            f"{metrics['profit_factor']:.3f}"
            if metrics["profit_factor"] is not None
            else "N/A"
        )

        print(
            f"{name:<6} "
            f"Return={metrics['return_pct']:>8.2f}% | "
            f"MDD={metrics['mdd_pct']:>7.2f}% | "
            f"Trades={metrics['trades']:>4} | "
            f"Win={metrics['win_rate_pct']:>6.2f}% | "
            f"PF={pf:>7} | "
            f"ATR Stops={metrics['atr_stops']:>4}"
        )

    if label == "FINAL_TEST":
        base_entries = entry_times["BASE"]
        filtered_entries = entry_times["A"]

        common = base_entries & filtered_entries
        removed = base_entries - filtered_entries
        added = filtered_entries - base_entries

        print("\\n========== BASE vs A ENTRY COMPARISON ==========")
        print(f"BASE entries: {len(base_entries)}")
        print(f"A entries: {len(filtered_entries)}")
        print(f"Common entries: {len(common)}")
        print(f"Removed entries: {len(removed)}")
        print(f"New entries: {len(added)}")

        for title, timestamps in (
            ("REMOVED", removed),
            ("NEW", added),
        ):
            print(f"\\n{title} ENTRY TIMESTAMPS")

            for timestamp in sorted(timestamps):
                print(
                    datetime.fromtimestamp(
                        timestamp / 1000,
                        UTC,
                    ).isoformat()
                )

    return rows


def main():
    print("\n========== AITrader ENTRY FILTER EXPERIMENT ==========")
    print(f"Exchange: {EXCHANGE}")
    print(f"Symbol: {SYMBOL}")
    print(f"Training quantile: {QUANTILE:.0%}")

    repository = SQLiteCandleRepository("data/aitrader.db")

    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    regime_times, _, detailed_regimes = build_market_regimes(
        repository,
        now_ms,
    )

    full_candles, full_signals = build_aligned_data(
        repository,
        now_ms,
    )

    if not full_candles:
        raise RuntimeError("No aligned candles available")

    data_end_ms = (
        full_candles[-1].timestamp_ms
        + TIMEFRAMES["1m"]
    )

    train_end = (
        data_end_ms
        - TRAIN_END_OFFSET_DAYS * DAY_MS
    )

    train_start = train_end - EVALUATION_DAYS * DAY_MS

    validation_end = (
        data_end_ms
        - VALIDATION_END_OFFSET_DAYS * DAY_MS
    )

    validation_start = (
        validation_end - EVALUATION_DAYS * DAY_MS
    )

    test_end = data_end_ms
    test_start = test_end - 60 * DAY_MS

    def select_period(start, end, days=EVALUATION_DAYS):
        selected = [
            index
            for index, candle in enumerate(full_candles)
            if start <= candle.timestamp_ms < end
        ]

        candles = [full_candles[i] for i in selected]
        signals = [full_signals[i] for i in selected]

        expected = days * 24 * 60

        if len(candles) != expected:
            raise RuntimeError(
                f"Unexpected candle count: "
                f"{len(candles)} != {expected}"
            )

        regimes = [
            get_regime_at(
                candle.timestamp_ms + TIMEFRAMES["1m"],
                regime_times,
                detailed_regimes,
            )
            for candle in candles
        ]

        if any(
            regime in ("warmup", "unavailable")
            for regime in regimes
        ):
            raise RuntimeError(
                "Incomplete market regime data"
            )

        return candles, signals, regimes

    train = select_period(train_start, train_end)
    validation = select_period(validation_start, validation_end)
    test = select_period(test_start, test_end, days=60)

    print("\n========== PERIODS ==========")
    print(
        "Training:",
        datetime.fromtimestamp(train_start / 1000, UTC),
        "to",
        datetime.fromtimestamp(train_end / 1000, UTC),
    )
    print(
        "Validation:",
        datetime.fromtimestamp(validation_start / 1000, UTC),
        "to",
        datetime.fromtimestamp(validation_end / 1000, UTC),
    )
    print(
        "Final Test:",
        datetime.fromtimestamp(test_start / 1000, UTC),
        "to",
        datetime.fromtimestamp(test_end / 1000, UTC),
    )

    print("\n========== LOADING INDICATORS ==========")

    series = {
        timeframe: load_indicator_series(
            repository,
            timeframe,
            now_ms,
        )
        for timeframe in TIMEFRAME_FILTERS
    }

    train_features = build_features(
        train[0],
        train[1],
        train[2],
        series,
    )

    validation_features = build_features(
        validation[0],
        validation[1],
        validation[2],
        series,
    )

    test_features = build_features(
        test[0],
        test[1],
        test[2],
        series,
    )

    # 모든 실험에 동일한 Bullish Transition 진입 조건 적용
    train_base_signals = filter_entries_by_regime(
        train[1], train[2], BULLISH_TRANSITION
    )
    validation_base_signals = filter_entries_by_regime(
        validation[1], validation[2], BULLISH_TRANSITION
    )
    test_base_signals = filter_entries_by_regime(
        test[1], test[2], BULLISH_TRANSITION
    )

    thresholds = build_thresholds(train_features)

    print("\n========== TRAINING THRESHOLDS ==========")

    for key, value in sorted(thresholds.items()):
        print(f"{key:<15}: {value:.6f}%")

    print("\n========== FINAL TEST FILTER DIAGNOSTICS ==========")

    valid_rows = [
        row for row in test_features
        if row is not None
    ]

    print(f"Valid test candidates: {len(valid_rows)}")

    for key, threshold in sorted(thresholds.items()):
        exceeded = sum(
            row[key] > threshold
            for row in valid_rows
        )

        print(
            f"{key:<15}: "
            f"Exceeded={exceeded:>3} / {len(valid_rows):>3} | "
            f"Threshold={threshold:.6f}%"
        )

    for name in ("A", "B", "C", "AB", "AC", "BC", "ABC"):
        blocked = sum(
            blocked_by_filter(row, name, thresholds)
            for row in valid_rows
        )

        print(
            f"Filter {name:<4}: "
            f"Blocked={blocked:>3} / {len(valid_rows):>3}"
        )

    replica_entry_times = diagnose_blocked_entries(
        test[0],
        test_base_signals,
        test_features,
        thresholds,
    )

    train_rows = run_experiments(
        "TRAIN",
        train[0],
        train_base_signals,
        train_features,
        thresholds,
    )

    validation_rows = run_experiments(
        "VALIDATION",
        validation[0],
        validation_base_signals,
        validation_features,
        thresholds,
    )

    test_rows = run_experiments(
        "FINAL_TEST",
        test[0],
        test_base_signals,
        test_features,
        thresholds,
    )

    actual_result = BacktestEngine().run(
        test[0],
        test_base_signals,
        create_config(),
    )

    actual_entry_times = {
        trade.entry_timestamp_ms
        for trade in actual_result.trades
    }

    common = replica_entry_times & actual_entry_times
    missing = actual_entry_times - replica_entry_times
    extra = replica_entry_times - actual_entry_times

    print("\\n========== ENGINE vs DIAGNOSTIC COMPARISON ==========")
    print(f"Actual engine entries : {len(actual_entry_times)}")
    print(f"Diagnostic entries    : {len(replica_entry_times)}")
    print(f"Matching entries      : {len(common)}")
    print(f"Missing entries       : {len(missing)}")
    print(f"Extra entries         : {len(extra)}")

    if missing:
        print("Missing timestamps:", sorted(missing))

    if extra:
        print("Extra timestamps:", sorted(extra))

    print(
        "[PASS] Entry timestamps match."
        if not missing and not extra
        else "[FAIL] Entry timestamps do not match."
    )

    all_rows = train_rows + validation_rows + test_rows

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print("\n========== EXPERIMENT COMPLETED ==========")
    print(f"Results: {OUTPUT_FILE}")
    print(f"Training candidates: {sum(x is not None for x in train_features)}")
    print(
        "Validation candidates:",
        sum(x is not None for x in validation_features),
    )
    print(
        "Final test candidates:",
        sum(x is not None for x in test_features),
    )


if __name__ == "__main__":
    main()
