import csv
from datetime import UTC, datetime
from pathlib import Path

from analyze_market_regimes_warmup_180d import (
    DAY_MS,
    DATA_LIMIT,
    EXCHANGE,
    SYMBOL,
    TIMEFRAMES,
    BULLISH_TRANSITION,
    build_aligned_data,
    build_market_regimes,
    get_regime_at,
    indicator_ready,
)

from experiment_transition_entry_filters import (
    EVALUATION_DAYS,
    TRAIN_END_OFFSET_DAYS,
    VALIDATION_END_OFFSET_DAYS,
    TIMEFRAME_FILTERS,
    FEATURES,
    QUANTILE,
    percentile,
    load_indicator_series,
    build_features,
    build_thresholds,
)

from aitrader.data.storage import SQLiteCandleRepository


OUTPUT_FILE = Path(
    "data/transition_feature_distribution.csv"
)

FEATURE_KEYS = [
    f"{timeframe}_{feature}"
    for timeframe in TIMEFRAME_FILTERS
    for feature in FEATURES
]

TARGET_ENTRY_TIME = datetime(
    2026, 11, 5, 0, 15, tzinfo=UTC
)

PERCENTILES = (0.25, 0.50, 0.75, 0.90, 0.95)


def format_time(timestamp_ms):
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        UTC,
    ).isoformat()


def get_valid_rows(features):
    return [
        (index, row)
        for index, row in enumerate(features)
        if row is not None
    ]


def print_distribution(label, rows, thresholds):
    print(f"\n========== {label} DISTRIBUTION ==========")

    print(f"Valid candidates: {len(rows)}")

    for key in FEATURE_KEYS:
        values = sorted(row[key] for _, row in rows)

        if not values:
            print(f"{key}: NO DATA")
            continue

        print(f"\n[{key}]")
        print(f"Min    : {values[0]:.6f}%")

        for q in PERCENTILES:
            print(
                f"P{int(q * 100):<5}: "
                f"{percentile(values, q):.6f}%"
            )

        print(f"Max    : {values[-1]:.6f}%")

        threshold = thresholds[key]
        exceeded = sum(value > threshold for value in values)

        print(f"Train threshold: {threshold:.6f}%")
        print(
            f"Above threshold: {exceeded}/{len(values)} "
            f"({exceeded / len(values) * 100:.2f}%)"
        )


def main():
    print("\n========== FEATURE DISTRIBUTION ANALYSIS ==========")
    print(f"Exchange: {EXCHANGE}")
    print(f"Symbol: {SYMBOL}")

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

    def select_period(start, end, days):
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
                f"Unexpected candle count: {len(candles)} != {expected}"
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
            raise RuntimeError("Incomplete market regime data")

        return candles, signals, regimes

    train = select_period(train_start, train_end, 120)
    validation = select_period(validation_start, validation_end, 120)
    test = select_period(test_start, test_end, 60)

    print("\n========== PERIODS ==========")
    for name, start, end in (
        ("TRAIN", train_start, train_end),
        ("VALIDATION", validation_start, validation_end),
        ("FINAL_TEST", test_start, test_end),
    ):
        print(f"{name}: {format_time(start)} to {format_time(end)}")

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
        train[0], train[1], train[2], series
    )
    validation_features = build_features(
        validation[0], validation[1], validation[2], series
    )
    test_features = build_features(
        test[0], test[1], test[2], series
    )

    train_rows = get_valid_rows(train_features)
    validation_rows = get_valid_rows(validation_features)
    test_rows = get_valid_rows(test_features)

    thresholds = build_thresholds(train_features)

    print("\n========== TRAINING THRESHOLDS ==========")
    for key in FEATURE_KEYS:
        print(f"{key:<15}: {thresholds[key]:.6f}%")

    print_distribution("TRAIN", train_rows, thresholds)
    print_distribution("VALIDATION", validation_rows, thresholds)
    print_distribution("FINAL_TEST", test_rows, thresholds)

    print("\n========== FINAL TEST CANDIDATES ==========")

    for index, row in test_rows:
        candle = test[0][index]
        entry_time_ms = candle.timestamp_ms + TIMEFRAMES["1m"]

        print(
            f"{format_time(entry_time_ms)} | "
            + " | ".join(
                f"{key}={row[key]:.6f}%"
                for key in FEATURE_KEYS
            )
        )

    print("\n========== TARGET ENTRY ANALYSIS ==========")

    target_ms = int(TARGET_ENTRY_TIME.timestamp() * 1000)
    target_found = False

    for index, row in test_rows:
        candle = test[0][index]
        entry_time_ms = candle.timestamp_ms + TIMEFRAMES["1m"]

        if entry_time_ms != target_ms:
            continue

        target_found = True

        print(f"Entry time: {format_time(entry_time_ms)}")
        print(f"Signal candle open: {format_time(candle.timestamp_ms)}")

        for key in FEATURE_KEYS:
            value = row[key]
            threshold = thresholds[key]
            difference = value - threshold
            ratio = value / threshold if threshold != 0 else float("inf")

            print(
                f"{key:<15}: "
                f"value={value:.6f}% | "
                f"threshold={threshold:.6f}% | "
                f"diff={difference:+.6f}% | "
                f"ratio={ratio:.3f}x"
            )

    if not target_found:
        print(
            "[NOT FOUND] Target entry timestamp is not a valid "
            "Bullish Transition feature candidate."
        )

    print("\n========== EXPORTING CANDIDATE DATA ==========")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "period",
            "signal_candle_time",
            "entry_time",
            *FEATURE_KEYS,
            *[f"{key}_above_train_threshold" for key in FEATURE_KEYS],
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for period_name, candles, rows in (
            ("TRAIN", train[0], train_rows),
            ("VALIDATION", validation[0], validation_rows),
            ("FINAL_TEST", test[0], test_rows),
        ):
            for index, row in rows:
                candle = candles[index]
                entry_time_ms = (
                    candle.timestamp_ms + TIMEFRAMES["1m"]
                )

                output = {
                    "period": period_name,
                    "signal_candle_time": format_time(
                        candle.timestamp_ms
                    ),
                    "entry_time": format_time(entry_time_ms),
                }

                for key in FEATURE_KEYS:
                    output[key] = row[key]
                    output[
                        f"{key}_above_train_threshold"
                    ] = row[key] > thresholds[key]

                writer.writerow(output)

    print(f"[COMPLETE] {OUTPUT_FILE}")
    print("Analysis only. No backtest or strategy files modified.")


if __name__ == "__main__":
    main()
