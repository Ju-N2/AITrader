import csv
from datetime import UTC, datetime
from pathlib import Path

from analyze_market_regimes_warmup_180d import (
    DAY_MS,
    TIMEFRAMES,
    build_aligned_data,
    build_market_regimes,
    BULLISH_TRANSITION,
    create_config,
)
from experiment_transition_entry_filters import (
    percentile,
    load_indicator_series,
    build_features,
)
from experiment_threshold_sensitivity import (
    EVALUATION_DAYS,
    TRAIN_END_OFFSET_DAYS,
    VALIDATION_END_OFFSET_DAYS,
    select_period,
)
from aitrader.backtesting.engine import BacktestEngine
from aitrader.data.storage import SQLiteCandleRepository


QUANTILES = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)

TARGET_FEATURES = (
    "15m_macd",
    "1h_macd",
)

DETAIL_FILE = Path(
    "data/macd_trade_capture_detail.csv"
)

SUMMARY_FILE = Path(
    "data/macd_trade_capture_summary.csv"
)


def iso_time(timestamp_ms):
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        UTC,
    ).isoformat()


def build_trade_records(
    period_name,
    period,
    features,
):
    candles = period[0]
    base_signals = period[3]

    result = BacktestEngine().run(
        candles,
        base_signals,
        create_config(),
    )

    timestamp_to_index = {
        candle.timestamp_ms: index
        for index, candle in enumerate(candles)
    }

    records = []
    missing = 0

    for trade in result.trades:
        signal_timestamp = (
            trade.entry_timestamp_ms
            - TIMEFRAMES["1m"]
        )

        index = timestamp_to_index.get(
            signal_timestamp
        )

        if index is None:
            missing += 1
            continue

        feature_row = features[index]

        if feature_row is None:
            missing += 1
            continue

        for feature in TARGET_FEATURES:
            records.append({
                "period": period_name,
                "entry_timestamp_ms": (
                    trade.entry_timestamp_ms
                ),
                "signal_timestamp_ms": (
                    signal_timestamp
                ),
                "exit_timestamp_ms": (
                    trade.exit_timestamp_ms
                ),
                "exit_reason": trade.exit_reason,
                "pnl": trade.pnl,
                "return_pct": trade.return_pct,
                "feature": feature,
                "feature_value": feature_row[feature],
            })

    print(
        f"{period_name}: "
        f"Completed trades={len(result.trades)}, "
        f"Matched trades={len(records) // len(TARGET_FEATURES)}, "
        f"Missing={missing}"
    )

    return records


def build_thresholds(train_features):
    rows = [
        row
        for row in train_features
        if row is not None
    ]

    thresholds = {}

    for feature in TARGET_FEATURES:
        values = [
            row[feature]
            for row in rows
        ]

        thresholds[feature] = {
            quantile: percentile(
                values,
                quantile,
            )
            for quantile in QUANTILES
        }

    return thresholds


def analyze_records(records, thresholds):
    detail_rows = []
    summary_rows = []

    for period in ("TRAIN", "VALIDATION"):
        for feature in TARGET_FEATURES:
            trades = [
                row
                for row in records
                if row["period"] == period
                and row["feature"] == feature
            ]

            wins = [
                row for row in trades
                if row["pnl"] > 0
            ]

            losses = [
                row for row in trades
                if row["pnl"] < 0
            ]

            for row in trades:
                detail = dict(row)

                detail["entry_timestamp_utc"] = (
                    iso_time(
                        row["entry_timestamp_ms"]
                    )
                )

                detail["signal_timestamp_utc"] = (
                    iso_time(
                        row["signal_timestamp_ms"]
                    )
                )

                detail["exit_timestamp_utc"] = (
                    iso_time(
                        row["exit_timestamp_ms"]
                    )
                )

                for quantile in QUANTILES:
                    threshold = thresholds[feature][
                        quantile
                    ]

                    detail[
                        f"threshold_q{int(quantile * 100)}"
                    ] = threshold

                    detail[
                        f"blocked_q{int(quantile * 100)}"
                    ] = (
                        row["feature_value"] > threshold
                    )

                detail_rows.append(detail)

            for quantile in QUANTILES:
                threshold = thresholds[feature][
                    quantile
                ]

                blocked = [
                    row for row in trades
                    if row["feature_value"] > threshold
                ]

                blocked_losses = [
                    row for row in blocked
                    if row["pnl"] < 0
                ]

                blocked_wins = [
                    row for row in blocked
                    if row["pnl"] > 0
                ]

                retained_wins = [
                    row for row in wins
                    if row["feature_value"] <= threshold
                ]

                retained = [
                    row for row in trades
                    if row["feature_value"] <= threshold
                ]

                blocked_pnl = sum(
                    row["pnl"] for row in blocked
                )

                retained_pnl = sum(
                    row["pnl"] for row in retained
                )

                summary_rows.append({
                    "period": period,
                    "feature": feature,
                    "quantile": int(quantile * 100),
                    "threshold_pct": threshold,
                    "baseline_trades": len(trades),
                    "baseline_wins": len(wins),
                    "baseline_losses": len(losses),
                    "blocked_trades": len(blocked),
                    "blocked_losses": len(blocked_losses),
                    "loss_capture_pct": (
                        len(blocked_losses) / len(losses) * 100
                        if losses else None
                    ),
                    "blocked_wins": len(blocked_wins),
                    "winner_retention_count": len(retained_wins),
                    "winner_retention_pct": (
                        len(retained_wins) / len(wins) * 100
                        if wins else None
                    ),
                    "blocked_pnl": blocked_pnl,
                    "retained_pnl": retained_pnl,
                    "baseline_pnl": sum(
                        row["pnl"] for row in trades
                    ),
                })

    return detail_rows, summary_rows


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            f"No rows to write: {path}"
        )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    print(
        "\n========== MACD TRADE CAPTURE ANALYSIS =========="
    )

    repository = SQLiteCandleRepository(
        "data/aitrader.db"
    )

    now_ms = int(
        datetime.now(UTC).timestamp() * 1000
    )

    regime_times, _, detailed_regimes = (
        build_market_regimes(
            repository,
            now_ms,
        )
    )

    full_candles, full_signals = (
        build_aligned_data(
            repository,
            now_ms,
        )
    )

    data_end_ms = (
        full_candles[-1].timestamp_ms
        + TIMEFRAMES["1m"]
    )

    train_end = (
        data_end_ms
        - TRAIN_END_OFFSET_DAYS * DAY_MS
    )

    train_start = (
        train_end
        - EVALUATION_DAYS * DAY_MS
    )

    validation_end = (
        data_end_ms
        - VALIDATION_END_OFFSET_DAYS * DAY_MS
    )

    validation_start = (
        validation_end
        - EVALUATION_DAYS * DAY_MS
    )

    print("\n========== PERIODS ==========")

    print(
        "TRAIN:",
        iso_time(train_start),
        "to",
        iso_time(train_end),
    )

    print(
        "VALIDATION:",
        iso_time(validation_start),
        "to",
        iso_time(validation_end),
    )

    train = select_period(
        full_candles,
        full_signals,
        regime_times,
        detailed_regimes,
        train_start,
        train_end,
    )

    validation = select_period(
        full_candles,
        full_signals,
        regime_times,
        detailed_regimes,
        validation_start,
        validation_end,
    )

    print("\n========== LOADING INDICATORS ==========")

    series = {
        timeframe: load_indicator_series(
            repository,
            timeframe,
            now_ms,
        )
        for timeframe in ("15m", "1h")
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

    print("\n========== TRAINING THRESHOLDS ==========")

    thresholds = build_thresholds(
        train_features
    )

    for feature in TARGET_FEATURES:
        print(f"\n{feature}")

        for quantile in QUANTILES:
            print(
                f"Q{quantile * 100:.0f}: "
                f"{thresholds[feature][quantile]:.6f}%"
            )

    print("\n========== MATCHING BASELINE TRADES ==========")

    records = []

    records.extend(
        build_trade_records(
            "TRAIN",
            train,
            train_features,
        )
    )

    records.extend(
        build_trade_records(
            "VALIDATION",
            validation,
            validation_features,
        )
    )

    detail_rows, summary_rows = analyze_records(
        records,
        thresholds,
    )

    write_csv(
        DETAIL_FILE,
        detail_rows,
    )

    write_csv(
        SUMMARY_FILE,
        summary_rows,
    )

    print("\n========== SUMMARY ==========")

    for row in summary_rows:
        loss_capture = row["loss_capture_pct"]
        winner_retention = row["winner_retention_pct"]

        loss_text = (
            f"{loss_capture:.1f}%"
            if loss_capture is not None
            else "N/A"
        )

        winner_text = (
            f"{winner_retention:.1f}%"
            if winner_retention is not None
            else "N/A"
        )

        print(
            f"{row['period']:<12} "
            f"{row['feature']:<10} "
            f"Q{row['quantile']} | "
            f"Blocked={row['blocked_trades']:>2} | "
            f"Loss capture={loss_text:>7} | "
            f"Winner retention={winner_text:>7} | "
            f"Blocked PnL={row['blocked_pnl']:>9.2f} | "
            f"Retained PnL={row['retained_pnl']:>9.2f}"
        )

    print("\n========== FILES CREATED ==========")
    print(DETAIL_FILE)
    print(SUMMARY_FILE)

    print("\n========== ANALYSIS COMPLETED ==========")


if __name__ == "__main__":
    main()
