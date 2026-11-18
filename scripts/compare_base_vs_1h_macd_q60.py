import csv
from datetime import UTC, datetime
from pathlib import Path

from analyze_market_regimes_warmup_180d import (
    DAY_MS,
    TIMEFRAMES,
    BULLISH_TRANSITION,
    build_aligned_data,
    build_market_regimes,
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
    apply_single_feature_filter,
    calculate_metrics,
)

from aitrader.backtesting.engine import BacktestEngine
from aitrader.data.storage import SQLiteCandleRepository


FEATURE = "1h_macd"
QUANTILE = 0.60

DETAIL_FILE = Path(
    "data/base_vs_1h_macd_q60_trades.csv"
)


def iso_time(timestamp_ms):
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        UTC,
    ).isoformat()


def trade_signature(trade):
    return (
        trade.entry_timestamp_ms,
        trade.exit_timestamp_ms,
        trade.exit_reason,
        round(trade.entry_price, 8),
        round(trade.exit_price, 8),
        round(trade.pnl, 8),
        round(trade.return_pct, 8),
    )


def trade_record(trade):
    return {
        "entry_timestamp_ms": trade.entry_timestamp_ms,
        "entry_timestamp_utc": iso_time(
            trade.entry_timestamp_ms
        ),
        "exit_timestamp_ms": trade.exit_timestamp_ms,
        "exit_timestamp_utc": iso_time(
            trade.exit_timestamp_ms
        ),
        "exit_reason": trade.exit_reason,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "quantity": trade.quantity,
        "entry_fee": trade.entry_fee,
        "exit_fee": trade.exit_fee,
        "pnl": trade.pnl,
        "return_pct": trade.return_pct,
    }


def compare_trades(base_result, filtered_result):
    base_by_entry = {
        trade.entry_timestamp_ms: trade
        for trade in base_result.trades
    }

    filtered_by_entry = {
        trade.entry_timestamp_ms: trade
        for trade in filtered_result.trades
    }

    if len(base_by_entry) != len(base_result.trades):
        raise RuntimeError(
            "Duplicate BASE entry timestamps detected"
        )

    if len(filtered_by_entry) != len(filtered_result.trades):
        raise RuntimeError(
            "Duplicate FILTERED entry timestamps detected"
        )

    all_entries = sorted(
        set(base_by_entry) | set(filtered_by_entry)
    )

    rows = []

    counts = {
        "UNCHANGED": 0,
        "CHANGED": 0,
        "REMOVED": 0,
        "NEW": 0,
    }

    for timestamp in all_entries:
        base_trade = base_by_entry.get(timestamp)
        filtered_trade = filtered_by_entry.get(timestamp)

        if base_trade is None:
            status = "NEW"
            base = None
            filtered = trade_record(filtered_trade)

        elif filtered_trade is None:
            status = "REMOVED"
            base = trade_record(base_trade)
            filtered = None

        elif (
            trade_signature(base_trade)
            == trade_signature(filtered_trade)
        ):
            status = "UNCHANGED"
            base = trade_record(base_trade)
            filtered = trade_record(filtered_trade)

        else:
            status = "CHANGED"
            base = trade_record(base_trade)
            filtered = trade_record(filtered_trade)

        counts[status] += 1

        rows.append({
            "status": status,
            "entry_timestamp_utc": iso_time(timestamp),
            "base_exit_timestamp_utc": (
                base["exit_timestamp_utc"]
                if base else ""
            ),
            "filtered_exit_timestamp_utc": (
                filtered["exit_timestamp_utc"]
                if filtered else ""
            ),
            "base_exit_reason": (
                base["exit_reason"] if base else ""
            ),
            "filtered_exit_reason": (
                filtered["exit_reason"] if filtered else ""
            ),
            "base_entry_price": (
                base["entry_price"] if base else ""
            ),
            "filtered_entry_price": (
                filtered["entry_price"] if filtered else ""
            ),
            "base_exit_price": (
                base["exit_price"] if base else ""
            ),
            "filtered_exit_price": (
                filtered["exit_price"] if filtered else ""
            ),
            "base_quantity": (
                base["quantity"] if base else ""
            ),
            "filtered_quantity": (
                filtered["quantity"] if filtered else ""
            ),
            "base_entry_fee": (
                base["entry_fee"] if base else ""
            ),
            "filtered_entry_fee": (
                filtered["entry_fee"] if filtered else ""
            ),
            "base_exit_fee": (
                base["exit_fee"] if base else ""
            ),
            "filtered_exit_fee": (
                filtered["exit_fee"] if filtered else ""
            ),
            "base_pnl": (
                base["pnl"] if base else ""
            ),
            "filtered_pnl": (
                filtered["pnl"] if filtered else ""
            ),
            "pnl_difference": (
                filtered["pnl"] - base["pnl"]
                if base and filtered else ""
            ),
            "base_return_pct": (
                base["return_pct"] if base else ""
            ),
            "filtered_return_pct": (
                filtered["return_pct"] if filtered else ""
            ),
        })

    return rows, counts


def print_metrics(period_name, label, result):
    metrics = calculate_metrics(result)

    print(
        f"{period_name:<12} "
        f"{label:<10} "
        f"Trades={metrics['trades']:>3} | "
        f"Return={metrics['return_pct']:>8.2f}% | "
        f"MDD={metrics['mdd_pct']:>7.2f}% | "
        f"Win={metrics['win_rate_pct']:>6.2f}% | "
        f"PF={metrics['profit_factor']} | "
        f"ATR Stops={metrics['atr_stops']:>3} | "
        f"Trade PnL={metrics['pnl']:>9.2f}"
    )

    return metrics


def main():
    print(
        "\n========== BASE VS 1H MACD Q60 =========="
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

    if not full_candles:
        raise RuntimeError(
            "No aligned candles available"
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

    # Final test: validation 종료 시점부터 최신 완료 캔들까지
    final_test = select_period(
        full_candles,
        full_signals,
        regime_times,
        detailed_regimes,
        validation_end,
        data_end_ms,
        expected_days=60,
    )

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

    final_test_features = build_features(
        final_test[0],
        final_test[1],
        final_test[2],
        series,
    )

    training_values = [
        row[FEATURE]
        for row in train_features
        if row is not None
    ]

    threshold = percentile(
        training_values,
        QUANTILE,
    )

    # 기존 실험에서 확정한 TRAIN Q60 임계값 검증
    expected_threshold = 0.136626

    if abs(threshold - expected_threshold) > 0.000001:
        raise RuntimeError(
            f"Q60 threshold changed: {threshold:.9f} "
            f"(expected approximately {expected_threshold:.6f}). "
            "Stop before evaluating FINAL_TEST."
        )

    print("\n========== FILTER ==========")
    print(f"Feature: {FEATURE}")
    print(f"Quantile: Q{QUANTILE * 100:.0f}")
    print(f"Threshold: {threshold:.6f}%")
    print(f"Training candidates: {len(training_values)}")

    all_rows = []

    for period_name, period, features in (
        ("TRAIN", train, train_features),
        ("VALIDATION", validation, validation_features),
        ("FINAL_TEST", final_test, final_test_features),
    ):
        print(f"\n========== {period_name} ==========")

        base_result = BacktestEngine().run(
            period[0],
            period[3],
            create_config(),
        )

        filtered_signals, blocked_count = (
            apply_single_feature_filter(
                period[3],
                features,
                FEATURE,
                threshold,
            )
        )

        filtered_result = BacktestEngine().run(
            period[0],
            filtered_signals,
            create_config(),
        )

        base_metrics = print_metrics(
            period_name,
            "BASE",
            base_result,
        )

        filtered_metrics = print_metrics(
            period_name,
            "Q60",
            filtered_result,
        )

        print(
            f"Blocked candidates: {blocked_count}"
        )

        print(
            f"Return difference: "
            f"{filtered_metrics['return_pct'] - base_metrics['return_pct']:+.2f}%p"
        )

        print(
            f"MDD difference: "
            f"{filtered_metrics['mdd_pct'] - base_metrics['mdd_pct']:+.2f}%p"
        )

        rows, counts = compare_trades(
            base_result,
            filtered_result,
        )

        for row in rows:
            row["period"] = period_name
            row["threshold"] = threshold

        all_rows.extend(rows)

        print("\n--- TRADE COMPARISON ---")

        for status, count in counts.items():
            print(f"{status:<12}: {count}")

        print(
            f"BASE trade count: {len(base_result.trades)}"
        )

        print(
            f"Q60 trade count: {len(filtered_result.trades)}"
        )

        print(
            f"BASE total trade PnL: "
            f"{base_metrics['pnl']:.2f}"
        )

        print(
            f"Q60 total trade PnL: "
            f"{filtered_metrics['pnl']:.2f}"
        )

        print("\n--- CHANGED / REMOVED / NEW ---")

        for row in rows:
            if row["status"] != "UNCHANGED":
                print(
                    f"{row['status']:<10} "
                    f"{row['entry_timestamp_utc']} | "
                    f"BASE PnL={row['base_pnl']} | "
                    f"Q60 PnL={row['filtered_pnl']} | "
                    f"BASE exit={row['base_exit_reason']} | "
                    f"Q60 exit={row['filtered_exit_reason']}"
                )

    if all_rows:
        with DETAIL_FILE.open(
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

    print("\n========== OUTPUT ==========")
    print(DETAIL_FILE)
    print("\n========== COMPARISON COMPLETED ==========")


if __name__ == "__main__":
    main()
