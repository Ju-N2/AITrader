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
    create_config,
    filter_entries_by_regime,
    get_regime_at,
)

from experiment_transition_entry_filters import (
    percentile,
    load_indicator_series,
    build_features,
)

from aitrader.backtesting.engine import BacktestEngine
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.strategy.models import MarketBias, StrategySignal


EVALUATION_DAYS = 120
TRAIN_END_OFFSET_DAYS = 180
VALIDATION_END_OFFSET_DAYS = 60

QUANTILES = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)

TARGET_FEATURES = (
    "1h_macd",
    "1h_atr",
    "15m_macd",
)

OUTPUT_FILE = Path(
    "data/transition_threshold_sensitivity.csv"
)


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

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else None
    )

    return {
        "return_pct": result.total_return_pct,
        "mdd_pct": result.max_drawdown_pct,
        "trades": result.total_trades,
        "win_rate_pct": result.win_rate_pct,
        "profit_factor": profit_factor,
        "atr_stops": result.stop_loss_count,
        "pnl": sum(
            trade.pnl for trade in result.trades
        ),
    }


def apply_single_feature_filter(
    signals,
    features,
    feature_key,
    threshold,
):
    filtered = []
    blocked_count = 0

    for signal, row in zip(signals, features):
        blocked = (
            signal.bias == MarketBias.BULLISH
            and row is not None
            and row[feature_key] > threshold
        )

        if blocked:
            blocked_count += 1

            filtered.append(
                StrategySignal(
                    exchange=signal.exchange,
                    symbol=signal.symbol,
                    timeframe=signal.timeframe,
                    timestamp_ms=signal.timestamp_ms,
                    bias=MarketBias.NEUTRAL,
                    score=0,
                    max_score=signal.max_score,
                    reasons=signal.reasons + (
                        f"Sensitivity filter blocked: "
                        f"{feature_key}",
                    ),
                    atr=signal.atr,
                )
            )
        else:
            filtered.append(signal)

    return filtered, blocked_count


def select_period(
    full_candles,
    full_signals,
    regime_times,
    detailed_regimes,
    start,
    end,
    expected_days=EVALUATION_DAYS,
):
    selected = [
        index
        for index, candle in enumerate(full_candles)
        if start <= candle.timestamp_ms < end
    ]

    candles = [full_candles[i] for i in selected]
    signals = [full_signals[i] for i in selected]

    expected = expected_days * 24 * 60

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

    base_signals = filter_entries_by_regime(
        signals,
        regimes,
        BULLISH_TRANSITION,
    )

    return candles, signals, regimes, base_signals


def run_one_experiment(
    period_name,
    candles,
    base_signals,
    features,
    feature_key,
    quantile,
    threshold,
):
    filtered_signals, blocked_count = (
        apply_single_feature_filter(
            base_signals,
            features,
            feature_key,
            threshold,
        )
    )

    result = BacktestEngine().run(
        candles,
        filtered_signals,
        create_config(),
    )

    metrics = calculate_metrics(result)

    row = {
        "period": period_name,
        "feature": feature_key,
        "quantile": quantile,
        "threshold": threshold,
        "blocked_candidates": blocked_count,
        **metrics,
    }

    pf = (
        f"{metrics['profit_factor']:.3f}"
        if metrics["profit_factor"] is not None
        else "N/A"
    )

    print(
        f"{period_name:<12} "
        f"{feature_key:<10} "
        f"Q{quantile * 100:.0f} "
        f"Threshold={threshold:.6f}% | "
        f"Blocked={blocked_count:>4} | "
        f"Trades={metrics['trades']:>3} | "
        f"Return={metrics['return_pct']:>8.2f}% | "
        f"MDD={metrics['mdd_pct']:>7.2f}% | "
        f"Win={metrics['win_rate_pct']:>6.2f}% | "
        f"PF={pf:>7} | "
        f"ATR Stops={metrics['atr_stops']:>3}"
    )

    return row


def main():
    print(
        "\n========== AITrader THRESHOLD SENSITIVITY =========="
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

    print("\n========== PERIODS ==========")

    for name, start, end in (
        ("TRAIN", train_start, train_end),
        (
            "VALIDATION",
            validation_start,
            validation_end,
        ),
    ):
        print(
            f"{name}: "
            f"{datetime.fromtimestamp(start / 1000, UTC)} "
            f"to "
            f"{datetime.fromtimestamp(end / 1000, UTC)}"
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

    train_rows = [
        row for row in train_features
        if row is not None
    ]

    validation_rows = [
        row for row in validation_features
        if row is not None
    ]

    print("\n========== CANDIDATE COUNTS ==========")
    print(f"TRAIN: {len(train_rows)}")
    print(f"VALIDATION: {len(validation_rows)}")

    all_results = []

    print("\n========== TRAINING THRESHOLDS ==========")

    thresholds = {}

    for feature_key in TARGET_FEATURES:
        values = [
            row[feature_key]
            for row in train_rows
        ]

        thresholds[feature_key] = {}

        print(f"\n{feature_key}: n={len(values)}")

        for quantile in QUANTILES:
            threshold = percentile(
                values,
                quantile,
            )

            thresholds[feature_key][quantile] = (
                threshold
            )

            print(
                f"Q{quantile * 100:.0f}: "
                f"{threshold:.6f}%"
            )

    print("\n========== BASELINE ==========")

    for period_name, period in (
        ("TRAIN", train),
        ("VALIDATION", validation),
    ):
        result = BacktestEngine().run(
            period[0],
            period[3],
            create_config(),
        )

        metrics = calculate_metrics(result)

        print(
            f"{period_name:<12} "
            f"Trades={metrics['trades']:>3} | "
            f"Return={metrics['return_pct']:>8.2f}% | "
            f"MDD={metrics['mdd_pct']:>7.2f}% | "
            f"Win={metrics['win_rate_pct']:>6.2f}% | "
            f"ATR Stops={metrics['atr_stops']:>3}"
        )

        all_results.append({
            "period": period_name,
            "feature": "BASE",
            "quantile": None,
            "threshold": None,
            "blocked_candidates": 0,
            **metrics,
        })

    print("\n========== SENSITIVITY EXPERIMENTS ==========")

    for feature_key in TARGET_FEATURES:
        for quantile in QUANTILES:
            threshold = thresholds[feature_key][
                quantile
            ]

            train_result = run_one_experiment(
                "TRAIN",
                train[0],
                train[3],
                train_features,
                feature_key,
                quantile,
                threshold,
            )

            validation_result = run_one_experiment(
                "VALIDATION",
                validation[0],
                validation[3],
                validation_features,
                feature_key,
                quantile,
                threshold,
            )

            all_results.extend([
                train_result,
                validation_result,
            ])

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
            fieldnames=list(all_results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n========== EXPERIMENT COMPLETED ==========")
    print(f"Results: {OUTPUT_FILE}")
    print(f"Experiments: {len(all_results)}")


if __name__ == "__main__":
    main()
