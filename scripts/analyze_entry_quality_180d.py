from bisect import bisect_right
from datetime import UTC, datetime
from statistics import mean, median

from analyze_market_regimes_warmup_180d import (
    ATR_MULTIPLIER,
    DATA_LIMIT,
    DAY_MS,
    EVALUATION_DAYS,
    EXCHANGE,
    MIN_WARMUP_DAYS,
    SIDEWAYS,
    SYMBOL,
    TIMEFRAMES,
    BacktestEngine,
    build_aligned_data,
    build_market_regimes,
    calculate_ema,
    create_config,
    filter_entries_by_regime,
    get_regime_at,
)

from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine


def build_indicator_history(repository, timeframe, now_ms):
    interval_ms = TIMEFRAMES[timeframe]

    candles = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        timeframe,
        limit=DATA_LIMIT,
    )

    closed = [
        candle
        for candle in candles
        if candle.timestamp_ms + interval_ms <= now_ms
    ]

    if len(closed) < 250:
        raise RuntimeError(
            f"Insufficient {timeframe} history: {len(closed)}"
        )

    snapshots = IndicatorEngine().calculate(closed)

    if len(snapshots) != len(closed):
        raise RuntimeError(
            f"{timeframe}: candle and indicator counts differ"
        )

    close_times = [
        candle.timestamp_ms + interval_ms
        for candle in closed
    ]

    print(
        f"{timeframe}: "
        f"{len(closed):,} candles | "
        f"{len(snapshots):,} snapshots"
    )

    return closed, close_times, snapshots


def get_completed_snapshot(history, timeframe, decision_time):
    candles, close_times, snapshots = history[timeframe]

    index = bisect_right(
        close_times,
        decision_time,
    ) - 1

    if index < 0:
        raise RuntimeError(
            f"No completed {timeframe} candle at {decision_time}"
        )

    candle = candles[index]
    snapshot = snapshots[index]

    if snapshot.rsi_14 is None:
        raise RuntimeError(
            f"{timeframe} RSI unavailable at {decision_time}"
        )

    return candle, snapshot, index


def build_entry_features(history, entry_time, signal):
    candle_4h, snapshot_4h, index_4h = get_completed_snapshot(
        history,
        "4h",
        entry_time,
    )

    candle_1h, snapshot_1h, _ = get_completed_snapshot(
        history,
        "1h",
        entry_time,
    )

    candle_15m, snapshot_15m, _ = get_completed_snapshot(
        history,
        "15m",
        entry_time,
    )

    closes_4h = [
        candle.close
        for candle in history["4h"][0]
    ]

    ema_50 = calculate_ema(closes_4h, 50)
    ema_200 = calculate_ema(closes_4h, 200)

    if index_4h < 6:
        raise RuntimeError("Insufficient EMA slope history")

    current_ema50 = ema_50[index_4h]
    previous_ema50 = ema_50[index_4h - 6]
    current_ema200 = ema_200[index_4h]

    if any(
        value is None
        for value in (
            current_ema50,
            previous_ema50,
            current_ema200,
        )
    ):
        raise RuntimeError("EMA values are unavailable")

    ema50_slope_24h_pct = (
        current_ema50 / previous_ema50 - 1
    ) * 100

    ema_gap_pct = (
        current_ema50 / current_ema200 - 1
    ) * 100

    close_ema200_distance_pct = (
        candle_4h.close / current_ema200 - 1
    ) * 100

    atr_15m_pct = (
        snapshot_15m.atr_14 / candle_15m.close * 100
        if snapshot_15m.atr_14 is not None
        else None
    )

    return {
        "entry_time": datetime.fromtimestamp(
            entry_time / 1000,
            UTC,
        ),
        "entry_price": None,
        "pnl": None,
        "exit_reason": None,
        "ema50_slope_24h_pct": ema50_slope_24h_pct,
        "ema50_ema200_gap_pct": ema_gap_pct,
        "close_ema200_distance_pct": close_ema200_distance_pct,
        "rsi_4h": snapshot_4h.rsi_14,
        "rsi_1h": snapshot_1h.rsi_14,
        "rsi_15m": snapshot_15m.rsi_14,
        "atr_15m_pct": atr_15m_pct,
        "mtf_score": signal.score,
        "mtf_bias": signal.bias.value,
        "completed_4h_time": candle_4h.datetime_utc,
        "completed_1h_time": candle_1h.datetime_utc,
        "completed_15m_time": candle_15m.datetime_utc,
    }


def print_feature_summary(name, rows):
    print(f"\n--- {name} ---")
    print(f"Trades: {len(rows)}")

    if not rows:
        return

    metrics = (
        ("EMA50 Slope 24h", "ema50_slope_24h_pct", "%"),
        ("EMA50-EMA200 Gap", "ema50_ema200_gap_pct", "%"),
        ("Close-EMA200 Distance", "close_ema200_distance_pct", "%"),
        ("4h RSI", "rsi_4h", ""),
        ("1h RSI", "rsi_1h", ""),
        ("15m RSI", "rsi_15m", ""),
        ("15m ATR / Price", "atr_15m_pct", "%"),
        ("MTF Score", "mtf_score", ""),
    )

    print(
        f"{'Metric':<27}"
        f"{'Mean':>12}"
        f"{'Median':>12}"
    )

    for label, key, suffix in metrics:
        values = [
            row[key]
            for row in rows
            if row[key] is not None
        ]

        if not values:
            continue

        print(
            f"{label:<27}"
            f"{mean(values):>11.3f}{suffix}"
            f"{median(values):>11.3f}{suffix}"
        )


def main():
    print("\n========== AITrader ENTRY QUALITY ANALYSIS ==========")
    print(f"Exchange: {EXCHANGE}")
    print(f"Symbol: {SYMBOL}")
    print(f"Evaluation: {EVALUATION_DAYS} days")
    print(f"ATR Multiplier: {ATR_MULTIPLIER}")

    repository = SQLiteCandleRepository("data/aitrader.db")

    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    # 1. 원본과 동일한 시장 국면 및 MTF 신호 생성
    regime_times, regimes_full, _ = build_market_regimes(
        repository,
        now_ms,
    )

    full_candles, full_signals = build_aligned_data(
        repository,
        now_ms,
    )

    if not full_candles:
        raise RuntimeError("No aligned candles")

    evaluation_end_ms = full_candles[-1].timestamp_ms + 60_000

    evaluation_start_ms = (
        evaluation_end_ms - EVALUATION_DAYS * DAY_MS
    )

    warmup_days = (
        evaluation_start_ms - full_candles[0].timestamp_ms
    ) / DAY_MS

    if warmup_days < MIN_WARMUP_DAYS:
        raise RuntimeError(
            f"Insufficient warmup history: {warmup_days:.2f} days"
        )

    selected = [
        index
        for index, candle in enumerate(full_candles)
        if evaluation_start_ms <= candle.timestamp_ms < evaluation_end_ms
    ]

    candles = [full_candles[index] for index in selected]
    signals = [full_signals[index] for index in selected]

    expected = EVALUATION_DAYS * 24 * 60

    if len(candles) != expected:
        raise RuntimeError(
            f"Expected {expected} candles, found {len(candles)}"
        )

    # 2. 평가 구간 시장 국면 매핑
    regimes = []

    for candle in candles:
        decision_time = candle.timestamp_ms + 60_000

        regimes.append(
            get_regime_at(
                decision_time,
                regime_times,
                regimes_full,
            )
        )

    if any(regime != SIDEWAYS for regime in regimes
           if regime == "unavailable"):
        raise RuntimeError("Unavailable market regime detected")

    # 3. 기존 횡보·전환장 전략 재현
    filtered_signals = filter_entries_by_regime(
        signals,
        regimes,
        SIDEWAYS,
    )

    result = BacktestEngine().run(
        candles,
        filtered_signals,
        create_config(),
    )

    print("\n========== FILTER REPRODUCTION ==========")
    print(f"Trades: {result.total_trades}")
    print(f"Final Equity: {result.final_equity:,.2f} USDT")
    print(f"Return: {result.total_return_pct:+.2f}%")
    print(f"MDD: {result.max_drawdown_pct:.2f}%")
    print(f"Profit Factor: {result.profit_factor}")

    if result.total_trades == 0:
        raise RuntimeError("No trades to analyze")

    # 4. 지표 데이터 준비
    print("\n========== BUILDING INDICATOR HISTORY ==========")

    history = {}

    for timeframe in ("4h", "1h", "15m"):
        history[timeframe] = build_indicator_history(
            repository,
            timeframe,
            now_ms,
        )

    # 5. 진입 결정 시점에 완료된 신호 조회
    signal_by_decision = {
        candle.timestamp_ms + 60_000: signal
        for candle, signal in zip(candles, signals)
    }

    rows = []

    print("\n========== INDIVIDUAL TRADE FEATURES ==========")

    for number, trade in enumerate(result.trades, 1):
        signal = signal_by_decision.get(
            trade.entry_timestamp_ms
        )

        if signal is None:
            raise RuntimeError(
                f"Entry signal missing for trade {number}"
            )

        features = build_entry_features(
            history,
            trade.entry_timestamp_ms,
            signal,
        )

        features["entry_price"] = trade.entry_price
        features["pnl"] = trade.pnl
        features["exit_reason"] = trade.exit_reason
        features["return_pct"] = trade.return_pct

        rows.append(features)

        print(f"\n--- Trade {number} ---")
        print(f"Entry: {features['entry_time']}")
        print(f"Entry Price: {trade.entry_price:.2f}")
        print(f"PnL: {trade.pnl:+.2f} USDT")
        print(f"Return: {trade.return_pct:+.3f}%")
        print(f"Exit Reason: {trade.exit_reason}")

        print(
            f"4h EMA50 Slope (24h): "
            f"{features['ema50_slope_24h_pct']:+.4f}%"
        )

        print(
            f"EMA50-EMA200 Gap: "
            f"{features['ema50_ema200_gap_pct']:+.4f}%"
        )

        print(
            f"Close-EMA200 Distance: "
            f"{features['close_ema200_distance_pct']:+.4f}%"
        )

        print(f"4h RSI: {features['rsi_4h']:.2f}")
        print(f"1h RSI: {features['rsi_1h']:.2f}")
        print(f"15m RSI: {features['rsi_15m']:.2f}")
        print(f"15m ATR / Price: {features['atr_15m_pct']:.4f}%")
        print(f"MTF Score: {features['mtf_score']}")
        print(f"MTF Bias: {features['mtf_bias']}")

    # 6. 승자와 패자 비교
    winners = [
        row for row in rows
        if row["pnl"] > 0
    ]

    losers = [
        row for row in rows
        if row["pnl"] < 0
    ]

    print("\n========== WINNERS VS LOSERS ==========")

    print_feature_summary("WINNERS", winners)
    print_feature_summary("LOSERS", losers)

    # 7. 최대 수익 거래 의존도 확인
    largest_winner = max(
        rows,
        key=lambda row: row["pnl"],
    )

    gross_profit = sum(
        row["pnl"] for row in winners
    )

    gross_loss = abs(sum(
        row["pnl"] for row in losers
    ))

    print("\n========== CONCENTRATION AUDIT ==========")
    print(
        f"Largest Winner: "
        f"{largest_winner['pnl']:+.2f} USDT"
    )

    if gross_profit > 0:
        print(
            f"Largest Winner Share: "
            f"{largest_winner['pnl'] / gross_profit * 100:.2f}%"
        )

    print(
        f"Net PnL Excluding Largest Winner: "
        f"{sum(row['pnl'] for row in rows) - largest_winner['pnl']:+.2f} USDT"
    )

    print(f"Gross Profit: {gross_profit:+.2f} USDT")
    print(f"Gross Loss: {-gross_loss:+.2f} USDT")

    print("\n========== ANALYSIS COMPLETED ==========")
    print("Only completed historical candles were used.")
    print("No strategy or backtest engine files were modified.")


if __name__ == "__main__":
    main()
