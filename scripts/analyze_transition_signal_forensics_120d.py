import csv
from bisect import bisect_right
from datetime import UTC, datetime
from pathlib import Path

from analyze_market_regimes_warmup_180d import (
    DAY_MS,
    DATA_LIMIT,
    EXCHANGE,
    SYMBOL,
    TIMEFRAMES,
    ATR_MULTIPLIER,
    build_aligned_data,
    build_market_regimes,
    create_config,
    filter_entries_by_regime,
    get_regime_at,
    BULLISH_TRANSITION,
    indicator_ready,
)

from aitrader.backtesting.engine import BacktestEngine
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine
from aitrader.strategy.models import MarketBias
from aitrader.strategy.multi_timeframe import MultiTimeframeEngine
from aitrader.strategy.technical import TechnicalTrendStrategy


EVALUATION_DAYS = 120

OUTPUT_FILE = Path(
    "data/transition_signal_forensics_120d.csv"
)

TIMEFRAME_ORDER = ("1m", "5m", "15m", "1h", "4h")

INDICATOR_FIELDS = (
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bollinger_upper",
    "bollinger_middle",
    "bollinger_lower",
    "atr_14",
)


def utc_string(timestamp_ms):
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        UTC,
    ).strftime("%Y-%m-%d %H:%M:%S")


def enum_value(value):
    return value.value if hasattr(value, "value") else value


def serialize_reasons(reasons):
    return " | ".join(str(reason) for reason in reasons)


def load_timeframe_series(repository, timeframe, interval_ms, now_ms):
    candles = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        timeframe,
        limit=DATA_LIMIT,
    )

    candles = [
        candle
        for candle in candles
        if candle.timestamp_ms + interval_ms <= now_ms
    ]

    snapshots = IndicatorEngine().calculate(candles)
    strategy = TechnicalTrendStrategy()

    close_times = []
    signals = []
    snapshot_map = {}

    for candle, snapshot in zip(candles, snapshots):
        if not indicator_ready(snapshot):
            continue

        signal = strategy.evaluate(candle, snapshot)

        close_time = candle.timestamp_ms + interval_ms

        close_times.append(close_time)
        signals.append(signal)
        snapshot_map[close_time] = snapshot

    print(
        f"{timeframe} forensic series: "
        f"{len(signals):,} completed signals"
    )

    return close_times, signals, snapshot_map


def get_timeframe_state(series, timeframe, decision_time):
    close_times, signals, snapshot_map = series[timeframe]

    index = bisect_right(close_times, decision_time) - 1

    if index < 0:
        return None, None, None

    signal_time = close_times[index]

    if (
        decision_time - signal_time
        > TIMEFRAMES[timeframe] * 3
    ):
        return None, None, None

    signal = signals[index]
    snapshot = snapshot_map[signal_time]

    return signal, snapshot, signal_time


def build_decision_snapshot(series, decision_time):
    current_signals = {}
    current_snapshots = {}
    signal_times = {}

    for timeframe in TIMEFRAME_ORDER:
        signal, snapshot, signal_time = get_timeframe_state(
            series,
            timeframe,
            decision_time,
        )

        if signal is not None:
            current_signals[timeframe] = signal
            current_snapshots[timeframe] = snapshot
            signal_times[timeframe] = signal_time

    if len(current_signals) == len(TIMEFRAME_ORDER):
        result = MultiTimeframeEngine(
            min_timeframes=5
        ).evaluate(current_signals)

        mtf_bias = enum_value(result.bias)
        mtf_score = result.weighted_score
        mtf_reasons = serialize_reasons(result.reasons)

    else:
        mtf_bias = "unavailable"
        mtf_score = None
        mtf_reasons = (
            "Insufficient timeframe data: "
            + ", ".join(
                timeframe
                for timeframe in TIMEFRAME_ORDER
                if timeframe not in current_signals
            )
        )

    return {
        "signals": current_signals,
        "snapshots": current_snapshots,
        "signal_times": signal_times,
        "mtf_bias": mtf_bias,
        "mtf_score": mtf_score,
        "mtf_reasons": mtf_reasons,
    }


def add_decision_fields(row, prefix, decision):
    row[f"{prefix}_mtf_bias"] = decision["mtf_bias"]
    row[f"{prefix}_mtf_weighted_score"] = decision["mtf_score"]
    row[f"{prefix}_mtf_reasons"] = decision["mtf_reasons"]

    for timeframe in TIMEFRAME_ORDER:
        signal = decision["signals"].get(timeframe)
        snapshot = decision["snapshots"].get(timeframe)
        signal_time = decision["signal_times"].get(timeframe)

        base = f"{prefix}_{timeframe}"

        row[f"{base}_signal_time_utc"] = (
            utc_string(signal_time)
            if signal_time is not None
            else None
        )

        row[f"{base}_bias"] = (
            enum_value(signal.bias)
            if signal is not None
            else "unavailable"
        )

        row[f"{base}_score"] = (
            signal.score if signal is not None else None
        )

        row[f"{base}_max_score"] = (
            signal.max_score if signal is not None else None
        )

        row[f"{base}_reasons"] = (
            serialize_reasons(signal.reasons)
            if signal is not None
            else None
        )

        for field in INDICATOR_FIELDS:
            row[f"{base}_{field}"] = (
                getattr(snapshot, field)
                if snapshot is not None
                else None
            )


def calculate_excursions(
    trade,
    candles,
    entry_index,
    exit_index,
    include_exit_candle,
):
    if entry_index is None or exit_index is None:
        return None, None

    end_index = (
        exit_index + 1
        if include_exit_candle
        else exit_index
    )

    relevant = candles[entry_index:end_index]

    if not relevant:
        return None, None

    highest = max(c.high for c in relevant)
    lowest = min(c.low for c in relevant)

    mfe = (
        highest / trade.entry_price - 1
    ) * 100

    mae = (
        lowest / trade.entry_price - 1
    ) * 100

    return mfe, mae


def main():
    print("\n========== TRANSITION SIGNAL FORENSICS ==========")

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

    data_end_ms = (
        full_candles[-1].timestamp_ms
        + TIMEFRAMES["1m"]
    )

    evaluation_end_ms = data_end_ms - 180 * DAY_MS

    evaluation_start_ms = (
        evaluation_end_ms - EVALUATION_DAYS * DAY_MS
    )

    selected = [
        index
        for index, candle in enumerate(full_candles)
        if (
            evaluation_start_ms
            <= candle.timestamp_ms
            < evaluation_end_ms
        )
    ]

    candles = [full_candles[index] for index in selected]
    signals = [full_signals[index] for index in selected]

    expected_count = EVALUATION_DAYS * 24 * 60

    if len(candles) != expected_count:
        raise RuntimeError(
            f"Unexpected evaluation candle count: "
            f"{len(candles)} != {expected_count}"
        )

    regimes = [
        get_regime_at(
            candle.timestamp_ms + TIMEFRAMES["1m"],
            regime_times,
            detailed_regimes,
        )
        for candle in candles
    ]

    filtered_signals = filter_entries_by_regime(
        signals,
        regimes,
        BULLISH_TRANSITION,
    )

    result = BacktestEngine().run(
        candles,
        filtered_signals,
        create_config(),
    )

    print("\n========== BACKTEST REPRODUCTION ==========")
    print(f"Evaluation candles: {len(candles):,}")
    print(f"Trades reproduced: {len(result.trades)}")
    print(f"Final equity: {result.final_equity:.2f} USDT")
    print(f"Return: {result.total_return_pct:.4f}%")
    print(f"MDD: {result.max_drawdown_pct:.4f}%")
    print(f"Win rate: {result.win_rate_pct:.2f}%")
    print(f"Profit factor: {result.profit_factor}")

    print("\n========== BUILDING FORENSIC SERIES ==========")

    series = {}

    for timeframe, interval_ms in TIMEFRAMES.items():
        series[timeframe] = load_timeframe_series(
            repository,
            timeframe,
            interval_ms,
            now_ms,
        )

    candle_index = {
        candle.timestamp_ms: index
        for index, candle in enumerate(candles)
    }

    rows = []

    for number, trade in enumerate(result.trades, start=1):
        entry_index = candle_index.get(
            trade.entry_timestamp_ms
        )

        exit_index = candle_index.get(
            trade.exit_timestamp_ms
        )

        entry_regime = get_regime_at(
            trade.entry_timestamp_ms,
            regime_times,
            detailed_regimes,
        )

        exit_regime = get_regime_at(
            trade.exit_timestamp_ms,
            regime_times,
            detailed_regimes,
        )

        # 진입 시각은 실제 다음 1m 캔들의 시가 체결 시각.
        # 해당 시각까지 완료된 상위 시간봉만 사용한다.
        entry_decision = build_decision_snapshot(
            series,
            trade.entry_timestamp_ms,
        )

        # 청산 시각은 청산을 유발한 직전 1m 캔들의
        # 의사결정 시각과 일치한다.
        exit_decision = build_decision_snapshot(
            series,
            trade.exit_timestamp_ms,
        )

        duration_minutes = (
            trade.exit_timestamp_ms
            - trade.entry_timestamp_ms
        ) / 60_000

        row = {
            "trade": number,
            "entry_time_utc": utc_string(
                trade.entry_timestamp_ms
            ),
            "exit_time_utc": utc_string(
                trade.exit_timestamp_ms
            ),
            "duration_minutes": round(
                duration_minutes,
                2,
            ),
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "quantity": trade.quantity,
            "entry_fee": trade.entry_fee,
            "exit_fee": trade.exit_fee,
            "pnl_usdt": trade.pnl,
            "return_pct": trade.return_pct,
            "exit_reason": trade.exit_reason,
            "entry_regime": entry_regime,
            "exit_regime": exit_regime,
            "entry_1m_decision_time_utc": utc_string(
                trade.entry_timestamp_ms
            ),
            "exit_1m_decision_time_utc": utc_string(
                trade.exit_timestamp_ms
            ),
        }

        add_decision_fields(
            row,
            "entry",
            entry_decision,
        )

        add_decision_fields(
            row,
            "pre_exit",
            exit_decision,
        )

        # MFE / MAE 계산
        # 기존 분석과 동일한 캔들 포함 규칙 유지.
        include_exit = True

        if trade.exit_reason == "strategy":
            include_exit = False

        elif "stop" in trade.exit_reason.lower():
            base_exit_price = (
                trade.exit_price
                / (1 - create_config().slippage_rate)
            )

            if entry_index is not None and entry_index > 0:
                entry_signal = signals[entry_index - 1]
                atr_value = entry_signal.atr

                if atr_value is not None:
                    stop_price = (
                        trade.entry_price
                        - atr_value * ATR_MULTIPLIER
                    )

                    if base_exit_price < stop_price - 1e-8:
                        include_exit = False

        mfe, mae = calculate_excursions(
            trade,
            candles,
            entry_index,
            exit_index,
            include_exit,
        )

        row["mfe_pct"] = mfe
        row["mae_pct"] = mae

        row["excursion_method"] = (
            "exit candle included"
            if include_exit
            else "exit candle excluded"
        )

        rows.append(row)

        print(
            f"\n[{number:02d}] "
            f"{utc_string(trade.entry_timestamp_ms)}"
            f" | PnL {trade.pnl:+.2f} USDT"
            f" | {trade.exit_reason}"
        )

        print(
            f"     Entry MTF: "
            f"{entry_decision['mtf_bias']} "
            f"({entry_decision['mtf_score']})"
        )

        print(
            f"     Pre-exit MTF: "
            f"{exit_decision['mtf_bias']} "
            f"({exit_decision['mtf_score']})"
        )

        print(
            f"     Regime: {entry_regime} -> {exit_regime}"
            f" | MFE: {mfe}"
            f" | MAE: {mae}"
        )

        for timeframe in TIMEFRAME_ORDER:
            entry_signal = entry_decision["signals"].get(
                timeframe
            )

            exit_signal = exit_decision["signals"].get(
                timeframe
            )

            entry_score = (
                entry_signal.score
                if entry_signal is not None
                else "N/A"
            )

            exit_score = (
                exit_signal.score
                if exit_signal is not None
                else "N/A"
            )

            print(
                f"     {timeframe}: "
                f"Entry {entry_score} -> "
                f"Pre-exit {exit_score}"
            )

    if not rows:
        raise RuntimeError(
            "No trades found in forensic analysis"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n========== SIGNAL FORENSICS COMPLETED ==========")
    print(f"Analyzed trades: {len(rows)}")
    print(f"CSV output: {OUTPUT_FILE}")
    print(f"Columns: {len(rows[0])}")


if __name__ == "__main__":
    main()
