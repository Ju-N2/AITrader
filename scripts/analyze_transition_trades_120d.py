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
    calculate_ema,
)

from aitrader.backtesting.engine import BacktestEngine
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine


EVALUATION_DAYS = 120
OUTPUT_FILE = Path("data/transition_trade_forensics_120d.csv")


def load_indicator_series(repository, timeframe, interval_ms, now_ms):
    candles = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        timeframe,
        limit=DATA_LIMIT,
    )

    candles = [
        candle for candle in candles
        if candle.timestamp_ms + interval_ms <= now_ms
    ]

    snapshots = IndicatorEngine().calculate(candles)

    close_times = [
        candle.timestamp_ms + interval_ms
        for candle in candles
    ]

    return candles, close_times, snapshots


def snapshot_at(close_times, snapshots, timestamp_ms):
    index = bisect_right(close_times, timestamp_ms) - 1

    if index < 0:
        return None

    return snapshots[index]


def value(snapshot, name):
    if snapshot is None:
        return None

    return getattr(snapshot, name)


def fmt(value, digits=2):
    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


def utc_string(timestamp_ms):
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        UTC,
    ).strftime("%Y-%m-%d %H:%M:%S")


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

    mfe = (highest / trade.entry_price - 1) * 100
    mae = (lowest / trade.entry_price - 1) * 100

    return mfe, mae


def main():
    print("\n========== TRANSITION TRADE FORENSICS ==========")

    repository = SQLiteCandleRepository("data/aitrader.db")
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    # 동일한 검증 데이터와 시장 레짐 구성
    regime_times, _, detailed_regimes = build_market_regimes(
        repository,
        now_ms,
    )

    full_candles, full_signals = build_aligned_data(
        repository,
        now_ms,
    )

    data_end_ms = full_candles[-1].timestamp_ms + 60_000
    evaluation_end_ms = data_end_ms - 180 * DAY_MS
    evaluation_start_ms = (
        evaluation_end_ms - EVALUATION_DAYS * DAY_MS
    )

    selected = [
        i for i, candle in enumerate(full_candles)
        if evaluation_start_ms <= candle.timestamp_ms < evaluation_end_ms
    ]

    candles = [full_candles[i] for i in selected]
    signals = [full_signals[i] for i in selected]

    if len(candles) != EVALUATION_DAYS * 24 * 60:
        raise RuntimeError(
            f"Unexpected evaluation candle count: {len(candles)}"
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

    print(f"\nEvaluation: {utc_string(candles[0].timestamp_ms)}")
    print(f"             ~ {utc_string(evaluation_end_ms)}")
    print(f"Trades: {len(result.trades)}")

    # 진입 시점에 이미 종료된 상위 시간봉 지표 계산
    indicator_data = {}

    for timeframe in ("15m", "1h", "4h"):
        indicator_data[timeframe] = load_indicator_series(
            repository,
            timeframe,
            TIMEFRAMES[timeframe],
            now_ms,
        )

    candle_index = {
        candle.timestamp_ms: i
        for i, candle in enumerate(candles)
    }

    rows = []

    for number, trade in enumerate(result.trades, start=1):
        entry_index = candle_index.get(trade.entry_timestamp_ms)
        exit_index = candle_index.get(trade.exit_timestamp_ms)

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

        duration_minutes = (
            trade.exit_timestamp_ms - trade.entry_timestamp_ms
        ) / 60_000

        row = {
            "trade": number,
            "entry_time_utc": utc_string(trade.entry_timestamp_ms),
            "exit_time_utc": utc_string(trade.exit_timestamp_ms),
            "duration_minutes": round(duration_minutes, 2),
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "quantity": trade.quantity,
            "pnl_usdt": trade.pnl,
            "return_pct": trade.return_pct,
            "exit_reason": trade.exit_reason,
            "entry_regime": entry_regime,
            "exit_regime": exit_regime,
        }

        # 시간봉별 진입 시점 지표
        for timeframe, (_, close_times, snapshots) in indicator_data.items():
            snapshot = snapshot_at(
                close_times,
                snapshots,
                trade.entry_timestamp_ms,
            )

            for name in (
                "ema_12",
                "ema_26",
                "rsi_14",
                "atr_14",
            ):
                row[f"{timeframe}_{name}"] = value(snapshot, name)

        # 4시간봉 EMA50 / EMA200
        h4_candles, h4_close_times, _ = indicator_data["4h"]

        h4_index = bisect_right(
            h4_close_times,
            trade.entry_timestamp_ms,
        ) - 1

        if h4_index >= 0:
            completed_h4 = h4_candles[:h4_index + 1]
            closes = [c.close for c in completed_h4]

            ema50 = calculate_ema(closes, 50)[-1]
            ema200 = calculate_ema(closes, 200)[-1]
        else:
            ema50 = None
            ema200 = None

        row["4h_ema_50"] = ema50
        row["4h_ema_200"] = ema200

        # MFE / MAE 계산
        # 진입 캔들은 포함. 시가 청산된 청산 캔들은 제외.
        include_exit = True

        if trade.exit_reason == "strategy":
            include_exit = False

        elif "stop" in trade.exit_reason.lower():
            # ATR 스톱의 실제 체결 가격을 기준으로
            # 갭 하락 시 시가 청산인지 확인
            base_exit_price = (
                trade.exit_price / (1 - create_config().slippage_rate)
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
            f"     Entry {trade.entry_price:.2f}"
            f" -> Exit {trade.exit_price:.2f}"
            f" | {duration_minutes:.1f} min"
        )

        print(
            f"     Regime {entry_regime} -> {exit_regime}"
            f" | MFE {fmt(mfe)}%"
            f" | MAE {fmt(mae)}%"
        )

        for timeframe in ("15m", "1h", "4h"):
            print(
                f"     {timeframe}: "
                f"EMA12={fmt(row[f'{timeframe}_ema_12'])}, "
                f"EMA26={fmt(row[f'{timeframe}_ema_26'])}, "
                f"RSI={fmt(row[f'{timeframe}_rsi_14'])}, "
                f"ATR={fmt(row[f'{timeframe}_atr_14'])}"
            )

        print(
            f"     4h EMA50={fmt(ema50)}, "
            f"EMA200={fmt(ema200)}"
        )

    if not rows:
        raise RuntimeError("No trades found in forensic analysis")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

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

    print("\n========== FORENSICS COMPLETED ==========")
    print(f"Analyzed trades: {len(rows)}")
    print(f"CSV output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
