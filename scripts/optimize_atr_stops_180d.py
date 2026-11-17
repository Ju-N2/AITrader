from bisect import bisect_right
from datetime import UTC, datetime

from aitrader.backtesting.engine import BacktestEngine
from aitrader.backtesting.models import BacktestConfig
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine
from aitrader.strategy.models import MarketBias, StrategySignal
from aitrader.strategy.multi_timeframe import MultiTimeframeEngine
from aitrader.strategy.technical import TechnicalTrendStrategy

EXCHANGE = "binance_spot"
SYMBOL = "BTC/USDT"

TIMEFRAMES = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}

INITIAL_CAPITAL = 10_000.0
ATR_MULTIPLIERS = [2.0, 2.5, 3.0]
COOLDOWN_BARS = 15


def indicator_ready(snapshot):
    return all(value is not None for value in [
        snapshot.sma_20,
        snapshot.sma_50,
        snapshot.ema_12,
        snapshot.ema_26,
        snapshot.rsi_14,
        snapshot.macd_histogram,
        snapshot.atr_14,
    ])


def build_signal_series(repository, timeframe, interval_ms, now_ms):
    candles = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        timeframe,
        limit=300000,
    )

    closed = [
        candle for candle in candles
        if candle.timestamp_ms + interval_ms <= now_ms
    ]

    if len(closed) < 50:
        raise RuntimeError(
            f"Insufficient data for {timeframe}: {len(closed)}"
        )

    snapshots = IndicatorEngine().calculate(closed)
    strategy = TechnicalTrendStrategy()

    close_times = []
    signals = []

    for candle, snapshot in zip(closed, snapshots):
        if not indicator_ready(snapshot):
            continue

        signal = strategy.evaluate(candle, snapshot)

        close_times.append(
            candle.timestamp_ms + interval_ms
        )
        signals.append(signal)

    print(f"{timeframe}: {len(signals)} completed signals")

    return close_times, signals


def build_aligned_data(repository, now_ms):
    signal_series = {}

    print("\n========== BUILDING SIGNALS ==========")

    for timeframe, interval_ms in TIMEFRAMES.items():
        signal_series[timeframe] = build_signal_series(
            repository,
            timeframe,
            interval_ms,
            now_ms,
        )

    candles_1m = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        "1m",
        limit=300000,
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
            close_times, signals = signal_series[timeframe]

            index = bisect_right(
                close_times,
                decision_time,
            ) - 1

            if index < 0:
                break

            signal_time = close_times[index]
            signal_age = decision_time - signal_time

            if signal_age > interval_ms * 3:
                break

            current_signals[timeframe] = signals[index]

        if len(current_signals) != len(TIMEFRAMES):
            unavailable += 1

            combined_signal = StrategySignal(
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
                    if "15m" in current_signals
                    else None
                ),
            )
        else:
            result = multi_engine.evaluate(current_signals)

            combined_signal = StrategySignal(
                exchange=EXCHANGE,
                symbol=SYMBOL,
                timeframe="1m",
                timestamp_ms=candle.timestamp_ms,
                bias=result.bias,
                score=max(
                    -5,
                    min(5, round(result.weighted_score))
                ),
                max_score=5,
                reasons=result.reasons,
                atr=current_signals["15m"].atr,
            )

        aligned_candles.append(candle)
        aligned_signals.append(combined_signal)

    print(f"\nAligned candles: {len(aligned_candles)}")
    print(f"Unavailable decisions: {unavailable}")

    if len(aligned_candles) < 100:
        raise RuntimeError("Insufficient aligned data")

    return aligned_candles, aligned_signals


def run_experiment(candles, signals, multiplier):
    config = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        fee_rate=0.001,
        slippage_rate=0.0005,
        position_fraction=0.25,
        atr_stop_multiplier=multiplier,
        atr_cooldown_bars=COOLDOWN_BARS,
    )

    result = BacktestEngine().run(
        candles,
        signals,
        config,
    )

    average_pnl = (
        sum(trade.pnl for trade in result.trades)
        / result.total_trades
        if result.total_trades else 0.0
    )

    strategy_exits = sum(
        1 for trade in result.trades
        if trade.exit_reason != "atr_stop"
    )

    return {
        "multiplier": multiplier,
        "result": result,
        "average_pnl": average_pnl,
        "strategy_exits": strategy_exits,
    }


def print_result(experiment):
    result = experiment["result"]

    total_cost = (
        result.total_fees
        + result.total_slippage_cost
    )

    print(f"\n{'=' * 58}")
    print(f"ATR MULTIPLIER: {experiment['multiplier']:.1f}")
    print(f"{'=' * 58}")

    print(f"Final Equity:       {result.final_equity:,.2f} USDT")
    print(f"Total Return:       {result.total_return_pct:+.2f}%")
    print(f"Maximum Drawdown:   {result.max_drawdown_pct:.2f}%")
    print(f"Total Trades:       {result.total_trades}")
    print(f"Win Rate:           {result.win_rate_pct:.2f}%")
    print(f"Profit Factor:      {result.profit_factor}")
    print(f"ATR Stop Losses:    {result.stop_loss_count}")
    print(f"Strategy Exits:     {experiment['strategy_exits']}")
    print(f"Average Trade PnL:  {experiment['average_pnl']:+.2f} USDT")
    print(f"Total Fees:         {result.total_fees:.2f} USDT")
    print(f"Slippage Cost:      {result.total_slippage_cost:.2f} USDT")
    print(f"Total Trading Cost: {total_cost:.2f} USDT")
    print(f"Cooldown Blocked:   {result.cooldown_blocked_entries}")

    # 청산 유형별 성과 분석
    grouped = {}

    for trade in result.trades:
        bucket = grouped.setdefault(
            trade.exit_reason,
            {
                "count": 0,
                "wins": 0,
                "pnl": 0.0,
                "returns": 0.0,
                "holding_ms": 0,
            },
        )

        bucket["count"] += 1
        bucket["wins"] += trade.pnl > 0
        bucket["pnl"] += trade.pnl
        bucket["returns"] += trade.return_pct
        bucket["holding_ms"] += (
            trade.exit_timestamp_ms - trade.entry_timestamp_ms
        )

    print("\\nExit Reason Breakdown:")

    for reason, bucket in sorted(grouped.items()):
        count = bucket["count"]

        print(f"  Reason: {reason}")
        print(f"    Trades: {count}")
        print(f"    Total PnL: {bucket['pnl']:+.2f} USDT")
        print(f"    Average PnL: {bucket['pnl'] / count:+.2f} USDT")
        print(f"    Win Rate: {bucket['wins'] / count * 100:.2f}%")
        print(f"    Average Return: {bucket['returns'] / count:+.2f}%")
        print(
            f"    Average Holding: "
            f"{bucket['holding_ms'] / count / 60000:.1f} minutes"
        )



def analyze_stop_recovery(candles, experiments):
    print("\\n========== POST-STOP PRICE RECOVERY ==========")
    print("Note: Price movement after exit, NOT hypothetical trade PnL.")

    timestamps = [candle.timestamp_ms for candle in candles]

    horizons = {
        "15m": 15 * 60_000,
        "1h": 60 * 60_000,
        "4h": 4 * 60 * 60_000,
    }

    for experiment in experiments:
        multiplier = experiment["multiplier"]

        stop_trades = [
            trade
            for trade in experiment["result"].trades
            if trade.exit_reason == "atr_stop"
        ]

        print(f"\\n--- ATR {multiplier:.1f} ---")
        print(f"ATR Stop Trades: {len(stop_trades)}")

        for label, horizon_ms in horizons.items():
            returns = []

            for trade in stop_trades:
                target_time = trade.exit_timestamp_ms + horizon_ms

                # 목표 시점까지 완전히 종료된 1분봉 중 가장 가까운 봉
                target_index = bisect_right(
                    timestamps,
                    target_time - 60_000,
                ) - 1

                exit_index = bisect_right(
                    timestamps,
                    trade.exit_timestamp_ms,
                ) - 1

                if (
                    target_index < exit_index
                    or target_index < 0
                    or trade.exit_price <= 0
                ):
                    continue

                target_price = candles[target_index].close

                recovery_pct = (
                    target_price / trade.exit_price - 1
                ) * 100

                returns.append(recovery_pct)

            if not returns:
                print(f"{label}: No available observations")
                continue

            positive = sum(value > 0 for value in returns)
            average = sum(returns) / len(returns)

            print(
                f"{label}: "
                f"Available={len(returns)}, "
                f"Above Exit={positive}/{len(returns)}, "
                f"Recovery Rate={positive / len(returns) * 100:.1f}%, "
                f"Average Change={average:+.3f}%"
            )


def main():
    print("\n========== AITrader 180D ATR OPTIMIZATION ==========")
    print(f"Exchange: {EXCHANGE}")
    print(f"Symbol: {SYMBOL}")
    print(f"ATR Multipliers: {ATR_MULTIPLIERS}")
    print(f"Cooldown: {COOLDOWN_BARS} bars")

    repository = SQLiteCandleRepository("data/aitrader.db")

    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    candles, signals = build_aligned_data(
        repository,
        now_ms,
    )

    experiments = []

    for multiplier in ATR_MULTIPLIERS:
        print(f"\nRunning ATR {multiplier:.1f} experiment...")

        experiment = run_experiment(
            candles,
            signals,
            multiplier,
        )

        experiments.append(experiment)
        print_result(experiment)

    print("\n\n========== COMPARISON SUMMARY ==========")

    print(
        f"{'ATR':>5} "
        f"{'Return':>10} "
        f"{'MDD':>8} "
        f"{'Trades':>8} "
        f"{'WinRate':>9} "
        f"{'PF':>8} "
        f"{'Stops':>7} "
        f"{'Avg PnL':>11}"
    )

    for experiment in experiments:
        result = experiment["result"]

        pf = (
            f"{result.profit_factor:.3f}"
            if result.profit_factor is not None
            else "N/A"
        )

        print(
            f"{experiment['multiplier']:>5.1f} "
            f"{result.total_return_pct:>+9.2f}% "
            f"{result.max_drawdown_pct:>7.2f}% "
            f"{result.total_trades:>8} "
            f"{result.win_rate_pct:>8.2f}% "
            f"{pf:>8} "
            f"{result.stop_loss_count:>7} "
            f"{experiment['average_pnl']:>+10.2f}"
        )

    analyze_stop_recovery(candles, experiments)

    print("\n========== OPTIMIZATION COMPLETED ==========")
    print("No existing strategy or engine files were modified.")


if __name__ == "__main__":
    main()
