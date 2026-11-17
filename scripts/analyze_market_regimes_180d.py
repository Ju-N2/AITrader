from bisect import bisect_right
from collections import Counter, defaultdict
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
ATR_MULTIPLIER = 3.0
COOLDOWN_BARS = 15
DATA_LIMIT = 300_000

BULLISH = "bullish"
BEARISH = "bearish"
SIDEWAYS = "sideways_transition"
WARMUP = "warmup"

REGIMES = (BULLISH, BEARISH, SIDEWAYS)


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


def calculate_ema(values, period):
    result = [None] * len(values)

    if len(values) < period:
        return result

    multiplier = 2 / (period + 1)

    current = sum(values[:period]) / period
    result[period - 1] = current

    for index in range(period, len(values)):
        current = (
            values[index] * multiplier
            + current * (1 - multiplier)
        )
        result[index] = current

    return result


def build_market_regimes(repository, now_ms):
    interval_ms = TIMEFRAMES["4h"]

    candles = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        "4h",
        limit=DATA_LIMIT,
    )

    closed = [
        candle
        for candle in candles
        if candle.timestamp_ms + interval_ms <= now_ms
    ]

    if len(closed) < 200:
        raise RuntimeError(
            f"Insufficient 4h candles for regime analysis: {len(closed)}"
        )

    closes = [candle.close for candle in closed]

    ema_50 = calculate_ema(closes, 50)
    ema_200 = calculate_ema(closes, 200)

    close_times = []
    regimes = []

    for candle, fast, slow in zip(closed, ema_50, ema_200):
        close_times.append(
            candle.timestamp_ms + interval_ms
        )

        if fast is None or slow is None:
            regimes.append(WARMUP)

        elif candle.close > slow and fast > slow:
            regimes.append(BULLISH)

        elif candle.close < slow and fast < slow:
            regimes.append(BEARISH)

        else:
            regimes.append(SIDEWAYS)

    print("\n========== MARKET REGIME MODEL ==========")
    print("Source: Binance Spot BTC/USDT 4h")
    print("Fast EMA: 50")
    print("Slow EMA: 200")
    print(f"Completed 4h candles: {len(closed):,}")
    print(f"Start: {closed[0].datetime_utc}")
    print(f"End: {closed[-1].datetime_utc}")

    return close_times, regimes


def get_regime_at(decision_time, close_times, regimes):
    index = bisect_right(close_times, decision_time) - 1

    if index < 0:
        return WARMUP

    signal_time = close_times[index]

    if decision_time - signal_time > TIMEFRAMES["4h"] * 3:
        return "unavailable"

    return regimes[index]


def build_signal_series(repository, timeframe, interval_ms, now_ms):
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

    print("\n========== BUILDING MTF SIGNALS ==========")

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
        limit=DATA_LIMIT,
    )

    candles_1m = [
        candle
        for candle in candles_1m
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

    print(f"\nAligned candles: {len(aligned_candles):,}")
    print(f"Unavailable MTF decisions: {unavailable}")

    if len(aligned_candles) < 100:
        raise RuntimeError("Insufficient aligned data")

    return aligned_candles, aligned_signals


def create_config():
    return BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        fee_rate=0.001,
        slippage_rate=0.0005,
        position_fraction=0.25,
        atr_stop_multiplier=ATR_MULTIPLIER,
        atr_cooldown_bars=COOLDOWN_BARS,
    )


def calculate_profit_factor(trades):
    gross_profit = sum(
        trade.pnl for trade in trades if trade.pnl > 0
    )

    gross_loss = abs(sum(
        trade.pnl for trade in trades if trade.pnl < 0
    ))

    if gross_loss == 0:
        return None

    return gross_profit / gross_loss


def print_backtest_result(label, result):
    profit_factor = (
        f"{result.profit_factor:.3f}"
        if result.profit_factor is not None
        else "N/A"
    )

    strategy_exits = sum(
        trade.exit_reason == "strategy"
        for trade in result.trades
    )

    costs = result.total_fees + result.total_slippage_cost

    print(f"\n--- {label} ---")
    print(f"Final Equity:       {result.final_equity:,.2f} USDT")
    print(f"Total Return:       {result.total_return_pct:+.2f}%")
    print(f"Maximum Drawdown:   {result.max_drawdown_pct:.2f}%")
    print(f"Trades:             {result.total_trades}")
    print(f"Win Rate:           {result.win_rate_pct:.2f}%")
    print(f"Profit Factor:      {profit_factor}")
    print(f"ATR Stops:          {result.stop_loss_count}")
    print(f"Strategy Exits:     {strategy_exits}")
    print(f"Trading Costs:      {costs:.2f} USDT")


def analyze_entry_regimes(
    result,
    regime_by_decision,
):
    grouped = defaultdict(list)

    for trade in result.trades:
        regime = regime_by_decision.get(
            trade.entry_timestamp_ms,
            "unmapped",
        )

        grouped[regime].append(trade)

    print("\n========== BASELINE TRADE ATTRIBUTION ==========")
    print("Trades are grouped by market regime at entry time.")

    print(
        f"{'Regime':<22}"
        f"{'Trades':>8}"
        f"{'WinRate':>10}"
        f"{'PF':>9}"
        f"{'ATR Stops':>12}"
        f"{'Total PnL':>15}"
    )

    for regime in (*REGIMES, WARMUP, "unavailable", "unmapped"):
        trades = grouped.get(regime, [])

        if not trades:
            continue

        wins = sum(trade.pnl > 0 for trade in trades)
        win_rate = wins / len(trades) * 100
        profit_factor = calculate_profit_factor(trades)

        pf_text = (
            f"{profit_factor:.3f}"
            if profit_factor is not None
            else "N/A"
        )

        stops = sum(
            trade.exit_reason == "atr_stop"
            for trade in trades
        )

        total_pnl = sum(trade.pnl for trade in trades)

        print(
            f"{regime:<22}"
            f"{len(trades):>8}"
            f"{win_rate:>9.2f}%"
            f"{pf_text:>9}"
            f"{stops:>12}"
            f"{total_pnl:>+14.2f}"
        )


def filter_entries_by_regime(signals, candles, regimes, allowed_regime):
    filtered = []

    for candle, signal, regime in zip(candles, signals, regimes):
        if (
            signal.bias == MarketBias.BULLISH
            and regime != allowed_regime
        ):
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
                        f"Entry blocked: regime is {regime}",
                    ),
                    atr=signal.atr,
                )
            )

        else:
            filtered.append(signal)

    return filtered


def main():
    print("\n========== AITrader MARKET REGIME ANALYSIS ==========")
    print(f"Exchange: {EXCHANGE}")
    print(f"Symbol: {SYMBOL}")
    print(f"ATR Multiplier: {ATR_MULTIPLIER}")
    print(f"Cooldown: {COOLDOWN_BARS} bars")
    print("Period: 180-day database history")

    repository = SQLiteCandleRepository("data/aitrader.db")
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    regime_close_times, regime_values = build_market_regimes(
        repository,
        now_ms,
    )

    candles, signals = build_aligned_data(
        repository,
        now_ms,
    )

    regimes = []
    regime_by_decision = {}

    for candle in candles:
        decision_time = candle.timestamp_ms + 60_000

        regime = get_regime_at(
            decision_time,
            regime_close_times,
            regime_values,
        )

        regimes.append(regime)
        regime_by_decision[decision_time] = regime

    regime_counts = Counter(regimes)

    print("\n========== MARKET REGIME DISTRIBUTION ==========")

    for regime in (*REGIMES, WARMUP, "unavailable"):
        count = regime_counts[regime]

        percentage = (
            count / len(regimes) * 100
            if regimes else 0
        )

        print(
            f"{regime:<22}"
            f"{count:>10,} "
            f"({percentage:>6.2f}%)"
        )

    engine = BacktestEngine()
    config = create_config()

    # 1. 기존 ATR 3.0 전략의 전체 기간 기준선
    baseline = engine.run(
        candles,
        signals,
        config,
    )

    print("\n========== BASELINE: NO REGIME FILTER ==========")
    print_backtest_result("Full 180D Strategy", baseline)

    # 2. 실제 거래를 진입 당시 시장 국면별로 분류
    analyze_entry_regimes(
        baseline,
        regime_by_decision,
    )

    # 3. 각 국면에서만 신규 매수를 허용하는 독립 실험
    print("\n========== REGIME ENTRY FILTER EXPERIMENTS ==========")
    print(
        "Bearish strategy exits remain active in every experiment."
    )
    print(
        "Only new bullish entries outside the selected regime are blocked."
    )

    for allowed_regime in REGIMES:
        filtered_signals = filter_entries_by_regime(
            signals,
            candles,
            regimes,
            allowed_regime,
        )

        result = engine.run(
            candles,
            filtered_signals,
            config,
        )

        print_backtest_result(
            f"Entries Allowed Only: {allowed_regime}",
            result,
        )

    print("\n========== ANALYSIS COMPLETED ==========")
    print("No existing strategy or engine files were modified.")


if __name__ == "__main__":
    main()
