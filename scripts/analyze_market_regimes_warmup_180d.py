from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import pairwise

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

DAY_MS = 86_400_000
DATA_LIMIT = 700_000
EVALUATION_DAYS = 180
MIN_WARMUP_DAYS = 150

INITIAL_CAPITAL = 10_000.0
ATR_MULTIPLIER = 3.0
COOLDOWN_BARS = 15

BULLISH = "bullish"
BEARISH = "bearish"
SIDEWAYS = "sideways_transition"
BULLISH_TRANSITION = "bullish_transition"
BEARISH_TRANSITION = "bearish_transition"
NEUTRAL_TRANSITION = "neutral_transition"

TRANSITION_REGIMES = (
    BULLISH_TRANSITION,
    BEARISH_TRANSITION,
    NEUTRAL_TRANSITION,
)

WARMUP = "warmup"


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


def validate_continuity(candles, timeframe, interval_ms):
    if len(candles) < 200:
        raise RuntimeError(
            f"Insufficient {timeframe} data: {len(candles)}"
        )

    gaps = sum(
        current.timestamp_ms - previous.timestamp_ms != interval_ms
        for previous, current in pairwise(candles)
    )

    print(
        f"{timeframe}: {len(candles):,} candles | "
        f"Time gaps: {gaps}"
    )

    if gaps:
        raise RuntimeError(
            f"Data gaps detected in {timeframe}: {gaps}"
        )


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

    validate_continuity(closed, "4h", interval_ms)

    closes = [candle.close for candle in closed]

    ema_50 = calculate_ema(closes, 50)
    ema_200 = calculate_ema(closes, 200)

    close_times = []
    regimes = []
    detailed_regimes = []

    for candle, fast, slow in zip(closed, ema_50, ema_200):
        close_times.append(
            candle.timestamp_ms + interval_ms
        )

        if fast is None or slow is None:
            regimes.append(WARMUP)
            detailed_regimes.append(WARMUP)

        elif candle.close > slow and fast > slow:
            regimes.append(BULLISH)
            detailed_regimes.append(BULLISH)

        elif candle.close < slow and fast < slow:
            regimes.append(BEARISH)
            detailed_regimes.append(BEARISH)

        else:
            regimes.append(SIDEWAYS)

            if candle.close > slow and fast <= slow:
                detailed_regimes.append(BULLISH_TRANSITION)

            elif candle.close < slow and fast >= slow:
                detailed_regimes.append(BEARISH_TRANSITION)

            else:
                detailed_regimes.append(NEUTRAL_TRANSITION)

    print("\n========== MARKET REGIME MODEL ==========")
    print("Source: Binance Spot BTC/USDT 4h")
    print("Fast EMA: 50")
    print("Slow EMA: 200")
    print(f"Completed 4h candles: {len(closed):,}")

    return close_times, regimes, detailed_regimes


def get_regime_at(decision_time, close_times, regimes):
    index = bisect_right(close_times, decision_time) - 1

    if index < 0:
        return WARMUP

    if decision_time - close_times[index] > TIMEFRAMES["4h"] * 3:
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

    validate_continuity(closed, timeframe, interval_ms)

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

    print(
        f"{timeframe}: {len(signals):,} completed signals"
    )

    return close_times, signals


def build_aligned_data(repository, now_ms):
    signal_series = {}

    print("\n========== BUILDING FULL HISTORY SIGNALS ==========")

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

    validate_continuity(candles_1m, "1m", 60_000)

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

            if decision_time - signal_time > interval_ms * 3:
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

    print(f"\nFull aligned candles: {len(aligned_candles):,}")
    print(f"Full history unavailable decisions: {unavailable}")

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


def print_result(label, result):
    pf = (
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
    print(f"Profit Factor:      {pf}")
    print(f"ATR Stops:          {result.stop_loss_count}")
    print(f"Strategy Exits:     {strategy_exits}")
    print(f"Trading Costs:      {costs:.2f} USDT")


def analyze_entry_regimes(result, regime_by_decision):
    grouped = defaultdict(list)

    for trade in result.trades:
        regime = regime_by_decision.get(
            trade.entry_timestamp_ms,
            "unmapped",
        )

        grouped[regime].append(trade)

    print("\n========== BASELINE TRADE ATTRIBUTION ==========")

    print(
        f"{'Regime':<22}"
        f"{'Trades':>8}"
        f"{'WinRate':>10}"
        f"{'PF':>9}"
        f"{'ATR Stops':>12}"
        f"{'Total PnL':>15}"
    )

    for regime in (BULLISH, BEARISH, SIDEWAYS, "unmapped"):
        trades = grouped.get(regime, [])

        if not trades:
            continue

        wins = sum(trade.pnl > 0 for trade in trades)
        win_rate = wins / len(trades) * 100

        pf = calculate_profit_factor(trades)

        pf_text = f"{pf:.3f}" if pf is not None else "N/A"

        stops = sum(
            trade.exit_reason == "atr_stop"
            for trade in trades
        )

        pnl = sum(trade.pnl for trade in trades)

        print(
            f"{regime:<22}"
            f"{len(trades):>8}"
            f"{win_rate:>9.2f}%"
            f"{pf_text:>9}"
            f"{stops:>12}"
            f"{pnl:>+14.2f}"
        )




def analyze_transition_entry_attribution(
    result,
    regime_by_decision,
    title,
):
    grouped = defaultdict(list)

    for trade in result.trades:
        regime = regime_by_decision.get(
            trade.entry_timestamp_ms,
            "unmapped",
        )
        grouped[regime].append(trade)

    print(f"\\n========== {title} ==========")

    print(
        f"{'Regime':<24}"
        f"{'Trades':>8}"
        f"{'WinRate':>10}"
        f"{'PF':>9}"
        f"{'ATR Stops':>12}"
        f"{'Total PnL':>15}"
    )

    for regime in (
        BULLISH,
        BEARISH,
        BULLISH_TRANSITION,
        BEARISH_TRANSITION,
        NEUTRAL_TRANSITION,
        "unmapped",
    ):
        trades = grouped.get(regime, [])

        if not trades:
            continue

        wins = sum(trade.pnl > 0 for trade in trades)
        win_rate = wins / len(trades) * 100

        pf = calculate_profit_factor(trades)
        pf_text = f"{pf:.3f}" if pf is not None else "N/A"

        stops = sum(
            trade.exit_reason == "atr_stop"
            for trade in trades
        )

        pnl = sum(trade.pnl for trade in trades)

        print(
            f"{regime:<24}"
            f"{len(trades):>8}"
            f"{win_rate:>9.2f}%"
            f"{pf_text:>9}"
            f"{stops:>12}"
            f"{pnl:>+14.2f}"
        )


def filter_entries_by_regime(signals, regimes, allowed_regime):
    filtered = []

    for signal, regime in zip(signals, regimes):
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



def print_trade_audit(trades):
    print("\\n========== SIDEWAYS TRANSITION TRADE AUDIT ==========")

    winners = [trade for trade in trades if trade.pnl > 0]
    losers = [trade for trade in trades if trade.pnl < 0]

    gross_profit = sum(trade.pnl for trade in winners)
    gross_loss = abs(sum(trade.pnl for trade in losers))

    largest_winner = max(
        winners,
        key=lambda trade: trade.pnl,
        default=None,
    )

    if largest_winner and gross_profit > 0:
        winner_concentration = (
            largest_winner.pnl / gross_profit * 100
        )
    else:
        winner_concentration = 0.0

    print(f"Total Trades: {len(trades)}")
    print(f"Winners: {len(winners)}")
    print(f"Losers: {len(losers)}")
    print(f"Gross Profit: {gross_profit:+.2f} USDT")
    print(f"Gross Loss: {-gross_loss:+.2f} USDT")
    print(f"Profit Factor: {calculate_profit_factor(trades)}")
    print(f"Largest Winner Share: {winner_concentration:.2f}%")

    print("\\nTrade Details:")

    for number, trade in enumerate(trades, 1):
        entry_time = datetime.fromtimestamp(
            trade.entry_timestamp_ms / 1000,
            UTC,
        )

        exit_time = datetime.fromtimestamp(
            trade.exit_timestamp_ms / 1000,
            UTC,
        )

        holding_minutes = (
            trade.exit_timestamp_ms - trade.entry_timestamp_ms
        ) / 60_000

        print(f"\\n--- Trade {number} ---")
        print(f"Entry: {entry_time}")
        print(f"Exit: {exit_time}")
        print(f"Exit Reason: {trade.exit_reason}")
        print(f"Entry Price: {trade.entry_price:.2f}")
        print(f"Exit Price: {trade.exit_price:.2f}")
        print(f"PnL: {trade.pnl:+.2f} USDT")
        print(f"Return: {trade.return_pct:+.3f}%")
        print(f"Holding: {holding_minutes:.1f} minutes")


def run_buy_hold(candles):
    fee_rate = 0.001
    slippage_rate = 0.0005
    position_fraction = 0.25

    budget = INITIAL_CAPITAL * position_fraction

    entry_price = candles[1].open * (1 + slippage_rate)
    notional = budget / (1 + fee_rate)
    entry_fee = notional * fee_rate
    quantity = notional / entry_price

    cash = INITIAL_CAPITAL - notional - entry_fee

    exit_price = candles[-1].close * (1 - slippage_rate)
    gross = quantity * exit_price
    exit_fee = gross * fee_rate

    final_equity = cash + gross - exit_fee

    btc_change = (
        candles[-1].close / candles[1].open - 1
    ) * 100

    net_return = (
        final_equity / INITIAL_CAPITAL - 1
    ) * 100

    print("\n========== BUY & HOLD BENCHMARK ==========")
    print(f"BTC Price Change: {btc_change:+.2f}%")
    print(f"Allocation: {position_fraction * 100:.0f}%")
    print(f"Final Equity: {final_equity:,.2f} USDT")
    print(f"Net Return: {net_return:+.2f}%")
    print(f"Trading Fees: {entry_fee + exit_fee:.2f} USDT")



def run_60d_segment_analysis(
    candles,
    signals,
    regimes,
    engine,
    config,
):
    segment_ms = 60 * DAY_MS
    expected_candles = 60 * 24 * 60
    start_ms = candles[0].timestamp_ms

    print()
    print("========== 60D SEGMENT ROBUSTNESS TEST ==========")
    print("Each segment uses independent initial capital.")
    print("Open positions are closed at segment boundaries.")

    for segment_number in range(3):
        segment_start = start_ms + segment_number * segment_ms
        segment_end = segment_start + segment_ms

        indices = [
            index
            for index, candle in enumerate(candles)
            if segment_start <= candle.timestamp_ms < segment_end
        ]

        segment_candles = [candles[i] for i in indices]
        segment_signals = [signals[i] for i in indices]
        segment_regimes = [regimes[i] for i in indices]

        if len(segment_candles) != expected_candles:
            raise RuntimeError(
                f"Segment {segment_number + 1}: "
                f"Expected {expected_candles}, "
                f"found {len(segment_candles)}"
            )

        baseline = engine.run(
            segment_candles,
            segment_signals,
            config,
        )

        sideways_signals = filter_entries_by_regime(
            segment_signals,
            segment_regimes,
            SIDEWAYS,
        )

        sideways = engine.run(
            segment_candles,
            sideways_signals,
            config,
        )

        start_time = datetime.fromtimestamp(
            segment_start / 1000,
            UTC,
        )

        end_time = datetime.fromtimestamp(
            segment_end / 1000,
            UTC,
        )

        baseline_pf = (
            f"{baseline.profit_factor:.3f}"
            if baseline.profit_factor is not None
            else "N/A"
        )

        sideways_pf = (
            f"{sideways.profit_factor:.3f}"
            if sideways.profit_factor is not None
            else "N/A"
        )

        print()
        print(f"--- SEGMENT {segment_number + 1} ---")
        print(f"Start: {start_time}")
        print(f"End:   {end_time}")
        print(f"Candles: {len(segment_candles):,}")

        print(
            f"Baseline: "
            f"Return={baseline.total_return_pct:+.2f}% | "
            f"MDD={baseline.max_drawdown_pct:.2f}% | "
            f"Trades={baseline.total_trades} | "
            f"WinRate={baseline.win_rate_pct:.2f}% | "
            f"PF={baseline_pf}"
        )

        print(
            f"Sideways: "
            f"Return={sideways.total_return_pct:+.2f}% | "
            f"MDD={sideways.max_drawdown_pct:.2f}% | "
            f"Trades={sideways.total_trades} | "
            f"WinRate={sideways.win_rate_pct:.2f}% | "
            f"PF={sideways_pf}"
        )

def analyze_trade_regime_paths(trades, candles, regimes):
    if len(candles) != len(regimes):
        raise RuntimeError("Candle and regime counts do not match")

    decision_times = [
        candle.timestamp_ms + 60_000
        for candle in candles
    ]

    print()
    print("========== TRADE REGIME TRANSITION AUDIT ==========")

    for number, trade in enumerate(trades, 1):
        start_index = bisect_right(
            decision_times,
            trade.entry_timestamp_ms - 1,
        )

        end_index = bisect_right(
            decision_times,
            trade.exit_timestamp_ms,
        )

        trade_regimes = regimes[start_index:end_index]

        if not trade_regimes:
            print(f"Trade {number}: No regime observations")
            continue

        counts = Counter(trade_regimes)
        total = len(trade_regimes)

        print()
        print(f"--- Trade {number} ---")
        print(
            f"Entry: "
            f"{datetime.fromtimestamp(trade.entry_timestamp_ms / 1000, UTC)}"
        )
        print(
            f"Exit: "
            f"{datetime.fromtimestamp(trade.exit_timestamp_ms / 1000, UTC)}"
        )
        print(f"PnL: {trade.pnl:+.2f} USDT")
        print(f"Exit Reason: {trade.exit_reason}")
        print(f"Entry Regime: {trade_regimes[0]}")
        print(f"Final Regime: {trade_regimes[-1]}")
        print(f"Observed 1m Decisions: {total}")

        print("Regime Duration:")

        for regime in (BULLISH, BEARISH, SIDEWAYS):
            count = counts[regime]

            if count:
                print(
                    f"  {regime}: "
                    f"{count} observations "
                    f"({count / total * 100:.1f}%)"
                )

        previous = trade_regimes[0]

        print("Regime Transitions:")

        transition_found = False

        for offset, current in enumerate(trade_regimes[1:], 1):
            if current != previous:
                transition_found = True

                change_time = decision_times[
                    start_index + offset
                ]

                print(
                    f"  {datetime.fromtimestamp(change_time / 1000, UTC)}: "
                    f"{previous} -> {current}"
                )

            previous = current

        if not transition_found:
            print("  No regime changes")
def main():
    print("\n========== AITrader WARMUP-ADJUSTED REGIME ANALYSIS ==========")
    print(f"Exchange: {EXCHANGE}")
    print(f"Symbol: {SYMBOL}")
    print(f"Evaluation: {EVALUATION_DAYS} days")
    print(f"ATR Multiplier: {ATR_MULTIPLIER}")
    print(f"Cooldown: {COOLDOWN_BARS} bars")

    repository = SQLiteCandleRepository("data/aitrader.db")

    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    (
        regime_close_times,
        regime_values,
        detailed_regime_values,
    ) = build_market_regimes(
        repository,
        now_ms,
    )

    full_candles, full_signals = build_aligned_data(
        repository,
        now_ms,
    )

    if not full_candles:
        raise RuntimeError("No aligned candles available")

    evaluation_end_ms = (
        full_candles[-1].timestamp_ms + 60_000
    )

    evaluation_start_ms = (
        evaluation_end_ms - EVALUATION_DAYS * DAY_MS
    )

    history_start_ms = full_candles[0].timestamp_ms

    warmup_days = (
        evaluation_start_ms - history_start_ms
    ) / DAY_MS

    print("\n========== EVALUATION WINDOW ==========")
    print(f"Warmup History: {warmup_days:.2f} days")

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

    expected_candles = EVALUATION_DAYS * 24 * 60

    if len(candles) != expected_candles:
        raise RuntimeError(
            f"Expected {expected_candles} evaluation candles, "
            f"found {len(candles)}"
        )

    start_time = datetime.fromtimestamp(
        candles[0].timestamp_ms / 1000,
        UTC,
    )

    end_time = datetime.fromtimestamp(
        evaluation_end_ms / 1000,
        UTC,
    )

    print(f"Evaluation Start: {start_time}")
    print(f"Evaluation End:   {end_time}")
    print(f"Evaluation Candles: {len(candles):,}")

    regimes = []
    regime_by_decision = {}
    detailed_regimes = []
    detailed_regime_by_decision = {}

    for candle in candles:
        decision_time = candle.timestamp_ms + 60_000

        regime = get_regime_at(
            decision_time,
            regime_close_times,
            regime_values,
        )

        regimes.append(regime)
        regime_by_decision[decision_time] = regime

        detailed_regime = get_regime_at(
            decision_time,
            regime_close_times,
            detailed_regime_values,
        )

        detailed_regimes.append(detailed_regime)
        detailed_regime_by_decision[decision_time] = detailed_regime

    regime_counts = Counter(regimes)
    detailed_regime_counts = Counter(detailed_regimes)

    warmup_count = regime_counts[WARMUP]
    unavailable_regimes = regime_counts["unavailable"]

    unavailable_signals = sum(
        "Insufficient timeframe data" in signal.reasons
        for signal in signals
    )

    print("\n========== EVALUATION DATA QUALITY ==========")
    print(f"Market Regime Warmup: {warmup_count}")
    print(f"Unavailable Regimes: {unavailable_regimes}")
    print(f"Unavailable MTF Decisions: {unavailable_signals}")

    detailed_warmup = detailed_regime_counts[WARMUP]
    detailed_unavailable = detailed_regime_counts["unavailable"]

    print(f"Detailed Regime Warmup: {detailed_warmup}")
    print(f"Detailed Regime Unavailable: {detailed_unavailable}")

    if (
        warmup_count
        or unavailable_regimes
        or unavailable_signals
        or detailed_warmup
        or detailed_unavailable
    ):
        raise RuntimeError(
            "Evaluation contains unavailable regime or MTF data. "
            "Do not interpret results until resolved."
        )

    print("\n========== MARKET REGIME DISTRIBUTION ==========")

    for regime in (BULLISH, BEARISH, SIDEWAYS):
        count = regime_counts[regime]
        percentage = count / len(regimes) * 100

        print(
            f"{regime:<22}"
            f"{count:>10,} "
            f"({percentage:>6.2f}%)"
        )

    print("\\n========== TRANSITION SUBTYPE DISTRIBUTION ==========")

    for regime in TRANSITION_REGIMES:
        count = detailed_regime_counts[regime]
        percentage = count / len(detailed_regimes) * 100

        print(
            f"{regime:<24}"
            f"{count:>10,} ({percentage:>6.2f}%)"
        )

    engine = BacktestEngine()
    config = create_config()

    baseline = engine.run(
        candles,
        signals,
        config,
    )

    print("\n========== BASELINE: NO REGIME FILTER ==========")
    print_result("Full 180D Strategy", baseline)

    analyze_entry_regimes(
        baseline,
        regime_by_decision,
    )

    analyze_transition_entry_attribution(
        baseline,
        detailed_regime_by_decision,
        "BASELINE DETAILED REGIME ATTRIBUTION",
    )

    run_buy_hold(candles)

    print("\n========== REGIME ENTRY FILTER EXPERIMENTS ==========")
    print("Only new bullish entries are filtered.")
    print("Existing positions can still exit normally.")

    for allowed_regime in (BULLISH, BEARISH, SIDEWAYS):
        filtered_signals = filter_entries_by_regime(
            signals,
            regimes,
            allowed_regime,
        )

        result = engine.run(
            candles,
            filtered_signals,
            config,
        )

        print_result(
            f"Entries Allowed Only: {allowed_regime}",
            result,
        )

        if allowed_regime == SIDEWAYS:
            print_trade_audit(result.trades)
            analyze_trade_regime_paths(result.trades, candles, regimes)

            analyze_transition_entry_attribution(
                result,
                detailed_regime_by_decision,
                "SIDEWAYS FILTER DETAILED ENTRY ATTRIBUTION",
            )

    print("\\n========== TRANSITION SUBTYPE FILTER EXPERIMENTS ==========")
    print("Only bullish entries in the selected subtype are allowed.")

    for allowed_transition in TRANSITION_REGIMES:
        filtered_signals = filter_entries_by_regime(
            signals,
            detailed_regimes,
            allowed_transition,
        )

        result = engine.run(
            candles,
            filtered_signals,
            config,
        )

        print_result(
            f"Entries Allowed Only: {allowed_transition}",
            result,
        )

        analyze_transition_entry_attribution(
            result,
            detailed_regime_by_decision,
            f"ENTRY ATTRIBUTION: {allowed_transition}",
        )

    run_60d_segment_analysis(
        candles,
        signals,
        regimes,
        engine,
        config,
    )

    print("\n========== ANALYSIS COMPLETED ==========")
    print("Warmup data was excluded from trading.")
    print("Only the latest 180 days were evaluated.")
    print("No existing strategy or engine files were modified.")


if __name__ == "__main__":
    main()
