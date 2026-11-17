from collections import Counter, defaultdict
from datetime import UTC, datetime

from analyze_market_regimes_warmup_180d import (
    BULLISH_TRANSITION,
    DAY_MS,
    EXCHANGE,
    SYMBOL,
    TIMEFRAMES,
    build_aligned_data,
    build_market_regimes,
    create_config,
    filter_entries_by_regime,
    get_regime_at,
)

from aitrader.backtesting.engine import BacktestEngine
from aitrader.data.storage import SQLiteCandleRepository

EVALUATION_DAYS = 120
MIN_WARMUP_DAYS = 35
EXPECTED_CANDLES = EVALUATION_DAYS * 24 * 60

INITIAL_CAPITAL = 10_000.0


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

    trading_costs = (
        result.total_fees
        + result.total_slippage_cost
    )

    print(f"\n--- {label} ---")
    print(f"Final Equity:       {result.final_equity:,.2f} USDT")
    print(f"Total Return:       {result.total_return_pct:+.2f}%")
    print(f"Maximum Drawdown:   {result.max_drawdown_pct:.2f}%")
    print(f"Trades:             {result.total_trades}")
    print(f"Win Rate:           {result.win_rate_pct:.2f}%")
    print(f"Profit Factor:      {pf}")
    print(f"ATR Stops:          {result.stop_loss_count}")
    print(f"Strategy Exits:     {strategy_exits}")
    print(f"Trading Costs:      {trading_costs:.2f} USDT")


def print_entry_attribution(result, regime_map, title):
    grouped = defaultdict(list)

    for trade in result.trades:
        regime = regime_map.get(
            trade.entry_timestamp_ms,
            "unmapped",
        )
        grouped[regime].append(trade)

    print(f"\n========== {title} ==========")

    for regime, trades in sorted(grouped.items()):
        wins = sum(trade.pnl > 0 for trade in trades)

        win_rate = wins / len(trades) * 100
        total_pnl = sum(trade.pnl for trade in trades)

        gross_profit = sum(
            trade.pnl for trade in trades if trade.pnl > 0
        )

        gross_loss = abs(sum(
            trade.pnl for trade in trades if trade.pnl < 0
        ))

        pf = (
            f"{gross_profit / gross_loss:.3f}"
            if gross_loss > 0
            else "N/A"
        )

        print(
            f"{regime:<24} "
            f"Trades={len(trades):>3} | "
            f"WinRate={win_rate:>6.2f}% | "
            f"PF={pf:>7} | "
            f"PnL={total_pnl:>+10.2f} USDT"
        )


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

    btc_return = (
        candles[-1].close / candles[1].open - 1
    ) * 100

    net_return = (
        final_equity / INITIAL_CAPITAL - 1
    ) * 100

    print("\n========== BUY & HOLD BENCHMARK ==========")
    print(f"BTC Price Change: {btc_return:+.2f}%")
    print(f"Position Allocation: {position_fraction * 100:.0f}%")
    print(f"Final Equity: {final_equity:,.2f} USDT")
    print(f"Net Return: {net_return:+.2f}%")
    print(f"Trading Fees: {entry_fee + exit_fee:.2f} USDT")


def main():
    print("\n========== AITrader EARLIER 120D VALIDATION ==========")
    print(f"Exchange: {EXCHANGE}")
    print(f"Symbol: {SYMBOL}")
    print(f"Evaluation Period: {EVALUATION_DAYS} days")
    print(f"Transition Filter: {BULLISH_TRANSITION}")

    repository = SQLiteCandleRepository("data/aitrader.db")

    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    # 1. 전체 데이터로 지표와 시장 국면 계산
    regime_times, _, detailed_regimes_full = build_market_regimes(
        repository,
        now_ms,
    )

    full_candles, full_signals = build_aligned_data(
        repository,
        now_ms,
    )

    if not full_candles:
        raise RuntimeError("No aligned candles available")

    # 2. 기존 최근 180일 구간보다 앞선 120일을 평가 구간으로 지정
    data_end_ms = full_candles[-1].timestamp_ms + 60_000

    evaluation_end_ms = data_end_ms - 180 * DAY_MS
    evaluation_start_ms = (
        evaluation_end_ms - EVALUATION_DAYS * DAY_MS
    )

    history_start_ms = full_candles[0].timestamp_ms

    warmup_days = (
        evaluation_start_ms - history_start_ms
    ) / DAY_MS

    print("\n========== VALIDATION WINDOW ==========")

    print(f"Available Warmup: {warmup_days:.2f} days")

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

    if len(candles) != EXPECTED_CANDLES:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDLES:,} candles, "
            f"found {len(candles):,}"
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

    # 3. 평가 기간의 세부 시장 국면 계산
    regimes = []
    regime_map = {}

    for candle in candles:
        decision_time = candle.timestamp_ms + TIMEFRAMES["1m"]

        regime = get_regime_at(
            decision_time,
            regime_times,
            detailed_regimes_full,
        )

        regimes.append(regime)
        regime_map[decision_time] = regime

    unavailable = sum(
        regime == "unavailable"
        for regime in regimes
    )

    warmup = sum(
        regime == "warmup"
        for regime in regimes
    )

    unavailable_signals = sum(
        "Insufficient timeframe data" in signal.reasons
        for signal in signals
    )

    print("\n========== DATA QUALITY ==========")
    print(f"Regime Warmup: {warmup}")
    print(f"Unavailable Regimes: {unavailable}")
    print(f"Unavailable MTF Decisions: {unavailable_signals}")

    if unavailable or warmup or unavailable_signals:
        raise RuntimeError(
            "Incomplete evaluation data. Validation stopped."
        )

    regime_counts = Counter(regimes)

    print("\n========== MARKET REGIME DISTRIBUTION ==========")

    for regime, count in sorted(regime_counts.items()):
        print(
            f"{regime:<24}"
            f"{count:>10,} "
            f"({count / len(regimes) * 100:>6.2f}%)"
        )

    # 4. 동일 조건으로 전략 비교
    engine = BacktestEngine()
    config = create_config()

    baseline = engine.run(
        candles,
        signals,
        config,
    )

    filtered_signals = filter_entries_by_regime(
        signals,
        regimes,
        BULLISH_TRANSITION,
    )

    transition_result = engine.run(
        candles,
        filtered_signals,
        config,
    )

    print("\n========== BASELINE STRATEGY ==========")
    print_result("No Regime Filter", baseline)

    print_entry_attribution(
        baseline,
        regime_map,
        "BASELINE ENTRY ATTRIBUTION",
    )

    print("\n========== BULLISH TRANSITION STRATEGY ==========")
    print_result(
        "Bullish Transition Only",
        transition_result,
    )

    print_entry_attribution(
        transition_result,
        regime_map,
        "TRANSITION ENTRY ATTRIBUTION",
    )

    # 5. Buy and Hold 비교
    run_buy_hold(candles)

    print("\n========== VALIDATION SUMMARY ==========")

    print(
        f"Baseline Return: "
        f"{baseline.total_return_pct:+.2f}%"
    )

    print(
        f"Bullish Transition Return: "
        f"{transition_result.total_return_pct:+.2f}%"
    )

    print(
        f"Return Difference: "
        f"{transition_result.total_return_pct - baseline.total_return_pct:+.2f}%p"
    )

    print("\n========== VALIDATION COMPLETED ==========")
    print("Earlier historical 120D period evaluated.")
    print("Latest 180D period was excluded from evaluation.")
    print("No existing strategy or engine files were modified.")


if __name__ == "__main__":
    main()
