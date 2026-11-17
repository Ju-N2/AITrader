from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone

from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine

from aitrader.strategy.technical import TechnicalTrendStrategy
from aitrader.strategy.multi_timeframe import MultiTimeframeEngine
from aitrader.strategy.models import StrategySignal, MarketBias

from aitrader.backtesting.engine import BacktestEngine
from aitrader.backtesting.models import BacktestConfig


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
        limit=200000,
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

    indicator_engine = IndicatorEngine()
    strategy = TechnicalTrendStrategy()

    snapshots = indicator_engine.calculate(closed)

    close_times = []
    signals = []

    for candle, snapshot in zip(closed, snapshots):

        if not indicator_ready(snapshot):
            continue

        signal = strategy.evaluate(candle, snapshot)

        # 이 신호가 사용 가능해지는 시각은 캔들 종료 시각
        close_time = candle.timestamp_ms + interval_ms

        close_times.append(close_time)
        signals.append(signal)

    print(
        f"{timeframe}: "
        f"{len(signals)} completed signals"
    )

    return close_times, signals


def main():
    repository = SQLiteCandleRepository("data/aitrader.db")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print("\n========== AITrader MTF BACKTEST ==========")

    # 1. 각 시간봉의 과거 분석 신호 생성
    signal_series = {}

    for timeframe, interval_ms in TIMEFRAMES.items():
        signal_series[timeframe] = build_signal_series(
            repository,
            timeframe,
            interval_ms,
            now_ms,
        )

    # 2. 1분봉을 기준으로 시뮬레이션
    candles_1m = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        "1m",
        limit=200000,
    )

    candles_1m = [
        candle for candle in candles_1m
        if candle.timestamp_ms + 60_000 <= now_ms
    ]

    multi_engine = MultiTimeframeEngine(
        min_timeframes=5,
    )

    aligned_candles = []
    aligned_signals = []

    unavailable = 0

    for candle in candles_1m:
        decision_time = candle.timestamp_ms + 60_000

        current_signals = {}

        # 각 시간봉의 최신 완료 신호 조회
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

        # 이 부분은 시간봉 반복문 바깥에 있어야 함
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
                    atr=current_signals["15m"].atr if "15m" in current_signals else None,
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

        # 1분봉 하나당 정확히 한 번 추가
        aligned_candles.append(candle)
        aligned_signals.append(combined_signal)

    if len(aligned_candles) < 100:
        print("[ERROR] Insufficient aligned data")
        print(f"Unavailable decisions: {unavailable}")
        return

    # 3. 백테스트
    config = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        fee_rate=0.001,
        slippage_rate=0.0005,
        position_fraction=0.25,
        atr_stop_multiplier=2.0,
        atr_cooldown_bars=15,
    )

    result = BacktestEngine().run(
        aligned_candles,
        aligned_signals,
        config,
    )

    bias_counts = Counter(
        signal.bias.value
        for signal in aligned_signals
    )

    print("\n========== DATA SUMMARY ==========")
    print(f"Exchange: {EXCHANGE}")
    print(f"Symbol: {SYMBOL}")
    print(f"Aligned candles: {len(aligned_candles)}")
    print(f"Unavailable decisions: {unavailable}")

    print("\n========== MARKET BIAS ==========")

    for bias in ["bullish", "bearish", "neutral"]:
        print(f"{bias.upper()}: {bias_counts[bias]}")

    print("\n========== MTF BACKTEST RESULT ==========")
    print(f"Initial Capital: {result.initial_capital:,.2f} USDT")
    print(f"Final Equity: {result.final_equity:,.2f} USDT")
    print(f"Total Return: {result.total_return_pct:+.2f}%")
    print(f"Maximum Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"Total Trades: {result.total_trades}")
    print(f"ATR Stop Losses: {result.stop_loss_count}")
    print(f"Cooldown Blocked Entries: {result.cooldown_blocked_entries}")
    print(f"Win Rate: {result.win_rate_pct:.2f}%")

    print(f"Total Fees: {result.total_fees:.2f} USDT")
    print(
        f"Slippage Cost: "
        f"{result.total_slippage_cost:.2f} USDT"
    )
    print(
        f"Total Trading Cost: "
        f"{result.total_fees + result.total_slippage_cost:.2f} USDT"
    )
    print(f"Total Turnover: {result.total_turnover:.2f} USDT")

    print(f"Profit Factor: {result.profit_factor}")

    print("\n========== TRADE AUDIT ==========")

    signal_lookup = {
        signal.timestamp_ms: signal
        for signal in aligned_signals
    }

    for number, trade in enumerate(result.trades, 1):

        entry_signal = signal_lookup.get(
            trade.entry_timestamp_ms - 60_000
        )

        exit_signal = signal_lookup.get(
            trade.exit_timestamp_ms - 60_000
        )

        entry_time = datetime.fromtimestamp(
            trade.entry_timestamp_ms / 1000,
            timezone.utc,
        )

        exit_time = datetime.fromtimestamp(
            trade.exit_timestamp_ms / 1000,
            timezone.utc,
        )

        entry_reason = (
            " | ".join(entry_signal.reasons)
            if entry_signal else "Unknown"
        )

        exit_reason = (
            " | ".join(exit_signal.reasons)
            if exit_signal else "Unknown"
        )

        print(f"\n--- Trade {number} ---")
        print(f"Entry Time: {entry_time}")
        print(f"Exit Time: {exit_time}")
        print(f"Exit Type: {trade.exit_reason}")
        print(f"Holding: {(trade.exit_timestamp_ms - trade.entry_timestamp_ms) / 60000:.1f} minutes")

        print(f"Entry Price: {trade.entry_price:.2f}")
        print(f"Exit Price: {trade.exit_price:.2f}")

        print(f"PnL: {trade.pnl:+.2f} USDT")
        print(f"Return: {trade.return_pct:+.2f}%")

        print(f"Entry Reason: {entry_reason}")
        print(f"Exit Reason: {exit_reason}")

    print("\n========== MTF BACKTEST COMPLETED ==========")


if __name__ == "__main__":
    main()
