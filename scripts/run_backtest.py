from datetime import datetime, timezone

from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine
from aitrader.strategy.technical import TechnicalTrendStrategy

from aitrader.backtesting.engine import BacktestEngine
from aitrader.backtesting.models import BacktestConfig


EXCHANGE = "binance_spot"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"

INTERVAL_MS = 60_000


def main():
    repository = SQLiteCandleRepository("data/aitrader.db")

    indicator_engine = IndicatorEngine()
    strategy = TechnicalTrendStrategy()
    backtest_engine = BacktestEngine()

    print("\n========== AITrader BACKTEST ==========")

    candles = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        TIMEFRAME,
        limit=20000,
    )

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # 진행 중인 캔들 제외
    candles = [
        candle
        for candle in candles
        if candle.timestamp_ms + INTERVAL_MS <= now_ms
    ]

    if len(candles) < 100:
        print("[ERROR] Insufficient historical data")
        return

    # 누락된 1분봉 확인
    missing = sum(
        max(
            0,
            (current.timestamp_ms - previous.timestamp_ms)
            // INTERVAL_MS - 1
        )
        for previous, current in zip(candles, candles[1:])
    )

    print(f"Exchange: {EXCHANGE}")
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Available candles: {len(candles)}")
    print(f"Missing intervals: {missing}")

    if missing:
        print("[WARNING] Missing intervals detected.")

    # 기술적 지표 계산
    snapshots = indicator_engine.calculate(candles)

    # 모든 필수 지표가 준비된 첫 번째 캔들 탐색
    start_index = next(
        (
            i for i, indicator in enumerate(snapshots)
            if all(value is not None for value in [
                indicator.sma_20,
                indicator.sma_50,
                indicator.ema_12,
                indicator.ema_26,
                indicator.rsi_14,
                indicator.macd_histogram,
            ])
        ),
        None,
    )

    if start_index is None:
        print("[ERROR] Indicators are insufficient")
        return

    candles = candles[start_index:]
    snapshots = snapshots[start_index:]

    # 각 캔들 기준 전략 신호 생성
    signals = [
        strategy.evaluate(candle, indicator)
        for candle, indicator in zip(candles, snapshots)
    ]

    config = BacktestConfig(
        initial_capital=10000.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
        position_fraction=0.25,
    )

    result = backtest_engine.run(
        candles,
        signals,
        config,
    )

    print("\n========== BACKTEST RESULT ==========")

    print(f"Initial Capital: {result.initial_capital:,.2f} USDT")
    print(f"Final Equity: {result.final_equity:,.2f} USDT")

    print(f"Total Return: {result.total_return_pct:+.2f}%")
    print(f"Maximum Drawdown: {result.max_drawdown_pct:.2f}%")

    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate_pct:.2f}%")

    if result.profit_factor is None:
        print("Profit Factor: N/A")
    else:
        print(f"Profit Factor: {result.profit_factor:.2f}")

    print(f"Total Fees: {result.total_fees:.4f} USDT")

    if result.trades:
        print("\n========== RECENT TRADES ==========")

        for trade in result.trades[-10:]:
            print(
                f"Entry: {trade.entry_price:.2f} | "
                f"Exit: {trade.exit_price:.2f} | "
                f"PnL: {trade.pnl:+.2f} USDT | "
                f"Return: {trade.return_pct:+.2f}%"
            )

    print("\n========== BACKTEST COMPLETED ==========")


if __name__ == "__main__":
    main()
