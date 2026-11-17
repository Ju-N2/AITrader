from datetime import datetime, timezone

from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine
from aitrader.strategy.technical import TechnicalTrendStrategy

from aitrader.backtesting.engine import BacktestEngine
from aitrader.backtesting.models import BacktestConfig


INITIAL_CAPITAL = 10000.0
POSITION_FRACTION = 0.25

EXCHANGE = "binance_spot"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"


def build_signals(candles):
    indicator_engine = IndicatorEngine()
    strategy = TechnicalTrendStrategy()

    snapshots = indicator_engine.calculate(candles)

    start = next(
        i for i, item in enumerate(snapshots)
        if all(value is not None for value in [
            item.sma_20,
            item.sma_50,
            item.ema_12,
            item.ema_26,
            item.rsi_14,
            item.macd_histogram,
        ])
    )

    candles = candles[start:]
    snapshots = snapshots[start:]

    signals = [
        strategy.evaluate(candle, indicator)
        for candle, indicator in zip(candles, snapshots)
    ]

    return candles, signals


def buy_and_hold(candles, fee_rate, slippage_rate):
    budget = INITIAL_CAPITAL * POSITION_FRACTION

    entry_price = candles[1].open * (1 + slippage_rate)

    notional = budget / (1 + fee_rate)
    entry_fee = notional * fee_rate

    quantity = notional / entry_price

    cash = INITIAL_CAPITAL - notional - entry_fee

    exit_price = candles[-1].close * (1 - slippage_rate)

    gross = quantity * exit_price
    exit_fee = gross * fee_rate

    final_equity = cash + gross - exit_fee
    total_fees = entry_fee + exit_fee

    return final_equity, total_fees


def main():
    repository = SQLiteCandleRepository("data/aitrader.db")

    candles = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        TIMEFRAME,
        limit=20000,
    )

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    candles = [
        candle for candle in candles
        if candle.timestamp_ms + 60_000 <= now_ms
    ]

    if len(candles) < 100:
        print("[ERROR] Insufficient data")
        return

    candles, signals = build_signals(candles)

    engine = BacktestEngine()

    scenarios = [
        ("A: No costs", 0.0, 0.0),
        ("B: Fee only", 0.001, 0.0),
        ("C: Fee + slippage", 0.001, 0.0005),
    ]

    print("\n========== BACKTEST DIAGNOSTICS ==========")
    print(f"Candles: {len(candles)}")
    print(f"Period: {candles[0].datetime_utc} ~ {candles[-1].datetime_utc}")

    for name, fee, slippage in scenarios:

        config = BacktestConfig(
            initial_capital=INITIAL_CAPITAL,
            fee_rate=fee,
            slippage_rate=slippage,
            position_fraction=POSITION_FRACTION,
        )

        result = engine.run(candles, signals, config)

        hold_equity, hold_fees = buy_and_hold(
            candles,
            fee,
            slippage,
        )

        hold_return = (
            hold_equity / INITIAL_CAPITAL - 1
        ) * 100

        print(f"\n--- {name} ---")
        print(f"Strategy Return: {result.total_return_pct:+.2f}%")
        print(f"Strategy Trades: {result.total_trades}")
        print(f"Strategy Win Rate: {result.win_rate_pct:.2f}%")
        print(f"Strategy Profit Factor: {result.profit_factor}")
        print(f"Strategy Fees: {result.total_fees:.2f} USDT")
        print(f"Slippage Cost: {result.total_slippage_cost:.2f} USDT")
        print(f"Total Trading Cost: {result.total_fees + result.total_slippage_cost:.2f} USDT")
        print(f"Total Turnover: {result.total_turnover:.2f} USDT")

        print(f"Buy & Hold Return: {hold_return:+.2f}%")
        print(f"Buy & Hold Fees: {hold_fees:.2f} USDT")

    print("\n========== DIAGNOSTICS COMPLETED ==========")


if __name__ == "__main__":
    main()
