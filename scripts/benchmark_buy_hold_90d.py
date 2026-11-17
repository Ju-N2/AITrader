from datetime import datetime, timezone

from aitrader.data.storage import SQLiteCandleRepository

INITIAL_CAPITAL = 10000.0
POSITION_FRACTION = 0.25
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005

EXCHANGE = "binance_spot"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"


def main():
    repository = SQLiteCandleRepository("data/aitrader.db")

    candles = repository.get_candles(
        EXCHANGE,
        SYMBOL,
        TIMEFRAME,
        limit=200000,
    )

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    candles = [
        candle for candle in candles
        if candle.timestamp_ms + 60_000 <= now_ms
    ]

    if len(candles) < 100:
        print("[ERROR] Insufficient data")
        return

    budget = INITIAL_CAPITAL * POSITION_FRACTION

    entry_price = candles[1].open * (1 + SLIPPAGE_RATE)
    notional = budget / (1 + FEE_RATE)
    entry_fee = notional * FEE_RATE
    quantity = notional / entry_price

    cash = INITIAL_CAPITAL - notional - entry_fee

    exit_price = candles[-1].close * (1 - SLIPPAGE_RATE)
    gross = quantity * exit_price
    exit_fee = gross * FEE_RATE

    final_equity = cash + gross - exit_fee
    total_fees = entry_fee + exit_fee
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100
    btc_return = (candles[-1].close / entry_price - 1) * 100

    print("\n========== BUY & HOLD 90D BENCHMARK ==========")
    print(f"Candles: {len(candles)}")
    print(f"Period: {candles[0].datetime_utc} ~ {candles[-1].datetime_utc}")
    print(f"Start Price: {entry_price:.2f}")
    print(f"End Price: {candles[-1].close:.2f}")
    print(f"BTC Price Change: {btc_return:+.2f}%")
    print(f"Initial Capital: {INITIAL_CAPITAL:.2f} USDT")
    print(f"Position Allocation: {POSITION_FRACTION * 100:.0f}%")
    print(f"Final Equity: {final_equity:.2f} USDT")
    print(f"Net Return: {total_return:+.2f}%")
    print(f"Total Fees: {total_fees:.2f} USDT")
    print("===============================================")


if __name__ == "__main__":
    main()
