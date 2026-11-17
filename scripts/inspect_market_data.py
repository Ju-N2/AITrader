from aitrader.data.storage import SQLiteCandleRepository


TARGETS = [
    ("upbit", "BTC/KRW"),
    ("binance_spot", "BTC/USDT"),
    ("binance_futures", "BTC/USDT:USDT"),
]


def main():
    repository = SQLiteCandleRepository("data/aitrader.db")

    print("\n========== AITrader DATABASE ==========")

    for exchange, symbol in TARGETS:
        print(f"\n[{exchange}] {symbol}")

        count = repository.count_candles(
            exchange,
            symbol,
            "1h",
        )

        if count == 0:
            print("[WARNING] No data found")
            continue

        candles = repository.get_candles(
            exchange,
            symbol,
            "1h",
            limit=5,
        )

        print(f"Total candles: {count}")
        print(f"Latest candle: {candles[-1].datetime_utc}")

        print("\nRecent candles:")

        for candle in candles:
            print(
                f"{candle.datetime_utc.isoformat()} | "
                f"O:{candle.open} "
                f"H:{candle.high} "
                f"L:{candle.low} "
                f"C:{candle.close} "
                f"V:{candle.volume}"
            )


if __name__ == "__main__":
    main()
