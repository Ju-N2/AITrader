from aitrader.exchanges.factory import ExchangeFactory
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.data.history import HistoricalDataDownloader


TIMEFRAMES = ["1m", "5m", "15m"]


def main():
    client = ExchangeFactory.create("binance_spot")

    repository = SQLiteCandleRepository("data/aitrader.db")

    try:
        client.load_markets()

        downloader = HistoricalDataDownloader(
            client,
            "binance_spot",
            repository,
        )

        print("========== BINANCE SPOT 90D BACKFILL ==========")

        for timeframe in TIMEFRAMES:
            try:
                print(f"\n========== {timeframe} ==========")

                processed = downloader.download(
                    "BTC/USDT",
                    timeframe,
                    days_override=90,
                )

                total = repository.count_candles(
                    "binance_spot",
                    "BTC/USDT",
                    timeframe,
                )

                print(f"[SUCCESS] Processed: {processed}")
                print(f"Total candles: {total}")

            except Exception as error:
                print(f"[FAILED] {timeframe}: {error}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
