from aitrader.data.history import HistoricalDataDownloader
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.exchanges.factory import ExchangeFactory


def main():
    repository = SQLiteCandleRepository("data/aitrader.db")

    client = ExchangeFactory.create("upbit")

    try:
        client.load_markets()

        downloader = HistoricalDataDownloader(
            client,
            "upbit",
            repository,
        )

        for timeframe in ["1m", "15m"]:
            try:
                print(f"\n========== Upbit {timeframe} ==========")

                processed = downloader.download(
                    "BTC/KRW",
                    timeframe,
                )

                total = repository.count_candles(
                    "upbit",
                    "BTC/KRW",
                    timeframe,
                )

                print(f"[SUCCESS] Processed: {processed}")
                print(f"Total stored: {total}")

            except Exception as error:
                print(f"[FAILED] {timeframe}: {error}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
