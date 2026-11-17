from aitrader.exchanges.factory import ExchangeFactory
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.data.history import HistoricalDataDownloader


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

        print("========== BINANCE 1M BACKFILL ==========")

        processed = downloader.download(
            "BTC/USDT",
            "1m",
        )

        total = repository.count_candles(
            "binance_spot",
            "BTC/USDT",
            "1m",
        )

        print(f"Processed: {processed}")
        print(f"Total candles: {total}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
