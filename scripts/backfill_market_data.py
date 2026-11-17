from aitrader.core.logger import setup_logger
from aitrader.data.history import (
    HistoricalDataDownloader,
    TIMEFRAME_CONFIG,
)
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.exchanges.factory import ExchangeFactory


TARGETS = [
    ("upbit", "BTC/KRW"),
    ("binance_spot", "BTC/USDT"),
    ("binance_futures", "BTC/USDT:USDT"),
]


def main():
    logger = setup_logger("aitrader.history")

    repository = SQLiteCandleRepository("data/aitrader.db")

    total_success = 0
    total_failed = 0

    for exchange_name, symbol in TARGETS:

        client = None

        try:
            print(f"\n========== {exchange_name} ==========")

            client = ExchangeFactory.create(exchange_name)
            client.load_markets()

            downloader = HistoricalDataDownloader(
                client,
                exchange_name,
                repository,
            )

            for timeframe, config in TIMEFRAME_CONFIG.items():

                try:
                    processed = downloader.download(
                        symbol,
                        timeframe,
                    )

                    total = repository.count_candles(
                        exchange_name,
                        symbol,
                        timeframe,
                    )

                    print(
                        f"[SUCCESS] {timeframe} | "
                        f"Period: {config['days']} days | "
                        f"Processed: {processed} | "
                        f"Total: {total}"
                    )

                    total_success += 1

                except Exception:
                    logger.exception(
                        "Historical download failed: %s %s",
                        exchange_name,
                        timeframe,
                    )

                    print(f"[FAILED] {timeframe}")
                    total_failed += 1

        except Exception:
            logger.exception(
                "Exchange history download failed: %s",
                exchange_name,
            )

            print(f"[FAILED] {exchange_name}")
            total_failed += len(TIMEFRAME_CONFIG)

        finally:
            if client:
                client.close()

    print("\n========== BACKFILL SUMMARY ==========")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_failed}")


if __name__ == "__main__":
    main()
