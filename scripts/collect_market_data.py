from aitrader.core.logger import setup_logger
from aitrader.data.service import MarketDataService
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.exchanges.factory import ExchangeFactory


TARGETS = [
    ("upbit", "BTC/KRW"),
    ("binance_spot", "BTC/USDT"),
    ("binance_futures", "BTC/USDT:USDT"),
]

TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

LIMIT = 100


def main():
    logger = setup_logger("aitrader.collect")

    repository = SQLiteCandleRepository(
        "data/aitrader.db"
    )

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for exchange_name, symbol in TARGETS:
        client = None

        try:
            print(f"\n========== {exchange_name} ==========")

            client = ExchangeFactory.create(exchange_name)

            markets = client.load_markets()

            if symbol not in markets:
                raise ValueError(f"Market not found: {symbol}")

            supported_timeframes = (
                getattr(client.exchange, "timeframes", None) or {}
            )

            for timeframe in TIMEFRAMES:

                if timeframe not in supported_timeframes:
                    print(f"[SKIP] Unsupported timeframe: {timeframe}")
                    skipped_count += 1
                    continue

                try:
                    service = MarketDataService(
                        client=client,
                        exchange_name=exchange_name,
                        repository=repository,
                    )

                    saved = service.collect_and_store(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=LIMIT,
                    )

                    total = repository.count_candles(
                        exchange_name,
                        symbol,
                        timeframe,
                    )

                    print(
                        f"[SUCCESS] {timeframe} | "
                        f"Processed: {saved} | Total: {total}"
                    )

                    success_count += 1

                except Exception:
                    logger.exception(
                        "Collection failed: %s %s",
                        exchange_name,
                        timeframe,
                    )

                    print(f"[FAILED] {timeframe}")
                    failed_count += 1

        except Exception:
            logger.exception(
                "Exchange connection failed: %s",
                exchange_name,
            )

            print(f"[FAILED] {exchange_name}")
            failed_count += 1

        finally:
            if client:
                client.close()

    print("\n========== COLLECTION SUMMARY ==========")
    print(f"Successful: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    main()
