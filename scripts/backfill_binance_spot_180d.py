from aitrader.core.exceptions import AITraderError
from aitrader.data.history import HistoricalDataDownloader
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.exchanges.factory import ExchangeFactory

EXCHANGE = "binance_spot"
SYMBOL = "BTC/USDT"
DAYS = 180

TIMEFRAMES = ("1m", "5m", "15m", "1h")


def main():
    client = ExchangeFactory.create(EXCHANGE)
    repository = SQLiteCandleRepository("data/aitrader.db")

    failures = []

    try:
        client.load_markets()

        downloader = HistoricalDataDownloader(
            client,
            EXCHANGE,
            repository,
        )

        print("=" * 60)
        print("BINANCE SPOT 180D HISTORICAL BACKFILL")
        print(f"Symbol: {SYMBOL}")
        print(f"Period: {DAYS} days")
        print(f"Timeframes: {', '.join(TIMEFRAMES)}")
        print("=" * 60)

        for timeframe in TIMEFRAMES:
            print(f"\n========== {timeframe} ==========")

            before = repository.count_candles(
                EXCHANGE,
                SYMBOL,
                timeframe,
            )

            print(f"Existing candles: {before:,}")

            try:
                processed = downloader.download(
                    SYMBOL,
                    timeframe,
                    days_override=DAYS,
                )

                after = repository.count_candles(
                    EXCHANGE,
                    SYMBOL,
                    timeframe,
                )

                print(f"Processed / Upserted: {processed:,}")
                print(f"Total candles: {after:,}")
                print(f"Net row increase: {after - before:,}")
                print(f"[SUCCESS] {timeframe}")

            except AITraderError as error:
                failures.append((timeframe, str(error)))
                print(f"[FAILED] {timeframe}: {error}")

        print("\n========== BACKFILL SUMMARY ==========")

        for timeframe in TIMEFRAMES:
            total = repository.count_candles(
                EXCHANGE,
                SYMBOL,
                timeframe,
            )
            status = (
                "FAILED"
                if any(item[0] == timeframe for item in failures)
                else "COMPLETED"
            )

            print(
                f"{timeframe}: "
                f"{total:,} candles | {status}"
            )

        if failures:
            print("\nFailed timeframes:")
            for timeframe, error in failures:
                print(f"- {timeframe}: {error}")

            raise SystemExit(1)

        print("\n[COMPLETE] Binance Spot 180D backfill finished.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
