from datetime import datetime, timezone

from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine
from aitrader.strategy.technical import TechnicalTrendStrategy


DATABASE = "data/aitrader.db"

TIMEFRAMES = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}

TARGETS = [
    ("upbit", "BTC/KRW"),
    ("binance_spot", "BTC/USDT"),
    ("binance_futures", "BTC/USDT:USDT"),
]

MIN_CANDLES = 50
MAX_STALE_INTERVALS = 3


def main():
    repository = SQLiteCandleRepository(DATABASE)
    indicator_engine = IndicatorEngine()
    strategy = TechnicalTrendStrategy()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    success = 0
    skipped = 0

    print("\n========== MULTI-TIMEFRAME ANALYSIS ==========")

    for exchange, symbol in TARGETS:
        print(f"\n{'=' * 50}")
        print(f"{exchange} | {symbol}")

        for timeframe, interval_ms in TIMEFRAMES.items():

            candles = repository.get_candles(
                exchange,
                symbol,
                timeframe,
                limit=500,
            )

            closed_candles = [
                candle
                for candle in candles
                if candle.timestamp_ms + interval_ms <= now_ms
            ]

            if len(closed_candles) < MIN_CANDLES:
                print(
                    f"[SKIP] {timeframe}: "
                    f"Insufficient candles ({len(closed_candles)})"
                )
                skipped += 1
                continue

            latest = closed_candles[-1]

            data_age = now_ms - (
                latest.timestamp_ms + interval_ms
            )

            if data_age < 0 or data_age > (
                interval_ms * MAX_STALE_INTERVALS
            ):
                print(f"[SKIP] {timeframe}: Stale or future data")
                skipped += 1
                continue

            snapshots = indicator_engine.calculate(closed_candles)

            signal = strategy.evaluate(
                latest,
                snapshots[-1],
            )

            print(
                f"[{timeframe:>3}] "
                f"{signal.bias.value.upper():8} "
                f"Score: {signal.score:+d}/5 "
                f"Price: {latest.close:,.2f}"
            )

            success += 1

    print("\n========== ANALYSIS SUMMARY ==========")
    print(f"Successful: {success}")
    print(f"Skipped: {skipped}")
    print(f"Total: {success + skipped}")


if __name__ == "__main__":
    main()
