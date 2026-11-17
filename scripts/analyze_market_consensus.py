from datetime import datetime, timezone

from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine
from aitrader.strategy.technical import TechnicalTrendStrategy
from aitrader.strategy.multi_timeframe import MultiTimeframeEngine


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


def main():
    repository = SQLiteCandleRepository(DATABASE)
    indicator_engine = IndicatorEngine()
    strategy = TechnicalTrendStrategy()
    consensus_engine = MultiTimeframeEngine()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print("\n========== AITrader MARKET CONSENSUS ==========")

    for exchange, symbol in TARGETS:

        print(f"\n{'=' * 55}")
        print(f"{exchange} | {symbol}")

        signals = {}

        for timeframe, interval_ms in TIMEFRAMES.items():

            candles = repository.get_candles(
                exchange,
                symbol,
                timeframe,
                limit=500,
            )

            closed = [
                candle
                for candle in candles
                if candle.timestamp_ms + interval_ms <= now_ms
            ]

            if len(closed) < 50:
                print(f"[SKIP] {timeframe}: insufficient data")
                continue

            latest = closed[-1]

            data_age = now_ms - (
                latest.timestamp_ms + interval_ms
            )

            if data_age < 0 or data_age > interval_ms * 3:
                print(f"[SKIP] {timeframe}: stale data")
                continue

            snapshots = indicator_engine.calculate(closed)

            signal = strategy.evaluate(
                latest,
                snapshots[-1],
            )

            signals[timeframe] = signal

            print(
                f"{timeframe:>3} | "
                f"{signal.bias.value.upper():8} | "
                f"{signal.score:+d}/5"
            )

        if not signals:
            print("[WARNING] No valid signals")
            continue

        result = consensus_engine.evaluate(signals)

        print("\n--- FINAL CONSENSUS ---")
        print(f"Market Bias: {result.bias.value.upper()}")
        print(f"Weighted Score: {result.weighted_score:+.2f} / 5")

        print(
            "Available: "
            + ", ".join(result.available_timeframes)
        )

        if result.missing_timeframes:
            print(
                "Missing: "
                + ", ".join(result.missing_timeframes)
            )

        for reason in result.reasons:
            print(f"- {reason}")

    print("\n========== ANALYSIS COMPLETED ==========")


if __name__ == "__main__":
    main()
