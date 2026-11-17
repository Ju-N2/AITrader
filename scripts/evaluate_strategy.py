from datetime import datetime, timezone

from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine
from aitrader.strategy.technical import TechnicalTrendStrategy


DATABASE = "data/aitrader.db"
TIMEFRAME = "1h"
INTERVAL_MS = 3_600_000
MAX_DATA_AGE_MS = INTERVAL_MS * 2

TARGETS = [
    ("upbit", "BTC/KRW"),
    ("binance_spot", "BTC/USDT"),
    ("binance_futures", "BTC/USDT:USDT"),
]


def main():
    repository = SQLiteCandleRepository(DATABASE)
    indicator_engine = IndicatorEngine()
    strategy = TechnicalTrendStrategy()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print("\n========== AITrader Strategy Analysis ==========")

    for exchange, symbol in TARGETS:
        print(f"\n[{exchange}] {symbol}")

        candles = repository.get_candles(
            exchange,
            symbol,
            TIMEFRAME,
            limit=500,
        )

        # 진행 중인 캔들 제외
        closed_candles = [
            candle
            for candle in candles
            if candle.timestamp_ms + INTERVAL_MS <= now_ms
        ]

        if len(closed_candles) < 50:
            print("[WARNING] Insufficient completed candle data")
            continue

        latest_candle = closed_candles[-1]

        # 오래된 데이터로 잘못된 신호를 만들지 않도록 검사
        data_age = now_ms - (
            latest_candle.timestamp_ms + INTERVAL_MS
        )

        if data_age > MAX_DATA_AGE_MS:
            print("[WARNING] Market data is stale. Analysis skipped.")
            continue

        snapshots = indicator_engine.calculate(closed_candles)

        latest_indicators = snapshots[-1]

        signal = strategy.evaluate(
            latest_candle,
            latest_indicators,
        )

        print(f"Analysis time: {latest_candle.datetime_utc}")
        print(f"Price: {latest_candle.close:,.2f}")

        print("\n--- Strategy Result ---")
        print(f"Market Bias: {signal.bias.value.upper()}")
        print(f"Score: {signal.score:+d} / {signal.max_score}")

        print("\n--- Analysis Reasons ---")

        for reason in signal.reasons:
            print(f"- {reason}")

    print("\n========== Analysis Completed ==========")


if __name__ == "__main__":
    main()
