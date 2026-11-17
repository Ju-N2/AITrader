from datetime import datetime, timezone

from aitrader.data.storage import SQLiteCandleRepository
from aitrader.indicators.engine import IndicatorEngine


DATABASE = "data/aitrader.db"
TIMEFRAME = "1h"

TARGETS = [
    ("upbit", "BTC/KRW"),
    ("binance_spot", "BTC/USDT"),
    ("binance_futures", "BTC/USDT:USDT"),
]

TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def show(value, digits=2):
    if value is None:
        return "계산 대기"

    return f"{value:,.{digits}f}"


def main():
    repository = SQLiteCandleRepository(DATABASE)
    engine = IndicatorEngine()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    interval_ms = TIMEFRAME_MS[TIMEFRAME]

    print("\n========== AITrader Indicator Analysis ==========")

    for exchange, symbol in TARGETS:
        print(f"\n[{exchange}] {symbol}")

        candles = repository.get_candles(
            exchange,
            symbol,
            TIMEFRAME,
            limit=500,
        )

        # 진행 중인 캔들은 분석에서 제외
        closed_candles = [
            candle
            for candle in candles
            if candle.timestamp_ms + interval_ms <= now_ms
        ]

        if not closed_candles:
            print("[WARNING] No completed candles available")
            continue

        snapshots = engine.calculate(closed_candles)

        latest_candle = closed_candles[-1]
        latest = snapshots[-1]

        print(f"Stored candles: {len(candles)}")
        print(f"Completed candles: {len(closed_candles)}")
        print(f"Analysis time: {latest_candle.datetime_utc}")
        print(f"Current close: {show(latest_candle.close)}")

        print("\n--- Moving Averages ---")
        print(f"SMA 20: {show(latest.sma_20)}")
        print(f"SMA 50: {show(latest.sma_50)}")
        print(f"EMA 12: {show(latest.ema_12)}")
        print(f"EMA 26: {show(latest.ema_26)}")

        print("\n--- Momentum ---")
        print(f"RSI 14: {show(latest.rsi_14)}")
        print(f"MACD: {show(latest.macd)}")
        print(f"MACD Signal: {show(latest.macd_signal)}")
        print(f"MACD Histogram: {show(latest.macd_histogram)}")

        print("\n--- Volatility ---")
        print(f"Bollinger Upper: {show(latest.bollinger_upper)}")
        print(f"Bollinger Middle: {show(latest.bollinger_middle)}")
        print(f"Bollinger Lower: {show(latest.bollinger_lower)}")
        print(f"ATR 14: {show(latest.atr_14)}")

    print("\n========== Analysis Completed ==========")


if __name__ == "__main__":
    main()
