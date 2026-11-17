from aitrader.core.exceptions import DataError
from aitrader.data.models import Candle

from aitrader.indicators.calculations import (
    sma,
    ema,
    rsi,
    macd,
    bollinger_bands,
    atr,
)

from aitrader.indicators.models import IndicatorSnapshot


class IndicatorEngine:
    """Calculate technical indicators from normalized candles."""

    def calculate(
        self,
        candles: list[Candle],
    ) -> list[IndicatorSnapshot]:

        if not candles:
            raise DataError("Cannot calculate indicators without candles")

        timestamps = [c.timestamp_ms for c in candles]

        if any(
            current <= previous
            for previous, current in zip(timestamps, timestamps[1:])
        ):
            raise DataError("Candles must be ordered by timestamp")

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        if any(
            value <= 0
            for value in closes + highs + lows
        ):
            raise DataError("Prices must be positive")

        sma_20 = sma(closes, 20)
        sma_50 = sma(closes, 50)

        ema_12 = ema(closes, 12)
        ema_26 = ema(closes, 26)

        rsi_14 = rsi(closes, 14)

        macd_line, macd_signal, macd_histogram = macd(closes)

        bb_upper, bb_middle, bb_lower = bollinger_bands(closes)

        atr_14 = atr(highs, lows, closes, 14)

        return [
            IndicatorSnapshot(
                timestamp_ms=candles[i].timestamp_ms,

                sma_20=sma_20[i],
                sma_50=sma_50[i],

                ema_12=ema_12[i],
                ema_26=ema_26[i],

                rsi_14=rsi_14[i],

                macd=macd_line[i],
                macd_signal=macd_signal[i],
                macd_histogram=macd_histogram[i],

                bollinger_upper=bb_upper[i],
                bollinger_middle=bb_middle[i],
                bollinger_lower=bb_lower[i],

                atr_14=atr_14[i],
            )
            for i in range(len(candles))
        ]
