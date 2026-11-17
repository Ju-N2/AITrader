from dataclasses import dataclass


@dataclass(frozen=True)
class IndicatorSnapshot:
    timestamp_ms: int

    sma_20: float | None
    sma_50: float | None

    ema_12: float | None
    ema_26: float | None

    rsi_14: float | None

    macd: float | None
    macd_signal: float | None
    macd_histogram: float | None

    bollinger_upper: float | None
    bollinger_middle: float | None
    bollinger_lower: float | None

    atr_14: float | None
