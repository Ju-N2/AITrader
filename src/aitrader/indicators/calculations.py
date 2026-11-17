from math import sqrt
from collections.abc import Sequence


def validate_period(period: int) -> None:
    if not isinstance(period, int) or isinstance(period, bool) or period < 1:
        raise ValueError("Period must be a positive integer")


def sma(values: Sequence[float], period: int) -> list[float | None]:
    """Simple Moving Average."""
    validate_period(period)

    result = [None] * len(values)

    if len(values) < period:
        return result

    window_sum = sum(values[:period])
    result[period - 1] = window_sum / period

    for i in range(period, len(values)):
        window_sum += values[i] - values[i - period]
        result[i] = window_sum / period

    return result


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential Moving Average using SMA initialization."""
    validate_period(period)

    result = [None] * len(values)

    if len(values) < period:
        return result

    alpha = 2 / (period + 1)

    previous = sum(values[:period]) / period
    result[period - 1] = previous

    for i in range(period, len(values)):
        previous = values[i] * alpha + previous * (1 - alpha)
        result[i] = previous

    return result


def rsi(values: Sequence[float], period: int = 14) -> list[float | None]:
    """Relative Strength Index using Wilder smoothing."""
    validate_period(period)

    result = [None] * len(values)

    if len(values) <= period:
        return result

    changes = [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]

    gains = [max(change, 0) for change in changes]
    losses = [max(-change, 0) for change in changes]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def calculate_value(gain: float, loss: float) -> float:
        if gain == 0 and loss == 0:
            return 50.0

        if loss == 0:
            return 100.0

        if gain == 0:
            return 0.0

        return 100 - 100 / (1 + gain / loss)

    result[period] = calculate_value(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        change = changes[i - 1]

        gain = max(change, 0)
        loss = max(-change, 0)

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        result[i] = calculate_value(avg_gain, avg_loss)

    return result


def bollinger_bands(
    values: Sequence[float],
    period: int = 20,
    stddev: float = 2.0,
) -> tuple[
    list[float | None],
    list[float | None],
    list[float | None],
]:
    """Bollinger Bands using population standard deviation."""
    validate_period(period)

    if stddev < 0:
        raise ValueError("Standard deviation multiplier cannot be negative")

    middle = sma(values, period)
    upper = [None] * len(values)
    lower = [None] * len(values)

    for i in range(period - 1, len(values)):
        mean = middle[i]
        window = values[i - period + 1:i + 1]

        variance = sum(
            (value - mean) ** 2 for value in window
        ) / period

        deviation = sqrt(variance) * stddev

        upper[i] = mean + deviation
        lower[i] = mean - deviation

    return upper, middle, lower


def macd(
    values: Sequence[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[
    list[float | None],
    list[float | None],
    list[float | None],
]:
    """MACD line, signal line and histogram."""

    validate_period(fast_period)
    validate_period(slow_period)
    validate_period(signal_period)

    if fast_period >= slow_period:
        raise ValueError("Fast period must be smaller than slow period")

    fast = ema(values, fast_period)
    slow = ema(values, slow_period)

    line = [None] * len(values)

    for i in range(len(values)):
        if fast[i] is not None and slow[i] is not None:
            line[i] = fast[i] - slow[i]

    first_valid = slow_period - 1

    valid_values = [
        value for value in line[first_valid:]
        if value is not None
    ]

    signal_values = ema(valid_values, signal_period)
    signal = [None] * len(values)

    for i, value in enumerate(signal_values):
        signal[first_valid + i] = value

    histogram = [
        line[i] - signal[i]
        if line[i] is not None and signal[i] is not None
        else None
        for i in range(len(values))
    ]

    return line, signal, histogram


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    """Average True Range using Wilder smoothing."""
    validate_period(period)

    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("OHLC arrays must have equal lengths")

    result = [None] * len(closes)

    if len(closes) < period:
        return result

    true_ranges = []

    for i in range(len(closes)):
        if highs[i] < lows[i]:
            raise ValueError("High cannot be lower than low")

        if i == 0:
            true_range = highs[i] - lows[i]
        else:
            true_range = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        true_ranges.append(true_range)

    previous = sum(true_ranges[:period]) / period
    result[period - 1] = previous

    for i in range(period, len(closes)):
        previous = (
            previous * (period - 1) + true_ranges[i]
        ) / period

        result[i] = previous

    return result
