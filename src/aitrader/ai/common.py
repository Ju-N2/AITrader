from math import isfinite


def numeric(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        return None

    return float(value)


def direction(first: float, second: float) -> int:
    if first > second:
        return 1

    if first < second:
        return -1

    return 0
