import pytest

from aitrader.core.exceptions import (
    AITraderError,
    ExchangeError,
    OrderError,
    InsufficientBalanceError,
    RiskError,
)


def test_custom_exception_inheritance():
    assert issubclass(ExchangeError, AITraderError)
    assert issubclass(OrderError, AITraderError)
    assert issubclass(InsufficientBalanceError, OrderError)
    assert issubclass(RiskError, AITraderError)


def test_exception_message():
    with pytest.raises(ExchangeError, match="API connection failed"):
        raise ExchangeError("API connection failed")


def test_insufficient_balance_exception():
    with pytest.raises(InsufficientBalanceError):
        raise InsufficientBalanceError("Insufficient balance")
