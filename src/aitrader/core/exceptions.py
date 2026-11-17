class AITraderError(Exception):
    """Base exception for AITrader."""


class ConfigurationError(AITraderError):
    """Configuration-related errors."""


class ExchangeError(AITraderError):
    """Exchange communication errors."""


class DataError(AITraderError):
    """Market data errors."""


class StrategyError(AITraderError):
    """Trading strategy errors."""


class RiskError(AITraderError):
    """Risk management errors."""


class OrderError(AITraderError):
    """Order execution errors."""


class InsufficientBalanceError(OrderError):
    """Insufficient balance for an order."""


class InvalidOrderError(OrderError):
    """Invalid order parameters."""


class StaleDataError(DataError):
    """Market data is too old to use safely."""

class StorageError(AITraderError):
    """Database storage-related errors."""