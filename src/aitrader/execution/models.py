from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from math import isfinite


class OrderStatus(str, Enum):
    FILLED = "filled"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    exchange: str
    symbol: str
    side: str
    reference_price: float
    client_order_id: str | None = None
    reduce_only: bool = False

    def __post_init__(self):
        from aitrader.risk.engine import OrderSide

        if not self.exchange.strip():
            raise ValueError("Exchange is required")

        if not self.symbol.strip():
            raise ValueError("Symbol is required")

        try:
            side = OrderSide(self.side)
        except (ValueError, TypeError):
            raise ValueError("Invalid order side") from None

        object.__setattr__(self, "side", side)

        if (
            isinstance(self.reference_price, bool)
            or not isinstance(self.reference_price, (int, float))
            or not isfinite(self.reference_price)
            or self.reference_price <= 0
        ):
            raise ValueError("Invalid reference price")

        if not isinstance(self.reduce_only, bool):
            raise ValueError("reduce_only must be boolean")

        if self.client_order_id is not None:
            if not self.client_order_id.strip():
                raise ValueError("Invalid client order ID")


@dataclass(frozen=True, slots=True)
class PaperExecutionConfig:
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005

    def __post_init__(self):
        for value in (self.fee_rate, self.slippage_rate):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 <= value < 1
            ):
                raise ValueError("Invalid execution configuration")


@dataclass(frozen=True, slots=True)
class OrderRecord:
    order_id: str
    client_order_id: str

    exchange: str
    symbol: str
    side: str

    status: OrderStatus

    requested_quantity: float
    filled_quantity: float

    reference_price: float
    average_fill_price: float | None

    filled_notional: float
    fee: float

    stop_price: float | None

    execution_mode: str
    created_at: datetime

    rejection_reason: str | None = None
    reduce_only: bool = False
