from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class Position:
    position_id: str

    exchange: str
    symbol: str
    market_type: str

    side: PositionSide
    status: PositionStatus

    entry_order_id: str

    entry_quantity: float
    remaining_quantity: float

    entry_price: float
    stop_price: float

    leverage: float

    entry_fee_total: float
    remaining_entry_fee: float
    exit_fee_total: float

    realized_pnl: float

    opened_at: datetime
    updated_at: datetime
    closed_at: datetime | None

    def unrealized_pnl(self, mark_price: float) -> float:
        """
        Unrealized PnL after allocating the remaining entry fee.
        Does not include a hypothetical exit fee.
        """
        if (
            isinstance(mark_price, bool)
            or not isinstance(mark_price, (int, float))
            or not isfinite(mark_price)
            or mark_price <= 0
        ):
            raise ValueError("Invalid mark price")

        if self.status != PositionStatus.OPEN:
            return 0.0

        if self.side == PositionSide.LONG:
            gross_pnl = (
                mark_price - self.entry_price
            ) * self.remaining_quantity
        else:
            gross_pnl = (
                self.entry_price - mark_price
            ) * self.remaining_quantity

        return gross_pnl - self.remaining_entry_fee

    def is_stop_triggered(self, mark_price: float) -> bool:
        if (
            isinstance(mark_price, bool)
            or not isinstance(mark_price, (int, float))
            or not isfinite(mark_price)
            or mark_price <= 0
        ):
            raise ValueError("Invalid mark price")

        if self.status != PositionStatus.OPEN:
            return False

        if self.side == PositionSide.LONG:
            return mark_price <= self.stop_price

        return mark_price >= self.stop_price
