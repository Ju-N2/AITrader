import math
import sqlite3

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from uuid import uuid4

from aitrader.execution.models import OrderRecord, OrderStatus
from aitrader.portfolio.models import (
    Position,
    PositionSide,
    PositionStatus,
)
from aitrader.risk.engine import MarketType, OrderSide


class PositionManager:
    """
    SQLite-backed portfolio position manager.

    Supports:
    - Paper execution records only
    - Spot long positions
    - Futures long/short positions
    - Partial and full exits
    - Realized and unrealized PnL
    - Stop monitoring
    - Persistent order deduplication

    This manager never submits exchange orders.
    """

    EPSILON = 1e-12

    def __init__(
        self,
        db_path: str | Path = "data/aitrader.db",
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = RLock()
        self.initialize()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(
            self.db_path,
            timeout=10,
        )
        connection.row_factory = sqlite3.Row

        try:
            connection.execute(
                "PRAGMA foreign_keys = ON"
            )
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def initialize(self):
        with self._lock:
            with self._transaction() as connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        position_id TEXT PRIMARY KEY,

                        exchange TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        market_type TEXT NOT NULL,

                        side TEXT NOT NULL,
                        status TEXT NOT NULL,

                        entry_order_id TEXT NOT NULL UNIQUE,

                        entry_quantity REAL NOT NULL,
                        remaining_quantity REAL NOT NULL,

                        entry_price REAL NOT NULL,
                        stop_price REAL NOT NULL,

                        leverage REAL NOT NULL,

                        entry_fee_total REAL NOT NULL,
                        remaining_entry_fee REAL NOT NULL,
                        exit_fee_total REAL NOT NULL,

                        realized_pnl REAL NOT NULL,

                        opened_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        closed_at TEXT
                    )
                """)

                connection.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_one_open_position_per_market
                    ON positions (
                        exchange,
                        symbol,
                        market_type
                    )
                    WHERE status = 'open'
                """)

                connection.execute("""
                    CREATE INDEX IF NOT EXISTS
                    idx_positions_lookup
                    ON positions (
                        exchange,
                        symbol,
                        status
                    )
                """)

                connection.execute("""
                    CREATE TABLE IF NOT EXISTS position_events (
                        order_id TEXT PRIMARY KEY,

                        position_id TEXT NOT NULL,

                        operation TEXT NOT NULL,

                        realized_pnl REAL NOT NULL DEFAULT 0,

                        applied_at TEXT NOT NULL,

                        FOREIGN KEY(position_id)
                            REFERENCES positions(position_id)
                    )
                """)

    @staticmethod
    def _utc_text(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)

        return value.astimezone(UTC).isoformat()

    @staticmethod
    def _number(value, name, allow_zero=False):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid {name}")

        if value < 0 if allow_zero else value <= 0:
            raise ValueError(f"Invalid {name}")

        return float(value)

    @classmethod
    def _validate_filled_order(cls, order: OrderRecord):
        if not isinstance(order, OrderRecord):
            raise TypeError("Expected OrderRecord")

        if OrderStatus(order.status) != OrderStatus.FILLED:
            raise ValueError("Only filled orders can update positions")

        if order.execution_mode != "paper":
            raise ValueError("Only paper execution is supported")

        if not order.exchange or not order.symbol:
            raise ValueError("Order market information is missing")

        try:
            side = OrderSide(order.side)
        except ValueError:
            raise ValueError("Invalid order side") from None

        quantity = cls._number(
            order.filled_quantity,
            "filled quantity",
        )

        price = cls._number(
            order.average_fill_price,
            "average fill price",
        )

        fee = cls._number(
            order.fee,
            "order fee",
            allow_zero=True,
        )

        return side, quantity, price, fee

    @staticmethod
    def _from_row(row) -> Position:
        return Position(
            position_id=row["position_id"],
            exchange=row["exchange"],
            symbol=row["symbol"],
            market_type=row["market_type"],
            side=PositionSide(row["side"]),
            status=PositionStatus(row["status"]),
            entry_order_id=row["entry_order_id"],
            entry_quantity=row["entry_quantity"],
            remaining_quantity=row["remaining_quantity"],
            entry_price=row["entry_price"],
            stop_price=row["stop_price"],
            leverage=row["leverage"],
            entry_fee_total=row["entry_fee_total"],
            remaining_entry_fee=row["remaining_entry_fee"],
            exit_fee_total=row["exit_fee_total"],
            realized_pnl=row["realized_pnl"],
            opened_at=datetime.fromisoformat(row["opened_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            closed_at=(
                datetime.fromisoformat(row["closed_at"])
                if row["closed_at"]
                else None
            ),
        )

    @staticmethod
    def _get_position_row(connection, position_id):
        return connection.execute("""
            SELECT *
            FROM positions
            WHERE position_id = ?
        """, (position_id,)).fetchone()

    def open_from_order(
        self,
        order: OrderRecord,
        market_type: MarketType | str,
        leverage: float = 1.0,
    ) -> Position:

        side, quantity, price, fee = (
            self._validate_filled_order(order)
        )

        try:
            market_type = MarketType(market_type)
        except ValueError:
            raise ValueError("Unsupported market type") from None

        leverage = self._number(leverage, "leverage")

        if leverage < 1:
            raise ValueError("Leverage must be at least 1")

        if market_type == MarketType.SPOT:
            if side == OrderSide.SELL:
                raise ValueError("Spot short positions are not supported")

            if leverage != 1:
                raise ValueError("Spot leverage must be 1")

        stop_price = self._number(
            order.stop_price,
            "stop price",
        )

        if side == OrderSide.BUY:
            if stop_price >= price:
                raise ValueError(
                    "Long stop must be below entry price"
                )
            position_side = PositionSide.LONG
        else:
            if stop_price <= price:
                raise ValueError(
                    "Short stop must be above entry price"
                )
            position_side = PositionSide.SHORT

        now = datetime.now(UTC)
        now_text = now.isoformat()

        position_id = uuid4().hex

        with self._lock:
            try:
                with self._transaction() as connection:

                    # Idempotent entry processing.
                    previous = connection.execute("""
                        SELECT position_id, operation
                        FROM position_events
                        WHERE order_id = ?
                    """, (order.order_id,)).fetchone()

                    if previous:
                        if previous["operation"] != "entry":
                            raise ValueError(
                                "Order ID was already used for another operation"
                            )

                        row = self._get_position_row(
                            connection,
                            previous["position_id"],
                        )
                        return self._from_row(row)

                    connection.execute("""
                        INSERT INTO positions (
                            position_id,
                            exchange,
                            symbol,
                            market_type,
                            side,
                            status,
                            entry_order_id,
                            entry_quantity,
                            remaining_quantity,
                            entry_price,
                            stop_price,
                            leverage,
                            entry_fee_total,
                            remaining_entry_fee,
                            exit_fee_total,
                            realized_pnl,
                            opened_at,
                            updated_at,
                            closed_at
                        )
                        VALUES (
                            ?, ?, ?, ?, ?, 'open', ?,
                            ?, ?, ?, ?, ?, ?, ?, 0, 0,
                            ?, ?, NULL
                        )
                    """, (
                        position_id,
                        order.exchange,
                        order.symbol,
                        market_type.value,
                        position_side.value,
                        order.order_id,
                        quantity,
                        quantity,
                        price,
                        stop_price,
                        leverage,
                        fee,
                        fee,
                        self._utc_text(order.created_at),
                        now_text,
                    ))

                    connection.execute("""
                        INSERT INTO position_events (
                            order_id,
                            position_id,
                            operation,
                            realized_pnl,
                            applied_at
                        )
                        VALUES (?, ?, 'entry', 0, ?)
                    """, (
                        order.order_id,
                        position_id,
                        now_text,
                    ))

                    row = self._get_position_row(
                        connection,
                        position_id,
                    )

                    return self._from_row(row)

            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "Cannot open position: duplicate order or "
                    "an open position already exists"
                ) from exc

    def apply_exit_order(
        self,
        position_id: str,
        order: OrderRecord,
    ) -> Position:

        exit_side, quantity, price, fee = (
            self._validate_filled_order(order)
        )

        now_text = datetime.now(UTC).isoformat()

        with self._lock:
            with self._transaction() as connection:

                # Duplicate exit order protection.
                previous = connection.execute("""
                    SELECT position_id, operation
                    FROM position_events
                    WHERE order_id = ?
                """, (order.order_id,)).fetchone()

                if previous:
                    if (
                        previous["position_id"] != position_id
                        or previous["operation"] != "exit"
                    ):
                        raise ValueError(
                            "Order ID was already processed"
                        )

                    row = self._get_position_row(
                        connection,
                        position_id,
                    )

                    return self._from_row(row)

                row = self._get_position_row(
                    connection,
                    position_id,
                )

                if row is None:
                    raise ValueError("Position not found")

                position = self._from_row(row)

                if position.status != PositionStatus.OPEN:
                    raise ValueError("Position is already closed")

                if order.exchange != position.exchange:
                    raise ValueError("Exchange mismatch")

                if order.symbol != position.symbol:
                    raise ValueError("Symbol mismatch")

                expected_exit_side = (
                    OrderSide.SELL
                    if position.side == PositionSide.LONG
                    else OrderSide.BUY
                )

                if exit_side != expected_exit_side:
                    raise ValueError(
                        "Exit order side conflicts with position"
                    )

                if quantity > position.remaining_quantity + self.EPSILON:
                    raise ValueError(
                        "Exit quantity exceeds remaining position"
                    )

                close_quantity = min(
                    quantity,
                    position.remaining_quantity,
                )

                # Allocate entry fee proportionally.
                allocation_ratio = (
                    close_quantity / position.remaining_quantity
                )

                allocated_entry_fee = (
                    position.remaining_entry_fee
                    * allocation_ratio
                )

                if position.side == PositionSide.LONG:
                    gross_pnl = (
                        price - position.entry_price
                    ) * close_quantity
                else:
                    gross_pnl = (
                        position.entry_price - price
                    ) * close_quantity

                # Net realized PnL includes entry and exit fees.
                realized_delta = (
                    gross_pnl
                    - allocated_entry_fee
                    - fee
                )

                remaining_quantity = (
                    position.remaining_quantity - close_quantity
                )

                remaining_entry_fee = (
                    position.remaining_entry_fee
                    - allocated_entry_fee
                )

                if remaining_quantity <= self.EPSILON:
                    remaining_quantity = 0.0
                    remaining_entry_fee = 0.0
                    status = PositionStatus.CLOSED
                    closed_at = now_text
                else:
                    status = PositionStatus.OPEN
                    closed_at = None

                new_realized_pnl = (
                    position.realized_pnl + realized_delta
                )

                connection.execute("""
                    UPDATE positions
                    SET
                        remaining_quantity = ?,
                        remaining_entry_fee = ?,
                        exit_fee_total = exit_fee_total + ?,
                        realized_pnl = ?,
                        status = ?,
                        updated_at = ?,
                        closed_at = ?
                    WHERE position_id = ?
                """, (
                    remaining_quantity,
                    remaining_entry_fee,
                    fee,
                    new_realized_pnl,
                    status.value,
                    now_text,
                    closed_at,
                    position_id,
                ))

                connection.execute("""
                    INSERT INTO position_events (
                        order_id,
                        position_id,
                        operation,
                        realized_pnl,
                        applied_at
                    )
                    VALUES (?, ?, 'exit', ?, ?)
                """, (
                    order.order_id,
                    position_id,
                    realized_delta,
                    now_text,
                ))

                updated = self._get_position_row(
                    connection,
                    position_id,
                )

                return self._from_row(updated)

    def get_realized_pnl_since(
        self,
        since: datetime,
        exchange: str | None = None,
        symbol: str | None = None,
    ) -> float:
        """Sum realized PnL from exit events since a UTC timestamp."""

        if not isinstance(since, datetime):
            raise ValueError("since must be a datetime")

        since_text = self._utc_text(since)

        query = """
            SELECT COALESCE(SUM(e.realized_pnl), 0.0)
            FROM position_events AS e
            JOIN positions AS p
                ON p.position_id = e.position_id
            WHERE e.operation = 'exit'
              AND e.applied_at >= ?
        """

        params = [since_text]

        if exchange is not None:
            query += " AND p.exchange = ?"
            params.append(exchange)

        if symbol is not None:
            query += " AND p.symbol = ?"
            params.append(symbol)

        with self._connection() as connection:
            row = connection.execute(
                query,
                params,
            ).fetchone()

        return float(row[0] or 0.0)

    def get_position(
        self,
        position_id: str,
    ) -> Position | None:

        with self._connection() as connection:
            row = self._get_position_row(
                connection,
                position_id,
            )

        return self._from_row(row) if row else None

    def get_open_positions(
        self,
        exchange: str | None = None,
        symbol: str | None = None,
    ) -> tuple[Position, ...]:

        query = """
            SELECT *
            FROM positions
            WHERE status = 'open'
        """
        params = []

        if exchange is not None:
            query += " AND exchange = ?"
            params.append(exchange)

        if symbol is not None:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY opened_at"

        with self._connection() as connection:
            rows = connection.execute(
                query,
                params,
            ).fetchall()

        return tuple(
            self._from_row(row)
            for row in rows
        )

    def list_positions(
        self,
        exchange: str | None = None,
        symbol: str | None = None,
    ) -> tuple[Position, ...]:

        query = "SELECT * FROM positions WHERE 1=1"
        params = []

        if exchange is not None:
            query += " AND exchange = ?"
            params.append(exchange)

        if symbol is not None:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY opened_at"

        with self._connection() as connection:
            rows = connection.execute(
                query,
                params,
            ).fetchall()

        return tuple(
            self._from_row(row)
            for row in rows
        )

    def is_stop_triggered(
        self,
        position_id: str,
        mark_price: float,
    ) -> bool:

        position = self.get_position(position_id)

        if position is None:
            raise ValueError("Position not found")

        return position.is_stop_triggered(mark_price)
