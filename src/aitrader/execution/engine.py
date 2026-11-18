from datetime import UTC, datetime
from math import isfinite
from threading import RLock
from uuid import uuid4

from aitrader.execution.models import (
    OrderRecord,
    OrderRequest,
    OrderStatus,
    PaperExecutionConfig,
)
from aitrader.risk.engine import OrderSide, RiskDecision


class OrderEngine:
    """
    Paper-only order execution engine.

    This engine never communicates with an exchange.
    It simulates immediate market order fills.
    """

    def __init__(
        self,
        config: PaperExecutionConfig | None = None,
    ):
        self.config = config or PaperExecutionConfig()

        self._orders: dict[str, OrderRecord] = {}
        self._client_orders: dict[tuple[str, str], str] = {}
        self._fingerprints: dict[tuple[str, str], tuple] = {}

        self._lock = RLock()

    @staticmethod
    def _fingerprint(request, decision):
        return (
            request.exchange.lower(),
            request.symbol,
            request.side,
            request.reference_price,
            decision.approved,
            decision.quantity,
            decision.stop_price,
            decision.rejection_reason,
            request.reduce_only,
        )

    def _rejected_record(
        self,
        request,
        order_id,
        client_order_id,
        reason,
        stop_price=None,
    ):
        return OrderRecord(
            order_id=order_id,
            client_order_id=client_order_id,
            exchange=request.exchange,
            symbol=request.symbol,
            side=request.side.value,
            status=OrderStatus.REJECTED,
            requested_quantity=0.0,
            filled_quantity=0.0,
            reference_price=request.reference_price,
            average_fill_price=None,
            filled_notional=0.0,
            fee=0.0,
            stop_price=stop_price,
            execution_mode="paper",
            created_at=datetime.now(UTC),
            rejection_reason=reason,
            reduce_only=request.reduce_only,
        )

    def submit_paper_order(
        self,
        request: OrderRequest,
        risk_decision: RiskDecision,
    ) -> OrderRecord:

        if not isinstance(request, OrderRequest):
            raise TypeError("request must be OrderRequest")

        if not isinstance(risk_decision, RiskDecision):
            raise TypeError("risk_decision must be RiskDecision")

        client_order_id = (
            request.client_order_id or uuid4().hex
        )

        key = (
            request.exchange.lower(),
            client_order_id,
        )

        fingerprint = self._fingerprint(
            request,
            risk_decision,
        )

        with self._lock:

            # Idempotency check.
            existing_id = self._client_orders.get(key)

            if existing_id is not None:
                if self._fingerprints[key] != fingerprint:
                    raise ValueError(
                        "Client order ID reused with different payload"
                    )

                return self._orders[existing_id]

            order_id = uuid4().hex

            # Only approved decisions may be executed.
            if not risk_decision.approved:
                record = self._rejected_record(
                    request,
                    order_id,
                    client_order_id,
                    risk_decision.rejection_reason
                    or "Risk approval rejected",
                )

            else:
                quantity = risk_decision.quantity
                stop_price = risk_decision.stop_price

                valid_quantity = (
                    isinstance(quantity, (int, float))
                    and not isinstance(quantity, bool)
                    and isfinite(quantity)
                    and quantity > 0
                )

                if request.reduce_only:
                    valid_stop = stop_price is None
                else:
                    valid_stop = (
                        isinstance(stop_price, (int, float))
                        and not isinstance(stop_price, bool)
                        and isfinite(stop_price)
                        and stop_price > 0
                    )

                if not valid_quantity:
                    record = self._rejected_record(
                        request,
                        order_id,
                        client_order_id,
                        "Invalid approved quantity",
                    )

                elif not valid_stop:
                    reason = (
                        "Reduce-only order must not include stop price"
                        if request.reduce_only
                        else "Approved order has no valid stop price"
                    )
                    record = self._rejected_record(
                        request,
                        order_id,
                        client_order_id,
                        reason,
                    )

                else:
                    side = request.side

                    if side == OrderSide.BUY:
                        fill_price = (
                            request.reference_price
                            * (1 + self.config.slippage_rate)
                        )
                        valid_stop_direction = (
                            request.reduce_only
                            or stop_price < fill_price
                        )

                    else:
                        fill_price = (
                            request.reference_price
                            * (1 - self.config.slippage_rate)
                        )
                        valid_stop_direction = (
                            request.reduce_only
                            or stop_price > fill_price
                        )

                    if not valid_stop_direction:
                        record = self._rejected_record(
                            request,
                            order_id,
                            client_order_id,
                            "Stop price is on the wrong side of fill price",
                            stop_price,
                        )

                    elif not isfinite(fill_price) or fill_price <= 0:
                        record = self._rejected_record(
                            request,
                            order_id,
                            client_order_id,
                            "Invalid simulated fill price",
                            stop_price,
                        )

                    else:
                        notional = quantity * fill_price
                        fee = notional * self.config.fee_rate

                        record = OrderRecord(
                            order_id=order_id,
                            client_order_id=client_order_id,
                            exchange=request.exchange,
                            symbol=request.symbol,
                            side=side.value,
                            status=OrderStatus.FILLED,
                            requested_quantity=quantity,
                            filled_quantity=quantity,
                            reference_price=request.reference_price,
                            average_fill_price=fill_price,
                            filled_notional=notional,
                            fee=fee,
                            stop_price=(
                                None if request.reduce_only else stop_price
                            ),
                            execution_mode="paper",
                            created_at=datetime.now(UTC),
                            reduce_only=request.reduce_only,
                        )

            self._orders[order_id] = record
            self._client_orders[key] = order_id
            self._fingerprints[key] = fingerprint

            return record

    def get_order(self, order_id: str) -> OrderRecord | None:
        with self._lock:
            return self._orders.get(order_id)

    def get_by_client_order_id(
        self,
        exchange: str,
        client_order_id: str,
    ) -> OrderRecord | None:

        with self._lock:
            order_id = self._client_orders.get(
                (exchange.lower(), client_order_id)
            )

            if order_id is None:
                return None

            return self._orders[order_id]

    def list_orders(
        self,
        exchange: str | None = None,
        symbol: str | None = None,
    ) -> tuple[OrderRecord, ...]:

        with self._lock:
            orders = tuple(self._orders.values())

        if exchange is not None:
            orders = tuple(
                order for order in orders
                if order.exchange.lower() == exchange.lower()
            )

        if symbol is not None:
            orders = tuple(
                order for order in orders
                if order.symbol == symbol
            )

        return orders
