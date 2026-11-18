import pytest

from aitrader.execution import (
    OrderEngine,
    OrderRequest,
    OrderStatus,
    PaperExecutionConfig,
)
from aitrader.risk.engine import OrderSide, RiskDecision


def make_decision(
    side=OrderSide.BUY,
    approved=True,
    stop_price=95.0,
    quantity=2.0,
):
    if side == OrderSide.SELL:
        stop_price = 105.0

    return RiskDecision(
        approved=approved,
        quantity=quantity,
        notional=quantity * 100,
        risk_budget=10.0,
        estimated_max_loss=8.0,
        margin_required=200.0,
        effective_leverage=1.0,
        stop_price=stop_price,
        rejection_reason=None if approved else "Risk rejected",
    )


def make_request(
    side=OrderSide.BUY,
    client_order_id="test-001",
):
    return OrderRequest(
        exchange="binance_spot",
        symbol="BTC/USDT",
        side=side,
        reference_price=100.0,
        client_order_id=client_order_id,
    )


def test_approved_buy_order_is_filled():
    engine = OrderEngine()

    record = engine.submit_paper_order(
        make_request(),
        make_decision(),
    )

    assert record.status == OrderStatus.FILLED
    assert record.filled_quantity == 2.0
    assert record.average_fill_price == pytest.approx(100.05)
    assert record.filled_notional == pytest.approx(200.10)
    assert record.fee == pytest.approx(0.2001)
    assert record.stop_price == 95.0
    assert record.execution_mode == "paper"


def test_approved_sell_order_is_filled():
    engine = OrderEngine()

    record = engine.submit_paper_order(
        make_request(OrderSide.SELL),
        make_decision(OrderSide.SELL),
    )

    assert record.status == OrderStatus.FILLED
    assert record.average_fill_price == pytest.approx(99.95)
    assert record.stop_price == 105.0


def test_unapproved_order_is_rejected():
    engine = OrderEngine()

    record = engine.submit_paper_order(
        make_request(),
        make_decision(approved=False),
    )

    assert record.status == OrderStatus.REJECTED
    assert record.filled_quantity == 0
    assert record.rejection_reason == "Risk rejected"


def test_missing_stop_price_is_rejected():
    engine = OrderEngine()

    decision = RiskDecision(
        approved=True,
        quantity=1.0,
    )

    record = engine.submit_paper_order(
        make_request(),
        decision,
    )

    assert record.status == OrderStatus.REJECTED
    assert record.rejection_reason == (
        "Approved order has no valid stop price"
    )


def test_wrong_long_stop_direction_is_rejected():
    engine = OrderEngine()

    decision = make_decision(stop_price=101.0)

    record = engine.submit_paper_order(
        make_request(),
        decision,
    )

    assert record.status == OrderStatus.REJECTED


def test_wrong_short_stop_direction_is_rejected():
    engine = OrderEngine()

    decision = RiskDecision(
        approved=True,
        quantity=1.0,
        stop_price=95.0,
    )

    record = engine.submit_paper_order(
        make_request(OrderSide.SELL),
        decision,
    )

    assert record.status == OrderStatus.REJECTED


def test_duplicate_client_order_is_idempotent():
    engine = OrderEngine()

    request = make_request()

    first = engine.submit_paper_order(
        request,
        make_decision(),
    )

    second = engine.submit_paper_order(
        request,
        make_decision(),
    )

    assert first.order_id == second.order_id
    assert len(engine.list_orders()) == 1


def test_client_order_id_cannot_be_reused_with_different_payload():
    engine = OrderEngine()

    engine.submit_paper_order(
        make_request(),
        make_decision(),
    )

    changed_request = OrderRequest(
        exchange="binance_spot",
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        reference_price=100.0,
        client_order_id="test-001",
    )

    with pytest.raises(ValueError):
        engine.submit_paper_order(
            changed_request,
            make_decision(),
        )


def test_order_lookup():
    engine = OrderEngine()

    record = engine.submit_paper_order(
        make_request(),
        make_decision(),
    )

    assert engine.get_order(record.order_id) == record

    assert engine.get_by_client_order_id(
        "binance_spot",
        "test-001",
    ) == record


def test_order_listing_filters():
    engine = OrderEngine()

    engine.submit_paper_order(
        make_request(),
        make_decision(),
    )

    second_request = OrderRequest(
        exchange="upbit",
        symbol="BTC/KRW",
        side=OrderSide.BUY,
        reference_price=100_000_000.0,
        client_order_id="upbit-001",
    )

    second_decision = RiskDecision(
        approved=True,
        quantity=0.001,
        stop_price=95_000_000.0,
    )

    engine.submit_paper_order(
        second_request,
        second_decision,
    )

    assert len(engine.list_orders()) == 2
    assert len(engine.list_orders(exchange="upbit")) == 1
    assert len(engine.list_orders(symbol="BTC/USDT")) == 1


def test_invalid_execution_config():
    with pytest.raises(ValueError):
        PaperExecutionConfig(fee_rate=-0.1)

    with pytest.raises(ValueError):
        PaperExecutionConfig(slippage_rate=1.0)


def test_invalid_order_request():
    with pytest.raises(ValueError):
        OrderRequest(
            exchange="binance_spot",
            symbol="BTC/USDT",
            side="invalid",
            reference_price=100.0,
        )

    with pytest.raises(ValueError):
        OrderRequest(
            exchange="binance_spot",
            symbol="BTC/USDT",
            side="buy",
            reference_price=0,
        )
