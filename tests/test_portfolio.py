from datetime import UTC, datetime
from dataclasses import replace

import pytest

from aitrader.execution.models import (
    OrderRecord,
    OrderStatus,
)
from aitrader.portfolio import (
    PositionManager,
    PositionSide,
    PositionStatus,
)
from aitrader.risk.engine import MarketType, OrderSide


def make_order(
    order_id,
    side=OrderSide.BUY,
    quantity=2.0,
    price=100.0,
    stop=95.0,
    fee=None,
    exchange="binance_spot",
    symbol="BTC/USDT",
):
    if fee is None:
        fee = quantity * price * 0.001

    return OrderRecord(
        order_id=order_id,
        client_order_id=f"client-{order_id}",
        exchange=exchange,
        symbol=symbol,
        side=side.value,
        status=OrderStatus.FILLED,
        requested_quantity=quantity,
        filled_quantity=quantity,
        reference_price=price,
        average_fill_price=price,
        filled_notional=quantity * price,
        fee=fee,
        stop_price=stop,
        execution_mode="paper",
        created_at=datetime.now(UTC),
    )


def make_manager(tmp_path):
    return PositionManager(tmp_path / "aitrader.db")


def test_open_spot_long_position(tmp_path):
    manager = make_manager(tmp_path)

    position = manager.open_from_order(
        make_order("entry-1"),
        MarketType.SPOT,
    )

    assert position.side == PositionSide.LONG
    assert position.status == PositionStatus.OPEN
    assert position.entry_price == 100.0
    assert position.entry_quantity == 2.0
    assert position.remaining_quantity == 2.0
    assert position.stop_price == 95.0


def test_open_futures_short_position(tmp_path):
    manager = make_manager(tmp_path)

    position = manager.open_from_order(
        make_order(
            "entry-2",
            side=OrderSide.SELL,
            stop=105.0,
        ),
        MarketType.FUTURES,
        leverage=2.0,
    )

    assert position.side == PositionSide.SHORT
    assert position.leverage == 2.0


def test_spot_short_position_is_rejected(tmp_path):
    manager = make_manager(tmp_path)

    with pytest.raises(ValueError):
        manager.open_from_order(
            make_order(
                "entry-3",
                side=OrderSide.SELL,
                stop=105.0,
            ),
            MarketType.SPOT,
        )


def test_rejected_order_cannot_open_position(tmp_path):
    manager = make_manager(tmp_path)

    order = make_order("entry-4")
    order = replace(
        order,
        status=OrderStatus.REJECTED,
    )

    with pytest.raises(ValueError):
        manager.open_from_order(order, MarketType.SPOT)


def test_live_order_is_rejected(tmp_path):
    manager = make_manager(tmp_path)

    order = make_order("entry-5")
    order = replace(
        order,
        execution_mode="live",
    )

    with pytest.raises(ValueError):
        manager.open_from_order(order, MarketType.SPOT)


def test_duplicate_entry_order_is_idempotent(tmp_path):
    manager = make_manager(tmp_path)
    order = make_order("entry-6")

    first = manager.open_from_order(order, MarketType.SPOT)
    second = manager.open_from_order(order, MarketType.SPOT)

    assert first.position_id == second.position_id
    assert len(manager.get_open_positions()) == 1


def test_second_open_position_same_market_is_rejected(tmp_path):
    manager = make_manager(tmp_path)

    manager.open_from_order(
        make_order("entry-7"),
        MarketType.SPOT,
    )

    with pytest.raises(ValueError):
        manager.open_from_order(
            make_order("entry-8"),
            MarketType.SPOT,
        )


def test_close_long_position_calculates_net_pnl(tmp_path):
    manager = make_manager(tmp_path)

    position = manager.open_from_order(
        make_order("entry-9"),
        MarketType.SPOT,
    )

    closed = manager.apply_exit_order(
        position.position_id,
        make_order(
            "exit-9",
            side=OrderSide.SELL,
            price=110.0,
            stop=95.0,
        ),
    )

    assert closed.status == PositionStatus.CLOSED
    assert closed.remaining_quantity == 0
    assert closed.realized_pnl == pytest.approx(19.58)
    assert closed.closed_at is not None


def test_close_short_position_calculates_net_pnl(tmp_path):
    manager = make_manager(tmp_path)

    position = manager.open_from_order(
        make_order(
            "entry-10",
            side=OrderSide.SELL,
            stop=105.0,
        ),
        MarketType.FUTURES,
        leverage=2.0,
    )

    closed = manager.apply_exit_order(
        position.position_id,
        make_order(
            "exit-10",
            side=OrderSide.BUY,
            price=90.0,
            stop=105.0,
        ),
    )

    assert closed.status == PositionStatus.CLOSED
    assert closed.realized_pnl == pytest.approx(19.62)


def test_partial_close_allocates_entry_fee(tmp_path):
    manager = make_manager(tmp_path)

    position = manager.open_from_order(
        make_order("entry-11"),
        MarketType.SPOT,
    )

    partial = manager.apply_exit_order(
        position.position_id,
        make_order(
            "exit-11a",
            side=OrderSide.SELL,
            quantity=1.0,
            price=110.0,
            fee=0.11,
        ),
    )

    assert partial.status == PositionStatus.OPEN
    assert partial.remaining_quantity == pytest.approx(1.0)
    assert partial.remaining_entry_fee == pytest.approx(0.1)
    assert partial.realized_pnl == pytest.approx(9.79)

    closed = manager.apply_exit_order(
        position.position_id,
        make_order(
            "exit-11b",
            side=OrderSide.SELL,
            quantity=1.0,
            price=90.0,
            fee=0.09,
        ),
    )

    assert closed.status == PositionStatus.CLOSED
    assert closed.realized_pnl == pytest.approx(-0.40)


def test_exit_quantity_cannot_exceed_position(tmp_path):
    manager = make_manager(tmp_path)

    position = manager.open_from_order(
        make_order("entry-12"),
        MarketType.SPOT,
    )

    with pytest.raises(ValueError):
        manager.apply_exit_order(
            position.position_id,
            make_order(
                "exit-12",
                side=OrderSide.SELL,
                quantity=3.0,
                price=110.0,
            ),
        )


def test_exit_side_must_match_position(tmp_path):
    manager = make_manager(tmp_path)

    position = manager.open_from_order(
        make_order("entry-13"),
        MarketType.SPOT,
    )

    with pytest.raises(ValueError):
        manager.apply_exit_order(
            position.position_id,
            make_order(
                "exit-13",
                side=OrderSide.BUY,
            ),
        )


def test_exit_order_market_must_match(tmp_path):
    manager = make_manager(tmp_path)

    position = manager.open_from_order(
        make_order("entry-14"),
        MarketType.SPOT,
    )

    with pytest.raises(ValueError):
        manager.apply_exit_order(
            position.position_id,
            make_order(
                "exit-14",
                side=OrderSide.SELL,
                exchange="upbit",
            ),
        )


def test_duplicate_exit_order_is_idempotent(tmp_path):
    manager = make_manager(tmp_path)

    position = manager.open_from_order(
        make_order("entry-15"),
        MarketType.SPOT,
    )

    exit_order = make_order(
        "exit-15",
        side=OrderSide.SELL,
        price=110.0,
    )

    first = manager.apply_exit_order(
        position.position_id,
        exit_order,
    )

    second = manager.apply_exit_order(
        position.position_id,
        exit_order,
    )

    assert first.realized_pnl == second.realized_pnl
    assert len(manager.list_positions()) == 1


def test_long_stop_trigger(tmp_path):
    manager = make_manager(tmp_path)

    position = manager.open_from_order(
        make_order("entry-16"),
        MarketType.SPOT,
    )

    assert not manager.is_stop_triggered(
        position.position_id,
        96.0,
    )

    assert manager.is_stop_triggered(
        position.position_id,
        95.0,
    )


def test_short_stop_trigger(tmp_path):
    manager = make_manager(tmp_path)

    position = manager.open_from_order(
        make_order(
            "entry-17",
            side=OrderSide.SELL,
            stop=105.0,
        ),
        MarketType.FUTURES,
    )

    assert not manager.is_stop_triggered(
        position.position_id,
        104.0,
    )

    assert manager.is_stop_triggered(
        position.position_id,
        105.0,
    )


def test_unrealized_pnl(tmp_path):
    manager = make_manager(tmp_path)

    position = manager.open_from_order(
        make_order("entry-18"),
        MarketType.SPOT,
    )

    assert position.unrealized_pnl(110.0) == pytest.approx(19.8)


def test_position_survives_restart(tmp_path):
    db_path = tmp_path / "aitrader.db"

    first_manager = PositionManager(db_path)

    created = first_manager.open_from_order(
        make_order("entry-19"),
        MarketType.SPOT,
    )

    second_manager = PositionManager(db_path)

    restored = second_manager.get_position(
        created.position_id
    )

    assert restored is not None
    assert restored.position_id == created.position_id
    assert restored.remaining_quantity == 2.0
    assert restored.status == PositionStatus.OPEN


def test_realized_pnl_since_uses_exit_event_dates(tmp_path):
    import sqlite3

    manager = make_manager(tmp_path)
    position = manager.open_from_order(
        make_order("entry-daily"), MarketType.SPOT
    )

    for order_id, price, fee in (
        ("exit-yesterday", 110.0, 0.11),
        ("exit-today", 90.0, 0.09),
    ):
        manager.apply_exit_order(
            position.position_id,
            make_order(
                order_id,
                side=OrderSide.SELL,
                quantity=1.0,
                price=price,
                fee=fee,
            ),
        )

    with sqlite3.connect(manager.db_path) as connection:
        connection.executemany(
            "UPDATE position_events SET applied_at = ? WHERE order_id = ?",
            [
                ("2026-11-17T23:59:59.000000+00:00", "exit-yesterday"),
                ("2026-11-18T00:00:00.000000+00:00", "exit-today"),
            ],
        )

    today = manager.get_realized_pnl_since(
        datetime(2026, 11, 18, tzinfo=UTC),
        exchange="binance_spot",
        symbol="BTC/USDT",
    )
    total = manager.get_realized_pnl_since(
        datetime(2026, 11, 17, tzinfo=UTC),
        exchange="binance_spot",
        symbol="BTC/USDT",
    )

    assert today == pytest.approx(-10.19)
    assert total == pytest.approx(-0.40)
