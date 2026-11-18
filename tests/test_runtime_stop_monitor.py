import logging
from types import SimpleNamespace

from aitrader.core.runtime import PaperTradingRuntime
from aitrader.execution.models import OrderStatus
from aitrader.portfolio.models import PositionStatus


class FakePositionManager:
    def __init__(self, position):
        self.position = position
        self.calls = 0

    def get_open_positions(self, exchange, symbol):
        self.calls += 1
        return (self.position,)


class FakeTradingService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def close_position(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def build_runtime(order_status, position_status):
    position = SimpleNamespace(
        position_id="test-position-001",
        is_stop_triggered=lambda price: price <= 95.0,
    )

    runtime = PaperTradingRuntime.__new__(PaperTradingRuntime)
    runtime.config = SimpleNamespace(
        exchange="binance_spot",
        symbol="BTC/USDT",
    )
    runtime.position_manager = FakePositionManager(position)
    runtime._fetch_price = lambda: 90.0
    runtime.logger = logging.getLogger("test.runtime.stop")

    runtime.trading_service = FakeTradingService(
        SimpleNamespace(
            order=SimpleNamespace(
                status=order_status,
                rejection_reason=(
                    "Test rejection"
                    if order_status == OrderStatus.REJECTED
                    else None
                ),
            ),
            position=SimpleNamespace(status=position_status),
        )
    )

    return runtime


def test_stop_monitor_confirms_completed_exit():
    runtime = build_runtime(
        OrderStatus.FILLED,
        PositionStatus.CLOSED,
    )

    result = runtime._monitor_stops(120000)

    assert result is True
    assert len(runtime.trading_service.calls) == 1


def test_rejected_stop_order_keeps_monitoring():
    runtime = build_runtime(
        OrderStatus.REJECTED,
        PositionStatus.OPEN,
    )

    first = runtime._monitor_stops(120000)
    second = runtime._monitor_stops(180000)

    assert first is False
    assert second is False
    assert runtime.position_manager.calls == 2
    assert len(runtime.trading_service.calls) == 2


def test_filled_order_does_not_confirm_open_position_exit():
    runtime = build_runtime(
        OrderStatus.FILLED,
        PositionStatus.OPEN,
    )

    result = runtime._monitor_stops(120000)

    assert result is False
    assert len(runtime.trading_service.calls) == 1


import logging
from datetime import UTC, datetime
from types import SimpleNamespace

from aitrader.core.paper_trading import PaperTradingService
from aitrader.core.runtime import PaperTradingRuntime
from aitrader.execution import OrderEngine
from aitrader.execution.models import OrderRecord, OrderStatus
from aitrader.portfolio import PositionManager, PositionStatus
from aitrader.risk.engine import MarketType


def test_runtime_restores_open_position_and_closes_stop_after_restart(tmp_path):
    db_path = tmp_path / "restart-stop.db"

    # Process 1: persist an open paper position.
    first_manager = PositionManager(db_path)
    entry_order = OrderRecord(
        order_id="restart-entry-001",
        client_order_id="restart-entry-client-001",
        exchange="binance_spot",
        symbol="BTC/USDT",
        side="buy",
        status=OrderStatus.FILLED,
        requested_quantity=1.0,
        filled_quantity=1.0,
        reference_price=100.0,
        average_fill_price=100.0,
        filled_notional=100.0,
        fee=0.1,
        stop_price=95.0,
        execution_mode="paper",
        created_at=datetime.now(UTC),
    )
    opened = first_manager.open_from_order(
        entry_order,
        MarketType.SPOT,
    )
    assert opened.status == PositionStatus.OPEN

    # Process 2: create new objects against the same SQLite database.
    restored_manager = PositionManager(db_path)
    restored = restored_manager.get_position(opened.position_id)

    assert restored is not None
    assert restored.status == PositionStatus.OPEN
    assert restored.remaining_quantity == 1.0

    service = PaperTradingService(
        ai_gate=None,
        order_engine=OrderEngine(),
        position_manager=restored_manager,
        notifier=None,
    )

    runtime = PaperTradingRuntime.__new__(PaperTradingRuntime)
    runtime.config = SimpleNamespace(
        exchange="binance_spot",
        symbol="BTC/USDT",
    )
    runtime.position_manager = restored_manager
    runtime.trading_service = service
    runtime._fetch_price = lambda: 90.0
    runtime.logger = logging.getLogger("test.runtime.restart")

    # The restored position is below its 95.0 stop and should be closed.
    stopped = runtime._monitor_stops(now_ms=1_800_000_000_000)

    final_position = restored_manager.get_position(
        opened.position_id
    )

    assert stopped is True
    assert final_position.status == PositionStatus.CLOSED
    assert final_position.remaining_quantity == 0.0
    assert restored_manager.get_open_positions(
        exchange="binance_spot",
        symbol="BTC/USDT",
    ) == ()


def test_stop_exit_is_processed_before_ohlcv_refresh():
    runtime = PaperTradingRuntime.__new__(PaperTradingRuntime)
    runtime.config = SimpleNamespace(
        exchange="binance_spot",
        symbol="BTC/USDT",
        execution_timeframe="1h",
    )
    runtime.runtime_key = "binance_spot:BTC/USDT:1h"
    runtime.repository = SimpleNamespace(
        get_candles=lambda *args, **kwargs: []
    )
    runtime.state_store = SimpleNamespace(
        save_processed=lambda *args: None
    )

    stop_calls = []
    runtime._monitor_stops = lambda now: (
        stop_calls.append(now) or True
    )

    def fail_refresh(now):
        raise AssertionError("OHLCV refresh should not run after stop exit")

    runtime._refresh_due_data = fail_refresh

    result = runtime.run_cycle(paper_mode=True)

    assert result == "stop_exit"
    assert len(stop_calls) == 1


def test_ohlcv_network_error_blocks_new_entry():
    runtime = PaperTradingRuntime.__new__(PaperTradingRuntime)

    runtime._monitor_stops = lambda now: False

    def fail_refresh(now):
        raise RuntimeError("Exchange API unavailable")

    runtime._refresh_due_data = fail_refresh

    try:
        runtime.run_cycle(paper_mode=True)
    except RuntimeError as exc:
        assert str(exc) == "Exchange API unavailable"
    else:
        raise AssertionError("Runtime should stop the cycle on data failure")
