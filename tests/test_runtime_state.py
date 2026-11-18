from aitrader.core.runtime import RuntimeStateStore


def test_daily_start_equity_is_persisted(tmp_path):
    db_path = tmp_path / "runtime.db"

    store = RuntimeStateStore(db_path)

    first = store.get_or_create_daily_start_equity(
        runtime_key="binance_spot:BTC/USDT:1h",
        utc_date="2026-11-18",
        start_equity=10000.0,
    )

    # A second call on the same day must not overwrite the baseline.
    second = store.get_or_create_daily_start_equity(
        runtime_key="binance_spot:BTC/USDT:1h",
        utc_date="2026-11-18",
        start_equity=9500.0,
    )

    # Reopening the store must preserve the same value.
    reopened = RuntimeStateStore(db_path)

    third = reopened.get_or_create_daily_start_equity(
        runtime_key="binance_spot:BTC/USDT:1h",
        utc_date="2026-11-18",
        start_equity=9000.0,
    )

    assert first == 10000.0
    assert second == 10000.0
    assert third == 10000.0


def test_daily_start_equity_is_separated_by_date_and_runtime(tmp_path):
    store = RuntimeStateStore(tmp_path / "runtime.db")

    day_one = store.get_or_create_daily_start_equity(
        "binance_spot:BTC/USDT:1h",
        "2026-11-18",
        10000.0,
    )

    day_two = store.get_or_create_daily_start_equity(
        "binance_spot:BTC/USDT:1h",
        "2026-11-19",
        10200.0,
    )

    another_runtime = store.get_or_create_daily_start_equity(
        "binance_spot:ETH/USDT:1h",
        "2026-11-18",
        5000.0,
    )

    assert day_one == 10000.0
    assert day_two == 10200.0
    assert another_runtime == 5000.0
from datetime import UTC
from types import SimpleNamespace

import pytest

from aitrader.core.runtime import PaperTradingRuntime, RuntimeStateStore
from aitrader.portfolio.models import PositionStatus
from aitrader.risk.engine import MarketType


def test_runtime_account_state_uses_daily_realized_pnl(tmp_path):
    class FakePositionManager:
        def list_positions(self, exchange, symbol):
            assert exchange == "binance_spot"
            assert symbol == "BTC/USDT"
            return (
                SimpleNamespace(
                    status=PositionStatus.CLOSED,
                    realized_pnl=-0.40,
                ),
            )

        def get_realized_pnl_since(self, since, exchange, symbol):
            assert since.tzinfo == UTC
            assert (since.hour, since.minute, since.second) == (0, 0, 0)
            assert exchange == "binance_spot"
            assert symbol == "BTC/USDT"
            return -10.19

    runtime = PaperTradingRuntime.__new__(PaperTradingRuntime)
    runtime.config = SimpleNamespace(
        exchange="binance_spot",
        symbol="BTC/USDT",
        market_type=MarketType.SPOT,
        leverage=1.0,
        initial_capital=10_000.0,
    )
    runtime.position_manager = FakePositionManager()
    runtime.state_store = RuntimeStateStore(tmp_path / "runtime.db")
    runtime.runtime_key = "binance_spot:BTC/USDT:1h"

    account = runtime._build_account_state()

    assert account.daily_realized_pnl == pytest.approx(-10.19)
    assert account.daily_start_equity == pytest.approx(10_009.79)
    assert account.account_equity == pytest.approx(9_999.60)
    assert account.open_positions == 0


def test_runtime_checkpoint_is_persisted_and_monotonic(tmp_path):
    db_path = tmp_path / "runtime.db"
    store = RuntimeStateStore(db_path)

    assert store.get_last_processed("binance_spot:BTC/USDT:1h") == 0

    store.save_processed("binance_spot:BTC/USDT:1h", 1000)

    reopened = RuntimeStateStore(db_path)
    assert reopened.get_last_processed("binance_spot:BTC/USDT:1h") == 1000

    # Older timestamps must never move the checkpoint backwards.
    reopened.save_processed("binance_spot:BTC/USDT:1h", 900)
    assert reopened.get_last_processed("binance_spot:BTC/USDT:1h") == 1000

    # Newer timestamps advance the checkpoint.
    reopened.save_processed("binance_spot:BTC/USDT:1h", 1100)
    assert reopened.get_last_processed("binance_spot:BTC/USDT:1h") == 1100

    # Separate runtimes maintain independent checkpoints.
    assert reopened.get_last_processed("binance_spot:ETH/USDT:1h") == 0
