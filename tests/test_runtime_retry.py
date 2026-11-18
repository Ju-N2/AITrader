import logging
from types import SimpleNamespace

from aitrader.core.runtime import PaperTradingRuntime


def test_runtime_uses_bounded_backoff_and_resets_after_success(monkeypatch):
    runtime = PaperTradingRuntime.__new__(PaperTradingRuntime)
    runtime.config = SimpleNamespace(
        poll_interval_seconds=5.0,
        exchange="binance_spot",
        symbol="BTC/USDT",
    )
    runtime.logger = logging.getLogger("test.runtime.retry")
    runtime.trading_service = SimpleNamespace(notifier=None)

    outcomes = iter((RuntimeError("offline"), "ok", RuntimeError("offline")))

    def run_cycle(paper_mode):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    runtime.run_cycle = run_cycle
    delays = []

    def fake_sleep(delay):
        delays.append(delay)
        if len(delays) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr("aitrader.core.runtime.time.sleep", fake_sleep)

    runtime.run_forever(paper_mode=True)

    assert delays == [10.0, 5.0, 10.0]


def test_runtime_backoff_is_capped(monkeypatch):
    runtime = PaperTradingRuntime.__new__(PaperTradingRuntime)
    runtime.config = SimpleNamespace(
        poll_interval_seconds=600.0,
        exchange="binance_spot",
        symbol="BTC/USDT",
    )
    runtime.logger = logging.getLogger("test.runtime.retry.cap")
    runtime.trading_service = SimpleNamespace(notifier=None)
    runtime.run_cycle = lambda paper_mode: (_ for _ in ()).throw(
        RuntimeError("offline")
    )
    delays = []

    def fake_sleep(delay):
        delays.append(delay)
        raise KeyboardInterrupt

    monkeypatch.setattr("aitrader.core.runtime.time.sleep", fake_sleep)

    runtime.run_forever(paper_mode=False)

    assert delays == [300.0]
