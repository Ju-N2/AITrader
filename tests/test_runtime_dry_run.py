import logging
import time
from types import SimpleNamespace as N

from aitrader.core.runtime import PaperTradingRuntime, TIMEFRAMES


def make_runtime():
    r = N()
    r.config = N(execution_timeframe="1h", market_type="spot", exchange="binance_spot", symbol="BTC/USDT")
    r.logger = logging.getLogger("test.runtime.dryrun")
    r.last_dry_run_decision_time = 0
    r.runtime_key = "binance_spot:BTC/USDT:1h"
    ts = int(time.time() * 1000) - 60_000
    signal = N(bias=N(value="bullish"), score=3, max_score=5)
    item = (ts, object(), object(), signal)
    r._refresh_due_data = lambda now: None
    r._build_signal_series = lambda tf, now: [item]
    r._build_aligned_signals = lambda series, dt: ("aligned")
    r.multi_engine = N(evaluate=lambda _: N(bias=N(value="bullish"), weighted_score=2.5))
    r.ai_engine = N(analyze=lambda *args: N(bias=N(value="bullish"), score=.5, confidence=.8, agreement=.8, regime=N(value="bull_trend"), volatility=N(value="normal"), reports=()))
    r._fetch_price = lambda: 100.0
    account = N(open_positions=0)
    r._build_account_state = lambda mark_price: account
    calls = []
    r.ai_gate = N(evaluate=lambda **kw: (calls.append(kw) or N(approved=False, rejection_reason="test dry run")))
    saved = []
    r.state_store = N(get_last_processed=lambda key: 0, save_processed=lambda key, ts: saved.append((key, ts)))
    r._monitor_stops = lambda now: False
    return r, ts, account, calls, saved


def test_same_dry_run_candle_is_not_analyzed_twice():
    r, _, _, calls, _ = make_runtime()
    assert PaperTradingRuntime.run_cycle(r, paper_mode=False) == "dry_run"
    assert PaperTradingRuntime.run_cycle(r, paper_mode=False) == "no_new_candle"
    assert len(calls) == 1


def test_dry_run_does_not_consume_paper_checkpoint():
    r, ts, account, _, saved = make_runtime()
    assert PaperTradingRuntime.run_cycle(r, paper_mode=False) == "dry_run"
    account.open_positions = 1
    result = PaperTradingRuntime.run_cycle(r, paper_mode=True)
    assert result == "position_open"
    assert saved == [(r.runtime_key, ts)]
