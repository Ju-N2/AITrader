from aitrader.execution import (
    OrderEngine,
    OrderRequest,
    OrderStatus,
    PaperExecutionConfig,
)
from aitrader.portfolio import (
    PositionManager,
    PositionSide,
    PositionStatus,
)
from aitrader.risk import (
    AccountRiskState,
    MarketType,
    OrderSide,
    RiskConfig,
    RiskDecision,
    RiskEngine,
    RiskManager,
)
from aitrader.strategy.models import MarketBias, StrategySignal
from aitrader.strategy.multi_timeframe import MultiTimeframeResult


def make_risk_manager():
    config = RiskConfig(
        risk_per_trade_fraction=0.01,
        max_position_fraction=0.25,
        max_total_exposure_fraction=0.50,
        max_daily_loss_fraction=0.02,
        max_leverage=3.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
        min_notional=10.0,
        max_open_positions=3,
    )
    return RiskManager(RiskEngine(config), atr_stop_multiplier=3.0)


def make_signal(bias):
    score = 4 if bias == MarketBias.BULLISH else -4
    return StrategySignal(
        exchange="binance_spot",
        symbol="BTC/USDT",
        timeframe="1m",
        timestamp_ms=1_700_000_000_000,
        bias=bias,
        score=score,
        max_score=5,
        reasons=("Integration test",),
        atr=100.0,
    )


def make_consensus(bias):
    return MultiTimeframeResult(
        exchange="binance_spot",
        symbol="BTC/USDT",
        bias=bias,
        weighted_score=2.5 if bias == MarketBias.BULLISH else -2.5,
        available_timeframes=("1m", "5m", "15m", "1h", "4h"),
        missing_timeframes=(),
        reasons=("Integration test",),
    )


def make_account(leverage=1.0):
    return AccountRiskState(
        account_equity=10_000.0,
        available_balance=10_000.0,
        daily_start_equity=10_000.0,
        leverage=leverage,
    )


def make_order_engine():
    return OrderEngine(
        PaperExecutionConfig(
            fee_rate=0.001,
            slippage_rate=0.0005,
        )
    )


def test_complete_spot_long_pipeline(tmp_path):
    risk_manager = make_risk_manager()
    order_engine = make_order_engine()
    portfolio = PositionManager(tmp_path / "pipeline.db")

    decision = risk_manager.evaluate(
        consensus=make_consensus(MarketBias.BULLISH),
        execution_signal=make_signal(MarketBias.BULLISH),
        entry_price=10_000.0,
        account=make_account(),
        market_type=MarketType.SPOT,
    )

    assert decision.approved
    assert decision.stop_price == 9_700.0

    request = OrderRequest(
        exchange="binance_spot",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        reference_price=10_000.0,
        client_order_id="integration-long-entry",
    )

    filled_order = order_engine.submit_paper_order(
        request,
        decision,
    )

    assert filled_order.status == OrderStatus.FILLED

    position = portfolio.open_from_order(
        filled_order,
        MarketType.SPOT,
    )

    assert position.status == PositionStatus.OPEN
    assert position.side == PositionSide.LONG
    assert position.entry_price == filled_order.average_fill_price
    assert position.stop_price == 9_700.0

    # Unrealized PnL and stop monitoring.
    assert position.unrealized_pnl(10_200.0) > 0
    assert portfolio.is_stop_triggered(
        position.position_id,
        9_700.0,
    )

    # Simulated exit.
    exit_request = OrderRequest(
        exchange="binance_spot",
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        reference_price=10_100.0,
        client_order_id="integration-long-exit",
    )

    exit_decision = RiskDecision(
        approved=True,
        quantity=position.remaining_quantity,
        stop_price=10_200.0,
    )

    exit_order = order_engine.submit_paper_order(
        exit_request,
        exit_decision,
    )

    assert exit_order.status == OrderStatus.FILLED

    closed_position = portfolio.apply_exit_order(
        position.position_id,
        exit_order,
    )

    assert closed_position.status == PositionStatus.CLOSED
    assert closed_position.remaining_quantity == 0
    assert closed_position.realized_pnl > 0

    # Replaying the same exit must not double-count PnL.
    replayed = portfolio.apply_exit_order(
        position.position_id,
        exit_order,
    )

    assert replayed.realized_pnl == closed_position.realized_pnl


def test_rejected_risk_decision_never_opens_position(tmp_path):
    risk_manager = make_risk_manager()
    order_engine = make_order_engine()
    portfolio = PositionManager(tmp_path / "rejected.db")

    decision = risk_manager.evaluate(
        consensus=make_consensus(MarketBias.NEUTRAL),
        execution_signal=make_signal(MarketBias.BULLISH),
        entry_price=10_000.0,
        account=make_account(),
        market_type=MarketType.SPOT,
    )

    assert not decision.approved

    order = order_engine.submit_paper_order(
        OrderRequest(
            exchange="binance_spot",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            reference_price=10_000.0,
            client_order_id="integration-rejected",
        ),
        decision,
    )

    assert order.status == OrderStatus.REJECTED

    try:
        portfolio.open_from_order(order, MarketType.SPOT)
    except ValueError:
        pass
    else:
        raise AssertionError("Rejected order opened a position")

    assert portfolio.get_open_positions() == ()


def test_complete_futures_short_pipeline(tmp_path):
    risk_manager = make_risk_manager()
    order_engine = make_order_engine()
    portfolio = PositionManager(tmp_path / "futures.db")

    decision = risk_manager.evaluate(
        consensus=make_consensus(MarketBias.BEARISH),
        execution_signal=make_signal(MarketBias.BEARISH),
        entry_price=10_000.0,
        account=make_account(leverage=2.0),
        market_type=MarketType.FUTURES,
    )

    assert decision.approved
    assert decision.stop_price == 10_300.0

    order = order_engine.submit_paper_order(
        OrderRequest(
            exchange="binance_spot",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            reference_price=10_000.0,
            client_order_id="integration-short-entry",
        ),
        decision,
    )

    assert order.status == OrderStatus.FILLED

    position = portfolio.open_from_order(
        order,
        MarketType.FUTURES,
        leverage=2.0,
    )

    assert position.side == PositionSide.SHORT
    assert position.status == PositionStatus.OPEN
    assert position.leverage == 2.0
    assert position.stop_price == 10_300.0
