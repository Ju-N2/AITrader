import pytest

from aitrader.risk import (
    MarketType,
    OrderSide,
    RiskConfig,
    RiskEngine,
    RiskRequest,
)


def make_config(**overrides):
    values = {
        "risk_per_trade_fraction": 0.01,
        "max_position_fraction": 0.25,
        "max_total_exposure_fraction": 0.50,
        "max_daily_loss_fraction": 0.02,
        "max_leverage": 3.0,
        "fee_rate": 0.001,
        "slippage_rate": 0.0005,
        "min_notional": 10.0,
        "max_open_positions": 3,
    }
    values.update(overrides)
    return RiskConfig(**values)


def make_request(**overrides):
    values = {
        "account_equity": 10_000.0,
        "available_balance": 10_000.0,
        "entry_price": 100.0,
        "stop_price": 95.0,
        "daily_start_equity": 10_000.0,
        "market_type": MarketType.SPOT,
        "side": OrderSide.BUY,
    }
    values.update(overrides)
    return RiskRequest(**values)


def test_approves_valid_spot_long():
    decision = RiskEngine(make_config()).evaluate(make_request())

    assert decision.approved is True
    assert decision.quantity > 0
    assert decision.notional > 0
    assert decision.estimated_max_loss <= decision.risk_budget
    assert decision.effective_leverage == 1.0


def test_spot_short_is_rejected():
    decision = RiskEngine(make_config()).evaluate(
        make_request(side=OrderSide.SELL)
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Spot short selling is not supported"


def test_valid_futures_short_is_approved():
    decision = RiskEngine(make_config()).evaluate(
        make_request(
            market_type=MarketType.FUTURES,
            side=OrderSide.SELL,
            entry_price=100.0,
            stop_price=105.0,
            leverage=2.0,
        )
    )

    assert decision.approved is True
    assert decision.effective_leverage == 2.0
    assert decision.quantity > 0


def test_daily_loss_limit_blocks_new_trade():
    decision = RiskEngine(make_config()).evaluate(
        make_request(daily_realized_pnl=-200.0)
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Daily loss limit reached"


def test_maximum_leverage_is_enforced():
    decision = RiskEngine(make_config(max_leverage=2.0)).evaluate(
        make_request(
            market_type=MarketType.FUTURES,
            leverage=3.0,
        )
    )

    assert decision.approved is False
    assert decision.rejection_reason == (
        "Requested leverage exceeds maximum"
    )


def test_invalid_long_stop_is_rejected():
    decision = RiskEngine(make_config()).evaluate(
        make_request(stop_price=101.0)
    )

    assert decision.approved is False
    assert "Long position stop" in decision.rejection_reason


def test_maximum_open_positions_blocks_trade():
    decision = RiskEngine(make_config()).evaluate(
        make_request(open_positions=3)
    )

    assert decision.approved is False
    assert decision.rejection_reason == (
        "Maximum open positions reached"
    )


def test_maximum_total_exposure_blocks_trade():
    decision = RiskEngine(make_config()).evaluate(
        make_request(current_exposure=5_000.0)
    )

    assert decision.approved is False
    assert decision.rejection_reason == (
        "Maximum total exposure reached"
    )


def test_available_balance_limits_quantity():
    decision = RiskEngine(make_config()).evaluate(
        make_request(available_balance=100.0)
    )

    assert decision.approved is True
    assert "available_balance" in decision.warnings
    assert decision.margin_required <= 100.0


def test_minimum_notional_is_enforced():
    decision = RiskEngine(
        make_config(min_notional=1_000.0)
    ).evaluate(
        make_request(available_balance=100.0)
    )

    assert decision.approved is False
    assert decision.rejection_reason == (
        "Order is below minimum notional"
    )


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        make_config(risk_per_trade_fraction=0)

    with pytest.raises(ValueError):
        make_config(max_leverage=0)


def test_non_finite_request_is_rejected():
    decision = RiskEngine(make_config()).evaluate(
        make_request(account_equity=float("nan"))
    )

    assert decision.approved is False
    assert "non-finite" in decision.rejection_reason
