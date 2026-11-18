from dataclasses import dataclass
from enum import Enum
from math import isfinite


class MarketType(str, Enum):
    SPOT = "spot"
    FUTURES = "futures"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """
    Risk policy.

    Fractions are decimal values:
    0.005 means 0.5%.
    """

    risk_per_trade_fraction: float
    max_position_fraction: float
    max_total_exposure_fraction: float
    max_daily_loss_fraction: float
    max_leverage: float

    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    min_notional: float = 0.0
    max_open_positions: int = 3

    def __post_init__(self):
        fractions = (
            self.risk_per_trade_fraction,
            self.max_position_fraction,
            self.max_total_exposure_fraction,
            self.max_daily_loss_fraction,
            self.max_leverage,
            self.fee_rate,
            self.slippage_rate,
            self.min_notional,
        )

        if not all(isfinite(value) for value in fractions):
            raise ValueError("Risk configuration must be finite")

        if not 0 < self.risk_per_trade_fraction <= 1:
            raise ValueError("Invalid risk_per_trade_fraction")

        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("Invalid max_position_fraction")

        if self.max_total_exposure_fraction <= 0:
            raise ValueError("Invalid max_total_exposure_fraction")

        if not 0 < self.max_daily_loss_fraction <= 1:
            raise ValueError("Invalid max_daily_loss_fraction")

        if self.max_leverage < 1:
            raise ValueError("max_leverage must be at least 1")

        if not 0 <= self.fee_rate < 1:
            raise ValueError("Invalid fee_rate")

        if not 0 <= self.slippage_rate < 1:
            raise ValueError("Invalid slippage_rate")

        if self.min_notional < 0:
            raise ValueError("Invalid min_notional")

        if (
            not isinstance(self.max_open_positions, int)
            or isinstance(self.max_open_positions, bool)
            or self.max_open_positions < 1
        ):
            raise ValueError("Invalid max_open_positions")


@dataclass(frozen=True, slots=True)
class RiskRequest:
    account_equity: float
    available_balance: float

    entry_price: float
    stop_price: float

    daily_start_equity: float

    market_type: MarketType | str
    side: OrderSide | str

    daily_realized_pnl: float = 0.0
    current_exposure: float = 0.0
    open_positions: int = 0
    leverage: float = 1.0


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool

    quantity: float = 0.0
    notional: float = 0.0

    risk_budget: float = 0.0
    estimated_max_loss: float = 0.0
    margin_required: float = 0.0

    effective_leverage: float = 1.0

    rejection_reason: str | None = None
    warnings: tuple[str, ...] = ()
    stop_price: float | None = None


class RiskEngine:
    """
    Exchange-independent risk validation and position sizing.

    This module does not submit orders.
    Quantity precision and exchange-specific limits must be
    validated separately before actual order execution.
    """

    def __init__(self, config: RiskConfig):
        self.config = config

    @staticmethod
    def _reject(reason: str, risk_budget: float = 0.0):
        return RiskDecision(
            approved=False,
            risk_budget=risk_budget,
            rejection_reason=reason,
        )

    @staticmethod
    def _valid_request_numbers(request: RiskRequest) -> bool:
        values = (
            request.account_equity,
            request.available_balance,
            request.entry_price,
            request.stop_price,
            request.daily_start_equity,
            request.daily_realized_pnl,
            request.current_exposure,
            request.leverage,
        )

        return all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            for value in values
        )

    def evaluate(self, request: RiskRequest) -> RiskDecision:
        config = self.config

        if not self._valid_request_numbers(request):
            return self._reject("Invalid or non-finite request values")

        if (
            request.account_equity <= 0
            or request.daily_start_equity <= 0
            or request.available_balance < 0
            or request.entry_price <= 0
            or request.stop_price <= 0
            or request.current_exposure < 0
            or request.leverage <= 0
        ):
            return self._reject("Invalid account, balance, or price values")

        if (
            not isinstance(request.open_positions, int)
            or isinstance(request.open_positions, bool)
            or request.open_positions < 0
        ):
            return self._reject("Invalid open_positions")

        try:
            market_type = MarketType(request.market_type)
            side = OrderSide(request.side)
        except ValueError:
            return self._reject("Unsupported market type or order side")

        risk_budget = (
            request.account_equity
            * config.risk_per_trade_fraction
        )

        # Global position count limit.
        if request.open_positions >= config.max_open_positions:
            return self._reject(
                "Maximum open positions reached",
                risk_budget,
            )

        # Daily loss protection.
        daily_loss_limit = (
            request.daily_start_equity
            * config.max_daily_loss_fraction
        )

        if request.daily_realized_pnl <= -daily_loss_limit:
            return self._reject(
                "Daily loss limit reached",
                risk_budget,
            )

        # Spot trading is long-only in this initial implementation.
        if market_type == MarketType.SPOT:
            if side != OrderSide.BUY:
                return self._reject(
                    "Spot short selling is not supported",
                    risk_budget,
                )

            if request.leverage != 1:
                return self._reject(
                    "Spot leverage must be 1",
                    risk_budget,
                )

            effective_leverage = 1.0

        else:
            if request.leverage > config.max_leverage:
                return self._reject(
                    "Requested leverage exceeds maximum",
                    risk_budget,
                )

            effective_leverage = request.leverage

        # Validate stop direction.
        if side == OrderSide.BUY:
            if request.stop_price >= request.entry_price:
                return self._reject(
                    "Long position stop must be below entry",
                    risk_budget,
                )

            entry_fill = (
                request.entry_price
                * (1 + config.slippage_rate)
            )
            stop_fill = (
                request.stop_price
                * (1 - config.slippage_rate)
            )

        else:
            if request.stop_price <= request.entry_price:
                return self._reject(
                    "Short position stop must be above entry",
                    risk_budget,
                )

            entry_fill = (
                request.entry_price
                * (1 - config.slippage_rate)
            )
            stop_fill = (
                request.stop_price
                * (1 + config.slippage_rate)
            )

        # Estimate loss per unit, including entry/exit fees.
        price_loss_per_unit = abs(entry_fill - stop_fill)

        fee_per_unit = (
            entry_fill + stop_fill
        ) * config.fee_rate

        loss_per_unit = price_loss_per_unit + fee_per_unit

        if not isfinite(loss_per_unit) or loss_per_unit <= 0:
            return self._reject(
                "Invalid estimated loss per unit",
                risk_budget,
            )

        # Risk-based quantity.
        risk_quantity = risk_budget / loss_per_unit

        # Maximum single-position notional.
        position_notional_limit = (
            request.account_equity
            * config.max_position_fraction
        )
        position_quantity = (
            position_notional_limit / request.entry_price
        )

        # Maximum remaining portfolio exposure.
        exposure_limit = (
            request.account_equity
            * config.max_total_exposure_fraction
        )

        remaining_exposure = (
            exposure_limit - request.current_exposure
        )

        if remaining_exposure <= 0:
            return self._reject(
                "Maximum total exposure reached",
                risk_budget,
            )

        exposure_quantity = (
            remaining_exposure / request.entry_price
        )

        # Funds required per unit.
        if market_type == MarketType.SPOT:
            funds_per_unit = (
                entry_fill * (1 + config.fee_rate)
            )
        else:
            funds_per_unit = (
                entry_fill / effective_leverage
                + entry_fill * config.fee_rate
            )

        if funds_per_unit <= 0:
            return self._reject(
                "Invalid funds requirement",
                risk_budget,
            )

        if request.available_balance <= 0:
            return self._reject(
                "Insufficient available balance",
                risk_budget,
            )

        funds_quantity = (
            request.available_balance / funds_per_unit
        )

        # Apply all independent quantity constraints.
        limits = {
            "position_limit": position_quantity,
            "exposure_limit": exposure_quantity,
            "available_balance": funds_quantity,
        }

        quantity = min(
            risk_quantity,
            *limits.values(),
        )

        if not isfinite(quantity) or quantity <= 0:
            return self._reject(
                "Calculated quantity is zero",
                risk_budget,
            )

        notional = quantity * request.entry_price

        if notional < config.min_notional:
            return self._reject(
                "Order is below minimum notional",
                risk_budget,
            )

        warnings = tuple(
            name
            for name, limit_quantity in limits.items()
            if limit_quantity < risk_quantity
        )

        estimated_max_loss = quantity * loss_per_unit
        margin_required = quantity * funds_per_unit

        return RiskDecision(
            approved=True,
            quantity=quantity,
            notional=notional,
            risk_budget=risk_budget,
            estimated_max_loss=estimated_max_loss,
            margin_required=margin_required,
            effective_leverage=effective_leverage,
            warnings=warnings,
            stop_price=request.stop_price,
        )
