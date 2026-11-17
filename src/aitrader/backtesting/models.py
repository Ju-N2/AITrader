from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    position_fraction: float = 0.25
    atr_stop_multiplier: float | None = None
    atr_cooldown_bars: int = 0

    def __post_init__(self):
        values = [
            self.initial_capital,
            self.fee_rate,
            self.slippage_rate,
            self.position_fraction,
        ]

        if not all(isfinite(value) for value in values):
            raise ValueError("Configuration values must be finite")

        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")

        if not 0 <= self.fee_rate < 1:
            raise ValueError("Invalid fee rate")

        if not 0 <= self.slippage_rate < 1:
            raise ValueError("Invalid slippage rate")

        if not 0 < self.position_fraction <= 1:
            raise ValueError("Invalid position fraction")

        if self.atr_stop_multiplier is not None:
            if (
                not isfinite(self.atr_stop_multiplier)
                or self.atr_stop_multiplier <= 0
            ):
                raise ValueError("Invalid ATR stop multiplier")

        if (
            not isinstance(self.atr_cooldown_bars, int)
            or isinstance(self.atr_cooldown_bars, bool)
            or self.atr_cooldown_bars < 0
        ):
            raise ValueError("Invalid ATR cooldown bars")


@dataclass(frozen=True)
class CompletedTrade:
    entry_timestamp_ms: int
    exit_timestamp_ms: int
    exit_reason: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_fee: float
    exit_fee: float
    pnl: float
    return_pct: float


@dataclass(frozen=True)
class BacktestResult:
    initial_capital: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    profit_factor: float | None
    total_fees: float
    total_slippage_cost: float
    total_turnover: float
    stop_loss_count: int
    cooldown_blocked_entries: int
    trades: tuple[CompletedTrade, ...]
    equity_curve: tuple[float, ...]
