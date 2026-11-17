from collections.abc import Sequence
from math import isfinite

from aitrader.core.exceptions import StrategyError
from aitrader.strategy.models import StrategySignal, MarketBias

from aitrader.backtesting.models import (
    BacktestConfig,
    BacktestResult,
    CompletedTrade,
)


class BacktestEngine:

    def run(
        self,
        candles: Sequence,
        signals: Sequence[StrategySignal],
        config: BacktestConfig | None = None,
    ) -> BacktestResult:

        config = config or BacktestConfig()

        if len(candles) < 2:
            raise StrategyError("At least two candles are required")

        if len(candles) != len(signals):
            raise StrategyError("Candle and signal counts must match")

        for i, (candle, signal) in enumerate(zip(candles, signals)):

            if candle.timestamp_ms != signal.timestamp_ms:
                raise StrategyError("Timestamp mismatch")

            if (
                candle.exchange != signal.exchange
                or candle.symbol != signal.symbol
                or candle.timeframe != signal.timeframe
            ):
                raise StrategyError("Metadata mismatch")

            if i > 0 and candle.timestamp_ms <= candles[i - 1].timestamp_ms:
                raise StrategyError("Candles must be chronological")

        cash = config.initial_capital
        position = None

        trades = []
        equity_curve = [cash]

        total_fees = 0.0
        total_slippage_cost = 0.0
        total_turnover = 0.0
        stop_loss_count = 0
        atr_cooldown_until_index = -1
        cooldown_blocked_entries = 0

        def close_position(base_exit_price, timestamp_ms, reason):
            nonlocal cash, position
            nonlocal total_fees, total_slippage_cost
            nonlocal total_turnover, stop_loss_count

            execution_price = (
                base_exit_price * (1 - config.slippage_rate)
            )

            quantity = position["quantity"]

            gross = quantity * execution_price
            exit_fee = gross * config.fee_rate
            proceeds = gross - exit_fee

            cash += proceeds

            original_cost = (
                position["notional"] + position["entry_fee"]
            )

            pnl = proceeds - original_cost

            trades.append(
                CompletedTrade(
                    entry_timestamp_ms=position["timestamp"],
                    exit_timestamp_ms=timestamp_ms,
                    exit_reason=reason,
                    entry_price=position["entry_price"],
                    exit_price=execution_price,
                    quantity=quantity,
                    entry_fee=position["entry_fee"],
                    exit_fee=exit_fee,
                    pnl=pnl,
                    return_pct=pnl / original_cost * 100,
                )
            )

            total_fees += exit_fee

            total_slippage_cost += (
                quantity * (base_exit_price - execution_price)
            )

            total_turnover += gross

            if reason == "atr_stop":
                stop_loss_count += 1

            position = None

        for i in range(len(candles) - 1):

            signal = signals[i]
            next_candle = candles[i + 1]

            exited_this_step = False

            # 1. 기존 포지션 청산 검사
            if position is not None:

                exit_price = None
                exit_reason = None

                # 전략의 하락 신호
                if signal.bias == MarketBias.BEARISH:

                    exit_price = next_candle.open
                    exit_reason = "strategy"

                # ATR 손절 검사
                elif position["stop_price"] is not None:

                    stop_price = position["stop_price"]

                    # 시가가 손절가 아래에서 시작한 경우
                    if next_candle.open <= stop_price:

                        exit_price = next_candle.open
                        exit_reason = "atr_stop"

                    # 장중 저가가 손절가에 도달한 경우
                    elif next_candle.low <= stop_price:

                        exit_price = stop_price
                        exit_reason = "atr_stop"

                if exit_price is not None:

                    close_position(
                        exit_price,
                        next_candle.timestamp_ms,
                        exit_reason,
                    )

                    if exit_reason == "atr_stop":
                        atr_cooldown_until_index = (
                            i + 1 + config.atr_cooldown_bars
                        )

                    exited_this_step = True

            # 2. 신규 매수
            if (
                position is None
                and not exited_this_step
                and signal.bias == MarketBias.BULLISH
                and i + 1 <= atr_cooldown_until_index
            ):
                cooldown_blocked_entries += 1

            if (
                position is None
                and not exited_this_step
                and signal.bias == MarketBias.BULLISH
                and i + 1 > atr_cooldown_until_index
            ):

                atr_value = signal.atr

                if config.atr_stop_multiplier is None:
                    can_enter = True
                else:
                    can_enter = (
                        atr_value is not None
                        and isfinite(atr_value)
                        and atr_value > 0
                    )

                if can_enter:

                    budget = cash * config.position_fraction

                    notional = budget / (1 + config.fee_rate)

                    entry_price = (
                        next_candle.open * (1 + config.slippage_rate)
                    )

                    quantity = notional / entry_price
                    entry_fee = notional * config.fee_rate

                    stop_price = None

                    if config.atr_stop_multiplier is not None:

                        stop_price = (
                            entry_price
                            - atr_value * config.atr_stop_multiplier
                        )

                    if stop_price is None or stop_price > 0:

                        cash -= notional + entry_fee

                        total_fees += entry_fee

                        total_slippage_cost += (
                            quantity * (
                                entry_price - next_candle.open
                            )
                        )

                        total_turnover += notional

                        position = {
                            "timestamp": next_candle.timestamp_ms,
                            "entry_price": entry_price,
                            "quantity": quantity,
                            "notional": notional,
                            "entry_fee": entry_fee,
                            "stop_price": stop_price,
                        }

                        # 진입한 캔들 안에서 손절 확인
                        if (
                            stop_price is not None
                            and next_candle.low <= stop_price
                        ):

                            if next_candle.open <= stop_price:
                                exit_price = next_candle.open
                            else:
                                exit_price = stop_price

                            close_position(
                                exit_price,
                                next_candle.timestamp_ms,
                                "atr_stop",
                            )

                            atr_cooldown_until_index = (
                                i + 1 + config.atr_cooldown_bars
                            )

                            exited_this_step = True

            # 3. 자산 평가
            equity = cash

            if position is not None:

                equity += (
                    position["quantity"] * next_candle.close
                )

            equity_curve.append(equity)

        # 4. 마지막 캔들에서 포지션 청산
        if position is not None:

            close_position(
                candles[-1].close,
                candles[-1].timestamp_ms,
                "end_of_test",
            )

            equity_curve.append(cash)

        final_equity = cash

        total_return = (
            final_equity / config.initial_capital - 1
        ) * 100

        peak = equity_curve[0]
        max_drawdown = 0.0

        for equity in equity_curve:

            peak = max(peak, equity)

            if peak > 0:

                drawdown = (
                    peak - equity
                ) / peak * 100

                max_drawdown = max(
                    max_drawdown,
                    drawdown,
                )

        winning_trades = [
            trade for trade in trades if trade.pnl > 0
        ]

        losing_trades = [
            trade for trade in trades if trade.pnl < 0
        ]

        win_rate = (
            len(winning_trades) / len(trades) * 100
            if trades else 0.0
        )

        gross_profit = sum(
            trade.pnl for trade in winning_trades
        )

        gross_loss = abs(sum(
            trade.pnl for trade in losing_trades
        ))

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0 else None
        )

        return BacktestResult(
            initial_capital=config.initial_capital,
            final_equity=final_equity,
            total_return_pct=total_return,
            max_drawdown_pct=max_drawdown,
            win_rate_pct=win_rate,
            total_trades=len(trades),
            profit_factor=profit_factor,
            total_fees=total_fees,
            total_slippage_cost=total_slippage_cost,
            total_turnover=total_turnover,
            stop_loss_count=stop_loss_count,
            cooldown_blocked_entries=cooldown_blocked_entries,
            trades=tuple(trades),
            equity_curve=tuple(equity_curve),
        )
