import logging
import sqlite3
import time
from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

from aitrader.ai import AIAnalysisEngine, AITradingGate
from aitrader.core.analysis_store import AIAnalysisStore
from aitrader.core.paper_trading import PaperTradingService
from aitrader.data.service import MarketDataService
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.execution import OrderEngine, PaperExecutionConfig
from aitrader.execution.models import OrderStatus
from aitrader.indicators.engine import IndicatorEngine
from aitrader.notifications import TelegramNotifier
from aitrader.portfolio import PositionManager
from aitrader.portfolio.models import PositionStatus
from aitrader.risk import (
    AccountRiskState,
    MarketType,
    RiskConfig,
    RiskEngine,
    RiskManager,
)
from aitrader.strategy.models import MarketBias
from aitrader.strategy.multi_timeframe import MultiTimeframeEngine
from aitrader.strategy.technical import TechnicalTrendStrategy

TIMEFRAMES = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    exchange: str = "binance_spot"
    symbol: str = "BTC/USDT"

    initial_capital: float = 10_000.0
    leverage: float = 1.0

    poll_interval_seconds: float = 5.0
    warmup_limit: int = 200

    execution_timeframe: str = "1h"

    def __post_init__(self):
        if self.exchange not in (
            "upbit",
            "binance_spot",
            "binance_futures",
        ):
            raise ValueError("Unsupported exchange")

        if not self.symbol:
            raise ValueError("Symbol is required")

        if (
            not isfinite(self.initial_capital)
            or self.initial_capital <= 0
        ):
            raise ValueError("Invalid initial capital")

        if (
            not isfinite(self.leverage)
            or self.leverage < 1
        ):
            raise ValueError("Invalid leverage")

        if self.exchange != "binance_futures" and self.leverage != 1:
            raise ValueError("Spot leverage must be 1")

        if (
            not isfinite(self.poll_interval_seconds)
            or self.poll_interval_seconds < 1
        ):
            raise ValueError("Invalid polling interval")

        if not 60 <= self.warmup_limit <= 1000:
            raise ValueError("Invalid warmup limit")

        if self.execution_timeframe not in TIMEFRAMES:
            raise ValueError("Unsupported execution timeframe")

    @property
    def market_type(self):
        if self.exchange == "binance_futures":
            return MarketType.FUTURES

        return MarketType.SPOT


class RuntimeStateStore:
    """Persistent runtime checkpoints stored in SQLite."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path, timeout=10) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS runtime_checkpoints (
                    runtime_key TEXT PRIMARY KEY,
                    last_processed_close_ms INTEGER NOT NULL
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS runtime_daily_equity (
                    runtime_key TEXT NOT NULL,
                    utc_date TEXT NOT NULL,
                    start_equity REAL NOT NULL
                        CHECK(start_equity >= 0),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (runtime_key, utc_date)
                )
            """)

    def get_last_processed(self, runtime_key):
        with sqlite3.connect(self.db_path, timeout=10) as connection:
            row = connection.execute("""
                SELECT last_processed_close_ms
                FROM runtime_checkpoints
                WHERE runtime_key = ?
            """, (runtime_key,)).fetchone()

        return row[0] if row else 0

    def save_processed(self, runtime_key, timestamp_ms):
        with sqlite3.connect(self.db_path, timeout=10) as connection:
            connection.execute("""
                INSERT INTO runtime_checkpoints (
                    runtime_key,
                    last_processed_close_ms
                )
                VALUES (?, ?)
                ON CONFLICT(runtime_key)
                DO UPDATE SET
                    last_processed_close_ms =
                    MAX(
                        runtime_checkpoints.last_processed_close_ms,
                        excluded.last_processed_close_ms
                    )
            """, (runtime_key, timestamp_ms))


    def get_or_create_daily_start_equity(
        self,
        runtime_key,
        utc_date,
        start_equity,
    ):
        """Return the persisted starting equity for a UTC trading day."""
        from math import isfinite

        if not isinstance(runtime_key, str) or not runtime_key:
            raise ValueError("Invalid runtime_key")

        if not isinstance(utc_date, str):
            raise ValueError("Invalid utc_date")

        try:
            parsed_date = datetime.fromisoformat(utc_date)
        except ValueError as exc:
            raise ValueError("Invalid utc_date") from exc

        if parsed_date.date().isoformat() != utc_date:
            raise ValueError("utc_date must use YYYY-MM-DD format")

        if (
            isinstance(start_equity, bool)
            or not isinstance(start_equity, (int, float))
            or not isfinite(start_equity)
            or start_equity < 0
        ):
            raise ValueError("Invalid start_equity")

        with sqlite3.connect(self.db_path, timeout=10) as connection:
            connection.execute("""
                INSERT OR IGNORE INTO runtime_daily_equity (
                    runtime_key,
                    utc_date,
                    start_equity,
                    created_at
                )
                VALUES (?, ?, ?, ?)
            """, (
                runtime_key,
                utc_date,
                float(start_equity),
                datetime.now(UTC).isoformat(),
            ))

            row = connection.execute("""
                SELECT start_equity
                FROM runtime_daily_equity
                WHERE runtime_key = ?
                  AND utc_date = ?
            """, (
                runtime_key,
                utc_date,
            )).fetchone()

        if row is None:
            raise RuntimeError("Failed to load daily starting equity")

        return float(row[0])


class PaperTradingRuntime:

    def __init__(
        self,
        config: RuntimeConfig,
        client,
        repository: SQLiteCandleRepository,
        notifier: TelegramNotifier | None = None,
    ):
        self.config = config
        self.client = client
        self.repository = repository

        self.logger = logging.getLogger("aitrader.runtime")
        self.notifier = notifier

        self.data_service = MarketDataService(
            client=client,
            exchange_name=config.exchange,
            repository=repository,
        )

        self.indicator_engine = IndicatorEngine()
        self.strategy = TechnicalTrendStrategy()

        self.multi_engine = MultiTimeframeEngine(
            min_timeframes=5,
        )

        self.ai_engine = AIAnalysisEngine()
        self.analysis_store = AIAnalysisStore(
            Path(repository.db_path).parent / "latest_ai_analysis.json"
        )

        self.risk_config = RiskConfig(
            risk_per_trade_fraction=0.01,
            max_position_fraction=0.25,
            max_total_exposure_fraction=0.50,
            max_daily_loss_fraction=0.02,
            max_leverage=3.0,
            fee_rate=0.001,
            slippage_rate=0.0005,
            min_notional=10.0,
            max_open_positions=1,
        )

        self.risk_manager = RiskManager(
            RiskEngine(self.risk_config),
            atr_stop_multiplier=3.0,
        )

        self.ai_gate = AITradingGate(
            risk_manager=self.risk_manager,
            min_confidence=0.70,
            min_agreement=0.60,
        )

        self.order_engine = OrderEngine(
            PaperExecutionConfig(
                fee_rate=0.001,
                slippage_rate=0.0005,
            )
        )

        self.position_manager = PositionManager(
            repository.db_path,
        )

        self.trading_service = PaperTradingService(
            ai_gate=self.ai_gate,
            order_engine=self.order_engine,
            position_manager=self.position_manager,
            notifier=notifier,
        )

        self.state_store = RuntimeStateStore(
            repository.db_path,
        )

        self.runtime_key = (
            f"{config.exchange}:"
            f"{config.symbol}:"
            f"{config.execution_timeframe}"
        )

        self.last_fetch_bucket = {}

        # DRY RUN deduplication is memory-only.
        # It must not consume persistent PAPER checkpoints.
        self.last_dry_run_decision_time = 0

    @staticmethod
    def indicator_ready(snapshot):
        return all(
            value is not None
            for value in (
                snapshot.sma_20,
                snapshot.sma_50,
                snapshot.ema_12,
                snapshot.ema_26,
                snapshot.rsi_14,
                snapshot.macd_histogram,
                snapshot.atr_14,
            )
        )

    def _get_recent_candles(self, timeframe):
        interval = TIMEFRAMES[timeframe]

        candles = self.repository.get_candles(
            self.config.exchange,
            self.config.symbol,
            timeframe,
            limit=self.config.warmup_limit,
        )

        now_ms = int(time.time() * 1000)

        return [
            candle
            for candle in candles
            if candle.timestamp_ms + interval <= now_ms
        ]

    def bootstrap(self):
        """Fetch initial historical data and check indicator warmup."""

        self.logger.info(
            "Starting runtime bootstrap | exchange=%s symbol=%s",
            self.config.exchange,
            self.config.symbol,
        )

        now_ms = int(time.time() * 1000)

        for timeframe, interval in TIMEFRAMES.items():

            saved = self.data_service.collect_and_store(
                symbol=self.config.symbol,
                timeframe=timeframe,
                limit=self.config.warmup_limit,
            )

            self.last_fetch_bucket[timeframe] = now_ms // interval

            candles = self._get_recent_candles(timeframe)

            if len(candles) < 50:
                raise RuntimeError(
                    f"Insufficient completed candles for {timeframe}: "
                    f"{len(candles)}"
                )

            self.logger.info(
                "Bootstrap completed | timeframe=%s fetched=%d "
                "completed=%d",
                timeframe,
                saved,
                len(candles),
            )

        self.logger.info("Runtime bootstrap successful")

    def _refresh_due_data(self, now_ms):
        """Fetch only after a new timeframe bucket starts."""

        for timeframe, interval in TIMEFRAMES.items():

            current_bucket = now_ms // interval
            previous_bucket = self.last_fetch_bucket.get(timeframe)

            if previous_bucket == current_bucket:
                continue

            latest = self.repository.get_candles(
                self.config.exchange,
                self.config.symbol,
                timeframe,
                limit=1,
            )

            since = (
                latest[-1].timestamp_ms + interval
                if latest
                else None
            )

            saved = self.data_service.collect_and_store(
                symbol=self.config.symbol,
                timeframe=timeframe,
                limit=1000,
                since=since,
            )

            self.last_fetch_bucket[timeframe] = current_bucket

            self.logger.info(
                "Market data refreshed | timeframe=%s saved=%d",
                timeframe,
                saved,
            )

    @staticmethod
    def _validate_candle_continuity(candles, timeframe):
        """Reject missing or irregular candles before signal generation."""

        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        expected_interval = TIMEFRAMES[timeframe]

        for previous, current in zip(candles, candles[1:]):
            actual_interval = (
                current.timestamp_ms - previous.timestamp_ms
            )

            if actual_interval != expected_interval:
                raise RuntimeError(
                    f"Candle continuity error | timeframe={timeframe} "
                    f"previous={previous.timestamp_ms} "
                    f"current={current.timestamp_ms} "
                    f"expected_interval={expected_interval} "
                    f"actual_interval={actual_interval}"
                )

        return True

    def _build_signal_series(self, timeframe, now_ms):
        interval = TIMEFRAMES[timeframe]

        candles = self.repository.get_candles(
            self.config.exchange,
            self.config.symbol,
            timeframe,
            limit=self.config.warmup_limit,
        )

        candles = [
            candle
            for candle in candles
            if candle.timestamp_ms + interval <= now_ms
        ]

        if len(candles) < 50:
            raise RuntimeError(
                f"Insufficient completed history for {timeframe}"
            )

        self._validate_candle_continuity(candles, timeframe)

        snapshots = self.indicator_engine.calculate(candles)

        result = []

        for candle, snapshot in zip(candles, snapshots):

            if not self.indicator_ready(snapshot):
                continue

            signal = self.strategy.evaluate(
                candle,
                snapshot,
            )

            close_time = candle.timestamp_ms + interval

            result.append((
                close_time,
                candle,
                snapshot,
                signal,
            ))

        if not result:
            raise RuntimeError(
                f"No indicator-ready signals for {timeframe}"
            )

        return result

    def _build_aligned_signals(self, series, decision_time):
        aligned = {}

        for timeframe, values in series.items():

            close_times = [
                item[0]
                for item in values
            ]

            index = bisect_right(
                close_times,
                decision_time,
            ) - 1

            if index < 0:
                return None

            item = values[index]

            signal_age = decision_time - item[0]

            if signal_age > TIMEFRAMES[timeframe] * 3:
                return None

            aligned[timeframe] = item[3]

        return aligned

    def _fetch_price(self):
        ticker = self.client.fetch_ticker(self.config.symbol)

        for key in ("last", "mark", "close"):
            value = ticker.get(key)

            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isfinite(value)
                and value > 0
            ):
                return float(value)

        raise RuntimeError("No valid market price available")

    def _monitor_stops(self, now_ms):
        positions = self.position_manager.get_open_positions(
            exchange=self.config.exchange,
            symbol=self.config.symbol,
        )

        if not positions:
            return False

        mark_price = self._fetch_price()
        stopped = False

        for position in positions:

            if not position.is_stop_triggered(mark_price):
                continue

            self.logger.warning(
                "Position stop triggered | id=%s price=%.4f",
                position.position_id,
                mark_price,
            )

            result = self.trading_service.close_position(
                position_id=position.position_id,
                mark_price=mark_price,
                client_order_id=(
                    f"stop-{position.position_id}-{now_ms // 60000}"
                ),
            )

            if (
                result.order.status == OrderStatus.FILLED
                and result.position.status == PositionStatus.CLOSED
            ):
                stopped = True
            else:
                self.logger.error(
                    "Position stop exit not completed | "
                    "id=%s order_status=%s reason=%s",
                    position.position_id,
                    result.order.status.value,
                    result.order.rejection_reason,
                )

        return stopped

    def _build_account_state(self, mark_price=None):
        positions = self.position_manager.list_positions(
            exchange=self.config.exchange,
            symbol=self.config.symbol,
        )

        open_positions = [
            position
            for position in positions
            if position.status == PositionStatus.OPEN
        ]

        realized_total = sum(
            position.realized_pnl
            for position in positions
        )

        unrealized = 0.0

        if mark_price is not None:
            unrealized = sum(
                position.unrealized_pnl(mark_price)
                for position in open_positions
            )

        exposure = sum(
            position.entry_price * position.remaining_quantity
            for position in open_positions
        )

        equity = max(
            0.0,
            self.config.initial_capital
            + realized_total
            + unrealized,
        )

        if self.config.market_type == MarketType.SPOT:
            locked_funds = exposure
        else:
            locked_funds = exposure / self.config.leverage

        available_balance = max(
            0.0,
            equity - locked_funds,
        )

        # Daily risk accounting uses UTC calendar days.
        today = datetime.now(UTC).date()

        day_start_utc = datetime(
            today.year,
            today.month,
            today.day,
            tzinfo=UTC,
        )

        # Sum individual exit events instead of cumulative position PnL.
        daily_realized = self.position_manager.get_realized_pnl_since(
            since=day_start_utc,
            exchange=self.config.exchange,
            symbol=self.config.symbol,
        )

        # Use realized equity before today's exits as the daily baseline.
        daily_start_candidate = max(
            0.0,
            self.config.initial_capital
            + realized_total
            - daily_realized,
        )

        # Persist the baseline so restarting Runtime does not reset it.
        daily_start_equity = (
            self.state_store.get_or_create_daily_start_equity(
                runtime_key=self.runtime_key,
                utc_date=today.isoformat(),
                start_equity=daily_start_candidate,
            )
        )

        return AccountRiskState(
            account_equity=equity,
            available_balance=available_balance,
            daily_start_equity=daily_start_equity,
            daily_realized_pnl=daily_realized,
            current_exposure=exposure,
            open_positions=len(open_positions),
            leverage=self.config.leverage,
        )

    def run_cycle(self, paper_mode=False):
        now_ms = int(time.time() * 1000)

        # 1. Monitor existing stops before refreshing candle history.
        if paper_mode:
            if self._monitor_stops(now_ms):
                latest = self.repository.get_candles(
                    self.config.exchange,
                    self.config.symbol,
                    self.config.execution_timeframe,
                    limit=1,
                )

                if latest:
                    close_time = (
                        latest[-1].timestamp_ms
                        + TIMEFRAMES[self.config.execution_timeframe]
                    )
                    self.state_store.save_processed(
                        self.runtime_key,
                        close_time,
                    )

                return "stop_exit"

        # 2. Refresh market data after protective stop checks.
        self._refresh_due_data(now_ms)

        # 3. Build completed signal history.
        series = {
            timeframe: self._build_signal_series(
                timeframe,
                now_ms,
            )
            for timeframe in TIMEFRAMES
        }

        execution_timeframe = self.config.execution_timeframe

        execution_item = series[execution_timeframe][-1]

        decision_time, candle, snapshot, execution_signal = execution_item

        # 4. Do not process old signals after a long outage.
        if now_ms - decision_time > TIMEFRAMES[execution_timeframe] * 2:
            self.logger.warning(
                "Skipping stale decision candle | close_time=%d",
                decision_time,
            )

            if paper_mode:
                self.state_store.save_processed(
                    self.runtime_key,
                    decision_time,
                )

            return "stale"

        # 5. Duplicate decision protection.
        if paper_mode:
            last_processed = self.state_store.get_last_processed(
                self.runtime_key
            )

            if decision_time <= last_processed:
                return "no_new_candle"

        elif decision_time <= self.last_dry_run_decision_time:
            return "no_new_candle"

        # 6. Align all five timeframes at the decision timestamp.
        aligned_signals = self._build_aligned_signals(
            series,
            decision_time,
        )

        if aligned_signals is None:
            self.logger.warning("Incomplete timeframe alignment")
            return "incomplete_alignment"

        consensus = self.multi_engine.evaluate(
            aligned_signals
        )

        # 7. Analyze the completed execution candle.
        analysis = self.ai_engine.analyze(
            candle,
            snapshot,
        )

        # Persist the latest AI analysis without affecting trading.
        try:
            self.analysis_store.save(analysis)
        except Exception:
            self.logger.exception(
                "Failed to persist latest AI analysis"
            )


        # Send AI analysis to Telegram, including neutral results.
        notifier = getattr(self, "notifier", None)

        if notifier is not None:
            send_analysis = getattr(
                notifier,
                "send_ai_analysis",
                None,
            )

            if callable(send_analysis):
                try:
                    send_analysis(
                        analysis,
                        consensus,
                        execution_signal,
                    )
                    self.logger.info(
                        "AI analysis Telegram notification sent"
                    )
                except Exception:
                    self.logger.exception(
                        "Failed to send AI analysis Telegram notification"
                    )

        # 8. Obtain current price.
        current_price = self._fetch_price()

        # 9. Build current account state.
        account = self._build_account_state(
            mark_price=current_price,
        )

        # 10. Dry-run mode never modifies orders or positions.
        if not paper_mode:
            decision = self.ai_gate.evaluate(
                analysis=analysis,
                consensus=consensus,
                execution_signal=execution_signal,
                entry_price=current_price,
                account=account,
                market_type=self.config.market_type,
            )

            self.logger.info(
                "DRY RUN | time=%d price=%.4f "
                "MTF=%s(score=%.2f) "
                "EXEC=%s(score=%d/%d) "
                "AI=%s(score=%+.3f conf=%.2f agreement=%.2f "
                "regime=%s volatility=%s) "
                "agents=[%s] decision=%s reason=%s",
                decision_time,
                current_price,
                consensus.bias.value,
                consensus.weighted_score,
                execution_signal.bias.value,
                execution_signal.score,
                execution_signal.max_score,
                analysis.bias.value,
                analysis.score,
                analysis.confidence,
                analysis.agreement,
                analysis.regime.value,
                analysis.volatility.value,
                ", ".join(
                    f"{report.agent_name}:{report.bias.value}"
                    f"({report.score:+.2f})"
                    for report in analysis.reports
                ),
                decision.approved,
                decision.rejection_reason,
            )

            self.last_dry_run_decision_time = decision_time
            return "dry_run"

        # 11. One open position per market in this initial runtime.
        if account.open_positions > 0:
            self.logger.info("Existing position found; entry skipped")

            self.state_store.save_processed(
                self.runtime_key,
                decision_time,
            )

            return "position_open"

        # 12. Avoid unnecessary rejected orders in a neutral market.
        if (
            consensus.bias == MarketBias.NEUTRAL
            or execution_signal.bias == MarketBias.NEUTRAL
        ):
            self.logger.info(
                "No directional setup | MTF=%s execution=%s",
                consensus.bias.value,
                execution_signal.bias.value,
            )

            self.state_store.save_processed(
                self.runtime_key,
                decision_time,
            )

            return "neutral"

        # 13. AI gate -> Risk -> Paper order -> Position.
        result = self.trading_service.enter(
            analysis=analysis,
            consensus=consensus,
            execution_signal=execution_signal,
            entry_price=current_price,
            account=account,
            market_type=self.config.market_type,
            client_order_id=(
                f"entry-{self.config.exchange}-"
                f"{self.config.symbol.replace('/', '_')}-"
                f"{decision_time}"
            ),
        )

        self.state_store.save_processed(
            self.runtime_key,
            decision_time,
        )

        self.logger.info(
            "Paper trading cycle completed | status=%s "
            "approved=%s order=%s position=%s",
            "filled" if result.position else "rejected_or_skipped",
            result.decision.approved,
            result.order.order_id if result.order else None,
            result.position.position_id if result.position else None,
        )

        return "paper_processed"

    def run_forever(self, paper_mode=False):
        mode = "PAPER TRADING" if paper_mode else "ANALYSIS ONLY"
        consecutive_failures = 0

        self.logger.info(
            "Runtime started | mode=%s exchange=%s symbol=%s",
            mode,
            self.config.exchange,
            self.config.symbol,
        )

        while True:
            try:
                result = self.run_cycle(
                    paper_mode=paper_mode,
                )

                self.logger.info("Runtime cycle: %s", result)
                consecutive_failures = 0

            except KeyboardInterrupt:
                self.logger.info("Runtime stopped by user")
                break

            except Exception as exc:
                consecutive_failures += 1
                self.logger.exception("Runtime cycle failed")

                if self.trading_service.notifier is not None:
                    self.trading_service.notify_system_error(
                        "PaperTradingRuntime",
                        type(exc).__name__,
                    )

                delay = min(
                    self.config.poll_interval_seconds
                    * (2 ** min(consecutive_failures, 10)),
                    300.0,
                )
            else:
                delay = self.config.poll_interval_seconds

            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                self.logger.info("Runtime stopped by user")
                break
