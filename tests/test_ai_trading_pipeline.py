from aitrader.ai import AIAnalysisEngine, AITradingGate
from aitrader.data.models import Candle
from aitrader.indicators.models import IndicatorSnapshot
from aitrader.risk import (
    AccountRiskState,
    MarketType,
    RiskConfig,
    RiskEngine,
    RiskManager,
)
from aitrader.strategy.models import MarketBias, StrategySignal
from aitrader.strategy.multi_timeframe import MultiTimeframeResult
from aitrader.execution import OrderEngine, OrderRequest, OrderStatus
from aitrader.portfolio import PositionManager, PositionStatus


def create_pipeline(tmp_path, atr=1.0):
    timestamp = 1_700_000_000_000

    candle = Candle(
        exchange="binance_spot",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=timestamp,
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
        volume=10.0,
    )

    indicators = IndicatorSnapshot(
        timestamp_ms=timestamp,
        sma_20=98.0,
        sma_50=95.0,
        ema_12=99.0,
        ema_26=97.0,
        rsi_14=60.0,
        macd=2.0,
        macd_signal=1.0,
        macd_histogram=1.0,
        bollinger_upper=105.0,
        bollinger_middle=100.0,
        bollinger_lower=95.0,
        atr_14=atr,
    )

    signal = StrategySignal(
        exchange="binance_spot",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=timestamp,
        bias=MarketBias.BULLISH,
        score=4,
        max_score=5,
        reasons=("Bullish signal",),
        atr=atr,
    )

    consensus = MultiTimeframeResult(
        exchange="binance_spot",
        symbol="BTC/USDT",
        bias=MarketBias.BULLISH,
        weighted_score=2.5,
        available_timeframes=("1m", "5m", "15m", "1h", "4h"),
        missing_timeframes=(),
        reasons=("Bullish consensus",),
    )

    account = AccountRiskState(
        account_equity=10_000.0,
        available_balance=10_000.0,
        daily_start_equity=10_000.0,
    )

    risk_config = RiskConfig(
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

    risk_manager = RiskManager(
        RiskEngine(risk_config),
        atr_stop_multiplier=3.0,
    )

    ai_engine = AIAnalysisEngine()
    ai_gate = AITradingGate(risk_manager)

    analysis = ai_engine.analyze(candle, indicators)

    order_engine = OrderEngine()
    portfolio = PositionManager(tmp_path / "pipeline.db")

    return {
        "signal": signal,
        "consensus": consensus,
        "account": account,
        "analysis": analysis,
        "gate": ai_gate,
        "orders": order_engine,
        "portfolio": portfolio,
    }


def test_ai_approved_trade_reaches_position_manager(tmp_path):
    pipeline = create_pipeline(tmp_path)

    decision = pipeline["gate"].evaluate(
        analysis=pipeline["analysis"],
        consensus=pipeline["consensus"],
        execution_signal=pipeline["signal"],
        entry_price=100.0,
        account=pipeline["account"],
        market_type=MarketType.SPOT,
    )

    assert decision.approved

    order = pipeline["orders"].submit_paper_order(
        OrderRequest(
            exchange="binance_spot",
            symbol="BTC/USDT",
            side="buy",
            reference_price=100.0,
            client_order_id="ai-pipeline-entry",
        ),
        decision,
    )

    assert order.status == OrderStatus.FILLED

    position = pipeline["portfolio"].open_from_order(
        order,
        MarketType.SPOT,
    )

    assert position.status == PositionStatus.OPEN
    assert position.remaining_quantity > 0
    assert position.stop_price == decision.stop_price


def test_high_volatility_blocks_position_creation(tmp_path):
    pipeline = create_pipeline(tmp_path, atr=2.0)

    decision = pipeline["gate"].evaluate(
        analysis=pipeline["analysis"],
        consensus=pipeline["consensus"],
        execution_signal=pipeline["signal"],
        entry_price=100.0,
        account=pipeline["account"],
        market_type=MarketType.SPOT,
    )

    assert not decision.approved
    assert "High volatility" in decision.rejection_reason

    order = pipeline["orders"].submit_paper_order(
        OrderRequest(
            exchange="binance_spot",
            symbol="BTC/USDT",
            side="buy",
            reference_price=100.0,
            client_order_id="ai-pipeline-blocked",
        ),
        decision,
    )

    assert order.status == OrderStatus.REJECTED
    assert pipeline["portfolio"].get_open_positions() == ()
