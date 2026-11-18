import json
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from aitrader.execution.models import OrderRecord, OrderStatus
from aitrader.notifications import (
    TelegramNotifier,
    TelegramNotificationError,
)
from aitrader.portfolio.models import (
    Position,
    PositionSide,
    PositionStatus,
)


class FakeResponse:
    def __init__(self, data):
        self.data = json.dumps(data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.data


@pytest.fixture
def notifier():
    return TelegramNotifier(
        bot_token="test-secret-token",
        chat_id="123456789",
    )


def success_response():
    return {
        "ok": True,
        "result": {
            "message_id": 100,
            "date": 1_700_000_000,
            "chat": {"id": 123456789},
            "text": "AITrader test",
        },
    }


def make_order():
    return OrderRecord(
        order_id="order-001",
        client_order_id="client-001",
        exchange="binance_spot",
        symbol="BTC/USDT",
        side="buy",
        status=OrderStatus.FILLED,
        requested_quantity=0.1,
        filled_quantity=0.1,
        reference_price=100_000.0,
        average_fill_price=100_050.0,
        filled_notional=10_005.0,
        fee=10.005,
        stop_price=97_000.0,
        execution_mode="paper",
        created_at=datetime.now(UTC),
    )


def make_position(status=PositionStatus.OPEN):
    now = datetime.now(UTC)

    return Position(
        position_id="position-001",
        exchange="binance_spot",
        symbol="BTC/USDT",
        market_type="spot",
        side=PositionSide.LONG,
        status=status,
        entry_order_id="entry-001",
        entry_quantity=0.1,
        remaining_quantity=0.1 if status == PositionStatus.OPEN else 0.0,
        entry_price=100_000.0,
        stop_price=97_000.0,
        leverage=1.0,
        entry_fee_total=10.0,
        remaining_entry_fee=10.0 if status == PositionStatus.OPEN else 0.0,
        exit_fee_total=0.0,
        realized_pnl=150.0,
        opened_at=now,
        updated_at=now,
        closed_at=now if status == PositionStatus.CLOSED else None,
    )


def test_send_message_success(monkeypatch, notifier):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data)
        return FakeResponse(success_response())

    monkeypatch.setattr(
        "aitrader.notifications.telegram.request.urlopen",
        fake_urlopen,
    )

    result = notifier.send_message("AITrader online")

    assert result.message_id == 100
    assert result.chat_id == "123456789"
    assert result.text == "AITrader online"
    assert captured["payload"]["text"] == "AITrader online"
    assert captured["payload"]["chat_id"] == "123456789"
    assert captured["timeout"] == 10.0
    assert "test-secret-token" in captured["url"]


def test_missing_credentials_rejected():
    notifier = TelegramNotifier("", "")

    assert not notifier.is_configured

    with pytest.raises(TelegramNotificationError):
        notifier.send_message("Test")


def test_empty_message_rejected(notifier):
    with pytest.raises(ValueError):
        notifier.send_message("   ")


def test_message_truncated_to_telegram_limit(monkeypatch, notifier):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data)
        return FakeResponse(success_response())

    monkeypatch.setattr(
        "aitrader.notifications.telegram.request.urlopen",
        fake_urlopen,
    )

    result = notifier.send_message("A" * 5000)

    assert len(result.text) == 4096
    assert len(captured["payload"]["text"]) == 4096


def test_api_rejection(monkeypatch, notifier):
    def fake_urlopen(req, timeout):
        return FakeResponse({
            "ok": False,
            "description": "Bad Request: chat not found",
        })

    monkeypatch.setattr(
        "aitrader.notifications.telegram.request.urlopen",
        fake_urlopen,
    )

    with pytest.raises(TelegramNotificationError, match="chat not found"):
        notifier.send_message("Test")


def test_http_error(monkeypatch, notifier):
    def fake_urlopen(req, timeout):
        raise HTTPError(
            req.full_url,
            400,
            "Bad Request",
            {},
            BytesIO(b'{"ok":false,"description":"Invalid chat"}'),
        )

    monkeypatch.setattr(
        "aitrader.notifications.telegram.request.urlopen",
        fake_urlopen,
    )

    with pytest.raises(TelegramNotificationError, match="Invalid chat"):
        notifier.send_message("Test")


def test_network_error_does_not_expose_token(monkeypatch, notifier):
    def fake_urlopen(req, timeout):
        raise URLError("Connection failed")

    monkeypatch.setattr(
        "aitrader.notifications.telegram.request.urlopen",
        fake_urlopen,
    )

    with pytest.raises(TelegramNotificationError) as exc:
        notifier.send_message("Test")

    assert "test-secret-token" not in str(exc.value)


def test_invalid_json_response(monkeypatch, notifier):
    def fake_urlopen(req, timeout):
        class Response(FakeResponse):
            def read(self):
                return b"not json"

        return Response({})

    monkeypatch.setattr(
        "aitrader.notifications.telegram.request.urlopen",
        fake_urlopen,
    )

    with pytest.raises(TelegramNotificationError, match="Invalid Telegram"):
        notifier.send_message("Test")


def test_send_order_notification(monkeypatch, notifier):
    def fake_urlopen(req, timeout):
        payload = json.loads(req.data)

        assert "주문 체결" in payload["text"]
        assert "BTC/USDT" in payload["text"]

        return FakeResponse(success_response())

    monkeypatch.setattr(
        "aitrader.notifications.telegram.request.urlopen",
        fake_urlopen,
    )

    result = notifier.send_order_notification(make_order())

    assert result.message_id == 100


def test_send_rejected_order_notification(monkeypatch, notifier):
    def fake_urlopen(req, timeout):
        payload = json.loads(req.data)

        assert "주문 거부" in payload["text"]
        assert "Insufficient balance" in payload["text"]

        return FakeResponse(success_response())

    monkeypatch.setattr(
        "aitrader.notifications.telegram.request.urlopen",
        fake_urlopen,
    )

    order = replace(
        make_order(),
        status=OrderStatus.REJECTED,
        rejection_reason="Insufficient balance",
    )

    notifier.send_order_notification(order)


def test_send_position_opened(monkeypatch, notifier):
    def fake_urlopen(req, timeout):
        payload = json.loads(req.data)

        assert "포지션 진입" in payload["text"]
        assert "LONG" in payload["text"]

        return FakeResponse(success_response())

    monkeypatch.setattr(
        "aitrader.notifications.telegram.request.urlopen",
        fake_urlopen,
    )

    notifier.send_position_opened(make_position())


def test_send_position_closed(monkeypatch, notifier):
    def fake_urlopen(req, timeout):
        payload = json.loads(req.data)

        assert "포지션 청산" in payload["text"]
        assert "150" in payload["text"]

        return FakeResponse(success_response())

    monkeypatch.setattr(
        "aitrader.notifications.telegram.request.urlopen",
        fake_urlopen,
    )

    notifier.send_position_closed(
        make_position(PositionStatus.CLOSED)
    )


def test_open_position_cannot_use_closed_notification(notifier):
    with pytest.raises(ValueError):
        notifier.send_position_closed(make_position())


def test_daily_summary(monkeypatch, notifier):
    def fake_urlopen(req, timeout):
        payload = json.loads(req.data)

        assert "일일 거래 요약" in payload["text"]
        assert "66.67%" in payload["text"]

        return FakeResponse(success_response())

    monkeypatch.setattr(
        "aitrader.notifications.telegram.request.urlopen",
        fake_urlopen,
    )

    notifier.send_daily_summary(
        date="2026-11-18",
        total_trades=3,
        wins=2,
        losses=1,
        realized_pnl=120.0,
        equity=10_120.0,
    )


def test_invalid_daily_statistics(notifier):
    with pytest.raises(ValueError):
        notifier.send_daily_summary(
            date="2026-11-18",
            total_trades=2,
            wins=2,
            losses=2,
            realized_pnl=100.0,
            equity=10_100.0,
        )


def test_system_error_notification(monkeypatch, notifier):
    def fake_urlopen(req, timeout):
        payload = json.loads(req.data)

        assert "시스템 오류" in payload["text"]
        assert "RiskEngine" in payload["text"]

        return FakeResponse(success_response())

    monkeypatch.setattr(
        "aitrader.notifications.telegram.request.urlopen",
        fake_urlopen,
    )

    notifier.send_system_error(
        "RiskEngine",
        "Unexpected risk calculation failure",
    )


def test_invalid_timeout():
    with pytest.raises(ValueError):
        TelegramNotifier(
            "token",
            "123",
            timeout=0,
        )
