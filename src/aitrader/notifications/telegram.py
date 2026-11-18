import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib import request, error

from aitrader.config.settings import get_settings
from aitrader.execution.models import OrderRecord, OrderStatus
from aitrader.portfolio.models import Position, PositionStatus


class TelegramNotificationError(Exception):
    """Telegram notification delivery error."""


@dataclass(frozen=True, slots=True)
class TelegramSendResult:
    chat_id: str
    message_id: int
    date: int
    text: str


class TelegramNotifier:
    """
    Outbound Telegram notification client.

    Uses Telegram Bot API sendMessage.
    Does not receive updates or execute commands.
    """

    API_BASE_URL = "https://api.telegram.org"
    MAX_MESSAGE_LENGTH = 4096

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout: float = 10.0,
    ):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Invalid timeout")

        self._bot_token = bot_token.strip()
        self._chat_id = str(chat_id).strip()
        self.timeout = float(timeout)

    @classmethod
    def from_settings(cls):
        settings = get_settings()

        return cls(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self._bot_token and self._chat_id)

    @staticmethod
    def _format_number(value):
        return f"{value:,.8f}".rstrip("0").rstrip(".")

    def send_message(self, text: str) -> TelegramSendResult:

        if not self.is_configured:
            raise TelegramNotificationError(
                "Telegram credentials are not configured"
            )

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Message text is required")

        # Telegram sendMessage allows 1-4096 characters.
        text = text[:self.MAX_MESSAGE_LENGTH]

        url = (
            f"{self.API_BASE_URL}/bot"
            f"{self._bot_token}/sendMessage"
        )

        payload = {
            "chat_id": self._chat_id,
            "text": text,
        }

        encoded = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        http_request = request.Request(
            url,
            data=encoded,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self.timeout,
            ) as response:
                raw_response = response.read()

        except error.HTTPError as exc:
            raw_response = exc.read()
            self._raise_api_error(raw_response)

        except (error.URLError, TimeoutError, OSError) as exc:
            # Do not expose request URL because it contains the bot token.
            raise TelegramNotificationError(
                "Telegram network request failed"
            ) from exc

        return self._parse_response(raw_response, text)

    @staticmethod
    def _raise_api_error(raw_response):
        try:
            data = json.loads(raw_response.decode("utf-8"))
            description = str(
                data.get("description", "Unknown Telegram API error")
            )[:300]
        except (ValueError, UnicodeDecodeError, AttributeError):
            description = "Telegram HTTP request failed"

        raise TelegramNotificationError(description)

    def _parse_response(
        self,
        raw_response: bytes,
        text: str,
    ) -> TelegramSendResult:

        try:
            data = json.loads(raw_response.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise TelegramNotificationError(
                "Invalid Telegram API response"
            ) from None

        if not isinstance(data, dict):
            raise TelegramNotificationError(
                "Unexpected Telegram API response"
            )

        if data.get("ok") is not True:
            description = str(
                data.get("description", "Telegram request rejected")
            )[:300]

            raise TelegramNotificationError(description)

        result = data.get("result")

        if not isinstance(result, dict):
            raise TelegramNotificationError(
                "Telegram response has no message result"
            )

        message_id = result.get("message_id")
        date = result.get("date")
        chat = result.get("chat", {})

        if (
            not isinstance(message_id, int)
            or isinstance(message_id, bool)
            or not isinstance(date, int)
            or isinstance(date, bool)
        ):
            raise TelegramNotificationError(
                "Invalid Telegram message metadata"
            )

        returned_chat_id = (
            str(chat.get("id", self._chat_id))
            if isinstance(chat, dict)
            else self._chat_id
        )

        return TelegramSendResult(
            chat_id=returned_chat_id,
            message_id=message_id,
            date=date,
            text=text,
        )


    @staticmethod
    def _ai_label(value: object) -> str:
        raw = str(getattr(value, "value", value))

        labels = {
            "bullish": "상승",
            "bearish": "하락",
            "neutral": "중립",
            "bull_trend": "상승 추세",
            "bear_trend": "하락 추세",
            "range_or_transition": "횡보 / 전환",
            "high_volatility": "고변동성",
            "low": "낮음",
            "normal": "보통",
            "high": "높음",
            "unknown": "알 수 없음",
        }

        return labels.get(raw, raw)

    def send_ai_analysis(
        self,
        analysis,
        consensus,
        execution_signal,
    ) -> TelegramSendResult:

        timestamp = datetime.fromtimestamp(
            analysis.timestamp_ms / 1000,
            tz=UTC,
        ).strftime("%Y-%m-%d %H:%M:%S")

        available = ", ".join(
            consensus.available_timeframes
        ) or "없음"

        lines = [
            "🧠 AITrader AI 분석 완료",
            "",
            f"거래소: {analysis.exchange}",
            f"종목: {analysis.symbol}",
            f"분석봉: {analysis.timeframe}",
            f"분석 시각: {timestamp} UTC",
            "",
            (
                f"AI 판단: {self._ai_label(analysis.bias)} "
                f"({analysis.score:+.3f})"
            ),
            f"신뢰도: {analysis.confidence:.1%}",
            f"의견 일치도: {analysis.agreement:.1%}",
            f"시장 국면: {self._ai_label(analysis.regime)}",
            f"변동성: {self._ai_label(analysis.volatility)}",
            "",
            f"멀티타임프레임: {self._ai_label(consensus.bias)}",
            f"MTF 점수: {consensus.weighted_score:+.2f}",
            f"사용 타임프레임: {available}",
            "",
            f"실행 전략: {self._ai_label(execution_signal.bias)}",
            (
                f"전략 점수: "
                f"{execution_signal.score}/"
                f"{execution_signal.max_score}"
            ),
        ]

        if analysis.reports:
            lines.extend(["", "📋 에이전트별 판단"])

            for report in analysis.reports[:8]:
                lines.append(
                    f"• {report.agent_name}: "
                    f"{self._ai_label(report.bias)} "
                    f"(점수 {report.score:+.2f}, "
                    f"신뢰도 {report.confidence:.0%})"
                )

                if report.reasons:
                    lines.append(
                        f"  └ {report.reasons[0][:150]}"
                    )

        if analysis.reasons:
            lines.extend(["", "💡 AI 분석 근거"])

            for reason in analysis.reasons[:3]:
                lines.append(f"• {reason[:180]}")

        if consensus.reasons:
            lines.extend(["", "📊 시장 추세 근거"])

            for reason in consensus.reasons[:3]:
                lines.append(f"• {reason[:180]}")

        return self.send_message("\n".join(lines))

    def send_order_notification(
        self,
        order: OrderRecord,
    ) -> TelegramSendResult:

        if order.status == OrderStatus.FILLED:
            message = (
                "✅ AITrader Paper 주문 체결\n"
                f"거래소: {order.exchange}\n"
                f"종목: {order.symbol}\n"
                f"방향: {order.side.upper()}\n"
                f"체결가: {self._format_number(order.average_fill_price)}\n"
                f"수량: {self._format_number(order.filled_quantity)}\n"
                f"체결 금액: {self._format_number(order.filled_notional)}\n"
                f"수수료: {self._format_number(order.fee)}\n"
                f"주문 ID: {order.order_id}"
            )
        else:
            message = (
                "⛔ AITrader 주문 거부\n"
                f"거래소: {order.exchange}\n"
                f"종목: {order.symbol}\n"
                f"방향: {order.side.upper()}\n"
                f"사유: {order.rejection_reason or 'Unknown'}\n"
                f"주문 ID: {order.order_id}"
            )

        return self.send_message(message)

    def send_position_opened(
        self,
        position: Position,
    ) -> TelegramSendResult:

        message = (
            "📈 AITrader 포지션 진입\n"
            f"거래소: {position.exchange}\n"
            f"종목: {position.symbol}\n"
            f"방향: {position.side.value.upper()}\n"
            f"진입 가격: {self._format_number(position.entry_price)}\n"
            f"수량: {self._format_number(position.remaining_quantity)}\n"
            f"손절 가격: {self._format_number(position.stop_price)}\n"
            f"레버리지: {position.leverage}x\n"
            f"포지션 ID: {position.position_id}"
        )

        return self.send_message(message)

    def send_position_closed(
        self,
        position: Position,
    ) -> TelegramSendResult:

        if position.status != PositionStatus.CLOSED:
            raise ValueError("Position is not closed")

        message = (
            "🏁 AITrader 포지션 청산\n"
            f"거래소: {position.exchange}\n"
            f"종목: {position.symbol}\n"
            f"방향: {position.side.value.upper()}\n"
            f"진입 가격: {self._format_number(position.entry_price)}\n"
            f"실현 손익: {self._format_number(position.realized_pnl)}\n"
            f"포지션 ID: {position.position_id}"
        )

        return self.send_message(message)

    def send_daily_summary(
        self,
        date: str,
        total_trades: int,
        wins: int,
        losses: int,
        realized_pnl: float,
        equity: float,
        unrealized_pnl: float = 0.0,
    ) -> TelegramSendResult:

        if (
            not isinstance(total_trades, int)
            or isinstance(total_trades, bool)
            or not isinstance(wins, int)
            or isinstance(wins, bool)
            or not isinstance(losses, int)
            or isinstance(losses, bool)
            or total_trades < 0
            or wins < 0
            or losses < 0
            or wins + losses > total_trades
        ):
            raise ValueError("Invalid daily trade statistics")

        values = (
            realized_pnl,
            equity,
            unrealized_pnl,
        )

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ):
            raise ValueError("Invalid daily financial values")

        win_rate = (
            wins / (wins + losses) * 100
            if wins + losses
            else 0.0
        )

        message = (
            "📊 AITrader 일일 거래 요약\n"
            f"날짜: {date}\n"
            f"총 거래: {total_trades}\n"
            f"승리: {wins}\n"
            f"패배: {losses}\n"
            f"승률: {win_rate:.2f}%\n"
            f"실현 손익: {self._format_number(realized_pnl)}\n"
            f"미실현 손익: {self._format_number(unrealized_pnl)}\n"
            f"계좌 평가액: {self._format_number(equity)}"
        )

        return self.send_message(message)

    def send_system_error(
        self,
        component: str,
        message: str,
    ) -> TelegramSendResult:

        if not component.strip():
            raise ValueError("Component name is required")

        if not message.strip():
            raise ValueError("Error message is required")

        text = (
            "🚨 AITrader 시스템 오류\n"
            f"모듈: {component[:200]}\n"
            f"내용: {message[:2500]}"
        )

        return self.send_message(text)
