import argparse
import logging
import signal
from pathlib import Path

from aitrader.core.instance_lock import (
    InstanceAlreadyRunningError,
    InstanceLock,
)
from aitrader.core.runtime import (
    PaperTradingRuntime,
    RuntimeConfig,
)
from aitrader.data.storage import SQLiteCandleRepository
from aitrader.exchanges.factory import ExchangeFactory
from aitrader.notifications import TelegramNotifier


def main():
    parser = argparse.ArgumentParser(
        description="AITrader Paper Trading Runtime"
    )

    parser.add_argument(
        "--exchange",
        choices=[
            "upbit",
            "binance_spot",
            "binance_futures",
        ],
        default="binance_spot",
    )

    parser.add_argument(
        "--symbol",
        default=None,
    )

    parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit",
    )

    parser.add_argument(
        "--paper",
        action="store_true",
        help="Enable simulated paper orders",
    )

    args = parser.parse_args()

    def stop_on_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_sigterm)

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    symbol = args.symbol

    if symbol is None:
        symbol = (
            "BTC/KRW"
            if args.exchange == "upbit"
            else "BTC/USDT"
        )

    notifier = TelegramNotifier.from_settings()

    if not notifier.is_configured:
        notifier = None

    config = RuntimeConfig(
        exchange=args.exchange,
        symbol=symbol,
        leverage=args.leverage,
        poll_interval_seconds=args.poll_seconds,
    )

    try:
        with InstanceLock(Path("data") / "aitrader.runtime.lock"):
            repository = SQLiteCandleRepository(
                "data/aitrader.db"
            )

            client = ExchangeFactory.create(
                config.exchange
            )

            try:
                runtime = PaperTradingRuntime(
                    config=config,
                    client=client,
                    repository=repository,
                    notifier=notifier,
                )

                runtime.bootstrap()

                if args.once:
                    mode = "PAPER" if args.paper else "DRY RUN"

                    print(f"\n========== AITrader {mode} ==========")

                    result = runtime.run_cycle(
                        paper_mode=args.paper,
                    )

                    print(f"Result: {result}")
                else:
                    runtime.run_forever(
                        paper_mode=args.paper,
                    )

            finally:
                client.close()
    except InstanceAlreadyRunningError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
