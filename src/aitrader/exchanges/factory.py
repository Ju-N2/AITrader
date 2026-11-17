from aitrader.exchanges.base import ExchangeClient
from aitrader.exchanges.clients import (
    UpbitClient,
    BinanceSpotClient,
    BinanceFuturesClient,
)


class ExchangeFactory:

    @staticmethod
    def create(name: str) -> ExchangeClient:
        exchanges = {
            "upbit": UpbitClient,
            "binance_spot": BinanceSpotClient,
            "binance_futures": BinanceFuturesClient,
        }

        client_class = exchanges.get(name.lower())

        if client_class is None:
            raise ValueError(f"Unsupported exchange: {name}")

        return client_class()
