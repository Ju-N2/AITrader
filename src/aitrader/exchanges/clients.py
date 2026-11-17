import ccxt

from aitrader.config.settings import get_settings
from aitrader.exchanges.base import ExchangeClient


class UpbitClient(ExchangeClient):
    def __init__(self):
        exchange = ccxt.upbit({
            "enableRateLimit": True,
            "rateLimit": 1200,
        })
        super().__init__(exchange)

    def get_exchange_name(self) -> str:
        return "Upbit"


class BinanceSpotClient(ExchangeClient):
    def __init__(self):
        settings = get_settings()

        exchange = ccxt.binance({
            "apiKey": settings.binance_api_key,
            "secret": settings.binance_secret_key,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
            },
        })
        super().__init__(exchange)

    def get_exchange_name(self) -> str:
        return "Binance Spot"


class BinanceFuturesClient(ExchangeClient):
    def __init__(self):
        settings = get_settings()

        exchange = ccxt.binanceusdm({
            "apiKey": settings.binance_api_key,
            "secret": settings.binance_secret_key,
            "enableRateLimit": True,
        })
        super().__init__(exchange)

    def get_exchange_name(self) -> str:
        return "Binance USD-M Futures"
