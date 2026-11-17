import pytest

from aitrader.exchanges.factory import ExchangeFactory
from aitrader.exchanges.clients import (
    UpbitClient,
    BinanceSpotClient,
    BinanceFuturesClient,
)


def test_create_upbit():
    client = ExchangeFactory.create("upbit")

    assert isinstance(client, UpbitClient)
    assert client.get_exchange_name() == "Upbit"


def test_create_binance_spot():
    client = ExchangeFactory.create("binance_spot")

    assert isinstance(client, BinanceSpotClient)
    assert client.get_exchange_name() == "Binance Spot"


def test_create_binance_futures():
    client = ExchangeFactory.create("binance_futures")

    assert isinstance(client, BinanceFuturesClient)
    assert client.get_exchange_name() == "Binance USD-M Futures"


def test_invalid_exchange():
    with pytest.raises(ValueError):
        ExchangeFactory.create("unknown")
