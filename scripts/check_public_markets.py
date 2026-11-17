from aitrader.exchanges.factory import ExchangeFactory


EXCHANGES = [
    ("Upbit", "upbit", "BTC/KRW"),
    ("Binance Spot", "binance_spot", "BTC/USDT"),
    ("Binance Futures", "binance_futures", "BTC/USDT:USDT"),
]


def main():
    for label, exchange_name, symbol in EXCHANGES:
        client = None

        try:
            print(f"\n[{label}] Connecting...")

            client = ExchangeFactory.create(exchange_name)

            markets = client.load_markets()

            if symbol not in markets:
                raise ValueError(f"Market not found: {symbol}")

            ticker = client.fetch_ticker(symbol)

            print(f"[SUCCESS] {label}")
            print(f"Symbol: {symbol}")
            print(f"Current Price: {ticker.get('last')}")
            print(f"Bid: {ticker.get('bid')}")
            print(f"Ask: {ticker.get('ask')}")
            print(f"24h Volume: {ticker.get('baseVolume')}")
            print(f"Updated: {ticker.get('datetime')}")

        except Exception as error:
            print(f"[FAILED] {label}")
            print(f"Error: {type(error).__name__}: {error}")

        finally:
            if client:
                client.close()


if __name__ == "__main__":
    main()
