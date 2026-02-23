import ccxt
import pandas as pd
from datetime import datetime


exchange = ccxt.binance()

symbol = "BTC/USDT"
timeframe = "1h"
limit = 1000

since = exchange.parse8601("2021-01-01T00:00:00Z")

all_candles = []

while True:
    candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    if len(candles) == 0:
        break

    since = candles[-1][0] +1 
    all_candles.extend(candles)


df = pd.DataFrame(
    all_candles,
    columns=["timestamp", "open", "high", "low", "close", "volume"]
)


df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

df.to_csv("data/raw/BTCUSDT_1h.csv", index=False)

print("Data saved successfully")