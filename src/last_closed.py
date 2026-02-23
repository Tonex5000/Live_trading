import ccxt
import pandas as pd

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "1h"
TZ = "Africa/Lagos"

exchange = ccxt.bybit({
    "enableRateLimit": True,
    "timeout": 30000,
    "options": {"defaultType": "swap", "adjustForTimeDifference": True},
})

candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=10)
df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(TZ)

print(df[["timestamp", "open", "high", "low", "close", "volume"]].to_string(index=False))
print("\nNow (Lagos):", pd.Timestamp.now(tz=TZ))
