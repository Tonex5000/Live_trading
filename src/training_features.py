from src.features import add_indicators

df = pd.read_csv("data/raw/BTCUSDT_1h.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

def add_training_features(df):
    df = add_indicators(df)

    df["future_close"] = df["close"].shift(-1)
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    df.dropna(inplace=True)
    return df
