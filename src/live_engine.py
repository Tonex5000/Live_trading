from fastapi import FastAPI
import pandas as pd
import joblib
import asyncio
import ccxt
from pydantic import BaseModel
from src.db import create_db_and_tables
from src.db import get_session
from src.models import Signal
from typing import Optional
from sqlmodel import select
from src.features import add_indicators
from src.signals import generate_signals, apply_risk_filter

app = FastAPI(title="Crypto ML Live Engine (ccxt version)")

# =========================
# CONFIG
# =========================
SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "30m"
MAX_BUFFER = 300
POLL_INTERVAL = 30  # seconds

# =========================
# Load ML Model
# =========================
model = joblib.load("models/xgb_signal_model.pkl")

# =========================
# Exchange Setup (ccxt)
# =========================
exchange = ccxt.bybit({
    "enableRateLimit": True,
    "timeout": 30000,
    "options": {
        "defaultType": "swap",
        "adjustForTimeDifference": True
    }
})

# =========================
# Global State
# =========================
candle_buffer = pd.DataFrame(
    columns=["timestamp", "open", "high", "low", "close", "volume"]
)

latest_signal = None


class SignalResponse(BaseModel):
    timestamp: Optional[str] = None
    signal: Optional[int] = None
    probability: Optional[float] = None
    confidence: Optional[float] = None
    position_size: Optional[float] = None
    buffer_size: int
    status: Optional[str] = None


# =========================
# Load Historical Data
# =========================
def load_historical_data():
    global candle_buffer

    print("Loading historical candles...")

    candles = exchange.fetch_ohlcv(
        SYMBOL,
        TIMEFRAME,
        limit=MAX_BUFFER
    )

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        .dt.tz_convert("Africa/Lagos")
    )

    # Make sure it's clean
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    candle_buffer = df.copy()

    print(f"Loaded {len(candle_buffer)} historical candles.")


# =========================
# Process Latest Buffer
# =========================
def process_buffer():
    global latest_signal

    df = add_indicators(candle_buffer.copy())
    df = df.dropna().reset_index(drop=True)

    if len(df) == 0:
        print("No valid rows after indicators.")
        return

    latest_row = df.iloc[-1:]

    signal, prob = generate_signals(latest_row, model, p_buy=0.60, p_sell=0.40, adx_min=20.0)

    prob = prob[0]
    signal = signal[0]

    confidence = max(abs(prob - 0.5) * 2, 0.15)

    atr_mean = df["atr"].rolling(50).mean().iloc[-1]
    atr = latest_row["atr"].values[0]

    # Expected R:R from strategy SL/TP profile (1.2 / 0.8 = 1.5).
    expected_rr = 1.5
    stop_loss = atr * 0.8
    take_profit = atr * 1.2
    win_prob = prob if signal == 1 else (1 - prob)
    expected_gross_edge = (win_prob * take_profit) - ((1 - win_prob) * stop_loss)
    fee_rate = 0.0006
    slippage_rate = 0.0004
    expected_cost_edge = (latest_row["close"].values[0] * 2) * (fee_rate + slippage_rate)
    expected_net_edge = expected_gross_edge - expected_cost_edge
    signal_filtered = apply_risk_filter(
        signal,
        confidence,
        atr,
        atr_mean,
        expected_rr=expected_rr,
        expected_net_edge=expected_net_edge,
        min_rr=1.5,
    )

    # IMPORTANT: use iloc[0] + isoformat() to preserve Lagos timezone properly
    ts = latest_row["timestamp"].iloc[0]

    latest_signal = {
        "timestamp": ts.isoformat(),
        "signal": int(signal_filtered),
        "probability": float(prob),
        "confidence": float(confidence),
        "position_size": float(confidence if signal_filtered != 0 else 0),
        "buffer_size": len(candle_buffer)
    }

    print("New Signal Generated:", latest_signal)

    def save_signal_to_db(signal_dict):
        with get_session() as session:
            row = Signal(
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                timestamp=signal_dict["timestamp"],
                signal=signal_dict["signal"],
                probability=signal_dict["probability"],
                confidence=signal_dict["confidence"],
                position_size=signal_dict["position_size"],
                buffer_size=signal_dict["buffer_size"],
            )
            session.add(row)
            try:
                session.commit()
            except Exception:
                session.rollback()

    save_signal_to_db(latest_signal)

# =========================
# Helper: convert fetch_ohlcv to Lagos DataFrame
# =========================
def ohlcv_to_df(candles):
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        .dt.tz_convert("Africa/Lagos")
    )
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df


# =========================
# Polling Loop (No WebSocket) - GAP-FREE
# =========================
async def polling_loop():
    global candle_buffer

    last_timestamp = candle_buffer.iloc[-1]["timestamp"]  # Lagos tz-aware

    print("Starting polling loop...")

    while True:
        try:
            # Fetch enough recent candles so we can backfill gaps if any
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=MAX_BUFFER)
            df = ohlcv_to_df(candles)

            # last row may be forming candle, so last CLOSED candle is -2
            if len(df) < 3:
                await asyncio.sleep(POLL_INTERVAL)
                continue

            closed_df = df.iloc[:-1]  # exclude forming candle

            # Only take candles newer than what we already have
            new_df = closed_df[closed_df["timestamp"] > last_timestamp].copy()

            if len(new_df) > 0:
                print(f"Backfilling {len(new_df)} missing closed candle(s).")

                candle_buffer = pd.concat([candle_buffer, new_df], ignore_index=True)

                # Deduplicate + keep last
                candle_buffer = candle_buffer.drop_duplicates(subset=["timestamp"], keep="last")

                # Keep last MAX_BUFFER rows
                candle_buffer = candle_buffer.sort_values("timestamp").iloc[-MAX_BUFFER:].reset_index(drop=True)

                process_buffer()

                last_timestamp = candle_buffer.iloc[-1]["timestamp"]

        except Exception as e:
            print("Polling error:", e)

        await asyncio.sleep(POLL_INTERVAL)


# =========================
# FastAPI Startup
# =========================
@app.on_event("startup")
async def startup_event():
    create_db_and_tables()
    load_historical_data()
    process_buffer()
    asyncio.create_task(polling_loop())


# =========================
# API Endpoint
# =========================
@app.get("/latest_signal", response_model=SignalResponse)
def get_latest_signal():
    if latest_signal is None:
        return {
            "status": "warming_up",
            "buffer_size": len(candle_buffer)
        }
    return SignalResponse(**latest_signal)


@app.get("/signals")
def get_signals(limit: int = 5):
    with get_session() as session:
        rows = session.exec(
            select(Signal)
            .where(Signal.symbol == SYMBOL, Signal.timeframe == TIMEFRAME)
            .order_by(Signal.created_at.desc())
            .limit(limit)
        ).all()
        return rows