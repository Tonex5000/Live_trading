import logging

from fastapi import FastAPI
from pydantic import BaseModel
from sqlmodel import select

from src.config import load_app_config
from src.db import session_scope
from src.models import DecisionEvent, EquitySnapshot, Position, Signal, Trade
from src.runtime import RuntimeController

app = FastAPI(title="Crypto ML Live Engine")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("live_engine")

APP_CONFIG = load_app_config()
controller = RuntimeController(APP_CONFIG)
SYMBOL = controller.symbol
TIMEFRAME = controller.timeframe

class SignalResponse(BaseModel):
    timestamp: str | None = None
    signal: int | None = None
    action: str | None = None
    raw_probability: float | None = None
    calibrated_probability: float | None = None
    probability: float | None = None
    confidence: float | None = None
    regime: str | None = None
    drawdown: float | None = None
    position_size: float | None = None
    risk_reason: str | None = None
    risk_explanation: str | None = None
    expected_value: float | None = None
    expected_rr: float | None = None
    effective_risk_pct: float | None = None
    dynamic_risk_pct: float | None = None
    execution_reason: str | None = None
    buffer_size: int
    status: str | None = None


class HealthResponse(BaseModel):
    status: str
    buffer_size: int
    mode: str


class PromoteRequest(BaseModel):
    new_model: str
    reason: str | None = "manual_promotion"


@app.on_event("startup")
async def startup_event():
    await controller.startup()


@app.on_event("shutdown")
async def shutdown_event():
    await controller.shutdown()


@app.get("/latest_signal", response_model=SignalResponse)
def get_latest_signal():
    latest_signal = controller.latest_signal()
    if latest_signal is None:
        return {"status": "warming_up", "buffer_size": len(controller.candle_buffer)}
    return SignalResponse(**latest_signal)

@app.get("/signals")
def get_signals(limit: int = 20):
    with session_scope() as session:
        rows = session.exec(
            select(Signal)
            .where(Signal.symbol == SYMBOL, Signal.timeframe == TIMEFRAME)
            .order_by(Signal.created_at.desc())
            .limit(limit)
        ).all()
        return rows

@app.get("/decision_events")
def get_decision_events(limit: int = 100):
    with session_scope() as session:
        rows = session.exec(select(DecisionEvent).order_by(DecisionEvent.created_at.desc()).limit(limit)).all()
        return rows


@app.get("/positions")
def get_positions(status: str = "OPEN"):
    with session_scope() as session:
        rows = session.exec(select(Position).where(Position.status == status.upper()).order_by(Position.opened_at.desc())).all()
        return rows


@app.get("/trades")
def get_trades(limit: int = 100):
    with session_scope() as session:
        rows = session.exec(select(Trade).order_by(Trade.closed_at.desc()).limit(limit)).all()
        return rows


@app.get("/equity")
def get_equity(limit: int = 200):
    with session_scope() as session:
        rows = session.exec(select(EquitySnapshot).order_by(EquitySnapshot.timestamp.desc()).limit(limit)).all()
        return rows


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(**controller.health())


@app.get("/metrics")
def metrics():
    return controller.metrics()


@app.get("/drift/latest")
def drift_latest(symbol: str | None = None):
    return controller.drift_latest(symbol=symbol)


@app.get("/drift/events")
def drift_events(symbol: str | None = None, dimension: str | None = None, limit: int = 100):
    return controller.drift_events(symbol=symbol, dimension=dimension, limit=limit)


@app.get("/performance/latest")
def performance_latest(model_version: str | None = None):
    return controller.performance_latest(model_version=model_version)


@app.get("/performance/history")
def performance_history(model_version: str | None = None, limit: int = 100):
    return controller.performance_history(model_version=model_version, limit=limit)


@app.get("/models/registry")
def models_registry():
    return controller.model_registry()


@app.post("/models/promote")
def models_promote(payload: PromoteRequest):
    return controller.manual_promote(new_model_version=payload.new_model, reason=payload.reason or "manual_promotion")


@app.get("/models/promotions")
def models_promotions(limit: int = 100):
    return controller.model_promotions(limit=limit)