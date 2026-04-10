from typing import Optional
from datetime import datetime

from sqlmodel import SQLModel, Field, Index


class Signal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    symbol: str = Field(index=True)
    timeframe: str = Field(index=True)
    timestamp: str = Field(index=True)

    signal: int
    raw_probability: float = 0.0
    calibrated_probability: float = 0.0
    probability: float
    confidence: float
    position_size: float
    buffer_size: int
    reason: str = ""
    reason_code: str = ""
    explanation: str = ""
    regime: str = "unknown"
    drawdown: float = 0.0
    expected_value: float = 0.0
    expected_rr: float = 0.0
    effective_risk_pct: float = 0.0
    dynamic_risk_pct: float = 0.0
    estimated_cost: float = 0.0

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    __table_args__ = (
        Index("uq_signal_unique", "symbol", "timeframe", "timestamp", unique=True),
    )


class Position(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    side: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    risk_reason: str = ""
    confidence: float = 0.0
    expected_value: float = 0.0
    expected_rr: float = 0.0
    effective_risk_pct: float = 0.0
    estimated_cost: float = 0.0
    regime: str = "unknown"
    drawdown: float = 0.0
    status: str = Field(default="OPEN", index=True)
    opened_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    closed_at: Optional[datetime] = Field(default=None, index=True)


class Trade(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    side: str
    entry_price: float
    exit_price: float
    size: float
    fees: float = 0.0
    slippage_cost: float = 0.0
    pnl: float
    realized: bool = True
    risk_reason: str = ""
    confidence: float = 0.0
    expected_value: float = 0.0
    expected_rr: float = 0.0
    effective_risk_pct: float = 0.0
    estimated_cost: float = 0.0
    regime: str = "unknown"
    drawdown: float = 0.0
    opened_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    closed_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    exit_reason: str = "close"


class EquitySnapshot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    balance: float
    equity: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class DecisionEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timeframe: str = Field(index=True)
    timestamp: str = Field(index=True)
    action: str
    approved: bool
    reason_code: str = Field(index=True)
    explanation: str = ""
    regime: str = "unknown"
    drawdown: float = 0.0
    confidence: float = 0.0
    raw_probability: float = 0.0
    calibrated_probability: float = 0.0
    probability: float = 0.0
    expected_value: float = 0.0
    expected_rr: float = 0.0
    effective_risk_pct: float = 0.0
    dynamic_risk_pct: float = 0.0
    position_size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    estimated_cost: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class DriftEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    symbol: str = Field(index=True)
    dimension: str = Field(index=True)  # feature | signal | execution
    score: float = 0.0
    status: str = Field(default="warming_up", index=True)  # warming_up | ok | warning | alert
    metadata_json: str = "{}"


class ModelPerformanceSnapshot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    model_version: str = Field(default="default", index=True)
    window_size: int = 0
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    total_pnl: float = 0.0
    rejection_rate: float = 0.0
    ev_realization_ratio: float = 0.0
    status: str = Field(default="healthy", index=True)


class ModelRegistryEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_version: str = Field(index=True, unique=True)
    model_path: str
    calibrator_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    is_champion: bool = Field(default=False, index=True)


class ModelPromotionEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    old_model: str = Field(index=True)
    new_model: str = Field(index=True)
    reason: str = ""