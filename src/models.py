from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, Index

class Signal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    symbol: str = Field(index=True)
    timeframe: str = Field(index=True)

    timestamp: str = Field(index=True)

    signal: int
    probability: float
    confidence: float
    position_size: float
    buffer_size: int

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    __table_args__ = (
        Index("uq_signal_unique", "symbol", "timeframe", "timestamp", unique=True),
    )