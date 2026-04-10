from dataclasses import dataclass
from enum import Enum


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategySignal:
    symbol: str
    timeframe: str
    action: SignalAction
    confidence: float
    probability: float  # calibrated probability used for decisions
    reason: str
    timestamp: str
    raw_probability: float = 0.0
    calibrated_probability: float = 0.0
    regime: str = "unknown"
