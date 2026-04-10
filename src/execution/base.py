from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.risk import RiskDecision
from src.strategies.signal_schema import StrategySignal


@dataclass
class ExecutionResult:
    executed: bool
    reason: str
    order_id: str = ""


class Executor(ABC):
    @abstractmethod
    def execute(self, signal: StrategySignal, risk: RiskDecision, market_price: float, timestamp: str) -> ExecutionResult:
        raise NotImplementedError
