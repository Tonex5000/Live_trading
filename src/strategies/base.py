from abc import ABC, abstractmethod
import pandas as pd

from src.strategies.signal_schema import StrategySignal


class BaseStrategy(ABC):
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> StrategySignal:
        raise NotImplementedError
