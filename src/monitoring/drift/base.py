from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class DriftDetector(ABC):
    @abstractmethod
    def fit_baseline(self, data: Iterable[float]) -> None:
        ...

    @abstractmethod
    def score(self, new_data: Iterable[float]) -> float:
        ...

    @abstractmethod
    def status(self, score: float) -> str:
        ...
