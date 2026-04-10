from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Tuple


class InMemoryBaselineStore:
    def __init__(self, maxlen: int = 500):
        self.maxlen = max(50, int(maxlen))
        self._store: Dict[Tuple[str, str], Deque[float]] = defaultdict(lambda: deque(maxlen=self.maxlen))

    def add(self, symbol: str, dimension: str, values: Iterable[float]) -> None:
        q = self._store[(symbol, dimension)]
        for val in values:
            q.append(float(val))

    def values(self, symbol: str, dimension: str) -> List[float]:
        return list(self._store[(symbol, dimension)])

    def count(self, symbol: str, dimension: str) -> int:
        return len(self._store[(symbol, dimension)])
