from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class RuntimeBackend(ABC):
    @abstractmethod
    def startup(self) -> None:
        ...

    @abstractmethod
    def on_new_bar(self, df_window: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def health(self, buffer_size: int) -> Dict[str, Any]:
        ...

    @abstractmethod
    def metrics(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...

    def latest_signal(self) -> Dict[str, Any] | None:
        return None

    def drift_observation(self) -> Dict[str, Any] | None:
        return None
