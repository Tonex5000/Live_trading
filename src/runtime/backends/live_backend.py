from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.runtime.backends.base import RuntimeBackend


class LiveBackend(RuntimeBackend):
    def startup(self) -> None:
        raise NotImplementedError("live mode backend is not implemented yet")

    def on_new_bar(self, df_window: pd.DataFrame) -> None:
        raise NotImplementedError("live mode backend is not implemented yet")

    def health(self, buffer_size: int) -> Dict[str, Any]:
        return {"status": "not_implemented", "buffer_size": buffer_size, "mode": "live"}

    def metrics(self) -> Dict[str, Any]:
        return {"mode": "live", "status": "not_implemented"}

    def shutdown(self) -> None:
        return None
