from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from src.config.schema import AppConfig


DEFAULT_APP_CONFIG = AppConfig()


def default_config_dict() -> Dict[str, Any]:
    return asdict(DEFAULT_APP_CONFIG)
