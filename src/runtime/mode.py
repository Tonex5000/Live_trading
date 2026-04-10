from enum import Enum


class RuntimeMode(str, Enum):
    PAPER = "paper"
    SHADOW_LIVE = "shadow_live"
    LIVE = "live"

    @classmethod
    def from_value(cls, value: str) -> "RuntimeMode":
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported runtime mode: {value}") from exc
