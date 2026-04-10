from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.config.defaults import default_config_dict
from src.config.schema import AppConfig

load_dotenv()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_from_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requested but PyYAML is not installed") from exc
        payload = yaml.safe_load(text)
        return payload or {}
    raise ValueError(f"unsupported config file type: {suffix}")


def _env_overrides() -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def setv(section: str, key: str, value: Any) -> None:
        out.setdefault(section, {})[key] = value

    # runtime
    if v := os.getenv("APP_MODE"):
        setv("runtime", "mode", v)
    if v := os.getenv("APP_SYMBOLS"):
        setv("runtime", "symbols", _parse_list(v))
    if v := os.getenv("APP_TIMEFRAME"):
        setv("runtime", "timeframe", v)
    if v := os.getenv("APP_POLL_INTERVAL"):
        setv("runtime", "poll_interval", int(v))
    if v := os.getenv("APP_MAX_BUFFER"):
        setv("runtime", "max_buffer", int(v))

    # data
    if v := os.getenv("APP_EXCHANGE_ID"):
        setv("data", "exchange_id", v)
    if v := os.getenv("APP_EXCHANGE_ENABLE_RATE_LIMIT"):
        setv("data", "enable_rate_limit", _parse_bool(v))
    if v := os.getenv("APP_EXCHANGE_TIMEOUT_MS"):
        setv("data", "timeout_ms", int(v))
    if v := os.getenv("APP_EXCHANGE_DEFAULT_TYPE"):
        setv("data", "default_type", v)

    # strategy
    if v := os.getenv("APP_MODEL_PATH"):
        setv("strategy", "model_path", v)
    if v := os.getenv("APP_CALIBRATOR_PATH"):
        setv("strategy", "calibrator_path", v)
    if v := os.getenv("APP_CALIBRATION_METHOD"):
        setv("strategy", "calibration_method", v)
    if v := os.getenv("APP_THRESHOLDS_MODE"):
        setv("strategy", "thresholds_mode", v)
    if v := os.getenv("APP_THRESHOLDS_PATH"):
        setv("strategy", "thresholds_path", v)
    if v := os.getenv("APP_P_BUY"):
        setv("strategy", "p_buy", float(v))
    if v := os.getenv("APP_P_SELL"):
        setv("strategy", "p_sell", float(v))
    if v := os.getenv("APP_ADX_MIN"):
        setv("strategy", "adx_min", float(v))
    if v := os.getenv("APP_ALLOW_LEGACY_FEATURE_FALLBACK"):
        setv("strategy", "allow_legacy_feature_fallback", _parse_bool(v))

    # risk
    risk_number_fields = {
        "APP_MAX_RISK_PER_TRADE": ("max_risk_per_trade", float),
        "APP_MAX_CONCURRENT_POSITIONS": ("max_concurrent_positions", int),
        "APP_COOLDOWN_MINUTES": ("cooldown_minutes", int),
        "APP_MIN_CONFIDENCE": ("min_confidence", float),
        "APP_CONFIDENCE_RISK_FLOOR": ("confidence_risk_floor", float),
        "APP_CONFIDENCE_RISK_CEILING": ("confidence_risk_ceiling", float),
        "APP_MIN_EXPECTED_VALUE": ("min_expected_value", float),
        "APP_ATR_STOP_MULT": ("atr_stop_mult", float),
        "APP_ATR_TP_MULT": ("atr_tp_mult", float),
        "APP_MIN_RR": ("min_rr", float),
        "APP_MAX_ATR_VOL_MULT": ("max_atr_vol_mult", float),
        "APP_ATR_RISK_CUT_MULT": ("atr_risk_cut_mult", float),
        "APP_ATR_RISK_MULTIPLIER": ("atr_risk_multiplier", float),
        "APP_MIN_QTY": ("min_qty", float),
        "APP_MIN_NOTIONAL": ("min_notional", float),
        "APP_QTY_PRECISION": ("qty_precision", int),
        "APP_PRICE_PRECISION": ("price_precision", int),
        "APP_STALE_SIGNAL_MINUTES": ("stale_signal_minutes", int),
        "APP_DRAWDOWN_THRESHOLD": ("drawdown_threshold", float),
        "APP_DRAWDOWN_RISK_MULTIPLIER": ("drawdown_risk_multiplier", float),
        "APP_MAX_TOTAL_RISK_EXPOSURE": ("max_total_risk_exposure", float),
    }
    for env_key, (cfg_key, caster) in risk_number_fields.items():
        raw = os.getenv(env_key)
        if raw is not None and raw != "":
            setv("risk", cfg_key, caster(raw))

    # execution
    if v := os.getenv("APP_FEE_RATE"):
        setv("execution", "fee_rate", float(v))
    if v := os.getenv("APP_SLIPPAGE_RATE"):
        setv("execution", "slippage_rate", float(v))

    # monitoring
    if v := os.getenv("APP_FEATURE_DRIFT_THRESHOLD"):
        setv("monitoring", "feature_drift_threshold", float(v))
    if v := os.getenv("APP_SIGNAL_DRIFT_THRESHOLD"):
        setv("monitoring", "signal_drift_threshold", float(v))
    if v := os.getenv("APP_EXECUTION_DRIFT_THRESHOLD"):
        setv("monitoring", "execution_drift_threshold", float(v))
    if v := os.getenv("APP_PERFORMANCE_WINDOW_TRADES"):
        setv("monitoring", "performance_window_trades", int(v))

    # governance
    if v := os.getenv("APP_MODEL_SELECTION_MODE"):
        setv("governance", "model_selection_mode", v)
    if v := os.getenv("APP_CHAMPION_MODEL_VERSION"):
        setv("governance", "champion_model_version", v)
    if v := os.getenv("APP_CHALLENGER_MODEL_VERSION"):
        setv("governance", "challenger_model_version", v)

    return out


def load_app_config(config_path: Optional[str] = None) -> AppConfig:
    payload = default_config_dict()

    resolved_path = config_path or os.getenv("APP_CONFIG_PATH", "").strip()
    if resolved_path:
        file_payload = _load_from_file(Path(resolved_path))
        payload = _deep_merge(payload, file_payload)

    payload = _deep_merge(payload, _env_overrides())
    return AppConfig.from_dict(payload)
