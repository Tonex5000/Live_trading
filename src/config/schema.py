from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


RuntimeMode = Literal["paper", "shadow_live", "live"]


@dataclass
class RuntimeConfig:
    mode: RuntimeMode = "paper"
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT:USDT"])
    timeframe: str = "30m"
    poll_interval: int = 30
    max_buffer: int = 300

    def validate(self) -> None:
        if self.mode not in {"paper", "shadow_live", "live"}:
            raise ValueError(f"invalid runtime.mode: {self.mode}")
        if not self.symbols:
            raise ValueError("runtime.symbols must not be empty")
        if self.poll_interval <= 0:
            raise ValueError("runtime.poll_interval must be > 0")
        if self.max_buffer < 50:
            raise ValueError("runtime.max_buffer must be >= 50")


@dataclass
class DataConfig:
    exchange_id: str = "bybit"
    enable_rate_limit: bool = True
    timeout_ms: int = 30000
    default_type: str = "swap"
    adjust_for_time_difference: bool = True

    def validate(self) -> None:
        if not self.exchange_id:
            raise ValueError("data.exchange_id is required")
        if self.timeout_ms <= 0:
            raise ValueError("data.timeout_ms must be > 0")


@dataclass
class StrategyConfig:
    model_path: str = "models/xgb_signal_model.pkl"
    calibrator_path: str = "models/prob_calibrator.pkl"
    calibration_method: str = "platt"
    thresholds_mode: str = "fixed"
    thresholds_path: str = "models/optimized_thresholds.json"
    p_buy: float = 0.60
    p_sell: float = 0.40
    adx_min: float = 20.0
    allow_legacy_feature_fallback: bool = False

    def validate(self) -> None:
        if self.calibration_method not in {"none", "platt", "isotonic"}:
            raise ValueError(f"invalid strategy.calibration_method: {self.calibration_method}")
        if not (0.0 < self.p_buy < 1.0 and 0.0 < self.p_sell < 1.0):
            raise ValueError("strategy p_buy/p_sell must be between 0 and 1")


@dataclass
class RiskSectionConfig:
    max_risk_per_trade: float = 0.01
    max_concurrent_positions: int = 1
    cooldown_minutes: int = 30
    min_confidence: float = 0.30
    confidence_risk_floor: float = 0.50
    confidence_risk_ceiling: float = 1.00
    min_expected_value: float = 0.0
    atr_stop_mult: float = 0.8
    atr_tp_mult: float = 1.2
    min_rr: float = 1.5
    max_atr_vol_mult: float = 1.8
    atr_risk_cut_mult: float = 1.3
    atr_risk_multiplier: float = 0.7
    min_qty: float = 0.001
    min_notional: float = 5.0
    qty_precision: int = 6
    price_precision: int = 2
    stale_signal_minutes: int = 120
    drawdown_threshold: float = 0.05
    drawdown_risk_multiplier: float = 0.5
    max_total_risk_exposure: float = 0.03

    def validate(self) -> None:
        if self.max_concurrent_positions < 1:
            raise ValueError("risk.max_concurrent_positions must be >= 1")
        if self.cooldown_minutes < 0:
            raise ValueError("risk.cooldown_minutes must be >= 0")


@dataclass
class ExecutionConfig:
    fee_rate: float = 0.0006
    slippage_rate: float = 0.0004

    def validate(self) -> None:
        if self.fee_rate < 0 or self.slippage_rate < 0:
            raise ValueError("execution fee/slippage must be >= 0")


@dataclass
class MonitoringConfig:
    feature_drift_threshold: float = 2.0
    signal_drift_threshold: float = 2.0
    execution_drift_threshold: float = 0.01
    performance_window_trades: int = 200

    def validate(self) -> None:
        if self.performance_window_trades < 20:
            raise ValueError("monitoring.performance_window_trades must be >= 20")


@dataclass
class GovernanceConfig:
    model_selection_mode: str = "champion_only"
    champion_model_version: str = "default"
    challenger_model_version: str = ""

    def validate(self) -> None:
        if self.model_selection_mode not in {"champion_only", "champion_challenger"}:
            raise ValueError("governance.model_selection_mode must be champion_only or champion_challenger")


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskSectionConfig = field(default_factory=RiskSectionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)

    def validate(self) -> None:
        self.runtime.validate()
        self.data.validate()
        self.strategy.validate()
        self.risk.validate()
        self.execution.validate()
        self.monitoring.validate()
        self.governance.validate()

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AppConfig":
        runtime = RuntimeConfig(**payload.get("runtime", {}))
        data = DataConfig(**payload.get("data", {}))
        strategy = StrategyConfig(**payload.get("strategy", {}))
        risk = RiskSectionConfig(**payload.get("risk", {}))
        execution = ExecutionConfig(**payload.get("execution", {}))
        monitoring = MonitoringConfig(**payload.get("monitoring", {}))
        governance = GovernanceConfig(**payload.get("governance", {}))
        cfg = cls(
            runtime=runtime,
            data=data,
            strategy=strategy,
            risk=risk,
            execution=execution,
            monitoring=monitoring,
            governance=governance,
        )
        cfg.validate()
        return cfg
