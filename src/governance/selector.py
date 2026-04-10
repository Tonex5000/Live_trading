from __future__ import annotations

from typing import Optional

import joblib

from src.config.schema import AppConfig
from src.governance.model_registry import ModelRegistryService
from src.ml.calibration import ProbabilityCalibrator
from src.strategies import MLStrategy, ThresholdConfig, load_thresholds


class ModelSelector:
    def __init__(self, config: AppConfig, registry: ModelRegistryService, *, symbol: str, timeframe: str):
        self.config = config
        self.registry = registry
        self.symbol = symbol
        self.timeframe = timeframe

    def _build_strategy(self, *, model_path: str, calibrator_path: str) -> MLStrategy:
        threshold_config = ThresholdConfig(
            mode=self.config.strategy.thresholds_mode,
            p_buy=self.config.strategy.p_buy,
            p_sell=self.config.strategy.p_sell,
            optimized_path=self.config.strategy.thresholds_path,
        )
        p_buy, p_sell = load_thresholds(threshold_config)
        model = joblib.load(model_path)
        calibrator = ProbabilityCalibrator.load(calibrator_path)
        return MLStrategy(
            model=model,
            symbol=self.symbol,
            timeframe=self.timeframe,
            p_buy=p_buy,
            p_sell=p_sell,
            adx_min=self.config.strategy.adx_min,
            allow_legacy_feature_fallback=self.config.strategy.allow_legacy_feature_fallback,
            probability_calibrator=calibrator,
        )

    def load_champion(self) -> MLStrategy:
        champion = self.registry.get_champion()
        if champion is None:
            return self._build_strategy(
                model_path=self.config.strategy.model_path,
                calibrator_path=self.config.strategy.calibrator_path,
            )
        return self._build_strategy(model_path=champion.model_path, calibrator_path=champion.calibrator_path)

    def load_challenger(self) -> Optional[MLStrategy]:
        challenger_version = self.config.governance.challenger_model_version
        if not challenger_version:
            return None
        challenger = self.registry.get_entry(challenger_version)
        if challenger is None:
            return None
        return self._build_strategy(model_path=challenger.model_path, calibrator_path=challenger.calibrator_path)
