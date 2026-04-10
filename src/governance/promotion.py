from __future__ import annotations

from typing import Optional

from src.governance.evaluator import ChallengerShadowStats, GovernanceEvaluator
from src.governance.model_registry import ModelRegistryService


class PromotionManager:
    def __init__(self, registry: ModelRegistryService, evaluator: GovernanceEvaluator):
        self.registry = registry
        self.evaluator = evaluator

    def maybe_promote(
        self,
        *,
        challenger_version: str,
        champion_status: str,
        champion_expectancy: float,
        drift_alert: bool,
        challenger_stats: ChallengerShadowStats,
    ) -> Optional[str]:
        promote, reason = self.evaluator.should_promote(
            champion_status=champion_status,
            champion_expectancy=champion_expectancy,
            drift_alert=drift_alert,
            challenger_stats=challenger_stats,
        )
        if not promote:
            return None
        event = self.registry.promote(new_model_version=challenger_version, reason=reason)
        if event is None:
            return None
        return event.reason

    def maybe_rollback(
        self,
        *,
        fallback_version: str,
        champion_status: str,
        drift_alert: bool,
    ) -> Optional[str]:
        rollback, reason = self.evaluator.should_rollback(
            champion_status=champion_status,
            drift_alert=drift_alert,
        )
        if not rollback:
            return None
        event = self.registry.promote(new_model_version=fallback_version, reason=f"rollback:{reason}")
        if event is None:
            return None
        return event.reason
