from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ChallengerShadowStats:
    samples: int = 0
    avg_expected_value: float = 0.0


class GovernanceEvaluator:
    def should_promote(
        self,
        *,
        champion_status: str,
        champion_expectancy: float,
        drift_alert: bool,
        challenger_stats: ChallengerShadowStats,
    ) -> Tuple[bool, str]:
        if challenger_stats.samples < 20:
            return False, "insufficient_challenger_samples"
        if drift_alert:
            return False, "drift_alert_active"
        if champion_status == "degraded" and challenger_stats.avg_expected_value > 0:
            return True, "champion_degraded_challenger_positive_ev"
        if champion_expectancy < 0 and challenger_stats.avg_expected_value > 0:
            return True, "negative_expectancy_challenger_positive_ev"
        return False, "no_promotion_trigger"

    def should_rollback(self, *, champion_status: str, drift_alert: bool) -> Tuple[bool, str]:
        if champion_status == "degraded" and drift_alert:
            return True, "champion_degraded_and_drift_alert"
        return False, "no_rollback_trigger"
