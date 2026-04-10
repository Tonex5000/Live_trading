from __future__ import annotations

from typing import Dict, List


def compute_trade_metrics(trades: List[object]) -> Dict[str, float]:
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "total_pnl": 0.0,
            "ev_realization_ratio": 0.0,
        }

    pnls = [float(t.pnl) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = float(sum(pnls))
    win_rate = float(len(wins) / total_trades)
    avg_win = float(sum(wins) / len(wins)) if wins else 0.0
    avg_loss = float(sum(losses) / len(losses)) if losses else 0.0
    expectancy = total_pnl / total_trades

    expected_total = float(sum(float(getattr(t, "expected_value", 0.0)) for t in trades))
    ev_realization_ratio = float(total_pnl / expected_total) if abs(expected_total) > 1e-9 else 0.0

    return {
        "total_trades": float(total_trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "total_pnl": total_pnl,
        "ev_realization_ratio": ev_realization_ratio,
    }


def compute_rejection_rate(decisions: List[object]) -> float:
    if not decisions:
        return 0.0
    rejected = sum(1 for d in decisions if not bool(getattr(d, "approved", False)))
    return float(rejected / len(decisions))


def detect_degradation(expectancy: float, ev_realization_ratio: float) -> str:
    if expectancy < 0 or ev_realization_ratio < 0.7:
        return "degraded"
    if expectancy < 0.1:
        return "caution"
    return "healthy"
