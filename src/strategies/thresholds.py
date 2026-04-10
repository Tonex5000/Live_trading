import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass
class ThresholdConfig:
    mode: str = "fixed"  # fixed | optimized
    p_buy: float = 0.60
    p_sell: float = 0.40
    optimized_path: str = "models/optimized_thresholds.json"


def optimize_thresholds(
    probs: Iterable[float],
    y_true: Iterable[int],
    buy_candidates=None,
    sell_candidates=None,
) -> Dict[str, float]:
    probs = np.asarray(list(probs), dtype=float)
    y_true = np.asarray(list(y_true), dtype=int)

    if buy_candidates is None:
        buy_candidates = np.arange(0.55, 0.81, 0.05)
    if sell_candidates is None:
        sell_candidates = np.arange(0.20, 0.46, 0.05)

    best = {"p_buy": 0.60, "p_sell": 0.40, "score": -1e9}

    for p_buy in buy_candidates:
        for p_sell in sell_candidates:
            if p_sell >= p_buy:
                continue
            pred = np.where(probs >= p_buy, 1, np.where(probs <= p_sell, -1, 0))
            # score = EV proxy + hit quality - abstention penalty
            buy_mask = pred == 1
            sell_mask = pred == -1
            ev_like = 0.0
            if buy_mask.any():
                ev_like += (y_true[buy_mask] == 1).mean() - (y_true[buy_mask] == -1).mean()
            if sell_mask.any():
                ev_like += (y_true[sell_mask] == -1).mean() - (y_true[sell_mask] == 1).mean()
            abstain_penalty = (pred == 0).mean() * 0.1
            score = ev_like - abstain_penalty
            if score > best["score"]:
                best = {"p_buy": float(p_buy), "p_sell": float(p_sell), "score": float(score)}

    return best


def save_optimized_thresholds(path: str, thresholds: Dict[str, float]) -> None:
    Path(path).write_text(json.dumps(thresholds, indent=2))


def load_thresholds(config: ThresholdConfig) -> Tuple[float, float]:
    if config.mode != "optimized":
        return config.p_buy, config.p_sell

    p = Path(config.optimized_path)
    if not p.exists():
        return config.p_buy, config.p_sell

    data = json.loads(p.read_text())
    return float(data.get("p_buy", config.p_buy)), float(data.get("p_sell", config.p_sell))
