from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.ml.calibration import ProbabilityCalibrator
from src.research.portfolio_simulator import PortfolioSimulationConfig, PortfolioStatefulSimulator
from src.research.stateful_simulator import StatefulBacktestEngine
from src.risk import RiskConfig, RiskManager
from src.strategies import MLStrategy


@dataclass
class WalkForwardConfig:
    train_bars: int
    calibration_bars: int
    test_bars: int
    step_bars: int
    mode: str = "rolling"


@dataclass
class CompositeObjectiveConfig:
    w_sharpe: float = 1.0
    w_return: float = 1.0
    w_profit_factor: float = 0.5
    w_drawdown: float = 1.0
    min_trade_count: int = 10
    penalty_low_trades: float = 2.0
    penalty_unstable_sharpe: float = 0.5
    max_drawdown_penalty_threshold: float = 0.20
    penalty_excessive_drawdown: float = 1.0


@dataclass
class StabilitySelectionConfig:
    use_stability_selection: bool = True

    # Fold weighting
    fold_weight_mode: str = "equal"  # equal | recency_weighted | regime_weighted | recency_and_regime_weighted
    recency_weight_method: str = "linear"  # linear | exponential
    recency_weight_strength: float = 1.0
    regime_weights: Dict[str, float] = None

    # Normalization
    stability_normalization: str = "zscore"  # zscore | rank | minmax
    use_weighted_metrics_for_selection: bool = True

    # Robust normalized score weights
    w_sharpe: float = 1.2
    w_sharpe_std: float = 0.8
    w_return: float = 1.0
    w_drawdown: float = 1.0
    w_return_std: float = 0.6
    w_stability_ratio: float = 0.4

    # Hard rejection thresholds
    min_sharpe_threshold: float = 0.0
    max_std_sharpe: float = 1.5
    min_trade_count_mean: float = 10.0
    max_losing_fold_pct: float = 0.5
    min_active_folds: int = 2

    # Penalties
    penalty_min_sharpe: float = 3.0
    penalty_low_trades: float = 2.0
    penalty_low_active_folds: float = 2.0
    penalty_losing_folds: float = 2.0

    # Runtime/diagnostics
    enable_legacy_comparison: bool = True
    enable_detailed_candidate_ranking: bool = True
    enable_pareto_diagnostics: bool = False
    enable_runtime_stats: bool = True

    # Simulation mode
    simulation_mode: str = "single_symbol"  # single_symbol | portfolio_multi_symbol
    portfolio_config: Optional[PortfolioSimulationConfig] = None

    epsilon: float = 1e-6
    param_std_threshold: float = 0.05

    def __post_init__(self):
        if self.regime_weights is None:
            self.regime_weights = {"trending": 1.0, "ranging": 1.0, "high_volatility": 1.0}
        if self.portfolio_config is None:
            self.portfolio_config = PortfolioSimulationConfig()


@dataclass
class ParameterGrid:
    p_buy: Iterable[float]
    p_sell: Iterable[float]
    min_confidence: Iterable[float]
    atr_stop_mult: Iterable[float]
    atr_tp_mult: Iterable[float]
    min_expected_value: Iterable[float]
    adx_min: Iterable[float]
    drawdown_risk_multiplier: Iterable[float] = (0.5,)
    atr_risk_multiplier: Iterable[float] = (0.7,)

    def iter_candidates(self) -> Iterable[Dict[str, float]]:
        keys = list(asdict(self).keys())
        values = [list(getattr(self, k)) for k in keys]
        for combo in product(*values):
            candidate = dict(zip(keys, combo))
            if candidate["p_sell"] >= candidate["p_buy"]:
                continue
            yield candidate


def _candidate_key(candidate: Dict[str, float]) -> str:
    return json.dumps(candidate, sort_keys=True)


def generate_walkforward_splits(n_rows: int, config: WalkForwardConfig) -> List[Dict[str, int]]:
    splits: List[Dict[str, int]] = []
    anchor = config.train_bars
    while True:
        train_end = anchor
        if config.mode == "rolling":
            train_start = train_end - config.train_bars
        elif config.mode == "expanding":
            train_start = 0
        else:
            raise ValueError(f"Unsupported walk-forward mode: {config.mode}")

        cal_start = train_end
        cal_end = cal_start + config.calibration_bars
        test_start = cal_end
        test_end = test_start + config.test_bars

        if test_end > n_rows:
            break

        splits.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "cal_start": cal_start,
                "cal_end": cal_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        anchor += config.step_bars

    return splits


def _max_drawdown_pct(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    arr = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(arr)
    dd = np.where(running_max > 0, (running_max - arr) / running_max, 0.0)
    return float(np.max(dd)) if len(dd) else 0.0


def _compute_metrics(trades: List[Dict], equity_curve: List[float], initial_capital: float) -> Dict[str, float]:
    pnls = np.asarray([t["pnl"] for t in trades], dtype=float) if trades else np.asarray([], dtype=float)
    returns = np.asarray([t.get("return", 0.0) for t in trades], dtype=float) if trades else np.asarray([], dtype=float)

    total_return = (equity_curve[-1] - initial_capital) / initial_capital if equity_curve else 0.0
    max_dd = _max_drawdown_pct(equity_curve)

    sharpe = 0.0
    if len(returns) > 1 and np.std(returns, ddof=1) > 0:
        sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(len(returns)))

    sortino = 0.0
    downside = returns[returns < 0]
    if len(returns) > 1 and len(downside) > 1 and np.std(downside, ddof=1) > 0:
        sortino = float((returns.mean() / np.std(downside, ddof=1)) * np.sqrt(len(returns)))

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 else (float("inf") if len(wins) > 0 else 0.0)
    win_rate = float(len(wins) / len(pnls)) if len(pnls) else 0.0
    avg_trade = float(pnls.mean()) if len(pnls) else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(total_return),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor if np.isfinite(profit_factor) else 10.0),
        "avg_trade": float(avg_trade),
        "expectancy": float(avg_trade),
        "trade_count": int(len(pnls)),
    }




def composite_score(metrics: Dict[str, float], cfg: CompositeObjectiveConfig) -> float:
    score = (
        cfg.w_sharpe * metrics["sharpe"]
        + cfg.w_return * metrics["total_return"]
        + cfg.w_profit_factor * min(metrics["profit_factor"], 5.0)
        - cfg.w_drawdown * metrics["max_drawdown"]
    )
    if metrics["trade_count"] < cfg.min_trade_count:
        score -= cfg.penalty_low_trades
    if metrics["trade_count"] < 2 or metrics["sharpe"] == 0.0:
        score -= cfg.penalty_unstable_sharpe
    if metrics["max_drawdown"] > cfg.max_drawdown_penalty_threshold:
        score -= cfg.penalty_excessive_drawdown
    return float(score)

def _dominant_regime(regime_counts: Dict[str, int]) -> str:
    if not regime_counts:
        return "unknown"
    return max(regime_counts.items(), key=lambda x: x[1])[0]


def _recency_weights(n: int, method: str, strength: float) -> np.ndarray:
    idx = np.arange(1, n + 1, dtype=float)
    if method == "linear":
        w = idx
    elif method == "exponential":
        w = np.exp(strength * (idx - 1) / max(1, n - 1))
    else:
        raise ValueError(f"Unsupported recency_weight_method: {method}")
    return w / np.sum(w)


def _fold_weights(eval_entries: List[Dict], cfg: StabilitySelectionConfig) -> np.ndarray:
    n = len(eval_entries)
    if n == 0:
        return np.asarray([], dtype=float)

    base = np.ones(n, dtype=float)

    if cfg.fold_weight_mode in {"recency_weighted", "recency_and_regime_weighted"}:
        base *= _recency_weights(n, cfg.recency_weight_method, cfg.recency_weight_strength)

    if cfg.fold_weight_mode in {"regime_weighted", "recency_and_regime_weighted"}:
        regime_w = np.asarray(
            [float(cfg.regime_weights.get(_dominant_regime(x.get("regimes", {})), 1.0)) for x in eval_entries],
            dtype=float,
        )
        if cfg.fold_weight_mode == "regime_weighted":
            base = regime_w
        else:
            base *= regime_w

    if cfg.fold_weight_mode == "equal":
        base = np.ones(n, dtype=float)

    s = float(np.sum(base))
    if s <= 0:
        return np.ones(n, dtype=float) / n
    return base / s


def summarize_candidate_metrics(eval_entries: List[Dict], cfg: StabilitySelectionConfig) -> Dict[str, float]:
    metrics_by_fold = [e["metrics"] for e in eval_entries]
    sharpe = np.asarray([m["sharpe"] for m in metrics_by_fold], dtype=float)
    ret = np.asarray([m["total_return"] for m in metrics_by_fold], dtype=float)
    dd = np.asarray([m["max_drawdown"] for m in metrics_by_fold], dtype=float)
    tc = np.asarray([m["trade_count"] for m in metrics_by_fold], dtype=float)

    weights = _fold_weights(eval_entries, cfg)

    summary = {
        # unweighted
        "mean_sharpe": float(np.mean(sharpe)) if len(sharpe) else 0.0,
        "std_sharpe": float(np.std(sharpe)) if len(sharpe) else 0.0,
        "min_sharpe": float(np.min(sharpe)) if len(sharpe) else 0.0,
        "mean_return": float(np.mean(ret)) if len(ret) else 0.0,
        "std_return": float(np.std(ret)) if len(ret) else 0.0,
        "max_drawdown_mean": float(np.mean(dd)) if len(dd) else 0.0,
        "max_drawdown_worst": float(np.max(dd)) if len(dd) else 0.0,
        "trade_count_mean": float(np.mean(tc)) if len(tc) else 0.0,
        "trade_count_std": float(np.std(tc)) if len(tc) else 0.0,
        # weighted
        "weighted_mean_sharpe": float(np.sum(weights * sharpe)) if len(sharpe) else 0.0,
        "weighted_mean_return": float(np.sum(weights * ret)) if len(ret) else 0.0,
        "weighted_drawdown": float(np.sum(weights * dd)) if len(dd) else 0.0,
        "weighted_trade_count": float(np.sum(weights * tc)) if len(tc) else 0.0,
        # diagnostics
        "profitable_fold_pct": float(np.mean(ret > 0)) if len(ret) else 0.0,
        "positive_sharpe_fold_pct": float(np.mean(sharpe > 0)) if len(sharpe) else 0.0,
        "best_worst_sharpe_gap": float(np.max(sharpe) - np.min(sharpe)) if len(sharpe) else 0.0,
        "stability_ratio": float(np.mean(sharpe) / (np.std(sharpe) + cfg.epsilon)) if len(sharpe) else 0.0,
        "active_folds": int(len(metrics_by_fold)),
        "losing_folds": int(np.sum(ret <= 0)) if len(ret) else 0,
        "weights": weights.tolist(),
    }
    return summary


def _hard_rejection_flags(summary: Dict[str, float], cfg: StabilitySelectionConfig) -> Dict[str, bool]:
    return {
        "reject_min_sharpe": summary["min_sharpe"] < cfg.min_sharpe_threshold,
        "reject_high_std_sharpe": summary["std_sharpe"] > cfg.max_std_sharpe,
        "reject_low_trades": summary["trade_count_mean"] < cfg.min_trade_count_mean,
        "reject_losing_folds": (summary["losing_folds"] / max(1, summary["active_folds"])) > cfg.max_losing_fold_pct,
        "reject_low_active_folds": summary["active_folds"] < cfg.min_active_folds,
    }


def _normalize_series(values: np.ndarray, method: str) -> np.ndarray:
    if len(values) == 0:
        return values
    if method == "zscore":
        std = np.std(values)
        if std == 0:
            return np.zeros_like(values)
        return (values - np.mean(values)) / std
    if method == "minmax":
        vmin, vmax = np.min(values), np.max(values)
        if vmax == vmin:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)
    if method == "rank":
        order = np.argsort(np.argsort(values))
        if len(values) == 1:
            return np.zeros_like(values)
        return order / float(len(values) - 1)
    raise ValueError(f"Unsupported stability_normalization: {method}")


def normalize_candidate_summaries(candidate_rows: List[Dict], cfg: StabilitySelectionConfig) -> List[Dict]:
    if not candidate_rows:
        return candidate_rows

    sharpe_key = "weighted_mean_sharpe" if cfg.use_weighted_metrics_for_selection else "mean_sharpe"
    return_key = "weighted_mean_return" if cfg.use_weighted_metrics_for_selection else "mean_return"
    dd_key = "weighted_drawdown" if cfg.use_weighted_metrics_for_selection else "max_drawdown_mean"

    keys = {
        "norm_sharpe": sharpe_key,
        "norm_sharpe_std": "std_sharpe",
        "norm_return": return_key,
        "norm_drawdown": dd_key,
        "norm_return_std": "std_return",
        "norm_stability_ratio": "stability_ratio",
    }

    normalized = {k: _normalize_series(np.asarray([r["candidate_summary"][v] for r in candidate_rows], dtype=float), cfg.stability_normalization) for k, v in keys.items()}

    for i, row in enumerate(candidate_rows):
        row["normalized_metrics"] = {k: float(arr[i]) for k, arr in normalized.items()}
    return candidate_rows


def robust_stability_score(row: Dict, cfg: StabilitySelectionConfig) -> Tuple[float, Dict[str, float]]:
    nm = row["normalized_metrics"]
    contributions = {
        "sharpe": cfg.w_sharpe * nm["norm_sharpe"],
        "sharpe_std": -cfg.w_sharpe_std * nm["norm_sharpe_std"],
        "return": cfg.w_return * nm["norm_return"],
        "drawdown": -cfg.w_drawdown * nm["norm_drawdown"],
        "return_std": -cfg.w_return_std * nm["norm_return_std"],
        "stability_ratio": cfg.w_stability_ratio * nm["norm_stability_ratio"],
    }
    score = float(sum(contributions.values()))

    flags = row["rejection_flags"]
    if flags["reject_min_sharpe"]:
        contributions["penalty_min_sharpe"] = -cfg.penalty_min_sharpe
        score -= cfg.penalty_min_sharpe
    if flags["reject_low_trades"]:
        contributions["penalty_low_trades"] = -cfg.penalty_low_trades
        score -= cfg.penalty_low_trades
    if flags["reject_losing_folds"]:
        contributions["penalty_losing_folds"] = -cfg.penalty_losing_folds
        score -= cfg.penalty_losing_folds
    if flags["reject_low_active_folds"]:
        contributions["penalty_low_active_folds"] = -cfg.penalty_low_active_folds
        score -= cfg.penalty_low_active_folds

    return score, contributions


def mark_pareto_front(candidate_rows: List[Dict]) -> List[Dict]:
    # maximize sharpe, minimize drawdown + instability
    for row in candidate_rows:
        row["is_pareto_efficient"] = True
    for i, a in enumerate(candidate_rows):
        sa = a["candidate_summary"]["weighted_mean_sharpe"]
        da = a["candidate_summary"]["weighted_drawdown"]
        ia = a["candidate_summary"]["std_sharpe"]
        for j, b in enumerate(candidate_rows):
            if i == j:
                continue
            sb = b["candidate_summary"]["weighted_mean_sharpe"]
            db = b["candidate_summary"]["weighted_drawdown"]
            ib = b["candidate_summary"]["std_sharpe"]
            if (sb >= sa and db <= da and ib <= ia) and (sb > sa or db < da or ib < ia):
                a["is_pareto_efficient"] = False
                break
    return candidate_rows


class WalkForwardRunner:
    def __init__(
        self,
        feature_cols: List[str],
        target_col: str = "target",
        initial_capital: float = 10_000.0,
        estimator_factory: Optional[Callable[[], object]] = None,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.initial_capital = initial_capital
        self.estimator_factory = estimator_factory or (
            lambda: XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        )

    @staticmethod
    def _assert_leakage_safe(split: Dict[str, int]) -> None:
        if not (
            split["train_start"] < split["train_end"] <= split["cal_start"] < split["cal_end"] <= split["test_start"] < split["test_end"]
        ):
            raise ValueError(f"Leakage-unsafe split boundaries detected: {split}")

    def _fit_model(self, train_df: pd.DataFrame):
        model = self.estimator_factory()
        model.fit(train_df[self.feature_cols], train_df[self.target_col])
        return model

    def _fit_calibrator(self, model, calibration_df: pd.DataFrame, method: str) -> ProbabilityCalibrator:
        if method == "none" or len(calibration_df) == 0:
            return ProbabilityCalibrator(method="none")
        raw_probs = model.predict_proba(calibration_df[self.feature_cols])[:, 1]
        y_true = (calibration_df[self.target_col] == 1).astype(int).values
        return ProbabilityCalibrator(method=method).fit(raw_probs, y_true)

    def _build_strategy_and_risk(self, model, calibrator: ProbabilityCalibrator, params: Dict[str, float], base_risk: RiskConfig):
        strategy = MLStrategy(
            model=model,
            symbol="BTC/USDT:USDT",
            timeframe="wf",
            p_buy=float(params["p_buy"]),
            p_sell=float(params["p_sell"]),
            adx_min=float(params["adx_min"]),
            allow_legacy_feature_fallback=False,
            probability_calibrator=calibrator,
        )
        risk_cfg = RiskConfig(
            **{
                **asdict(base_risk),
                **{
                    "min_confidence": float(params["min_confidence"]),
                    "atr_stop_mult": float(params["atr_stop_mult"]),
                    "atr_tp_mult": float(params["atr_tp_mult"]),
                    "min_expected_value": float(params["min_expected_value"]),
                    "drawdown_risk_multiplier": float(params.get("drawdown_risk_multiplier", base_risk.drawdown_risk_multiplier)),
                    "atr_risk_multiplier": float(params.get("atr_risk_multiplier", base_risk.atr_risk_multiplier)),
                },
            }
        )
        return strategy, RiskManager(risk_cfg), risk_cfg

    def _simulate_stateful(self, df: pd.DataFrame, model, calibrator: ProbabilityCalibrator, params: Dict[str, float], base_risk: RiskConfig) -> Dict:
        strategy, risk_manager, risk_cfg = self._build_strategy_and_risk(model, calibrator, params, base_risk)
        sim = StatefulBacktestEngine(risk_manager=risk_manager, risk_config=risk_cfg, initial_capital=self.initial_capital)
        run = sim.run(df=df, strategy=strategy)
        run["metrics"] = _compute_metrics(run["trades"], run["equity_curve"], self.initial_capital)
        return run

    def _simulate_portfolio(
        self,
        symbol_dfs: Dict[str, pd.DataFrame],
        model,
        calibrator: ProbabilityCalibrator,
        params: Dict[str, float],
        base_risk: RiskConfig,
        portfolio_cfg: PortfolioSimulationConfig,
    ) -> Dict:
        strategies = {}
        strategy, risk_manager, risk_cfg = self._build_strategy_and_risk(model, calibrator, params, base_risk)
        for sym in symbol_dfs.keys():
            strategies[sym] = MLStrategy(
                model=strategy.model,
                symbol=sym,
                timeframe="wf",
                p_buy=strategy.p_buy,
                p_sell=strategy.p_sell,
                adx_min=strategy.adx_min,
                allow_legacy_feature_fallback=False,
                probability_calibrator=calibrator,
            )

        sim = PortfolioStatefulSimulator(risk_manager=risk_manager, risk_config=risk_cfg, config=portfolio_cfg)
        return sim.run(symbol_dfs=symbol_dfs, strategies=strategies)

    def _simulate_legacy_stateless(self, df: pd.DataFrame, model, calibrator: ProbabilityCalibrator, params: Dict[str, float], base_risk: RiskConfig) -> Dict:
        strategy, risk_manager, _ = self._build_strategy_and_risk(model, calibrator, params, base_risk)
        equity = self.initial_capital
        peak = self.initial_capital
        trades = []
        eq = [equity]
        for i in range(1, len(df) - 1):
            hist = df.iloc[: i + 1]
            row = hist.iloc[-1]
            nxt = df.iloc[i + 1]
            sig = strategy.generate(hist)
            dec = risk_manager.evaluate(
                signal=sig,
                latest_price=float(row["close"]),
                atr=float(row["atr"]),
                account_balance=float(equity),
                open_positions_count=0,
                has_open_position_for_symbol=False,
                last_trade_at=None,
                now=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                atr_mean=float(hist["atr"].rolling(50, min_periods=1).mean().iloc[-1]),
                signal_timestamp=sig.timestamp,
                current_drawdown=(peak - equity) / peak if peak > 0 else 0.0,
                open_risk_exposure_pct=0.0,
            )
            if not dec.approved:
                eq.append(equity)
                continue
            entry = float(nxt["open"])
            exit_price = float(nxt["close"])
            qty = dec.position_size
            pnl = (exit_price - entry) * qty if sig.action.value == "BUY" else (entry - exit_price) * qty
            prev = equity
            equity += pnl
            peak = max(peak, equity)
            trades.append({"pnl": pnl, "return": pnl / prev if prev > 0 else 0.0})
            eq.append(equity)
        return {"metrics": _compute_metrics(trades, eq, self.initial_capital)}

    @staticmethod
    def _compare_new_vs_old(stateful_metrics: Dict[str, float], legacy_metrics: Dict[str, float]) -> Dict[str, float]:
        keys = ["sharpe", "max_drawdown", "total_return", "trade_count"]
        return {f"delta_{k}": float(stateful_metrics.get(k, 0.0) - legacy_metrics.get(k, 0.0)) for k in keys}

    @staticmethod
    def _parameter_stability(per_fold_best_params: List[Dict[str, float]], cfg: StabilitySelectionConfig) -> Dict[str, Dict[str, float]]:
        if not per_fold_best_params:
            return {}
        out = {}
        for k in sorted(per_fold_best_params[0].keys()):
            vals = np.asarray([float(x[k]) for x in per_fold_best_params], dtype=float)
            std = float(np.std(vals))
            out[k] = {
                "mean": float(np.mean(vals)),
                "std": std,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "is_stable": bool(std <= cfg.param_std_threshold),
            }
        return out

    def run(
        self,
        df: pd.DataFrame,
        wf_config: WalkForwardConfig,
        parameter_grid: ParameterGrid,
        objective_config: CompositeObjectiveConfig,
        base_risk_config: RiskConfig,
        output_dir: str = "models/walkforward",
        calibration_method: str = "none",
        include_legacy_comparison: Optional[bool] = None,
        stability_config: Optional[StabilitySelectionConfig] = None,
    ) -> Dict:
        cfg = stability_config or StabilitySelectionConfig()
        # backward-compatible argument
        if include_legacy_comparison is not None:
            cfg.enable_legacy_comparison = include_legacy_comparison

        timers = {}
        t0 = time.perf_counter()

        if cfg.simulation_mode == "portfolio_multi_symbol":
            if not isinstance(df, dict):
                raise ValueError("portfolio_multi_symbol requires df as Dict[str, DataFrame]")
            any_df = next(iter(df.values()))
            splits = generate_walkforward_splits(len(any_df), wf_config)
        else:
            splits = generate_walkforward_splits(len(df), wf_config)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # stage: fold artifacts
        s = time.perf_counter()
        fold_contexts = []
        for split in splits:
            self._assert_leakage_safe(split)
            if cfg.simulation_mode == "portfolio_multi_symbol":
                train_symbol = {s: sdf.iloc[split["train_start"] : split["train_end"]].reset_index(drop=True) for s, sdf in df.items()}
                cal_symbol = {s: sdf.iloc[split["cal_start"] : split["cal_end"]].reset_index(drop=True) for s, sdf in df.items()}
                test_symbol = {s: sdf.iloc[split["test_start"] : split["test_end"]].reset_index(drop=True) for s, sdf in df.items()}

                train_df = pd.concat(list(train_symbol.values()), axis=0, ignore_index=True)
                cal_df = pd.concat(list(cal_symbol.values()), axis=0, ignore_index=True)
                model = self._fit_model(train_df)
                calibrator = self._fit_calibrator(model, cal_df, method=calibration_method)
                fold_contexts.append(
                    {
                        "split": split,
                        "cal_symbol_dfs": cal_symbol,
                        "test_symbol_dfs": test_symbol,
                        "model": model,
                        "calibrator": calibrator,
                    }
                )
            else:
                train_df = df.iloc[split["train_start"] : split["train_end"]].reset_index(drop=True)
                cal_df = df.iloc[split["cal_start"] : split["cal_end"]].reset_index(drop=True)
                test_df = df.iloc[split["test_start"] : split["test_end"]].reset_index(drop=True)
                model = self._fit_model(train_df)
                calibrator = self._fit_calibrator(model, cal_df, method=calibration_method)
                fold_contexts.append({"split": split, "cal_df": cal_df, "test_df": test_df, "model": model, "calibrator": calibrator})
        timers["fit_folds_sec"] = time.perf_counter() - s

        candidates = list(parameter_grid.iter_candidates())

        # shared candidate-fold cache
        s = time.perf_counter()
        eval_cache: Dict[Tuple[int, str], Dict] = {}
        cache_hits = 0
        total_evals = 0

        def eval_candidate_fold(fold_idx: int, candidate: Dict[str, float]) -> Dict:
            nonlocal cache_hits, total_evals
            key = (fold_idx, _candidate_key(candidate))
            if key in eval_cache:
                cache_hits += 1
                return eval_cache[key]
            total_evals += 1
            ctx = fold_contexts[fold_idx]
            if cfg.simulation_mode == "portfolio_multi_symbol":
                res = self._simulate_portfolio(
                    ctx["cal_symbol_dfs"],
                    ctx["model"],
                    ctx["calibrator"],
                    candidate,
                    base_risk_config,
                    cfg.portfolio_config,
                )
            else:
                res = self._simulate_stateful(ctx["cal_df"], ctx["model"], ctx["calibrator"], candidate, base_risk_config)
            eval_cache[key] = {
                "fold_id": fold_idx,
                "candidate_id": key[1],
                "params": candidate,
                "metrics": res["metrics"],
                "rejections": res.get("rejections", {}),
                "regimes": res.get("regimes", {}),
            }
            return eval_cache[key]

        candidate_rows = []
        for cand in candidates:
            cid = _candidate_key(cand)
            fold_entries = [eval_candidate_fold(i, cand) for i in range(len(fold_contexts))]
            summary = summarize_candidate_metrics(fold_entries, cfg)
            flags = _hard_rejection_flags(summary, cfg)
            candidate_rows.append(
                {
                    "candidate_id": cid,
                    "params": cand,
                    "fold_entries": fold_entries,
                    "candidate_summary": summary,
                    "rejection_flags": flags,
                }
            )
        timers["evaluate_candidates_sec"] = time.perf_counter() - s

        # normalize + robust scoring
        s = time.perf_counter()
        candidate_rows = normalize_candidate_summaries(candidate_rows, cfg)
        for row in candidate_rows:
            score, contrib = robust_stability_score(row, cfg)
            row["stability_score"] = score
            row["score_contributions"] = contrib

        if cfg.enable_pareto_diagnostics:
            candidate_rows = mark_pareto_front(candidate_rows)
        else:
            for row in candidate_rows:
                row["is_pareto_efficient"] = False

        ranked = sorted(candidate_rows, key=lambda x: x["stability_score"], reverse=True)
        timers["score_rank_sec"] = time.perf_counter() - s

        # test evaluation and per-fold artifacts
        s = time.perf_counter()
        per_fold = []
        per_fold_best_params = []

        for fold_id, ctx in enumerate(fold_contexts):
            # per-fold best from cached calibration metrics (no recomputation)
            fold_best = max(
                candidates,
                key=lambda cand: composite_score(eval_candidate_fold(fold_id, cand)["metrics"], objective_config),
            )
            per_fold_best_params.append(fold_best)

            selected = ranked[0]["params"] if cfg.use_stability_selection else fold_best
            selected_source = "global_stability_selection" if cfg.use_stability_selection else "per_fold_objective_selection"
            selected_id = _candidate_key(selected)
            selected_row = next(x for x in ranked if x["candidate_id"] == selected_id)

            if cfg.simulation_mode == "portfolio_multi_symbol":
                test_eval = self._simulate_portfolio(
                    ctx["test_symbol_dfs"],
                    ctx["model"],
                    ctx["calibrator"],
                    selected,
                    base_risk_config,
                    cfg.portfolio_config,
                )
                any_test_df = next(iter(ctx["test_symbol_dfs"].values()))
                start_ts = str(any_test_df["timestamp"].iloc[0])
                end_ts = str(any_test_df["timestamp"].iloc[-1])
            else:
                test_eval = self._simulate_stateful(ctx["test_df"], ctx["model"], ctx["calibrator"], selected, base_risk_config)
                start_ts = str(ctx["test_df"]["timestamp"].iloc[0])
                end_ts = str(ctx["test_df"]["timestamp"].iloc[-1])
            legacy = None
            delta = None
            if cfg.enable_legacy_comparison and cfg.simulation_mode != "portfolio_multi_symbol":
                legacy = self._simulate_legacy_stateless(ctx["test_df"], ctx["model"], ctx["calibrator"], selected, base_risk_config)
                delta = self._compare_new_vs_old(test_eval["metrics"], legacy["metrics"])

            fold_artifact = {
                "fold_id": fold_id,
                "ranges": ctx["split"],
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "calibration_method": calibration_method,
                "candidate_used": selected,
                "candidate_source": selected_source,
                "consistency_flags": selected_row["rejection_flags"],
                "candidate_stability_metrics": selected_row["candidate_summary"],
                "test_metrics": test_eval["metrics"],
                "position_metrics": test_eval.get("position_metrics", {}),
                "trade_count": test_eval["metrics"]["trade_count"],
                "drawdown_summary": {"max_drawdown": test_eval["metrics"]["max_drawdown"]},
                "equity_curve_summary": {"start_equity": test_eval["equity_curve"][0], "end_equity": test_eval["equity_curve"][-1]},
                "regime_summary": test_eval.get("regimes", {}),
                "rejection_counts": test_eval.get("rejections", {}),
                "per_symbol_metrics": test_eval.get("per_symbol_metrics", []),
                "portfolio_equity_curve": test_eval.get("equity_curve", []),
                "model_metadata": {"type": type(ctx["model"]).__name__},
                "legacy_comparison": {
                    "legacy_metrics": legacy["metrics"] if legacy else None,
                    "difference": delta,
                },
                "leakage_audit": {
                    "split_safe": True,
                    "history_window_used": "df.iloc[:i+1] for signal and risk state",
                    "calibration_seen_test": False,
                    "optimization_seen_test": False,
                },
            }
            fold_path = out_dir / f"fold_{fold_id}_result.json"
            fold_path.write_text(json.dumps(fold_artifact, indent=2))
            fold_artifact["artifact_path"] = str(fold_path)
            per_fold.append(fold_artifact)
        timers["test_eval_sec"] = time.perf_counter() - s

        summary = self._aggregate_results(per_fold)
        summary["best_candidate_params"] = ranked[0]["params"] if ranked else None
        summary["best_candidate_stability"] = ranked[0]["candidate_summary"] if ranked else {}
        summary["selection_mode"] = "stability" if cfg.use_stability_selection else "per_fold"
        summary["parameter_stability"] = self._parameter_stability(per_fold_best_params, cfg)

        top = ranked if cfg.enable_detailed_candidate_ranking else ranked[:5]
        summary["top_candidates"] = [
            {
                "rank": i + 1,
                "params": c["params"],
                "stability_score": c["stability_score"],
                "candidate_summary": c["candidate_summary"],
                "normalized_metrics": c.get("normalized_metrics", {}),
                "score_contributions": c.get("score_contributions", {}),
                "rejection_flags": c["rejection_flags"],
                "is_pareto_efficient": c.get("is_pareto_efficient", False),
            }
            for i, c in enumerate(top[:5])
        ]

        if cfg.enable_runtime_stats:
            timers["total_runtime_sec"] = time.perf_counter() - t0
            summary["runtime_stats"] = {
                "total_candidates": len(candidates),
                "total_folds": len(fold_contexts),
                "total_candidate_fold_evaluations": total_evals,
                "cache_reuse_count": cache_hits,
                **{k: float(v) for k, v in timers.items()},
            }

        (out_dir / "per_fold_results.json").write_text(json.dumps(per_fold, indent=2))
        (out_dir / "walkforward_summary.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame([f["test_metrics"] | {"fold_id": f["fold_id"]} for f in per_fold]).to_csv(out_dir / "fold_metrics.csv", index=False)
        if cfg.simulation_mode == "portfolio_multi_symbol":
            (out_dir / "portfolio_per_fold_results.json").write_text(json.dumps(per_fold, indent=2))
            (out_dir / "portfolio_walkforward_summary.json").write_text(json.dumps(summary, indent=2))
            all_equity_rows = []
            all_symbol_rows = []
            all_rejections = []
            for f in per_fold:
                fid = f["fold_id"]
                for erow in f.get("portfolio_equity_curve", []):
                    all_equity_rows.append({"fold_id": fid, **erow})
                for srow in f.get("per_symbol_metrics", []):
                    all_symbol_rows.append({"fold_id": fid, **srow})
                for r, c in f.get("rejection_counts", {}).items():
                    all_rejections.append({"fold_id": fid, "reason": r, "count": c})
            if all_equity_rows:
                pd.DataFrame(all_equity_rows).to_csv(out_dir / "portfolio_equity_curve.csv", index=False)
            if all_symbol_rows:
                pd.DataFrame(all_symbol_rows).to_csv(out_dir / "per_symbol_metrics.csv", index=False)
            if all_rejections:
                pd.DataFrame(all_rejections).to_csv(out_dir / "portfolio_rejections.csv", index=False)
        pd.DataFrame(
            [
                {
                    "candidate_id": c["candidate_id"],
                    **{f"param_{k}": v for k, v in c["params"].items()},
                    **c["candidate_summary"],
                    **{f"norm_{k}": v for k, v in c.get("normalized_metrics", {}).items()},
                    **{f"contrib_{k}": v for k, v in c.get("score_contributions", {}).items()},
                    "stability_score": c["stability_score"],
                    **{f"flag_{k}": v for k, v in c["rejection_flags"].items()},
                    "is_pareto_efficient": c.get("is_pareto_efficient", False),
                }
                for c in ranked
            ]
        ).to_csv(out_dir / "candidate_stability_ranking.csv", index=False)

        return {"summary": summary, "folds": per_fold, "candidate_ranking": ranked}

    @staticmethod
    def _aggregate_results(per_fold: List[Dict]) -> Dict:
        if not per_fold:
            return {"fold_count": 0}

        test_metrics = [f["test_metrics"] for f in per_fold]
        metric_names = ["total_return", "sharpe", "max_drawdown", "profit_factor", "trade_count", "win_rate", "expectancy"]

        aggregate = {}
        for m in metric_names:
            vals = np.asarray([tm[m] for tm in test_metrics], dtype=float)
            aggregate[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}

        best_fold = max(per_fold, key=lambda x: x["test_metrics"]["sharpe"])
        worst_fold = min(per_fold, key=lambda x: x["test_metrics"]["sharpe"])

        p_buy_values = [f["candidate_used"]["p_buy"] for f in per_fold]
        p_sell_values = [f["candidate_used"]["p_sell"] for f in per_fold]

        threshold_stability = {
            "p_buy_std": float(np.std(p_buy_values)),
            "p_sell_std": float(np.std(p_sell_values)),
            "p_buy_range": [float(np.min(p_buy_values)), float(np.max(p_buy_values))],
            "p_sell_range": [float(np.min(p_sell_values)), float(np.max(p_sell_values))],
        }

        return {
            "fold_count": len(per_fold),
            "aggregate_metrics": aggregate,
            "stability": {"threshold_stability": threshold_stability},
            "best_fold": {"fold_id": best_fold["fold_id"], "sharpe": best_fold["test_metrics"]["sharpe"]},
            "worst_fold": {"fold_id": worst_fold["fold_id"], "sharpe": worst_fold["test_metrics"]["sharpe"]},
        }
