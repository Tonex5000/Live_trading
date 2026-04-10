from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.risk import RiskConfig, RiskManager
from src.research.stateful_simulator import SimPosition
from src.strategies.signal_schema import SignalAction


@dataclass
class SymbolMetadata:
    symbol: str
    group: str = "default"
    priority_rank: int = 0
    min_qty: Optional[float] = None
    min_notional: Optional[float] = None
    qty_precision: Optional[int] = None
    price_precision: Optional[int] = None


@dataclass
class PortfolioSimulationConfig:
    simulation_mode: str = "single_symbol"  # single_symbol | portfolio_multi_symbol
    initial_capital: float = 10_000.0
    max_open_positions: int = 3
    max_total_risk_exposure: float = 0.03
    max_symbol_allocation_pct: float = 0.5
    candidate_ranking_mode: str = "highest_expected_value"  # highest_expected_value | highest_confidence | highest_risk_adjusted_score | symbol_order
    allow_multiple_symbols: bool = True
    one_position_per_symbol: bool = True

    enable_group_constraints: bool = False
    group_exposure_limits: Dict[str, float] = field(default_factory=dict)
    symbol_metadata: Dict[str, SymbolMetadata] = field(default_factory=dict)

    timestamp_alignment_mode: str = "union"
    missing_bar_policy: str = "skip"

    enable_portfolio_runtime_stats: bool = True
    enable_portfolio_diagnostics: bool = True
    enable_allocation_competition_logs: bool = True

    # Correlation-aware risk
    correlation_window: int = 100
    correlation_threshold: float = 0.7
    correlation_penalty_strength: float = 0.5
    max_correlated_exposure: float = 0.03
    enable_correlation_risk: bool = True
    enable_correlation_rejection: bool = True
    enable_correlation_scaling: bool = True
    use_absolute_correlation: bool = True

    # Lightweight cluster controls
    cluster_threshold: float = 0.7
    max_cluster_positions: int = 2
    max_cluster_exposure: float = 0.6
    diagnostics_output_dir: Optional[str] = None
    enable_iterative_allocation: bool = True
    max_refinement_iterations: int = 8
    allocation_step_fraction: float = 0.08
    min_improvement_threshold: float = 1e-6
    donor_pool_size: int = 2
    receiver_pool_size: int = 2
    max_repair_iterations: int = 6
    repair_shrink_factor: float = 0.9
    use_marginal_utility_refinement: bool = True
    w_ev: float = 1.0
    w_corr: float = 0.15
    w_drawdown: float = 0.1
    w_concentration: float = 0.2
    w_diversification: float = 0.1
    use_projection_repair: bool = True
    max_projection_iterations: int = 10
    projection_tolerance: float = 1e-8
    fallback_to_simple_repair: bool = True
    use_delta_marginal_utility: bool = True
    marginal_probe_fraction: float = 0.02
    marginal_probe_min_notional: float = 10.0
    enable_projection_diagnostics: bool = True
    enable_utility_weight_calibration: bool = False
    utility_weight_grid: Dict[str, List[float]] = field(default_factory=dict)
    utility_validation_objective_weights: Dict[str, float] = field(default_factory=lambda: {"sharpe": 1.0, "drawdown": 0.5, "return": 0.5, "stability": 0.5})
    utility_turnover_penalty: float = 0.0
    utility_avg_corr_penalty: float = 0.1
    utility_concentration_penalty: float = 0.1
    enable_utility_sensitivity_reporting: bool = True
    enable_execution_realism: bool = True
    spread_bps: float = 5.0
    base_slippage_bps: float = 2.0
    slippage_size_coefficient: float = 0.5
    slippage_volatility_coefficient: float = 0.2
    max_participation_rate: float = 0.1
    execution_delay_bars: int = 0
    stale_signal_bars: int = 2
    enable_gap_aware_fills: bool = True
    enable_stress_testing: bool = False
    stress_scenarios: Dict[str, Dict[str, float]] = field(default_factory=dict)
    max_spread_bps_for_entry: float = 30.0
    min_liquidity_threshold: float = 1.0
    max_fill_cost_bps: float = 50.0


class PortfolioStatefulSimulator:
    def __init__(self, risk_manager: RiskManager, risk_config: RiskConfig, config: PortfolioSimulationConfig):
        self.risk_manager = risk_manager
        self.risk_config = risk_config
        self.config = config

    @staticmethod
    def _to_dt(value) -> datetime:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            return ts.tz_convert(None).to_pydatetime()
        return ts.to_pydatetime()

    def _execution_cost_bps(self, requested_size: float, bar_volume: float, atr: float, price: float) -> float:
        base = float(self.config.base_slippage_bps)
        participation = abs(requested_size) / max(bar_volume, 1e-9)
        size_term = self.config.slippage_size_coefficient * max(0.0, participation * 100.0)
        vol_term = self.config.slippage_volatility_coefficient * (atr / max(price, 1e-9))
        return max(0.0, base + size_term + vol_term * 10_000.0)

    def _entry_fill(self, action: SignalAction, market_price: float, requested_size: float = 0.0, bar_volume: float = 1.0, atr: float = 0.0) -> Tuple[float, Dict]:
        spread_adj = (self.config.spread_bps / 10_000.0) * market_price * 0.5 if self.config.enable_execution_realism else 0.0
        slip_bps = self._execution_cost_bps(requested_size, bar_volume, atr, market_price) if self.config.enable_execution_realism else self.risk_config.slippage_rate * 10_000.0
        slip_adj = (slip_bps / 10_000.0) * market_price
        if action == SignalAction.BUY:
            fill = market_price + spread_adj + slip_adj
        else:
            fill = market_price - spread_adj - slip_adj
        return fill, {"spread_cost": spread_adj, "slippage_cost": slip_adj, "cost_bps": float(self.config.spread_bps * 0.5 + slip_bps)}

    def _exit_fill(self, side: str, exit_price: float, requested_size: float = 0.0, bar_volume: float = 1.0, atr: float = 0.0) -> float:
        spread_adj = (self.config.spread_bps / 10_000.0) * exit_price * 0.5 if self.config.enable_execution_realism else 0.0
        slip_bps = self._execution_cost_bps(requested_size, bar_volume, atr, exit_price) if self.config.enable_execution_realism else self.risk_config.slippage_rate * 10_000.0
        slip_adj = (slip_bps / 10_000.0) * exit_price
        return exit_price - spread_adj - slip_adj if side == SignalAction.BUY.value else exit_price + spread_adj + slip_adj

    def _max_fillable_size(self, row: pd.Series) -> float:
        vol = float(row.get("volume", np.nan))
        if not np.isfinite(vol) or vol <= 0:
            vol = self.config.min_liquidity_threshold * 10.0
        return max(0.0, vol * self.config.max_participation_rate)

    def _build_timeline(self, symbol_dfs: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
        ts = set()
        for df in symbol_dfs.values():
            ts.update(pd.to_datetime(df["timestamp"]).tolist())
        return sorted(ts)

    def _build_returns_matrix(self, symbol_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        close_frames: List[pd.Series] = []
        for sym, df in symbol_dfs.items():
            local = df[["timestamp", "close"]].copy()
            local["timestamp"] = pd.to_datetime(local["timestamp"])
            local = local.drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp").sort_index()
            close_frames.append(local["close"].rename(sym))

        if not close_frames:
            return pd.DataFrame()

        close_matrix = pd.concat(close_frames, axis=1).sort_index()
        return np.log(close_matrix).diff()

    def _corr_matrix_at(self, returns_matrix: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
        if returns_matrix.empty:
            return pd.DataFrame()
        hist = returns_matrix.loc[:pd.Timestamp(ts)].tail(int(self.config.correlation_window))
        if hist.shape[0] < 2:
            return pd.DataFrame(index=returns_matrix.columns, columns=returns_matrix.columns, dtype=float)
        return hist.corr(min_periods=2)

    def _corr_value(self, corr_matrix: pd.DataFrame, a: str, b: str) -> float:
        if a == b:
            return 1.0
        if corr_matrix.empty or a not in corr_matrix.index or b not in corr_matrix.columns:
            return 0.0
        raw = corr_matrix.loc[a, b]
        if pd.isna(raw):
            return 0.0
        raw_f = float(raw)
        return abs(raw_f) if self.config.use_absolute_correlation else raw_f

    def _position_allocation_pct(self, symbol: str, open_positions: Dict[str, SimPosition], prices: Dict[str, float], equity: float) -> float:
        pos = open_positions.get(symbol)
        if pos is None or equity <= 0:
            return 0.0
        px = prices.get(symbol, pos.entry_price)
        return abs(pos.size * px) / equity

    def _open_risk_exposure_pct(self, open_positions: Dict[str, SimPosition], equity: float) -> float:
        if equity <= 0:
            return 0.0
        risk = sum(abs(p.size * p.stop_distance) for p in open_positions.values())
        return risk / equity

    def _group_exposure(self, open_positions: Dict[str, SimPosition], prices: Dict[str, float], equity: float) -> Dict[str, float]:
        out = {}
        if equity <= 0:
            return out
        for sym, pos in open_positions.items():
            md = self.config.symbol_metadata.get(sym, SymbolMetadata(symbol=sym))
            grp = md.group
            px = prices.get(sym, pos.entry_price)
            out[grp] = out.get(grp, 0.0) + abs(pos.size * px) / equity
        return out

    def _rank_candidates(self, candidates: List[Dict]) -> List[Dict]:
        if self.config.enable_correlation_risk:
            for cand in candidates:
                weighted_corr = float(cand.get("corr", {}).get("weighted_corr", 0.0))
                corr_penalty = max(0.0, 1.0 - weighted_corr)
                cand["rank_score"] = float(cand["decision"].expected_value) * corr_penalty
            return sorted(candidates, key=lambda x: x.get("rank_score", 0.0), reverse=True)
        if self.config.candidate_ranking_mode == "highest_confidence":
            return sorted(candidates, key=lambda x: x["decision"].confidence, reverse=True)
        if self.config.candidate_ranking_mode == "highest_risk_adjusted_score":
            return sorted(candidates, key=lambda x: (x["decision"].expected_value / max(1e-9, x["decision"].estimated_cost + 1e-9)), reverse=True)
        if self.config.candidate_ranking_mode == "symbol_order":
            return sorted(candidates, key=lambda x: x["symbol"])
        return sorted(candidates, key=lambda x: x["decision"].expected_value, reverse=True)

    def _candidate_corr_context(
        self,
        symbol: str,
        open_positions: Dict[str, SimPosition],
        current_prices: Dict[str, float],
        equity: float,
        corr_matrix: pd.DataFrame,
    ) -> Dict:
        if not open_positions or equity <= 0:
            return {
                "avg_corr": 0.0,
                "weighted_corr": 0.0,
                "effective_correlated_exposure": 0.0,
                "open_symbols": [],
                "cluster_symbols": [symbol],
                "cluster_exposure": 0.0,
                "cluster_open_positions": 0,
                "corr_by_symbol": {},
            }

        corr_by_symbol: Dict[str, float] = {}
        open_symbols = []
        exposures = []
        for osym, pos in open_positions.items():
            px = current_prices.get(osym, pos.entry_price)
            exp = abs(pos.size * px) / max(equity, 1e-9)
            open_symbols.append(osym)
            exposures.append(exp)
            corr_by_symbol[osym] = self._corr_value(corr_matrix, symbol, osym)

        total_exp = float(sum(exposures))
        corr_vals = [corr_by_symbol[s] for s in open_symbols]
        avg_corr = float(np.mean(corr_vals)) if corr_vals else 0.0
        if total_exp > 0:
            weights = [e / total_exp for e in exposures]
            weighted_corr = float(sum(w * c for w, c in zip(weights, corr_vals)))
        else:
            weighted_corr = 0.0
        effective_corr_exposure = float(sum(e * c for e, c in zip(exposures, corr_vals)))

        symbols = list(open_positions.keys()) + [symbol]
        cluster_symbols = self._cluster_members(symbol, symbols, corr_matrix)
        cluster_open_positions = sum(1 for s in open_positions if s in cluster_symbols)
        cluster_exposure = 0.0
        for csym in cluster_symbols:
            if csym not in open_positions:
                continue
            pos = open_positions[csym]
            px = current_prices.get(csym, pos.entry_price)
            cluster_exposure += abs(pos.size * px) / max(equity, 1e-9)

        return {
            "avg_corr": avg_corr,
            "weighted_corr": weighted_corr,
            "effective_correlated_exposure": effective_corr_exposure,
            "open_symbols": open_symbols,
            "cluster_symbols": sorted(cluster_symbols),
            "cluster_exposure": float(cluster_exposure),
            "cluster_open_positions": int(cluster_open_positions),
            "corr_by_symbol": corr_by_symbol,
        }

    def _cluster_members(self, symbol: str, symbols: List[str], corr_matrix: pd.DataFrame) -> List[str]:
        uniq = sorted(set(symbols))
        adjacency: Dict[str, set] = {s: set() for s in uniq}
        for i, a in enumerate(uniq):
            for b in uniq[i + 1 :]:
                corr = self._corr_value(corr_matrix, a, b)
                if corr >= self.config.cluster_threshold:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        seen = set()
        stack = [symbol]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(list(adjacency.get(cur, set()) - seen))
        return list(seen) if seen else [symbol]

    def _portfolio_utility(self, rows: List[Dict], equity: float) -> Dict[str, float]:
        if not rows:
            return {"utility": 0.0, "ev_component": 0.0, "corr_penalty": 0.0, "drawdown_proxy": 0.0, "concentration_penalty": 0.0, "diversification_bonus": 0.0}
        notionals = np.asarray([max(0.0, r["final_size"] * r["entry_fill"]) for r in rows], dtype=float)
        total_notional = float(notionals.sum())
        if total_notional <= 0:
            return {"utility": 0.0, "ev_component": 0.0, "corr_penalty": 0.0, "drawdown_proxy": 0.0, "concentration_penalty": 0.0, "diversification_bonus": 0.0}

        weights = notionals / max(total_notional, 1e-9)
        ev_component = float(sum(max(0.0, r["adjusted_ev"]) * w for r, w in zip(rows, weights)))
        corr_penalty = float(sum(float(r.get("corr", {}).get("weighted_corr", 0.0)) * w for r, w in zip(rows, weights)))
        drawdown_proxy = float(sum((r["final_size"] * float(r["adjusted_decision"].stop_distance) / max(equity, 1e-9)) * w for r, w in zip(rows, weights)))
        concentration_penalty = float(np.sum(np.square(weights)))
        diversification_bonus = float(1.0 - concentration_penalty)
        utility = (
            self.config.w_ev * ev_component
            - self.config.w_corr * corr_penalty
            - self.config.w_drawdown * drawdown_proxy
            - self.config.w_concentration * concentration_penalty
            + self.config.w_diversification * diversification_bonus
        )
        return {
            "utility": float(utility),
            "ev_component": ev_component,
            "corr_penalty": corr_penalty,
            "drawdown_proxy": drawdown_proxy,
            "concentration_penalty": concentration_penalty,
            "diversification_bonus": diversification_bonus,
        }

    def _evaluate_feasibility(self, rows: List[Dict], equity: float, cash: float) -> Dict:
        binding = []
        max_additional: Dict[str, float] = {}
        feasible = True

        total_notional = 0.0
        total_risk = 0.0
        for r in rows:
            n = max(0.0, r["final_size"] * r["entry_fill"])
            total_notional += n
            total_risk += (r["final_size"] * float(r["adjusted_decision"].stop_distance)) / max(equity, 1e-9)

            if r["final_size"] < 0:
                feasible = False
            if n > self.config.max_symbol_allocation_pct * equity + 1e-12:
                feasible = False
                if "symbol_cap" not in binding:
                    binding.append("symbol_cap")
            corr_ctx = r.get("corr", {})
            cluster_after = float(corr_ctx.get("cluster_exposure", 0.0)) + (n / max(equity, 1e-9))
            if cluster_after > self.config.max_cluster_exposure + 1e-12:
                feasible = False
                if "cluster_cap" not in binding:
                    binding.append("cluster_cap")
            wc = float(corr_ctx.get("weighted_corr", 0.0))
            corr_after = float(corr_ctx.get("effective_correlated_exposure", 0.0)) + (n / max(equity, 1e-9)) * max(wc, 0.0)
            if self.config.enable_correlation_risk and corr_after > self.config.max_correlated_exposure + 1e-12:
                feasible = False
                if "correlated_exposure_cap" not in binding:
                    binding.append("correlated_exposure_cap")

            max_additional[r["symbol"]] = max(
                0.0,
                min(
                    (r["max_size"] - r["final_size"]) * r["entry_fill"],
                    self.config.max_symbol_allocation_pct * equity - n,
                ),
            )

        if total_notional > cash + 1e-12:
            feasible = False
            binding.append("capital_cap")
        if total_risk > self.config.max_total_risk_exposure + 1e-12:
            feasible = False
            binding.append("total_risk_cap")
        if len(rows) > self.config.max_open_positions:
            feasible = False
            binding.append("max_open_positions")

        return {"feasible": feasible, "binding_constraints": binding, "max_additional_allocation_by_candidate": max_additional}

    def _repair_allocation(self, rows: List[Dict], equity: float, cash: float) -> Dict:
        if self.config.use_projection_repair:
            proj = self._projection_repair(rows, equity, cash)
            if proj["status"]["feasible"] or not self.config.fallback_to_simple_repair:
                return proj

        return self._simple_shrink_repair(rows, equity, cash)

    def _project_total_capital(self, rows: List[Dict], cash: float) -> None:
        total_notional = sum(max(0.0, r["final_size"] * r["entry_fill"]) for r in rows)
        if total_notional <= cash:
            return
        scale = cash / max(total_notional, 1e-9)
        for r in rows:
            r["final_size"] *= scale

    def _project_total_risk(self, rows: List[Dict], equity: float) -> None:
        total_risk = sum((r["final_size"] * float(r["adjusted_decision"].stop_distance)) / max(equity, 1e-9) for r in rows)
        if total_risk <= self.config.max_total_risk_exposure:
            return
        scale = self.config.max_total_risk_exposure / max(total_risk, 1e-9)
        for r in rows:
            r["final_size"] *= scale

    def _project_symbol_caps(self, rows: List[Dict], equity: float) -> None:
        for r in rows:
            notional = max(0.0, r["final_size"] * r["entry_fill"])
            cap = self.config.max_symbol_allocation_pct * equity
            if notional > cap:
                r["final_size"] = cap / max(r["entry_fill"], 1e-9)

    def _project_cluster_caps(self, rows: List[Dict], equity: float) -> None:
        for r in rows:
            notional = max(0.0, r["final_size"] * r["entry_fill"])
            corr_ctx = r.get("corr", {})
            cluster_after = float(corr_ctx.get("cluster_exposure", 0.0)) + notional / max(equity, 1e-9)
            if cluster_after > self.config.max_cluster_exposure:
                allowed = max(0.0, self.config.max_cluster_exposure - float(corr_ctx.get("cluster_exposure", 0.0)))
                r["final_size"] = min(r["final_size"], (allowed * equity) / max(r["entry_fill"], 1e-9))

    def _project_correlated_exposure(self, rows: List[Dict], equity: float) -> None:
        if not self.config.enable_correlation_risk:
            return
        for r in rows:
            notional = max(0.0, r["final_size"] * r["entry_fill"])
            corr_ctx = r.get("corr", {})
            wc = max(0.0, float(corr_ctx.get("weighted_corr", 0.0)))
            corr_after = float(corr_ctx.get("effective_correlated_exposure", 0.0)) + (notional / max(equity, 1e-9)) * wc
            if corr_after > self.config.max_correlated_exposure and wc > 0:
                allowed = max(0.0, self.config.max_correlated_exposure - float(corr_ctx.get("effective_correlated_exposure", 0.0)))
                r["final_size"] = min(r["final_size"], (allowed * equity) / max(wc * r["entry_fill"], 1e-9))

    def _projection_repair(self, rows: List[Dict], equity: float, cash: float) -> Dict:
        repair_iterations = 0
        steps_applied: List[str] = []
        for _ in range(max(0, int(self.config.max_projection_iterations))):
            status = self._evaluate_feasibility(rows, equity, cash)
            if status["feasible"]:
                status["projection_steps"] = steps_applied
                return {"rows": rows, "status": status, "repair_iterations": repair_iterations}
            before = np.asarray([r["final_size"] for r in rows], dtype=float)
            self._project_total_capital(rows, cash)
            self._project_total_risk(rows, equity)
            self._project_symbol_caps(rows, equity)
            self._project_cluster_caps(rows, equity)
            self._project_correlated_exposure(rows, equity)
            after = np.asarray([r["final_size"] for r in rows], dtype=float)
            repair_iterations += 1
            steps_applied.append("|".join(status["binding_constraints"]))
            if np.max(np.abs(after - before)) < self.config.projection_tolerance:
                break
        status = self._evaluate_feasibility(rows, equity, cash)
        status["projection_steps"] = steps_applied
        return {"rows": rows, "status": status, "repair_iterations": repair_iterations}

    def _simple_shrink_repair(self, rows: List[Dict], equity: float, cash: float) -> Dict:
        repair_iterations = 0
        for _ in range(max(0, int(self.config.max_repair_iterations))):
            status = self._evaluate_feasibility(rows, equity, cash)
            if status["feasible"]:
                return {"rows": rows, "status": status, "repair_iterations": repair_iterations}
            repair_iterations += 1
            b = status["binding_constraints"]
            if not b:
                break
            if "capital_cap" in b:
                total_notional = sum(max(0.0, r["final_size"] * r["entry_fill"]) for r in rows)
                if total_notional > 0:
                    scale = min(1.0, cash / max(total_notional, 1e-9))
                    for r in rows:
                        r["final_size"] *= scale
            if "total_risk_cap" in b:
                total_risk = sum((r["final_size"] * float(r["adjusted_decision"].stop_distance)) / max(equity, 1e-9) for r in rows)
                if total_risk > 0:
                    scale = min(1.0, self.config.max_total_risk_exposure / max(total_risk, 1e-9))
                    for r in rows:
                        r["final_size"] *= scale
            if any(x in b for x in ["symbol_cap", "cluster_cap", "correlated_exposure_cap"]):
                for r in rows:
                    r["final_size"] *= float(self.config.repair_shrink_factor)

        return {"rows": rows, "status": self._evaluate_feasibility(rows, equity, cash), "repair_iterations": repair_iterations}

    def _optimize_allocations(
        self,
        ts: pd.Timestamp,
        ranked: List[Dict],
        open_positions: Dict[str, SimPosition],
        equity: float,
        cash: float,
    ) -> Dict:
        if not ranked:
            return {"allocated": [], "logs": [], "binding": {}}

        slots_left = max(0, self.config.max_open_positions - len(open_positions))
        working = []
        logs = []
        for cand in ranked:
            dec = cand["adjusted_decision"]
            entry_fill = float(cand["entry_fill"])
            max_size = max(0.0, float(dec.position_size))
            risk_per_unit = float(dec.stop_distance) / max(equity, 1e-9)
            corr_ctx = cand.get("corr", {})
            weighted_corr = float(corr_ctx.get("weighted_corr", 0.0))
            raw_ev = float(cand["decision"].expected_value)
            adjusted_ev = raw_ev * (1.0 - weighted_corr) if self.config.enable_correlation_risk else raw_ev
            score = max(0.0, adjusted_ev) / max(risk_per_unit, 1e-9)
            working.append(
                {
                    **cand,
                    "max_size": max_size,
                    "risk_per_unit": risk_per_unit,
                    "raw_ev": raw_ev,
                    "adjusted_ev": adjusted_ev,
                    "score": score,
                    "entry_fill": entry_fill,
                }
            )

        if slots_left <= 0:
            for w in working:
                logs.append({"timestamp": ts.isoformat(), "symbol": w["symbol"], "decision": "rejected", "reason": "allocation_not_selected"})
            return {"allocated": [], "logs": logs, "binding": {"max_open_positions": True}}

        # Keep top by score if slot-limited.
        working = sorted(working, key=lambda x: (x["score"], x["adjusted_ev"]), reverse=True)
        selected = working[:slots_left]
        dropped = working[slots_left:]
        for d in dropped:
            logs.append({"timestamp": ts.isoformat(), "symbol": d["symbol"], "decision": "rejected", "reason": "allocation_not_selected"})

        total_score = sum(x["score"] for x in selected)
        base_weights = {x["symbol"]: ((x["score"] / total_score) if total_score > 0 else (1.0 / max(len(selected), 1))) for x in selected}

        total_risk_available = max(0.0, self.config.max_total_risk_exposure - self._open_risk_exposure_pct(open_positions, equity))
        binding = {"cash": False, "risk": False, "cluster": False, "correlated_exposure": False, "symbol_allocation": False}
        allocated = []
        for item in selected:
            sym = item["symbol"]
            base_weight = float(base_weights[sym])
            target_notional = cash * base_weight
            base_size = target_notional / max(item["entry_fill"], 1e-9)
            size = min(item["max_size"], base_size)

            # hard caps converted to size caps
            symbol_cap_size = (self.config.max_symbol_allocation_pct * equity) / max(item["entry_fill"], 1e-9)
            if size > symbol_cap_size:
                binding["symbol_allocation"] = True
                size = symbol_cap_size

            corr_ctx = item.get("corr", {})
            weighted_corr = float(corr_ctx.get("weighted_corr", 0.0))
            alloc_pct = (size * item["entry_fill"]) / max(equity, 1e-9)
            cluster_exposure_after = float(corr_ctx.get("cluster_exposure", 0.0)) + alloc_pct
            if cluster_exposure_after > self.config.max_cluster_exposure:
                binding["cluster"] = True
                allowed_alloc = max(0.0, self.config.max_cluster_exposure - float(corr_ctx.get("cluster_exposure", 0.0)))
                size = min(size, (allowed_alloc * equity) / max(item["entry_fill"], 1e-9))

            if self.config.enable_correlation_risk and weighted_corr > 0:
                allowed_extra = max(0.0, self.config.max_correlated_exposure - float(corr_ctx.get("effective_correlated_exposure", 0.0)))
                corr_cap_size = (allowed_extra * equity) / max(weighted_corr * item["entry_fill"], 1e-9)
                if size > corr_cap_size:
                    binding["correlated_exposure"] = True
                    size = max(0.0, corr_cap_size)

            allocated.append(
                {
                    **item,
                    "base_weight": base_weight,
                    "size_pre_global_scale": max(0.0, size),
                }
            )

        total_notional = sum(x["size_pre_global_scale"] * x["entry_fill"] for x in allocated)
        total_inc_risk = sum(x["size_pre_global_scale"] * float(x["adjusted_decision"].stop_distance) / max(equity, 1e-9) for x in allocated)
        cash_scale = min(1.0, cash / max(total_notional, 1e-9)) if total_notional > 0 else 1.0
        risk_scale = min(1.0, total_risk_available / max(total_inc_risk, 1e-9)) if total_inc_risk > 0 else 1.0
        global_scale = min(cash_scale, risk_scale)
        if cash_scale < 1.0:
            binding["cash"] = True
        if risk_scale < 1.0:
            binding["risk"] = True

        for x in allocated:
            x["initial_size"] = x["size_pre_global_scale"] * global_scale
            x["initial_notional"] = x["initial_size"] * x["entry_fill"]
            x["initial_weight"] = x["initial_notional"] / max(cash, 1e-9)
            x["final_size"] = x["initial_size"]
            x["final_weight"] = x["initial_weight"]

        refinement_iterations = 0
        improvement_delta = 0.0
        binding_constraints_seen: List[str] = []
        avg_marginal_utility = 0.0
        concentration_penalty = 0.0
        diversification_bonus = 0.0
        utility_before = self._portfolio_utility(allocated, equity)
        utility_after = utility_before
        repair_iterations = 0
        donor_pool_symbols: List[str] = []
        receiver_pool_symbols: List[str] = []
        projection_steps: List[str] = []
        feasibility_status = "feasible"
        if self.config.enable_iterative_allocation and len(allocated) > 1:
            max_iters = max(0, int(self.config.max_refinement_iterations))
            step_fraction = float(max(0.0, min(0.5, self.config.allocation_step_fraction)))
            min_improve = float(max(0.0, self.config.min_improvement_threshold))
            donor_k = max(1, int(self.config.donor_pool_size))
            receiver_k = max(1, int(self.config.receiver_pool_size))

            for _ in range(max_iters):
                active = [r for r in allocated if r["final_size"] > 0]
                if len(active) < 2:
                    break
                total_alloc_notional = sum(r["final_size"] * r["entry_fill"] for r in active)
                if total_alloc_notional <= 0:
                    break
                for r in active:
                    n = r["final_size"] * r["entry_fill"]
                    r["efficiency"] = r["adjusted_ev"] / max(n, 1e-9)
                    corr_pen = float(r.get("corr", {}).get("weighted_corr", 0.0))
                    concentration_impact = n / max(total_alloc_notional, 1e-9)
                    diversification_factor = max(0.0, 1.0 - corr_pen)
                    feasibility_factor = float(max(0.0, self._evaluate_feasibility([r], equity, cash)["max_additional_allocation_by_candidate"].get(r["symbol"], 0.0)))
                    if self.config.use_delta_marginal_utility:
                        probe_notional = max(self.config.marginal_probe_min_notional, total_alloc_notional * self.config.marginal_probe_fraction)
                        base_u = self._portfolio_utility(active, equity)["utility"]
                        add_rows = [{**x} for x in active]
                        add_map = {x["symbol"]: x for x in add_rows}
                        add_map[r["symbol"]]["final_size"] += probe_notional / max(r["entry_fill"], 1e-9)
                        rem_rows = [{**x} for x in active]
                        rem_map = {x["symbol"]: x for x in rem_rows}
                        rem_map[r["symbol"]]["final_size"] = max(0.0, rem_map[r["symbol"]]["final_size"] - probe_notional / max(r["entry_fill"], 1e-9))
                        utility_delta_add = self._portfolio_utility(add_rows, equity)["utility"] - base_u
                        utility_delta_remove = self._portfolio_utility(rem_rows, equity)["utility"] - base_u
                        r["utility_delta_add"] = float(utility_delta_add)
                        r["utility_delta_remove"] = float(utility_delta_remove)
                        r["marginal_probe_size"] = float(probe_notional)
                        r["marginal_utility"] = float(utility_delta_add - 0.5 * utility_delta_remove + 1e-9 * feasibility_factor)
                    else:
                        r["marginal_utility"] = (
                            self.config.w_ev * max(0.0, r["adjusted_ev"])
                            + self.config.w_diversification * diversification_factor
                            - self.config.w_corr * corr_pen
                            - self.config.w_concentration * concentration_impact
                            + 1e-9 * feasibility_factor
                        )
                        r["utility_delta_add"] = 0.0
                        r["utility_delta_remove"] = 0.0
                        r["marginal_probe_size"] = 0.0

                donors = sorted(active, key=lambda r: (r["efficiency"], r["symbol"]))[:donor_k]
                receivers = sorted(active, key=lambda r: (-r["marginal_utility"], r["symbol"]))[:receiver_k]
                donor_pool_symbols = [d["symbol"] for d in donors]
                receiver_pool_symbols = [r["symbol"] for r in receivers]
                if not donors or not receivers:
                    break
                receiver_scores = np.asarray([max(0.0, r["marginal_utility"]) for r in receivers], dtype=float)
                if receiver_scores.sum() <= 0:
                    break
                receiver_scores = receiver_scores / receiver_scores.sum()

                donor_total_notional = sum(d["final_size"] * d["entry_fill"] for d in donors)
                move_notional_total = min(total_alloc_notional * step_fraction, donor_total_notional * 0.5)
                if move_notional_total <= 0:
                    break

                proposal = [{**r} for r in allocated]
                pmap = {p["symbol"]: p for p in proposal}
                for d in donors:
                    dprop = pmap[d["symbol"]]
                    d_move = move_notional_total * ((d["final_size"] * d["entry_fill"]) / max(donor_total_notional, 1e-9))
                    dprop["final_size"] = max(0.0, dprop["final_size"] - d_move / max(dprop["entry_fill"], 1e-9))

                for idx, rcv in enumerate(receivers):
                    rprop = pmap[rcv["symbol"]]
                    add_notional = move_notional_total * float(receiver_scores[idx])
                    rprop["final_size"] = rprop["final_size"] + add_notional / max(rprop["entry_fill"], 1e-9)

                repair = self._repair_allocation(proposal, equity, cash)
                proposal = repair["rows"]
                repair_iterations = max(repair_iterations, int(repair["repair_iterations"]))
                status = repair["status"]
                binding_constraints_seen.extend(status["binding_constraints"])
                projection_steps.extend(status.get("projection_steps", []))
                feasibility_status = "feasible" if status["feasible"] else "infeasible"
                new_utility = self._portfolio_utility(proposal, equity)
                if status["feasible"] and (new_utility["utility"] - utility_after["utility"]) > min_improve:
                    allocated = proposal
                    refinement_iterations += 1
                    improvement_delta += float(new_utility["utility"] - utility_after["utility"])
                    utility_after = new_utility
                else:
                    break

            for r in allocated:
                r["final_weight"] = (r["final_size"] * r["entry_fill"]) / max(cash, 1e-9)
            if allocated:
                avg_marginal_utility = float(np.mean([r.get("marginal_utility", 0.0) for r in allocated]))
            concentration_penalty = float(utility_after["concentration_penalty"])
            diversification_bonus = float(utility_after["diversification_bonus"])
        else:
            concentration_penalty = float(utility_before["concentration_penalty"])
            diversification_bonus = float(utility_before["diversification_bonus"])

        out = []
        zero_scaled = []
        for x in allocated:
            final_size = x["final_size"]
            final_notional = final_size * x["entry_fill"]
            reason = "allocated" if final_size > 0 else "allocation_scaled_to_zero"
            logs.append(
                {
                    "timestamp": ts.isoformat(),
                    "symbol": x["symbol"],
                    "raw_ev": x["raw_ev"],
                    "adjusted_ev": x["adjusted_ev"],
                    "weighted_corr": float(x.get("corr", {}).get("weighted_corr", 0.0)),
                    "avg_corr": float(x.get("corr", {}).get("avg_corr", 0.0)),
                    "base_weight": x["base_weight"],
                    "final_weight": x["final_weight"],
                    "base_size": float(x["decision"].position_size),
                    "adjusted_size_before_allocation": float(x["adjusted_decision"].position_size),
                    "final_size": final_size,
                    "initial_weight": float(x["initial_weight"]),
                    "refined_weight": float(x["final_weight"]),
                    "initial_size": float(x["initial_size"]),
                    "refined_size": float(final_size),
                    "efficiency_before": float(x["adjusted_ev"] / max(x["initial_notional"], 1e-9)) if x["initial_notional"] > 0 else 0.0,
                    "efficiency_after": float(x["adjusted_ev"] / max(final_notional, 1e-9)) if final_notional > 0 else 0.0,
                    "marginal_utility_before": float(x.get("marginal_utility", 0.0)),
                    "marginal_utility_after": float(x.get("marginal_utility", 0.0)),
                    "marginal_probe_size": float(x.get("marginal_probe_size", 0.0)),
                    "utility_delta_add": float(x.get("utility_delta_add", 0.0)),
                    "utility_delta_remove": float(x.get("utility_delta_remove", 0.0)),
                    "donor_pool": "|".join(sorted(set(donor_pool_symbols))),
                    "receiver_pool": "|".join(sorted(set(receiver_pool_symbols))),
                    "binding_constraints": "|".join(sorted(set(binding_constraints_seen))),
                    "concentration_penalty": concentration_penalty,
                    "diversification_bonus": diversification_bonus,
                    "utility_before": float(utility_before["utility"]),
                    "utility_after": float(utility_after["utility"]),
                    "repair_iterations": repair_iterations,
                    "projection_steps": "|".join(projection_steps),
                    "feasibility_status": feasibility_status if len(binding_constraints_seen) else "feasible",
                    "iteration_count": refinement_iterations,
                    "improvement_delta": float(improvement_delta),
                    "decision": "allocated" if final_size > 0 else "rejected",
                    "reason": reason,
                    "binding_cash": binding["cash"],
                    "binding_risk": binding["risk"],
                    "binding_cluster": binding["cluster"],
                    "binding_correlated_exposure": binding["correlated_exposure"],
                    "binding_symbol_allocation": binding["symbol_allocation"],
                    "scaling_factor": global_scale,
                }
            )
            if final_size > 0:
                out.append(x)
            else:
                zero_scaled.append(x["symbol"])
        return {
            "allocated": out,
            "logs": logs,
            "binding": binding,
            "selected_symbols": [x["symbol"] for x in allocated],
            "zero_scaled_symbols": zero_scaled,
        }

    def calibrate_utility_weights(self, symbol_dfs: Dict[str, pd.DataFrame], strategies: Dict[str, object], output_dir: Optional[str] = None) -> Dict:
        if not self.config.utility_weight_grid:
            return {"recommended": None, "ranking": []}

        keys = sorted(self.config.utility_weight_grid.keys())
        candidates: List[Dict[str, float]] = []
        def _build(idx: int, cur: Dict[str, float]):
            if idx >= len(keys):
                candidates.append(dict(cur))
                return
            k = keys[idx]
            for v in self.config.utility_weight_grid.get(k, []):
                cur[k] = float(v)
                _build(idx + 1, cur)
        _build(0, {})

        ranking = []
        for cand in candidates:
            cfg = replace(self.config, **cand, enable_utility_weight_calibration=False)
            sim = PortfolioStatefulSimulator(self.risk_manager, self.risk_config, cfg)
            # validation-only slice to avoid test leakage (last 30% ignored for calibration)
            val_dfs = {}
            for s, df in symbol_dfs.items():
                n = len(df)
                start = int(n * 0.5)
                end = int(n * 0.8)
                val_dfs[s] = df.iloc[start:end].reset_index(drop=True)
            out = sim.run(val_dfs, strategies)
            m = out["metrics"]
            obj_w = self.config.utility_validation_objective_weights
            composite = (
                obj_w.get("sharpe", 1.0) * m.get("sharpe", 0.0)
                + obj_w.get("return", 0.5) * m.get("total_return", 0.0)
                - obj_w.get("drawdown", 0.5) * m.get("max_drawdown", 0.0)
                - obj_w.get("stability", 0.5) * abs(m.get("average_active_correlation", 0.0))
                - self.config.utility_turnover_penalty * m.get("turnover", 0.0)
                - self.config.utility_avg_corr_penalty * m.get("average_active_correlation", 0.0)
                - self.config.utility_concentration_penalty * (1.0 - m.get("diversification_score", 0.0))
            )
            ranking.append({"weights": cand, "score": float(composite), "metrics": m})

        ranking = sorted(ranking, key=lambda x: x["score"], reverse=True)
        out = {"recommended": ranking[0]["weights"] if ranking else None, "runner_up": ranking[1]["weights"] if len(ranking) > 1 else None, "ranking": ranking}
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for i, r in enumerate(ranking):
                row = {"rank": i + 1, "score": r["score"]}
                row.update({f"w_{k}": v for k, v in r["weights"].items()})
                rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "utility_weight_ranking.csv", index=False)
            sens = {
                "recommended": out["recommended"],
                "runner_up": out["runner_up"],
                "score_variance": float(np.var([r["score"] for r in ranking])) if ranking else 0.0,
                "num_candidates": len(ranking),
            }
            (out_dir / "utility_sensitivity_summary.json").write_text(json.dumps(sens, indent=2))
        return out

    def run_stress_tests(self, symbol_dfs: Dict[str, pd.DataFrame], strategies: Dict[str, object], output_dir: Optional[str] = None) -> Dict:
        base = self.run(symbol_dfs, strategies)
        scenarios = self.config.stress_scenarios or {
            "spread_widening": {"spread_bps": self.config.spread_bps * 2},
            "slippage_increase": {"base_slippage_bps": self.config.base_slippage_bps * 2},
            "volatility_shock": {"slippage_volatility_coefficient": self.config.slippage_volatility_coefficient * 2},
            "correlation_spike": {"w_corr": self.config.w_corr * 1.5},
            "liquidity_reduction": {"max_participation_rate": max(0.01, self.config.max_participation_rate * 0.5)},
            "delayed_execution": {"execution_delay_bars": self.config.execution_delay_bars + 1},
        }
        rows = []
        for name, overrides in scenarios.items():
            cfg = replace(self.config, **overrides)
            sim = PortfolioStatefulSimulator(self.risk_manager, self.risk_config, cfg)
            out = sim.run(symbol_dfs, strategies)
            m = out["metrics"]
            rows.append(
                {
                    "scenario": name,
                    "total_return": m.get("total_return", 0.0),
                    "sharpe": m.get("sharpe", 0.0),
                    "max_drawdown": m.get("max_drawdown", 0.0),
                    "turnover": m.get("turnover", 0.0),
                    "fill_rate": m.get("fill_rate", 0.0),
                    "degradation": base["metrics"].get("total_return", 0.0) - m.get("total_return", 0.0),
                }
            )
        summary = {"base_metrics": base["metrics"], "scenario_count": len(rows), "avg_degradation": float(np.mean([r["degradation"] for r in rows])) if rows else 0.0}
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "stress_scenario_results.csv", index=False)
            (out_dir / "stress_test_summary.json").write_text(json.dumps(summary, indent=2))
        return {"summary": summary, "scenarios": rows}

    def run(self, symbol_dfs: Dict[str, pd.DataFrame], strategies: Dict[str, object]) -> Dict:
        cash = float(self.config.initial_capital)
        realized_pnl = 0.0
        equity = float(self.config.initial_capital)
        peak_equity = equity

        open_positions: Dict[str, SimPosition] = {}
        last_trade_at: Dict[str, datetime] = {}

        trade_log: List[Dict] = []
        rejection_counts: Dict[str, int] = {}
        per_symbol_pnl: Dict[str, float] = {s: 0.0 for s in symbol_dfs}
        per_symbol_trades: Dict[str, int] = {s: 0 for s in symbol_dfs}
        regime_counts: Dict[str, int] = {}

        equity_curve = []
        exposure_timeline = []
        open_positions_timeline = []
        allocation_competition = []
        portfolio_correlation_log: List[Dict] = []
        correlation_rejections: List[Dict] = []
        accepted_entries: List[Dict] = []
        allocation_log: List[Dict] = []
        cluster_limit_hits = {"cluster_position_limit": 0, "cluster_exposure_limit": 0}
        accepted_weighted_corr: List[float] = []
        max_correlated_exposure_observed = 0.0
        capital_utilization_series: List[float] = []
        allocation_efficiency_series: List[float] = []
        diversification_series: List[float] = []
        active_corr_series: List[float] = []
        binding_constraints_per_step: List[float] = []
        concentration_penalty_series: List[float] = []
        diversification_bonus_series: List[float] = []
        accepted_marginal_utility_series: List[float] = []
        requested_fill_count = 0
        filled_count = 0
        partial_fill_count = 0
        spread_cost_total = 0.0
        slippage_cost_total = 0.0

        indices = {s: {pd.Timestamp(t): i for i, t in enumerate(pd.to_datetime(df["timestamp"]))} for s, df in symbol_dfs.items()}
        timeline = self._build_timeline(symbol_dfs)
        returns_matrix = self._build_returns_matrix(symbol_dfs)

        for ts in timeline:
            current_prices = {}
            # exits first
            for sym, pos in list(open_positions.items()):
                idx = indices[sym].get(pd.Timestamp(ts))
                if idx is None:
                    continue
                row = symbol_dfs[sym].iloc[idx]
                high, low, close, open_px = float(row["high"]), float(row["low"]), float(row["close"]), float(row.get("open", row["close"]))
                current_prices[sym] = close

                if pos.side == SignalAction.BUY.value:
                    stop_hit = low <= pos.stop_loss
                    tp_hit = high >= pos.take_profit
                    if stop_hit and tp_hit:
                        exit_price, exit_reason = pos.stop_loss, "stop_and_target_same_bar_stop_first"
                    elif stop_hit:
                        exit_price, exit_reason = pos.stop_loss, "stop_loss"
                    elif tp_hit:
                        exit_price, exit_reason = pos.take_profit, "take_profit"
                    else:
                        continue
                    if self.config.enable_gap_aware_fills and open_px < pos.stop_loss and exit_reason.startswith("stop"):
                        exit_price = min(exit_price, open_px)
                    exit_fill = self._exit_fill(pos.side, exit_price, requested_size=pos.size, bar_volume=float(row.get("volume", np.nan)), atr=float(row.get("atr", 0.0)))
                    entry_notional = pos.entry_price * pos.size
                    exit_notional = exit_fill * pos.size
                    fees = (abs(entry_notional) + abs(exit_notional)) * self.risk_config.fee_rate
                    pnl = (exit_fill - pos.entry_price) * pos.size - fees
                else:
                    stop_hit = high >= pos.stop_loss
                    tp_hit = low <= pos.take_profit
                    if stop_hit and tp_hit:
                        exit_price, exit_reason = pos.stop_loss, "stop_and_target_same_bar_stop_first"
                    elif stop_hit:
                        exit_price, exit_reason = pos.stop_loss, "stop_loss"
                    elif tp_hit:
                        exit_price, exit_reason = pos.take_profit, "take_profit"
                    else:
                        continue
                    if self.config.enable_gap_aware_fills and open_px > pos.stop_loss and exit_reason.startswith("stop"):
                        exit_price = max(exit_price, open_px)
                    exit_fill = self._exit_fill(pos.side, exit_price, requested_size=pos.size, bar_volume=float(row.get("volume", np.nan)), atr=float(row.get("atr", 0.0)))
                    entry_notional = pos.entry_price * pos.size
                    exit_notional = exit_fill * pos.size
                    fees = (abs(entry_notional) + abs(exit_notional)) * self.risk_config.fee_rate
                    pnl = (pos.entry_price - exit_fill) * pos.size - fees

                cash += exit_notional
                realized_pnl += pnl
                per_symbol_pnl[sym] += pnl
                per_symbol_trades[sym] += 1
                trade_log.append({"symbol": sym, "pnl": pnl, "exit_reason": exit_reason, "timestamp": ts.isoformat()})
                last_trade_at[sym] = self._to_dt(ts)
                del open_positions[sym]

            # mark equity
            unrealized = 0.0
            for sym, pos in open_positions.items():
                idx = indices[sym].get(pd.Timestamp(ts))
                if idx is None:
                    continue
                close = float(symbol_dfs[sym].iloc[idx]["close"])
                current_prices[sym] = close
                unrealized += (close - pos.entry_price) * pos.size if pos.side == SignalAction.BUY.value else (pos.entry_price - close) * pos.size

            equity = cash + unrealized
            peak_equity = max(peak_equity, equity)

            # generate candidates
            corr_matrix = self._corr_matrix_at(returns_matrix, ts)
            candidates = []
            for sym, df in symbol_dfs.items():
                idx = indices[sym].get(pd.Timestamp(ts))
                if idx is None:
                    continue
                hist = df.iloc[: idx + 1]
                row = hist.iloc[-1]
                signal = strategies[sym].generate(hist)
                regime_counts[signal.regime] = regime_counts.get(signal.regime, 0) + 1

                drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                decision = self.risk_manager.evaluate(
                    signal=signal,
                    latest_price=float(row["close"]),
                    atr=float(row["atr"]),
                    account_balance=equity,
                    open_positions_count=len(open_positions),
                    has_open_position_for_symbol=(sym in open_positions if self.config.one_position_per_symbol else False),
                    last_trade_at=last_trade_at.get(sym),
                    now=self._to_dt(ts),
                    atr_mean=float(hist["atr"].rolling(50, min_periods=1).mean().iloc[-1]),
                    signal_timestamp=signal.timestamp,
                    current_drawdown=drawdown,
                    open_risk_exposure_pct=self._open_risk_exposure_pct(open_positions, equity),
                    available_cash=cash,
                    symbol_allocation_pct=self._position_allocation_pct(sym, open_positions, current_prices, equity),
                )

                if not decision.approved:
                    rejection_counts[decision.reason] = rejection_counts.get(decision.reason, 0) + 1
                    continue
                corr_ctx = self._candidate_corr_context(sym, open_positions, current_prices, equity, corr_matrix)
                candidates.append(
                    {
                        "symbol": sym,
                        "signal": signal,
                        "decision": decision,
                        "idx": idx,
                        "row": row,
                        "corr": corr_ctx,
                        "candidate_meta": {
                            "expected_value": float(decision.expected_value),
                            "calibrated_probability": float(signal.calibrated_probability or signal.probability),
                            "expected_rr": float(decision.expected_rr),
                            "adjusted_risk_pct": float(decision.effective_risk_pct),
                            "min_qty": self.risk_config.min_qty,
                            "min_notional": self.risk_config.min_notional,
                            "symbol_group": self.config.symbol_metadata.get(sym, SymbolMetadata(symbol=sym)).group,
                        },
                    }
                )

            ranked = self._rank_candidates(candidates)
            accepted = []
            alloc_ready = []
            for cand in ranked:
                sym = cand["symbol"]
                dec = cand["decision"]
                corr_ctx = cand.get("corr", {})
                row = cand["row"]
                close = float(row["close"])

                avg_corr = float(corr_ctx.get("avg_corr", 0.0))
                weighted_corr = float(corr_ctx.get("weighted_corr", 0.0))
                effective_corr_exposure = float(corr_ctx.get("effective_correlated_exposure", 0.0))
                max_correlated_exposure_observed = max(max_correlated_exposure_observed, effective_corr_exposure)

                corr_excess = max(0.0, weighted_corr - self.config.correlation_threshold) if self.config.enable_correlation_risk else 0.0
                penalty_factor = max(0.0, 1.0 - self.config.correlation_penalty_strength * corr_excess)
                adjusted_risk_pct = (
                    float(dec.effective_risk_pct * penalty_factor)
                    if self.config.enable_correlation_risk and self.config.enable_correlation_scaling
                    else float(dec.effective_risk_pct)
                )
                scaling_ratio = adjusted_risk_pct / max(dec.effective_risk_pct, 1e-12)
                adjusted_size = float(dec.position_size * scaling_ratio)
                adjusted_dec = replace(
                    dec,
                    effective_risk_pct=adjusted_risk_pct,
                    dynamic_risk_pct=adjusted_risk_pct,
                    position_size=adjusted_size,
                    notional=adjusted_size * close,
                )

                corr_log_row = {
                    "timestamp": ts.isoformat(),
                    "symbol": sym,
                    "open_symbols": "|".join(corr_ctx.get("open_symbols", [])),
                    "avg_corr": avg_corr,
                    "weighted_corr": weighted_corr,
                    "effective_correlated_exposure": effective_corr_exposure,
                    "base_risk_pct": float(dec.effective_risk_pct),
                    "adjusted_risk_pct": adjusted_risk_pct,
                    "cluster_symbols": "|".join(corr_ctx.get("cluster_symbols", [])),
                    "cluster_open_positions": int(corr_ctx.get("cluster_open_positions", 0)),
                    "cluster_exposure": float(corr_ctx.get("cluster_exposure", 0.0)),
                    "accepted": False,
                    "rejection_reason": "",
                }

                if self.config.enable_correlation_risk and self.config.enable_correlation_rejection and weighted_corr > self.config.correlation_threshold:
                    rejection_counts["correlation_exceeded"] = rejection_counts.get("correlation_exceeded", 0) + 1
                    corr_log_row["rejection_reason"] = "correlation_exceeded"
                    correlation_rejections.append(corr_log_row.copy())
                    portfolio_correlation_log.append(corr_log_row)
                    continue

                if self.config.enable_correlation_risk and self.config.enable_correlation_rejection and effective_corr_exposure > self.config.max_correlated_exposure:
                    rejection_counts["correlated_exposure_limit"] = rejection_counts.get("correlated_exposure_limit", 0) + 1
                    corr_log_row["rejection_reason"] = "correlated_exposure_limit"
                    correlation_rejections.append(corr_log_row.copy())
                    portfolio_correlation_log.append(corr_log_row)
                    continue

                cluster_positions_after = int(corr_ctx.get("cluster_open_positions", 0)) + 1
                if cluster_positions_after > self.config.max_cluster_positions:
                    rejection_counts["cluster_position_limit"] = rejection_counts.get("cluster_position_limit", 0) + 1
                    cluster_limit_hits["cluster_position_limit"] += 1
                    corr_log_row["rejection_reason"] = "cluster_position_limit"
                    correlation_rejections.append(corr_log_row.copy())
                    portfolio_correlation_log.append(corr_log_row)
                    continue
                alloc_ready.append(
                    {
                        **cand,
                        "adjusted_decision": adjusted_dec,
                        "entry_fill": close,
                        "corr_log_row": corr_log_row,
                    }
                )

            alloc_result = self._optimize_allocations(ts, alloc_ready, open_positions, equity, cash)
            allocation_log.extend(alloc_result["logs"])
            step_logs = [x for x in alloc_result["logs"] if x.get("timestamp") == ts.isoformat()]
            if step_logs:
                binding_constraints_per_step.append(float(np.mean([len(str(x.get("binding_constraints", "")).split("|")) if x.get("binding_constraints") else 0 for x in step_logs])))
                concentration_penalty_series.append(float(np.mean([float(x.get("concentration_penalty", 0.0)) for x in step_logs])))
                diversification_bonus_series.append(float(np.mean([float(x.get("diversification_bonus", 0.0)) for x in step_logs])))
                accepted_marginal_utility_series.extend([float(x.get("marginal_utility_after", 0.0)) for x in step_logs if x.get("decision") == "allocated"])
            if alloc_result["binding"].get("cluster"):
                cluster_limit_hits["cluster_exposure_limit"] += 1
            selected_symbols = set(alloc_result.get("selected_symbols", []))
            zero_scaled_symbols = set(alloc_result.get("zero_scaled_symbols", []))

            for cand in alloc_ready:
                if cand["symbol"] in zero_scaled_symbols:
                    rejection_counts["allocation_scaled_to_zero"] = rejection_counts.get("allocation_scaled_to_zero", 0) + 1
                    row = cand["corr_log_row"]
                    row["rejection_reason"] = "allocation_scaled_to_zero"
                    portfolio_correlation_log.append(row)
                elif cand["symbol"] not in selected_symbols:
                    rejection_counts["allocation_not_selected"] = rejection_counts.get("allocation_not_selected", 0) + 1
                    row = cand["corr_log_row"]
                    row["rejection_reason"] = "allocation_not_selected"
                    portfolio_correlation_log.append(row)

            for item in alloc_result["allocated"]:
                requested_fill_count += 1
                sym = item["symbol"]
                adjusted_dec = item["adjusted_decision"]
                final_size = float(item["final_size"])
                exec_idx = int(item.get("idx", 0)) + int(self.config.execution_delay_bars)
                sym_df = symbol_dfs[sym]
                if exec_idx >= len(sym_df):
                    rejection_counts["stale_signal_rejection"] = rejection_counts.get("stale_signal_rejection", 0) + 1
                    row = item["corr_log_row"]
                    row["rejection_reason"] = "stale_signal_rejection"
                    portfolio_correlation_log.append(row)
                    continue
                if int(self.config.execution_delay_bars) > int(self.config.stale_signal_bars):
                    rejection_counts["stale_signal_rejection"] = rejection_counts.get("stale_signal_rejection", 0) + 1
                    row = item["corr_log_row"]
                    row["rejection_reason"] = "stale_signal_rejection"
                    portfolio_correlation_log.append(row)
                    continue

                exec_row = sym_df.iloc[exec_idx]
                px = float(exec_row.get("open", exec_row["close"])) if self.config.execution_delay_bars > 0 else float(exec_row["close"])
                max_fillable = self._max_fillable_size(exec_row)
                fill_size = min(final_size, max_fillable)
                if fill_size <= 0:
                    rejection_counts["liquidity_rejection"] = rejection_counts.get("liquidity_rejection", 0) + 1
                    row = item["corr_log_row"]
                    row["rejection_reason"] = "liquidity_rejection"
                    portfolio_correlation_log.append(row)
                    continue
                is_partial = fill_size < final_size
                filled_count += 1
                if is_partial:
                    partial_fill_count += 1
                entry_fill, fill_diag = self._entry_fill(
                    item["signal"].action,
                    px,
                    requested_size=fill_size,
                    bar_volume=float(exec_row.get("volume", np.nan)),
                    atr=float(exec_row.get("atr", 0.0)),
                )
                if self.config.spread_bps > self.config.max_spread_bps_for_entry:
                    rejection_counts["spread_too_wide"] = rejection_counts.get("spread_too_wide", 0) + 1
                    continue
                if fill_diag["cost_bps"] > self.config.max_fill_cost_bps:
                    rejection_counts["fill_quality_too_poor"] = rejection_counts.get("fill_quality_too_poor", 0) + 1
                    continue
                spread_cost_total += float(fill_diag["spread_cost"])
                slippage_cost_total += float(fill_diag["slippage_cost"])
                notional = abs(entry_fill * fill_size)
                alloc = notional / max(equity, 1e-9)

                if final_size < self.risk_config.min_qty or notional < self.risk_config.min_notional:
                    rejection_counts["allocation_scaled_to_zero"] = rejection_counts.get("allocation_scaled_to_zero", 0) + 1
                    row = item["corr_log_row"]
                    row["rejection_reason"] = "allocation_scaled_to_zero"
                    portfolio_correlation_log.append(row)
                    continue

                if self.config.enable_group_constraints:
                    md = self.config.symbol_metadata.get(sym, SymbolMetadata(symbol=sym))
                    grp = md.group
                    grp_exp = self._group_exposure(open_positions, current_prices, equity)
                    if grp_exp.get(grp, 0.0) + alloc > self.config.group_exposure_limits.get(grp, 1.0):
                        rejection_counts["group_exposure_limit"] = rejection_counts.get("group_exposure_limit", 0) + 1
                        row = item["corr_log_row"]
                        row["rejection_reason"] = "group_exposure_limit"
                        portfolio_correlation_log.append(row)
                        continue

                open_positions[sym] = SimPosition(
                    symbol=sym,
                    side=item["signal"].action.value,
                    entry_price=entry_fill,
                    size=fill_size,
                    stop_loss=adjusted_dec.stop_loss,
                    take_profit=adjusted_dec.take_profit,
                    stop_distance=adjusted_dec.stop_distance,
                    take_profit_distance=adjusted_dec.take_profit_distance,
                    entry_time=self._to_dt(ts),
                )
                cash -= notional
                accepted.append(sym)
                weighted_corr = float(item.get("corr", {}).get("weighted_corr", 0.0))
                effective_corr_exposure = float(item.get("corr", {}).get("effective_correlated_exposure", 0.0))
                accepted_weighted_corr.append(weighted_corr)
                accepted_entries.append(
                    {
                        "timestamp": ts.isoformat(),
                        "symbol": sym,
                        "base_position_size": float(item["decision"].position_size),
                        "adjusted_position_size": float(item["adjusted_decision"].position_size),
                        "allocated_position_size": final_size,
                        "partial_fill": bool(is_partial),
                        "fill_cost_bps": float(fill_diag["cost_bps"]),
                        "base_risk_pct": float(item["decision"].effective_risk_pct),
                        "adjusted_risk_pct": float(item["adjusted_decision"].effective_risk_pct),
                        "weighted_corr": weighted_corr,
                        "effective_correlated_exposure": effective_corr_exposure,
                        "allocation_weight": float(item["final_weight"]),
                    }
                )
                row = item["corr_log_row"]
                row["accepted"] = True
                portfolio_correlation_log.append(row)

            # allocation-level diagnostics
            if ranked:
                achieved = sum(max(0.0, r.get("adjusted_ev", 0.0)) * r.get("final_weight", 0.0) for r in alloc_result["allocated"])
                theoretical = max([max(0.0, r.get("adjusted_ev", 0.0)) for r in alloc_result["allocated"]] + [0.0])
                allocation_efficiency_series.append(achieved / max(theoretical, 1e-9) if theoretical > 0 else 0.0)
                capital_utilization_series.append(sum(r.get("final_weight", 0.0) for r in alloc_result["allocated"]))

            if self.config.enable_allocation_competition_logs and len(ranked) > 1:
                allocation_competition.append(
                    {
                        "timestamp": ts.isoformat(),
                        "candidates": [x["symbol"] for x in ranked],
                        "accepted": accepted,
                        "rejected": [x["symbol"] for x in ranked if x["symbol"] not in accepted],
                    }
                )

            equity_curve.append({"timestamp": ts.isoformat(), "cash": cash, "equity": equity, "realized_pnl": realized_pnl, "unrealized_pnl": unrealized})
            exposure_timeline.append({"timestamp": ts.isoformat(), "open_risk_exposure_pct": self._open_risk_exposure_pct(open_positions, equity)})
            open_positions_timeline.append({"timestamp": ts.isoformat(), "open_positions": len(open_positions)})
            if open_positions:
                notionals = []
                syms = list(open_positions.keys())
                for sym, pos in open_positions.items():
                    px = current_prices.get(sym, pos.entry_price)
                    notionals.append(abs(pos.size * px))
                weights = np.asarray(notionals, dtype=float)
                weights = weights / max(weights.sum(), 1e-9)
                diversification_series.append(float(1.0 - np.sum(np.square(weights))))

                pair_corr = []
                for i, a in enumerate(syms):
                    for b in syms[i + 1 :]:
                        pair_corr.append(self._corr_value(corr_matrix, a, b))
                active_corr_series.append(float(np.mean(pair_corr)) if pair_corr else 0.0)
            else:
                diversification_series.append(0.0)
                active_corr_series.append(0.0)

        eq_values = np.asarray([x["equity"] for x in equity_curve], dtype=float)
        returns = np.diff(eq_values) / np.maximum(eq_values[:-1], 1e-9) if len(eq_values) > 1 else np.asarray([])
        sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(len(returns))) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0
        sortino = float((returns.mean() / returns[returns < 0].std(ddof=1)) * np.sqrt(len(returns))) if len(returns[returns < 0]) > 1 and returns[returns < 0].std(ddof=1) > 0 else 0.0
        running_max = np.maximum.accumulate(eq_values) if len(eq_values) else np.asarray([])
        drawdowns = (running_max - eq_values) / np.maximum(running_max, 1e-9) if len(eq_values) else np.asarray([])
        max_dd = float(np.max(drawdowns)) if len(drawdowns) else 0.0
        total_return = float((eq_values[-1] - self.config.initial_capital) / self.config.initial_capital) if len(eq_values) else 0.0

        pnl_array = np.asarray([t["pnl"] for t in trade_log], dtype=float) if trade_log else np.asarray([])
        wins = pnl_array[pnl_array > 0]
        losses = pnl_array[pnl_array < 0]
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) else (float("inf") if len(wins) else 0.0)

        if self.config.diagnostics_output_dir:
            out_dir = Path(self.config.diagnostics_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(portfolio_correlation_log).to_csv(out_dir / "portfolio_correlation_log.csv", index=False)
            pd.DataFrame(correlation_rejections).to_csv(out_dir / "correlation_rejections.csv", index=False)
            pd.DataFrame(allocation_log).to_csv(out_dir / "portfolio_allocation_log.csv", index=False)

        return {
            "metrics": {
                "total_return": total_return,
                "annualized_return": total_return,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": max_dd,
                "calmar": (total_return / max_dd) if max_dd > 0 else 0.0,
                "win_rate": float(len(wins) / len(pnl_array)) if len(pnl_array) else 0.0,
                "profit_factor": float(profit_factor if np.isfinite(profit_factor) else 10.0),
                "expectancy": float(pnl_array.mean()) if len(pnl_array) else 0.0,
                "trade_count": int(len(pnl_array)),
                "avg_open_positions": float(np.mean([x["open_positions"] for x in open_positions_timeline])) if open_positions_timeline else 0.0,
                "exposure_pct_mean": float(np.mean([x["open_risk_exposure_pct"] for x in exposure_timeline])) if exposure_timeline else 0.0,
                "turnover": float(sum(abs(t["pnl"]) for t in trade_log) / max(self.config.initial_capital, 1e-9)) if trade_log else 0.0,
                "correlation_rejections": int(
                    rejection_counts.get("correlation_exceeded", 0) + rejection_counts.get("correlated_exposure_limit", 0)
                ),
                "avg_accepted_weighted_correlation": float(np.mean(accepted_weighted_corr)) if accepted_weighted_corr else 0.0,
                "max_correlated_exposure_observed": float(max_correlated_exposure_observed),
                "cluster_position_limit_hits": int(cluster_limit_hits["cluster_position_limit"]),
                "cluster_exposure_limit_hits": int(cluster_limit_hits["cluster_exposure_limit"]),
                "capital_utilization_pct": float(np.mean(capital_utilization_series)) if capital_utilization_series else 0.0,
                "allocation_efficiency": float(np.mean(allocation_efficiency_series)) if allocation_efficiency_series else 0.0,
                "diversification_score": float(np.mean(diversification_series)) if diversification_series else 0.0,
                "average_active_correlation": float(np.mean(active_corr_series)) if active_corr_series else 0.0,
                "avg_binding_constraints_per_step": float(np.mean(binding_constraints_per_step)) if binding_constraints_per_step else 0.0,
                "avg_concentration_penalty": float(np.mean(concentration_penalty_series)) if concentration_penalty_series else 0.0,
                "avg_diversification_bonus": float(np.mean(diversification_bonus_series)) if diversification_bonus_series else 0.0,
                "avg_marginal_utility_accepted": float(np.mean(accepted_marginal_utility_series)) if accepted_marginal_utility_series else 0.0,
                "fill_rate": float(filled_count / max(requested_fill_count, 1)),
                "partial_fill_rate": float(partial_fill_count / max(filled_count, 1)),
                "average_slippage_cost": float(slippage_cost_total / max(filled_count, 1)),
                "average_spread_cost": float(spread_cost_total / max(filled_count, 1)),
                "stale_signal_rejection_count": int(rejection_counts.get("stale_signal_rejection", 0)),
                "liquidity_rejection_count": int(rejection_counts.get("liquidity_rejection", 0)),
                "stress_degradation_score": 0.0,
            },
            "equity_curve": equity_curve,
            "exposure_timeline": exposure_timeline,
            "open_positions_timeline": open_positions_timeline,
            "trade_log": trade_log,
            "rejections": rejection_counts,
            "allocation_competition": allocation_competition,
            "portfolio_correlation_log": portfolio_correlation_log,
            "correlation_rejections": correlation_rejections,
            "accepted_entries": accepted_entries,
            "portfolio_allocation_log": allocation_log,
            "per_symbol_metrics": [
                {"symbol": s, "pnl": per_symbol_pnl.get(s, 0.0), "trade_count": per_symbol_trades.get(s, 0)} for s in symbol_dfs
            ],
            "regimes": regime_counts,
        }
