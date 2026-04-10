from __future__ import annotations

import asyncio
import contextlib
from typing import Optional

import ccxt
import pandas as pd

from src.config.schema import AppConfig
from src.db import create_db_and_tables
from src.execution import PaperExecutor
from src.features import add_indicators
from src.governance import (
    ChallengerShadowStats,
    GovernanceEvaluator,
    ModelRegistryService,
    ModelSelector,
    PromotionManager,
)
from src.monitoring.drift import DriftObservation, DriftService
from src.monitoring.performance import PerformanceService
from src.paper_engine import PaperTradingEngine
from src.risk import RiskConfig, RiskManager
from src.runtime.backends.base import RuntimeBackend
from src.runtime.backends.live_backend import LiveBackend
from src.runtime.backends.paper_backend import PaperBackend
from src.runtime.backends.shadow_backend import ShadowBackend
from src.runtime.mode import RuntimeMode


class RuntimeController:
    def __init__(self, config: AppConfig):
        self.config = config
        self.symbol = config.runtime.symbols[0]
        self.timeframe = config.runtime.timeframe
        self.max_buffer = config.runtime.max_buffer
        self.poll_interval = config.runtime.poll_interval

        self.exchange = self._build_exchange()
        create_db_and_tables()

        self.registry = ModelRegistryService()
        self.registry.ensure_default_champion(
            model_version=self.config.governance.champion_model_version,
            model_path=self.config.strategy.model_path,
            calibrator_path=self.config.strategy.calibrator_path,
        )
        self.selector = ModelSelector(config, self.registry, symbol=self.symbol, timeframe=self.timeframe)
        self.evaluator = GovernanceEvaluator()
        self.promotion_manager = PromotionManager(self.registry, self.evaluator)

        self.backend = self._build_backend()
        self.challenger_strategy = self.selector.load_challenger()
        self.challenger_stats = ChallengerShadowStats()
        self.drift_service = DriftService(
            feature_threshold=self.config.monitoring.feature_drift_threshold,
            signal_threshold=self.config.monitoring.signal_drift_threshold,
            execution_threshold=self.config.monitoring.execution_drift_threshold,
            baseline_window=self.config.monitoring.performance_window_trades,
            min_samples=max(20, self.config.monitoring.performance_window_trades // 4),
        )
        self.performance_service = PerformanceService(
            model_version=self.config.governance.champion_model_version,
            window_size=self.config.monitoring.performance_window_trades,
            snapshot_interval_bars=1,
        )

        self.candle_buffer = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        self._polling_task: Optional[asyncio.Task] = None

    def _build_exchange(self):
        exchange_cls = getattr(ccxt, self.config.data.exchange_id, ccxt.bybit)
        return exchange_cls(
            {
                "enableRateLimit": self.config.data.enable_rate_limit,
                "timeout": self.config.data.timeout_ms,
                "options": {
                    "defaultType": self.config.data.default_type,
                    "adjustForTimeDifference": self.config.data.adjust_for_time_difference,
                },
            }
        )

    def _build_risk_manager(self):
        risk_cfg = RiskConfig(
            max_risk_per_trade=self.config.risk.max_risk_per_trade,
            max_concurrent_positions=self.config.risk.max_concurrent_positions,
            cooldown_minutes=self.config.risk.cooldown_minutes,
            min_confidence=self.config.risk.min_confidence,
            confidence_risk_floor=self.config.risk.confidence_risk_floor,
            confidence_risk_ceiling=self.config.risk.confidence_risk_ceiling,
            min_expected_value=self.config.risk.min_expected_value,
            atr_stop_mult=self.config.risk.atr_stop_mult,
            atr_tp_mult=self.config.risk.atr_tp_mult,
            min_rr=self.config.risk.min_rr,
            max_atr_vol_mult=self.config.risk.max_atr_vol_mult,
            atr_risk_cut_mult=self.config.risk.atr_risk_cut_mult,
            atr_risk_multiplier=self.config.risk.atr_risk_multiplier,
            fee_rate=self.config.execution.fee_rate,
            slippage_rate=self.config.execution.slippage_rate,
            min_qty=self.config.risk.min_qty,
            min_notional=self.config.risk.min_notional,
            qty_precision=self.config.risk.qty_precision,
            price_precision=self.config.risk.price_precision,
            stale_signal_minutes=self.config.risk.stale_signal_minutes,
            drawdown_threshold=self.config.risk.drawdown_threshold,
            drawdown_risk_multiplier=self.config.risk.drawdown_risk_multiplier,
            max_total_risk_exposure=self.config.risk.max_total_risk_exposure,
        )
        return RiskManager(risk_cfg), risk_cfg

    def _build_backend(self) -> RuntimeBackend:
        mode = RuntimeMode.from_value(self.config.runtime.mode)
        strategy = self.selector.load_champion()
        risk_manager, risk_cfg = self._build_risk_manager()

        if mode == RuntimeMode.PAPER:
            paper_engine = PaperTradingEngine(
                starting_balance=10_000.0,
                fee_rate=risk_cfg.fee_rate,
                slippage_rate=risk_cfg.slippage_rate,
            )
            executor = PaperExecutor(engine=paper_engine)
            return PaperBackend(
                symbol=self.symbol,
                timeframe=self.timeframe,
                strategy=strategy,
                risk_manager=risk_manager,
                paper_engine=paper_engine,
                executor=executor,
            )

        if mode == RuntimeMode.SHADOW_LIVE:
            return ShadowBackend(
                symbol=self.symbol,
                timeframe=self.timeframe,
                strategy=strategy,
                risk_manager=risk_manager,
            )

        return LiveBackend()

    @staticmethod
    def ohlcv_to_df(candles):
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Africa/Lagos")
        df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        return df

    def load_historical_data(self) -> None:
        candles = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.max_buffer)
        self.candle_buffer = self.ohlcv_to_df(candles)

    def process_buffer(self) -> None:
        self.backend.on_new_bar(self.candle_buffer.copy())
        obs_payload = self.backend.drift_observation()
        if obs_payload:
            obs = DriftObservation(
                timestamp=obs_payload["timestamp"],
                symbol=obs_payload["symbol"],
                feature_values=obs_payload.get("feature_values", []),
                signal_values=obs_payload.get("signal_values", []),
                execution_values=obs_payload.get("execution_values", []),
            )
            self.drift_service.process(obs)
        self.performance_service.update()
        self._update_challenger_shadow()
        self._run_governance_loop()

    def _update_challenger_shadow(self) -> None:
        if self.challenger_strategy is None:
            return
        if len(self.candle_buffer) < 30:
            return
        df = add_indicators(self.candle_buffer.copy()).dropna().reset_index(drop=True)
        if len(df) == 0:
            return
        signal = self.challenger_strategy.generate(df)
        prev = self.challenger_stats.avg_expected_value * self.challenger_stats.samples
        self.challenger_stats.samples += 1
        self.challenger_stats.avg_expected_value = (prev + float(signal.confidence)) / self.challenger_stats.samples

    def _run_governance_loop(self) -> None:
        if self.config.governance.model_selection_mode != "champion_challenger":
            return
        challenger_version = self.config.governance.challenger_model_version
        if not challenger_version:
            return

        latest_perf = self.performance_service.latest()
        champion_status = latest_perf.status if latest_perf is not None else "healthy"
        champion_expectancy = float(latest_perf.expectancy) if latest_perf is not None else 0.0

        latest_drift = self.drift_service.latest(symbol=self.symbol)
        drift_alert = any(
            (d.get("status") == "alert")
            for d in latest_drift.get("dimensions", {}).values()
        )

        reason = self.promotion_manager.maybe_promote(
            challenger_version=challenger_version,
            champion_status=champion_status,
            champion_expectancy=champion_expectancy,
            drift_alert=drift_alert,
            challenger_stats=self.challenger_stats,
        )
        if reason:
            self.backend.strategy = self.selector.load_champion()
            self.challenger_strategy = self.selector.load_challenger()
            self.challenger_stats = ChallengerShadowStats()

    async def polling_loop(self) -> None:
        last_timestamp = self.candle_buffer.iloc[-1]["timestamp"]
        while True:
            try:
                candles = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.max_buffer)
                df = self.ohlcv_to_df(candles)
                if len(df) < 3:
                    await asyncio.sleep(self.poll_interval)
                    continue

                closed_df = df.iloc[:-1]
                new_df = closed_df[closed_df["timestamp"] > last_timestamp].copy()
                if len(new_df) > 0:
                    self.candle_buffer = pd.concat([self.candle_buffer, new_df], ignore_index=True)
                    self.candle_buffer = self.candle_buffer.drop_duplicates(subset=["timestamp"], keep="last")
                    self.candle_buffer = self.candle_buffer.sort_values("timestamp").iloc[-self.max_buffer :].reset_index(drop=True)
                    self.process_buffer()
                    last_timestamp = self.candle_buffer.iloc[-1]["timestamp"]
            except Exception:
                pass

            await asyncio.sleep(self.poll_interval)

    async def startup(self) -> None:
        create_db_and_tables()
        self.load_historical_data()
        self.backend.startup()
        self.process_buffer()
        self._polling_task = asyncio.create_task(self.polling_loop())

    async def shutdown(self) -> None:
        if self._polling_task is not None:
            self._polling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._polling_task
        self.backend.shutdown()

    def latest_signal(self):
        return self.backend.latest_signal()

    def health(self):
        return self.backend.health(len(self.candle_buffer))

    def metrics(self):
        return self.backend.metrics()

    def drift_latest(self, symbol: Optional[str] = None):
        return self.drift_service.latest(symbol=symbol)

    def drift_events(self, symbol: Optional[str] = None, dimension: Optional[str] = None, limit: int = 100):
        return self.drift_service.events(symbol=symbol, dimension=dimension, limit=limit)

    def performance_latest(self, model_version: Optional[str] = None):
        return self.performance_service.latest(model_version=model_version)

    def performance_history(self, model_version: Optional[str] = None, limit: int = 100):
        return self.performance_service.history(model_version=model_version, limit=limit)

    def model_registry(self):
        return self.registry.list_entries()

    def model_promotions(self, limit: int = 100):
        return self.registry.promotions(limit=limit)

    def manual_promote(self, new_model_version: str, reason: str = "manual_promotion"):
        event = self.registry.promote(new_model_version=new_model_version, reason=reason)
        if event is not None:
            self.backend.strategy = self.selector.load_champion()
            self.challenger_strategy = self.selector.load_challenger()
            self.challenger_stats = ChallengerShadowStats()
        return event
