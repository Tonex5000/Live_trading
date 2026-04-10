import logging

from src.execution.base import ExecutionResult, Executor
from src.paper_engine import PaperTradingEngine
from src.risk import RiskDecision
from src.strategies.signal_schema import StrategySignal

logger = logging.getLogger(__name__)


class PaperExecutor(Executor):
    def __init__(self, engine: PaperTradingEngine):
        self.engine = engine

    def execute(self, signal: StrategySignal, risk: RiskDecision, market_price: float, timestamp: str) -> ExecutionResult:
        if not risk.approved:
            logger.info(
                "trade_rejected",
                extra={
                    "symbol": signal.symbol,
                    "action": signal.action.value,
                    "reason": risk.reason,
                    "timestamp": timestamp,
                },
            )
            return ExecutionResult(executed=False, reason=risk.reason)

        position_id = self.engine.open_trade(
            symbol=signal.symbol,
            action=signal.action,
            market_price=market_price,
            size=risk.position_size,
            stop_loss=risk.stop_loss,
            take_profit=risk.take_profit,
        )

        logger.info(
            "trade_executed",
            extra={
                "symbol": signal.symbol,
                "action": signal.action.value,
                "position_id": position_id,
                "size": risk.position_size,
                "stop_loss": risk.stop_loss,
                "take_profit": risk.take_profit,
                "timestamp": timestamp,
            },
        )

        return ExecutionResult(executed=True, reason="paper_trade_opened", order_id=str(position_id))
