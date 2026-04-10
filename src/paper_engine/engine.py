from datetime import datetime
from typing import Dict, List, Optional

from sqlmodel import select

from src.db import session_scope
from src.models import EquitySnapshot, Position, Trade
from src.risk import RiskDecision
from src.strategies.signal_schema import SignalAction


class PaperTradingEngine:
    def __init__(self, starting_balance: float = 10000.0, fee_rate: float = 0.0006, slippage_rate: float = 0.0004):
        self.starting_balance = starting_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def _get_or_create_balance(self) -> float:
        with session_scope() as session:
            latest = session.exec(select(EquitySnapshot).order_by(EquitySnapshot.timestamp.desc())).first()
            if latest:
                return float(latest.balance)

            snap = EquitySnapshot(
                balance=float(self.starting_balance),
                equity=float(self.starting_balance),
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )
            session.add(snap)
            return float(self.starting_balance)

    def account_balance(self) -> float:
        return self._get_or_create_balance()

    def get_drawdown_stats(self) -> Dict[str, float]:
        with session_scope() as session:
            snaps = list(session.exec(select(EquitySnapshot).order_by(EquitySnapshot.timestamp.asc())).all())
            if not snaps:
                return {"current_drawdown": 0.0, "max_drawdown": 0.0}

            equities = [float(x.equity) for x in snaps]
            running_peak = equities[0]
            max_dd = 0.0
            for eq in equities:
                running_peak = max(running_peak, eq)
                dd = (running_peak - eq) / running_peak if running_peak > 0 else 0.0
                max_dd = max(max_dd, dd)

            current_peak = max(equities)
            current_dd = (current_peak - equities[-1]) / current_peak if current_peak > 0 else 0.0
            return {"current_drawdown": current_dd, "max_drawdown": max_dd}

    def open_risk_exposure_pct(self) -> float:
        with session_scope() as session:
            positions = list(session.exec(select(Position).where(Position.status == "OPEN")).all())
            latest_snapshot = session.exec(select(EquitySnapshot).order_by(EquitySnapshot.timestamp.desc())).first()
            balance = float(latest_snapshot.balance) if latest_snapshot else float(self.starting_balance)
            if balance <= 0:
                return 0.0

            risk_notional = 0.0
            for p in positions:
                risk_notional += abs((p.entry_price - p.stop_loss) * p.size)

            return float(risk_notional / balance)

    def open_positions(self) -> List[Position]:
        with session_scope() as session:
            return list(session.exec(select(Position).where(Position.status == "OPEN")).all())

    def has_open_position_for_symbol(self, symbol: str) -> bool:
        with session_scope() as session:
            position = session.exec(
                select(Position).where(Position.status == "OPEN", Position.symbol == symbol)
            ).first()
            return position is not None

    def last_trade_time(self, symbol: str):
        with session_scope() as session:
            trade = session.exec(select(Trade).where(Trade.symbol == symbol).order_by(Trade.closed_at.desc())).first()
            return trade.closed_at if trade else None

    def _entry_fill(self, action: SignalAction, market_price: float) -> float:
        if action == SignalAction.BUY:
            return market_price * (1 + self.slippage_rate)
        return market_price * (1 - self.slippage_rate)

    def _exit_fill_for_side(self, side: str, exit_price: float) -> float:
        if side == SignalAction.BUY.value:
            return exit_price * (1 - self.slippage_rate)
        return exit_price * (1 + self.slippage_rate)

    def open_trade(
        self,
        symbol: str,
        action: SignalAction,
        market_price: float,
        size: float,
        stop_loss: float,
        take_profit: float,
        risk: Optional[RiskDecision] = None,
    ) -> int:
        entry_fill = self._entry_fill(action, market_price)
        with session_scope() as session:
            position = Position(
                symbol=symbol,
                side=action.value,
                entry_price=float(entry_fill),
                size=float(size),
                stop_loss=float(stop_loss),
                take_profit=float(take_profit),
                risk_reason=risk.reason if risk else "",
                confidence=risk.confidence if risk else 0.0,
                expected_value=risk.expected_value if risk else 0.0,
                expected_rr=risk.expected_rr if risk else 0.0,
                effective_risk_pct=risk.effective_risk_pct if risk else 0.0,
                estimated_cost=risk.estimated_cost if risk else 0.0,
                regime=risk.regime if risk else "unknown",
                drawdown=risk.drawdown if risk else 0.0,
                status="OPEN",
            )
            session.add(position)
            session.flush()
            return int(position.id)

    def mark_to_market(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        with session_scope() as session:
            positions = list(session.exec(select(Position).where(Position.status == "OPEN")).all())
            latest_snapshot = session.exec(select(EquitySnapshot).order_by(EquitySnapshot.timestamp.desc())).first()
            balance = float(latest_snapshot.balance) if latest_snapshot else float(self.starting_balance)
            realized_pnl = float(latest_snapshot.realized_pnl) if latest_snapshot else 0.0

            unrealized = 0.0
            for pos in positions:
                current_price = current_prices.get(pos.symbol)
                if current_price is None:
                    continue
                if pos.side == SignalAction.BUY.value:
                    unrealized += (current_price - pos.entry_price) * pos.size
                else:
                    unrealized += (pos.entry_price - current_price) * pos.size

            equity = balance + unrealized
            session.add(
                EquitySnapshot(
                    balance=balance,
                    equity=equity,
                    unrealized_pnl=unrealized,
                    realized_pnl=realized_pnl,
                )
            )
            return {"balance": balance, "equity": equity, "unrealized_pnl": unrealized, "realized_pnl": realized_pnl}

    def _close_position(self, session, position: Position, exit_price: float, exit_reason: str):
        exit_fill = self._exit_fill_for_side(position.side, exit_price)
        entry_notional = abs(position.entry_price * position.size)
        exit_notional = abs(exit_fill * position.size)
        fees = (entry_notional + exit_notional) * self.fee_rate

        slippage_cost = abs((exit_price - exit_fill) * position.size)

        if position.side == SignalAction.BUY.value:
            gross = (exit_fill - position.entry_price) * position.size
        else:
            gross = (position.entry_price - exit_fill) * position.size

        pnl = gross - fees

        position.status = "CLOSED"
        position.closed_at = datetime.utcnow()

        session.add(
            Trade(
                symbol=position.symbol,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=exit_fill,
                size=position.size,
                fees=fees,
                slippage_cost=slippage_cost,
                pnl=pnl,
                realized=True,
                risk_reason=position.risk_reason,
                confidence=position.confidence,
                expected_value=position.expected_value,
                expected_rr=position.expected_rr,
                effective_risk_pct=position.effective_risk_pct,
                estimated_cost=position.estimated_cost,
                regime=position.regime,
                drawdown=position.drawdown,
                opened_at=position.opened_at,
                closed_at=position.closed_at,
                exit_reason=exit_reason,
            )
        )

        latest_snapshot = session.exec(select(EquitySnapshot).order_by(EquitySnapshot.timestamp.desc())).first()
        balance = float(latest_snapshot.balance) if latest_snapshot else float(self.starting_balance)
        realized = float(latest_snapshot.realized_pnl) if latest_snapshot else 0.0

        new_balance = balance + pnl
        new_realized = realized + pnl
        session.add(
            EquitySnapshot(
                balance=new_balance,
                equity=new_balance,
                unrealized_pnl=0.0,
                realized_pnl=new_realized,
            )
        )

    def process_exits(self, current_prices: Dict[str, float], current_bars: Optional[Dict[str, Dict[str, float]]] = None) -> int:
        """If OHLC is provided, uses high/low touch logic; otherwise falls back to last price checks."""
        closed = 0
        with session_scope() as session:
            positions = list(session.exec(select(Position).where(Position.status == "OPEN")).all())
            for pos in positions:
                price = current_prices.get(pos.symbol)
                bar = (current_bars or {}).get(pos.symbol)

                if price is None and bar is None:
                    continue

                if bar is not None:
                    high = float(bar.get("high", price if price is not None else 0.0))
                    low = float(bar.get("low", price if price is not None else 0.0))
                else:
                    high = low = float(price)

                if pos.side == SignalAction.BUY.value:
                    stop_hit = low <= pos.stop_loss
                    target_hit = high >= pos.take_profit
                    if stop_hit and target_hit:
                        self._close_position(session, pos, pos.stop_loss, "stop_and_target_same_bar_stop_first")
                        closed += 1
                    elif stop_hit:
                        self._close_position(session, pos, pos.stop_loss, "stop_loss")
                        closed += 1
                    elif target_hit:
                        self._close_position(session, pos, pos.take_profit, "take_profit")
                        closed += 1
                else:
                    stop_hit = high >= pos.stop_loss
                    target_hit = low <= pos.take_profit
                    if stop_hit and target_hit:
                        self._close_position(session, pos, pos.stop_loss, "stop_and_target_same_bar_stop_first")
                        closed += 1
                    elif stop_hit:
                        self._close_position(session, pos, pos.stop_loss, "stop_loss")
                        closed += 1
                    elif target_hit:
                        self._close_position(session, pos, pos.take_profit, "take_profit")
                        closed += 1
        return closed
