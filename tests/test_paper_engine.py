from sqlmodel import delete

from src.db import create_db_and_tables, session_scope
from src.models import EquitySnapshot, Position, Trade
from src.paper_engine import PaperTradingEngine
from src.strategies.signal_schema import SignalAction


def reset_db():
    create_db_and_tables()
    with session_scope() as session:
        session.exec(delete(Trade))
        session.exec(delete(Position))
        session.exec(delete(EquitySnapshot))


def test_open_and_close_trade_with_take_profit():
    reset_db()
    engine = PaperTradingEngine(starting_balance=10000.0)

    engine.open_trade(
        symbol="BTC/USDT:USDT",
        action=SignalAction.BUY,
        market_price=100.0,
        size=1.0,
        stop_loss=95.0,
        take_profit=105.0,
    )

    closed = engine.process_exits({"BTC/USDT:USDT": 106.0})
    assert closed == 1

    from sqlmodel import select
    with session_scope() as session:
        trades = session.exec(select(Trade)).all()
    assert len(trades) == 1
    assert trades[0].pnl != 0


def test_mark_to_market_updates_equity():
    reset_db()
    engine = PaperTradingEngine(starting_balance=10000.0)

    engine.open_trade(
        symbol="BTC/USDT:USDT",
        action=SignalAction.BUY,
        market_price=100.0,
        size=2.0,
        stop_loss=90.0,
        take_profit=120.0,
    )

    snap = engine.mark_to_market({"BTC/USDT:USDT": 110.0})
    assert snap["equity"] > snap["balance"]
