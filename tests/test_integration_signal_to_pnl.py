import pandas as pd
from sqlmodel import delete

from src.db import create_db_and_tables, session_scope
from src.execution import PaperExecutor
from src.features import add_indicators
from src.models import EquitySnapshot, Position, Trade
from src.paper_engine import PaperTradingEngine
from src.risk import RiskConfig, RiskManager
from src.strategies import MLStrategy


class AlwaysBuyModel:
    feature_names_in_ = ["rsi", "ema_20", "ema_50", "ema_200", "atr", "adx", "trend", "strong_trend"]

    def predict_proba(self, X):
        return [[0.1, 0.9] for _ in range(len(X))]


def reset_db():
    create_db_and_tables()
    with session_scope() as session:
        session.exec(delete(Trade))
        session.exec(delete(Position))
        session.exec(delete(EquitySnapshot))


def test_signal_to_trade_to_realized_pnl():
    reset_db()
    model = AlwaysBuyModel()
    strategy = MLStrategy(model=model, symbol="BTC/USDT:USDT", timeframe="30m")
    risk = RiskManager(RiskConfig(max_risk_per_trade=0.01, max_concurrent_positions=2, cooldown_minutes=0))
    engine = PaperTradingEngine(starting_balance=10000.0)
    executor = PaperExecutor(engine)

    # Build enough bars for indicators to be valid.
    rows = []
    for i in range(250):
        price = 100 + i * 0.2
        rows.append(
            {
                "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=30 * i),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.3,
                "volume": 10 + i,
            }
        )

    df = add_indicators(pd.DataFrame(rows)).dropna().reset_index(drop=True)
    signal = strategy.generate(df)

    latest_price = float(df.iloc[-1]["close"])
    atr = float(df.iloc[-1]["atr"])

    decision = risk.evaluate(
        signal=signal,
        latest_price=latest_price,
        atr=atr,
        account_balance=engine.account_balance(),
        open_positions_count=0,
        has_open_position_for_symbol=False,
        last_trade_at=None,
        now=pd.Timestamp("2026-01-02T00:00:00Z").to_pydatetime(),
    )

    result = executor.execute(signal=signal, risk=decision, market_price=latest_price, timestamp=signal.timestamp)
    assert result.executed is True

    engine.process_exits({"BTC/USDT:USDT": decision.take_profit + 1})

    from sqlmodel import select
    with session_scope() as session:
        trades = session.exec(select(Trade)).all()

    assert len(trades) == 1
    assert trades[0].pnl > 0
