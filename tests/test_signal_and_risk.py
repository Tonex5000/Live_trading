from datetime import datetime, timedelta

from src.risk import RiskConfig, RiskManager
from src.strategies.signal_schema import SignalAction, StrategySignal


def _signal(action=SignalAction.BUY, confidence=0.8, probability=0.8):
    return StrategySignal(
        symbol="BTC/USDT:USDT",
        timeframe="30m",
        action=action,
        confidence=confidence,
        probability=probability,
        reason="test",
        timestamp=datetime.utcnow().isoformat(),
    )


def _manager():
    return RiskManager(
        RiskConfig(
            max_risk_per_trade=0.01,
            max_concurrent_positions=1,
            cooldown_minutes=30,
            min_confidence=0.3,
            min_expected_value=0.0,
            min_rr=1.4,
            min_qty=0.001,
            min_notional=5.0,
            qty_precision=6,
            price_precision=2,
        )
    )


def test_hold_gets_rejected():
    d = _manager().evaluate(
        signal=_signal(action=SignalAction.HOLD),
        latest_price=100,
        atr=1,
        account_balance=10000,
        open_positions_count=0,
        has_open_position_for_symbol=False,
        last_trade_at=None,
        now=datetime.utcnow(),
    )
    assert not d.approved
    assert d.reason == "hold_signal"


def test_low_confidence_gets_rejected():
    d = _manager().evaluate(
        signal=_signal(confidence=0.2),
        latest_price=100,
        atr=1,
        account_balance=10000,
        open_positions_count=0,
        has_open_position_for_symbol=False,
        last_trade_at=None,
        now=datetime.utcnow(),
    )
    assert not d.approved
    assert d.reason == "low_confidence"


def test_ev_below_threshold_gets_rejected():
    m = RiskManager(RiskConfig(min_expected_value=1000.0))
    d = m.evaluate(
        signal=_signal(confidence=0.9, probability=0.55),
        latest_price=50000,
        atr=100,
        account_balance=10000,
        open_positions_count=0,
        has_open_position_for_symbol=False,
        last_trade_at=None,
        now=datetime.utcnow(),
    )
    assert not d.approved
    assert d.reason == "ev_below_threshold"


def test_cooldown_rejection():
    d = _manager().evaluate(
        signal=_signal(),
        latest_price=50000,
        atr=100,
        account_balance=10000,
        open_positions_count=0,
        has_open_position_for_symbol=False,
        last_trade_at=datetime.utcnow() - timedelta(minutes=5),
        now=datetime.utcnow(),
    )
    assert not d.approved
    assert d.reason == "cooldown_active"


def test_duplicate_symbol_rejection():
    d = _manager().evaluate(
        signal=_signal(),
        latest_price=50000,
        atr=100,
        account_balance=10000,
        open_positions_count=0,
        has_open_position_for_symbol=True,
        last_trade_at=None,
        now=datetime.utcnow(),
    )
    assert not d.approved
    assert d.reason == "duplicate_symbol_open"


def test_invalid_inputs_rejection():
    d = _manager().evaluate(
        signal=_signal(),
        latest_price=0,
        atr=100,
        account_balance=10000,
        open_positions_count=0,
        has_open_position_for_symbol=False,
        last_trade_at=None,
        now=datetime.utcnow(),
    )
    assert not d.approved
    assert d.reason == "invalid_market_inputs"


def test_invalid_stop_distance_rejection():
    m = RiskManager(RiskConfig(atr_stop_mult=0.0))
    d = m.evaluate(
        signal=_signal(),
        latest_price=50000,
        atr=100,
        account_balance=10000,
        open_positions_count=0,
        has_open_position_for_symbol=False,
        last_trade_at=None,
        now=datetime.utcnow(),
    )
    assert not d.approved
    assert d.reason == "invalid_stop_distance"


def test_qty_below_min_rejection():
    m = RiskManager(RiskConfig(min_qty=10.0, qty_precision=2))
    d = m.evaluate(
        signal=_signal(),
        latest_price=50000,
        atr=100,
        account_balance=10000,
        open_positions_count=0,
        has_open_position_for_symbol=False,
        last_trade_at=None,
        now=datetime.utcnow(),
    )
    assert not d.approved
    assert d.reason == "qty_below_minimum"


def test_notional_below_min_rejection():
    m = RiskManager(RiskConfig(min_notional=1_000_000.0))
    d = m.evaluate(
        signal=_signal(),
        latest_price=50000,
        atr=100,
        account_balance=10000,
        open_positions_count=0,
        has_open_position_for_symbol=False,
        last_trade_at=None,
        now=datetime.utcnow(),
    )
    assert not d.approved
    assert d.reason == "notional_below_minimum"


def test_approved_trade_path():
    d = _manager().evaluate(
        signal=_signal(confidence=0.9, probability=0.9),
        latest_price=100,
        atr=1,
        account_balance=10000,
        open_positions_count=0,
        has_open_position_for_symbol=False,
        last_trade_at=None,
        now=datetime.utcnow(),
        atr_mean=95,
    )
    assert d.approved
    assert d.reason == "approved"
    assert d.position_size > 0
    assert d.expected_value > 0


def test_confidence_scaling_affects_effective_risk():
    m = _manager()
    low = m.evaluate(
        signal=_signal(confidence=0.4, probability=0.9),
        latest_price=100,
        atr=1,
        account_balance=10000,
        open_positions_count=0,
        has_open_position_for_symbol=False,
        last_trade_at=None,
        now=datetime.utcnow(),
    )
    high = m.evaluate(
        signal=_signal(confidence=0.9, probability=0.9),
        latest_price=100,
        atr=1,
        account_balance=10000,
        open_positions_count=0,
        has_open_position_for_symbol=False,
        last_trade_at=None,
        now=datetime.utcnow(),
    )
    assert high.effective_risk_pct > low.effective_risk_pct
