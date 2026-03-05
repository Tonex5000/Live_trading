import numpy as np
import matplotlib.pyplot as plt


def backtest(
    df,
    signals,
    probs,
    initial_capital=10_000,
    risk_pct=0.01,
    fee_rate=0.0006,
    slippage_rate=0.0004,
    min_prob=0.60,
    min_rr=1.5,
    min_adx=20.0,
    min_trades_required=500,
):
    # Risk cap per trade in requested range [0.5%, 1.0%].
    risk_pct = min(max(risk_pct, 0.005), 0.01)

    equity = initial_capital
    peak_equity = initial_capital
    profits = []
    equity_curve = []
    risk_multiplier = 1.0
    loss_streak = 0
    max_loss_streak = 3

    for i in range(len(df) - 1):
        equity_curve.append(equity)

        signal = signals[i]
        prob = probs[i]

        if signal == 0:
            continue

        # Probability threshold.
        if signal == 1 and prob < min_prob:
            continue
        if signal == -1 and prob > (1 - min_prob):
            continue

        # Regime filter.
        adx = df.loc[i, "adx"]
        if adx <= min_adx:
            continue

        entry_price = df.loc[i, "close"]
        exit_price = df.loc[i + 1, "close"]

        confidence = max(abs(prob - 0.5) * 2, 0.15)
        atr = df.loc[i, "atr"]
        atr_mean = df["atr"].iloc[: i + 1].rolling(50, min_periods=1).mean().iloc[-1]

        if atr > atr_mean * 1.8:
            continue

        stop_loss = atr * 0.8
        take_profit = atr * 1.2
        expected_rr = take_profit / stop_loss if stop_loss > 0 else 0

        # R:R filter.
        if expected_rr < min_rr:
            continue

        # Expected net edge after fees/slippage must be positive.
        win_prob = prob if signal == 1 else (1 - prob)
        expected_gross_edge = (win_prob * take_profit) - ((1 - win_prob) * stop_loss)
        expected_cost_edge = (entry_price * 2) * (fee_rate + slippage_rate)
        expected_net_edge = expected_gross_edge - expected_cost_edge
        if expected_net_edge <= 0:
            continue

        risk_per_trade = equity * risk_pct * risk_multiplier
        position_size = (risk_per_trade / stop_loss) * confidence if stop_loss > 0 else 0

        gross_pnl = (exit_price - entry_price) * position_size if signal == 1 else (entry_price - exit_price) * position_size
        gross_pnl = max(min(gross_pnl, take_profit * position_size), -stop_loss * position_size)

        # Fees + slippage on entry and exit.
        trade_notional = (entry_price + exit_price) * position_size
        trade_costs = trade_notional * (fee_rate + slippage_rate)
        pnl = gross_pnl - trade_costs

        equity += pnl
        profits.append(pnl)

        if pnl < 0:
            loss_streak += 1
        else:
            loss_streak = 0

        if loss_streak >= max_loss_streak:
            risk_multiplier = max(0.3, risk_multiplier * 0.7)
        else:
            risk_multiplier = min(1.0, risk_multiplier * 1.05)

        peak_equity = max(peak_equity, equity)
        drawdown_now = equity - peak_equity
        if drawdown_now < -0.05 * peak_equity:
            risk_multiplier *= 0.7

    profits = np.array(profits)
    wins = profits[profits > 0]
    losses = profits[profits <= 0]
    equity_curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_curve) if len(equity_curve) > 0 else np.array([])
    drawdown = equity_curve - running_max if len(equity_curve) > 0 else np.array([])

    total_trades = len(profits)
    gross_expectancy = profits.mean() if total_trades > 0 else 0
    net_expectancy = gross_expectancy

    metrics = {
        "Max Drawdown": float(drawdown.min()) if len(drawdown) > 0 else 0,
        "Total trades": total_trades,
        "Winning trades": len(wins),
        "Losing trades": len(losses),
        "Win rate": len(wins) / total_trades if total_trades > 0 else 0,
        "Total PnL": float(profits.sum()) if total_trades > 0 else 0,
        "Average Win": float(wins.mean()) if len(wins) > 0 else 0,
        "Average Loss": float(losses.mean()) if len(losses) > 0 else 0,
        "Win/Loss Ratio": abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else np.inf,
        "Risk % per trade": risk_pct,
        "Fee rate": fee_rate,
        "Slippage rate": slippage_rate,
        "Net expectancy per trade": float(net_expectancy),
        "Trades validation (500-1000)": min_trades_required <= total_trades <= 1000,
        "Expectancy positive after costs": net_expectancy > 0,
    }

    return equity_curve, drawdown, metrics


def plot_backtest(equity_curve, drawdown):
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(equity_curve, label="Equity Curve")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(drawdown, label="Drawdown", color="red")
    plt.legend()
    plt.tight_layout()
    plt.show()