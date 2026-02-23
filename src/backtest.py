import numpy as np
import matplotlib.pyplot as plt

def backtest(df, signals, probs, initial_capital=10_000, risk_pct=0.003):
    equity = initial_capital
    peak_equity = initial_capital
    profits = []
    equity_curve = []
    risk_multiplier = 1.0
    loss_streak = 0
    max_loss_streak = 3

    prob_map = dict(zip(df.index, probs))

    for i in range(len(df)-1):
        equity_curve.append(equity)

        signal = signals[i]
        if signal == 0:
            continue

        entry_price = df.loc[i, "close"]
        exit_price = df.loc[i+1, "close"]

        # Confidence sizing
        confidence = max(abs(prob_map[i] - 0.5) * 2, 0.15)
        atr = df.loc[i, "atr"]
        atr_mean = df["atr"].iloc[:i].rolling(50).mean().iloc[-1]

        # Volatility filter
        if atr > atr_mean * 1.8:
            continue

        # Risk & SL/TP
        risk_per_trade = equity * risk_pct * risk_multiplier
        stop_loss = atr * 0.8
        take_profit = atr * 2.2
        position_size = (risk_per_trade / stop_loss) * confidence

        pnl = (exit_price - entry_price) * position_size if signal == 1 else (entry_price - exit_price) * position_size
        pnl = max(min(pnl, take_profit * position_size), -stop_loss * position_size)

        equity += pnl
        profits.append(pnl)

        # Loss streak
        if pnl < 0:
            loss_streak += 1
        else:
            loss_streak = 0

        if loss_streak >= max_loss_streak:
            risk_multiplier = max(0.3, risk_multiplier * 0.7)
        else:
            risk_multiplier = min(1.0, risk_multiplier * 1.05)

        # Drawdown control
        peak_equity = max(peak_equity, equity)
        drawdown_now = equity - peak_equity
        if drawdown_now < -0.05 * peak_equity:
            risk_multiplier *= 0.7

    # Metrics
    profits = np.array(profits)
    wins = profits[profits>0]
    losses = profits[profits<=0]
    equity_curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - running_max

    metrics = {
        "Max Drawdown": drawdown.min(),
        "Total trades": len(profits),
        "Winning trades": len(wins),
        "Losing trades": len(losses),
        "Win rate": len(wins)/len(profits) if len(profits) > 0 else 0,
        "Total PnL": profits.sum(),
        "Average Win": wins.mean() if len(wins)>0 else 0,
        "Average Loss": losses.mean() if len(losses)>0 else 0,
        "Win/Loss Ratio": abs(wins.mean()/losses.mean()) if len(losses)>0 else np.inf
    }

    return equity_curve, drawdown, metrics

def plot_backtest(equity_curve, drawdown):
    plt.figure(figsize=(14,6))
    plt.subplot(2,1,1)
    plt.plot(equity_curve, label="Equity Curve")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(drawdown, label="Drawdown", color="red")
    plt.legend()
    plt.tight_layout()
    plt.show()
