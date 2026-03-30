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
    max_trades_validation=1000,
    max_notional_pct=1.0,      # max position notional as % of equity (1.0 = 100%)
    min_stop_pct=0.002,        # minimum stop distance = 0.2% of entry
    atr_stop_mult=0.8,
    atr_tp_mult=1.2,
    max_atr_vol_mult=1.8,
    max_loss_streak=3,
    drawdown_risk_cut=0.05,    # reduce risk after 5% DD
    hard_stop_drawdown=0.10,   # stop trading after 10% DD
):
    """
    Improved backtest with:
    - capped risk per trade in [0.5%, 1%]
    - ADX regime filter
    - probability threshold
    - ATR volatility filter
    - true stop-loss / take-profit check using next candle OHLC
    - exposure cap
    - more realistic transaction cost handling
    - adaptive risk reduction on loss streak and drawdown
    """

    required_cols = {"open", "high", "low", "close", "adx", "atr"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if len(signals) != len(df) or len(probs) != len(df):
        raise ValueError("Length of df, signals, and probs must match.")

    # Keep requested trade risk between 0.5% and 1.0%.
    risk_pct = min(max(risk_pct, 0.005), 0.01)

    equity = float(initial_capital)
    peak_equity = float(initial_capital)

    equity_curve = []
    profits = []
    trade_returns = []

    risk_multiplier = 1.0
    loss_streak = 0
    trading_halted = False

    trade_log = []

    for i in range(len(df) - 1):
        equity_curve.append(equity)

        # Hard shutdown if drawdown gets too deep.
        current_drawdown_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if current_drawdown_pct >= hard_stop_drawdown:
            trading_halted = True

        if trading_halted:
            continue

        signal = signals[i]
        prob = probs[i]

        if signal == 0:
            continue

        # Probability threshold
        if signal == 1 and prob < min_prob:
            continue
        if signal == -1 and prob > (1 - min_prob):
            continue

        # Regime filter
        adx = float(df.loc[i, "adx"])
        if adx <= min_adx:
            continue

        entry_price_raw = float(df.loc[i, "close"])
        next_open = float(df.loc[i + 1, "open"])
        next_high = float(df.loc[i + 1, "high"])
        next_low = float(df.loc[i + 1, "low"])
        next_close = float(df.loc[i + 1, "close"])

        if entry_price_raw <= 0:
            continue

        atr = float(df.loc[i, "atr"])
        atr_mean = float(df["atr"].iloc[: i + 1].rolling(50, min_periods=1).mean().iloc[-1])

        # Skip abnormal volatility spikes
        if atr > atr_mean * max_atr_vol_mult:
            continue

        # Confidence from probability
        # 0.15 floor so very small edge does not become zero sizing
        confidence = max(abs(prob - 0.5) * 2, 0.15)

        # Stop/target distances
        stop_distance = max(atr * atr_stop_mult, entry_price_raw * min_stop_pct)
        take_profit_distance = atr * atr_tp_mult

        if stop_distance <= 0 or take_profit_distance <= 0:
            continue

        expected_rr = take_profit_distance / stop_distance
        if expected_rr < min_rr:
            continue

        # Use entry at next bar open for a more realistic fill assumption
        entry_price = next_open

        # Estimate expected edge in price units
        win_prob = prob if signal == 1 else (1 - prob)
        expected_gross_edge = (win_prob * take_profit_distance) - ((1 - win_prob) * stop_distance)
        expected_cost_edge = (entry_price * 2) * (fee_rate + slippage_rate)
        expected_net_edge = expected_gross_edge - expected_cost_edge

        if expected_net_edge <= 0:
            continue

        # Base allowed capital risk on current equity
        effective_risk_pct = risk_pct * risk_multiplier
        risk_per_trade = equity * effective_risk_pct

        # Raw size from stop distance
        raw_position_size = (risk_per_trade / stop_distance) * confidence

        if raw_position_size <= 0:
            continue

        # Exposure cap: max notional as % of equity
        max_notional = equity * max_notional_pct
        max_position_size = max_notional / entry_price if entry_price > 0 else 0.0
        position_size = min(raw_position_size, max_position_size)

        if position_size <= 0:
            continue

        # Define stop/target prices
        if signal == 1:
            stop_price = entry_price - stop_distance
            target_price = entry_price + take_profit_distance
        else:
            stop_price = entry_price + stop_distance
            target_price = entry_price - take_profit_distance

        # Slippage-adjusted prices
        # Worse fills for both entry and exit
        if signal == 1:
            entry_fill = entry_price * (1 + slippage_rate)
        else:
            entry_fill = entry_price * (1 - slippage_rate)

        exit_reason = "close"

        # True OHLC stop/target simulation on next candle
        if signal == 1:
            stop_hit = next_low <= stop_price
            target_hit = next_high >= target_price

            if stop_hit and target_hit:
                # Conservative assumption: stop gets hit first
                exit_price = stop_price
                exit_reason = "stop_and_target_same_bar_stop_first"
            elif stop_hit:
                exit_price = stop_price
                exit_reason = "stop"
            elif target_hit:
                exit_price = target_price
                exit_reason = "target"
            else:
                exit_price = next_close
                exit_reason = "close"

            exit_fill = exit_price * (1 - slippage_rate)
            gross_pnl = (exit_fill - entry_fill) * position_size

        else:
            stop_hit = next_high >= stop_price
            target_hit = next_low <= target_price

            if stop_hit and target_hit:
                # Conservative assumption: stop gets hit first
                exit_price = stop_price
                exit_reason = "stop_and_target_same_bar_stop_first"
            elif stop_hit:
                exit_price = stop_price
                exit_reason = "stop"
            elif target_hit:
                exit_price = target_price
                exit_reason = "target"
            else:
                exit_price = next_close
                exit_reason = "close"

            exit_fill = exit_price * (1 + slippage_rate)
            gross_pnl = (entry_fill - exit_fill) * position_size

        # Fees charged on both entry and exit notionals
        entry_notional = abs(entry_fill * position_size)
        exit_notional = abs(exit_fill * position_size)
        trade_costs = (entry_notional + exit_notional) * fee_rate

        pnl = gross_pnl - trade_costs
        prev_equity = equity
        equity += pnl
        profits.append(pnl)

        trade_return = pnl / prev_equity if prev_equity > 0 else 0.0
        trade_returns.append(trade_return)

        # Loss streak logic
        if pnl < 0:
            loss_streak += 1
        else:
            loss_streak = 0

        # Risk adjustment
        if loss_streak >= max_loss_streak:
            risk_multiplier = max(0.3, risk_multiplier * 0.7)
        else:
            # Recover risk slowly
            risk_multiplier = min(1.0, risk_multiplier * 1.02)

        # Peak / drawdown logic
        peak_equity = max(peak_equity, equity)
        drawdown_now_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        if drawdown_now_pct >= drawdown_risk_cut:
            risk_multiplier = max(0.3, risk_multiplier * 0.85)

        if drawdown_now_pct >= hard_stop_drawdown:
            trading_halted = True

        trade_log.append(
            {
                "index": i,
                "signal": signal,
                "prob": prob,
                "entry_price": entry_fill,
                "exit_price": exit_fill,
                "position_size": position_size,
                "pnl": pnl,
                "equity": equity,
                "risk_multiplier": risk_multiplier,
                "exit_reason": exit_reason,
            }
        )

    # Append final equity for better curve completeness
    equity_curve.append(equity)

    profits = np.array(profits, dtype=float)
    trade_returns = np.array(trade_returns, dtype=float)
    equity_curve = np.array(equity_curve, dtype=float)

    if len(equity_curve) > 0:
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = equity_curve - running_max
        drawdown_pct = np.where(running_max > 0, drawdown / running_max, 0.0)
    else:
        drawdown = np.array([], dtype=float)
        drawdown_pct = np.array([], dtype=float)

    wins = profits[profits > 0]
    losses = profits[profits <= 0]

    total_trades = len(profits)
    net_expectancy = profits.mean() if total_trades > 0 else 0.0
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    final_equity = equity_curve[-1] if len(equity_curve) > 0 else initial_capital
    total_pnl = final_equity - initial_capital
    total_return_pct = (total_pnl / initial_capital) if initial_capital > 0 else 0.0

    # Simple Sharpe-like metric from trade returns
    if len(trade_returns) > 1 and np.std(trade_returns, ddof=1) > 0:
        sharpe_like = (np.mean(trade_returns) / np.std(trade_returns, ddof=1)) * np.sqrt(len(trade_returns))
    else:
        sharpe_like = 0.0

    metrics = {
        "Initial Capital": float(initial_capital),
        "Final Equity": float(final_equity),
        "Total PnL": float(total_pnl),
        "Total Return %": float(total_return_pct),
        "Max Drawdown": float(drawdown.min()) if len(drawdown) > 0 else 0.0,
        "Max Drawdown %": float(drawdown_pct.min()) if len(drawdown_pct) > 0 else 0.0,
        "Total trades": int(total_trades),
        "Winning trades": int(len(wins)),
        "Losing trades": int(len(losses)),
        "Win rate": float(win_rate),
        "Average Win": float(avg_win),
        "Average Loss": float(avg_loss),
        "Win/Loss Ratio": float(win_loss_ratio),
        "Risk % per trade": float(risk_pct),
        "Fee rate": float(fee_rate),
        "Slippage rate": float(slippage_rate),
        "Net expectancy per trade": float(net_expectancy),
        "Sharpe-like": float(sharpe_like),
        "Trades validation (500-1000)": bool(min_trades_required <= total_trades <= max_trades_validation),
        "Expectancy positive after costs": bool(net_expectancy > 0),
        "Trading halted by hard drawdown": bool(trading_halted),
    }

    return equity_curve, drawdown, metrics, trade_log


def plot_backtest(equity_curve, drawdown):
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(equity_curve, label="Equity Curve")
    plt.title("Backtest Equity Curve")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(drawdown, label="Drawdown")
    plt.title("Drawdown")
    plt.legend()

    plt.tight_layout()
    plt.show()


def print_metrics(metrics):
    print("\nBacktest Metrics")
    print("=" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value}")


# Example usage:
# equity_curve, drawdown, metrics, trade_log = backtest(df, signals, probs)
# print_metrics(metrics)
# plot_backtest(equity_curve, drawdown)