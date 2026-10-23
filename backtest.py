"""Causal, next-open execution backtest.

Signals/positions are computed after each bar closes. The desired position at
bar t is filled at open[t+1]. P&L is marked from open to open, with turnover
costs charged at the execution open. No order is filled after the final bar.
"""
import numpy as np
import pandas as pd


def backtest_next_open(
    data: pd.DataFrame,
    target_positions: pd.Series,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Return per-bar equity and turnover; no financing or slippage model.

    Costs apply to absolute position changes, including two units on reversal.
    Return on bar t is position filled at open[t] times open[t+1]/open[t]-1.
    """
    if cost_bps < 0 or not np.isfinite(cost_bps):
        raise ValueError("cost_bps must be nonnegative and finite")
    if len(data) != len(target_positions):
        raise ValueError("data and positions must have equal lengths")
    if len(data) < 2:
        raise ValueError("at least two bars are required")
    if not data.index.equals(target_positions.index):
        raise ValueError("data and positions must have identical indices")
    opens = data["open"].astype(float)
    if not np.isfinite(opens.to_numpy()).all() or (opens <= 0).any():
        raise ValueError("open prices must be positive and finite")
    targets = pd.to_numeric(target_positions, errors="raise")
    if targets.isna().any() or not targets.isin([-1, 0, 1]).all():
        raise ValueError("target positions must be -1, 0 or 1")

    # Position at the open of bar t comes from the close of bar t-1.
    positions = targets.shift(1, fill_value=0).astype(int)
    turnover = positions.diff().fillna(positions.iloc[0]).abs()
    # P&L on [open_t, open_(t+1)) is attributed to the position at open_t.
    next_open_return = opens.shift(-1).div(opens).sub(1).fillna(0.0)
    gross = positions * next_open_return
    costs = turnover * cost_bps / 10000.0
    # The final open can execute an order and incur cost but earns no later P&L.
    net = gross - costs
    equity = (1 + net).cumprod()
    buy_hold = opens.div(opens.iloc[0])
    return pd.DataFrame({
        "executed_position": positions,
        "turnover": turnover,
        "gross_return": gross,
        "cost": costs,
        "net_return": net,
        "equity": equity,
        "buy_hold_equity": buy_hold,
    }, index=data.index)
