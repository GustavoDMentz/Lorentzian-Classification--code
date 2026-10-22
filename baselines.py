"""Fixed-parameter RSI(14) baseline: long below 30, short above 70.

Signals are computed from closing prices only; fills happen next open in
backtest_next_open. No hyperparameter optimization is performed.
"""
import pandas as pd
import talib


def rsi_target_positions(data: pd.DataFrame, period: int = 14,
                         oversold: float = 30., overbought: float = 70.) -> pd.Series:
    if period < 2 or not 0 < oversold < overbought < 100:
        raise ValueError("invalid RSI parameters")
    close = data["close"].astype(float)
    rsi = pd.Series(talib.RSI(close.to_numpy(), timeperiod=period), index=data.index)
    targets = pd.Series(0, index=data.index, dtype=int)
    targets.loc[rsi < oversold] = 1
    targets.loc[rsi > overbought] = -1
    return targets


def compare_rsi_holdout(strategy, data: pd.DataFrame, split_index: int,
                        cost_bps: float = 10.0) -> dict:
    """Compare Lorentzian and fixed RSI on identical chronological test bars."""
    from evaluation import evaluate_holdout
    from backtest import backtest_next_open
    if not isinstance(split_index, int) or not 1 <= split_index < len(data):
        raise ValueError("invalid split_index")
    lorentz = evaluate_holdout(strategy, data, split_index, cost_bps)["test"]
    rsi_targets = rsi_target_positions(data)
    rsi = backtest_next_open(data, rsi_targets, cost_bps).iloc[split_index:].copy()
    rsi["holdout_equity"] = (1 + rsi["net_return"]).cumprod()
    return {"lorentzian": lorentz, "rsi14": rsi}
