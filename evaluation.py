"""Connect strategy signals to the next-open execution model.

The full dataset is used only for feature calculation. Each prediction at t
is causal by construction (only matured labels are eligible), and the order
is filled at open[t+1]. Evaluation must be done on held-out periods.
"""
import pandas as pd
from backtest import backtest_next_open


def evaluate_strategy(strategy, data: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    """Generate causal predictions and simulate next-open execution.

    Requires OHLC columns; returns signals alongside the execution ledger.
    This function does not tune parameters or assert predictive validity.
    """
    required = {"open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    if len(data) < 2:
        raise ValueError("at least two bars are required")
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("data index must be unique and sorted")
    signals = strategy.generate_signals(data)
    if not signals.index.equals(data.index):
        raise ValueError("strategy output index must match input")
    ledger = backtest_next_open(data, signals["position"], cost_bps=cost_bps)
    return pd.concat([signals, ledger], axis=1)
