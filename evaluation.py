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


def evaluate_holdout(
    strategy,
    data: pd.DataFrame,
    split_index: int,
    cost_bps: float = 10.0,
) -> dict:
    """Evaluate a chronological holdout with historical warmup preserved.

    Run the causal model on the entire history, then score only the test
    period. The split must be fixed *before* inspecting holdout performance.
    An execution at the first test open may reflect the previous bar's signal.
    No parameter selection or fitting may use the holdout.
    """
    if not isinstance(split_index, int) or not 1 <= split_index < len(data):
        raise ValueError("split_index must leave nonempty train and test periods")
    full = evaluate_strategy(strategy, data, cost_bps=cost_bps)
    test = full.iloc[split_index:].copy()
    # Rebase both curves at the start of the holdout.
    test["holdout_equity"] = (1 + test["net_return"]).cumprod()
    test["holdout_buy_hold_equity"] = (
        data["open"].iloc[split_index:].astype(float)
        / float(data["open"].iloc[split_index])
    )
    return {
        "train": full.iloc[:split_index].copy(),
        "test": test,
        "split_index": split_index,
    }
