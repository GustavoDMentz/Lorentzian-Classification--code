"""Causal rolling-performance selector for the predeclared stateful systems.

At close[t], rank strategies using only realized open-to-open returns through
open[t] (the last complete interval ends at open[t]). Select one strategy or
cash; execute its target at open[t+1]. No hindsight regime labels.
"""
import numpy as np
import pandas as pd
from backtest import backtest_next_open
from stateful_systems import system_targets
from walk_forward import expanding_folds


def adaptive_targets(data, lookback=336, cost_bps=10., min_hold=24):
    if lookback < 48 or min_hold < 1 or cost_bps < 0:
        raise ValueError("invalid selector settings")
    systems = system_targets(data)
    opens = data["open"].astype(float)
    # At close[t], open[t] is known; the latest fully realized open interval
    # is [open[t-1], open[t]]. Never use open[t+1] for selection.
    completed = opens.pct_change().fillna(0.).to_numpy()
    names = list(systems)
    history = []
    for name in names:
        positions = systems[name].shift(1, fill_value=0).to_numpy(dtype=int)
        previous = np.r_[0, positions[:-1]]
        turnover = np.abs(positions - previous)
        # Realized net return at open[t] belongs to the position held since
        # open[t-1]; entry cost paid at open[t-1] is included in that interval.
        completed_net = np.r_[0., positions[:-1] * completed[1:] -
                              turnover[:-1] * cost_bps / 10000.]
        history.append(completed_net)
    returns = pd.DataFrame(np.asarray(history).T, index=data.index, columns=names)
    # A strictly historical score, with no reoptimization of indicator params.
    scores = returns.rolling(lookback, min_periods=lookback).sum()
    chosen = np.zeros(len(data), dtype=int)
    selection = []
    current = None
    last_switch = -min_hold
    for t in range(len(data)):
        row = scores.iloc[t]
        best = None
        if row.notna().all():
            winner = row.idxmax()
            if row[winner] > 0:
                best = winner
        if t - last_switch >= min_hold and best != current:
            current = best
            last_switch = t
        chosen[t] = int(systems[current].iloc[t]) if current is not None else 0
        selection.append(current if current is not None else "CASH")
    return pd.Series(chosen, index=data.index), pd.Series(selection, index=data.index)


def evaluate_adaptive(data, train_bars=1440, test_bars=336, cost_bps=10.,
                      lookback=336, min_hold=24):
    targets, selection = adaptive_targets(data, lookback, cost_bps, min_hold)
    folds = expanding_folds(len(data), train_bars, test_bars)
    records = []
    for start, end in folds:
        ledger = backtest_next_open(data.iloc[start:end],
                                    targets.iloc[start:end], cost_bps)
        records.append({
            "test_start": str(data.index[start]),
            "test_end": str(data.index[end - 1]),
            "test_return": float(ledger["equity"].iloc[-1] - 1),
            "gross_0bps": float((1 + ledger["gross_return"]).prod() - 1),
            "turnover": float(ledger["turnover"].sum()),
            "cash_selected_bars": int((selection.iloc[start:end] == "CASH").sum()),
            "strategy_switches": int((selection.iloc[start:end] !=
                                     selection.iloc[start:end].shift(1)).sum()) - 1,
        })
    frame = pd.DataFrame(records)
    result = pd.DataFrame([{
        "strategy": f"Adaptive rolling {lookback}h hold {min_hold}h",
        "folds": len(frame),
        "compounded_return": float(np.prod(1 + frame["test_return"]) - 1),
        "gross_0bps": float(np.prod(1 + frame["gross_0bps"]) - 1),
        "positive_folds": int((frame["test_return"] > 0).sum()),
        "worst_fold_return": float(frame["test_return"].min()),
        "turnover": float(frame["turnover"].sum()),
        "strategy_switches": int(frame["strategy_switches"].sum()),
    }])
    return result, frame, selection
