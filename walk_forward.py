"""Deterministic expanding-window walk-forward optimizer for fixed-rule RSI.

Parameters are selected on past observations only. Each test fold starts
with a flat portfolio and is scored using next-open execution. Warmup is
calculated using the full history, but no future bar affects past signals.
"""
from itertools import product

import numpy as np
import pandas as pd

from backtest import backtest_next_open
from baselines import rsi_target_positions


def expanding_folds(length: int, train_bars: int, test_bars: int):
    if train_bars < 2 or test_bars < 2 or length < train_bars + test_bars:
        raise ValueError("insufficient history or invalid window sizes")
    return [(end, min(end + test_bars, length))
            for end in range(train_bars, length, test_bars)]


def fold_returns(data: pd.DataFrame, targets: pd.Series, start: int, end: int,
                 cost_bps: float) -> pd.DataFrame:
    """Score a fold independently; previous signals supply feature warmup only.

    The first test open is flat. The first eligible fill is the second test
    open, using the signal from the first test close. This avoids an
    unaccounted pre-test position or cost.
    """
    if not 0 <= start < end <= len(data) or end - start < 2:
        raise ValueError("invalid fold bounds")
    local = data.iloc[start:end]
    local_targets = targets.iloc[start:end]
    return backtest_next_open(local, local_targets, cost_bps)


def score_fold(ledger: pd.DataFrame) -> float:
    """Net log growth; nonpositive equity is a losing configuration."""
    equity = float(ledger["equity"].iloc[-1])
    if not np.isfinite(equity) or equity <= 0:
        return float("-inf")
    return float(np.log(equity))


def optimize_rsi_walk_forward(data: pd.DataFrame, train_bars: int,
                              test_bars: int, cost_bps: float = 10.0,
                              periods=(7, 14, 21),
                              oversold_levels=(25., 30., 35.),
                              overbought_levels=(65., 70., 75.)) -> pd.DataFrame:
    """Select RSI settings on each expanding training window, then test once."""
    if cost_bps < 0 or not np.isfinite(cost_bps):
        raise ValueError("invalid cost_bps")
    folds = expanding_folds(len(data), train_bars, test_bars)
    candidates = [(int(p), float(lo), float(hi))
                  for p, lo, hi in product(periods, oversold_levels, overbought_levels)
                  if p >= 2 and 0 < lo < hi < 100]
    if not candidates:
        raise ValueError("empty valid parameter grid")
    targets = {params: rsi_target_positions(
        data, period=params[0], oversold=params[1], overbought=params[2]
    ) for params in candidates}
    records = []
    for start, end in folds:
        # The training interval is [0,start); no test-period price enters selection.
        best = max(candidates, key=lambda params: (
            score_fold(fold_returns(data, targets[params], 0, start, cost_bps)),
            -candidates.index(params)
        ))
        test_ledger = fold_returns(data, targets[best], start, end, cost_bps)
        records.append({
            "train_end": data.index[start - 1],
            "test_start": data.index[start],
            "test_end": data.index[end - 1],
            "period": best[0],
            "oversold": best[1],
            "overbought": best[2],
            "train_log_growth": score_fold(
                fold_returns(data, targets[best], 0, start, cost_bps)),
            "test_return": float(test_ledger["equity"].iloc[-1] - 1),
            "test_turnover": float(test_ledger["turnover"].sum()),
            "test_bars": end - start,
        })
    return pd.DataFrame.from_records(records)
