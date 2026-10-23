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
            "test_gross_return": float((1 + test_ledger["gross_return"]).prod() - 1),
            "test_cost_sum": float(test_ledger["cost"].sum()),
            "test_long_bars": int((test_ledger["executed_position"] == 1).sum()),
            "test_short_bars": int((test_ledger["executed_position"] == -1).sum()),
            "test_flat_bars": int((test_ledger["executed_position"] == 0).sum()),
            "test_turnover": float(test_ledger["turnover"].sum()),
            "test_bars": end - start,
        })
    return pd.DataFrame.from_records(records)


def optimize_lorentzian_walk_forward(
    data: pd.DataFrame,
    train_bars: int,
    test_bars: int,
    cost_bps: float = 10.0,
    neighbors_grid=(4, 8),
    regime_grid=(0.15, 0.25),
    max_bars_back: int = 500,
    strategy_factory=None,
) -> pd.DataFrame:
    """Causal expanding-window search with the same scoring as RSI.

    Predictions are cached for each candidate over the full series. This is
    safe only because the strategy's features and kNN labels are causal:
    the four-bar label at i is eligible no earlier than bar i+4.
    The holdout is never used to select candidate parameters.
    """
    from importlib.machinery import SourceFileLoader
    import importlib.util
    from pathlib import Path

    if cost_bps < 0 or not np.isfinite(cost_bps):
        raise ValueError("invalid cost_bps")
    folds = expanding_folds(len(data), train_bars, test_bars)
    candidates = [(int(k), float(r)) for k, r in product(neighbors_grid, regime_grid)
                  if int(k) >= 1 and 0 <= float(r) <= 1]
    if not candidates or max_bars_back < 4:
        raise ValueError("invalid Lorentzian parameter grid")
    if strategy_factory is None:
        path = Path(__file__).with_name("Código Principal")
        loader = SourceFileLoader("lorentzian_wf", str(path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        strategy_factory = module.LorentzianClassificationStrategy

    targets = {}
    for k, regime in candidates:
        strategy = strategy_factory(
            neighbors_count=k,
            max_bars_back=max_bars_back,
            regime_threshold=regime,
        )
        signals = strategy.generate_signals(data)
        if not signals.index.equals(data.index):
            raise ValueError("strategy index mismatch")
        targets[(k, regime)] = signals["position"]

    records = []
    for start, end in folds:
        # Equal train-score and test-score conventions for both strategies.
        scores = {
            params: score_fold(fold_returns(data, targets[params], 0, start, cost_bps))
            for params in candidates
        }
        best = max(candidates, key=lambda params: (scores[params], -candidates.index(params)))
        ledger = fold_returns(data, targets[best], start, end, cost_bps)
        records.append({
            "train_end": data.index[start - 1],
            "test_start": data.index[start],
            "test_end": data.index[end - 1],
            "neighbors_count": best[0],
            "regime_threshold": best[1],
            "max_bars_back": max_bars_back,
            "train_log_growth": scores[best],
            "test_return": float(ledger["equity"].iloc[-1] - 1),
            "test_gross_return": float((1 + ledger["gross_return"]).prod() - 1),
            "test_cost_sum": float(ledger["cost"].sum()),
            "test_long_bars": int((ledger["executed_position"] == 1).sum()),
            "test_short_bars": int((ledger["executed_position"] == -1).sum()),
            "test_flat_bars": int((ledger["executed_position"] == 0).sum()),
            "test_turnover": float(ledger["turnover"].sum()),
            "test_bars": end - start,
        })
    return pd.DataFrame.from_records(records)
