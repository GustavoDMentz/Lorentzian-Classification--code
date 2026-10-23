"""Summarize aligned walk-forward folds without arithmetic-return aggregation."""
import numpy as np
import pandas as pd


def summarize_walk_forward(rsi: pd.DataFrame, lorentzian: pd.DataFrame,
                           data: pd.DataFrame) -> pd.DataFrame:
    """Compound net fold returns and compare same-window buy-and-hold.

    Buy-and-hold is measured from each test fold's first open to last open,
    consistent with the available next-open P&L horizon. Fold results
    include a flat first test open and may exclude a final position close.
    """
    required = {"test_start", "test_end", "test_return", "test_bars"}
    for name, frame in (("RSI", rsi), ("Lorentzian", lorentzian)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} has no test folds")
    if len(rsi) != len(lorentzian):
        raise ValueError("fold counts differ")
    for col in ("test_start", "test_end", "test_bars"):
        if not rsi[col].reset_index(drop=True).equals(
            lorentzian[col].reset_index(drop=True)
        ):
            raise ValueError(f"fold alignment differs: {col}")

    hold = []
    for start, end in zip(rsi["test_start"], rsi["test_end"]):
        first = data.loc[start, "open"]
        last = data.loc[end, "open"]
        if not np.isfinite(first) or not np.isfinite(last) or first <= 0:
            raise ValueError("invalid buy-and-hold open prices")
        hold.append(float(last / first - 1))
    report = []
    for name, returns in (
        ("RSI optimized", rsi["test_return"].to_numpy(dtype=float)),
        ("Lorentzian optimized", lorentzian["test_return"].to_numpy(dtype=float)),
        ("Buy and hold", np.asarray(hold, dtype=float)),
    ):
        if not np.isfinite(returns).all() or (returns < -1).any():
            raise ValueError("invalid fold returns")
        equity = np.cumprod(1 + returns)
        peaks = np.maximum.accumulate(np.r_[1., equity])
        drawdown = equity / peaks[1:] - 1
        report.append({
            "strategy": name,
            "folds": len(returns),
            "compounded_return": float(equity[-1] - 1),
            "positive_folds": int((returns > 0).sum()),
            "worst_fold_return": float(returns.min()),
            "fold_boundary_max_drawdown": float(drawdown.min()),
        })
    return pd.DataFrame(report)
