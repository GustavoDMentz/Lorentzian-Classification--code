"""Run a predeclared chronological holdout on local OHLC CSV data.

Usage:
  python run_evaluation.py prices.csv --split 2024-01-01 --cost-bps 10
The CSV must have date, open, high, low, close columns.
No data is downloaded; no parameters are optimized on the holdout.
"""
import argparse
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import evaluate_holdout


def load_strategy():
    path = Path(__file__).with_name("Código Principal")
    loader = SourceFileLoader("lorentzian_strategy", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.LorentzianClassificationStrategy


def load_ohlc_csv(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    data["date"] = pd.to_datetime(data["date"], errors="raise", utc=True)
    if data["date"].isna().any() or data["date"].duplicated().any():
        raise ValueError("date values must be unique and nonmissing")
    data = data.set_index("date").sort_index()
    for col in ("open", "high", "low", "close"):
        data[col] = pd.to_numeric(data[col], errors="raise")
    ohlc = data[["open", "high", "low", "close"]]
    if not np.isfinite(ohlc.to_numpy()).all() or (ohlc <= 0).any():
        raise ValueError("OHLC values must be finite and positive")
    if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high must be at least open, close, and low")
    if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low must be at most open, close, and high")
    return data


def holdout_report(data: pd.DataFrame, split: str, cost_bps: float = 10.0,
                   strategy=None) -> dict:
    cutoff = pd.Timestamp(split)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    split_index = int(data.index.searchsorted(cutoff, side="left"))
    if split_index < 1 or split_index >= len(data):
        raise ValueError("split date must leave nonempty train and holdout periods")
    if strategy is None:
        strategy = load_strategy()()
    result = evaluate_holdout(strategy, data, split_index, cost_bps)
    test = result["test"]
    return {
        "train_bars": split_index,
        "holdout_bars": len(test),
        "holdout_start": str(test.index[0]),
        "holdout_end": str(test.index[-1]),
        "cost_bps_per_position_unit": cost_bps,
        "strategy_return_pct": round((test["holdout_equity"].iloc[-1] - 1) * 100, 4),
        "buy_hold_return_pct": round((test["holdout_buy_hold_equity"].iloc[-1] - 1) * 100, 4),
        "turnover_units": round(float(test["turnover"].sum()), 4),
        "note": "Descriptive holdout only; no parameter tuning, financing, slippage, or spread model.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Local CSV with date,open,high,low,close")
    parser.add_argument("--split", required=True, help="Predeclared ISO holdout start date")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    import json
    report = holdout_report(load_ohlc_csv(args.csv), args.split, args.cost_bps)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
