"""Independent, frozen-rule external validation on three UNSEEN crypto assets.

Rules are not optimized on these assets. Only ETH/USDT 2022-2023 contributed
to the candidate selection, 2024 to validation; ETH/USDT 2025 and the other
coins' histories are an exploratory external check. This is NOT a live forecast.

Spot, long/USDT only, zero cash yield, no margin or lookahead:
signal at close[t] -> execute open[t+1] -> earn open[t+1]:open[t+2].
Trading fee+slippage proxy 10/20/40 bps per turnover unit, no funding/tax.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

RULES = ("30d-price-above-SMA", "14d-momentum-over-10pct",
         "14d-high-breakout-3d-low-exit", "buy-and-hold", "cash")
SYMBOLS = ("BTC-USDT", "SOL-USDT", "ADA-USDT")


def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("duplicate/unsorted timestamps")
    if len(df) != 35064 or not df.index.equals(pd.date_range(
        "2022-01-01", "2025-12-31 23:00", freq="h", tz="UTC"
    )):
        raise ValueError("incomplete hourly 2022-2025 candles")
    if not np.isfinite(df[["open", "high", "low", "close", "volume"]]).all().all():
        raise ValueError("nonfinite price or volume")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("nonpositive prices")
    return df


def decision(data, rule):
    close = data["close"]
    n = len(data)
    if rule == "30d-price-above-SMA":
        ma = close.rolling(24*30, min_periods=24*30).mean()
        target = (close > ma).to_numpy(dtype=np.int8)
    elif rule == "14d-momentum-over-10pct":
        target = (close.pct_change(24*14) > .10).to_numpy(dtype=np.int8)
    elif rule == "14d-high-breakout-3d-low-exit":
        entry = (close > data["high"].shift(1).rolling(
            24*14, min_periods=24*14).max()).to_numpy()
        exit_ = (close < data["low"].shift(1).rolling(
            24*3, min_periods=24*3).min()).to_numpy()
        target = np.zeros(n, dtype=np.int8)
        held = False
        for t in range(n):
            if held and exit_[t]:
                held = False
            if not held and entry[t]:
                held = True
            target[t] = int(held)
    elif rule == "buy-and-hold":
        target = np.ones(n, dtype=np.int8)
    elif rule == "cash":
        target = np.zeros(n, dtype=np.int8)
    else:
        raise ValueError(rule)
    return target


def evaluate(data, target, start, end, bps=10):
    if bps < 0 or not 0 <= start < end <= len(data):
        raise ValueError("invalid cost/window")
    target = np.asarray(target, dtype=np.int8)
    if len(target) != len(data) or not np.isin(target, [0, 1]).all():
        raise ValueError("positions must be binary and fully aligned")
    # Shift decisions by a full hour, just like backtest_next_open().
    executed = np.r_[0, target[:-1]][start:end]
    transitions = np.abs(executed-np.r_[0, executed[:-1]])
    prices = data["open"].to_numpy(dtype=float)
    future = np.r_[prices[1:]/prices[:-1]-1, 0.][start:end]
    gains = executed*future - transitions*bps/10000.
    wealth = np.cumprod(1.+gains)
    dd = wealth/np.maximum.accumulate(np.r_[1., wealth])[-len(wealth):]-1
    return {
        "net_return": float(wealth[-1]-1),
        "gross_return": float(np.prod(1+executed*future)-1),
        "max_drawdown": float(dd.min()),
        "turnover": int(transitions.sum()),
        "exposure": float(executed.mean()),
        "annualized_sharpe": float(
            np.sqrt(24*365)*gains.mean()/gains.std()) if gains.std() else 0.,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="research_data")
    args = parser.parse_args()
    root = Path(args.input_dir)
    all_results, manifest = [], {}
    for symbol in SYMBOLS:
        file = root / f"{symbol.replace('-', '_')}_1h.csv"
        data = load_data(file)
        manifest[symbol] = {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                            "rows": len(data),
                            "first": str(data.index[0]),
                            "last": str(data.index[-1])}
        for rule in RULES:
            target = decision(data, rule)
            for year in range(2022, 2026):
                indices = np.flatnonzero(data.index.year == year)
                for bps in (0, 10, 20, 40):
                    result = evaluate(data, target, int(indices[0]),
                                      int(indices[-1]+1), bps)
                    all_results.append(dict(asset=symbol, rule=rule, year=year,
                                            cost_bps=bps, **result))
    report = pd.DataFrame(all_results)
    report.to_csv(root / "external_validation.csv", index=False)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("\n2022-2025 FROZEN RULES: outside ETH, no parameter tuning")
    print(report[report.cost_bps == 10].pivot_table(
        index=["asset", "rule"], columns="year",
        values="net_return").to_string(float_format=lambda v:f"{v:+.2%}"))
    print("See external_validation.csv, manifest.json for reproducibility.")


if __name__ == "__main__":
    main()
