"""Run the ETH-USDT 1h RSI vs Lorentzian walk-forward experiment.

Example:
 python run_walk_forward.py eth_usdt_1h.csv --train-bars 1440 --test-bars 336 --cost-bps 10 --output-dir results

Caution: Lorentzian kNN is a Python loop; long histories can take hours.
Results are research-only, not evidence of profitable trading.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from comparison import summarize_walk_forward
from run_evaluation import load_ohlc_csv
from walk_forward import optimize_rsi_walk_forward, optimize_lorentzian_walk_forward


def run_experiment(data, train_bars=1440, test_bars=336, cost_bps=10.0,
                   neighbors_grid=(4, 8), regime_grid=(0.15, 0.25),
                   rsi_periods=(7, 14, 21), rsi_oversold=(25., 30., 35.),
                   rsi_overbought=(65., 70., 75.), max_bars_back=500,
                   lorentzian_factory=None):
    if not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError("data must have a strictly increasing unique index")
    if not np.isfinite(data[["open", "high", "low", "close"]].to_numpy()).all():
        raise ValueError("nonfinite OHLC data")
    if len(data) < train_bars + test_bars:
        raise ValueError("insufficient data for one full test fold")
    rsi = optimize_rsi_walk_forward(
        data, train_bars, test_bars, cost_bps, periods=rsi_periods,
        oversold_levels=rsi_oversold, overbought_levels=rsi_overbought
    )
    lorentzian = optimize_lorentzian_walk_forward(
        data, train_bars, test_bars, cost_bps, neighbors_grid=neighbors_grid,
        regime_grid=regime_grid, max_bars_back=max_bars_back,
        strategy_factory=lorentzian_factory
    )
    comparison = summarize_walk_forward(rsi, lorentzian, data)
    config = {
        "asset": "ETH-USDT", "timeframe": "1h", "source": "KuCoin spot",
        "train_bars": train_bars, "test_bars": test_bars,
        "cost_bps_per_turnover_unit": cost_bps,
        "rsi_periods": list(rsi_periods),
        "rsi_oversold": list(rsi_oversold),
        "rsi_overbought": list(rsi_overbought),
        "lorentzian_neighbors": list(neighbors_grid),
        "lorentzian_regime": list(regime_grid),
        "lorentzian_max_bars_back": max_bars_back,
        "first_bar": str(data.index[0]), "last_bar": str(data.index[-1]),
        "n_bars": len(data),
        "caveats": [
            "Folds start flat; final fold positions are not force-closed.",
            "No slippage, spread, borrowing, funding, or market impact.",
            "Drawdown measured only at fold boundaries.",
            "Parameter grid sizes differ; no equal-search-budget claim.",
            "Last partial test fold may have fewer bars.",
        ],
    }
    return {"rsi": rsi, "lorentzian": lorentzian,
            "comparison": comparison, "config": config}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--train-bars", type=int, default=1440)
    parser.add_argument("--test-bars", type=int, default=336)
    parser.add_argument("--cost-bps", type=float, default=10.)
    parser.add_argument("--max-bars-back", type=int, default=500)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    data = load_ohlc_csv(args.csv)
    result = run_experiment(data, train_bars=args.train_bars,
                            test_bars=args.test_bars, cost_bps=args.cost_bps,
                            max_bars_back=args.max_bars_back)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key in ("rsi", "lorentzian", "comparison"):
        result[key].to_csv(out / f"{key}.csv", index=False)
    (out / "config.json").write_text(
        json.dumps(result["config"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print(result["comparison"].to_string(index=False))
    print(f"Saved fold reports and configuration in {out}")


if __name__ == "__main__":
    main()
