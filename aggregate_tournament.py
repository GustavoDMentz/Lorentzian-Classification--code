"""Aggregate per-regime indicator leaderboards without conflating time horizons."""
import argparse
from pathlib import Path
import pandas as pd


def aggregate(root: Path, filename: str = "indicator_leaderboard.csv") -> pd.DataFrame:
    files = sorted(root.glob(f"*/{filename}"))
    if not files:
        raise ValueError("no tournament leaderboards found")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["period"] = path.parent.name
        frames.append(df)
    all_periods = pd.concat(frames, ignore_index=True)
    # Equal weighting of regimes (not an investable compounded portfolio).
    report = all_periods.groupby("strategy", as_index=False).agg(
        regimes=("period", "nunique"),
        mean_regime_return=("compounded_return", "mean"),
        worst_regime_return=("compounded_return", "min"),
        positive_regimes=("compounded_return", lambda s: int((s > 0).sum())),
        total_turnover=("turnover", "sum"),
    )
    return report.sort_values(["mean_regime_return", "worst_regime_return"],
                              ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = aggregate(args.root)
    report.to_csv(args.root / "indicator_cross_regime.csv", index=False)
    print("\nCROSS-REGIME INDICATOR LEADERBOARD (equal-weight descriptive mean):")
    print(report.to_string(index=False))
    systems = aggregate(args.root, "stateful_systems_leaderboard.csv")
    systems.to_csv(args.root / "stateful_cross_regime.csv", index=False)
    print("\nCROSS-REGIME STATEFUL SYSTEMS (fixed rules, no optimization):")
    print(systems.to_string(index=False))
    adaptive = aggregate(args.root, "adaptive_leaderboard.csv")
    adaptive.to_csv(args.root / "adaptive_cross_regime.csv", index=False)
    print("\nCROSS-REGIME ADAPTIVE SELECTOR:")
    print(adaptive.to_string(index=False))


if __name__ == "__main__":
    main()
