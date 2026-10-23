"""Leakage-aware descriptive analysis of Kraken OOB order-book snapshots.

Usage: python research/analyze_kraken_oob.py research_data/kraken_oob/snapshots.csv
Works with concatenated CSV exports; does not claim trade execution.
"""
import argparse
import csv
from datetime import datetime, timezone
from statistics import mean, median


def load(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("error") or not row.get("best_bid") or not row.get("best_ask"):
                continue
            try:
                t = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
                if t.tzinfo is None:
                    continue
                bid, ask = float(row["best_bid"]), float(row["best_ask"])
                imbalance = float(row["imbalance_10"])
                if not (0 < bid < ask and -1 <= imbalance <= 1):
                    continue
                rows[t.astimezone(timezone.utc)] = (bid, ask, imbalance)
            except (ValueError, KeyError, OverflowError):
                continue
    return sorted((t, *v) for t, v in rows.items())


def analyze(rows, horizon_seconds=60, tolerance_seconds=20):
    """Match next observation in [horizon, horizon+tolerance] only.

    Never bridges gaps between separate 15-minute collection windows.
    Return is future mid-price change; net proxy deducts entry+exit
    quoted half-spreads (not fees, impact or fills).
    """
    pairs = []
    for i, (t, bid, ask, imbalance) in enumerate(rows):
        target = t.timestamp() + horizon_seconds
        for later in rows[i+1:]:
            future_t, future_bid, future_ask, _ = later
            delta = future_t.timestamp() - target
            if delta < 0:
                continue
            if delta > tolerance_seconds:
                break
            mid = (bid + ask) / 2
            future_mid = (future_bid + future_ask) / 2
            gross = future_mid / mid - 1
            # Long buy at current ask and sell at future bid; no fees.
            executable_proxy = future_bid / ask - 1
            pairs.append((imbalance, gross, executable_proxy))
            break
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--tolerance", type=int, default=20)
    args = p.parse_args()
    if args.horizon <= 0 or args.tolerance < 0:
        p.error("horizon must be positive; tolerance nonnegative")
    rows = load(args.csv_path)
    pairs = analyze(rows, args.horizon, args.tolerance)
    print(f"Valid unique snapshots: {len(rows)}; matched forward observations: {len(pairs)}")
    if not pairs:
        print("Insufficient matched observations; no predictive inference.")
        return
    spreads = [100 * (ask-bid)/((ask+bid)/2) for _, bid, ask, _ in rows]
    print(f"Median quoted spread: {median(spreads):.3f}%")
    print(f"Mean forward mid return: {100*mean(x[1] for x in pairs):.4f}%")
    print(f"Mean buy-ask/sell-bid proxy: {100*mean(x[2] for x in pairs):.4f}% (before fees)")
    for label, selected in (
        ("imbalance > +0.2", [x for x in pairs if x[0] > .2]),
        ("imbalance < -0.2", [x for x in pairs if x[0] < -.2]),
    ):
        if selected:
            print(f"{label}: n={len(selected)} mean_mid_return={100*mean(x[1] for x in selected):.4f}%")
        else:
            print(f"{label}: n=0")
    print("Exploratory only: overlapping returns, nonrandom windows, and no executable fill model.")


if __name__ == "__main__":
    main()
