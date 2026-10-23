"""Frozen ETH-trained ML strategy applied unchanged to BTC and SOL KuCoin spot.

Training ETH 2022-2023, selection ETH 2024, holdout ETH/BTC/SOL 2025.
No retraining on BTC/SOL. Signal at UTC daily close[t] -> trade next
day open[t+1] -> mark open[t+2]. Zero leverage, long/cash, cash earns zero.
This reproduces the exploratory HistGradientBoosting configuration.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

SYMBOLS = ("ETH-USDT", "BTC-USDT", "SOL-USDT")
FEATURES = None
THRESHOLD = 0.50
COSTS_BPS = (0, 10, 20, 40, 60)


def hourly_to_daily(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    expected = pd.date_range(df.index.min(), df.index.max(),
                             freq="h", tz="UTC")
    if df.index.max() < pd.Timestamp("2026-01-03", tz="UTC"):
        raise ValueError("2026 validation requires data beyond 2025")
    if not df.index.equals(expected):
        raise ValueError(f"{path}: hourly candles missing, duplicated or out of order")
    required = ["open", "high", "low", "close", "volume"]
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError("non-finite OHLCV")
    if (df[required[:4]] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError("invalid prices or volume")
    return df.resample("1D").agg({"open":"first", "high":"max", "low":"min",
                                   "close":"last", "volume":"sum"})


def make_features(d):
    c = d["close"]
    feat = pd.DataFrame(index=d.index)
    for n in (1, 3, 7, 14, 30):
        feat[f"ret_{n}"] = c.pct_change(n)
    for n in (7, 14, 30):
        feat[f"vol_{n}"] = c.pct_change().rolling(n).std()
        feat[f"price_ma_{n}"] = c / c.rolling(n).mean() - 1
    feat["range"] = (d.high - d.low) / c
    feat["volume_rel"] = np.log(d.volume / d.volume.rolling(30).mean())
    feat["volume_change"] = d.volume.pct_change(7)
    feat["weekday"] = d.index.dayofweek / 6
    forward = d.open.shift(-2) / d.open.shift(-1) - 1
    ok = feat.replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & forward.notna()
    return feat.loc[ok], forward.loc[ok]


def evaluate(signal, forward, year, bps):
    # Decisions at close[t]; each decision buys/sells at open[t+1].
    # Costs are charged on that entry/exit, not on prior day's returns.
    sig = signal.astype(int)
    turn = sig.diff().abs().fillna(sig.iloc[0])
    mask = sig.index.year == year
    gains = sig.loc[mask] * forward.loc[mask] - turn.loc[mask] * bps / 10000
    equity = (1 + gains).cumprod()
    drawdown = equity / np.maximum.accumulate(np.r_[1., equity.to_numpy()])[-len(equity):] - 1
    return dict(net_return=float(equity.iloc[-1] - 1),
                max_drawdown=float(drawdown.min()),
                turnover=int(turn.loc[mask].sum()),
                exposure=float(sig.loc[mask].mean()),
                observations=int(mask.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="research_data")
    args = parser.parse_args()
    root = Path(args.input_dir)
    dataset, checksums, daily_data = {}, {}, {}
    for symbol in SYMBOLS:
        path = root / f"{symbol.replace('-', '_')}_1h.csv"
        checksums[symbol] = hashlib.sha256(path.read_bytes()).hexdigest()
        daily_data[symbol] = hourly_to_daily(path)
        dataset[symbol] = make_features(daily_data[symbol])

    train_x, train_forward = dataset["ETH-USDT"]
    train_mask = train_x.index < pd.Timestamp("2023-12-30", tz="UTC")
    train_y = (train_forward.loc[train_mask] > 0).astype(int)
    model = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=7, l2_regularization=10,
        learning_rate=.04, min_samples_leaf=30, random_state=42
    )
    model.fit(train_x.loc[train_mask], train_y)

    rows = []
    for symbol in SYMBOLS:
        features, forward = dataset[symbol]
        prob = pd.Series(model.predict_proba(features)[:, 1],
                         index=features.index)
        signals = {
            "frozen_eth_ml": (prob >= THRESHOLD).astype(int),
            "ma30": (daily_data[symbol].close.reindex(features.index) >
                     daily_data[symbol].close.rolling(30).mean().reindex(features.index)).astype(int),
            "buy_hold": pd.Series(1, index=features.index),
            "cash": pd.Series(0, index=features.index),
        }
        for strategy, signal in signals.items():
            for year in (2024, 2025, 2026):
                for cost in COSTS_BPS:
                    rows.append(dict(asset=symbol, strategy=strategy, year=year,
                                     cost_bps=cost,
                                     **evaluate(signal, forward, year, cost)))
        pd.DataFrame({"date":features.index.astype(str),
                      "prob_up":prob.to_numpy(),
                      "position":signals["frozen_eth_ml"].to_numpy()}).to_csv(
                          root / f"{symbol.replace('-', '_')}_ml_signals.csv", index=False)

    report = pd.DataFrame(rows)
    report.to_csv(root / "frozen_eth_ml_cross_asset.csv", index=False)
    manifest = dict(
        trained_on="ETH-USDT KuCoin daily 2022-2023",
        selected_on="ETH-USDT 2024 (exploratory model selection)",
        externally_tested_on="BTC-USDT and SOL-USDT 2024-2026; no refitting",
        held_out="ETH-USDT 2025-2026 (2025 inspected before 2026 run)",
        model="HistGradientBoostingClassifier",
        model_parameters=model.get_params(),
        threshold=THRESHOLD, features=list(train_x.columns),
        data_sha256=checksums,
        caveat="Research only; slippage, liquidity and model-selection bias remain.")
    (root / "frozen_eth_ml_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)+"\n", encoding="utf-8")
    print("\nFROZEN ETH MODEL vs BENCHMARKS (10bps, no refitting)")
    print(report[report.cost_bps == 10].pivot_table(
        index=["asset", "strategy"], columns="year", values="net_return"
    ).to_string(float_format=lambda x:f"{x:+.2%}"))
    print("\nFull costs and drawdowns: frozen_eth_ml_cross_asset.csv")


if __name__ == "__main__":
    main()
