"""Read-only Kraken OOB/USD market observer. No credentials, orders, or trading.

Usage: python research/kraken_oob_observer.py --samples 30 --interval 10
All timestamps UTC. Raw public API responses are preserved as JSONL.
"""
import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://api.kraken.com/0/public/"
FIELDS = ("timestamp_utc", "pair", "best_bid", "best_ask", "spread_pct",
          "bid_depth_10", "ask_depth_10", "imbalance_10", "last_price",
          "volume_24h", "error")


def get(endpoint, **params):
    url = BASE + endpoint + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "research-oob-observer/1.0"})
    with urllib.request.urlopen(req, timeout=15) as response:
        payload = json.load(response)
    if payload.get("error"):
        raise RuntimeError("; ".join(payload["error"]))
    return payload["result"]


def resolve_pair():
    """Resolve the *actual* Kraken market; do not assume OOBUSD exists."""
    pairs = get("AssetPairs")
    candidates = []
    for key, info in pairs.items():
        ws = str(info.get("wsname", "")).upper()
        alt = str(info.get("altname", "")).upper()
        base = str(info.get("base", "")).upper()
        quote = str(info.get("quote", "")).upper()
        if (ws == "OOB/USD" or alt == "OOBUSD" or
                (base in ("OOB", "XOOB") and quote in ("USD", "ZUSD"))):
            if info.get("status", "online") == "online":
                candidates.append((key, info))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one active OOB/USD market, found {len(candidates)}; verify Kraken AssetPairs")
    return candidates[0][0]


def summarize(pair, book, ticker, timestamp):
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids or not asks:
        raise ValueError("Empty order book")
    bid, ask = float(bids[0][0]), float(asks[0][0])
    if bid <= 0 or ask <= 0 or bid >= ask:
        raise ValueError("Invalid or crossed order book")
    bd = sum(float(x[0]) * float(x[1]) for x in bids[:10])
    ad = sum(float(x[0]) * float(x[1]) for x in asks[:10])
    return dict(timestamp_utc=timestamp, pair=pair, best_bid=bid, best_ask=ask,
                spread_pct=100 * (ask-bid)/((ask+bid)/2),
                bid_depth_10=bd, ask_depth_10=ad,
                imbalance_10=(bd-ad)/(bd+ad) if bd+ad else 0,
                last_price=ticker["c"][0], volume_24h=ticker["v"][1], error="")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--interval", type=float, default=10)
    p.add_argument("--output-dir", default="research_data/kraken_oob")
    args = p.parse_args()
    if not 1 <= args.samples <= 100000 or args.interval < 5:
        p.error("samples must be 1..100000 and interval >= 5 seconds")
    pair = resolve_pair()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, raw_path = out / "snapshots.csv", out / "raw.jsonl"
    with csv_path.open("a", newline="", encoding="utf-8") as csvfile, raw_path.open("a", encoding="utf-8") as raw:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
        if csvfile.tell() == 0:
            writer.writeheader()
        for i in range(args.samples):
            stamp = datetime.now(timezone.utc).isoformat()
            try:
                depth_result = get("Depth", pair=pair, count=25)
                ticker_result = get("Ticker", pair=pair)
                book = next(iter(depth_result.values()))
                ticker = next(iter(ticker_result.values()))
                row = summarize(pair, book, ticker, stamp)
                raw.write(json.dumps({"timestamp_utc": stamp, "pair": pair,
                                      "depth": book, "ticker": ticker}, ensure_ascii=False) + "\n")
                print(f"{stamp} spread={row['spread_pct']:.3f}% imbalance={row['imbalance_10']:+.3f}", flush=True)
            except (ValueError, RuntimeError, OSError, KeyError, IndexError) as exc:
                row = dict.fromkeys(FIELDS, "")
                row.update(timestamp_utc=stamp, pair=pair, error=str(exc))
                print(f"{stamp} ERROR: {exc}", flush=True)
            writer.writerow(row)
            csvfile.flush()
            raw.flush()
            if i + 1 < args.samples:
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
