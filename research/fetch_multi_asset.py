"""Fetch unadjusted KuCoin spot 1-hour OHLCV without API credentials.

No missing timestamps are silently forward-filled. Outputs one CSV per asset,
UTC / end-exclusive. Intentionally bounded to the predeclared three symbols.
"""
import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://api.kucoin.com/api/v1/market/candles"
SYMBOLS = ("BTC-USDT", "SOL-USDT", "ADA-USDT")
HOUR = 3600


def utc_epoch(iso):
    obj = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if obj.tzinfo is None:
        obj = obj.replace(tzinfo=dt.timezone.utc)
    return int(obj.astimezone(dt.timezone.utc).timestamp())


def fetch(symbol, start, end, opener=urlopen, pause=time.sleep):
    if symbol not in SYMBOLS:
        raise ValueError("symbol not in predeclared verification universe")
    first, last = utc_epoch(start), utc_epoch(end)
    if first >= last or first % HOUR or last % HOUR:
        raise ValueError("UTC start/end must be aligned hours, start < end")
    candles = {}
    for left in range(first, last, 1499 * HOUR):
        right = min(left + 1499 * HOUR, last)
        request = Request(
            API + "?" + urlencode({"symbol": symbol, "type": "1hour",
                                  "startAt": left, "endAt": right - 1}),
            headers={"User-Agent": "causal-cross-asset-research/1.0",
                     "Accept": "application/json"},
        )
        for attempt in range(5):
            try:
                with opener(request, timeout=35) as stream:
                    payload = json.load(stream)
                if payload.get("code") != "200000" or not isinstance(payload.get("data"), list):
                    raise ValueError(f"invalid response for {symbol}")
                break
            except (HTTPError, URLError, OSError, TimeoutError) as exc:
                if attempt == 4 or (isinstance(exc, HTTPError)
                                    and exc.code not in (429, 500, 502, 503, 504)):
                    raise
                pause(2 ** attempt)
        for row in payload["data"]:
            if len(row) < 6:
                raise ValueError("short candle")
            stamp = int(row[0])
            if not first <= stamp < last:
                continue
            open_, close, high, low, volume = map(float, row[1:6])
            if not 0 < low <= min(open_, close) <= max(open_, close) <= high or volume < 0:
                raise ValueError(f"invalid OHLCV {symbol} at {stamp}")
            candles[stamp] = (open_, high, low, close, volume)
        pause(0.25)
    stamps = range(first, last, HOUR)
    missing = [stamp for stamp in stamps if stamp not in candles]
    if missing:
        sample = dt.datetime.fromtimestamp(missing[0], dt.timezone.utc).isoformat()
        raise ValueError(f"{symbol}: {len(missing)} missing hourly candles, first {sample}")
    return [(dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(), *candles[t])
            for t in stamps]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--output-dir", default="research_data")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for symbol in SYMBOLS:
        rows = fetch(symbol, args.start, args.end)
        path = output / f"{symbol.replace('-', '_')}_1h.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["date", "open", "high", "low", "close", "volume"])
            writer.writerows(rows)
        print(f"{symbol}: {len(rows)} contiguous hourly bars saved to {path}", flush=True)


if __name__ == "__main__":
    main()
