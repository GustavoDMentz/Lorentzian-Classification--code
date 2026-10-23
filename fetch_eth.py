"""Download historical ETH-USD hourly candles from Coinbase Exchange public API.

No API key required. Coinbase returns at most 300 candles per request.
Usage: python fetch_eth.py --start 2023-01-01 --end 2024-01-01 --output eth_usd_1h.csv
Dates are UTC and end-exclusive. Missing hourly candles fail closed.
"""
import argparse
import csv
import datetime as dt
import json
import time
from urllib.parse import urlencode
from urllib.request import urlopen

BASE_URL = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
HOUR = 3600
PAGE_HOURS = 299


def parse_utc(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fetch_hourly(start, end, opener=urlopen, pause=time.sleep):
    start_dt, end_dt = parse_utc(start), parse_utc(end)
    first, last = int(start_dt.timestamp()), int(end_dt.timestamp())
    if first % HOUR or last % HOUR or first >= last:
        raise ValueError("start/end must be UTC hour boundaries with start < end")
    observations = {}
    for page_start in range(first, last, PAGE_HOURS * HOUR):
        page_end = min(page_start + PAGE_HOURS * HOUR, last)
        query = urlencode({
            "start": dt.datetime.fromtimestamp(page_start, dt.timezone.utc).isoformat(),
            "end": dt.datetime.fromtimestamp(page_end, dt.timezone.utc).isoformat(),
            "granularity": HOUR,
        })
        request_url = BASE_URL + "?" + query
        # A bounded retry handles transient HTTP/network errors.
        for attempt in range(3):
            try:
                with opener(request_url, timeout=30) as response:
                    rows = json.load(response)
                if not isinstance(rows, list):
                    raise ValueError("unexpected Coinbase candle response")
                break
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                pause(2 ** attempt)
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                raise ValueError("invalid candle record")
            stamp, low, high, open_, close, volume = row[:6]
            stamp = int(stamp)
            if first <= stamp < last and stamp % HOUR == 0:
                observations[stamp] = (float(open_), float(high), float(low),
                                       float(close), float(volume))
        pause(0.35)
    expected = range(first, last, HOUR)
    missing = [stamp for stamp in expected if stamp not in observations]
    if missing:
        first_missing = dt.datetime.fromtimestamp(missing[0], dt.timezone.utc).isoformat()
        raise ValueError(f"{len(missing)} missing hourly candles; first: {first_missing}")
    return [
        (dt.datetime.fromtimestamp(stamp, dt.timezone.utc).isoformat(),
         *observations[stamp])
        for stamp in expected
    ]


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC ISO date, inclusive")
    parser.add_argument("--end", required=True, help="UTC ISO date, exclusive")
    parser.add_argument("--output", default="eth_usd_1h.csv")
    args = parser.parse_args()
    rows = fetch_hourly(args.start, args.end)
    save_csv(rows, args.output)
    print(f"Saved {len(rows)} hourly ETH-USD candles to {args.output}")


if __name__ == "__main__":
    main()
