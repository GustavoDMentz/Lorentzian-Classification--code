"""Download KuCoin spot ETH-USDT hourly candles (UTC, end-exclusive).

KuCoin GET /api/v1/market/candles, type=1hour, max 1500 rows.
Response rows: [timestamp, open, close, high, low, volume, turnover].
No keys needed. Missing hourly bars fail closed rather than being fabricated.
"""
import argparse
import csv
import datetime as dt
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

API = "https://api.kucoin.com/api/v1/market/candles"
HOUR = 3600
PAGE_HOURS = 1499


def utc_timestamp(value):
    date = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        date = date.replace(tzinfo=dt.timezone.utc)
    return int(date.astimezone(dt.timezone.utc).timestamp())


def fetch_hourly(start, end, opener=urlopen, pause=time.sleep):
    first, last = utc_timestamp(start), utc_timestamp(end)
    if first % HOUR or last % HOUR or first >= last:
        raise ValueError("start/end must be UTC hour boundaries with start < end")
    candles = {}
    for left in range(first, last, PAGE_HOURS * HOUR):
        right = min(left + PAGE_HOURS * HOUR, last)
        url = API + "?" + urlencode({
            "symbol": "ETH-USDT", "type": "1hour",
            "startAt": left, "endAt": right - 1,
        })
        request = Request(url, headers={"User-Agent": "eth-walk-forward-research/1.0",
                                        "Accept": "application/json"})
        for attempt in range(4):
            try:
                with opener(request, timeout=30) as response:
                    payload = json.load(response)
                if payload.get("code") != "200000" or not isinstance(payload.get("data"), list):
                    raise ValueError("unexpected KuCoin response")
                break
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                if attempt == 3 or (isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504)):
                    raise
                pause(2 ** attempt)
        for row in payload["data"]:
            if not isinstance(row, list) or len(row) < 7:
                raise ValueError("invalid KuCoin candle record")
            stamp = int(row[0])
            if first <= stamp < last and stamp % HOUR == 0:
                open_, close, high, low, volume = map(float, row[1:6])
                if not (0 < low <= min(open_, close) <= max(open_, close) <= high):
                    raise ValueError(f"invalid OHLC at {stamp}")
                candles[stamp] = (open_, high, low, close, volume)
        pause(0.25)
    missing = [t for t in range(first, last, HOUR) if t not in candles]
    if missing:
        stamp = dt.datetime.fromtimestamp(missing[0], dt.timezone.utc).isoformat()
        raise ValueError(f"{len(missing)} missing ETH-USDT hourly candles; first: {stamp}")
    return [(dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(), *candles[t])
            for t in range(first, last, HOUR)]


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default="eth_usdt_1h.csv")
    args = parser.parse_args()
    rows = fetch_hourly(args.start, args.end)
    save_csv(rows, args.output)
    print(f"Saved {len(rows)} KuCoin ETH-USDT hourly candles to {args.output}")


if __name__ == "__main__":
    main()
