"""Deterministic tests for Coinbase hourly history pagination."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import datetime as dt

from fetch_eth import fetch_hourly, save_csv


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


class DownloaderTests(unittest.TestCase):
    def fake_opener(self, url, timeout):
        query = parse_qs(urlsplit(url).query)
        start = int(dt.datetime.fromisoformat(query["start"][0]).timestamp())
        end = int(dt.datetime.fromisoformat(query["end"][0]).timestamp())
        # Mock Coinbase's descending timestamp order and inclusive end.
        candles = [[t, 90., 110., 100., 105., 3.]
                   for t in range(end, start - 1, -3600)]
        return FakeResponse(json.dumps(candles).encode())

    def test_pagination_dedup_and_utc_order(self):
        rows = fetch_hourly("2024-01-01", "2024-01-26", opener=self.fake_opener,
                            pause=lambda _: None)
        self.assertEqual(len(rows), 25 * 24)
        self.assertTrue(rows[0][0].startswith("2024-01-01T00:00:00"))
        self.assertTrue(rows[-1][0].startswith("2024-01-25T23:00:00"))
        self.assertEqual(rows[0][1:5], (100., 110., 90., 105.))

    def test_missing_candle_fails_closed(self):
        def empty_opener(url, timeout):
            return FakeResponse(b"[]")
        with self.assertRaisesRegex(ValueError, "missing hourly candles"):
            fetch_hourly("2024-01-01", "2024-01-02",
                         opener=empty_opener, pause=lambda _: None)

    def test_bad_range_rejected(self):
        with self.assertRaises(ValueError):
            fetch_hourly("2024-01-02", "2024-01-01",
                         opener=self.fake_opener, pause=lambda _: None)

    def test_csv_schema(self):
        rows = fetch_hourly("2024-01-01", "2024-01-01T02:00:00",
                            opener=self.fake_opener, pause=lambda _: None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eth.csv"
            save_csv(rows, path)
            self.assertEqual(path.read_text().splitlines()[0],
                             "date,open,high,low,close,volume")
