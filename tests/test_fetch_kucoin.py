"""Offline contract tests for KuCoin ETH-USDT candle collection."""
import io
import json
import unittest
from urllib.parse import parse_qs, urlsplit

from fetch_kucoin import fetch_hourly


class Response(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


class KucoinTests(unittest.TestCase):
    def opener(self, request, timeout):
        url = request.full_url
        query = parse_qs(urlsplit(url).query)
        self.assertEqual(query["symbol"], ["ETH-USDT"])
        self.assertEqual(query["type"], ["1hour"])
        first, last = int(query["startAt"][0]), int(query["endAt"][0])
        candles = [[str(t), "100", "105", "110", "90", "3", "300"]
                   for t in range(last - last % 3600, first - 1, -3600)]
        return Response(json.dumps({"code": "200000", "data": candles}).encode())

    def test_multi_page_order_and_schema(self):
        rows = fetch_hourly("2024-01-01", "2024-04-01",
                            opener=self.opener, pause=lambda _: None)
        self.assertEqual(len(rows), 91 * 24)
        self.assertEqual(rows[0][1:], (100., 110., 90., 105., 3.))
        self.assertTrue(rows[-1][0].startswith("2024-03-31T23:00:00"))

    def test_missing_candles_are_rejected(self):
        def empty(request, timeout):
            return Response(b'{"code":"200000","data":[]}')
        with self.assertRaisesRegex(ValueError, "missing ETH-USDT"):
            fetch_hourly("2024-01-01", "2024-01-02",
                         opener=empty, pause=lambda _: None)

    def test_invalid_api_response_rejected(self):
        def invalid(request, timeout):
            return Response(b'{"code":"400100","data":[]}')
        with self.assertRaisesRegex(ValueError, "unexpected KuCoin"):
            fetch_hourly("2024-01-01", "2024-01-02",
                         opener=invalid, pause=lambda _: None)
