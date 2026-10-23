import unittest
from datetime import datetime, timedelta, timezone
from research.analyze_kraken_oob import analyze


class ForwardAnalysisTests(unittest.TestCase):
    def test_one_minute_and_no_gap_leakage(self):
        start = datetime(2026, 10, 23, tzinfo=timezone.utc)
        rows = [(start, 99, 101, .5),
                (start + timedelta(seconds=60), 100, 102, -.5),
                (start + timedelta(hours=3), 101, 103, .2)]
        pairs = analyze(rows, horizon_seconds=60, tolerance_seconds=20)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], .5)
        self.assertAlmostEqual(pairs[0][2], 100/101 - 1)

    def test_future_only(self):
        start = datetime(2026, 10, 23, tzinfo=timezone.utc)
        rows = [(start, 99, 101, 0),
                (start + timedelta(seconds=45), 100, 102, 0)]
        self.assertEqual(analyze(rows), [])


if __name__ == "__main__":
    unittest.main()
