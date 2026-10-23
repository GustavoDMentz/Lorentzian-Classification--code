"""CSV validation and predeclared holdout reporting tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from run_evaluation import load_ohlc_csv, holdout_report


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, tz="UTC"),
            "open": [100., 110., 121., 133.1, 146.41],
            "high": [101., 111., 122., 134., 147.],
            "low": [99., 109., 120., 132., 145.],
            "close": [100., 110., 121., 133.1, 146.41],
        })

    def test_csv_loader_sorts_and_parses_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.csv"
            self.data.iloc[::-1].to_csv(path, index=False)
            loaded = load_ohlc_csv(str(path))
            self.assertTrue(loaded.index.is_monotonic_increasing)
            self.assertEqual(loaded["open"].iloc[0], 100.)

    def test_rejects_invalid_ohlc_and_duplicate_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.csv"
            bad = self.data.copy()
            bad.loc[0, "high"] = 90.
            bad.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_ohlc_csv(str(path))
            bad = self.data.copy()
            bad.loc[1, "date"] = bad.loc[0, "date"]
            bad.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_ohlc_csv(str(path))

    def test_holdout_report_uses_requested_date(self):
        data = self.data.set_index("date")
        strategy = Mock()
        strategy.generate_signals.return_value = pd.DataFrame(
            {"position": [0, 1, 1, 0, 0]}, index=data.index
        )
        report = holdout_report(data, "2024-01-04", cost_bps=0, strategy=strategy)
        self.assertEqual(report["train_bars"], 3)
        self.assertEqual(report["holdout_bars"], 2)
        self.assertAlmostEqual(report["strategy_return_pct"], 10.)
        with self.assertRaises(ValueError):
            holdout_report(data, "2023-01-01", strategy=strategy)
