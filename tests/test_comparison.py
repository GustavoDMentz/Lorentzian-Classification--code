"""Walk-forward report aggregation invariants."""
import unittest

import pandas as pd

from comparison import summarize_walk_forward


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"open": [100., 110., 121., 133.1, 146.41, 161.051]})
        self.rsi = pd.DataFrame({
            "test_start": [1, 3], "test_end": [2, 4],
            "test_bars": [2, 2], "test_return": [0.1, -0.1]
        })
        self.lorentz = self.rsi.copy()
        self.lorentz["test_return"] = [0.05, 0.05]

    def test_compounds_instead_of_summing(self):
        report = summarize_walk_forward(self.rsi, self.lorentz, self.data)
        rsi = report.set_index("strategy").loc["RSI optimized"]
        lorentz = report.set_index("strategy").loc["Lorentzian optimized"]
        self.assertAlmostEqual(rsi["compounded_return"], -0.01)
        self.assertAlmostEqual(lorentz["compounded_return"], 0.1025)
        self.assertEqual(int(lorentz["positive_folds"]), 2)

    def test_buy_hold_uses_same_fold_boundaries(self):
        report = summarize_walk_forward(self.rsi, self.lorentz, self.data)
        buy_hold = report.set_index("strategy").loc["Buy and hold"]
        self.assertAlmostEqual(buy_hold["compounded_return"], 0.21)

    def test_rejects_misaligned_folds(self):
        wrong = self.lorentz.copy()
        wrong.loc[1, "test_start"] = 4
        with self.assertRaisesRegex(ValueError, "alignment"):
            summarize_walk_forward(self.rsi, wrong, self.data)

    def test_rejects_missing_or_invalid_returns(self):
        wrong = self.lorentz.copy()
        wrong.loc[0, "test_return"] = -1.5
        with self.assertRaisesRegex(ValueError, "invalid fold returns"):
            summarize_walk_forward(self.rsi, wrong, self.data)
