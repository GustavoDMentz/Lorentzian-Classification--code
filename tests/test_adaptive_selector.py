"""Regression checks for causal strategy selection."""
import unittest
import numpy as np
import pandas as pd
from adaptive_selector import adaptive_targets, evaluate_adaptive


class AdaptiveTests(unittest.TestCase):
    def data(self, n=800):
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        c = 100 * np.exp(np.cumsum(0.002 * np.sin(np.arange(n) / 11) + 0.0002))
        return pd.DataFrame({"open": c, "high": c * 1.01,
                             "low": c * .99, "close": c}, index=idx)

    def test_future_prices_cannot_change_past_selection(self):
        a = self.data()
        b = a.copy()
        b.iloc[680:, :] *= 1.8
        xa, sa = adaptive_targets(a, lookback=100)
        xb, sb = adaptive_targets(b, lookback=100)
        pd.testing.assert_series_equal(xa.iloc[:679], xb.iloc[:679])
        pd.testing.assert_series_equal(sa.iloc[:679], sb.iloc[:679])

    def test_initial_cash_and_valid_positions(self):
        targets, selected = adaptive_targets(self.data(), lookback=100)
        self.assertTrue((targets.iloc[:99] == 0).all())
        self.assertTrue(targets.isin([-1, 0, 1]).all())
        self.assertTrue((selected.iloc[:99] == "CASH").all())

    def test_fold_accounting(self):
        summary, folds, selection = evaluate_adaptive(
            self.data(), train_bars=300, test_bars=100, lookback=100)
        self.assertEqual(len(folds), 5)
        self.assertTrue(np.isfinite(summary["compounded_return"]).all())
        self.assertEqual(len(selection), 800)


if __name__ == "__main__":
    unittest.main()
