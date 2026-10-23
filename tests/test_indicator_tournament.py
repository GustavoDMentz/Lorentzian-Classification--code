"""Deterministic, causal tournament regression checks."""
import unittest
import numpy as np
import pandas as pd

from indicator_tournament import indicator_targets, tournament


class TournamentTests(unittest.TestCase):
    def data(self, n=280):
        close = 100 * np.exp(np.cumsum(0.001 + 0.01 * np.sin(np.arange(n) / 11)))
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame({"open": close, "high": close * 1.01,
                             "low": close * .99, "close": close}, index=idx)

    def test_target_values_and_finite_outputs(self):
        targets = indicator_targets(self.data())
        self.assertEqual(len(targets), 9)
        for series in targets.values():
            self.assertTrue(series.isin((-1, 0, 1)).all())
            self.assertEqual(len(series), 280)

    def test_future_candles_do_not_change_earlier_signals(self):
        original = self.data()
        changed = original.copy()
        changed.iloc[250:, :] *= 2
        a, b = indicator_targets(original), indicator_targets(changed)
        for name in a:
            pd.testing.assert_series_equal(a[name].iloc[:249], b[name].iloc[:249])

    def test_ranking_and_fold_accounting(self):
        summary, folds = tournament(self.data(), 200, 40, 10)
        self.assertEqual(len(summary), 9)
        self.assertEqual(len(folds), 18)
        self.assertTrue((summary["folds"] == 2).all())
        self.assertTrue(np.isfinite(summary["compounded_return"]).all())
        self.assertTrue((summary["turnover"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
