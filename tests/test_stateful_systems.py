"""Causality and state machine invariants for full trading systems."""
import unittest
import numpy as np
import pandas as pd
from stateful_systems import _stateful, system_targets, evaluate_systems


class StatefulSystemsTests(unittest.TestCase):
    def make_data(self, n=450):
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        c = 100 * np.exp(np.cumsum(0.002 * np.sin(np.arange(n) / 13) + 0.0002))
        return pd.DataFrame({"open": c, "high": c * 1.01,
                             "low": c * 0.99, "close": c}, index=idx)

    def test_state_persists_until_exit(self):
        idx = pd.RangeIndex(5)
        a = _stateful(idx, [0, 1, 0, 0, 0], [0]*5,
                      [0, 0, 0, 1, 0], [0]*5, allow_reverse=False)
        self.assertEqual(a.tolist(), [0, 1, 1, 0, 0])

    def test_future_changes_do_not_affect_past(self):
        a = self.make_data()
        b = a.copy()
        b.iloc[400:, :] *= 1.6
        x, y = system_targets(a), system_targets(b)
        self.assertEqual(set(x), set(y))
        self.assertEqual(len(x), 10)
        for name in x:
            pd.testing.assert_series_equal(x[name].iloc[:399], y[name].iloc[:399])
            self.assertTrue(x[name].isin([-1, 0, 1]).all())

    def test_folds_and_costs(self):
        summary, folds = evaluate_systems(self.make_data(), 250, 100, 10)
        self.assertEqual(len(summary), 10)
        self.assertEqual(len(folds), 20)
        self.assertTrue(np.isfinite(summary["compounded_return"]).all())
        self.assertTrue((summary["turnover"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
