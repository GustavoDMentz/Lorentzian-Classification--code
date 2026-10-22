"""Walk-forward invariants: folds, no future leakage, deterministic selection."""
import unittest

import numpy as np
import pandas as pd

from walk_forward import expanding_folds, optimize_rsi_walk_forward, fold_returns


class WalkForwardTests(unittest.TestCase):
    def make_data(self, n=150):
        close = 100 * np.exp(np.cumsum(np.sin(np.arange(n) / 6) * 0.01))
        return pd.DataFrame({
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        })

    def test_expanding_nonoverlapping_folds(self):
        self.assertEqual(expanding_folds(110, 50, 20),
                         [(50, 70), (70, 90), (90, 110)])
        with self.assertRaises(ValueError):
            expanding_folds(30, 20, 20)

    def test_future_changes_do_not_affect_prior_selection(self):
        a = self.make_data()
        b = a.copy()
        b.loc[120:, ["open", "high", "low", "close"]] *= 1.5
        kw = dict(train_bars=60, test_bars=20, cost_bps=10,
                  periods=(7, 14), oversold_levels=(30.,), overbought_levels=(70.,))
        x = optimize_rsi_walk_forward(a, **kw)
        y = optimize_rsi_walk_forward(b, **kw)
        pd.testing.assert_frame_equal(x.iloc[:3], y.iloc[:3])

    def test_deterministic_and_cost_sensitive(self):
        data = self.make_data()
        kw = dict(train_bars=60, test_bars=30, periods=(7, 14),
                  oversold_levels=(30.,), overbought_levels=(70.,))
        x = optimize_rsi_walk_forward(data, cost_bps=10, **kw)
        y = optimize_rsi_walk_forward(data, cost_bps=10, **kw)
        pd.testing.assert_frame_equal(x, y)
        self.assertEqual(len(x), 3)
        self.assertTrue((x["test_bars"] == 30).all())

    def test_fold_starts_flat(self):
        data = self.make_data(10)
        targets = pd.Series([1] * 10)
        out = fold_returns(data, targets, 4, 9, 10)
        self.assertEqual(out["executed_position"].iloc[0], 0)
        self.assertEqual(out["turnover"].iloc[0], 0)
        self.assertEqual(out["executed_position"].iloc[1], 1)
