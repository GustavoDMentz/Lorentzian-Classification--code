"""Exact-value regression tests for next-open execution and costs."""
import unittest

import pandas as pd
from backtest import backtest_next_open


class BacktestTests(unittest.TestCase):
    def test_next_open_no_same_bar_lookahead(self):
        data = pd.DataFrame({"open": [100., 110., 121., 133.1]})
        targets = pd.Series([1, 1, 0, 0])
        out = backtest_next_open(data, targets)
        self.assertEqual(out["executed_position"].tolist(), [0, 1, 1, 0])
        self.assertAlmostEqual(out["gross_return"].iloc[0], 0)
        self.assertAlmostEqual(out["gross_return"].iloc[1], 0.1)
        self.assertAlmostEqual(out["gross_return"].iloc[2], 0.1)
        self.assertAlmostEqual(out["equity"].iloc[-1], 1.21)

    def test_reversal_charges_two_units(self):
        data = pd.DataFrame({"open": [100., 100., 100., 100.]})
        targets = pd.Series([1, -1, -1, -1])
        out = backtest_next_open(data, targets, cost_bps=10)
        self.assertEqual(out["turnover"].tolist(), [0, 1, 2, 0])
        self.assertAlmostEqual(out["cost"].sum(), 0.003)
        self.assertAlmostEqual(out["equity"].iloc[-1], 0.999 * 0.998)

    def test_future_price_does_not_change_past_returns(self):
        a = pd.DataFrame({"open": [100., 101., 102., 103., 104.]})
        b = a.copy()
        b.loc[4, "open"] = 200.
        targets = pd.Series([1, 1, 1, 1, 1])
        x = backtest_next_open(a, targets)
        y = backtest_next_open(b, targets)
        pd.testing.assert_series_equal(x["net_return"].iloc[:3], y["net_return"].iloc[:3])

    def test_invalid_inputs(self):
        data = pd.DataFrame({"open": [100., 101.]})
        with self.assertRaises(ValueError):
            backtest_next_open(data, pd.Series([2, 0]))
        with self.assertRaises(ValueError):
            backtest_next_open(data, pd.Series([0, 0]), cost_bps=-1)
        with self.assertRaises(ValueError):
            backtest_next_open(data, pd.Series([0]))
