"""Regression tests for the fixed RSI benchmark and equal evaluation windows."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from baselines import rsi_target_positions, compare_rsi_holdout


class RSIBaselineTests(unittest.TestCase):
    def test_thresholds_and_warmup(self):
        data = pd.DataFrame({"close": [100.] * 6})
        fake_rsi = np.array([np.nan, 20., 30., 50., 70., 80.])
        with patch("baselines.talib.RSI", return_value=fake_rsi):
            result = rsi_target_positions(data)
        self.assertEqual(result.tolist(), [0, 1, 0, 0, 0, -1])

    def test_no_future_rsi_dependency(self):
        a = pd.DataFrame({"close": [100. + i for i in range(80)]})
        b = a.copy()
        b.loc[79, "close"] = 1000.
        x = rsi_target_positions(a)
        y = rsi_target_positions(b)
        pd.testing.assert_series_equal(x.iloc[:79], y.iloc[:79])

    def test_same_holdout_and_costs(self):
        n = 90
        close = np.linspace(100., 120., n)
        data = pd.DataFrame({
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        })
        strategy = Mock()
        strategy.generate_signals.return_value = pd.DataFrame(
            {"position": [0] * n}, index=data.index
        )
        result = compare_rsi_holdout(strategy, data, split_index=60, cost_bps=10)
        self.assertEqual(len(result["lorentzian"]), 30)
        self.assertEqual(len(result["rsi14"]), 30)
        self.assertTrue(result["lorentzian"].index.equals(result["rsi14"].index))
        self.assertAlmostEqual(result["lorentzian"]["holdout_equity"].iloc[-1], 1.)
