"""Integration smoke tests for ETH walk-forward orchestration."""
import unittest

import numpy as np
import pandas as pd

from run_walk_forward import run_experiment


class FakeStrategy:
    def __init__(self, neighbors_count, max_bars_back, regime_threshold):
        self.neighbors_count = neighbors_count

    def generate_signals(self, data):
        positions = np.zeros(len(data), dtype=int)
        positions[30:] = 1 if self.neighbors_count == 4 else -1
        return pd.DataFrame({"position": positions}, index=data.index)


class RunnerTests(unittest.TestCase):
    def data(self, n=120):
        prices = 100 * np.exp(np.cumsum(np.sin(np.arange(n) / 5) * .01))
        return pd.DataFrame({"open": prices, "high": prices + 1,
                             "low": prices - 1, "close": prices},
                            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))

    def test_end_to_end_fold_alignment(self):
        result = run_experiment(
            self.data(), train_bars=60, test_bars=20, cost_bps=10,
            rsi_periods=(7,), rsi_oversold=(30.,),
            rsi_overbought=(70.,), neighbors_grid=(4, 8),
            regime_grid=(.2,), lorentzian_factory=FakeStrategy
        )
        self.assertEqual(len(result["rsi"]), 3)
        self.assertEqual(len(result["lorentzian"]), 3)
        self.assertEqual(len(result["comparison"]), 3)
        self.assertEqual(result["config"]["n_bars"], 120)

    def test_insufficient_history_rejected(self):
        with self.assertRaisesRegex(ValueError, "insufficient data"):
            run_experiment(self.data(50), train_bars=40, test_bars=20,
                           lorentzian_factory=FakeStrategy)

    def test_unsorted_index_rejected(self):
        data = self.data()
        data = data.iloc[::-1]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            run_experiment(data, train_bars=60, test_bars=20,
                           lorentzian_factory=FakeStrategy)
