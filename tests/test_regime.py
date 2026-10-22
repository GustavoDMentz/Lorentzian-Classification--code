"""Regression tests for causal regime classification."""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

source_path = Path(__file__).resolve().parents[1] / "Código Principal"
spec = importlib.util.spec_from_loader("lorentzian", SourceFileLoader("lorentzian", str(source_path)))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Strategy = module.LorentzianClassificationStrategy


class RegimeTests(unittest.TestCase):
    def setUp(self):
        self.strategy = Strategy()

    def test_monotonic_trend_passes_after_warmup(self):
        data = pd.DataFrame({"close": np.arange(1., 102.)})
        result = self.strategy.regime_filter(data, 0.2)
        self.assertFalse(result.iloc[:50].any())
        self.assertTrue(result.iloc[50:].all())

    def test_choppy_prices_are_rejected(self):
        close = np.array([100. + (i % 2) for i in range(101)])
        result = self.strategy.regime_filter(pd.DataFrame({"close": close}), 0.2)
        self.assertFalse(result.any())

    def test_flat_prices_are_rejected(self):
        result = self.strategy.regime_filter(pd.DataFrame({"close": [100.] * 101}), 0.2)
        self.assertFalse(result.any())

    def test_future_changes_do_not_affect_past_filter(self):
        a = pd.DataFrame({"close": np.arange(1., 102.)})
        b = a.copy()
        b.loc[80:, "close"] = 1000.
        first = self.strategy.regime_filter(a, 0.2)
        second = self.strategy.regime_filter(b, 0.2)
        pd.testing.assert_series_equal(first.iloc[:80], second.iloc[:80])

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            self.strategy.regime_filter(pd.DataFrame({"close": [1.] * 101}), -0.1)


if __name__ == "__main__":
    unittest.main()
