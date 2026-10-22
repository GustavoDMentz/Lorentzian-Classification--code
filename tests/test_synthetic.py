"""Synthetic data must include drift changes before prices are accumulated."""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / "Código Principal"
spec = importlib.util.spec_from_loader("lorentzian", SourceFileLoader("lorentzian", str(path)))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SyntheticDataTests(unittest.TestCase):
    def test_reproducible_and_positive(self):
        a = module.generate_synthetic_prices(250, seed=17)
        b = module.generate_synthetic_prices(250, seed=17)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue((a > 0).all())

    def test_drift_is_applied_to_actual_prices(self):
        n, seed = 250, 17
        prices = module.generate_synthetic_prices(n, seed)
        actual_returns = np.diff(np.log(np.r_[100., prices]))
        baseline = np.random.default_rng(seed).normal(0.0005, 0.02, n)
        np.testing.assert_allclose(actual_returns[:100], baseline[:100], atol=1e-12)
        self.assertTrue(np.all(np.isclose(np.abs(actual_returns[100:150] - baseline[100:150]), 0.001, atol=1e-12)))
        self.assertTrue(np.all(np.isclose(np.abs(actual_returns[200:250] - baseline[200:250]), 0.001, atol=1e-12)))
        np.testing.assert_allclose(actual_returns[150:200], baseline[150:200], atol=1e-12)

    def test_invalid_length(self):
        with self.assertRaises(ValueError):
            module.generate_synthetic_prices(0)
