"""Vectorized kNN must match the legacy scalar reference exactly."""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

path = Path(__file__).resolve().parents[1] / "Código Principal"
loader = SourceFileLoader("lorentzian_perf", str(path))
spec = importlib.util.spec_from_loader(loader.name, loader)
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)


def scalar_reference(strategy, features, t):
    if t < strategy.neighbors_count or not np.isfinite(features.iloc[t].to_numpy()).all():
        return 0
    distances, labels = [], []
    for i in range(max(0, t - strategy.max_bars_back),
                   max(max(0, t - strategy.max_bars_back), t - 3)):
        a, b = features.iloc[t].to_numpy(), features.iloc[i].to_numpy()
        if np.isfinite(b).all():
            distances.append(np.log1p(np.abs(a - b)).sum())
            labels.append(strategy.training_labels[i])
    if len(labels) < strategy.neighbors_count:
        return 0
    chosen = np.asarray(labels)[np.argsort(distances)[:strategy.neighbors_count]]
    values, counts = np.unique(chosen, return_counts=True)
    return int(values[np.argmax(counts)])


class VectorizedKnnTests(unittest.TestCase):
    def test_matches_scalar_with_nan_and_ties(self):
        rng = np.random.default_rng(123)
        matrix = rng.normal(size=(100, 5))
        matrix[20, 2] = np.nan
        matrix[45] = matrix[44]
        features = pd.DataFrame(matrix)
        for neighbors in (1, 4, 8):
            strategy = module.LorentzianClassificationStrategy(
                neighbors_count=neighbors, max_bars_back=30)
            strategy.training_labels = rng.choice([-1, 0, 1], len(matrix)).tolist()
            for t in range(len(matrix)):
                self.assertEqual(strategy.knn_predict(features, t),
                                 scalar_reference(strategy, features, t))

    def test_future_labels_still_excluded(self):
        strategy = module.LorentzianClassificationStrategy(
            neighbors_count=1, max_bars_back=20)
        features = pd.DataFrame({"x": [10., 10., 10., 10., 0., 0., 0., 0., 0.]})
        strategy.training_labels = [1, 1, 1, 1, 1, -1, -1, -1, -1]
        self.assertEqual(strategy.knn_predict(features, 8), 1)
