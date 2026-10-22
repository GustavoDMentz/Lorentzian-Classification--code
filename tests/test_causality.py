"""Regression tests for causal four-bar labels and neighbor eligibility."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("lorentzian", Path(__file__).resolve().parents[1] / "Código Principal")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Strategy = module.LorentzianClassificationStrategy


class CausalityTests(unittest.TestCase):
    def test_unmatured_labels_are_neutral(self):
        s = Strategy()
        data = pd.DataFrame({"close": np.arange(1.0, 11.0)})
        self.assertEqual([s.classify_bar(data, i) for i in range(6, 10)], [0] * 4)
        self.assertEqual(s.classify_bar(data, 5), 1)

    def test_nearest_neighbors_exclude_unmatured_labels(self):
        s = Strategy(neighbors_count=1, max_bars_back=20, feature_count=1)
        features = pd.DataFrame({"f1": [10., 10., 10., 10., 0., 0., 0., 0., 0., 0.]})
        # At t=8, only indices <=4 have observable 4-bar outcomes.
        # Index 7 is closest but must never influence the prediction.
        s.training_labels = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
        self.assertEqual(s.knn_predict(features, 8), 1)

    def test_future_prices_do_not_change_past_prediction(self):
        s = Strategy(neighbors_count=1, max_bars_back=20, feature_count=1)
        features = pd.DataFrame({"f1": [0.] * 12})
        a = pd.DataFrame({"close": [100.] * 12})
        b = a.copy()
        b.loc[9:, "close"] = 200.
        for data in (a, b):
            s.training_labels = [s.classify_bar(data, i) for i in range(len(data))]
            prediction = s.knn_predict(features, 8)
            if data is a:
                first = prediction
            else:
                self.assertEqual(prediction, first)


if __name__ == "__main__":
    unittest.main()
