"""Position transitions must distinguish entries, exits and reversals."""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

path = Path(__file__).resolve().parents[1] / "Código Principal"
spec = importlib.util.spec_from_loader("lorentzian", SourceFileLoader("lorentzian", str(path)))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PositionTests(TestCase):
    def run_predictions(self, predictions):
        s = module.LorentzianClassificationStrategy()
        n = len(predictions)
        data = pd.DataFrame({"close": [100.] * n})
        features = pd.DataFrame({"f1": [0.] * n})
        with patch.object(s, "calculate_all_features", return_value=features), \
             patch.object(s, "apply_filters", return_value=pd.Series([True] * n)), \
             patch.object(s, "knn_predict", side_effect=lambda _features, i: predictions[i]):
            return s.generate_signals(data)

    def test_all_transitions_and_reversals(self):
        result = self.run_predictions([0, 1, 1, 0, -1, 1, -1, 0])
        self.assertEqual(result["position"].tolist(), [0, 1, 1, 0, -1, 1, -1, 0])
        self.assertEqual(result["signal"].tolist(), [0, 1, 0, -1, -1, 2, -2, 1])
        self.assertEqual(result["action"].tolist(), [
            "HOLD", "ENTER_LONG", "HOLD", "EXIT_LONG",
            "ENTER_SHORT", "REVERSE_TO_LONG", "REVERSE_TO_SHORT", "EXIT_SHORT"
        ])

    def test_neutral_predictions_close_positions(self):
        result = self.run_predictions([0, -1, 0, 0, 1, 0])
        self.assertEqual(result["signal"].tolist(), [0, -1, 1, 0, 1, -1])

    def test_no_orders_when_flat(self):
        result = self.run_predictions([0, 0, 0, 0])
        self.assertTrue((result["signal"] == 0).all())
        self.assertTrue((result["action"] == "HOLD").all())
