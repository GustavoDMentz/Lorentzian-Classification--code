"""Integration checks for signal-to-fill timing."""
import unittest
from unittest.mock import Mock

import pandas as pd
from evaluation import evaluate_strategy, evaluate_holdout


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "open": [100., 110., 121., 133.1],
            "high": [101., 111., 122., 134.],
            "low": [99., 109., 120., 132.],
            "close": [100., 110., 121., 133.1],
        })

    def test_signal_at_close_executes_next_open(self):
        strategy = Mock()
        strategy.generate_signals.return_value = pd.DataFrame({
            "prediction": [1, 1, 0, 0],
            "position": [0, 1, 1, 0],
            "signal": [0, 1, 0, -1],
        })
        out = evaluate_strategy(strategy, self.data, cost_bps=0)
        self.assertEqual(out["executed_position"].tolist(), [0, 0, 1, 1])
        self.assertAlmostEqual(out["gross_return"].iloc[1], 0)
        self.assertAlmostEqual(out["gross_return"].iloc[2], 0.1)
        self.assertAlmostEqual(out["equity"].iloc[-1], 1.1)

    def test_costs_are_passed_through(self):
        strategy = Mock()
        strategy.generate_signals.return_value = pd.DataFrame({
            "position": [0, 1, -1, 0],
        })
        out = evaluate_strategy(strategy, self.data, cost_bps=10)
        self.assertEqual(out["turnover"].tolist(), [0, 0, 1, 2])
        self.assertAlmostEqual(out["cost"].sum(), 0.003)

    def test_holdout_preserves_warmup_and_rebases_returns(self):
        strategy = Mock()
        strategy.generate_signals.return_value = pd.DataFrame({
            "position": [0, 1, 1, 0],
        })
        out = evaluate_holdout(strategy, self.data, split_index=2, cost_bps=0)
        strategy.generate_signals.assert_called_once()
        self.assertEqual(len(out["train"]), 2)
        self.assertEqual(len(out["test"]), 2)
        self.assertEqual(out["test"]["executed_position"].tolist(), [1, 1])
        self.assertAlmostEqual(out["test"]["holdout_equity"].iloc[-1], 1.1)
        self.assertAlmostEqual(out["test"]["holdout_buy_hold_equity"].iloc[-1], 1.1)

    def test_holdout_invalid_split(self):
        strategy = Mock()
        for split in (0, -1, len(self.data), 1.5):
            with self.subTest(split=split):
                with self.assertRaises(ValueError):
                    evaluate_holdout(strategy, self.data, split)

    def test_invalid_input_rejected(self):
        strategy = Mock()
        with self.assertRaises(ValueError):
            evaluate_strategy(strategy, self.data.drop(columns="open"))
        with self.assertRaises(ValueError):
            evaluate_strategy(strategy, self.data.iloc[::-1])
