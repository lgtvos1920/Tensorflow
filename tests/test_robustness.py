"""
Robustness & Edge-Case Tests for C-MAPSS FD001 Predict RUL Pipeline
Author: Member A (Data & ML Lead)
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import RULPredictor, predict_rul


class TestPipelineRobustness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = RULPredictor()
        cls.feature_order = cls.predictor.feature_order
        
        # Load sample payloads for testing
        payloads_path = os.path.join(cls.predictor.model_dir, "sample_payloads.json")
        with open(payloads_path, "r") as f:
            cls.sample_payloads = json.load(f)

    def test_missing_values_rejection(self):
        """Test that sequences containing NaN or None values raise ValueError."""
        seq = [{f: 0.5 for f in self.feature_order} for _ in range(30)]
        seq[10]["sensor_2"] = np.nan
        with self.assertRaises(ValueError) as ctx:
            predict_rul(seq)
        self.assertIn("non-finite values", str(ctx.exception))

    def test_sensor_noise_robustness(self):
        """Test that small Gaussian sensor noise preserves prediction stability."""
        clean_seq = self.sample_payloads["engine_54_successful"]["sequence_30_cycle_payload"]
        res_clean = predict_rul(clean_seq)
        
        # Add small Gaussian noise (std=0.01) to numeric sensor values
        np.random.seed(42)
        noisy_seq = []
        for row in clean_seq:
            noisy_row = {}
            for k, v in row.items():
                noisy_row[k] = float(v + np.random.normal(0, 0.01))
            noisy_seq.append(noisy_row)
            
        res_noisy = predict_rul(noisy_seq)
        
        # Prediction shift under 0.01 noise should be < 1.5 cycles
        rul_diff = abs(res_clean["estimated_rul"] - res_noisy["estimated_rul"])
        print(f"Noise Robustness Test -> Clean RUL: {res_clean['estimated_rul']}, Noisy RUL: {res_noisy['estimated_rul']} (Diff: {rul_diff:.2f})")
        self.assertLess(rul_diff, 1.5)

    def test_feature_order_invariance(self):
        """Test that re-ordering dictionary keys produces identical predictions."""
        clean_seq = self.sample_payloads["engine_54_successful"]["sequence_30_cycle_payload"]
        
        # Reverse key ordering in input dictionaries
        reordered_seq = []
        for row in clean_seq:
            reversed_keys = list(row.keys())[::-1]
            reordered_row = {k: row[k] for k in reversed_keys}
            reordered_seq.append(reordered_row)
            
        res_orig = predict_rul(clean_seq)
        res_reordered = predict_rul(reordered_seq)
        
        self.assertEqual(res_orig["estimated_rul"], res_reordered["estimated_rul"])
        self.assertEqual(res_orig["lower_bound"], res_reordered["lower_bound"])
        self.assertEqual(res_orig["upper_bound"], res_reordered["upper_bound"])

    def test_difficult_trajectory_engine_74(self):
        """Test prediction robustness on Engine 74 (difficult non-linear trajectory)."""
        engine_74_seq = self.sample_payloads["engine_74_difficult"]["sequence_30_cycle_payload"]
        res = predict_rul(engine_74_seq)
        
        self.assertEqual(res["window_length"], 30)
        self.assertIn("CRITICAL", res["risk_level"])
        self.assertIn("qualified engineering inspection", res["recommendation"].lower())
        self.assertTrue(0.0 <= res["estimated_rul"] <= 125.0)
        self.assertTrue(0.0 <= res["lower_bound"] <= 125.0)
        self.assertTrue(0.0 <= res["upper_bound"] <= 125.0)
        self.assertTrue(res["lower_bound"] <= res["estimated_rul"] <= res["upper_bound"])


if __name__ == "__main__":
    unittest.main()
