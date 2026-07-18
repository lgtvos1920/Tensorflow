"""
C-MAPSS FD001 Predict RUL Unified Interface
Author: Member A (Data & ML Lead)
"""

import json
import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from src.data_loader import REPO_ROOT
except ImportError:
    from data_loader import REPO_ROOT

MODEL_DIR = os.path.join(REPO_ROOT, "models")


class RULPredictor:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.model = joblib.load(os.path.join(model_dir, "baseline_model.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        
        with open(os.path.join(model_dir, "feature_order.json"), "r") as f:
            self.feature_order = json.load(f)
            
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
            
        self.intervals = self.metadata.get("empirical_prediction_intervals", {
            "lower_quantile_5": -15.0,
            "upper_quantile_95": 15.0
        })

    def assess_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate data quality score based on feature completeness and null values."""
        missing_feats = [f for f in self.feature_order if f not in df.columns]
        if missing_feats:
            present_ratio = (len(self.feature_order) - len(missing_feats)) / len(self.feature_order)
            return round(present_ratio, 2)
            
        # Check non-null values
        null_count = df[self.feature_order].isnull().sum().sum()
        total_cells = len(df) * len(self.feature_order)
        if total_cells == 0:
            return 0.0
        return round(float(1.0 - (null_count / total_cells)), 4)

    def determine_risk_and_recommendation(self, estimated_rul: float) -> tuple:
        """Determine risk level classification and actionable maintenance recommendation."""
        if estimated_rul <= 15.0:
            risk = "CRITICAL"
            rec = "IMMEDIATE ACTION REQUIRED: Engine approaching end-of-life. Schedule immediate engine swap / overhaul."
        elif estimated_rul <= 30.0:
            risk = "HIGH"
            rec = "HIGH PRIORITY: Operational lifetime under 30 cycles. Plan maintenance within the next 5 flight cycles."
        elif estimated_rul <= 60.0:
            risk = "MEDIUM"
            rec = "ELEVATED MONITORING: Engine wear progressing. Conduct detailed sensor inspection during next depot visit."
        else:
            risk = "LOW"
            rec = "NORMAL OPERATION: Engine operating within nominal degradation bounds. Routine monitoring active."
        return risk, rec

    def predict_rul(self, input_data) -> dict:
        """
        Unified prediction interface for downstream services (Members B & C).
        Accepts dict, list of dicts (sequence), Series, or DataFrame.
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.Series):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("input_data must be a pandas DataFrame, Series, dict, or list of dicts")

        quality_score = self.assess_data_quality(df)
        
        # Fill missing required feature columns with 0 if any missing to avoid crash
        for f in self.feature_order:
            if f not in df.columns:
                df[f] = 0.0

        # Extract features in exact training order as DataFrame with column names
        X_df = df[self.feature_order]
        
        # Scale features
        X_scaled = self.scaler.transform(X_df)
        
        # Point prediction (last row if sequence passed)
        pred_rul_array = self.model.predict(X_df.values)
        latest_pred = float(pred_rul_array[-1])
        
        # Calculate empirical prediction bounds
        lower_b = max(0.0, float(latest_pred + self.intervals.get("lower_quantile_5", -15.0)))
        upper_b = float(latest_pred + self.intervals.get("upper_quantile_95", 15.0))
        
        risk_level, recommendation = self.determine_risk_and_recommendation(latest_pred)
        
        return {
            "model_name": self.metadata.get("model_name", "RandomForestRegressor_FD001"),
            "version": self.metadata.get("version", "1.0.0"),
            "feature_order": self.feature_order,
            "window_length": self.metadata.get("window_length", 30),
            "estimated_rul": round(latest_pred, 2),
            "lower_bound": round(lower_b, 2),
            "upper_bound": round(upper_b, 2),
            "risk_level": risk_level,
            "data_quality_score": quality_score,
            "recommendation": recommendation
        }


# Global instance & exportable function
_predictor_instance = None

def predict_rul(input_data) -> dict:
    """Standalone function wrapping RULPredictor.predict_rul()."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = RULPredictor()
    return _predictor_instance.predict_rul(input_data)


if __name__ == "__main__":
    predictor = RULPredictor()
    sample_input = {f: 0.5 for f in predictor.feature_order}
    res = predictor.predict_rul(sample_input)
    print("Sample Predict RUL Output:")
    print(json.dumps(res, indent=2))
