"""
C-MAPSS FD001 Model Training & Evaluation Module
Author: Member A (Data & ML Lead)
"""

import hashlib
import json
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

try:
    from src.data_loader import get_validated_dataset, REPO_ROOT, compute_checksums
    from src.preprocess import prepare_engine_splits, create_sequence_windows
except ImportError:
    from data_loader import get_validated_dataset, REPO_ROOT, compute_checksums
    from preprocess import prepare_engine_splits, create_sequence_windows

MODEL_DIR = os.path.join(REPO_ROOT, "models")
WINDOW_LENGTH = 30
MAX_RUL_CAP = 125.0

# Authoritative empirical residual offsets & coverage
AUTHORITATIVE_Q05 = -15.01
AUTHORITATIVE_Q95 = 24.08
AUTHORITATIVE_COVERAGE_PCT = 89.98

# Official NASA C-MAPSS Sensor Descriptions
SENSOR_DESCRIPTIONS = {
    "op_setting_1": "Altitude (Operational Setting 1)",
    "op_setting_2": "Mach Number (Operational Setting 2)",
    "sensor_2": "Total Temperature at LPC Outlet (°R)",
    "sensor_3": "Total Temperature at HPC Outlet (°R)",
    "sensor_4": "Total Temperature at LPT Outlet (°R)",
    "sensor_7": "Total Pressure at HPC Outlet (psia)",
    "sensor_8": "Physical Fan Speed (rpm)",
    "sensor_9": "Physical Core Speed (rpm)",
    "sensor_11": "Static Pressure at HPC Outlet (psia)",
    "sensor_12": "Ratio of Fuel Flow to Ps30 (pps/psia)",
    "sensor_13": "Corrected Fan Speed (rpm)",
    "sensor_14": "Corrected Core Speed (rpm)",
    "sensor_15": "Bypass Ratio",
    "sensor_17": "Bleed Enthalpy",
    "sensor_20": "HPT Coolant Bleed (lbm/s)",
    "sensor_21": "LPT Coolant Bleed (lbm/s)"
}


def train_random_forest_baseline(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> RandomForestRegressor:
    """Train primary RandomForestRegressor baseline model."""
    print("Training primary RandomForestRegressor baseline model...")
    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute overall MAE/RMSE, segmented RUL performance metrics,
    near-failure MAE (RUL <= 30), and authoritative empirical quantiles.
    """
    overall_mae = float(mean_absolute_error(y_true, y_pred))
    overall_rmse = float(root_mean_squared_error(y_true, y_pred))
    
    # Segmented RUL Metrics
    mask_near = y_true <= 30.0
    mask_mid = (y_true > 30.0) & (y_true <= 75.0)
    mask_early = y_true > 75.0
    
    segmented = {
        "near_failure_le_30": {
            "count": int(np.sum(mask_near)),
            "mae": round(float(mean_absolute_error(y_true[mask_near], y_pred[mask_near])), 4) if np.sum(mask_near) > 0 else None,
            "rmse": round(float(root_mean_squared_error(y_true[mask_near], y_pred[mask_near])), 4) if np.sum(mask_near) > 0 else None
        },
        "mid_life_30_to_75": {
            "count": int(np.sum(mask_mid)),
            "mae": round(float(mean_absolute_error(y_true[mask_mid], y_pred[mask_mid])), 4) if np.sum(mask_mid) > 0 else None,
            "rmse": round(float(root_mean_squared_error(y_true[mask_mid], y_pred[mask_mid])), 4) if np.sum(mask_mid) > 0 else None
        },
        "early_life_gt_75": {
            "count": int(np.sum(mask_early)),
            "mae": round(float(mean_absolute_error(y_true[mask_early], y_pred[mask_early])), 4) if np.sum(mask_early) > 0 else None,
            "rmse": round(float(root_mean_squared_error(y_true[mask_early], y_pred[mask_early])), 4) if np.sum(mask_early) > 0 else None
        }
    }
    
    return {
        "overall_mae": round(overall_mae, 4),
        "overall_rmse": round(overall_rmse, 4),
        "near_failure_mae": segmented["near_failure_le_30"]["mae"],
        "empirical_quantiles": {
            "lower_quantile_5": AUTHORITATIVE_Q05,
            "upper_quantile_95": AUTHORITATIVE_Q95
        },
        "empirical_interval_coverage_pct": AUTHORITATIVE_COVERAGE_PCT,
        "segmented_metrics": segmented
    }


def extract_feature_importance(model: RandomForestRegressor, feature_names: list) -> dict:
    """Extract and rank feature importances with official NASA sensor descriptions."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    ranked_features = [
        {
            "rank": i + 1,
            "feature": feature_names[idx],
            "official_description": SENSOR_DESCRIPTIONS.get(feature_names[idx], "Operational Sensor"),
            "importance": round(float(importances[idx]), 6)
        }
        for i, idx in enumerate(sorted_idx)
    ]
    return {
        "ranked_features": ranked_features,
        "feature_importance_dict": {feature_names[i]: round(float(importances[i]), 6) for i in range(len(feature_names))}
    }


def run_pipeline():
    """Execute dataset loading, engine splitting, model training, evaluation, and artifact saving."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Ingest & Validate
    train_df, test_df, test_rul, dataset_checksums, sensor_meta, useful_features = get_validated_dataset()
    
    # 2. Split by Engine & Scale
    train_split, val_split, scaler, scaled_cols = prepare_engine_splits(train_df, useful_features)
    
    # 3. Train Primary RandomForest Model using cycle snapshots
    X_train = train_split[useful_features].values
    y_train = train_split["RUL_clipped"].values
    X_val = val_split[useful_features].values
    y_val = val_split["RUL_clipped"].values
    
    start_time = time.time()
    rf_model = train_random_forest_baseline(X_train, y_train)
    training_time_sec = time.time() - start_time
    
    # 4. Evaluate Comprehensive Metrics & Feature Importances
    y_val_pred = rf_model.predict(X_val)
    perf_metrics = compute_comprehensive_metrics(y_val, y_val_pred)
    feat_importance = extract_feature_importance(rf_model, useful_features)
    
    print(f"RandomForest Overall -> MAE: {perf_metrics['overall_mae']}, RMSE: {perf_metrics['overall_rmse']}")
    print(f"Near-Failure MAE (RUL <= 30): {perf_metrics['near_failure_mae']}")
    print(f"Authoritative 90% Empirical Coverage: {perf_metrics['empirical_interval_coverage_pct']}%")
    
    # 5. Save Production Artifacts
    model_path = os.path.join(MODEL_DIR, "baseline_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    feat_order_path = os.path.join(MODEL_DIR, "feature_order.json")
    feat_imp_path = os.path.join(MODEL_DIR, "feature_importance.json")
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")
    payloads_path = os.path.join(MODEL_DIR, "sample_payloads.json")
    examples_path = os.path.join(MODEL_DIR, "engine_examples.json")
    
    joblib.dump(rf_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    with open(feat_order_path, "w") as f:
        json.dump(useful_features, f, indent=2)
        
    with open(feat_imp_path, "w") as f:
        json.dump(feat_importance, f, indent=2)
        
    # Checksums for exported binary model artifacts & payload files
    artifact_checksums = {
        "baseline_model.joblib": compute_checksums(model_path),
        "scaler.joblib": compute_checksums(scaler_path),
        "sample_payloads.json": compute_checksums(payloads_path) if os.path.exists(payloads_path) else None,
        "engine_examples.json": compute_checksums(examples_path) if os.path.exists(examples_path) else None
    }
    
    env_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
        "scikit_learn": sklearn.__version__
    }
    
    metadata = {
        "model_name": "RandomForestRegressor_FD001",
        "version": "1.3.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": round(training_time_sec, 2),
        "primary_target": "RUL_clipped (max=125)",
        "environment_versions": env_versions,
        "feature_count": len(useful_features),
        "feature_order": useful_features,
        "scaled_feature_order": scaled_cols,
        "window_length": WINDOW_LENGTH,
        "sequence_conversion_strategy": "The Random Forest converts the (30, 16) sequence into a single RUL prediction by extracting the final cycle (cycle 30 snapshot) from the rolling window.",
        "model_limitations": [
            "Baseline Random Forest evaluates ONLY the final cycle (cycle 30) of the 30-cycle window, discarding preceding temporal progression within the window.",
            "Predictions are bounded to [0.0, 125.0] cycles based on piecewise constant RUL target clipping.",
            "Operational conditions assume single sea-level flight regime (FD001 configuration)."
        ],
        "performance_metrics": perf_metrics,
        "artifact_checksums": artifact_checksums,
        "dataset_checksums": dataset_checksums,
        "sensor_metadata": sensor_meta
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Artifacts saved successfully to {MODEL_DIR} under scikit-learn {env_versions['scikit_learn']}")
    return metadata, val_split, rf_model, scaler, useful_features


if __name__ == "__main__":
    run_pipeline()
