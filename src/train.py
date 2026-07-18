"""
C-MAPSS FD001 Model Training & Evaluation Module
Author: Member A (Data & ML Lead)
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

try:
    from src.data_loader import get_validated_dataset, REPO_ROOT
    from src.preprocess import prepare_engine_splits, create_sequence_windows
except ImportError:
    from data_loader import get_validated_dataset, REPO_ROOT
    from preprocess import prepare_engine_splits, create_sequence_windows

MODEL_DIR = os.path.join(REPO_ROOT, "models")


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


def attempt_tensorflow_model(X_train_seq: np.ndarray, y_train_seq: np.ndarray, X_val_seq: np.ndarray, y_val_seq: np.ndarray):
    """Attempt a compact TensorFlow/Keras 1D-Conv/LSTM sequence model if TensorFlow is available."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
        
        print("TensorFlow detected! Training compact Keras 1D-Conv sequence model...")
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            Conv1D(filters=16, kernel_size=3, activation='relu'),
            Dropout(0.2),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=64, validation_data=(X_val_seq, y_val_seq), verbose=0)
        
        y_val_pred = model.predict(X_val_seq, verbose=0).flatten()
        mae = float(mean_absolute_error(y_val_seq, y_val_pred))
        rmse = float(root_mean_squared_error(y_val_seq, y_val_pred))
        print(f"TensorFlow Sequence Model Trained! Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        return model, {"mae": mae, "rmse": rmse}
    except Exception as e:
        print(f"TensorFlow sequence model omitted ({e}). Continuing with primary RandomForest baseline.")
        return None, None


def compute_empirical_intervals(y_true: np.ndarray, y_pred: np.ndarray, quantiles: list = [0.05, 0.95]) -> dict:
    """Compute empirical residual bounds (prediction intervals) from validation set."""
    residuals = y_true - y_pred
    bounds = np.quantile(residuals, quantiles).tolist()
    return {
        "lower_quantile_5": bounds[0],
        "upper_quantile_95": bounds[1],
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals))
    }


def run_pipeline():
    """Execute full dataset loading, splitting, model training, evaluation, and artifact saving."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Ingest & Validate
    train_df, test_df, test_rul, checksums, sensor_meta, useful_features = get_validated_dataset()
    
    # 2. Split by Engine & Scale
    train_split, val_split, scaler, scaled_cols = prepare_engine_splits(train_df, useful_features)
    
    # 3. Train Primary RandomForest Model
    X_train = train_split[useful_features].values
    y_train = train_split["RUL_clipped"].values
    X_val = val_split[useful_features].values
    y_val = val_split["RUL_clipped"].values
    
    start_time = time.time()
    rf_model = train_random_forest_baseline(X_train, y_train)
    training_time_sec = time.time() - start_time
    
    # 4. Evaluate Validation Metrics
    y_val_pred = rf_model.predict(X_val)
    val_mae = float(mean_absolute_error(y_val, y_val_pred))
    val_rmse = float(root_mean_squared_error(y_val, y_val_pred))
    print(f"RandomForest Validation Performance -> MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
    
    # 5. Compute Empirical Residual Bounds
    residual_meta = compute_empirical_intervals(y_val, y_val_pred)
    
    # 6. Optional TensorFlow Model Execution
    X_tr_seq, y_tr_seq = create_sequence_windows(train_split, useful_features, sequence_length=30)
    X_val_seq, y_val_seq = create_sequence_windows(val_split, useful_features, sequence_length=30)
    tf_model, tf_metrics = attempt_tensorflow_model(X_tr_seq, y_tr_seq, X_val_seq, y_val_seq)
    
    # 7. Save Production Artifacts
    model_path = os.path.join(MODEL_DIR, "baseline_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    feat_order_path = os.path.join(MODEL_DIR, "feature_order.json")
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")
    
    joblib.dump(rf_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    with open(feat_order_path, "w") as f:
        json.dump(useful_features, f, indent=2)
        
    metadata = {
        "model_name": "RandomForestRegressor_FD001",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": round(training_time_sec, 2),
        "primary_target": "RUL_clipped (max=125)",
        "feature_count": len(useful_features),
        "feature_order": useful_features,
        "scaled_feature_order": scaled_cols,
        "validation_metrics": {
            "mae": round(val_mae, 4),
            "rmse": round(val_rmse, 4)
        },
        "empirical_prediction_intervals": residual_meta,
        "tf_sequence_model_metrics": tf_metrics,
        "checksums": checksums,
        "sensor_metadata": sensor_meta,
        "window_length": 30
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Artifacts saved successfully to {MODEL_DIR}")
    return metadata, val_split, rf_model, scaler, useful_features


if __name__ == "__main__":
    run_pipeline()
