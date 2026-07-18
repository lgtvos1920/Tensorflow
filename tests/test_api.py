from fastapi.testclient import TestClient
import pytest
import os
import json
from app.main import app
from app.config import BASE_DIR, WINDOW_SIZE, NUM_FEATURES, FEATURE_ORDER
from app.api.endpoints import predictor_service

client = TestClient(app)

# Load official engine payloads from Member A's artifacts file
payloads_file = os.path.join(BASE_DIR, "models", "sample_payloads.json")
if os.path.exists(payloads_file):
    with open(payloads_file, "r") as f:
        SAMPLE_PAYLOADS = json.load(f)
else:
    raise FileNotFoundError(f"Missing official integration payloads at {payloads_file}")

# Format the dict payloads into lists of lists of floats to match the API contract
def get_sensor_window_payload(raw_case):
    seq = raw_case["sequence_30_cycle_payload"]
    return [[row[feat] for feat in FEATURE_ORDER] for row in seq]

def test_health_endpoint():
    """
    Verifies /health endpoint returns 200 OK and correct structure.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "api_version" in data
    assert isinstance(data["model_loaded"], bool)

def test_model_info_endpoint():
    """
    Verifies /model-info returns metadata matching the active Random Forest model.
    """
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "RandomForestRegressor_FD001"
    assert data["model_version"] == "1.3.0"
    assert "framework" in data
    assert data["expected_window_size"] == WINDOW_SIZE
    assert data["expected_num_features"] == NUM_FEATURES
    assert "metrics" in data
    assert "MAE" in data["metrics"]
    assert "RMSE" in data["metrics"]

def test_predict_rul_engine_54_valid():
    """
    Tests /predict/rul using the official Engine 54 payload.
    Verifies the newly added prediction interval level, coverage, description, and strategy.
    """
    case_data = SAMPLE_PAYLOADS["engine_54_successful"]
    window = get_sensor_window_payload(case_data)
    
    payload = {
        "engine_id": case_data["unit_nr"],
        "cycle": case_data["total_cycles"],
        "sensor_window": window
    }
    
    response = client.post("/predict/rul", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["engine_id"] == 54
    assert data["cycle"] == case_data["total_cycles"]
    assert "estimated_rul" in data
    assert "prediction_interval_lower" in data
    assert "prediction_interval_upper" in data
    assert data["prediction_interval_level"] == 0.90
    assert data["prediction_interval_coverage"] == 0.8998
    assert data["prediction_interval_description"] == "Bounds use validation-residual 5th and 95th percentiles."
    assert "sequence_conversion_strategy" in data
    assert data["risk_level"] in ("Low", "Medium", "High", "Critical")
    assert 0.0 <= data["data_quality_score"] <= 1.0
    assert "recommendation" in data
    assert data["model_name"] == "RandomForestRegressor_FD001"
    assert data["model_version"] == "1.3.0"

def test_predict_rul_engine_74_valid():
    """
    Tests /predict/rul using the official Engine 74 payload.
    """
    case_data = SAMPLE_PAYLOADS["engine_74_difficult"]
    window = get_sensor_window_payload(case_data)
    
    payload = {
        "engine_id": case_data["unit_nr"],
        "cycle": case_data["total_cycles"],
        "sensor_window": window
    }
    response = client.post("/predict/rul", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["engine_id"] == 74
    assert data["cycle"] == case_data["total_cycles"]
    assert "estimated_rul" in data
    assert data["prediction_interval_level"] == 0.90
    assert data["prediction_interval_coverage"] == 0.8998
    assert data["prediction_interval_description"] == "Bounds use validation-residual 5th and 95th percentiles."
    assert "sequence_conversion_strategy" in data
    assert data["risk_level"] in ("Low", "Medium", "High", "Critical")

def test_predict_rul_invalid_dimensions():
    """
    Tests that sequences with incorrect shapes (not 30x16) are rejected.
    """
    # Malformed features: 15 features instead of 16
    bad_features_window = [[1.0] * (NUM_FEATURES - 1)] * WINDOW_SIZE
    payload = {
        "engine_id": 54,
        "cycle": 257,
        "sensor_window": bad_features_window
    }
    response = client.post("/predict/rul", json=payload)
    assert response.status_code == 422
    
    # Malformed timesteps: 29 timesteps instead of 30
    bad_timesteps_window = [[1.0] * NUM_FEATURES] * (WINDOW_SIZE - 1)
    payload_time = {
        "engine_id": 54,
        "cycle": 257,
        "sensor_window": bad_timesteps_window
    }
    response_time = client.post("/predict/rul", json=payload_time)
    assert response_time.status_code == 422

def test_predict_rul_non_finite():
    """
    Tests that requests with non-finite values (NaN, Inf) are rejected.
    """
    bad_window = [[1.0] * NUM_FEATURES] * WINDOW_SIZE
    bad_window[0][0] = float("nan")
    
    payload = {
        "engine_id": 54,
        "cycle": 257,
        "sensor_window": bad_window
    }
    response = client.post("/predict/rul", json=payload)
    assert response.status_code == 422

def test_missing_artifacts_behavior():
    """
    Simulates missing model artifacts to test the explicit model-unavailable state.
    Health endpoint should remain online with model_loaded: false,
    and predictions/model-info should return 503 Service Unavailable.
    """
    model_dir = os.path.join(BASE_DIR, "models")
    model_file = os.path.join(model_dir, "baseline_model.joblib")
    temp_file = os.path.join(model_dir, "baseline_model.joblib.bak")
    
    # If the file exists, temporarily rename it to simulate missing model
    if os.path.exists(model_file):
        os.rename(model_file, temp_file)
        
    try:
        # Trigger reload of the predictor service to check directory
        predictor_service.__init__()
        
        # 1. Health check should still return 200 but model_loaded: False
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["model_loaded"] is False
        
        # 2. Predict should return 503 Service Unavailable
        case_data = SAMPLE_PAYLOADS["engine_54_successful"]
        window = get_sensor_window_payload(case_data)
        payload = {
            "engine_id": case_data["unit_nr"],
            "cycle": case_data["total_cycles"],
            "sensor_window": window
        }
        pred_resp = client.post("/predict/rul", json=payload)
        assert pred_resp.status_code == 503
        
        # 3. Model Info should return 503 Service Unavailable
        info_resp = client.get("/model-info")
        assert info_resp.status_code == 503
        
    finally:
        # Restore the model file and restore service state
        if os.path.exists(temp_file):
            os.rename(temp_file, model_file)
        predictor_service.__init__()
        
    # Verify it restored successfully
    assert predictor_service.model_loaded is True
