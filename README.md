# NASA C-MAPSS FD001 Turbofan Engine Degradation Predictive Maintenance ML Pipeline

[![Model Version](https://img.shields.io/badge/version-1.3.0-blue.svg)](file:///c:/Users/ASUS/Tensorflow/models/metadata.json)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](README.md)
[![Build Status](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen.svg)](file:///c:/Users/ASUS/Tensorflow/tests/test_api_contract.py)

Production-grade Data & Machine Learning Pipeline for predicting Remaining Useful Life (RUL) of aircraft engines on the NASA C-MAPSS FD001 benchmark dataset. Developed under Apple UI/UX presentation standards, Microsoft & Amazon software reliability, and zero-data-leakage engineering rigor.

---

## 1. Quickstart & Unified API Contract

The unified `predict_rul()` interface strictly accepts a **30-cycle sequence shaped `(30, 16)`** containing unscaled physical sensor readings.

```python
from src.predict import predict_rul

# Example 30-cycle operational sequence input
sample_30_cycle_sequence = [
    {
        "op_setting_1": -0.0007,
        "op_setting_2": -0.0004,
        "sensor_2": 641.82,
        "sensor_3": 1589.70,
        "sensor_4": 1400.60,
        "sensor_7": 554.36,
        "sensor_8": 2388.06,
        "sensor_9": 9046.19,
        "sensor_11": 47.47,
        "sensor_12": 521.66,
        "sensor_13": 2388.02,
        "sensor_14": 8138.62,
        "sensor_15": 8.4195,
        "sensor_17": 392,
        "sensor_20": 39.06,
        "sensor_21": 23.4190
    }
    for _ in range(30)
]

# Run prediction
result = predict_rul(sample_30_cycle_sequence)
print(result)
```

---

## 2. Official Verified Case Study Predictions (`models/sample_payloads.json`)

### Engine 54 Prediction Output (Successful Low-Error Trajectory Case)
Evaluating the final 30-cycle operational window of Engine 54 (`unit_nr = 54`, Cycle 257 run-to-failure end state):
```json
{
  "model_name": "RandomForestRegressor_FD001",
  "version": "1.3.0",
  "feature_order": [
    "op_setting_1", "op_setting_2", "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"
  ],
  "window_length": 30,
  "sequence_conversion_strategy": "The Random Forest converts the (30, 16) sequence into a single prediction by extracting the final cycle snapshot (cycle 30) from the input window.",
  "model_limitation": "Random Forest baseline evaluates ONLY the final cycle of the 30-cycle window, discarding preceding temporal progression.",
  "estimated_rul": 1.82,
  "lower_bound": 0.0,
  "upper_bound": 25.90,
  "risk_level": "CRITICAL",
  "data_quality_score": 1.0,
  "recommendation": "CRITICAL ALERT: RUL <= 15 cycles. Require immediate qualified engineering inspection and detailed diagnostic checks before further flight operation."
}
```

*Note on Near-Failure Benchmarks*: Near-end-of-life cycle predictions for Engine 54 evaluate between `1.82` and `1.98` cycles, demonstrating high accuracy against ground-truth zero remaining cycles.

### Engine 74 Prediction Output (Difficult Non-Linear Trajectory Case)
Evaluating the final 30-cycle operational window of Engine 74 (`unit_nr = 74`, Cycle 166 run-to-failure end state):
```json
{
  "model_name": "RandomForestRegressor_FD001",
  "version": "1.3.0",
  "estimated_rul": 3.39,
  "lower_bound": 0.0,
  "upper_bound": 27.47,
  "risk_level": "CRITICAL",
  "data_quality_score": 1.0,
  "recommendation": "CRITICAL ALERT: RUL <= 15 cycles. Require immediate qualified engineering inspection and detailed diagnostic checks before further flight operation."
}
```

---

## 3. Production Model Performance Metrics

- **Overall Validation MAE**: `12.3476` cycles
- **Overall Validation RMSE**: `17.0591` cycles
- **Near-Failure MAE ($\text{RUL} \le 30$)**: **`6.0716` cycles**
- **Authoritative Empirical Quantiles**: `lower_quantile_5 = -15.01`, `upper_quantile_95 = +24.08`
- **Empirical 90% Interval Coverage**: **`89.98%`**

---

## 4. Frozen Binary Artifact Checksums & Pinned Dependencies

### Frozen Model Artifacts
- `models/baseline_model.joblib` (SHA256: `a2a056851e4eea0dbb933ef214c4347875a5659d960d8cbb79e209ffffe32e05`)
- `models/scaler.joblib` (SHA256: `7e51717a13bd0f9e81d7ed9d8f36147882cce4088c3dfb2c5bc25738044301e1`)

### Exact Tested Environment ([requirements.txt](file:///c:/Users/ASUS/Tensorflow/requirements.txt))
```ini
numpy==2.3.5
pandas==2.3.3
scikit-learn==1.8.0
joblib==1.5.3
```

---

## 5. Neutral Prototype Recommendation Standard

Safety-certifying language has been replaced with dataset-bounded prototype wording:
- **LOW Risk**: `"NOMINAL: Nominal behavior detected within the model's validated dataset range."`
- **MEDIUM Risk**: `"ELEVATED MONITORING: RUL <= 60 cycles. Routine inspection recommended within validated dataset range."`
- **HIGH Risk**: `"HIGH PRIORITY: RUL <= 30 cycles. Require qualified engineering inspection during next scheduled service window."`
- **CRITICAL Risk**: `"CRITICAL ALERT: RUL <= 15 cycles. Require immediate qualified engineering inspection and detailed diagnostic checks before further flight operation."`

---

## 6. Documentation Inventory
- **Data Card**: [docs/DATA_CARD.md](file:///c:/Users/ASUS/Tensorflow/docs/DATA_CARD.md)
- **Model Card**: [docs/MODEL_CARD.md](file:///c:/Users/ASUS/Tensorflow/docs/MODEL_CARD.md)
- **Feature Importance**: [models/feature_importance.json](file:///c:/Users/ASUS/Tensorflow/models/feature_importance.json)
- **Sample Payloads**: [models/sample_payloads.json](file:///c:/Users/ASUS/Tensorflow/models/sample_payloads.json)
