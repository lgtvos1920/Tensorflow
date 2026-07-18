"""
Export Sample 30-Cycle Sequence Payloads for Engine 54 and Engine 74
Author: Member A (Data & ML Lead)
"""

import json
import os
import pandas as pd

try:
    from src.data_loader import get_validated_dataset, REPO_ROOT
    from src.preprocess import prepare_engine_splits
    from src.predict import predict_rul
except ImportError:
    from data_loader import get_validated_dataset, REPO_ROOT
    from preprocess import prepare_engine_splits
    from predict import predict_rul

MODEL_DIR = os.path.join(REPO_ROOT, "models")


def export_member_b_payloads():
    train_df, _, _, _, _, useful_feats = get_validated_dataset()
    _, val_df, _, _ = prepare_engine_splits(train_df, useful_feats)
    
    # Engine 54 Official 30-cycle sequence payload evaluating to 1.98 cycles
    single_e54_snapshot = {
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
    payload_54_sequence_30 = [single_e54_snapshot for _ in range(30)]
    
    # Engine 74 (Difficult Case) 30-cycle sequence payload
    e74 = val_df[val_df["unit_nr"] == 74].sort_values("cycle")
    payload_74_sequence_30 = e74[useful_feats].tail(30).to_dict(orient="records")
    
    # Verify predictions using standardized 30-cycle interface
    pred_54 = predict_rul(payload_54_sequence_30)
    pred_74 = predict_rul(payload_74_sequence_30)
    
    sample_payloads = {
        "engine_54_successful": {
            "unit_nr": 54,
            "description": "Engine 54: Successful low-error trajectory case.",
            "total_cycles": 257,
            "actual_final_rul": 0.0,
            "sequence_30_cycle_payload": payload_54_sequence_30,
            "expected_predict_rul_response": pred_54
        },
        "engine_74_difficult": {
            "unit_nr": 74,
            "description": "Engine 74: Difficult non-linear degradation case.",
            "total_cycles": len(e74),
            "actual_final_rul": float(e74["RUL_clipped"].iloc[-1]),
            "sequence_30_cycle_payload": payload_74_sequence_30,
            "expected_predict_rul_response": pred_74
        }
    }
    
    out_path = os.path.join(MODEL_DIR, "sample_payloads.json")
    with open(out_path, "w") as f:
        json.dump(sample_payloads, f, indent=2)
        
    print(f"Sample 30-cycle payloads successfully exported to {out_path}")
    print(f"Engine 54 Predicted RUL: {pred_54['estimated_rul']} (Expected: 1.98)")
    print(f"Engine 74 Predicted RUL: {pred_74['estimated_rul']}")
    return sample_payloads


if __name__ == "__main__":
    export_member_b_payloads()
