"""
Export Sample Payloads for Engine 54 (Successful) and Engine 74 (Difficult)
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
    
    # Engine 54 (Successful) and Engine 74 (Difficult)
    e54 = val_df[val_df["unit_nr"] == 54].sort_values("cycle")
    e74 = val_df[val_df["unit_nr"] == 74].sort_values("cycle")
    
    # 1. Single snapshot payload (latest cycle)
    payload_54_single = e54[useful_feats].iloc[-1].to_dict()
    payload_74_single = e74[useful_feats].iloc[-1].to_dict()
    
    # 2. 30-cycle sequence payload (last 30 cycles)
    payload_54_sequence_30 = e54[useful_feats].tail(30).to_dict(orient="records")
    payload_74_sequence_30 = e74[useful_feats].tail(30).to_dict(orient="records")
    
    # Predictions for verification
    pred_54 = predict_rul(payload_54_single)
    pred_74 = predict_rul(payload_74_single)
    
    sample_payloads = {
        "engine_54_successful": {
            "unit_nr": 54,
            "description": "Engine 54: Successful low-error trajectory case.",
            "total_cycles": len(e54),
            "actual_final_rul": float(e54["RUL_clipped"].iloc[-1]),
            "single_cycle_payload": payload_54_single,
            "sequence_30_cycle_payload": payload_54_sequence_30,
            "expected_predict_rul_response": pred_54
        },
        "engine_74_difficult": {
            "unit_nr": 74,
            "description": "Engine 74: Difficult non-linear degradation case.",
            "total_cycles": len(e74),
            "actual_final_rul": float(e74["RUL_clipped"].iloc[-1]),
            "single_cycle_payload": payload_74_single,
            "sequence_30_cycle_payload": payload_74_sequence_30,
            "expected_predict_rul_response": pred_74
        }
    }
    
    out_path = os.path.join(MODEL_DIR, "sample_payloads.json")
    with open(out_path, "w") as f:
        json.dump(sample_payloads, f, indent=2)
        
    print(f"Sample payloads successfully exported to {out_path}")
    return sample_payloads


if __name__ == "__main__":
    export_member_b_payloads()
