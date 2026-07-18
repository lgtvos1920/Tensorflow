import streamlit as st
import httpx
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import json

# Set page config for a premium, widescreen layout
st.set_page_config(
    page_title="AeroShield | Predictive Maintenance Integration",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Apple-inspired dark mode UI & glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main container override */
    .stApp {
        background-color: #0b0f19;
        color: #f5f5f7;
    }
    
    /* Header layout */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 30px;
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ffffff 30%, #a0a0a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #8e8e93;
        margin-top: 4px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Glassmorphic Cards */
    .card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 24px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* Card internal structures */
    .card-title {
        font-size: 0.85rem;
        color: #8e8e93;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
    }
    .card-value {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .card-subtitle {
        font-size: 0.85rem;
        color: #8e8e93;
        margin-top: 8px;
    }
    
    /* Custom Risk Badges */
    .badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        text-align: center;
        margin-top: 10px;
    }
    .badge-low {
        background: rgba(52, 199, 89, 0.12);
        color: #30d158;
        border: 1px solid rgba(52, 199, 89, 0.25);
    }
    .badge-medium {
        background: rgba(255, 214, 10, 0.12);
        color: #ffd60a;
        border: 1px solid rgba(255, 214, 10, 0.25);
    }
    .badge-high {
        background: rgba(255, 159, 10, 0.12);
        color: #ff9f0a;
        border: 1px solid rgba(255, 159, 10, 0.25);
    }
    .badge-critical {
        background: rgba(255, 69, 58, 0.12);
        color: #ff453a;
        border: 1px solid rgba(255, 69, 58, 0.25);
    }
    
    /* Recommendation Card Styling */
    .rec-box {
        border-left: 5px solid;
        border-radius: 12px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.02);
        margin-top: 10px;
    }
    .rec-box-low { border-left-color: #30d158; }
    .rec-box-medium { border-left-color: #ffd60a; }
    .rec-box-high { border-left-color: #ff9f0a; }
    .rec-box-critical { border-left-color: #ff453a; }
    
    /* Health Badges */
    .health-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 10px;
    }
    .dot-green { background-color: #30d158; box-shadow: 0 0 10px #30d158; }
    .dot-red { background-color: #ff453a; box-shadow: 0 0 10px #ff453a; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration & Environment variables
# ---------------------------------------------------------
st.sidebar.markdown("<h2 style='letter-spacing: -0.5px;'>Configuration</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
backend_url = st.sidebar.text_input("API Gateway Endpoint", value=DEFAULT_BACKEND)

st.sidebar.markdown("### Case Study Selector")

# Base directory for resolving models relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load engine examples and metadata from Member A's artifacts dynamically
@st.cache_data
def load_model_artifacts():
    examples_path = os.path.join(BASE_DIR, "models", "engine_examples.json")
    metadata_path = os.path.join(BASE_DIR, "models", "metadata.json")
    
    with open(examples_path, "r") as f:
        examples = json.load(f)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    return examples, metadata

try:
    examples, metadata = load_model_artifacts()
    feature_order = metadata.get("feature_order", [])
    window_length = metadata.get("window_length", 30)
    num_features = len(feature_order)
    model_name = metadata.get("model_name", "RandomForestRegressor_FD001")
    model_version = metadata.get("version", "1.0.0")
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Support Engine 54 and Engine 74 case studies from engine_examples.json
engine_options = {
    54: {
        "key": "successful_engine_example",
        "label": "Engine #054 (successful low-error case)",
        "rmse": examples["successful_engine_example"]["metrics"]["rmse"],
        "mae": examples["successful_engine_example"]["metrics"]["mae"]
    },
    74: {
        "key": "difficult_engine_example",
        "label": "Engine #074 (difficult high-error case)",
        "rmse": examples["difficult_engine_example"]["metrics"]["rmse"],
        "mae": examples["difficult_engine_example"]["metrics"]["mae"]
    }
}

selected_engine_id = st.sidebar.selectbox(
    "Active Turbofan Engine",
    options=list(engine_options.keys()),
    format_func=lambda x: engine_options[x]["label"]
)

# Extract trajectory history for selected engine
active_case = examples[engine_options[selected_engine_id]["key"]]
max_cycles = active_case["data"]["total_cycles"]
latest_features = active_case["data"]["latest_unscaled_features"]

# Cycle selector slider (starts at window_length so we always have a full 30-cycle window)
selected_cycle = st.sidebar.slider(
    "Operational Flight Cycle",
    min_value=window_length,
    max_value=max_cycles,
    value=min(100, max_cycles)
)

# ---------------------------------------------------------
# Dynamic Sensor Window Simulator
# ---------------------------------------------------------
def simulate_sensor_window(engine_id: int, target_cycle: int, total_cycles: int, final_features: dict, feat_order: list) -> list:
    """
    Interpolates sensor readings back to cycle 1 starting from nominal values,
    and returns a sequence of 30 cycles ending at target_cycle.
    """
    np.random.seed(engine_id * 13 + target_cycle)
    
    # We need a 30-cycle window ending at target_cycle
    window_cycles = np.arange(target_cycle - window_length + 1, target_cycle + 1)
    
    simulated_window = []
    for c in window_cycles:
        row = []
        # Fraction of engine operational lifetime
        frac = c / total_cycles
        
        for f_name in feat_order:
            final_val = final_features[f_name]
            
            # Predict realistic drift depending on whether sensor typically rises or falls
            # e.g., T24, T30, T50, Nc, Ps30, BPR, ht rise; P30, Nf, etc. fall
            is_rising = any(x in f_name for x in ["sensor_2", "sensor_3", "sensor_4", "sensor_8", "sensor_9", "sensor_11", "sensor_13", "sensor_14", "sensor_15", "sensor_17"])
            if is_rising:
                nominal = final_val * 0.92
            else:
                nominal = final_val * 1.08
                
            # Linear wear interpolation with random gaussian noise
            noise_scale = abs(final_val) * 0.003
            noise = np.random.normal(0, noise_scale)
            value = nominal + (final_val - nominal) * frac + noise
            row.append(round(value, 4))
            
        simulated_window.append(row)
        
    return simulated_window

# ---------------------------------------------------------
# API Connection Diagnostic
# ---------------------------------------------------------
api_health = {"status": "Disconnected", "latency": 0.0, "model_loaded": False}
try:
    start_time = time.perf_counter()
    response = httpx.get(f"{backend_url}/health", timeout=1.0)
    end_time = time.perf_counter()
    if response.status_code == 200:
        data = response.json()
        api_health["status"] = "Healthy"
        api_health["latency"] = (end_time - start_time) * 1000
        api_health["model_loaded"] = data.get("model_loaded", False)
except Exception:
    api_health["status"] = "Disconnected"

# Header Section
st.markdown(f"""
<div class="header-container">
    <div>
        <h1 class="brand-title">AERO-SHIELD</h1>
        <div class="brand-subtitle">Model Integration & Control Center</div>
    </div>
    <div>
        <div class="health-indicator">
            <span class="dot {'dot-green' if api_health['status'] == 'Healthy' and api_health['model_loaded'] else 'dot-red'}"></span>
            <span>API Gateway: {api_health['status']} {"(Model Active)" if api_health['model_loaded'] else "(Model Inactive)" if api_health['status'] == 'Healthy' else ''}</span>
            {f"<span style='color: #8e8e93; font-size: 0.85rem; margin-left: 10px;'>({api_health['latency']:.1f}ms)</span>" if api_health['status'] == 'Healthy' else ''}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Real Prediction Query
# ---------------------------------------------------------
sensor_window_data = simulate_sensor_window(
    selected_engine_id, selected_cycle, max_cycles, latest_features, feature_order
)

prediction_result = None
prediction_error = None

if api_health["status"] == "Healthy":
    if api_health["model_loaded"]:
        payload = {
            "engine_id": selected_engine_id,
            "cycle": selected_cycle,
            "sensor_window": sensor_window_data
        }
        try:
            resp = httpx.post(f"{backend_url}/predict/rul", json=payload, timeout=2.0)
            if resp.status_code == 200:
                prediction_result = resp.json()
            elif resp.status_code == 503:
                prediction_error = "Server reports model artifacts are unavailable (503)."
            else:
                prediction_error = f"API Error: Server returned status {resp.status_code}"
        except Exception as ex:
            prediction_error = f"Connection Timeout / Failure: {type(ex).__name__}"
    else:
        prediction_error = "Model files are not loaded on the backend FastAPI server."
else:
    prediction_error = "API is disconnected. Please start the backend FastAPI server."

# ---------------------------------------------------------
# Main Panel UI
# ---------------------------------------------------------

# Section: Model Stats & Metrics Cards
st.markdown("### Model & Verification Metadata")
col_diag1, col_diag2, col_diag3, col_diag4 = st.columns(4)

with col_diag1:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Active Model</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #fff;">{model_name}</div>
        <div class="card-subtitle">Version: {model_version}</div>
    </div>
    """, unsafe_allow_html=True)

with col_diag2:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Case Study Target</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #fff;">Engine #{selected_engine_id:03d}</div>
        <div class="card-subtitle">Total runtime: {max_cycles} flight cycles</div>
    </div>
    """, unsafe_allow_html=True)

with col_diag3:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Historic Validation MAE</div>
        <div style="font-size: 1.5rem; font-weight: 800; color: #ffd60a;">{engine_options[selected_engine_id]['mae']:.3f}</div>
        <div class="card-subtitle">Mean Absolute Error (cycles)</div>
    </div>
    """, unsafe_allow_html=True)

with col_diag4:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Historic Validation RMSE</div>
        <div style="font-size: 1.5rem; font-weight: 800; color: #ffd60a;">{engine_options[selected_engine_id]['rmse']:.3f}</div>
        <div class="card-subtitle">Root Mean Squared Error (cycles)</div>
    </div>
    """, unsafe_allow_html=True)

# Main results layout if prediction was successful
if prediction_result:
    st.markdown("### Remaining Useful Life (RUL) Insights")
    
    # 4 KPI Columns
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Estimated RUL</div>
            <div class="card-value">{prediction_result['estimated_rul']:.1f} <span style="font-size: 1.1rem; font-weight: 400; color: #8e8e93;">Cycles</span></div>
            <div class="card-subtitle">Remaining operational lifetime</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi2:
        pi_low = prediction_result["prediction_interval_lower"]
        pi_high = prediction_result["prediction_interval_upper"]
        st.markdown(f"""
        <div class="card">
            <div class="card-title">empirical 90% prediction interval</div>
            <div class="card-value" style="font-size: 2rem;">[{pi_low:.1f}, {pi_high:.1f}]</div>
            <div class="card-subtitle">Empirical validation bounds</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi3:
        risk = prediction_result["risk_level"]
        badge_class = f"badge-{risk.lower()}"
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Risk Level</div>
            <div>
                <span class="badge {badge_class}">{risk}</span>
            </div>
            <div class="card-subtitle">System warning escalation rank</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi4:
        dq = prediction_result["data_quality_score"]
        dq_percent = dq * 100
        dq_color = "#30d158" if dq >= 0.9 else "#ffd60a" if dq >= 0.7 else "#ff453a"
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Telemetry Integrity</div>
            <div class="card-value">{dq_percent:.1f}%</div>
            <div style="background-color: rgba(255,255,255,0.08); border-radius: 4px; height: 6px; margin-top: 10px; width: 100%;">
                <div style="background-color: {dq_color}; border-radius: 4px; height: 6px; width: {dq_percent}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Human-Review and Recommendation Section
    risk_lower = prediction_result["risk_level"].lower()
    
    st.markdown("### Maintenance Operations & Review")
    col_rec, col_sign = st.columns([2, 1])
    
    with col_rec:
        st.markdown(f"""
        <div class="card" style="height: 100%;">
            <div class="card-title">Maintenance Recommendation</div>
            <div class="rec-box rec-box-{risk_lower}">
                <h4 style="margin: 0 0 8px 0; color: #fff; font-weight: 600;">Action Plan ({prediction_result["risk_level"]} Risk)</h4>
                <p style="margin: 0; font-size: 1.05rem; line-height: 1.5; color: #d1d1d6;">{prediction_result["recommendation"]}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_sign:
        # Microsoft/Amazon style sign-off card (no PII collection)
        with st.container(border=True):
            st.subheader("Human-Review Acknowledgment")
            st.info("Acknowledge review of prediction trajectory prior to logging action item in depot dispatch queue.")
            certify_check = st.checkbox("I acknowledge that I have reviewed the prediction trajectory and confirm receipt of the maintenance recommendation.")
            
            if certify_check:
                st.success("✓ Recommendation review acknowledged.")
else:
    st.markdown("### Remaining Useful Life (RUL) Insights")
    st.warning(f"⚠️ Predictions Unavailable: {prediction_error}")
    st.markdown("""
    <div style="text-align: center; padding: 40px; background: rgba(255,0,0,0.05); border: 1px dashed rgba(255,0,0,0.2); border-radius: 12px; margin-bottom: 30px;">
        <h4 style="color: #ff453a; margin-top:0;">FastAPI Serving Server Offline / Unavailable</h4>
        <p style="color: #8e8e93; margin-bottom:0;">Please run the backend server in your shell to activate inference: <br/>
        <code>uvicorn backend.app.main:app --port 8000</code></p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Historic Trajectory Plot (Actual vs Predicted)
# ---------------------------------------------------------
st.markdown("### Engine Lifetime Trajectory (Case Study)")

# Extract full cycle lists from engine_examples.json
hist_cycles = active_case["data"]["cycles"]
hist_actual = active_case["data"]["actual_rul"]
hist_pred = active_case["data"]["predicted_rul"]

# Empirical prediction intervals bounds (constant offsets from metadata.json)
intervals_meta = metadata.get("empirical_prediction_intervals", {})
lower_offset = intervals_meta.get("lower_quantile_5", -34.28)
upper_offset = intervals_meta.get("upper_quantile_95", 24.08)

# Calculate trajectory bounds
hist_pred_lower = [max(0.0, float(val + lower_offset)) for val in hist_pred]
hist_pred_upper = [float(val + upper_offset) for val in hist_pred]

fig_traj = go.Figure()

# Plot shaded empirical confidence interval area
fig_traj.add_trace(go.Scatter(
    x=hist_cycles + hist_cycles[::-1],
    y=hist_pred_upper + hist_pred_lower[::-1],
    fill='toself',
    fillcolor='rgba(255, 214, 10, 0.08)',
    line=dict(color='rgba(255, 255, 255, 0)'),
    hoverinfo="skip",
    name="Empirical 90% Prediction Interval"
))

# Plot Predicted RUL line
fig_traj.add_trace(go.Scatter(
    x=hist_cycles,
    y=hist_pred,
    mode='lines',
    name="Predicted RUL (RandomForest)",
    line=dict(color='#ffd60a', width=3),
    hovertemplate="Cycle %{x}<br>Predicted: %{y:.2f} cycles"
))

# Plot Actual RUL line
fig_traj.add_trace(go.Scatter(
    x=hist_cycles,
    y=hist_actual,
    mode='lines',
    name="Actual RUL",
    line=dict(color='#30d158', width=2, dash='dash'),
    hovertemplate="Cycle %{x}<br>Actual: %{y:.2f} cycles"
))

# Vertical line highlighting current cycle slider
fig_traj.add_vline(
    x=selected_cycle,
    line_width=2,
    line_color="#ff453a",
    annotation_text=f"Current: Cycle {selected_cycle}",
    annotation_position="top right"
)

fig_traj.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8e8e93', family='Outfit'),
    xaxis=dict(
        title="Operational Flight Cycles",
        gridcolor='rgba(255,255,255,0.05)',
        zerolinecolor='rgba(255,255,255,0.05)'
    ),
    yaxis=dict(
        title="Remaining Useful Life (Cycles)",
        gridcolor='rgba(255,255,255,0.05)',
        zerolinecolor='rgba(255,255,255,0.05)',
        range=[-5, 200]
    ),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=450,
    margin=dict(l=0, r=0, t=10, b=0)
)

st.plotly_chart(fig_traj, use_container_width=True)

# ---------------------------------------------------------
# Simulated Sensor Trends View
# ---------------------------------------------------------
st.markdown("### Active Telemetry Windows (Last 30 Cycles)")
st.markdown("Explore normalized trends of the 16 model features inside the current prediction sequence:")

selected_sensors_to_plot = st.multiselect(
    "Select Features to Visualize",
    options=feature_order,
    default=["sensor_2", "sensor_3", "sensor_4", "sensor_11"]
)

if selected_sensors_to_plot:
    # Build history dataframe of simulated sensors up to selected_cycle
    full_history = []
    # We construct historical values from cycle 1 up to selected_cycle
    for c in range(1, selected_cycle + 1):
        # Seed based on cycle index so noise is deterministic
        np.random.seed(selected_engine_id * 13 + c)
        frac = c / max_cycles
        row = {"cycle": c}
        for f_name in feature_order:
            final_val = latest_features[f_name]
            is_rising = any(x in f_name for x in ["sensor_2", "sensor_3", "sensor_4", "sensor_8", "sensor_9", "sensor_11", "sensor_13", "sensor_14", "sensor_15", "sensor_17"])
            if is_rising:
                nominal = final_val * 0.92
            else:
                nominal = final_val * 1.08
            noise_scale = abs(final_val) * 0.003
            noise = np.random.normal(0, noise_scale)
            row[f_name] = nominal + (final_val - nominal) * frac + noise
        full_history.append(row)
        
    history_df = pd.DataFrame(full_history)
    
    fig_sensors = go.Figure()
    for s in selected_sensors_to_plot:
        # Z-score normalization for clean comparative view
        vals = history_df[s].values
        mean_val = np.mean(vals)
        std_val = np.std(vals) if np.std(vals) > 0 else 1.0
        normalized = (vals - mean_val) / std_val
        
        fig_sensors.add_trace(go.Scatter(
            x=history_df["cycle"],
            y=normalized,
            mode='lines',
            name=f"{s} (norm)",
            line=dict(width=2),
            hovertemplate=f"Cycle %{{x}}<br>{s} (norm): %{{y:.2f}}"
        ))
        
    # Vertical line representing the start of the current 30-cycle prediction window
    window_start_cycle = max(1, selected_cycle - window_length + 1)
    fig_sensors.add_vline(
        x=window_start_cycle,
        line_width=1.5,
        line_dash="dash",
        line_color="#ffd60a",
        annotation_text="Window Start",
        annotation_position="top left"
    )
    
    # Highlight current cycle
    fig_sensors.add_vline(
        x=selected_cycle,
        line_width=2,
        line_color="#ff453a",
        annotation_text="Current Cycle",
        annotation_position="top right"
    )

    fig_sensors.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8e8e93', family='Outfit'),
        xaxis=dict(
            title="Operational Cycles",
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.05)'
        ),
        yaxis=dict(
            title="Sensor Standardized Readings (Normalized)",
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.05)'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=380,
        margin=dict(l=0, r=0, t=10, b=0)
    )
    
    st.plotly_chart(fig_sensors, use_container_width=True)
else:
    st.info("Select features to visualize standard sensor trends.")
