"""Interactive Streamlit Demo UI for Recovery Decision Engine."""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from src.simulation.schemas import FailureContext
from src.modeling.predictor import estimate_all_actions
from src.policy.engine import evaluate_policy
from src.policy.costs import ACTION_COSTS
from src.state_machine.fsm import RecoveryFSM

# Page Configuration
st.set_page_config(
    page_title="Recovery Decision Engine — Interactive Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .card-execute {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        padding: 1.2rem;
        border-radius: 10px;
        color: #065f46;
    }
    .card-escalate {
        background-color: #fffbe6;
        border: 2px solid #f59e0b;
        padding: 1.2rem;
        border-radius: 10px;
        color: #92400e;
    }
    .card-abstain {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        padding: 1.2rem;
        border-radius: 10px;
        color: #991b1b;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ Recovery Decision Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Economics-Aware Subscription Recovery with Bayesian Uncertainty Gating</div>', unsafe_allow_html=True)

# Preset Scenarios
st.sidebar.markdown("### 🎯 One-Click Demo Presets")

preset_col1, preset_col2, preset_col3 = st.sidebar.columns(3)

default_decline = "insufficient_funds"
default_day = 1
default_tenure = 12
default_value = 999.0
default_retry = 0
default_hours = 2.0

if preset_col1.button("🟢 Execute"):
    default_decline = "insufficient_funds"
    default_day = 1  # Payday!
    default_tenure = 14
    default_value = 999.0
    default_retry = 0
    default_hours = 2.0

if preset_col2.button("🟡 Escalate"):
    default_decline = "expired_card"
    default_day = 15
    default_tenure = 2
    default_value = 1499.0
    default_retry = 2
    default_hours = 12.0

if preset_col3.button("🔴 Abstain"):
    default_decline = "stolen_card"
    default_day = 10
    default_tenure = 4
    default_value = 499.0
    default_retry = 1
    default_hours = 24.0

# Sidebar Form Input
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Event Context Parameters")

with st.sidebar.form(key="context_form"):
    decline_code = st.selectbox(
        "Decline Code",
        ["insufficient_funds", "issuer_unavailable", "expired_card", "do_not_honor", "stolen_card"],
        index=["insufficient_funds", "issuer_unavailable", "expired_card", "do_not_honor", "stolen_card"].index(default_decline),
    )
    day_of_month = st.slider("Day of Month (Payday = 1, 30, 31)", 1, 31, default_day)
    subscription_value = st.number_input("Subscription Value (₹)", min_value=99.0, max_value=10000.0, value=default_value, step=100.0)
    customer_tenure_months = st.number_input("Customer Tenure (Months)", min_value=0, max_value=120, value=default_tenure)
    retry_count = st.number_input("Prior Retries Count", min_value=0, max_value=10, value=default_retry)
    hours_since_failure = st.number_input("Hours Elapsed Since Failure", min_value=0.0, max_value=168.0, value=default_hours)
    prior_recovery_outcome = st.selectbox("Prior Recovery Outcome", ["none", "recovered", "churned"])
    tau_threshold = st.slider("Uncertainty Threshold (tau)", 0.10, 0.50, 0.25, 0.05)

    submit_btn = st.form_submit_button("🚀 Evaluate Recovery Decision")

# Main Content Execution
ctx = FailureContext(
    event_id=f"demo_evt_{datetime.utcnow().strftime('%H%M%S')}",
    decline_code=decline_code,
    retry_count=retry_count,
    hours_since_failure=hours_since_failure,
    day_of_month=day_of_month,
    customer_tenure_months=customer_tenure_months,
    subscription_value=subscription_value,
    prior_recovery_outcome=prior_recovery_outcome,
)

# Run Inference & Policy Engine
estimates = estimate_all_actions(ctx)
decision = evaluate_policy(ctx.event_id, ctx.subscription_value, estimates, tau_threshold=tau_threshold)

# Display Decision Banner
col_banner, col_metrics = st.columns([1.5, 1])

with col_banner:
    if decision.chosen == "execute":
        st.markdown(f"""
            <div class="card-execute">
                <h2 style="margin:0;">✅ DECISION: EXECUTE ACTION</h2>
                <h3 style="margin-top:0.5rem; color:#047857;">Chosen Action: <code>{decision.action.upper()}</code></h3>
                <p>Expected Recovery Value: <b>₹{decision.expected_value:,.2f}</b></p>
                <p>Reasoning: <i>{decision.reasoning.get("decision_rule_triggered")}</i></p>
            </div>
        """, unsafe_allow_html=True)
    elif decision.chosen == "escalate":
        st.markdown(f"""
            <div class="card-escalate">
                <h2 style="margin:0;">⚠️ DECISION: ESCALATE TO HUMAN</h2>
                <h3 style="margin-top:0.5rem; color:#b45309;">Candidate Action: <code>{decision.action.upper()}</code></h3>
                <p>Expected Recovery Value: <b>₹{decision.expected_value:,.2f}</b></p>
                <p>Reasoning: <i>{decision.reasoning.get("decision_rule_triggered")}</i></p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="card-abstain">
                <h2 style="margin:0;">🛑 DECISION: ABSTAIN FROM INTERVENTION</h2>
                <h3 style="margin-top:0.5rem; color:#b91c1c;">No Action Attempted</h3>
                <p>Expected Recovery Value: <b>₹{decision.expected_value:,.2f}</b></p>
                <p>Reasoning: <i>{decision.reasoning.get("decision_rule_triggered")}</i></p>
            </div>
        """, unsafe_allow_html=True)

with col_metrics:
    st.metric("Subscription Value", f"₹{ctx.subscription_value:,.2f}")
    st.metric("Decline Risk Code", ctx.decline_code)
    st.metric("Uncertainty Gate (tau)", f"{tau_threshold:.2f}")

st.markdown("---")
st.markdown("### 📊 Candidate Action Estimates & EV Breakdown")

# Format Table
table_data = []
for est in estimates:
    ev = round((est.p_success * ctx.subscription_value) - est.cost, 2)
    hdi_width = round(est.hdi_high - est.hdi_low, 4)
    is_chosen = (est.action == decision.action) if decision.action else False

    table_data.append({
        "Action": est.action + (" ⭐ (Chosen)" if is_chosen else ""),
        "P(Success) Mean": f"{est.p_success*100:.1f}%",
        "90% Credible Interval": f"[{est.hdi_low*100:.1f}%, {est.hdi_high*100:.1f}%]",
        "HDI Width": f"{hdi_width:.4f}",
        "Intervention Cost": f"₹{est.cost:.2f}",
        "Expected Value (EV)": f"₹{ev:,.2f}",
    })

df_estimates = pd.DataFrame(table_data)
st.dataframe(df_estimates, use_container_width=True)

st.markdown("---")
st.markdown("### 📜 Audit Log Inspection")
st.json({
    "event_id": decision.event_id,
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "context": ctx.model_dump(),
    "decision": decision.model_dump(),
})
