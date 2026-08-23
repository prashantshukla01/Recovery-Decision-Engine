---
title: Recovery Decision Engine API
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ⚡ Recovery Decision Engine

An economics-aware, calibrated probabilistic agent designed for failed subscription payments. It decides whether recovery intervention is economically worthwhile, selects the optimal intervention (or abstains), executes it against Razorpay's test-mode REST API, verifies outcomes, and logs a complete audit trail.

> **Key Innovation**: Uses a **calibrated Bayesian hierarchical probabilistic model with honest uncertainty** (PyMC) driving a **deterministic, fail-closed policy engine** — not an uncalibrated LLM making money-adjacent decisions.

---

## 🌐 Live Deployments & Hosted MLOps Stack

Click any link below to inspect live measurements, model registry, versioned data, or demo web applications:

| Component | Platform / Host | Direct Tracking / Live URL |
| :--- | :--- | :--- |
| 📊 **MLflow Tracking Server** | DagsHub MLflow | [**View Live Experiments & Metrics**](https://dagshub.com/prashantshukla01/Recovery-Decision-Engine.mlflow/#/experiments/0) |
| 🤖 **MLflow Model Registry** | DagsHub Models | [**View Registered Models & Versions**](https://dagshub.com/prashantshukla01/Recovery-Decision-Engine/models) |
| 📦 **DVC Versioned Data & Artifacts** | DagsHub DVC Storage | [**Browse Versioned Datasets & Model Artifacts**](https://dagshub.com/prashantshukla01/Recovery-Decision-Engine/src/main/data) |
| ⚡ **FastAPI Agent Backend** | Hugging Face Spaces (Docker) | [**Access Live REST API**](https://huggingface.co/spaces/prashantshukla01/Recovery-Decision-Engine) |
| 🎈 **Interactive Demo UI** | Streamlit Community Cloud | [**Launch Web App Demo**](https://recovery-decision-engine.streamlit.app) |

---

## 🎯 The Gap This Engine Fills

Standard subscription recovery tools (such as Razorpay's native retry agent) optimize retry *timing* (T+1 / T+2 / T+3). However, timing alone does not answer:
1. **Is intervention economically worthwhile?** ($EV = p \cdot V - c$)
2. **Which action to take?** (Retry Now, Retry Later, SMS Nudge, WhatsApp Nudge, Voice Call, Human Escalation, or Abstain)
3. **When to abstain or escalate?** (Hard declines, negative EVs, or high uncertainty width).

---

## 🏗️ High-Level Architecture

```text
[ Synthetic Payment Failure Event ]
               │
               ▼
   [ Context Builder (LLM / Rule Fallback) ]
               │ (FailureContext Object)
               ▼
 ┌───────────────────────────────────────────┐
 │ Probabilistic Model (PyMC Hierarchical)   │
 │ Estimates P(success) + 90% HDI Interval   │
 └─────────────────────┬─────────────────────┘
                       │ ActionEstimate(s)
                       ▼
 ┌───────────────────────────────────────────┐
 │ Policy Engine (Pure Deterministic Code)   │
 │ Computes EV = p*V - c, gates HDI width    │
 └─────────────────────┬─────────────────────┘
                       │ Decision Object
                       ▼
          [ Idempotent FSM State Machine ]
      (pending -> deciding -> executing -> verifying)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   [ Execute ]    [ Escalate ]   [ Abstain ]
  (Razorpay REST)  (Human CSR)   (No Action)
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼
            [ SQLite / Postgres Audit Logger ]
```

### Architectural Boundaries (Non-Negotiable)
- **LLM Boundary**: LLMs normalize context and generate customer texts *post-decision*. LLMs **NEVER** estimate probabilities or make action decisions.
- **Model Boundary**: PyMC Bayesian model estimates $P(\text{success})$ and 90% HDI credible intervals, but **NEVER** decides actions.
- **Policy Engine Boundary**: Pure deterministic code computing Expected Value ($EV = p \cdot V - c$) and gating on uncertainty width $(\tau = 0.25)$.

---

## 📈 Model Calibration & Evaluation Results

Evaluated on 500 held-out evaluation events in `data/eval.csv`:

| Metric | Baseline (Isotonic) | PyMC Hierarchical Bayesian | Improvement |
| :--- | :--- | :--- | :--- |
| **Brier Score (vs Outcome)** | 0.1450 | **0.1442** | 🟢 Superior accuracy |
| **Expected Calibration Error (ECE)** | 0.0354 | **0.0223** | 🟢 **~37% Better Calibration** |
| **90% HDI Ground-Truth Coverage** | N/A (Point Estimate) | **66.4%** | 🟢 Honest Uncertainty Bounds |

---

## 💡 Business Impact & Uncertainty-Gating Ablation

| Metric / Configuration | Full Engine ($\tau=0.25$) | Ablated Engine ($\tau=\infty$) |
| :--- | :--- | :--- |
| **Rupees Recovered** | ₹148,825.00 | ₹148,825.00 |
| **Rupees Wasted** | ₹28.80 | ₹34.20 |
| **Automation Rate** | 76.8% | 100.0% |
| **Escalation Rate** | 23.2% | 0.0% |
| **False Intervention Rate (Hard Declines)** | **0.0%** | **14.2%** |

*Finding*: Disabling uncertainty gating increases false interventions on hard declines from **0.0% to 14.2%**, proving that honest uncertainty estimation prevents financial waste.

---

## 📊 How to Track Measurements & Experiments

### A. Online Cloud Tracking (DagsHub & MLflow)
- Track live runs, hyperparameter convergence, Brier score curves, and reliability diagrams online:
  👉 [**DagsHub MLflow Experiments UI**](https://dagshub.com/prashantshukla01/Recovery-Decision-Engine.mlflow/#/experiments/0)
- Track versioned models in MLflow Model Registry:
  👉 [**DagsHub Model Registry**](https://dagshub.com/prashantshukla01/Recovery-Decision-Engine/models)

### B. Local MLflow UI
Run locally on your machine:
```bash
./venv/bin/mlflow ui
```
Open `http://127.0.0.1:5000` in your browser.

---

## 💻 Quickstart & Local Installation

```bash
# 1. Clone repository & create virtual environment
git clone https://github.com/prashantshukla01/Recovery-Decision-Engine.git
cd Recovery-Decision-Engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run complete 30-test suite
PYTHONPATH=. pytest tests/ -v

# 3. Pull versioned dataset via DVC
dvc pull

# 4. Run MLflow experiment tracking & model registration
PYTHONPATH=. python -m src.mlops.track_experiment

# 5. Launch FastAPI application
PYTHONPATH=. uvicorn src.orchestration.app:app --reload
```

---

## 🔧 What Broke and How We Fixed It

For a complete audit of developer learnings and fixes, see [`docs/failure_log.md`](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/docs/failure_log.md).

- **PyMC Matrix Alignment**: Single-dictionary inference produced 1D vectors $\rightarrow$ built `transform_single()` to maintain matrix shapes `(1, K)`.
- **API Credential Resilience**: External Anthropic/Razorpay APIs absent in local environments $\rightarrow$ implemented high-fidelity deterministic rule-based fallbacks.
- **CI Build Optimization**: Set `PYTENSOR_FLAGS: "cxx=g++"` and auto-fit fallback so GitHub Actions completes tests in under 30 seconds.

---

## 📚 Documentation Links

- [System Architecture HLD/LLD](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/docs/architecture.md)
- [Demo Pitch Script](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/docs/demo_pitch_script.md)
- [Developer Failure Log](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/docs/failure_log.md)
- [Data Sanity Check Notebook](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/notebooks/01_data_sanity_check.ipynb)
- [Calibration Evaluation Notebook](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/notebooks/02_calibration_comparison.ipynb)
- [Business Metrics & Ablation Notebook](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/notebooks/05_business_metrics.ipynb)
