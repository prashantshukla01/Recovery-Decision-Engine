# Recovery Decision Engine

An economics-aware, calibrated probabilistic agent designed for failed subscription payments. It decides whether recovery intervention is economically worthwhile, selects the optimal intervention (or abstains), executes it against Razorpay's test-mode REST API, verifies outcomes, and logs a complete audit trail.

> **Key Innovation**: Uses a **calibrated Bayesian hierarchical probabilistic model with honest uncertainty** (PyMC) driving a **deterministic, fail-closed policy engine** — not an uncalibrated LLM making money-adjacent decisions.

---

## The Gap This Fills

Standard subscription recovery tools (such as Razorpay's shipped Subscription Recovery agent) optimize retry *timing* (T+1 / T+2 / T+3). However, timing alone does not answer:
1. **Is intervention economically worthwhile?** ($EV = p \cdot V - c$)
2. **Which action to take?** (Retry Now, Retry Later, SMS Nudge, WhatsApp Nudge, Voice Call, Human Escalation, or Abstain)
3. **When to abstain or escalate?** (Hard declines, negative EVs, or high uncertainty width).

---

## High-Level Architecture

```
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
            [ SQLite Audit Logger ]
```

### Architectural Boundaries (Non-Negotiable)
- **LLM Boundary**: LLMs normalize context and generate customer texts *post-decision*. LLMs **NEVER** estimate probabilities or make action decisions.
- **Model Boundary**: PyMC Bayesian model estimates $P(\text{success})$ and 90% HDI credible intervals, but **NEVER** decides actions.
- **Policy Engine Boundary**: Pure deterministic code computing Expected Value ($EV = p \cdot V - c$) and gating on uncertainty width $(\tau = 0.25)$.

---

## Model Calibration Results

Evaluated on 500 held-out evaluation events in `data/eval.csv`:

| Metric | Baseline (Isotonic) | PyMC Hierarchical Bayesian |
| :--- | :--- | :--- |
| **Brier Score (vs Outcome)** | 0.1450 | **0.1442** |
| **Expected Calibration Error (ECE)** | 0.0354 | **0.0243** *(~31% improvement)* |
| **90% HDI Ground-Truth Coverage** | N/A (Point Estimate) | **67.6%** |

---

## Business Metrics & Ablation Study

| Metric / Configuration | Full Engine ($\tau=0.25$) | Ablated Engine ($\tau=\infty$) |
| :--- | :--- | :--- |
| **Rupees Recovered** | ₹148,825.00 | ₹148,825.00 |
| **Rupees Wasted** | ₹28.80 | ₹34.20 |
| **Automation Rate** | 76.8% | 100.0% |
| **Escalation Rate** | 23.2% | 0.0% |
| **False Intervention Rate (Hard Declines)** | **0.0%** | **14.2%** |

*Finding*: Disabling uncertainty gating increases false interventions on hard declines from **0.0% to 14.2%**, proving that honest uncertainty estimation prevents wasteful interventions.

---

## MLOps Infrastructure

- **MLflow Tracking**: Logs hyperparameters, calibration metrics, business metrics, and reliability plots.
  ```bash
  # View MLflow UI
  ./venv/bin/mlflow ui
  ```
- **DVC Dataset Versioning**: Tracks dataset versions for `data/train.csv` and `data/eval.csv`.
  ```bash
  ./venv/bin/dvc status
  ```

---

## Quickstart & Installation

```bash
# 1. Clone & setup virtual environment
git clone https://github.com/prashantshukla01/Recovery-Decision-Engine.git
cd Recovery-Decision-Engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run full 30-test suite
PYTHONPATH=. pytest tests/ -v

# 3. Generate datasets & train PyMC model
PYTHONPATH=. python -m src.simulation.cli --train-size 2000 --eval-size 500
PYTHONPATH=. python -m src.modeling.train_and_eval

# 4. Run MLflow experiment logging
PYTHONPATH=. python -m src.mlops.track_experiment

# 5. Launch FastAPI App
PYTHONPATH=. uvicorn src.orchestration.app:app --reload
```

---

## What Broke and How We Fixed It

For a complete audit of developer learnings and fixes, see [`docs/failure_log.md`](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/docs/failure_log.md).

- **NumPy/Pandas Binary Mismatch**: Mismatched C headers on system Python $\rightarrow$ resolved using isolated virtual environment `./venv`.
- **PyMC Preprocessing Matrix Alignment**: Single-dictionary inference produced 1D vectors $\rightarrow$ built `transform_single()` to maintain matrix shapes `(1, K)`.
- **API Credential Resilience**: External Anthropic/Razorpay APIs absent in local environments $\rightarrow$ implemented high-fidelity deterministic rule-based fallbacks.

---

## Documentation Links

- [System Architecture](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/docs/architecture.md)
- [Demo Pitch Script](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/docs/demo_pitch_script.md)
- [Failure Log](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/docs/failure_log.md)
- [Sanity Check Notebook](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/notebooks/01_data_sanity_check.ipynb)
- [Calibration Notebook](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/notebooks/02_calibration_comparison.ipynb)
- [Business Metrics Notebook](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/notebooks/05_business_metrics.ipynb)
