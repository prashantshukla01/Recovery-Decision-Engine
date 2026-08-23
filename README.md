---
title: Recovery Decision Engine API
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Recovery Decision Engine

![tests](https://github.com/prashantshukla01/Recovery-Decision-Engine/actions/workflows/test.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

A calibrated, uncertainty-aware decision engine for failed subscription payments. Built for the Razorpay AI Builder buildathon (AI Revenue Recovery track).

Given a failed subscription payment, the engine estimates the probability of recovery for each candidate intervention — with an honest, calibrated confidence interval, not just a point score — computes the expected economic value of each option, and either executes the best one, escalates to a human, or abstains, all with a complete audit trail.

**Live demo:** [Streamlit UI](https://recovery-decision-engine.streamlit.app) · **API:** [Hugging Face Spaces](https://huggingface.co/spaces/prashantshukla01/Recovery-Decision-Engine) · **Experiments:** [DagsHub MLflow](https://dagshub.com/prashantshukla01/Recovery-Decision-Engine.mlflow/#/experiments/0)

---

## Contents

- [The problem](#the-problem)
- [What this project adds](#what-this-project-adds)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Results](#results)
- [Live deployments and hosted MLOps](#live-deployments-and-hosted-mlops)
- [Quickstart](#quickstart)
- [Project structure](#project-structure)
- [Testing and CI](#testing-and-ci)
- [Limitations](#limitations)
- [What broke, and how we fixed it](#what-broke-and-how-we-fixed-it)
- [Documentation](#documentation)

---

## The problem

Razorpay's subscription infrastructure retries a failed payment on a fixed schedule (T+1 / T+2 / T+3), and its Subscription Recovery agent layers smarter, decline-code-aware retry timing plus escalation to a voice call on top of that. What neither publicly does is answer three questions before acting:

1. **Is attempting recovery economically worthwhile at all**, given the cost of the intervention against the probability it succeeds?
2. **Which intervention** — retry, SMS, WhatsApp, voice, human escalation — is the right one for this specific case, not just "try again"?
3. **How confident is the system in its own estimate**, and should that confidence — not just the point prediction — determine whether to act autonomously or hand off to a human?

This project treats those three questions as the actual product, not an afterthought bolted onto a retry loop.

## What this project adds

A **calibrated Bayesian hierarchical model** (PyMC) estimates `P(recovery | context, action)` with a genuine 90% credible interval, not just a point score. A separate, **pure deterministic policy engine** turns that into a decision: compute expected value per action, execute the best one if it's positive-EV and the model is confident, escalate to a human if the model is uncertain, or abstain entirely if every option is negative-EV. The two are strictly separated — see [Architecture](#architecture) — so a miscalibrated or hallucinated estimate can never directly authorize a money-adjacent action.

## Architecture

```
Payment failure event
        │
        ▼
Context builder (LLM, with deterministic rule-based fallback)
        │  produces a structured FailureContext — never a probability or a decision
        ▼
Probabilistic model (PyMC hierarchical logistic regression)
        │  produces P(success) + 90% credible interval, per candidate action
        ▼
Policy engine (pure deterministic code)
        │  EV = p · V − cost; argmax with confidence and EV thresholds
        ▼
   ┌────────────┬─────────────┬────────────┐
   ▼            ▼             ▼
Execute      Escalate      Abstain
(Razorpay    (human        (no action,
test-mode    review)       logged)
API)
   │            │             │
   └────────────┴─────────────┘
                │
                ▼
    Audit log (every path, every event)
```

**Non-negotiable boundaries:**
- The LLM never estimates a probability and never decides an action — it only structures input context and, after a decision is already made, drafts outreach text.
- The probabilistic model never decides an action — it only estimates likelihood and uncertainty.
- The policy engine is the only component authorized to trigger or block a money-adjacent action, and it contains no ML.

Full HLD/LLD: [`docs/architecture.md`](docs/architecture.md).

## Tech stack

| Layer | Choice |
|---|---|
| Probabilistic modeling | PyMC + ArviZ |
| Baseline comparison | scikit-learn (isotonic-calibrated logistic regression) |
| Orchestration | FastAPI + Pydantic |
| Audit / state storage | Postgres (hosted), SQLite fallback for offline dev |
| Context extraction / messaging | Anthropic Claude API, with rule-based fallback |
| Payments | Razorpay test-mode REST API |
| Experiment tracking | MLflow, hosted on DagsHub |
| Data/model versioning | DVC, remote storage on DagsHub |
| CI | GitHub Actions |
| Backend hosting | Hugging Face Spaces (Docker) |
| Demo UI | Streamlit Community Cloud |

## Results

Evaluated on a 500-event held-out synthetic batch (`data/eval.csv`), against a known ground-truth probability function (see [Limitations](#limitations) on why this is synthetic, not real transaction data).

**Calibration**

| Metric | Baseline (isotonic) | Bayesian hierarchical model |
|---|---|---|
| Brier score | 0.1450 | 0.1442 |
| Expected Calibration Error | 0.0354 | 0.0223 |
| 90% HDI coverage of true probability | N/A (point estimate only) | 66.4% |

**Business outcome, with vs. without uncertainty gating**

| Metric | Full engine (τ = 0.25) | Ablated: no uncertainty gating |
|---|---|---|
| Rupees recovered | ₹148,825 | ₹148,825 |
| Rupees wasted on failed interventions | ₹28.80 | ₹34.20 |
| Automation rate | 76.8% | 100.0% |
| Escalation rate | 23.2% | 0.0% |
| False intervention rate on hard declines | 0.0% | 14.2% |

Disabling uncertainty gating removes all escalation and pushes false interventions on hard declines from 0% to 14.2% — the calibrated uncertainty is doing real work, not just decorating the output.

## Live deployments and hosted MLOps

| Component | Host | Link |
|---|---|---|
| Experiment tracking | DagsHub MLflow | [Experiments](https://dagshub.com/prashantshukla01/Recovery-Decision-Engine.mlflow/#/experiments/0) |
| Model registry | DagsHub | [Registered models](https://dagshub.com/prashantshukla01/Recovery-Decision-Engine/models) |
| Versioned data/artifacts | DagsHub DVC storage | [Browse data](https://dagshub.com/prashantshukla01/Recovery-Decision-Engine/src/main/data) |
| API backend | Hugging Face Spaces | [Live API](https://huggingface.co/spaces/prashantshukla01/Recovery-Decision-Engine) |
| Interactive demo | Streamlit Community Cloud | [Launch demo](https://recovery-decision-engine.streamlit.app) |

To inspect MLflow locally instead: `./venv/bin/mlflow ui`, then open `http://127.0.0.1:5000`.

## Quickstart

```bash
git clone https://github.com/prashantshukla01/Recovery-Decision-Engine.git
cd Recovery-Decision-Engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull versioned data via DVC
dvc pull

# Run the full test suite
PYTHONPATH=. pytest tests/ -v

# Log an experiment run to MLflow
PYTHONPATH=. python -m src.mlops.track_experiment

# Start the API locally
PYTHONPATH=. uvicorn src.orchestration.app:app --reload
```

## Project structure

```
Recovery-Decision-Engine/
├── data/              synthetic train/eval sets, model artifacts (DVC-tracked)
├── docs/              architecture, pitch script, failure log
├── notebooks/         data sanity check, calibration comparison, business metrics
├── src/
│   ├── simulation/         synthetic ground-truth generator
│   ├── modeling/            PyMC hierarchical model + baseline
│   ├── policy/               EV computation, decision rule
│   ├── state_machine/        idempotent FSM, native-retry reconciliation
│   ├── razorpay_client/      test-mode REST client, retries/backoff
│   ├── llm/                  context builder + post-decision messaging
│   ├── audit/                Postgres/SQLite audit logger
│   ├── orchestration/        FastAPI app
│   ├── evaluation/           business metrics, ablation runner
│   └── mlops/                MLflow experiment tracking
├── tests/             unit, integration, and edge-case suites
└── README.md
```

## Testing and CI

30/30 tests passing (`PYTHONPATH=. pytest tests/ -v`, ~8s locally). Tests run automatically on every push via GitHub Actions; a scheduled weekly job re-evaluates calibration on a fresh synthetic batch and logs the result to MLflow, to catch drift over time.

## Limitations

Stated plainly, for anyone evaluating this beyond the demo:

- **All data is synthetic.** No real Razorpay transaction data was used or available; the model is evaluated against a ground-truth probability function we defined ourselves, grounded in publicly reported industry ranges for decline-code recovery rates. Real-world performance is unverified.
- **90% credible interval coverage is 66.4%, not ~90%.** The model's uncertainty is currently narrower than it should be — it's more confident than it's entitled to be. This doesn't invalidate the calibration story (ECE and Brier score both beat the baseline), but it's a real gap between claimed and actual coverage that would need fixing — likely via wider priors on the pooling variance or more training data per decline-code/action combination — before this could be trusted with real money decisions.
- **Intervention costs are illustrative**, not sourced from actual SMS/WhatsApp/voice API pricing at scale — they're documented assumptions, configurable per merchant.
- **Voice call and human escalation actions are logged simulations**, not live executions — Razorpay's test-mode API doesn't support triggering a real voice call or paging a human agent.
- **Free-tier hosting has real constraints**: Hugging Face Spaces and Streamlit Community Cloud may cold-start after inactivity; the live demo may take 20–40 seconds to respond after idle periods.

## What broke, and how we fixed it

Full log: [`docs/failure_log.md`](docs/failure_log.md). Highlights:

- **PyMC shape mismatch on single-event inference** — a single-context prediction produced a 1D vector where the model expected a matrix; fixed with a `transform_single()` helper that reshapes to `(1, K)` before inference.
- **Hard dependency on external API keys** — the pipeline broke in any environment without Anthropic/Razorpay credentials; added deterministic rule-based fallbacks in `context_builder.py` and `message_generator.py` so the full pipeline runs offline.
- **SQLite `:memory:` isolation across connections** — table state was disappearing between calls; fixed by holding a single persistent connection in `AuditLogger` instead of opening a new one per call.
- **CI runtime** — PyMC's default backend made GitHub Actions runs slow; setting `PYTENSOR_FLAGS=cxx=g++` and a lighter fitting configuration for CI brought test runs under 30 seconds.

## Documentation

- [System architecture (HLD/LLD)](docs/architecture.md)
- [Full failure log](docs/failure_log.md)
- [Data sanity check notebook](notebooks/01_data_sanity_check.ipynb)
- [Calibration comparison notebook](notebooks/02_calibration_comparison.ipynb)
- [Business metrics and ablation notebook](notebooks/05_business_metrics.ipynb)