# System Architecture — Recovery Decision Engine

The **Recovery Decision Engine** is a production-minded fintech agent that decides whether a failed subscription payment is economically recoverable, selects the best intervention among several bounded options (or abstains), executes it against Razorpay's test-mode REST API, verifies outcome, and maintains an append-only audit trail.

---

## 1. High-Level Architecture (HLD)

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

---

## 2. Component Design & Non-Negotiable Boundaries

### A. Context Builder (`src/llm/context_builder.py`)
- **Responsibility**: Normalizes raw webhook payloads or log metadata into a structured `FailureContext` object.
- **LLM Boundary Rule**: Uses Anthropic Claude API with a deterministic rule-based fallback parser. **NEVER outputs probabilities or intervention decisions.**

### B. Probabilistic Model (`src/modeling/bayesian_model.py`)
- **Responsibility**: Fits a Bayesian hierarchical logistic regression model in PyMC:
  $$\text{logit}(p_i) = \alpha + u_{\text{decline}[i]} + v_{\text{action}[i]} + \beta^T x_i$$
  with partial pooling hyperpriors $\sigma_g, \sigma_a \sim \text{HalfNormal}(1)$.
- **Output**: Returns `ActionEstimate` containing mean posterior $P(\text{success})$ and 90% HDI credible interval $[hdi_{\text{low}}, hdi_{\text{high}}]$.

### C. Policy Engine (`src/policy/engine.py`)
- **Responsibility**: Pure deterministic code evaluating Expected Economic Value ($EV = p \cdot V - c$) and HDI width uncertainty gating ($\tau = 0.25$).
- **Output**: Produces an auditable `Decision` object (`execute`, `escalate`, `abstain`).

### D. Idempotent State Machine (`src/state_machine/fsm.py`)
- **Responsibility**: Enforces lifecycle transitions (`pending -> deciding -> executing -> verifying -> resolved | escalated | failed | aborted`).
- **Idempotency**: Repeated transitions on an event return `duplicate_ignored` without modifying state.

### E. Razorpay REST Client (`src/razorpay_client/client.py`)
- **Responsibility**: Thin REST API test-mode wrapper handling retries with exponential backoff on 5xx/network errors (max 3 retries) and idempotency header support (`X-Razorpay-Idempotency-Key`).

### F. Persistent Audit Logger (`src/audit/logger.py`)
- **Responsibility**: SQLite append-only logger (`data/audit.db`) storing `AuditRecord` entries for every single event flow.

---

## 3. MLOps Integration

- **MLflow**: Tracks model hyperparameters, calibration metrics (Brier score, ECE, 90% HDI coverage), financial evaluation metrics, and reliability diagram artifacts.
- **DVC**: Manages data version control for `data/train.csv` and `data/eval.csv`.
