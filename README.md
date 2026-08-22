# Recovery Decision Engine

An economics-aware, calibrated probabilistic agent designed for failed subscription payments. It evaluates whether recovery intervention is economically worthwhile, selects the optimal intervention (or abstains), and executes it with complete auditability.

## Architecture

- **Event Source**: Simulated `payment.failed` webhooks (synthetic ground truth).
- **Context Builder**: Normalizes failure metadata into structured `FailureContext`.
- **Probabilistic Model**: Bayesian hierarchical logistic regression (PyMC) estimating $P(\text{success})$ and 90% HDI credible intervals.
- **Policy Engine**: Pure deterministic code computing Expected Value ($EV = p \cdot V - c$) and gating on uncertainty width.
- **Action Executor & Verifier**: Triggers test-mode actions and reconciles outcomes.
- **Audit Log**: Append-only log of all contexts, estimates, decisions, and outcomes.

## Getting Started

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Tests
```bash
PYTHONPATH=. pytest tests/ -v
```
