# Demo Pitch Script — 5-Minute Technical Interview Walkthrough

This pitch outline is structured for presenting the **Recovery Decision Engine** to an interviewer or hackathon judging panel.

---

## 1. Problem & Gap (0:00 – 1:00)

- **Problem**: 15–20% of recurring subscription payments fail in India due to issuer outages, expired cards, or temporary insufficient funds.
- **Industry Gap**: Standard tools (like Razorpay's native retry schedule) optimize retry *timing* (T+1/T+2/T+3). They do not evaluate whether an intervention is **economically worthwhile**, which channel to pick (SMS, WhatsApp, Voice, Human), or when to **abstain**.
- **Engine Novelty**: Combines PyMC Bayesian hierarchical modeling (honest uncertainty) with a 100% deterministic, fail-closed policy engine ($EV = p \cdot V - c$).

---

## 2. System Architecture & Boundaries (1:00 – 2:00)

- Highlight the **Non-Negotiable LLM Boundary**:
  - LLMs normalize metadata and generate customer text *post-decision*.
  - LLMs **NEVER** estimate probabilities or make execution decisions.
- Point to the **Policy Engine**:
  - Pure, deterministic, fail-closed code.
  - Uncertainty gating threshold $\tau = 0.25$: if $HDI_{\text{high}} - HDI_{\text{low}} > \tau$, escalate to human.

---

## 3. Live Execution Demo (2:00 – 3:00)

Run the live agent pipeline or FastAPI endpoints:

1. **Executed Action (Soft Decline + Payday)**:
   - Event: `insufficient_funds` on 1st of month.
   - Result: `execute` $\rightarrow$ `nudge_whatsapp` (Positive EV = ₹699.60, narrow HDI width).
2. **Escalated Action (High Uncertainty)**:
   - Event: High point estimate EV, but wide posterior credible interval ($HDI_{\text{width}} = 0.40 > 0.25$).
   - Result: `escalate` $\rightarrow$ Human Representative.
3. **Abstained Action (Hard Decline / Negative EV)**:
   - Event: `stolen_card` or `do_not_honor`.
   - Result: `abstain` (Prevents false interventions and saves waste).

Show persistent SQLite Audit Record in [`data/audit.db`](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/data/audit.db).

---

## 4. Calibration & MLOps Tracking (3:00 – 4:00)

Demonstrate MLOps infrastructure:
- Launch `mlflow ui` to showcase logged parameters, Brier scores, ECE, business metrics, and reliability diagrams.
- **Calibration Metrics**:
  - PyMC Expected Calibration Error (ECE): **0.0243** vs Isotonic Baseline **0.0354** (~31% improvement).
  - Brier Score: **0.1442**.
- Demonstrate DVC data version control (`dvc status`, `data/train.csv.dvc`).

---

## 5. Business Impact & Ablation Study (4:00 – 5:00)

- Show **Ablation Study** results:
  - Disabling uncertainty gating increases false interventions on hard declines significantly.
  - Proves why honest uncertainty estimation prevents financial waste.
- **Edge-Case Hardening**: Highlight the 30-test suite covering duplicate webhooks, API timeouts, rate limits, out-of-order events, and contradictory signals.
