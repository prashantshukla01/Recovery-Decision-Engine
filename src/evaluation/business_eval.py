"""Business metrics evaluation and Ablation Study runner."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.simulation.schemas import FailureContext
from src.modeling.predictor import estimate_all_actions, ActionEstimate
from src.policy.engine import evaluate_policy
from src.policy.costs import ACTION_COSTS


def run_batch_evaluation(eval_df: pd.DataFrame, tau_threshold: float = 0.25, model_path: str = "data/pymc_model_idata.pkl") -> Dict[str, Any]:
    """Runs a batch evaluation of FailureContexts through policy engine and computes business metrics."""
    total_events = len(eval_df)
    hard_decline_events = 0
    hard_decline_interventions = 0

    rupees_recovered = 0.0
    rupees_wasted = 0.0
    rupees_avoided = 0.0

    count_executed = 0
    count_escalated = 0
    count_abstained = 0

    for _, row in eval_df.iterrows():
        ctx = FailureContext(
            event_id=str(row["event_id"]),
            decline_code=str(row["decline_code"]),
            retry_count=int(row["retry_count"]),
            hours_since_failure=float(row["hours_since_failure"]),
            day_of_month=int(row["day_of_month"]),
            customer_tenure_months=int(row["customer_tenure_months"]),
            subscription_value=float(row["subscription_value"]),
            prior_recovery_outcome=str(row["prior_recovery_outcome"]),
        )

        estimates = estimate_all_actions(ctx, model_path=model_path)
        decision = evaluate_policy(ctx.event_id, ctx.subscription_value, estimates, tau_threshold=tau_threshold)

        true_outcome = int(row["outcome"])
        is_hard_decline = ctx.decline_code in ("stolen_card", "do_not_honor")

        if is_hard_decline:
            hard_decline_events += 1

        if decision.chosen == "execute":
            count_executed += 1
            cost = ACTION_COSTS.get(decision.action, 0.0)

            if is_hard_decline:
                hard_decline_interventions += 1

            if true_outcome == 1:
                rupees_recovered += ctx.subscription_value
            else:
                rupees_wasted += cost

        elif decision.chosen == "escalate":
            count_escalated += 1
            if is_hard_decline:
                # Avoided wasting cost on hopeless hard decline
                rupees_avoided += ctx.subscription_value

        elif decision.chosen == "abstain":
            count_abstained += 1
            if is_hard_decline:
                rupees_avoided += ctx.subscription_value

    false_intervention_rate = hard_decline_interventions / hard_decline_events if hard_decline_events > 0 else 0.0
    automation_rate = (count_executed + count_abstained) / total_events
    escalation_rate = count_escalated / total_events

    return {
        "total_events": total_events,
        "tau_threshold": tau_threshold,
        "rupees_recovered": round(rupees_recovered, 2),
        "rupees_wasted": round(rupees_wasted, 2),
        "rupees_avoided": round(rupees_avoided, 2),
        "count_executed": count_executed,
        "count_escalated": count_escalated,
        "count_abstained": count_abstained,
        "automation_rate": round(automation_rate, 4),
        "escalation_rate": round(escalation_rate, 4),
        "false_intervention_rate": round(false_intervention_rate, 4),
        "hard_decline_events": hard_decline_events,
        "hard_decline_interventions": hard_decline_interventions,
    }


def run_ablation_study(eval_df: pd.DataFrame, model_path: str = "data/pymc_model_idata.pkl") -> Dict[str, Any]:
    """Runs ablation study comparing Uncertainty-Gated Engine (tau=0.25) vs Point-Estimate Only (tau=999.0)."""
    full_engine_metrics = run_batch_evaluation(eval_df, tau_threshold=0.25, model_path=model_path)
    ablated_engine_metrics = run_batch_evaluation(eval_df, tau_threshold=999.0, model_path=model_path)

    false_intervention_increase = ablated_engine_metrics["false_intervention_rate"] - full_engine_metrics["false_intervention_rate"]
    wasted_rupees_increase = ablated_engine_metrics["rupees_wasted"] - full_engine_metrics["rupees_wasted"]

    return {
        "full_engine": full_engine_metrics,
        "ablated_engine": ablated_engine_metrics,
        "false_intervention_rate_delta": round(false_intervention_increase, 4),
        "wasted_rupees_delta": round(wasted_rupees_increase, 2),
    }
