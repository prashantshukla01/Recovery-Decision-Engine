"""Pure, deterministic policy engine enforcing EV decision rule and uncertainty gating."""

from typing import List, Dict, Any, Optional
from src.simulation.schemas import FailureContext
from src.modeling.predictor import ActionEstimate
from src.policy.schemas import Decision
from src.policy.costs import ACTION_COSTS

# Default maximum credible interval width threshold for escalation
DEFAULT_TAU_THRESHOLD: float = 0.25


def compute_expected_value(p_success: float, subscription_value: float, cost: float) -> float:
    """Computes Expected Economic Value: EV = p_success * subscription_value - cost."""
    return round((p_success * subscription_value) - cost, 2)


def evaluate_policy(
    event_id: str,
    subscription_value: float,
    estimates: List[ActionEstimate],
    tau_threshold: float = DEFAULT_TAU_THRESHOLD,
) -> Decision:
    """Evaluates candidate action estimates deterministically and outputs a Decision object.

    Decision Rules:
        1. EV(a) = p_a * V - c_a for every candidate action.
        2. a* = argmax_a EV(a).
        3. If max EV < 0 for all candidate actions -> chosen = "abstain", action = None.
        4. Elif HDI_width(a*) > tau -> chosen = "escalate", action = a*.
        5. Else -> chosen = "execute", action = a*.
    """
    if not estimates:
        # Safety fail-closed default: if estimates list is empty, abstain immediately
        return Decision(
            event_id=event_id,
            chosen="abstain",
            action=None,
            expected_value=0.0,
            reasoning={"error": "Empty action estimates list provided", "fail_closed": True},
        )

    ev_map: Dict[str, float] = {}
    hdi_widths: Dict[str, float] = {}
    candidate_details: Dict[str, Any] = {}

    best_action: Optional[ActionEstimate] = None
    max_ev: float = float("-inf")

    for est in estimates:
        act = est.action
        cost = est.cost if est.cost is not None else ACTION_COSTS.get(act, 0.0)
        ev = compute_expected_value(est.p_success, subscription_value, cost)
        hdi_width = round(est.hdi_high - est.hdi_low, 4)

        ev_map[act] = ev
        hdi_widths[act] = hdi_width
        candidate_details[act] = {
            "p_success": est.p_success,
            "hdi_low": est.hdi_low,
            "hdi_high": est.hdi_high,
            "hdi_width": hdi_width,
            "cost": cost,
            "expected_value": ev,
        }

        if ev > max_ev:
            max_ev = ev
            best_action = est

    reasoning = {
        "subscription_value": subscription_value,
        "tau_threshold": tau_threshold,
        "max_expected_value": max_ev,
        "best_candidate_action": best_action.action if best_action else None,
        "candidate_evaluations": candidate_details,
    }

    # Policy Rule 1: All candidate EVs negative
    if max_ev < 0.0 or best_action is None or best_action.action == "stop":
        reasoning["decision_rule_triggered"] = "all_negative_ev_or_stop"
        return Decision(
            event_id=event_id,
            chosen="abstain",
            action=None,
            expected_value=max(max_ev, 0.0),
            reasoning=reasoning,
        )

    best_hdi_width = hdi_widths.get(best_action.action, 0.0)

    # Policy Rule 2: Uncertainty width exceeds tau threshold -> Escalate to human
    if best_hdi_width > tau_threshold:
        reasoning["decision_rule_triggered"] = f"hdi_width_exceeds_tau ({best_hdi_width} > {tau_threshold})"
        return Decision(
            event_id=event_id,
            chosen="escalate",
            action=best_action.action,
            expected_value=max_ev,
            reasoning=reasoning,
        )

    # Policy Rule 3: Confident positive EV -> Execute action
    reasoning["decision_rule_triggered"] = "confident_positive_ev"
    return Decision(
        event_id=event_id,
        chosen="execute",
        action=best_action.action,
        expected_value=max_ev,
        reasoning=reasoning,
    )
