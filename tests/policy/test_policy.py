"""pytest test suite for Phase 3 Policy Engine."""

import pytest
from src.modeling.predictor import ActionEstimate
from src.policy.engine import evaluate_policy, compute_expected_value
from src.policy.schemas import Decision


def test_expected_value_calculation():
    """Verify EV formula: p_success * sub_val - cost."""
    # 50% chance on ₹1000 subscription with ₹0 cost -> EV = 500
    assert compute_expected_value(0.50, 1000.0, 0.0) == 500.0
    # 10% chance on ₹500 subscription with ₹50 human cost -> EV = 50 - 50 = 0
    assert compute_expected_value(0.10, 500.0, 50.0) == 0.0
    # 2% chance on ₹1000 subscription with ₹50 human cost -> EV = 20 - 50 = -30
    assert compute_expected_value(0.02, 1000.0, 50.0) == -30.0


def test_all_negative_ev_abstains():
    """Verify policy abstains when all candidate actions yield negative expected value."""
    estimates = [
        ActionEstimate(action="retry_now", p_success=0.01, hdi_low=0.0, hdi_high=0.02, cost=10.0),
        ActionEstimate(action="nudge_whatsapp", p_success=0.02, hdi_low=0.01, hdi_high=0.03, cost=20.0),
        ActionEstimate(action="escalate_human", p_success=0.05, hdi_low=0.02, hdi_high=0.08, cost=50.0),
        ActionEstimate(action="stop", p_success=0.0, hdi_low=0.0, hdi_high=0.0, cost=0.0),
    ]

    decision = evaluate_policy("evt_neg_001", subscription_value=499.0, estimates=estimates)

    assert isinstance(decision, Decision)
    assert decision.chosen == "abstain"
    assert decision.action is None
    assert decision.expected_value == 0.0
    assert decision.reasoning["decision_rule_triggered"] == "all_negative_ev_or_stop"


def test_confident_positive_ev_executes():
    """Verify policy chooses 'execute' when best action has positive EV and narrow HDI width."""
    estimates = [
        ActionEstimate(action="retry_now", p_success=0.60, hdi_low=0.55, hdi_high=0.65, cost=0.0),
        ActionEstimate(action="nudge_whatsapp", p_success=0.70, hdi_low=0.65, hdi_high=0.75, cost=0.40),
        ActionEstimate(action="stop", p_success=0.0, hdi_low=0.0, hdi_high=0.0, cost=0.0),
    ]

    # WhatsApp EV = 0.70 * 1000 - 0.40 = 699.60, HDI width = 0.10 <= 0.25
    decision = evaluate_policy("evt_exec_001", subscription_value=1000.0, estimates=estimates, tau_threshold=0.25)

    assert decision.chosen == "execute"
    assert decision.action == "nudge_whatsapp"
    assert decision.expected_value == 699.60
    assert decision.reasoning["decision_rule_triggered"] == "confident_positive_ev"


def test_uncertain_positive_ev_escalates():
    """Verify policy escalates to human when best action has positive EV but HDI width exceeds tau."""
    estimates = [
        ActionEstimate(action="retry_now", p_success=0.20, hdi_low=0.15, hdi_high=0.25, cost=0.0),
        # WhatsApp has high point estimate (0.75) but wide uncertainty [0.45, 0.85] (width=0.40 > tau=0.25)
        ActionEstimate(action="nudge_whatsapp", p_success=0.75, hdi_low=0.45, hdi_high=0.85, cost=0.40),
        ActionEstimate(action="stop", p_success=0.0, hdi_low=0.0, hdi_high=0.0, cost=0.0),
    ]

    decision = evaluate_policy("evt_esc_001", subscription_value=1000.0, estimates=estimates, tau_threshold=0.25)

    assert decision.chosen == "escalate"
    assert decision.action == "nudge_whatsapp"
    assert "hdi_width_exceeds_tau" in decision.reasoning["decision_rule_triggered"]


def test_empty_estimates_fails_closed():
    """Verify empty action estimates fails closed into an abstain decision."""
    decision = evaluate_policy("evt_empty", subscription_value=499.0, estimates=[])
    assert decision.chosen == "abstain"
    assert decision.action is None
    assert decision.reasoning["fail_closed"] is True
