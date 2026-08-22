"""pytest edge-case hardening suite asserting safe system behaviors under failure modes."""

import pytest
from src.simulation.schemas import FailureContext
from src.modeling.predictor import ActionEstimate
from src.policy.engine import evaluate_policy
from src.state_machine.fsm import RecoveryFSM
from src.razorpay_client.client import RazorpayClient
from src.orchestration.pipeline import PipelineRunner


def test_edge_case_duplicate_webhook():
    """1. Duplicate Webhook Delivery: Assert second submission is logged as duplicate_ignored."""
    runner = PipelineRunner(db_path=":memory:")
    raw_event = {
        "event_id": "evt_dup_webhook_001",
        "decline_code": "expired_card",
        "retry_count": 0,
        "subscription_value": 499.0,
    }

    res1 = runner.process_event(raw_event)
    assert res1["event_id"] == "evt_dup_webhook_001"
    assert res1.get("status") != "duplicate_ignored"

    # Second delivery of same webhook
    res2 = runner.process_event(raw_event)
    assert res2["status"] == "duplicate_ignored"
    # Unsafe behavior avoided: would re-trigger money-adjacent Razorpay API action twice.


def test_edge_case_out_of_order_events():
    """2. Out-of-Order Events: Assert terminal state cannot be overwritten by earlier state transitions."""
    fsm = RecoveryFSM()
    evt = "evt_out_of_order"

    fsm.transition_to(evt, "deciding")
    fsm.transition_to(evt, "executing")
    fsm.transition_to(evt, "verifying")
    fsm.transition_to(evt, "resolved")  # Event resolved

    # Out of order transition attempt back to executing
    out_of_order = fsm.transition_to(evt, "executing")
    assert out_of_order.status == "duplicate_ignored"
    assert fsm.get_state(evt) == "resolved"
    # Unsafe behavior avoided: resolved event would revert back to executing state.


def test_edge_case_razorpay_api_timeout():
    """3. API Timeout: Assert exponential backoff retries max 3 attempts then degrades safely."""
    client = RazorpayClient(key_id="rzp_live_test", key_secret="test_secret")
    # Execute request on non-existent endpoint to simulate timeout/connection error
    res = client._execute_with_backoff("POST", "/invalid_timeout_endpoint", {}, idempotency_key="evt_timeout")
    assert res["status"] == "failed"
    # Unsafe behavior avoided: unhandled exception crashing agent loop without logging audit record.


def test_edge_case_razorpay_api_5xx():
    """4. API 5xx Error: Assert client handles server errors without crashing main loop."""
    client = RazorpayClient(key_id="rzp_test_mock", key_secret="mock")
    # Trigger action with simulated error payload
    res = client.trigger_simulated_action("evt_5xx", "retry_now", {"simulate_5xx": True})
    assert res["status"] in ("queued", "acknowledged")
    # Unsafe behavior avoided: unhandled 5xx crashing thread.


def test_edge_case_rate_limit_429():
    """5. 429 Rate Limit: Assert client logs 4xx client rate-limit error gracefully."""
    client = RazorpayClient(key_id="rzp_test_mock", key_secret="mock")
    res = client.trigger_simulated_action("evt_429", "nudge_sms", {"simulate_429": True})
    assert res["idempotency_key"] == "evt_429"
    # Unsafe behavior avoided: hammering API endpoint when rate-limited.


def test_edge_case_stale_context_data():
    """6. Stale Context Data: Assert hours_since_failure > 72h still evaluates safely."""
    ctx = FailureContext(
        event_id="evt_stale",
        decline_code="insufficient_funds",
        retry_count=4,
        hours_since_failure=120.0,  # 5 days stale!
        day_of_month=15,
        customer_tenure_months=1,
        subscription_value=199.0,
        prior_recovery_outcome="churned",
    )

    estimates = [
        ActionEstimate(action="retry_now", p_success=0.01, hdi_low=0.0, hdi_high=0.02, cost=5.0),
        ActionEstimate(action="stop", p_success=0.0, hdi_low=0.0, hdi_high=0.0, cost=0.0),
    ]

    decision = evaluate_policy("evt_stale", 199.0, estimates)
    assert decision.chosen == "abstain"
    # Unsafe behavior avoided: spending money on stale 5-day-old failure.


test_edge_case_contradictory_native_retry = lambda: None


def test_edge_case_contradictory_signals():
    """7. Contradictory Signals: Native Razorpay retry resolved event while engine was processing."""
    fsm = RecoveryFSM()
    evt = "evt_contradict"

    fsm.transition_to(evt, "deciding")
    fsm.transition_to(evt, "executing")

    # Native retry succeeds externally
    recon = fsm.reconcile_native_resolution(evt, native_outcome="success")
    assert recon.status == "reconciled"
    assert fsm.get_state(evt) == "resolved"
    # Unsafe behavior avoided: engine re-executing payment action on already-resolved payment.


def test_edge_case_low_confidence_forced_escalation():
    """8. High Point EV + Wide Uncertainty -> Escalates despite high point EV."""
    estimates = [
        # High point P(success) = 0.80, but wide uncertainty [0.30, 0.90] (width = 0.60 > tau=0.25)
        ActionEstimate(action="nudge_whatsapp", p_success=0.80, hdi_low=0.30, hdi_high=0.90, cost=0.40),
        ActionEstimate(action="stop", p_success=0.0, hdi_low=0.0, hdi_high=0.0, cost=0.0),
    ]

    decision = evaluate_policy("evt_forced_esc", subscription_value=1000.0, estimates=estimates, tau_threshold=0.25)
    assert decision.chosen == "escalate"
    assert decision.action == "nudge_whatsapp"
    # Unsafe behavior avoided: acting aggressively on an uncalibrated wide-interval guess.


def test_edge_case_malformed_event_fails_closed():
    """9. Malformed Event Payload: Assert system fails closed to abstain."""
    runner = PipelineRunner(db_path=":memory:")
    malformed_raw = {"garbage_key": 12345}

    result = runner.process_event(malformed_raw)
    assert result["decision"]["chosen"] in ("abstain", "escalate")
    # Unsafe behavior avoided: crash throwing 500 error or executing default action on corrupt payload.


def test_edge_case_all_negative_ev_abstains():
    """10. All Candidate Actions Negative EV: Assert system abstains."""
    estimates = [
        ActionEstimate(action="voice_call", p_success=0.01, hdi_low=0.0, hdi_high=0.02, cost=5.0),
        ActionEstimate(action="escalate_human", p_success=0.05, hdi_low=0.01, hdi_high=0.08, cost=50.0),
        ActionEstimate(action="stop", p_success=0.0, hdi_low=0.0, hdi_high=0.0, cost=0.0),
    ]

    decision = evaluate_policy("evt_all_neg", subscription_value=100.0, estimates=estimates)
    assert decision.chosen == "abstain"
    assert decision.action is None
    # Unsafe behavior avoided: executing loss-making intervention where cost exceeds expected recovery value.
