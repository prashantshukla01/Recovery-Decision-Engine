"""pytest test suite for Phase 1 synthetic data simulator."""

import pytest
import numpy as np
import pandas as pd
from src.simulation.generator import (
    generate_dataset,
    generate_synthetic_context,
    compute_ground_truth_p_success,
    ACTIONS,
    DECLINE_CODES,
)
from src.simulation.schemas import FailureContext, SimulationRecord


def test_generator_determinism():
    """Verify that generation is strictly deterministic given a fixed random seed."""
    _, df1 = generate_dataset(num_events=100, seed=42, sample_actions_per_event=True)
    _, df2 = generate_dataset(num_events=100, seed=42, sample_actions_per_event=True)

    pd.testing.assert_frame_equal(df1, df2)


def test_schema_and_column_bounds():
    """Verify that all columns exist, are non-null, and conform to valid ranges."""
    _, df = generate_dataset(num_events=200, seed=99, sample_actions_per_event=True)

    expected_columns = [
        "event_id",
        "decline_code",
        "retry_count",
        "hours_since_failure",
        "day_of_month",
        "customer_tenure_months",
        "subscription_value",
        "prior_recovery_outcome",
        "action",
        "true_p_success",
        "outcome",
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"
        assert df[col].isnull().sum() == 0, f"Null values found in column: {col}"

    assert (df["retry_count"] >= 0).all()
    assert (df["hours_since_failure"] >= 0.0).all()
    assert df["day_of_month"].between(1, 31).all()
    assert (df["customer_tenure_months"] >= 0).all()
    assert (df["subscription_value"] > 0).all()
    assert df["decline_code"].isin(DECLINE_CODES).all()
    assert df["action"].isin(ACTIONS).all()
    assert df["true_p_success"].between(0.0, 1.0).all()
    assert df["outcome"].isin([0, 1]).all()


def test_hard_decline_probabilities_low_and_bounded():
    """Verify hard declines (stolen_card, do_not_honor) have true P(success) ~ 0.01 to 0.07 across 1,000 samples."""
    rng = np.random.default_rng(1234)

    for hard_code in ["stolen_card", "do_not_honor"]:
        for _ in range(500):
            ctx = FailureContext(
                event_id=f"test_{rng.integers(100000)}",
                decline_code=hard_code,
                retry_count=int(rng.integers(0, 5)),
                hours_since_failure=float(rng.uniform(0, 48)),
                day_of_month=int(rng.integers(1, 32)),
                customer_tenure_months=int(rng.integers(0, 36)),
                subscription_value=499.0,
                prior_recovery_outcome="none",
            )

            for act in ACTIONS:
                p_true = compute_ground_truth_p_success(ctx, act)
                if act == "stop":
                    assert p_true == 0.0
                else:
                    assert 0.01 <= p_true <= 0.07, (
                        f"Hard decline {hard_code} with action {act} produced p_true={p_true}, "
                        f"expected between 0.01 and 0.07"
                    )


def test_soft_decline_action_variation():
    """Verify soft declines exhibit meaningful P(success) variation across actions and payday proximity."""
    ctx_payday = FailureContext(
        event_id="test_payday",
        decline_code="insufficient_funds",
        retry_count=0,
        hours_since_failure=2.0,
        day_of_month=1,  # Payday!
        customer_tenure_months=12,
        subscription_value=999.0,
        prior_recovery_outcome="none",
    )

    ctx_midmonth = FailureContext(
        event_id="test_midmonth",
        decline_code="insufficient_funds",
        retry_count=0,
        hours_since_failure=2.0,
        day_of_month=15,  # Mid-month (far from payday)
        customer_tenure_months=12,
        subscription_value=999.0,
        prior_recovery_outcome="none",
    )

    p_payday_whatsapp = compute_ground_truth_p_success(ctx_payday, "nudge_whatsapp")
    p_midmonth_whatsapp = compute_ground_truth_p_success(ctx_midmonth, "nudge_whatsapp")

    # WhatsApp nudge on payday should have higher success than mid-month for insufficient funds
    assert p_payday_whatsapp > p_midmonth_whatsapp + 0.15

    # Expired card: nudges work better than retries
    ctx_expired = FailureContext(
        event_id="test_expired",
        decline_code="expired_card",
        retry_count=1,
        hours_since_failure=5.0,
        day_of_month=10,
        customer_tenure_months=6,
        subscription_value=499.0,
        prior_recovery_outcome="none",
    )

    p_expired_retry = compute_ground_truth_p_success(ctx_expired, "retry_now")
    p_expired_nudge = compute_ground_truth_p_success(ctx_expired, "nudge_whatsapp")
    assert p_expired_nudge > p_expired_retry + 0.20
