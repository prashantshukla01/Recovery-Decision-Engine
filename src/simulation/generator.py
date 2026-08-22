"""Synthetic Data Generator with hidden ground-truth probability mechanics.

Ground-truth probability formula:
    logit(p_true) = g0 + g_decline[code] + g_action[action]
                   + g_interaction[code, action]
                   + g_retry * retry_count
                   + g_payday * payday_proximity
                   + g_tenure * customer_tenure_months

    outcome ~ Bernoulli(sigmoid(logit(p_true)))
"""

import math
import uuid
import numpy as np
import pandas as pd
from typing import List, Tuple
from src.simulation.schemas import FailureContext, SimulationRecord

ACTIONS = [
    "retry_now",
    "retry_later",
    "nudge_sms",
    "nudge_whatsapp",
    "voice_call",
    "escalate_human",
    "stop",
]

DECLINE_CODES = [
    "insufficient_funds",
    "issuer_unavailable",
    "expired_card",
    "do_not_honor",
    "stolen_card",
]


def calculate_payday_proximity(day_of_month: int) -> float:
    """Computes payday proximity score in [0.0, 1.0].

    Paydays in India typically occur on the 1st and 30th/31st (end of month).
    Proximity decays exponentially with distance from these dates.
    """
    dist_to_1st = abs(day_of_month - 1)
    dist_to_end = min(abs(day_of_month - 30), abs(day_of_month - 31))
    min_dist = min(dist_to_1st, dist_to_end)
    # Exponential decay: score is 1.0 on payday, ~0.37 at 3 days away
    return math.exp(-min_dist / 3.0)


def compute_ground_truth_p_success(context: FailureContext, action: str) -> float:
    """Computes exact ground-truth P(success) for a given FailureContext and Action."""
    if action == "stop":
        # Action 'stop' means no intervention is attempted, zero probability of recovery
        return 0.0

    # Base intercept
    g0 = -0.2  # Baseline logit (~45% base success in neutral conditions)

    # Decline code main effects (logit adjustments)
    g_decline = {
        "insufficient_funds": -0.4,  # Moderate friction, heavily dependent on payday proximity
        "issuer_unavailable": 0.6,   # High recovery potential since outage is temporary
        "expired_card": -1.2,        # Requires customer action to update billing details
        "do_not_honor": -3.5,        # Hard decline: bank rejected transaction permanently
        "stolen_card": -4.0,         # Hard decline: card reported stolen / blocked
    }

    # Action main effects (logit adjustments)
    g_action = {
        "retry_now": -0.1,        # Quick retries work best for transient errors
        "retry_later": 0.3,       # Delayed retries allow issuer recovery or salary deposit
        "nudge_sms": 0.2,         # Low friction customer reminder
        "nudge_whatsapp": 0.4,    # Higher engagement customer notification
        "voice_call": 0.7,        # Direct personal contact, higher conversion
        "escalate_human": 1.1,    # High-touch human agent resolution
    }

    # Context feature coefficients
    payday_prox = calculate_payday_proximity(context.day_of_month)

    # Feature weights
    g_payday = 1.4      # Strong positive impact near payday for insufficient funds
    g_retry = -0.35     # Diminishing returns with each consecutive failure
    g_tenure = 0.02     # Slight loyalty boost for long-tenured customers (max +0.5 over 24m)

    # Specific decline-action interactions
    g_interaction = 0.0
    code = context.decline_code

    if code in ("stolen_card", "do_not_honor"):
        # For hard declines, override/attenuate positive action effects — card is blocked/invalid
        # Keeps true P(success) strictly in the ~0.01 - 0.05 range
        logit_raw = g0 + g_decline[code] + 0.1 * g_action.get(action, 0.0) + 0.05 * g_retry * context.retry_count
        p_true = 1.0 / (1.0 + math.exp(-logit_raw))
        return float(np.clip(p_true, 0.01, 0.05))

    if code == "insufficient_funds":
        # Retries near payday are very effective; nudges prompt manual top-up
        if action in ("retry_later", "nudge_whatsapp", "nudge_sms"):
            g_interaction += 0.5 * payday_prox
        elif action == "retry_now":
            g_interaction -= 0.3  # Immediate retry before payday rarely works for insufficient funds

    elif code == "issuer_unavailable":
        # Technical outages resolve over time; retries work well, customer nudges do not add value
        if action in ("retry_now", "retry_later"):
            g_interaction += 0.8
        elif action in ("nudge_sms", "nudge_whatsapp"):
            g_interaction -= 0.4

    elif code == "expired_card":
        # Technical retries fail on expired cards; customer notifications prompt update
        if action in ("retry_now", "retry_later"):
            g_interaction -= 1.0
        elif action in ("nudge_sms", "nudge_whatsapp", "voice_call", "escalate_human"):
            g_interaction += 1.2

    # Compute total logit
    logit = (
        g0
        + g_decline[code]
        + g_action.get(action, 0.0)
        + g_interaction
        + (g_payday * payday_prox if code == "insufficient_funds" else 0.2 * payday_prox)
        + (g_retry * context.retry_count)
        + (g_tenure * min(context.customer_tenure_months, 36))
    )

    p_true = 1.0 / (1.0 + math.exp(-logit))
    return float(np.clip(p_true, 0.0, 1.0))


def generate_synthetic_context(rng: np.random.Generator, event_idx: int) -> FailureContext:
    """Generates a single synthetic FailureContext object."""
    hex_suffix = f"{rng.integers(0, 0xFFFFFFFF):08x}"
    event_id = f"evt_{event_idx:06d}_{hex_suffix}"

    # Distribution over decline codes: 45% insufficient_funds, 20% issuer_unavailable, 15% expired_card, 12% do_not_honor, 8% stolen_card
    decline_code = rng.choice(
        DECLINE_CODES,
        p=[0.45, 0.20, 0.15, 0.12, 0.08],
    )

    retry_count = int(rng.choice([0, 1, 2, 3, 4], p=[0.50, 0.25, 0.15, 0.07, 0.03]))
    hours_since_failure = round(float(rng.exponential(scale=12.0)), 1)
    day_of_month = int(rng.integers(1, 32))
    customer_tenure_months = int(rng.geometric(p=0.08))  # Mean ~12.5 months
    subscription_value = float(rng.choice([199.0, 499.0, 999.0, 1499.0, 2499.0], p=[0.25, 0.40, 0.20, 0.10, 0.05]))
    prior_recovery_outcome = rng.choice(["none", "recovered", "churned"], p=[0.70, 0.20, 0.10])

    return FailureContext(
        event_id=event_id,
        decline_code=decline_code,
        retry_count=retry_count,
        hours_since_failure=hours_since_failure,
        day_of_month=day_of_month,
        customer_tenure_months=customer_tenure_months,
        subscription_value=subscription_value,
        prior_recovery_outcome=prior_recovery_outcome,
    )


def generate_dataset(
    num_events: int,
    seed: int = 42,
    sample_actions_per_event: bool = False,
) -> Tuple[List[SimulationRecord], pd.DataFrame]:
    """Generates a synthetic dataset of FailureContexts and Action outcomes.

    If sample_actions_per_event is True, samples 1 random action per event (typical for train/eval CSVs).
    If False, generates records for all candidate actions per event.
    """
    rng = np.random.default_rng(seed)
    records: List[SimulationRecord] = []
    rows = []

    for idx in range(num_events):
        ctx = generate_synthetic_context(rng, idx)
        actions_to_eval = [rng.choice(ACTIONS)] if sample_actions_per_event else ACTIONS

        for act in actions_to_eval:
            p_true = compute_ground_truth_p_success(ctx, act)
            outcome = int(rng.binomial(1, p_true))

            rec = SimulationRecord(
                context=ctx,
                action=act,
                true_p_success=p_true,
                outcome=outcome,
            )
            records.append(rec)

            row = {
                "event_id": ctx.event_id,
                "decline_code": ctx.decline_code,
                "retry_count": ctx.retry_count,
                "hours_since_failure": ctx.hours_since_failure,
                "day_of_month": ctx.day_of_month,
                "customer_tenure_months": ctx.customer_tenure_months,
                "subscription_value": ctx.subscription_value,
                "prior_recovery_outcome": ctx.prior_recovery_outcome,
                "action": act,
                "true_p_success": p_true,
                "outcome": outcome,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return records, df
