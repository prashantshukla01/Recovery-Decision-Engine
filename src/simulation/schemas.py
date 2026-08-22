"""Pydantic schemas for FailureContext and Simulation records."""

from typing import Literal
from pydantic import BaseModel, Field


class FailureContext(BaseModel):
    """Core schema representing normalized failure event context."""

    event_id: str = Field(..., description="Unique event idempotency key")
    decline_code: Literal[
        "insufficient_funds",
        "issuer_unavailable",
        "expired_card",
        "do_not_honor",
        "stolen_card",
    ]
    retry_count: int = Field(ge=0, description="Number of prior retry attempts")
    hours_since_failure: float = Field(ge=0.0, description="Hours elapsed since initial decline")
    day_of_month: int = Field(ge=1, le=31, description="Day of month (1-31)")
    customer_tenure_months: int = Field(ge=0, description="Customer subscription tenure in months")
    subscription_value: float = Field(gt=0.0, description="Subscription billing amount in INR")
    prior_recovery_outcome: Literal["none", "recovered", "churned"] = Field(
        default="none", description="Outcome of previous failure recovery attempts"
    )


class SimulationRecord(BaseModel):
    """Ground-truth simulation record including action, hidden true_p_success, and outcome."""

    context: FailureContext
    action: Literal[
        "retry_now",
        "retry_later",
        "nudge_sms",
        "nudge_whatsapp",
        "voice_call",
        "escalate_human",
        "stop",
    ]
    true_p_success: float = Field(ge=0.0, le=1.0, description="Hidden ground-truth recovery probability")
    outcome: int = Field(ge=0, le=1, description="Sampled Bernoulli outcome (1=recovered, 0=failed)")
