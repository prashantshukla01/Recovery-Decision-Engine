"""Pydantic schema for Policy Decision outputs."""

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class Decision(BaseModel):
    """Output decision object produced strictly by the policy engine."""

    event_id: str = Field(..., description="Idempotency key of the failure event")
    chosen: Literal["execute", "escalate", "abstain"] = Field(
        ..., description="Final decision choice"
    )
    action: Optional[str] = Field(
        default=None, description="Chosen intervention action name if chosen=='execute' or 'escalate'"
    )
    expected_value: float = Field(
        ..., description="Calculated expected economic value in rupees"
    )
    reasoning: Dict[str, Any] = Field(
        ..., description="Complete audit trail of all candidate EVs, HDI widths, and thresholds"
    )
