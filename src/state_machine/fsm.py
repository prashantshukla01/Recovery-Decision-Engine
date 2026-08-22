"""Idempotent Finite State Machine (FSM) for payment recovery event lifecycle."""

from typing import Dict, List, Optional, Literal, Tuple
from pydantic import BaseModel, Field

StateName = Literal[
    "pending",
    "deciding",
    "executing",
    "verifying",
    "resolved",
    "escalated",
    "failed",
    "aborted",
]

TERMINAL_STATES = {"resolved", "escalated", "failed", "aborted"}

VALID_TRANSITIONS: Dict[StateName, List[StateName]] = {
    "pending": ["deciding", "aborted"],
    "deciding": ["executing", "escalated", "aborted"],
    "executing": ["verifying", "failed", "escalated"],
    "verifying": ["resolved", "failed", "escalated"],
    "resolved": [],
    "escalated": [],
    "failed": [],
    "aborted": [],
}


class StateTransition(BaseModel):
    """Log record of a state transition."""

    event_id: str
    from_state: StateName
    to_state: StateName
    status: Literal["success", "duplicate_ignored", "reconciled", "invalid_transition"]
    message: str


class RecoveryFSM:
    """In-memory or persistent Finite State Machine with strict idempotency & reconciliation."""

    def __init__(self):
        # Maps event_id -> current_state
        self._states: Dict[str, StateName] = {}
        # Maps event_id -> list of StateTransition history
        self._history: Dict[str, List[StateTransition]] = {}

    def get_state(self, event_id: str) -> StateName:
        """Returns the current state of an event (defaults to 'pending' if uninitialized)."""
        return self._states.get(event_id, "pending")

    def get_history(self, event_id: str) -> List[StateTransition]:
        """Returns complete transition history for an event."""
        return list(self._history.get(event_id, []))

    def transition_to(self, event_id: str, target_state: StateName, reason: str = "") -> StateTransition:
        """Attempts a state transition with idempotency check.

        Rules:
        - If event is in a terminal state, repeated transition returns duplicate_ignored no-op.
        - If transition is invalid, returns invalid_transition.
        - Otherwise updates state and appends to history log.
        """
        current_state = self.get_state(event_id)

        # Idempotency check: event already in terminal state or target state
        if current_state in TERMINAL_STATES:
            record = StateTransition(
                event_id=event_id,
                from_state=current_state,
                to_state=target_state,
                status="duplicate_ignored",
                message=f"Event {event_id} is already in terminal state '{current_state}'. Transition ignored.",
            )
            self._history.setdefault(event_id, []).append(record)
            return record

        if current_state == target_state:
            record = StateTransition(
                event_id=event_id,
                from_state=current_state,
                to_state=target_state,
                status="duplicate_ignored",
                message=f"Event {event_id} is already in state '{target_state}'. Transition ignored.",
            )
            self._history.setdefault(event_id, []).append(record)
            return record

        # Valid transition check
        allowed_next = VALID_TRANSITIONS.get(current_state, [])
        if target_state not in allowed_next:
            record = StateTransition(
                event_id=event_id,
                from_state=current_state,
                to_state=target_state,
                status="invalid_transition",
                message=f"Cannot transition from '{current_state}' to '{target_state}'. Allowed: {allowed_next}",
            )
            self._history.setdefault(event_id, []).append(record)
            return record

        # Execute transition
        self._states[event_id] = target_state
        record = StateTransition(
            event_id=event_id,
            from_state=current_state,
            to_state=target_state,
            status="success",
            message=f"Successfully transitioned from '{current_state}' to '{target_state}'. Reason: {reason}",
        )
        self._history.setdefault(event_id, []).append(record)
        return record

    def reconcile_native_resolution(self, event_id: str, native_outcome: str) -> StateTransition:
        """Reconciles an external Razorpay native retry outcome (e.g. T+1 retry succeeded independently)."""
        current_state = self.get_state(event_id)

        target_state: StateName = "resolved" if native_outcome == "success" else "failed"

        if current_state == "resolved" and target_state == "resolved":
            record = StateTransition(
                event_id=event_id,
                from_state=current_state,
                to_state=target_state,
                status="duplicate_ignored",
                message="Native retry resolution matches engine state (both resolved). No-op.",
            )
        else:
            self._states[event_id] = target_state
            record = StateTransition(
                event_id=event_id,
                from_state=current_state,
                to_state=target_state,
                status="reconciled",
                message=f"Native Razorpay retry reconciled event to '{target_state}' from '{current_state}'.",
            )

        self._history.setdefault(event_id, []).append(record)
        return record
