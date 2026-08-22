"""pytest test suite for Phase 3 Idempotent Finite State Machine (FSM)."""

import pytest
from src.state_machine.fsm import RecoveryFSM, StateTransition


def test_fsm_valid_lifecycle():
    """Verify normal state lifecycle: pending -> deciding -> executing -> verifying -> resolved."""
    fsm = RecoveryFSM()
    evt = "evt_lifecycle_001"

    assert fsm.get_state(evt) == "pending"

    tr1 = fsm.transition_to(evt, "deciding")
    assert tr1.status == "success"
    assert fsm.get_state(evt) == "deciding"

    tr2 = fsm.transition_to(evt, "executing")
    assert tr2.status == "success"
    assert fsm.get_state(evt) == "executing"

    tr3 = fsm.transition_to(evt, "verifying")
    assert tr3.status == "success"
    assert fsm.get_state(evt) == "verifying"

    tr4 = fsm.transition_to(evt, "resolved")
    assert tr4.status == "success"
    assert fsm.get_state(evt) == "resolved"


def test_fsm_idempotency_duplicate_ignored():
    """Verify repeated transition on an already-resolved event returns duplicate_ignored no-op."""
    fsm = RecoveryFSM()
    evt = "evt_dup_001"

    fsm.transition_to(evt, "deciding")
    fsm.transition_to(evt, "aborted")
    assert fsm.get_state(evt) == "aborted"

    # Attempt repeated transition on aborted event
    dup = fsm.transition_to(evt, "executing")
    assert dup.status == "duplicate_ignored"
    assert fsm.get_state(evt) == "aborted"


def test_fsm_native_reconciliation():
    """Verify external Razorpay native retry reconciliation."""
    fsm = RecoveryFSM()
    evt = "evt_recon_001"

    fsm.transition_to(evt, "deciding")
    fsm.transition_to(evt, "executing")

    rec = fsm.reconcile_native_resolution(evt, native_outcome="success")
    assert rec.status == "reconciled"
    assert fsm.get_state(evt) == "resolved"


def test_fsm_invalid_transition():
    """Verify invalid transition attempt returns invalid_transition status."""
    fsm = RecoveryFSM()
    evt = "evt_invalid_001"

    # Pending directly to resolved is invalid
    invalid = fsm.transition_to(evt, "resolved")
    assert invalid.status == "invalid_transition"
    assert fsm.get_state(evt) == "pending"
