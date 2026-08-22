"""Integration pytest suite for Phase 4 agent orchestration pipeline."""

import os
import pytest
from fastapi.testclient import TestClient
from src.orchestration.app import app
from src.orchestration.pipeline import PipelineRunner
from src.audit.logger import AuditLogger


@pytest.fixture
def test_pipeline_runner(tmp_path):
    """Fixture initializing a PipelineRunner with a temporary SQLite database."""
    db_file = str(tmp_path / "test_audit.db")
    runner = PipelineRunner(db_path=db_file)
    return runner


def test_end_to_end_single_event_pipeline(test_pipeline_runner):
    """Integration test running a single event end-to-end through context -> model -> policy -> razorpay -> audit."""
    raw_event = {
        "event_id": "evt_integ_001",
        "decline_code": "expired_card",
        "retry_count": 0,
        "hours_since_failure": 2.0,
        "day_of_month": 10,
        "customer_tenure_months": 8,
        "subscription_value": 999.0,
        "prior_recovery_outcome": "none",
    }

    result = test_pipeline_runner.process_event(raw_event, tau_threshold=0.30)

    assert result["event_id"] == "evt_integ_001"
    assert "decision" in result
    assert result["decision"]["chosen"] in ("execute", "escalate", "abstain")

    # Verify persistent SQLite audit log record
    audit_rec = test_pipeline_runner.audit_logger.get_record("evt_integ_001")
    assert audit_rec is not None
    assert audit_rec.event_id == "evt_integ_001"
    assert audit_rec.context.decline_code == "expired_card"
    assert len(audit_rec.estimates) == 7
    assert len(audit_rec.state_transitions) >= 2


def test_fastapi_endpoints():
    """Test API endpoints /health, /api/v1/event, and /api/v1/audit/{event_id} via TestClient."""
    client = TestClient(app)

    # Health check
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    # Single event endpoint
    payload = {
        "event_id": "evt_fastapi_001",
        "decline_code": "insufficient_funds",
        "retry_count": 0,
        "hours_since_failure": 1.0,
        "day_of_month": 1,  # Payday! High P(success) expected
        "customer_tenure_months": 12,
        "subscription_value": 1499.0,
        "prior_recovery_outcome": "none",
    }
    resp_event = client.post("/api/v1/event", json=payload)
    assert resp_event.status_code == 200
    data = resp_event.json()
    assert data["event_id"] == "evt_fastapi_001"
    assert "decision" in data

    # Audit endpoint
    resp_audit = client.get("/api/v1/audit/evt_fastapi_001")
    assert resp_audit.status_code == 200
    audit_data = resp_audit.json()
    assert audit_data["event_id"] == "evt_fastapi_001"
