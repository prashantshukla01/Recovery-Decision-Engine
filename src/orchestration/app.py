"""FastAPI application serving Recovery Decision Engine agent pipeline endpoints."""

from typing import Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

load_dotenv()
from src.orchestration.pipeline import PipelineRunner

app = FastAPI(
    title="Recovery Decision Engine API",
    description="Agentic payment recovery decisioning engine with Bayesian uncertainty gating and deterministic policy enforcement.",
    version="1.0.0",
)

runner = PipelineRunner()


class EventRequest(BaseModel):
    """Raw failure event submission request."""

    event_id: str = Field(..., description="Idempotency key for event")
    decline_code: str = Field("insufficient_funds", description="Razorpay decline error code")
    retry_count: int = Field(0, ge=0, description="Prior retry attempts")
    hours_since_failure: float = Field(2.0, ge=0.0, description="Hours elapsed")
    day_of_month: int = Field(15, ge=1, le=31, description="Day of month")
    customer_tenure_months: int = Field(6, ge=0, description="Customer tenure in months")
    subscription_value: float = Field(499.0, gt=0.0, description="Subscription amount in INR")
    prior_recovery_outcome: str = Field("none", description="Previous failure outcome")


class BatchEventRequest(BaseModel):
    """Batch event processing request."""

    events: List[EventRequest]


@app.get("/health")
def healthcheck():
    """Healthcheck endpoint."""
    return {"status": "healthy", "service": "Recovery Decision Engine"}


@app.post("/api/v1/event")
def process_single_event(payload: EventRequest, tau: float = Query(0.25, description="HDI width tau threshold")):
    """Processes a single payment failure event through the recovery engine."""
    try:
        result = runner.process_event(payload.model_dump(), tau_threshold=tau)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/batch")
def process_batch_events(payload: BatchEventRequest, tau: float = Query(0.25, description="HDI width tau threshold")):
    """Processes a batch of payment failure events."""
    results = []
    for req in payload.events:
        res = runner.process_event(req.model_dump(), tau_threshold=tau)
        results.append(res)
    return {"total_processed": len(results), "results": results}


@app.get("/api/v1/audit/{event_id}")
def get_audit_log(event_id: str):
    """Retrieves full persistent audit record for a given event_id."""
    record = runner.audit_logger.get_record(event_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"No audit record found for event_id '{event_id}'")
    return record.model_dump()
