"""LLM Context Builder normalizing raw failure metadata into FailureContext.

CRITICAL RULE: This component NEVER outputs a probability or an action decision.
"""

import os
import json
import logging
import httpx
from typing import Dict, Any, Optional
from src.simulation.schemas import FailureContext

logger = logging.getLogger("llm_context_builder")


def fallback_rule_based_context(raw_event: Dict[str, Any]) -> FailureContext:
    """Deterministic rule-based fallback parser when LLM is unavailable or fails."""
    event_id = str(raw_event.get("event_id", raw_event.get("id", "evt_unknown")))

    # Extract decline code from raw payload
    raw_code = str(raw_event.get("decline_code", raw_event.get("error_code", "insufficient_funds"))).lower()
    if "stolen" in raw_code or "lost" in raw_code:
        decline_code = "stolen_card"
    elif "do_not_honor" in raw_code or "reject" in raw_code:
        decline_code = "do_not_honor"
    elif "expired" in raw_code:
        decline_code = "expired_card"
    elif "issuer" in raw_code or "down" in raw_code or "unavailable" in raw_code:
        decline_code = "issuer_unavailable"
    else:
        decline_code = "insufficient_funds"

    retry_count = int(raw_event.get("retry_count", raw_event.get("attempts", 0)))
    hours_since_failure = float(raw_event.get("hours_since_failure", raw_event.get("elapsed_hours", 2.0)))
    day_of_month = int(raw_event.get("day_of_month", 15))
    day_of_month = max(1, min(31, day_of_month))

    customer_tenure = int(raw_event.get("customer_tenure_months", raw_event.get("tenure", 6)))
    sub_val = float(raw_event.get("subscription_value", raw_event.get("amount", 499.0)))
    prior_outcome = str(raw_event.get("prior_recovery_outcome", "none"))
    if prior_outcome not in ("none", "recovered", "churned"):
        prior_outcome = "none"

    return FailureContext(
        event_id=event_id,
        decline_code=decline_code,
        retry_count=retry_count,
        hours_since_failure=hours_since_failure,
        day_of_month=day_of_month,
        customer_tenure_months=customer_tenure,
        subscription_value=sub_val,
        prior_recovery_outcome=prior_outcome,
    )


def build_failure_context(raw_event: Dict[str, Any]) -> FailureContext:
    """Constructs FailureContext using Claude API with deterministic fallback."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        logger.info("[LLM Context Builder] ANTHROPIC_API_KEY absent; using deterministic rule-based fallback.")
        return fallback_rule_based_context(raw_event)

    try:
        # Call Anthropic API
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        prompt = f"""You are a payment failure metadata parser. Normalize the following raw event JSON into a clean structured JSON with fields:
- event_id (string)
- decline_code (one of: "insufficient_funds", "issuer_unavailable", "expired_card", "do_not_honor", "stolen_card")
- retry_count (integer >= 0)
- hours_since_failure (float >= 0.0)
- day_of_month (integer 1-31)
- customer_tenure_months (integer >= 0)
- subscription_value (float > 0.0)
- prior_recovery_outcome (one of: "none", "recovered", "churned")

Raw Event:
{json.dumps(raw_event)}

Output ONLY valid JSON matching this schema."""

        body = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }

        with httpx.Client(timeout=4.0) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", json=body, headers=headers)

        if resp.status_code == 200:
            content = resp.json()["content"][0]["text"]
            parsed_dict = json.loads(content)
            return FailureContext(**parsed_dict)
        else:
            logger.warning(f"[LLM Context Builder] Anthropic API returned status {resp.status_code}. Falling back.")
            return fallback_rule_based_context(raw_event)

    except Exception as exc:
        logger.warning(f"[LLM Context Builder] LLM parsing exception: {exc}. Falling back to rule-based parser.")
        return fallback_rule_based_context(raw_event)
