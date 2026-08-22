"""LLM Message Generator producing customer communication text AFTER decisioning.

CRITICAL RULE: This component is ONLY called post-decision and NEVER influences the EV or intervention choice.
"""

import os
import logging
import httpx
from src.simulation.schemas import FailureContext
from src.policy.schemas import Decision

logger = logging.getLogger("llm_message_generator")


def fallback_message_text(context: FailureContext, action: str) -> str:
    """Fallback template message generator when LLM is unavailable."""
    amount_str = f"₹{context.subscription_value:,.2f}"

    if action == "nudge_whatsapp":
        if context.decline_code == "expired_card":
            return f"Hi! Your subscription payment of {amount_str} could not be processed as your card has expired. Please update your payment details here to avoid service interruption."
        elif context.decline_code == "insufficient_funds":
            return f"Hi! Your subscription payment of {amount_str} was unsuccessful. Please check your bank balance or update your payment method to keep your subscription active."
        else:
            return f"Hi! Your payment of {amount_str} for your subscription failed. Please tap here to complete payment."

    elif action == "nudge_sms":
        return f"Payment of {amount_str} failed. Update your card/UPI details to keep your subscription active."

    return f"Subscription payment of {amount_str} update required."


def generate_customer_message(context: FailureContext, decision: Decision) -> str:
    """Generates customer notification text for chosen nudge actions post-decision."""
    if decision.chosen != "execute" or decision.action not in ("nudge_sms", "nudge_whatsapp"):
        raise ValueError(f"Message generator invoked invalidly for decision choice '{decision.chosen}' and action '{decision.action}'")

    action = decision.action
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return fallback_message_text(context, action)

    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        channel = "WhatsApp" if action == "nudge_whatsapp" else "SMS"
        prompt = f"""Generate a short, polite, high-converting payment reminder message for a customer whose subscription billing failed.
Context:
- Channel: {channel}
- Subscription Value: ₹{context.subscription_value}
- Reason: {context.decline_code}

Keep it under 30 words for SMS or under 50 words for WhatsApp. Return ONLY the message body."""

        body = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 150,
            "messages": [{"role": "user", "content": prompt}],
        }

        with httpx.Client(timeout=3.0) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", json=body, headers=headers)

        if resp.status_code == 200:
            return resp.json()["content"][0]["text"].strip()
        else:
            return fallback_message_text(context, action)

    except Exception as exc:
        logger.warning(f"[LLM Message Generator] Exception: {exc}. Using fallback message.")
        return fallback_message_text(context, action)
