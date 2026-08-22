"""Razorpay Test-Mode REST API Client with idempotency and exponential backoff retries."""

import os
import time
import httpx
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("razorpay_client")


class RazorpayClient:
    """Razorpay REST API test-mode client."""

    def __init__(
        self,
        key_id: Optional[str] = None,
        key_secret: Optional[str] = None,
        base_url: str = "https://api.razorpay.com/v1",
        max_retries: int = 3,
    ):
        self.key_id = key_id or os.environ.get("RAZORPAY_KEY_ID", "rzp_test_mock_key")
        self.key_secret = key_secret or os.environ.get("RAZORPAY_KEY_SECRET", "rzp_test_mock_secret")
        self.base_url = base_url
        self.max_retries = max_retries
        self.use_mock = self.key_id.startswith("rzp_test_mock") or not self.key_secret

    def _execute_with_backoff(
        self,
        method: str,
        endpoint: str,
        payload: Dict[str, Any],
        idempotency_key: str,
    ) -> Dict[str, Any]:
        """Executes an HTTP request with exponential backoff on 5xx or connection errors."""
        if self.use_mock:
            # Test-mode simulated API response
            logger.info(f"[MOCK RAZORPAY API] {method} {endpoint} (idempotency: {idempotency_key})")
            return {
                "id": f"rzp_sub_{idempotency_key[:12]}",
                "entity": "subscription_recovery_action",
                "status": "acknowledged",
                "idempotency_key": idempotency_key,
                "payload": payload,
            }

        headers = {
            "X-Razorpay-Idempotency-Key": idempotency_key,
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        auth = (self.key_id, self.key_secret)

        attempt = 0
        backoff = 0.5  # initial backoff in seconds

        while attempt < self.max_retries:
            attempt += 1
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.request(method, url, json=payload, headers=headers, auth=auth)

                if response.status_code in [200, 201, 202]:
                    return response.json()
                elif response.status_code >= 500:
                    logger.warning(f"Razorpay API 5xx error (attempt {attempt}/{self.max_retries}): {response.status_code}")
                else:
                    # 4xx client errors should fail fast without retry
                    logger.error(f"Razorpay API 4xx error: {response.status_code} - {response.text}")
                    return {"status": "failed", "error": response.text, "code": response.status_code}

            except (httpx.RequestError, httpx.TimeoutException) as exc:
                logger.warning(f"Razorpay API connection error (attempt {attempt}/{self.max_retries}): {exc}")

            if attempt < self.max_retries:
                time.sleep(backoff)
                backoff *= 2.0  # exponential backoff

        return {"status": "failed", "error": f"Failed after {self.max_retries} attempts"}

    def trigger_retry(self, event_id: str, subscription_id: str, retry_delay_hours: float = 0.0) -> Dict[str, Any]:
        """Triggers a payment retry via Razorpay test-mode REST API."""
        payload = {
            "event_id": event_id,
            "subscription_id": subscription_id,
            "action": "retry",
            "delay_hours": retry_delay_hours,
        }
        return self._execute_with_backoff("POST", "/subscriptions/retry", payload, idempotency_key=event_id)

    def send_notification(self, event_id: str, channel: str, message_text: str, customer_phone: str = "+919999999999") -> Dict[str, Any]:
        """Sends a notification (SMS / WhatsApp) via Razorpay notification service."""
        payload = {
            "event_id": event_id,
            "channel": channel,
            "message": message_text,
            "phone": customer_phone,
        }
        return self._execute_with_backoff("POST", "/notifications/send", payload, idempotency_key=event_id)

    def trigger_simulated_action(self, event_id: str, action: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles voice call and human escalation simulated test paths."""
        logger.info(f"[SIMULATED ACTION] Action '{action}' triggered for event {event_id}: {details}")
        return {
            "id": f"sim_{action}_{event_id[:10]}",
            "action": action,
            "status": "queued",
            "simulated": True,
            "idempotency_key": event_id,
        }
