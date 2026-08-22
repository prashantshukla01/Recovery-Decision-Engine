"""Intervention cost configuration (in INR)."""

from typing import Dict

# Documented illustrative intervention cost assumptions
ACTION_COSTS: Dict[str, float] = {
    "retry_now": 0.0,        # Automated API call, negligible marginal cost
    "retry_later": 0.0,      # Scheduled automated retry, negligible cost
    "nudge_sms": 0.20,       # Transactional SMS gateway (~₹0.15 - ₹0.25)
    "nudge_whatsapp": 0.40,  # WhatsApp Business API session (~₹0.35 - ₹0.50)
    "voice_call": 3.50,      # Automated IVR voice call (~₹2 - ₹5)
    "escalate_human": 50.0,  # Human customer success representative time (~₹40 - ₹60)
    "stop": 0.0,             # Abstain / no action
}
