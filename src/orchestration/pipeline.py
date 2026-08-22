"""End-to-end agent orchestration pipeline wiring context building, modeling, policy decisioning, FSM, execution, and audit logging."""

import logging
from typing import Dict, Any, List, Optional
from src.llm.context_builder import build_failure_context
from src.modeling.predictor import estimate_all_actions, ActionEstimate
from src.policy.engine import evaluate_policy
from src.state_machine.fsm import RecoveryFSM
from src.razorpay_client.client import RazorpayClient
from src.llm.message_generator import generate_customer_message
from src.audit.logger import AuditLogger, AuditRecord

logger = logging.getLogger("agent_pipeline")


class PipelineRunner:
    """Orchestrates end-to-end payment failure recovery workflow."""

    def __init__(self, db_path: str = "data/audit.db", model_path: str = "data/pymc_model_idata.pkl"):
        self.fsm = RecoveryFSM()
        self.audit_logger = AuditLogger(db_path=db_path)
        self.razorpay_client = RazorpayClient()
        self.model_path = model_path

    def process_event(self, raw_event: Dict[str, Any], tau_threshold: float = 0.25) -> Dict[str, Any]:
        """Executes full agent loop for a single raw event."""
        # 1. Normalize raw event -> FailureContext
        context = build_failure_context(raw_event)
        event_id = context.event_id

        # 2. Check FSM idempotency & transition to 'deciding'
        t1 = self.fsm.transition_to(event_id, "deciding", reason="Event received by agent pipeline")
        if t1.status == "duplicate_ignored":
            logger.info(f"Duplicate event {event_id} ignored by FSM.")
            audit = self.audit_logger.get_record(event_id)
            return {
                "event_id": event_id,
                "status": "duplicate_ignored",
                "audit_record": audit.model_dump() if audit else None,
            }

        # 3. Model Inference: Estimate P(success) & credible intervals per candidate action
        estimates: List[ActionEstimate] = estimate_all_actions(context, model_path=self.model_path)

        # 4. Policy Engine: Compute EVs & Uncertainty Gating
        decision = evaluate_policy(
            event_id=event_id,
            subscription_value=context.subscription_value,
            estimates=estimates,
            tau_threshold=tau_threshold,
        )

        action_result: Optional[Dict[str, Any]] = None
        outcome: Optional[str] = None

        # 5. Handle Policy Choices & FSM Transitions
        if decision.chosen == "abstain":
            self.fsm.transition_to(event_id, "aborted", reason="Policy engine selected abstain (EV < 0)")
            outcome = "aborted"

        elif decision.chosen == "escalate":
            self.fsm.transition_to(event_id, "escalated", reason="Policy engine escalated due to high uncertainty")
            # Trigger simulated human escalation ticket
            action_result = self.razorpay_client.trigger_simulated_action(
                event_id, "escalate_human", {"reason": decision.reasoning.get("decision_rule_triggered")}
            )
            outcome = "escalated"

        elif decision.chosen == "execute":
            self.fsm.transition_to(event_id, "executing", reason=f"Executing action '{decision.action}'")
            act = decision.action

            # Trigger action via Razorpay client
            if act in ("retry_now", "retry_later"):
                delay = 24.0 if act == "retry_later" else 0.0
                action_result = self.razorpay_client.trigger_retry(
                    event_id=event_id,
                    subscription_id=f"sub_{event_id[:8]}",
                    retry_delay_hours=delay,
                )

            elif act in ("nudge_sms", "nudge_whatsapp"):
                # Generate localized customer message post-decision
                msg_text = generate_customer_message(context, decision)
                channel = "whatsapp" if act == "nudge_whatsapp" else "sms"
                action_result = self.razorpay_client.send_notification(
                    event_id=event_id,
                    channel=channel,
                    message_text=msg_text,
                )
                action_result["generated_message"] = msg_text

            elif act in ("voice_call", "escalate_human"):
                action_result = self.razorpay_client.trigger_simulated_action(
                    event_id, act, {"subscription_value": context.subscription_value}
                )

            # 6. Verification Step
            self.fsm.transition_to(event_id, "verifying", reason="Verifying action execution status")
            if action_result and action_result.get("status") in ("acknowledged", "queued", "success"):
                self.fsm.transition_to(event_id, "resolved", reason="Action successfully executed and verified")
                outcome = "success"
            else:
                self.fsm.transition_to(event_id, "failed", reason="Action execution failed or API error")
                outcome = "failure"

        # 7. Write complete AuditRecord to SQLite
        history_str = [f"{t.from_state}->{t.to_state}:{t.status}" for t in self.fsm.get_history(event_id)]
        audit_rec = AuditRecord(
            event_id=event_id,
            context=context,
            estimates=estimates,
            decision=decision,
            outcome=outcome,
            state_transitions=history_str,
        )
        self.audit_logger.log_record(audit_rec)

        return {
            "event_id": event_id,
            "context": context.model_dump(),
            "decision": decision.model_dump(),
            "action_result": action_result,
            "outcome": outcome,
            "state_transitions": history_str,
        }
