"""
Guardrail 7: HUMAN-IN-THE-LOOP (HITL) GUARDRAILS
Ensures human approval for critical decisions and actions.

Checks:
  - Human approval before financial transactions
  - Approval before sending emails / alerts
  - Manual verification of critical outputs
  - Escalation to human experts
  - Approval before modifying or deleting data
"""
import uuid
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.utils.logger import logger, audit_log

settings = get_settings()


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    ESCALATED = "escalated"


class ActionCategory(str, Enum):
    FINANCIAL = "financial"
    COMMUNICATION = "communication"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    CRITICAL_ADVISORY = "critical_advisory"
    SYSTEM_CHANGE = "system_change"


# Actions that always require human approval
ALWAYS_REQUIRE_APPROVAL = {
    ActionCategory.FINANCIAL,
    ActionCategory.DATA_DELETION,
    ActionCategory.SYSTEM_CHANGE,
}

# Confidence threshold below which human review is triggered
CONFIDENCE_ESCALATION_THRESHOLD = 0.50


@dataclass
class ApprovalRequest:
    request_id: str
    action_category: ActionCategory
    description: str
    payload: Dict[str, Any]
    requester: str
    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewer_notes: str = ""
    approved_by: str = ""


@dataclass
class HITLResult:
    approved: bool
    request_id: str
    status: ApprovalStatus
    message: str
    requires_human: bool = False
    pending_request: Optional[ApprovalRequest] = None


class HITLGuardrail:
    """
    Human-in-the-Loop guardrail that queues critical actions
    for human approval and provides escalation paths.
    """

    def __init__(self):
        # In-memory approval queue (production: use a database)
        self._approval_queue: Dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def request_approval(
        self,
        action_category: ActionCategory,
        description: str,
        payload: Dict[str, Any],
        requester: str = "agent",
        session_id: str = "unknown",
        confidence_score: float = 1.0,
    ) -> HITLResult:
        """
        Check whether the action needs human approval.
        If HITL is disabled globally, auto-approve low-risk actions.
        """
        needs_approval = self._needs_human_approval(
            action_category, confidence_score
        )

        if not needs_approval:
            audit_log(
                "HITL_AUTO_APPROVED",
                category=action_category.value,
                session=session_id,
            )
            return HITLResult(
                approved=True,
                request_id="auto",
                status=ApprovalStatus.APPROVED,
                message="Action auto-approved (low risk, high confidence).",
            )

        # Create approval request
        req = ApprovalRequest(
            request_id=str(uuid.uuid4())[:8],
            action_category=action_category,
            description=description,
            payload=payload,
            requester=requester,
            session_id=session_id,
        )

        with self._lock:
            self._approval_queue[req.request_id] = req

        audit_log(
            "HITL_APPROVAL_REQUESTED",
            request_id=req.request_id,
            category=action_category.value,
            session=session_id,
        )
        logger.info(
            f"HITL approval requested: id={req.request_id}, "
            f"category={action_category.value}"
        )

        return HITLResult(
            approved=False,
            request_id=req.request_id,
            status=ApprovalStatus.PENDING,
            message=(
                f"Action requires human approval (request_id={req.request_id}). "
                f"Reason: {action_category.value} action."
            ),
            requires_human=True,
            pending_request=req,
        )

    def process_approval(
        self,
        request_id: str,
        approved: bool,
        reviewer: str = "human_reviewer",
        notes: str = "",
    ) -> HITLResult:
        """
        Process a human approval/rejection decision.
        Called by the Streamlit UI when the reviewer acts.
        """
        with self._lock:
            req = self._approval_queue.get(request_id)
            if not req:
                return HITLResult(
                    approved=False,
                    request_id=request_id,
                    status=ApprovalStatus.REJECTED,
                    message=f"Approval request '{request_id}' not found.",
                )

            if self._is_timed_out(req):
                req.status = ApprovalStatus.TIMEOUT
                audit_log("HITL_TIMEOUT", request_id=request_id)
                return HITLResult(
                    approved=False,
                    request_id=request_id,
                    status=ApprovalStatus.TIMEOUT,
                    message="Approval request has timed out.",
                )

            req.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
            req.reviewer_notes = notes
            req.approved_by = reviewer

        audit_log(
            "HITL_DECISION",
            request_id=request_id,
            decision=req.status.value,
            reviewer=reviewer,
        )

        return HITLResult(
            approved=approved,
            request_id=request_id,
            status=req.status,
            message=(
                f"Action {'approved' if approved else 'rejected'} by {reviewer}."
            ),
        )

    def escalate_to_expert(
        self,
        query: str,
        context: str,
        session_id: str = "unknown",
    ) -> HITLResult:
        """
        Escalate a query to a human agricultural expert.
        In production this would trigger a notification.
        """
        escalation_id = str(uuid.uuid4())[:8]
        audit_log(
            "HITL_ESCALATED_TO_EXPERT",
            escalation_id=escalation_id,
            session=session_id,
        )
        logger.info(f"Escalated to expert: {escalation_id}")

        return HITLResult(
            approved=False,
            request_id=escalation_id,
            status=ApprovalStatus.ESCALATED,
            message=(
                "Your question has been escalated to a human agricultural expert. "
                "You will receive a response within 24 hours. "
                f"Reference ID: {escalation_id}"
            ),
            requires_human=True,
        )

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Return all pending (non-expired) approval requests."""
        with self._lock:
            return [
                r
                for r in self._approval_queue.values()
                if r.status == ApprovalStatus.PENDING
                and not self._is_timed_out(r)
            ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _needs_human_approval(
        self, category: ActionCategory, confidence: float
    ) -> bool:
        if not settings.hitl_enabled:
            return False
        if category in ALWAYS_REQUIRE_APPROVAL:
            return True
        if (
            category == ActionCategory.COMMUNICATION
            and settings.financial_transaction_approval
        ):
            return True
        if confidence < CONFIDENCE_ESCALATION_THRESHOLD:
            return True
        return False

    def _is_timed_out(self, req: ApprovalRequest) -> bool:
        elapsed = time.monotonic() - req.created_at
        return elapsed > settings.approval_timeout_seconds
