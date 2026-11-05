"""
Guardrail 9: ETHICAL & COMPLIANCE GUARDRAILS
Ensures responsible and compliant AI behaviour.

Checks:
  - Bias detection (demographic and regional bias in recommendations)
  - Fairness validation (equal treatment across user groups)
  - Explainability (decision rationale attached to every response)
  - Regulatory compliance (GDPR-like principles)
  - Responsible AI principles
  - Audit trails (immutable logs of all decisions)
"""
import hashlib
import json
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.utils.logger import logger, audit_log

settings = get_settings()

# ---------------------------------------------------------------------------
# Demographic / regional bias indicators
# ---------------------------------------------------------------------------
BIAS_INDICATORS = {
    "caste_bias": [
        r"\b(upper|lower)\s+caste\b",
        r"\b(brahmin|dalit|jat|yadav)\s+(farmer|people)\b",
    ],
    "gender_bias": [
        r"\b(women|female)\s+farmers?\s+(cannot|shouldn'?t|should\s+not)\b",
        r"\b(only\s+men|men\s+only)\s+(can|should)\b",
    ],
    "regional_bias": [
        r"\b(north|south|east|west)\s+india\s+is\s+(better|worse|superior|inferior)\b",
        r"\b(hindi[- ]speaking)\s+farmers?\s+(are\s+better|are\s+smarter)\b",
    ],
    "economic_bias": [
        r"\b(poor|rich)\s+farmers?\s+(don'?t|cannot|should\s+not)\b",
    ],
}

# ---------------------------------------------------------------------------
# GDPR / Indian PDPB compliance: fields that need consent for processing
# ---------------------------------------------------------------------------
CONSENT_REQUIRED_FIELDS = {
    "aadhaar", "phone", "email", "location", "gps", "bank_account",
    "income", "caste", "religion", "health_data",
}


@dataclass
class AuditEntry:
    entry_id: str
    timestamp: str
    session_id: str
    event_type: str
    agent_name: str
    input_summary: str
    output_summary: str
    decision: str
    confidence: float
    bias_flags: List[str]
    compliance_flags: List[str]
    user_role: str
    metadata: Dict[str, Any]


@dataclass
class EthicsCheckResult:
    passed: bool
    bias_detected: List[str] = field(default_factory=list)
    fairness_score: float = 1.0
    compliance_issues: List[str] = field(default_factory=list)
    explanation: str = ""
    audit_id: str = ""
    warnings: List[str] = field(default_factory=list)


class EthicalGuardrail:
    """
    Comprehensive ethical guardrail covering bias, fairness,
    explainability, compliance, and immutable audit trails.
    """

    def __init__(self):
        self._bias_patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in BIAS_INDICATORS.items()
        }
        if settings.audit_trail_enabled:
            self._init_audit_db()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def check_response(
        self,
        response: str,
        query: str = "",
        agent_name: str = "agent",
        session_id: str = "unknown",
        user_role: str = "farmer",
        confidence: float = 1.0,
        tool_used: Optional[str] = None,
    ) -> EthicsCheckResult:
        """
        Full ethical check on an AI-generated response.
        """
        result = EthicsCheckResult(passed=True)

        # 1. Bias detection
        bias_found = self._detect_bias(response)
        if bias_found:
            result.bias_detected = bias_found
            result.passed = False
            result.warnings.append(
                f"Biased content detected in response: {bias_found}. "
                "Response blocked."
            )
            audit_log(
                "ETHICS_BIAS_DETECTED",
                session=session_id,
                bias_types=bias_found,
            )

        # 2. Fairness validation
        result.fairness_score = self._compute_fairness_score(response, query)
        if result.fairness_score < 0.60:
            result.warnings.append(
                f"Low fairness score ({result.fairness_score:.2f}). "
                "Response may not apply equally to all farmer demographics."
            )

        # 3. Compliance check
        compliance_issues = self._check_compliance(response, query)
        if compliance_issues:
            result.compliance_issues = compliance_issues
            result.warnings.extend(compliance_issues)

        # 4. Explainability
        if settings.explainability_enabled:
            result.explanation = self._generate_explanation(
                query, agent_name, tool_used, confidence
            )

        # 5. Write audit entry
        if settings.audit_trail_enabled:
            result.audit_id = self._write_audit(
                session_id=session_id,
                event_type="response_generated",
                agent_name=agent_name,
                input_summary=query[:200],
                output_summary=response[:200],
                decision="blocked" if not result.passed else "allowed",
                confidence=confidence,
                bias_flags=result.bias_detected,
                compliance_flags=result.compliance_issues,
                user_role=user_role,
            )

        return result

    def validate_data_consent(
        self,
        data_fields: List[str],
        consent_given: bool,
        session_id: str = "unknown",
    ) -> EthicsCheckResult:
        """
        Check that data requiring consent is only processed if
        consent has been explicitly given.
        """
        result = EthicsCheckResult(passed=True)
        sensitive = [f for f in data_fields if f.lower() in CONSENT_REQUIRED_FIELDS]

        if sensitive and not consent_given:
            result.passed = False
            result.compliance_issues = [
                f"Processing of sensitive fields {sensitive} requires explicit consent."
            ]
            audit_log(
                "ETHICS_CONSENT_VIOLATION",
                fields=sensitive,
                session=session_id,
            )

        return result

    def get_audit_log(
        self, session_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        """Retrieve recent audit entries from the database."""
        if not settings.audit_trail_enabled:
            return []
        try:
            db_path = self._get_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT * FROM audit_trail WHERE session_id=? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM audit_trail ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            columns = [col[0] for col in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Audit log retrieval failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _detect_bias(self, text: str) -> List[str]:
        if not settings.bias_detection_enabled:
            return []
        found = []
        for category, patterns in self._bias_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    found.append(category)
                    break
        return found

    def _compute_fairness_score(self, response: str, query: str) -> float:
        """
        Simple heuristic: check that recommendations aren't
        conditioned on demographic qualifiers.
        """
        demographic_qualifiers = [
            r"\b(only|just)\s+(men|women|upper|lower|rich|poor)\b",
            r"\b(not\s+suitable\s+for)\s+(small|marginal|tribal|backward)\b",
        ]
        deductions = sum(
            1
            for p in demographic_qualifiers
            if re.search(p, response, re.IGNORECASE)
        )
        return max(0.0, 1.0 - deductions * 0.25)

    def _check_compliance(self, response: str, query: str) -> List[str]:
        """Check for basic GDPR / PDPB compliance signals."""
        issues = []
        # Check if response contains raw PII
        pii_patterns = [
            (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "Aadhaar number"),
            (r"\b[6-9]\d{9}\b", "Phone number"),
            (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "Email"),
        ]
        for pattern, label in pii_patterns:
            if re.search(pattern, response):
                issues.append(
                    f"Compliance: Response may contain {label}. "
                    "Verify it is necessary and consented."
                )
        return issues

    def _generate_explanation(
        self,
        query: str,
        agent_name: str,
        tool_used: Optional[str],
        confidence: float,
    ) -> str:
        """Generate a human-readable explanation of how the answer was produced."""
        tool_note = (
            f"Data was retrieved using the '{tool_used}' tool." if tool_used
            else "This response was generated from the knowledge base."
        )
        return (
            f"This advice was generated by the '{agent_name}' agent in response "
            f"to your query. {tool_note} "
            f"Confidence level: {confidence*100:.0f}%. "
            "Always consult a certified agronomist for critical decisions."
        )

    # ------------------------------------------------------------------
    # Audit database
    # ------------------------------------------------------------------
    def _get_db_path(self) -> Path:
        path = Path("data/smart_farming_audit.db")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _init_audit_db(self) -> None:
        try:
            conn = sqlite3.connect(self._get_db_path())
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_trail (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    event_type TEXT,
                    agent_name TEXT,
                    input_summary TEXT,
                    output_summary TEXT,
                    decision TEXT,
                    confidence REAL,
                    bias_flags TEXT,
                    compliance_flags TEXT,
                    user_role TEXT,
                    metadata TEXT
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Audit DB init failed (non-fatal): {exc}")

    def _write_audit(self, **kwargs) -> str:
        entry_id = str(uuid.uuid4())
        try:
            conn = sqlite3.connect(self._get_db_path())
            conn.execute(
                """
                INSERT INTO audit_trail VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    entry_id,
                    datetime.utcnow().isoformat(),
                    kwargs.get("session_id", ""),
                    kwargs.get("event_type", ""),
                    kwargs.get("agent_name", ""),
                    kwargs.get("input_summary", ""),
                    kwargs.get("output_summary", ""),
                    kwargs.get("decision", ""),
                    kwargs.get("confidence", 1.0),
                    json.dumps(kwargs.get("bias_flags", [])),
                    json.dumps(kwargs.get("compliance_flags", [])),
                    kwargs.get("user_role", ""),
                    json.dumps(kwargs.get("metadata", {})),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Audit write failed (non-fatal): {exc}")
        return entry_id
