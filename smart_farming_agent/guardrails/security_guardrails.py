"""
Guardrail 5: SECURITY GUARDRAILS
Protects the Agentic AI application from security threats.

Checks:
  - Prompt injection protection (deep pattern scanning)
  - Jailbreak detection
  - SQL injection prevention
  - Cross-agent communication security (HMAC signing)
  - Credential and secret management (no plaintext secrets)
"""
import hashlib
import hmac
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.utils.logger import logger, audit_log

settings = get_settings()

# ---------------------------------------------------------------------------
# Jailbreak detection patterns
# ---------------------------------------------------------------------------
JAILBREAK_PATTERNS = [
    r"do\s+anything\s+now",
    r"\bdan\b",                         # "Do Anything Now" persona
    r"grandfather\s+(trick|exploit|hack)",
    r"hypothetically\s+speaking\s*,?\s*(how|what|tell)",
    r"in\s+a\s+fictional\s+(world|story|universe).*?(how|what|can\s+you)",
    r"for\s+(research|educational)\s+purposes.*?(how\s+to|instructions)",
    r"write\s+(a\s+)?story\s+where.*?(hacking|cracking|exploit)",
    r"as\s+a\s+(researcher|professor|scientist).*?(help\s+me\s+(hack|crack|bypass))",
    r"evil\s+(twin|mode|version)",
    r"no\s+restrictions",
    r"unrestricted\s+(ai|mode|version)",
    r"without\s+(any\s+)?(rules|restrictions|limits|guardrails)",
    r"override\s+(safety|restrictions?|filter|rules?)",
    r"you\s+are\s+(not|no\s+longer)\s+bound\s+by",
]

# ---------------------------------------------------------------------------
# SQL injection patterns
# ---------------------------------------------------------------------------
SQL_INJECTION_PATTERNS = [
    r"(?i)(union\s+select|union\s+all\s+select)",
    r"(?i)(drop\s+table|drop\s+database|truncate\s+table)",
    r"(?i)(insert\s+into|delete\s+from|update\s+\w+\s+set)",
    r"(?i)(exec\s*\(|execute\s*\()",
    r"(?i)(xp_cmdshell|sp_executesql)",
    r"(?i)(\bor\b\s+\d+\s*=\s*\d+|\band\b\s+\d+\s*=\s*\d+)",
    r"--\s*$",                           # SQL comment injection
    r";\s*(drop|delete|insert|update)",
    r"(?i)(sleep\s*\(\d+\)|waitfor\s+delay)",
    r"(?i)(char\s*\(\d+\)|ascii\s*\(|substring\s*\(.*?from)",
    r"(?i)(load_file\s*\(|into\s+outfile\s+['\"])",
]

# ---------------------------------------------------------------------------
# Credential / secret leak patterns (for inter-agent messages)
# ---------------------------------------------------------------------------
SECRET_LEAK_PATTERNS = [
    re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(secret[_-]?key|secret)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-._~+/]{20,}"),
    re.compile(r"(?i)(aws|gcp|azure)[_-]?(access[_-]?key|secret)\s*[:=]\s*\S+"),
]


@dataclass
class SecurityCheckResult:
    passed: bool
    threat_type: Optional[str] = None
    reason: Optional[str] = None
    sanitized_input: Optional[str] = None
    risk_score: float = 0.0
    warnings: List[str] = field(default_factory=list)


class SecurityGuardrail:
    """
    Multi-layer security guardrail covering injection, jailbreak,
    SQL injection, cross-agent signing, and secret management.
    """

    def __init__(self):
        self._jailbreak_compiled = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in JAILBREAK_PATTERNS
        ]
        self._sql_compiled = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in SQL_INJECTION_PATTERNS
        ]
        # Failed attempt tracker: ip/session -> (count, first_fail_time)
        self._failed_attempts: Dict[str, Tuple[int, float]] = defaultdict(
            lambda: (0, time.monotonic())
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def check_input(
        self,
        text: str,
        session_id: str = "unknown",
        client_id: str = "unknown",
    ) -> SecurityCheckResult:
        """Full security scan on raw user input."""

        # Lockout check
        if self._is_locked_out(client_id):
            return SecurityCheckResult(
                passed=False,
                threat_type="LOCKOUT",
                reason=(
                    f"Too many failed attempts. "
                    f"Try again after {settings.lockout_duration_minutes} minutes."
                ),
                risk_score=1.0,
            )

        # SQL injection check
        if settings.sql_injection_prevention:
            hit, snippet = self._check_sql(text)
            if hit:
                self._record_failed_attempt(client_id)
                audit_log(
                    "SECURITY_SQL_INJECTION",
                    snippet=snippet,
                    session=session_id,
                )
                return SecurityCheckResult(
                    passed=False,
                    threat_type="SQL_INJECTION",
                    reason="SQL injection pattern detected.",
                    risk_score=0.95,
                )

        # Jailbreak check
        if settings.jailbreak_detection_enabled:
            hit, snippet = self._check_jailbreak(text)
            if hit:
                self._record_failed_attempt(client_id)
                audit_log(
                    "SECURITY_JAILBREAK",
                    snippet=snippet,
                    session=session_id,
                )
                return SecurityCheckResult(
                    passed=False,
                    threat_type="JAILBREAK",
                    reason="Jailbreak attempt detected.",
                    risk_score=0.90,
                )

        # Secret leak check in user message
        leaked, pattern_name = self._check_secret_leak(text)
        if leaked:
            audit_log(
                "SECURITY_SECRET_IN_INPUT",
                pattern=pattern_name,
                session=session_id,
            )
            return SecurityCheckResult(
                passed=False,
                threat_type="SECRET_LEAK",
                reason=(
                    f"Potential credential/secret found in input ({pattern_name}). "
                    "Do not share API keys or passwords."
                ),
                risk_score=0.85,
            )

        return SecurityCheckResult(passed=True, risk_score=0.0)

    def sign_agent_message(self, payload: str, agent_id: str) -> str:
        """
        Generate an HMAC-SHA256 signature for cross-agent messages.
        Agents must verify this before acting on received messages.
        """
        key = (settings.app_secret_key + agent_id).encode()
        sig = hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()
        return sig

    def verify_agent_message(
        self, payload: str, signature: str, agent_id: str
    ) -> bool:
        """Verify HMAC signature of a cross-agent message."""
        expected = self.sign_agent_message(payload, agent_id)
        valid = hmac.compare_digest(expected, signature)
        if not valid:
            audit_log("SECURITY_AGENT_SIG_FAIL", agent=agent_id)
        return valid

    def sanitize_for_db(self, value: str) -> str:
        """
        Strip SQL-dangerous characters for an extra layer of protection
        before any DB query string interpolation (parameterized queries
        remain the primary defence).
        """
        return re.sub(r"[;'\"\\]", "", value)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _check_jailbreak(self, text: str) -> Tuple[bool, Optional[str]]:
        for pattern in self._jailbreak_compiled:
            match = pattern.search(text)
            if match:
                return True, match.group()[:80]
        return False, None

    def _check_sql(self, text: str) -> Tuple[bool, Optional[str]]:
        for pattern in self._sql_compiled:
            match = pattern.search(text)
            if match:
                return True, match.group()[:80]
        return False, None

    def _check_secret_leak(self, text: str) -> Tuple[bool, Optional[str]]:
        for pattern in SECRET_LEAK_PATTERNS:
            match = pattern.search(text)
            if match:
                return True, pattern.pattern[:40]
        return False, None

    def _record_failed_attempt(self, client_id: str) -> None:
        count, first_fail = self._failed_attempts[client_id]
        self._failed_attempts[client_id] = (count + 1, first_fail)

    def _is_locked_out(self, client_id: str) -> bool:
        count, first_fail = self._failed_attempts[client_id]
        window = settings.lockout_duration_minutes * 60
        if time.monotonic() - first_fail > window:
            # Reset after lockout period
            self._failed_attempts[client_id] = (0, time.monotonic())
            return False
        return count >= settings.max_failed_attempts
