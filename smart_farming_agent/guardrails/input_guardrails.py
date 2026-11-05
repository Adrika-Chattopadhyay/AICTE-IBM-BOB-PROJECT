"""
Guardrail 1: INPUT GUARDRAILS
Validates and sanitizes user inputs before passing them to the AI agent.

Checks:
  - Input length validation
  - Prompt injection detection
  - Malicious input detection
  - PII (Personally Identifiable Information) detection
  - Inappropriate content filtering
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.utils.logger import logger, audit_log

settings = get_settings()

# ---------------------------------------------------------------------------
# Prompt injection patterns (covers most known attack vectors)
# ---------------------------------------------------------------------------
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(your\s+)?(system|prior|previous)\s+(prompt|instructions?|rules?)",
    r"you\s+are\s+now\s+(a\s+)?(?!a\s+farming)",  # role-hijacking
    r"act\s+as\s+(if\s+you\s+are\s+)?(?!a\s+(farming|agriculture))",
    r"forget\s+(everything|all)\s+(you\s+know|above)",
    r"new\s+instructions?:\s*",
    r"system\s*prompt\s*[:=]",
    r"<\s*system\s*>",
    r"\[INST\].*override",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"sudo\s+(mode|override)",
    r"bypass\s+(safety|filter|guardrail)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"roleplay\s+as",
]

# ---------------------------------------------------------------------------
# Malicious input patterns
# ---------------------------------------------------------------------------
MALICIOUS_PATTERNS = [
    r"<script[^>]*>.*?</script>",  # XSS
    r"javascript\s*:",
    r"on\w+\s*=\s*['\"]",          # HTML event handlers
    r"eval\s*\(",
    r"exec\s*\(",
    r"__import__",
    r"os\.system",
    r"subprocess\.",
    r"open\s*\(\s*['\"]",          # file ops
]

# ---------------------------------------------------------------------------
# Inappropriate content keywords (farming-context scoped)
# ---------------------------------------------------------------------------
INAPPROPRIATE_KEYWORDS = {
    "violence", "hate", "kill", "murder", "bomb", "weapon", "drug",
    "narcotics", "terrorism", "abuse", "harassment", "explicit",
}

# ---------------------------------------------------------------------------
# PII regex patterns
# ---------------------------------------------------------------------------
PII_PATTERNS = {
    "aadhaar": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "pan_card": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "phone_india": r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
    "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "credit_card": r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
    "passport": r"\b[A-Z]\d{7}\b",
    "bank_account": r"\b\d{9,18}\b",
    "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
}


@dataclass
class InputValidationResult:
    is_valid: bool
    cleaned_input: str
    blocked_reason: Optional[str] = None
    warnings: list = field(default_factory=list)
    pii_detected: list = field(default_factory=list)


class InputGuardrail:
    """
    Validates user input through multiple safety checks before
    it reaches the AI model or any downstream tool.
    """

    def __init__(self):
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in PROMPT_INJECTION_PATTERNS
        ]
        self._malicious_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in MALICIOUS_PATTERNS
        ]
        self._pii_patterns = {
            k: re.compile(v) for k, v in PII_PATTERNS.items()
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def validate(self, user_input: str, session_id: str = "unknown") -> InputValidationResult:
        """
        Run all input checks.  Returns an InputValidationResult whose
        `is_valid` flag gates whether the input should proceed.
        """
        result = InputValidationResult(is_valid=True, cleaned_input=user_input)

        # 1. Length check
        length_ok, reason = self._check_length(user_input)
        if not length_ok:
            result.is_valid = False
            result.blocked_reason = reason
            audit_log("INPUT_BLOCKED", reason=reason, session=session_id)
            return result

        # 2. Malicious input check (hard block)
        malicious, reason = self._check_malicious(user_input)
        if malicious:
            result.is_valid = False
            result.blocked_reason = f"Malicious input detected: {reason}"
            audit_log("INPUT_BLOCKED_MALICIOUS", reason=reason, session=session_id)
            return result

        # 3. Prompt injection detection (hard block)
        injected, reason = self._check_prompt_injection(user_input)
        if injected:
            result.is_valid = False
            result.blocked_reason = f"Prompt injection detected: {reason}"
            audit_log("INPUT_BLOCKED_INJECTION", reason=reason, session=session_id)
            return result

        # 4. PII detection (warn + redact, don't block)
        pii_found, cleaned = self._check_pii(user_input)
        if pii_found and settings.pii_detection_enabled:
            result.pii_detected = pii_found
            result.cleaned_input = cleaned
            result.warnings.append(
                f"PII detected and redacted: {', '.join(pii_found)}"
            )
            audit_log("PII_REDACTED", types=pii_found, session=session_id)

        # 5. Inappropriate content (soft block with warning)
        inappropriate, words = self._check_inappropriate(result.cleaned_input)
        if inappropriate:
            result.is_valid = False
            result.blocked_reason = (
                f"Inappropriate content detected. This assistant provides "
                f"agricultural advice only. Detected terms: {words}"
            )
            audit_log(
                "INPUT_BLOCKED_INAPPROPRIATE", terms=words, session=session_id
            )
            return result

        logger.debug(f"Input validation passed for session={session_id}")
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _check_length(self, text: str):
        stripped = text.strip()
        if len(stripped) < settings.min_input_length:
            return False, f"Input too short (min {settings.min_input_length} chars)."
        if len(stripped) > settings.max_input_length:
            return False, (
                f"Input too long ({len(stripped)} chars). "
                f"Max allowed: {settings.max_input_length}."
            )
        return True, None

    def _check_malicious(self, text: str):
        for pattern in self._malicious_patterns:
            match = pattern.search(text)
            if match:
                return True, match.group()[:60]
        return False, None

    def _check_prompt_injection(self, text: str):
        for pattern in self._injection_patterns:
            match = pattern.search(text)
            if match:
                return True, match.group()[:80]
        return False, None

    def _check_pii(self, text: str):
        found_types = []
        cleaned = text
        for pii_type, pattern in self._pii_patterns.items():
            if pattern.search(cleaned):
                found_types.append(pii_type)
                cleaned = pattern.sub(f"[{pii_type.upper()}_REDACTED]", cleaned)
        return found_types, cleaned

    def _check_inappropriate(self, text: str):
        lower = text.lower()
        found = [w for w in INAPPROPRIATE_KEYWORDS if w in lower]
        return bool(found), found
