"""
Guardrail 2: OUTPUT GUARDRAILS
Validates AI-generated responses before displaying them to users.

Checks:
  - Response length validation
  - Hallucination / confidence detection
  - Factual accuracy validation (keyword grounding)
  - Toxicity and harmful content filtering
  - Sensitive information filtering
  - Response format validation (JSON, XML)
"""
import re
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.utils.logger import logger, audit_log

settings = get_settings()

# ---------------------------------------------------------------------------
# Toxicity keywords (extended for output context)
# ---------------------------------------------------------------------------
TOXIC_KEYWORDS = {
    "idiot", "stupid", "dumb", "useless", "worthless", "hate", "kill",
    "die", "suicide", "drug", "weapon", "abuse", "racist", "sexist",
    "discriminat",
}

# ---------------------------------------------------------------------------
# Sensitive output patterns (should not appear in responses)
# ---------------------------------------------------------------------------
SENSITIVE_OUTPUT_PATTERNS = {
    "api_key": re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*\S+"),
    "credit_card": re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
    "phone": re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"),
    "aadhaar": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
}

# ---------------------------------------------------------------------------
# Hallucination indicators (hedge words that signal uncertainty)
# ---------------------------------------------------------------------------
HALLUCINATION_HEDGES = [
    r"\bi\s+(think|believe|suppose|guess|assume)\b",
    r"\bprobably\b",
    r"\bmight\s+be\b",
    r"\bnot\s+sure\b",
    r"\bi\s+(don'?t|do\s+not)\s+know\b",
    r"\bI'm\s+not\s+certain\b",
    r"\bperhaps\b",
    r"\bcould\s+be\s+(wrong|incorrect)\b",
]

# ---------------------------------------------------------------------------
# Farming domain grounding keywords (for factual accuracy check)
# ---------------------------------------------------------------------------
FARMING_DOMAIN_KEYWORDS = {
    "crop", "soil", "weather", "rain", "irrigation", "fertilizer",
    "pesticide", "harvest", "season", "temperature", "humidity",
    "plant", "seed", "yield", "market", "price", "mandi", "farm",
    "agriculture", "water", "nitrogen", "phosphorus", "potassium",
    "rabi", "kharif", "zaid", "sowing", "reaping", "pest", "disease",
    "organic", "subsidy", "scheme", "government", "acre", "hectare",
}


@dataclass
class OutputValidationResult:
    is_valid: bool
    sanitized_output: str
    confidence_score: float = 1.0
    blocked_reason: Optional[str] = None
    warnings: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OutputGuardrail:
    """
    Validates and sanitizes AI-generated responses before
    they are displayed to the end user.
    """

    def __init__(self):
        self._hedge_patterns = [
            re.compile(p, re.IGNORECASE) for p in HALLUCINATION_HEDGES
        ]

    def validate(
        self,
        response: str,
        context: str = "",
        expected_format: Optional[str] = None,
        session_id: str = "unknown",
    ) -> OutputValidationResult:
        """
        Run all output checks. Returns an OutputValidationResult.
        `expected_format` can be 'json', 'xml', or None for plain text.
        """
        result = OutputValidationResult(is_valid=True, sanitized_output=response)

        # 1. Length validation
        if len(response) > settings.max_output_length:
            response = response[: settings.max_output_length] + "\n\n[Response truncated for safety]"
            result.warnings.append("Response truncated to maximum allowed length.")
            result.sanitized_output = response

        if len(response.strip()) < 5:
            result.is_valid = False
            result.blocked_reason = "Response is empty or too short."
            return result

        # 2. Toxicity check (hard block)
        toxic, words = self._check_toxicity(response)
        if toxic:
            result.is_valid = False
            result.blocked_reason = (
                "Response contains harmful or toxic content and cannot be displayed."
            )
            audit_log(
                "OUTPUT_BLOCKED_TOXIC",
                session=session_id,
                terms=words,
            )
            return result

        # 3. Sensitive information filtering
        sanitized, found_sensitive = self._filter_sensitive(response)
        if found_sensitive:
            result.sanitized_output = sanitized
            result.warnings.append(
                f"Sensitive data masked in response: {found_sensitive}"
            )
            audit_log("OUTPUT_SENSITIVE_MASKED", session=session_id, types=found_sensitive)

        # 4. Hallucination detection (confidence scoring)
        hedge_count = self._count_hedges(response)
        confidence = max(0.0, 1.0 - hedge_count * 0.15)
        result.confidence_score = round(confidence, 2)
        if confidence < settings.hallucination_threshold:
            result.warnings.append(
                f"Low confidence response (score={confidence:.2f}). "
                "Please verify with a local agricultural expert."
            )
            result.metadata["low_confidence"] = True

        # 5. Factual accuracy — domain grounding
        domain_score = self._check_domain_relevance(response)
        result.metadata["domain_relevance_score"] = round(domain_score, 2)
        if domain_score < settings.factual_accuracy_min_score and len(response.split()) > 20:
            result.warnings.append(
                "Response may contain information outside the agricultural domain. "
                "Verify with trusted sources."
            )

        # 6. Format validation (if required)
        if expected_format:
            fmt_valid, fmt_error = self._validate_format(
                result.sanitized_output, expected_format
            )
            if not fmt_valid:
                result.warnings.append(f"Response format issue ({expected_format}): {fmt_error}")

        logger.debug(
            f"Output validation: valid={result.is_valid}, confidence={result.confidence_score}, session={session_id}"
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _check_toxicity(self, text: str):
        lower = text.lower()
        found = [w for w in TOXIC_KEYWORDS if w in lower]
        return bool(found), found

    def _filter_sensitive(self, text: str):
        cleaned = text
        found = []
        if not settings.sensitive_info_filter_enabled:
            return cleaned, found
        for label, pattern in SENSITIVE_OUTPUT_PATTERNS.items():
            if pattern.search(cleaned):
                found.append(label)
                cleaned = pattern.sub(f"[{label.upper()}_MASKED]", cleaned)
        return cleaned, found

    def _count_hedges(self, text: str) -> int:
        return sum(1 for p in self._hedge_patterns if p.search(text))

    def _check_domain_relevance(self, text: str) -> float:
        words = set(re.findall(r"\b\w+\b", text.lower()))
        if not words:
            return 0.0
        matched = words & FARMING_DOMAIN_KEYWORDS
        return len(matched) / max(len(FARMING_DOMAIN_KEYWORDS) * 0.15, 1)

    def _validate_format(self, text: str, fmt: str):
        fmt = fmt.lower().strip()
        if fmt == "json":
            try:
                json.loads(text)
                return True, None
            except json.JSONDecodeError as e:
                return False, str(e)
        if fmt == "xml":
            try:
                ET.fromstring(text)
                return True, None
            except ET.ParseError as e:
                return False, str(e)
        return True, None  # unknown format — pass through
