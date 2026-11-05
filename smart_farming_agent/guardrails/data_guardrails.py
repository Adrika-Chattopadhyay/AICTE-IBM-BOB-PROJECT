"""
Guardrail 4: DATA GUARDRAILS
Protects data accessed and processed by agents.

Checks:
  - Data privacy protection
  - Data masking and anonymization
  - Role-based data access
  - Database access restrictions
  - Prevention of unauthorized data retrieval
"""
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.utils.logger import logger, audit_log

settings = get_settings()

# ---------------------------------------------------------------------------
# Role → accessible data categories
# ---------------------------------------------------------------------------
ROLE_DATA_PERMISSIONS: Dict[str, List[str]] = {
    "farmer": [
        "weather", "crop_recommendations", "market_prices",
        "pest_alerts", "govt_schemes", "soil_reports",
    ],
    "advisor": [
        "weather", "crop_recommendations", "market_prices",
        "pest_alerts", "govt_schemes", "soil_reports",
        "farmer_profiles_anonymized", "regional_analytics",
    ],
    "admin": [
        "weather", "crop_recommendations", "market_prices",
        "pest_alerts", "govt_schemes", "soil_reports",
        "farmer_profiles", "regional_analytics",
        "system_logs", "audit_trails", "user_management",
    ],
}

# ---------------------------------------------------------------------------
# Sensitive field definitions per data category
# ---------------------------------------------------------------------------
SENSITIVE_FIELDS: Dict[str, List[str]] = {
    "farmer_profiles": ["aadhaar", "phone", "bank_account", "address", "gps_location"],
    "market_prices": [],  # no sensitive fields
    "system_logs": ["ip_address", "session_id"],
    "audit_trails": ["user_id", "ip_address"],
}

# ---------------------------------------------------------------------------
# Fields that must be masked for non-admin roles
# ---------------------------------------------------------------------------
ALWAYS_MASK_FIELDS = {
    "aadhaar", "pan", "passport", "bank_account", "ifsc",
    "credit_card", "debit_card", "cvv", "pin", "password",
    "secret", "private_key", "api_key", "token",
}


@dataclass
class DataAccessResult:
    allowed: bool
    data_category: str
    sanitized_record: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    masked_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DataGuardrail:
    """
    Enforces role-based data access control, masks sensitive fields,
    and prevents unauthorized data retrieval.
    """

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def check_access(
        self,
        data_category: str,
        user_role: str = "farmer",
        session_id: str = "unknown",
    ) -> DataAccessResult:
        """
        Check if `user_role` is permitted to access `data_category`.
        """
        permitted = ROLE_DATA_PERMISSIONS.get(user_role, [])
        if data_category not in permitted:
            audit_log(
                "DATA_ACCESS_DENIED",
                category=data_category,
                role=user_role,
                session=session_id,
            )
            return DataAccessResult(
                allowed=False,
                data_category=data_category,
                reason=(
                    f"Role '{user_role}' does not have access to data category "
                    f"'{data_category}'."
                ),
            )
        audit_log(
            "DATA_ACCESS_GRANTED",
            category=data_category,
            role=user_role,
            session=session_id,
        )
        return DataAccessResult(allowed=True, data_category=data_category)

    def sanitize_record(
        self,
        record: Dict[str, Any],
        user_role: str = "farmer",
        data_category: str = "unknown",
        session_id: str = "unknown",
    ) -> DataAccessResult:
        """
        Mask or remove sensitive fields from a data record
        before returning it to the caller.
        """
        result = DataAccessResult(
            allowed=True,
            data_category=data_category,
            sanitized_record=dict(record),
        )

        sensitive_for_category = set(SENSITIVE_FIELDS.get(data_category, []))
        fields_to_mask = ALWAYS_MASK_FIELDS | sensitive_for_category

        # Non-admin roles also mask anonymized farmer data
        if user_role != "admin":
            fields_to_mask |= {"farmer_id", "full_name", "aadhaar", "phone", "address"}

        masked = []
        for key in list(result.sanitized_record.keys()):
            if key.lower() in fields_to_mask:
                original_val = str(result.sanitized_record[key])
                result.sanitized_record[key] = self._hash_value(original_val)
                masked.append(key)

        if masked:
            result.masked_fields = masked
            audit_log(
                "DATA_FIELDS_MASKED",
                fields=masked,
                role=user_role,
                session=session_id,
            )

        return result

    def prevent_bulk_retrieval(
        self,
        requested_records: int,
        user_role: str = "farmer",
        session_id: str = "unknown",
    ) -> bool:
        """
        Return True (allowed) only if the requested record count is within role limits.
        """
        limits = {"farmer": 50, "advisor": 500, "admin": 10_000}
        cap = limits.get(user_role, 10)
        if requested_records > cap:
            audit_log(
                "DATA_BULK_RETRIEVAL_BLOCKED",
                requested=requested_records,
                limit=cap,
                role=user_role,
                session=session_id,
            )
            logger.warning(
                f"Bulk data retrieval blocked: {requested_records} > {cap} for role '{user_role}'"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_value(value: str) -> str:
        """One-way hash for anonymizing sensitive field values."""
        return "***" + hashlib.sha256(value.encode()).hexdigest()[:8] + "***"
