"""
Guardrail 3: TOOL EXECUTION GUARDRAILS
Controls how AI agents interact with external tools and APIs.

Checks:
  - Tool access permissions
  - API authorization
  - Function-call validation
  - Restricted tool execution
  - Prevention of unauthorized actions
"""
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.utils.logger import logger, audit_log

settings = get_settings()

# ---------------------------------------------------------------------------
# Tool permission registry
# Each tool entry defines: allowed_roles, requires_auth, is_destructive
# ---------------------------------------------------------------------------
TOOL_REGISTRY: Dict[str, Dict] = {
    "weather": {
        "description": "Get current weather and forecast data",
        "allowed_roles": ["farmer", "advisor", "admin"],
        "requires_auth": False,
        "is_destructive": False,
        "required_params": ["location"],
    },
    "soil": {
        "description": "Get soil composition and health data",
        "allowed_roles": ["farmer", "advisor", "admin"],
        "requires_auth": False,
        "is_destructive": False,
        "required_params": ["latitude", "longitude"],
    },
    "crop": {
        "description": "Get crop recommendation based on soil and season",
        "allowed_roles": ["farmer", "advisor", "admin"],
        "requires_auth": False,
        "is_destructive": False,
        "required_params": ["soil_type", "season"],
    },
    "market_price": {
        "description": "Get current mandi/market prices",
        "allowed_roles": ["farmer", "advisor", "admin"],
        "requires_auth": True,
        "is_destructive": False,
        "required_params": ["commodity"],
    },
    "pest_control": {
        "description": "Get pest identification and control recommendations",
        "allowed_roles": ["farmer", "advisor", "admin"],
        "requires_auth": False,
        "is_destructive": False,
        "required_params": ["crop", "symptoms"],
    },
    "scheme_info": {
        "description": "Get government scheme and subsidy information",
        "allowed_roles": ["farmer", "advisor", "admin"],
        "requires_auth": False,
        "is_destructive": False,
        "required_params": ["state"],
    },
    # --- Restricted tools (admin only) ---
    "database_write": {
        "description": "Write data to the knowledge base",
        "allowed_roles": ["admin"],
        "requires_auth": True,
        "is_destructive": True,
        "required_params": ["table", "data"],
    },
    "send_alert": {
        "description": "Send SMS/push alert to farmer",
        "allowed_roles": ["admin", "advisor"],
        "requires_auth": True,
        "is_destructive": False,
        "required_params": ["recipient", "message"],
    },
}

# ---------------------------------------------------------------------------
# Blocked tool names (absolute deny-list)
# ---------------------------------------------------------------------------
BLOCKED_TOOLS = {
    "shell", "exec", "os_command", "file_write", "file_delete",
    "system", "subprocess", "eval", "network_scan", "port_scan",
}


@dataclass
class ToolExecutionResult:
    allowed: bool
    tool_name: str
    reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolExecutionGuardrail:
    """
    Validates every tool call before execution.
    Maintains per-session rate-limit counters.
    """

    def __init__(self):
        # session_id -> list of call timestamps
        self._call_log: Dict[str, List[float]] = defaultdict(list)
        # session_id -> total tool calls
        self._session_totals: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def validate_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        session_id: str = "unknown",
        user_role: str = "farmer",
    ) -> ToolExecutionResult:
        """
        Validate a requested tool call.  Returns ToolExecutionResult.
        """

        # 1. Absolute block-list
        if tool_name.lower() in BLOCKED_TOOLS:
            audit_log(
                "TOOL_BLOCKED_DENY_LIST",
                tool=tool_name,
                session=session_id,
            )
            return ToolExecutionResult(
                allowed=False,
                tool_name=tool_name,
                reason=f"Tool '{tool_name}' is permanently restricted.",
            )

        # 2. Allowed tools list check
        if tool_name not in settings.allowed_tools_list:
            audit_log(
                "TOOL_NOT_PERMITTED",
                tool=tool_name,
                session=session_id,
            )
            return ToolExecutionResult(
                allowed=False,
                tool_name=tool_name,
                reason=f"Tool '{tool_name}' is not in the permitted tools list.",
            )

        # 3. Registry check
        tool_spec = TOOL_REGISTRY.get(tool_name)
        if not tool_spec:
            return ToolExecutionResult(
                allowed=False,
                tool_name=tool_name,
                reason=f"Tool '{tool_name}' is not registered.",
            )

        # 4. Role-based access control
        if user_role not in tool_spec["allowed_roles"]:
            audit_log(
                "TOOL_ROLE_DENIED",
                tool=tool_name,
                role=user_role,
                session=session_id,
            )
            return ToolExecutionResult(
                allowed=False,
                tool_name=tool_name,
                reason=f"Role '{user_role}' does not have permission to use '{tool_name}'.",
            )

        # 5. Required parameter validation
        missing = [
            p for p in tool_spec["required_params"] if p not in params or params[p] is None
        ]
        if missing:
            return ToolExecutionResult(
                allowed=False,
                tool_name=tool_name,
                reason=f"Missing required parameters for '{tool_name}': {missing}",
            )

        # 6. Session-level rate limiting (per minute)
        if not self._check_rate_limit(session_id):
            audit_log(
                "TOOL_RATE_LIMIT",
                tool=tool_name,
                session=session_id,
            )
            return ToolExecutionResult(
                allowed=False,
                tool_name=tool_name,
                reason=(
                    f"Rate limit exceeded: max {settings.api_rate_limit_per_minute} "
                    "tool calls per minute."
                ),
            )

        # 7. Session total cap
        self._session_totals[session_id] += 1
        if self._session_totals[session_id] > settings.max_tool_calls_per_session:
            audit_log(
                "TOOL_SESSION_CAP_EXCEEDED",
                tool=tool_name,
                session=session_id,
            )
            return ToolExecutionResult(
                allowed=False,
                tool_name=tool_name,
                reason=(
                    f"Session tool call limit reached ({settings.max_tool_calls_per_session})."
                ),
            )

        # 8. Destructive action warning
        result = ToolExecutionResult(allowed=True, tool_name=tool_name)
        if tool_spec.get("is_destructive"):
            result.warnings.append(
                f"Tool '{tool_name}' performs a destructive action. Audit logged."
            )
            audit_log(
                "TOOL_DESTRUCTIVE_EXEC",
                tool=tool_name,
                session=session_id,
                role=user_role,
            )

        logger.debug(f"Tool '{tool_name}' approved for session={session_id}, role={user_role}")
        return result

    def reset_session(self, session_id: str) -> None:
        """Reset counters when a new session starts."""
        self._call_log.pop(session_id, None)
        self._session_totals.pop(session_id, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _check_rate_limit(self, session_id: str) -> bool:
        """Sliding-window rate limiter (1-minute window)."""
        now = time.monotonic()
        window = 60.0
        calls = self._call_log[session_id]
        # Remove calls older than the window
        calls[:] = [t for t in calls if now - t < window]
        if len(calls) >= settings.api_rate_limit_per_minute:
            return False
        calls.append(now)
        return True
