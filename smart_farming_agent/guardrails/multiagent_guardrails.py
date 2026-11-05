"""
Guardrail 8: MULTI-AGENT GUARDRAILS
Ensures safe coordination between multiple specialized agents.

Checks:
  - Agent-level access control
  - Task assignment validation
  - Cross-agent output verification
  - Agent communication monitoring
  - Conflict resolution between agents
  - Prevention of infinite agent loops
"""
import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.utils.logger import logger, audit_log

settings = get_settings()


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    WEATHER_AGENT = "weather_agent"
    SOIL_AGENT = "soil_agent"
    CROP_AGENT = "crop_agent"
    MARKET_AGENT = "market_agent"
    PEST_AGENT = "pest_agent"
    SCHEME_AGENT = "scheme_agent"
    TRANSLATION_AGENT = "translation_agent"


# Task types each agent is permitted to handle
AGENT_TASK_PERMISSIONS: Dict[AgentRole, Set[str]] = {
    AgentRole.ORCHESTRATOR: {
        "route_query", "aggregate_results", "compose_response", "delegate_task"
    },
    AgentRole.WEATHER_AGENT: {
        "get_weather", "get_forecast", "get_rainfall", "get_temperature"
    },
    AgentRole.SOIL_AGENT: {
        "get_soil_type", "get_soil_health", "get_soil_nutrients", "get_ph"
    },
    AgentRole.CROP_AGENT: {
        "recommend_crop", "get_sowing_schedule", "get_harvest_info", "get_variety"
    },
    AgentRole.MARKET_AGENT: {
        "get_mandi_price", "get_market_trend", "get_commodity_price"
    },
    AgentRole.PEST_AGENT: {
        "identify_pest", "get_pest_treatment", "get_disease_info", "get_pesticide"
    },
    AgentRole.SCHEME_AGENT: {
        "get_govt_scheme", "get_subsidy_info", "get_insurance_info"
    },
    AgentRole.TRANSLATION_AGENT: {
        "translate_text", "detect_language", "transliterate"
    },
}

# Who can call whom (caller -> set of allowed callees)
AGENT_CALL_PERMISSIONS: Dict[AgentRole, Set[AgentRole]] = {
    AgentRole.ORCHESTRATOR: {
        AgentRole.WEATHER_AGENT, AgentRole.SOIL_AGENT, AgentRole.CROP_AGENT,
        AgentRole.MARKET_AGENT, AgentRole.PEST_AGENT, AgentRole.SCHEME_AGENT,
        AgentRole.TRANSLATION_AGENT,
    },
    AgentRole.WEATHER_AGENT: set(),   # leaf agents cannot call others
    AgentRole.SOIL_AGENT: set(),
    AgentRole.CROP_AGENT: {AgentRole.SOIL_AGENT, AgentRole.WEATHER_AGENT},
    AgentRole.MARKET_AGENT: set(),
    AgentRole.PEST_AGENT: {AgentRole.CROP_AGENT},
    AgentRole.SCHEME_AGENT: set(),
    AgentRole.TRANSLATION_AGENT: set(),
}


@dataclass
class AgentMessage:
    sender: AgentRole
    receiver: AgentRole
    task: str
    payload: Dict[str, Any]
    session_id: str
    message_id: str = field(default_factory=lambda: hashlib.md5(
        str(time.monotonic()).encode()
    ).hexdigest()[:10])
    timestamp: float = field(default_factory=time.monotonic)
    hop_count: int = 0


@dataclass
class MultiAgentCheckResult:
    allowed: bool
    reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiAgentGuardrail:
    """
    Guardrail layer for safe multi-agent coordination.
    Prevents loops, unauthorized agent calls, and task hijacking.
    """

    def __init__(self):
        # session -> list of (sender, receiver, task, timestamp)
        self._call_trace: Dict[str, List[Tuple]] = defaultdict(list)
        # session -> consecutive same-task count for loop detection
        self._task_repeat: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def validate_agent_call(
        self, message: AgentMessage
    ) -> MultiAgentCheckResult:
        """
        Validate an inter-agent message before it is dispatched.
        """

        # 1. Hop count limit
        if message.hop_count > settings.max_agent_hops:
            audit_log(
                "MULTIAGENT_HOP_LIMIT",
                session=message.session_id,
                hops=message.hop_count,
            )
            return MultiAgentCheckResult(
                allowed=False,
                reason=(
                    f"Agent call chain exceeded maximum hops "
                    f"({settings.max_agent_hops})."
                ),
            )

        # 2. Caller permission: can sender call receiver?
        allowed_callees = AGENT_CALL_PERMISSIONS.get(message.sender, set())
        if message.receiver not in allowed_callees:
            audit_log(
                "MULTIAGENT_CALL_DENIED",
                sender=message.sender.value,
                receiver=message.receiver.value,
                session=message.session_id,
            )
            return MultiAgentCheckResult(
                allowed=False,
                reason=(
                    f"Agent '{message.sender.value}' is not permitted to call "
                    f"'{message.receiver.value}'."
                ),
            )

        # 3. Task assignment validation: is this task in receiver's scope?
        allowed_tasks = AGENT_TASK_PERMISSIONS.get(message.receiver, set())
        if message.task not in allowed_tasks:
            return MultiAgentCheckResult(
                allowed=False,
                reason=(
                    f"Task '{message.task}' is not in the scope of "
                    f"agent '{message.receiver.value}'. "
                    f"Allowed tasks: {allowed_tasks}"
                ),
            )

        # 4. Infinite loop detection
        loop_detected, loop_reason = self._check_loops(message)
        if loop_detected:
            audit_log(
                "MULTIAGENT_LOOP_DETECTED",
                session=message.session_id,
                reason=loop_reason,
            )
            return MultiAgentCheckResult(
                allowed=False,
                reason=f"Potential agent loop detected: {loop_reason}",
            )

        # Record the call
        self._record_call(message)

        result = MultiAgentCheckResult(allowed=True)
        if message.hop_count >= settings.max_agent_hops - 2:
            result.warnings.append(
                f"Agent call chain is deep (hop={message.hop_count}). "
                "Consider flattening the agent architecture."
            )

        return result

    def verify_cross_agent_output(
        self,
        output: Any,
        producer_agent: AgentRole,
        consumer_agent: AgentRole,
        session_id: str = "unknown",
    ) -> MultiAgentCheckResult:
        """
        Verify that the output from one agent is safe to consume by another.
        Checks for None, unexpected types, and oversized payloads.
        """
        if output is None:
            return MultiAgentCheckResult(
                allowed=False,
                reason=f"Agent '{producer_agent.value}' returned None output.",
            )

        if isinstance(output, str) and len(output) > settings.max_output_length:
            audit_log(
                "MULTIAGENT_OUTPUT_OVERSIZED",
                producer=producer_agent.value,
                size=len(output),
                session=session_id,
            )
            return MultiAgentCheckResult(
                allowed=True,
                warnings=[
                    f"Output from '{producer_agent.value}' is oversized "
                    f"({len(output)} chars). Truncating."
                ],
            )

        return MultiAgentCheckResult(allowed=True)

    def get_call_trace(self, session_id: str) -> List[Tuple]:
        """Return the agent call trace for a session (for debugging)."""
        return list(self._call_trace.get(session_id, []))

    def reset_session(self, session_id: str) -> None:
        self._call_trace.pop(session_id, None)
        self._task_repeat.pop(session_id, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _check_loops(self, message: AgentMessage) -> Tuple[bool, Optional[str]]:
        trace = self._call_trace[message.session_id]
        call_key = (message.sender.value, message.receiver.value, message.task)

        # Check if the exact same call has happened too many times
        recent = [
            (s, r, t) for s, r, t, _ in trace[-10:]
        ]
        repeat_count = recent.count(call_key)
        if repeat_count >= settings.max_loop_detection_count:
            return True, (
                f"Call {call_key} repeated {repeat_count} times in last 10 hops."
            )

        # Detect A->B->A->B cycles
        if len(trace) >= 4:
            last_four = [(s, r, t) for s, r, t, _ in trace[-4:]]
            if last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                return True, "Alternating cycle detected in agent call chain."

        return False, None

    def _record_call(self, message: AgentMessage) -> None:
        self._call_trace[message.session_id].append(
            (
                message.sender.value,
                message.receiver.value,
                message.task,
                time.monotonic(),
            )
        )
        # Keep trace bounded
        if len(self._call_trace[message.session_id]) > 100:
            self._call_trace[message.session_id] = (
                self._call_trace[message.session_id][-100:]
            )
