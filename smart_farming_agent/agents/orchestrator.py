"""
Orchestrator Agent — routes farmer queries to specialized sub-agents,
aggregates responses, and applies all guardrails before returning
a final answer to the UI layer.

Pipeline per query:
  1.  Input Guardrails (security → input → data)
  2.  Language detection and optional translation
  3.  Intent classification → route to specialist agent(s)
  4.  Tool execution (with tool guardrails)
  5.  RAG-enhanced response generation (Granite)
  6.  Output Guardrails (output → ethical)
  7.  HITL check (if confidence low or action critical)
  8.  Return final result
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.rag.rag_pipeline import RAGPipeline
from smart_farming_agent.rag.granite_client import GraniteClient
from smart_farming_agent.tools.farming_tools import TOOL_INSTANCES, get_tool
from smart_farming_agent.guardrails.input_guardrails import InputGuardrail
from smart_farming_agent.guardrails.output_guardrails import OutputGuardrail
from smart_farming_agent.guardrails.tool_guardrails import ToolExecutionGuardrail
from smart_farming_agent.guardrails.data_guardrails import DataGuardrail
from smart_farming_agent.guardrails.security_guardrails import SecurityGuardrail
from smart_farming_agent.guardrails.operational_guardrails import OperationalGuardrail
from smart_farming_agent.guardrails.hitl_guardrails import (
    HITLGuardrail, ActionCategory, HITLResult,
)
from smart_farming_agent.guardrails.multiagent_guardrails import (
    MultiAgentGuardrail, AgentMessage, AgentRole,
)
from smart_farming_agent.guardrails.ethical_guardrails import EthicalGuardrail
from smart_farming_agent.utils.logger import logger, audit_log

settings = get_settings()


# ---------------------------------------------------------------------------
# Intent → Agent / Tool mapping
# ---------------------------------------------------------------------------
INTENT_MAP = {
    "weather": {
        "keywords": ["weather", "rain", "forecast", "temperature", "humidity",
                     "monsoon", "mausam", "barish", "tapman"],
        "tool": "weather",
        "agent": AgentRole.WEATHER_AGENT,
        "data_category": "weather",
    },
    "market_price": {
        "keywords": ["price", "rate", "mandi", "market", "bhav", "daam",
                     "sell", "selling", "trading", "cost", "rs", "rupee"],
        "tool": "market_price",
        "agent": AgentRole.MARKET_AGENT,
        "data_category": "market_prices",
    },
    "soil": {
        "keywords": ["soil", "mitti", "ph", "nutrient", "nitrogen", "fertilizer",
                     "organic", "clay", "sandy", "black", "alluvial", "khad"],
        "tool": "soil",
        "agent": AgentRole.SOIL_AGENT,
        "data_category": "soil_reports",
    },
    "crop": {
        "keywords": ["crop", "grow", "sow", "plant", "seed", "recommend",
                     "fasal", "kharif", "rabi", "zaid", "harvest", "yield",
                     "variety", "paddy", "wheat", "cotton", "sugarcane"],
        "tool": "crop",
        "agent": AgentRole.CROP_AGENT,
        "data_category": "crop_recommendations",
    },
    "pest": {
        "keywords": ["pest", "insect", "disease", "blight", "fungal", "bacteria",
                     "worm", "aphid", "spray", "kida", "rog", "pesticide",
                     "symptoms", "leaves", "stem", "wilt", "spot", "rust"],
        "tool": "pest_control",
        "agent": AgentRole.PEST_AGENT,
        "data_category": "crop_recommendations",
    },
    "scheme": {
        "keywords": ["scheme", "subsidy", "government", "yojana", "loan", "insurance",
                     "card", "credit", "pm-kisan", "fasal bima", "kusum",
                     "registration", "apply", "benefit", "help"],
        "tool": "scheme_info",
        "agent": AgentRole.SCHEME_AGENT,
        "data_category": "govt_schemes",
    },
}


@dataclass
class AgentResponse:
    """Final structured response returned to the UI."""
    success: bool
    answer: str
    session_id: str
    intent: str = "general"
    tool_data: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)
    explanation: str = ""
    audit_id: str = ""
    requires_human_approval: bool = False
    pending_approval_id: Optional[str] = None
    execution_time_ms: float = 0.0
    demo_mode: bool = False
    blocked_reason: Optional[str] = None


class OrchestratorAgent:
    """
    Central agent that coordinates all specialist agents and guardrails.
    Exposes a single `process_query()` method for the UI layer.
    """

    def __init__(self):
        # Core components
        self._granite = GraniteClient()
        self._rag = RAGPipeline(granite_client=self._granite)

        # Guardrails
        self._input_g = InputGuardrail()
        self._output_g = OutputGuardrail()
        self._tool_g = ToolExecutionGuardrail()
        self._data_g = DataGuardrail()
        self._security_g = SecurityGuardrail()
        self._ops_g = OperationalGuardrail()
        self._hitl = HITLGuardrail()
        self._multiagent_g = MultiAgentGuardrail()
        self._ethics_g = EthicalGuardrail()

        logger.info("OrchestratorAgent initialized.")

    # ------------------------------------------------------------------
    # Main public entry point
    # ------------------------------------------------------------------
    def process_query(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_role: str = "farmer",
        location: Optional[str] = None,
        language: str = "en",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> AgentResponse:
        """
        Full pipeline: guardrails → tool → RAG → guardrails → response.
        """
        start_time = time.monotonic()
        session_id = session_id or str(uuid.uuid4())[:8]

        audit_log("QUERY_START", session=session_id, input_len=len(user_input))

        # ---- 1. Security check (highest priority) -------------------------
        sec_result = self._security_g.check_input(
            user_input, session_id=session_id, client_id=session_id
        )
        if not sec_result.passed:
            return self._blocked_response(
                sec_result.reason or "Security check failed.",
                session_id,
                start_time,
            )

        # ---- 2. Input validation ------------------------------------------
        input_result = self._input_g.validate(user_input, session_id=session_id)
        if not input_result.is_valid:
            return self._blocked_response(
                input_result.blocked_reason or "Input validation failed.",
                session_id,
                start_time,
            )
        clean_input = input_result.cleaned_input

        # ---- 3. Operational check (iterations / budget) -------------------
        iter_result = self._ops_g.check_iteration(session_id, "orchestrator")
        if not iter_result.allowed:
            return self._blocked_response(iter_result.reason, session_id, start_time)

        # ---- 4. Intent classification ------------------------------------
        intent = self._classify_intent(clean_input)
        tool_data: Optional[Dict] = None
        intent_config = INTENT_MAP.get(intent, {})

        # ---- 5. Multi-agent routing + tool execution ----------------------
        if intent_config:
            tool_name = intent_config["tool"]
            target_agent = intent_config["agent"]
            data_category = intent_config["data_category"]

            # Multi-agent validation
            msg = AgentMessage(
                sender=AgentRole.ORCHESTRATOR,
                receiver=target_agent,
                task=f"get_{intent}",
                payload={"query": clean_input, "location": location},
                session_id=session_id,
            )
            ma_result = self._multiagent_g.validate_agent_call(msg)
            if not ma_result.allowed:
                logger.warning(f"Multi-agent check failed: {ma_result.reason}")

            # Data access check
            data_result = self._data_g.check_access(data_category, user_role, session_id)

            # Tool execution guardrail
            tool_params = self._build_tool_params(
                intent, clean_input, location, latitude, longitude
            )
            tool_check = self._tool_g.validate_tool_call(
                tool_name, tool_params, session_id=session_id, user_role=user_role
            )

            if tool_check.allowed and data_result.allowed:
                try:
                    tool_instance = get_tool(tool_name)
                    if tool_instance:
                        tool_data = self._ops_g.run_with_timeout(
                            tool_instance.call,
                            **tool_params,
                            timeout=settings.tool_timeout_seconds,
                        )
                        logger.debug(f"Tool '{tool_name}' executed successfully.")
                except TimeoutError:
                    logger.warning(f"Tool '{tool_name}' timed out.")
                    tool_data = {"error": "Tool timed out. Using knowledge base only."}
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Tool '{tool_name}' error: {exc}")
                    tool_data = {"error": str(exc)}

        # ---- 6. RAG generation -------------------------------------------
        rag_result = self._rag.query(
            question=clean_input,
            top_k=3,
            language=language,
            location=location,
            season=self._detect_season(),
        )

        # Enhance answer with tool data
        answer = self._enhance_with_tool_data(
            rag_result["answer"], tool_data, intent
        )

        # ---- 7. Output guardrails ----------------------------------------
        out_result = self._output_g.validate(
            answer, context=clean_input, session_id=session_id
        )
        if not out_result.is_valid:
            answer = (
                "I apologize, but I cannot provide that response. "
                "Please rephrase your farming question."
            )
        else:
            answer = out_result.sanitized_output

        confidence = min(rag_result["confidence"], out_result.confidence_score)

        # ---- 8. Ethics / Bias check --------------------------------------
        ethics_result = self._ethics_g.check_response(
            response=answer,
            query=clean_input,
            agent_name="orchestrator",
            session_id=session_id,
            user_role=user_role,
            confidence=confidence,
            tool_used=intent_config.get("tool") if intent_config else None,
        )
        if not ethics_result.passed:
            answer = (
                "I cannot provide this response as it may contain biased content. "
                "Please ask a general farming question."
            )
        all_warnings = (
            input_result.warnings
            + out_result.warnings
            + ethics_result.warnings
        )

        # ---- 9. HITL check -----------------------------------------------
        hitl_result: Optional[HITLResult] = None
        if confidence < settings.human_escalation_threshold:
            hitl_result = self._hitl.escalate_to_expert(
                query=clean_input,
                context=answer,
                session_id=session_id,
            )

        execution_ms = round((time.monotonic() - start_time) * 1000, 1)
        audit_log(
            "QUERY_COMPLETE",
            session=session_id,
            intent=intent,
            confidence=confidence,
            exec_ms=execution_ms,
        )

        return AgentResponse(
            success=True,
            answer=answer,
            session_id=session_id,
            intent=intent,
            tool_data=tool_data,
            confidence=confidence,
            warnings=all_warnings,
            explanation=ethics_result.explanation,
            audit_id=ethics_result.audit_id,
            requires_human_approval=hitl_result is not None and hitl_result.requires_human,
            pending_approval_id=hitl_result.request_id if hitl_result else None,
            execution_time_ms=execution_ms,
            demo_mode=rag_result.get("demo_mode", False),
        )

    def get_pending_approvals(self) -> List:
        """Return pending HITL approval requests (for admin UI)."""
        return self._hitl.get_pending_requests()

    def approve_action(self, request_id: str, approved: bool, reviewer: str = "admin") -> HITLResult:
        """Process a human approval decision."""
        return self._hitl.process_approval(request_id, approved, reviewer)

    def get_audit_log(self, session_id: Optional[str] = None):
        """Return audit trail entries."""
        return self._ethics_g.get_audit_log(session_id=session_id, limit=50)

    def reset_session(self, session_id: str) -> None:
        """Reset all per-session counters."""
        self._ops_g.reset_session_iterations(session_id)
        self._tool_g.reset_session(session_id)
        self._multiagent_g.reset_session(session_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _classify_intent(self, text: str) -> str:
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for intent, config in INTENT_MAP.items():
            score = sum(1 for kw in config["keywords"] if kw in text_lower)
            if score > 0:
                scores[intent] = score
        if not scores:
            return "general"
        return max(scores, key=scores.get)

    def _build_tool_params(
        self,
        intent: str,
        query: str,
        location: Optional[str],
        lat: Optional[float],
        lon: Optional[float],
    ) -> Dict[str, Any]:
        if intent == "weather":
            return {"location": location or "Mumbai"}
        if intent == "market_price":
            commodity = self._extract_commodity(query)
            return {"commodity": commodity, "state": location or "Maharashtra"}
        if intent == "soil":
            return {
                "latitude": lat or 19.07,
                "longitude": lon or 72.88,
            }
        if intent == "crop":
            return {
                "soil_type": "Alluvial",
                "season": self._detect_season(),
                "state": location or "Maharashtra",
            }
        if intent == "pest":
            crop, symptoms = self._extract_pest_params(query)
            return {"crop": crop, "symptoms": symptoms}
        if intent == "scheme":
            return {"state": location or "Maharashtra", "keyword": self._extract_scheme_keyword(query)}
        return {}

    def _extract_commodity(self, text: str) -> str:
        commodities = [
            "tomato", "onion", "potato", "wheat", "rice", "maize",
            "cotton", "soybean", "groundnut", "sugarcane", "banana",
            "mango", "chilli", "garlic", "mustard",
        ]
        text_lower = text.lower()
        for c in commodities:
            if c in text_lower:
                return c
        return "wheat"  # default

    def _extract_pest_params(self, text: str) -> tuple:
        crops = ["rice", "wheat", "cotton", "tomato", "potato", "maize", "soybean"]
        crop = "rice"
        for c in crops:
            if c in text.lower():
                crop = c
                break
        return crop, text

    def _extract_scheme_keyword(self, text: str) -> str:
        keywords = ["insurance", "loan", "subsidy", "credit", "organic", "solar", "pump"]
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                return kw
        return ""

    def _detect_season(self) -> str:
        month = time.localtime().tm_mon
        if 6 <= month <= 9:
            return "kharif"
        if 10 <= month <= 12 or month <= 2:
            return "rabi"
        return "zaid"

    def _enhance_with_tool_data(
        self, rag_answer: str, tool_data: Optional[Dict], intent: str
    ) -> str:
        if not tool_data or "error" in tool_data:
            return rag_answer

        tool_summary = ""
        if intent == "weather" and "temperature_c" in tool_data:
            tool_summary = (
                f"\n\n📡 **Live Weather Update ({tool_data.get('location', '')}):**\n"
                f"🌡️ Temperature: {tool_data['temperature_c']}°C "
                f"(Feels like {tool_data.get('feels_like_c', '-')}°C)\n"
                f"💧 Humidity: {tool_data.get('humidity_pct', '-')}%\n"
                f"🌤️ Condition: {tool_data.get('description', '-')}\n"
                f"💨 Wind: {tool_data.get('wind_speed_kmh', '-')} km/h\n"
                f"🌾 {tool_data.get('farming_advisory', '')}"
            )
        elif intent == "market_price" and "modal_price_inr" in tool_data:
            tool_summary = (
                f"\n\n📊 **Live Market Price ({tool_data.get('commodity', '')}):**\n"
                f"💰 Modal Price: ₹{tool_data['modal_price_inr']}/quintal\n"
                f"📉 Min: ₹{tool_data.get('min_price_inr', '-')} | "
                f"📈 Max: ₹{tool_data.get('max_price_inr', '-')}\n"
                + (f"🏛️ MSP: ₹{tool_data['msp_if_applicable']}/quintal"
                   if tool_data.get('msp_if_applicable') else "")
            )
        elif intent == "crop" and "recommended_crops" in tool_data:
            crops = ", ".join(tool_data["recommended_crops"])
            tool_summary = (
                f"\n\n🌱 **Crop Recommendations for {tool_data.get('season', '').title()}:**\n"
                f"Best crops: **{crops}**\n"
                f"📅 Sowing window: {tool_data.get('sowing_window', '-')}"
            )
        elif intent == "pest" and "diagnosis" in tool_data:
            diag = tool_data["diagnosis"]
            tool_summary = (
                f"\n\n🔬 **Pest/Disease Diagnosis:**\n"
                f"🦠 Issue: **{diag.get('pest', 'Unknown')}**\n"
                f"⚠️ Severity: {diag.get('severity', '-')}\n"
                f"💊 Chemical control: {diag.get('chemical', '-')}\n"
                f"🌿 Biological control: {diag.get('biological', '-')}"
            )
        elif intent == "scheme" and "schemes" in tool_data:
            scheme_list = "\n".join(
                f"• **{s['name']}**: {s['benefit']}"
                for s in tool_data["schemes"][:3]
            )
            tool_summary = f"\n\n🏛️ **Government Schemes:**\n{scheme_list}"

        return rag_answer + tool_summary if tool_summary else rag_answer

    def _blocked_response(
        self, reason: str, session_id: str, start_time: float
    ) -> AgentResponse:
        exec_ms = round((time.monotonic() - start_time) * 1000, 1)
        return AgentResponse(
            success=False,
            answer=(
                f"⚠️ Your request could not be processed.\n\n"
                f"**Reason:** {reason}\n\n"
                "Please ask a legitimate farming-related question."
            ),
            session_id=session_id,
            execution_time_ms=exec_ms,
            blocked_reason=reason,
        )
