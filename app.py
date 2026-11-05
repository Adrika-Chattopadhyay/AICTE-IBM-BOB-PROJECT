"""
Smart Farming AI Agent — Streamlit Application
IBM Granite powered | RAG | 9-Layer Guardrails

Run: streamlit run app.py
"""
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# Load .env before importing project modules
from dotenv import load_dotenv
load_dotenv()

from smart_farming_agent.agents.orchestrator import OrchestratorAgent, AgentResponse
from smart_farming_agent.guardrails.hitl_guardrails import ApprovalStatus
from smart_farming_agent.config.settings import get_settings

settings = get_settings()

# ===========================================================================
# Page configuration (must be first Streamlit call)
# ===========================================================================
st.set_page_config(
    page_title="KisanMitra — Smart Farming AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://kvk.icar.gov.in",
        "Report a bug": None,
        "About": "KisanMitra — AI-powered smart farming advisor powered by IBM Granite.",
    },
)

# ===========================================================================
# Custom CSS
# ===========================================================================
st.markdown(
    """
<style>
/* ---- Main layout ---- */
.main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px; }

/* ---- Chat bubbles ---- */
.chat-user {
    background: #e8f5e9; border-radius: 18px 18px 4px 18px;
    padding: 12px 16px; margin: 8px 0; margin-left: 15%;
    border-left: 4px solid #2e7d32; color: #1b5e20;
}
.chat-agent {
    background: #e3f2fd; border-radius: 18px 18px 18px 4px;
    padding: 12px 16px; margin: 8px 0; margin-right: 15%;
    border-left: 4px solid #1565c0; color: #0d47a1;
}
.chat-warning {
    background: #fff8e1; border-radius: 8px;
    padding: 8px 12px; margin: 4px 0;
    border-left: 3px solid #f9a825; font-size: 0.85em;
}
.chat-blocked {
    background: #ffebee; border-radius: 8px;
    padding: 10px 14px; margin: 8px 0;
    border-left: 4px solid #c62828; color: #b71c1c;
}

/* ---- Metric cards ---- */
.metric-card {
    background: #f8f9fa; border: 1px solid #dee2e6;
    border-radius: 8px; padding: 12px; text-align: center;
}

/* ---- Guardrail badges ---- */
.badge-green { background:#d4edda; color:#155724; padding:3px 8px;
               border-radius:12px; font-size:0.8em; font-weight:600; }
.badge-yellow { background:#fff3cd; color:#856404; padding:3px 8px;
                border-radius:12px; font-size:0.8em; font-weight:600; }
.badge-red { background:#f8d7da; color:#721c24; padding:3px 8px;
             border-radius:12px; font-size:0.8em; font-weight:600; }

/* ---- Sidebar ---- */
.sidebar-section { background:#f0f4f8; border-radius:8px; padding:10px; margin-bottom:10px; }

/* ---- Hero banner ---- */
.hero-banner {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
    color: white; padding: 20px 24px; border-radius: 12px;
    margin-bottom: 16px;
}
.hero-banner h1 { margin:0; font-size:1.8em; }
.hero-banner p { margin:4px 0 0; opacity:0.9; font-size:0.95em; }

/* ---- Demo mode banner ---- */
.demo-banner {
    background: #fff3cd; border: 1px solid #ffc107;
    border-radius: 8px; padding: 8px 14px; margin-bottom: 12px;
    font-size: 0.88em;
}
</style>
""",
    unsafe_allow_html=True,
)

# ===========================================================================
# Session state initialization
# ===========================================================================
def init_session():
    defaults = {
        "session_id": str(uuid.uuid4())[:8],
        "messages": [],
        "agent": None,
        "query_count": 0,
        "pending_approvals": [],
        "user_role": "farmer",
        "language": "en",
        "location": "",
        "latitude": None,
        "longitude": None,
        "guardrail_stats": {
            "total": 0, "blocked": 0, "warnings": 0, "escalated": 0
        },
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state["agent"] is None:
        with st.spinner("🔄 Initializing AI agent..."):
            st.session_state["agent"] = OrchestratorAgent()


# ===========================================================================
# Sidebar
# ===========================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌾 KisanMitra Settings")

        # --- User profile ---
        st.markdown("### 👤 User Profile")
        st.session_state["user_role"] = st.selectbox(
            "Role",
            ["farmer", "advisor", "admin"],
            index=["farmer", "advisor", "admin"].index(st.session_state["user_role"]),
            help="Your role determines data access permissions.",
        )
        st.session_state["language"] = st.selectbox(
            "Language",
            ["en", "hi", "mr", "ta", "te", "kn", "gu", "pa", "bn", "ml"],
            format_func=lambda x: {
                "en": "🇬🇧 English", "hi": "🇮🇳 Hindi", "mr": "🇮🇳 Marathi",
                "ta": "🇮🇳 Tamil", "te": "🇮🇳 Telugu", "kn": "🇮🇳 Kannada",
                "gu": "🇮🇳 Gujarati", "pa": "🇮🇳 Punjabi",
                "bn": "🇮🇳 Bengali", "ml": "🇮🇳 Malayalam",
            }[x],
        )

        # --- Location ---
        st.markdown("### 📍 Location")
        st.session_state["location"] = st.text_input(
            "Your location (city/district)",
            value=st.session_state["location"],
            placeholder="e.g. Nashik, Pune, Amritsar",
        )
        col1, col2 = st.columns(2)
        with col1:
            st.session_state["latitude"] = st.number_input(
                "Latitude", value=st.session_state["latitude"] or 19.07,
                min_value=8.0, max_value=37.0, step=0.01, format="%.4f"
            )
        with col2:
            st.session_state["longitude"] = st.number_input(
                "Longitude", value=st.session_state["longitude"] or 72.88,
                min_value=68.0, max_value=97.0, step=0.01, format="%.4f"
            )

        st.divider()

        # --- Quick questions ---
        st.markdown("### 💡 Quick Questions")
        quick_questions = [
            "🌤️ What is the weather today?",
            "🌱 What crop should I grow this season?",
            "💰 What is the mandi price for tomatoes?",
            "🐛 Rice leaves have brown spots. What to do?",
            "🏛️ Which government schemes are available for me?",
            "🌿 How to improve my soil health?",
        ]
        for q in quick_questions:
            if st.button(q, key=f"quick_{q[:20]}", use_container_width=True):
                st.session_state["quick_input"] = q[2:].strip()  # remove emoji
                st.rerun()

        st.divider()

        # --- Guardrail Stats ---
        st.markdown("### 🛡️ Guardrail Stats")
        stats = st.session_state["guardrail_stats"]
        c1, c2 = st.columns(2)
        c1.metric("Total", stats["total"])
        c2.metric("Blocked", stats["blocked"], delta=None)
        c1.metric("Warnings", stats["warnings"])
        c2.metric("Escalated", stats["escalated"])

        st.divider()

        # --- Session controls ---
        st.markdown("### ⚙️ Session")
        col_a, col_b = st.columns(2)
        if col_a.button("🗑️ Clear Chat", use_container_width=True):
            agent: OrchestratorAgent = st.session_state["agent"]
            agent.reset_session(st.session_state["session_id"])
            st.session_state["messages"] = []
            st.session_state["session_id"] = str(uuid.uuid4())[:8]
            st.session_state["guardrail_stats"] = {
                "total": 0, "blocked": 0, "warnings": 0, "escalated": 0
            }
            st.rerun()
        if col_b.button("🔄 New Session", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.markdown(
            f"<small>Session ID: <code>{st.session_state['session_id']}</code></small>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# Chat interface
# ===========================================================================
def render_chat():
    # Hero banner
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🌾 KisanMitra — Smart Farming AI Agent</h1>
            <p>Powered by IBM Granite | RAG | 9-Layer Guardrails | Ask in your language</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Demo mode notice
    agent: OrchestratorAgent = st.session_state["agent"]

    # Sample questions strip
    st.markdown("**Ask me about:** Weather | Crop Advice | Market Prices | Pest Control | Govt Schemes | Soil Health")

    # ── Chat history ─────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">👨‍🌾 <strong>You:</strong> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-agent">🌾 <strong>KisanMitra:</strong></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(msg["content"])

                # Warnings
                for w in msg.get("warnings", []):
                    st.markdown(
                        f'<div class="chat-warning">⚠️ {w}</div>',
                        unsafe_allow_html=True,
                    )

                # Meta row
                meta_cols = st.columns([2, 2, 2, 2, 2])
                meta_cols[0].caption(f"🎯 Intent: `{msg.get('intent', 'general')}`")
                conf = msg.get("confidence", 1.0)
                conf_label = "🟢 High" if conf > 0.75 else ("🟡 Medium" if conf > 0.50 else "🔴 Low")
                meta_cols[1].caption(f"Confidence: {conf_label} ({conf:.0%})")
                meta_cols[2].caption(f"⏱️ {msg.get('exec_ms', 0):.0f}ms")
                if msg.get("demo_mode"):
                    meta_cols[3].caption("🔵 Demo Mode")
                if msg.get("audit_id"):
                    meta_cols[4].caption(f"📋 Audit: `{msg['audit_id'][:6]}`")

                # Explanation (expandable)
                if msg.get("explanation"):
                    with st.expander("ℹ️ Explanation (Responsible AI)", expanded=False):
                        st.info(msg["explanation"])

                # Tool data (expandable)
                if msg.get("tool_data") and not msg["tool_data"].get("error"):
                    with st.expander("📊 Raw Tool Data", expanded=False):
                        st.json(msg["tool_data"])

                # Human escalation notice
                if msg.get("requires_human"):
                    st.warning(
                        f"🙋 **Human Expert Escalation** — Reference: `{msg.get('approval_id', '')}`\n\n"
                        "Your question has been escalated to a human agricultural expert. "
                        "You will receive a response within 24 hours."
                    )

    # ── Input box ─────────────────────────────────────────────────────────────
    st.divider()

    # Handle quick-question injection
    prefill = st.session_state.pop("quick_input", "")

    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([8, 1])
        with col_input:
            user_input = st.text_input(
                "Ask your farming question:",
                value=prefill,
                placeholder="e.g. What is the best crop for black soil in kharif season?",
                label_visibility="collapsed",
            )
        with col_send:
            submitted = st.form_submit_button("Send 🚀", use_container_width=True)

    if submitted and user_input.strip():
        _process_user_input(user_input.strip())
        st.rerun()


def _process_user_input(user_input: str):
    """Process user input through the full agent pipeline."""
    agent: OrchestratorAgent = st.session_state["agent"]
    stats = st.session_state["guardrail_stats"]
    stats["total"] += 1

    # Add user message to history
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input,
    })

    with st.spinner("🤔 KisanMitra is thinking..."):
        response: AgentResponse = agent.process_query(
            user_input=user_input,
            session_id=st.session_state["session_id"],
            user_role=st.session_state["user_role"],
            location=st.session_state["location"] or None,
            language=st.session_state["language"],
            latitude=st.session_state.get("latitude"),
            longitude=st.session_state.get("longitude"),
        )

    # Update stats
    if not response.success:
        stats["blocked"] += 1
    if response.warnings:
        stats["warnings"] += len(response.warnings)
    if response.requires_human_approval:
        stats["escalated"] += 1

    # Add agent message to history
    st.session_state["messages"].append({
        "role": "agent",
        "content": response.answer,
        "intent": response.intent,
        "confidence": response.confidence,
        "warnings": response.warnings,
        "explanation": response.explanation,
        "audit_id": response.audit_id,
        "tool_data": response.tool_data,
        "exec_ms": response.execution_time_ms,
        "demo_mode": response.demo_mode,
        "requires_human": response.requires_human_approval,
        "approval_id": response.pending_approval_id,
    })

    st.session_state["query_count"] += 1


# ===========================================================================
# Guardrails Dashboard Tab
# ===========================================================================
def render_guardrails_dashboard():
    st.markdown("## 🛡️ Guardrails Dashboard")
    st.markdown("All 9 guardrail layers protecting this application:")

    guardrail_info = [
        {
            "number": "1",
            "name": "Input Guardrails",
            "icon": "🔍",
            "checks": ["Input length validation", "Prompt injection detection",
                       "Malicious input detection", "PII detection & redaction",
                       "Inappropriate content filtering"],
            "status": "Active",
        },
        {
            "number": "2",
            "name": "Output Guardrails",
            "icon": "📤",
            "checks": ["Hallucination detection", "Factual accuracy validation",
                       "Toxicity & harmful content filtering",
                       "Sensitive information filtering",
                       "Response format validation (JSON/XML)"],
            "status": "Active",
        },
        {
            "number": "3",
            "name": "Tool Execution Guardrails",
            "icon": "🔧",
            "checks": ["Tool access permissions (RBAC)", "API authorization",
                       "Function-call parameter validation",
                       "Restricted tool deny-list",
                       "Session-level rate limiting"],
            "status": "Active",
        },
        {
            "number": "4",
            "name": "Data Guardrails",
            "icon": "🗄️",
            "checks": ["Data privacy protection", "Field masking & anonymization",
                       "Role-based data access (farmer/advisor/admin)",
                       "Database access restrictions",
                       "Bulk retrieval prevention"],
            "status": "Active",
        },
        {
            "number": "5",
            "name": "Security Guardrails",
            "icon": "🔒",
            "checks": ["Prompt injection protection", "Jailbreak detection",
                       "SQL injection prevention",
                       "Cross-agent HMAC signing",
                       "Credential & secret management"],
            "status": "Active",
        },
        {
            "number": "6",
            "name": "Operational Guardrails",
            "icon": "⚙️",
            "checks": ["Max execution time (60s)", "Token usage limits (2048/req)",
                       "API rate limiting (30/min)", "Max agent iterations (10)",
                       "Cost monitoring & budget alerts"],
            "status": "Active",
        },
        {
            "number": "7",
            "name": "Human-in-the-Loop Guardrails",
            "icon": "🙋",
            "checks": ["Human approval for financial actions",
                       "Approval for communication actions",
                       "Low-confidence escalation to experts",
                       "Approval for data modification/deletion",
                       "Timeout management for approvals"],
            "status": "Active",
        },
        {
            "number": "8",
            "name": "Multi-Agent Guardrails",
            "icon": "🤝",
            "checks": ["Agent-level RBAC", "Task assignment validation",
                       "Cross-agent output verification",
                       "Infinite loop detection",
                       "Max agent hop count (5)"],
            "status": "Active",
        },
        {
            "number": "9",
            "name": "Ethical & Compliance Guardrails",
            "icon": "⚖️",
            "checks": ["Bias detection (caste/gender/regional)",
                       "Fairness validation",
                       "Explainability for every response",
                       "GDPR/PDPB compliance checks",
                       "Immutable SQLite audit trail"],
            "status": "Active",
        },
    ]

    # 3-column layout
    for i in range(0, len(guardrail_info), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(guardrail_info):
                g = guardrail_info[i + j]
                with col:
                    with st.expander(
                        f"{g['icon']} **{g['number']}. {g['name']}**",
                        expanded=False,
                    ):
                        st.markdown(
                            f'<span class="badge-green">✅ {g["status"]}</span>',
                            unsafe_allow_html=True,
                        )
                        for check in g["checks"]:
                            st.markdown(f"• {check}")

    st.divider()
    st.markdown("### 📊 Session Statistics")
    stats = st.session_state["guardrail_stats"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Queries", st.session_state["query_count"])
    c2.metric("Total Guardrail Checks", stats["total"])
    c3.metric("Requests Blocked", stats["blocked"])
    c4.metric("Warnings Issued", stats["warnings"])
    c5.metric("Expert Escalations", stats["escalated"])


# ===========================================================================
# HITL Approvals Tab
# ===========================================================================
def render_hitl_panel():
    st.markdown("## 🙋 Human-in-the-Loop Approvals")
    agent: OrchestratorAgent = st.session_state["agent"]
    pending = agent.get_pending_approvals()

    if not pending:
        st.success("✅ No pending approvals at this time.")
        return

    for req in pending:
        with st.expander(
            f"🟡 Pending: [{req.request_id}] {req.action_category.value.title()} — {req.description[:60]}",
            expanded=True,
        ):
            st.json({
                "request_id": req.request_id,
                "category": req.action_category.value,
                "description": req.description,
                "requester": req.requester,
                "session_id": req.session_id,
            })
            c1, c2, c3 = st.columns([2, 1, 1])
            notes = c1.text_input("Reviewer notes:", key=f"notes_{req.request_id}")
            if c2.button("✅ Approve", key=f"approve_{req.request_id}", type="primary"):
                result = agent.approve_action(req.request_id, True, "admin", notes)
                st.success(f"Approved! {result.message}")
                st.rerun()
            if c3.button("❌ Reject", key=f"reject_{req.request_id}"):
                result = agent.approve_action(req.request_id, False, "admin", notes)
                st.error(f"Rejected. {result.message}")
                st.rerun()


# ===========================================================================
# Audit Trail Tab
# ===========================================================================
def render_audit_trail():
    st.markdown("## 📋 Audit Trail")
    st.markdown("Immutable log of all AI decisions and guardrail actions.")

    agent: OrchestratorAgent = st.session_state["agent"]
    show_session_only = st.checkbox(
        "Show only current session", value=True
    )
    session_filter = st.session_state["session_id"] if show_session_only else None
    entries = agent.get_audit_log(session_id=session_filter)

    if not entries:
        st.info("No audit entries yet. Start a conversation to generate audit logs.")
        return

    st.markdown(f"**{len(entries)} audit entries found**")
    for entry in entries:
        with st.expander(
            f"[{entry.get('timestamp', '')[:19]}] {entry.get('event_type', '')} — {entry.get('agent_name', '')}",
            expanded=False,
        ):
            cols = st.columns(4)
            cols[0].markdown(f"**Decision:** `{entry.get('decision', '-')}`")
            cols[1].markdown(f"**Confidence:** `{entry.get('confidence', 1.0):.0%}`")
            cols[2].markdown(f"**Role:** `{entry.get('user_role', '-')}`")
            cols[3].markdown(f"**Session:** `{entry.get('session_id', '-')}`")

            if entry.get("input_summary"):
                st.markdown(f"**Query:** {entry['input_summary']}")
            if entry.get("output_summary"):
                st.markdown(f"**Response:** {entry['output_summary']}")
            if entry.get("bias_flags") and entry["bias_flags"] != "[]":
                st.warning(f"Bias flags: {entry['bias_flags']}")


# ===========================================================================
# About Tab
# ===========================================================================
def render_about():
    st.markdown(
        """
## 🌾 About KisanMitra — Smart Farming AI Agent

### Challenge
Small-scale farmers face information asymmetry — they lack timely access to localized agricultural guidance 
on weather, soil, market prices, and pest control. This gap leads to suboptimal crop choices, yield loss, 
and reduced income.

### Solution
**KisanMitra** is an AI agent for smart farming advice, powered by **IBM Granite** (via watsonx.ai) and 
**RAG (Retrieval-Augmented Generation)**. It provides real-time, localized advice in multiple Indian languages.

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | IBM Granite 3.8B Instruct (watsonx.ai) |
| **RAG** | FAISS vector store + SentenceTransformers |
| **Framework** | LangChain + Python |
| **UI** | Streamlit |
| **Guardrails** | 9-layer custom guardrail system |
| **APIs** | OpenWeatherMap, Agmarknet, ISRIC SoilGrids |
| **Audit Trail** | SQLite |
| **Config** | Pydantic Settings + .env |

### IBM Cloud Services Used
- **IBM watsonx.ai** (Lite tier) — Granite LLM inference
- **IBM Granite 3.8B Instruct** — Primary language model
- **IBM Slate 125M English** — Embedding model

### Guardrails (9 Layers)
1. **Input** — Injection, PII, malicious content detection
2. **Output** — Hallucination, toxicity, format validation  
3. **Tool Execution** — RBAC, rate limiting, parameter validation
4. **Data** — Masking, role-based access, bulk retrieval prevention
5. **Security** — Jailbreak, SQL injection, HMAC signing
6. **Operational** — Token budget, execution timeout, iteration cap
7. **Human-in-the-Loop** — Approval workflows, expert escalation
8. **Multi-Agent** — Loop detection, task validation, hop limits
9. **Ethical** — Bias detection, fairness, audit trail, compliance

### Supported Languages
English, Hindi, Marathi, Tamil, Telugu, Kannada, Gujarati, Punjabi, Bengali, Malayalam

### Project Team
**AICTE IBM SkillsBuild AI Project 2024**
        """
    )


# ===========================================================================
# Main App
# ===========================================================================
def main():
    init_session()
    render_sidebar()

    # Top-level tabs
    tab_chat, tab_guardrails, tab_hitl, tab_audit, tab_about = st.tabs([
        "💬 Chat",
        "🛡️ Guardrails",
        "🙋 Approvals",
        "📋 Audit Trail",
        "ℹ️ About",
    ])

    with tab_chat:
        render_chat()

    with tab_guardrails:
        render_guardrails_dashboard()

    with tab_hitl:
        render_hitl_panel()

    with tab_audit:
        render_audit_trail()

    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
