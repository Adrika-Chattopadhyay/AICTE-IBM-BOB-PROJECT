# 🌾 KisanMitra — Smart Farming AI Agent

> **AI Agent for Smart Farming Advice** powered by **IBM Granite** (watsonx.ai) + **RAG** + **9-Layer Guardrails**  
> *AICTE IBM SkillsBuild AI Project 2024*

---

## 📋 Overview

KisanMitra bridges the agricultural knowledge gap for small-scale farmers by delivering **real-time, localized farming advice** through an AI agent. Farmers can ask questions in their local language and receive expert guidance on:

- 🌤️ **Weather** — Real-time forecasts and farming advisories
- 🌱 **Crop Advice** — Season & soil-specific recommendations
- 💰 **Market Prices** — Live mandi/APMC rates
- 🐛 **Pest Control** — Diagnosis and treatment plans
- 🏛️ **Govt Schemes** — Subsidies, insurance, loans
- 🌿 **Soil Health** — Fertilizer and amendment recommendations

---

## 🏗️ Architecture

```
User Query (Streamlit UI)
         │
         ▼
┌─────────────────────────────────────┐
│   9-LAYER GUARDRAILS PIPELINE       │
│  1. Security (Jailbreak/SQL inject) │
│  2. Input (PII/Injection/Length)    │
│  3. Operational (Tokens/Iterations) │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     ORCHESTRATOR AGENT              │
│  Intent Classification              │
│  Multi-Agent Routing                │
│  Tool Execution (with guardrails)   │
└─────────────────────────────────────┘
         │
    ┌────┴─────┐
    ▼          ▼
 Tools      RAG Pipeline
(Weather,  (FAISS + IBM Granite)
 Market,
 Soil,
 Crop,
 Pest,
 Schemes)
         │
         ▼
┌─────────────────────────────────────┐
│   OUTPUT GUARDRAILS                 │
│  4. Output (Hallucination/Toxicity) │
│  5. Data (Masking/RBAC)             │
│  6. HITL (Human Approval)           │
│  7. Multi-Agent (Loop Detection)    │
│  8. Ethical (Bias/Audit)            │
└─────────────────────────────────────┘
         │
         ▼
  Final Response (Streamlit UI)
```

---

## 🛡️ 9-Layer Guardrail System

| # | Guardrail | Key Checks |
|---|-----------|------------|
| 1 | **Input** | Prompt injection, PII detection, malicious content, length validation |
| 2 | **Output** | Hallucination detection, toxicity filtering, format validation |
| 3 | **Tool Execution** | RBAC, rate limiting (30/min), parameter validation |
| 4 | **Data** | Field masking, role-based access, bulk retrieval cap |
| 5 | **Security** | Jailbreak detection, SQL injection, HMAC signing |
| 6 | **Operational** | Token budget (2048/req), execution timeout (60s), iteration cap |
| 7 | **Human-in-the-Loop** | Expert escalation, approval workflows |
| 8 | **Multi-Agent** | Loop detection, hop limits (5), task validation |
| 9 | **Ethical** | Bias detection, fairness scoring, audit trail, GDPR |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- IBM Cloud Lite account (free): [cloud.ibm.com](https://cloud.ibm.com)
- OpenWeatherMap API key (free): [openweathermap.org](https://openweathermap.org/api)
- data.gov.in API key (free): [data.gov.in](https://data.gov.in)

### Installation

```bash
# 1. Clone / download the project
cd AICTE_IBM-BOB-PROJECT

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp env.example .env
# Edit .env with your credentials (see Configuration section below)

# 5. Run the app
streamlit run app.py
```

### Windows One-Click Launch
```
Double-click run.bat
```

---

## ⚙️ Configuration (`.env`)

Copy `env.example` to `.env` and fill in your credentials:

```env
# MANDATORY - IBM watsonx.ai
IBM_API_KEY=your_ibm_cloud_api_key_here
IBM_PROJECT_ID=your_watsonx_project_id_here
IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com

# OPTIONAL - External APIs (app works in demo mode without these)
OPENWEATHER_API_KEY=your_openweathermap_api_key_here
DATA_GOV_API_KEY=your_data_gov_in_api_key_here
```

> **Demo Mode**: Without API keys, the app runs in demo mode with realistic sample data. All guardrails still function.

---

## 📁 Project Structure

```
AICTE_IBM-BOB-PROJECT/
├── app.py                          # Streamlit application (main entry point)
├── env.example                     # Environment variables template
├── requirements.txt                # Python dependencies
├── run.bat                         # Windows launcher
├── run.sh                          # Linux/Mac launcher
├── README.md                       # This file
└── smart_farming_agent/
    ├── __init__.py
    ├── agents/
    │   ├── __init__.py
    │   └── orchestrator.py         # Main orchestrator agent
    ├── rag/
    │   ├── __init__.py
    │   ├── granite_client.py       # IBM Granite LLM wrapper
    │   └── rag_pipeline.py         # RAG pipeline (FAISS + Granite)
    ├── tools/
    │   ├── __init__.py
    │   └── farming_tools.py        # Weather, Market, Soil, Crop, Pest, Scheme tools
    ├── guardrails/
    │   ├── __init__.py
    │   ├── input_guardrails.py     # Guardrail 1: Input validation
    │   ├── output_guardrails.py    # Guardrail 2: Output validation
    │   ├── tool_guardrails.py      # Guardrail 3: Tool execution control
    │   ├── data_guardrails.py      # Guardrail 4: Data protection
    │   ├── security_guardrails.py  # Guardrail 5: Security threats
    │   ├── operational_guardrails.py # Guardrail 6: Resource control
    │   ├── hitl_guardrails.py      # Guardrail 7: Human-in-the-Loop
    │   ├── multiagent_guardrails.py # Guardrail 8: Multi-agent safety
    │   └── ethical_guardrails.py   # Guardrail 9: Ethics & compliance
    ├── config/
    │   ├── __init__.py
    │   └── settings.py             # Centralized configuration
    ├── utils/
    │   ├── __init__.py
    │   └── logger.py               # Structured logging
    └── data/
        └── knowledge_base/
            └── farming_knowledge.txt  # Agricultural knowledge base
```

---

## 🌐 Supported Languages

English, Hindi (हिंदी), Marathi (मराठी), Tamil (தமிழ்), Telugu (తెలుగు),  
Kannada (ಕನ್ನಡ), Gujarati (ગુજરાતી), Punjabi (ਪੰਜਾਬੀ), Bengali (বাংলা), Malayalam (മലയാളം)

---

## 📊 Streamlit App Tabs

| Tab | Description |
|-----|-------------|
| 💬 Chat | Main conversational interface |
| 🛡️ Guardrails | Dashboard showing all 9 guardrail layers and stats |
| 🙋 Approvals | Human-in-the-Loop approval queue (admin) |
| 📋 Audit Trail | Immutable log of all AI decisions |
| ℹ️ About | Project overview and technology stack |

---

## 🏛️ IBM Technology Used

- **IBM watsonx.ai** (Lite tier) — Foundation model inference
- **IBM Granite 3.8B Instruct** — Primary conversational model
- **IBM Slate 125M English Retriever** — Embedding model for RAG

---

## 📜 License

MIT License — Free to use for educational and non-commercial purposes.

---

*Made with ❤️ for Indian farmers | AICTE IBM SkillsBuild AI Project 2024*
