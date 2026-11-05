#!/bin/bash
# KisanMitra - Smart Farming AI Agent Launcher
# Powered by IBM Granite + RAG + 9-Layer Guardrails

echo "============================================================"
echo "  KisanMitra - Smart Farming AI Agent"
echo "  Powered by IBM Granite + RAG + 9-Layer Guardrails"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    exit 1
fi

# Create virtualenv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt --quiet

# Copy env.example to .env if .env does not exist
if [ ! -f ".env" ]; then
    echo "Copying env.example to .env ..."
    cp env.example .env
    echo "IMPORTANT: Edit .env and add your IBM API Key before running!"
fi

echo ""
echo "Starting Streamlit application..."
echo "Open your browser at: http://localhost:8501"
echo ""
streamlit run app.py --server.port 8501 --server.address localhost
