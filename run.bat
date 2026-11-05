@echo off
echo ============================================================
echo   KisanMitra - Smart Farming AI Agent
echo   Powered by IBM Granite + RAG + 9-Layer Guardrails
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Install requirements if needed
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing requirements...
pip install -r requirements.txt --quiet

REM Copy env.example to .env if .env does not exist
if not exist ".env" (
    echo Copying env.example to .env ...
    copy env.example .env
    echo IMPORTANT: Edit .env and add your IBM API Key before running!
)

echo.
echo Starting Streamlit application...
echo Open your browser at: http://localhost:8501
echo.
streamlit run app.py --server.port 8501 --server.address localhost

pause
