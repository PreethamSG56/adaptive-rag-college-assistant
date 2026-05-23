@echo off
REM start.bat — Activate venv and launch the app (Windows CMD)
REM Usage: start.bat

SET SCRIPT_DIR=%~dp0
SET VENV_PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe
SET VENV_PIP=%SCRIPT_DIR%venv\Scripts\pip.exe
SET VENV_STREAMLIT=%SCRIPT_DIR%venv\Scripts\streamlit.exe

REM Create venv if missing
IF NOT EXIST "%VENV_PYTHON%" (
    echo Creating virtual environment...
    python -m venv "%SCRIPT_DIR%venv"
    echo Installing dependencies...
    "%VENV_PIP%" install -r "%SCRIPT_DIR%requirements.txt" -q
    echo Downloading spaCy language model...
    "%VENV_PYTHON%" -m spacy download en_core_web_sm
    echo Setup complete!
)

REM Activate venv
CALL "%SCRIPT_DIR%venv\Scripts\activate.bat"

REM Launch app
echo Starting College Notes RAG Assistant...
"%VENV_STREAMLIT%" run "%SCRIPT_DIR%app.py"
