# start.ps1 — Activate venv and launch the app
# Usage: .\start.ps1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvActivate = Join-Path $scriptDir "venv\Scripts\Activate.ps1"
$venvPython   = Join-Path $scriptDir "venv\Scripts\python.exe"

# ── Create venv if it doesn't exist ────────────────────────────────────────
if (-not (Test-Path $venvActivate)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv "$scriptDir\venv"
    Write-Host "Installing dependencies..." -ForegroundColor Cyan
    & "$scriptDir\venv\Scripts\pip.exe" install -r "$scriptDir\requirements.txt" -q
    Write-Host "Downloading spaCy language model..." -ForegroundColor Cyan
    & $venvPython -m spacy download en_core_web_sm
    Write-Host "Setup complete!" -ForegroundColor Green
}

# ── Activate venv ──────────────────────────────────────────────────────────
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& $venvActivate

# ── Launch Streamlit ───────────────────────────────────────────────────────
Write-Host "Starting College Notes RAG Assistant..." -ForegroundColor Green
& "$scriptDir\venv\Scripts\streamlit.exe" run "$scriptDir\app.py"
