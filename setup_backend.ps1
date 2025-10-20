# Setup and Run Backend Server
# This script activates virtual environment and runs the backend

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Helpdesk Agent - Backend Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".\agent\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv agent" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\agent\Scripts\Activate.ps1"

# Check if uvicorn is installed
Write-Host "Checking dependencies..." -ForegroundColor Green
$uvicornInstalled = & python -c "import uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing missing dependencies..." -ForegroundColor Yellow
    pip install fastapi uvicorn[standard] python-multipart
}

# Check Groq API key
if (-not $env:GROQ_API_KEY) {
    Write-Host ""
    Write-Host "WARNING: GROQ_API_KEY not set!" -ForegroundColor Yellow
    Write-Host "Set it with: `$env:GROQ_API_KEY = 'your-key'" -ForegroundColor Yellow
    Write-Host "Get free key: https://console.groq.com/keys" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Enter to continue anyway, or Ctrl+C to exit..."
    Read-Host
}

# Run backend
Write-Host ""
Write-Host "Starting Helpdesk Agent Backend..." -ForegroundColor Green
Write-Host "Server will run on: http://localhost:8004" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python -m uvicorn backend:app --host 0.0.0.0 --port 8004 --reload

