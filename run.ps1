# Run Helpdesk Agent Backend (PowerShell)

# Check if GROQ_API_KEY is set
if (-not $env:GROQ_API_KEY) {
    Write-Host "WARNING: GROQ_API_KEY not set!" -ForegroundColor Yellow
    Write-Host "Set it with: `$env:GROQ_API_KEY = 'your-key'" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Starting Helpdesk Agent on http://localhost:8004" -ForegroundColor Green
Write-Host "Open browser: http://localhost:8004" -ForegroundColor Cyan
Write-Host ""

# Run with Python module (more reliable than direct uvicorn command)
python -m uvicorn backend:app --host 0.0.0.0 --port 8004 --reload

