# 🚀 Quick Start Guide

## Fixed! Ready to Run

✅ **Issue 1 FIXED**: PowerShell script encoding error  
✅ **Issue 2 FIXED**: Uvicorn installed successfully

---

## Run the Server (Choose ONE option)

### ✅ Option 1: PowerShell Script (RECOMMENDED)
```powershell
.\run.ps1
```

### ✅ Option 2: Python Module
```powershell
python -m uvicorn backend:app --host 0.0.0.0 --port 8004 --reload
```

### ✅ Option 3: Direct Python
```powershell
python backend.py
```

---

## After Server Starts

You'll see:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
✅ Graph initialized
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8004
```

Then open your browser to:
**http://localhost:8004**

---

## Troubleshooting

### If you see "GROQ_API_KEY not set"
```powershell
$env:GROQ_API_KEY = "your-groq-api-key-here"
```

### If port 8004 is already in use
```powershell
python -m uvicorn backend:app --host 0.0.0.0 --port 9000 --reload
```
(Then open http://localhost:9000)

---

## What to Do Next

1. ✅ **Submit a ticket** via the web form
2. ✅ **Upload a file** (CSV/XLSX/JSON)
3. ✅ **Click "Process All Tickets"**
4. ✅ **View results** instantly

---

**You're all set! Just run one of the commands above and open http://localhost:8004** 🎉

