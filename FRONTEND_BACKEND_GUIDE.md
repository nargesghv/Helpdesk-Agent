# 🚀 Frontend + Backend Setup Guide

## Overview

Your helpdesk agent now has a **complete web interface**:
- ✅ Beautiful frontend UI
- ✅ REST API backend (port 8004)
- ✅ File upload support
- ✅ Real-time ticket processing
- ✅ Results visualization

---

## Quick Start

### 1. Set API Key (PowerShell)
```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
```

### 2. Run the Server

**Option A: Using run script (PowerShell)**
```powershell
.\run.ps1
```

**Option B: Using run script (Bash/Linux/Mac)**
```bash
chmod +x run.sh
./run.sh
```

**Option C: Direct command**
```powershell
uvicorn backend:app --host 0.0.0.0 --port 8004 --reload
```

### 3. Open Browser
Navigate to: **http://localhost:8004**

---

## Features

### 📝 Submit Tickets
- Fill form with issue details
- Tickets auto-saved to `input/` folder
- Instant feedback

### 📁 Upload Files
- Drag & drop CSV/XLSX/JSON files
- Bulk ticket import
- View uploaded files

### 🚀 Process Tickets
- One-click processing
- Real-time progress
- Beautiful results display

### 📊 View Results
- See all evidence tickets
- Resolution status (True/False)
- Troubleshooting steps with citations

---

## File Structure

```
helpdesk agent/
├── backend.py              # FastAPI server (NEW)
├── frontend.html           # Web UI (NEW)
├── run.sh                  # Bash run script (NEW)
├── run.ps1                 # PowerShell run script (NEW)
├── ticket_graph.py         # Core logic
├── requirements.txt        # Dependencies
├── Data/                   # Historical tickets (KB)
├── input/                  # New tickets (created auto)
├── output/                 # Results (created auto)
└── artifacts/              # Cached indexes
```

---

## API Endpoints

The backend exposes these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/health` | GET | Health check |
| `/submit-ticket` | POST | Submit one ticket |
| `/upload-file` | POST | Upload CSV/XLSX/JSON |
| `/process-tickets` | POST | Process all tickets |
| `/list-input` | GET | List input files |
| `/clear-input` | DELETE | Clear input folder |
| `/list-results` | GET | List result files |

---

## How It Works

### Workflow

```
┌─────────────────────────────────────────┐
│  1. USER SUBMITS TICKET                 │
│     Via form or file upload             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  2. SAVED TO input/ FOLDER              │
│     As JSON or uploaded file            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  3. USER CLICKS "PROCESS ALL TICKETS"   │
│     Triggers backend processing         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  4. BACKEND PROCESSES                   │
│     • Load tickets from input/          │
│     • Run through LangGraph pipeline    │
│     • Generate solutions via Groq       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  5. RESULTS DISPLAYED                   │
│     • Troubleshooting steps             │
│     • Evidence tickets                  │
│     • Resolved status                   │
└─────────────────────────────────────────┘
```

---

## Usage Examples

### Submit Single Ticket

1. Open http://localhost:8004
2. Fill in the form:
   - **Issue**: "VPN connection timeout"
   - **Description**: "VPN disconnects after 5 minutes"
   - **Category**: "Network"
3. Click "Submit Ticket"
4. ✅ Ticket saved to `input/` folder

### Upload Bulk Tickets

1. Prepare a CSV file:
   ```csv
   Issue,Description,Category
   VPN timeout,VPN disconnects,Network
   Email not syncing,Outlook not working,Email
   Printer offline,Cannot connect,Hardware
   ```
2. Click "Upload Ticket File"
3. Select your CSV
4. ✅ All tickets imported

### Process Tickets

1. Click "🚀 Process All Tickets"
2. Wait for processing (shows spinner)
3. View results:
   - Solution bullets with citations
   - Evidence from similar tickets
   - Resolved status for each

---

## Frontend Features

### Beautiful UI
- Modern gradient design
- Responsive layout (mobile-friendly)
- Smooth animations
- Clear status messages

### Real-time Feedback
- Upload progress
- Processing status
- Success/error messages
- File list updates

### Results Display
- Formatted solution bullets
- Evidence tickets with details
- Resolved status badges
- Easy-to-read layout

---

## Configuration

### Change Port

Edit `backend.py` (last line):
```python
uvicorn.run(app, host="0.0.0.0", port=8004)  # Change 8004
```

Or run with custom port:
```bash
uvicorn backend:app --port 9000
```

### Customize Frontend

Edit `frontend.html`:
- Colors in `<style>` section
- Layout in `<body>` section
- API URL in `<script>` section

### API Configuration

Edit `initialize_graph()` in `backend.py`:
```python
graph = build_graph(
    alpha=0.6,              # Semantic vs keyword weight
    k=5,                    # Top-k retrieval
    groq_model="llama-3.1-8b-instant",
    deterministic=False,    # Enable sampling
    force_rebuild=False     # Use cache
)
```

---

## Troubleshooting

### ❌ "Cannot connect to API"

**Problem**: Frontend can't reach backend

**Solutions**:
1. Check backend is running: `http://localhost:8004/health`
2. Check firewall isn't blocking port 8004
3. Try: `--host 127.0.0.1` instead of `0.0.0.0`

### ❌ "GROQ_API_KEY not set"

**Problem**: Missing API key

**Solution**:
```powershell
$env:GROQ_API_KEY = "your-key-here"
```

### ❌ "No tickets found in input folder"

**Problem**: Input folder is empty

**Solutions**:
1. Submit a ticket via form
2. Upload a file
3. Manually add files to `input/` folder

### ❌ "Processing failed"

**Problem**: Error during ticket processing

**Solutions**:
1. Check backend console for errors
2. Verify Data/ folder has historical tickets
3. Try rebuilding: Delete `artifacts/` folder

### ❌ Slow first request

**Problem**: Initial load takes time

**Explanation**: First request builds indexes (normal)

**Solution**: Be patient (~5-10 seconds first time)

---

## Advanced Usage

### Run in Production

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn backend:app --workers 4 --bind 0.0.0.0:8004
```

### Add HTTPS

```bash
# Generate self-signed cert
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Run with SSL
uvicorn backend:app --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8004
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8004"]
```

Build and run:
```bash
docker build -t helpdesk-agent .
docker run -p 8004:8004 -e GROQ_API_KEY=your-key helpdesk-agent
```

---

## Development

### Hot Reload

The `--reload` flag enables auto-reload on code changes:
```bash
uvicorn backend:app --reload
```

### Debug Mode

Add print statements in `backend.py`:
```python
print(f"Processing ticket: {ticket}")
```

View in console output.

### Test API

Use the test client:
```bash
python test_api.py
```

Or use curl:
```bash
curl http://localhost:8004/health
```

---

## Security Notes

⚠️ **Current Setup**: Development mode

**For Production**:
1. Add authentication (API keys, OAuth)
2. Restrict CORS origins
3. Use HTTPS
4. Add rate limiting
5. Sanitize inputs
6. Add logging/monitoring
7. Secure API key storage

---

## Next Steps

1. ✅ **Test the system**
   ```powershell
   .\run.ps1
   ```

2. ✅ **Submit test tickets**
   - Use the form
   - Upload a file

3. ✅ **Process tickets**
   - Click "Process All Tickets"
   - View results

4. ✅ **Integrate**
   - Connect your frontend
   - Add to workflow
   - Deploy to production

---

## Support

- **Backend API docs**: http://localhost:8004/docs
- **System architecture**: See `document.md`
- **General usage**: See `README.md`

---

## Summary

You now have:
- ✅ Beautiful web interface
- ✅ REST API backend
- ✅ File upload support
- ✅ Real-time processing
- ✅ Results visualization
- ✅ Production-ready code

**Your helpdesk agent is now a complete web application!** 🎉

---

*Last updated: 2025*

