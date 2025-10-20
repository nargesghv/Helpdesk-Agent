# Backend API Guide

## Overview

The Helpdesk Agent Backend exposes your ticket processing system as a REST API running on **port 8004**.

## Quick Start

```powershell
# Ensure GROQ_API_KEY is set
$env:GROQ_API_KEY = "your-key"

# Start the server
python backend.py
```

The server will start on `http://localhost:8004`

## Features

✅ **RESTful API** - Standard HTTP/JSON endpoints  
✅ **Auto-generated docs** - Interactive Swagger UI at `/docs`  
✅ **Single + Batch processing** - Process 1 or many tickets  
✅ **Health checks** - Monitor system status  
✅ **Statistics** - View KB metrics  
✅ **CORS enabled** - Call from web frontends  

---

## API Endpoints

### 1. Health Check
**GET** `/health`

Check system status and configuration.

**Response:**
```json
{
  "status": "healthy",
  "groq_api_configured": true,
  "artifacts_exist": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

---

### 2. Knowledge Base Stats
**GET** `/stats`

Get statistics about the knowledge base.

**Response:**
```json
{
  "kb_size": 29,
  "ready": true,
  "resolved_distribution": {
    "True": 17,
    "False": 12
  },
  "top_categories": {
    "Network": 8,
    "Email": 6,
    "Hardware": 5
  },
  "embed_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

---

### 3. Process Single Ticket
**POST** `/process/single`

Process one ticket and get troubleshooting guidance.

**Request Body:**
```json
{
  "issue": "VPN connection timeout",
  "description": "VPN disconnects after 5 minutes of usage",
  "category": "Network"
}
```

**Response:**
```json
{
  "ticket_id": 1,
  "issue": "VPN connection timeout",
  "direction_bullets": "- Update VPN client to latest version [#TCKT-1011]\n- Check VPN configuration settings [#TCKT-1011]\n- Verify firewall isn't blocking VPN ports [#TCKT-1022]\n- Caveat: Ensure admin rights before modifying settings",
  "evidence": [
    {
      "ticket_id": "TCKT-1011",
      "Issue": "VPN timeout",
      "Category": "Network",
      "Resolution": "Updated VPN settings",
      "Resolved": true,
      "score": 0.89
    }
  ],
  "logs": [
    "DataProcessingAgent: loading from Data",
    "RetrieverAgent: 5 candidates",
    "CuratorAgent: prepared 5 evidence rows",
    "PlannerAgent: plan generated via Groq"
  ],
  "processed_at": "2024-01-15T10:35:00"
}
```

---

### 4. Process Batch
**POST** `/process/batch`

Process multiple tickets at once.

**Request Body:**
```json
{
  "tickets": [
    {
      "issue": "Email not syncing",
      "description": "Outlook not working on mobile",
      "category": "Email"
    },
    {
      "issue": "Printer offline",
      "description": "Cannot connect to printer",
      "category": "Hardware"
    }
  ]
}
```

**Response:**
```json
[
  {
    "ticket_id": 1,
    "issue": "Email not syncing",
    "direction_bullets": "...",
    "evidence": [...],
    "logs": [...],
    "processed_at": "2024-01-15T10:40:00"
  },
  {
    "ticket_id": 2,
    "issue": "Printer offline",
    "direction_bullets": "...",
    "evidence": [...],
    "logs": [...],
    "processed_at": "2024-01-15T10:40:05"
  }
]
```

---

### 5. Rebuild Indexes
**POST** `/rebuild`

Force rebuild of knowledge base and indexes (use after adding new data to `Data/`).

**Response:**
```json
{
  "status": "success",
  "message": "Indexes rebuilt successfully",
  "timestamp": "2024-01-15T10:45:00"
}
```

---

## Testing

### Using cURL

```bash
# Health check
curl http://localhost:8004/health

# Process ticket
curl -X POST http://localhost:8004/process/single \
  -H "Content-Type: application/json" \
  -d '{"issue":"VPN timeout","description":"VPN disconnects","category":"Network"}'
```

### Using Python

```python
import requests

# Single ticket
response = requests.post(
    "http://localhost:8004/process/single",
    json={
        "issue": "VPN connection timeout",
        "description": "VPN disconnects after 5 minutes",
        "category": "Network"
    }
)

result = response.json()
print(result['direction_bullets'])
```

### Using the Test Client

```powershell
python test_api.py
```

---

## Interactive Documentation

The API includes auto-generated interactive documentation:

### Swagger UI
**http://localhost:8004/docs**
- Try endpoints directly in browser
- See request/response schemas
- Test authentication

### ReDoc
**http://localhost:8004/redoc**
- Alternative documentation UI
- Better for reading/printing
- Cleaner layout

---

## Configuration

The backend reads configuration from:

- **Environment Variables:**
  - `GROQ_API_KEY` - Required for LLM generation

- **Hardcoded Defaults:**
  - `alpha=0.6` - Semantic vs keyword weight
  - `k=5` - Top-k retrieval
  - `groq_model=llama-3.1-8b-instant` - Groq model
  - `deterministic=False` - Enable sampling

To change these, modify `backend.py` in the `initialize_graph()` function.

---

## Architecture

```
┌─────────────────────┐
│   HTTP Request      │
│   (port 8004)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   FastAPI App       │
│   • Routes          │
│   • Validation      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   LangGraph         │
│   • 5 agents        │
│   • State mgmt      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Response          │
│   (JSON)            │
└─────────────────────┘
```

---

## Performance

- **First request:** ~3-5 seconds (loads KB + builds indexes)
- **Subsequent requests:** ~0.5-1 second (cache hit)
- **Groq API latency:** ~0.3-0.5 seconds per ticket
- **Batch processing:** Parallel-friendly (consider async if needed)

---

## Error Handling

The API returns standard HTTP status codes:

- **200** - Success
- **422** - Validation error (bad request body)
- **500** - Server error (processing failed)

Error response example:
```json
{
  "detail": "Processing error: GROQ_API_KEY not set"
}
```

---

## Security Notes

⚠️ **Current State:** Development mode (CORS open, no auth)

**For Production:**
1. Add authentication (API keys, OAuth)
2. Restrict CORS origins
3. Add rate limiting
4. Use HTTPS
5. Implement input sanitization
6. Add request logging/monitoring

---

## Troubleshooting

### Server won't start
- Check: Is port 8004 already in use?
- Solution: Kill existing process or change port

### 500 errors on requests
- Check: Is `GROQ_API_KEY` set?
- Check: Is `Data/` folder present with tickets?
- Check logs in console

### Slow first request
- Normal: First request loads KB and builds indexes
- Solution: Use `/rebuild` endpoint to pre-warm

### Different results each run
- Cause: `deterministic=False` (default)
- Solution: Set `deterministic=True` in `initialize_graph()`

---

## Deployment

### Local Production

```powershell
# Use Gunicorn (Linux/Mac)
gunicorn backend:app --workers 4 --bind 0.0.0.0:8004

# Use Uvicorn (Windows)
uvicorn backend:app --host 0.0.0.0 --port 8004 --workers 4
```

### Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8004
CMD ["python", "backend.py"]
```

Build and run:
```bash
docker build -t helpdesk-agent .
docker run -p 8004:8004 -e GROQ_API_KEY=your-key helpdesk-agent
```

---

## Next Steps

1. ✅ Test endpoints with `test_api.py`
2. ✅ Explore interactive docs at `/docs`
3. ✅ Integrate with your frontend/workflow
4. ✅ Add monitoring and logging
5. ✅ Secure for production use

---

**Need help?** Check `README.md` or `document.md` for full system documentation.

