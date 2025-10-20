# Helpdesk Ticket Assistant (LangGraph + FAISS + BM25 + Groq Llama 3.1)

A production-ready pipeline that builds a knowledge base from historical tickets, retrieves similar cases with hybrid search, and generates validated troubleshooting steps using Groq’s Llama 3.1.

## Requirements

- Python 3.10+
- Windows PowerShell or bash/zsh
- Optional: virtual environment
- Install dependencies: `pip install -r requirements.txt`

---

## 🏗️ ARCHITECTURE OVERVIEW

### System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: New Tickets                           │
│                    (input/new_tickets.csv)                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. DATA PROCESSING AGENT                                       │
│     • Load old tickets from Data/                               │
│     • Unify schema                                              │
│     • Clean & deduplicate                                       │
│     • Build indexes (FAISS + BM25)                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. RETRIEVER AGENT                                             │
│     • Hybrid search (60% semantic, 40% keyword)                 │
│     • Find top-k similar tickets                                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. CURATOR AGENT                                               │
│     • Format evidence                                           │
│     • Prepare structured data                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. PLANNER AGENT (LLM)                                         │
│     • Generate solution bullets                                 │
│     • Use Llama-3.1-8B-Instruct                                 │
│     • Cite evidence tickets                                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. VALIDATOR AGENT                                             │
│     • Ensure bullet format                                      │
│     • Deduplicate similar bullets                               │
│     • Merge citations                                           │
│     • Add caveat if missing                                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                 OUTPUT: Results                                 │
│     • JSON with direction + evidence                            │
│     • Markdown (human-readable)                                 │
│     • output/result_001.json, result_001.md                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🌐 Quick Start (Web Interface - RECOMMENDED)

### 1. Set API Key
```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
```

### 2. Run Backend Server
```powershell
# PowerShell
.\run.ps1

# Or Bash/Linux/Mac
./run.sh

# Or direct command
uvicorn backend:app --host 0.0.0.0 --port 8004 --reload
```

### 3. Open Browser
Navigate to: **http://localhost:8004**

You'll see a beautiful web interface where you can:
- ✅ Submit tickets via form
- ✅ Upload CSV/XLSX/JSON files
- ✅ Process all tickets with one click
- ✅ View results instantly

---

## 💻 CLI Mode (Alternative)

Run from command line:
```powershell
python ticket_graph.py --data_dir Data --input_dir input --out_dir output --rebuild
```

## What You Get
- `output/result_XXX.json` with bullets, evidence, logs
- `output/result_XXX.md` human-readable summary
- `artifacts/` cached KB and FAISS index for fast reruns

## How It Works
- DataProcessingAgent: loads `Data/`, standardizes schema, redacts PII, builds FAISS + BM25
- RetrieverAgent: hybrid search over all tickets (Resolved=True/False included)
- CuratorAgent: formats evidence with `Resolved` copied from the source
- PlannerAgent (Groq): generates 3–5 cited bullets; instructed to avoid failed steps
- ValidatorAgent: enforces format, merges duplicates, ensures single caveat

See `document.md` for full architecture and flowcharts.

## Installation (step-by-step)

1) Clone this repo
2) Create a virtual environment (optional)
   - PowerShell
     ```powershell
     python -m venv agent
     .\agent\Scripts\Activate.ps1
     ```
   - bash/zsh
     ```bash
     python -m venv agent
     source agent/bin/activate
     ```
3) Install packages
   ```powershell
   pip install -r requirements.txt
   ```
4) Prepare data
   - Put historical tickets in `Data/` (csv/xlsx/json)
   - Put new tickets in `input/` (csv/xlsx/json)
5) Run CLI
   ```powershell
   python ticket_graph.py --data_dir Data --input_dir input --out_dir output --rebuild
   ```

## 🌐 Backend API (Port 8004)

### Start the API Server
```powershell
python backend.py
```

The API runs on **http://localhost:8004** with:
- Interactive docs: http://localhost:8004/docs
- Alternative docs: http://localhost:8004/redoc

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + system status |
| `/stats` | GET | Knowledge base statistics |
| `/process/single` | POST | Process one ticket |
| `/process/batch` | POST | Process multiple tickets |
| `/rebuild` | POST | Force rebuild indexes |

### Example: Process Single Ticket

```python
import requests

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

### Example: Process Batch

```python
response = requests.post(
    "http://localhost:8004/process/batch",
    json={
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
)

results = response.json()
for r in results:
    print(f"Ticket {r['ticket_id']}: {r['issue']}")
```

### Test the API

Run the included test client:
```powershell
python test_api.py
```

## FAQ

- Q: I only see Resolved=True in evidence.
  - A: Delete `artifacts/` or run with `--rebuild`. Old cache may predate schema changes.
- Q: Excel file won’t load.
  - A: `pip install openpyxl` (already in requirements.txt). Ensure file isn’t locked by Excel.
- Q: How do I get repeatable results?
  - A: Add `--deterministic` to disable sampling.
- Q: Can I change the model?
  - A: Use `--groq_model llama-3.1-70b-versatile` (ensure your key has access).

## Configuration
- `--alpha` (default 0.6): semantic vs keyword weight
- `--k` (default 5): number of retrieved candidates
- `--groq_model` (default `llama-3.1-8b-instant`): Groq model id
- `--deterministic`: temperature=0.0/top_p=1.0
- `--rebuild`: ignore cache and rebuild indexes

## Troubleshooting
- Only Resolved=True in evidence → remove `artifacts/` or use `--rebuild`.
- Missing Excel support → `pip install openpyxl`.
- Groq errors → verify `$env:GROQ_API_KEY`, network, or try again later.

## Notes
- Retrieval includes both resolved and unresolved tickets; Resolved is surfaced to the LLM and outputs to flag unsuccessful attempts.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` for speed/quality balance.

## License
MIT (or adapt as needed)
