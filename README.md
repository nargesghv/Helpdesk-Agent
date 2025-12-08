# AI-Powered Helpdesk Agent

Intelligent ticket resolution assistant using LangGraph, FAISS, BM25, and Meta Llama 3.1.

## Features

- 🔍 Hybrid retrieval (semantic + keyword search)
- 🤖 AI-powered troubleshooting suggestions
- 📊 Multi-agent architecture with LangGraph
- ⚡ Fast processing (3-5 seconds per ticket)
- 🎯 Evidence-based recommendations with citations

## Tech Stack

- **Backend**: FastAPI, Python 3.10
- **AI/ML**: Meta Llama 3.1 via Groq, Sentence Transformers, FAISS, BM25
- **Orchestration**: LangGraph

## Quick Start

### Prerequisites

- Python 3.10+
- Groq API key (get free at https://console.groq.com/keys)

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/helpdesk-agent.git
cd helpdesk-agent

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export GROQ_API_KEY="your_key_here"  # Linux/Mac
$env:GROQ_API_KEY="your_key_here"    # Windows PowerShell

# Run server
python backend.py
```

### Usage

1. Open http://localhost:8004/docs for interactive API
2. Upload historical tickets via `/upload-file`
3. Submit new ticket via `/submit-ticket`
4. Process tickets via `/process-tickets`
5. Get AI-generated troubleshooting steps!

## API Endpoints

- `GET /health` - Health check
- `POST /upload-file` - Upload historical tickets
- `POST /submit-ticket` - Submit new ticket
- `POST /process-tickets` - Process all pending tickets
- `GET /docs` - Interactive API documentation

## Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

See `render.yaml` for deployment configuration.

## Project Structure

```
├── backend.py           # FastAPI server
├── ticket_graph.py      # LangGraph multi-agent pipeline
├── requirements.txt     # Python dependencies
├── render.yaml         # Render deployment config
├── frontend.html       # Web UI (optional)
└── Data/              # Historical tickets (CSV/Excel/JSON)
```

## License

MIT

## Author

Built as part of AI/ML coursework.
