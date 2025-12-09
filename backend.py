"""
Helpdesk Agent Backend API
FastAPI server with file upload and ticket processing
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd

# Import the lightweight ticket processing logic
from ticket_graph_lite import build_graph, GraphState

# -------------------- Pydantic Models --------------------

class TicketInput(BaseModel):
    """Single ticket input"""
    issue: str = Field(..., description="Issue title")
    description: str = Field(..., description="Problem description")
    category: Optional[str] = Field(None, description="Category")

class ProcessResponse(BaseModel):
    """Processing response"""
    status: str
    message: str
    results_count: int
    results: List[Dict[str, Any]]

# -------------------- FastAPI App --------------------

app = FastAPI(
    title="Helpdesk Agent",
    description="AI-powered ticket resolution assistant",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------- Global State --------------------

graph = None
base_state: GraphState = {}

def initialize_graph():
    """Initialize the lightweight pipeline"""
    global graph, base_state
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set")
    
    graph = build_graph()
    
    base_state = {
        "data_dir": "Data",
        "kb_df": None,
        "raw_df": None,
        "faiss_index": None,
        "embeddings": None,
        "bm25": None,
        "bm25_tokens": None,
        "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "artifacts_dir": "./artifacts",
        "ready": False,
        "ticket": {"Issue": "", "Description": "", "Category": ""},
        "candidates": [],
        "evidence": [],
        "plan": "",
        "logs": []
    }
    
    print("✅ Graph initialized")

# -------------------- API Endpoints --------------------

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("Data", exist_ok=True)
    initialize_graph()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve frontend"""
    frontend_path = Path("frontend.html")
    if frontend_path.exists():
        return frontend_path.read_text(encoding='utf-8')
    return """
    <html>
        <body>
            <h1>Helpdesk Agent API</h1>
            <p>Frontend not found. API is running at /docs</p>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check"""
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    artifacts_exist = os.path.exists("./artifacts/kb.parquet")
    
    return {
        "status": "healthy" if groq_configured else "degraded",
        "groq_configured": groq_configured,
        "artifacts_exist": artifacts_exist,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a ticket file (CSV/XLSX/JSON) to input folder
    """
    try:
        # Validate file extension
        filename = file.filename
        if not filename.lower().endswith(('.csv', '.xlsx', '.json')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only CSV, XLSX, JSON allowed"
            )
        
        # Save to input folder
        file_path = os.path.join("input", filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "status": "success",
            "message": f"File '{filename}' uploaded successfully",
            "file_path": file_path
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit-ticket")
async def submit_ticket(ticket: TicketInput):
    """
    Submit a single ticket via form (saves to input folder as JSON)
    """
    try:
        # Create ticket data
        ticket_data = {
            "Issue": ticket.issue,
            "Description": ticket.description,
            "Category": ticket.category or ""
        }
        
        # Save to input folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ticket_{timestamp}.json"
        file_path = os.path.join("input", filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([ticket_data], f, indent=2)
        
        return {
            "status": "success",
            "message": "Ticket submitted successfully",
            "file": filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-tickets", response_model=ProcessResponse)
async def process_tickets():
    """
    Process all tickets in input folder and save results to output folder
    """
    global graph, base_state
    
    if graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")
    
    try:
        # Read tickets from input folder
        tickets = []
        input_files = list(Path("input").glob("*"))
        
        if not input_files:
            raise HTTPException(status_code=400, detail="No tickets found in input folder")
        
        # Load tickets from all files
        for file_path in input_files:
            ext = file_path.suffix.lower()
            try:
                if ext == '.csv':
                    df = pd.read_csv(file_path)
                elif ext == '.xlsx':
                    df = pd.read_excel(file_path)
                elif ext == '.json':
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data if isinstance(data, list) else [data])
                else:
                    continue
                
                # Extract tickets
                for _, row in df.iterrows():
                    tickets.append({
                        "Issue": str(row.get("Issue", row.get("issue", ""))),
                        "Description": str(row.get("Description", row.get("description", ""))),
                        "Category": str(row.get("Category", row.get("category", "")))
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not tickets:
            raise HTTPException(status_code=400, detail="No valid tickets found in files")
        
        # Process each ticket
        results = []
        for i, ticket in enumerate(tickets, start=1):
            try:
                # Update state
                state = base_state.copy()
                state["ticket"] = ticket
                state["logs"] = []
                
                # Process through graph
                result = graph["invoke"](state)
                
                # Persist artifacts for next ticket
                base_state.update({
                    "kb_df": result.get("kb_df"),
                    "raw_df": result.get("raw_df"),
                    "faiss_index": result.get("faiss_index"),
                    "embeddings": result.get("embeddings"),
                    "bm25": result.get("bm25"),
                    "bm25_tokens": result.get("bm25_tokens"),
                    "embed_model_name": result.get("embed_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                    "ready": True
                })
                
                # Format result
                evidence = [{
                    "ticket_id": e.get("ticket_id", ""),
                    "Issue": e.get("Issue", ""),
                    "Category": e.get("Category", ""),
                    "Resolution": e.get("Resolution", ""),
                    "Resolved": str(e.get("Resolved", "")),
                } for e in result.get("evidence", [])]
                
                ticket_result = {
                    "ticket_number": i,
                    "issue": ticket["Issue"],
                    "category": ticket["Category"],
                    "direction_bullets": result.get("plan", ""),
                    "evidence": evidence,
                    "logs": result.get("logs", [])
                }
                
                results.append(ticket_result)
                
                # Save individual result
                output_file = os.path.join("output", f"result_{i:03d}.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(ticket_result, f, indent=2)
                
            except Exception as e:
                results.append({
                    "ticket_number": i,
                    "issue": ticket.get("Issue", "Unknown"),
                    "category": ticket.get("Category", ""),
                    "direction_bullets": f"- Caveat: Processing failed - {str(e)}",
                    "evidence": [],
                    "logs": [f"ERROR: {str(e)}"]
                })
        
        return ProcessResponse(
            status="success",
            message=f"Processed {len(results)} ticket(s)",
            results_count=len(results),
            results=results
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.delete("/clear-input")
async def clear_input():
    """Clear all files from input folder"""
    try:
        for file in Path("input").glob("*"):
            file.unlink()
        return {"status": "success", "message": "Input folder cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-input")
async def list_input():
    """List files in input folder"""
    try:
        files = [f.name for f in Path("input").glob("*")]
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-results")
async def list_results():
    """List result files in output folder"""
    try:
        files = sorted([f.name for f in Path("output").glob("result_*.json")])
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Main --------------------

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  WARNING: GROQ_API_KEY not set!")
        print("Set it: $env:GROQ_API_KEY='your-key' (PowerShell)")
    
    print("🚀 Starting Helpdesk Agent on http://localhost:8004")
    print("📱 Open browser: http://localhost:8004")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        log_level="info"
    )
