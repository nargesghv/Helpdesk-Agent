# 📖 Step-by-Step Explanation: ticket_graph.py

**Complete walkthrough of your helpdesk RAG system**

---

## 🎯 **HIGH-LEVEL OVERVIEW**

Your system is a **multi-agent RAG (Retrieval-Augmented Generation) pipeline** that:
1. **Loads** old resolved tickets from `Data/` folder
2. **Indexes** them for fast search (FAISS + BM25)
3. **Retrieves** similar tickets when a new problem arrives
4. **Generates** solution steps using Groq's Llama 3.1 8B
5. **Validates** and formats the output

---

## 📂 **FILE STRUCTURE** (Lines 1-25)

### **Header Comments (Lines 1-11)**
```python
# ticket_graph.py
# Multi-agent LangGraph RAG for Helpdesk Ticket Assistance
# Nodes:
#  - DataProcessingAgent: load/clean/index old tickets (Data/)
#  - RetrieverAgent: hybrid search (FAISS + MiniLM + BM25)
#  - CuratorAgent: evidence shaping (shows Category)
#  - PlannerAgent (Groq): Llama 3.1 -> 3–5 bullets + citations
#  - ValidatorAgent: format, de-dup, single Caveat
```

**Purpose:** Documents the 5 agents in your pipeline

---

### **Imports (Lines 13-25)**
```python
import os, re, json, glob, time, argparse, traceback, unicodedata, hashlib, difflib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, TypedDict, Optional

import numpy as np
import pandas as pd

# Retrieval
import faiss                           # Vector similarity search
from sentence_transformers import SentenceTransformer  # Embeddings
from rank_bm25 import BM25Okapi       # Keyword search

# Orchestration
from langgraph.graph import StateGraph, END
```

**Key Libraries:**
- **faiss**: Facebook's library for fast vector search
- **sentence_transformers**: Creates semantic embeddings
- **rank_bm25**: Traditional keyword search (TF-IDF based)
- **langgraph**: Orchestrates the multi-agent workflow

---

## 🛠️ **SECTION 1: UTILITIES** (Lines 27-120)

### **1.1 Column Name Standardization Map (Lines 29-39)**

```python
_STD_MAP = {
    "ticket_id": ["ticket id", "id", "ticketid", "ticket-id"],
    "issue": ["issue", "title", "subject", "summary"],
    "description": ["description", "problem", "details", "body"],
    "resolution": ["resolution", "solution", "fix", "steps", "how resolved", "comment"],
    "category": ["category", "type", "queue", "group"],
    "date": ["date", "created", "created at", "opened"],
    "agent name": ["agent name", "agent", "assignee", "owner"],
    "resolved": ["resolved", "is resolved", "status", "closed"],
}
```

**Purpose:** Handles inconsistent column naming across different data sources
- Your CSV might have "Title", another has "Issue", another has "Subject"
- This maps all variants to canonical names

---

### **1.2 Column Standardization Function (Lines 41-88)**

```python
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names case-insensitively and map synonyms to canonical names."""
```

**What it does:**
1. **Maps lowercase → actual column names**
   ```python
   lower_to_actual = {c.lower().strip(): c for c in df.columns}
   # {"title": "Title", "description": "Description"}
   ```

2. **Finds matching columns** using the synonym map
   ```python
   for canon, alts in _STD_MAP.items():
       hit = find_any([a.lower() for a in alts])
       if hit and hit != canon:
           rename[hit] = canon
   # Renames "Title" → "issue", "Problem" → "description"
   ```

3. **Creates missing columns** with defaults
   ```python
   for c in ["issue", "description", "resolution", ...]:
       if c not in df.columns:
           df[c] = "" if c != "resolved" else True
   ```

4. **Creates Title-case aliases** for backward compatibility
   ```python
   if "Issue" not in df.columns and "issue" in df.columns:
       df["Issue"] = df["issue"]
   ```

**Result:** Every DataFrame has consistent column names regardless of source

---

### **1.3 Text Normalization (Lines 90-93)**

```python
def _norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)  # Unicode normalization
    return re.sub(r"\s+", " ", s).strip()  # Remove extra whitespace
```

**Purpose:** Clean text consistently
- Handles None values → empty string
- Normalizes Unicode (café vs cafe)
- Removes extra spaces/tabs/newlines

---

### **1.4 PII Redaction (Lines 95-99)**

```python
def _pii_redact(s: str) -> str:
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", s)
    s = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]", s)
    s = re.sub(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[PHONE]", s)
    return s
```

**Purpose:** Privacy protection
- Replaces `john@example.com` → `[EMAIL]`
- Replaces `192.168.1.1` → `[IP]`
- Replaces `+1-555-1234` → `[PHONE]`

---

### **1.5 Canonical Text Creation (Lines 101-108)**

```python
def _canonical(row: pd.Series) -> str:
    title = _norm(row.get("Issue") or row.get("Title") or row.get("Subject"))
    desc  = _norm(row.get("Description"))
    reso  = _norm(row.get("Resolution"))
    cat   = _norm(row.get("Category"))
    text  = f"Title: {title}\nCategory: {cat}\nProblem: {desc}\nResolution: {reso}"
    return _pii_redact(text)
```

**Purpose:** Create structured text for search indexing

**Example:**
```
Input row: {"Issue": "VPN timeout", "Description": "Can't connect", ...}

Output:
Title: VPN timeout
Category: Network
Problem: Can't connect
Resolution: Updated VPN settings
```

This structured format helps both FAISS and BM25 understand the ticket better.

---

### **1.6 Row Hashing for Deduplication (Lines 110-113)**

```python
def _stable_row_hash(row: pd.Series) -> str:
    base = f"{_norm(row.get('Issue'))}|{_norm(row.get('Description'))}|{_norm(row.get('Resolution'))}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
```

**Purpose:** Create unique ID for duplicate detection
- Combines Issue + Description + Resolution
- SHA-256 hash ensures identical tickets get same hash
- First 16 characters is enough for uniqueness

---

## 🗂️ **SECTION 2: STATE MANAGEMENT** (Lines 115-132)

```python
class GraphState(TypedDict):
    # Global artifacts (built once by DataProcessingAgent)
    data_dir: str
    kb_df: Optional[pd.DataFrame]           # Knowledge base (resolved tickets)
    raw_df: Optional[pd.DataFrame]          # All tickets (including unresolved)
    faiss_index: Optional[Any]              # Vector similarity index
    embeddings: Optional[np.ndarray]        # Sentence embeddings matrix
    bm25: Optional[BM25Okapi]               # Keyword search index
    bm25_tokens: Optional[List[List[str]]]  # Tokenized corpus for BM25
    embed_model_name: str
    artifacts_dir: str
    ready: bool

    # Per-ticket fields (changes for each new ticket)
    ticket: Dict[str, Any]                  # Current ticket being processed
    candidates: List[Tuple[int, float]]     # Retrieved similar tickets (index, score)
    evidence: List[Dict[str, Any]]          # Formatted evidence for LLM
    plan: str                               # Generated solution bullets
    logs: List[str]                         # Processing logs
```

**Purpose:** LangGraph state that flows through all agents
- **Global artifacts**: Built once, reused for all tickets
- **Per-ticket**: Reset for each new ticket

---

## 🤖 **SECTION 3: DATA PROCESSING AGENT** (Lines 134-253)

This is the **first agent** - loads and indexes old tickets.

### **3.1 Class Definition (Lines 136-142)**

```python
@dataclass
class DataProcessingAgent:
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    artifacts_dir: str = "./artifacts"
    force_rebuild: bool = False
    shared_embedder: Optional[SentenceTransformer] = None
```

**Configuration:**
- **embed_model_name**: Which sentence-transformer model to use
- **artifacts_dir**: Where to cache processed data
- **force_rebuild**: If True, ignore cache and rebuild
- **shared_embedder**: Pre-loaded model (avoids reloading)

---

### **3.2 Loading Source Files (Lines 144-168)**

```python
def _load_sources(self, data_dir: str) -> pd.DataFrame:
    frames = []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        low = fname.lower()
        try:
            if low.endswith(".csv"):
                frames.append(pd.read_csv(path))
            elif low.endswith(".xlsx"):
                frames.append(pd.read_excel(path))
            elif low.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                frames.append(pd.DataFrame(obj) if isinstance(obj, (list, dict)) else pd.read_json(path))
        except Exception:
            if low.endswith(".json"):
                frames.append(pd.read_json(path))
    
    if not frames:
        raise ValueError("No files found in Data/ (old tickets).")
    return pd.concat(frames, ignore_index=True)
```

**Step-by-step:**
1. **List all files** in `Data/` folder
2. **For each file**:
   - If `.csv`: Load with pandas
   - If `.xlsx`: Load Excel file
   - If `.json`: Load JSON (handles both array and dict formats)
3. **Concatenate** all DataFrames into one big DataFrame
4. **Return** unified dataset

---

### **3.3 Schema Unification (Lines 170-180)**

```python
def _unify(self, df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names (case-insensitive + synonym mapping)
    df = _standardize_columns(df)

    df["Resolved"] = df["Resolved"].astype(bool)
    for col in ["Issue", "Description", "Resolution", "Category", "Agent Name"]:
        df[col] = df[col].map(_norm)

    df["Description"] = df["Description"].map(_pii_redact)
    df["Resolution"]  = df["Resolution"].map(_pii_redact)
    df["canonical"]   = df.apply(_canonical, axis=1)
    return df
```

**Step-by-step:**
1. **Standardize columns** (uses the helper from earlier)
2. **Convert Resolved to boolean** (handles "Yes", "True", 1, etc.)
3. **Normalize text** in key columns (remove extra spaces, Unicode)
4. **Redact PII** in Description and Resolution
5. **Create canonical text** for each row (structured format for search)

---

### **3.4 Deduplication (Lines 182-187)**

```python
def _dedupe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    df["_hash"] = df.apply(_stable_row_hash, axis=1)
    before = len(df)
    df = df.drop_duplicates(subset=["_hash"], keep="first").drop(columns=["_hash"])
    removed = before - len(df)
    return df, removed
```

**Step-by-step:**
1. **Create hash** for each row (Issue+Description+Resolution)
2. **Count** rows before dedup
3. **Drop duplicates** (keeps first occurrence)
4. **Remove hash column** (temporary)
5. **Return** cleaned DataFrame + count of removed duplicates

---

### **3.5 Building Search Indexes (Lines 189-198)**

```python
def _build_indexes(self, kb_df: pd.DataFrame):
    # Embeddings (semantic search)
    embedder = self.shared_embedder or SentenceTransformer(self.embed_model_name)
    X = embedder.encode(kb_df["canonical"].tolist(), 
                        normalize_embeddings=True, 
                        batch_size=64, 
                        show_progress_bar=True)
    X = np.asarray(X, dtype="float32")
    
    # FAISS index (fast similarity search)
    index = faiss.IndexFlatIP(X.shape[1])  # Inner product (cosine sim with normalized)
    index.add(X)

    # BM25 (keyword search)
    tokens = [t.lower().split() for t in kb_df["canonical"].tolist()]
    bm25 = BM25Okapi(tokens)
    
    return index, X, bm25, tokens
```

**Step-by-step:**

**For Semantic Search (FAISS):**
1. **Load embedder model** (sentence-transformers)
2. **Encode all canonical texts** → 384-dimensional vectors
3. **Normalize embeddings** (for cosine similarity)
4. **Create FAISS index** (IndexFlatIP = inner product)
5. **Add all vectors** to index

**For Keyword Search (BM25):**
1. **Tokenize** canonical texts (split on whitespace, lowercase)
2. **Build BM25 index** (TF-IDF based ranking)

**Result:** Two complementary search indexes
- FAISS: Understands meaning ("VPN timeout" ~ "VPN disconnection")
- BM25: Finds exact keywords ("VPN" must be present)

---

### **3.6 Main Processing Logic (Lines 200-253)**

```python
def __call__(self, state: GraphState) -> GraphState:
    logs = state.get("logs", [])
    
    # Try to load from cache first
    if not self.force_rebuild:
        cached = self._try_cache()
        if cached and cached[0] is not None:
            # Cache hit! Skip expensive processing
            kb, faiss_index, bm25, tokens = cached
            # Update state with cached data
            state.update({...})
            return state
```

**Step 1: Try Cache** (Lines 202-225)
- Check if `artifacts/` folder has pre-built indexes
- If yes, load them (saves 30-60 seconds!)
- If no, continue to build from scratch

```python
    # No cache, build from scratch
    data_dir = state["data_dir"]
    logs.append(f"DataProcessingAgent: loading from {data_dir}")
    
    raw = self._load_sources(data_dir)  # Load all files
    raw = self._unify(raw)               # Standardize schema
```

**Step 2: Load & Unify** (Lines 227-230)
- Load all CSV/XLSX/JSON from Data/
- Standardize columns, normalize text, redact PII

```python
    # Filter to resolved tickets only
    kb = raw[raw["Resolved"] == True].copy()
    kb, removed = self._dedupe(kb)
    
    logs.append(f"Loaded rows: {len(raw)} | KB rows (resolved): {len(kb)} | Deduped removed: {removed}")
```

**Step 3: Filter & Dedupe** (Lines 232-235)
- Keep only resolved tickets (those with solutions)
- Remove duplicates
- Log statistics

```python
    # Build indexes
    logs.append("Building indexes (MiniLM + FAISS + BM25)…")
    t0 = time.time()
    faiss_index, embeddings, bm25, tokens = self._build_indexes(kb)
    logs.append(f"Indexes built in {time.time() - t0:.1f}s")
```

**Step 4: Index Building** (Lines 237-241)
- Build FAISS (semantic) + BM25 (keyword) indexes
- Time the operation
- This takes ~2-5 seconds depending on # of tickets

```python
    # Cache for next time
    self._cache_artifacts(kb, faiss_index, tokens, self.artifacts_dir)
```

**Step 5: Cache** (Line 243)
- Save processed data to disk
- Next run will be instant!

```python
    # Update state
    state.update({
        "raw_df": raw,
        "kb_df": kb.reset_index(drop=True),
        "faiss_index": faiss_index,
        "embeddings": embeddings,
        "bm25": bm25,
        "bm25_tokens": tokens,
        "embed_model_name": self.embed_model_name,
        "ready": True,
        "logs": logs
    })
    return state
```

**Step 6: Return State** (Lines 245-253)
- Package everything into state
- Mark as `ready=True`
- Return to LangGraph

---

## 🔍 **SECTION 4: HYBRID SEARCH INDEX** (Lines 255-293)

This combines semantic (FAISS) and keyword (BM25) search.

### **4.1 Class Definition (Lines 257-262)**

```python
@dataclass
class HybridIndex:
    kb_df: pd.DataFrame             # Knowledge base
    faiss_index: Any                # FAISS vector index
    bm25: BM25Okapi                 # BM25 keyword index
    embedder: SentenceTransformer   # For encoding queries
```

---

### **4.2 Search Method (Lines 264-293)**

```python
def search(self, query_text: str, k: int = 8, alpha: float = 0.6) -> List[Tuple[int, float]]:
```

**Parameters:**
- `query_text`: The new ticket text ("VPN timeout...")
- `k`: How many results to return (default 8)
- `alpha`: Weight for semantic (0.6 = 60% semantic, 40% keyword)

**Step 1: Semantic Search (FAISS)**
```python
qv = self.embedder.encode([query_text.lower()], normalize_embeddings=True)
D, I = self.faiss_index.search(np.array(qv, dtype="float32"), k * 5)
sem_hits = [(i, s) for i, s in zip(I[0].tolist(), D[0].tolist()) if i >= 0]
```
- Encode query to 384-dim vector
- Search FAISS for top k*5 similar vectors
- Get (index, similarity_score) pairs
- Filter out invalid indices (-1)

**Step 2: Normalize Semantic Scores**
```python
sem_vals = np.array([s for _, s in sem_hits]) if sem_hits else np.array([0.0])
smin, smax = float(sem_vals.min()), float(sem_vals.max())
sem_norm = {idx: (score - smin) / (smax - smin + 1e-9) for idx, score in sem_hits}
```
- Get all semantic scores
- Normalize to [0, 1] range
- Store in dictionary {index: normalized_score}

**Step 3: Keyword Search (BM25)**
```python
bm_arr = self.bm25.get_scores(query_text.lower().split())
bmin, bmax = float(np.min(bm_arr)), float(np.max(bm_arr))
bm_norm = (bm_arr - bmin) / (bmax - bmin + 1e-9)
```
- Tokenize query (split on whitespace)
- Get BM25 scores for all documents
- Normalize to [0, 1] range

**Step 4: Combine Scores**
```python
bm_top = np.argsort(-bm_arr)[:k*5].tolist()
cand = set([i for i, _ in sem_hits] + bm_top)

scored = []
for idx in cand:
    s = sem_norm.get(idx, 0.0)        # Semantic score (0 if not in top-k*5)
    b = float(bm_norm[idx])           # BM25 score
    scored.append((idx, alpha*s + (1-alpha)*b))  # Hybrid score
```
- Get top k*5 from BM25
- Union of semantic and BM25 candidates
- For each candidate:
  - Get semantic score (or 0)
  - Get BM25 score
  - Combine: `0.6 * semantic + 0.4 * BM25`

**Step 5: Sort and Return**
```python
scored.sort(key=lambda x: x[1], reverse=True)
return scored[:k]
```
- Sort by hybrid score (high to low)
- Return top k results

**Result:** List of (ticket_index, hybrid_score) tuples

---

## 🎣 **SECTION 5: RETRIEVER AGENT** (Lines 295-321)

Finds similar tickets for a new problem.

```python
class RetrieverAgent:
    def __init__(self, embedder: SentenceTransformer, alpha=0.6, k=5):
        self.embedder = embedder
        self.alpha, self.k = alpha, k

    def __call__(self, state: GraphState) -> GraphState:
        kb = state["kb_df"]
        fa = state["faiss_index"]
        bm = state["bm25"]
        
        # Check if artifacts ready
        if kb is None or len(kb) == 0 or fa is None or bm is None:
            state["candidates"] = []
            state["logs"].append("RetrieverAgent: KB empty or artifacts missing")
            return state

        # Build query from new ticket
        t = state["ticket"]
        q_title = _norm(t.get("Issue") or t.get("Title"))
        q_desc  = _norm(t.get("Description"))
        q = _pii_redact(f"Title: {q_title}\nProblem: {q_desc}")

        # Search!
        idx = HybridIndex(kb, fa, bm, self.embedder)
        state["candidates"] = idx.search(q, k=self.k, alpha=self.alpha)
        state["logs"].append(f"RetrieverAgent: {len(state['candidates'])} candidates")
        return state
```

**Step-by-step:**
1. **Get artifacts** from state (KB, FAISS, BM25)
2. **Check if ready** (empty KB → skip)
3. **Build query text** from new ticket (same format as canonical)
4. **Search** using HybridIndex
5. **Store candidates** in state (list of (index, score))
6. **Log** and return

**Result:** `state["candidates"]` = [(42, 0.89), (17, 0.81), (55, 0.78), ...]

---

## 📋 **SECTION 6: CURATOR AGENT** (Lines 323-347)

Formats raw candidates into structured evidence for the LLM.

```python
class CuratorAgent:
    def __call__(self, state: GraphState) -> GraphState:
        kb = state["kb_df"]
        cand = state.get("candidates", [])
        
        # No candidates?
        if kb is None or len(kb) == 0 or not cand:
            state["evidence"] = []
            state["logs"].append("CuratorAgent: no candidates")
            return state

        # Format each candidate
        evid = []
        for idx, score in cand:
            r = kb.iloc[int(idx)]  # Get row from knowledge base
            evid.append({
                "ticket_id": r["ticket_id"],
                "Issue": _norm(r.get("Issue","")),
                "Description": _norm(r.get("Description","")),
                "Resolution": _norm(r.get("Resolution","")),
                "Category": _norm(r.get("Category","")),
                "Date": _norm(r.get("Date","")),
                "score": float(round(score, 4))
            })
        
        state["evidence"] = evid
        state["logs"].append(f"CuratorAgent: prepared {len(evid)} evidence rows")
        return state
```

**Step-by-step:**
1. **Get candidates** from retrieval [(42, 0.89), (17, 0.81), ...]
2. **For each candidate**:
   - Get the actual ticket row from knowledge base
   - Extract fields: ticket_id, Issue, Description, Resolution, Category, Date, score
   - Clean/normalize text
3. **Store as evidence** list of dicts
4. **Log** and return

**Result:** `state["evidence"]` = structured, clean data ready for LLM

---

## 🤖 **SECTION 7: PLANNER AGENT (LLM)** (Lines 349-424)

**This is where the AI magic happens!**

### **7.1 Initialization (Lines 354-366)**

```python
class PlannerAgent:
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "llama-3.1-8b-instant", 
                 deterministic: bool = False):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set...")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.deterministic = deterministic
        print(f"✅ Using Groq model: {model} | deterministic={deterministic}")
```

**What it does:**
- Gets API key from environment variable
- Creates Groq client
- Stores model name and settings

---

### **7.2 Building the Prompt (Lines 368-397)**

```python
def _messages(self, ticket: Dict[str, Any], evidence_rows: List[Dict[str, Any]]):
    q_title = _norm(ticket.get("Issue") or ticket.get("Title"))
    q_desc  = _norm(ticket.get("Description"))

    # Format evidence
    ev_lines = []
    for r in evidence_rows:
        ev_lines.append(
            f"Ticket {r['ticket_id']} ({r.get('Category','Unknown')}):\n"
            f"  Problem: {_norm(r.get('Description',''))}\n"
            f"  Resolution: {_norm(r.get('Resolution',''))}"
        )
    ev_str = "\n\n".join(ev_lines)
```

**Step 1: Format Evidence**

Example output:
```
Ticket TCKT-1011 (Network):
  Problem: VPN keeps disconnecting
  Resolution: Updated VPN settings

Ticket TCKT-1022 (Network):
  Problem: Cannot connect to VPN
  Resolution: Reinstalled VPN client
```

```python
    sys = (
        "You are an IT helpdesk assistant. Suggest troubleshooting steps based ONLY on the similar "
        "resolved tickets provided.\n\nSTRICT RULES:\n"
        "1) Provide exactly 3-5 specific, actionable bullet points\n"
        "2) Each bullet MUST cite at least one ticket using [#TICKET_ID]\n"
        "3) Only suggest steps explicitly mentioned or directly implied by evidence\n"
        "4) Start each bullet with '- '\n"
        "5) Keep each bullet under 80 words\n"
        "6) End with exactly one caveat bullet: '- Caveat: ...'\n"
        "7) No text outside the bullets"
    )
```

**Step 2: System Message**
- Sets the role (IT helpdesk assistant)
- Defines strict rules for output format
- Ensures citations and caveats

```python
    user = (
        f"NEW TICKET\nTitle: {q_title}\nProblem: {q_desc}\n\n"
        f"SIMILAR RESOLVED TICKETS:\n{ev_str}\n\n"
        f"Generate the bullets now, following the STRICT RULES exactly."
    )
    
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]
```

**Step 3: User Message**
- Presents new ticket
- Shows evidence
- Asks for bullets

---

### **7.3 Calling the LLM (Lines 399-424)**

```python
def __call__(self, state: GraphState) -> GraphState:
    # No evidence → emit caveat-only
    if not state.get("evidence"):
        state["plan"] = "- Caveat: No similar resolved tickets found; gather diagnostics or escalate."
        state["logs"].append("PlannerAgent: no evidence; emitted caveat-only")
        return state

    messages = self._messages(state["ticket"], state["evidence"])

    temperature = 0.0 if self.deterministic else 0.2
    top_p = 1.0 if self.deterministic else 0.9

    # Retry logic (handles rate limits, timeouts)
    backoff = 1.0
    for attempt in range(4):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=350,
            )
            text = (resp.choices[0].message.content or "").strip()
            
            # Extract bullets (in case LLM adds extra text)
            m = re.search(r"(\n?- .*)", text, flags=re.S)
            state["plan"] = (m.group(1).strip() if m else text) or "- Caveat: No guidance."
            state["logs"].append(f"PlannerAgent: plan generated via Groq ({self.model})")
            return state
            
        except Exception as e:
            # Retry on rate limits/timeouts
            if attempt < 3 and any(tok in str(e).lower() for tok in ("429", "rate", "timeout")):
                time.sleep(backoff)
                backoff *= 2
                continue
            # Failed after retries
            state["plan"] = "- Caveat: Could not generate guidance due to API error."
            state["logs"].append(f"PlannerAgent: API error - {e}")
            return state
```

**Step-by-step:**

1. **Check evidence** - If none, return caveat-only
2. **Build messages** - System + User prompts
3. **Set parameters**:
   - `temperature=0.2`: Focused (not creative)
   - `top_p=0.9`: Nucleus sampling
   - `max_tokens=350`: Enough for 5 bullets + caveat

4. **Try API call** (with retry):
   - Call Groq API
   - Get response
   - Extract bullets (remove any extra text)
   - Store in state

5. **Handle errors**:
   - Rate limit (429) → wait and retry (exponential backoff)
   - Timeout → retry
   - Other errors → fallback caveat

**Result:** `state["plan"]` = bullet points with citations

Example:
```
- Update VPN client to latest version [#TCKT-1011]
- Check VPN configuration settings [#TCKT-1011]
- Verify firewall isn't blocking VPN ports [#TCKT-1022]
- Caveat: Ensure user has admin rights before modifying settings
```

---

## ✅ **SECTION 8: VALIDATOR AGENT** (Lines 426-497)

Post-processes LLM output to ensure quality.

### **8.1 Initialization (Lines 428-432)**

```python
class ValidatorAgent:
    def __init__(self, sim_threshold: float = 0.85, max_len: int = 400):
        self.sim_threshold = sim_threshold  # 85% similar → merge bullets
        self.max_len = max_len              # Clip overly long bullets
        self._stop = set("the a an of to in on for with and or if is are be by from at as into over under".split())
```

---

### **8.2 Helper Methods (Lines 434-456)**

```python
def _extract_citations(self, line: str):
    """Extract ticket IDs and clean text"""
    cites = re.findall(r"\[#([A-Za-z0-9_-]+)\]", line)
    text = re.sub(r"\s*\[#([A-Za-z0-9_-]+)\]\s*", "", line).strip()
    return text, cites
    # "Check VPN [#TKT1] [#TKT2]" → ("Check VPN", ["TKT1", "TKT2"])
```

```python
def _normalize_for_match(self, line: str):
    """Normalize for similarity comparison"""
    if line.startswith("- "): line = line[2:]
    text, _ = self._extract_citations(line)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    toks = [t for t in text.split() if t not in self._stop]
    return " ".join(toks)
    # "- Check the VPN settings [#TKT1]" → "check vpn settings"
```

```python
def _merge_citations(self, base_line: str, new_cites):
    """Merge citations from similar bullets"""
    text, cites = self._extract_citations(base_line)
    merged = list(dict.fromkeys(cites + new_cites))  # Dedupe, preserve order
    if merged:
        text = text.rstrip(". ")
        text = f"{text} " + " ".join(f"[#{cid}]" for cid in merged)
    return text
```

---

### **8.3 Main Validation Logic (Lines 458-497)**

```python
def __call__(self, state: GraphState) -> GraphState:
    plan = state.get("plan", "").strip()
    
    # Empty plan?
    if not plan:
        state["plan"] = "- Caveat: No guidance could be generated from current evidence."
        return state

    # Split into lines
    lines = [ln.strip() for ln in plan.splitlines() if ln.strip()]
    
    # Ensure all start with "- "
    lines = ["- " + re.sub(r"^-+\s*", "", ln) if not ln.startswith("- ") else ln for ln in lines]

    # Separate caveats from normal bullets
    normal_lines, caveats = [], []
    for ln in lines:
        if re.match(r"^-+\s*caveat:\s*", ln, flags=re.I):
            caveats.append(ln)
        else:
            normal_lines.append(ln)

    # Deduplicate similar bullets
    deduped = []
    for ln in normal_lines:
        norm = self._normalize_for_match(ln)
        _, cites = self._extract_citations(ln)
        merged = False
        
        # Check against existing bullets
        for i, ex in enumerate(deduped):
            ratio = difflib.SequenceMatcher(None, norm, self._normalize_for_match(ex)).ratio()
            if ratio >= self.sim_threshold:  # 85% similar
                # Merge this into existing bullet
                deduped[i] = "- " + self._merge_citations(deduped[i][2:], cites)
                merged = True
                break
        
        if not merged:
            deduped.append(ln)

    # Ensure exactly one caveat
    if caveats:
        caveat = caveats[0]
    else:
        caveat = "- Caveat: Validate environment specifics (OS, versions, permissions) before applying steps."

    # Clip overly long bullets
    out_lines = []
    for ln in deduped + [caveat]:
        if len(ln) > self.max_len:
            ln = ln[: self.max_len - 1] + "…"
        out_lines.append(ln)

    state["plan"] = "\n".join(out_lines)
    state["logs"].append("ValidatorAgent: plan validated (de-duplicated & citations merged)")
    return state
```

**Step-by-step:**

1. **Check if empty** → fallback caveat
2. **Split into lines**
3. **Ensure "- " prefix** on all bullets
4. **Separate caveats** from normal bullets
5. **Deduplicate similar bullets**:
   - For each bullet, check against existing
   - If 85%+ similar → merge citations
   - Example: "Check VPN [#TKT1]" + "Verify VPN [#TKT2]" → "Check VPN [#TKT1] [#TKT2]"
6. **Ensure exactly one caveat** (add default if missing)
7. **Clip long bullets** (>400 chars)
8. **Return cleaned plan**

**Result:** Clean, deduplicated bullets with merged citations

---

## 🔄 **SECTION 9: GRAPH BUILDER** (Lines 499-521)

Wires all agents together with LangGraph.

```python
def build_graph(alpha=0.6, k=5, groq_model="llama-3.1-8b-instant", deterministic=False) -> StateGraph:
    g = StateGraph(GraphState)

    # Create shared embedder (used by both processor and retriever)
    shared_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Instantiate all agents
    processor = DataProcessingAgent(shared_embedder=shared_embedder)
    retriever =  RetrieverAgent(shared_embedder, alpha=alpha, k=k)
    curator   =  CuratorAgent()
    planner   =  PlannerAgent(model=groq_model, deterministic=deterministic)
    validator =  ValidatorAgent()

    # Add nodes to graph
    g.add_node("process_data", processor)
    g.add_node("retrieve",     retriever)
    g.add_node("curate",       curator)
    g.add_node("plan",         planner)
    g.add_node("validate",     validator)

    # Define flow (linear pipeline)
    g.set_entry_point("process_data")
    g.add_edge("process_data", "retrieve")
    g.add_edge("retrieve", "curate")
    g.add_edge("curate", "plan")
    g.add_edge("plan", "validate")
    g.add_edge("validate", END)
    
    return g
```

**Creates this flow:**
```
START → DataProcessing → Retriever → Curator → Planner → Validator → END
```

Each node transforms the state and passes it to the next.

---

## 📥📤 **SECTION 10: I/O HELPERS** (Lines 523-605)

### **10.1 Reading New Tickets (Lines 525-589)**

```python
def _read_new_tickets(input_path: str) -> List[Dict[str, Any]]:
    tickets = []
    for fp in glob.glob(os.path.join(input_path, "*")):
        low = fp.lower()
        try:
            if low.endswith(".csv"):
                df = pd.read_csv(fp)
            elif low.endswith(".xlsx"):
                df = pd.read_excel(fp)
            elif low.endswith(".json"):
                with open(fp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                df = pd.DataFrame(obj) if isinstance(obj, (list, dict)) else pd.read_json(fp)
            else:
                continue
        except Exception:
            # Fallback for malformed JSON
            if low.endswith(".json"):
                df = pd.read_json(fp)
            else:
                continue

        # Standardize columns
        df = _standardize_columns(df)

        # Extract tickets
        for _, row in df.iterrows():
            tickets.append({
                "Issue": _norm(row.get("Issue", "")),
                "Description": _norm(row.get("Description", "")),
            })
    return tickets
```

**Purpose:** Load new tickets from `input/` folder
- Handles CSV/XLSX/JSON
- Standardizes columns
- Returns list of ticket dicts

---

### **10.2 Writing Results (Lines 591-605)**

```python
def _write_outputs(out_dir: str, ticket_idx: int, plan: str, evidence: List[Dict[str, Any]], logs: List[str]):
    os.makedirs(out_dir, exist_ok=True)
    
    # Write JSON
    out_json = {"direction_bullets": plan, "evidence": evidence, "logs": logs}
    with open(os.path.join(out_dir, f"result_{ticket_idx:03d}.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    # Write Markdown (human-readable)
    md = ["# Suggested direction", plan, "\n# Evidence"]
    for e in evidence:
        title = e.get("title") or e.get("Issue") or ""
        md.append(
            f"- **{e.get('ticket_id','')}** — *{title}* — **Category:** {e.get('Category','')} — "
            f"*Resolution:* {e.get('Resolution','')} (score {e.get('score','')})"
        )
    md.append("\n# Logs")
    md.extend(f"- {l}" for l in logs)
    
    with open(os.path.join(out_dir, f"result_{ticket_idx:03d}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
```

**Purpose:** Save results to `output/` folder
- Creates both JSON (machine-readable) and Markdown (human-readable)
- Each ticket gets result_001.json, result_001.md, etc.

---

## 🎬 **SECTION 11: MAIN CLI** (Lines 607-695)

### **11.1 Main Processing Function (Lines 609-683)**

```python
def run_once(data_dir: str, input_dir: str, out_dir: str, alpha=0.6, k=5,
              groq_model="llama-3.1-8b-instant", deterministic=False):
    
    # Build and compile graph
    graph = build_graph(alpha=alpha, k=k, groq_model=groq_model, deterministic=deterministic).compile()

    # Initialize state
    base_state: GraphState = {
        "data_dir": data_dir,
        "kb_df": None, "raw_df": None,
        "faiss_index": None, "embeddings": None,
        "bm25": None, "bm25_tokens": None,
        "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "artifacts_dir": "./artifacts",
        "ready": False,
        "ticket": {"Issue":"", "Description":""},
        "candidates": [], "evidence": [],
        "plan": "", "logs": []
    }

    # Load new tickets
    new_tickets = _read_new_tickets(input_dir)
    if not new_tickets:
        print(f"No new tickets found in {input_dir}")
        return

    # Process each ticket
    for i, t in enumerate(new_tickets, start=1):
        try:
            # Set current ticket
            base_state["ticket"] = t
            
            # Run through entire pipeline!
            result = graph.invoke(base_state)
            
            # Format evidence for output
            evidence_cards = [{
                "ticket_id": r["ticket_id"],
                "title": r["Issue"],
                "Category": r["Category"],
                "Resolution": r["Resolution"],
                "snippet": (r["Description"][:280] + "…") if len(r["Description"]) > 280 else r["Description"],
                "score": r["score"],
                "Date": r["Date"]
            } for r in result.get("evidence", [])]

            # Write results
            _write_outputs(out_dir, i, result["plan"], evidence_cards, result["logs"])

            # Persist artifacts for next ticket (avoid reprocessing)
            base_state.update({
                "kb_df": result.get("kb_df"),
                "raw_df": result.get("raw_df"),
                "faiss_index": result.get("faiss_index"),
                "embeddings": result.get("embeddings"),
                "bm25": result.get("bm25"),
                "bm25_tokens": result.get("bm25_tokens"),
                "embed_model_name": result.get("embed_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                "ready": True,
                "logs": []
            })
            
            print(f"✅ Processed ticket {i}/{len(new_tickets)}: {t.get('Issue','N/A')[:60]}")
            
        except Exception as e:
            print(f"❌ ERROR processing ticket {i}/{len(new_tickets)} ({t.get('Issue','N/A')})")
            print(f"   Error: {e}")
            
            # Write error file with debug info
            err_file = os.path.join(out_dir, f"error_{i:03d}.txt")
            with open(err_file, "w", encoding="utf-8") as ef:
                ef.write(f"Ticket: {t}\n\nError: {e}\n\n{traceback.format_exc()}")
                
                # Debug snapshot
                ef.write("\n\n=== DEBUG SNAPSHOT ===\n")
                try:
                    if 'result' in locals():
                        ef.write(f"State keys: {list(result.keys())}\n")
                        ef.write(f"KB rows: {len(result.get('kb_df', [])) if result.get('kb_df') is not None else 'None'}\n")
                        ef.write(f"Candidates: {len(result.get('candidates', []))}\n")
                        ef.write(f"Evidence: {len(result.get('evidence', []))}\n")
                    else:
                        ef.write("Result not available (error before graph invoke)\n")
                except Exception as debug_err:
                    ef.write(f"Debug snapshot failed: {debug_err}\n")
            
            print(f"   Details saved to: {err_file}")
            continue

    print(f"\n✅ Completed! Processed {len(new_tickets)} ticket(s). Results saved to {out_dir}")
```

**Step-by-step:**

1. **Build graph** with all agents
2. **Initialize state** (empty at first)
3. **Load new tickets** from input/
4. **For each ticket**:
   - Set as current ticket in state
   - **Run graph** (all 5 agents execute)
   - Format output
   - **Write results** to output/
   - **Persist artifacts** (so next ticket is faster)
   - Print progress
5. **Handle errors**:
   - Write error file
   - Add debug snapshot
   - Continue with next ticket

---

### **11.2 Argument Parsing (Lines 685-695)**

```python
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",  default="Data",   help="Folder with old tickets")
    ap.add_argument("--input_dir", default="input",  help="Folder with new tickets")
    ap.add_argument("--out_dir",   default="output", help="Folder to write results")
    ap.add_argument("--alpha",     type=float, default=0.6, help="Hybrid weight (semantic)")
    ap.add_argument("--k",         type=int,   default=5,   help="Top-k retrieved")
    ap.add_argument("--groq_model", default="llama-3.1-8b-instant", help="Groq model")
    ap.add_argument("--deterministic", action="store_true", help="Use temp=0.0 for reproducible")
    
    args = ap.parse_args()
    run_once(args.data_dir, args.input_dir, args.out_dir,
              alpha=args.alpha, k=args.k,
              groq_model=args.groq_model, deterministic=args.deterministic)
```

**Allows customization via command line:**
```bash
python ticket_graph.py --data_dir Data --input_dir input --out_dir output --k 10 --alpha 0.7
```

---

## 🎬 **COMPLETE FLOW EXAMPLE**

Let's trace one ticket through the entire system:

### **Input:** New ticket arrives
```
Issue: "VPN connection timeout"
Description: "VPN times out frequently during use"
```

### **Step 1: DataProcessingAgent**
- Loads Data/ folder (29 tickets)
- Filters to resolved (17 tickets)
- Deduplicates (removes 2)
- Builds FAISS index (15 tickets → 15 × 384 vectors)
- Builds BM25 index (15 tickets → tokenized)
- **Output:** Artifacts ready

### **Step 2: RetrieverAgent**
- Query: "Title: VPN connection timeout\nProblem: VPN times out frequently..."
- FAISS search → top 25 semantic matches
- BM25 search → top 25 keyword matches
- Combine with α=0.6 (60% semantic, 40% keyword)
- **Output:** Top 5: [(42, 0.89), (17, 0.81), (55, 0.78), (12, 0.72), (33, 0.68)]

### **Step 3: CuratorAgent**
- For each index (42, 17, 55, 12, 33):
  - Get ticket from KB
  - Extract fields
  - Format nicely
- **Output:** 5 evidence dicts with Issue, Description, Resolution, Category, score

### **Step 4: PlannerAgent (Groq LLM)**
- Builds prompt with new ticket + 5 evidence tickets
- Calls Groq API (Llama 3.1 8B)
- LLM generates:
```
- Update VPN client to latest version [#TCKT-1011]
- Check VPN configuration settings in network preferences [#TCKT-1011]
- Verify firewall isn't blocking VPN ports [#TCKT-1022]
- Test connection with different VPN protocol [#TCKT-1022]
- Caveat: Ensure user has admin rights before modifying settings
```
- **Output:** Raw bullets

### **Step 5: ValidatorAgent**
- Checks format (all have "- ")
- Checks for duplicates (none found)
- Ensures exactly one caveat (yes)
- Clips long bullets (none >400 chars)
- **Output:** Validated plan

### **Step 6: Write Results**
- Saves to output/result_001.json
- Saves to output/result_001.md
- **Done!**

---

## 🎯 **KEY TAKEAWAYS**

### **Your System is:**

1. **Hybrid RAG**: Combines semantic (FAISS) + keyword (BM25) search
2. **Multi-agent**: 5 specialized agents, each with single responsibility
3. **LLM-powered**: Uses Groq's Llama 3.1 8B for fast, cheap generation
4. **Validated**: Post-processes LLM output for quality
5. **Production-ready**: Caching, error handling, logging
6. **Flexible**: Handles varied data sources, column names, formats

### **Flow Summary:**
```
New Ticket → Search Similar → Format Evidence → LLM Generate → Validate → Output
```

### **Technologies:**
- **LangGraph**: Orchestration
- **FAISS**: Vector search (Facebook AI)
- **BM25**: Keyword search (traditional IR)
- **Groq**: Fast LLM API (Llama 3.1)
- **sentence-transformers**: Embeddings

---

**That's your entire system explained step-by-step!** 🎉

Each component has a clear purpose and they all work together to deliver high-quality, cited solution recommendations for IT helpdesk tickets!

