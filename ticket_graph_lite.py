# ticket_graph_lite.py - Memory-optimized version for deployment
# Uses BM25 only (no FAISS/embeddings) and OpenAI GPT-4o-mini

import os, re, json, time, unicodedata, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, TypedDict, Optional

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from openai import OpenAI

# -------------------- Utilities --------------------

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

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df.columns):
        return df
    lower_to_actual = {c.lower().strip(): c for c in df.columns}

    def find_any(names):
        for n in names:
            if n in lower_to_actual:
                return lower_to_actual[n]
        return None

    rename = {}
    for canon, alts in _STD_MAP.items():
        hit = find_any([a.lower() for a in alts])
        if hit and hit != canon:
            rename[hit] = canon
    if rename:
        df = df.rename(columns=rename)

    if "ticket_id" not in df.columns:
        df["ticket_id"] = np.arange(1, len(df) + 1)
    for c in ["issue", "description", "resolution", "category", "date", "agent name", "resolved"]:
        if c not in df.columns:
            df[c] = "" if c != "resolved" else np.nan

    alias_map = {
        "Issue": "issue",
        "Description": "description",
        "Resolution": "resolution",
        "Category": "category",
        "Date": "date",
        "Agent Name": "agent name",
        "Resolved": "resolved",
    }
    for Title, lower in alias_map.items():
        if Title not in df.columns and lower in df.columns:
            df[Title] = df[lower]
    return df

def _norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()

def _pii_redact(s: str) -> str:
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", s)
    s = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]", s)
    s = re.sub(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[PHONE]", s)
    return s

def _canonical(row: pd.Series) -> str:
    title = _norm(row.get("Issue") or row.get("Title") or row.get("Subject"))
    desc  = _norm(row.get("Description"))
    reso  = _norm(row.get("Resolution"))
    cat   = _norm(row.get("Category"))
    text  = f"Title: {title}\nCategory: {cat}\nProblem: {desc}\nResolution: {reso}"
    return _pii_redact(text)

def _stable_row_hash(row: pd.Series) -> str:
    base = f"{_norm(row.get('Issue'))}|{_norm(row.get('Description'))}|{_norm(row.get('Resolution'))}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

# -------------------- State --------------------

class GraphState(TypedDict):
    data_dir: str
    kb_df: Optional[pd.DataFrame]
    bm25: Optional[BM25Okapi]
    bm25_tokens: Optional[List[List[str]]]
    ready: bool
    ticket: Dict[str, Any]
    candidates: List[Tuple[int, float]]
    evidence: List[Dict[str, Any]]
    plan: str
    logs: List[str]

# -------------------- Lightweight Data Processing --------------------

@dataclass
class LiteDataProcessingAgent:
    """Memory-optimized data processing - BM25 only, no embeddings"""
    
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

    def _unify(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _standardize_columns(df)
        for col in ["Issue", "Description", "Resolution", "Category", "Agent Name"]:
            df[col] = df[col].map(_norm)
        df["Description"] = df["Description"].map(_pii_redact)
        df["Resolution"]  = df["Resolution"].map(_pii_redact)
        df["canonical"]   = df.apply(_canonical, axis=1)
        return df

    def _dedupe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        df["_hash"] = df.apply(_stable_row_hash, axis=1)
        before = len(df)
        df = df.drop_duplicates(subset=["_hash"], keep="first").drop(columns=["_hash"])
        removed = before - len(df)
        return df, removed

    def __call__(self, state: GraphState) -> GraphState:
        logs = state.get("logs", [])
        if state.get("ready"):
            logs.append("DataProcessingAgent: using existing KB")
            state["logs"] = logs
            return state

        data_dir = state["data_dir"]
        logs.append(f"DataProcessingAgent: loading from {data_dir}")

        raw = self._load_sources(data_dir)
        raw = self._unify(raw)
        kb = raw.copy()
        kb, removed = self._dedupe(kb)
        logs.append(f"Loaded rows: {len(raw)} | KB rows: {len(kb)} | Deduped: {removed}")

        if len(kb) == 0:
            state.update({
                "kb_df": kb, "bm25": None, "bm25_tokens": None,
                "ready": True, "logs": logs
            })
            return state

        # Build BM25 only (no FAISS)
        logs.append("Building BM25 index (lightweight)...")
        tokens = [t.lower().split() for t in kb["canonical"].tolist()]
        bm25 = BM25Okapi(tokens)
        logs.append("BM25 index built successfully")

        state.update({
            "kb_df": kb.reset_index(drop=True),
            "bm25": bm25,
            "bm25_tokens": tokens,
            "ready": True,
            "logs": logs
        })
        return state

# -------------------- Lightweight Retriever --------------------

class LiteRetrieverAgent:
    """BM25-only retrieval (no semantic search)"""
    
    def __init__(self, k=5):
        self.k = k

    def __call__(self, state: GraphState) -> GraphState:
        kb = state["kb_df"]
        bm = state["bm25"]
        
        if kb is None or len(kb) == 0 or bm is None:
            state["candidates"] = []
            state["logs"].append("RetrieverAgent: KB empty")
            return state

        t = state["ticket"]
        q_title = _norm(t.get("Issue") or t.get("Title"))
        q_desc  = _norm(t.get("Description"))
        q = _pii_redact(f"Title: {q_title}\nProblem: {q_desc}")
        cat = _norm(t.get("Category", ""))

        # BM25 search only
        query_tokens = q.lower().split()
        scores = bm.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(-scores)[:self.k * 3].tolist()
        
        # Filter by category if provided
        if cat:
            cat_norm = cat.lower()
            filtered = []
            for idx in top_indices:
                if _norm(kb.iloc[int(idx)]["Category"]).lower() == cat_norm:
                    filtered.append((idx, float(scores[idx])))
            candidates = filtered[:self.k]
        else:
            candidates = [(idx, float(scores[idx])) for idx in top_indices[:self.k]]
        
        state["candidates"] = candidates
        state["logs"].append(
            f"RetrieverAgent: {len(candidates)} candidates (BM25 only)"
            + (f" category='{cat}'" if cat else "")
        )
        return state

# -------------------- Curator --------------------

class CuratorAgent:
    def __call__(self, state: GraphState) -> GraphState:
        kb = state["kb_df"]
        cand = state.get("candidates", [])
        if kb is None or len(kb) == 0 or not cand:
            state["evidence"] = []
            state["logs"].append("CuratorAgent: no candidates")
            return state

        evid = []
        for idx, _score in cand:
            r = kb.iloc[int(idx)]
            evid.append({
                "ticket_id": r["ticket_id"],
                "Issue": _norm(r.get("Issue","")),
                "Description": _norm(r.get("Description","")),
                "Resolution": _norm(r.get("Resolution","")),
                "Category": _norm(r.get("Category","")),
                "Date": _norm(r.get("Date","")),
                "Resolved": r.get("Resolved", ""),
            })
        state["evidence"] = evid
        state["logs"].append(f"CuratorAgent: prepared {len(evid)} evidence rows")
        return state

# -------------------- OpenAI Planner --------------------

class OpenAIPlannerAgent:
    """Uses OpenAI GPT-4o-mini instead of Groq"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        print(f"✅ Using OpenAI model: {model}")

    def _messages(self, ticket: Dict[str, Any], evidence_rows: List[Dict[str, Any]]):
        q_title = _norm(ticket.get("Issue") or ticket.get("Title"))
        q_desc  = _norm(ticket.get("Description"))

        ev_lines = []
        for r in evidence_rows:
            ev_lines.append(
                f"Ticket {r['ticket_id']} ({r.get('Category','Unknown')} | Resolved: {r.get('Resolved','')}):\n"
                f"  Problem: {_norm(r.get('Description',''))}\n"
                f"  Resolution: {_norm(r.get('Resolution',''))}"
            )
        ev_str = "\n\n".join(ev_lines)

        sys = (
            "You are an IT helpdesk assistant. Suggest troubleshooting steps based ONLY on the similar "
            "tickets provided.\n\nSTRICT RULES:\n"
            "1) Provide exactly 3-5 specific, actionable bullet points\n"
            "2) Each bullet MUST cite at least one ticket using [#TICKET_ID]\n"
            "3) Only suggest steps explicitly mentioned or directly implied by evidence\n"
            "4) Start each bullet with '- '\n"
            "5) Keep each bullet under 80 words\n"
            "6) If any evidence shows Resolved: False, call it out explicitly\n"
            "7) End with exactly one caveat bullet: '- Caveat: ...'\n"
            "8) No text outside the bullets"
        )
        user = (
            f"NEW TICKET\nTitle: {q_title}\nProblem: {q_desc}\n\n"
            f"SIMILAR OLD TICKETS:\n{ev_str}\n\n"
            f"Generate the bullets now, following the STRICT RULES exactly."
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    def __call__(self, state: GraphState) -> GraphState:
        if not state.get("evidence"):
            state["plan"] = "- Caveat: No similar tickets found; gather diagnostics or escalate."
            state["logs"].append("PlannerAgent: no evidence")
            return state

        messages = self._messages(state["ticket"], state["evidence"])

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=400,
                )
                text = (resp.choices[0].message.content or "").strip()
                m = re.search(r"(\n?- .*)", text, flags=re.S)
                state["plan"] = (m.group(1).strip() if m else text) or "- Caveat: No guidance."
                state["logs"].append(f"PlannerAgent: plan generated via {self.model}")
                return state
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                state["plan"] = "- Caveat: Could not generate guidance due to API error."
                state["logs"].append(f"PlannerAgent: API error - {e}")
                return state

# -------------------- Validator --------------------

class ValidatorAgent:
    def __call__(self, state: GraphState) -> GraphState:
        plan = state.get("plan", "").strip()
        if not plan:
            state["plan"] = "- Caveat: No guidance could be generated."
            return state
        
        lines = [ln.strip() for ln in plan.splitlines() if ln.strip()]
        lines = ["- " + re.sub(r"^-+\s*", "", ln) if not ln.startswith("- ") else ln for ln in lines]
        
        # Ensure caveat
        has_caveat = any(re.match(r"^-+\s*caveat:\s*", ln, flags=re.I) for ln in lines)
        if not has_caveat:
            lines.append("- Caveat: Validate environment specifics before applying steps.")
        
        state["plan"] = "\n".join(lines)
        state["logs"].append("ValidatorAgent: plan validated")
        return state

# -------------------- Lightweight Graph Builder --------------------

def build_lite_graph() -> dict:
    """Build lightweight graph without LangGraph dependency"""
    
    processor = LiteDataProcessingAgent()
    retriever = LiteRetrieverAgent(k=5)
    curator = CuratorAgent()
    planner = OpenAIPlannerAgent()
    validator = ValidatorAgent()
    
    def run_pipeline(state: GraphState) -> GraphState:
        """Execute agents sequentially"""
        state = processor(state)
        state = retriever(state)
        state = curator(state)
        state = planner(state)
        state = validator(state)
        return state
    
    return {"invoke": run_pipeline}

# For compatibility with backend.py
def build_graph(**kwargs):
    """Compatibility wrapper"""
    return build_lite_graph()

GraphState = GraphState  # Export for type hints

