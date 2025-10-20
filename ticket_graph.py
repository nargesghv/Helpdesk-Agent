# ticket_graph.py
# Multi-agent LangGraph RAG for Helpdesk Ticket Assistance
#
# Pipeline:
#   New Ticket → Load Old Tickets → (strict same-category) Search → Curate Evidence
#   → LLM Plan (Groq Llama 3.1) → Validate → Save Output
#
# Rules:
#   - Strict same-category retrieval (fallback to all if new ticket has no category).
#   - NO effectiveness anywhere in outputs.
#   - In #Evidence, show Resolved: True/False/Unknown for each source ticket.
#   - Suggested direction bullets: no effectiveness labels.

import os, re, json, glob, time, argparse, traceback, unicodedata, hashlib, difflib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, TypedDict, Optional

import numpy as np
import pandas as pd

# Retrieval
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Orchestration
from langgraph.graph import StateGraph, END

# -------------------- Utilities --------------------

_STD_MAP = {
    "ticket_id": ["ticket id", "id", "ticketid", "ticket-id"],
    "issue": ["issue", "title", "subject, summary", "summary", "subject"],
    "description": ["description", "problem", "details", "body"],
    "resolution": ["resolution", "solution", "fix", "steps", "how resolved", "comment"],
    "category": ["category", "type", "queue", "group"],
    "date": ["date", "created", "created at", "opened"],
    "agent name": ["agent name", "agent", "assignee, owner", "owner"],
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

    # Title-case aliases used downstream
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

def _to_py_bool(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "y", "resolved", "closed"}:
        return True
    if s in {"false", "0", "no", "n", "open"}:
        return False
    return None

def _bool_to_label(b) -> str:
    return "True" if b is True else "False" if b is False else "Unknown"

# JSON default to safely serialize exotic types
import datetime as _dt
import pathlib as _pl
import uuid as _uuid

def _json_default(o):
    try:
        import numpy as _np
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.bool_,)):
            return bool(o)
        if isinstance(o, (_np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    try:
        import pandas as _pd
        if isinstance(o, _pd.Timestamp):
            return o.isoformat()
        if o is _pd.NA:
            return None
    except Exception:
        pass
    if isinstance(o, (_dt.datetime, _dt.date)):
        return o.isoformat()
    if isinstance(o, (_pl.Path, _uuid.UUID)):
        return str(o)
    if isinstance(o, set):
        return list(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)

# -------------------- State --------------------

class GraphState(TypedDict):
    data_dir: str
    kb_df: Optional[pd.DataFrame]
    raw_df: Optional[pd.DataFrame]
    faiss_index: Optional[Any]
    embeddings: Optional[np.ndarray]
    bm25: Optional[BM25Okapi]
    bm25_tokens: Optional[List[List[str]]]
    embed_model_name: str
    artifacts_dir: str
    ready: bool

    ticket: Dict[str, Any]
    candidates: List[Tuple[int, float]]
    evidence: List[Dict[str, Any]]
    plan: str
    logs: List[str]

# -------------------- DataProcessingAgent --------------------

@dataclass
class DataProcessingAgent:
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    artifacts_dir: str = "./artifacts"
    force_rebuild: bool = False
    shared_embedder: Optional[SentenceTransformer] = None

    def _try_load_cache(self):
        idx_path = os.path.join(self.artifacts_dir, "kb.index")
        kb_path  = os.path.join(self.artifacts_dir, "kb.parquet")
        if not (os.path.exists(idx_path) and os.path.exists(kb_path)):
            return None
        try:
            kb = pd.read_parquet(kb_path)
            index = faiss.read_index(idx_path)
            tokens = [t.lower().split() for t in kb["canonical"].tolist()]
            bm25 = BM25Okapi(tokens)
            return (kb.reset_index(drop=True), index, bm25, tokens)
        except Exception:
            return None

    def _save_cache(self, kb_df, faiss_index):
        try:
            os.makedirs(self.artifacts_dir, exist_ok=True)
            faiss.write_index(faiss_index, os.path.join(self.artifacts_dir, "kb.index"))
            kb_df.to_parquet(os.path.join(self.artifacts_dir, "kb.parquet"))
        except Exception:
            pass

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

    def _build_indexes(self, kb_df: pd.DataFrame):
        embedder = self.shared_embedder or SentenceTransformer(self.embed_model_name)
        X = embedder.encode(
            kb_df["canonical"].tolist(),
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True
        )
        X = np.asarray(X, dtype="float32")
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        tokens = [t.lower().split() for t in kb_df["canonical"].tolist()]
        bm25 = BM25Okapi(tokens)
        return index, X, bm25, tokens

    def __call__(self, state: GraphState) -> GraphState:
        logs = state.get("logs", [])
        if state.get("ready") and not self.force_rebuild:
            logs.append("DataProcessingAgent: using existing artifacts")
            state["logs"] = logs
            return state

        data_dir = state["data_dir"]
        logs.append(f"DataProcessingAgent: loading from {data_dir}")

        if not self.force_rebuild and os.path.exists(self.artifacts_dir):
            cached = self._try_load_cache()
            if cached:
                kb, faiss_index, bm25, tokens = cached
                raw = kb
                logs.append("Cache hit: loaded KB + FAISS from artifacts/")
                state.update({
                    "raw_df": raw, "kb_df": kb,
                    "faiss_index": faiss_index, "embeddings": None,
                    "bm25": bm25, "bm25_tokens": tokens,
                    "embed_model_name": self.embed_model_name,
                    "ready": True, "logs": logs
                })
                return state

        raw = self._load_sources(data_dir)
        raw = self._unify(raw)
        # Use ALL tickets (resolved or not)
        kb = raw.copy()
        kb, removed = self._dedupe(kb)
        logs.append(f"Loaded rows: {len(raw)} | KB rows: {len(kb)} | Deduped removed: {removed}")

        if len(kb) == 0:
            logs.append("KB is empty.")
            state.update({
                "raw_df": raw, "kb_df": kb, "faiss_index": None, "embeddings": None,
                "bm25": None, "bm25_tokens": None, "embed_model_name": self.embed_model_name,
                "ready": True, "logs": logs
            })
            return state

        logs.append("Building indexes (MiniLM + FAISS + BM25)…")
        t0 = time.time()
        faiss_index, embeddings, bm25, tokens = self._build_indexes(kb)
        logs.append(f"Indexes built in {time.time() - t0:.1f}s")

        self._save_cache(kb.reset_index(drop=True), faiss_index)

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

# -------------------- HybridIndex --------------------

@dataclass
class HybridIndex:
    kb_df: pd.DataFrame
    faiss_index: Any
    bm25: BM25Okapi
    embedder: SentenceTransformer

    def search(self, query_text: str, k: int = 8, alpha: float = 0.6,
               category: Optional[str] = None) -> List[Tuple[int, float]]:
        qv = self.embedder.encode([query_text.lower()], normalize_embeddings=True)
        D, I = self.faiss_index.search(np.array(qv, dtype="float32"), k * 10)
        sem_hits = [(i, s) for i, s in zip(I[0].tolist(), D[0].tolist()) if i >= 0]
        sem_vals = np.array([s for _, s in sem_hits]) if sem_hits else np.array([0.0])
        smin, smax = float(sem_vals.min()), float(sem_vals.max())
        sem_norm = {idx: (score - smin) / (smax - smin + 1e-9) for idx, score in sem_hits}

        bm_arr = self.bm25.get_scores(query_text.lower().split())
        bmin, bmax = float(np.min(bm_arr)), float(np.max(bm_arr))
        bm_norm = (bm_arr - bmin) / (bmax - bmin + 1e-9)

        bm_top = np.argsort(-bm_arr)[:k*10].tolist()
        cand = set([i for i, _ in sem_hits] + bm_top)

        if category and _norm(category):
            cat_norm = _norm(category).lower()
            def same_cat(i):
                return _norm(self.kb_df.iloc[int(i)]["Category"]).lower() == cat_norm
            cand = {i for i in cand if same_cat(i)}

        scored = []
        for idx in cand:
            s = sem_norm.get(idx, 0.0)
            b = float(bm_norm[idx])
            scored.append((idx, alpha*s + (1-alpha)*b))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

# -------------------- Agents --------------------

class RetrieverAgent:
    def __init__(self, embedder: SentenceTransformer, alpha=0.6, k=5):
        self.embedder = embedder
        self.alpha, self.k = alpha, k

    def __call__(self, state: GraphState) -> GraphState:
        kb = state["kb_df"]; fa = state["faiss_index"]; bm = state["bm25"]
        if kb is None or len(kb) == 0 or fa is None or bm is None:
            state["candidates"] = []
            state["logs"].append("RetrieverAgent: KB empty or artifacts missing; skipping retrieval")
            return state

        t = state["ticket"]
        q_title = _norm(t.get("Issue") or t.get("Title"))
        q_desc  = _norm(t.get("Description"))
        q = _pii_redact(f"Title: {q_title}\nProblem: {q_desc}")
        cat = _norm(t.get("Category", ""))

        idx = HybridIndex(kb, fa, bm, self.embedder)
        state["candidates"] = idx.search(q, k=self.k, alpha=self.alpha, category=cat if cat else None)
        state["logs"].append(
            f"RetrieverAgent: {len(state['candidates'])} candidates"
            + (f" (category='{cat}')" if cat else " (no category filter)")
        )
        return state

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
                # Carry through original Resolved value from source ticket
                "Resolved": r.get("Resolved", ""),
            })
        state["evidence"] = evid
        state["logs"].append(f"CuratorAgent: prepared {len(evid)} evidence rows")
        return state

# -------- PlannerAgent (Groq Llama-3.1) --------
from groq import Groq

class PlannerAgent:
    """
    Uses Groq Chat Completions API with Meta Llama 3.1
    Set GROQ_API_KEY in your environment.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant", deterministic: bool = False):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Create a key at https://console.groq.com/keys and export it.")
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.deterministic = deterministic
        print(f"✅ Using Groq model: {model} | deterministic={deterministic}")

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
            "6) If any evidence shows Resolved: False or an unsuccessful attempt, call it out explicitly and avoid recommending that step unless justified.\n"
            "7) End with exactly one caveat bullet: '- Caveat: ...'\n"
            "8) No text outside the bullets"
        )
        user = (
            f"NEW TICKET\nTitle: {q_title}\nProblem: {q_desc}\n\n"
            f"SIMILAR OLD TICKETS (same category when provided):\n{ev_str}\n\n"
            f"Generate the bullets now, following the STRICT RULES exactly."
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    def __call__(self, state: GraphState) -> GraphState:
        if not state.get("evidence"):
            state["plan"] = "- Caveat: No similar tickets found; gather diagnostics or escalate."
            state["logs"].append("PlannerAgent: no evidence; emitted caveat-only")
            return state

        messages = self._messages(state["ticket"], state["evidence"])

        temperature = 0.0 if self.deterministic else 0.2
        top_p = 1.0 if self.deterministic else 0.9

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
                m = re.search(r"(\n?- .*)", text, flags=re.S)
                state["plan"] = (m.group(1).strip() if m else text) or "- Caveat: No guidance."
                state["logs"].append(f"PlannerAgent: plan generated via Groq ({self.model})")
                return state
            except Exception as e:
                err = str(e)
                if attempt < 3 and any(tok in err.lower() for tok in ("429", "rate", "timeout", "temporar")):
                    time.sleep(backoff); backoff *= 2
                    continue
                state["plan"] = "- Caveat: Could not generate guidance due to API error."
                state["logs"].append(f"PlannerAgent: API error - {e}")
                return state

class ValidatorAgent:
    """Format bullets, de-dup similar, enforce exactly one Caveat."""
    def __init__(self, sim_threshold: float = 0.85, max_len: int = 400):
        self.sim_threshold = sim_threshold
        self.max_len = max_len
        self._stop = set("the a an of to in on for with and or if is are be by from at as into over under".split())

    def _extract_citations(self, line: str):
        cites = re.findall(r"\[#([A-Za-z0-9_-]+)\]", line)
        text = re.sub(r"\s*\[#([A-Za-z0-9_-]+)\]\s*", "", line).strip()
        return text, cites

    def _normalize_for_match(self, line: str):
        if line.startswith("- "): line = line[2:]
        text, _ = self._extract_citations(line)
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        toks = [t for t in text.split() if t not in self._stop]
        return " ".join(toks)

    def _merge_citations(self, base_line: str, new_cites):
        text, cites = self._extract_citations(base_line)
        merged = list(dict.fromkeys(cites + new_cites))
        if merged:
            text = text.rstrip(". ")
            text = f"{text} " + " ".join(f"[#{cid}]" for cid in merged)
        return text

    def __call__(self, state: GraphState) -> GraphState:
        plan = state.get("plan", "").strip()
        if not plan:
            state["plan"] = "- Caveat: No guidance could be generated from current evidence."
            state["logs"].append("ValidatorAgent: empty plan fallback")
            return state

        lines = [ln.strip() for ln in plan.splitlines() if ln.strip()]
        lines = ["- " + re.sub(r"^-+\s*", "", ln) if not ln.startswith("- ") else ln for ln in lines]

        normal_lines, caveats = [], []
        for ln in lines:
            if re.match(r"^-+\s*caveat:\s*", ln, flags=re.I):
                caveats.append(ln)
            else:
                normal_lines.append(ln)

        deduped = []
        for ln in normal_lines:
            norm = self._normalize_for_match(ln)
            _, cites = self._extract_citations(ln)
            merged = False
            for i, ex in enumerate(deduped):
                ratio = difflib.SequenceMatcher(None, norm, self._normalize_for_match(ex)).ratio()
                if ratio >= self.sim_threshold:
                    deduped[i] = "- " + self._merge_citations(deduped[i][2:], cites)
                    merged = True
                    break
            if not merged:
                deduped.append(ln)

        caveat = caveats[0] if caveats else "- Caveat: Validate environment specifics (OS, versions, permissions) before applying steps."

        out_lines = []
        for ln in deduped + [caveat]:
            if len(ln) > self.max_len:
                ln = ln[: self.max_len - 1] + "…"
            out_lines.append(ln)

        state["plan"] = "\n".join(out_lines)
        state["logs"].append("ValidatorAgent: plan validated (de-duplicated & citations merged)")
        return state

# -------------------- Graph Builder --------------------

def build_graph(alpha=0.6, k=5, groq_model="llama-3.1-8b-instant", deterministic=False, force_rebuild: bool = False) -> StateGraph:
    g = StateGraph(GraphState)

    shared_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    processor = DataProcessingAgent(shared_embedder=shared_embedder, force_rebuild=force_rebuild)
    retriever =  RetrieverAgent(shared_embedder, alpha=alpha, k=k)
    curator   =  CuratorAgent()
    planner   =  PlannerAgent(model=groq_model, deterministic=deterministic)
    validator =  ValidatorAgent()

    g.add_node("process_data", processor)
    g.add_node("retrieve",     retriever)
    g.add_node("curate",       curator)
    g.add_node("plan",         planner)
    g.add_node("validate",     validator)

    g.set_entry_point("process_data")
    g.add_edge("process_data", "retrieve")
    g.add_edge("retrieve", "curate")
    g.add_edge("curate", "plan")
    g.add_edge("plan", "validate")
    g.add_edge("validate", END)
    return g

# -------------------- I/O Helpers --------------------

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
            if low.endswith(".json"):
                df = pd.read_json(fp)
            else:
                continue

        df = _standardize_columns(df)
        for _, row in df.iterrows():
            tickets.append({
                "Issue": _norm(row.get("Issue", "")),
                "Description": _norm(row.get("Description", "")),
                "Category": _norm(row.get("Category", "")),
            })
    return tickets

def _write_outputs(out_dir: str, ticket_idx: int, plan: str, evidence: List[Dict[str, Any]], logs: List[str]):
    os.makedirs(out_dir, exist_ok=True)

    out_json = {"direction_bullets": plan, "evidence": evidence, "logs": logs}
    with open(os.path.join(out_dir, f"result_{ticket_idx:03d}.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2, default=_json_default)

    md = ["# Suggested direction", plan, "\n# Evidence"]
    for e in evidence:
        title = e.get("title") or e.get("Issue") or ""
        md.append(
            f"- **{e.get('ticket_id','')}** — *{title}* — **Category:** {e.get('Category','')} — "
            f"*Resolution:* {e.get('Resolution','')} — **Resolved:** {e.get('Resolved','')}"
        )
    md.append("\n# Logs")
    md.extend(f"- {l}" for l in logs)
    with open(os.path.join(out_dir, f"result_{ticket_idx:03d}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

# -------------------- Runner --------------------

def run_once(data_dir: str, input_dir: str, out_dir: str, alpha=0.6, k=5,
             groq_model="llama-3.1-8b-instant", deterministic=False, force_rebuild: bool = False):
    graph = build_graph(alpha=alpha, k=k, groq_model=groq_model, deterministic=deterministic, force_rebuild=force_rebuild).compile()

    base_state: GraphState = {
        "data_dir": data_dir,
        "kb_df": None, "raw_df": None,
        "faiss_index": None, "embeddings": None,
        "bm25": None, "bm25_tokens": None,
        "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "artifacts_dir": "./artifacts",
        "ready": False,
        "ticket": {"Issue":"", "Description":"", "Category":""},
        "candidates": [], "evidence": [],
        "plan": "", "logs": []
    }

    new_tickets = _read_new_tickets(input_dir)
    if not new_tickets:
        print(f"No new tickets found in {input_dir}")
        return

    for i, t in enumerate(new_tickets, start=1):
        try:
            base_state["ticket"] = t
            result = graph.invoke(base_state)

            evidence_cards = [{
                "ticket_id": r["ticket_id"],
                "title": r["Issue"],
                "Category": r["Category"],
                "Resolution": r["Resolution"],
                "Resolved": r["Resolved"],  # True/False/Unknown in evidence
                "snippet": (r["Description"][:280] + "…") if len(r["Description"]) > 280 else r["Description"],
                "Date": r.get("Date", "")
            } for r in result.get("evidence", [])]

            _write_outputs(out_dir, i, result["plan"], evidence_cards, result["logs"])

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
            err_file = os.path.join(out_dir, f"error_{i:03d}.txt")
            with open(err_file, "w", encoding="utf-8") as ef:
                ef.write(f"Ticket: {t}\n\nError: {e}\n\n{traceback.format_exc()}")
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",  default="Data",   help="Folder with old tickets (csv/xlsx/json)")
    ap.add_argument("--input_dir", default="input",  help="Folder with new tickets (csv/xlsx/json)")
    ap.add_argument("--out_dir",   default="output", help="Folder to write results")
    ap.add_argument("--alpha",     type=float, default=0.6, help="Hybrid weight (semantic emphasis)")
    ap.add_argument("--k",         type=int,   default=5,   help="Top-k retrieved")
    ap.add_argument("--groq_model", default="llama-3.1-8b-instant",
                    help="Groq Llama 3.1 model id (e.g., llama-3.1-8b-instant or llama-3.1-70b-versatile)")
    ap.add_argument("--deterministic", action="store_true",
                    help="Use temperature=0.0/top_p=1.0 for reproducible bullets")
    ap.add_argument("--rebuild", action="store_true", help="Ignore cached artifacts and rebuild KB/indexes")
    args = ap.parse_args()
    run_once(args.data_dir, args.input_dir, args.out_dir,
             alpha=args.alpha, k=args.k,
             groq_model=args.groq_model, deterministic=args.deterministic,
             force_rebuild=args.rebuild)
