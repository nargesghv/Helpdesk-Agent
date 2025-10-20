# Helpdesk Agent - Detailed Architecture and Workflow

This document describes the system architecture, LangGraph workflow, data pipeline, retrieval strategy, LLM planning, validation logic, and runtime behavior for the helpdesk ticket assistant.

## 1. System Goals
- Build a knowledge base from historical tickets (mixed CSV/XLSX/JSON).
- Retrieve the most similar tickets for a new issue using hybrid search (FAISS + BM25).
- Generate concise, cited troubleshooting steps via Groq Llama 3.1.
- Preserve evidence (with Resolved status) and logs for auditability.

## 2. End-to-End Flow
1) Load and unify old tickets from `Data/` (schema normalization + PII redaction).
2) Build hybrid retrieval artifacts:
   - Semantic vectors (Sentence Transformers + FAISS)
   - Keyword index (BM25)
3) For each new ticket in `input/`:
   - Retrieve similar tickets using hybrid search.
   - Curate evidence records (include Resolved = True/False/other).
   - Plan with LLM (Groq) using evidence; enforce strict bullet rules.
   - Validate bullets (formatting, deduplication, single caveat).
   - Write JSON and Markdown results to `output/`.

All steps are orchestrated by LangGraph.

## 3. LangGraph Workflow
Nodes and edges form a linear pipeline:

- process_data → retrieve → curate → plan → validate → END

Flowchart (mermaid):

```mermaid
flowchart TD
    A[Start] --> B[process_data\nDataProcessingAgent]
    B --> C[retrieve\nRetrieverAgent]
    C --> D[curate\nCuratorAgent]
    D --> E[plan\nPlannerAgent (Groq)]
    E --> F[validate\nValidatorAgent]
    F --> G[END]
```

### 3.1 State (TypedDict `GraphState`)
- Global artifacts: `kb_df`, `faiss_index`, `bm25`, `bm25_tokens`, `embed_model_name`, `ready`.
- Per-ticket: `ticket`, `candidates`, `evidence`, `plan`, `logs`.

### 3.2 Nodes
- DataProcessingAgent
  - Loads files from `Data/`
  - Standardizes schema and text
  - Builds FAISS + BM25 indexes
  - Caches artifacts to `artifacts/`
- RetrieverAgent
  - Builds query from new ticket (Issue + Description)
  - Runs hybrid search (FAISS + BM25)
  - Optional category filter (from the new ticket)
- CuratorAgent
  - Transforms raw matches into structured evidence dicts
  - Evidence includes Issue/Description/Resolution/Category/Date/Resolved
- PlannerAgent (Groq)
  - Formats prompt with new ticket + evidence (shows Resolved)
  - Instructs model to avoid steps that failed (Resolved=False)
  - Returns 3–5 bullets with citations and a single caveat
- ValidatorAgent
  - Ensures formatting, merges similar bullets, enforces one caveat, clips long lines

## 4. Data Processing
### 4.1 Column Standardization
- Handles case/alias variants: Title → Issue, Problem → Description, etc.
- Ensures canonical Title-case columns exist for downstream use.

### 4.2 Cleaning & PII Redaction
- Unicode normalization and whitespace cleanup
- Redaction of emails, IPv4 addresses, and phone numbers

### 4.3 Canonical Text Construction
- Structured text used for embeddings and BM25 indexing:
  - Title, Category, Problem, Resolution

### 4.4 Deduplication
- Stable hash on Issue|Description|Resolution; drop duplicate rows.

### 4.5 Indexing
- SentenceTransformer encodes canonical text; FAISS IndexFlatIP stores normalized vectors.
- BM25 built from tokenized canonical text.

## 5. Hybrid Retrieval
- Semantic (FAISS) and keyword (BM25) scores are normalized and combined:
  - score = α*semantic + (1-α)*bm25 (default α=0.6)
- Candidate set = top-k*10 from FAISS ∪ top-k*10 from BM25.
- Optional category constraint using the new ticket’s Category.
- Returns top-k candidates (index, score).

## 6. Evidence Curation
- For each candidate index, fetch row from KB and build evidence dict:
  - `ticket_id`, `Issue`, `Description`, `Resolution`, `Category`, `Date`, `Resolved`.
- Resolved value is passed through from source without normalization.

## 7. Planning (Groq Llama 3.1)
### 7.1 Prompt Structure
- System: strict rules (3–5 bullets, citations [#TICKET_ID], one caveat, no extra text).
- Additional rule: if evidence shows Resolved=False, call it out and avoid repeating that step unless justified.
- User: NEW TICKET (Issue + Description) and SIMILAR OLD TICKETS (Problem/Resolution/Resolved per ticket).

### 7.2 Decoding Controls
- Deterministic mode: temperature=0.0, top_p=1.0 (reproducible outputs).
- Default (non-deterministic): temperature=0.2, top_p=0.9 (slight variation).

### 7.3 Resilience
- 4 attempts with exponential backoff for transient API errors.

## 8. Validation
- Normalizes bullet prefix "- ".
- Splits caveats vs main bullets; keeps one caveat.
- Deduplicates similar bullets by semantic similarity (difflib) and merges citations.
- Clips very long lines.

## 9. Outputs
- JSON: `{ direction_bullets, evidence[], logs[] }`.
- Markdown: human-readable bullets, evidence with Category/Resolution/Resolved, and logs.

## 10. Caching and Rebuilds
- Artifacts persisted to `artifacts/` (`kb.parquet`, `kb.index`).
- Use `--rebuild` to ignore cache and rebuild KB + indexes (ensures unresolved tickets are included after schema changes).

## 11. Operational Notes
- Groq key: set `GROQ_API_KEY` in environment.
- The system does not filter by Resolved; both True and False contribute to retrieval.
- Resolved is surfaced in evidence and prompt to inform the model about unsuccessful attempts.

## 12. Quick Troubleshooting
- Seeing only Resolved=True: delete `artifacts/` or run with `--rebuild`.
- Missing columns: standardization creates reasonable defaults; review source files if values are unexpected.
- API errors: check `GROQ_API_KEY`, network, or Groq status; see logs and output error files.

## 13. Extensibility
- Add new fields to evidence (e.g., product, platform) and include in canonical text.
- Swap embedding models or change α to rebalance semantic vs keyword matching.
- Extend Validator to enforce domain-specific formatting or metadata.

---

This document reflects the current behavior of `ticket_graph.py` and how the product operates in production-like conditions.
