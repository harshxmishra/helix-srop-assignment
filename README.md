# Helix SROP — Stateful RAG Orchestration Pipeline

An AI Support Concierge for the Helix B2B SaaS platform. Handles product knowledge questions via RAG and account queries via internal tools — in a single stateful conversation that survives process restarts.

---

## Setup

```bash
git clone <repo>
cd helix-srop

pip install -e ".[dev]"

cp .env.example .env
# Edit .env — set GOOGLE_API_KEY

# Ingest product docs into vector store
python -m app.rag.ingest --path docs/

# Start server
uvicorn app.main:app --reload
```

Server runs at `http://localhost:8000`. Swagger UI at `/docs`.

### Docker

```bash
cp .env.example .env  # set GOOGLE_API_KEY
docker compose up
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/sessions` | Create session (`user_id`, `plan_tier`) |
| `POST` | `/v1/chat/{session_id}` | Send message → `reply`, `routed_to`, `trace_id` |
| `GET`  | `/v1/traces/{trace_id}` | Inspect turn: tool calls, chunks, latency |
| `GET`  | `/healthz` | Health check |

### Quick test

```bash
SESSION=$(curl -s -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","plan_tier":"pro"}' | jq -r .session_id)

# Knowledge query
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content":"How do I rotate a deploy key?"}' | jq .

# Account query
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content":"Show my recent builds"}' | jq .

# Streaming (E3)
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"content":"What is the artifact registry?"}'

# Idempotent replay (E1)
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: req-abc-001" \
  -d '{"content":"How do I rotate a deploy key?"}' | jq .
```

---

## Architecture

```
POST /v1/chat/{session_id}
    │
    ▼
Pipeline
  ├─ Load SessionState from SQLite (sessions.state JSON)
  ├─ E5: Guardrails check (out-of-scope refusal, PII redaction)
  ├─ E1: Idempotency-Key cache lookup
  ├─ Build root_agent (state injected into instruction)
  │
  └─ ADK InMemoryRunner
       │
       ├─ knowledge_agent ──► search_docs ──► ChromaDB
       │                        (E4: LLM reranker on top-20)
       ├─ account_agent ────► get_recent_builds
       │                    ► get_account_status
       └─ escalation_agent ► create_ticket ──► tickets table

  ├─ Collect events: tool calls, routing decision, final text
  ├─ Write AgentTrace to DB
  ├─ Update SessionState (turn_count, last_agent)
  └─ Return PipelineResult

Storage:
  SQLite: sessions, messages, agent_traces, idempotency_keys, tickets
  ChromaDB: product doc chunks (./chroma_db)
```

---

## Design Decisions

### State Persistence — Pattern 3 (Instruction Injection)

`SessionState` (`user_id`, `plan_tier`, `last_agent`, `turn_count`) stored as JSON in `sessions.state` column. Each turn: load from DB → inject into root agent instruction → save updated state.

**Why:** Simplest. No ADK session plumbing. State survives restart because SQLite persists on disk. Small enough (<200 bytes) to fit in instruction without wasting context.

**Tradeoff:** Stateless ADK session per turn (no message history re-hydration). Sufficient for support Q&A; not ideal for complex multi-turn narratives.

### Chunking — Heading-Aware + Sentence Sub-Chunking

Split on `##` / `###` markdown headings. Sections longer than 512 chars sub-chunked by sentence with 1-sentence overlap.

**Why:** Product docs are structured by feature area. Heading splits preserve semantic boundaries better than fixed-size. Sentence sub-chunking avoids breaking mid-thought on long sections.

### Embeddings — Google gemini-embedding-001

Same model at ingest time (`RETRIEVAL_DOCUMENT`) and query time (`RETRIEVAL_QUERY`). Both calls via `google.genai` SDK.

### Routing — ADK AgentTool (LLM tool-selection)

Root agent has `knowledge_agent`, `account_agent`, and `escalation_agent` wrapped as `AgentTool`. The LLM selects the correct tool — no regex or keyword matching in routing logic.

---

## Extensions

| Ext | Status | Description |
|-----|--------|-------------|
| E1 | ✅ | **Idempotency** — `Idempotency-Key` header; replay returns cached response |
| E2 | ✅ | **Escalation agent** — `create_ticket()` → `tickets` table; ticket ID in state |
| E3 | ✅ | **Streaming SSE** — `Accept: text/event-stream` streams tokens, final event has metadata |
| E4 | ✅ | **Reranking** — retrieve top-20, LLM-as-judge scores relevance 1–10, return top-k |
| E5 | ✅ | **Guardrails** — out-of-scope refusal; PII (email/phone/SSN/card) redacted in logs |
| E6 | ✅ | **Docker** — `Dockerfile` + `docker-compose.yml`, volumes for DB + ChromaDB |
| E7 | ✅ | **Eval harness** — `python eval/run_eval.py`; 14 labeled queries, reports routing accuracy |

---

## Testing

```bash
pytest -q
# Expected: 6 passed, 1 skipped

# Run eval harness (server must be running)
python eval/run_eval.py
```

Tests cover:
- Session creation
- Multi-turn state persistence (plan_tier survives turn 2)
- 404 on unknown session
- E1 idempotency key cache
- E5 guardrails refusal
- RAG chunker unit test

The search_docs integration test (`test_search_docs_returns_results_with_chunk_ids`) requires the vector store to be seeded first and is skipped on clean clone.

---

## Known Limitations

- **Account tools return mock data.** `get_recent_builds` and `get_account_status` return static fixtures. The wiring (ADK tool invocation) is real; the data source is not.
- **No full message history re-hydration.** Each turn sees only `SessionState` (user_id, plan_tier, etc.), not prior conversation turns. Context from earlier turns is not in scope.
- **E3 streams word-by-word.** ADK does not expose partial token streaming; words are streamed from the completed response. True streaming requires a streaming-capable ADK runner.
- **E4 reranking adds one extra LLM call.** This increases latency per knowledge query. Acceptable for correctness; could be made optional via config flag.

---

## Time Spent

| Phase | Time |
|-------|------|
| Setup, deps, ADK API exploration | 25 min |
| RAG ingest + search_docs | 35 min |
| ADK agents (knowledge, account, root) | 30 min |
| pipeline.py (state, events, traces, timeout) | 40 min |
| API routes (sessions, chat, traces, errors) | 20 min |
| Tests + fixtures | 20 min |
| Extensions E1–E7 | 60 min |
| README | 15 min |
| **Total** | **~245 min (~4 hours)** |
