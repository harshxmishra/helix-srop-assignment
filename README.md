# Helix SROP — AI Support Concierge

Stateful RAG orchestration pipeline with multi-agent routing. Knowledge questions → RAG retrieval. Account queries → database tools. State survives process restart.

**Status:** Core 70 pts complete. `pytest -q` passes (4 passed, 1 skipped).

## Setup

```bash
git clone <your-repo>
cd helix-srop
pip install -e ".[dev]"
echo "GOOGLE_API_KEY=sk_..." > .env
python -m app.rag.ingest --path docs/
uvicorn app.main:app --reload
```

Ingest takes ~2 min (batches embeddings). Server starts on `http://localhost:8000`.

## Quick Test

```bash
# Create session
SESSION=$(curl -s -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u_demo","plan_tier":"pro"}' | jq -r .session_id)

# Knowledge query
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content":"How do I rotate a deploy key?"}' | jq .

# Follow-up (state persists)
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content":"What is my plan tier?"}' | jq .
```

## Architecture

```
POST /v1/chat/{session_id}
    ↓
Pipeline:
  1. Load SessionState from sessions.state (JSON, SQLite)
  2. Build root_agent with state injected into instruction
  3. Run ADK orchestrator (InMemoryRunner + InMemorySessionService)
  4. Collect events: tool calls, routing decision, final text
  5. Write AgentTrace + Message rows
  6. Update state (turn_count, last_agent)
  ↓
Routes via AgentTool (LLM-driven, not string parsing):
  ├── knowledge_agent
  │   └── search_docs (ChromaDB vector store)
  └── account_agent
      ├── get_recent_builds (mock)
      └── get_account_status (mock)
```

## Design Decisions

### State Persistence: Pattern 3 (Instruction Injection)

Load `SessionState` from DB (JSON column) → stringify into root agent's instruction each turn → save updated state back to DB.

**Why:** Simplest. State is small (user_id, plan_tier, last_agent, turn_count). No ADK session plumbing needed. Survives restart because SQLite persists.

**Tradeoff:** Uses a bit more context per turn but avoids re-hydrating full message history. For short sessions (<10 turns) negligible.

### Chunking Strategy: Heading-Aware + Sentence Sub-Chunking

Split on `##` and `###` markdown headings. Long sections (>512 chars) further split by sentence with 1-sentence overlap.

**Why:** Product docs are structured by feature/topic. Heading-aware splits preserve semantic boundaries. Sentence sub-chunking handles long sections without breaking mid-idea. Overlap at boundaries maintains context.

**Example:** "## Deploy Keys" → chunk_001 ("## Deploy Keys\n\nA deploy key..."), chunk_002 ("...stored securely. You can rotate..."), etc.

**Justification in rag-guide.md:** Better retrieval quality than fixed-size; more coherent than pure sentence splitting.

### Vector Store: ChromaDB PersistentClient

In-memory for demo speed, persistent to `./chroma_db` for restarts.

**Why:** Already in deps. Simple API. Works for assignment scale (≈2K docs chunks). Sync client → wrap calls in `asyncio.to_thread`.

## Known Limitations

- **Mock account data:** `get_recent_builds` and `get_account_status` return static data (not querying real DB). Wiring is correct; implementation is stub.
- **No conversation context re-use:** Each turn starts fresh with state context only (not full message history). Suitable for support queries; might miss multi-turn narrative context.
- **Score threshold:** Retrieved chunks below 0.3 similarity are filtered; unanswerable queries fall back to "I don't have documentation on that."
- **No extensions:** E1–E7 not implemented (idempotency, escalation, SSE, reranking, guardrails, Docker, eval).
- **ADK event extraction:** Routed_to inferred from `event.author` or first tool call name; fallback to "smalltalk" if no tools fire. Defensive but not production-hardened.

## Assumptions Made

- `google.genai` SDK (not deprecated `google.generativeai`) available and properly versioned.
- SQLite async requires `greenlet` — installed via deps.
- Docs in `docs/*.md` are Markdown with optional YAML frontmatter (product_area, title, tags).
- Test environment has in-memory SQLite; production uses persistent SQLite file.

## What I'd Do With More Time

1. **E2 Escalation agent** — `create_ticket(summary, priority)` writes to `tickets` table, ticket_id stored in state.
2. **E1 Idempotency** — `Idempotency-Key` header + DB cache of responses by (session_id, key).
3. **Reranking** (E4) — LLM-as-judge rerank top-20 retrieval to top-5 before sending to root agent.
4. **Guardrails** (E5) — Refusal on out-of-scope queries; PII redaction in logs.
5. **Real DB tools** — Replace mock account tools with actual queries to a build history table.

## Time Spent

| Phase | Time |
|-------|------|
| Setup + deps + ADK API exploration | 25 min |
| RAG ingest + search_docs | 35 min |
| ADK agents (knowledge + account + root) | 30 min |
| pipeline.py (state, events, traces, timeout) | 40 min |
| API routes (sessions, chat, traces, errors) | 20 min |
| Tests + mock_adk fixture | 20 min |
| SDK migration (google.generativeai → google.genai) | 15 min |
| README + memory | 10 min |
| **Total** | ~195 min (~3.3 hours) |

Stayed within 3–4 hour estimate. Debugged ADK event API variations and SDK deprecation; no major surprises.

## Testing

```bash
pytest -q
# 4 passed, 1 skipped
# - test_create_session: POST /v1/sessions works
# - test_knowledge_query_routes_correctly: multi-turn state persistence
# - test_session_not_found_returns_404: error handling
# - test_chunker_produces_non_empty_chunks: RAG unit test
# - test_search_docs_returns_results_with_chunk_ids: skipped (needs seeded Chroma)
```

Run integration test (mocked LLM): `pytest tests/test_api.py::test_knowledge_query_routes_correctly -v`

Run chunker test: `pytest tests/test_retriever.py::test_chunker_produces_non_empty_chunks -v`

## Extensions Completed

- [x] Core 70 pts
- [ ] E1: Idempotency
- [ ] E2: Escalation agent
- [ ] E3: Streaming SSE
- [ ] E4: Reranking
- [ ] E5: Guardrails
- [ ] E6: Docker
- [ ] E7: Eval harness

## Notes

- **State pattern choice is documented** — see pipeline.py docstring for Pattern 3 rationale.
- **Chunking strategy justified** — see rag-guide.md in docs/ and ingest.py docstring.
- **No API keys committed** — .env not in repo; configure at runtime.
- **Trace quality** — GET /v1/traces/{trace_id} returns all tool calls, chunk IDs, latency for debugging.
