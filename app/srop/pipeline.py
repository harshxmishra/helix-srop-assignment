"""
SROP pipeline — called by the chat route for every user turn.

Flow:
  1. Load session from DB (raises SessionNotFoundError on miss)
  2. Deserialise SessionState from session.state JSON
  3. Build root agent with state injected into instruction (Pattern 3)
  4. Run ADK in-memory runner, collecting events with a timeout
  5. Extract: final reply text, which sub-agent ran, tool calls, chunk IDs
  6. Write AgentTrace row to DB
  7. Update SessionState (turn_count, last_agent) and persist
  8. Insert Message rows (user + assistant)
  9. Return PipelineResult

State persistence — Pattern 3:
  SessionState is stored as JSON in sessions.state (SQLite).
  Each turn loads it, injects it into the agent instruction, then saves
  the updated state back. State survives process restarts because it lives
  in the DB, not memory.
"""
import asyncio
import time
import uuid
from dataclasses import dataclass

import structlog
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import build_root_agent
from app.api.errors import SessionNotFoundError, UpstreamTimeoutError
from app.db.models import AgentTrace, Message, Session
from app.settings import settings
from app.srop.state import SessionState

log = structlog.get_logger(__name__)


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    """
    Execute one conversational turn for session_id.

    Raises:
        SessionNotFoundError: session_id does not exist in DB
        UpstreamTimeoutError: LLM did not respond within settings.llm_timeout_seconds
    """
    start_ms = time.monotonic()

    # ------------------------------------------------------------------ #
    # 1. Load session + state
    # ------------------------------------------------------------------ #
    result = await db.execute(select(Session).where(Session.session_id == session_id))
    session = result.scalar_one_or_none()
    if session is None:
        raise SessionNotFoundError(f"Session {session_id!r} not found")

    state = SessionState.from_db_dict(session.state)
    log.info("pipeline.start", session_id=session_id, turn=state.turn_count)

    # ------------------------------------------------------------------ #
    # 2. Build ADK agent + runner for this turn
    # ------------------------------------------------------------------ #
    root_agent = build_root_agent(state)
    runner = InMemoryRunner(agent=root_agent, app_name="helix_srop")
    svc = InMemorySessionService()
    adk_session = await svc.create_session(app_name="helix_srop", user_id=state.user_id)

    new_message = Content(role="user", parts=[Part(text=user_message)])

    # ------------------------------------------------------------------ #
    # 3. Collect ADK events (with timeout)
    # ------------------------------------------------------------------ #
    final_text: str = ""
    routed_to: str = "smalltalk"
    tool_calls: list[dict] = []
    chunk_ids: list[str] = []

    async def _collect() -> None:
        nonlocal final_text, routed_to

        async for event in runner.run_async(
            user_id=state.user_id,
            session_id=adk_session.id,
            new_message=new_message,
        ):
            # Capture function calls made during this turn
            for fc in (event.get_function_calls() or []):
                tool_calls.append({
                    "tool_name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                    "result": None,
                })

            # Capture function results — attach to the last matching pending call
            for fr in (event.get_function_responses() or []):
                for tc in reversed(tool_calls):
                    if tc["tool_name"] == fr.name and tc["result"] is None:
                        tc["result"] = (
                            fr.response
                            if isinstance(fr.response, (str, dict, list))
                            else str(fr.response)
                        )
                        break

            # Final response — grab text and the author agent name
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = "".join(
                        p.text
                        for p in event.content.parts
                        if hasattr(p, "text") and p.text
                    )
                author = getattr(event, "author", "") or ""
                if "knowledge" in author:
                    routed_to = "knowledge"
                elif "account" in author:
                    routed_to = "account"

        # Fallback: infer routed_to from first tool call name if still "smalltalk"
        if routed_to == "smalltalk" and tool_calls:
            first_tool = tool_calls[0]["tool_name"]
            if "knowledge" in first_tool:
                routed_to = "knowledge"
            elif "account" in first_tool:
                routed_to = "account"

    try:
        await asyncio.wait_for(_collect(), timeout=float(settings.llm_timeout_seconds))
    except asyncio.TimeoutError as exc:
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        ) from exc

    if not final_text:
        final_text = "I'm sorry, I couldn't generate a response. Please try again."

    # ------------------------------------------------------------------ #
    # 4. Extract retrieved chunk IDs from search_docs results
    # ------------------------------------------------------------------ #
    for tc in tool_calls:
        if tc["tool_name"] == "search_docs":
            result_data = tc.get("result")
            if isinstance(result_data, list):
                for item in result_data:
                    if isinstance(item, dict) and "chunk_id" in item:
                        chunk_ids.append(item["chunk_id"])

    # ------------------------------------------------------------------ #
    # 5. Persist trace, messages, and updated state in one commit
    # ------------------------------------------------------------------ #
    trace_id = str(uuid.uuid4())
    latency_ms = int((time.monotonic() - start_ms) * 1000)

    db.add(AgentTrace(
        trace_id=trace_id,
        session_id=session_id,
        routed_to=routed_to,
        tool_calls=tool_calls,
        retrieved_chunk_ids=chunk_ids,
        latency_ms=latency_ms,
    ))
    db.add(Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=user_message,
        trace_id=trace_id,
    ))
    db.add(Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=final_text,
        trace_id=trace_id,
    ))

    # Update state — turn_count and last_agent
    state.turn_count += 1
    state.last_agent = routed_to  # type: ignore[assignment]
    session.state = state.to_db_dict()

    await db.commit()

    log.info(
        "pipeline.done",
        session_id=session_id,
        trace_id=trace_id,
        routed_to=routed_to,
        latency_ms=latency_ms,
    )
    return PipelineResult(content=final_text, routed_to=routed_to, trace_id=trace_id)
