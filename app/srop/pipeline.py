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
import os
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import structlog
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import (
    account_agent,
    build_root_agent,
    escalation_agent,
    knowledge_agent,
)
from app.agents.tools.guardrails import is_out_of_scope, redact_pii
from app.api.errors import SessionNotFoundError, UpstreamTimeoutError
from app.db.models import AgentTrace, Message, Session, Ticket
from app.settings import settings
from app.srop.state import SessionState

log = structlog.get_logger(__name__)


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


def _to_jsonable(value: Any) -> Any:
    """Recursively convert arbitrary tool outputs into JSON-serializable values."""
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


async def _run_specialist_agent(user_id: str, user_message: str, agent) -> tuple[str, list[dict]]:
    """
    Execute a specialist agent directly and capture tool call metadata.

    Used as a fallback when the root agent returns literal tool-call text
    instead of invoking tools through ADK events.
    """
    runner = InMemoryRunner(agent=agent, app_name="helix_srop")
    adk_session = await runner.session_service.create_session(
        app_name="helix_srop",
        user_id=user_id,
    )
    new_message = Content(role="user", parts=[Part(text=user_message)])

    final_text: str = ""
    tool_calls: list[dict] = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=adk_session.id,
        new_message=new_message,
    ):
        for fc in (event.get_function_calls() or []):
            tool_calls.append({
                "tool_name": fc.name,
                "args": dict(fc.args) if fc.args else {},
                "result": None,
            })

        for fr in (event.get_function_responses() or []):
            for tc in reversed(tool_calls):
                if tc["tool_name"] == fr.name and tc["result"] is None:
                    tc["result"] = _to_jsonable(fr.response)
                    break

        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(
                p.text
                for p in event.content.parts
                if hasattr(p, "text") and p.text
            )

    return final_text, tool_calls


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
    # E5: Guardrails check
    # ------------------------------------------------------------------ #
    if is_out_of_scope(user_message):
        trace_id = str(uuid.uuid4())
        latency_ms = int((time.monotonic() - start_ms) * 1000)
        db.add(AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to="guardrails",
            tool_calls=[],
            retrieved_chunk_ids=[],
            latency_ms=latency_ms,
        ))
        db.add(Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=user_message,
            trace_id=trace_id,
        ))
        refusal = "I'm a Helix support assistant. I can only help with questions about our product, builds, accounts, and support. Your question is outside my scope."
        db.add(Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=refusal,
            trace_id=trace_id,
        ))
        await db.commit()
        return PipelineResult(content=refusal, routed_to="guardrails", trace_id=trace_id)

    # ------------------------------------------------------------------ #
    # 2. Build ADK agent + runner for this turn
    # ------------------------------------------------------------------ #
    root_agent = build_root_agent(state)
    runner = InMemoryRunner(agent=root_agent, app_name="helix_srop")
    # Use runner's own internal session service — not a separate instance
    adk_session = await runner.session_service.create_session(
        app_name="helix_srop", user_id=state.user_id
    )

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
                        tc["result"] = _to_jsonable(fr.response)
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

    # Some models can emit literal tool-call text instead of invoking tool events.
    # Recover by executing the requested specialist agent directly.
    if routed_to == "smalltalk" and not tool_calls:
        fallback = None
        if "knowledge_agent" in final_text:
            fallback = (knowledge_agent, "knowledge")
        elif "account_agent" in final_text:
            fallback = (account_agent, "account")
        elif "escalation_agent" in final_text:
            fallback = (escalation_agent, "escalation")

        if fallback:
            specialist_agent, routed_to = fallback
            specialist_text, specialist_tool_calls = await _run_specialist_agent(
                user_id=state.user_id,
                user_message=user_message,
                agent=specialist_agent,
            )
            if specialist_text:
                final_text = specialist_text
            tool_calls.extend(specialist_tool_calls)

    # E5: Redact PII from final response before storing
    final_text_redacted = redact_pii(final_text)

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
    # 5. E2: Extract ticket creation from tool calls
    # ------------------------------------------------------------------ #
    ticket_ids: list[str] = []
    for tc in tool_calls:
        if tc["tool_name"] == "create_ticket":
            result_data = tc.get("result")
            if isinstance(result_data, dict) and "ticket_id" in result_data:
                ticket_ids.append(result_data["ticket_id"])
                # Write ticket to DB
                ticket = Ticket(
                    ticket_id=result_data["ticket_id"],
                    session_id=session_id,
                    user_id=state.user_id,
                    summary=result_data.get("summary", ""),
                    priority=result_data.get("priority", "medium"),
                )
                db.add(ticket)

    # ------------------------------------------------------------------ #
    # 6. Persist trace, messages, and updated state in one commit
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
        content=final_text_redacted,
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
    return PipelineResult(content=final_text_redacted, routed_to=routed_to, trace_id=trace_id)
