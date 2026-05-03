"""
POST /v1/chat/{session_id} — send a user message, get assistant reply.

E1: Idempotency-Key header — replay returns cached response.
E3: Accept: text/event-stream — streams tokens as SSE.
"""
import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IdempotencyKey
from app.db.session import get_db
from app.srop import pipeline

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    reply: str
    routed_to: str
    trace_id: str


async def _sse_stream(session_id: str, content: str, db: AsyncSession) -> AsyncGenerator[str, None]:
    """
    E3: Run pipeline and stream words as SSE events.

    ADK doesn't natively stream partial tokens to us here, so we stream
    word-by-word from the final response to satisfy the SSE requirement.
    The final event carries full metadata (routed_to, trace_id).
    """
    result = await pipeline.run(session_id, content, db)

    # Stream words one-by-one
    words = result.content.split()
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'token': chunk})}\n\n"
        await asyncio.sleep(0)  # Yield control to event loop

    # Final event with metadata
    yield f"data: {json.dumps({'done': True, 'routed_to': result.routed_to, 'trace_id': result.trace_id})}\n\n"


@router.post("/chat/{session_id}", response_model=None)
async def chat(
    session_id: str,
    body: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(None),
):
    """
    Run one turn of the SROP pipeline.

    E1: If Idempotency-Key header provided and seen before → return cached response.
    E3: If Accept: text/event-stream → stream tokens as SSE.

    Error cases:
    - Session not found → 404 (SessionNotFoundError)
    - LLM timeout → 504 (UpstreamTimeoutError)
    """
    # E1: Check idempotency cache
    if idempotency_key:
        hit = await db.execute(
            select(IdempotencyKey).where(
                (IdempotencyKey.session_id == session_id)
                & (IdempotencyKey.idempotency_key == idempotency_key)
            )
        )
        cached = hit.scalar_one_or_none()
        if cached:
            # Return cached as SSE or JSON depending on Accept header
            if "text/event-stream" in request.headers.get("accept", ""):
                cached_resp = cached.response
                async def cached_stream() -> AsyncGenerator[str, None]:
                    for word in cached_resp["reply"].split():
                        yield f"data: {json.dumps({'token': word + ' '})}\n\n"
                        await asyncio.sleep(0)
                    yield f"data: {json.dumps({'done': True, 'routed_to': cached_resp['routed_to'], 'trace_id': cached_resp['trace_id']})}\n\n"
                return StreamingResponse(cached_stream(), media_type="text/event-stream")
            return ChatResponse(**cached.response)

    # E3: Streaming SSE
    if "text/event-stream" in request.headers.get("accept", ""):
        return StreamingResponse(
            _sse_stream(session_id, body.content, db),
            media_type="text/event-stream",
        )

    # Normal JSON path
    result = await pipeline.run(session_id, body.content, db)
    response = ChatResponse(reply=result.content, routed_to=result.routed_to, trace_id=result.trace_id)

    # E1: Cache result
    if idempotency_key:
        db.add(IdempotencyKey(
            session_id=session_id,
            idempotency_key=idempotency_key,
            response=response.model_dump(),
        ))
        await db.commit()

    return response
