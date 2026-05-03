"""
POST /v1/chat/{session_id} — send a user message, get assistant reply.
Supports E1 Idempotency-Key header for replay protection.
"""
from fastapi import APIRouter, Depends, Header
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
    routed_to: str   # which sub-agent handled this turn
    trace_id: str


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(None),
) -> ChatResponse:
    """
    Run one turn of the SROP pipeline.

    E1: If Idempotency-Key header provided and we've seen it before,
    return cached response without re-running LLM.

    Error cases:
    - Session not found → 404
    - LLM timeout → 504
    """
    # E1: Check cache
    if idempotency_key:
        result = await db.execute(
            select(IdempotencyKey).where(
                (IdempotencyKey.session_id == session_id)
                & (IdempotencyKey.idempotency_key == idempotency_key)
            )
        )
        cached = result.scalar_one_or_none()
        if cached:
            return ChatResponse(**cached.response)

    # Run pipeline (LLM call happens here)
    result = await pipeline.run(session_id, body.content, db)
    response = ChatResponse(reply=result.content, routed_to=result.routed_to, trace_id=result.trace_id)

    # E1: Cache result if idempotency_key provided
    if idempotency_key:
        cache_entry = IdempotencyKey(
            session_id=session_id,
            idempotency_key=idempotency_key,
            response=response.model_dump(),
        )
        db.add(cache_entry)
        await db.commit()

    return response
