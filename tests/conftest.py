"""
Test fixtures.

Key fixtures:
- `client`: async test client with in-memory SQLite DB
- `mock_adk`: patches pipeline.run so tests don't hit the real LLM
- `db`: raw async session for direct DB inspection
"""
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.db.session import get_db
from app.main import app

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    """Create all tables before each test and drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    """Yield a raw async DB session for direct inspection in tests."""
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db: AsyncSession):
    """
    Async HTTP test client with DB overridden to in-memory SQLite.

    The same `db` session is shared so mock_adk can write traces
    that are visible to subsequent GET /v1/traces calls.
    """
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_adk(monkeypatch, db: AsyncSession):
    """
    Patch pipeline.run at the ADK boundary — tests never call a real LLM.

    Routing heuristic mirrors real intent routing:
    - knowledge questions → routed_to="knowledge", returns chunk IDs
    - plan/tier queries   → routed_to="knowledge"
    - build/account       → routed_to="account"
    - smalltalk           → routed_to="smalltalk"

    Writes an AgentTrace row so GET /v1/traces/{trace_id} succeeds.
    """
    from app.db.models import AgentTrace
    from app.srop.pipeline import PipelineResult

    async def mock_run(session_id: str, message: str, _db: AsyncSession) -> PipelineResult:
        msg = message.lower()

        if any(kw in msg for kw in ("plan", "tier", "my plan")):
            routed = "knowledge"
            reply = "Your plan is pro."
            chunks = []
        elif any(kw in msg for kw in ("rotate", "deploy key", "how do", "what is")):
            routed = "knowledge"
            reply = "To rotate a deploy key, go to Settings > Deploy Keys. [chunk_abc123]"
            chunks = ["chunk_abc123", "chunk_def456"]
        elif any(kw in msg for kw in ("build", "status", "account")):
            routed = "account"
            reply = "Your recent builds: build_001 (passed), build_002 (failed)."
            chunks = []
        else:
            routed = "smalltalk"
            reply = "Hello! How can I help you today?"
            chunks = []

        trace_id = str(uuid.uuid4())
        _db.add(AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to=routed,
            tool_calls=[],
            retrieved_chunk_ids=chunks,
            latency_ms=1,
        ))
        await _db.commit()
        return PipelineResult(content=reply, routed_to=routed, trace_id=trace_id)

    monkeypatch.setattr("app.srop.pipeline.run", mock_run)
