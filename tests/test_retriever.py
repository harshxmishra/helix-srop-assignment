"""
Unit tests for RAG retrieval.

test_chunker_produces_non_empty_chunks — pure unit test, no external deps.
test_search_docs_returns_results_with_chunk_ids — integration test, skipped
  unless vector store is seeded (run: python -m app.rag.ingest --path docs/).
"""
import pytest


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings."""
    from app.rag.ingest import chunk_markdown

    text = "# Header\n\nSome content.\n\n## Section 2\n\nMore content here."
    chunks = chunk_markdown(text, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks)


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids():
    """search_docs must return chunk IDs and scores in [0, 1].

    Requires the vector store to be seeded:
        python -m app.rag.ingest --path docs/
    """
    import os
    from pathlib import Path

    from app.settings import settings

    chroma_dir = Path(settings.chroma_persist_dir)
    if not chroma_dir.exists() or not any(chroma_dir.iterdir()):
        pytest.skip("Vector store not seeded — run: python -m app.rag.ingest --path docs/")

    from app.agents.tools.search_docs import search_docs

    results = await search_docs("how to rotate a deploy key", k=3)
    assert len(results) > 0
    assert all(r.chunk_id for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)
