"""
search_docs tool — used by KnowledgeAgent.

Queries the ChromaDB vector store for relevant documentation chunks.
Returns chunk IDs, similarity scores, and content for citations.
"""
import asyncio
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types

from app.rag.ingest import get_chroma_collection
from app.settings import settings

# Minimum similarity score to include a result (filters noise)
SCORE_THRESHOLD = 0.3


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict  # e.g. {"product_area": "security", "source": "deploy-keys.md"}


def _embed_query(query: str) -> list[float]:
    """Embed query text using the same model as ingest (task_type=RETRIEVAL_QUERY)."""
    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.embed_content(
        model="models/text-embedding-004",
        contents=query,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    # Single string content → response.embeddings has one entry
    return response.embeddings[0].values


async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> list[DocChunk]:
    """
    Search the vector store for the top-k most relevant documentation chunks.

    Args:
        query: natural language question from the user
        k: number of chunks to return
        product_area: optional metadata filter (e.g. "security", "ci-cd")

    Returns:
        List of DocChunk ordered by descending similarity score (best first).
        Chunks below SCORE_THRESHOLD are excluded.
    """
    # Embed the query in a thread (blocking genai call)
    query_embedding = await asyncio.to_thread(_embed_query, query)

    # Build optional metadata filter
    where: dict | None = {"product_area": product_area} if product_area else None

    def _query() -> dict:
        collection = get_chroma_collection()
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            kwargs["where"] = where
        return collection.query(**kwargs)

    results = await asyncio.to_thread(_query)

    chunks: list[DocChunk] = []
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for chunk_id, distance, doc, meta in zip(ids, distances, documents, metadatas):
        # Chroma cosine distance: 0 = identical, 2 = opposite → convert to [0,1] score
        score = round(1.0 - distance, 4)
        if score < SCORE_THRESHOLD:
            continue
        chunks.append(DocChunk(
            chunk_id=chunk_id,
            score=score,
            content=doc,
            metadata=meta or {},
        ))

    return sorted(chunks, key=lambda c: c.score, reverse=True)
