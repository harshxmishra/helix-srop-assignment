"""
search_docs tool — used by KnowledgeAgent.

Queries the ChromaDB vector store for relevant documentation chunks.
Returns chunk IDs, similarity scores, and content for citations.

E4: LLM-as-judge reranker. Retrieves top-20, reranks to top-k.
"""
import asyncio
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types

from app.rag.ingest import get_chroma_collection
from app.settings import settings

# Minimum similarity score to include a result (filters noise)
SCORE_THRESHOLD = 0.3

# E4: retrieve wider candidate set before reranking
RERANK_CANDIDATE_MULTIPLIER = 4


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict  # e.g. {"product_area": "security", "source": "deploy-keys.md"}


def _embed_query(query: str) -> list[float]:
    """Embed query text using gemini-embedding-001 (task_type=RETRIEVAL_QUERY)."""
    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=query,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def _rerank_with_llm(query: str, chunks: list[DocChunk], top_k: int) -> list[DocChunk]:
    """
    E4: LLM-as-judge reranker.

    Ask Gemini to rank candidates by relevance to query.
    Returns top_k chunks in reranked order.

    Scoring: LLM assigns 1-10 relevance; chunks sorted by LLM score.
    Falls back to original order on parse failure.
    """
    if len(chunks) <= top_k:
        return chunks

    # Build prompt with numbered chunks
    chunk_listing = "\n\n".join(
        f"[{i+1}] (chunk_id: {c.chunk_id})\n{c.content[:300]}"
        for i, c in enumerate(chunks)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Rate each chunk's relevance to the query from 1 (irrelevant) to 10 (highly relevant).\n"
        f"Respond with ONLY a comma-separated list of scores in order, e.g.: 8,3,9,5,2\n\n"
        f"Chunks:\n{chunk_listing}"
    )

    try:
        client = genai.Client(api_key=settings.google_api_key)
        resp = client.models.generate_content(
            model=settings.adk_model,
            contents=prompt,
        )
        text = resp.text.strip()
        scores = [float(s.strip()) for s in text.split(",")]
        if len(scores) != len(chunks):
            return chunks[:top_k]

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        return [c for _, c in ranked[:top_k]]
    except Exception:
        # Fallback: return original order
        return chunks[:top_k]


async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> list[DocChunk]:
    """
    Search vector store for top-k relevant chunks.
    E4: Retrieves up to k*4 candidates, reranks with LLM-as-judge, returns top k.

    Args:
        query: natural language question from the user
        k: number of chunks to return after reranking
        product_area: optional metadata filter (e.g. "security", "ci-cd")

    Returns:
        List of DocChunk ordered by reranked relevance (best first).
        Chunks below SCORE_THRESHOLD are excluded before reranking.
    """
    query_embedding = await asyncio.to_thread(_embed_query, query)

    where: dict | None = {"product_area": product_area} if product_area else None

    # E4: fetch more candidates for reranking
    candidate_k = min(k * RERANK_CANDIDATE_MULTIPLIER, 20)

    def _query() -> dict:
        collection = get_chroma_collection()
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": candidate_k,
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
        score = round(1.0 - distance, 4)
        if score < SCORE_THRESHOLD:
            continue
        chunks.append(DocChunk(
            chunk_id=chunk_id,
            score=score,
            content=doc,
            metadata=meta or {},
        ))

    if not chunks:
        return []

    # E4: Rerank with LLM-as-judge
    reranked = await asyncio.to_thread(_rerank_with_llm, query, chunks, k)
    return reranked
