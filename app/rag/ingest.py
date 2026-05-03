"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 512 --chunk-overlap 64

Reads markdown files, chunks them, embeds, and writes to the vector store.

Chunking strategy: heading-aware.
Split on ## / ### headings so each chunk maps to one logical section.
Long sections are sub-chunked by sentence to stay under chunk_size chars.
This preserves topic coherence better than fixed-size splitting for structured
markdown docs like these product reference pages.
"""
import argparse
import asyncio
import hashlib
import re
from pathlib import Path

import chromadb
import yaml
from google import genai
from google.genai import types as genai_types

from app.settings import settings

COLLECTION_NAME = "helix_docs"


def get_chroma_collection() -> chromadb.Collection:
    """Return (or create) the persistent Chroma collection."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def extract_metadata(file_path: Path, text: str) -> tuple[dict, str]:
    """
    Extract YAML frontmatter from markdown text.

    Returns (metadata_dict, body_text_without_frontmatter).
    If no frontmatter found, returns ({"source": filename}, original_text).
    """
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {"source": file_path.name}, text
    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        metadata = {}
    metadata["source"] = file_path.name
    body = text[match.end():]
    return metadata, body


def _chunk_by_sentence(text: str, max_chars: int, overlap_sentences: int = 1) -> list[str]:
    """Split text into sentence-bounded chunks of at most max_chars characters."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:]
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def chunk_markdown(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split markdown text into chunks using heading-aware strategy.

    Splits on ## and ### headings so each chunk corresponds to one section.
    Sections longer than chunk_size chars are further split by sentence.
    The overlap parameter is passed to sentence sub-chunking for context
    preservation at boundaries.
    """
    sections = re.split(r"\n(?=#{2,3} )", text)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(_chunk_by_sentence(section, max_chars=chunk_size))
    return [c for c in chunks if c.strip()]


def _make_chunk_id(file_path: Path, index: int) -> str:
    """Deterministic chunk ID — stable across re-ingests of the same content."""
    raw = f"{file_path}::{index}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a batch of texts using Google text-embedding-004 (google.genai SDK)."""
    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.embed_content(
        model="models/text-embedding-004",
        contents=texts,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    # response.embeddings is a list[ContentEmbedding], each with .values
    return [emb.values for emb in response.embeddings]


async def ingest_directory(docs_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """
    Walk docs_path, chunk and embed every .md file, upsert into vector store.

    Stable chunk IDs mean re-ingesting the same file is idempotent (upsert).
    Embeddings are generated in batches of 20 to respect rate limits.
    ChromaDB's PersistentClient is synchronous — calls run in a thread.
    """
    md_files = list(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        metadata, body = extract_metadata(file_path, text)
        chunks = chunk_markdown(body, chunk_size, chunk_overlap)
        print(f"  {file_path.name}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = _make_chunk_id(file_path, i)
            all_ids.append(chunk_id)
            all_documents.append(chunk)
            # Chroma requires metadata values to be str/int/float/bool
            safe_meta = {
                k: (", ".join(v) if isinstance(v, list) else str(v) if v is not None else "")
                for k, v in metadata.items()
            }
            all_metadatas.append(safe_meta)

    if not all_ids:
        print("No chunks to ingest.")
        return

    # Embed in batches of 20
    batch_size = 20
    total_batches = (len(all_documents) + batch_size - 1) // batch_size
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i : i + batch_size]
        batch_embeddings = await asyncio.to_thread(_embed_texts, batch)
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded batch {i // batch_size + 1}/{total_batches}")

    # Upsert to Chroma in a thread (sync client)
    def _upsert() -> None:
        collection = get_chroma_collection()
        collection.upsert(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_documents,
            metadatas=all_metadatas,
        )

    await asyncio.to_thread(_upsert)
    print(f"Ingest complete. Upserted {len(all_ids)} chunks.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
