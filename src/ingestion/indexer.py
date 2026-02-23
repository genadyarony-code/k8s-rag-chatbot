"""
Writes chunks into two indexes: ChromaDB for vector search, and a BM25 index
pickled to disk as a fallback for when you want to run without the embeddings API.

The schema version is written into both indexes at build time and checked on load.
Change the chunking logic, bump the version, and the mismatch will fail loudly
instead of silently mixing old and new chunks in the same collection.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

from src.config.settings import settings


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list) -> list[list[float]]:
    """
    Embeds all chunks using text-embedding-3-small (1536 dimensions).
    Sends in batches of 100 to stay within the API's input limits.
    Returns vectors in the same order as the input list.
    """
    client = OpenAI(api_key=settings.openai_api_key)
    texts = [chunk.content for chunk in chunks]

    BATCH_SIZE = 100
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(
            input=batch,
            model=settings.embedding_model,
        )
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


# ── ChromaDB ───────────────────────────────────────────────────────────────────

def load_to_chroma(chunks: list, embeddings: list[list[float]]) -> None:
    """
    Loads chunks and their embeddings into the ChromaDB collection.

    Checks the stored schema version before writing. If the collection exists
    with a different version, raises ValueError — run ingest.py --force to
    drop and rebuild it cleanly.
    """
    client = chromadb.PersistentClient(path=settings.chroma_db_path)

    # Check schema version if the collection already exists
    try:
        existing = client.get_collection(settings.chroma_collection_name)
        stored_version = existing.metadata.get("chunk_schema_version", "unknown")
        if stored_version != settings.chunk_schema_version:
            raise ValueError(
                f"Index version mismatch: stored={stored_version}, "
                f"current={settings.chunk_schema_version}. "
                f"Run: python scripts/ingest.py --force"
            )
    except Exception as e:
        if "does not exist" not in str(e).lower() and "InvalidCollection" not in type(e).__name__:
            raise

    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={
            "hnsw:space": "cosine",
            "chunk_schema_version": settings.chunk_schema_version,
        },
    )

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.content for c in chunks],
        metadatas=[
            {
                "source": c.source,
                "doc_type": c.doc_type,
                "section_title": c.section_title,
                "page_number": c.page_number if c.page_number is not None else -1,
                "char_count": c.char_count,
                "chunk_schema_version": settings.chunk_schema_version,
            }
            for c in chunks
        ],
    )


# ── BM25 ───────────────────────────────────────────────────────────────────────

def build_bm25_index(chunks: list) -> None:
    """
    Builds an Okapi BM25 index and pickles it to bm25_index/bm25.pkl.
    The chunks list is stored alongside the index so retrieve_node can
    reconstruct full results without re-loading from ChromaDB.
    """
    tokenized_corpus = [chunk.content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    Path("./bm25_index").mkdir(exist_ok=True)
    with open("./bm25_index/bm25.pkl", "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": chunks,
                "chunk_schema_version": settings.chunk_schema_version,
            },
            f,
        )
