"""
Builds both search indexes from the raw documents. Run this once before starting
the API, and again with --force if you change the chunking strategy or add documents.

    python scripts/ingest.py           # first run, or after adding documents
    python scripts/ingest.py --force   # drop existing index and rebuild from scratch

Writes data/processed/index_meta.json when done. The API reads this on startup
and refuses to start if it's missing or reports an error.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add the project root to sys.path so src.* imports work without installing the package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

#  ANSI Colors 
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"


def main(force: bool = False) -> None:
    from src.config.settings import settings
    from src.ingestion.indexer import build_bm25_index, embed_chunks, load_to_chroma
    from src.ingestion.loaders import load_html, load_pdf_smart
    from src.ingestion.preprocessor import process_document

    print("=== K8s RAG Chatbot — Ingestion Pipeline ===")
    print(f"Schema version: {settings.chunk_schema_version}\n")

    # force: drop the existing ChromaDB collection before rebuilding
    if force:
        import chromadb
        print(f"{YELLOW}⚠️  --force: deleting existing ChromaDB collection...{RESET}")
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        try:
            client.delete_collection(settings.chroma_collection_name)
            print("   ChromaDB collection deleted.")
        except Exception:
            pass  # collection didn't exist — that's fine

    #  1. Load 
    print("Loading documents...")
    raw_docs = [
        load_html(Path("data/raw/k8s_concepts.html"), "concepts"),
        load_pdf_smart(Path("data/raw/k8s_in_action_ch1-10.pdf"), "book"),
        load_html(Path("data/raw/k8s_troubleshooting.html"), "troubleshooting"),
    ]
    for doc in raw_docs:
        print(
            f"  {doc.source}: {len(doc.content):,} chars "
            f"| loader: {doc.metadata.get('loader', 'unknown')}"
        )

    #  2. Preprocess 
    print("\nPreprocessing & chunking...")
    all_chunks = []
    for doc in raw_docs:
        chunks = process_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc.source}: {len(chunks)} chunks")
    print(f"Total chunks: {len(all_chunks)}")

    #  3. Save processed chunks (useful for inspection and debugging)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
        json.dump([vars(c) for c in all_chunks], f, indent=2, default=str)
    print("Chunks saved → data/processed/chunks.json")

    #  4. Embed 
    print("\nEmbedding chunks (this may take a minute)...")
    embeddings = embed_chunks(all_chunks)
    print(f"Embeddings created: {len(embeddings)} vectors")

    #  5. Index: ChromaDB 
    print("Loading to ChromaDB...")
    chroma_ok = False
    try:
        load_to_chroma(all_chunks, embeddings)
        chroma_ok = True
        print(f"{GREEN}✓ ChromaDB indexed successfully{RESET}")
    except Exception as e:
        print(f"{RED}✗ ChromaDB indexing failed: {e}{RESET}")

    #  6. Index: BM25 
    print("Building BM25 fallback index...")
    bm25_ok = False
    try:
        build_bm25_index(all_chunks)
        bm25_ok = True
        print(f"{GREEN}✓ BM25 index built successfully{RESET}")
    except Exception as e:
        print(f"{RED}✗ BM25 indexing failed: {e}{RESET}")

    # 7. Write index_meta.json 
    # Written even on partial failure so the API can report a clear error on startup
    meta = {
        "chunk_schema_version": settings.chunk_schema_version,
        "chunk_count": len(all_chunks),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "chroma_ok": chroma_ok,
        "bm25_ok": bm25_ok,
    }
    with open("data/processed/index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Index manifest saved → data/processed/index_meta.json")

    # 8. Final status 
    success = chroma_ok and bm25_ok
    status = (
        f"{GREEN}✓ SUCCESS{RESET}"
        if success
        else f"{RED}✗ PARTIAL FAILURE{RESET}"
    )
    print(
        f"\n{status} | "
        f"Schema: {settings.chunk_schema_version} | "
        f"Chunks: {len(all_chunks)} | "
        f"chroma_ok={chroma_ok} | bm25_ok={bm25_ok}"
    )
    if not success:
        print(
            f"{YELLOW}⚠️  FastAPI will refuse to start. "
            f"Run: python scripts/ingest.py --force{RESET}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K8s RAG Chatbot — Ingestion Pipeline"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing ChromaDB collection and rebuild from scratch",
    )
    args = parser.parse_args()
    main(force=args.force)
