"""
Splits a RawDocument into Chunks ready to embed and index.

Each chunk gets a section title prefix before going into ChromaDB — something like
"[Pod Scheduling]\nchunk text here". The idea is that the chunk carries its own
context so a retrieval hit on "resource limits" still tells the model it came from
the scheduling section, even if that heading isn't in the chunk text itself.

RecursiveCharacterTextSplitter breaks on paragraph boundaries first, then lines,
then words — usually keeps related sentences together better than a fixed-size split.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Optional

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings


# ── Chunk Dataclass ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str        # uuid4
    content: str         # text content, prefixed with [section_title]
    source: str          # original filename
    doc_type: str        # concepts / book / troubleshooting
    section_title: str   # nearest heading before this chunk
    page_number: Optional[int]
    char_count: int


# ── Text Utilities ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Basic noise removal without losing information.
    Collapses extra whitespace, trims repeated blank lines,
    strips page-number artifacts (e.g. "- 42 -"), and removes control characters.
    """
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n[-–]\s*\d+\s*[-–]\n", "\n", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def extract_section_title(text_before_chunk: str) -> str:
    """
    Finds the last ## Title ## marker before this chunk's position in the document.
    These markers are inserted by the HTML loader from heading tags.
    Returns "General" if no marker is found.
    """
    matches = list(re.finditer(r"## (.+?) ##", text_before_chunk))
    return matches[-1].group(1) if matches else "General"


def extract_page_number(text_before_chunk: str) -> Optional[int]:
    """Finds the last [PAGE N] marker before this chunk. Returns None for HTML sources."""
    matches = list(re.finditer(r"\[PAGE (\d+)\]", text_before_chunk))
    return int(matches[-1].group(1)) if matches else None


# ── Main Processor ─────────────────────────────────────────────────────────────

def process_document(raw_doc) -> list[Chunk]:
    """Cleans and splits a RawDocument into a list of Chunks with metadata."""
    cleaned = clean_text(raw_doc.content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,       # 1000
        chunk_overlap=settings.chunk_overlap,  # 200
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    text_chunks = splitter.split_text(cleaned)
    chunks: list[Chunk] = []

    for chunk_text in text_chunks:
        # Locate this chunk in the cleaned text to find the heading above it
        search_key = chunk_text[:50]
        chunk_start = cleaned.find(search_key)
        text_before = cleaned[:chunk_start] if chunk_start > 0 else ""

        section = extract_section_title(text_before)
        enriched_content = f"[{section}]\n{chunk_text}"

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            content=enriched_content,
            source=raw_doc.source,
            doc_type=raw_doc.doc_type,
            section_title=section,
            page_number=extract_page_number(text_before),
            char_count=len(chunk_text),
        )
        chunks.append(chunk)

    return chunks
