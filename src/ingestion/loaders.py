"""
Loads raw documents from disk and returns a RawDocument ready for the preprocessor.

HTML files go through BeautifulSoup — strips nav/header/script noise and converts
heading tags to ## markers that the preprocessor uses to tag chunks with section titles.

PDFs try Docling first (better at layout, tables, and YAML formatting) and fall back
to PyMuPDF + pdfplumber if Docling fails. The fallback prints a conspicuous warning
box — silently degraded content quality is harder to spot than a loud notice.

The k8s book PDF is capped at pages 33-340 (chapters 1-10 only; the rest is appendices).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

# ── ANSI Colors ────────────────────────────────────────────────────────────────
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RESET  = "\033[0m"

# Pages to load from each PDF (0-indexed, inclusive).
# Anything not listed here gets loaded in full.
PDF_PAGE_RANGES: dict[str, tuple[int, int]] = {
    "k8s_in_action_ch1-10.pdf": (33, 340),  # chapters 1-10 only
}


def _get_page_range(path: Path) -> tuple[int, int]:
    """Returns (start, end) 0-indexed inclusive, or (0, -1) for the full document."""
    return PDF_PAGE_RANGES.get(path.name, (0, -1))


# ── RawDocument Dataclass ──────────────────────────────────────────────────────

@dataclass
class RawDocument:
    content: str
    source: str
    doc_type: Literal["concepts", "book", "troubleshooting"]
    file_format: Literal["html", "pdf", "markdown"]
    tables: List[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # metadata must include a "loader" key — used for logging in ingest.py


# ── HTML Loader ────────────────────────────────────────────────────────────────

def load_html(path: Path, doc_type: str) -> RawDocument:
    """
    Parses HTML with BeautifulSoup.

    Strips: nav, footer, header, script, style, aside, iframe, noscript
    Keeps:  h1-h4, p, li, code, pre, table
    Headings become ## markers so the preprocessor can tag chunks with section titles.
    Lines shorter than 20 characters are filtered out (page numbers, stray labels, etc.).
    """
    from bs4 import BeautifulSoup

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    for tag in soup.find_all(["nav", "footer", "header", "script", "style",
                               "aside", "iframe", "noscript"]):
        tag.decompose()

    content_parts: List[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li",
                               "code", "pre", "table"]):
        if tag.name in ("h1", "h2", "h3", "h4"):
            text = tag.get_text(strip=True)
            if text:
                content_parts.append(f"\n## {text} ##\n")
        else:
            text = tag.get_text(strip=True)
            if len(text) > 20:
                content_parts.append(text)

    return RawDocument(
        content="\n".join(content_parts),
        source=path.name,
        doc_type=doc_type,
        file_format="html",
        metadata={"loader": "beautifulsoup"},
    )


# ── PDF Loader: Docling ────────────────────────────────────────────────────────

def load_pdf_docling(path: Path, doc_type: str) -> RawDocument:
    """
    Converts PDF to Markdown via Docling. Handles YAML indentation, tables,
    and multi-column layouts better than text-extraction approaches.
    Applies page range filtering according to PDF_PAGE_RANGES.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    markdown_text = result.document.export_to_markdown()

    start, end = _get_page_range(path)
    if end != -1:
        total_pages = (
            len(result.document.pages)
            if hasattr(result.document, "pages")
            else -1
        )
        if total_pages > 0:
            char_per_page = len(markdown_text) / total_pages
            start_char = int(start * char_per_page)
            end_char = int((end + 1) * char_per_page)
            markdown_text = markdown_text[start_char:end_char]

    page_count = (
        len(result.document.pages)
        if hasattr(result.document, "pages")
        else -1
    )

    return RawDocument(
        content=markdown_text,
        source=path.name,
        doc_type=doc_type,
        file_format="pdf",
        metadata={
            "loader": "docling",
            "has_tables": "| " in markdown_text,
            "page_count": page_count,
        },
    )


# ── PDF Loader: Hybrid (PyMuPDF + pdfplumber) ──────────────────────────────────

def load_pdf_hybrid(path: Path, doc_type: str) -> RawDocument:
    """
    Fallback PDF loader for when Docling isn't available or fails.

    Uses PyMuPDF for text extraction (with [PAGE N] markers) and pdfplumber
    for table detection only. Tables are converted to Markdown and injected
    inline. Applies page range filtering according to PDF_PAGE_RANGES.
    """
    import fitz  # PyMuPDF
    import pdfplumber

    content_parts: List[str] = []
    tables_found: List[dict] = []

    doc = fitz.open(str(path))
    page_count = len(doc)

    start, end = _get_page_range(path)
    end = end if end != -1 else page_count - 1
    end = min(end, page_count - 1)  # safety clamp in case the PDF has fewer pages

    # Open pdfplumber once for all pages — reopening per page is O(n²) on file size
    try:
        plumber_pdf = pdfplumber.open(str(path))
    except Exception as e:
        logging.warning(f"pdfplumber failed to open {path.name}: {e}")
        plumber_pdf = None

    for page_num in range(start, end + 1):
        page = doc[page_num]
        blocks = page.get_text("blocks", sort=True, flags=11)  # type: ignore[arg-type]
        display_page = page_num + 1  # 1-indexed for display
        content_parts.append(f"\n[PAGE {display_page}]\n")

        page_blocks: List[str] = []
        for block in blocks:
            if len(block) < 5:
                continue
            text = block[4].strip()
            if text:
                page_blocks.append(text)

        content_parts.append("\n\n".join(page_blocks))

        # pdfplumber only for tables — PyMuPDF handles text
        if plumber_pdf is not None:
            try:
                if page_num < len(plumber_pdf.pages):
                    plumber_page = plumber_pdf.pages[page_num]
                    tables = plumber_page.find_tables({
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "snap_tolerance": 3,
                    })
                    for idx, table in enumerate(tables):
                        try:
                            md_table = table.to_markdown(index=False)
                        except Exception:
                            md_table = str(table.extract())
                        tables_found.append({
                            "page": display_page,
                            "table_index": idx,
                            "markdown": md_table,
                        })
                        content_parts.append(
                            f"\n[TABLE {idx + 1} on page {display_page}]\n"
                            f"{md_table}\n[/TABLE]\n"
                        )
            except Exception as e:
                logging.warning(f"pdfplumber failed on page {display_page}: {e}")

    if plumber_pdf is not None:
        plumber_pdf.close()

    return RawDocument(
        content="\n".join(content_parts),
        source=path.name,
        doc_type=doc_type,
        file_format="pdf",
        tables=tables_found,
        metadata={
            "loader": "pymupdf+pdfplumber",
            "has_tables": len(tables_found) > 0,
            "table_count": len(tables_found),
            "page_count": page_count,
        },
    )


# ── PDF Loader: Smart (Docling → fallback to Hybrid) ──────────────────────────

def load_pdf_smart(path: Path, doc_type: str) -> RawDocument:
    """
    Tries Docling first. If it fails for any reason, falls back to the hybrid loader.
    The fallback warning box is intentionally loud — degraded content quality
    shouldn't go unnoticed.
    """
    logging.info(f"{GREEN}[Loader] Attempting Docling for {path.name}...{RESET}")

    try:
        result = load_pdf_docling(path, doc_type)
        logging.info(f"{GREEN}[Loader] ✓ Docling succeeded{RESET}")
        return result

    except Exception as e:
        logging.warning(
            f"\n{YELLOW}"
            f"╔══════════════════════════════════════════════════════╗\n"
            f"║  ⚠️  DOCLING FALLBACK TRIGGERED                      ║\n"
            f"║  File   : {path.name:<42}║\n"
            f"║  Reason : {str(e)[:42]:<42}║\n"
            f"║  Action : Falling back to PyMuPDF + pdfplumber       ║\n"
            f"║  Impact : YAML indentation & tables may be degraded  ║\n"
            f"╚══════════════════════════════════════════════════════╝"
            f"{RESET}\n"
        )
        return load_pdf_hybrid(path, doc_type)
