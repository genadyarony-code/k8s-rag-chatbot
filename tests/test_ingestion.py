import pytest
from pathlib import Path
from src.ingestion.preprocessor import clean_text, extract_section_title, extract_page_number, process_document
from src.ingestion.loaders import RawDocument


class TestCleanText:
    def test_removes_multiple_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_reduces_multiple_newlines(self):
        result = clean_text("line1\n\n\n\nline2")
        assert "line1\n\nline2" in result

    def test_removes_control_characters(self):
        result = clean_text("hello\x00world")
        assert "\x00" not in result

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"


class TestExtractSectionTitle:
    def test_extracts_last_section(self):
        text = "intro\n## First Section ##\ncontent\n## Second Section ##\nmore"
        assert extract_section_title(text) == "Second Section"

    def test_returns_general_when_no_marker(self):
        assert extract_section_title("plain text without markers") == "General"

    def test_handles_empty_string(self):
        assert extract_section_title("") == "General"

    def test_extracts_k8s_specific_titles(self):
        text = "## CrashLoopBackOff ##\nsome content"
        assert extract_section_title(text) == "CrashLoopBackOff"


class TestExtractPageNumber:
    def test_extracts_page_number(self):
        text = "[PAGE 1]\ncontent\n[PAGE 5]\nmore content"
        assert extract_page_number(text) == 5

    def test_returns_none_when_no_marker(self):
        assert extract_page_number("no page markers") is None


class TestProcessDocument:
    def test_returns_list_of_chunks(self, sample_raw_document):
        chunks = process_document(sample_raw_document)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_has_required_fields(self, sample_raw_document):
        chunks = process_document(sample_raw_document)
        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert chunk.content != ""
            assert chunk.source == "test.html"
            assert chunk.doc_type == "concepts"
            assert chunk.section_title != ""
            assert chunk.char_count > 0

    def test_chunk_contains_section_header(self, sample_raw_document):
        """section title חייב להיות בתוך content עצמו לשיפור embedding"""
        chunks = process_document(sample_raw_document)
        assert any("Pod Scheduling" in chunk.content for chunk in chunks)

    def test_chunk_ids_are_unique(self, sample_raw_document):
        chunks = process_document(sample_raw_document)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_size_within_limits(self, sample_raw_document):
        from src.config.settings import settings
        chunks = process_document(sample_raw_document)
        for chunk in chunks:
            # נותנים margin של 20% בגלל overlap
            assert chunk.char_count <= settings.chunk_size * 1.2
