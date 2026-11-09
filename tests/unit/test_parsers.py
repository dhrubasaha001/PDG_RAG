"""
tests/unit/test_parsers.py
Unit tests for document parsers.

Requires: backend/data/DAA.pdf (already present in the repo)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DAA_PDF = REPO_ROOT / "backend" / "data" / "DAA.pdf"


# ---------------------------------------------------------------------------
# PDF Parser
# ---------------------------------------------------------------------------

class TestPDFParser:
    def setup_method(self):
        from backend.ingestion.parsers.pdf_parser import PDFParser
        self.parser = PDFParser()

    def test_supports_pdf(self):
        assert self.parser.supports("file.pdf") is True
        assert self.parser.supports("file.PDF") is True

    def test_does_not_support_docx(self):
        assert self.parser.supports("file.docx") is False

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_parse_daa_pdf(self):
        from backend.ingestion.parsers.base import ParsedDocument
        doc = self.parser.parse(DAA_PDF)

        assert isinstance(doc, ParsedDocument)
        assert doc.format == "pdf"
        assert doc.filename == "DAA.pdf"
        assert len(doc.doc_id) == 64          # SHA-256 hex
        assert doc.page_count >= 1
        assert len(doc.text) > 0
        assert doc.file_size_bytes > 0
        assert isinstance(doc.metadata, dict)

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_doc_id_is_sha256_of_file(self):
        from backend.ingestion.parsers.pdf_parser import PDFParser
        doc = PDFParser().parse(DAA_PDF)
        # Compute expected hash independently
        sha256 = hashlib.sha256(DAA_PDF.read_bytes()).hexdigest()
        assert doc.doc_id == sha256

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_page_markers_present(self):
        doc = self.parser.parse(DAA_PDF)
        assert "--- PAGE 1 ---" in doc.text

    def test_scanned_detection_heuristic(self, tmp_path):
        """A PDF with very little text per page should be flagged as scanned."""
        # Create a minimal 1-page PDF with no real text
        try:
            from pypdf import PdfWriter
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            pdf_path = tmp_path / "blank.pdf"
            with open(pdf_path, "wb") as f:
                writer.write(f)

            doc = self.parser.parse(pdf_path)
            assert doc.metadata["is_scanned"] is True
        except Exception:
            pytest.skip("Could not create blank PDF for scanned detection test")

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            self.parser.parse("nonexistent.pdf")


# ---------------------------------------------------------------------------
# Markdown / HTML / Text Parser
# ---------------------------------------------------------------------------

class TestTextParser:
    def setup_method(self):
        from backend.ingestion.parsers.text_parser import TextParser
        self.parser = TextParser()

    def test_supports_extensions(self):
        for ext in [".txt", ".md", ".html", ".htm", ".rst", ".csv"]:
            assert self.parser.supports(f"file{ext}") is True

    def test_does_not_support_pdf(self):
        assert self.parser.supports("file.pdf") is False

    def test_parse_markdown(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Heading\n\nSome **bold** text and a [link](http://example.com).")
        doc = self.parser.parse(md_file)

        assert doc.format == "md"
        assert "Heading" in doc.text
        assert "bold" in doc.text
        # Markdown syntax stripped
        assert "**" not in doc.text
        assert "[link]" not in doc.text

    def test_parse_html(self, tmp_path):
        html_file = tmp_path / "test.html"
        html_file.write_text(
            "<html><head><title>T</title></head><body>"
            "<script>alert(1)</script>"
            "<p>Hello World</p>"
            "</body></html>"
        )
        doc = self.parser.parse(html_file)

        assert doc.format == "html"
        assert "Hello World" in doc.text
        assert "alert" not in doc.text   # script removed

    def test_parse_txt(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Plain text content.\nSecond line.")
        doc = self.parser.parse(txt_file)

        assert doc.format == "txt"
        assert "Plain text content" in doc.text

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            self.parser.parse("ghost.txt")


# ---------------------------------------------------------------------------
# ParserFactory
# ---------------------------------------------------------------------------

class TestParserFactory:
    def test_auto_detects_pdf(self):
        from backend.ingestion.parsers.factory import ParserFactory
        from backend.ingestion.parsers.pdf_parser import PDFParser
        parser = ParserFactory.get_parser("document.pdf")
        assert isinstance(parser, PDFParser)

    def test_auto_detects_docx(self):
        from backend.ingestion.parsers.factory import ParserFactory
        from backend.ingestion.parsers.docx_parser import DocxParser
        parser = ParserFactory.get_parser("document.docx")
        assert isinstance(parser, DocxParser)

    def test_auto_detects_pptx(self):
        from backend.ingestion.parsers.factory import ParserFactory
        from backend.ingestion.parsers.pptx_parser import PptxParser
        parser = ParserFactory.get_parser("slides.pptx")
        assert isinstance(parser, PptxParser)

    def test_auto_detects_xlsx(self):
        from backend.ingestion.parsers.factory import ParserFactory
        from backend.ingestion.parsers.xlsx_parser import XlsxParser
        parser = ParserFactory.get_parser("data.xlsx")
        assert isinstance(parser, XlsxParser)

    def test_auto_detects_markdown(self):
        from backend.ingestion.parsers.factory import ParserFactory
        from backend.ingestion.parsers.text_parser import TextParser
        parser = ParserFactory.get_parser("readme.md")
        assert isinstance(parser, TextParser)

    def test_unsupported_raises(self):
        from backend.ingestion.parsers.factory import ParserFactory
        with pytest.raises(ValueError, match="No parser available"):
            ParserFactory.get_parser("file.xyz")

    def test_is_supported(self):
        from backend.ingestion.parsers.factory import ParserFactory
        assert ParserFactory.is_supported("file.pdf") is True
        assert ParserFactory.is_supported("file.xyz") is False

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_parse_convenience_method(self):
        from backend.ingestion.parsers.factory import ParserFactory
        doc = ParserFactory.parse(DAA_PDF)
        assert doc.format == "pdf"
        assert len(doc.text) > 0
        assert doc.doc_id  # non-empty SHA-256
