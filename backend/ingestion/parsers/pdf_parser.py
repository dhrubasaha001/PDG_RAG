"""
backend/ingestion/parsers/pdf_parser.py
PDF document parser using pypdf.

Scanned-PDF detection:
  If the average extracted characters per page < SCANNED_CHARS_THRESHOLD
  we flag the document as scanned and log a warning.
  Scanned docs are still returned — the text will just be sparse/empty.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from backend.core.logging import get_logger
from backend.ingestion.parsers.base import BaseParser, ParsedDocument

logger = get_logger("ingestion.pdf_parser")

# Heuristic: fewer than this many chars/page → likely scanned
SCANNED_CHARS_THRESHOLD = 100

# Separator injected between pages so the chunker can recover page numbers
PAGE_SEPARATOR = "\n\n--- PAGE {n} ---\n\n"


class PDFParser(BaseParser):
    """Parse text-native PDF files using pypdf."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    def supports(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc_id = self.compute_doc_id(path)
        file_size = self.file_size(path)

        reader = PdfReader(str(path))
        page_count = len(reader.pages)

        # ------------------------------------------------------------------ #
        # Extract text page-by-page, injecting page markers
        # ------------------------------------------------------------------ #
        pages_text: list[str] = []
        total_chars = 0

        for n, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            total_chars += len(raw)
            pages_text.append(PAGE_SEPARATOR.format(n=n) + raw)

        full_text = "".join(pages_text)

        # ------------------------------------------------------------------ #
        # Scanned-PDF detection
        # ------------------------------------------------------------------ #
        avg_chars_per_page = total_chars / max(page_count, 1)
        is_scanned = avg_chars_per_page < SCANNED_CHARS_THRESHOLD

        if is_scanned:
            logger.warning(
                "scanned_pdf_detected",
                doc_id=doc_id,
                filename=path.name,
                avg_chars_per_page=round(avg_chars_per_page, 1),
                message="OCR not available — text content will be sparse",
            )

        # ------------------------------------------------------------------ #
        # Metadata extraction from PDF info dict
        # ------------------------------------------------------------------ #
        info = reader.metadata or {}
        metadata = {
            "title": _safe_str(info.get("/Title", path.stem)),
            "author": _safe_str(info.get("/Author", "")),
            "created_at": _safe_str(info.get("/CreationDate", "")),
            "language": "",
            "is_scanned": is_scanned,
        }

        logger.info(
            "pdf_parsed",
            doc_id=doc_id,
            filename=path.name,
            page_count=page_count,
            total_chars=total_chars,
            is_scanned=is_scanned,
        )

        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            format="pdf",
            text=full_text,
            page_count=page_count,
            file_path=str(path),
            file_size_bytes=file_size,
            metadata=metadata,
        )


def _safe_str(value: object) -> str:
    """Convert a PDF metadata value to a clean string."""
    if value is None:
        return ""
    return str(value).strip()
