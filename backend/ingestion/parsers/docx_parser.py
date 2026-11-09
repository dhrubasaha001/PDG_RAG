"""
backend/ingestion/parsers/docx_parser.py
DOCX document parser using python-docx.

Preserves heading hierarchy as markdown-style markers so the chunker
can use them as natural split points.
Tables are serialised as pipe-delimited rows.
"""

from __future__ import annotations

from pathlib import Path

try:
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn
except ImportError as exc:
    raise ImportError("python-docx is required: pip install python-docx") from exc

from backend.core.logging import get_logger
from backend.ingestion.parsers.base import BaseParser, ParsedDocument

logger = get_logger("ingestion.docx_parser")

# Heading style name → markdown prefix
_HEADING_PREFIX = {
    "heading 1": "# ",
    "heading 2": "## ",
    "heading 3": "### ",
    "heading 4": "#### ",
}


class DocxParser(BaseParser):
    """Parse .docx files, preserving headings and tables."""

    SUPPORTED_EXTENSIONS = {".docx"}

    def supports(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"DOCX not found: {path}")

        doc_id = self.compute_doc_id(path)
        file_size = self.file_size(path)

        doc = DocxDocument(str(path))
        parts: list[str] = []

        for block in _iter_block_items(doc):
            if block["type"] == "paragraph":
                para = block["element"]
                style_name = para.style.name.lower() if para.style else ""
                prefix = _HEADING_PREFIX.get(style_name, "")
                text = para.text.strip()
                if text:
                    parts.append(f"{prefix}{text}")
            elif block["type"] == "table":
                table = block["element"]
                parts.append(_table_to_text(table))

        full_text = "\n\n".join(parts)

        # ------------------------------------------------------------------ #
        # Metadata from core properties
        # ------------------------------------------------------------------ #
        core = doc.core_properties
        metadata = {
            "title": core.title or path.stem,
            "author": core.author or "",
            "created_at": str(core.created) if core.created else "",
            "language": core.language or "",
            "is_scanned": False,
        }

        logger.info(
            "docx_parsed",
            doc_id=doc_id,
            filename=path.name,
            total_chars=len(full_text),
        )

        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            format="docx",
            text=full_text,
            page_count=0,          # python-docx doesn't expose page count reliably
            file_path=str(path),
            file_size_bytes=file_size,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_block_items(doc: "DocxDocument"):
    """
    Yield paragraphs and tables in document order.
    python-docx iterates paragraphs and tables separately by default;
    this function walks the XML body to preserve interleaved order.
    """
    from docx.oxml.ns import qn as _qn
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            from docx.text.paragraph import Paragraph
            yield {"type": "paragraph", "element": Paragraph(child, doc)}
        elif tag == "tbl":
            from docx.table import Table
            yield {"type": "table", "element": Table(child, doc)}


def _table_to_text(table) -> str:
    """Serialise a docx Table as pipe-delimited plain text."""
    rows: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append(" | ".join(cells))
    return "\n".join(rows)
