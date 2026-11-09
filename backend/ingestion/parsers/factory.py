"""
backend/ingestion/parsers/factory.py
ParserFactory — auto-detects document format by extension + MIME type
and returns the appropriate parser instance.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from backend.core.logging import get_logger
from backend.ingestion.parsers.base import BaseParser, ParsedDocument
from backend.ingestion.parsers.pdf_parser import PDFParser
from backend.ingestion.parsers.docx_parser import DocxParser
from backend.ingestion.parsers.pptx_parser import PptxParser
from backend.ingestion.parsers.xlsx_parser import XlsxParser
from backend.ingestion.parsers.text_parser import TextParser

logger = get_logger("ingestion.parser_factory")

# Ordered list of parser instances — first match wins
_PARSERS: list[BaseParser] = [
    PDFParser(),
    DocxParser(),
    PptxParser(),
    XlsxParser(),
    TextParser(),
]

# Supported MIME types (supplementary to extension check)
_SUPPORTED_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/plain",
    "text/html",
    "text/markdown",
    "text/csv",
    "text/x-rst",
}

# Extension → human-readable format name (used in error messages)
_EXT_LABELS = {
    ".pdf": "PDF",
    ".docx": "Word Document",
    ".pptx": "PowerPoint Presentation",
    ".xlsx": "Excel Spreadsheet",
    ".xls": "Excel Spreadsheet (legacy)",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".html": "HTML",
    ".htm": "HTML",
    ".txt": "Plain Text",
    ".rst": "reStructuredText",
    ".csv": "CSV",
}


class ParserFactory:
    """
    Auto-detect the right parser for a given file and parse it.

    Usage:
        doc = ParserFactory.parse("path/to/file.pdf")
        # or
        parser = ParserFactory.get_parser("path/to/file.pdf")
        doc = parser.parse("path/to/file.pdf")
    """

    @staticmethod
    def get_parser(file_path: str | Path) -> BaseParser:
        """
        Return the first parser that supports this file.

        Args:
            file_path: Path to the file.

        Returns:
            A BaseParser subclass instance.

        Raises:
            ValueError: If no parser supports the file format.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        for parser in _PARSERS:
            if parser.supports(path):
                logger.debug(
                    "parser_selected",
                    filename=path.name,
                    parser=type(parser).__name__,
                )
                return parser

        # Provide a helpful error
        mime, _ = mimetypes.guess_type(str(path))
        label = _EXT_LABELS.get(suffix, suffix or "unknown")
        raise ValueError(
            f"No parser available for '{path.name}' "
            f"(format: {label}, MIME: {mime or 'unknown'}). "
            f"Supported extensions: {', '.join(sorted(_EXT_LABELS.keys()))}"
        )

    @staticmethod
    def parse(file_path: str | Path) -> ParsedDocument:
        """
        Convenience method: auto-detect format and parse in one call.

        Args:
            file_path: Path to the file.

        Returns:
            ParsedDocument with full text and metadata.
        """
        parser = ParserFactory.get_parser(file_path)
        return parser.parse(file_path)

    @staticmethod
    def is_supported(file_path: str | Path) -> bool:
        """Return True if the file can be parsed by any registered parser."""
        path = Path(file_path)
        return any(p.supports(path) for p in _PARSERS)

    @staticmethod
    def supported_extensions() -> list[str]:
        """Return a sorted list of all supported file extensions."""
        return sorted(_EXT_LABELS.keys())
