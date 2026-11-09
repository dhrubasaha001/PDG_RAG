"""
backend/ingestion/parsers/base.py
Abstract base classes and shared data structures for document parsers.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class ParsedDocument:
    """Normalised output of any document parser."""

    doc_id: str              # SHA-256 hex digest of raw file bytes
    filename: str            # original filename
    format: str              # pdf | docx | pptx | xlsx | md | html | txt
    text: str                # full extracted plain text
    page_count: int          # logical page / slide / sheet count
    file_path: str           # absolute path where the file lives on disk
    file_size_bytes: int     # raw file size in bytes
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys (best-effort, format-dependent):
    #   title        str
    #   author       str
    #   created_at   str   (ISO-8601 or free-form from doc metadata)
    #   language     str   (detected or declared)
    #   is_scanned   bool  (PDF only: True when OCR is required)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseParser(ABC):
    """Every format-specific parser must implement this interface."""

    @abstractmethod
    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Extract text and metadata from *file_path*.

        Args:
            file_path: Path to the document on disk.

        Returns:
            ParsedDocument with all fields populated.

        Raises:
            ValueError: If the file cannot be parsed by this parser.
            FileNotFoundError: If the file does not exist.
        """

    @abstractmethod
    def supports(self, file_path: str | Path) -> bool:
        """Return True if this parser can handle the given file."""

    # ------------------------------------------------------------------
    # Shared utilities available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def compute_doc_id(file_path: str | Path) -> str:
        """Return the SHA-256 hex digest of the raw file bytes."""
        path = Path(file_path)
        sha256 = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65_536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def file_size(file_path: str | Path) -> int:
        """Return file size in bytes."""
        return Path(file_path).stat().st_size
