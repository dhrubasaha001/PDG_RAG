"""
backend/ingestion/parsers/pptx_parser.py
PowerPoint parser using python-pptx.

Each slide is prefixed with a '--- SLIDE {n} ---' marker.
Text from all shapes (titles, bodies, notes) is extracted.
"""

from __future__ import annotations

from pathlib import Path

try:
    from pptx import Presentation
    from pptx.util import Pt
except ImportError as exc:
    raise ImportError("python-pptx is required: pip install python-pptx") from exc

from backend.core.logging import get_logger
from backend.ingestion.parsers.base import BaseParser, ParsedDocument

logger = get_logger("ingestion.pptx_parser")

SLIDE_SEPARATOR = "\n\n--- SLIDE {n} ---\n\n"


class PptxParser(BaseParser):
    """Parse .pptx files, extracting text from slides and notes."""

    SUPPORTED_EXTENSIONS = {".pptx"}

    def supports(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"PPTX not found: {path}")

        doc_id = self.compute_doc_id(path)
        file_size = self.file_size(path)

        prs = Presentation(str(path))
        slide_count = len(prs.slides)

        parts: list[str] = []
        presentation_title = ""

        for n, slide in enumerate(prs.slides, start=1):
            slide_parts: list[str] = [SLIDE_SEPARATOR.format(n=n)]

            for shape in slide.shapes:
                if not shape.has_text_frame:
                    continue
                for para in shape.text_frame.paragraphs:
                    text = para.text.strip()
                    if text:
                        slide_parts.append(text)

            # Extract speaker notes
            if slide.has_notes_slide:
                notes_frame = slide.notes_slide.notes_text_frame
                notes_text = notes_frame.text.strip() if notes_frame else ""
                if notes_text:
                    slide_parts.append(f"[Notes] {notes_text}")

            parts.append("\n".join(slide_parts))

            # Use the first non-empty text on slide 1 as the presentation title
            if n == 1 and len(slide_parts) > 1:
                presentation_title = slide_parts[1]

        full_text = "\n".join(parts)

        metadata = {
            "title": presentation_title or path.stem,
            "author": "",
            "created_at": "",
            "language": "",
            "is_scanned": False,
        }

        logger.info(
            "pptx_parsed",
            doc_id=doc_id,
            filename=path.name,
            slide_count=slide_count,
            total_chars=len(full_text),
        )

        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            format="pptx",
            text=full_text,
            page_count=slide_count,
            file_path=str(path),
            file_size_bytes=file_size,
            metadata=metadata,
        )
