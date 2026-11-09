"""
backend/ingestion/parsers/text_parser.py
Plain-text family parser: .md, .html, .htm, .txt, .rst, .csv

Markdown → strip syntax, return plain text
HTML     → BeautifulSoup text extraction (scripts/styles removed)
TXT/RST  → read as-is
CSV      → preserve as-is (tab-like structure)
"""

from __future__ import annotations

from pathlib import Path

from backend.core.logging import get_logger
from backend.ingestion.parsers.base import BaseParser, ParsedDocument

logger = get_logger("ingestion.text_parser")

SUPPORTED = {".md", ".markdown", ".html", ".htm", ".txt", ".rst", ".csv"}


class TextParser(BaseParser):
    """Parse plain-text family files."""

    def supports(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in SUPPORTED

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        doc_id = self.compute_doc_id(path)
        file_size = self.file_size(path)
        suffix = path.suffix.lower()

        raw = path.read_text(encoding="utf-8", errors="replace")

        if suffix in {".md", ".markdown"}:
            text = _strip_markdown(raw)
            fmt = "md"
        elif suffix in {".html", ".htm"}:
            text = _strip_html(raw)
            fmt = "html"
        else:
            text = raw
            fmt = "txt"

        metadata = {
            "title": path.stem,
            "author": "",
            "created_at": "",
            "language": "",
            "is_scanned": False,
        }

        logger.info(
            "text_parsed",
            doc_id=doc_id,
            filename=path.name,
            format=fmt,
            total_chars=len(text),
        )

        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            format=fmt,
            text=text,
            page_count=1,
            file_path=str(path),
            file_size_bytes=file_size,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """
    Convert markdown to plain text by stripping common syntax.
    Uses markdown-it-py if available; falls back to regex stripping.
    """
    try:
        from markdown_it import MarkdownIt  # type: ignore
        md = MarkdownIt()
        tokens = md.parse(text)
        return _extract_token_text(tokens)
    except ImportError:
        pass

    import re
    # Remove fenced code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    # Remove headings markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)
    # Remove links
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove images
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    # Remove blockquotes
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_token_text(tokens: list) -> str:
    """Recursively extract inline text from markdown-it token tree."""
    parts: list[str] = []
    for token in tokens:
        if token.type == "inline" and token.children:
            for child in token.children:
                if child.type == "text":
                    parts.append(child.content)
        elif hasattr(token, "children") and token.children:
            parts.append(_extract_token_text(token.children))
    return " ".join(p for p in parts if p)


def _strip_html(text: str) -> str:
    """Extract visible text from HTML, removing scripts and styles."""
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(text, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style", "head", "meta", "link"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        import re
        # Fallback: naive tag stripping
        text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        return text.strip()
