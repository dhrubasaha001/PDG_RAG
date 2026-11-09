"""
backend/ingestion/parsers/xlsx_parser.py
Excel parser using openpyxl.

Each worksheet is prefixed with a '--- SHEET: {name} ---' marker.
Rows are tab-separated; empty rows are skipped.
"""

from __future__ import annotations

from pathlib import Path

try:
    import openpyxl
except ImportError as exc:
    raise ImportError("openpyxl is required: pip install openpyxl") from exc

from backend.core.logging import get_logger
from backend.ingestion.parsers.base import BaseParser, ParsedDocument

logger = get_logger("ingestion.xlsx_parser")

SHEET_SEPARATOR = "\n\n--- SHEET: {name} ---\n\n"


class XlsxParser(BaseParser):
    """Parse .xlsx files, extracting each sheet as tab-separated text."""

    SUPPORTED_EXTENSIONS = {".xlsx", ".xls"}

    def supports(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"XLSX not found: {path}")

        doc_id = self.compute_doc_id(path)
        file_size = self.file_size(path)

        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        sheet_count = len(wb.sheetnames)

        parts: list[str] = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            sheet_parts: list[str] = [SHEET_SEPARATOR.format(name=sheet_name)]
            row_count = 0

            for row in ws.iter_rows(values_only=True):
                # Convert all cells to strings, skip entirely-None rows
                cells = [str(cell) if cell is not None else "" for cell in row]
                if all(c == "" for c in cells):
                    continue
                sheet_parts.append("\t".join(cells))
                row_count += 1

            if row_count > 0:
                parts.append("\n".join(sheet_parts))

        wb.close()
        full_text = "\n".join(parts)

        metadata = {
            "title": path.stem,
            "author": "",
            "created_at": "",
            "language": "",
            "is_scanned": False,
            "sheet_count": sheet_count,
        }

        logger.info(
            "xlsx_parsed",
            doc_id=doc_id,
            filename=path.name,
            sheet_count=sheet_count,
            total_chars=len(full_text),
        )

        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            format="xlsx",
            text=full_text,
            page_count=sheet_count,
            file_path=str(path),
            file_size_bytes=file_size,
            metadata=metadata,
        )
