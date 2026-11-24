"""
Document Indexer
================
Erzeugt den Document_Index (document.json) aus einem Canonical_PDF
(siehe .kiro/specs/pdf-preview-annotation, Anforderungen 2.1, 2.2, 2.5).

Enthaelt:
  - Seitenzahl
  - gedruckte Seitenlabels (best-effort, mit Style-Expansion D/R/A)
  - Text je Seite inkl. Start-/End-Offsets im Gesamttext
  - Page_Marker als Offset-Bereiche je Seite

PyMuPDF ist primaer; pypdf dient als Text-Fallback.
"""

import logging
from pathlib import Path
from typing import List, Optional

from ..annotation.model import DocumentIndex, Marker, PageText

logger = logging.getLogger(__name__)

PAGE_SEPARATOR = "\n"


class IndexBuildError(Exception):
    """Wird geworfen, wenn aus einem PDF kein Index erzeugt werden kann."""


class DocumentIndexer:
    """Baut den Document_Index aus einem Canonical_PDF."""

    def __init__(self, page_separator: str = PAGE_SEPARATOR):
        self.page_separator = page_separator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, pdf_path: Path, doc_id: str) -> DocumentIndex:
        """Erzeugt den Index.

        Raises:
            IndexBuildError: wenn das PDF nicht lesbar ist.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise IndexBuildError(f"PDF nicht gefunden: {pdf_path}")

        page_texts, labels = self._read_pages(pdf_path)

        pages: List[PageText] = []
        markers: List[Marker] = []
        offset = 0
        for page_no, text in enumerate(page_texts, start=1):
            text = text or ""
            start = offset
            end = start + len(text)
            pages.append(PageText(page=page_no, text=text, start=start, end=end))
            markers.append(Marker(page=page_no, start=start, end=end))
            offset = end + len(self.page_separator)

        return DocumentIndex(
            doc_id=doc_id,
            page_count=len(pages),
            pages=pages,
            page_labels=labels,
            page_markers=markers,
        )

    # ------------------------------------------------------------------
    # PDF-Lesen
    # ------------------------------------------------------------------

    def _read_pages(self, pdf_path: Path):
        try:
            import fitz  # PyMuPDF

            with fitz.open(pdf_path) as doc:
                texts = [page.get_text("text") for page in doc]
                labels = _expand_page_labels(doc)
                return texts, labels
        except ImportError:
            logger.warning("PyMuPDF nicht verfuegbar, nutze pypdf-Fallback")
        except Exception as exc:
            raise IndexBuildError(f"PDF nicht lesbar (PyMuPDF): {exc}") from exc

        try:
            import pypdf

            with open(pdf_path, "rb") as fh:
                reader = pypdf.PdfReader(fh)
                texts = [page.extract_text() or "" for page in reader.pages]
                return texts, None
        except Exception as exc:
            raise IndexBuildError(f"PDF nicht lesbar (pypdf): {exc}") from exc


# --------------------------------------------------------------------------
# Seitenlabel-Expansion
# --------------------------------------------------------------------------

def _format_number(value: int, style: Optional[str]) -> str:
    if style == "D" or style is None:
        return str(value)
    if style == "R":
        return _to_roman(value).upper()
    if style == "r":
        return _to_roman(value).lower()
    if style == "A":
        return _to_alpha(value).upper()
    if style == "a":
        return _to_alpha(value).lower()
    return str(value)


def _to_roman(value: int) -> str:
    if value <= 0:
        return ""
    table = [
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    ]
    out = []
    for arabic, roman in table:
        while value >= arabic:
            out.append(roman)
            value -= arabic
    return "".join(out)


def _to_alpha(value: int) -> str:
    if value <= 0:
        return ""
    # 1 -> a, 26 -> z, 27 -> aa ...
    letters = []
    while value > 0:
        value -= 1
        letters.append(chr(ord("a") + value % 26))
        value //= 26
    return "".join(reversed(letters))


def _expand_page_labels(doc) -> Optional[List[str]]:
    """Expandiert PyMuPDF-Seitenlabel-Definitionen zu einer Liste je Seite."""
    try:
        raw = doc.get_page_labels() if hasattr(doc, "get_page_labels") else None
    except Exception:
        return None
    if not raw:
        return None

    page_count = doc.page_count
    labels: List[str] = [""] * page_count
    for entry in raw:
        start = int(entry.get("startpage", 0))
        first = int(entry.get("firstpagenum", 1))
        style = entry.get("style")
        prefix = entry.get("prefix", "") or ""
        # Ende = naechster Eintrag oder Dokumentende
        next_starts = [
            int(e.get("startpage", 0)) for e in raw if int(e.get("startpage", 0)) > start
        ]
        end = min(next_starts) if next_starts else page_count
        for idx, page_index in enumerate(range(start, end)):
            if 0 <= page_index < page_count:
                labels[page_index] = prefix + _format_number(first + idx, style)
    return labels


__all__ = ["DocumentIndexer", "IndexBuildError", "PAGE_SEPARATOR"]