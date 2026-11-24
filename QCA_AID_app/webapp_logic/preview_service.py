"""
Preview Service
===============
Eine Quelle fuer Dokumentvorschau und Annotation im Webapp-Prozess.

Aufgabe: Eingabedokumente (PDF, DOCX, DOC, HTML, MD, TXT) einmalig in ein
Canonical_PDF konvertieren, ablegen und cachen, damit Vorschau und
Annotation dieselbe Geometrie verwenden
(siehe .kiro/specs/pdf-preview-annotation, Anforderungen 1.7-1.9, 11.6).

Warum eine eigene Schicht statt direkter DocumentStore-Aufrufe im UI:
  - Streamlit fuehrt bei jedem Widget-Event das Skript neu aus. Ohne
    In-Memory-Cache wuerde jeder Rerun konvertieren, hashen und indexieren.
  - Konvertierungsfehler sollen die UI nicht abbrechen (Anforderung 1.9),
    sondern als ``PreviewUnavailable`` mit Begruendungen ankommen.

Headless testbar: keine Streamlit-Imports, keine UI-Zustaende.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from QCA_AID_assets.annotation.model import DocumentIndex
from QCA_AID_assets.preprocessing.document_index import DocumentIndexer
from QCA_AID_assets.preprocessing.document_store import (
    STATUS_CACHED,
    STATUS_CONVERTED,
    DocumentStore,
    IngestResult,
)

logger = logging.getLogger(__name__)

#: Formate, die in der Vorschau angeboten werden (Reihenfolge = Anzeige).
SUPPORTED_INPUT_EXTENSIONS: Tuple[str, ...] = (
    ".pdf",
    ".docx",
    ".doc",
    ".html",
    ".htm",
    ".md",
    ".txt",
)

#: Fallback-Seitenmass (A4) wenn PyMuPDF nicht verfuegbar ist.
DEFAULT_PAGE_SIZE: Tuple[float, float] = (595.0, 842.0)

DOCUMENT_JSON = "document.json"
CANONICAL_PDF = "canonical.pdf"
#: Neben datei mit dem SHA-256 der Quelle; invalidiert den Textindex.
INDEX_FINGERPRINT = "index.source_hash"


class PreviewUnavailable(RuntimeError):
    """Das Dokument kann nicht als PDF bereitgestellt werden."""

    def __init__(self, src: Path, reasons: Optional[List[str]] = None):
        self.src = Path(src)
        self.reasons = list(reasons or [])
        message = f"Vorschau nicht verfuegbar: {self.src.name}"
        if self.reasons:
            message += " — " + "; ".join(self.reasons)
        super().__init__(message)


# ---------------------------------------------------------------------------
# Eingabedokumente
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputDocument:
    """Ein Dokument im Eingabeverzeichnis."""

    path: Path
    size_bytes: int
    modified: float

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def doc_id(self) -> str:
        return doc_id_for(self.path)

    def format_size(self) -> str:
        size = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024.0 or unit == "GB":
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} GB"  # pragma: no cover - unerreichbar


def doc_id_for(path: Path) -> str:
    """Bildet einen stabilen, dateisystemtauglichen doc_id.

    Der Name bleibt lesbar (er taucht in Pfaden, Ankern und Exporten auf);
    ein kurzer Hash-Suffix verhindert Kollisionen bei gleichen Stem-Namen
    in unterschiedlichen Verzeichnissen.
    """
    path = Path(path)
    stem = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_") or "doc"
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:6]
    return f"{stem}_{digest}"


def list_input_documents(
    directory: Path,
    extensions: Optional[Tuple[str, ...]] = None,
) -> List[InputDocument]:
    """Listet die vorschaubaren Dokumente eines Verzeichnisses."""
    directory = Path(directory)
    allowed = {ext.lower() for ext in (extensions or SUPPORTED_INPUT_EXTENSIONS)}
    if not directory.is_dir():
        return []
    documents: List[InputDocument] = []
    for entry in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_file():
            continue
        if entry.name.startswith(".") or entry.name.startswith("~$"):
            continue
        if entry.suffix.lower() not in allowed:
            continue
        stat = entry.stat()
        documents.append(
            InputDocument(path=entry, size_bytes=stat.st_size, modified=stat.st_mtime)
        )
    return documents


# ---------------------------------------------------------------------------
# Vorschau-Dokument
# ---------------------------------------------------------------------------


@dataclass
class PreviewDocument:
    """Aufbereitetes Dokument fuer Vorschau und Annotation."""

    doc_id: str
    source_path: Path
    pdf_path: Path
    page_count: int
    page_labels: Optional[List[str]] = None
    index: Optional[DocumentIndex] = None
    converter: Optional[str] = None
    status: str = STATUS_CACHED
    warnings: List[str] = field(default_factory=list)

    @property
    def has_text_layer(self) -> bool:
        """False bei reinen Bild-PDFs (keine OCR, siehe Entscheidung 2026-11-23)."""
        if self.index is None:
            return True
        return any(page.text.strip() for page in self.index.pages)

    @property
    def page_sizes(self) -> List[Tuple[float, float]]:
        return page_sizes(self.pdf_path, self.page_count)


def page_sizes(
    pdf_path: Path,
    page_count: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """Liefert (Breite, Hoehe) je Seite im PDF-Koordinatensystem.

    Faellt ohne PyMuPDF auf das A4-Fallback zurueck, damit der Viewer auch
    ohne optionale Abhaengigkeit rendert.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:  # pragma: no cover - PyMuPDF ist Pflichtabhaengigkeit
        return [DEFAULT_PAGE_SIZE] * (page_count or 1)

    sizes: List[Tuple[float, float]] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            rect = page.rect
            sizes.append((float(rect.width), float(rect.height)))
    return sizes or [DEFAULT_PAGE_SIZE]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class PreviewService:
    """Konvertiert, cached und indexiert Dokumente fuer die UI."""

    def __init__(
        self,
        root: Path,
        store: Optional[DocumentStore] = None,
        indexer: Optional[DocumentIndexer] = None,
    ):
        self.root = Path(root)
        self.store = store or DocumentStore(self.root)
        self.indexer = indexer or DocumentIndexer()
        self._cache: Dict[str, PreviewDocument] = {}

    # -- Cache ---------------------------------------------------------

    @staticmethod
    def _cache_key(src: Path) -> str:
        src = Path(src)
        try:
            stat = src.stat()
        except OSError:
            return f"{src}|missing"
        return f"{src.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"

    def cached(self, src: Path) -> Optional[PreviewDocument]:
        return self._cache.get(self._cache_key(src))

    def invalidate(self, src: Optional[Path] = None) -> None:
        if src is None:
            self._cache.clear()
        else:
            self._cache.pop(self._cache_key(src), None)

    # -- Ingest --------------------------------------------------------

    def get(self, src: Path, doc_id: Optional[str] = None, refresh: bool = False) -> PreviewDocument:
        """Liefert das Vorschau-Dokument (konvertiert beim ersten Aufruf).

        Raises:
            PreviewUnavailable: wenn die Konvertierung scheitert.
        """
        src = Path(src)
        if not src.is_file():
            raise PreviewUnavailable(src, ["Datei nicht gefunden"])

        key = self._cache_key(src)
        if not refresh:
            hit = self._cache.get(key)
            if hit is not None:
                return hit

        result = self.store.ingest(src, doc_id or doc_id_for(src))
        if not result.ok or result.pdf_path is None:
            raise PreviewUnavailable(src, result.reasons)

        index, warnings = self._index_for(result)
        document = PreviewDocument(
            doc_id=result.doc_id,
            source_path=src,
            pdf_path=Path(result.pdf_path),
            page_count=index.page_count if index else 1,
            page_labels=index.page_labels if index else None,
            index=index,
            converter=result.converter,
            status=result.status,
            warnings=list(result.warnings) + warnings,
        )
        self._cache[key] = document
        return document

    def _index_for(self, result: IngestResult) -> Tuple[Optional[DocumentIndex], List[str]]:
        """Laedt document.json oder baut es neu; Fehler sind nicht fatal."""
        warnings: List[str] = []
        json_path = Path(result.doc_dir) / DOCUMENT_JSON
        pdf_path = Path(result.pdf_path)
        fingerprint_path = Path(result.doc_dir) / INDEX_FINGERPRINT

        # Der Index gehoert zum Inhalt der Quelldatei, nicht nur zum doc_id:
        # nach einer erneuten Konvertierung (geaenderte Quelldatei) muss
        # document.json neu erzeugt werden. Der Fingerprint ist der SHA-256 der
        # Quelle - robuster als Dateizeiten, weil das Canonical_PDF bei jedem
        # Ingest neu kopiert wird.
        fingerprint_matches = False
        if json_path.is_file() and fingerprint_path.is_file() and result.source_hash:
            try:
                fingerprint_matches = (
                    fingerprint_path.read_text(encoding="utf-8").strip()
                    == result.source_hash
                )
            except OSError:
                fingerprint_matches = False

        if json_path.is_file() and fingerprint_matches:
            try:
                return (
                    DocumentIndex.from_dict(
                        json.loads(json_path.read_text(encoding="utf-8"))
                    ),
                    warnings,
                )
            except (ValueError, KeyError, OSError) as exc:
                warnings.append(f"document.json unlesbar, wird neu erzeugt: {exc}")
        elif json_path.is_file():
            if result.source_hash and not fingerprint_path.is_file():
                warnings.append(
                    "Textindex ohne Inhalts-Fingerprint - wird einmalig neu erzeugt"
                )
            else:
                warnings.append("Quelldatei geaendert - Textindex wird neu erzeugt")

        try:
            index = self.indexer.build(pdf_path, result.doc_id)
        except Exception as exc:  # noqa: BLE001 - Indexfehler duerfen die UI nicht abbrechen
            logger.warning("Indexierung fehlgeschlagen fuer %s: %s", result.doc_id, exc)
            warnings.append(f"Textindex nicht erzeugbar: {exc}")
            return None, warnings

        try:
            self.store.write_index(result.doc_id, index.to_dict())
            self._write_fingerprint(fingerprint_path, result.source_hash)
        except OSError as exc:  # pragma: no cover - Schreibfehler
            warnings.append(f"document.json nicht schreibbar: {exc}")
        return index, warnings

    @staticmethod
    def _write_fingerprint(path: Path, source_hash: Optional[str]) -> None:
        """Merkt sich den Quell-Hash neben dem Index (Invalidierung)."""
        if not source_hash:
            return
        path.write_text(source_hash, encoding="utf-8")

    # -- Verzeichnis ---------------------------------------------------

    def documents(self, input_dir: Path) -> List[InputDocument]:
        return list_input_documents(input_dir)

    def not_convertible(self) -> Dict[str, Dict]:
        """Dokumente, deren Konvertierung dauerhaft scheitert."""
        return self.store.load_not_convertible()


__all__ = [
    "SUPPORTED_INPUT_EXTENSIONS",
    "DEFAULT_PAGE_SIZE",
    "PreviewUnavailable",
    "InputDocument",
    "PreviewDocument",
    "PreviewService",
    "doc_id_for",
    "list_input_documents",
    "page_sizes",
]
