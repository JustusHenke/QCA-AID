"""
Document Store
==============
Ablage und Caching der konvertierten Dokumente fuer Vorschau und Annotation
(siehe .kiro/specs/pdf-preview-annotation, Anforderungen 1.7-1.9, 9.3, 9.4).

Layout:
  <root>/converted_cache/<sha256>/canonical.pdf     (Inhalts-Hash-Cache)
  <root>/documents/<doc_id>/original<ext>           (unveraenderte Quelle)
  <root>/documents/<doc_id>/canonical.pdf           (Anzeige + Annotation)
  <root>/documents/<doc_id>/document.json           (via write_index, Task 5)
  <root>/documents/_not_convertible.json            (Fehlerregister)

Prinzip: Konvertierungsfehler brechen die Verarbeitung NICHT ab; das Dokument
wird als 'not_convertible' registriert und der Rest laeuft weiter.
"""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .document_converter import ConvertError, ConvertResult, DocumentConverter, is_valid_pdf

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONVERT_MB = 200

STATUS_CONVERTED = "converted"
STATUS_CACHED = "cached"
STATUS_NOT_CONVERTIBLE = "not_convertible"


@dataclass
class IngestResult:
    """Ergebnis der Ablage eines Dokuments."""

    doc_id: str
    src: Path
    status: str
    doc_dir: Path
    pdf_path: Optional[Path] = None
    converter: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_hash: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status in (STATUS_CONVERTED, STATUS_CACHED)


class DocumentStore:
    """Verwaltet Cache und Ablage der Canonical_PDFs."""

    def __init__(
        self,
        root: Path,
        converter: Optional[DocumentConverter] = None,
        max_convert_mb: float = DEFAULT_MAX_CONVERT_MB,
    ):
        self.root = Path(root)
        self.converter = converter or DocumentConverter()
        self.max_convert_mb = max_convert_mb
        self.cache_dir = self.root / "converted_cache"
        self.documents_dir = self.root / "documents"
        self.not_convertible_path = self.documents_dir / "_not_convertible.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Hash und Pfade
    # ------------------------------------------------------------------

    @staticmethod
    def content_hash(src: Path) -> str:
        digest = hashlib.sha256()
        with open(src, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def doc_dir(self, doc_id: str) -> Path:
        return self.documents_dir / doc_id

    def _check_size(self, src: Path) -> Optional[str]:
        size_mb = src.stat().st_size / (1024 * 1024)
        if size_mb > self.max_convert_mb:
            return (
                f"Datei zu gross: {size_mb:.1f} MB > "
                f"Limit {self.max_convert_mb:.0f} MB"
            )
        return None

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, src: Path, doc_id: str) -> IngestResult:
        """Konvertiert (oder cached) ein Dokument und legt es ab.

        Wirft keine Konvertierungsfehler; scheitert die Konvertierung, wird das
        Dokument als 'not_convertible' registriert.
        """
        src = Path(src)
        doc_dir = self.doc_dir(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        size_error = self._check_size(src)
        if size_error:
            return self._mark_not_convertible(doc_id, src, doc_dir, [size_error])

        try:
            digest = self.content_hash(src)
        except OSError as exc:
            return self._mark_not_convertible(
                doc_id, src, doc_dir, [f"Datei nicht lesbar: {exc}"]
            )

        cache_entry = self.cache_dir / digest
        cached_pdf = cache_entry / "canonical.pdf"
        cached = is_valid_pdf(cached_pdf)

        result: Optional[ConvertResult] = None
        reasons: List[str] = []
        if not cached:
            try:
                result = self.converter.convert(src, cache_entry, canonical_name="canonical.pdf")
            except ConvertError as exc:
                return self._mark_not_convertible(doc_id, src, doc_dir, exc.reasons)
            except Exception as exc:  # unerwartete Fehler isolieren
                return self._mark_not_convertible(
                    doc_id, src, doc_dir, [f"unerwarteter Konvertierungsfehler: {exc}"]
                )
            if not is_valid_pdf(cached_pdf):
                reasons.append("Konverter lieferte kein gueltiges PDF")
                return self._mark_not_convertible(doc_id, src, doc_dir, reasons)

        # Kopie in die Dokumentablage
        target_pdf = doc_dir / "canonical.pdf"
        shutil.copyfile(cached_pdf, target_pdf)

        # Original fuer Nachvollziehbarkeit mitsichern
        original_target = doc_dir / f"original{src.suffix.lower()}"
        if src.resolve() != original_target.resolve():
            shutil.copyfile(src, original_target)

        # Ein frueherer Fehlversuch (z.B. fehlender Konverter) ist damit erledigt.
        self._clear_not_convertible(doc_id)

        return IngestResult(
            doc_id=doc_id,
            src=src,
            status=STATUS_CACHED if (cached and result is None) else STATUS_CONVERTED,
            doc_dir=doc_dir,
            pdf_path=target_pdf,
            converter=result.converter if result else "cache",
            reasons=[],
            warnings=result.warnings if result else [],
            source_hash=digest,
        )

    def ingest_directory(self, input_dir: Path, extensions: Optional[Iterable[str]] = None) -> List[IngestResult]:
        """Konvertiert alle passenden Dateien im Verzeichnis.

        Fehler eines Dokuments brechen die Verarbeitung nicht ab.
        """
        input_dir = Path(input_dir)
        allowed = {ext.lower() for ext in extensions} if extensions else None
        results: List[IngestResult] = []
        for src in sorted(input_dir.iterdir()):
            if not src.is_file():
                continue
            if allowed is not None and src.suffix.lower() not in allowed:
                continue
            results.append(self.ingest(src, doc_id=src.stem))
        return results

    # ------------------------------------------------------------------
    # Index (von Task 5 genutzt)
    # ------------------------------------------------------------------

    def write_index(self, doc_id: str, index_dict: Dict) -> Path:
        """Schreibt document.json in die Dokumentablage."""
        doc_dir = self.doc_dir(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        path = doc_dir / "document.json"
        path.write_text(json.dumps(index_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Fehlerregister
    # ------------------------------------------------------------------

    def _mark_not_convertible(
        self, doc_id: str, src: Path, doc_dir: Path, reasons: List[str]
    ) -> IngestResult:
        logger.warning("Dokument %s nicht konvertierbar: %s", doc_id, " | ".join(reasons))
        registry = self.load_not_convertible()
        registry[doc_id] = {
            "source": str(src),
            "reasons": reasons,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save_not_convertible(registry)
        return IngestResult(
            doc_id=doc_id,
            src=src,
            status=STATUS_NOT_CONVERTIBLE,
            doc_dir=doc_dir,
            pdf_path=None,
            converter=None,
            reasons=reasons,
        )

    def _clear_not_convertible(self, doc_id: str) -> None:
        """Entfernt einen Eintrag, sobald die Konvertierung gelingt.

        Ohne das bliebe nach dem Nachinstallieren eines Konverters der alte
        Fehlergrund im Register stehen und die UI wuerde weiter einen Fehler
        behaupten, obwohl die Vorschau funktioniert.
        """
        registry = self.load_not_convertible()
        if registry.pop(doc_id, None) is None:
            return
        try:
            self._save_not_convertible(registry)
        except OSError as exc:  # pragma: no cover - Schreibfehler
            logger.warning("Fehlerregister nicht aktualisierbar: %s", exc)

    def load_not_convertible(self) -> Dict:
        if not self.not_convertible_path.exists():
            return {}
        try:
            return json.loads(self.not_convertible_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_not_convertible(self, registry: Dict) -> None:
        self.not_convertible_path.parent.mkdir(parents=True, exist_ok=True)
        self.not_convertible_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )


__all__ = [
    "DocumentStore",
    "IngestResult",
    "STATUS_CONVERTED",
    "STATUS_CACHED",
    "STATUS_NOT_CONVERTIBLE",
    "DEFAULT_MAX_CONVERT_MB",
]