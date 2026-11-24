"""
PDF Annotator
=============
Annotiert Original-PDFs mit kodierten Textstellen
(siehe .kiro/specs/pdf-preview-annotation, Anforderung 1.1/4.1).

Vollstaendig neu implementiert: statt geratener Fuzzy-Rechtecke werden
Anker exakt aufgeloest (AnchorResolver), Quads je Textzeile erzeugt und als
Highlight-Annotationen mit Multiply-Blend geschrieben (PdfBakingExporter).

Die oeffentliche API (``PDFAnnotator(results_exporter)`` und
``annotate_pdf_with_codings(...)``) bleibt unveraendert, damit ResultsExporter
ohne Anpassung weiterlaeuft.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...annotation.anchor import AnchorResolver
from ...annotation.export_pdf import PdfBakingExporter
from ...annotation.model import Coding
from ...preprocessing.document_index import DocumentIndexer
from .paths import annotated_pdf_path

# Entfernt Pfad-/Datei-Artefakte, ohne den eigentlichen Inhalt zu verlieren.
_PATH_ARTIFACT_RE = re.compile(
    r"(file:///\S+|[A-Za-z]:\\\\[^\s]+|/Users/\S+|OneDrive\S*|Projekte/Forschung\S*)"
)

MAX_QUOTE_CHARS = 240


class PDFAnnotator:
    """Erzeugt annotierte PDFs aus QCA-AID-Kodierungen."""

    def __init__(self, results_exporter: Any):
        self.results_exporter = results_exporter
        # Kompatibilitaetsattribute (frueher fuer die Fuzzy-Heuristik genutzt).
        self.fuzzy_match_threshold = 0.85
        self.verbose = False
        self.category_colors: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Oeffentliche API (unveraendert)
    # ------------------------------------------------------------------

    def annotate_pdf_with_codings(
        self,
        pdf_path: str,
        codings: List[Dict],
        chunks: Dict[str, List[str]],
        output_path: str = None,
    ) -> Optional[str]:
        """Annotiert pdf_path mit den Kodierungen und speichert das Ergebnis.

        Args:
            pdf_path: Original-PDF
            codings: Kodierungs-Dicts (segment_id, category, ...)
            chunks: doc_name -> Liste der Chunk-Texte
            output_path: Zielpfad. Default:
                ``<pdf-verzeichnis>/Annotated/<stem>_QCA_annotiert_<stamp>.pdf``
                (gleicher Namensstempel wie die Excel-Ausgabe).

        Returns:
            Pfad des annotierten PDFs oder None.
        """
        source = Path(pdf_path)
        if not source.exists():
            print(f"PDF-Fehler: {source.name} existiert nicht")
            return None

        try:
            index = DocumentIndexer().build(source, doc_id=source.stem)
        except Exception as exc:
            print(f"PDF-Fehler {source.name}: {exc}")
            return None

        resolver = AnchorResolver()
        coding_objects = self._build_codings(source, index, resolver, codings, chunks)
        if not coding_objects:
            print(f"Warnung: Keine auflösbaren Kodierungen für {source.name}")
            return None

        if not output_path:
            output_path = str(annotated_pdf_path(source.parent, source.stem))

        try:
            exporter = PdfBakingExporter(category_colors=self._export_colors())
            result = exporter.bake(
                source, coding_objects, Path(output_path), legend=True, margin_notes=True
            )
        except Exception as exc:
            print(f"PDF-Fehler {source.name}: {exc}")
            return None

        anchored = sum(1 for c in coding_objects if c.anchor and c.anchor.anchored)
        print(
            f"PDF-Export: {source.name} -> {Path(result).name} "
            f"({anchored}/{len(coding_objects)} Kodierungen verankert)"
        )
        return str(result)

    # ------------------------------------------------------------------
    # Anker-Aufbau
    # ------------------------------------------------------------------

    def _build_codings(self, source: Path, index, resolver, codings, chunks) -> List[Coding]:
        objects: List[Coding] = []
        for raw in codings:
            segment_id = str(raw.get("segment_id", ""))
            if "_chunk_" not in segment_id:
                continue

            doc_name, chunk_part = segment_id.split("_chunk_", 1)
            try:
                chunk_id = int(chunk_part.split("-")[0])
            except ValueError:
                continue

            doc_chunks = chunks.get(doc_name)
            if not doc_chunks or chunk_id >= len(doc_chunks):
                continue

            quote = self._select_quote(str(doc_chunks[chunk_id]))
            if not quote:
                continue

            page = self._estimate_page(index, quote)
            anchor = resolver.resolve(
                source, doc_id=index.doc_id, page=page, quote=quote
            )

            category = raw.get("category") or "Nicht kodiert"
            objects.append(
                Coding(
                    segment_id=segment_id,
                    doc_id=index.doc_id,
                    category=category,
                    subcategories=list(raw.get("subcategories") or []),
                    confidence=self._confidence(raw.get("confidence")),
                    justification=raw.get("justification"),
                    page=page,
                    anchor=anchor,
                )
            )
        return objects

    @staticmethod
    def _select_quote(text: str) -> str:
        """Waehlt ein ankerbares Zitat (Pfad-Artefakte entfernt, Wortgrenze)."""
        cleaned = _PATH_ARTIFACT_RE.sub(" ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) < 20:
            return ""

        quote = cleaned[:MAX_QUOTE_CHARS]
        last_space = quote.rfind(" ")
        if last_space > 20:
            quote = quote[:last_space]
        return quote.strip()

    @staticmethod
    def _estimate_page(index, quote: str) -> int:
        """Findet die Seite, deren Text das Zitat enthaelt (best effort)."""
        needle = re.sub(r"\s+", " ", quote).lower()[:120]
        for page in index.pages:
            haystack = re.sub(r"\s+", " ", page.text).lower()
            if needle and needle in haystack:
                return page.page
        return 1

    @staticmethod
    def _confidence(value) -> Optional[float]:
        if isinstance(value, dict):
            value = value.get("total")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _export_colors(self) -> Dict[str, str]:
        colors = getattr(self.results_exporter, "category_colors", None)
        if isinstance(colors, dict):
            return {k: v for k, v in colors.items() if isinstance(v, str) and v.startswith("#")}
        return {}


__all__ = ["PDFAnnotator"]