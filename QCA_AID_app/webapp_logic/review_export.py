"""
Rezensionspaket
===============
Schreibt den Stand aus dem Prüfen-Tab weg: annotiertes PDF, Kodierungen
(XLSX mit Änderungsblatt) und ein Entscheidungsprotokoll (JSON).

Ablage und Namensstempel folgen der Analyse: eigener Unterordner
``output/review/`` und Zeitstempel ``JJJJMMTT_HHMMSS`` im Dateinamen. Die
Analyse-Ausgabe selbst wird nie überschrieben - der Rezensionsstand ist ein
eigenes Artefakt (Nachvollziehbarkeit, Anforderung 8.5/8.6).

Die Logik ist headless (testbar); die Streamlit-Verdrahtung steht in
``webapp_components.review_ui``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from QCA_AID_assets.annotation.export_data import export_review_xlsx
from QCA_AID_assets.annotation.export_pdf import PdfBakingExporter
from QCA_AID_assets.annotation.model import Coding
from QCA_AID_assets.utils.export.paths import STAMP_FORMAT, export_stamp

logger = logging.getLogger(__name__)

#: Unterordner in ``output/`` für Rezensions-Exporte.
REVIEW_SUBDIR = "review"


@dataclass(frozen=True)
class ReviewPackage:
    """Ergebnis eines Rezensions-Exports."""

    folder: Path
    pdf: Path
    xlsx: Path
    json: Path
    codings: int
    decisions: int

    @property
    def files(self) -> List[Path]:
        return [self.pdf, self.xlsx, self.json]


def export_paths(output_dir: Path, doc_stem: str, stamp: Optional[str] = None) -> Dict[str, Path]:
    """Zielpfade des Rezensionspakets (ohne etwas anzulegen)."""
    stamp = stamp or export_stamp()
    folder = Path(output_dir) / REVIEW_SUBDIR
    return {
        "folder": folder,
        "pdf": folder / f"{doc_stem}_annotiert_{stamp}.pdf",
        "xlsx": folder / f"{doc_stem}_review_{stamp}.xlsx",
        "json": folder / f"{doc_stem}_review_{stamp}.json",
    }


def build_decisions(
    codings: Sequence[Coding],
    original: Optional[Sequence[Coding]] = None,
) -> List[Dict[str, object]]:
    """Feldweise Entscheidungen je Kodierung (Basis für XLSX/JSON)."""
    before = {coding.segment_id: coding for coding in (original or [])}
    decisions: List[Dict[str, object]] = []
    for coding in codings:
        previous = before.get(coding.segment_id)
        changes: Dict[str, Dict[str, str]] = {}
        if previous is not None:
            if previous.category != coding.category:
                changes["kategorie"] = {"vorher": previous.category, "nachher": coding.category}
            if list(previous.subcategories) != list(coding.subcategories):
                changes["subkategorien"] = {
                    "vorher": "; ".join(previous.subcategories),
                    "nachher": "; ".join(coding.subcategories),
                }
            if previous.status != coding.status:
                changes["status"] = {"vorher": previous.status, "nachher": coding.status}
            if previous.created_by != coding.created_by:
                changes["erstellt_von"] = {
                    "vorher": previous.created_by,
                    "nachher": coding.created_by,
                }
        elif coding.status != "proposed":
            changes["status"] = {"vorher": "", "nachher": coding.status}

        decisions.append(
            {
                "segment_id": coding.segment_id,
                "kategorie": coding.category,
                "subkategorien": list(coding.subcategories),
                "status": coding.status,
                "erstellt_von": coding.created_by,
                "seite": coding.page if coding.page is not None else (
                    coding.anchor.page if coding.anchor else None
                ),
                "verankert": bool(coding.anchor and coding.anchor.anchored),
                "begruendung": coding.justification or "",
                "aenderungen": changes,
            }
        )
    return decisions


def export_review_package(
    pdf_path: Path,
    codings: Sequence[Coding],
    output_dir: Path,
    doc_stem: str,
    document_label: str = "",
    source_label: str = "",
    original: Optional[Sequence[Coding]] = None,
    stamp: Optional[str] = None,
    exporter: Optional[PdfBakingExporter] = None,
) -> ReviewPackage:
    """Schreibt PDF + XLSX + JSON des Rezensionsstands.

    Args:
        pdf_path: kanonisches PDF der Vorschau (Quelle für die Annotationen).
        codings: aktueller Stand inkl. Anker.
        output_dir: ``output/`` des Projekts.
        doc_stem: Dateiname ohne Endung (Namensstempel der Exporte).
        original: Stand beim Laden - nur so sind Umkodierungen im
            Änderungsblatt sichtbar.
        exporter: optionaler PdfBakingExporter (Tests/abweichende Farben).
    """
    paths = export_paths(output_dir, doc_stem, stamp=stamp)
    folder: Path = paths["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    exported_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    (exporter or PdfBakingExporter()).bake(
        pdf_path=Path(pdf_path),
        codings=codings,
        out_path=paths["pdf"],
    )

    export_review_xlsx(
        codings,
        paths["xlsx"],
        original=original,
        document_label=document_label,
        source_label=source_label,
        exported_at=exported_at,
    )

    decisions = build_decisions(codings, original)
    changed = [entry for entry in decisions if entry["aenderungen"]]
    payload = {
        "dokument": document_label,
        "quelle": source_label,
        "exportiert_am": exported_at,
        "anzahl_kodierungen": len(codings),
        "anzahl_geaendert": len(changed),
        "kodierungen": decisions,
    }
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        "Rezensionspaket geschrieben: %s (%d Kodierungen, %d geändert)",
        folder,
        len(codings),
        len(changed),
    )
    return ReviewPackage(
        folder=folder,
        pdf=paths["pdf"],
        xlsx=paths["xlsx"],
        json=paths["json"],
        codings=len(codings),
        decisions=len(changed),
    )


__all__ = [
    "REVIEW_SUBDIR",
    "STAMP_FORMAT",
    "ReviewPackage",
    "export_paths",
    "build_decisions",
    "export_review_package",
]
