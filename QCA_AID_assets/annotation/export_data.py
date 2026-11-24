"""
Data Export
===========
Exportiert Kodierungen als CSV/XLSX mit vollstaendiger Anker-Information
(siehe .kiro/specs/pdf-preview-annotation, Anforderungen 8.5, 8.6).

Spalten: Dokument, Seite, Seitenlabel, quote, Kategorie, Subkategorien,
Konfidenz, Status, Verankert (anchored/unanchored), Begruendung, Segment.
Unanchored-Kodierungen werden explizit als solche gekennzeichnet.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .model import Coding

logger = logging.getLogger(__name__)

COLUMNS = [
    "dokument",
    "seite",
    "seitenlabel",
    "quote",
    "kategorie",
    "subkategorien",
    "konfidenz",
    "status",
    "verankert",
    "begruendung",
    "segment_id",
]

SUB_SEPARATOR = "; "


def build_rows(codings: Sequence[Coding]) -> List[Dict[str, object]]:
    """Baut die Exportzeilen aus den Kodierungen."""
    rows: List[Dict[str, object]] = []
    for coding in codings:
        anchor = coding.anchor
        anchored = bool(anchor and anchor.anchored)
        page = coding.page if coding.page is not None else (anchor.page if anchor else None)
        page_label = anchor.page_label if anchor else None
        quote = anchor.quote if anchor else ""

        rows.append(
            {
                "dokument": coding.doc_id,
                "seite": page if page is not None else "",
                "seitenlabel": page_label or "",
                "quote": quote,
                "kategorie": coding.category,
                "subkategorien": SUB_SEPARATOR.join(coding.subcategories),
                "konfidenz": coding.confidence if coding.confidence is not None else "",
                "status": coding.status,
                "verankert": "anchored" if anchored else "unanchored",
                "begruendung": coding.justification or "",
                "segment_id": coding.segment_id,
            }
        )
    return rows


def export_csv(codings: Sequence[Coding], path: Path) -> Path:
    """Schreibt die Kodierungen als CSV (UTF-8 mit BOM fuer Excel)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(codings)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_xlsx(codings: Sequence[Coding], path: Path, sheet_name: str = "Kodierungen") -> Path:
    """Schreibt die Kodierungen als XLSX."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openpyxl nicht installiert") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    sheet.append(COLUMNS)
    for cell in sheet[1]:
        cell.font = Font(bold=True)

    rows = build_rows(codings)
    for row in rows:
        sheet.append([row[key] for key in COLUMNS])

    for index, _ in enumerate(COLUMNS, start=1):
        letter = sheet.cell(row=1, column=index).column_letter
        sheet.column_dimensions[letter].width = max(12, len(COLUMNS[index - 1]) + 4)

    workbook.save(str(path))
    return path


def export(codings: Sequence[Coding], path: Path) -> Path:
    """Exportiert je nach Endung nach CSV oder XLSX."""
    suffix = Path(path).suffix.lower()
    if suffix == ".xlsx":
        return export_xlsx(codings, path)
    if suffix == ".csv":
        return export_csv(codings, path)
    raise ValueError(f"nicht unterstuetzte Endung: {suffix}")


# --------------------------------------------------------------------------
# Rezensionsstand (Kodierungen + Entscheidungen)
# --------------------------------------------------------------------------

REVIEW_COLUMNS = [
    "dokument",
    "seite",
    "seitenlabel",
    "segment_id",
    "kategorie",
    "subkategorien",
    "konfidenz",
    "status",
    "erstellt_von",
    "verankert",
    "quote",
    "begruendung",
    "textanker",
]

_CHANGE_FIELDS = ("kategorie", "subkategorien", "status", "erstellt_von")


def build_review_rows(codings: Sequence[Coding]) -> List[Dict[str, object]]:
    """Zeilen des Rezensionsstands (deutsche Spalten, ein Datensatz je Kodierung)."""
    rows: List[Dict[str, object]] = []
    for coding in codings:
        anchor = coding.anchor
        anchored = bool(anchor and anchor.anchored)
        page = coding.page if coding.page is not None else (anchor.page if anchor else None)
        rows.append(
            {
                "dokument": coding.doc_id,
                "seite": page if page is not None else "",
                "seitenlabel": (anchor.page_label if anchor else None) or "",
                "segment_id": coding.segment_id,
                "kategorie": coding.category,
                "subkategorien": SUB_SEPARATOR.join(coding.subcategories),
                "konfidenz": coding.confidence if coding.confidence is not None else "",
                "status": coding.status,
                "erstellt_von": coding.created_by,
                "verankert": "ja" if anchored else "nein",
                "quote": (anchor.quote if anchor else "") or "",
                "begruendung": coding.justification or "",
                # Textanker (Spec text-anker-export, R6): dieselbe ' | '-
                # Liste wie in der Analyse-XLSX - ohne Anker leer.
                "textanker": " | ".join(coding.text_anchors),
            }
        )
    return rows


def review_changes(
    codings: Sequence[Coding],
    original: Optional[Sequence[Coding]] = None,
) -> List[Dict[str, object]]:
    """Feldweise Aenderungen gegenueber dem geladenen Stand.

    Ohne ``original`` bleiben nur die Entscheidungen uebrig, die ohne Vorlage
    erkennbar sind (Status, Kodierer). Eine Zeile je geaendertem Feld:
    ``segment_id, kategorie, feld, vorher, nachher``.
    """
    before = {coding.segment_id: coding for coding in (original or [])}
    changes: List[Dict[str, object]] = []
    for coding in codings:
        previous = before.get(coding.segment_id)
        pairs: Dict[str, tuple] = {}
        if previous is not None:
            if previous.category != coding.category:
                pairs["kategorie"] = (previous.category, coding.category)
            if list(previous.subcategories) != list(coding.subcategories):
                pairs["subkategorien"] = (
                    SUB_SEPARATOR.join(previous.subcategories),
                    SUB_SEPARATOR.join(coding.subcategories),
                )
            if previous.status != coding.status:
                pairs["status"] = (previous.status, coding.status)
            if previous.created_by != coding.created_by:
                pairs["erstellt_von"] = (previous.created_by, coding.created_by)
        elif coding.status != "proposed":
            pairs["status"] = ("", coding.status)

        for field, (old, new) in pairs.items():
            changes.append(
                {
                    "segment_id": coding.segment_id,
                    "kategorie": coding.category,
                    "feld": field,
                    "vorher": old,
                    "nachher": new,
                }
            )
    return changes


def export_review_xlsx(
    codings: Sequence[Coding],
    path: Path,
    original: Optional[Sequence[Coding]] = None,
    document_label: str = "",
    source_label: str = "",
    exported_at: str = "",
) -> Path:
    """Schreibt den Rezensionsstand als XLSX.

    Blätter: ``Kodierungen`` (aktueller Stand), ``Aenderungen`` (eine Zeile je
    geändertem Feld) und ``Meta`` (Herkunft und Zählung).
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openpyxl nicht installiert") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Kodierungen"
    sheet.append(REVIEW_COLUMNS)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in build_review_rows(codings):
        sheet.append([row[key] for key in REVIEW_COLUMNS])
    _autosize(sheet, REVIEW_COLUMNS)

    changes = review_changes(codings, original)
    change_sheet = workbook.create_sheet("Aenderungen")
    change_columns = ["segment_id", "kategorie", "feld", "vorher", "nachher"]
    change_sheet.append(change_columns)
    for cell in change_sheet[1]:
        cell.font = Font(bold=True)
    for row in changes:
        change_sheet.append([row[key] for key in change_columns])
    _autosize(change_sheet, change_columns)

    meta = workbook.create_sheet("Meta")
    meta.append(["feld", "wert"])
    for cell in meta[1]:
        cell.font = Font(bold=True)
    meta_rows = [
        ("dokument", document_label or ""),
        ("quelle", source_label or ""),
        ("exportiert_am", exported_at),
        ("kodierungen", len(codings)),
        ("bestaetigt", sum(1 for c in codings if c.status == "accepted")),
        ("verworfen", sum(1 for c in codings if c.status == "rejected")),
        ("offen", sum(1 for c in codings if c.status == "proposed")),
        ("ohne_anker", sum(1 for c in codings if not (c.anchor and c.anchor.anchored))),
        ("aenderungen", len(changes)),
    ]
    for key, value in meta_rows:
        meta.append([key, value])
    _autosize(meta, ["feld", "wert"])

    workbook.save(str(path))
    return path


def _autosize(sheet, columns: Sequence[str]) -> None:
    for index, name in enumerate(columns, start=1):
        letter = sheet.cell(row=1, column=index).column_letter
        sheet.column_dimensions[letter].width = max(12, len(name) + 4)


__all__ = [
    "build_rows",
    "export_csv",
    "export_xlsx",
    "export",
    "COLUMNS",
    "REVIEW_COLUMNS",
    "build_review_rows",
    "review_changes",
    "export_review_xlsx",
]