"""
Review-Entscheidungen in die Analyse-Arbeitsmappe schreiben
===========================================================

Die Analyse-XLSX ist die zentrale Datei der Weiterarbeit. Der Prüfen-Tab
schreibt seine Entscheidungen deshalb optional direkt dorthin - aber streng
additiv:

* neue Spalten **rechts neben** den Bestandsspalten von
  ``Kodierungsergebnisse``: ``Review_Status``, ``Hauptkategorie_Review``,
  ``Subkategorien_Review``, ``Review_Änderung``, ``Review_Stand``,
* neues Blatt ``Review_Entscheidungen`` mit einer Zeile je Kodierung
  (vorher/nachher),
* Bestandsspalten, Häufigkeitsanalysen, Intercoder-Bericht, Kategorien usw.
  bleiben unverändert,
* vor jedem Schreiben entsteht eine Sicherung in ``<output>/backup/``.

Gelesen wird der Stand wieder in :func:`review_service.coding_from_record`:
liegt eine ``Hauptkategorie_Review`` vor, gilt sie als aktueller Code, der
Status kommt aus ``Review_Status``. Damit sind die Entscheidungen über
Sitzungen hinweg im zentralen Dokument sichtbar und reproduzierbar.

Die Logik ist headless (testbar); die Streamlit-Verdrahtung steht in
``webapp_components.review_ui``.
"""

from __future__ import annotations

import logging
import shutil
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from QCA_AID_assets.annotation.model import Coding
from QCA_AID_assets.utils.export.paths import export_stamp

logger = logging.getLogger(__name__)

#: Blatt mit den Kodierungen (Ziel der neuen Spalten).
MAIN_SHEET = "Kodierungsergebnisse"
#: Blatt mit den Entscheidungen (wird bei jedem Schreiben neu aufgebaut).
DECISION_SHEET = "Review_Entscheidungen"
#: Unterordner für Sicherungen vor dem Schreiben.
BACKUP_SUBDIR = "backup"

#: Spaltennamen der Analyse-XLSX (mit Fallbacks für ältere Exporte).
COL_KEY = ("Original_Chunk_ID", "Chunk_Nr", "Segment_ID")
COL_DOCUMENT = ("Dokument", "Datei", "Quelle")
COL_CATEGORY = ("Hauptkategorie", "Kategorie")
COL_SUBCATEGORIES = ("Subkategorien", "Subkategorie")

#: Neue Spalten (in dieser Reihenfolge angehängt).
STATUS_COLUMN = "Review_Status"
CATEGORY_COLUMN = "Hauptkategorie_Review"
SUBCATEGORY_COLUMN = "Subkategorien_Review"
CHANGE_COLUMN = "Review_Änderung"
STAMP_COLUMN = "Review_Stand"

REVIEW_COLUMNS: Tuple[str, ...] = (
    STATUS_COLUMN,
    CATEGORY_COLUMN,
    SUBCATEGORY_COLUMN,
    CHANGE_COLUMN,
    STAMP_COLUMN,
)

#: Statuswerte, wie sie in der Arbeitsmappe stehen.
STATUS_CONFIRMED = "bestätigt"
STATUS_REJECTED = "verworfen"
STATUS_OPEN = "offen"

#: Coding.status -> Spaltenwert.
STATUS_BY_CODING = {
    "accepted": STATUS_CONFIRMED,
    "rejected": STATUS_REJECTED,
    "proposed": STATUS_OPEN,
}

#: Spaltenwert -> Coding.status (Rückweg beim Laden).
CODING_BY_STATUS = {
    STATUS_CONFIRMED: "accepted",
    "bestaetigt": "accepted",
    "akzeptiert": "accepted",
    STATUS_REJECTED: "rejected",
    STATUS_OPEN: "proposed",
}

#: Zellfarben: grün bestätigt, rot verworfen, gelb offen, orange geändert.
STATUS_FILLS = {
    STATUS_CONFIRMED: "C8E6C9",
    STATUS_REJECTED: "FFCDD2",
    STATUS_OPEN: "FFF2CC",
}
CHANGE_FILL = "FFE0B2"

#: Trennzeichen der Subkategorien in der Analyse-XLSX (wie im Export).
SUBCATEGORY_JOIN = ", "

#: Spalten des Entscheidungsblatts.
DECISION_COLUMNS: Tuple[str, ...] = (
    "Segment_ID",
    "Dokument",
    "Hauptkategorie_vorher",
    "Hauptkategorie_nachher",
    "Subkategorien_vorher",
    "Subkategorien_nachher",
    "Status",
    "Änderung",
    "Stand",
)


class WritebackError(RuntimeError):
    """Die Arbeitsmappe konnte nicht ergänzt werden."""


@dataclass(frozen=True)
class WritebackResult:
    """Ergebnis eines Schreibvorgangs."""

    path: Path
    backup: Optional[Path]
    sheet: str
    rows: int
    matched: int
    confirmed: int
    rejected: int
    open: int
    changed: int

    @property
    def summary(self) -> str:
        return (
            f"{self.matched} Kodierung(en) übernommen — "
            f"{self.confirmed} bestätigt, {self.changed} geändert, "
            f"{self.rejected} verworfen, {self.open} offen"
        )


# ---------------------------------------------------------------------------
# Kleine Helfer
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _key(value: Any) -> str:
    return _text(value).casefold()


def split_subcategories(value: Any) -> List[str]:
    """Subkategorien aus einer Zelle lesen (Trenner ``;`` oder ``,``)."""
    text = _text(value)
    if not text:
        return []
    parts = text.replace(";", ",").split(",")
    return [part.strip() for part in parts if part.strip()]


def _same_list(left: Sequence[str], right: Sequence[str]) -> bool:
    return [_key(item) for item in left] == [_key(item) for item in right]


def _find_header(
    sheet: Any, candidates: Sequence[str]
) -> Tuple[int, int]:
    """Findet Zeile und Spalte eines Spaltennamens (erste 5 Zeilen)."""
    wanted = {name.casefold() for name in candidates}
    for row in range(1, min(sheet.max_row, 5) + 1):
        for column in range(1, sheet.max_column + 1):
            if _key(sheet.cell(row=row, column=column).value) in wanted:
                return row, column
    raise WritebackError(
        f"Spalte {candidates[0]} nicht gefunden - ist das eine Analyse-Arbeitsmappe?"
    )


def _header_map(sheet: Any, header_row: int) -> Dict[str, int]:
    """Spaltenname (casefold) -> Spaltennummer der Kopfzeile."""
    mapping: Dict[str, int] = {}
    for column in range(1, sheet.max_column + 1):
        name = _key(sheet.cell(row=header_row, column=column).value)
        if name and name not in mapping:
            mapping[name] = column
    return mapping


def _ensure_column(sheet: Any, header: Dict[str, int], header_row: int, name: str) -> int:
    """Spalte anlegen (rechts anhängen) und Spaltennummer liefern."""
    existing = header.get(_key(name))
    if existing:
        return existing
    column = sheet.max_column + 1
    reference = sheet.cell(row=header_row, column=max(1, sheet.max_column))
    cell = sheet.cell(row=header_row, column=column, value=name)
    cell._style = copy(reference._style)  # Kopfzeilen-Optik übernehmen
    column_letter = cell.column_letter
    sheet.column_dimensions[column_letter].width = max(
        14.0, sheet.column_dimensions[column_letter].width or 0.0
    )
    header[_key(name)] = column
    return column


def _main_sheet(workbook: Any, sheet_name: Optional[str] = None) -> Any:
    if sheet_name and sheet_name in workbook.sheetnames:
        return workbook[sheet_name]
    if MAIN_SHEET in workbook.sheetnames:
        return workbook[MAIN_SHEET]
    return workbook.worksheets[0]


def _fill(color: str) -> Any:
    from openpyxl.styles import PatternFill

    return PatternFill(fill_type="solid", start_color=color, end_color=color)


# ---------------------------------------------------------------------------
# Zählwerte (für die Vorschau in der UI)
# ---------------------------------------------------------------------------


def change_summary(
    codings: Sequence[Coding],
    original: Optional[Sequence[Coding]] = None,
) -> Dict[str, int]:
    """Zählt bestätigt/verworfen/offen und echte Code-Änderungen.

    ``original`` ist der Stand beim Laden des Prüfen-Tabs; ohne ihn gelten nur
    Statuswechsel als Änderung.
    """
    before = {coding.segment_id: coding for coding in (original or [])}
    counts = {"total": 0, "confirmed": 0, "rejected": 0, "open": 0, "changed": 0}
    for coding in codings:
        counts["total"] += 1
        status = STATUS_BY_CODING.get(coding.status, STATUS_OPEN)
        if status == STATUS_CONFIRMED:
            counts["confirmed"] += 1
        elif status == STATUS_REJECTED:
            counts["rejected"] += 1
        else:
            counts["open"] += 1
        previous = before.get(coding.segment_id)
        if previous is not None and (
            _key(previous.category) != _key(coding.category)
            or not _same_list(previous.subcategories, coding.subcategories)
        ):
            counts["changed"] += 1
    return counts


# ---------------------------------------------------------------------------
# Schreiben
# ---------------------------------------------------------------------------


def write_review_into_xlsx(
    xlsx_path: Path,
    codings: Sequence[Coding],
    stamp: Optional[str] = None,
    sheet_name: Optional[str] = None,
    backup: bool = True,
) -> WritebackResult:
    """Ergänzt die Analyse-Arbeitsmappe um die Review-Spalten und das Blatt.

    Args:
        xlsx_path: Analyse-XLSX (wird ergänzt, nicht neu erzeugt)
        codings: Stand aus dem Prüfen-Tab
        stamp: Zeitstempel (Default: jetzt)
        sheet_name: Hauptblatt (Default: ``Kodierungsergebnisse``)
        backup: vorher eine Kopie nach ``<output>/backup/`` legen

    Raises:
        WritebackError: fehlende Datei, falsches Format oder gesperrte Datei.
    """
    from openpyxl import load_workbook

    path = Path(xlsx_path)
    if not path.is_file():
        raise WritebackError(f"Datei nicht gefunden: {path}")
    if not codings:
        raise WritebackError("Keine Kodierungen geladen - nichts zu übernehmen.")

    stamp = stamp or export_stamp()

    try:
        workbook = load_workbook(path)
    except Exception as exc:  # noqa: BLE001 - openpyxl wirft heterogen
        raise WritebackError(f"Arbeitsmappe nicht lesbar: {exc}") from exc

    sheet = _main_sheet(workbook, sheet_name)
    header_row, key_column = _find_header(sheet, COL_KEY)
    header = _header_map(sheet, header_row)

    category_column = header.get(_key(COL_CATEGORY[0])) or header.get(_key(COL_CATEGORY[1]))
    subcategory_column = header.get(_key(COL_SUBCATEGORIES[0])) or header.get(
        _key(COL_SUBCATEGORIES[1])
    )
    document_column = None
    for name in COL_DOCUMENT:
        if header.get(_key(name)):
            document_column = header[_key(name)]
            break

    status_column = _ensure_column(sheet, header, header_row, STATUS_COLUMN)
    review_category_column = _ensure_column(sheet, header, header_row, CATEGORY_COLUMN)
    review_subcategory_column = _ensure_column(sheet, header, header_row, SUBCATEGORY_COLUMN)
    change_column = _ensure_column(sheet, header, header_row, CHANGE_COLUMN)
    stamp_column = _ensure_column(sheet, header, header_row, STAMP_COLUMN)

    by_key = {_key(coding.segment_id): coding for coding in codings}

    counts = {"matched": 0, "confirmed": 0, "rejected": 0, "open": 0, "changed": 0}
    decisions: List[Dict[str, str]] = []

    for row in range(header_row + 1, sheet.max_row + 1):
        key = _key(sheet.cell(row=row, column=key_column).value)
        coding = by_key.get(key)
        if coding is None:
            continue

        original_category = _text(
            sheet.cell(row=row, column=category_column).value
        ) if category_column else ""
        original_subcategories = (
            split_subcategories(sheet.cell(row=row, column=subcategory_column).value)
            if subcategory_column
            else []
        )

        status = STATUS_BY_CODING.get(coding.status, STATUS_OPEN)
        changed_fields: List[str] = []
        if _key(original_category) != _key(coding.category):
            changed_fields.append("Hauptkategorie")
        if not _same_list(original_subcategories, coding.subcategories):
            changed_fields.append("Subkategorien")

        sheet.cell(row=row, column=status_column, value=status).fill = _fill(
            STATUS_FILLS[status]
        )
        sheet.cell(row=row, column=review_category_column, value=coding.category)
        sheet.cell(
            row=row,
            column=review_subcategory_column,
            value=SUBCATEGORY_JOIN.join(coding.subcategories),
        )
        change_cell = sheet.cell(
            row=row, column=change_column, value=", ".join(changed_fields)
        )
        if changed_fields:
            change_cell.fill = _fill(CHANGE_FILL)
        sheet.cell(row=row, column=stamp_column, value=stamp)

        counts["matched"] += 1
        counter = {STATUS_CONFIRMED: "confirmed", STATUS_REJECTED: "rejected"}.get(status, "open")
        counts[counter] += 1
        if changed_fields:
            counts["changed"] += 1

        decisions.append(
            {
                "Segment_ID": coding.segment_id,
                "Dokument": _text(
                    sheet.cell(row=row, column=document_column).value
                ) if document_column else "",
                "Hauptkategorie_vorher": original_category,
                "Hauptkategorie_nachher": coding.category,
                "Subkategorien_vorher": SUBCATEGORY_JOIN.join(original_subcategories),
                "Subkategorien_nachher": SUBCATEGORY_JOIN.join(coding.subcategories),
                "Status": status,
                "Änderung": ", ".join(changed_fields),
                "Stand": stamp,
            }
        )

    if not counts["matched"]:
        raise WritebackError(
            "Keine Zeile der Arbeitsmappe passte zu den geladenen Kodierungen "
            f"(Schlüsselspalte: {COL_KEY[0]})."
        )

    _rebuild_decision_sheet(workbook, sheet, decisions, stamp)

    backup_path: Optional[Path] = None
    if backup:
        backup_dir = path.parent / BACKUP_SUBDIR
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{path.stem}_vor-review_{stamp}{path.suffix}"
        shutil.copy2(path, backup_path)

    try:
        workbook.save(path)
    except PermissionError as exc:  # Datei in Excel geöffnet
        raise WritebackError(
            f"{path.name} ist gesperrt (in Excel geöffnet?). Bitte schließen und erneut versuchen."
        ) from exc

    logger.info("Review-Übernahme: %s (%s)", path, counts)
    return WritebackResult(
        path=path,
        backup=backup_path,
        sheet=sheet.title,
        rows=sheet.max_row - header_row,
        matched=counts["matched"],
        confirmed=counts["confirmed"],
        rejected=counts["rejected"],
        open=counts["open"],
        changed=counts["changed"],
    )


def _rebuild_decision_sheet(
    workbook: Any,
    main_sheet: Any,
    decisions: Sequence[Dict[str, str]],
    stamp: str,
) -> None:
    """Legt das Blatt ``Review_Entscheidungen`` neu an (idempotent)."""
    from openpyxl.styles import Alignment, Font

    if DECISION_SHEET in workbook.sheetnames:
        del workbook[DECISION_SHEET]

    index = workbook.sheetnames.index(main_sheet.title) + 1
    sheet = workbook.create_sheet(DECISION_SHEET, index=index)

    sheet.append(list(DECISION_COLUMNS))
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="center", wrap_text=True)

    for decision in decisions:
        sheet.append([decision.get(column, "") for column in DECISION_COLUMNS])

    for row in range(2, sheet.max_row + 1):
        status = _text(sheet.cell(row=row, column=7).value)
        if status in STATUS_FILLS:
            sheet.cell(row=row, column=7).fill = _fill(STATUS_FILLS[status])
        if _text(sheet.cell(row=row, column=8).value):
            sheet.cell(row=row, column=8).fill = _fill(CHANGE_FILL)

    widths = {
        "A": 22,
        "B": 26,
        "C": 28,
        "D": 28,
        "E": 46,
        "F": 46,
        "G": 12,
        "H": 24,
        "I": 16,
    }
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = "A2"
    last_column = sheet.cell(row=1, column=len(DECISION_COLUMNS)).column_letter
    sheet.auto_filter.ref = f"A1:{last_column}{max(1, sheet.max_row)}"

    # Kopfzeile des Entscheidungsblatts trägt den Stand des letzten Schreibens.
    note = sheet.cell(row=sheet.max_row + 2, column=1, value=f"Stand: {stamp}")
    note.font = Font(italic=True, size=9)


__all__ = [
    "MAIN_SHEET",
    "DECISION_SHEET",
    "BACKUP_SUBDIR",
    "REVIEW_COLUMNS",
    "DECISION_COLUMNS",
    "STATUS_COLUMN",
    "CATEGORY_COLUMN",
    "SUBCATEGORY_COLUMN",
    "CHANGE_COLUMN",
    "STAMP_COLUMN",
    "STATUS_CONFIRMED",
    "STATUS_REJECTED",
    "STATUS_OPEN",
    "CODING_BY_STATUS",
    "WritebackError",
    "WritebackResult",
    "split_subcategories",
    "change_summary",
    "write_review_into_xlsx",
]
