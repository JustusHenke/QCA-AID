"""
Tests fuer das Zurueckschreiben der Review-Entscheidungen in die Analyse-XLSX.

Regel: die Arbeitsmappe wird *ergaenzt* (neue Spalten + Blatt
``Review_Entscheidungen``, Sicherung in ``backup/``), nie ersetzt. Der
Pruefen-Tab liest den Stand beim naechsten Laden wieder ein.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from QCA_AID_assets.annotation.model import Coding  # noqa: E402
from webapp_logic.review_service import codings_from_records, load_codings_from_xlsx  # noqa: E402
from webapp_logic.review_writeback import (  # noqa: E402
    CATEGORY_COLUMN,
    CHANGE_COLUMN,
    DECISION_COLUMNS,
    DECISION_SHEET,
    STATUS_COLUMN,
    STAMP_COLUMN,
    SUBCATEGORY_COLUMN,
    WritebackError,
    change_summary,
    split_subcategories,
    write_review_into_xlsx,
)

STAMP = "20261124_123000"
DOC = "text_demo_wiss.pdf"


def coding(
    segment_id: str,
    category: str,
    subcategories=("A",),
    status: str = "proposed",
) -> Coding:
    return Coding(
        segment_id=segment_id,
        doc_id="text_demo_wiss",
        category=category,
        subcategories=list(subcategories),
        status=status,
    )


def make_workbook(path: Path) -> Path:
    """Kleine Arbeitsmappe im Format des Analyse-Exports."""
    frame = pd.DataFrame(
        [
            {
                "Dokument": DOC,
                "Chunk_Nr": "TEXT_DEMO_WISS-0-1",
                "Text": "Die Befragten unter Druck geraten.",
                "Kodiert": True,
                "Hauptkategorie": "Legitimationsstrategien",
                "Subkategorien": "Diskurs_Bildung, Gruendungsmotive",
                "Begründung": "explizit",
                "Textanker": "Die Befragten unter Druck geraten.",
                "Konfidenz": "0.90",
                "Original_Chunk_ID": "TEXT_DEMO_WISS-0-1",
            },
            {
                "Dokument": DOC,
                "Chunk_Nr": "TEXT_DEMO_WISS-0-2",
                "Text": "Staatliche Verfahren binden Ressourcen.",
                "Kodiert": True,
                "Hauptkategorie": "Strukturelle_Rahmenbedingungen",
                "Subkategorien": "Verfahren",
                "Begründung": "explizit",
                "Textanker": "Staatliche Verfahren binden Ressourcen.",
                "Konfidenz": "0.70",
                "Original_Chunk_ID": "TEXT_DEMO_WISS-0-2",
            },
            {
                "Dokument": "text_interview.docx",
                "Chunk_Nr": "TEXT_INTERV_-0-1",
                "Text": "Lehre und Forschung gleichzeitig.",
                "Kodiert": True,
                "Hauptkategorie": "Lehrpraxis_und_kultur",
                "Subkategorien": "Lehrqualitaet",
                "Begründung": "explizit",
                "Textanker": "Lehre und Forschung gleichzeitig.",
                "Konfidenz": "0.80",
                "Original_Chunk_ID": "TEXT_INTERV_-0-1",
            },
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Kodierungsergebnisse", index=False)
        pd.DataFrame({"Hinweis": ["bleibt unverändert"]}).to_excel(
            writer, sheet_name="Haeufigkeiten", index=False
        )
    return path


def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)


# --------------------------------------------------------------------------
# Rueckweg (Laden)
# --------------------------------------------------------------------------

def test_split_subcategories_accepts_comma_and_semicolon():
    assert split_subcategories("A, B; C") == ["A", "B", "C"]
    assert split_subcategories(None) == []


def test_workbook_without_review_columns_stays_open():
    record = {
        "Hauptkategorie": "Kategorie_A",
        "Subkategorien": "A, B",
        "Original_Chunk_ID": "X-1",
        "Text": "Ein hinreichend langer Text fuer die Zitatauswahl.",
    }
    (coding_item,) = codings_from_records([record], doc_id="x", doc_name=None)
    assert coding_item.status == "proposed"
    assert coding_item.category == "Kategorie_A"
    assert coding_item.subcategories == ["A", "B"]
    assert coding_item.created_by == "ai"


def test_review_columns_are_read_back_as_state():
    record = {
        "Hauptkategorie": "Kategorie_A",
        "Subkategorien": "A, B",
        "Original_Chunk_ID": "X-1",
        "Text": "Ein hinreichend langer Text fuer die Zitatauswahl.",
        STATUS_COLUMN: "bestätigt",
        CATEGORY_COLUMN: "Kategorie_A",
        SUBCATEGORY_COLUMN: "A",
    }
    (coding_item,) = codings_from_records([record], doc_id="x", doc_name=None)
    assert coding_item.status == "accepted"
    assert coding_item.subcategories == ["A"]
    assert coding_item.created_by == "ai"


def test_review_category_change_marks_the_coding_as_human():
    record = {
        "Hauptkategorie": "Kategorie_A",
        "Subkategorien": "A",
        "Original_Chunk_ID": "X-1",
        "Text": "Ein hinreichend langer Text fuer die Zitatauswahl.",
        STATUS_COLUMN: "bestätigt",
        CATEGORY_COLUMN: "Kategorie_B",
        SUBCATEGORY_COLUMN: "C",
    }
    (coding_item,) = codings_from_records([record], doc_id="x", doc_name=None)
    assert coding_item.category == "Kategorie_B"
    assert coding_item.subcategories == ["C"]
    assert coding_item.created_by == "human"


# --------------------------------------------------------------------------
# Schreiben
# --------------------------------------------------------------------------

def test_writeback_adds_columns_and_keeps_the_rest(tmp_path):
    path = make_workbook(tmp_path / "QCA-AID_Analysis_deductive_20261124_120000.xlsx")
    codings = [
        coding(
            "TEXT_DEMO_WISS-0-1",
            "Legitimationsstrategien",
            ["Diskurs_Bildung", "Gruendungsmotive"],
            "accepted",
        ),
        coding("TEXT_DEMO_WISS-0-2", "Steuerung_und_Governance", ["Verfahren"], "accepted"),
    ]

    result = write_review_into_xlsx(path, codings, stamp=STAMP)

    assert result.matched == 2
    assert result.confirmed == 2
    assert result.changed == 1  # nur die zweite Zeile: Kategorie geändert
    sheet = read_sheet(path, "Kodierungsergebnisse")
    assert sheet.loc[0, STATUS_COLUMN] == "bestätigt"
    assert sheet.loc[0, CATEGORY_COLUMN] == "Legitimationsstrategien"
    assert sheet.loc[1, CATEGORY_COLUMN] == "Steuerung_und_Governance"
    assert sheet.loc[1, SUBCATEGORY_COLUMN] == "Verfahren"
    assert sheet.loc[1, CHANGE_COLUMN] == "Hauptkategorie"
    assert sheet.loc[0, STAMP_COLUMN] == STAMP
    # Bestandsspalten unangetastet
    assert sheet.loc[1, "Hauptkategorie"] == "Strukturelle_Rahmenbedingungen"
    assert sheet.loc[1, "Begründung"] == "explizit"
    # Fremdes Blatt unangetastet
    assert read_sheet(path, "Haeufigkeiten").loc[0, "Hinweis"] == "bleibt unverändert"


def test_writeback_leaves_rows_of_other_documents_untouched(tmp_path):
    path = make_workbook(tmp_path / "analyse.xlsx")
    write_review_into_xlsx(path, [coding("TEXT_DEMO_WISS-0-1", "X", ["A"], "accepted")], stamp=STAMP)

    sheet = read_sheet(path, "Kodierungsergebnisse")
    other = sheet[sheet["Original_Chunk_ID"] == "TEXT_INTERV_-0-1"].iloc[0]
    assert pd.isna(other[STATUS_COLUMN])
    assert pd.isna(other[CATEGORY_COLUMN])


def test_writeback_creates_a_backup_outside_the_source_pattern(tmp_path):
    path = make_workbook(tmp_path / "QCA-AID_Analysis_deductive_20261124_120000.xlsx")
    result = write_review_into_xlsx(path, [coding("TEXT_DEMO_WISS-0-1", "X", ["A"], "accepted")], stamp=STAMP)

    assert result.backup is not None and result.backup.is_file()
    assert result.backup.parent.name == "backup"
    # Sicherung ist der Stand VOR dem Schreiben
    backup_sheet = read_sheet(result.backup, "Kodierungsergebnisse")
    assert STATUS_COLUMN not in backup_sheet.columns


def test_writeback_can_skip_the_backup(tmp_path):
    path = make_workbook(tmp_path / "analyse.xlsx")
    result = write_review_into_xlsx(path, [coding("TEXT_DEMO_WISS-0-1", "X", ["A"])], stamp=STAMP)
    assert result.backup is not None
    result2 = write_review_into_xlsx(
        path, [coding("TEXT_DEMO_WISS-0-1", "X", ["A"])], stamp=STAMP, backup=False
    )
    assert result2.backup is None


def test_decision_sheet_lists_before_and_after(tmp_path):
    path = make_workbook(tmp_path / "analyse.xlsx")
    write_review_into_xlsx(
        path,
        [
            coding("TEXT_DEMO_WISS-0-1", "Legitimationsstrategien", ["Neu"], "accepted"),
            coding("TEXT_DEMO_WISS-0-2", "Strukturelle_Rahmenbedingungen", ["Verfahren"], "rejected"),
        ],
        stamp=STAMP,
    )

    sheet = read_sheet(path, DECISION_SHEET)
    assert list(sheet.columns) == list(DECISION_COLUMNS)
    first = sheet[sheet["Segment_ID"] == "TEXT_DEMO_WISS-0-1"].iloc[0]
    assert first["Hauptkategorie_vorher"] == "Legitimationsstrategien"
    assert first["Hauptkategorie_nachher"] == "Legitimationsstrategien"
    assert first["Subkategorien_vorher"] == "Diskurs_Bildung, Gruendungsmotive"
    assert first["Subkategorien_nachher"] == "Neu"
    assert first["Status"] == "bestätigt"
    assert first["Änderung"] == "Subkategorien"

    second = sheet[sheet["Segment_ID"] == "TEXT_DEMO_WISS-0-2"].iloc[0]
    assert second["Status"] == "verworfen"
    assert pd.isna(second["Änderung"])


def test_writeback_is_repeatable(tmp_path):
    """Zweiter Lauf ersetzt die Entscheidungen, ohne Spalten zu duplizieren."""
    path = make_workbook(tmp_path / "analyse.xlsx")
    write_review_into_xlsx(path, [coding("TEXT_DEMO_WISS-0-1", "A", ["A"], "accepted")], stamp=STAMP)
    write_review_into_xlsx(path, [coding("TEXT_DEMO_WISS-0-1", "B", ["B"], "rejected")], stamp="20261124_130000")

    sheet = read_sheet(path, "Kodierungsergebnisse")
    assert list(sheet.columns).count(STATUS_COLUMN) == 1
    assert sheet.loc[0, STATUS_COLUMN] == "verworfen"
    assert sheet.loc[0, CATEGORY_COLUMN] == "B"
    assert sheet.loc[0, STAMP_COLUMN] == "20261124_130000"
    decisions = read_sheet(path, DECISION_SHEET)
    assert len(decisions[decisions["Segment_ID"] == "TEXT_DEMO_WISS-0-1"]) == 1


def test_round_trip_through_the_loader(tmp_path):
    """Nach dem Schreiben liest der Prüfen-Tab den Stand wieder ein."""
    path = make_workbook(tmp_path / "analyse.xlsx")
    write_review_into_xlsx(
        path,
        [
            coding("TEXT_DEMO_WISS-0-1", "Legitimationsstrategien", ["Diskurs_Bildung"], "accepted"),
            coding("TEXT_DEMO_WISS-0-2", "Steuerung_und_Governance", ["Verfahren"], "rejected"),
        ],
        stamp=STAMP,
    )

    codings = load_codings_from_xlsx(path, doc_id="text_demo_wiss", doc_name=DOC)
    by_id = {item.segment_id: item for item in codings}
    assert by_id["TEXT_DEMO_WISS-0-1"].status == "accepted"
    assert by_id["TEXT_DEMO_WISS-0-2"].status == "rejected"
    assert by_id["TEXT_DEMO_WISS-0-2"].category == "Steuerung_und_Governance"


def test_writeback_rejects_a_foreign_workbook(tmp_path):
    path = tmp_path / "fremd.xlsx"
    pd.DataFrame({"Spalte": [1]}).to_excel(path, index=False)

    with pytest.raises(WritebackError):
        write_review_into_xlsx(path, [coding("X-1", "A", ["A"])], stamp=STAMP)


def test_writeback_without_codings_is_an_error(tmp_path):
    path = make_workbook(tmp_path / "analyse.xlsx")
    with pytest.raises(WritebackError):
        write_review_into_xlsx(path, [], stamp=STAMP)


def test_writeback_without_matching_rows_is_an_error(tmp_path):
    path = make_workbook(tmp_path / "analyse.xlsx")
    with pytest.raises(WritebackError):
        write_review_into_xlsx(path, [coding("UNBEKANNT-9", "A", ["A"])], stamp=STAMP)


# --------------------------------------------------------------------------
# Vorschau-Zaehlwerte
# --------------------------------------------------------------------------

def test_change_summary_counts_status_and_edits():
    original = [
        coding("s1", "A", ["A"]),
        coding("s2", "B", ["B"]),
        coding("s3", "C", ["C"]),
    ]
    current = [
        coding("s1", "A", ["A"], "accepted"),
        coding("s2", "Z", ["B"], "accepted"),  # Kategorie geändert
        coding("s3", "C", ["C"], "rejected"),
    ]

    counts = change_summary(current, original)

    assert counts == {"total": 3, "confirmed": 2, "rejected": 1, "open": 0, "changed": 1}


def test_change_summary_without_original_counts_only_status():
    counts = change_summary([coding("s1", "A", ["A"], "accepted")])
    assert counts["confirmed"] == 1 and counts["changed"] == 0


# --------------------------------------------------------------------------
# Textanker bleibt erhalten (Spec text-anker-export, R7)
# --------------------------------------------------------------------------

def test_writeback_behaelt_textanker_spalte_unveraendert(tmp_path):
    """R7: Additives Schreiben beruehrt die Textanker-Wortlautspalte nicht."""
    path = make_workbook(tmp_path / "QCA-AID_Analysis_deductive_20261124_120000.xlsx")
    before = read_sheet(path, "Kodierungsergebnisse")["Textanker"].tolist()

    write_review_into_xlsx(
        path,
        [
            coding("TEXT_DEMO_WISS-0-1", "Legitimationsstrategien", ["Diskurs_Bildung"], "accepted"),
            coding("TEXT_DEMO_WISS-0-2", "Steuerung_und_Governance", ["Verfahren"], "rejected"),
        ],
        stamp=STAMP,
    )

    sheet = read_sheet(path, "Kodierungsergebnisse")
    assert "Textanker" in sheet.columns
    assert sheet["Textanker"].tolist() == before  # Wortlaut unveraendert
    assert sheet.loc[0, "Textanker"] == "Die Befragten unter Druck geraten."
    # Die neuen Review-Spalten liegen rechts daneben, ohne zu verschieben
    assert sheet.columns.get_loc("Textanker") < sheet.columns.get_loc(STATUS_COLUMN)


def test_round_trip_nach_writeback_liefert_textanker_erneut_ein(tmp_path):
    """Nach dem Schreiben liest der Pruefen-Tab dieselben Anker wieder ein."""
    path = make_workbook(tmp_path / "analyse.xlsx")
    write_review_into_xlsx(
        path,
        [coding("TEXT_DEMO_WISS-0-1", "Legitimationsstrategien", ["Diskurs_Bildung"], "accepted")],
        stamp=STAMP,
    )

    codings = load_codings_from_xlsx(path, doc_id="text_demo_wiss", doc_name=DOC)
    by_id = {item.segment_id: item for item in codings}
    assert by_id["TEXT_DEMO_WISS-0-1"].text_anchors == [
        "Die Befragten unter Druck geraten."
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
