"""
Tests fuer den Datenexport (Task 13).

Prueft Anforderungen 8.5, 8.6: Spalten vollstaendig, Unanchored kenntlich,
CSV/XLSX korrekt geschrieben.
"""

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QCA_AID_assets.annotation.export_data import (  # noqa: E402
    COLUMNS,
    build_rows,
    export,
    export_csv,
    export_xlsx,
)
from QCA_AID_assets.annotation.model import Anchor, Coding  # noqa: E402


def _anchor(page=2, label=None, quote="unter Druck", anchored=True):
    return Anchor(
        doc_id="doc1", page=page, page_label=label, quote=quote,
        quads=[[0, 0, 10, 10]] if anchored else [], anchored=anchored,
    )


def _coding(segment_id="s1", category="Problemdruck", anchor=None):
    return Coding(
        segment_id=segment_id, doc_id="doc1", category=category,
        subcategories=["Zeit", "Last"], confidence=0.87,
        justification="weil", anchor=anchor or _anchor(),
    )


# --------------------------------------------------------------------------
# Zeilenaufbau
# --------------------------------------------------------------------------

def test_rows_have_all_columns():
    rows = build_rows([_coding()])
    assert set(rows[0].keys()) == set(COLUMNS)
    assert rows[0]["dokument"] == "doc1"
    assert rows[0]["seite"] == 2
    assert rows[0]["kategorie"] == "Problemdruck"
    assert rows[0]["subkategorien"] == "Zeit; Last"
    assert rows[0]["konfidenz"] == 0.87
    assert rows[0]["quote"] == "unter Druck"


def test_unanchored_is_marked():
    coding = Coding(
        segment_id="s1", doc_id="doc1", category="X",
        anchor=_anchor(anchored=False),
    )
    rows = build_rows([coding])
    assert rows[0]["verankert"] == "unanchored"


def test_anchored_is_marked():
    rows = build_rows([_coding()])
    assert rows[0]["verankert"] == "anchored"


def test_page_falls_back_to_anchor():
    coding = Coding(segment_id="s1", doc_id="doc1", category="X",
                    page=None, anchor=_anchor(page=7))
    assert build_rows([coding])[0]["seite"] == 7


def test_page_label_exported():
    coding = _coding(anchor=_anchor(page=3, label="iii"))
    assert build_rows([coding])[0]["seitenlabel"] == "iii"


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

def test_export_csv(tmp_path):
    path = export_csv([_coding(), _coding("s2")], tmp_path / "rows.csv")

    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        assert reader.fieldnames == COLUMNS
        rows = list(reader)
    assert len(rows) == 2


def test_csv_escapes_special_characters(tmp_path):
    coding = _coding(anchor=_anchor(quote='Zitat mit "Anfuehrung"; und Semikolon'))
    path = export_csv([coding], tmp_path / "rows.csv")

    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    assert rows[0]["quote"] == 'Zitat mit "Anfuehrung"; und Semikolon'


def test_csv_header_only_when_empty(tmp_path):
    path = export_csv([], tmp_path / "empty.csv")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    assert rows == [COLUMNS]


# --------------------------------------------------------------------------
# XLSX
# --------------------------------------------------------------------------

def test_export_xlsx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = export_xlsx([_coding(), _coding("s2")], tmp_path / "rows.xlsx")

    workbook = openpyxl.load_workbook(path)
    sheet = workbook.active
    header = [c.value for c in sheet[1]]
    assert header == COLUMNS
    assert sheet.max_row == 3


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def test_export_dispatch_by_suffix(tmp_path):
    csv_path = export([_coding()], tmp_path / "a.csv")
    xlsx_ok = export([_coding()], tmp_path / "a.xlsx")
    assert csv_path.exists() and xlsx_ok.exists()


def test_unsupported_suffix_raises(tmp_path):
    with pytest.raises(ValueError):
        export([_coding()], tmp_path / "a.json")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))