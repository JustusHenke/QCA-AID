"""
Tests fuer das Rezensionspaket (Export aus dem Pruefen-Tab).

Prueft Ablage/Namensstempel, die drei Artefakte (PDF, XLSX mit Aenderungsblatt,
JSON-Protokoll) und dass die Analyse-Ausgabe unberuehrt bleibt.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fitz = pytest.importorskip("fitz")
openpyxl = pytest.importorskip("openpyxl")

from QCA_AID_assets.annotation.anchor import AnchorResolver  # noqa: E402
from QCA_AID_assets.annotation.model import Coding  # noqa: E402
from QCA_AID_app.webapp_logic.review_export import (  # noqa: E402
    REVIEW_SUBDIR,
    build_decisions,
    export_paths,
    export_review_package,
)

STAMP = "20261124_120000"


def make_pdf(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Die Befragten berichten von Zeitdruck.", fontsize=11)
    path = tmp_path / "src.pdf"
    doc.save(path)
    doc.close()
    return path


def make_coding(pdf, category="Zeitdruck", subcategories=(), status="proposed"):
    anchor = AnchorResolver().resolve(
        pdf, doc_id="d", page=1, quote="berichten von Zeitdruck"
    )
    return Coding(
        segment_id="seg-1",
        doc_id="d",
        category=category,
        subcategories=list(subcategories),
        status=status,
        created_by="ai",
        anchor=anchor,
    )


# --------------------------------------------------------------------------
# Pfade und Stempel
# --------------------------------------------------------------------------

def test_paths_use_review_subfolder_and_stamp(tmp_path):
    paths = export_paths(tmp_path, "text_demo_wiss", stamp=STAMP)

    assert paths["folder"] == tmp_path / REVIEW_SUBDIR
    assert paths["pdf"].name == f"text_demo_wiss_annotiert_{STAMP}.pdf"
    assert paths["xlsx"].name == f"text_demo_wiss_review_{STAMP}.xlsx"
    assert paths["json"].name == f"text_demo_wiss_review_{STAMP}.json"
    assert paths["pdf"].parent == paths["xlsx"].parent == paths["json"].parent


def test_stamp_defaults_to_now(tmp_path):
    paths = export_paths(tmp_path, "doc")

    stamp = paths["pdf"].stem.rsplit("_", 2)
    assert len(stamp[-1]) == 6 and len(stamp[-2]) == 8  # HHMMSS und JJJJMMTT


# --------------------------------------------------------------------------
# Paket
# --------------------------------------------------------------------------

def test_package_writes_pdf_xlsx_and_json(tmp_path):
    pdf = make_pdf(tmp_path)
    coding = make_coding(pdf)
    out_dir = tmp_path / "output"

    package = export_review_package(
        pdf_path=pdf,
        codings=[coding],
        output_dir=out_dir,
        doc_stem="text_demo_wiss",
        document_label="text_demo_wiss.pdf",
        source_label="QCA-AID_Analysis_abductive_x.xlsx",
        original=[coding],
        stamp=STAMP,
    )

    assert package.folder == out_dir / REVIEW_SUBDIR
    assert package.folder.is_dir()
    for path in package.files:
        assert path.is_file(), path
    assert package.codings == 1
    assert package.decisions == 0


def test_package_leaves_the_analysis_output_untouched(tmp_path):
    pdf = make_pdf(tmp_path)
    coding = make_coding(pdf)
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    analysis = out_dir / "QCA-AID_Analysis_abductive_20261123_225136.xlsx"
    analysis.write_bytes(b"original")

    export_review_package(
        pdf_path=pdf,
        codings=[coding],
        output_dir=out_dir,
        doc_stem="doc",
        stamp=STAMP,
    )

    assert analysis.read_bytes() == b"original"
    assert sorted(p.name for p in out_dir.iterdir() if p.is_file()) == [analysis.name]


def test_pdf_contains_highlight(tmp_path):
    pdf = make_pdf(tmp_path)
    coding = make_coding(pdf)
    package = export_review_package(
        pdf_path=pdf,
        codings=[coding],
        output_dir=tmp_path / "output",
        doc_stem="doc",
        stamp=STAMP,
    )

    doc = fitz.open(package.pdf)
    highlights = [a for a in doc[0].annots() if a.type[0] == fitz.PDF_ANNOT_HIGHLIGHT]
    assert len(highlights) == 1
    doc.close()


def test_xlsx_has_current_state_and_changes(tmp_path):
    pdf = make_pdf(tmp_path)
    before = make_coding(pdf, category="Zeitdruck")
    after = Coding(
        segment_id="seg-1",
        doc_id="d",
        category="Steuerung",
        subcategories=["Legitimität"],
        status="accepted",
        created_by="human",
        anchor=before.anchor,
    )
    package = export_review_package(
        pdf_path=pdf,
        codings=[after],
        output_dir=tmp_path / "output",
        doc_stem="doc",
        document_label="doc.pdf",
        original=[before],
        stamp=STAMP,
    )

    workbook = openpyxl.load_workbook(package.xlsx)
    assert workbook.sheetnames == ["Kodierungen", "Aenderungen", "Meta"]

    sheet = workbook["Kodierungen"]
    header = [cell.value for cell in sheet[1]]
    assert "kategorie" in header and "status" in header
    row = [cell.value for cell in sheet[2]]
    assert row[header.index("kategorie")] == "Steuerung"
    assert row[header.index("status")] == "accepted"

    changes = workbook["Aenderungen"]
    fields = {row[2] for row in changes.iter_rows(min_row=2, values_only=True)}
    assert {"kategorie", "subkategorien", "status", "erstellt_von"} <= fields

    meta = {row[0]: row[1] for row in workbook["Meta"].iter_rows(min_row=2, values_only=True)}
    assert meta["dokument"] == "doc.pdf"
    assert meta["kodierungen"] == 1


def test_json_protocol_lists_decisions(tmp_path):
    pdf = make_pdf(tmp_path)
    before = make_coding(pdf)
    after = Coding(
        segment_id="seg-1",
        doc_id="d",
        category="Zeitdruck",
        status="rejected",
        created_by="human",
        anchor=before.anchor,
    )
    package = export_review_package(
        pdf_path=pdf,
        codings=[after],
        output_dir=tmp_path / "output",
        doc_stem="doc",
        document_label="doc.pdf",
        original=[before],
        stamp=STAMP,
    )

    payload = json.loads(package.json.read_text(encoding="utf-8"))
    assert payload["dokument"] == "doc.pdf"
    assert payload["anzahl_geaendert"] == 1
    entry = payload["kodierungen"][0]
    assert entry["aenderungen"]["status"] == {"vorher": "proposed", "nachher": "rejected"}


# --------------------------------------------------------------------------
# Entscheidungen (headless)
# --------------------------------------------------------------------------

def test_decisions_without_changes_is_empty(tmp_path):
    pdf = make_pdf(tmp_path)
    coding = make_coding(pdf)

    assert [d for d in build_decisions([coding], [coding]) if d["aenderungen"]] == []


def test_decisions_detect_recode_and_status(tmp_path):
    pdf = make_pdf(tmp_path)
    before = make_coding(pdf)
    after = make_coding(pdf, category="Neu", subcategories=["X"], status="accepted")

    changes = build_decisions([after], [before])[0]["aenderungen"]

    assert changes["kategorie"] == {"vorher": "Zeitdruck", "nachher": "Neu"}
    assert changes["subkategorien"]["nachher"] == "X"
    assert changes["status"] == {"vorher": "proposed", "nachher": "accepted"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
