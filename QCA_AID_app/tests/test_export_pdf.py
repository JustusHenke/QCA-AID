"""
Tests fuer den PdfBakingExporter (Task 12).

Prueft Anforderungen 8.1-8.4: Highlight mit /QuadPoints, Multiply-Blend,
Legenden-Seite, Marginalien. Anker werden real ueber den AnchorResolver
aus einem generierten PDF gewonnen.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fitz = pytest.importorskip("fitz")

from QCA_AID_assets.annotation.anchor import AnchorResolver  # noqa: E402
from QCA_AID_assets.annotation.export_pdf import PdfBakingExporter  # noqa: E402
from QCA_AID_assets.annotation.model import Coding  # noqa: E402


def make_pdf(tmp_path, lines=("Die Befragten unter Druck geraten.",)):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 100
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    path = tmp_path / "src.pdf"
    doc.save(path)
    doc.close()
    return path


def make_coding(pdf, quote, category="Problemdruck", justification=None, page=1):
    anchor = AnchorResolver().resolve(pdf, doc_id="d", page=page, quote=quote)
    return Coding(
        segment_id="d_chunk_0", doc_id="d", category=category,
        anchor=anchor, justification=justification, confidence=0.87,
    )


# --------------------------------------------------------------------------
# Highlights
# --------------------------------------------------------------------------

def test_bake_writes_highlight_with_quads(tmp_path):
    pdf = make_pdf(tmp_path)
    coding = make_coding(pdf, "unter Druck geraten")
    out = PdfBakingExporter().bake(pdf, [coding], tmp_path / "out.pdf", legend=False, margin_notes=False)

    doc = fitz.open(out)
    highlights = [a for a in doc[0].annots() if a.type[0] == fitz.PDF_ANNOT_HIGHLIGHT]
    assert len(highlights) >= 1
    # QuadPoints vorhanden
    assert doc.xref_get_key(highlights[0].xref, "QuadPoints")[0] != "null"


def test_bake_applies_multiply_blend(tmp_path):
    pdf = make_pdf(tmp_path)
    coding = make_coding(pdf, "unter Druck geraten")
    out = PdfBakingExporter().bake(pdf, [coding], tmp_path / "out.pdf", legend=False, margin_notes=False)

    doc = fitz.open(out)
    annot = list(doc[0].annots())[0]
    kind, value = doc.xref_get_key(annot.xref, "BM")
    assert kind == "name"
    assert value == "/Multiply"


def test_bake_sets_opacity(tmp_path):
    pdf = make_pdf(tmp_path)
    coding = make_coding(pdf, "unter Druck geraten")
    exporter = PdfBakingExporter(opacity=0.3)
    out = exporter.bake(pdf, [coding], tmp_path / "out.pdf", legend=False, margin_notes=False)

    doc = fitz.open(out)
    annot = list(doc[0].annots())[0]
    assert abs(annot.opacity - 0.3) < 0.05


# --------------------------------------------------------------------------
# Legende
# --------------------------------------------------------------------------

def test_legend_page_is_appended(tmp_path):
    pdf = make_pdf(tmp_path)
    coding = make_coding(pdf, "unter Druck geraten")
    original_pages = fitz.open(pdf).page_count
    out = PdfBakingExporter().bake(pdf, [coding], tmp_path / "out.pdf", legend=True, margin_notes=False)

    doc = fitz.open(out)
    assert doc.page_count == original_pages + 1
    # Legende am Ende: Seitenzahlen des Dokuments bleiben unveraendert.
    assert "Legende" in doc[original_pages].get_text()
    assert "Legende" not in doc[0].get_text()


# --------------------------------------------------------------------------
# Marginalien
# --------------------------------------------------------------------------

def test_margin_notes_added(tmp_path):
    pdf = make_pdf(tmp_path)
    coding = make_coding(pdf, "unter Druck geraten", justification="weil Druck")
    out = PdfBakingExporter().bake(pdf, [coding], tmp_path / "out.pdf", legend=False, margin_notes=True)

    doc = fitz.open(out)
    text_annots = [a for a in doc[0].annots() if a.type[0] == fitz.PDF_ANNOT_TEXT]
    assert text_annots
    assert "Problemdruck" in text_annots[0].info.get("content", "")


def test_margin_notes_can_be_disabled(tmp_path):
    pdf = make_pdf(tmp_path)
    coding = make_coding(pdf, "unter Druck geraten")
    out = PdfBakingExporter().bake(pdf, [coding], tmp_path / "out.pdf", legend=False, margin_notes=False)

    doc = fitz.open(out)
    assert [a for a in doc[0].annots() if a.type[0] == fitz.PDF_ANNOT_TEXT] == []


def test_highlight_info_contains_category_and_justification(tmp_path):
    pdf = make_pdf(tmp_path)
    coding = make_coding(pdf, "unter Druck geraten", justification="weil Druck")
    out = PdfBakingExporter().bake(pdf, [coding], tmp_path / "out.pdf", legend=False, margin_notes=False)

    doc = fitz.open(out)
    annot = list(doc[0].annots())[0]
    content = annot.info.get("content", "")
    assert "Problemdruck" in content
    assert "weil Druck" in content


# --------------------------------------------------------------------------
# Negativfaelle
# --------------------------------------------------------------------------

def test_unanchored_coding_is_skipped(tmp_path):
    pdf = make_pdf(tmp_path)
    anchor = AnchorResolver().resolve(pdf, doc_id="d", page=1, quote="nicht vorhanden")
    coding = Coding(segment_id="s1", doc_id="d", category="X", anchor=anchor)
    out = PdfBakingExporter().bake(pdf, [coding], tmp_path / "out.pdf", legend=False, margin_notes=False)

    doc = fitz.open(out)
    assert list(doc[0].annots()) == []


def test_multiple_categories_get_distinct_colors(tmp_path):
    pdf = make_pdf(tmp_path, lines=("Erster Satz hier.", "Zweiter Satz dort."))
    c1 = make_coding(pdf, "Erster Satz", category="A")
    c2 = make_coding(pdf, "Zweiter Satz", category="B")
    out = PdfBakingExporter().bake(pdf, [c1, c2], tmp_path / "out.pdf", legend=False, margin_notes=False)

    doc = fitz.open(out)
    colors = {tuple(round(v, 2) for v in a.colors.get("stroke", ())) for a in doc[0].annots()}
    assert len(colors) >= 2


# --------------------------------------------------------------------------
# Mehrfachkodierung: ein Highlight je Textstelle, Schraffur fuer die weiteren
# --------------------------------------------------------------------------

def _shared_anchor_codings(pdf, quote="Erster Satz"):
    anchor = AnchorResolver().resolve(pdf, doc_id="d", page=1, quote=quote)
    first = Coding(segment_id="s1", doc_id="d", category="A", anchor=anchor)
    second = Coding(segment_id="s2", doc_id="d", category="B", anchor=anchor)
    return first, second


def test_shared_text_area_gets_one_highlight_and_a_hatch(tmp_path):
    pdf = make_pdf(tmp_path, lines=("Erster Satz hier.", "Zweiter Satz dort."))
    first, second = _shared_anchor_codings(pdf)
    out = PdfBakingExporter().bake(
        pdf, [first, second], tmp_path / "out.pdf", legend=False, margin_notes=False
    )

    doc = fitz.open(out)
    highlights = [a for a in doc[0].annots() if a.type[0] == fitz.PDF_ANNOT_HIGHLIGHT]
    assert len(highlights) == 1  # nicht zwei uebereinanderliegende Markierungen
    # Die zweite Kodierung wird als Schraffur gezeichnet (duenne Linien).
    thin = [
        drawing for drawing in doc[0].get_drawings()
        if drawing.get("color") and 0 < (drawing.get("width") or 0) < 2
    ]
    assert thin


def test_distinct_text_areas_still_get_their_own_highlights(tmp_path):
    pdf = make_pdf(tmp_path, lines=("Erster Satz hier.", "Zweiter Satz dort."))
    c1 = make_coding(pdf, "Erster Satz", category="A")
    c2 = make_coding(pdf, "Zweiter Satz", category="B")
    out = PdfBakingExporter().bake(
        pdf, [c1, c2], tmp_path / "out.pdf", legend=False, margin_notes=False
    )

    doc = fitz.open(out)
    highlights = [a for a in doc[0].annots() if a.type[0] == fitz.PDF_ANNOT_HIGHLIGHT]
    assert len(highlights) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))