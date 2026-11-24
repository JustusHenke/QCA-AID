"""
Tests fuer den AnchorResolver (Task 6).

Prueft die exakte Aufloesung und die Quad-Geometrie
(Anforderungen 4.1-4.4, 4.7, 4.8):
  - Treffer mit zeilenweisen Quads
  - mehrzeiliger Abschnitt -> mehrere Quads
  - Mehrspaltigkeit -> getrennte Quads
  - glyph-genaue Hoehe
  - nicht gefunden -> unanchored
  - search_for-Fallback
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fitz = pytest.importorskip("fitz")

from QCA_AID_assets.annotation.anchor import AnchorResolver  # noqa: E402


def _pdf(tmp_path, lines, name="doc.pdf"):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 100
    for text in lines:
        page.insert_text((72, y), text, fontsize=11)
        y += 16
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path


# --------------------------------------------------------------------------
# Einzeilig / mehrzeilig
# --------------------------------------------------------------------------

def test_single_line_anchor(tmp_path):
    pdf = _pdf(tmp_path, ["Die Befragten unter Druck geraten und"])
    resolver = AnchorResolver()

    anchor = resolver.resolve(
        pdf, doc_id="d", page=1, quote="unter Druck geraten",
        prefix="Die Befragten ", suffix=" und",
    )

    assert anchor.anchored
    assert len(anchor.quads) == 1
    assert anchor.quote == "unter Druck geraten"
    assert anchor.prefix == "Die Befragten "


def test_multiline_anchor_yields_multiple_quads(tmp_path):
    pdf = _pdf(tmp_path, ["Die Befragten unter Druck geraten und",
                          "mussten priorisieren."])
    resolver = AnchorResolver()

    anchor = resolver.resolve(
        pdf, doc_id="d", page=1, quote="geraten und mussten priorisieren"
    )

    assert anchor.anchored
    assert len(anchor.quads) == 2
    ys = sorted(q[1] for q in anchor.quads)
    assert ys[0] != ys[1]


def test_quad_is_glyph_sized(tmp_path):
    pdf = _pdf(tmp_path, ["Die Befragten unter Druck geraten"])
    anchor = AnchorResolver().resolve(pdf, doc_id="d", page=1, quote="unter Druck")

    assert anchor.anchored
    x0, y0, x1, y1 = anchor.quads[0]
    assert 4 < (y1 - y0) < 25  # eine Textzeile, nicht die halbe Seite
    assert x1 > x0


# --------------------------------------------------------------------------
# Mehrspaltigkeit
# --------------------------------------------------------------------------

def test_columns_get_separate_quads(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Absatz eins", fontsize=11)
    page.insert_text((400, 100), "Absatz zwei", fontsize=11)
    pdf = tmp_path / "cols.pdf"
    doc.save(pdf)
    doc.close()

    anchor = AnchorResolver().resolve(
        pdf, doc_id="d", page=1, quote="eins Absatz zwei"
    )

    assert anchor.anchored
    assert len(anchor.quads) >= 2
    left = min(q[0] for q in anchor.quads)
    right = max(q[0] for q in anchor.quads)
    assert right - left > 100  # ueber die Spaltenluecke


# --------------------------------------------------------------------------
# Fallback und Negativfaelle
# --------------------------------------------------------------------------

def test_search_for_fallback_used_when_word_run_empty(tmp_path, monkeypatch):
    pdf = _pdf(tmp_path, ["Die Befragten unter Druck geraten"])
    resolver = AnchorResolver()
    monkeypatch.setattr(resolver, "_quads_from_word_run", lambda page, quote, prefix="", suffix="": [])

    anchor = resolver.resolve(pdf, doc_id="d", page=1, quote="unter Druck")

    assert anchor.anchored
    assert anchor.quads


def test_not_found_is_unanchored(tmp_path):
    pdf = _pdf(tmp_path, ["Die Befragten unter Druck geraten"])
    anchor = AnchorResolver().resolve(
        pdf, doc_id="d", page=1, quote="voellig anderer satz"
    )

    assert not anchor.anchored
    assert anchor.quads == []


def test_empty_quote_is_unanchored(tmp_path):
    pdf = _pdf(tmp_path, ["etwas text"])
    anchor = AnchorResolver().resolve(pdf, doc_id="d", page=1, quote="   ")
    assert not anchor.anchored
    assert anchor.quads == []


def test_page_out_of_range_is_unanchored(tmp_path):
    pdf = _pdf(tmp_path, ["etwas text"])
    anchor = AnchorResolver().resolve(pdf, doc_id="d", page=99, quote="etwas text")
    assert not anchor.anchored


# --------------------------------------------------------------------------
# Metadaten
# --------------------------------------------------------------------------

def test_page_label_and_doc_id_preserved(tmp_path):
    pdf = _pdf(tmp_path, ["etwas text"])
    anchor = AnchorResolver().resolve(
        pdf, doc_id="d1", page=1, quote="etwas text", page_label="iv"
    )
    assert anchor.doc_id == "d1"
    assert anchor.page == 1
    assert anchor.page_label == "iv"
    ok, errors = anchor.validate()
    assert ok, errors


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))