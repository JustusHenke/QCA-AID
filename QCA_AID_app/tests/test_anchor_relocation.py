"""
Tests fuer Anchor-Relocation und Unanchored-Handling (Task 7).

Prueft Anforderung 4.5 (Relocation via quote/prefix/suffix) und
4.6 (keine Fehlplatzierung, ehrlich unanchored).
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fitz = pytest.importorskip("fitz")

from QCA_AID_assets.annotation.anchor import AnchorResolver  # noqa: E402


def _multi_page_pdf(tmp_path, pages, name="doc.pdf"):
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page(width=595, height=842)
        y = 100
        for text in lines:
            page.insert_text((72, y), text, fontsize=11)
            y += 16
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path


LOREM = ["Die Befragten unter Druck geraten und mussten priorisieren sofort."]


# --------------------------------------------------------------------------
# Fuzzy auf derselben Seite
# --------------------------------------------------------------------------

def test_fuzzy_relocation_with_one_token_variation(tmp_path):
    pdf = _multi_page_pdf(tmp_path, [LOREM])
    resolver = AnchorResolver()

    # 'musste' statt 'mussten' -> 6/7 Tokens = 0.857 >= 0.85
    anchor = resolver.resolve(
        pdf,
        doc_id="d",
        page=1,
        quote="Die Befragten unter Druck geraten und musste priorisieren",
        search_neighbors=False,
    )

    assert anchor.anchored
    assert anchor.page == 1
    assert anchor.quads


def test_exact_match_preferred_over_fuzzy(tmp_path):
    pdf = _multi_page_pdf(tmp_path, [LOREM])
    anchor = AnchorResolver().resolve(
        pdf, doc_id="d", page=1, quote="unter Druck geraten"
    )
    assert anchor.anchored


def test_higher_threshold_rejects_relocation(tmp_path):
    pdf = _multi_page_pdf(tmp_path, [LOREM])
    anchor = AnchorResolver().resolve(
        pdf,
        doc_id="d",
        page=1,
        quote="Die Befragten unter Druck geraten musste priorisieren",
        fuzzy_threshold=0.99,
        search_neighbors=False,
    )
    assert not anchor.anchored


# --------------------------------------------------------------------------
# Relocation auf Nachbarseiten
# --------------------------------------------------------------------------

def test_neighbor_page_relocation(tmp_path):
    pages = [
        ["Voellig anderer Inhalt ohne Bezug."],
        ["Die Befragten unter Druck geraten und musste priorisieren"],
    ]
    pdf = _multi_page_pdf(tmp_path, pages)

    anchor = AnchorResolver().resolve(
        pdf,
        doc_id="d",
        page=1,
        quote="Die Befragten unter Druck geraten und musste priorisieren",
        page_label="1",
        search_neighbors=True,
    )

    assert anchor.anchored
    assert anchor.page == 2
    assert anchor.page_label is None  # Label gilt fuer die alte Seite


def test_neighbor_search_can_be_disabled(tmp_path):
    pages = [
        ["Voellig anderer Inhalt ohne Bezug."],
        ["Die Befragten unter Druck geraten und musste priorisieren"],
    ]
    pdf = _multi_page_pdf(tmp_path, pages)

    anchor = AnchorResolver().resolve(
        pdf,
        doc_id="d",
        page=1,
        quote="Die Befragten unter Druck geraten und musste priorisieren",
        search_neighbors=False,
    )

    assert not anchor.anchored


# --------------------------------------------------------------------------
# Unanchored (4.6)
# --------------------------------------------------------------------------

def test_completely_different_quote_is_unanchored(tmp_path):
    pdf = _multi_page_pdf(tmp_path, [LOREM])
    anchor = AnchorResolver().resolve(
        pdf, doc_id="d", page=1, quote="etwas ganz anderes hier", search_neighbors=False
    )

    assert not anchor.anchored
    assert anchor.quads == []
    ok, errors = anchor.validate()  # kein falsch platzierter Anker
    assert ok, errors


def test_relocation_never_produces_partial_wrong_quads(tmp_path):
    pdf = _multi_page_pdf(tmp_path, [LOREM])
    anchor = AnchorResolver().resolve(
        pdf, doc_id="d", page=1, quote="xyz qqq zzz", search_neighbors=False
    )
    assert anchor.quads == []
    assert not anchor.anchored


# --------------------------------------------------------------------------
# Kontext-Tie-Break
# --------------------------------------------------------------------------

def test_prefix_and_suffix_help_disambiguation(tmp_path):
    lines = [
        "Erstes Vorkommen unter Druck geraten hier.",
        "Zweites Vorkommen unter Druck geraten dort.",
    ]
    pdf = _multi_page_pdf(tmp_path, [lines])

    anchor = AnchorResolver().resolve(
        pdf,
        doc_id="d",
        page=1,
        quote="unter Druck geraten",
        prefix="Zweites Vorkommen ",
        suffix=" dort.",
        search_neighbors=False,
    )

    assert anchor.anchored
    # Der Kontext sollte den zweiten Treffer bevorzugen (weiter unten auf der Seite).
    assert anchor.quads[0][1] > 100


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))