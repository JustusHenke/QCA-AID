"""
Tests fuer die Annotation-Datenmodelle (Task 1).

Prueft Validierung und Invarianten aus Anforderung 7:
  - Anchor: Pflichtfelder, Quads-Integritaet, anchored-Konsistenz
  - Coding: Ankerpflicht (7.4), Inferenz->Begruendung (7.5), Wertebereiche
  - DocumentIndex: pages/page_count, Marker-Grenzen
  - RenderOptions: Modus, Opacity, Blend
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QCA_AID_assets.annotation.model import (  # noqa: E402
    Anchor,
    Coding,
    DocumentIndex,
    Marker,
    PageText,
    RenderOptions,
)


# --------------------------------------------------------------------------
# Anchor
# --------------------------------------------------------------------------

def test_anchor_valid_when_anchored():
    anchor = Anchor(
        doc_id="doc1",
        page=3,
        quote="unter Druck geraten",
        prefix="Die Befragten ",
        suffix=" und mussten priorisieren.",
        quads=[[10, 20, 100, 34], [10, 36, 90, 50]],
        anchored=True,
    )
    ok, errors = anchor.validate()
    assert ok, errors


def test_anchor_invalid_page():
    anchor = Anchor(doc_id="doc1", page=0, quote="x")
    ok, errors = anchor.validate()
    assert not ok
    assert any("page" in e for e in errors)


def test_anchor_anchored_requires_quote_and_quads():
    no_quote = Anchor(doc_id="doc1", page=1, quote="", quads=[[0, 0, 1, 1]], anchored=True)
    ok, errors = no_quote.validate()
    assert not ok

    no_quads = Anchor(doc_id="doc1", page=1, quote="abc", quads=[], anchored=True)
    ok, errors = no_quads.validate()
    assert not ok


def test_anchor_rejects_inverted_quad():
    anchor = Anchor(
        doc_id="doc1", page=1, quote="abc", quads=[[100, 20, 10, 34]], anchored=True
    )
    ok, errors = anchor.validate()
    assert not ok
    assert any("invertiert" in e for e in errors)


def test_anchor_roundtrip():
    anchor = Anchor(
        doc_id="doc1",
        page=2,
        quote="q",
        quads=[[0, 0, 10, 10]],
        anchored=True,
        page_label="ii",
    )
    assert Anchor.from_dict(anchor.to_dict()) == anchor


# --------------------------------------------------------------------------
# Coding
# --------------------------------------------------------------------------

def _valid_anchor() -> Anchor:
    return Anchor(
        doc_id="doc1",
        page=1,
        quote="Text",
        quads=[[0, 0, 10, 10]],
        anchored=True,
    )


def test_coding_valid_with_anchor():
    coding = Coding(
        segment_id="doc1_chunk_0",
        doc_id="doc1",
        category="Problemdruck",
        confidence=0.9,
        anchor=_valid_anchor(),
    )
    ok, errors = coding.validate()
    assert ok, errors


def test_coding_requires_anchor_by_default():
    coding = Coding(segment_id="s1", doc_id="doc1", category="Kat")
    ok, errors = coding.validate()
    assert not ok
    assert any("Anker" in e for e in errors)


def test_coding_anchor_optional_when_not_required():
    coding = Coding(segment_id="s1", doc_id="doc1", category="Kat")
    ok, errors = coding.validate(require_anchor=False)
    assert ok, errors


def test_coding_unanchored_anchor_is_invalid():
    coding = Coding(
        segment_id="s1",
        doc_id="doc1",
        category="Kat",
        anchor=Anchor(doc_id="doc1", page=1, quote="x", anchored=False),
    )
    ok, errors = coding.validate()
    assert not ok
    assert any("unanchored" in e for e in errors)


def test_coding_inference_requires_justification():
    coding = Coding(
        segment_id="s1",
        doc_id="doc1",
        category="Kat",
        inference=True,
        anchor=_valid_anchor(),
    )
    ok, errors = coding.validate()
    assert not ok
    assert any("Begruendung" in e for e in errors)

    coding.justification = "weil abgeleitet"
    ok, errors = coding.validate()
    assert ok, errors


def test_coding_rejects_bad_enum_and_confidence():
    coding = Coding(
        segment_id="s1",
        doc_id="doc1",
        category="Kat",
        status="bogus",
        created_by="robot",
        confidence=1.5,
        anchor=_valid_anchor(),
    )
    ok, errors = coding.validate()
    assert not ok
    assert any("status" in e for e in errors)
    assert any("created_by" in e for e in errors)
    assert any("confidence" in e for e in errors)


def test_coding_roundtrip_with_anchor():
    coding = Coding(
        segment_id="s1",
        doc_id="doc1",
        category="Kat",
        subcategories=["A"],
        confidence=0.5,
        justification="j",
        anchor=_valid_anchor(),
    )
    assert Coding.from_dict(coding.to_dict()) == coding


def test_coding_roundtrip_without_anchor():
    coding = Coding(segment_id="s1", doc_id="doc1", category="Kat")
    restored = Coding.from_dict(coding.to_dict())
    assert restored.anchor is None
    assert restored == coding


# --------------------------------------------------------------------------
# DocumentIndex
# --------------------------------------------------------------------------

def test_document_index_valid():
    index = DocumentIndex(
        doc_id="doc1",
        page_count=2,
        pages=[
            PageText(page=1, text="a", start=0, end=1),
            PageText(page=2, text="b", start=1, end=2),
        ],
        page_labels=["i", "1"],
        page_markers=[Marker(page=1, start=0, end=1), Marker(page=2, start=1, end=2)],
    )
    ok, errors = index.validate()
    assert ok, errors


def test_document_index_page_count_mismatch():
    index = DocumentIndex(
        doc_id="doc1",
        page_count=3,
        pages=[PageText(page=1, text="a", start=0, end=1)],
    )
    ok, errors = index.validate()
    assert not ok
    assert any("page_count" in e for e in errors)


def test_document_index_marker_out_of_range():
    index = DocumentIndex(
        doc_id="doc1",
        page_count=1,
        pages=[PageText(page=1, text="a", start=0, end=1)],
        page_markers=[Marker(page=5, start=0, end=1)],
    )
    ok, errors = index.validate()
    assert not ok
    assert any("ausserhalb" in e for e in errors)


def test_document_index_roundtrip():
    index = DocumentIndex(
        doc_id="doc1",
        page_count=1,
        pages=[PageText(page=1, text="a", start=0, end=1)],
        page_markers=[Marker(page=1, start=0, end=1)],
    )
    assert DocumentIndex.from_dict(index.to_dict()) == index


# --------------------------------------------------------------------------
# RenderOptions
# --------------------------------------------------------------------------

def test_render_options_valid_and_defaults():
    opts = RenderOptions()
    ok, errors = opts.validate()
    assert ok, errors
    assert opts.mode == "highlight"
    assert opts.blend == "multiply"


def test_render_options_invalid_values():
    opts = RenderOptions(mode="glow", opacity=2.0, blend="screen")
    ok, errors = opts.validate()
    assert not ok
    assert any("mode" in e for e in errors)
    assert any("opacity" in e for e in errors)
    assert any("blend" in e for e in errors)


def test_render_options_roundtrip():
    opts = RenderOptions(mode="underline", active_coding_id="c1", opacity=0.3, blend="normal")
    assert RenderOptions.from_dict(opts.to_dict()) == opts


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
