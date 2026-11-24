"""
Tests fuer den DocumentIndexer (Task 5).

Prueft Seitenzahl, Seitenlabels, Text-Offsets und Page_Marker
(Anforderungen 2.1, 2.2, 2.5).
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fitz = pytest.importorskip("fitz")

from QCA_AID_assets.preprocessing.document_index import (  # noqa: E402
    PAGE_SEPARATOR,
    DocumentIndexer,
    IndexBuildError,
)


def make_pdf(path: Path, page_texts, labels=None) -> Path:
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    if labels is not None:
        doc.set_page_labels(labels)
    doc.save(path)
    doc.close()
    return path


# --------------------------------------------------------------------------
# Aufbau
# --------------------------------------------------------------------------

def test_build_page_count_and_validation(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["Seite eins", "Seite zwei", "Seite drei"])
    index = DocumentIndexer().build(pdf, doc_id="a")

    assert index.page_count == 3
    assert len(index.pages) == 3
    ok, errors = index.validate()
    assert ok, errors


def test_offsets_are_contiguous(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["aaa", "bbbb", "cc"])
    index = DocumentIndexer().build(pdf, doc_id="a")

    for prev, nxt in zip(index.pages, index.pages[1:]):
        assert nxt.start == prev.end + len(PAGE_SEPARATOR)


def test_page_text_located_at_offset(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["eins", "zwei", "drei"])
    index = DocumentIndexer().build(pdf, doc_id="a")

    full = PAGE_SEPARATOR.join(p.text for p in index.pages)
    second = index.pages[1]
    assert full[second.start:second.end] == second.text
    assert "zwei" in second.text


def test_markers_match_page_ranges(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["x", "y"])
    index = DocumentIndexer().build(pdf, doc_id="a")

    assert len(index.page_markers) == 2
    for page, marker in zip(index.pages, index.page_markers):
        assert marker.page == page.page
        assert marker.start == page.start
        assert marker.end == page.end


def test_page_for_offset(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["eins", "zwei"])
    index = DocumentIndexer().build(pdf, doc_id="a")

    assert index.page_for_offset(index.pages[0].start) == 1
    assert index.page_for_offset(index.pages[1].start) == 2


def test_marked_text_contains_markers(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["eins", "zwei"])
    index = DocumentIndexer().build(pdf, doc_id="a")

    marked = index.to_marked_text()
    assert "<<PAGE 1>>" in marked
    assert "<<END PAGE 1>>" in marked
    assert "<<PAGE 2>>" in marked


# --------------------------------------------------------------------------
# Seitenlabels
# --------------------------------------------------------------------------

def test_page_labels_expanded(tmp_path):
    labels = [
        {"startpage": 0, "prefix": "", "style": "D", "firstpagenum": 1},
        {"startpage": 1, "prefix": "", "style": "r", "firstpagenum": 1},
    ]
    pdf = make_pdf(tmp_path / "a.pdf", ["cover", "einleitung"], labels=labels)
    index = DocumentIndexer().build(pdf, doc_id="a")

    assert index.page_labels is not None
    assert index.page_labels[0] == "1"
    assert index.page_labels[1] == "i"


# --------------------------------------------------------------------------
# Kantenfaelle
# --------------------------------------------------------------------------

def test_empty_page_is_kept(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["text", "", "text2"])
    index = DocumentIndexer().build(pdf, doc_id="a")

    assert index.page_count == 3
    assert index.pages[1].text == ""
    assert index.pages[1].start == index.pages[1].end


def test_missing_file_raises(tmp_path):
    with pytest.raises(IndexBuildError):
        DocumentIndexer().build(tmp_path / "nope.pdf", doc_id="x")


def test_write_index_roundtrip_via_store(tmp_path):
    from QCA_AID_assets.preprocessing.document_store import DocumentStore
    from QCA_AID_assets.annotation.model import DocumentIndex

    pdf = make_pdf(tmp_path / "a.pdf", ["eins"])
    index = DocumentIndexer().build(pdf, doc_id="a")

    store = DocumentStore(tmp_path / "proj")
    path = store.write_index("a", index.to_dict())

    restored = DocumentIndex.from_dict(__import__("json").loads(path.read_text(encoding="utf-8")))
    assert restored == index


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))