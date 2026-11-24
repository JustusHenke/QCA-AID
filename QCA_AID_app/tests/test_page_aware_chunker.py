"""
Tests fuer den seitenbewussten Chunker (Task 8).

Prueft Anforderung 2.3/2.4: Chunks tragen Seitenzuordnung, Offsets bleiben
konsistent, seitenuebergreifende Chunks werden korrekt markiert.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fitz = pytest.importorskip("fitz")

from QCA_AID_assets.preprocessing.document_index import DocumentIndexer  # noqa: E402
from QCA_AID_assets.preprocessing.page_aware_chunker import PageAwareChunker  # noqa: E402


def make_index(tmp_path, pages, name="doc.pdf"):
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page(width=595, height=842)
        y = 100
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 16
    pdf = tmp_path / name
    doc.save(pdf)
    doc.close()
    return DocumentIndexer().build(pdf, doc_id="d")


PAGES = [
    ["Satz eins. Satz zwei."],
    ["Satz drei. Satz vier."],
]


# --------------------------------------------------------------------------
# Grundverhalten
# --------------------------------------------------------------------------

def test_chunks_have_sequential_ids_and_pages(tmp_path):
    index = make_index(tmp_path, PAGES)
    chunker = PageAwareChunker(chunk_size=40, overlap=0)

    chunks = chunker.chunk_document(index)

    assert chunks
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert [c.segment_id for c in chunks] == [f"d_chunk_{i}" for i in range(len(chunks))]
    for c in chunks:
        assert 1 <= c.page_start <= index.page_count
        assert 1 <= c.page_end <= index.page_count
        assert c.page_start <= c.page_end


def test_offsets_match_full_text(tmp_path):
    index = make_index(tmp_path, PAGES)
    full = index.full_text()
    chunks = PageAwareChunker(chunk_size=40, overlap=10).chunk_document(index)

    for c in chunks:
        assert full[c.start:c.end] == c.text
        assert c.doc_id == "d"


def test_pages_assigned_per_chunk(tmp_path):
    index = make_index(tmp_path, PAGES)
    chunks = PageAwareChunker(chunk_size=20, overlap=0).chunk_document(index)

    # Mindestens ein Chunk auf Seite 1 und einer auf Seite 2.
    assert any(c.page_start == 1 for c in chunks)
    assert any(c.page_start == 2 for c in chunks)


# --------------------------------------------------------------------------
# Seitenuebergreifende Chunks
# --------------------------------------------------------------------------

def test_large_chunk_spans_pages(tmp_path):
    index = make_index(tmp_path, PAGES)
    chunks = PageAwareChunker(chunk_size=1000, overlap=0).chunk_document(index)

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    # Text enthaelt Inhalt beider Seiten
    assert "Satz eins" in chunks[0].text
    assert "Satz vier" in chunks[0].text


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------

def test_overlap_produces_shared_region(tmp_path):
    index = make_index(tmp_path, PAGES)
    chunks = PageAwareChunker(chunk_size=25, overlap=15).chunk_document(index)

    assert len(chunks) >= 2
    overlaps = [chunks[i].start < chunks[i - 1].end for i in range(1, len(chunks))]
    assert any(overlaps)


# --------------------------------------------------------------------------
# Kanten
# --------------------------------------------------------------------------

def test_empty_text_yields_no_chunks(tmp_path):
    index = make_index(tmp_path, [[""]])
    assert PageAwareChunker().chunk_document(index) == []


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        PageAwareChunker(chunk_size=0)
    with pytest.raises(ValueError):
        PageAwareChunker(chunk_size=100, overlap=100)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))