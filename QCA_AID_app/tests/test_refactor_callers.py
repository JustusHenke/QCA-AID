"""
Regressions-Tests fuer Task 14 (Full-Refactor).

Sichert die oeffentlichen Schnittstellen der abgeloesten Komponenten:
  - DocumentToPDFConverter.convert() -> str|None
  - PDFAnnotator.annotate_pdf_with_codings() -> str|None (jetzt mit Ankern)
  - DocumentReader.read_documents()/read_documents_sync() inkl. indexes
  - MaterialLoader.load_documents() ueber die zentrale Pipeline
"""

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fitz = pytest.importorskip("fitz")

from QCA_AID_assets.utils.export.converters import DocumentToPDFConverter  # noqa: E402
from QCA_AID_assets.utils.export.pdf_annotator import PDFAnnotator  # noqa: E402
from QCA_AID_assets.utils.io.document_reader import DocumentReader  # noqa: E402


def make_pdf(path: Path, lines=("Die Befragten unter Druck geraten.",)) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 100
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    doc.save(path)
    doc.close()
    return path


# --------------------------------------------------------------------------
# DocumentToPDFConverter
# --------------------------------------------------------------------------

def test_converter_passthrough_pdf(tmp_path):
    src = make_pdf(tmp_path / "quelle.pdf")
    converter = DocumentToPDFConverter(str(tmp_path / "out"))

    result = converter.convert(str(src))

    assert result is not None
    assert Path(result).exists()
    assert Path(result).name == "quelle.pdf"


def test_converter_unknown_format_returns_none(tmp_path):
    src = tmp_path / "datei.xyz"
    src.write_text("nope")
    converter = DocumentToPDFConverter(str(tmp_path / "out"))

    assert converter.convert(str(src)) is None


def test_converter_cleanup_is_safe(tmp_path):
    converter = DocumentToPDFConverter(str(tmp_path / "out"))
    converter.cleanup_temp_pdfs()  # darf nicht werfen


# --------------------------------------------------------------------------
# PDFAnnotator
# --------------------------------------------------------------------------

CHUNK_TEXT = "Die Befragten unter Druck geraten und mussten priorisieren."
CODING = {
    "segment_id": "doc_chunk_0",
    "category": "Problemdruck",
    "subcategories": ["Zeitkonflikt"],
    "confidence": {"total": 0.9},
    "justification": "explizite Problemnennung",
}


def test_pdf_annotator_writes_highlight(tmp_path):
    src = make_pdf(tmp_path / "doc.pdf", lines=(CHUNK_TEXT,))
    out = tmp_path / "annotiert.pdf"
    annotator = PDFAnnotator(results_exporter=None)

    result = annotator.annotate_pdf_with_codings(
        str(src), [dict(CODING)], {"doc": [CHUNK_TEXT]}, str(out)
    )

    assert result is not None and Path(result).exists()
    doc = fitz.open(result)
    highlights = []
    for page in doc:
        highlights += [a for a in page.annots() if a.type[0] == fitz.PDF_ANNOT_HIGHLIGHT]
    assert highlights
    assert doc.xref_get_key(highlights[0].xref, "BM")[1] == "/Multiply"


def test_pdf_annotator_returns_none_without_resolvable_codings(tmp_path):
    src = make_pdf(tmp_path / "doc.pdf")
    bad = dict(CODING)
    bad["segment_id"] = "doc_chunk_99"
    annotator = PDFAnnotator(results_exporter=None)

    assert annotator.annotate_pdf_with_codings(
        str(src), [bad], {"doc": [CHUNK_TEXT]}, str(tmp_path / "out.pdf")
    ) is None


def test_pdf_annotator_skips_path_artifacts(tmp_path):
    src = make_pdf(tmp_path / "doc.pdf", lines=(CHUNK_TEXT,))
    chunk = "file:///C:/Users/x/OneDrive/Projekte/Forschung/doc.txt " + CHUNK_TEXT
    annotator = PDFAnnotator(results_exporter=None)

    result = annotator.annotate_pdf_with_codings(
        str(src), [dict(CODING)], {"doc": [chunk]}, str(tmp_path / "out.pdf")
    )
    assert result is not None


# --------------------------------------------------------------------------
# DocumentReader
# --------------------------------------------------------------------------

def test_document_reader_sync_and_indexes(tmp_path):
    make_pdf(tmp_path / "studie.pdf", lines=("Erste Seite Text.",))
    reader = DocumentReader(str(tmp_path))

    documents = reader.read_documents_sync()

    assert "studie.pdf" in documents
    assert "Erste Seite Text" in documents["studie.pdf"]
    assert "studie.pdf" in reader.indexes
    assert reader.indexes["studie.pdf"].page_count == 1


def test_document_reader_async_wrapper(tmp_path):
    make_pdf(tmp_path / "studie.pdf")
    reader = DocumentReader(str(tmp_path))

    documents = asyncio.run(reader.read_documents())

    assert "studie.pdf" in documents


def test_document_reader_text_has_no_page_markers(tmp_path):
    make_pdf(tmp_path / "studie.pdf")
    reader = DocumentReader(str(tmp_path))

    text = reader.read_documents_sync()["studie.pdf"]

    assert "<<PAGE" not in text


# --------------------------------------------------------------------------
# MaterialLoader
# --------------------------------------------------------------------------

def test_material_loader_uses_pipeline(tmp_path):
    from QCA_AID_assets.preprocessing.material_loader import MaterialLoader

    (tmp_path / "notiz.txt").write_text("Ein kurzer Text.", encoding="utf-8")
    try:
        loader = MaterialLoader(data_dir=str(tmp_path))
    except OSError:
        pytest.skip("spaCy-Modell de_core_news_sm nicht installiert")

    documents = loader.load_documents()

    assert "notiz.txt" in documents
    assert "kurzer Text" in documents["notiz.txt"]


def test_material_loader_has_legacy_fallback():
    from QCA_AID_assets.preprocessing.material_loader import MaterialLoader

    assert hasattr(MaterialLoader, "_load_documents_legacy")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))