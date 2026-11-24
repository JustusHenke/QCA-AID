"""
Tests fuer den PreviewService (Canonical_PDF-Cache fuer die UI).

Deckt: Dokumentliste, doc_id-Bildung, Ingest+Index, In-Memory-Cache,
Fehlerpfad (PreviewUnavailable) und Seitenmasse.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

fitz = pytest.importorskip("fitz")

from QCA_AID_assets.preprocessing.document_store import (  # noqa: E402
    STATUS_CONVERTED,
    STATUS_NOT_CONVERTIBLE,
    IngestResult,
)
from webapp_logic.preview_service import (  # noqa: E402
    SUPPORTED_INPUT_EXTENSIONS,
    PreviewService,
    PreviewUnavailable,
    doc_id_for,
    list_input_documents,
    page_sizes,
)


def make_pdf(path: Path, lines=("Die Befragten berichten von Zeitdruck.",), size=(595, 842)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=size[0], height=size[1])
    y = 100
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    doc.save(str(path))
    doc.close()
    return path


class CountingStore:
    """Minimaler Store-Ersatz, der Ingest-Aufrufe zaehlt."""

    def __init__(self, real_store, result_factory=None):
        self.real = real_store
        self.calls = 0
        self._factory = result_factory

    def ingest(self, src, doc_id):
        self.calls += 1
        if self._factory is not None:
            return self._factory(src, doc_id)
        return self.real.ingest(src, doc_id)

    def write_index(self, doc_id, index_dict):
        return self.real.write_index(doc_id, index_dict)


class TestInputDocuments:
    def test_lists_supported_extensions_only(self, tmp_path):
        for name in ("a.pdf", "b.txt", "c.docx", "d.md", "e.html", "f.xlsx"):
            (tmp_path / name).write_bytes(b"x")

        names = [doc.name for doc in list_input_documents(tmp_path)]

        assert names == ["a.pdf", "b.txt", "c.docx", "d.md", "e.html"]
        assert ".xlsx" not in SUPPORTED_INPUT_EXTENSIONS

    def test_ignores_hidden_and_lock_files(self, tmp_path):
        (tmp_path / ".hidden.pdf").write_bytes(b"x")
        (tmp_path / "~$gesperrt.docx").write_bytes(b"x")
        (tmp_path / "echt.pdf").write_bytes(b"x")

        assert [doc.name for doc in list_input_documents(tmp_path)] == ["echt.pdf"]

    def test_missing_directory_returns_empty(self, tmp_path):
        assert list_input_documents(tmp_path / "gibt-es-nicht") == []

    def test_document_exposes_size_and_extension(self, tmp_path):
        target = tmp_path / "klein.TXT"
        target.write_bytes(b"12345")

        document = list_input_documents(tmp_path)[0]

        assert document.extension == ".txt"
        assert document.size_bytes == 5
        assert document.format_size() == "5 B"


class TestDocId:
    def test_stable_and_filesystem_safe(self, tmp_path):
        target = make_pdf(tmp_path / "Interview 01 (A).pdf")

        first = doc_id_for(target)

        assert first == doc_id_for(target)
        assert first.startswith("interview_01_a")
        assert all(char.isalnum() or char == "_" for char in first)

    def test_same_stem_in_different_dirs_differs(self, tmp_path):
        first = tmp_path / "a" / "befragung.pdf"
        second = tmp_path / "b" / "befragung.pdf"
        first.parent.mkdir()
        second.parent.mkdir()
        make_pdf(first)
        make_pdf(second)

        assert doc_id_for(first) != doc_id_for(second)


class TestPreviewService:
    def test_get_builds_index_and_document_json(self, tmp_path):
        source = make_pdf(tmp_path / "input" / "rede.pdf", lines=("Zeile eins.", "Zeile zwei."))
        service = PreviewService(tmp_path / "work")

        document = service.get(source)

        assert document.page_count == 1
        assert document.pdf_path.name == "canonical.pdf"
        assert document.pdf_path.is_file()
        assert document.index is not None
        assert "Zeile eins." in document.index.pages[0].text
        assert (document.pdf_path.parent / "document.json").is_file()

    def test_second_call_uses_memory_cache(self, tmp_path):
        source = make_pdf(tmp_path / "rede.pdf")
        service = PreviewService(tmp_path / "work")
        store = CountingStore(service.store)
        service.store = store

        service.get(source)
        service.get(source)

        assert store.calls == 1

    def test_refresh_forces_new_ingest(self, tmp_path):
        source = make_pdf(tmp_path / "rede.pdf")
        service = PreviewService(tmp_path / "work")
        store = CountingStore(service.store)
        service.store = store

        service.get(source)
        service.get(source, refresh=True)

        assert store.calls == 2

    def test_changed_file_invalidates_cache(self, tmp_path):
        source = make_pdf(tmp_path / "rede.pdf")
        service = PreviewService(tmp_path / "work")
        service.get(source)

        make_pdf(source, lines=("Anderer Inhalt nach Aenderung.",))
        document = service.get(source)

        assert document.index is not None
        assert "Anderer Inhalt" in document.index.pages[0].text

    def test_missing_file_raises(self, tmp_path):
        service = PreviewService(tmp_path / "work")

        with pytest.raises(PreviewUnavailable) as error:
            service.get(tmp_path / "fehlt.pdf")

        assert "nicht gefunden" in " ".join(error.value.reasons)

    def test_conversion_failure_carries_reasons(self, tmp_path):
        source = tmp_path / "alt.doc"
        source.write_bytes(b"kein echtes doc")
        service = PreviewService(tmp_path / "work")

        def failing(src, doc_id):
            return IngestResult(
                doc_id=doc_id,
                src=Path(src),
                status=STATUS_NOT_CONVERTIBLE,
                doc_dir=tmp_path / "work" / "documents" / doc_id,
                pdf_path=None,
                reasons=["LibreOffice-Binary fehlt"],
            )

        service.store = CountingStore(service.store, result_factory=failing)

        with pytest.raises(PreviewUnavailable) as error:
            service.get(source)

        assert error.value.reasons == ["LibreOffice-Binary fehlt"]
        assert "alt.doc" in str(error.value)

    def test_corrupt_document_json_is_rebuilt(self, tmp_path):
        source = make_pdf(tmp_path / "rede.pdf")
        service = PreviewService(tmp_path / "work")
        first = service.get(source)
        (first.pdf_path.parent / "document.json").write_text("{kaputt", encoding="utf-8")

        service.invalidate()
        document = service.get(source)

        assert document.index is not None
        assert any("neu erzeugt" in warning for warning in document.warnings)

    def test_image_only_pdf_has_no_text_layer(self, tmp_path):
        path = tmp_path / "scan.pdf"
        doc = fitz.open()
        doc.new_page(width=595, height=842)
        doc.save(str(path))
        doc.close()
        service = PreviewService(tmp_path / "work")

        document = service.get(path)

        assert document.has_text_layer is False

    def test_status_is_converted_on_first_ingest(self, tmp_path):
        source = make_pdf(tmp_path / "rede.pdf")
        service = PreviewService(tmp_path / "work")

        assert service.get(source).status == STATUS_CONVERTED


class TestPageSizes:
    def test_reports_real_page_geometry(self, tmp_path):
        source = make_pdf(tmp_path / "quer.pdf", size=(842, 595))

        sizes = page_sizes(source)

        assert sizes == [(842.0, 595.0)]

    def test_preview_document_page_sizes(self, tmp_path):
        source = make_pdf(tmp_path / "rede.pdf")
        document = PreviewService(tmp_path / "work").get(source)

        assert document.page_sizes[0][0] == pytest.approx(595.0)
