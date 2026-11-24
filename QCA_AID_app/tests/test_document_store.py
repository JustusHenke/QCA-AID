"""
Tests fuer den DocumentStore (Task 4).

Prueft Cache und Ablage (Anforderungen 1.7-1.9, 9.3, 9.4):
  - Inhalts-Hash-Cache wird wiederverwendet
  - canonical.pdf + original landen in der Dokumentablage
  - Konvertierungsfehler -> not_convertible, ohne Abbruch
  - Groessenlimit -> not_convertible
  - ingest_directory verarbeitet mehrere Dateien inkl. Fehler
  - write_index schreibt document.json
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QCA_AID_assets.preprocessing.document_converter import (  # noqa: E402
    ConvertError,
    ConvertResult,
)
from QCA_AID_assets.preprocessing.document_store import (  # noqa: E402
    DocumentStore,
    STATUS_CACHED,
    STATUS_CONVERTED,
    STATUS_NOT_CONVERTIBLE,
)

MINIMAL_PDF = b"%PDF-1.4\n%%EOF\n"


class FakeConverter:
    """Schreibt ein Minimal-PDF und zaehlt Aufrufe."""

    def __init__(self):
        self.calls = 0

    def convert(self, src, out_dir, canonical_name="canonical.pdf"):
        self.calls += 1
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf = out_dir / canonical_name
        pdf.write_bytes(MINIMAL_PDF)
        return ConvertResult(
            pdf_path=pdf, page_count=1, page_labels=None, converter="fake"
        )


class FailingConverter:
    def __init__(self):
        self.calls = 0

    def convert(self, src, out_dir, canonical_name="canonical.pdf"):
        self.calls += 1
        raise ConvertError(Path(src), ["fake: bewusst fehlgeschlagen"])


def _src(tmp_path: Path, name="doc.txt", content="hallo welt") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# Cache & Ablage
# --------------------------------------------------------------------------

def test_first_ingest_converts_and_places_files(tmp_path):
    store = DocumentStore(tmp_path / "proj", converter=FakeConverter())
    src = _src(tmp_path)

    result = store.ingest(src, doc_id="doc")

    assert result.status == STATUS_CONVERTED
    assert result.ok
    assert result.pdf_path.exists()
    assert (result.doc_dir / "canonical.pdf").exists()
    assert (result.doc_dir / "original.txt").exists()


def test_second_ingest_hits_cache(tmp_path):
    fake = FakeConverter()
    store = DocumentStore(tmp_path / "proj", converter=fake)
    src = _src(tmp_path)

    first = store.ingest(src, doc_id="doc")
    second = store.ingest(src, doc_id="doc")

    assert first.status == STATUS_CONVERTED
    assert second.status == STATUS_CACHED
    assert fake.calls == 1


def test_changed_content_invalidates_cache(tmp_path):
    fake = FakeConverter()
    store = DocumentStore(tmp_path / "proj", converter=fake)
    src = _src(tmp_path, content="version 1")
    store.ingest(src, doc_id="doc")

    src.write_text("version 2", encoding="utf-8")
    second = store.ingest(src, doc_id="doc")

    assert second.status == STATUS_CONVERTED
    assert fake.calls == 2


# --------------------------------------------------------------------------
# Fehlerfaelle
# --------------------------------------------------------------------------

def test_conversion_failure_marks_not_convertible(tmp_path):
    store = DocumentStore(tmp_path / "proj", converter=FailingConverter())
    src = _src(tmp_path)

    result = store.ingest(src, doc_id="kaputt")

    assert result.status == STATUS_NOT_CONVERTIBLE
    assert not result.ok
    assert result.pdf_path is None
    registry = store.load_not_convertible()
    assert "kaputt" in registry
    assert registry["kaputt"]["reasons"]


def test_successful_ingest_clears_stale_error_entry(tmp_path):
    """Nach einem gelungenen Lauf darf kein alter Fehlergrund stehen bleiben."""

    class SwitchableConverter:
        def __init__(self):
            self.fail = True

        def convert(self, src, out_dir, canonical_name="canonical.pdf"):
            if self.fail:
                raise ConvertError(Path(src), ["libreoffice: nicht verfuegbar"])
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pdf = out_dir / canonical_name
            pdf.write_bytes(MINIMAL_PDF)
            return ConvertResult(
                pdf_path=pdf, page_count=1, page_labels=None, converter="fake"
            )

    converter = SwitchableConverter()
    store = DocumentStore(tmp_path / "proj", converter=converter)
    src = _src(tmp_path)

    failed = store.ingest(src, doc_id="doc")
    assert failed.status == STATUS_NOT_CONVERTIBLE
    assert "doc" in store.load_not_convertible()

    converter.fail = False
    succeeded = store.ingest(src, doc_id="doc")

    assert succeeded.ok
    assert "doc" not in store.load_not_convertible()


def test_oversize_file_is_rejected_without_conversion(tmp_path):
    fake = FakeConverter()
    store = DocumentStore(tmp_path / "proj", converter=fake, max_convert_mb=0.000001)
    src = _src(tmp_path, content="x" * 100)

    result = store.ingest(src, doc_id="gross")

    assert result.status == STATUS_NOT_CONVERTIBLE
    assert any("zu gross" in r for r in result.reasons)
    assert fake.calls == 0


def test_ingest_directory_continues_after_failure(tmp_path):
    fake = FakeConverter()
    store = DocumentStore(tmp_path / "proj", converter=fake)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _src(input_dir, "a.txt")
    _src(input_dir, "b.txt", content="anderer inhalt")

    results = store.ingest_directory(input_dir, extensions=[".txt"])

    assert len(results) == 2
    assert all(r.ok for r in results)


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------

def test_write_index_writes_document_json(tmp_path):
    store = DocumentStore(tmp_path / "proj", converter=FakeConverter())
    store.ingest(_src(tmp_path), doc_id="doc")

    path = store.write_index("doc", {"doc_id": "doc", "page_count": 1})

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["page_count"] == 1


# --------------------------------------------------------------------------
# Hash
# --------------------------------------------------------------------------

def test_content_hash_stable_and_distinct(tmp_path):
    a = _src(tmp_path, "a.txt", "gleicher inhalt")
    b = _src(tmp_path, "b.txt", "gleicher inhalt")
    c = _src(tmp_path, "c.txt", "anderer inhalt")

    ha = DocumentStore.content_hash(a)
    assert ha == DocumentStore.content_hash(b)
    assert ha != DocumentStore.content_hash(c)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))