"""
Tests fuer den DocumentConverter (Task 2).

Prueft die Fallback-Kette und Trigger aus Anforderung 9.2:
  - Binary fehlt (available() False) -> naechste Stufe
  - StageError (Exit != 0 / Timeout) -> naechste Stufe
  - leeres/ungueltiges Ausgabe-PDF -> naechste Stufe
  - erschoepfte Kette -> ConvertError mit Gruenden
  - unbekannte Endung -> ConvertError
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QCA_AID_assets.preprocessing.document_converter import (  # noqa: E402
    ConversionStage,
    ConvertError,
    ConverterFactory,
    DocumentConverter,
    StageError,
    _run_subprocess,
    is_valid_pdf,
    pdf_page_count,
)

MINIMAL_PDF = b"%PDF-1.4\n%%EOF\n"


def _write_minimal_pdf(path: Path) -> Path:
    path.write_bytes(MINIMAL_PDF)
    return path


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeFail(ConversionStage):
    name = "fake_fail"

    def run(self, src, out_pdf, timeout):
        raise StageError("bewusst fehlgeschlagen")


class FakeEmpty(ConversionStage):
    name = "fake_empty"

    def run(self, src, out_pdf, timeout):
        out_pdf.write_bytes(b"not a pdf")


class FakeOk(ConversionStage):
    name = "fake_ok"

    def run(self, src, out_pdf, timeout):
        out_pdf.write_bytes(MINIMAL_PDF)


class FakeUnavailable(ConversionStage):
    name = "fake_unavailable"

    def available(self):
        return False

    def run(self, src, out_pdf, timeout):  # pragma: no cover - wird nie erreicht
        raise AssertionError("nicht verfuegbar, darf nicht laufen")


# --------------------------------------------------------------------------
# PDF-Hilfsfunktionen
# --------------------------------------------------------------------------

def test_is_valid_pdf(tmp_path):
    good = _write_minimal_pdf(tmp_path / "good.pdf")
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    assert is_valid_pdf(good)
    assert not is_valid_pdf(empty)
    assert not is_valid_pdf(tmp_path / "missing.pdf")


def test_pdf_page_count_never_crashes(tmp_path):
    result = pdf_page_count(_write_minimal_pdf(tmp_path / "x.pdf"))
    assert isinstance(result, int)


# --------------------------------------------------------------------------
# Fallback-Kette
# --------------------------------------------------------------------------

def test_fallback_to_next_stage_on_failure(tmp_path):
    factory = ConverterFactory(chains={".pdf": [FakeFail, FakeOk]})
    converter = DocumentConverter(factory=factory)
    src = _write_minimal_pdf(tmp_path / "src.pdf")

    result = converter.convert(src, tmp_path / "out")

    assert result.converter == "fake_ok"
    assert is_valid_pdf(result.pdf_path)


def test_unavailable_stage_is_skipped(tmp_path):
    factory = ConverterFactory(chains={".pdf": [FakeUnavailable, FakeOk]})
    converter = DocumentConverter(factory=factory)
    src = _write_minimal_pdf(tmp_path / "src.pdf")

    result = converter.convert(src, tmp_path / "out")
    assert result.converter == "fake_ok"


def test_invalid_output_triggers_next_stage(tmp_path):
    factory = ConverterFactory(chains={".pdf": [FakeEmpty, FakeOk]})
    converter = DocumentConverter(factory=factory)
    src = _write_minimal_pdf(tmp_path / "src.pdf")

    result = converter.convert(src, tmp_path / "out")
    assert result.converter == "fake_ok"


def test_exhausted_chain_raises_converterror(tmp_path):
    factory = ConverterFactory(chains={".pdf": [FakeFail, FakeUnavailable, FakeEmpty]})
    converter = DocumentConverter(factory=factory)
    src = _write_minimal_pdf(tmp_path / "src.pdf")

    with pytest.raises(ConvertError) as exc:
        converter.convert(src, tmp_path / "out")
    assert len(exc.value.reasons) == 3


def test_unknown_extension_raises(tmp_path):
    converter = DocumentConverter(factory=ConverterFactory(chains={}))
    src = tmp_path / "src.xyz"
    src.write_text("hello")

    with pytest.raises(ConvertError) as exc:
        converter.convert(src, tmp_path / "out")
    assert "Endung" in exc.value.reasons[0]


def test_passthrough_stage_copies_pdf(tmp_path):
    factory = ConverterFactory(chains={".pdf": [FakeOk]})
    converter = DocumentConverter(factory=factory)
    src = _write_minimal_pdf(tmp_path / "src.pdf")

    result = converter.convert(src, tmp_path / "out")
    assert result.pdf_path.name == "canonical.pdf"
    assert result.pdf_path.exists()


def test_custom_canonical_name(tmp_path):
    factory = ConverterFactory(chains={".pdf": [FakeOk]})
    converter = DocumentConverter(factory=factory)
    src = _write_minimal_pdf(tmp_path / "src.pdf")

    result = converter.convert(src, tmp_path / "out", canonical_name="doc.pdf")
    assert result.pdf_path.name == "doc.pdf"


# --------------------------------------------------------------------------
# Unterprozess-Timeout (echter Codepfad)
# --------------------------------------------------------------------------

def test_run_subprocess_timeout_raises_stageerror():
    with pytest.raises(StageError) as exc:
        _run_subprocess([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.5)
    assert "Timeout" in str(exc.value)


def test_run_subprocess_missing_binary_raises_stageerror():
    with pytest.raises(StageError):
        _run_subprocess(["definitely-not-a-binary-xyz"], timeout=2)


def test_run_subprocess_nonzero_exit_raises_stageerror():
    with pytest.raises(StageError) as exc:
        _run_subprocess([sys.executable, "-c", "import sys; sys.exit(3)"], timeout=5)
    assert "Exit 3" in str(exc.value)


# --------------------------------------------------------------------------
# Abhaengigkeitserklaerung (PyMuPDF traegt die Text-Fallbacks)
# --------------------------------------------------------------------------

def test_pymupdf_is_a_core_requirement():
    """Ohne PyMuPDF greifen die DOCX/MD/TXT-Fallbacks nicht - es ist Pflicht."""
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "PyMuPDF" in requirements
    assert "PyMuPDF" in pyproject.split("[project.optional-dependencies]")[0]


def test_optional_converters_are_not_core_requirements():
    """weasyprint/docling bleiben optional: sie brauchen native Bibliotheken."""
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    active = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert not any(line.lower().startswith(("weasyprint", "docling")) for line in active)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
