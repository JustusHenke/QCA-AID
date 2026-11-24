"""
Tests fuer die Ablageregel annotierter PDFs (Analysis-Pipeline).

Regel seit 0.15.3:
  * annotierte PDFs liegen in ``<output>/Annotated/``
  * Dateiname mit Namensstempel der Excel-Ausgabe
    (``<stem>_QCA_annotiert_YYYYMMDD_HHMMSS.pdf``)
  * unannotierte Zwischen-PDFs (Konvertierung DOCX/TXT -> PDF) werden nicht
    im Ausgabeverzeichnis abgelegt
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fitz = pytest.importorskip("fitz")

from QCA_AID_assets.utils.export.converters import DocumentToPDFConverter  # noqa: E402
from QCA_AID_assets.utils.export.paths import (  # noqa: E402
    ANNOTATED_SUBDIR,
    annotated_pdf_path,
    conversion_workspace,
    export_stamp,
)

STAMP_RE = re.compile(r"_\d{8}_\d{6}\.pdf$")

CHUNK_TEXT = "Die Befragten unter Druck geraten und mussten priorisieren."


def make_pdf(path: Path, lines=(CHUNK_TEXT,)) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 100
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    doc.save(path)
    doc.close()
    return path


def coding(segment_id: str = "doc_chunk_0", category: str = "Problemdruck") -> dict:
    return {
        "segment_id": segment_id,
        "document": "doc.pdf",
        "category": category,
        "subcategories": ["Zeitkonflikt"],
        "confidence": {"total": 0.9},
        "justification": "explizite Problemnennung",
    }


# --------------------------------------------------------------------------
# Pfadregel
# --------------------------------------------------------------------------

def test_annotated_pdf_path_uses_subfolder_and_stamp(tmp_path):
    path = annotated_pdf_path(tmp_path, "text_demo_wiss", "20261124_120100")

    assert path.parent == tmp_path / ANNOTATED_SUBDIR
    assert path.name == "text_demo_wiss_QCA_annotiert_20261124_120100.pdf"
    assert path.parent.is_dir()  # Ordner wird angelegt


def test_export_stamp_matches_excel_convention():
    assert re.fullmatch(r"\d{8}_\d{6}", export_stamp())


def test_conversion_workspace_is_removed(tmp_path):
    with conversion_workspace(tmp_path) as workdir:
        (workdir / "zwischen.pdf").write_bytes(b"%PDF-1.4")
        assert workdir.is_dir()
        # Zwischen-PDFs entstehen nie im Ausgabeverzeichnis
        assert not any(tmp_path.iterdir())

    assert not workdir.exists()


# --------------------------------------------------------------------------
# Konverter
# --------------------------------------------------------------------------

def test_converter_can_write_into_a_work_dir(tmp_path):
    src = make_pdf(tmp_path / "quelle.pdf")
    out_dir = tmp_path / "output"
    work_dir = tmp_path / "work"

    converter = DocumentToPDFConverter(str(out_dir))
    result = converter.convert(str(src), target_dir=str(work_dir))

    assert result is not None
    assert Path(result).parent == work_dir
    assert Path(result).name == "quelle.pdf"
    assert not out_dir.exists() or not list(out_dir.iterdir())


# --------------------------------------------------------------------------
# ResultsExporter
# --------------------------------------------------------------------------

@pytest.fixture()
def exporter(tmp_path):
    from QCA_AID_assets.export.results_exporter import ResultsExporter

    return ResultsExporter(str(tmp_path / "output"), attribute_labels={})


def test_exporter_writes_annotated_pdf_into_annotated_subfolder(tmp_path, exporter):
    data_dir = tmp_path / "input"
    data_dir.mkdir()
    make_pdf(data_dir / "doc.pdf")

    created = exporter.export_annotated_pdfs(
        codings=[coding()], chunks={"doc": [CHUNK_TEXT]}, data_dir=str(data_dir)
    )

    assert len(created) == 1
    path = Path(created[0])
    assert path.exists()
    assert path.parent == Path(exporter.output_dir) / ANNOTATED_SUBDIR
    assert path.name.startswith("doc_QCA_annotiert_")
    assert STAMP_RE.search(path.name)
    # im Ausgabeverzeichnis selbst liegt kein PDF
    assert not list(Path(exporter.output_dir).glob("*.pdf"))


def test_exporter_uses_one_stamp_for_all_files(tmp_path, exporter):
    data_dir = tmp_path / "input"
    data_dir.mkdir()
    make_pdf(data_dir / "eins.pdf")
    make_pdf(data_dir / "zwei.pdf")

    created = exporter.export_annotated_pdfs(
        codings=[
            coding("eins_chunk_0"),
            {**coding("zwei_chunk_0"), "document": "zwei.pdf"},
        ],
        chunks={"eins": [CHUNK_TEXT], "zwei": [CHUNK_TEXT]},
        data_dir=str(data_dir),
    )

    stamps = {STAMP_RE.search(Path(p).name).group(0) for p in created}
    assert len(created) == 2
    assert len(stamps) == 1


def test_all_formats_export_leaves_no_unannotated_pdf(tmp_path, exporter):
    data_dir = tmp_path / "input"
    data_dir.mkdir()
    make_pdf(data_dir / "doc.pdf")

    created = exporter.export_annotated_pdfs_all_formats(
        codings=[coding()], chunks={"doc": [CHUNK_TEXT]}, data_dir=str(data_dir)
    )

    assert len(created) == 1
    output = Path(exporter.output_dir)
    assert Path(created[0]).parent == output / ANNOTATED_SUBDIR
    assert STAMP_RE.search(Path(created[0]).name)
    assert not list(output.glob("*.pdf"))


def test_pdf_annotator_default_path_follows_the_rule(tmp_path):
    from QCA_AID_assets.utils.export.pdf_annotator import PDFAnnotator

    src = make_pdf(tmp_path / "doc.pdf")
    annotator = PDFAnnotator(results_exporter=None)

    result = annotator.annotate_pdf_with_codings(
        str(src), [coding()], {"doc": [CHUNK_TEXT]}
    )

    assert result is not None
    path = Path(result)
    assert path.parent == tmp_path / ANNOTATED_SUBDIR
    assert path.name.startswith("doc_QCA_annotiert_")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
