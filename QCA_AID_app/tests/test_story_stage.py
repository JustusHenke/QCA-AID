"""
Tests fuer den Python-Fallback der Dokumentkonvertierung (PyMuPDF Story).

Hintergrund: ohne LibreOffice/Pandoc/WeasyPrint-Runtime war bisher keine
Vorschau fuer .txt/.md/.html moeglich. StoryStage schliesst diese Luecke
ohne externe Binaries - und erzeugt dabei einen echten Textlayer, damit
Anker auch in konvertierten Textdokumenten funktionieren.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fitz = pytest.importorskip("fitz")

from QCA_AID_assets.preprocessing.document_converter import (  # noqa: E402
    ConverterFactory,
    DocumentConverter,
    DocxTextStage,
    HtmlStoryStage,
    StageError,
    StoryStage,
    _read_text,
)


class TestReadText:
    def test_utf8_with_bom(self, tmp_path):
        target = tmp_path / "bom.txt"
        target.write_bytes("\ufeffInhalt".encode("utf-8"))

        assert _read_text(target) == "Inhalt"

    def test_cp1252_fallback(self, tmp_path):
        target = tmp_path / "alt.txt"
        target.write_bytes("Gr\u00fc\u00dfe".encode("cp1252"))

        assert _read_text(target) == "Gr\u00fc\u00dfe"


class TestStoryStage:
    def test_txt_becomes_pdf_with_text_layer(self, tmp_path):
        src = tmp_path / "interview.txt"
        src.write_text(
            "Die Befragten berichten von Zeitdruck.\n\nZweitens fehlen Ressourcen.",
            encoding="utf-8",
        )
        out = tmp_path / "out" / "canonical.pdf"
        out.parent.mkdir()

        StoryStage().run(src, out, timeout=30)

        assert out.is_file()
        with fitz.open(str(out)) as doc:
            text = doc[0].get_text()
        assert "Zeitdruck" in text
        assert "Ressourcen" in text

    def test_paragraphs_stay_separate(self, tmp_path):
        src = tmp_path / "absaetze.txt"
        src.write_text("Absatz eins.\n\nAbsatz zwei.", encoding="utf-8")
        out = tmp_path / "canonical.pdf"

        StoryStage().run(src, out, timeout=30)

        with fitz.open(str(out)) as doc:
            text = doc[0].get_text()
        assert "Absatz eins." in text
        assert "Absatz zwei." in text

    def test_html_is_escaped_in_text_mode(self, tmp_path):
        src = tmp_path / "roh.txt"
        src.write_text("<b>nicht fett</b> & Zeichen", encoding="utf-8")
        out = tmp_path / "canonical.pdf"

        StoryStage().run(src, out, timeout=30)

        with fitz.open(str(out)) as doc:
            text = doc[0].get_text()
        assert "<b>" in text

    def test_long_text_spans_multiple_pages(self, tmp_path):
        src = tmp_path / "lang.txt"
        src.write_text("\n\n".join(["Ein Absatz mit Inhalt."] * 300), encoding="utf-8")
        out = tmp_path / "canonical.pdf"

        StoryStage().run(src, out, timeout=30)

        with fitz.open(str(out)) as doc:
            assert doc.page_count > 1

    def test_empty_file_raises(self, tmp_path):
        src = tmp_path / "leer.txt"
        src.write_text("   \n\n  ", encoding="utf-8")

        with pytest.raises(StageError):
            StoryStage().run(src, tmp_path / "canonical.pdf", timeout=30)

    def test_available(self):
        assert StoryStage().available() is True


class TestHtmlStoryStage:
    def test_html_renders_structure(self, tmp_path):
        src = tmp_path / "seite.html"
        src.write_text(
            "<html><body><h1>Ueberschrift</h1><p>Ein Absatz.</p></body></html>",
            encoding="utf-8",
        )
        out = tmp_path / "canonical.pdf"

        HtmlStoryStage().run(src, out, timeout=30)

        with fitz.open(str(out)) as doc:
            text = doc[0].get_text()
        assert "Ueberschrift" in text
        assert "Ein Absatz." in text


class TestDocxTextStage:
    def _make_docx(self, path: Path) -> Path:
        docx = pytest.importorskip("docx")
        document = docx.Document()
        document.add_heading("Interview 01", level=1)
        document.add_paragraph("Die Befragten berichten von Zeitdruck.")
        table = document.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "Rolle"
        table.rows[0].cells[1].text = "Lehre"
        document.save(str(path))
        return path

    def test_docx_becomes_pdf_with_text(self, tmp_path):
        src = self._make_docx(tmp_path / "interview.docx")
        out = tmp_path / "canonical.pdf"

        DocxTextStage().run(src, out, timeout=30)

        with fitz.open(str(out)) as doc:
            text = doc[0].get_text()
        assert "Zeitdruck" in text
        assert "Interview 01" in text

    def test_table_row_is_kept(self, tmp_path):
        src = self._make_docx(tmp_path / "interview.docx")
        out = tmp_path / "canonical.pdf"

        DocxTextStage().run(src, out, timeout=30)

        with fitz.open(str(out)) as doc:
            assert "Rolle | Lehre" in doc[0].get_text()

    def test_empty_docx_raises(self, tmp_path):
        docx = pytest.importorskip("docx")
        document = docx.Document()
        path = tmp_path / "leer.docx"
        document.save(str(path))

        with pytest.raises(StageError):
            DocxTextStage().run(path, tmp_path / "canonical.pdf", timeout=30)

    def test_docx_chain_ends_with_python_fallback(self):
        chain = ConverterFactory.DEFAULT_CHAINS[".docx"]

        assert chain[-1] is DocxTextStage


class TestChains:
    def test_txt_chain_ends_with_python_fallback(self):
        chain = ConverterFactory.DEFAULT_CHAINS[".txt"]
        assert chain[-1] is StoryStage
        assert StoryStage in chain

    def test_md_chain_ends_with_python_fallback(self):
        assert ConverterFactory.DEFAULT_CHAINS[".md"][-1] is StoryStage

    def test_html_chain_ends_with_html_fallback(self):
        assert ConverterFactory.DEFAULT_CHAINS[".html"][-1] is HtmlStoryStage

    def test_txt_conversion_works_without_external_binaries(self, tmp_path):
        """Der gesamte Converter faellt bis auf die Python-Stufe durch."""
        src = tmp_path / "notiz.txt"
        src.write_text("Ein Satz fuer die Vorschau.", encoding="utf-8")

        result = DocumentConverter().convert(src, tmp_path / "out")

        assert result.page_count >= 1
        assert result.pdf_path.is_file()
        assert result.converter  # Name der erfolgreichen Stufe
