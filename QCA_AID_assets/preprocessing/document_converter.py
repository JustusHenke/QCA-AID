"""
Document Converter
==================
Konvertiert Eingabedokumente in ein kanonisches PDF fuer Vorschau und
Annotation (siehe .kiro/specs/pdf-preview-annotation, Anforderungen 1 & 9).

Aufbau:
  - ConversionStage: isolierte Konvertierungsstufe (eine Bibliothek/ein Tool)
  - DocumentConverter: faehrt je Endung eine geordnete Fallback-Kette ab
  - ConverterFactory: Registry Endung -> Stufenliste

Fallback-Trigger (Anforderung 9.2): fehlendes Binary (available() False),
Timeout, Exit-Code != 0, leeres/ungueltiges Ausgabe-PDF. Jede Stufe laeuft
isoliert; Fehler werden gesammelt und die naechste Stufe versucht.
"""

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_TIMEOUT = 120.0


class ConvertError(Exception):
    """Wird geworfen, wenn die gesamte Fallback-Kette einer Datei scheitert."""

    def __init__(self, src: Path, reasons: List[str]):
        self.src = src
        self.reasons = reasons
        super().__init__(
            f"Konvertierung von {src.name} fehlgeschlagen: " + " | ".join(reasons)
        )


class StageError(Exception):
    """Signalisiert innerhalb einer Stufe, dass die naechste Stufe greifen soll."""


@dataclass
class ConvertResult:
    """Ergebnis einer erfolgreichen Konvertierung."""

    pdf_path: Path
    page_count: int
    page_labels: Optional[List[str]]
    converter: str
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# PDF-Hilfsfunktionen
# --------------------------------------------------------------------------

def is_valid_pdf(path: Path) -> bool:
    """Prueft, ob eine Datei existiert, nicht leer ist und mit %PDF beginnt."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with open(path, "rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


def pdf_page_count(path: Path) -> int:
    """Ermittelt die Seitenzahl best-effort (PyMuPDF, dann pypdf)."""
    try:
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        pass
    try:
        import pypdf

        with open(path, "rb") as fh:
            return len(pypdf.PdfReader(fh).pages)
    except Exception:
        return 0


def pdf_page_labels(path: Path) -> Optional[List[str]]:
    """Ermittelt gedruckte Seitenlabels best-effort via PyMuPDF."""
    try:
        import fitz

        with fitz.open(path) as doc:
            labels = doc.get_page_labels() if hasattr(doc, "get_page_labels") else None
            if labels:
                return [str(label) for label in labels]
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# Konvertierungsstufen
# --------------------------------------------------------------------------

class ConversionStage:
    """Basisklasse einer Konvertierungsstufe."""

    name = "stage"

    def available(self) -> bool:
        """Gibt an, ob die Stufe auf diesem System einsatzbereit ist."""
        return True

    def run(self, src: Path, out_pdf: Path, timeout: float) -> None:
        """Erzeugt out_pdf aus src. Hebt StageError bei Fehlschlag."""
        raise NotImplementedError


def _run_subprocess(cmd: Sequence[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            list(cmd), timeout=timeout, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired as exc:
        raise StageError(f"Timeout nach {timeout}s") from exc
    except FileNotFoundError as exc:
        raise StageError("Binary nicht gefunden") from exc
    except OSError as exc:
        raise StageError(f"Prozessfehler: {exc}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        tail = stderr[-1] if stderr else f"Exit {result.returncode}"
        raise StageError(f"Exit {result.returncode}: {tail}")
    return result


class PassthroughStage(ConversionStage):
    """Uebernimmt ein bereits vorhandenes PDF unveraendert."""

    name = "passthrough"

    def run(self, src: Path, out_pdf: Path, timeout: float) -> None:
        if not is_valid_pdf(src):
            raise StageError("Quelle ist kein gueltiges PDF")
        shutil.copyfile(src, out_pdf)


class LibreOfficeStage(ConversionStage):
    """Konvertiert Office-Formate via LibreOffice Headless."""

    name = "libreoffice"

    def _binary(self) -> Optional[str]:
        return shutil.which("soffice") or shutil.which("libreoffice")

    def available(self) -> bool:
        return self._binary() is not None

    def run(self, src: Path, out_pdf: Path, timeout: float) -> None:
        binary = self._binary()
        if binary is None:
            raise StageError("LibreOffice-Binary fehlt")
        with tempfile.TemporaryDirectory() as tmp:
            _run_subprocess(
                [binary, "--headless", "--convert-to", "pdf", "--outdir", tmp, str(src)],
                timeout,
            )
            produced = Path(tmp) / (src.stem + ".pdf")
            if not is_valid_pdf(produced):
                raise StageError("LibreOffice lieferte kein gueltiges PDF")
            shutil.copyfile(produced, out_pdf)


class WeasyPrintStage(ConversionStage):
    """Rendert HTML/Markdown/Text via WeasyPrint zu PDF."""

    name = "weasyprint"

    def available(self) -> bool:
        try:
            import weasyprint  # noqa: F401

            return True
        except Exception:
            return False

    def run(self, src: Path, out_pdf: Path, timeout: float) -> None:
        try:
            from weasyprint import HTML
        except Exception as exc:  # pragma: no cover - available() faengt das ab
            raise StageError("weasyprint nicht importierbar") from exc

        suffix = src.suffix.lower()
        if suffix == ".html":
            payload = src.read_text(encoding="utf-8", errors="replace")
        else:
            # Markdown/Text: als vorformatierten Block einbetten.
            text = src.read_text(encoding="utf-8", errors="replace")
            payload = f"<html><body><pre>{_escape_html(text)}</pre></body></html>"

        try:
            HTML(string=payload).write_pdf(str(out_pdf))
        except Exception as exc:
            raise StageError(f"WeasyPrint-Renderfehler: {exc}") from exc
        if not is_valid_pdf(out_pdf):
            raise StageError("WeasyPrint lieferte kein gueltiges PDF")


class PandocStage(ConversionStage):
    """Konvertiert Markdown/HTML/Text via pandoc zu PDF."""

    name = "pandoc"

    def available(self) -> bool:
        return shutil.which("pandoc") is not None

    def run(self, src: Path, out_pdf: Path, timeout: float) -> None:
        binary = shutil.which("pandoc")
        if binary is None:
            raise StageError("pandoc-Binary fehlt")
        _run_subprocess([binary, str(src), "-o", str(out_pdf)], timeout)
        if not is_valid_pdf(out_pdf):
            raise StageError("pandoc lieferte kein gueltiges PDF")


class DoclingStage(ConversionStage):
    """Fallback fuer DOCX via docling (Konvertierung nach HTML + WeasyPrint)."""

    name = "docling"

    def available(self) -> bool:
        try:
            import docling  # noqa: F401
            import weasyprint  # noqa: F401

            return True
        except Exception:
            return False

    def run(self, src: Path, out_pdf: Path, timeout: float) -> None:
        try:
            from docling.document_converter import DocumentConverter as _Docling
            from weasyprint import HTML
        except Exception as exc:  # pragma: no cover
            raise StageError("docling nicht importierbar") from exc
        try:
            result = _Docling().convert(str(src))
            html = result.document.export_to_html()
        except Exception as exc:
            raise StageError(f"docling-Konvertierung fehlgeschlagen: {exc}") from exc
        try:
            HTML(string=html).write_pdf(str(out_pdf))
        except Exception as exc:
            raise StageError(f"WeasyPrint-Renderfehler: {exc}") from exc
        if not is_valid_pdf(out_pdf):
            raise StageError("docling/WeasyPrint lieferte kein gueltiges PDF")


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


class StoryStage(ConversionStage):
    """Reiner Python-Fallback ohne externe Binaries (PyMuPDF Story).

    Deckt Text, Markdown (als Klartext) und HTML-Grundstruktur ab, damit auch
    ohne LibreOffice/Pandoc eine Vorschau mit Textlayer entsteht. Bewusst
    letzte Stufe: die externen Konverter liefern bessere Typografie.
    """

    name = "pymupdf-story"
    MODE_TEXT = "text"
    MODE_HTML = "html"

    def __init__(self, mode: str = MODE_TEXT):
        self.mode = mode

    def available(self) -> bool:
        try:
            import fitz  # noqa: F401
        except ImportError:
            return False
        return hasattr(fitz, "Story") and hasattr(fitz, "DocumentWriter")

    def run(self, src: Path, out_pdf: Path, timeout: float) -> None:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover
            raise StageError("PyMuPDF nicht verfuegbar") from exc

        raw = _read_text(src)
        if not raw.strip():
            raise StageError("Datei enthaelt keinen Text")

        if self.mode == self.MODE_HTML:
            body = raw
        else:
            paragraphs = [
                f"<p>{_escape_html(block.strip())}</p>"
                for block in re.split(r"\n\s*\n", raw)
                if block.strip()
            ]
            body = "\n".join(paragraphs)

        _render_html_to_pdf(body, out_pdf)


def _render_html_to_pdf(body: str, out_pdf: Path) -> None:
    """Rendert HTML-Körper via PyMuPDF Story seitenweise in ein PDF."""
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise StageError("PyMuPDF nicht verfuegbar") from exc

    try:
        story = fitz.Story(html=f"<body>{body}</body>", user_css=_STORY_CSS)
        writer = fitz.DocumentWriter(str(out_pdf))
        page = fitz.paper_rect("a4")  # A4 in Punkten
        margin = 56.0
        content = fitz.Rect(margin, margin, page.width - margin, page.height - margin)
        while True:
            device = writer.begin_page(page)
            more, _ = story.place(content)
            story.draw(device)
            writer.end_page()
            if not more:
                break
        writer.close()
    except Exception as exc:
        raise StageError(f"PyMuPDF-Renderfehler: {exc}") from exc

    if not is_valid_pdf(out_pdf):
        raise StageError("PyMuPDF lieferte kein gueltiges PDF")


class DocxTextStage(ConversionStage):
    """Fallback fuer DOCX ohne LibreOffice: Text via python-docx + Story.

    Absaetze und Tabellenzellen werden uebernommen, Ueberschriften als
    h1/h2/h3 - genug Struktur fuer Vorschau und Textanker. Layout-Treue
    (Spalten, Grafiken) liefert nur LibreOffice.
    """

    name = "python-docx"

    def available(self) -> bool:
        try:
            import docx  # noqa: F401
            import fitz  # noqa: F401
        except ImportError:
            return False
        return True

    def run(self, src: Path, out_pdf: Path, timeout: float) -> None:
        try:
            from docx import Document as _DocxDocument
        except ImportError as exc:  # pragma: no cover
            raise StageError("python-docx nicht verfuegbar") from exc
        try:
            document = _DocxDocument(str(src))
        except Exception as exc:
            raise StageError(f"DOCX nicht lesbar: {exc}") from exc

        parts: List[str] = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            level = _heading_level(paragraph)
            if level:
                parts.append(f"<h{level}>{_escape_html(text)}</h{level}>")
            else:
                parts.append(f"<p>{_escape_html(text)}</p>")

        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(f"<p>{_escape_html(' | '.join(cells))}</p>")

        if not parts:
            raise StageError("Dokument enthaelt keinen Text")

        _render_html_to_pdf("\n".join(parts), out_pdf)


def _heading_level(paragraph: Any) -> int:
    """Liest die Ueberschriftenebene aus dem Absatzstil (0 = keine)."""
    style = getattr(paragraph, "style", None)
    name = str(getattr(style, "name", "") or "")
    if not name.lower().startswith("heading"):
        return 0
    digits = "".join(char for char in name if char.isdigit())
    level = int(digits) if digits else 1
    return min(max(level, 1), 3)


_STORY_CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.4; }
p { margin: 0 0 6pt 0; }
h1, h2, h3 { font-weight: bold; margin: 10pt 0 4pt 0; }
"""


def _read_text(src: Path) -> str:
    """Liest Textdateien tolerant (UTF-8, dann Windows-1252, dann Latin-1)."""
    data = Path(src).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class HtmlStoryStage(StoryStage):
    """HTML-Fallback: rendert das HTML-Dokument direkt mit PyMuPDF Story."""

    name = "pymupdf-html"

    def __init__(self):
        super().__init__(mode=StoryStage.MODE_HTML)


# --------------------------------------------------------------------------
# Factory und Converter
# --------------------------------------------------------------------------

class ConverterFactory:
    """Registry Endung -> geordnete Liste von Stage-Klassen."""

    DEFAULT_CHAINS: Dict[str, List[type]] = {
        ".pdf": [PassthroughStage],
        ".docx": [LibreOfficeStage, DoclingStage, DocxTextStage],
        ".doc": [LibreOfficeStage, DoclingStage],
        ".html": [WeasyPrintStage, PandocStage, HtmlStoryStage],
        ".htm": [WeasyPrintStage, PandocStage, HtmlStoryStage],
        ".md": [PandocStage, WeasyPrintStage, StoryStage],
        ".txt": [WeasyPrintStage, PandocStage, StoryStage],
    }

    def __init__(self, chains: Optional[Dict[str, List[type]]] = None):
        self.chains = chains if chains is not None else dict(self.DEFAULT_CHAINS)

    def stages_for(self, src: Path) -> List[ConversionStage]:
        """Erzeugt die Stufenobjekte fuer die Endung von src."""
        stage_classes = self.chains.get(src.suffix.lower(), [])
        return [cls() for cls in stage_classes]


class DocumentConverter:
    """Faehrt je Dokument die Fallback-Kette ab und liefert ein Canonical_PDF."""

    def __init__(
        self,
        factory: Optional[ConverterFactory] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.factory = factory or ConverterFactory()
        self.timeout = timeout

    def convert(self, src: Path, out_dir: Path, canonical_name: str = "canonical.pdf") -> ConvertResult:
        """Konvertiert src nach out_dir/canonical_name.

        Args:
            src: Quelldatei
            out_dir: Zielverzeichnis
            canonical_name: Name des erzeugten PDFs

        Returns:
            ConvertResult

        Raises:
            ConvertError: wenn keine Stufe erfolgreich war
        """
        src = Path(src)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = out_dir / canonical_name

        stages = self.factory.stages_for(src)
        if not stages:
            raise ConvertError(src, [f"kein Konverter fuer Endung '{src.suffix}'"])

        reasons: List[str] = []
        warnings: List[str] = []
        for stage in stages:
            if not stage.available():
                reasons.append(f"{stage.name}: nicht verfuegbar")
                continue
            try:
                stage.run(src, out_pdf, self.timeout)
            except StageError as exc:
                reasons.append(f"{stage.name}: {exc}")
                continue
            except Exception as exc:  # unerwartete Fehler isolieren
                reasons.append(f"{stage.name}: unerwarteter Fehler: {exc}")
                continue

            if not is_valid_pdf(out_pdf):
                reasons.append(f"{stage.name}: leeres/ungueltiges PDF")
                continue

            return ConvertResult(
                pdf_path=out_pdf,
                page_count=pdf_page_count(out_pdf),
                page_labels=pdf_page_labels(out_pdf),
                converter=stage.name,
                warnings=warnings,
            )

        raise ConvertError(src, reasons)


__all__ = [
    "DocumentConverter",
    "ConverterFactory",
    "ConvertResult",
    "ConvertError",
    "StageError",
    "ConversionStage",
    "PassthroughStage",
    "LibreOfficeStage",
    "WeasyPrintStage",
    "PandocStage",
    "DoclingStage",
    "is_valid_pdf",
    "pdf_page_count",
    "pdf_page_labels",
]
