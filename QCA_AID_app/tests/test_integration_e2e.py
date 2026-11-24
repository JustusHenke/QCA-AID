"""
Gesamtintegration und Golden-Files (Task 15).

Deckt die komplette Kette ab (Anforderungen 4.1, 5.7, 6.7, 9.5):
  Ingest (DocumentStore) -> Cache/Ablage -> DocumentIndex -> PageAwareChunks
  -> AnchorResolver (Quads) -> OverlayRenderer -> PdfBakingExporter -> Datenexport

Die Konvertierung nutzt einen deterministischen Test-Konverter (PyMuPDF), damit
die Integration unabhaengig von LibreOffice/WeasyPrint/Pandoc pruefbar ist.
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fitz = pytest.importorskip("fitz")

from QCA_AID_assets.annotation.anchor import AnchorResolver  # noqa: E402
from QCA_AID_assets.annotation.export_data import build_rows  # noqa: E402
from QCA_AID_assets.annotation.export_pdf import PdfBakingExporter  # noqa: E402
from QCA_AID_assets.annotation.model import Coding, RenderOptions  # noqa: E402
from QCA_AID_assets.annotation.renderer import DEFAULT_PALETTE, OverlayRenderer  # noqa: E402
from QCA_AID_assets.preprocessing.document_converter import (  # noqa: E402
    ConvertResult,
    pdf_page_count,
)
from QCA_AID_assets.preprocessing.document_index import DocumentIndexer  # noqa: E402
from QCA_AID_assets.preprocessing.document_store import DocumentStore  # noqa: E402
from QCA_AID_assets.preprocessing.page_aware_chunker import PageAwareChunker  # noqa: E402
from QCA_AID_app.viewer.pdf_viewer_component import (  # noqa: E402
    ViewerConfig,
    build_viewer_html,
)

SOURCE_LINES = [
    "Die Befragten unter Druck geraten und mussten priorisieren.",
    "Zweiter Abschnitt beschreibt die Folgen der Belastung.",
    "Dritter Abschnitt fasst die Empfehlungen zusammen.",
]


class FixedTextConverter:
    """Deterministischer Konverter: schreibt SOURCE_LINES als PDF-Seiten."""

    def __init__(self):
        self.calls = 0

    def convert(self, src, out_dir, canonical_name="canonical.pdf"):
        self.calls += 1
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf = out_dir / canonical_name

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        y = 100
        for line in SOURCE_LINES:
            page.insert_text((72, y), line, fontsize=11)
            y += 16
        doc.save(pdf)
        doc.close()

        return ConvertResult(
            pdf_path=pdf, page_count=pdf_page_count(pdf), page_labels=None, converter="fixed"
        )


def _source_file(tmp_path: Path, suffix: str) -> Path:
    path = tmp_path / f"studie{suffix}"
    if suffix == ".docx":
        path.write_bytes(b"PK\x03\x04 dummy docx")
    elif suffix == ".html":
        path.write_text("<html><body>Studie</body></html>", encoding="utf-8")
    else:
        path.write_text("\n".join(SOURCE_LINES), encoding="utf-8")
    return path


def _run_pipeline(tmp_path, suffix):
    store = DocumentStore(tmp_path / "proj", converter=FixedTextConverter())
    src = _source_file(tmp_path, suffix)
    ingest = store.ingest(src, doc_id="studie")
    assert ingest.ok, ingest.reasons

    index = DocumentIndexer().build(ingest.pdf_path, doc_id="studie")
    store.write_index("studie", index.to_dict())
    chunks = PageAwareChunker(chunk_size=120, overlap=0).chunk_document(index)
    anchor = AnchorResolver().resolve(
        ingest.pdf_path, doc_id="studie", page=1, quote="unter Druck geraten"
    )
    return store, ingest, index, chunks, anchor


# --------------------------------------------------------------------------
# End-to-End
# --------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".txt", ".docx", ".html"])
def test_end_to_end_pipeline_all_formats(tmp_path, suffix):
    store, ingest, index, chunks, anchor = _run_pipeline(tmp_path, suffix)

    assert ingest.status == "converted"
    assert index.page_count >= 1
    assert index.validate()[0]
    assert (ingest.doc_dir / "document.json").exists()

    assert chunks and chunks[0].page_start == 1
    assert anchor.anchored and anchor.quads

    coding = Coding(
        segment_id="studie_chunk_0", doc_id="studie", category="Problemdruck",
        anchor=anchor, confidence=0.9,
    )
    ok, errors = coding.validate()
    assert ok, errors

    # Overlay
    svg = OverlayRenderer().build_page_svg(1, [anchor], [coding], RenderOptions())
    assert 'data-coding-id="studie_chunk_0"' in svg
    assert 'tabindex="0"' in svg and 'role="button"' in svg

    # PDF-Export
    out_pdf = PdfBakingExporter().bake(
        ingest.pdf_path, [coding], tmp_path / "annotiert.pdf",
        legend=False, margin_notes=False,
    )
    assert out_pdf.exists()
    doc = fitz.open(out_pdf)
    highlights = [a for a in doc[0].annots() if a.type[0] == fitz.PDF_ANNOT_HIGHLIGHT]
    assert highlights

    # Datenexport
    rows = build_rows([coding])
    assert rows[0]["verankert"] == "anchored"
    assert rows[0]["seite"] == 1


def test_second_ingest_is_served_from_cache(tmp_path):
    converter = FixedTextConverter()
    store = DocumentStore(tmp_path / "proj", converter=converter)
    src = _source_file(tmp_path, ".txt")

    first = store.ingest(src, "studie")
    second = store.ingest(src, "studie")

    assert first.status == "converted"
    assert second.status == "cached"
    assert converter.calls == 1


# --------------------------------------------------------------------------
# Golden File: Quad entspricht den Wort-Boxen (±2 pt)
# --------------------------------------------------------------------------

def test_golden_quads_match_word_boxes(tmp_path):
    _, ingest, _, _, anchor = _run_pipeline(tmp_path, ".txt")

    doc = fitz.open(ingest.pdf_path)
    words = doc[0].get_text("words")
    tokens = [re.sub(r"\W+", "", w[4]).lower() for w in words]
    target = ["unter", "druck", "geraten"]

    start = next(
        i for i in range(len(tokens) - len(target) + 1)
        if tokens[i:i + len(target)] == target
    )
    run = words[start:start + len(target)]
    expected = (
        min(w[0] for w in run), min(w[1] for w in run),
        max(w[2] for w in run), max(w[3] for w in run),
    )

    assert len(anchor.quads) == 1
    for got, exp in zip(anchor.quads[0], expected):
        assert abs(got - exp) <= 2.0, (anchor.quads[0], expected)


# --------------------------------------------------------------------------
# Anker-Invarianz
# --------------------------------------------------------------------------

def test_anchor_is_deterministic(tmp_path):
    _, ingest, _, _, anchor = _run_pipeline(tmp_path, ".txt")
    resolver = AnchorResolver()

    again = resolver.resolve(ingest.pdf_path, doc_id="studie", page=1,
                             quote="unter Druck geraten")
    assert again.quads == anchor.quads

    reopened = DocumentIndexer().build(ingest.pdf_path, doc_id="studie")
    assert reopened.page_count == 1


# --------------------------------------------------------------------------
# Barrierefreiheit
# --------------------------------------------------------------------------

def _relative_luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    channels = []
    for i in (0, 2, 4):
        c = int(value[i:i + 2], 16) / 255.0
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_palette_has_sufficient_contrast_with_black_text():
    for color in DEFAULT_PALETTE:
        luminance = _relative_luminance(color)
        contrast = (luminance + 0.05) / 0.05  # gegen schwarzen Text
        assert contrast >= 4.5, f"{color}: Kontrast {contrast:.1f}"


def test_viewer_has_focus_and_motion_styles():
    html = build_viewer_html(ViewerConfig(pdf_data_uri="data:application/pdf;base64,AA=="))
    assert ":focus-visible" in html
    assert "prefers-reduced-motion" in html


# --------------------------------------------------------------------------
# Validierung (9.5)
# --------------------------------------------------------------------------

def test_validators_flag_corrupted_anchor(tmp_path):
    _, _, index, _, anchor = _run_pipeline(tmp_path, ".txt")
    assert index.validate()[0]

    broken = Coding(
        segment_id="s", doc_id="studie", category="X",
        anchor=type(anchor)(doc_id="studie", page=1, quote="x", quads=[], anchored=True),
    )
    ok, errors = broken.validate()
    assert not ok
    assert any("Quad" in e for e in errors)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))