"""
Tests fuer die TextViewer-Komponente (Task 10).

Prueft den HTML-Generator headless (Anforderungen 3.1-3.7, 6.6):
Toolbar, Seitenlabels, Overlay-Einbettung, Virtualisierung, JS-API, Bruecke.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QCA_AID_app.viewer.pdf_viewer_component import (  # noqa: E402
    ViewerConfig,
    build_viewer_html,
    pdf_to_data_uri,
    render_pdf_viewer,
)

DATA_URI = "data:application/pdf;base64,JVBERi0xLjQK"


def _config(**kwargs):
    base = dict(
        pdf_data_uri=DATA_URI,
        doc_id="doc-1",
        page_labels=["i", "1", "2"],
        overlays={2: '<svg class="qca-overlay"><rect data-coding-id="s9"/></svg>'},
        initial_page=2,
        height=600,
    )
    base.update(kwargs)
    return ViewerConfig(**base)


# --------------------------------------------------------------------------
# Grundgeruest
# --------------------------------------------------------------------------

def test_no_unresolved_placeholders():
    html = build_viewer_html(_config())
    assert "__LIB_URL__" not in html
    assert "__PDF_DATA__" not in html
    assert "__INITIAL_PAGE__" not in html


def test_pdfjs_urls_and_worker():
    html = build_viewer_html(_config())
    assert "pdf.min.mjs" in html
    assert "pdf.worker.min.mjs" in html
    assert "GlobalWorkerOptions.workerSrc" in html


def test_pdf_payload_and_doc_id():
    html = build_viewer_html(_config())
    assert DATA_URI in html
    assert 'data-doc-id="doc-1"' in html


def test_initial_page_and_height():
    html = build_viewer_html(_config(initial_page=3, height=512))
    assert 'value="3"' in html
    assert 'parseInt("3", 10)' in html
    assert "height: 512px" in html


# --------------------------------------------------------------------------
# Seitenlabels
# --------------------------------------------------------------------------

def test_page_label_options_rendered():
    html = build_viewer_html(_config())
    assert '<option value="1">i</option>' in html
    assert '<option value="2">1</option>' in html


def test_labels_are_html_escaped():
    html = build_viewer_html(_config(page_labels=['<b>"x"</b>']))
    assert "&lt;b&gt;" in html
    assert "<b>" not in html


# --------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------

def test_overlay_svg_embedded():
    html = build_viewer_html(_config())
    assert 'id="qca-overlays"' in html
    assert 'data-coding-id=\\"s9\\"' in html or "data-coding-id" in html
    assert "attachOverlay" in html


def test_overlay_json_escapes_closing_script():
    html = build_viewer_html(_config(overlays={1: "</script><svg/>"}))
    assert "</script><svg/>" not in html


# --------------------------------------------------------------------------
# Optionen
# --------------------------------------------------------------------------

def test_thumbnails_can_be_disabled():
    html = build_viewer_html(_config(show_thumbnails=False))
    assert 'const SHOW_THUMBS = "0"' in html


def test_search_can_be_disabled():
    html = build_viewer_html(_config(enable_search=False))
    assert 'id="qca-search"' not in html


def test_search_present_by_default():
    html = build_viewer_html(_config())
    assert 'id="qca-search"' in html
    assert "async function search" in html


# --------------------------------------------------------------------------
# Virtualisierung & API & Bruecke
# --------------------------------------------------------------------------

def test_virtualization_present():
    html = build_viewer_html(_config())
    assert "IntersectionObserver" in html
    assert "renderPage" in html


def test_global_api_exposed():
    html = build_viewer_html(_config())
    assert "window.qcaViewer" in html
    for name in ("gotoPage", "setActiveCoding", "scrollToQuad", "search"):
        assert name in html


def test_bridge_uses_postmessage():
    html = build_viewer_html(_config())
    assert "window.parent.postMessage" in html
    assert 'source: "qca-viewer"' in html
    assert '"page-change"' in html


def test_accessibility_controls():
    html = build_viewer_html(_config())
    assert 'aria-label="Vorherige Seite"' in html
    assert 'role="toolbar"' in html
    assert "prefers-reduced-motion" in html


# --------------------------------------------------------------------------
# Helper & Wrapper
# --------------------------------------------------------------------------

def test_pdf_to_data_uri(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    uri = pdf_to_data_uri(pdf)
    assert uri.startswith("data:application/pdf;base64,")


def test_render_pdf_viewer_delegates_to_streamlit(monkeypatch):
    import streamlit.components.v1 as components

    captured = {}

    def fake_html(html, **kwargs):
        captured["html"] = html
        captured["kwargs"] = kwargs

    monkeypatch.setattr(components, "html", fake_html)

    render_pdf_viewer(_config(height=400))

    assert "window.qcaViewer" in captured["html"]
    assert captured["kwargs"]["height"] == 400


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))