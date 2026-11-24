"""
Tests fuer den Viewer-Rueckkanal (Task 11, Anforderung 6.3/6.7).

Klick auf eine Kodierung im PDF soll die Kodierung in der Pruef-Ansicht
selektieren; Seitenwechsel merken sich die Seite. getestet wird:
  - Component-Host-HTML existiert und enthaelt das Streamlit-Protokoll
  - Viewer-HTML sendet Ereignisse mit fortlaufender seq
  - render_pdf_viewer uebergibt Component-Values an on_event (Fallback zu
    components.html ohne Rueckkanal)
  - review_ui._handle_viewer_event setzt Seite/aktive Kodierung und
    uebergeht bereits verarbeitete Ereignisse
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent.parent
for path in (str(REPO_ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from QCA_AID_app.viewer import pdf_viewer_component as pvc  # noqa: E402
from QCA_AID_app.viewer.pdf_viewer_component import (  # noqa: E402
    HOST_DIR,
    ViewerConfig,
    build_viewer_html,
)


def _config(**overrides):
    base = dict(
        pdf_data_uri="data:application/pdf;base64,AAAA",
        doc_id="doc_1",
        overlays={1: "<svg/>"},
        initial_page=2,
        height=420,
        show_thumbnails=False,
    )
    base.update(overrides)
    return ViewerConfig(**base)


# --------------------------------------------------------------------------
# Bausteine
# --------------------------------------------------------------------------

def test_component_host_exists_with_streamlit_protocol():
    index = HOST_DIR / "index.html"

    assert index.is_file()
    source = index.read_text(encoding="utf-8")
    assert "streamlit:componentReady" in source
    assert "streamlit:render" in source
    assert "streamlit:setComponentValue" in source
    assert "qca-viewer" in source
    # Live-Update statt Neuaufbau: HTML-Vergleich, Overlay- und Seitenkommandos
    assert 'args.html !== lastHtml' in source
    assert "set-overlays" in source
    assert "goto-page" in source
    assert "lastHtml = args.html" in source


def test_viewer_html_emits_events_with_sequence_numbers():
    source = build_viewer_html(_config())

    assert "seq: ++SEQ" in source
    assert 'source: "qca-viewer"' in source


def test_viewer_accepts_live_overlay_and_page_commands():
    source = build_viewer_html(_config())

    assert "set-overlays" in source
    assert "goto-page" in source
    assert "let overlays" in source  # Overlays sind zur Laufzeit ersetzbar


def test_overlay_payload_can_stay_out_of_the_html():
    with_payload = build_viewer_html(_config(overlays={1: "<svg>dato</svg>"}))
    without_payload = build_viewer_html(_config(overlays={1: "<svg>dato</svg>"}), embed_overlays=False)

    assert "dato" in with_payload
    assert "dato" not in without_payload


def test_component_args_carry_page_and_overlays(monkeypatch):
    calls = []

    class FakeComponent:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return None

    monkeypatch.setattr(pvc, "_viewer_component", lambda: FakeComponent())

    pvc.render_pdf_viewer(_config(initial_page=4, overlays={2: "<svg/>"}), key="k")

    assert calls[0]["page"] == 4
    assert calls[0]["overlays"] == {2: "<svg/>"}


def test_initial_page_is_not_baked_into_component_html(monkeypatch):
    """Sonst wuerde jede Seitenbewegung den Viewer neu bauen (Springen)."""
    calls = []

    class FakeComponent:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return None

    monkeypatch.setattr(pvc, "_viewer_component", lambda: FakeComponent())

    pvc.render_pdf_viewer(_config(initial_page=4), key="k")

    assert 'value="__INITIAL_PAGE__"' not in calls[0]["html"]
    assert 'value="1"' in calls[0]["html"]


def test_component_declaration_uses_host_directory():
    assert (HOST_DIR / "index.html").is_file()


def test_component_is_packaged():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "viewer/_component_host/*.html" in pyproject


# --------------------------------------------------------------------------
# render_pdf_viewer
# --------------------------------------------------------------------------

def test_render_forwards_component_value_to_on_event(monkeypatch):
    calls = []

    class FakeComponent:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return {"source": "qca-viewer", "type": "coding-click", "codingId": "s1"}

    monkeypatch.setattr(pvc, "_viewer_component", lambda: FakeComponent())

    seen = []
    pvc.render_pdf_viewer(_config(), key="viewer_x", on_event=seen.append)

    assert seen == [{"source": "qca-viewer", "type": "coding-click", "codingId": "s1"}]
    assert calls[0]["key"] == "viewer_x"
    assert calls[0]["height"] == 420


def test_render_falls_back_without_component(monkeypatch):
    monkeypatch.setattr(pvc, "_viewer_component", lambda: None)
    monkeypatch.setattr(pvc, "_DECLARATION_FAILED", True)

    seen = []
    pvc.render_pdf_viewer(_config(), key="viewer_x", on_event=seen.append)

    assert seen == []  # kein Rueckkanal, kein Absturz


def test_render_without_key_has_no_return_channel(monkeypatch):
    # Fallback-Pfad: ohne key muss der klassische html-Renderer greifen.
    import streamlit.components.v1 as components

    captured = {}
    monkeypatch.setattr(
        components,
        "html",
        lambda body, **kwargs: captured.update(body=body, **kwargs),
    )

    pvc.render_pdf_viewer(_config(), key=None, on_event=lambda payload: None)

    assert "qca-viewer" in captured["body"]
    assert captured["height"] == 420


# --------------------------------------------------------------------------
# review_ui-Rueckkanal
# --------------------------------------------------------------------------

class FakeSession(dict):
    def __getattr__(self, name):  # noqa: D105
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class RerunSignalled(Exception):
    pass


@pytest.fixture()
def review_session(monkeypatch):
    from QCA_AID_app.webapp_components import review_ui

    session = FakeSession()
    monkeypatch.setattr(review_ui.st, "session_state", session, raising=False)
    monkeypatch.setattr(
        review_ui.st, "rerun", lambda: (_ for _ in ()).throw(RerunSignalled())
    )
    return review_ui, session


def _click_payload(seq=1, coding_id="s1", page=3):
    return {"source": "qca-viewer", "type": "coding-click", "seq": seq,
            "codingId": coding_id, "docId": "doc_1", "page": page}


def test_coding_click_selects_coding_and_reruns(review_session):
    review_ui, session = review_session

    with pytest.raises(RerunSignalled):
        review_ui._handle_viewer_event(_click_payload())

    assert session[review_ui.STATE_ACTIVE] == "s1"
    assert session[review_ui.STATE_PAGE] == 3
    assert session[review_ui.STATE_LAST_EVENT] == ("coding-click", "s1", 3, 1)


def test_repeated_event_is_ignored(review_session):
    review_ui, session = review_session

    with pytest.raises(RerunSignalled):
        review_ui._handle_viewer_event(_click_payload())
    session[review_ui.STATE_ACTIVE] = None  # Beweis: kein zweiter Rerun

    review_ui._handle_viewer_event(_click_payload())  # gleicher seq

    assert session[review_ui.STATE_ACTIVE] is None


def test_new_sequence_reruns_again(review_session):
    review_ui, session = review_session

    with pytest.raises(RerunSignalled):
        review_ui._handle_viewer_event(_click_payload(seq=1, coding_id="s1"))

    with pytest.raises(RerunSignalled):
        review_ui._handle_viewer_event(_click_payload(seq=2, coding_id="s1"))

    assert session[review_ui.STATE_LAST_EVENT] == ("coding-click", "s1", 3, 2)


def test_page_change_updates_page_and_reruns(review_session):
    review_ui, session = review_session

    with pytest.raises(RerunSignalled):
        review_ui._handle_viewer_event(
            {"source": "qca-viewer", "type": "page-change", "seq": 5, "page": 7, "docId": "d"}
        )

    assert session[review_ui.STATE_PAGE] == 7
    assert review_ui.STATE_ACTIVE not in session


def test_foreign_payload_is_ignored(review_session):
    review_ui, session = review_session

    review_ui._handle_viewer_event({"source": "other", "type": "page-change", "seq": 1})
    review_ui._handle_viewer_event("kein json")
    review_ui._handle_viewer_event(None)

    assert not session


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
