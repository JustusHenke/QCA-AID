"""
Tests fuer die Viewer-Bridge (Task 11).

Prueft Anforderungen 6.3, 6.7, 7.1 (Ereignis-Parsing, Zustandsuebergaenge,
App->Viewer-Kommandos).
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QCA_AID_app.viewer.viewer_bridge import (  # noqa: E402
    ViewerEvent,
    ViewerState,
    apply_event,
    apply_events,
    build_command,
    goto_page_command,
    VIEWER_SOURCE,
)


def payload(**kwargs):
    base = {"source": VIEWER_SOURCE}
    base.update(kwargs)
    return base


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_parse_dict_page_change():
    event = ViewerEvent.parse(payload(type="page-change", docId="d", page=4))
    assert event is not None
    assert event.type == "page-change"
    assert event.page == 4
    assert event.doc_id == "d"


def test_parse_json_string():
    raw = json.dumps(payload(type="coding-click", codingId="s1", page=2))
    event = ViewerEvent.parse(raw)
    assert event is not None
    assert event.coding_id == "s1"
    assert event.page == 2


def test_parse_page_coerced_from_float():
    event = ViewerEvent.parse(payload(type="page-change", page=3.0))
    assert event.page == 3
    assert isinstance(event.page, int)


def test_parse_rejects_foreign_source():
    assert ViewerEvent.parse({"source": "other", "type": "page-change"}) is None
    assert ViewerEvent.parse("nicht json") is None
    assert ViewerEvent.parse(42) is None


def test_parse_rejects_unknown_type():
    assert ViewerEvent.parse(payload(type="something-else")) is None


# --------------------------------------------------------------------------
# Zustandsuebergaenge
# --------------------------------------------------------------------------

def test_page_change_updates_state():
    state = ViewerState()
    state = apply_event(state, ViewerEvent(type="page-change", page=5))
    assert state.current_page == 5


def test_coding_click_sets_active_and_page():
    state = ViewerState()
    state = apply_event(state, ViewerEvent(type="coding-click", coding_id="s9", page=3))
    assert state.active_coding_id == "s9"
    assert state.current_page == 3


def test_search_hit_and_ready_and_error():
    state = ViewerState()
    state = apply_event(state, ViewerEvent(type="search-hit", page=2, term="druck"))
    assert state.last_search == {"page": 2, "term": "druck"}

    state = apply_event(state, ViewerEvent(type="ready"))
    assert state.ready is True

    state = apply_event(state, ViewerEvent(type="error", message="kaputt"))
    assert state.error == "kaputt"


def test_apply_events_ignores_foreign_payloads():
    state = apply_events(
        ViewerState(),
        [
            {"source": "other", "type": "page-change", "page": 9},
            payload(type="page-change", page=7),
        ],
    )
    assert state.current_page == 7


def test_state_roundtrip():
    state = ViewerState(current_page=3, active_coding_id="s1", ready=True)
    assert ViewerState.from_dict(state.to_dict()) == state


# --------------------------------------------------------------------------
# Kommandos
# --------------------------------------------------------------------------

def test_goto_page_command():
    js = goto_page_command(4)
    assert "qcaViewer" in js
    assert "gotoPage(4)" in js


def test_set_active_coding_command_is_json_escaped():
    js = build_command("set_active_coding", coding_id='weird"id')
    assert '\\"id' in js


def test_scroll_to_quad_command():
    js = build_command("scroll_to_quad", rect=[1, 2, 3, 4])
    assert "scrollToQuad([1, 2, 3, 4])" in js


def test_unknown_command_raises():
    with pytest.raises(ValueError):
        build_command("explode")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))