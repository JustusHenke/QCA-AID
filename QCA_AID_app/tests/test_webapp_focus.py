"""
Unit-Tests fuer Fokus-Flags (ux-usability-refactoring, Task 4.1).

st.session_state funktioniert auch ohne Streamlit-Runtime (Bare Mode) -
pro Testsitz wird vor jedem Test zurueckgesetzt.

Requirements: 11.2, 11.3, 12.3
"""

import sys
from pathlib import Path

import pytest
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from webapp_components.focus import (  # noqa: E402
    FOCUS_KEY,
    SECTION_FORSCHUNGSFRAGE,
    SECTION_KATEGORIEN,
    SECTION_MODEL,
    auto_expand,
    clear_focus,
    consume_focus,
    pop_focus,
    set_focus,
)


@pytest.fixture(autouse=True)
def _reset_session_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


class TestSetAndPopFocus:
    def test_set_and_pop_matching_tab(self):
        set_focus("codebook", SECTION_KATEGORIEN)
        assert pop_focus("codebook") == SECTION_KATEGORIEN
        # Flag geloescht
        assert FOCUS_KEY not in st.session_state
        assert pop_focus("codebook") is None

    def test_pop_wrong_tab_preserves_flag(self):
        set_focus("codebook", SECTION_KATEGORIEN)
        assert pop_focus("config") is None
        # Flag bleibt erhalten - Ziel-Tab kann es spaeter lesen
        assert pop_focus("codebook") == SECTION_KATEGORIEN

    def test_pop_missing_flag_is_none(self):
        assert pop_focus("config") is None


class TestConsumeFocus:
    def test_only_matching_section_consumes(self):
        set_focus("codebook", SECTION_KATEGORIEN)
        assert consume_focus("codebook", SECTION_FORSCHUNGSFRAGE) is False
        # Noch da - falsche Section hat nicht geloescht
        assert consume_focus("codebook", SECTION_KATEGORIEN) is True
        # Nach Match geloescht
        assert consume_focus("codebook", SECTION_KATEGORIEN) is False

    def test_consume_wrong_tab_preserves(self):
        set_focus("codebook", SECTION_KATEGORIEN)
        assert consume_focus("config", SECTION_KATEGORIEN) is False
        assert consume_focus("codebook", SECTION_KATEGORIEN) is True


class TestAutoExpand:
    def test_auto_expand_mirrors_consume(self):
        set_focus("config", SECTION_MODEL)
        assert auto_expand("config", SECTION_MODEL) is True
        assert auto_expand("config", SECTION_MODEL) is False


class TestClearFocus:
    def test_clear_removes_flag(self):
        set_focus("codebook", SECTION_KATEGORIEN)
        clear_focus()
        assert FOCUS_KEY not in st.session_state
        assert pop_focus("codebook") is None

    def test_clear_without_flag_is_safe(self):
        clear_focus()  # kein Fehler
