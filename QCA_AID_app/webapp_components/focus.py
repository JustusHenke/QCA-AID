"""
Focus
=====
Fokus-Flags fuer zielgerichtete Tab-Uebergaenge (Jump-Links).

Ein einmaliges Flag ``ux_focus = {"tab": ..., "section": ...}`` im Session
State ersetzt fehlende Scroll-Anker in Streamlit: ``set_focus`` wird vom
Ausloeser gerufen (z. B. Readiness-"Beheben"-Button, Dirty-Leiste-Verwerfen),
die Ziel-Section konsumiert das Flag (Info-Box und/oder Expander aufklappen).

Semantik (design.md):
- Das Flag wird IMMER nach dem Lesen geloescht (einmalig pro Rerun).
- ``consume_focus`` loescht NUR bei Match - mehrere Sections eines Tabs
  koennen gefahrlos fragen, ohne sich gegenseitig das Flag wegzunehmen.

Requirements (ux-usability-refactoring): 11.2, 11.3, 12.3
"""

from __future__ import annotations

import streamlit as st

FOCUS_KEY = "ux_focus"


def set_focus(tab: str, section: str) -> None:
    """Setzt das Fokus-Flag auf eine Section in einem Tab (ausloesende Seite)."""
    st.session_state[FOCUS_KEY] = {"tab": tab, "section": section}


def pop_focus(tab: str) -> str | None:
    """Liest und loescht das Fokus-Flag, falls es fuer ``tab`` gilt.

    Returns:
        section-Key oder None.
    """
    data = st.session_state.get(FOCUS_KEY)
    if data and data.get("tab") == tab:
        del st.session_state[FOCUS_KEY]
        return data.get("section")
    return None


def consume_focus(tab: str, section: str) -> bool:
    """True, wenn DIESER Tab + DIESE Section fokussiert ist; loescht dann das Flag.

    Sections rufen das am eigenen Einstieg - mehrere Sections eines Tabs
    koennen gefahrlos nacheinander fragen, da nur der Match loescht.
    """
    data = st.session_state.get(FOCUS_KEY)
    if data and data.get("tab") == tab and data.get("section") == section:
        del st.session_state[FOCUS_KEY]
        return True
    return False


def auto_expand(tab: str, section: str) -> bool:
    """Convenience fuer ``st.expander(..., expanded=auto_expand(tab, section))`` (R12.3)."""
    return consume_focus(tab, section)


def render_focus_banner(
    section_label: str,
    message: str | None = None,
    section_key: str | None = None,
) -> bool:
    """Zeigt eine Info-Box, wenn DIESE Section fokussiert ist (R11.3).

    Args:
        section_label: Anzeigename der Section (z. B. "Forschungsfrage")
        message: Optionale ergaenzende Info; Standard erklaert den Sprung.

    Returns:
        True, wenn fokussiert ( Banner wurde gerendert).
    """
    key = section_key or _label_to_section_key(section_label)
    if not consume_focus(_current_tab(), key):
        return False
    st.info(
        message
        or f"👉 **{section_label}**: Hier können Sie das gemeldete Problem beheben."
    )
    return True


def clear_focus() -> None:
    """Entfernt das Flag (z. B. nach Verwerfen von Aenderungen)."""
    if FOCUS_KEY in st.session_state:
        del st.session_state[FOCUS_KEY]


def _current_tab() -> str:
    """Aktiver Tab-Key aus dem Session State (Kompatibilitaet zu webapp.py)."""
    return st.session_state.get("active_tab", "")


def _label_to_section_key(label: str) -> str:
    """Abschnitts-Keys sind lowercase ohne Umlaute-Sonderbehandlung - wir
    erlauben auch den Aufruf mit dem Anzeigenamen."""
    return label.lower().strip()


# Bekannte Section-Keys (fuer set_focus-Aufrufer als Konstanten wiederverwendbar)
SECTION_FORSCHUNGSFRAGE = "forschungsfrage"
SECTION_KODIERREGELN = "kodierregeln"
SECTION_KATEGORIEN = "kategorien"
SECTION_MODEL = "model"
SECTION_CHUNKS = "chunks"
SECTION_CODER = "coder"
SECTION_ATTRIBUTE_LABELS = "attribute_labels"
SECTION_INPUT_FILES = "input_files"
