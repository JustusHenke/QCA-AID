"""
Pagination
==========
Gemeinsame Paginierungs-Komponente fuer Dateilisten (R13).

ABWEICHUNG zum Designtext (dokumentiert, analog Task 4.1): R13.1 verlangt
``st.pagination`` als native Streamlit-Komponente. Diese existiert in
Streamlit 1.51.0 NICHT (``hasattr(st, "pagination") is False``). Statt des
nicht verfuegbaren nativen Elements ersetzt diese gemeinsame Komponente die
bisher drei duplizierten Chevron-Nav-Implementierungen (:material/chevron_left:/:material/chevron_right:) in analysis_ui (2x) und
explorer_ui (1x).

Verhalten:
- Seitengroesse 10/25/50 als Select; Auswahl wird im Session-State
  ``files_per_page`` persistiert (R13.2, geteilt ueber alle Listen).
- Seitennavigation :material/chevron_left:/:material/chevron_right: mit Clamping (Seite bleibt gueltig, wenn sich
  Seitengroesse oder Listenlaenge aendern).
- Die Pro-Datei-Aktionen (Vorschau, Ordner oeffnen) bleiben unveraendert
  (R13.3) - die Aufrufer paginieren weiter mit ``file_manager.paginate_files``.

Requirements (ux-usability-refactoring): 13.1 (Abweichung), 13.2, 13.3
"""

from __future__ import annotations

from typing import Tuple

import streamlit as st

PAGE_SIZE_OPTIONS = (10, 25, 50)


def compute_window(total_items: int, current_page: int, page_size: int) -> Tuple[int, int]:
    """Reine Funktion: clamp(page) + total_pages.

    Args:
        total_items: Gesamtzahl der Eintraege.
        current_page: Angeforderte Seite (1-basiert).
        page_size: Eintraege pro Seite (>0).

    Returns:
        (gueltige_seite, total_pages)
    """
    if page_size <= 0:
        page_size = 1
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    page = min(max(1, current_page), total_pages)
    return page, total_pages


def render_pagination_controls(total_items: int, key_prefix: str) -> Tuple[int, int, int]:
    """Zeigt Seitengroesse-Select + Chevron-Navigation (:material/chevron_left:/:material/chevron_right:) und liefert (page, page_size, total_pages).

    Args:
        total_items: Gesamtzahl der Eintraege in der Liste.
        key_prefix: Eindeutiges Praefix (z. B. "input_files") - daraus werden
            die Widget-Keys ``{prefix}_page``/``{prefix}_prev``/... und der
            bestehende Session-State-Key ``{prefix}_page`` gebildet.

    Returns:
        (page, page_size, total_pages)
    """
    # R13.2: Seitengroesse wird im geteilten Session-State persistiert.
    # Widget-Keys sind pro Liste eindeutig (mehrere Listen koennen im selben
    # Tab sichtbar sein), der Wert landet aber in ``files_per_page``.
    current_size = st.session_state.get("files_per_page", 10)
    index = PAGE_SIZE_OPTIONS.index(current_size) if current_size in PAGE_SIZE_OPTIONS else 0
    page_size = st.selectbox(
        "Dateien pro Seite",
        options=list(PAGE_SIZE_OPTIONS),
        index=index,
        key=f"{key_prefix}_size",
    )
    if page_size != st.session_state.get("files_per_page"):
        st.session_state["files_per_page"] = page_size

    current_page = st.session_state.get(f"{key_prefix}_page", 1)
    page, total_pages = compute_window(total_items, current_page, page_size)
    if page != current_page:
        st.session_state[f"{key_prefix}_page"] = page

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button(":material/chevron_left: Zurück", disabled=(page <= 1), key=f"{key_prefix}_prev"):
            st.session_state[f"{key_prefix}_page"] = max(1, page - 1)
            st.rerun()

    with col2:
        st.markdown(
            f"<div style='text-align: center'>Seite {page} von {total_pages}</div>",
            unsafe_allow_html=True,
        )

    with col3:
        if st.button("Weiter :material/play_arrow:", disabled=(page >= total_pages), key=f"{key_prefix}_next"):
            st.session_state[f"{key_prefix}_page"] = min(total_pages, page + 1)
            st.rerun()

    return page, page_size, total_pages
