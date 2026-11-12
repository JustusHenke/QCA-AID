"""
Dialog Helpers
==============
Zentrale @st.dialog-Wrapper fuer Bestaetigungsdialoge, damit das Confirm-Muster
nicht pro Datei kopiert wird.

- confirm_removal: Bestaetigung destruktiver Aktionen (R7)
- confirm_explorer_start: Zusammenfassung + Bestaetigung vor LLM-Analysen (R6)
- confirm_discard: Bestaetigung des Verwerfens ungespeicherter Aenderungen (R4.3)

Die Summary-Builder sind reine Funktionen (ohne Streamlit-Interaktion) und
somit unit-testbar (siehe tests/test_webapp_labels.py).

Requirements (ux-usability-refactoring): 5.x indirekt, 6.1, 6.2, 6.5, 7.1, 7.2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import streamlit as st

# Analysetypen, die LLM-Aufrufe ausloesen (Kosten-Hinweis im Start-Dialog)
LLM_COST_ANALYSIS_TYPES = frozenset(
    {"summary_paraphrase", "summary_reasoning", "custom_summary", "sentiment_analysis"}
)


def build_start_summary(
    explorer_config_data: Any,
    selected_file: str | None = None,
) -> dict[str, Any]:
    """Baut die Zusammenfassung fuer den Explorer-Start-Dialog (reine Funktion).

    Args:
        explorer_config_data: ExplorerConfigData (duck-typed: analysis_configs, base_config)
        selected_file: Pfad zur ausgewaehlten Analysedatei oder None

    Returns:
        Dict mit file_name, active_names, active_count, inactive_count,
        provider, model, llm_costs
    """
    configs = list(getattr(explorer_config_data, "analysis_configs", None) or [])
    active = [c for c in configs if getattr(c, "active", False)]
    base_config = getattr(explorer_config_data, "base_config", None) or {}

    return {
        "file_name": Path(selected_file).name if selected_file else None,
        "active_names": [getattr(c, "name", "?") or "?" for c in active],
        "active_count": len(active),
        "inactive_count": len(configs) - len(active),
        "provider": base_config.get("provider"),
        "model": base_config.get("model"),
        "llm_costs": any(
            getattr(c, "analysis_type", "") in LLM_COST_ANALYSIS_TYPES for c in active
        ),
    }


def format_start_summary_text(summary: dict[str, Any]) -> str:
    """Formatiert die Start-Zusammenfassung als Markdown (reine Funktion)."""
    file_name = summary.get("file_name")
    names = summary.get("active_names") or []
    lines = [
        f"**Datei:** {file_name or '–'}",
        f"**Aktive Analysen ({len(names)}):** " + (", ".join(names) if names else "keine"),
    ]
    inactive = summary.get("inactive_count") or 0
    if inactive:
        lines[-1] += f" ({inactive} deaktivierte Analyse(n) werden übersprungen)"
    provider = summary.get("provider")
    model = summary.get("model")
    lines.append(f"**Modell:** {provider or '–'} / {model or '–'}")
    if summary.get("llm_costs"):
        lines.append(":material/savings: **Es fallen LLM-Kosten an.**")
    return "\n\n".join(lines)


@st.dialog("Wirklich entfernen?")
def confirm_removal(item_label: str, on_confirm: Callable[[], None]) -> None:
    """Bestaetigungsdialog fuer destruktive Aktionen (R7).

    Args:
        item_label: Anzeigename des zu entfernenden Elements
        on_confirm: Wird bei Bestaetigung ausgefuehrt (Entfernen + Rerun durch Aufrufer)
    """
    st.warning(f"Möchten Sie „{item_label}“ wirklich entfernen?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Entfernen", type="primary", use_container_width=True, key="dlg_remove_confirm"):
            on_confirm()

    with col2:
        if st.button("Abbrechen", use_container_width=True, key="dlg_remove_cancel"):
            st.rerun()


@st.dialog("Ungespeicherte Änderungen verwerfen?")
def confirm_discard(views_text: str, on_confirm: Callable[[], None]) -> None:
    """Bestaetigungsdialog vor dem Verwerfen ungespeicherter Aenderungen (R4.3).

    Args:
        views_text: Beschreibung der betroffenen Views
            (Ergebnis von dirty_state_bar.describe_dirty_state()).
        on_confirm: Wird bei Bestaetigung ausgefuehrt (Laedt von Festplatte neu).
    """
    st.warning(
        "Möchten Sie die ungespeicherten Änderungen wirklich verwerfen?\n\n"
        f"**Betroffen:** {views_text}\n\n"
        "Die Änderungen werden durch den Stand von der Festplatte ersetzt."
    )
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Verwerfen", type="primary", use_container_width=True, key="dlg_discard_confirm"
        ):
            on_confirm()

    with col2:
        if st.button("Abbrechen", use_container_width=True, key="dlg_discard_cancel"):
            st.rerun()


@st.dialog("Explorer starten")
def confirm_explorer_start(summary: dict[str, Any], on_confirm: Callable[[], None]) -> None:
    """Bestaetigungsdialog vor dem Explorer-Start mit Kosten-Hinweis (R6).

    Args:
        summary: Ergebnis von build_start_summary()
        on_confirm: Wird bei Bestaetigung ausgefuehrt (startet die Analysen)
    """
    st.markdown(format_start_summary_text(summary))
    col1, col2 = st.columns(2)

    with col1:
        if st.button(":material/play_arrow: Starten", type="primary", use_container_width=True, key="dlg_explorer_start"):
            on_confirm()

    with col2:
        if st.button("Abbrechen", use_container_width=True, key="dlg_explorer_cancel"):
            st.rerun()
