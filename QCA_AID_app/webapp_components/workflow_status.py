"""
Workflow Status
===============
Klickbare Workflow-Steps in der Sidebar (R1) + "Weiter"-CTAs am Tab-Ende
(R1.3, R1.4).

Ersetzt die reine Textliste in ``webapp.py → render_config_info()``.
Die Status-Logik liegt im zentralen Readiness-Layer
(``webapp_logic/readiness.py``) - eine Quelle fuer Sidebar-Steps,
Tab-Badges (R4.5) und Jump-Links (R11).

Requirements (ux-usability-refactoring): 1.1, 1.2, 1.3, 1.4
"""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

from webapp_components.icons import ICON_STATUS_OK, ICON_STATUS_WARN

# (tab_key, Label, Beschreibung) - Reihenfolge = Workflow-Reihenfolge
WORKFLOW_STEPS: list[tuple[str, str, str]] = [
    ("config", ":material/settings: Konfiguration", "Konfiguration laden/erstellen"),
    ("codebook", ":material/menu_book: Codebook", "Forschungsfrage und Kategorien definieren"),
    ("analysis", ":material/biotech: Analyse", "Eingabedateien bereitstellen und Analyse starten"),
    ("explorer", ":material/bar_chart: Explorer", "Ergebnisse explorieren"),
]

#: Naechster Schritt pro Tab (fuer die "Weiter"-CTAs am Tab-Ende, R1.3)
NEXT_STEP: dict[str, str] = {
    "config": "codebook",
    "codebook": "analysis",
    "analysis": "explorer",
}

#: Anzeigelabel der naechsten Schritte (fuer die CTA-Beschriftung)
STEP_LABELS: dict[str, str] = {tab: label for tab, label, _ in WORKFLOW_STEPS}

#: Mapping Readiness-Status-Key -> tab_key (get_workflow_status liefert
#: "input_files" statt "analysis" als Schluessel)
_STATUS_KEYS: dict[str, str] = {
    "config": "config",
    "codebook": "codebook",
    "analysis": "input_files",
    "explorer": "explorer",
}


def step_status(workflow_status: Mapping[str, Any]) -> dict[str, bool]:
    """Mapt den Readiness-Status auf die Workflow-Steps (reine Funktion).

    Args:
        workflow_status: Ergebnis von ``get_workflow_status(ctx)`` mit den
            Keys config / codebook / input_files / explorer.

    Returns:
        Mapping tab_key -> is_complete fuer alle WORKFLOW_STEPS.
    """
    return {
        tab_key: bool(workflow_status.get(_STATUS_KEYS[tab_key], False))
        for tab_key, _, _ in WORKFLOW_STEPS
    }


def switch_to_tab(tab_key: str) -> None:
    """Tab-Wechsel ohne Import von webapp.py (vermeidet Zirkelimport).

    Setzt beide Session-State-Keys (active_tab_key + active_tab fuer
    Kompatibilitaet) und triggert einen Rerun.
    """
    st.session_state.active_tab_key = tab_key
    # active_tab durchgaengig als lowercase-Key (Dispatch + focus._current_tab)
    st.session_state.active_tab = tab_key
    # Programmatische Navigation markieren - Anwendung erst im naechsten Run
    # vor der Radio-Instanziierung (Widget hier schon instanziiert).
    st.session_state["nav_programmatic"] = tab_key
    st.rerun()


def _ensure_codebook_loaded() -> None:
    """Codebook-Fallback: Session State zuerst, sonst von Datei laden.

    Damit zeigen die Sidebar-Steps einen realistischen Codebook-Status,
    auch wenn der Codebook-Tab noch nie geoeffnet war.
    """
    codebook_data = st.session_state.get("codebook_data", None)
    if codebook_data is not None:
        return

    from webapp_logic.codebook_manager import CodebookManager  # noqa: PLC0415

    codebook_manager = CodebookManager(st.session_state.project_manager.get_root_directory())
    success, codebook_data, _errors = codebook_manager.load_codebook()
    if success and codebook_data:
        st.session_state.codebook_data = codebook_data


def scroll_to_top() -> None:
    """Scrollt das Haupt-Fenster nach oben (nach Tab-Wechsel).

    Streamlit behaelt die Scroll-Position bei Reruns - nach einem
    programmatischen Tab-Wechsel (Weiter-CTA, Workflow-Step, Handoff) soll
    der neue Tab aber von oben beginnen (Nutzer-Feedback zu R1.2).

    Der Aufruf muss im Render-Durchlauf des NEUEN Tabs erfolgen; er wird
    aus render_navigation() bei erkanntem Tab-Wechsel getriggert.
    """
    from streamlit.components.v1 import html  # noqa: PLC0415

    js = """
    <script>
    (function() {
        const doc = window.parent.document;
        const selectors = [
            'section.main',
            '[data-testid="stMain"]',
            '[data-testid="stAppViewContainer"]',
            '[data-testid="stAppViewContainer"] > div',
            '.stApp'
        ];
        const scrollToTop = function() {
            selectors.forEach(function(sel) {
                doc.querySelectorAll(sel).forEach(function(el) {
                    try { el.scrollTop = 0; } catch (e) {}
                });
            });
            window.parent.scrollTo(0, 0);
        };
        // Sofort + mit kurzen Wiederholungen: Streamlit stellt die Scroll-
        // Position nach dem Rerun wieder her - der Retry setzt danach erneut
        // auf 0, damit der neue Tab von oben beginnt.
        scrollToTop();
        let tries = 0;
        const iv = setInterval(function() {
            scrollToTop();
            if (++tries >= 6) { clearInterval(iv); }
        }, 60);
    })();
    </script>
    """
    html(js, height=0)


def render_workflow_status() -> dict[str, bool]:
    """Zeigt klickbare Workflow-Steps in der Sidebar (R1.1, R1.2).

    Returns:
        Mapping tab_key -> is_complete (z. B. fuer weitergehende Anzeigen).
    """
    if "config_data" not in st.session_state:
        # webapp.lazy_load_tab_data ist hier nicht importierbar (Zirkelimport);
        # ohne Config gibt es keinen sinnvollen Status.
        return {tab: False for tab, _, _ in WORKFLOW_STEPS}

    _ensure_codebook_loaded()

    from webapp_logic.readiness import build_context_from_session, get_workflow_status  # noqa: PLC0415

    status = step_status(get_workflow_status(build_context_from_session()))

    for tab_key, label, description in WORKFLOW_STEPS:
        is_complete = status.get(tab_key, False)
        # Statusabhaengige Faerbung: erledigt = gruen, offen = neutral (erbt Textfarbe)
        icon = ICON_STATUS_OK if is_complete else ":material/hourglass_top:"
        button_label = f"{icon} {label}"
        help_text = description

        if st.sidebar.button(
            button_label,
            key=f"workflow_step_{tab_key}",
            use_container_width=True,
            help=help_text,
        ):
            switch_to_tab(tab_key)

    return status


def render_next_step_cta(current_tab: str) -> None:
    """Rendert die "Weiter"-CTA am Tab-Ende (R1.3, R1.4).

    - Keine Blocker im Readiness-Check des Tabs: primaere "Weiter"-Schaltflaeche
      zum naechsten Workflow-Schritt.
    - Blocker vorhanden: Schaltflaeche deaktiviert + Hinweis auf den ersten
      offenen Blocker.
    """
    next_tab = NEXT_STEP.get(current_tab)
    if next_tab is None:
        return

    next_label = STEP_LABELS.get(next_tab, next_tab)
    cta_key = f"next_step_cta_{current_tab}"

    blockers = []
    try:
        from webapp_logic.readiness import build_context_from_session, get_blockers  # noqa: PLC0415

        ctx = build_context_from_session()
        blockers = get_blockers(ctx, current_tab)
    except Exception:
        # Ohne Readiness-Kontext keine Blocker unterstellen
        blockers = []

    if blockers:
        st.button(
            f"Weiter zu {next_label}",
            key=cta_key,
            disabled=True,
            use_container_width=True,
            help=f"Erst beheben: {blockers[0].message}",
        )
        st.caption(f"{ICON_STATUS_WARN} {blockers[0].message}")
    else:
        if st.button(
            f"Weiter zu {next_label} →",
            key=cta_key,
            type="primary",
            use_container_width=True,
        ):
            switch_to_tab(next_tab)
