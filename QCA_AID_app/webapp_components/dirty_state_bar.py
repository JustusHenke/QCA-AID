"""
Dirty State Bar
===============
Zentrale Leiste fuer ungespeicherte Aenderungen (R4) + Tab-Badges (R4.5).

- Die Leiste wird in webapp.py DIREKT unter render_navigation() gerendert
  (design.md-Entscheidung: kein position:fixed-CSS-Hacking; die Leiste ist
  ohne Scrollen sichtbar und robust gegen Streamlit-Updates).
- Save/Reload-Callbacks pro Tab sind in der Registry ``DIRTY_HANDLERS``
  gebuendelt; das "Speichern" sichert ALLE betroffenen Views, das
  "Verwerfen" laedt nach einmaligem Bestaetigungsdialog ALLE betroffenen
  Views von der Festplatte neu (Reload-Fehler -> Error-Box, Dirty bleibt).
- Tab-Labels werden um :material/warning: (Dirty) und :material/error:
  (Readiness-Blocker) erweitert.

Die Label-/Badge-Bildner sind reine Funktionen (ohne Streamlit-Rerun-
Effekte) und unit-testbar (siehe tests/test_webapp_labels.py).

Requirements (ux-usability-refactoring): 4.1, 4.2, 4.3, 4.4, 4.5
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import streamlit as st

from .icons import ICON_STATUS_ERROR, ICON_STATUS_WARN

# ---------------------------------------------------------------------------
# Konstanten (reine Daten, unit-testbar)
# ---------------------------------------------------------------------------

#: Session-State-Flag pro Tab. Der Analyse-Tab hat keinen persistenten
#: Editierzustand und daher keinen Eintrag.
DIRTY_FLAG_KEYS: dict[str, str] = {
    "config": "config_modified",
    "codebook": "codebook_modified",
    "explorer": "explorer_config_modified",
}

#: Anzeigename pro Tab (fuer die Leisten-Beschreibung).
VIEW_NAMES: dict[str, str] = {
    "config": "Konfiguration",
    "codebook": "Codebook",
    "explorer": "Explorer-Konfiguration",
}

#: Reihenfolge, in der beim Sammel-Speichern/-Verwerfen behandelt wird
#: (Config zuerst, da Config + Codebook eine Datei teilen).
PROCESSING_ORDER: tuple[str, ...] = ("config", "codebook", "explorer")


# ---------------------------------------------------------------------------
# Reine Funktionen (unit-testbar)
# ---------------------------------------------------------------------------

def get_dirty_flags(state: Mapping[str, Any]) -> dict[str, bool]:
    """Liest die ``*_modified``-Flags aus einem Mapping (z. B. st.session_state).

    Fehlende Keys gelten als nicht-dirty.
    """
    return {tab: bool(state.get(flag, False)) for tab, flag in DIRTY_FLAG_KEYS.items()}


def dirty_tabs(flags: Mapping[str, bool]) -> list[str]:
    """Tab-Keys, die dirty sind (in PROCESSING_ORDER)."""
    return [tab for tab in PROCESSING_ORDER if flags.get(tab)]


def describe_dirty_state(flags: Mapping[str, bool]) -> str:
    """Menschliche Beschreibung der betroffenen Views (fuer Leiste + Dialog)."""
    return ", ".join(VIEW_NAMES[tab] for tab in dirty_tabs(flags))


def apply_tab_badges(
    base_labels: Mapping[str, str],
    dirty_flags: Mapping[str, bool] | None = None,
    blocker_tabs: Mapping[str, Any] | Any = (),
) -> dict[str, str]:
    """Baut Tab-Labels mit Badges (R4.5).

    Args:
        base_labels: Mapping tab_key -> Basislabel (ohne Badges).
        dirty_flags: Mapping tab_key -> bool (:material/warning: wenn True).
        blocker_tabs: Iterable von Tab-Keys mit Readiness-Blockern (:material/error:).

    Returns:
        Mapping tab_key -> Label mit optionalen Badges (" :material/warning:",
        " :material/error:").
        Beide Badges koennen kombiniert auftreten (Reihenfolge: warning dann error).
    """
    blockers = set(blocker_tabs or ())
    result: dict[str, str] = {}
    for tab, label in base_labels.items():
        badges = ""
        if dirty_flags and dirty_flags.get(tab):
            badges += f" {ICON_STATUS_WARN}"
        if tab in blockers:
            badges += f" {ICON_STATUS_ERROR}"
        result[tab] = f"{label}{badges}"
    return result


def compute_blocker_tabs() -> set[str]:
    """Tab-Keys mit Readiness-Blockern (severity=error) fuer das :material/error:-Badge (R4.5).

    Nutzt den zentralen Readiness-Layer. Schlaegt der Kontextaufbau fehl
    (z. B. config_data noch nicht geladen), werden keine Blocker gemeldet.
    """
    if "config_data" not in st.session_state or "project_manager" not in st.session_state:
        return set()

    from webapp_logic.readiness import (  # noqa: PLC0415 - lokal, keeps module light
        TAB_ANALYSIS,
        TAB_CODEBOOK,
        TAB_CONFIG,
        TAB_EXPLORER,
        build_context_from_session,
        get_blockers,
    )

    ctx = build_context_from_session()
    return {
        tab
        for tab in (TAB_CONFIG, TAB_CODEBOOK, TAB_ANALYSIS, TAB_EXPLORER)
        if get_blockers(ctx, tab)
    }


# ---------------------------------------------------------------------------
# Save/Reload-Callbacks (R4.2, R4.3)
# ---------------------------------------------------------------------------

def _current_config_file_and_format() -> tuple[str | None, str | None]:
    """Gespeicherter Config-Dateipfad + abgeleitetes Format (oder Defaults)."""
    file_path = st.session_state.get("current_config_filepath")
    fmt = None
    if file_path:
        fmt = "json" if str(file_path).lower().endswith(".json") else "xlsx"
    return file_path, fmt


def _save_config() -> bool:
    """Quick-Save der Haupt-Konfiguration (ohne Speicher-Dialog).

    Speichert unter dem zuletzt geladenen Pfad (bzw. Standard-Pfad) und
    liest die Datei danach zurueck - wie im Config-Save-Dialog - damit
    Widgets beim naechsten Rerun ``config_modified`` nicht erneut setzen.
    """
    from webapp_logic.config_manager import ConfigManager  # noqa: PLC0415

    config_manager = st.session_state.config_manager
    config = st.session_state.config_data
    file_path, fmt = _current_config_file_and_format()

    if file_path:
        success, errors = config_manager.save_config(
            config=config, file_path=file_path, format=fmt or "json"
        )
    else:
        success, errors = config_manager.save_config(config=config)

    if not success:
        st.error(f":material/cancel: Fehler beim Speichern der Konfiguration: {', '.join(errors)}")
        return False

    # Reload wie im Save-Dialog (R4.2-Verhalten der bestehenden Speichern-UI)
    reload_ok, reloaded, reload_errors = config_manager.load_config(
        file_path=file_path, format=fmt
    )
    if reload_ok and reloaded:
        st.session_state.config_data = reloaded
        st.session_state.config_modified = False
        return True

    st.warning(
        ":material/save: Konfiguration gespeichert, aber Rücklesen fehlgeschlagen – "
        "Dirty-Status bleibt erhalten."
    )
    return False


def _reload_config() -> bool:
    """Laedt die Konfiguration von der Festplatte neu (Verwerfen, R4.3).

    Bei Fehler: Error-Box, ``config_modified`` bleibt gesetzt (niemals
    stilles Verwerfen bei Fehler).
    """
    file_path, fmt = _current_config_file_and_format()
    success, config_data, errors = st.session_state.config_manager.load_config(
        file_path=file_path, format=fmt
    )
    if success and config_data:
        st.session_state.config_data = config_data
        st.session_state.config_modified = False
        return True

    st.error(
        ":material/cancel: Verwerfen fehlgeschlagen – Konfiguration konnte nicht von der "
        f"Festplatte geladen werden: {', '.join(errors)} "
        "Ungespeicherte Änderungen bleiben erhalten."
    )
    return False


def _resolve_codebook_save_path() -> Path:
    """Pfad-Logik wie codebook_ui.save_codebook_to_current_file (ohne Side-Effects)."""
    project_root = Path(st.session_state.project_manager.get_root_directory())

    stored = st.session_state.get("current_config_filepath")
    if stored and Path(stored).parent.exists():
        return Path(stored)

    json_path = project_root / "QCA-AID-Codebook.json"
    xlsx_path = project_root / "QCA-AID-Codebook.xlsx"
    if json_path.exists():
        return json_path
    if xlsx_path.exists():
        return xlsx_path
    return json_path  # Noch keine Datei: Standard-JSON im Projekt-Root


def _save_codebook() -> bool:
    """Quick-Save des Codebooks in die aktuelle Datei (R4.2).

    Loescht NUR ``codebook_modified`` - die Konfiguration kann zusaetzlich
    eigene, ungespeicherte Aenderungen haben und wird separat behandelt.
    """
    from webapp_logic.codebook_manager import CodebookManager  # noqa: PLC0415

    codebook = st.session_state.get("codebook_data")
    if codebook is None:
        st.error(":material/cancel: Kein Codebook im Speicher – Speichern nicht möglich.")
        return False

    save_path = _resolve_codebook_save_path()
    fmt = "json" if save_path.suffix.lower() == ".json" else "xlsx"
    analysis_mode = getattr(st.session_state.get("config_data"), "analysis_mode", None)

    manager = CodebookManager(st.session_state.project_manager.get_root_directory())
    success, errors = manager.save_codebook(
        codebook=codebook,
        file_path=str(save_path),
        format=fmt,
        analysis_mode=analysis_mode,
    )
    if success:
        st.session_state.codebook_modified = False
        return True

    st.error(
        f":material/cancel: Fehler beim Speichern des Codebooks ({save_path.name}): "
        f"{'; '.join(errors)}"
    )
    return False


def _reload_codebook() -> bool:
    """Laedt das Codebook von der Festplatte neu (Verwerfen, R4.3)."""
    from webapp_logic.codebook_manager import CodebookManager  # noqa: PLC0415

    file_path, _ = _current_config_file_and_format()
    fmt = None
    if file_path:
        fmt = "json" if str(file_path).lower().endswith(".json") else "xlsx"

    manager = CodebookManager(st.session_state.project_manager.get_root_directory())
    success, codebook_data, errors = manager.load_codebook(
        file_path=file_path, format=fmt
    )
    if success and codebook_data:
        st.session_state.codebook_data = codebook_data
        st.session_state.codebook_modified = False
        st.session_state.codebook_loaded_from = "file"
        return True

    st.error(
        ":material/cancel: Verwerfen fehlgeschlagen – Codebook konnte nicht von der "
        f"Festplatte geladen werden: {'; '.join(errors)} "
        "Ungespeicherte Änderungen bleiben erhalten."
    )
    return False


def _save_explorer() -> bool:
    """Quick-Save der Explorer-Konfiguration in beiden Formaten (R3.2-R3.4).

    Partial-Success (mindestens ein Format persistiert) loescht das
    Dirty-Flag mit einer gebuendelten Warning - analog zur bestehenden
    Save-Logik im Explorer-Tab.
    """
    manager = st.session_state.get("explorer_config_manager")
    config = st.session_state.get("explorer_config_data")
    if manager is None or config is None:
        st.error(":material/cancel: Explorer-Konfiguration nicht geladen – Speichern nicht möglich.")
        return False

    success_json, errors_json = manager.save_config(config, format="json")
    success_xlsx, errors_xlsx = manager.save_config(config, format="xlsx")

    if success_json and success_xlsx:
        st.session_state.explorer_config_modified = False
        return True
    if success_json or success_xlsx:
        failed_label = "XLSX" if success_json else "JSON"
        failed_errors = errors_xlsx if success_json else errors_json
        st.session_state.explorer_config_modified = False
        st.warning(
            f":material/warning: Teilweise gespeichert ({failed_label} fehlgeschlagen): "
            f"{'; '.join(failed_errors)}"
        )
        return True

    st.error(
        ":material/cancel: Fehler beim Speichern der Explorer-Konfiguration: "
        f"JSON: {'; '.join(errors_json)}, XLSX: {'; '.join(errors_xlsx)}"
    )
    return False


def _reload_explorer() -> bool:
    """Laedt die Explorer-Konfiguration von der Festplatte neu (Verwerfen, R4.3)."""
    manager = st.session_state.get("explorer_config_manager")
    if manager is None or "explorer_config_data" not in st.session_state:
        st.error(":material/cancel: Explorer-Konfiguration nicht geladen – Verwerfen nicht möglich.")
        return False

    success, config_data, errors = manager.load_config()
    if success and config_data:
        st.session_state.explorer_config_data = config_data
        st.session_state.explorer_config_loaded_from = "file"
        st.session_state.explorer_config_modified = False
        return True

    st.error(
        ":material/cancel: Verwerfen fehlgeschlagen – Explorer-Konfiguration konnte nicht "
        f"von der Festplatte geladen werden: {'; '.join(errors)} "
        "Ungespeicherte Änderungen bleiben erhalten."
    )
    return False


#: Registry: Save/Reload-Callbacks pro Tab. In webapp.py wird nur
#: render_dirty_bar() gerendert; die Callbacks liegen hier gebuendelt.
DIRTY_HANDLERS: dict[str, dict[str, Callable[[], bool]]] = {
    "config": {"save": _save_config, "reload": _reload_config},
    "codebook": {"save": _save_codebook, "reload": _reload_codebook},
    "explorer": {"save": _save_explorer, "reload": _reload_explorer},
}


# ---------------------------------------------------------------------------
# Sammel-Aktionen + Rendering (R4.1-R4.4)
# ---------------------------------------------------------------------------

def _save_all(flags: Mapping[str, bool]) -> None:
    """Speichert alle betroffenen Views (R4.2)."""
    failed: list[str] = []
    for tab_key in dirty_tabs(flags):
        if not DIRTY_HANDLERS[tab_key]["save"]():
            failed.append(VIEW_NAMES[tab_key])

    if failed:
        st.error(f":material/cancel: Speichern fehlgeschlagen für: {', '.join(failed)}")
        return

    st.toast(":material/save: Alle Änderungen gespeichert")
    st.rerun()


def _discard_all() -> None:
    """Verwirft alle ungespeicherten Aenderungen durch Reload von Festplatte
    (R4.3). Wird als on_confirm aus dem Bestaetigungsdialog gerufen."""
    flags = get_dirty_flags(st.session_state)
    failed: list[str] = []
    for tab_key in dirty_tabs(flags):
        if not DIRTY_HANDLERS[tab_key]["reload"]():
            failed.append(VIEW_NAMES[tab_key])

    if failed:
        st.error(
            f":material/cancel: Verwerfen fehlgeschlagen für: {', '.join(failed)} – "
            "ungespeicherte Änderungen bleiben erhalten."
        )
        return

    # Verworfene Fokus-Flags sind ungueltig geworden
    from webapp_components.focus import clear_focus  # noqa: PLC0415

    clear_focus()
    st.toast("↩ Änderungen verworfen")
    st.rerun()


def render_dirty_bar() -> None:
    """Zeigt die Dirty-Leiste direkt unter der Tab-Navigation (R4.1).

    Design-Abweichung zu Req 4.1 ("Sticky-Leiste unten im Viewport"): Die
    Leiste sitzt stattdessen direkt unter render_navigation() - dokumentiert
    in design.md (Kernentscheidungen). Bei keinen Dirty-Flags: kein Output
    (R4.4).
    """
    flags = get_dirty_flags(st.session_state)
    if not any(flags.values()):
        return

    views = describe_dirty_state(flags)
    st.warning(f":material/warning: **Ungespeicherte Änderungen** ({views})")

    col_save, col_discard, _ = st.columns([1, 1, 3])
    with col_save:
        if st.button(
            ":material/save: Speichern", type="primary", key="dirty_bar_save", use_container_width=True
        ):
            _save_all(flags)
    with col_discard:
        if st.button("↩ Verwerfen", key="dirty_bar_discard", use_container_width=True):
            from webapp_components.dialog_helpers import confirm_discard  # noqa: PLC0415

            confirm_discard(views, on_confirm=_discard_all)
