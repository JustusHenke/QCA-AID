"""
Readiness Layer
===============
Zentrale Workflow-Bereitschaftspruefung fuer die QCA-AID Webapp.

Extrahiert die readiness-Logik, die zuvor inline in
``webapp_components/analysis_ui.py`` (render_analysis_tab) und
``webapp.py`` (render_config_info) dupliziert war. Eine Quelle fuer:
- klickbare Workflow-Steps in der Sidebar (R1)
- Tab-Badges (R4.5)
- Readiness-Issues mit Jump-Links im Analyse-Tab (R11)

Bewusst OHNE Streamlit-Import: reine Datei-/State-Pruefungen,
damit unit-testbar (siehe tests/test_webapp_readiness.py).

Requirements (ux-usability-refactoring): 1.1, 1.4, 4.5, 11.1, 11.4
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

# Tab-Keys (kompatibel zu webapp.py switch_to_tab)
TAB_CONFIG = "config"
TAB_CODEBOOK = "codebook"
TAB_ANALYSIS = "analysis"
TAB_EXPLORER = "explorer"

# Section-Keys fuer Fokus-Flags (ux_focus), siehe design.md
SECTION_FORSCHUNGSFRAGE = "forschungsfrage"
SECTION_KODIERREGELN = "kodierregeln"
SECTION_KATEGORIEN = "kategorien"
SECTION_MODEL = "model"
SECTION_INPUT_FILES = "input_files"

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    """Ein Readiness-Fund mit optionaler Behebungszuordnung (Jump-Link)."""

    message: str
    severity: Severity = "error"
    fix_tab: Optional[str] = None  # tab_key, zu dem gesprungen werden kann
    fix_section: Optional[str] = None  # section_key fuer ux_focus-Flag
    open_folder: Optional[Path] = None  # Alternative: Ordner oeffnen
    hint: str = ""


@dataclass
class ReadinessContext:
    """Alle Abhaengigkeiten der Checks. Manager werden duck-getypt genutzt."""

    project_root: Path
    config_data: Any  # ConfigData (duck-typed)
    config_manager: Any = None  # ConfigManager oder Stub mit json_path/xlsx_path
    codebook_data: Any = None  # CodebookData oder None
    analysis_mode: Optional[str] = None  # aus config_data, falls vorhanden


# ---------------------------------------------------------------------------
# Einzelne Checks (bool-Pruefungen, auch fuer Sidebar-Steps nutzbar)
# ---------------------------------------------------------------------------

def config_is_saved(config_manager: Any) -> Optional[bool]:
    """True/False ob Config-Datei existiert; None, wenn kein Manager bekannt."""
    if config_manager is None:
        return None
    return bool(
        getattr(config_manager, "json_path", None) and Path(config_manager.json_path).exists()
    ) or bool(
        getattr(config_manager, "xlsx_path", None) and Path(config_manager.xlsx_path).exists()
    )


def has_categories(codebook_data: Any) -> bool:
    return bool(
        codebook_data is not None
        and isinstance(getattr(codebook_data, "deduktive_kategorien", None), dict)
        and len(codebook_data.deduktive_kategorien) > 0
    )


def has_research_question(codebook_data: Any) -> bool:
    ff = getattr(codebook_data, "forschungsfrage", None) if codebook_data is not None else None
    return bool(ff and str(ff).strip())


def codebook_ready(codebook_data: Any, analysis_mode: Optional[str]) -> bool:
    """Grounded Mode: Forschungsfrage reicht; sonst Kategorien erforderlich."""
    if analysis_mode == "grounded":
        return has_research_question(codebook_data)
    return has_categories(codebook_data)


def resolve_dir(project_root: Path, rel_dir: str) -> Path:
    """Relatives Verzeichnis gegen Projekt-Root aufloesen."""
    path = Path(rel_dir)
    return path if path.is_absolute() else Path(project_root) / path


def get_input_dir(ctx: ReadinessContext) -> Path:
    return resolve_dir(ctx.project_root, getattr(ctx.config_data, "data_dir", "input"))


def get_output_dir(ctx: ReadinessContext) -> Path:
    return resolve_dir(ctx.project_root, getattr(ctx.config_data, "output_dir", "output"))


def has_input_files(ctx: ReadinessContext) -> bool:
    input_dir = get_input_dir(ctx)
    if not input_dir.exists():
        return False
    return any(
        list(input_dir.glob("*.txt"))
        + list(input_dir.glob("*.pdf"))
        + list(input_dir.glob("*.docx"))
    )


def has_results(ctx: ReadinessContext) -> bool:
    output_dir = get_output_dir(ctx)
    return output_dir.exists() and any(output_dir.glob("QCA-AID_Analysis_*.xlsx"))


# ---------------------------------------------------------------------------
# Issues pro Tab
# ---------------------------------------------------------------------------

def _config_issues(ctx: ReadinessContext) -> list[Issue]:
    issues: list[Issue] = []
    config = ctx.config_data

    is_valid, config_errors = config.validate() if config is not None else (False, ["keine Konfiguration"])
    if not is_valid:
        issues.append(
            Issue(
                message=f"Konfiguration ungültig: {', '.join(list(config_errors)[:2])}",
                severity="error",
                fix_tab=TAB_CONFIG,
                fix_section=SECTION_MODEL,
            )
        )

    saved = config_is_saved(ctx.config_manager)
    if saved is False:
        issues.append(
            Issue(
                message="Konfiguration noch nicht gespeichert",
                severity="warning",
                fix_tab=TAB_CONFIG,
            )
        )
    return issues


def _codebook_issues(ctx: ReadinessContext, forschung_frage_as_error: bool) -> list[Issue]:
    """Codebook-Funde. Der Analyse-Tab wertet die Forschungsfrage als Blocker (Pflicht),
    der Codebook-Tab nur als Empfehlung - daher der Schalter."""
    issues: list[Issue] = []
    mode = ctx.analysis_mode or getattr(ctx.config_data, "analysis_mode", None)
    codebook = ctx.codebook_data

    if codebook is None:
        issues.append(
            Issue(
                message="Codebook konnte nicht geladen werden",
                severity="error",
                fix_tab=TAB_CODEBOOK,
                fix_section=SECTION_KATEGORIEN,
            )
        )
        return issues

    if not has_categories(codebook):
        if mode == "grounded":
            issues.append(
                Issue(
                    message=(
                        "Codebook enthält keine Kategorien (im Grounded Mode OK – "
                        "Hauptkategorien werden emergent gebildet)"
                    ),
                    severity="warning",
                )
            )
        else:
            issues.append(
                Issue(
                    message="Codebook enthält keine Kategorien",
                    severity="error",
                    fix_tab=TAB_CODEBOOK,
                    fix_section=SECTION_KATEGORIEN,
                )
            )

    if not has_research_question(codebook):
        issues.append(
            Issue(
                message="Forschungsfrage fehlt im Codebook",
                severity="error" if forschung_frage_as_error else "warning",
                fix_tab=TAB_CODEBOOK,
                fix_section=SECTION_FORSCHUNGSFRAGE,
            )
        )
        return issues

    # Nur pruefen, wenn Kategorien vorhanden sind (wie bisherige Inline-Logik)
    without_subcats = [
        name
        for name, cat in getattr(codebook, "deduktive_kategorien", {}).items()
        if hasattr(cat, "subcategories") and (not cat.subcategories or len(cat.subcategories) == 0)
    ]
    if without_subcats:
        issues.append(
            Issue(
                message=f"{len(without_subcats)} Kategorie(n) ohne Subkategorien",
                severity="warning",
                fix_tab=TAB_CODEBOOK,
                fix_section=SECTION_KATEGORIEN,
            )
        )
    return issues


def _input_issues(ctx: ReadinessContext) -> list[Issue]:
    issues: list[Issue] = []
    input_dir = get_input_dir(ctx)

    if not input_dir.exists():
        issues.append(
            Issue(
                message=f"Eingabeverzeichnis existiert nicht: {input_dir.name}",
                severity="error",
                fix_tab=None,
                open_folder=ctx.project_root,
            )
        )
        return issues

    input_files = (
        list(input_dir.glob("*.txt"))
        + list(input_dir.glob("*.pdf"))
        + list(input_dir.glob("*.docx"))
    )
    if not input_files:
        issues.append(
            Issue(
                message="Keine Eingabedateien (.txt, .pdf, .docx) gefunden",
                severity="error",
                fix_tab=None,
                open_folder=input_dir,
            )
        )
        return issues

    empty_files = [f.name for f in input_files if f.stat().st_size == 0]
    if empty_files:
        issues.append(
            Issue(
                message=f"{len(empty_files)} leere Datei(en) gefunden",
                severity="warning",
                fix_tab=None,
                open_folder=input_dir,
            )
        )
    return issues


def _output_issues(ctx: ReadinessContext) -> list[Issue]:
    output_dir = get_output_dir(ctx)
    if not output_dir.exists():
        return [Issue(message=f"Ausgabeverzeichnis wird erstellt: {output_dir.name}", severity="warning")]
    if not has_results(ctx):
        return [
            Issue(
                message="Keine Analyseergebnisse gefunden – führen Sie zuerst eine Analyse durch",
                severity="warning",
            )
        ]
    return []


def _explorer_issues(ctx: ReadinessContext) -> list[Issue]:
    """Explorer-Tab: bei fehlenden Ergebnissen IMMER Jump-Link zum Analyse-Tab
    (R9.5), unabhaengig davon, ob das Output-Verzeichnis schon existiert."""
    if has_results(ctx):
        return []
    return [
        Issue(
            message="Keine Analyseergebnisse gefunden – führen Sie zuerst eine Analyse durch",
            severity="warning",
            fix_tab=TAB_ANALYSIS,
        )
    ]


def get_readiness_issues(ctx: ReadinessContext, tab_key: str) -> list[Issue]:
    """Alle Issues fuer einen Tab. Eine Quelle fuer Sidebar-Badges und Jump-Links.

    - config:    Config-Validitaet + gespeichert
    - codebook:  Kategorien (Forschungsfrage als Empfehlung, wie codebook_ui)
    - analysis:  Config + Codebook (Forschungsfrage = Blocker) + Eingabedateien + Output
    - explorer:  Ergebnisse vorhanden
    """
    if tab_key == TAB_CONFIG:
        return _config_issues(ctx)
    if tab_key == TAB_CODEBOOK:
        return _codebook_issues(ctx, forschung_frage_as_error=False)
    if tab_key == TAB_ANALYSIS:
        issues = _config_issues(ctx)
        issues.extend(_codebook_issues(ctx, forschung_frage_as_error=True))
        issues.extend(_input_issues(ctx))
        issues.extend(_output_issues(ctx))
        return issues
    if tab_key == TAB_EXPLORER:
        return _explorer_issues(ctx)
    return []


def get_workflow_status(ctx: ReadinessContext) -> dict[str, bool]:
    """Completeness der Sidebar-Workflow-Steps (klickbar ab R1)."""
    saved = config_is_saved(ctx.config_manager)
    mode = ctx.analysis_mode or getattr(ctx.config_data, "analysis_mode", None)
    return {
        TAB_CONFIG: bool(saved) if saved is not None else False,
        TAB_CODEBOOK: codebook_ready(ctx.codebook_data, mode),
        "input_files": has_input_files(ctx),
        TAB_EXPLORER: has_results(ctx),
    }


def get_blockers(ctx: ReadinessContext, tab_key: str) -> list[Issue]:
    return [i for i in get_readiness_issues(ctx, tab_key) if i.severity == "error"]


def build_context_from_session() -> ReadinessContext:
    """Bequemer Aufbau aus st.session_state.

    Streamlit wird NUR lokal importiert, damit das Modul ohne Streamlit-Laufzeit
    unit-testbar bleibt (siehe tests/test_webapp_readiness.py).

    Voraussetzung: 'config_data' wurde bereits geladen (webapp.main lazy_load_tab_data).
    """
    import streamlit as st  # noqa: PLC0415 - absichtlich lokal

    config_data = st.session_state.get("config_data")
    return ReadinessContext(
        project_root=Path(st.session_state.project_manager.get_root_directory()),
        config_data=config_data,
        config_manager=st.session_state.get("config_manager"),
        codebook_data=st.session_state.get("codebook_data"),
        analysis_mode=getattr(config_data, "analysis_mode", None),
    )
