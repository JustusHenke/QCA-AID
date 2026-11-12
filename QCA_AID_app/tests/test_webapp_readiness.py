"""
Unit-Tests fuer den Readiness-Layer (ux-usability-refactoring, Task 1.1).

Readiness-Checks sind reine Datei-/State-Pruefungen ohne Streamlit -
daher direkt gegen Temp-Projektverzeichnisse testbar.

Requirements: 1.1, 4.5, 11.1
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

# Pfad-Setup: QCA_AID_app importierbar machen (wie in webapp.py)
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from webapp_logic.readiness import (  # noqa: E402
    Issue,
    ReadinessContext,
    TAB_ANALYSIS,
    TAB_CODEBOOK,
    TAB_CONFIG,
    TAB_EXPLORER,
    codebook_ready,
    get_blockers,
    get_readiness_issues,
    get_workflow_status,
    has_categories,
    has_research_question,
)


class StubConfigManager:
    """Duck-typed Manager mit json_path/xlsx_path (wie ConfigManager)."""

    def __init__(self, project_root: Path, saved_json: bool = False, saved_xlsx: bool = False):
        self.json_path = project_root / "QCA-AID-Codebook-Config.json"
        self.xlsx_path = project_root / "QCA-AID-Codebook-Config.xlsx"
        self._saved_json = saved_json
        self._saved_xlsx = saved_xlsx
        if saved_json:
            self.json_path.touch()
        if saved_xlsx:
            self.xlsx_path.touch()


def make_config(saved: bool = True, analysis_mode: str = "deductive", valid: bool = True):
    def validate():
        return (valid, [] if valid else ["Fehler A", "Fehler B"])

    return SimpleNamespace(
        data_dir="input",
        output_dir="output",
        analysis_mode=analysis_mode,
        validate=validate,
    )


def make_codebook(
    with_categories: bool = True,
    with_forschungsfrage: bool = True,
    subcats: bool = True,
):
    categories = {}
    if with_categories:
        categories["Kategorie 1"] = SimpleNamespace(
            subcategories=["Sub 1"] if subcats else []
        )
    return SimpleNamespace(
        forschungsfrage="Wie wirkt X?" if with_forschungsfrage else "  ",
        kodierregeln={"general": [], "format": [], "exclusion": []},
        deduktive_kategorien=categories,
    )


def make_context(tmp_path: Path, *, config_saved: bool = True, **kwargs) -> ReadinessContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    # WICHTIG: Default-Stub nur anlegen, wenn kein eigener Manager übergeben wird -
    # ein Stub mit saved_json=True wuerde sonst tmp_path fuer Overriding-Tests
    # "verschmutzen" (Datei existiert, obwohl der ersetzte Manager nichts speicherte).
    if "config_manager" not in kwargs:
        kwargs["config_manager"] = StubConfigManager(tmp_path, saved_json=config_saved)
    defaults = dict(
        project_root=tmp_path,
        config_data=make_config(),
        codebook_data=make_codebook(),
        analysis_mode="deductive",
    )
    defaults.update(kwargs)
    return ReadinessContext(**defaults)


# ---------------------------------------------------------------------------
# Grundlegende Checks
# ---------------------------------------------------------------------------


class TestBasicChecks:
    def test_has_categories(self):
        assert has_categories(make_codebook(with_categories=True))
        assert not has_categories(make_codebook(with_categories=False))
        assert not has_categories(None)

    def test_has_research_question(self):
        assert has_research_question(make_codebook())
        assert not has_research_question(make_codebook(with_forschungsfrage=False))
        assert not has_research_question(None)

    def test_codebook_ready_deductive_needs_categories(self):
        assert codebook_ready(make_codebook(), "deductive")
        assert not codebook_ready(make_codebook(with_categories=False), "deductive")

    def test_codebook_ready_grounded_needs_only_forschungsfrage(self):
        cb = make_codebook(with_categories=False, with_forschungsfrage=True)
        assert codebook_ready(cb, "grounded")
        cb2 = make_codebook(with_categories=False, with_forschungsfrage=False)
        assert not codebook_ready(cb2, "grounded")


class TestInputFiles:
    def test_no_input_dir_is_error_with_folder_fix(self, tmp_path):
        ctx = make_context(tmp_path)
        issues = [i for i in get_readiness_issues(ctx, TAB_ANALYSIS) if "Eingabeverzeichnis" in i.message]
        assert len(issues) == 1
        issue = issues[0]
        assert issue.severity == "error"
        assert issue.open_folder == tmp_path
        assert issue.fix_tab is None  # kein Jump-Link, Ordner-oeffnen-Alternative

    def test_empty_input_dir_is_error(self, tmp_path):
        (tmp_path / "input").mkdir()
        ctx = make_context(tmp_path)
        issues = [i for i in get_readiness_issues(ctx, TAB_ANALYSIS) if "Keine Eingabedateien" in i.message]
        assert len(issues) == 1
        assert issues[0].open_folder == tmp_path / "input"

    def test_empty_files_warn(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("Inhalt")
        (input_dir / "b.txt").write_text("")
        ctx = make_context(tmp_path)
        issues = [i for i in get_readiness_issues(ctx, TAB_ANALYSIS) if "leere Datei" in i.message]
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_txt_pdf_docx_recognized(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("x")
        (input_dir / "b.pdf").write_text("x")
        (input_dir / "c.docx").write_text("x")
        ctx = make_context(tmp_path)
        messages = [i.message for i in get_readiness_issues(ctx, TAB_ANALYSIS)]
        assert not any("Eingabedatei" in m for m in messages)


# ---------------------------------------------------------------------------
# Issues pro Tab
# ---------------------------------------------------------------------------


class TestAnalysisTabIssues:
    def test_full_project_no_blockers(self, tmp_path):
        (tmp_path / "input").mkdir()
        (tmp_path / "input" / "a.txt").write_text("x")
        (tmp_path / "output").mkdir()
        (tmp_path / "output" / "QCA-AID_Analysis_test.xlsx").touch()
        ctx = make_context(tmp_path)
        assert get_blockers(ctx, TAB_ANALYSIS) == []

    def test_invalid_config_is_blocker(self, tmp_path):
        (tmp_path / "input").mkdir()
        (tmp_path / "input" / "a.txt").write_text("x")
        ctx = make_context(tmp_path, config_data=make_config(valid=False))
        blockers = get_blockers(ctx, TAB_ANALYSIS)
        assert any(i.fix_tab == TAB_CONFIG and "ungültig" in i.message for i in blockers)

    def test_unsaved_config_is_warning(self, tmp_path):
        ctx = make_context(tmp_path, config_manager=StubConfigManager(tmp_path))
        warnings = [
            i
            for i in get_readiness_issues(ctx, TAB_ANALYSIS)
            if i.severity == "warning" and "nicht gespeichert" in i.message
        ]
        assert len(warnings) == 1

    def test_missing_forschungsfrage_is_blocker_in_analysis(self, tmp_path):
        (tmp_path / "input").mkdir()
        (tmp_path / "input" / "a.txt").write_text("x")
        ctx = make_context(tmp_path, codebook_data=make_codebook(with_forschungsfrage=False))
        blockers = get_blockers(ctx, TAB_ANALYSIS)
        assert any(
            i.fix_tab == TAB_CODEBOOK
            and i.fix_section == "forschungsfrage"
            and i.severity == "error"
            for i in blockers
        )

    def test_missing_categories_blocker_unless_grounded(self, tmp_path):
        (tmp_path / "input").mkdir()
        (tmp_path / "input" / "a.txt").write_text("x")
        ctx = make_context(tmp_path, codebook_data=make_codebook(with_categories=False))
        assert any(
            i.message == "Codebook enthält keine Kategorien"
            and i.severity == "error"
            and i.fix_section == "kategorien"
            for i in get_blockers(ctx, TAB_ANALYSIS)
        )
        # Grounded Mode: nur Warning
        ctx_grounded = make_context(
            tmp_path,
            codebook_data=make_codebook(with_categories=False),
            analysis_mode="grounded",
            config_data=make_config(analysis_mode="grounded"),
        )
        grounded_msgs = [i.message for i in get_readiness_issues(ctx_grounded, TAB_ANALYSIS)]
        assert any("Grounded Mode OK" in m for m in grounded_msgs)
        assert not any(
            i.message == "Codebook enthält keine Kategorien"
            and i.severity == "error"
            for i in get_readiness_issues(ctx_grounded, TAB_ANALYSIS)
        )

    def test_categories_without_subcats_warn(self, tmp_path):
        (tmp_path / "input").mkdir()
        (tmp_path / "input" / "a.txt").write_text("x")
        ctx = make_context(tmp_path, codebook_data=make_codebook(subcats=False))
        warnings = [i for i in get_readiness_issues(ctx, TAB_ANALYSIS) if "ohne Subkategorien" in i.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"
        assert warnings[0].fix_section == "kategorien"


class TestPerTabSemantics:
    def test_codebook_tab_forschungsfrage_is_warning_only(self, tmp_path):
        ctx = make_context(tmp_path)
        ctx.codebook_data = make_codebook(with_forschungsfrage=False)
        issues = [i for i in get_readiness_issues(ctx, TAB_CODEBOOK) if "Forschungsfrage" in i.message]
        assert len(issues) == 1
        assert issues[0].severity == "warning"  # im Codebook-Tab Empfehlung

    def test_explorer_tab_reports_missing_results(self, tmp_path):
        ctx = make_context(tmp_path)
        issues = get_readiness_issues(ctx, TAB_EXPLORER)
        assert any(i.fix_tab == TAB_ANALYSIS and "Keine Analyseergebnisse" in i.message for i in issues)

    def test_explorer_tab_ok_with_results(self, tmp_path):
        (tmp_path / "output").mkdir()
        (tmp_path / "output" / "QCA-AID_Analysis_x.xlsx").touch()
        ctx = make_context(tmp_path)
        assert get_readiness_issues(ctx, TAB_EXPLORER) == []

    def test_config_tab_reports_validity_and_save_state(self, tmp_path):
        # Zwei getrennte Projektverzeichnisse: gespeicherte Datei im ersten
        # Kontext wuerde den zweiten sonst "verschmutzen".
        ctx = make_context(tmp_path / "saved", config_data=make_config(valid=False))
        issues = get_readiness_issues(ctx, TAB_CONFIG)
        assert any("ungültig" in i.message for i in issues)
        ctx_unsaved = make_context(tmp_path / "unsaved", config_saved=False)
        issues = get_readiness_issues(ctx_unsaved, TAB_CONFIG)
        assert any("nicht gespeichert" in i.message for i in issues)


class TestWorkflowStatus:
    def test_full_project_all_steps_complete(self, tmp_path):
        (tmp_path / "input").mkdir()
        (tmp_path / "input" / "a.txt").write_text("x")
        (tmp_path / "output").mkdir()
        (tmp_path / "output" / "QCA-AID_Analysis_x.xlsx").touch()
        ctx = make_context(tmp_path)
        status = get_workflow_status(ctx)
        assert status == {
            TAB_CONFIG: True,
            TAB_CODEBOOK: True,
            "input_files": True,
            TAB_EXPLORER: True,
        }

    def test_empty_project_nothing_complete(self, tmp_path):
        ctx = make_context(tmp_path, config_saved=False, codebook_data=None)
        status = get_workflow_status(ctx)
        assert not any(status.values())

    def test_grounded_project_ready_with_forschungsfrage_only(self, tmp_path):
        (tmp_path / "input").mkdir()
        (tmp_path / "input" / "a.txt").write_text("x")
        ctx = make_context(
            tmp_path,
            codebook_data=make_codebook(with_categories=False),
            analysis_mode="grounded",
            config_data=make_config(analysis_mode="grounded"),
        )
        assert get_workflow_status(ctx)[TAB_CODEBOOK] is True


# ---------------------------------------------------------------------------
# Issue-Verhalten
# ---------------------------------------------------------------------------


class TestIssueDataclass:
    def test_issue_defaults(self):
        issue = Issue(message="Test")
        assert issue.severity == "error"
        assert issue.fix_tab is None
        assert issue.fix_section is None
        assert issue.open_folder is None
        assert issue.hint == ""

    def test_issue_is_frozen(self):
        issue = Issue(message="Test")
        with pytest.raises(Exception):
            issue.message = "anderer"  # type: ignore[misc]
