"""
Tests fuer die Verdrahtung des Pruefen-Tabs.

Sichert: Tab-Reihenfolge in webapp.py, Workflow-Kette in der Sidebar,
Readiness-Eintrag und die kopflosen Bausteine der Review-UI.
"""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent
for path in (str(APP_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from QCA_AID_assets.annotation.model import RENDER_MODES, Anchor, Coding  # noqa: E402
from webapp_components import review_ui  # noqa: E402
from webapp_components.workflow_status import (  # noqa: E402
    NEXT_STEP,
    WORKFLOW_STEPS,
    step_status,
)
from webapp_logic import readiness  # noqa: E402

WEBAPP_SOURCE = (APP_DIR / "webapp.py").read_text(encoding="utf-8")
ANALYSIS_SOURCE = (APP_DIR / "webapp_components" / "analysis_ui.py").read_text(encoding="utf-8")


class TestNavigation:
    def test_nav_tabs_follow_workflow_order(self):
        import importlib

        webapp = importlib.import_module("webapp")
        ids = [tab["id"] for tab in webapp.NAV_TABS]

        assert ids == ["config", "codebook", "analysis", "review", "explorer"]

    def test_every_nav_tab_is_part_of_workflow_steps(self):
        import importlib

        webapp = importlib.import_module("webapp")
        nav_ids = [tab["id"] for tab in webapp.NAV_TABS]
        step_ids = [tab for tab, _, _ in WORKFLOW_STEPS]

        assert nav_ids == step_ids

    def test_dispatch_handles_review(self):
        assert re.search(r'active_tab == "review"', WEBAPP_SOURCE)
        assert "render_review_tab()" in WEBAPP_SOURCE

    def test_review_tab_has_icon_and_label(self):
        import importlib

        webapp = importlib.import_module("webapp")
        review = next(tab for tab in webapp.NAV_TABS if tab["id"] == "review")

        assert "Prüfen" in review["label"]
        assert review["label"].startswith(":material/")

    def test_programmatic_tab_name_maps_to_key(self):
        assert '"Prüfen": "review"' in WEBAPP_SOURCE


class TestWorkflowChain:
    def test_review_sits_between_analysis_and_explorer(self):
        assert NEXT_STEP["analysis"] == "review"
        assert NEXT_STEP["review"] == "explorer"

    def test_review_status_comes_from_readiness(self):
        status = step_status(
            {"config": True, "codebook": True, "input_files": True, "review": True}
        )

        assert status["review"] is True

    def test_missing_review_status_is_false(self):
        assert step_status({})["review"] is False


class TestReadiness:
    def make_context(self, tmp_path, *, input_files: bool, results: bool):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        if input_files:
            (input_dir / "interview.txt").write_text("Inhalt", encoding="utf-8")
        if results:
            (output_dir / "QCA-AID_Analysis_deductive_2026.xlsx").write_bytes(b"x")

        return readiness.ReadinessContext(
            project_root=tmp_path,
            config_data=SimpleNamespace(data_dir="input", output_dir="output", analysis_mode="deductive"),
            config_manager=None,
            codebook_data=None,
            analysis_mode="deductive",
        )

    def test_review_ready_when_results_exist(self, tmp_path):
        context = self.make_context(tmp_path, input_files=True, results=True)

        assert readiness.get_workflow_status(context)[readiness.TAB_REVIEW] is True

    def test_review_not_ready_without_results(self, tmp_path):
        context = self.make_context(tmp_path, input_files=True, results=False)

        assert readiness.get_workflow_status(context)[readiness.TAB_REVIEW] is False

    def test_missing_inputs_warn_with_jump_to_analysis(self, tmp_path):
        context = self.make_context(tmp_path, input_files=False, results=False)

        issues = readiness.get_readiness_issues(context, readiness.TAB_REVIEW)

        assert issues
        assert all(issue.severity == "warning" for issue in issues)
        assert any(issue.fix_tab == readiness.TAB_ANALYSIS for issue in issues)

    def test_no_warnings_when_everything_present(self, tmp_path):
        context = self.make_context(tmp_path, input_files=True, results=True)

        assert readiness.get_readiness_issues(context, readiness.TAB_REVIEW) == []


    def test_no_silent_cross_document_fallback(self):
        """Nicht kodierte Dokumente duerfen keine fremden Kodierungen laden."""
        source = (APP_DIR / "webapp_components" / "review_ui.py").read_text(encoding="utf-8")

        assert "doc_name=None" not in source
        assert "load_coding_source(source)" in source
        assert "codings_from_records(records" in source


class TestReviewUiSurface:
    def test_render_functions_exist(self):
        assert callable(review_ui.render_review_tab)
        assert callable(review_ui.render_document_preview)

    def test_render_modes_cover_model_modes(self):
        assert tuple(review_ui.RENDER_MODE_LABELS) == RENDER_MODES

    def test_workspace_is_taller_than_compact(self):
        assert review_ui.WORKSPACE_HEIGHT > review_ui.COMPACT_HEIGHT

    def test_viewer_config_carries_pdf_and_overlays(self, tmp_path):
        pytest.importorskip("fitz")
        import fitz

        pdf = tmp_path / "canonical.pdf"
        doc = fitz.open()
        doc.new_page(width=595, height=842)
        doc.save(str(pdf))
        doc.close()

        document = review_ui.PreviewDocument(
            doc_id="doc_1",
            source_path=tmp_path / "quelle.pdf",
            pdf_path=pdf,
            page_count=1,
            page_labels=["i"],
        )

        config = review_ui.viewer_config_for(
            document, overlays={1: "<svg/>"}, height=300, initial_page=1
        )

        assert config.pdf_data_uri.startswith("data:application/pdf;base64,")
        assert config.doc_id == "doc_1"
        assert config.overlays == {1: "<svg/>"}
        assert config.height == 300
        assert config.page_labels == ["i"]


class TestPreviewFailureHandling:
    """Die Vorschau darf nie den gesamten Tab abbrechen (Anforderung 1.9)."""

    def _patch_service(self, monkeypatch, tmp_path, service):
        monkeypatch.setattr(review_ui, "get_preview_service", lambda root: service)
        monkeypatch.setattr(review_ui, "_project_root", lambda: tmp_path)
        src = tmp_path / "interview.docx"
        src.write_text("inhalt", encoding="utf-8")
        return src

    def test_unexpected_error_returns_none_instead_of_raising(self, monkeypatch, tmp_path):
        class Kaputt:
            def get(self, src):
                raise RuntimeError("Konverter-Bibliothek defekt")

        src = self._patch_service(monkeypatch, tmp_path, Kaputt())

        assert review_ui._load_document(src) is None

    def test_conversion_failure_returns_none(self, monkeypatch, tmp_path):
        class Fehlt:
            def get(self, src):
                raise review_ui.PreviewUnavailable(
                    src, ["libreoffice: nicht verfuegbar", "docling: nicht verfuegbar"]
                )

        src = self._patch_service(monkeypatch, tmp_path, Fehlt())

        assert review_ui._load_document(src) is None

    def test_success_returns_document(self, monkeypatch, tmp_path):
        marker = SimpleNamespace(doc_id="d1")

        class Ok:
            def get(self, src):
                return marker

        src = self._patch_service(monkeypatch, tmp_path, Ok())

        assert review_ui._load_document(src) is marker

    def test_converter_hints_name_the_missing_tool(self):
        hints = review_ui._converter_hints(
            ["libreoffice: nicht verfuegbar", "docling: nicht verfuegbar"]
        )

        assert len(hints) == 2
        assert "LibreOffice" in hints[0]
        assert "docling" in hints[1]

    def test_hints_only_for_missing_tools(self):
        assert review_ui._converter_hints(["libreoffice: Exit 1: kaputt"]) == []
        assert review_ui._converter_hints([]) == []

    def test_analysis_preview_uses_failure_safe_loader(self):
        source = (APP_DIR / "webapp_components" / "review_ui.py").read_text(encoding="utf-8")

        assert source.count("_load_document(") >= 3
        assert "except Exception as exc:  # noqa: BLE001 - Vorschau ist nicht kritisch" in source

    def test_panel_shows_only_current_page_codings(self):
        source = (APP_DIR / "webapp_components" / "review_ui.py").read_text(encoding="utf-8")

        # Kein Seiten-Filter-Dropdown mehr: das Panel folgt der Viewer-Seite.
        assert "review_page_filter" not in source
        assert "page=current_page" in source

    def test_page_resets_on_document_switch(self):
        source = (APP_DIR / "webapp_components" / "review_ui.py").read_text(encoding="utf-8")

        assert "STATE_PAGE_DOC" in source
        assert "st.session_state[STATE_PAGE] = None" in source

    def test_page_change_triggers_panel_refresh(self):
        source = (APP_DIR / "webapp_components" / "review_ui.py").read_text(encoding="utf-8")
        handler = source[source.index("def _handle_viewer_event"):source.index("def _available_code_options")]

        assert handler.count("st.rerun()") == 2  # coding-click und page-change


class TestExcelWriteback:
    """Der Pruefen-Tab schreibt seine Entscheidungen additiv in die Analyse-XLSX."""

    def source(self) -> str:
        return (APP_DIR / "webapp_components" / "review_ui.py").read_text(encoding="utf-8")

    def test_export_expander_offers_the_writeback(self):
        source = self.source()

        assert "_render_writeback_section(codings, selected_source)" in source
        assert "write_review_into_xlsx(" in source
        assert "In Analyse-XLSX übernehmen" in source

    def test_writeback_needs_an_explicit_confirmation(self):
        source = self.source()

        assert 'key="review_writeback_confirm"' in source
        assert "disabled=not confirmed" in source

    def test_writeback_only_for_the_loaded_source(self):
        source = self.source()

        assert "if st.session_state.get(STATE_SOURCE) != str(source)" in source

    def test_writeback_is_headless_testable(self):
        from webapp_logic import review_writeback

        assert callable(review_writeback.write_review_into_xlsx)
        assert review_writeback.STATUS_COLUMN in review_writeback.REVIEW_COLUMNS
        assert review_writeback.DECISION_SHEET == "Review_Entscheidungen"


class TestAnalysisPreviewWiring:
    def test_analysis_tab_uses_shared_preview(self):
        assert "from webapp_components.review_ui import render_document_preview" in ANALYSIS_SOURCE
        assert "render_document_preview(Path(selected_file.path))" in ANALYSIS_SOURCE

    def test_raw_text_area_preview_is_gone(self):
        assert "get_file_preview" not in ANALYSIS_SOURCE
        assert 'st.text_area(\n                    "Dateiinhalt"' not in ANALYSIS_SOURCE

    def test_analysis_tab_lists_all_convertible_formats(self):
        for extension in (".pdf", ".docx", ".doc", ".html", ".md", ".txt"):
            assert f'"{extension}"' in ANALYSIS_SOURCE
