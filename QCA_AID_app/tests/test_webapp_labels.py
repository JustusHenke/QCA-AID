"""
Unit-Tests fuer reine Label-/Summary-Builder der Webapp-Dialoge
(ux-usability-refactoring, Task 2.1 und Task 6.1/6.2).

Nur Streamlit-freie Funktionen werden getestet (@st.dialog-Wrapper und
Leisten-Callbacks werden durch Import ueberdeckt, aber nicht gerendert).

Requirements: 4.1, 4.5, 6.1, 6.2, 6.5
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Pfad-Setup wie in test_webapp_readiness.py
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from webapp_components.dirty_state_bar import (  # noqa: E402
    DIRTY_FLAG_KEYS,
    VIEW_NAMES,
    apply_tab_badges,
    describe_dirty_state,
    dirty_tabs,
    get_dirty_flags,
)
from webapp_components.dialog_helpers import (  # noqa: E402
    LLM_COST_ANALYSIS_TYPES,
    build_start_summary,
    format_start_summary_text,
)


def make_analysis(name: str, active: bool, analysis_type: str):
    return SimpleNamespace(name=name, active=active, analysis_type=analysis_type)


def make_explorer_config(analyses, provider="openai", model="gpt-4o-mini"):
    return SimpleNamespace(
        analysis_configs=analyses,
        base_config={"provider": provider, "model": model},
    )


class TestBuildStartSummary:
    def test_active_filtering_and_counts(self):
        config = make_explorer_config(
            [
                make_analysis("Netz", True, "netzwerk"),
                make_analysis("Heatmap", False, "heatmap"),
                make_analysis("Sunburst", True, "sunburst"),
            ]
        )
        summary = build_start_summary(config, selected_file=r"C:\out\QCA-AID_Analysis_x.xlsx")
        assert summary["active_count"] == 2
        assert summary["inactive_count"] == 1
        assert summary["active_names"] == ["Netz", "Sunburst"]
        assert summary["file_name"] == "QCA-AID_Analysis_x.xlsx"

    def test_llm_costs_flag_true_for_llm_types(self):
        for llm_type in LLM_COST_ANALYSIS_TYPES:
            config = make_explorer_config([make_analysis("X", True, llm_type)])
            assert build_start_summary(config)["llm_costs"] is True, llm_type

    def test_llm_costs_flag_false_for_visualizations(self):
        config = make_explorer_config(
            [make_analysis("Netz", True, "netzwerk"), make_analysis("Sun", True, "sunburst")]
        )
        assert build_start_summary(config)["llm_costs"] is False

    def test_llm_costs_ignores_inactive_llm_analyses(self):
        config = make_explorer_config(
            [make_analysis("Zusammenfassung", False, "summary_paraphrase")]
        )
        summary = build_start_summary(config)
        assert summary["llm_costs"] is False
        assert summary["active_count"] == 0

    def test_provider_and_model_from_base_config(self):
        config = make_explorer_config([], provider="anthropic", model="claude-sonnet-4-5")
        summary = build_start_summary(config)
        assert summary["provider"] == "anthropic"
        assert summary["model"] == "claude-sonnet-4-5"

    def test_empty_config_is_safe(self):
        summary = build_start_summary(SimpleNamespace(analysis_configs=None, base_config={}))
        assert summary["active_count"] == 0
        assert summary["active_names"] == []
        assert summary["file_name"] is None
        assert summary["llm_costs"] is False

    def test_no_file_selected(self):
        summary = build_start_summary(make_explorer_config([]), selected_file=None)
        assert summary["file_name"] is None


class TestFormatStartSummary:
    def test_contains_key_facts(self):
        summary = build_start_summary(
            make_explorer_config(
                [make_analysis("Netzwerk", True, "netzwerk")], provider="openai", model="gpt-4o"
            ),
            selected_file=r"C:\out\analysis.xlsx",
        )
        text = format_start_summary_text(summary)
        assert "analysis.xlsx" in text
        assert "Netzwerk" in text
        assert "openai" in text
        assert "gpt-4o" in text
        assert "LLM-Kosten" not in text  # netzwerk ruft kein LLM

    def test_llm_cost_warning_when_relevant(self):
        summary = build_start_summary(
            make_explorer_config([make_analysis("Zusammenfassung", True, "summary_paraphrase")])
        )
        text = format_start_summary_text(summary)
        assert "LLM-Kosten" in text

    def test_inactive_count_hint(self):
        summary = build_start_summary(
            make_explorer_config(
                [
                    make_analysis("Aktiv", True, "netzwerk"),
                    make_analysis("Inaktiv", False, "heatmap"),
                ]
            )
        )
        text = format_start_summary_text(summary)
        assert "übersprungen" in text

    def test_empty_summary_readable(self):
        text = format_start_summary_text(build_start_summary(make_explorer_config([])))
        assert "keine" in text
        assert "(0)" in text


# ---------------------------------------------------------------------------
# Task 6.1 / 6.2: Dirty-Flags, Dirty-Beschreibung, Tab-Badges
# ---------------------------------------------------------------------------

BASE_LABELS = {
    "config": "⚙️ Konfiguration",
    "codebook": "📚 Codebook",
    "analysis": "🔬 Analyse",
    "explorer": "📊 Explorer",
}


class TestGetDirtyFlags:
    def test_all_flags_false_when_missing(self):
        flags = get_dirty_flags({})
        assert flags == {"config": False, "codebook": False, "explorer": False}

    def test_reads_existing_flags(self):
        state = {
            "config_modified": True,
            "codebook_modified": False,
            "explorer_config_modified": True,
        }
        flags = get_dirty_flags(state)
        assert flags["config"] is True
        assert flags["codebook"] is False
        assert flags["explorer"] is True

    def test_truthy_values_coerced_to_bool(self):
        assert get_dirty_flags({"config_modified": 1})["config"] is True
        assert get_dirty_flags({"config_modified": ""})["config"] is False

    def test_flag_key_mapping_is_complete(self):
        assert set(DIRTY_FLAG_KEYS) == {"config", "codebook", "explorer"}
        assert "analysis" not in DIRTY_FLAG_KEYS  # Analyse-Tab hat keinen Dirty-State


class TestDirtyTabsAndDescription:
    def test_dirty_tabs_in_processing_order(self):
        flags = {"config": False, "codebook": True, "explorer": True}
        assert dirty_tabs(flags) == ["codebook", "explorer"]

    def test_no_dirty_tabs(self):
        assert dirty_tabs({"config": False, "codebook": False, "explorer": False}) == []

    def test_describe_single_view(self):
        text = describe_dirty_state({"config": True, "codebook": False, "explorer": False})
        assert text == VIEW_NAMES["config"]

    def test_describe_multiple_views(self):
        text = describe_dirty_state({"config": True, "codebook": True, "explorer": False})
        assert "Konfiguration" in text
        assert "Codebook" in text

    def test_describe_empty(self):
        assert describe_dirty_state({"config": False, "codebook": False, "explorer": False}) == ""


class TestApplyTabBadges:
    def test_no_badges_when_clean(self):
        labels = apply_tab_badges(BASE_LABELS)
        assert labels == BASE_LABELS

    def test_dirty_badge_only_on_dirty_tabs(self):
        labels = apply_tab_badges(BASE_LABELS, dirty_flags={"config": True})
        assert labels["config"].endswith("⚠️")
        assert labels["codebook"] == "📚 Codebook"
        assert labels["analysis"] == "🔬 Analyse"  # Analyse-Tab ist nie dirty

    def test_blocker_badge(self):
        labels = apply_tab_badges(BASE_LABELS, blocker_tabs={"codebook", "analysis"})
        assert labels["codebook"].endswith("❗")
        assert labels["analysis"].endswith("❗")
        assert labels["config"] == "⚙️ Konfiguration"

    def test_combined_badges(self):
        labels = apply_tab_badges(
            BASE_LABELS, dirty_flags={"codebook": True}, blocker_tabs={"codebook"}
        )
        assert labels["codebook"].endswith("⚠️ ❗")

    def test_unknown_blocker_tab_ignored(self):
        labels = apply_tab_badges(BASE_LABELS, blocker_tabs={"nonexistent"})
        assert labels == BASE_LABELS
