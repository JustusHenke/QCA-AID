"""
Unit-Tests fuer den Workflow-Status-Mapper (ux-usability-refactoring, Task 9.1).

step_status() ist eine reine Funktion ohne Streamlit-Rerun-Effekte.

Requirements: 1.1, 1.2
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from webapp_components.workflow_status import (  # noqa: E402
    NEXT_STEP,
    WORKFLOW_STEPS,
    step_status,
)


class TestStepStatus:
    def test_maps_input_files_to_analysis(self):
        status = step_status(
            {"config": True, "codebook": True, "input_files": True, "explorer": False}
        )
        assert status["analysis"] is True  # input_files -> analysis-Step
        assert status["explorer"] is False

    def test_all_complete(self):
        status = step_status(
            {"config": True, "codebook": True, "input_files": True, "explorer": True}
        )
        assert all(status.values())

    def test_missing_keys_are_false(self):
        status = step_status({})
        assert set(status) == {"config", "codebook", "analysis", "explorer"}
        assert not any(status.values())

    def test_truthy_coercion(self):
        status = step_status({"config": 1, "input_files": "yes"})
        assert status["config"] is True
        assert status["analysis"] is True


class TestWorkflowStructure:
    def test_steps_cover_all_tabs_in_order(self):
        keys = [tab for tab, _, _ in WORKFLOW_STEPS]
        assert keys == ["config", "codebook", "analysis", "explorer"]

    def test_next_step_chain(self):
        assert NEXT_STEP == {
            "config": "codebook",
            "codebook": "analysis",
            "analysis": "explorer",
        }
        assert "explorer" not in NEXT_STEP  # letzter Schritt hat keinen Nachfolger


# ---------------------------------------------------------------------------
# Task 12: Pagination (R13) - compute_window als reine Funktion
# ---------------------------------------------------------------------------

from webapp_components.pagination import compute_window  # noqa: E402


class TestComputeWindow:
    def test_basic_pages(self):
        page, total = compute_window(45, 1, 10)
        assert (page, total) == (1, 5)

    def test_clamps_high_page(self):
        page, total = compute_window(25, 99, 10)
        assert (page, total) == (3, 3)

    def test_clamps_low_page(self):
        page, total = compute_window(25, 0, 10)
        assert (page, total) == (1, 3)

    def test_empty_list(self):
        page, total = compute_window(0, 1, 10)
        assert (page, total) == (1, 1)

    def test_exact_multiple(self):
        page, total = compute_window(20, 2, 10)
        assert (page, total) == (2, 2)

    def test_invalid_page_size_falls_back(self):
        page, total = compute_window(25, 1, 0)
        assert (page, total) == (1, 25)
