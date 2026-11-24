"""
Tests fuer die Codeauswahl in der Pruef-Ansicht.

Sichert Anforderung 6.5 (Change): andere Kategorie vergeben, Subcodes
aendern, neue Codes anlegen - ohne die uebrigen Felder einer Kodierung
zu verlieren.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent.parent
for path in (str(REPO_ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from QCA_AID_app.viewer.code_options import (  # noqa: E402
    CodeOption,
    build_code_options,
    code_label,
    find_option,
    iter_codebook_categories,
    option_index,
    parse_code_text,
    recode,
)
from QCA_AID_assets.annotation.model import Anchor, Coding  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

class FakeCategory:
    def __init__(self, name, subcategories=None):
        self.name = name
        self.subcategories = subcategories or {}


class FakeCodebook:
    def __init__(self, categories):
        self.deduktive_kategorien = categories


def _codebook():
    return FakeCodebook(
        {
            "Steuerung": FakeCategory("Steuerung", {"Legitimität": "...", "Effizienz": "..."}),
            "Akteure": FakeCategory("Akteure", {}),
        }
    )


def _coding(segment_id="s1", category="Alt", subcategories=()):
    return Coding(
        segment_id=segment_id,
        doc_id="d",
        category=category,
        subcategories=list(subcategories),
        justification="weil",
        confidence=0.7,
        anchor=Anchor(doc_id="d", page=1, quote="q", quads=[[0, 0, 1, 1]], anchored=True),
    )


# --------------------------------------------------------------------------
# Codebook lesen
# --------------------------------------------------------------------------

def test_categories_read_from_dataclass_codebook():
    categories = dict(iter_codebook_categories(_codebook()))

    assert categories["Steuerung"] == ["Effizienz", "Legitimität"] or set(
        categories["Steuerung"]
    ) == {"Legitimität", "Effizienz"}
    assert categories["Akteure"] == []


def test_categories_read_from_dict_codebook():
    data = {"deduktive_kategorien": {"Frames": {"name": "Frames", "subcategories": {"A": ""}}}}

    assert dict(iter_codebook_categories(data))["Frames"] == ["A"]


def test_missing_codebook_yields_nothing():
    assert list(iter_codebook_categories(None)) == []
    assert list(iter_codebook_categories(object())) == []


# --------------------------------------------------------------------------
# Optionen
# --------------------------------------------------------------------------

def test_options_contain_category_and_each_subcode():
    options = build_code_options(_codebook())
    labels = [option.label for option in options]

    assert "Akteure" in labels
    assert code_label("Steuerung", ["Legitimität"]) in labels
    assert code_label("Steuerung", ["Effizienz"]) in labels


def test_options_are_sorted_and_unique():
    options = build_code_options(_codebook(), codings=[_coding(category="Steuerung")])
    labels = [option.label for option in options]

    assert labels == sorted(labels, key=str.casefold)
    assert len(labels) == len(set(labels))


def test_existing_codings_extend_the_options():
    """Induktive Codes stehen nicht im Codebook, muessen aber waehlbar sein."""
    options = build_code_options(_codebook(), codings=[_coding(category="Emergent", subcategories=["Neu"])])

    assert find_option(options, "Emergent", ["Neu"]) is not None


def test_optionless_coding_sets_only_extras():
    options = build_code_options(None, codings=[_coding(category="Nur", subcategories=[])])

    assert [option.label for option in options] == ["Nur"]


def test_find_option_and_index():
    options = build_code_options(_codebook())

    assert find_option(options, "Akteure").category == "Akteure"
    assert find_option(options, "Fehlt") is None
    index = option_index(options, "Steuerung", ["Effizienz"])
    assert options[index].matches("Steuerung", ["Effizienz"])
    assert option_index(options, "Fehlt") is None


# --------------------------------------------------------------------------
# Freitext
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Steuerung", ("Steuerung", ())),
        ("Steuerung › Legitimität", ("Steuerung", ("Legitimität",))),
        ("Steuerung > Legitimität, Effizienz", ("Steuerung", ("Legitimität", "Effizienz"))),
        ("  Akteure >  Verbände ; Politik ", ("Akteure", ("Verbände", "Politik"))),
    ],
)
def test_parse_code_text(text, expected):
    option = parse_code_text(text)

    assert option is not None
    assert (option.category, option.subcategories) == expected


@pytest.mark.parametrize("text", ["", "   ", "› Sub ohne Kategorie"])
def test_parse_code_text_rejects_empty_category(text):
    assert parse_code_text(text) is None


# --------------------------------------------------------------------------
# Umschreiben
# --------------------------------------------------------------------------

def test_recode_changes_code_and_marks_human_accepted():
    coding = _coding(category="Alt", subcategories=["X"])

    changed = recode(coding, CodeOption("Neu", ("Y", "Z")))

    assert changed.category == "Neu"
    assert changed.subcategories == ["Y", "Z"]
    assert changed.created_by == "human"
    assert changed.status == "accepted"


def test_recode_preserves_justification_confidence_and_anchor():
    coding = _coding()

    changed = recode(coding, CodeOption("Neu"))

    assert changed.justification == coding.justification
    assert changed.confidence == coding.confidence
    assert changed.anchor is coding.anchor
    assert changed.segment_id == coding.segment_id


def test_recode_does_not_mutate_original():
    coding = _coding()
    recode(coding, CodeOption("Neu"))

    assert coding.category == "Alt"
    assert coding.status == "proposed"
    assert coding.created_by == "ai"


def test_recode_can_keep_status():
    coding = _coding()

    kept = recode(coding, CodeOption("Neu"), accept=False)

    assert kept.status == "proposed"
    assert kept.created_by == "human"


def test_recode_drops_subcodes_when_option_has_none():
    coding = _coding(subcategories=["X", "Y"])

    assert recode(coding, CodeOption("Neu")).subcategories == []


# --------------------------------------------------------------------------
# Verdrahtung im Panel
# --------------------------------------------------------------------------

def test_panel_rows_expose_justification_and_inference():
    from QCA_AID_app.viewer.coding_panel import build_coding_rows

    coding = _coding()
    coding.inference = True

    row = build_coding_rows([coding])[0]

    assert row.justification == "weil"
    assert row.inference is True


def test_panel_accepts_recode_parameters():
    import inspect

    from QCA_AID_app.viewer.coding_panel import render_coding_panel

    parameters = inspect.signature(render_coding_panel).parameters

    assert "code_options" in parameters
    assert "on_recode" in parameters
    assert "on_select" in parameters


def test_recode_dialog_has_a_cancel_button():
    """Der Aenderungs-Dialog ist inline und laesst sich abbrechen."""
    source = (APP_DIR / "viewer" / "coding_panel.py").read_text(encoding="utf-8")

    assert "Abbrechen" in source
    assert "editing" in source
    assert 'st.session_state[edit_key] = None' in source


def test_panel_click_on_code_selects_the_coding():
    source = (APP_DIR / "viewer" / "coding_panel.py").read_text(encoding="utf-8")
    review = (APP_DIR / "webapp_components" / "review_ui.py").read_text(encoding="utf-8")

    assert "on_select(row.segment_id)" in source
    assert "on_select=_select_coding" in review
    assert "def _select_coding(" in review


def test_review_tab_wires_recode_callbacks():
    source = (APP_DIR / "webapp_components" / "review_ui.py").read_text(encoding="utf-8")

    assert "code_options=code_options" in source
    assert "on_recode=_apply_coding_change" in source
    assert "_available_code_options" in source


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
