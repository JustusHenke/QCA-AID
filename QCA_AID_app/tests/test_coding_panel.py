"""
Tests fuer das CodingPanel (Task 11).

Prueft Anforderungen 6.3-6.5, 7.1-7.3: Zeilenaufbau, Filter, Kennzahlen,
Statusuebergaenge und Invarianten.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QCA_AID_assets.annotation.model import Anchor, Coding  # noqa: E402
from QCA_AID_app.viewer.coding_panel import (  # noqa: E402
    ACTION_ACCEPT,
    ACTION_CHANGE,
    ACTION_REJECT,
    apply_action,
    build_coding_rows,
    panel_header,
    panel_regions,
    panel_summary,
    render_coding_panel,
)


def _anchor(page=1, anchored=True, quads=None):
    return Anchor(
        doc_id="d", page=page, quote="q",
        quads=(quads if quads is not None else [[0, 0, 10, 10]]) if anchored else [],
        anchored=anchored,
    )


def _coding(segment_id="s1", category="A", page=None, anchor=None, status="proposed"):
    return Coding(
        segment_id=segment_id, doc_id="d", category=category,
        page=page, anchor=anchor, status=status,
    )


# --------------------------------------------------------------------------
# Zeilen
# --------------------------------------------------------------------------

def test_rows_sorted_and_flagged():
    codings = [
        _coding("s2", "B", page=2, anchor=_anchor(2)),
        _coding("s1", "A", page=1, anchor=_anchor(1)),
    ]
    rows = build_coding_rows(codings, active_id="s2")
    assert [r.segment_id for r in rows] == ["s1", "s2"]
    assert rows[1].active is True
    assert rows[0].active is False


def test_rows_filter_by_page():
    codings = [_coding("s1", page=1), _coding("s2", page=2)]
    rows = build_coding_rows(codings, page=2)
    assert [r.segment_id for r in rows] == ["s2"]


def test_rows_filter_by_status():
    codings = [
        _coding("s1", status="proposed"),
        _coding("s2", status="accepted"),
    ]
    rows = build_coding_rows(codings, status="accepted")
    assert [r.segment_id for r in rows] == ["s2"]


def test_page_falls_back_to_anchor():
    coding = _coding("s1", page=None, anchor=_anchor(page=5))
    rows = build_coding_rows([coding])
    assert rows[0].page == 5


def test_anchored_flag():
    anchored = _coding("s1", anchor=_anchor(anchored=True))
    unanchored = _coding("s2", anchor=_anchor(anchored=False))
    rows = build_coding_rows([anchored, unanchored])
    flags = {r.segment_id: r.anchored for r in rows}
    assert flags["s1"] is True
    assert flags["s2"] is False


# --------------------------------------------------------------------------
# Kennzahlen
# --------------------------------------------------------------------------

def test_panel_summary_counts():
    codings = [
        _coding("s1", status="proposed", anchor=_anchor()),
        _coding("s2", status="accepted", anchor=_anchor()),
        _coding("s3", status="rejected", anchor=_anchor(anchored=False)),
    ]
    summary = panel_summary(codings)
    assert summary == {
        "total": 3, "proposed": 1, "accepted": 1, "rejected": 1, "unanchored": 1,
    }


def test_panel_regions_counts_distinct_text_areas():
    shared = _anchor(quads=[[0, 0, 10, 10]])
    other = _anchor(quads=[[0, 20, 10, 30]])
    codings = [
        _coding("s1", category="A", anchor=shared),
        _coding("s2", category="B", anchor=shared),
        _coding("s3", category="A", anchor=other),
        _coding("s4", anchor=None),
    ]
    assert panel_regions(codings) == 2


def test_panel_header_is_scoped_to_the_page():
    page1 = _anchor(page=1)
    page2 = _anchor(page=2)
    codings = [
        _coding("s1", "A", page=1, anchor=page1),
        _coding("s2", "B", page=1, anchor=page1),
        _coding("s3", "C", page=2, anchor=page2),
    ]
    header = panel_header(codings, page=1)
    assert "auf S. 1" in header
    assert "2 ·" in header            # nur die zwei Kodierungen dieser Seite
    assert "1 Textstelle" in header   # s1 und s2 teilen die Textstelle
    assert "1 ohne Anker" not in header


def test_panel_header_without_page_counts_everything():
    codings = [
        _coding("s1", anchor=_anchor(page=1)),
        _coding("s2", anchor=_anchor(page=2)),
    ]
    header = panel_header(codings)
    assert "auf S." not in header
    assert "2 ·" in header


def test_rows_flag_shared_text_area():
    shared = _anchor(quads=[[0, 0, 10, 10]])
    rows = build_coding_rows(
        [
            _coding("s1", category="A", anchor=shared),
            _coding("s2", category="B", anchor=shared),
        ]
    )
    assert {r.segment_id: r.same_region for r in rows} == {"s1": 2, "s2": 2}


def test_rows_carry_a_quote_preview():
    anchor = Anchor(
        doc_id="d", page=1, quote="  Ein   sehr langes Zitat\nueber zwei Zeilen  ",
        quads=[[0, 0, 10, 10]], anchored=True,
    )
    row = build_coding_rows([_coding("s1", anchor=anchor)])[0]
    assert row.quote == "Ein sehr langes Zitat ueber zwei Zeilen"


def test_quote_preview_is_cut_at_a_word_boundary():
    anchor = Anchor(
        doc_id="d", page=1, quote="wort " * 60,
        quads=[[0, 0, 10, 10]], anchored=True,
    )
    row = build_coding_rows([_coding("s1", anchor=anchor)])[0]
    assert row.quote.endswith("…")
    assert len(row.quote) <= 113


# --------------------------------------------------------------------------
# Aktionen
# --------------------------------------------------------------------------

def test_accept_and_reject_transitions():
    coding = _coding("s1", anchor=_anchor())
    assert apply_action(coding, ACTION_ACCEPT).status == "accepted"
    assert apply_action(coding, ACTION_REJECT).status == "rejected"


def test_change_updates_fields_preserving_rest():
    coding = _coding("s1", category="Alt", anchor=_anchor())
    changed = apply_action(coding, ACTION_CHANGE, category="Neu", subcategories=["x"])
    assert changed.category == "Neu"
    assert changed.subcategories == ["x"]
    assert changed.segment_id == "s1"
    assert changed.anchor == coding.anchor


def test_apply_action_does_not_mutate_original():
    coding = _coding("s1", anchor=_anchor())
    apply_action(coding, ACTION_ACCEPT)
    assert coding.status == "proposed"


def test_unknown_action_raises():
    with pytest.raises(ValueError):
        apply_action(_coding(), "delete")


def test_accepted_coding_without_anchor_fails_invariant():
    coding = _coding("s1", anchor=None)
    accepted = apply_action(coding, ACTION_ACCEPT)
    ok, errors = accepted.validate()
    assert not ok
    assert any("Anker" in e for e in errors)


# --------------------------------------------------------------------------
# Renderer vorhanden
# --------------------------------------------------------------------------

def test_render_coding_panel_is_callable():
    assert callable(render_coding_panel)


# --------------------------------------------------------------------------
# Render-Smoke (Streamlit AppTest)
# --------------------------------------------------------------------------

_PANEL_SCRIPT = '''
import sys
from pathlib import Path
sys.path.insert(0, r"{root}")

import streamlit as st

from QCA_AID_assets.annotation.model import Anchor, Coding
from QCA_AID_app.viewer.coding_panel import render_coding_panel

anchor = Anchor(doc_id="d", page=1, quote="Ein Zitat", quads=[[0, 0, 10, 10]], anchored=True)
codings = [
    Coding(segment_id="s1", doc_id="d", category="Kategorie_A", anchor=anchor),
    Coding(segment_id="s2", doc_id="d", category="Kategorie_B", anchor=anchor),
]
st.session_state.setdefault("selected", None)
st.session_state.setdefault("recoded", None)

render_coding_panel(
    codings,
    active_id="s1",
    page=1,
    key_prefix="t",
    on_select=lambda sid: st.session_state.__setitem__("selected", sid),
    on_action=lambda sid, action: st.session_state.__setitem__("actioned", (sid, action)),
    on_recode=lambda sid, option: st.session_state.__setitem__("recoded", (sid, option.label)),
)
st.write("selected=" + str(st.session_state["selected"]))
'''


def _panel_app(tmp_path):
    from streamlit.testing.v1 import AppTest

    script = tmp_path / "panel_app.py"
    script.write_text(_PANEL_SCRIPT.format(root=REPO_ROOT), encoding="utf-8")
    return AppTest.from_file(str(script), default_timeout=30).run()


def test_panel_renders_rows_and_lets_the_code_be_clicked(tmp_path):
    at = _panel_app(tmp_path)

    assert not at.exception
    labels = [button.label for button in at.button]
    assert any("Kategorie_B" in label for label in labels)
    assert any("Ändern" in label for label in labels)

    at.button(key="t_sel_s2").click().run()

    assert at.session_state["selected"] == "s2"
    assert not at.exception


def test_edit_dialog_opens_and_can_be_cancelled(tmp_path):
    at = _panel_app(tmp_path)

    assert not any(button.label == "Abbrechen" for button in at.button)

    at.button(key="t_edit_s1").click().run()
    assert at.session_state["t_editing"] == "s1"
    assert any(button.label == "Abbrechen" for button in at.button)
    assert any(widget.key == "t_code_new_s1" for widget in at.text_input)

    at.button(key="t_cancel_s1").click().run()
    assert at.session_state["t_editing"] is None
    assert not at.exception


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

# --------------------------------------------------------------------------
# Textanker-Regionen im Panel (Spec text-anker-export, R13)
# --------------------------------------------------------------------------


def _multi_anchor_coding(segment_id="s1", pages=(1, 2)):
    anchors = [
        Anchor(
            doc_id="doc", page=page, quote=f"Zitat auf Seite {page} mit etwas Text",
            quads=[[72, 100, 300, 114]], anchored=True,
        )
        for page in pages
    ]
    return Coding(
        segment_id=segment_id, doc_id="doc", category="Kat",
        anchors=anchors, text_anchors=["a", "b"],
    )


class TestAnchorViews:
    def test_alle_regionen_als_zeilen(self):
        rows = build_coding_rows([_multi_anchor_coding()])
        assert len(rows) == 1
        views = rows[0].anchor_views
        assert [v.index for v in views] == [0, 1]
        assert [v.page for v in views] == [1, 2]
        assert all("Zitat auf Seite" in v.quote for v in views)

    def test_nur_verankerte_regionen(self):
        coding = _multi_anchor_coding()
        unanchored = Anchor(doc_id="doc", page=3, quote="offen", anchored=False)
        coding.anchors.append(unanchored)
        rows = build_coding_rows([coding])
        assert [v.page for v in rows[0].anchor_views] == [1, 2]

    def test_active_flag_mit_index(self):
        coding = _multi_anchor_coding()
        rows = build_coding_rows(
            [coding], active_id="s1", active_anchor_index=1
        )
        flags = [v.active for v in rows[0].anchor_views]
        assert flags == [False, True]

    def test_active_flag_ohne_index_alle_aus(self):
        coding = _multi_anchor_coding()
        rows = build_coding_rows([coding], active_id="s1")
        assert all(not v.active for v in rows[0].anchor_views)

    def test_single_anchor_zeile(self):
        coding = _multi_anchor_coding(pages=(1,))
        rows = build_coding_rows([coding])
        assert len(rows[0].anchor_views) == 1
        assert rows[0].quote.startswith("Zitat auf Seite 1")

    def test_unverankerte_kodierung_leere_views(self):
        coding = Coding(
            segment_id="s", doc_id="doc", category="K",
            anchor=Anchor(doc_id="doc", page=1, quote="x", anchored=False),
        )
        rows = build_coding_rows([coding])
        assert rows[0].anchor_views == []
        assert rows[0].anchored is False


class TestOverlayMapMultiRegion:
    def test_overlay_map_liefert_beide_seiten(self):
        import sys as _sys

        _app_dir = str(Path(__file__).resolve().parent.parent)
        if _app_dir not in _sys.path:
            _sys.path.insert(0, _app_dir)
        from webapp_logic.review_service import build_overlay_map

        coding = _multi_anchor_coding()
        overlays = build_overlay_map(
            page_sizes=[(595, 842), (595, 842)],
            codings=[coding],
        )
        assert sorted(overlays) == [1, 2]

    def test_overlay_map_active_anchor_index_aktzeptiert(self):
        import sys as _sys

        _app_dir = str(Path(__file__).resolve().parent.parent)
        if _app_dir not in _sys.path:
            _sys.path.insert(0, _app_dir)
        from webapp_logic.review_service import build_overlay_map

        coding = _multi_anchor_coding(pages=(1, 1))
        overlays = build_overlay_map(
            page_sizes=[(595, 842)],
            codings=[coding],
            active_coding_id="s1",
            active_anchor_index=1,
        )
        assert 1 in overlays
