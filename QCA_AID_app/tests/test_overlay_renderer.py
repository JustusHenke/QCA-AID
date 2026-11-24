"""
Tests fuer den OverlayRenderer (Task 9).

Prueft Anforderungen 5.1-5.7, 6.1, 6.2, 3.6:
  - Quads -> In-Text-Highlight, Multiply-Blend, Opacity
  - Mehrfachkodierung: aktiv in-text, weitere als Margin-Stripes
  - Darstellungsmodi highlight/underline/margin/bracket
  - unanchored wird nicht gezeichnet
  - Barrierefreiheit (role/tabindex/aria-label/title)
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QCA_AID_assets.annotation.model import Anchor, Coding, RenderOptions  # noqa: E402
from QCA_AID_assets.annotation.renderer import SVG_NS, OverlayRenderer  # noqa: E402

NS = {"s": SVG_NS}


def _anchor(page=1, quads=None, quote="text", anchored=True):
    return Anchor(
        doc_id="d", page=page, quote=quote,
        quads=quads if quads is not None else [[72, 100, 300, 114]],
        anchored=anchored,
    )


def _coding(segment_id="s1", category="Kat", anchor=None, confidence=None):
    return Coding(
        segment_id=segment_id, doc_id="d", category=category,
        anchor=anchor or _anchor(), confidence=confidence,
    )


def _layers(svg_str):
    root = ET.fromstring(svg_str)
    groups = {g.get("class"): g for g in root.findall("s:g", NS)}
    return root, groups


def _solid(layer):
    """Rechtecke mit Flaechenfarbe (Basis-Highlight, nicht Schraffur)."""
    return [
        r for r in layer.findall("s:rect", NS)
        if not str(r.get("fill") or "").startswith("url(")
    ]


def _hatched(layer):
    """Rechtecke mit Schraffurmuster (weitere Kodierung derselben Stelle)."""
    return [
        r for r in layer.findall("s:rect", NS)
        if str(r.get("fill") or "").startswith("url(")
    ]


# --------------------------------------------------------------------------
# Basis-Highlight
# --------------------------------------------------------------------------

def test_highlight_rects_from_quads():
    anchor = _anchor(quads=[[72, 100, 300, 114], [72, 118, 280, 132]])
    coding = _coding(anchor=anchor)
    svg = OverlayRenderer().build_page_svg(1, [anchor], [coding], RenderOptions())

    root, groups = _layers(svg)
    hl = groups["qca-highlights"]
    rects = hl.findall("s:rect", NS)
    assert len(rects) == 2


def test_multiply_blend_and_opacity():
    anchor = _anchor()
    svg = OverlayRenderer().build_page_svg(
        1, [anchor], [_coding(anchor=anchor)], RenderOptions(opacity=0.35)
    )
    _, groups = _layers(svg)
    style = groups["qca-highlights"].get("style")
    assert "mix-blend-mode:multiply" in style
    assert "opacity:0.35" in style


def test_viewbox_uses_page_size():
    anchor = _anchor()
    svg = OverlayRenderer().build_page_svg(
        1, [anchor], [_coding(anchor=anchor)], RenderOptions(), page_size=(400, 600)
    )
    root, _ = _layers(svg)
    assert root.get("viewBox") == "0 0 400 600"


# --------------------------------------------------------------------------
# Mehrfachkodierung
# --------------------------------------------------------------------------

def test_multiple_codings_active_in_text_others_in_margin():
    anchor = _anchor(quads=[[72, 100, 300, 114]])
    c1 = _coding("s1", "A", anchor)
    c2 = _coding("s2", "B", anchor)
    svg = OverlayRenderer().build_page_svg(
        1, [anchor], [c1, c2], RenderOptions(active_coding_id="s1")
    )

    _, groups = _layers(svg)
    hl = groups["qca-highlights"]
    margin_rects = groups["qca-margin-overlay"].findall("s:rect", NS)
    assert len(_solid(hl)) == 1
    assert len(_hatched(hl)) == 1
    assert len(margin_rects) == 1
    assert _solid(hl)[0].get("fill") != margin_rects[0].get("fill")  # verschiedene Kategorien


def test_separate_segments_both_highlighted():
    a1 = _anchor()
    a2 = _anchor(quads=[[72, 140, 300, 154]])
    c1 = _coding("s1", "A", a1)
    c2 = Coding(segment_id="s2", doc_id="d", category="B", anchor=a2)

    svg = OverlayRenderer().build_page_svg(
        1, [a1, a2], [c1, c2], RenderOptions()
    )
    _, groups = _layers(svg)
    rects = groups["qca-highlights"].findall("s:rect", NS)
    assert len(rects) == 2
    assert {r.get("fill") for r in rects} == {rects[0].get("fill"), rects[1].get("fill")}


def test_same_text_area_is_not_painted_over_itself():
    """Zwei Kodierungen desselben Chunks teilen die Quads.

    Dann darf die zweite Kodierung die erste nicht exakt uebermalen - sonst
    sieht man im Viewer nur eine der beiden (genau der Fall "7 Kodierungen,
    aber kaum sichtbare Markierungen").
    """
    anchor = _anchor(quads=[[72, 100, 300, 114], [72, 118, 280, 132]])
    c1 = _coding("s1", "Legitimation", anchor)
    c2 = _coding("s2", "Struktur", anchor)

    svg = OverlayRenderer().build_page_svg(
        1, [anchor, anchor], [c1, c2], RenderOptions()
    )
    _, groups = _layers(svg)
    hl = groups["qca-highlights"]
    margin = groups["qca-margin-overlay"].findall("s:rect", NS)

    # Erste Kodierung: In-Text-Highlight ueber beide Zeilen.
    assert [r.get("data-segment-id") for r in _solid(hl)] == ["s1", "s1"]
    # Zweite Kodierung: Schraffur ueber dieselben Quads (deckungsgleich,
    # aber sichtbar) plus eigener, klickbarer Margin-Stripe.
    assert [r.get("data-segment-id") for r in _hatched(hl)] == ["s2", "s2"]
    assert [r.get("data-segment-id") for r in margin] == ["s2"]
    assert _solid(hl)[0].get("fill") != margin[0].get("fill")


def test_active_coding_of_a_shared_area_wins_the_highlight():
    anchor = _anchor(quads=[[72, 100, 300, 114]])
    c1 = _coding("s1", "A", anchor)
    c2 = _coding("s2", "B", anchor)

    svg = OverlayRenderer().build_page_svg(
        1, [anchor, anchor], [c1, c2], RenderOptions(active_coding_id="s2")
    )
    _, groups = _layers(svg)
    hl = groups["qca-highlights"]
    margin = groups["qca-margin-overlay"].findall("s:rect", NS)
    assert [r.get("data-segment-id") for r in _solid(hl)] == ["s2"]
    assert [r.get("data-segment-id") for r in _hatched(hl)] == ["s1"]
    assert [r.get("data-segment-id") for r in margin] == ["s1"]


def test_active_coding_is_emphasised_and_others_dimmmed():
    a1 = _anchor(quads=[[72, 100, 300, 114]])
    a2 = _anchor(quads=[[72, 140, 300, 154]])
    svg = OverlayRenderer().build_page_svg(
        1,
        [a1, a2],
        [_coding("s1", "A", a1), _coding("s2", "B", a2)],
        RenderOptions(active_coding_id="s1"),
    )
    _, groups = _layers(svg)
    solid = {
        r.get("data-segment-id"): r.get("opacity")
        for r in _solid(groups["qca-highlights"])
    }
    assert solid["s1"] == "1.00"      # ausgewaehlt: volle Deckkraft
    assert solid["s2"] == "0.35"      # nicht ausgewaehlt: abgeschwaecht


def test_without_selection_nothing_is_dimmmed():
    a1 = _anchor(quads=[[72, 100, 300, 114]])
    a2 = _anchor(quads=[[72, 140, 300, 154]])
    svg = OverlayRenderer().build_page_svg(
        1,
        [a1, a2],
        [_coding("s1", "A", a1), _coding("s2", "B", a2)],
        RenderOptions(),
    )
    _, groups = _layers(svg)
    assert all(r.get("opacity") is None for r in _solid(groups["qca-highlights"]))


def test_hatching_can_be_switched_off():
    anchor = _anchor(quads=[[72, 100, 300, 114]])
    svg = OverlayRenderer().build_page_svg(
        1,
        [anchor, anchor],
        [_coding("s1", "A", anchor), _coding("s2", "B", anchor)],
        RenderOptions(hatch_shared=False),
    )
    _, groups = _layers(svg)
    assert _hatched(groups["qca-highlights"]) == []
    assert len(groups["qca-margin-overlay"].findall("s:rect", NS)) == 1


# --------------------------------------------------------------------------
# Modi
# --------------------------------------------------------------------------

def test_underline_mode_is_thin():
    anchor = _anchor(quads=[[72, 100, 300, 114]])
    svg = OverlayRenderer().build_page_svg(
        1, [anchor], [_coding(anchor=anchor)], RenderOptions(mode="underline")
    )
    _, groups = _layers(svg)
    rect = groups["qca-highlights"].findall("s:rect", NS)[0]
    assert float(rect.get("height")) < 3


def test_margin_mode_has_no_in_text_highlight():
    anchor = _anchor()
    c1 = _coding("s1", "A", anchor)
    svg = OverlayRenderer().build_page_svg(
        1, [anchor], [c1], RenderOptions(mode="margin")
    )
    _, groups = _layers(svg)
    assert groups["qca-highlights"].findall("s:rect", NS) == []
    assert len(groups["qca-margin-overlay"].findall("s:rect", NS)) == 1


def test_bracket_mode_uses_paths():
    anchor = _anchor()
    svg = OverlayRenderer().build_page_svg(
        1, [anchor], [_coding(anchor=anchor)], RenderOptions(mode="bracket")
    )
    _, groups = _layers(svg)
    assert len(groups["qca-margin-overlay"].findall("s:path", NS)) == 1


# --------------------------------------------------------------------------
# Filter & Negativfaelle
# --------------------------------------------------------------------------

def test_unanchored_anchor_not_drawn():
    anchor = _anchor(quads=[], anchored=False)
    svg = OverlayRenderer().build_page_svg(
        1, [anchor], [_coding(anchor=anchor)], RenderOptions()
    )
    root, groups = _layers(svg)
    assert root.findall(".//s:rect", NS) == []


def test_other_page_anchor_filtered():
    anchor = _anchor(page=2)
    svg = OverlayRenderer().build_page_svg(
        1, [anchor], [_coding(anchor=anchor)], RenderOptions()
    )
    _, groups = _layers(svg)
    assert groups["qca-highlights"].findall("s:rect", NS) == []


def test_invalid_options_raise():
    anchor = _anchor()
    with pytest.raises(ValueError):
        OverlayRenderer().build_page_svg(
            1, [anchor], [_coding(anchor=anchor)], RenderOptions(mode="glow")
        )


# --------------------------------------------------------------------------
# Barrierefreiheit
# --------------------------------------------------------------------------

def test_accessibility_attributes_present():
    anchor = _anchor()
    coding = _coding("s1", "Problemdruck", anchor, confidence=0.87)
    svg = OverlayRenderer().build_page_svg(1, [anchor], [coding], RenderOptions())

    _, groups = _layers(svg)
    rect = groups["qca-highlights"].findall("s:rect", NS)[0]
    assert rect.get("role") == "button"
    assert rect.get("tabindex") == "0"
    assert "Problemdruck" in rect.get("aria-label")
    title = rect.find("s:title", NS)
    assert title is not None and "0.87" in title.text


# --------------------------------------------------------------------------
# Farben
# --------------------------------------------------------------------------

def test_distinct_categories_get_distinct_colors():
    a1 = _anchor()
    a2 = _anchor(quads=[[72, 140, 300, 154]])
    c1 = _coding("s1", "A", a1)
    c2 = Coding(segment_id="s2", doc_id="d", category="B", anchor=a2)
    svg = OverlayRenderer().build_page_svg(1, [a1, a2], [c1, c2], RenderOptions())

    _, groups = _layers(svg)
    fills = [r.get("fill") for r in groups["qca-highlights"].findall("s:rect", NS)]
    assert len(set(fills)) == 2


def test_explicit_category_color_wins():
    anchor = _anchor()
    renderer = OverlayRenderer(category_colors={"Kat": "#123456"})
    svg = renderer.build_page_svg(1, [anchor], [_coding(anchor=anchor)], RenderOptions())

    _, groups = _layers(svg)
    assert groups["qca-highlights"].findall("s:rect", NS)[0].get("fill") == "#123456"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))