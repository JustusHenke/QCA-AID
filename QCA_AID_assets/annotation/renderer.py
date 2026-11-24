"""
Overlay Renderer
================
Erzeugt das Annotation-Overlay fuer den TextViewer als SVG
(siehe .kiro/specs/pdf-preview-annotation, Anforderungen 5.1-5.7, 6.1, 6.2, 3.6).

Prinzip (SOTA):
  - Highlights liegen als Quads direkt ueber dem Text
  - Multiply-Blending + reduzierte Deckkraft, Text bleibt lesbar
  - aktive Kodierung = In-Text-Highlight, weitere = gestapelte Margin-Stripes
  - Darstellungsmodi: highlight, underline, margin, bracket
  - jedes Element traegt data-coding-id/data-segment-id fuer die Interaktion
    sowie role/tabindex/aria-label fuer Barrierefreiheit

Das SVG nutzt das PDF-Koordinatensystem (points) als viewBox und wird im
Viewer absolut ueber die PDF.js-Seite gelegt. PDF-Baking folgt in Task 12.
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

from .model import Anchor, Coding, RenderOptions

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

# Gut unterscheidbare, helle Kategoriefarben (deuteranopie-freundlich sortiert).
DEFAULT_PALETTE = [
    "#f6a5a5",  # rot
    "#a5c8f6",  # blau
    "#a5e0b0",  # gruen
    "#f6d78a",  # gelb
    "#d6a5f6",  # violett
    "#8fd8d8",  # tuerkis
    "#f6c0a5",  # orange
    "#c8c8a5",  # oliv
]

MARGIN_STRIPE_WIDTH = 4.0
MARGIN_STRIPE_GAP = 2.0
MARGIN_OFFSET = 6.0
UNDERLINE_HEIGHT = 1.8
HATCH_SIZE = 7.0
#: Deckkraft der nicht ausgewaehlten Kodierungen, sobald eine Kodierung aktiv ist.
DIM_OPACITY = 0.35


class OverlayRenderer:
    """Baut SVG-Overlays aus Ankern und Kodierungen."""

    def __init__(self, palette: Optional[Sequence[str]] = None,
                 category_colors: Optional[Dict[str, str]] = None):
        self.palette = list(palette) if palette else list(DEFAULT_PALETTE)
        self.category_colors = dict(category_colors) if category_colors else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_page_svg(
        self,
        page: int,
        anchors: Sequence[Anchor],
        codings: Sequence[Coding],
        opts: RenderOptions,
        page_size: Tuple[float, float] = (595.0, 842.0),
    ) -> str:
        """Erzeugt das SVG fuer eine Seite.

        Args:
            page: 1-basierte Seitennummer
            anchors: Anker (nur die mit passender Seite werden genutzt)
            codings: Kodierungen (per segment_id den Ankern zugeordnet)
            opts: Darstellungsoptionen
            page_size: (Breite, Hoehe) im PDF-Koordinatensystem
        """
        ok, errors = opts.validate()
        if not ok:
            raise ValueError("; ".join(errors))

        width, height = page_size
        svg = ET.Element(
            _q("svg"),
            {
                "class": "qca-overlay",
                "width": "100%",
                "height": "100%",
                "viewBox": f"0 0 {width} {height}",
                "preserveAspectRatio": "none",
            },
        )
        defs = ET.SubElement(svg, _q("defs"))

        highlight_layer = ET.SubElement(
            svg,
            _q("g"),
            {
                "class": "qca-highlights",
                "style": f"mix-blend-mode:{opts.blend};opacity:{opts.opacity}",
            },
        )
        margin_layer = ET.SubElement(
            svg, _q("g"), {"class": "qca-margin-overlay"}
        )

        page_anchors = [a for a in anchors if a.page == page and a.anchored and a.quads]
        by_region = self._group_by_region(page, page_anchors, codings)
        colors = self._assign_colors(codings)
        emphasize = bool(opts.active_coding_id) and any(
            coding.segment_id == opts.active_coding_id for coding in codings
        )

        for _region, pairs in by_region.items():
            base_coding = self._active_coding(pairs, opts.active_coding_id)

            if opts.mode in ("highlight", "underline"):
                base_pairs = [p for p in pairs if p[0] is base_coding]
                other_pairs = [p for p in pairs if p[0] is not base_coding]
                for coding, anchor in base_pairs:
                    color = colors.get(coding.category, self.palette[0])
                    self._add_in_text(
                        highlight_layer, coding, anchor, color, opts,
                        opacity=_pair_emphasis(coding, anchor, opts, emphasize),
                    )
                for idx, (coding, anchor) in enumerate(other_pairs):
                    color = colors.get(coding.category, self.palette[0])
                    # Textmarker-Modus: weitere Kodierungen derselben Textstelle
                    # als Schraffur ueber die gleichen Quads - so bleibt die
                    # Flaeche deckungsgleich und beide Kategorien sind sichtbar.
                    if opts.mode == "highlight" and opts.hatch_shared:
                        self._add_hatch(
                            defs, highlight_layer, coding, anchor, color, opts,
                            opacity=_pair_emphasis(coding, anchor, opts, emphasize),
                        )
                    self._add_margin_stripe(margin_layer, coding, anchor, color, stack=idx)
            elif opts.mode == "margin":
                for idx, (coding, anchor) in enumerate(pairs):
                    color = colors.get(coding.category, self.palette[0])
                    self._add_margin_stripe(margin_layer, coding, anchor, color, stack=idx)
            elif opts.mode == "bracket":
                for idx, (coding, anchor) in enumerate(pairs):
                    color = colors.get(coding.category, self.palette[0])
                    self._add_bracket(margin_layer, coding, anchor, color, idx)

        return ET.tostring(svg, encoding="unicode")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _group_by_region(self, page: int, anchors: Sequence[Anchor], codings: Sequence[Coding]):
        """Gruppiert (coding, anchor)-Paare je Textstelle fuer die aktuelle Seite.

        Schluessel ist die Geometrie des Ankers, nicht die segment_id: derselbe
        Chunk kann mehrfach kodiert sein (z. B. Legitimationsstrategien *und*
        Strukturelle_Rahmenbedingungen). Liefe jede Kodierung als eigene
        Gruppe, wuerde die zweite Markierung exakt ueber der ersten liegen und
        sie unsichtbar machen. So bekommt die Textstelle ein In-Text-Highlight,
        die weiteren Kodierungen ihre eigenen Margin-Stripes.
        """
        groups: Dict[tuple, List[Tuple[Coding, Anchor]]] = {}

        # 1) Kanonisch: die Anker haengen an den Kodierungen - alle
        #    Regionen (Spec text-anker-export, R10). Eine Kodierung mit
        #    mehreren Textstellen liefert mehrere (coding, anchor)-Paare,
        #    auch auf verschiedenen Seiten.
        pairs_from_codings: List[Tuple[Coding, Anchor]] = []
        for coding in codings:
            for anchor in coding.anchors:
                if anchor.anchored and anchor.quads and anchor.page == page:
                    pairs_from_codings.append((coding, anchor))
        if pairs_from_codings:
            for coding, anchor in pairs_from_codings:
                groups.setdefault(region_key(anchor), []).append((coding, anchor))
            return groups

        # 2) Fallback: separate Ankerliste, positionsgleich zu den Kodierungen.
        if len(anchors) == len(codings):
            for coding, anchor in zip(codings, anchors):
                if anchor.anchored and anchor.quads and anchor.page == page:
                    groups.setdefault(region_key(anchor), []).append((coding, anchor))
        return groups

    def _active_coding(self, pairs, active_coding_id) -> Coding:
        """Waehlt die aktive Kodierung eines Segments (eine Instanz)."""
        if active_coding_id:
            for coding, _ in pairs:
                if coding.segment_id == active_coding_id:
                    return coding
        return pairs[0][0]

    def _add_in_text(self, layer, coding: Coding, anchor: Anchor, color: str,
                     opts: RenderOptions, opacity: Optional[float] = None):
        for quad in anchor.quads:
            x0, y0, x1, y1 = quad
            if opts.mode == "underline":
                rect = _add_rect(
                    layer, x0, max(y0, y1 - UNDERLINE_HEIGHT), x1 - x0, UNDERLINE_HEIGHT,
                    color, coding, opacity=opacity,
                )
            else:
                rect = _add_rect(
                    layer, x0, y0, x1 - x0, y1 - y0, color, coding, rx=1.5, opacity=opacity,
                )
            _decorate(rect, coding)

    def _add_hatch(self, defs, layer, coding: Coding, anchor: Anchor, color: str,
                   opts: RenderOptions, opacity: Optional[float] = None):
        """Schraffur ueber die Quads einer weiteren Kodierung derselben Stelle.

        Die Flaeche bleibt deckungsgleich mit dem Textmarker der Basiskodierung;
        die diagonale Schraffur macht die zweite Kategorie sichtbar, ohne die
        erste zu ueberdecken.
        """
        pattern_id = _hatch_id(coding.segment_id)
        if not any(pattern.get("id") == pattern_id for pattern in defs):
            pattern = ET.SubElement(
                defs,
                _q("pattern"),
                {
                    "id": pattern_id,
                    "patternUnits": "userSpaceOnUse",
                    "width": f"{HATCH_SIZE:.2f}",
                    "height": f"{HATCH_SIZE:.2f}",
                    "patternTransform": "rotate(45)",
                },
            )
            ET.SubElement(
                pattern,
                _q("rect"),
                {
                    "width": f"{HATCH_SIZE:.2f}",
                    "height": f"{HATCH_SIZE:.2f}",
                    "fill": color,
                    "opacity": "0.35",
                },
            )
            ET.SubElement(
                pattern,
                _q("line"),
                {
                    "x1": "0",
                    "y1": "0",
                    "x2": "0",
                    "y2": f"{HATCH_SIZE:.2f}",
                    "stroke": color,
                    "stroke-width": f"{HATCH_SIZE * 0.35:.2f}",
                    "opacity": "0.9",
                },
            )

        for quad in anchor.quads:
            x0, y0, x1, y1 = quad
            rect = _add_rect(
                layer, x0, y0, x1 - x0, y1 - y0, f"url(#{pattern_id})", coding,
                rx=1.5, opacity=opacity,
            )
            _decorate(rect, coding)

    def _add_margin_stripe(self, layer, coding: Coding, anchor: Anchor, color: str, stack: int = 0):
        x_min = min(q[0] for q in anchor.quads)
        y0 = min(q[1] for q in anchor.quads)
        y1 = max(q[3] for q in anchor.quads)
        x = max(1.0, x_min - MARGIN_OFFSET - (stack + 1) * (MARGIN_STRIPE_WIDTH + MARGIN_STRIPE_GAP))
        rect = _add_rect(layer, x, y0, MARGIN_STRIPE_WIDTH, max(2.0, y1 - y0), color, coding, rx=1.0)
        _decorate(rect, coding)

    def _add_bracket(self, layer, coding: Coding, anchor: Anchor, color: str, stack: int):
        x_min = min(q[0] for q in anchor.quads)
        y0 = min(q[1] for q in anchor.quads)
        y1 = max(q[3] for q in anchor.quads)
        x = max(1.0, x_min - MARGIN_OFFSET - stack * (MARGIN_STRIPE_WIDTH + MARGIN_STRIPE_GAP))
        tick = 3.0
        path_data = (
            f"M {x + tick:.2f} {y0:.2f} "
            f"L {x:.2f} {y0:.2f} "
            f"L {x:.2f} {y1:.2f} "
            f"L {x + tick:.2f} {y1:.2f}"
        )
        path = ET.SubElement(
            layer,
            _q("path"),
            {
                "d": path_data,
                "fill": "none",
                "stroke": color,
                "stroke-width": "1.6",
                "stroke-linecap": "round",
            },
        )
        _decorate(path, coding)

    # ------------------------------------------------------------------
    # Farben
    # ------------------------------------------------------------------

    def _assign_colors(self, codings: Sequence[Coding]) -> Dict[str, str]:
        colors: Dict[str, str] = {}
        palette_index = 0
        for coding in codings:
            category = coding.category
            if category in colors:
                continue
            if category in self.category_colors:
                colors[category] = self.category_colors[category]
            else:
                colors[category] = self.palette[palette_index % len(self.palette)]
                palette_index += 1
        return colors


# --------------------------------------------------------------------------
# SVG-Hilfsfunktionen
# --------------------------------------------------------------------------

def region_key(anchor: Anchor) -> tuple:
    """Identitaet einer Textstelle: Seite + gerundete Quad-Geometrie.

    Zwei Kodierungen desselben Chunks liefern identische Quads - unabhaengig
    von Segment-ID, Kategorie oder Zitatlaenge.
    """
    quads = anchor.quads or []
    return (
        anchor.page,
        tuple(
            (round(q[0], 1), round(q[1], 1), round(q[2], 1), round(q[3], 1))
            for q in quads
        ),
    )


def _q(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def _hatch_id(segment_id: str) -> str:
    """SVG-taugliche ID fuer das Schraffurmuster einer Kodierung."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", str(segment_id))
    return f"qca-hatch-{safe}"


def _emphasis(coding: Coding, opts: RenderOptions, active: bool) -> Optional[float]:
    """Deckkraft einer Kodierung, wenn eine andere ausgewaehlt ist.

    Die ausgewaehlte Kodierung bleibt voll sichtbar, alle anderen werden
    abgeschwaecht - so springt die Auswahl im Viewer sofort ins Auge.
    """
    if not active or not opts.active_coding_id:
        return None
    return 1.0 if coding.segment_id == opts.active_coding_id else DIM_OPACITY


def _pair_emphasis(
    coding: Coding, anchor: Anchor, opts: RenderOptions, active: bool
) -> Optional[float]:
    """Deckkraft einer konkreten Region (Spec text-anker-export, R13).

    Basis ist die Kodierungs-Deckkraft (_emphasis). Ist eine konkrete
    Region der aktiven Kodierung gewaehlt (active_anchor_index), werden
    die uebrigen Regionen derselben Kodierung abgeschwaecht - die
    gewaehlte Textstelle bleibt voll deckend.
    """
    base = _emphasis(coding, opts, active)
    if opts.active_anchor_index is None:
        return base
    if coding.segment_id != opts.active_coding_id:
        return base
    try:
        index = coding.anchors.index(anchor)
    except ValueError:
        return base
    return DIM_OPACITY if index != opts.active_anchor_index else None


def _add_rect(layer, x, y, w, h, color, coding: Coding, rx: float = 0.0,
              opacity: Optional[float] = None):
    attrs = {
        "x": f"{x:.2f}",
        "y": f"{y:.2f}",
        "width": f"{max(0.0, w):.2f}",
        "height": f"{max(0.0, h):.2f}",
        "fill": color,
    }
    if rx:
        attrs["rx"] = f"{rx:.2f}"
    if opacity is not None:
        attrs["opacity"] = f"{opacity:.2f}"
    attrs["data-coding-id"] = coding.segment_id
    attrs["data-segment-id"] = coding.segment_id
    attrs["role"] = "button"
    attrs["tabindex"] = "0"
    attrs["pointer-events"] = "auto"
    return ET.SubElement(layer, _q("rect"), attrs)


def _decorate(element, coding: Coding) -> None:
    element.set("aria-label", f"Kodierung {coding.category}: {coding.segment_id}")
    title = ET.SubElement(element, _q("title"))
    confidence = f" ({coding.confidence:.2f})" if coding.confidence is not None else ""
    title.text = f"{coding.category}{confidence}"


__all__ = ["OverlayRenderer", "DEFAULT_PALETTE", "SVG_NS", "region_key"]