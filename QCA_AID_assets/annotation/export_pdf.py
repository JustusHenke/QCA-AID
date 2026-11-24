"""
PDF Baking Exporter
===================
Schreibt Kodierungen dauerhaft in ein PDF: Highlight-Annotationen mit
/QuadPoints, Multiply-Blend, Legenden-Seite und Marginalien
(siehe .kiro/specs/pdf-preview-annotation, Anforderungen 8.1-8.4).

Unterschied zum OverlayRenderer (Task 9): dieser Exporter erzeugt ein
eigenstaendiges, viewer-unabhaengiges PDF (Archiv/Weitergabe), waehrend der
Renderer nur zur Anzeige dient.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .model import Anchor, Coding
from .renderer import DEFAULT_PALETTE, region_key

logger = logging.getLogger(__name__)

HIGHLIGHT_OPACITY = 0.4
BLEND_MODE = "Multiply"

#: Schraffur fuer weitere Kodierungen derselben Textstelle (Linienabstand in pt).
HATCH_SPACING = 6.0
HATCH_WIDTH = 1.1
HATCH_OPACITY = 0.6


class PdfBakingExporter:
    """Bettet Kodierungen als Annotationen in ein PDF ein."""

    def __init__(self, category_colors: Optional[Dict[str, str]] = None,
                 opacity: float = HIGHLIGHT_OPACITY):
        self.category_colors = dict(category_colors) if category_colors else {}
        self.opacity = opacity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bake(
        self,
        pdf_path: Path,
        codings: Sequence[Coding],
        out_path: Path,
        anchors: Optional[Sequence[Anchor]] = None,
        legend: bool = True,
        margin_notes: bool = True,
    ) -> Path:
        """Erzeugt ein annotiertes PDF.

        Args:
            pdf_path: Quell-PDF (Canonical_PDF)
            codings: Kodierungen (Anker haengt i. d. R. an der Kodierung)
            out_path: Zielpfad
            anchors: optionale zusaetzliche Anker (falls nicht an Kodierungen)
            legend: Legenden-Seite einfuegen
            margin_notes: Marginalien (Text-Notizen) hinzufuegen
        """
        import fitz

        pdf_path = Path(pdf_path)
        out_path = Path(out_path)
        doc = fitz.open(pdf_path)

        pairs = self._pairs(codings, anchors)
        colors = self._assign_colors(codings)
        page_size = (doc[0].rect.width, doc[0].rect.height) if doc.page_count else (595, 842)

        self._annotate(doc, fitz, pairs, colors, margin_notes)

        if legend:
            self._add_legend_page(doc, fitz, colors, page_size)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_path), garbage=3, deflate=True)
        doc.close()
        return out_path

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _pairs(self, codings, anchors):
        pairs: List[tuple] = []
        for coding in codings:
            if coding.anchor is not None:
                pairs.append((coding, coding.anchor))
        if anchors:
            for anchor in anchors:
                if anchor.anchored and anchor.quads:
                    pairs.append((None, anchor))
        return pairs

    @staticmethod
    def _group_regions(pairs) -> List[List[tuple]]:
        """Gruppiert die Paare nach Textstelle (Seite + Quad-Geometrie).

        Mehrfachkodierungen desselben Chunks teilen die Quads. Laege fuer jede
        eine eigene Highlight-Annotation, wuerde die zweite die erste exakt
        ueberdecken - im PDF sieht man dann nur eine Kodierung.
        """
        regions: Dict[tuple, List[tuple]] = {}
        for pair in pairs:
            _coding, anchor = pair
            regions.setdefault(region_key(anchor), []).append(pair)
        return list(regions.values())

    def _annotate(self, doc, fitz, pairs, colors, margin_notes: bool) -> int:
        count = 0
        for region in self._group_regions(pairs):
            base_coding, base_anchor = region[0]
            if not (base_anchor.anchored and base_anchor.quads):
                continue
            if not (1 <= base_anchor.page <= doc.page_count):
                continue
            page = doc.load_page(base_anchor.page - 1)
            category = base_coding.category if base_coding is not None else ""
            color = colors.get(category, (0.8, 0.8, 0.8))

            rects = [fitz.Rect(q[0], q[1], q[2], q[3]) for q in base_anchor.quads]
            annot = page.add_highlight_annot(rects)
            annot.set_colors(stroke=color)
            annot.set_opacity(self.opacity)
            annot.set_blendmode(BLEND_MODE)
            annot.set_info(content=self._info_text(base_coding, base_anchor))
            annot.update()
            count += 1

            if margin_notes:
                self._add_margin_note(page, fitz, base_anchor, base_coding)

            # Weitere Kodierungen derselben Textstelle: Schraffur in ihrer
            # Kategoriefarbe, damit beide Kodierungen sichtbar bleiben.
            for coding, anchor in region[1:]:
                if not (anchor.anchored and anchor.quads):
                    continue
                hatch_color = colors.get(
                    coding.category if coding is not None else "", (0.8, 0.8, 0.8)
                )
                self._draw_hatch(page, fitz, anchor, hatch_color)
                if margin_notes:
                    self._add_margin_note(page, fitz, anchor, coding)

        return count

    def _draw_hatch(self, page, fitz, anchor: Anchor, color: tuple) -> None:
        """Zeichnet eine 45-Grad-Schraffur in die Quads einer Kodierung."""
        shape = page.new_shape()
        for quad in anchor.quads:
            x0, y0, x1, y1 = quad
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            # Linien der Form y = c - x, auf das Quad beschnitten.
            c = x0 + y0 + HATCH_SPACING
            while c < x1 + y1:
                xa = max(x0, c - y1)
                xb = min(x1, c - y0)
                if xb - xa > 0.5:
                    shape.draw_line(
                        fitz.Point(xa, c - xa), fitz.Point(xb, c - xb)
                    )
                c += HATCH_SPACING
        shape.finish(
            color=color,
            width=HATCH_WIDTH,
            stroke_opacity=HATCH_OPACITY,
            closePath=False,
        )
        shape.commit(overlay=True)

    @staticmethod
    def _info_text(coding: Optional[Coding], anchor: Anchor) -> str:
        lines = [f"Kategorie: {coding.category if coding else '(ohne)'}"]
        if coding is not None:
            if coding.subcategories:
                lines.append("Subkategorien: " + ", ".join(coding.subcategories))
            if coding.confidence is not None:
                lines.append(f"Konfidenz: {coding.confidence:.2f}")
            if coding.justification:
                lines.append(f"Begruendung: {coding.justification}")
        lines.append(f"Zitat: {anchor.quote[:200]}")
        return "\n".join(lines)

    def _add_margin_note(self, page, fitz, anchor: Anchor, coding: Optional[Coding]) -> None:
        x_min = min(q[0] for q in anchor.quads)
        y_top = min(q[1] for q in anchor.quads)
        point = fitz.Point(max(2.0, x_min - 30.0), y_top)
        text = (coding.category if coding else "Kodierung")
        if coding is not None and coding.justification:
            text += ": " + coding.justification[:120]
        note = page.add_text_annot(point, text, icon="Note")
        note.set_info(title="QCA-AID")
        note.update()

    # ------------------------------------------------------------------
    # Legende
    # ------------------------------------------------------------------

    def _add_legend_page(self, doc, fitz, colors, page_size) -> None:
        # Legende ans Ende: die Seitenzahlen des Dokuments bleiben so
        # unveraendert (Seitenverweise aus dem Pruefen-Tab stimmen weiter).
        page = doc.new_page(width=page_size[0], height=page_size[1])
        page.insert_text((50, 60), "QCA-AID Kategorien-Legende", fontsize=18)
        y = 110
        for category, color in sorted(colors.items()):
            rect = fitz.Rect(50, y - 12, 80, y + 2)
            page.draw_rect(rect, color=color, fill=color)
            page.insert_text((90, y), category, fontsize=12)
            y += 26
        page.insert_text((50, y + 20),
                         "Highlights: kodierte Textstellen (Multiply-Blend).",
                         fontsize=10)
        page.insert_text((50, y + 36),
                         "Schraffur: weitere Kodierung derselben Textstelle.",
                         fontsize=10)
        page.insert_text((50, y + 52),
                         "Marginalien: Kategorie und Begruendung je Kodierung.",
                         fontsize=10)

    # ------------------------------------------------------------------
    # Farben
    # ------------------------------------------------------------------

    def _assign_colors(self, codings) -> Dict[str, tuple]:
        colors: Dict[str, tuple] = {}
        index = 0
        for coding in codings:
            category = coding.category
            if category in colors:
                continue
            hex_color = self.category_colors.get(category) or DEFAULT_PALETTE[
                index % len(DEFAULT_PALETTE)
            ]
            index += 1
            colors[category] = _hex_to_rgb(hex_color)
        return colors


def _hex_to_rgb(hex_color: str) -> tuple:
    value = hex_color.lstrip("#")
    try:
        r = int(value[0:2], 16) / 255.0
        g = int(value[2:4], 16) / 255.0
        b = int(value[4:6], 16) / 255.0
        return (round(r, 4), round(g, 4), round(b, 4))
    except (ValueError, IndexError):
        return (0.8, 0.8, 0.8)


__all__ = [
    "PdfBakingExporter",
    "HIGHLIGHT_OPACITY",
    "BLEND_MODE",
    "HATCH_SPACING",
    "HATCH_WIDTH",
    "HATCH_OPACITY",
]