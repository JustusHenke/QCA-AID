"""
Anchor Resolver
===============
Loest Kodierungen gegen ein Canonical_PDF in Anker auf: TextQuoteSelector
(quote/prefix/suffix) plus geometrische Quads je Textzeile
(siehe .kiro/specs/pdf-preview-annotation, Anforderungen 4.1-4.4, 4.7, 4.8).

Geometrie-Prinzip (SOTA):
  - Zeilenweise Quads statt einer Bounding-Box
  - Wort-Boxen aus `get_text("words")` (glyph-genau, Ascender/Descender)
  - Gruppierung je (block, line); horizontale Luecken (Spalten) splitten Runs
  - Fallback: `search_for(quote, quads=True)`

Relocation und Unanchored-Handling folgen in Task 7.
"""

import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .model import Anchor

logger = logging.getLogger(__name__)

# Ab dieser horizontalen Luecke (pt) gilt ein Zeilenabschnitt als neue Spalte.
DEFAULT_COLUMN_GAP_PT = 25.0
# Mindestaehnlichkeit fuer die Fuzzy-Relocation.
DEFAULT_FUZZY_THRESHOLD = 0.85


def _norm_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class AnchorResolver:
    """Loest Textabschnitte in Anker mit zeilenweisen Quads auf."""

    def __init__(self, column_gap_pt: float = DEFAULT_COLUMN_GAP_PT):
        self.column_gap_pt = column_gap_pt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        pdf_path: Path,
        doc_id: str,
        page: int,
        quote: str,
        prefix: str = "",
        suffix: str = "",
        page_label: Optional[str] = None,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        search_neighbors: bool = True,
    ) -> Anchor:
        """Loest einen Textabschnitt auf Seite `page` (1-basiert) auf.

        Reihenfolge: exakt auf der Seite -> fuzzy auf der Seite ->
        fuzzy auf Nachbarseiten -> unanchored.
        """
        import fitz

        with fitz.open(Path(pdf_path)) as doc:
            return self.resolve_in_doc(
                doc=doc,
                doc_id=doc_id,
                page=page,
                quote=quote,
                prefix=prefix,
                suffix=suffix,
                page_label=page_label,
                fuzzy_threshold=fuzzy_threshold,
                search_neighbors=search_neighbors,
            )

    def resolve_in_doc(
        self,
        doc,
        doc_id: str,
        page: int,
        quote: str,
        prefix: str = "",
        suffix: str = "",
        page_label: Optional[str] = None,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        search_neighbors: bool = True,
    ) -> Anchor:
        """Wie ``resolve``, aber mit bereits geöffnetem PyMuPDF-Dokument.

        Für gestaffelte Anker: mehrere Zitatlängen eines Chunks werden gegen
        dasselbe Dokument aufgeloest, ohne es je Stufe neu zu oeffnen.
        """
        anchor = Anchor(
            doc_id=doc_id,
            page=page,
            quote=quote,
            prefix=prefix,
            suffix=suffix,
            page_label=page_label,
            quads=[],
            anchored=False,
        )
        if not quote or not quote.strip():
            return anchor

        candidates = [page]
        if search_neighbors:
            candidates += [page - 1, page + 1]

        for candidate in candidates:
            if not (1 <= candidate <= doc.page_count):
                continue
            page_obj = doc.load_page(candidate - 1)

            quads = self.resolve_quads(page_obj, quote, prefix, suffix)
            if not quads:
                quads = self._quads_from_fuzzy_run(
                    page_obj, quote, prefix, suffix, fuzzy_threshold
                )
            if quads:
                anchor.quads = quads
                anchor.page = candidate
                # Seitenlabel nur behalten, wenn die Seite unveraendert ist.
                if candidate != page:
                    anchor.page_label = None
                break

        anchor.anchored = bool(anchor.quads)
        return anchor

    def resolve_quads(self, page, quote: str, prefix: str = "", suffix: str = "") -> List[List[float]]:
        """Liefert die Quads [x0, y0, x1, y1] fuer einen Textabschnitt."""
        quads = self._quads_from_word_run(page, quote, prefix, suffix)
        if quads:
            return quads
        return self._quads_from_search(page, quote)

    # ------------------------------------------------------------------
    # Strategie 1: Wort-Lauf (glyph-genau, spaltensicher)
    # ------------------------------------------------------------------

    def _quads_from_word_run(self, page, quote: str, prefix: str = "", suffix: str = "") -> List[List[float]]:
        target = _norm_tokens(quote)
        if not target:
            return []

        words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
        if not words:
            return []

        pdf_tokens = [_norm_word(w[4]) for w in words]
        matches = _find_exact_runs(pdf_tokens, target)
        if not matches:
            return []

        # Bei mehreren Treffern entscheidet der Textkontext (prefix/suffix).
        chosen = matches[0]
        if len(matches) > 1:
            prefix_tokens = _norm_tokens(prefix)
            suffix_tokens = _norm_tokens(suffix)
            if prefix_tokens or suffix_tokens:
                chosen = max(
                    matches,
                    key=lambda run: _context_score(
                        pdf_tokens, run[0], len(run), prefix_tokens, suffix_tokens
                    ),
                )

        matched = [words[i] for i in chosen]
        return self._line_quads(matched)

    def _line_quads(self, words: Sequence[tuple]) -> List[List[float]]:
        """Gruppiert Wort-Boxen je Zeile und splittet an Spaltenluecken."""
        lines: dict = {}
        for w in words:
            key = (w[5], w[6])  # (block_no, line_no)
            lines.setdefault(key, []).append(w)

        quads: List[List[float]] = []
        for key in sorted(lines.keys()):
            row = sorted(lines[key], key=lambda w: w[0])
            for run in _split_runs(row, self.column_gap_pt):
                x0 = min(w[0] for w in run)
                y0 = min(w[1] for w in run)
                x1 = max(w[2] for w in run)
                y1 = max(w[3] for w in run)
                quads.append([round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)])
        return quads

    # ------------------------------------------------------------------
    # Strategie 2: search_for (Fallback)
    # ------------------------------------------------------------------

    def _quads_from_search(self, page, quote: str) -> List[List[float]]:
        try:
            hits = page.search_for(quote, quads=True)
        except Exception:
            return []
        quads: List[List[float]] = []
        for hit in hits:
            rect = hit.rect
            quads.append(
                [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)]
            )
        return quads

    # ------------------------------------------------------------------
    # Strategie 3: Fuzzy-Relocation (quote + prefix/suffix)
    # ------------------------------------------------------------------

    def _quads_from_fuzzy_run(
        self,
        page,
        quote: str,
        prefix: str,
        suffix: str,
        threshold: float,
    ) -> List[List[float]]:
        target = _norm_tokens(quote)
        if not target:
            return []
        words = page.get_text("words")
        if not words:
            return []
        pdf_tokens = [_norm_word(w[4]) for w in words]
        best = _find_best_fuzzy_run(
            pdf_tokens, target, _norm_tokens(prefix), _norm_tokens(suffix), threshold
        )
        if best is None:
            return []
        start, end = best
        return self._line_quads(words[start:end])


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------

def _norm_word(word: str) -> str:
    tokens = _norm_tokens(word)
    return tokens[0] if len(tokens) == 1 else "".join(tokens)


def _find_exact_runs(pdf_tokens: Sequence[str], target: Sequence[str]) -> List[List[int]]:
    """Findet alle exakten Vorkommen von target als fortlaufenden Lauf."""
    n = len(target)
    results: List[List[int]] = []
    for start in range(0, len(pdf_tokens) - n + 1):
        if list(pdf_tokens[start:start + n]) == list(target):
            results.append(list(range(start, start + n)))
    return results


def _split_runs(row: Sequence[tuple], gap_pt: float) -> List[List[tuple]]:
    """Splittet eine sortierte Wortzeile an grossen horizontalen Luecken."""
    if not row:
        return []
    runs: List[List[tuple]] = [[row[0]]]
    for prev, cur in zip(row, row[1:]):
        if cur[0] - prev[2] > gap_pt:
            runs.append([cur])
        else:
            runs[-1].append(cur)
    return runs


def _find_best_fuzzy_run(
    pdf_tokens: Sequence[str],
    target: Sequence[str],
    prefix_tokens: Sequence[str],
    suffix_tokens: Sequence[str],
    threshold: float,
):
    """Findet den besten fuzzy Lauf; Kontext (prefix/suffix) dient als Tie-Break."""
    n = len(target)
    if n == 0 or len(pdf_tokens) < n - 1:
        return None

    best_score = None
    best_span = None
    for size in (n - 1, n, n + 1):
        if size <= 0:
            continue
        last_start = len(pdf_tokens) - size
        if last_start < 0:
            continue
        for start in range(0, last_start + 1):
            window = pdf_tokens[start:start + size]
            ratio = SequenceMatcher(None, list(target), list(window)).ratio()
            if ratio < threshold:
                continue
            ctx = _context_score(pdf_tokens, start, size, prefix_tokens, suffix_tokens)
            score = (round(ratio, 4), round(ctx, 4))
            if best_score is None or score > best_score:
                best_score = score
                best_span = (start, start + size)
    return best_span


def _context_score(
    pdf_tokens: Sequence[str],
    start: int,
    size: int,
    prefix_tokens: Sequence[str],
    suffix_tokens: Sequence[str],
) -> float:
    """Aehnlichkeit des Textkontexts links/rechts eines Laufs."""
    ctx = 0.0
    if prefix_tokens:
        ps = start - len(prefix_tokens)
        if ps >= 0:
            ctx += SequenceMatcher(None, list(prefix_tokens), pdf_tokens[ps:start]).ratio()
    if suffix_tokens:
        se = start + size
        if se + len(suffix_tokens) <= len(pdf_tokens):
            ctx += SequenceMatcher(
                None, list(suffix_tokens), pdf_tokens[se:se + len(suffix_tokens)]
            ).ratio()
    return ctx


__all__ = ["AnchorResolver", "DEFAULT_COLUMN_GAP_PT"]