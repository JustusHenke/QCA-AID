"""
Coding Panel
============
Kodierungsliste mit Status und Aktionen (Confirm/Change/Reject)
(siehe .kiro/specs/pdf-preview-annotation, Anforderungen 6.3-6.5, 7.1-7.3).

Reine Logik (headless testbar):
  - build_coding_rows: gefilterte, sortierte Anzeigezeilen
  - panel_summary: Kennzahlen (proposed/accepted/rejected, unanchored)
  - apply_action: Statusuebergaenge und Feld-Aenderungen

Duenn (Streamlit):
  - render_coding_panel: rendert Zeilen mit Popover und Aktionsbuttons
"""

import logging
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence

from QCA_AID_assets.annotation.model import Coding
from .code_options import CodeOption, code_label, option_index, parse_code_text

logger = logging.getLogger(__name__)

ACTION_ACCEPT = "accept"
ACTION_REJECT = "reject"
ACTION_CHANGE = "change"
ALLOWED_ACTIONS = (ACTION_ACCEPT, ACTION_REJECT, ACTION_CHANGE)


@dataclass
class CodingRow:
    """Anzeigezeile einer Kodierung."""

    segment_id: str
    category: str
    subcategories: List[str]
    confidence: Optional[float]
    status: str
    page: Optional[int]
    anchored: bool
    active: bool
    justification: Optional[str] = None
    inference: bool = False
    quote: str = ""
    same_region: int = 1
    blocks: int = 1


def _page_of(coding: Coding) -> Optional[int]:
    if coding.page is not None:
        return coding.page
    if coding.anchor is not None:
        return coding.anchor.page
    return None


def _region_key(coding: Coding) -> Optional[tuple]:
    """Identitaet der Textstelle (Seite + Quad-Geometrie).

    Mehrfachkodierungen desselben Chunks teilen sich Anker und Quads - sie
    zeigen also auf dieselbe Textstelle und werden im Panel als solche
    gekennzeichnet.
    """
    anchor = coding.anchor
    if anchor is None or not anchor.quads:
        return None
    return (
        anchor.page,
        tuple(
            (round(q[0], 1), round(q[1], 1), round(q[2], 1), round(q[3], 1))
            for q in anchor.quads
        ),
    )


def _region_counts(codings: Sequence[Coding]) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    for coding in codings:
        key = _region_key(coding)
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _spread(anchor, gap_pt: float = 24.0) -> int:
    """Anzahl zusammenhaengender Bloecke, die eine Markierung abdeckt.

    Chunks, die aus mehreren Spalten/Absaetzen zusammengesetzt sind, ergeben
    eine Markierung mit Spruengen ueber die Seite. Das ist kein Fehler, sollte
    aber sichtbar sein ("Markierung verteilt").
    """
    quads = list(getattr(anchor, "quads", None) or ())
    if len(quads) <= 1:
        return 1 if quads else 0
    ordered = sorted(quads, key=lambda q: (round(q[1], 1), q[0]))
    blocks = 1
    prev_bottom = ordered[0][3]
    for quad in ordered[1:]:
        if quad[1] - prev_bottom > gap_pt:
            blocks += 1
        prev_bottom = max(prev_bottom, quad[3])
    return blocks


def _quote_preview(text: Optional[str], limit: int = 110) -> str:
    """Einzeilige Vorschau des Zitats (fuer die Panelzeile)."""
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + " …"


def panel_regions(codings: Sequence[Coding], page: Optional[int] = None) -> int:
    """Anzahl unterschiedlicher Textstellen (Seite optional gefiltert)."""
    keys = set()
    for coding in codings:
        if page is not None and _page_of(coding) != page:
            continue
        key = _region_key(coding)
        if key is not None:
            keys.add(key)
    return len(keys)


def panel_header(codings: Sequence[Coding], page: Optional[int] = None) -> str:
    """Kopfzeile des Panels - bei Seitenbindung nur die Zahlen dieser Seite."""
    visible = [
        coding for coding in codings if page is None or _page_of(coding) == page
    ]
    summary = panel_summary(visible)
    regions = panel_regions(visible)
    scope = f" auf S. {page}" if page is not None else ""
    return (
        f"**Kodierungen{scope}** — {summary['total']} · "
        f"{regions} Textstelle{'n' if regions != 1 else ''} · "
        f"{summary['proposed']} Vorschläge · {summary['accepted']} bestätigt · "
        f"{summary['rejected']} verworfen · {summary['unanchored']} ohne Anker"
    )


def build_coding_rows(
    codings: Sequence[Coding],
    active_id: Optional[str] = None,
    page: Optional[int] = None,
    status: Optional[str] = None,
) -> List[CodingRow]:
    """Baut die Anzeigezeilen, gefiltert nach Seite/Status und sortiert."""
    region_counts = _region_counts(codings)
    rows: List[CodingRow] = []
    for coding in codings:
        coding_page = _page_of(coding)
        if page is not None and coding_page != page:
            continue
        if status is not None and coding.status != status:
            continue
        key = _region_key(coding)
        rows.append(
            CodingRow(
                segment_id=coding.segment_id,
                category=coding.category,
                subcategories=list(coding.subcategories),
                confidence=coding.confidence,
                status=coding.status,
                page=coding_page,
                anchored=bool(coding.anchor and coding.anchor.anchored),
                active=coding.segment_id == active_id,
                justification=coding.justification,
                inference=bool(coding.inference),
                quote=_quote_preview(coding.anchor.quote if coding.anchor else ""),
                same_region=region_counts.get(key, 1) if key is not None else 1,
                blocks=_spread(coding.anchor) or 1,
            )
        )
    rows.sort(key=lambda r: (r.page if r.page is not None else 0, r.segment_id))
    return rows


def panel_summary(codings: Sequence[Coding]) -> Dict[str, int]:
    """Kennzahlen fuer den Panel-Kopf."""
    summary = {"total": len(codings), "proposed": 0, "accepted": 0, "rejected": 0, "unanchored": 0}
    for coding in codings:
        if coding.status in ("proposed", "accepted", "rejected"):
            summary[coding.status] += 1
        if not (coding.anchor and coding.anchor.anchored):
            summary["unanchored"] += 1
    return summary


def apply_action(
    coding: Coding,
    action: str,
    category: Optional[str] = None,
    subcategories: Optional[Sequence[str]] = None,
) -> Coding:
    """Wendet eine Nutzeraktion an und liefert eine neue Kodierung.

    Raises:
        ValueError: bei unbekannter Aktion.
    """
    if action == ACTION_ACCEPT:
        return replace(coding, status="accepted")
    if action == ACTION_REJECT:
        return replace(coding, status="rejected")
    if action == ACTION_CHANGE:
        updates: Dict[str, object] = {}
        if category is not None:
            updates["category"] = category
        if subcategories is not None:
            updates["subcategories"] = list(subcategories)
        return replace(coding, **updates)
    raise ValueError(f"unbekannte Aktion: {action}")


# --------------------------------------------------------------------------
# Streamlit-Renderer (duenn)
# --------------------------------------------------------------------------

_STATUS_LABEL = {"proposed": "Vorschlag", "accepted": "Bestätigt", "rejected": "Verworfen"}


def _render_recode_controls(
    row: CodingRow,
    options: List[CodeOption],
    key_prefix: str,
    on_recode: Callable[[str, CodeOption], None],
) -> None:
    """Auswahl eines anderen Codes bzw. Aenderung der Subcodes."""
    import streamlit as st

    current = code_label(row.category, row.subcategories)
    st.markdown("**Code ändern**")
    st.caption(f"Aktuell: {current or '—'}")

    if options:
        labels = [option.label for option in options]
        index = option_index(options, row.category, row.subcategories) or 0
        choice = st.selectbox(
            "Vorhandener Code",
            options=labels,
            index=index,
            key=f"{key_prefix}_code_{row.segment_id}",
            help="Kategorien und Subcodes aus dem Codebook und aus den geladenen Kodierungen.",
        )
        if st.button(
            "Diesen Code übernehmen",
            key=f"{key_prefix}_code_ok_{row.segment_id}",
            use_container_width=True,
        ):
            on_recode(row.segment_id, options[labels.index(choice)])

    st.caption("Neuer Code im Format `Kategorie › Subcode, Subcode2`:")
    custom = st.text_input(
        "Neuer Code",
        value="",
        key=f"{key_prefix}_code_new_{row.segment_id}",
        label_visibility="collapsed",
        placeholder="z.B. Steuerung › Legitimität",
    )
    if st.button(
        "Anlegen und übernehmen",
        key=f"{key_prefix}_code_new_ok_{row.segment_id}",
        use_container_width=True,
        disabled=not custom.strip(),
    ):
        option = parse_code_text(custom)
        if option is None:
            st.warning("Bitte zuerst eine Kategorie angeben.")
        else:
            on_recode(row.segment_id, option)


def render_coding_panel(
    codings: Sequence[Coding],
    active_id: Optional[str] = None,
    page: Optional[int] = None,
    key_prefix: str = "coding",
    on_action: Optional[Callable[[str, str], None]] = None,
    code_options: Optional[Sequence[CodeOption]] = None,
    on_recode: Optional[Callable[[str, CodeOption], None]] = None,
    on_select: Optional[Callable[[str], None]] = None,
) -> None:
    """Rendert die Kodierungsliste (Streamlit).

    Args:
        on_action: Callback (segment_id, action) fuer Confirm/Reject.
        code_options: Auswaehlbare Codes (Codebook + vorhandene Kodierungen).
            Ohne sie steht nur der Freitext fuer neue Codes zur Verfuegung.
        on_recode: Callback (segment_id, CodeOption) fuer "anderen Code
            vergeben" / "Subcode aendern". Der Dialog bleibt inline und hat
            einen eigenen "Abbrechen"-Knopf.
        on_select: Callback (segment_id), wenn der Code angeklickt wird - die
            Kodierung wird aktiv und im Viewer hervorgehoben.
    """
    import streamlit as st

    rows = build_coding_rows(codings, active_id=active_id, page=page)
    visible = [c for c in codings if page is None or _page_of(c) == page]
    edit_key = f"{key_prefix}_editing"

    st.markdown(panel_header(codings, page=page))
    if page is not None and len(visible) != len(codings):
        st.caption(
            f"Insgesamt {len(codings)} Kodierungen im Dokument — "
            f"{len(visible)} davon auf dieser Seite."
        )

    if not rows:
        st.info("Keine Kodierungen auf dieser Seite.")
        return

    for row in rows:
        with st.container(border=True):
            if row.active:
                st.markdown(
                    '<div style="background:#dbeafe;border-left:4px solid #2563eb;'
                    'padding:4px 10px;border-radius:4px;margin-bottom:6px;'
                    'font-size:13px;">Aktive Kodierung</div>',
                    unsafe_allow_html=True,
                )
            cols = st.columns([3, 1, 1])
            with cols[0]:
                marker = "▶ " if row.active else ""
                anchor_note = "" if row.anchored else " ⚠︎ ohne Anker"
                if on_select is not None:
                    if st.button(
                        f"{marker}{row.category}{anchor_note}",
                        key=f"{key_prefix}_sel_{row.segment_id}",
                        type="tertiary",
                        use_container_width=True,
                        help="Diese Kodierung auswählen und im Viewer hervorheben",
                    ):
                        on_select(row.segment_id)
                else:
                    st.markdown(f"{marker}**{row.category}**{anchor_note}")
                if row.subcategories:
                    st.caption("Sub: " + ", ".join(row.subcategories))
                if row.quote:
                    st.caption(f"„{row.quote}“", help="Textstelle im Dokument")
                if row.same_region > 1:
                    st.caption(
                        f"↔ gleiche Textstelle wie {row.same_region - 1} weitere "
                        f"Kodierung{'en' if row.same_region > 2 else ''}"
                    )
                if row.blocks > 1:
                    st.caption(
                        f"⇅ Markierung verteilt über {row.blocks} Abschnitte",
                        help=(
                            "Der Chunk enthält Text aus mehreren Spalten bzw. Absätzen "
                            "(z. B. Fließtext und Abstract). Die Markierung springt "
                            "deshalb über die Seite."
                        ),
                    )
            with cols[1]:
                st.caption(_STATUS_LABEL.get(row.status, row.status))
                if row.page is not None:
                    st.caption(f"S. {row.page}")
            with cols[2]:
                if on_action is not None:
                    if st.button("✓", key=f"{key_prefix}_ok_{row.segment_id}",
                                 help="Bestätigen", disabled=row.status == "accepted"):
                        on_action(row.segment_id, ACTION_ACCEPT)
                    if st.button("✗", key=f"{key_prefix}_no_{row.segment_id}",
                                 help="Verwerfen", disabled=row.status == "rejected"):
                        on_action(row.segment_id, ACTION_REJECT)

            if on_recode is not None:
                editing = st.session_state.get(edit_key) == row.segment_id
                if st.button(
                    "✎ Ändern" if not editing else "✎ Ändern …",
                    key=f"{key_prefix}_edit_{row.segment_id}",
                    use_container_width=True,
                    help="Anderen Code vergeben oder Subcodes ändern",
                ):
                    st.session_state[edit_key] = None if editing else row.segment_id
                    st.rerun()
                if editing:
                    with st.container(border=True):

                        def _apply(segment_id: str, option: CodeOption) -> None:
                            st.session_state[edit_key] = None
                            on_recode(segment_id, option)

                        _render_recode_controls(
                            row,
                            list(code_options or ()),
                            key_prefix,
                            _apply,
                        )
                        if st.button(
                            "Abbrechen",
                            key=f"{key_prefix}_cancel_{row.segment_id}",
                            use_container_width=True,
                            help="Änderung verwerfen und Dialog schließen",
                        ):
                            st.session_state[edit_key] = None
                            st.rerun()

            with st.popover("Details", use_container_width=True):
                st.write(f"Segment: `{row.segment_id}`")
                if row.confidence is not None:
                    st.write(f"Konfidenz: {row.confidence:.2f}")
                st.write("Status: " + _STATUS_LABEL.get(row.status, row.status))
                if row.justification:
                    st.caption(f"Begründung: {row.justification}")
                if row.inference:
                    st.caption("Inferenz-Kodierung (aus dem Kontext erschlossen)")


__all__ = [
    "CodingRow",
    "build_coding_rows",
    "panel_summary",
    "apply_action",
    "render_coding_panel",
    "ACTION_ACCEPT",
    "ACTION_REJECT",
    "ACTION_CHANGE",
]