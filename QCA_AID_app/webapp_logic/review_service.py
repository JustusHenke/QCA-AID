"""
Review Service
==============
Ueberfuehrt Analyseergebnisse in ankerbasierte Kodierungen und erzeugt die
SVG-Overlays fuer die Dokumentvorschau
(siehe .kiro/specs/pdf-preview-annotation, Anforderungen 2.1-2.6, 4.1-4.6, 6.1-6.2).

Datenfluss:
  Analyse-XLSX --load_codings_from_xlsx--> Coding (ohne Anker)
               --anchor_codings---------> Coding mit Anchor (quads)
               --build_overlay_map------> {Seite: SVG}

Die Zitatauswahl ist bewusst konservativ: aus dem Segmenttext wird ein
zusammenhaengender Ausschnitt gewaehlt, der im PDF wortgetreu auffindbar ist.
Findet sich kein Ausschnitt, bleibt die Kodierung ohne Anker (``anchor=None``)
und wird im Panel als "ohne Anker" markiert - nie falsch platziert.

Headless testbar: keine Streamlit-Imports.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from QCA_AID_assets.annotation.anchor import AnchorResolver
from QCA_AID_assets.annotation.model import (
    Anchor,
    Coding,
    DocumentIndex,
    RenderOptions,
)
from QCA_AID_assets.annotation.renderer import OverlayRenderer

logger = logging.getLogger(__name__)

#: Maximale Laenge des Zitats (Zeichen). Laengere Zitate werden an einer
#: Wortgrenze gekuerzt, damit die Markierung lesbar bleibt.
QUOTE_MAX_CHARS = 240

#: Gestaffelte Ziellaengen fuer Anker, von lang nach kurz probiert
#: (Variante 1: gestufte Zitatlaengen mit Fallback). Ein Chunk wird moeglichst
#: weit markiert - die erste Stufe deckt den ganzen Chunk ab, solange er in
#: eine Seite passt; scheitert eine Stufe im PDF (Silbentrennung, Layout,
#: seitenuebergreifender Chunk), greift die naechste. Die letzte Stufe ist
#: der konservative Kopf - die Kodierung wird nie "ohne Anker" nur weil ein
#: langer Abschnitt nicht 1:1 auffindbar ist.
QUOTE_STAGES: Tuple[int, ...] = (1200, 800, 500, QUOTE_MAX_CHARS)

#: Obergrenze fuer den in der Kodierung mitgefuehrten Originaltext (Puffer,
#: damit die Stufen aus dem vollen Chunktext arbeiten koennen).
QUOTE_SOURCE_LIMIT = 4000

#: Trennzeichen fuer Subkategorien in der Analyse-XLSX.
SUBCATEGORY_SEPARATOR = ";"

#: Spaltennamen der Analyse-XLSX (mit Fallbacks fuer aeltere Exporte).
COL_DOCUMENT = ("Dokument", "Datei", "Quelle")
COL_TEXT = ("Text", "Textstelle", "Zitat", "Segment")
COL_CATEGORY = ("Hauptkategorie", "Kategorie")
COL_SUBCATEGORIES = ("Subkategorien", "Subkategorie")
COL_JUSTIFICATION = ("Begründung", "Begruendung", "Rechtfertigung")
COL_CONFIDENCE = ("Konfidenz", "Confidence")
COL_CODED = ("Kodiert",)
COL_SEGMENT = ("Original_Chunk_ID", "Chunk_Nr", "Segment_ID", "Segment")
COL_INFERENCE = ("Inferenz", "Inference")

#: Spalten, die der Pruefen-Tab beim Zurueckschreiben in die Analyse-XLSX
#: anlegt (siehe webapp_logic.review_writeback). Liegen sie vor, gelten sie
#: als aktueller Stand - die Bestandsspalten bleiben der Ursprungszustand.
COL_STATUS = ("Review_Status",)
COL_REVIEW_CATEGORY = ("Hauptkategorie_Review",)
COL_REVIEW_SUBCATEGORIES = ("Subkategorien_Review",)

#: Pfad-/Dateiartefakte, die nie als Zitat taugen (Anforderung: ehrliche Anker).
_PATH_ARTIFACT_RE = re.compile(
    r"(file:///\S+|[A-Za-z]:\\[^\s]+|/Users/\S+|/home/\S+|OneDrive\S*|\.(?:pdf|docx?|txt|md)\b)",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")

#: Werte, die "nicht kodiert" bedeuten.
_FALSE_VALUES = {"", "0", "false", "nein", "no", "nan", "none", "nicht kodiert"}


# ---------------------------------------------------------------------------
# Dokumentnamen
# ---------------------------------------------------------------------------


def normalize_doc_name(value: Any) -> str:
    r"""Vergleichsform eines Dokumentnamens: ohne Pfad, Endung, Gross/Klein.

    ``C:\...\input\Text_Interview.DOCX`` und ``text_interview`` sind damit
    derselbe Name. So schlaegt beim Laden kein Kodierungssatz fehl, nur weil
    die Analyse den Namen mit Pfad oder Endung schreibt.
    """
    text = _as_text(value)
    if not text:
        return ""
    text = text.replace("\\", "/").rsplit("/", 1)[-1]
    return Path(text).stem.casefold().strip()


# ---------------------------------------------------------------------------
# Zitatauswahl
# ---------------------------------------------------------------------------


def normalize_ws(text: str) -> str:
    """Vereinheitlicht Whitespace fuer den Vergleich mit dem PDF-Text."""
    return _WHITESPACE_RE.sub(" ", (text or "").replace("\u00ad", "")).strip()


def select_quote(text: str, max_chars: int = QUOTE_MAX_CHARS) -> str:
    """Waehlt einen zitierfaehigen Ausschnitt aus einem Segmenttext.

    Entfernt Pfad-/Dateiartefakte und kuerzt an einer Wortgrenze. Gibt einen
    leeren String zurueck, wenn nach der Bereinigung nichts uebrig bleibt.
    """
    if not text:
        return ""
    cleaned = _PATH_ARTIFACT_RE.sub(" ", str(text))
    cleaned = normalize_ws(cleaned)
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned

    cut = cleaned[:max_chars]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.strip()


def quote_stages(text: str, stages: Sequence[int] = QUOTE_STAGES) -> List[str]:
    """Zitierfaehige Kandidaten eines Chunktexts, von lang nach kurz.

    Die Stufen sind Praefixe desselben bereinigten Texts (Kuerzung jeweils an
    einer Wortgrenze) - der Seitenkopf ist also identisch, und die
    Seitenfindung gilt fuer alle Stufen gleichermaßen.
    """
    candidates: List[str] = []
    for max_chars in stages:
        candidate = select_quote(text, max_chars=max_chars)
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def find_page_for_quote(
    index: Optional[DocumentIndex],
    quote: str,
    min_head_chars: int = 40,
) -> Optional[int]:
    """Findet die Seite, auf der das Zitat (normalisiert) beginnt.

    Massgeblich ist der Anfang des Zitats: laeuft es ueber das Seitenende
    hinaus, reicht ein uebereinstimmender Kopf von ``min_head_chars`` Zeichen.
    So landet eine Kodierung auf der Seite, auf der sie *anfaengt* - nie auf
    einer Folgeseite, auf der nur noch der Rest steht.
    """
    if index is None or not quote:
        return None
    target = normalize_ws(quote)
    if not target:
        return None

    seed_length = min(min_head_chars, len(target))
    seed = target[:seed_length]
    # An einer Wortgrenze enden: der normalisierte Seitentext hat am Ende kein
    # Leerzeichen, ein roher Zeichen-Schnitt wuerde dort scheitern.
    if seed_length < len(target) and " " in seed:
        seed = seed.rsplit(" ", 1)[0]
    seed = seed.strip()
    if not seed or len(seed) < min(20, len(target)):
        return None

    required = len(seed)
    for page in index.pages:
        text = normalize_ws(page.text)
        position = text.find(seed)
        if position < 0:
            continue
        common = 0
        limit = min(len(target), len(text) - position)
        while common < limit and target[common] == text[position + common]:
            common += 1
        if common >= required:
            return page.page
    return None


# ---------------------------------------------------------------------------
# XLSX -> Codings
# ---------------------------------------------------------------------------


def _first(record: Mapping[str, Any], columns: Sequence[str]) -> Any:
    for column in columns:
        if column in record:
            value = record[column]
            if value is not None:
                return value
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _FALSE_VALUES:
        return False
    return text in {"1", "true", "ja", "yes", "x", "wahr"}


def _as_confidence(value: Any) -> Optional[float]:
    """Liest Konfidenz aus Zahl, String oder verschachteltem Dict."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("total", value.get("overall"))
        if value is None:
            return None
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    if 0.0 <= number <= 1.0:
        return number
    if 0.0 <= number <= 100.0:
        return number / 100.0
    return None


def _split_subcategories(value: Any) -> List[str]:
    text = _as_text(value)
    if not text:
        return []
    parts = re.split(rf"[{re.escape(SUBCATEGORY_SEPARATOR)},]", text)
    return [part.strip() for part in parts if part.strip()]


def _apply_review_state(
    record: Mapping[str, Any],
    category: str,
    subcategories: List[str],
) -> Tuple[str, List[str], str, str]:
    """Übernimmt zurückgeschriebene Review-Spalten einer XLSX-Zeile.

    Liegt ``Hauptkategorie_Review`` vor, ist das der aktuelle Code; die
    Bestandsspalten bleiben der Ursprungszustand (Grundlage für vorher/nachher).
    ``Review_Status`` wird auf den Coding-Status abgebildet.

    Returns:
        (Kategorie, Subkategorien, Status, created_by)
    """
    from .review_writeback import CODING_BY_STATUS

    review_category = _as_text(_first(record, COL_REVIEW_CATEGORY))
    status = CODING_BY_STATUS.get(_as_text(_first(record, COL_STATUS)).casefold(), "proposed")

    if not review_category or review_category.lower() in _FALSE_VALUES:
        # Ohne Review-Spalten bleibt die Kodierung ein offener Vorschlag.
        return category, subcategories, "proposed", "ai"

    review_subcategories = _split_subcategories(_first(record, COL_REVIEW_SUBCATEGORIES))
    changed = normalize_ws(review_category).casefold() != normalize_ws(category).casefold()
    created_by = "human" if changed else "ai"
    return review_category, review_subcategories, status, created_by


def coding_from_record(
    record: Mapping[str, Any],
    doc_id: str,
    fallback_segment_id: str,
) -> Optional[Coding]:
    """Baut eine Kodierung aus einer XLSX-Zeile; None wenn nicht kodiert."""
    if "Kodiert" in record and not _as_bool(record.get("Kodiert"), default=True):
        return None

    category = _as_text(_first(record, COL_CATEGORY))
    if not category or category.lower() in _FALSE_VALUES:
        return None

    subcategories = _split_subcategories(_first(record, COL_SUBCATEGORIES))
    category, subcategories, status, created_by = _apply_review_state(
        record, category, subcategories
    )

    raw_text = _as_text(_first(record, COL_TEXT))
    # Voller bereinigter Text (Puffer QUOTE_SOURCE_LIMIT): anchor_codings
    # probiert daraus gestaffelte Zitatlaengen - der 240-Zeichen-Kopf ist nur
    # die letzte Stufe, nicht das Zitat selbst.
    quote = select_quote(raw_text, max_chars=QUOTE_SOURCE_LIMIT)
    if not quote:
        return None

    justification = _as_text(_first(record, COL_JUSTIFICATION)) or None
    inference = _as_bool(_first(record, COL_INFERENCE), default=False)
    # Anforderung 7.5: Inferenz-Kodierungen brauchen immer eine Begruendung.
    if inference and not justification:
        inference = False

    return Coding(
        segment_id=_as_text(_first(record, COL_SEGMENT)) or fallback_segment_id,
        doc_id=doc_id,
        category=category,
        subcategories=subcategories,
        confidence=_as_confidence(_first(record, COL_CONFIDENCE)),
        justification=justification,
        created_by=created_by,
        status=status,
        inference=inference,
        # Der Anker traegt den TextQuoteSelector (quote) bereits, ist aber noch
        # nicht aufgeloest (anchored=False) - die Seite wird spaeter gesucht.
        anchor=Anchor(doc_id=doc_id, page=1, quote=quote, anchored=False),
    )


def codings_from_records(
    records: Iterable[Mapping[str, Any]],
    doc_id: str,
    doc_name: Optional[str] = None,
) -> List[Coding]:
    """Wandelt XLSX-Zeilen in Kodierungen (nur die des Dokuments).

    Dokumentnamen werden normalisiert verglichen (ohne Pfad, Endung und
    Gross/Klein). Ohne ``doc_name`` werden alle Zeilen uebernommen - dieser
    Modus ist fuer den stillen Fallback *nicht* gedacht.
    """
    codings: List[Coding] = []
    target = normalize_doc_name(doc_name) if doc_name is not None else None
    for position, record in enumerate(records, start=1):
        if target is not None:
            name = normalize_doc_name(_first(record, COL_DOCUMENT))
            if name and name != target:
                continue
        coding = coding_from_record(
            record, doc_id=doc_id, fallback_segment_id=f"{doc_id}_row_{position}"
        )
        if coding is not None:
            codings.append(coding)
    return codings


def document_names_in_source(records: Iterable[Mapping[str, Any]]) -> List[str]:
    """Normalisierte Dokumentnamen, die in der Kodierquelle vorkommen."""
    names: List[str] = []
    for record in records:
        name = normalize_doc_name(_first(record, COL_DOCUMENT))
        if name and name not in names:
            names.append(name)
    return names


def load_coding_source(
    path: Path,
    sheet_name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Liest eine Analyse-XLSX roh ein (Records + vorhandene Dokumentnamen).

    Die UI benutzt das, um "Dokument hat keine Kodierungen" von "falsche
    Quelle/Namensabweichung" unterscheiden zu koennen - statt still alle
    Kodierungen anderer Dokumente zu laden.

    Raises:
        FileNotFoundError: wenn die Datei fehlt.
        ValueError: wenn die Tabelle nicht lesbar ist.
    """
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    sheet: Any = sheet_name or 0
    try:
        frame = pd.read_excel(path, sheet_name=sheet)
    except ValueError as exc:
        raise ValueError(f"Tabelle nicht lesbar: {exc}") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    records = frame.to_dict(orient="records")
    return records, document_names_in_source(records)


def load_codings_from_xlsx(
    path: Path,
    doc_id: str,
    doc_name: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> List[Coding]:
    """Liest Kodierungen aus einer Analyse-XLSX.

    Raises:
        FileNotFoundError: wenn die Datei fehlt.
        ValueError: wenn keine passende Tabelle lesbar ist.
    """
    records, _ = load_coding_source(path, sheet_name=sheet_name)
    return codings_from_records(records, doc_id=doc_id, doc_name=doc_name)


def available_coding_sources(output_dir: Path, pattern: str = "QCA-AID_Analysis_*.xlsx") -> List[Path]:
    """Listet Analysedateien (neueste zuerst) als Kodierquelle."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    files = [p for p in output_dir.glob(pattern) if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


# ---------------------------------------------------------------------------
# Anker
# ---------------------------------------------------------------------------


def anchor_codings(
    pdf_path: Path,
    codings: Sequence[Coding],
    index: Optional[DocumentIndex] = None,
    resolver: Optional[AnchorResolver] = None,
    fuzzy_threshold: float = 0.85,
) -> List[Coding]:
    """Loest fuer jede Kodierung einen Anker auf (unveraenderlich).

    Bereits verankerte Kodierungen bleiben unangetastet. Fuer den Chunktext
    werden gestaffelte Zitatlaengen probiert (QUOTE_STAGES, von lang nach
    kurz): die laengste auffindbare Stufe wird markiert, die kuerzeste Stufe
    greift nur, wenn alle laengeren scheitern. Kodierungen ohne auffindbares
    Zitat bekommen ``anchor=None`` und bleiben im Panel als "ohne Anker"
    sichtbar.
    """
    resolver = resolver or AnchorResolver()
    resolved: List[Coding] = []

    pdf_path = Path(pdf_path)
    # Das PDF wird einmal geoeffnet und an alle Stufen durchgereicht
    # (resolve_in_doc); eigene Resolver-Implementierungen ohne diese Methode
    # bekommen weiterhin pdf_path und oeffnen selbst.
    in_doc_resolver = getattr(resolver, "resolve_in_doc", None)
    pdf_doc = None
    if in_doc_resolver is not None:
        import fitz

        pdf_doc = fitz.open(pdf_path)

    try:
        for coding in codings:
            if coding.anchor is not None and coding.anchor.anchored:
                resolved.append(coding)
                continue

            quote = select_quote(
                coding.anchor.quote if coding.anchor else "",
                max_chars=QUOTE_SOURCE_LIMIT,
            )
            if not quote:
                resolved.append(_without_anchor(coding, quote))
                continue

            # Seite bevorzugt aus dem Textindex, sonst die (ggf. vorlaeufige) Angabe.
            page = find_page_for_quote(index, quote) or coding.page
            if page is None and coding.anchor is not None:
                page = coding.anchor.page
            if page is None:
                resolved.append(_without_anchor(coding, quote))
                continue

            anchor = None
            chosen_quote = quote
            for stage_quote in quote_stages(quote):
                try:
                    if in_doc_resolver is not None:
                        candidate = in_doc_resolver(
                            doc=pdf_doc,
                            doc_id=coding.doc_id,
                            page=page,
                            quote=stage_quote,
                            page_label=_page_label(index, page),
                            fuzzy_threshold=fuzzy_threshold,
                        )
                    else:
                        candidate = resolver.resolve(
                            pdf_path=pdf_path,
                            doc_id=coding.doc_id,
                            page=page,
                            quote=stage_quote,
                            page_label=_page_label(index, page),
                            fuzzy_threshold=fuzzy_threshold,
                        )
                except Exception as exc:  # noqa: BLE001 - eine Stufe kippt nicht alles
                    logger.warning(
                        "Anker-Aufloesung fehlgeschlagen (%s, Stufe %d): %s",
                        coding.segment_id,
                        len(stage_quote),
                        exc,
                    )
                    continue
                if candidate.anchored:
                    anchor = candidate
                    chosen_quote = stage_quote
                    break

            if anchor is None or not anchor.anchored:
                resolved.append(_without_anchor(coding, chosen_quote))
                continue

            resolved.append(
                Coding(
                    segment_id=coding.segment_id,
                    doc_id=coding.doc_id,
                    category=coding.category,
                    subcategories=list(coding.subcategories),
                    confidence=coding.confidence,
                    justification=coding.justification,
                    created_by=coding.created_by,
                    status=coding.status,
                    inference=coding.inference,
                    page=anchor.page,
                    anchor=anchor,
                )
            )
    finally:
        if pdf_doc is not None:
            pdf_doc.close()
    return resolved


def _without_anchor(coding: Coding, quote: str = "") -> Coding:
    """Kopie ohne aufgeloesten Anker - der Zitattext bleibt erhalten."""
    anchor = None
    if quote:
        page = coding.anchor.page if coding.anchor is not None else 1
        anchor = Anchor(
            doc_id=coding.doc_id,
            page=page,
            quote=quote,
            page_label=coding.anchor.page_label if coding.anchor else None,
            anchored=False,
        )
    return Coding(
        segment_id=coding.segment_id,
        doc_id=coding.doc_id,
        category=coding.category,
        subcategories=list(coding.subcategories),
        confidence=coding.confidence,
        justification=coding.justification,
        created_by=coding.created_by,
        status=coding.status,
        inference=coding.inference,
        page=coding.page,
        anchor=anchor,
    )


def _page_label(index: Optional[DocumentIndex], page: int) -> Optional[str]:
    if index is None or not index.page_labels:
        return None
    if 1 <= page <= len(index.page_labels):
        return index.page_labels[page - 1]
    return None


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


def build_overlay_map(
    page_sizes: Sequence[Tuple[float, float]],
    codings: Sequence[Coding],
    mode: str = "highlight",
    active_coding_id: Optional[str] = None,
    opacity: float = 0.4,
    blend: str = "multiply",
    renderer: Optional[OverlayRenderer] = None,
) -> Dict[int, str]:
    """Erzeugt je Seite mit Kodierungen ein SVG-Overlay.

    Args:
        page_sizes: (Breite, Hoehe) je Seite, Index 0 = Seite 1.
        codings: Kodierungen (nur verankerte werden gezeichnet).
        mode: Darstellungsmodus (highlight/underline/margin/bracket).
        active_coding_id: segment_id der aktiven Kodierung (In-Text-Highlight).
    """
    renderer = renderer or OverlayRenderer()
    options = RenderOptions(
        mode=mode,
        active_coding_id=active_coding_id,
        opacity=opacity,
        blend=blend,
    )

    anchors = [c.anchor for c in codings if c.anchor is not None and c.anchor.anchored]
    if not anchors:
        return {}

    pages = sorted({anchor.page for anchor in anchors})
    overlays: Dict[int, str] = {}
    for page in pages:
        if page < 1 or page > len(page_sizes):
            continue
        overlays[page] = renderer.build_page_svg(
            page=page,
            anchors=anchors,
            codings=codings,
            opts=options,
            page_size=page_sizes[page - 1],
        )
    return overlays


__all__ = [
    "QUOTE_MAX_CHARS",
    "QUOTE_STAGES",
    "QUOTE_SOURCE_LIMIT",
    "SUBCATEGORY_SEPARATOR",
    "normalize_doc_name",
    "normalize_ws",
    "select_quote",
    "quote_stages",
    "find_page_for_quote",
    "coding_from_record",
    "codings_from_records",
    "document_names_in_source",
    "load_coding_source",
    "load_codings_from_xlsx",
    "available_coding_sources",
    "anchor_codings",
    "build_overlay_map",
]
