"""
Review UI
=========
Dokumentvorschau mit Annotation (Kodierungen sichtbar und pruefbar).

Zwei Einstiegspunkte, ein Bauteil:
  - ``render_document_preview()`` — kompakte Vorschau im Analyse-Tab
    (ersetzt die fruehere reine Quelltext-Anzeige)
  - ``render_review_tab()`` — eigener Arbeitsplatz: Viewer + Kodierliste
    + Bestaetigen/Verwerfen + Export

Die Logik liegt in ``webapp_logic.preview_service`` (Canonical_PDF) und
``webapp_logic.review_service`` (Kodierungen, Anker, Overlay). Hier steht nur
die Streamlit-Verdrahtung.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import streamlit as st

from webapp_logic.review_export import REVIEW_SUBDIR, export_review_package
from webapp_logic.review_writeback import (
    BACKUP_SUBDIR,
    CHANGE_COLUMN,
    DECISION_SHEET,
    MAIN_SHEET,
    STATUS_COLUMN,
    WritebackError,
    change_summary,
    write_review_into_xlsx,
)
from QCA_AID_assets.annotation.model import Coding
from viewer import (
    CodeOption,
    ViewerConfig,
    ViewerEvent,
    build_code_options,
    build_coding_rows,
    panel_summary,
    pdf_to_data_uri,
    recode,
    render_pdf_viewer,
)
from viewer.coding_panel import ACTION_ACCEPT, ACTION_REJECT, render_coding_panel
from webapp_components.workflow_status import switch_to_tab
from webapp_logic.preview_service import (
    PreviewDocument,
    PreviewService,
    PreviewUnavailable,
    list_input_documents,
    page_sizes,
)
from webapp_logic.review_service import (
    anchor_codings,
    available_coding_sources,
    build_overlay_map,
    codings_from_records,
    load_coding_source,
    normalize_doc_name,
)

logger = logging.getLogger(__name__)

#: Vorschau-Hoehe im Analyse-Tab (kompakt) bzw. im Pruefen-Tab (Arbeitsplatz).
COMPACT_HEIGHT = 420
WORKSPACE_HEIGHT = 680

#: Ab dieser Groesse wird das PDF nicht mehr inline (data:-URI) eingebettet.
MAX_INLINE_PDF_MB = 25.0

#: Konkrete Abhilfe je fehlender Konvertierungsstufe (Schlüssel = Stufenname
#: in den Fehlergründen, z.B. ``libreoffice: nicht verfuegbar``).
CONVERTER_HINTS: Dict[str, str] = {
    "libreoffice": "LibreOffice installieren (soffice im PATH) – beste Layout-Treue für DOCX/DOC.",
    "docling": "optional: `pip install docling` für strukturierte Konvertierung.",
    "python-docx": "`pip install python-docx` – Text-Fallback für DOCX.",
    "pandoc": "Pandoc installieren (pandoc.org) – für MD, HTML und TXT.",
    "weasyprint": "`pip install weasyprint` – rendert HTML/TXT als PDF.",
    "pymupdf-story": "PyMuPDF (`pip install pymupdf`) – Text-Fallback für MD/TXT.",
}

#: Darstellungsmodi mit Beschriftung
RENDER_MODE_LABELS: Dict[str, str] = {
    "highlight": "Textmarker",
    "underline": "Unterstreichen",
    "margin": "Randmarkierung",
    "bracket": "Klammer",
}

STATE_CODINGS = "review_codings"
STATE_SOURCE = "review_source"
STATE_ACTIVE = "review_active_segment"
STATE_MODE = "review_render_mode"
STATE_PAGE = "review_page"
STATE_DOC = "review_doc_id"
STATE_PENDING_DOC = "review_pending_doc_path"
STATE_LAST_EVENT = "review_last_viewer_event"
STATE_PAGE_DOC = "review_page_doc_id"
#: Stand beim Laden - Bezugspunkt fuer das Aenderungsblatt des Exports.
STATE_ORIGINAL = "review_original_codings"


# ---------------------------------------------------------------------------
# Service-Zugriff
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _preview_service(root: str) -> PreviewService:
    """Ein Service je Projekt-Root (Cache ueberlebt Reruns)."""
    return PreviewService(Path(root))


def get_preview_service(project_root: Path) -> PreviewService:
    return _preview_service(str(Path(project_root)))


def _project_root() -> Path:
    return Path(st.session_state.project_manager.get_root_directory())


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else _project_root() / path


# ---------------------------------------------------------------------------
# Viewer-Bausteine
# ---------------------------------------------------------------------------


def viewer_config_for(
    document: PreviewDocument,
    overlays: Optional[Dict[int, str]] = None,
    height: int = COMPACT_HEIGHT,
    initial_page: int = 1,
    show_thumbnails: bool = False,
) -> ViewerConfig:
    """Baut die Viewer-Konfiguration aus einem PreviewDocument."""
    try:
        mtime_ns = document.pdf_path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return ViewerConfig(
        pdf_data_uri=_cached_data_uri(str(document.pdf_path), mtime_ns),
        doc_id=document.doc_id,
        page_labels=document.page_labels,
        overlays=overlays or {},
        initial_page=initial_page,
        height=height,
        show_thumbnails=show_thumbnails,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_data_uri(pdf_path: str, mtime_ns: int = 0) -> str:
    """Base64-Einbettung des PDFs, gecacht ueber Pfad + Aenderungszeit.

    Ohne diesen Cache wuerde jeder Widget-Klick in Streamlit das komplette PDF
    erneut base64-kodieren.
    """
    return pdf_to_data_uri(Path(pdf_path))


def _too_large(document: PreviewDocument) -> bool:
    try:
        size_mb = document.pdf_path.stat().st_size / (1024 * 1024)
    except OSError:
        return False
    return size_mb > MAX_INLINE_PDF_MB


def _document_meta_caption(document: PreviewDocument) -> None:
    parts = [f"{document.page_count} Seite(n)"]
    if document.converter:
        parts.append(f"erzeugt mit {document.converter}")
    if not document.has_text_layer:
        parts.append("ohne Textlayer (keine OCR)")
    st.caption(" · ".join(parts))
    for warning in document.warnings:
        st.caption(f":material/warning: {warning}")


def _render_failure(src: Path, error: PreviewUnavailable) -> None:
    st.warning(f":material/warning: Für **{src.name}** ist keine PDF-Vorschau möglich.")
    if error.reasons:
        with st.expander(":material/search: Ursachen", expanded=True):
            for reason in error.reasons:
                st.markdown(f"- {reason}")
            hints = _converter_hints(error.reasons)
            if hints:
                st.markdown("**Abhilfe:**")
                for hint in hints:
                    st.markdown(f"- {hint}")
    st.caption(
        "Die Datei bleibt unverändert im Eingabeverzeichnis. Fehlt ein Konverter "
        "(LibreOffice, Pandoc), betrifft das nur die Vorschau."
    )


def _converter_hints(reasons: List[str]) -> List[str]:
    """Übersetzt 'nicht verfuegbar'-Gründe in konkrete Abhilfe."""
    hints: List[str] = []
    for reason in reasons:
        if "nicht verfuegbar" not in reason:
            continue
        stage = reason.split(":", 1)[0].strip().lower()
        hint = CONVERTER_HINTS.get(stage)
        if hint and hint not in hints:
            hints.append(hint)
    return hints


def _load_document(src: Path) -> Optional[PreviewDocument]:
    """Holt das Vorschau-Dokument, ohne den Tab abzubrechen.

    Konvertierungsfehler enden als Hinweis in der Vorschau (Anforderung 1.9).
    Auch unerwartete Fehler werden gefangen: eine defekte Konverter-Bibliothek
    darf nicht den gesamten Tab lahmlegen.
    """
    service = get_preview_service(_project_root())
    try:
        return service.get(src)
    except PreviewUnavailable as error:
        _render_failure(src, error)
        return None
    except Exception as exc:  # noqa: BLE001 - Vorschau ist nicht kritisch
        logger.exception("Vorschau für %s fehlgeschlagen", src)
        st.warning(f":material/warning: Für **{src.name}** ist keine PDF-Vorschau möglich.")
        with st.expander(":material/search: Technischer Fehler"):
            st.code(f"{type(exc).__name__}: {exc}")
        st.caption(
            "Die Datei bleibt unverändert im Eingabeverzeichnis; Analyse und "
            "Export sind davon nicht betroffen."
        )
        return None


def preview_document(
    src: Path,
    codings: Optional[List[Coding]] = None,
    active_coding_id: Optional[str] = None,
    mode: str = "highlight",
    height: int = COMPACT_HEIGHT,
    show_thumbnails: bool = False,
    initial_page: int = 1,
    on_event: Optional[Callable[[Dict], None]] = None,
) -> Optional[PreviewDocument]:
    """Konvertiert (gecacht) und rendert die Dokumentvorschau.

    Returns:
        PreviewDocument oder None, wenn die Konvertierung scheitert.
    """
    document = _load_document(src)
    if document is None:
        return None

    if _too_large(document):
        st.info(
            f":material/info: {src.name} ist größer als {MAX_INLINE_PDF_MB:.0f} MB "
            "und wird nicht inline angezeigt."
        )
        st.download_button(
            ":material/download: Canonical PDF herunterladen",
            data=document.pdf_path.read_bytes(),
            file_name=f"{src.stem}.pdf",
            mime="application/pdf",
            key=f"dl_{document.doc_id}",
        )
        return document

    overlays: Dict[int, str] = {}
    if codings:
        overlays = build_overlay_map(
            page_sizes=page_sizes(document.pdf_path, document.page_count),
            codings=codings,
            mode=mode,
            active_coding_id=active_coding_id,
        )

    render_pdf_viewer(
        viewer_config_for(
            document,
            overlays=overlays,
            height=height,
            initial_page=initial_page,
            show_thumbnails=show_thumbnails,
        ),
        key=f"viewer_{document.doc_id}_{height}",
        on_event=on_event,
    )
    _document_meta_caption(document)
    return document


# ---------------------------------------------------------------------------
# Kompakter Einstieg (Analyse-Tab)
# ---------------------------------------------------------------------------


def render_document_preview(src: Path, compact: bool = True) -> None:
    """Vorschau eines Eingabedokuments fuer den Analyse-Tab."""
    st.markdown("**Vorschau:**")
    document = preview_document(
        src,
        height=COMPACT_HEIGHT,
        show_thumbnails=False,
        initial_page=st.session_state.get(STATE_PAGE, 1),
        on_event=_handle_viewer_event,
    )
    if document is None:
        return

    codings = st.session_state.get(STATE_CODINGS) or []
    doc_codings = [c for c in codings if c.doc_id == document.doc_id]
    if doc_codings:
        summary = panel_summary(doc_codings)
        st.caption(
            f":material/fact_check: {summary['total']} Kodierung(en) vorhanden — "
            "im Prüfen-Tab bearbeitbar."
        )

    if st.button(
        ":material/open_in_new: Im Prüfen-Tab öffnen",
        key=f"open_review_{document.doc_id}",
        use_container_width=True,
    ):
        st.session_state[STATE_PENDING_DOC] = str(src)
        st.session_state[STATE_DOC] = document.doc_id
        switch_to_tab("review")


# ---------------------------------------------------------------------------
# Pruefen-Tab
# ---------------------------------------------------------------------------


def _select_document(documents) -> Optional[Path]:
    """Dokumentauswahl (Vorauswahl aus Analyse-Tab via review_pending_doc_path)."""
    labels = {f"{doc.name} ({doc.format_size()})": doc.path for doc in documents}
    if not labels:
        return None

    pending = st.session_state.pop(STATE_PENDING_DOC, None)
    widget_key = "review_document_select"

    # Streamlit ignoriert `index`, sobald der Widget-State existiert. Fuer die
    # Vorauswahl aus dem Analyse-Tab wird deshalb der Widget-State direkt
    # gesetzt - und dann KEIN index uebergeben (sonst StreamlitAPIException).
    if pending:
        for label, path in labels.items():
            if str(path) == str(pending):
                st.session_state[widget_key] = label
                break

    selected = st.selectbox(
        "Dokument",
        options=list(labels.keys()),
        key=widget_key,
        **({"index": _default_document_index(labels)} if not pending else {}),
    )
    return labels.get(selected)


def _default_document_index(labels: Dict[str, Path]) -> int:
    """Vorauswahl: erstes bereits als PDF vorliegendes Dokument, sonst 0.

    PDFs brauchen keinen Konverter und sind damit immer sofort sichtbar.
    """
    for index, path in enumerate(labels.values()):
        if Path(path).suffix.lower() == ".pdf":
            return index
    return 0


def _load_codings(document: PreviewDocument, doc_name: str, source: Path, refresh: bool) -> None:
    """Laedt Kodierungen aus der Analysedatei, ankert sie und legt sie in den State.

    Nie einen stillen Fallback ueber alle Dokumente: Ist die Datei nicht
    kodiert, muss das im Panel sichtbar bleiben (0 Kodierungen) - sonst
    erscheinen Kodierungen *anderer* Dokumente und anchern alle nicht.
    """
    try:
        records, source_docs = load_coding_source(source)
    except (FileNotFoundError, ValueError) as error:
        st.error(f":material/cancel: Kodierungen nicht lesbar: {error}")
        return

    raw = codings_from_records(records, doc_id=document.doc_id, doc_name=doc_name)

    if not raw:
        target = normalize_doc_name(doc_name)
        if not records:
            st.warning(
                f":material/warning: {source.name} enthält keine Zeilen — "
                "kein geeigneter Export?"
            )
        elif source_docs and target not in source_docs:
            st.warning(
                f":material/warning: In {source.name} gibt es Kodierungen für "
                f"**{', '.join(source_docs)}** — für „{doc_name}“ keine. "
                "Passen Quelle und Dokument zusammen?"
            )
        else:
            st.info(
                f":material/info: Für „{doc_name}“ stehen in {source.name} keine "
                "Kodierungen — es gab keine relevanten Segmente. Das Dokument "
                "ist damit vollständig geprüft."
            )
        st.session_state[STATE_CODINGS] = []
        st.session_state[STATE_SOURCE] = str(source)
        st.session_state[STATE_ACTIVE] = None
        return

    if not raw:
        st.warning(
            f":material/warning: In {source.name} wurden keine Kodierungen "
            f"für „{doc_name}“ gefunden."
        )
        st.session_state[STATE_CODINGS] = []
        return

    with st.spinner("Anker werden aufgelöst …"):
        anchored = anchor_codings(
            pdf_path=document.pdf_path,
            codings=raw,
            index=document.index,
        )
    st.session_state[STATE_CODINGS] = anchored
    st.session_state[STATE_ORIGINAL] = anchored
    st.session_state[STATE_SOURCE] = str(source)
    st.session_state[STATE_ACTIVE] = None

    summary = panel_summary(anchored)
    st.success(
        f":material/check_circle: {summary['total']} Kodierung(en) geladen, "
        f"davon {summary['total'] - summary['unanchored']} verankert."
    )
    if refresh:
        st.rerun()


def _apply_coding_action(segment_id: str, action: str) -> None:
    codings: List[Coding] = list(st.session_state.get(STATE_CODINGS) or [])
    for index, coding in enumerate(codings):
        if coding.segment_id != segment_id:
            continue
        if action == ACTION_ACCEPT:
            codings[index] = replace(coding, status="accepted")
        elif action == ACTION_REJECT:
            codings[index] = replace(coding, status="rejected")
        break
    st.session_state[STATE_CODINGS] = codings
    st.rerun()


def _apply_coding_change(segment_id: str, option: CodeOption) -> None:
    """Vergibt einen anderen Code (oder aendert die Subcodes)."""
    codings: List[Coding] = list(st.session_state.get(STATE_CODINGS) or [])
    for index, coding in enumerate(codings):
        if coding.segment_id != segment_id:
            continue
        codings[index] = recode(coding, option)
        break
    st.session_state[STATE_CODINGS] = codings
    st.session_state[STATE_ACTIVE] = segment_id
    st.rerun()


def _select_coding(segment_id: str) -> None:
    """Klick auf den Code im Panel: Kodierung aktiv setzen.

    Der Viewer bekommt ``active_coding_id`` ueber die Overlays und hebt die
    Markierung hervor, waehrend die uebrigen abgeschwaecht werden.
    """
    codings: List[Coding] = list(st.session_state.get(STATE_CODINGS) or [])
    st.session_state[STATE_ACTIVE] = segment_id
    for coding in codings:
        if coding.segment_id == segment_id and coding.page:
            st.session_state[STATE_PAGE] = coding.page
            break
    st.rerun()


def _handle_viewer_event(payload) -> None:
    """Rueckkanal des Viewers: Seitenwechsel und Klick auf eine Kodierung.

    Ein Component-Value bleibt zwischen Reruns stehen; deshalb merkt sich der
    State die zuletzt verarbeitete Ereignis-Kennung (seq) und uebergeht
    Wiederholungen. Seitenwechsel brauchen keinen Rerun - STATE_PAGE wird
    beim naechsten Rerun gelesen.
    """
    event = ViewerEvent.parse(payload)
    if event is None:
        return
    seq = (payload or {}).get("seq")
    fingerprint = (event.type, event.coding_id, event.page, seq)
    if st.session_state.get(STATE_LAST_EVENT) == fingerprint:
        return
    st.session_state[STATE_LAST_EVENT] = fingerprint

    if event.type == "coding-click" and event.coding_id:
        st.session_state[STATE_ACTIVE] = event.coding_id
        if event.page:
            st.session_state[STATE_PAGE] = event.page
        st.rerun()
    if event.type == "page-change" and event.page:
        st.session_state[STATE_PAGE] = event.page
        # Rerun damit das Panel die Kodierungen dieser Seite zeigt; der Viewer
        # wird dabei nicht neu gebaut (Overlays laufen als Live-Update), also
        # springt die Ansicht nicht.
        st.rerun()


def _available_code_options(codings: List[Coding]) -> List[CodeOption]:
    """Codes aus dem geladenen Codebook plus die bereits vergebenen Codes."""
    return build_code_options(st.session_state.get("codebook_data"), codings)


def _code_options_caption(options: List[CodeOption]) -> str:
    """Kurzinfo, woher die auswaehlbaren Codes stammen."""
    if options and st.session_state.get("codebook_data") is not None:
        return (
            f":material/menu_book: {len(options)} Codes aus Codebook und "
            "geladenen Kodierungen wählbar („Ändern“ je Kodierung)."
        )
    if options:
        return (
            f":material/menu_book: {len(options)} Codes aus den geladenen "
            "Kodierungen wählbar — kein Codebook geladen."
        )
    return (
        ":material/info: Kein Codebook geladen: neue Codes im Feld "
        "„Neuer Code“ je Kodierung anlegen."
    )


def render_review_tab() -> None:
    """Arbeitsplatz: Dokument lesen, Kodierungen pruefen, Ergebnis exportieren."""
    config = st.session_state.config_data
    input_dir = _resolve(config.data_dir)
    output_dir = _resolve(config.output_dir)

    st.subheader(":material/fact_check: Kodierungen prüfen")
    st.caption(
        "Originaldokument neben der Kodierung: markierte Stellen sind anklickbar, "
        "rechts stehen Kategorie, Konfidenz und Begründung."
    )

    documents = list_input_documents(input_dir)
    if not documents:
        st.info(
            ":material/info: Keine vorschaubaren Dokumente im Eingabeverzeichnis "
            f"({input_dir})."
        )
        st.code(str(input_dir))
        return

    sources = available_coding_sources(output_dir)

    top = st.columns([3, 3, 2])
    with top[0]:
        selected_path = _select_document(documents)
    with top[1]:
        source_labels = {p.name: p for p in sources}
        selected_source_label = st.selectbox(
            "Kodierquelle (Analyseergebnis)",
            options=["— keine —"] + list(source_labels.keys()),
            index=0,
            key="review_source_select",
            help="Kodierungen stammen aus der Analyse-XLSX im Ausgabeverzeichnis.",
        )
        selected_source = source_labels.get(selected_source_label)
    with top[2]:
        mode = st.selectbox(
            "Darstellung",
            options=list(RENDER_MODE_LABELS.keys()),
            format_func=lambda key: RENDER_MODE_LABELS[key],
            index=list(RENDER_MODE_LABELS).index(
                st.session_state.get(STATE_MODE, "highlight")
                if st.session_state.get(STATE_MODE) in RENDER_MODE_LABELS
                else "highlight"
            ),
            key=STATE_MODE,
        )

    if selected_path is None:
        return

    document = _load_document(selected_path)
    if document is None:
        return

    # Seitenposition ist dokumentspezifisch: bei Dokumentwechsel zuruecksetzen.
    if st.session_state.get(STATE_PAGE_DOC) != document.doc_id:
        st.session_state[STATE_PAGE_DOC] = document.doc_id
        st.session_state[STATE_PAGE] = None

    st.session_state[STATE_DOC] = document.doc_id

    # -- Kodierungen laden ------------------------------------------------
    load_col, info_col = st.columns([2, 5])
    with load_col:
        load_clicked = st.button(
            ":material/download: Kodierungen laden",
            use_container_width=True,
            disabled=selected_source is None,
        )

    # Erst laden, dann den Hinweis rendern - sonst zeigt ein Klick "Quelle
    # gewechselt" gleichzeitig mit der Erfolgsmeldung an.
    if load_clicked and selected_source is not None:
        _load_codings(document, selected_path.name, selected_source, refresh=False)

    with info_col:
        if not sources:
            st.caption("Keine Analyseergebnisse gefunden — zuerst eine Analyse ausführen.")
        elif st.session_state.get(STATE_SOURCE) != str(selected_source):
            st.caption(
                "Quelle wählen und „Kodierungen laden“ — die Kodierungen dieser "
                "Datei sind noch nicht geladen."
            )

    codings: List[Coding] = [
        coding
        for coding in (st.session_state.get(STATE_CODINGS) or [])
        if coding.doc_id == document.doc_id
    ]
    active_id = st.session_state.get(STATE_ACTIVE)

    left, right = st.columns([3, 2], gap="large")

    with left:
        preview_document(
            selected_path,
            codings=codings,
            active_coding_id=active_id,
            mode=mode,
            height=WORKSPACE_HEIGHT,
            show_thumbnails=True,
            initial_page=st.session_state.get(STATE_PAGE, 1),
            on_event=_handle_viewer_event,
        )

    with right:
        if not codings:
            st.info(
                ":material/info: Noch keine Kodierungen geladen. Wählen Sie eine "
                "Kodierquelle und laden Sie die Kodierungen."
            )
        else:
            # Nur die Kodierungen der aktuellen Viewer-Seite - die Seite folgt
            # dem Rueckkanal des Viewers (page-change), nicht einer Dropdown-
            # Auswahl, die bei jedem Seitenwechsel veraltet waere.
            current_page = st.session_state.get(STATE_PAGE)
            if current_page is None:
                st.caption(
                    ":material/filter_alt: Solange keine Seite gewählt ist, "
                    "werden alle Kodierungen gezeigt."
                )
            code_options = _available_code_options(codings)
            st.caption(_code_options_caption(code_options))
            render_coding_panel(
                codings,
                active_id=active_id,
                page=current_page,
                key_prefix="review",
                on_action=_apply_coding_action,
                code_options=code_options,
                on_recode=_apply_coding_change,
                on_select=_select_coding,
            )

    # -- Export -----------------------------------------------------------
    if codings:
        with st.expander(":material/save: Export", expanded=False):
            summary = panel_summary(codings)
            st.caption(
                f"{summary['accepted']} bestätigt · {summary['proposed']} offen · "
                f"{summary['rejected']} verworfen · {summary['unanchored']} ohne Anker"
            )
            st.caption(
                "Der Export ändert die Analyse-Ausgabe nicht. Er schreibt ein "
                f"eigenes Paket nach `{output_dir.name}/"
                f"{REVIEW_SUBDIR}/` — annotiertes PDF, Kodierungen (XLSX) und "
                "Entscheidungsprotokoll (JSON), alle mit demselben Zeitstempel."
            )

            if st.button(
                ":material/save: Rezensionspaket exportieren",
                use_container_width=True,
                help=(
                    "Annotiertes PDF (Highlights + Schraffur für "
                    "Mehrfachkodierungen), Kodierungen inkl. Änderungsblatt "
                    "und JSON-Protokoll."
                ),
            ):
                try:
                    package = export_review_package(
                        pdf_path=document.pdf_path,
                        codings=codings,
                        output_dir=output_dir,
                        doc_stem=selected_path.stem,
                        document_label=selected_path.name,
                        source_label=selected_source.name if selected_source else "",
                        original=st.session_state.get(STATE_ORIGINAL),
                    )
                except Exception as error:  # noqa: BLE001 - UI-Fehler anzeigen
                    logger.exception("Rezensions-Export fehlgeschlagen")
                    st.error(f":material/cancel: Export fehlgeschlagen: {error}")
                else:
                    st.session_state["review_export_files"] = [
                        str(path) for path in package.files
                    ]
                    st.success(
                        f":material/check_circle: Gespeichert in `{package.folder}` "
                        f"({package.codings} Kodierungen, {package.decisions} Änderungen)."
                    )

            for path_string in st.session_state.get("review_export_files", []):
                path = Path(path_string)
                if not path.is_file():
                    continue
                st.download_button(
                    f":material/download: {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    use_container_width=True,
                    key=f"download_{path.name}",
                )

            _render_writeback_section(codings, selected_source)


def _render_writeback_section(codings: List[Coding], source: Optional[Path]) -> None:
    """Entscheidungen additiv in die Analyse-Arbeitsmappe schreiben.

    Die Analyse-XLSX bleibt die zentrale Datei der Weiterarbeit: hier bekommt
    sie die Review-Spalten und das Entscheidungsblatt - Bestandsspalten,
    Häufigkeiten und Intercoder-Bericht bleiben unverändert. Vorher entsteht
    eine Sicherung, damit nichts unwiederbringlich überschrieben wird.
    """
    st.divider()
    st.markdown("**Entscheidungen in die Analyse-XLSX übernehmen**")

    if source is None:
        st.caption(
            "Zuerst eine Kodierquelle (Analyse-XLSX) wählen und die Kodierungen laden."
        )
        return

    if st.session_state.get(STATE_SOURCE) != str(source):
        st.caption(
            f"Die geladenen Kodierungen gehören nicht zu `{source.name}` — bitte "
            "zuerst laden, dann übernehmen."
        )
        return

    counts = change_summary(codings, st.session_state.get(STATE_ORIGINAL))
    st.caption(
        f"{counts['confirmed']} bestätigt · {counts['changed']} geändert · "
        f"{counts['rejected']} verworfen · {counts['open']} offen "
        f"· {counts['total']} gesamt"
    )
    st.caption(
        f"Die Arbeitsmappe wird ergänzt, nicht ersetzt: Spalten `{STATUS_COLUMN}`, "
        "`Hauptkategorie_Review`, `Subkategorien_Review`, "
        f"`{CHANGE_COLUMN}` und `Review_Stand` in `{MAIN_SHEET}`, dazu das Blatt "
        f"`{DECISION_SHEET}` mit vorher/nachher. Häufigkeiten, Intercoder-Bericht "
        f"und Kategorien bleiben unverändert; vorher entsteht eine Sicherung in "
        f"`{source.parent.name}/{BACKUP_SUBDIR}/`."
    )

    confirmed = st.checkbox(
        "Arbeitsmappe ergänzen (Sicherung wird angelegt)",
        key="review_writeback_confirm",
    )

    if st.button(
        ":material/table_view: In Analyse-XLSX übernehmen",
        use_container_width=True,
        disabled=not confirmed,
        help="Schreibt Status und finalen Code je Kodierung in die Analyse-XLSX.",
    ):
        try:
            result = write_review_into_xlsx(source, codings)
        except WritebackError as error:
            st.error(f":material/cancel: Übernahme fehlgeschlagen: {error}")
        else:
            st.success(f":material/check_circle: {result.summary}.")
            if result.backup is not None:
                st.caption(f"Sicherung: `{result.backup}`")
            st.caption(
                "Beim nächsten „Kodierungen laden“ liest der Prüfen-Tab diesen "
                "Stand wieder ein — `Review_Status` bestimmt dann "
                "bestätigt/verworfen, `Hauptkategorie_Review` den gültigen Code."
            )


__all__ = [
    "COMPACT_HEIGHT",
    "WORKSPACE_HEIGHT",
    "RENDER_MODE_LABELS",
    "viewer_config_for",
    "preview_document",
    "render_document_preview",
    "render_review_tab",
    "get_preview_service",
]
