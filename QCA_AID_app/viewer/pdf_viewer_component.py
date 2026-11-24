"""
PDF Text Viewer Component
=========================
Eingebetteter PDF.js-Viewer fuer die Dokumentenvorschau
(siehe .kiro/specs/pdf-preview-annotation, Anforderungen 3.1-3.7, 6.6).

Aufbau:
  - `build_viewer_html(config)` erzeugt ein vollstaendiges HTML-Dokument mit
    PDF.js, Toolbar (Seiten/Labels/Zoom/Suche), Thumbnail-Leiste,
    Annotation-Overlay je Seite und Viewport-Virtualisierung.
  - `render_pdf_viewer(...)` ist der duenne Streamlit-Wrapper
    (`st.components.v1.html`).

Bruecke (Task 11): Das JS stellt `window.qcaViewer` bereit und meldet
Ereignisse via `postMessage` an das Elternfenster:
  {source: "qca-viewer", type: "page-change" | "coding-click" | "ready", ...}
"""

import base64
import html
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

DEFAULT_PDFJS_VERSION = "4.6.82"
DEFAULT_PDFJS_BASE = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/{version}"
DEFAULT_HEIGHT = 720

logger = logging.getLogger(__name__)


@dataclass
class ViewerConfig:
    """Konfiguration des TextViewers."""

    pdf_data_uri: str
    doc_id: str = "doc"
    page_labels: Optional[List[str]] = None
    overlays: Dict[int, str] = field(default_factory=dict)  # page -> SVG
    initial_page: int = 1
    height: int = DEFAULT_HEIGHT
    pdfjs_base: str = DEFAULT_PDFJS_BASE
    pdfjs_version: str = DEFAULT_PDFJS_VERSION
    show_thumbnails: bool = True
    enable_search: bool = True


def pdf_to_data_uri(pdf_path: Path) -> str:
    """Kodiert ein PDF als data:-URI (kein Server-Roundtrip im Viewer)."""
    data = Path(pdf_path).read_bytes()
    return "data:application/pdf;base64," + base64.b64encode(data).decode("ascii")


HOST_DIR = Path(__file__).resolve().parent / "_component_host"


def _declare_viewer_component():
    """Registriert den Viewer-Host (Rueckkanal fuer Viewer-Ereignisse).

    Liefert None, wenn die Deklaration fehlschlaegt (z.B. fehlendes
    Host-Verzeichnis bei einer unvollstaendigen Installation) - der Viewer
    funktioniert dann ohne Rueckkanal weiter.
    """
    try:
        import streamlit.components.v1 as components

        if not (HOST_DIR / "index.html").is_file():
            logger.warning("Viewer-Host fehlt (%s) - ohne Rueckkanal", HOST_DIR)
            return None
        return components.declare_component("qca_viewer", path=str(HOST_DIR))
    except Exception as exc:  # noqa: BLE001 - Rueckkanal ist optional
        logger.warning("Viewer-Component nicht registrierbar (%s) - ohne Rueckkanal", exc)
        return None


_VIEWER_COMPONENT = None
_DECLARATION_FAILED = False


def _viewer_component():
    """Lazy: deklariert die Component beim ersten Aufruf (im Streamlit-Lauf)."""
    global _VIEWER_COMPONENT, _DECLARATION_FAILED
    if _VIEWER_COMPONENT is not None or _DECLARATION_FAILED:
        return _VIEWER_COMPONENT
    _VIEWER_COMPONENT = _declare_viewer_component()
    _DECLARATION_FAILED = _VIEWER_COMPONENT is None
    return _VIEWER_COMPONENT


def render_pdf_viewer(
    config: ViewerConfig,
    key: Optional[str] = None,
    on_event: Optional[Callable[[Dict], None]] = None,
) -> None:
    """Streamlit-Wrapper: rendert den Viewer und liefert Viewer-Ereignisse.

    Args:
        key: Stabiler Widget-Key (Pflicht fuer den Rueckkanal).
        on_event: Callback (payload: dict) fuer ``coding-click`` /
            ``page-change`` / ``ready`` / ``error``. Ohne Rueckkanal wird er
            nie aufgerufen; die Vorschau funktioniert trotzdem.
    """
    component = _viewer_component()
    if component is not None and key:
        payload = component(
            # Overlays und Zielseite laufen als eigene Argumente: der Host
            # aktualisiert sie live im laufenden Viewer statt das iframe neu
            # zu bauen (kein Springen bei Kodierungsaktionen). Die Zielseite
            # wird deshalb NICHT ins HTML gebacken - sonst wuerde jede
            # Seitenbewegung einen Neuaufbau ausloesen.
            html=build_viewer_html(replace(config, initial_page=1), embed_overlays=False),
            height=config.height,
            page=config.initial_page,
            overlays=config.overlays,
            key=key,
            default=None,
        )
        if payload and on_event is not None:
            on_event(payload)
        return

    # Rueckkanal nicht verfuegbar (alter Streamlit-Flow: st.components.html).
    import streamlit.components.v1 as components

    components.html(build_viewer_html(config), height=config.height, scrolling=True)


# --------------------------------------------------------------------------
# HTML-Aufbau
# --------------------------------------------------------------------------

def build_viewer_html(config: ViewerConfig, embed_overlays: bool = True) -> str:
    """Erzeugt das vollstaendige Viewer-HTML.

    Args:
        embed_overlays: Wenn False, bleibt der Overlay-Slot leer und die
            Overlays werden als eigenes Component-Argument uebergeben. Der
            Component-Host kann sie dann *live* in den laufenden Viewer
            schieben, statt das iframe neu zu bauen (kein Springen beim
            Bestaetigen/Laden). Der Fallback-Weg (st.components.html) braucht
            die eingebetteten Overlays.
    """
    base = config.pdfjs_base.format(version=config.pdfjs_version).rstrip("/")
    lib_url = f"{base}/pdf.min.mjs"
    worker_url = f"{base}/pdf.worker.min.mjs"

    labels = config.page_labels or []
    nav_options = _build_label_options(labels)
    overlay_hosts = _build_overlay_hosts(config.overlays if embed_overlays else {})
    thumbnails = "1" if config.show_thumbnails else "0"
    search_block = (
        '<input id="qca-search" type="search" placeholder="Volltextsuche …" '
        'aria-label="Volltextsuche">'
        if config.enable_search
        else ""
    )

    template = _TEMPLATE
    replacements = {
        "__LIB_URL__": html.escape(lib_url, quote=True),
        "__WORKER_URL__": html.escape(worker_url, quote=True),
        "__PDF_DATA__": config.pdf_data_uri,
        "__DOC_ID__": html.escape(config.doc_id, quote=True),
        "__INITIAL_PAGE__": str(max(1, int(config.initial_page))),
        "__HEIGHT__": str(int(config.height)),
        "__NAV_OPTIONS__": nav_options,
        "__OVERLAY_HOSTS__": overlay_hosts,
        "__SHOW_THUMBS__": thumbnails,
        "__SEARCH_BLOCK__": search_block,
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def _build_label_options(labels: List[str]) -> str:
    if not labels:
        return ""
    options = []
    for index, label in enumerate(labels, start=1):
        safe = html.escape(str(label), quote=True)
        options.append(f'<option value="{index}">{safe}</option>')
    return "".join(options)


def _build_overlay_hosts(overlays: Dict[int, str]) -> str:
    """Bettet die SVG-Overlays in ihre Seiten-Container ein (Template-Slot)."""
    if not overlays:
        return ""
    import json

    # JSON-sicher einbetten; das JS haengt sie an die passenden Seiten.
    payload = {str(page): svg for page, svg in overlays.items()}
    return (
        '<script id="qca-overlays" type="application/json">'
        + json.dumps(payload).replace("</", "<\\/")
        + "</script>"
    )


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dokumentvorschau</title>
<style>
  :root { --toolbar-h: 42px; --thumb-w: 96px; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:#f3f3f3; color:#1b1b1b; }
  #qca-viewer { height: __HEIGHT__px; display:flex; flex-direction:column; }
  .qca-toolbar { height:var(--toolbar-h); display:flex; gap:8px; align-items:center;
                 padding:0 10px; background:#fff; border-bottom:1px solid #ddd; flex-wrap:nowrap; }
  .qca-toolbar button, .qca-toolbar input, .qca-toolbar select {
    font-size:13px; padding:4px 8px; border:1px solid #ccc; border-radius:4px; background:#fff;
  }
  .qca-toolbar button:hover { background:#f0f0f0; }
  .qca-toolbar button:focus-visible, .qca-toolbar input:focus-visible,
  .qca-toolbar select:focus-visible { outline:2px solid #2563eb; outline-offset:1px; }
  #qca-page-input { width:56px; text-align:center; }
  .qca-spacer { flex:1; }
  #qca-label { font-size:12px; color:#555; }
  .qca-body { flex:1; display:flex; min-height:0; }
  #qca-thumbs { width:var(--thumb-w); overflow-y:auto; background:#eaeaea;
                border-right:1px solid #ddd; padding:6px; }
  #qca-thumbs[hidden] { display:none; }
  #qca-thumbs .thumb { margin:0 0 8px; cursor:pointer; border:2px solid transparent; }
  #qca-thumbs .thumb.active { border-color:#2563eb; }
  #qca-thumbs canvas { width:100%; display:block; background:#fff; }
  #qca-scroll { flex:1; overflow:auto; padding:16px; scroll-behavior:smooth; }
  .qca-page { position:relative; margin:0 auto 16px; background:#fff;
              box-shadow:0 1px 4px rgba(0,0,0,.15); }
  .qca-page canvas { display:block; }
  .qca-text-layer { position:absolute; inset:0; overflow:hidden; line-height:1;
                    opacity:1; pointer-events:auto; }
  .qca-text-layer span { position:absolute; color:transparent; white-space:pre;
                         transform-origin:0 0; cursor:text; }
  .qca-overlay-host { position:absolute; inset:0; pointer-events:none; }
  .qca-overlay-host svg { position:absolute; inset:0; width:100%; height:100%; }
  .qca-overlay-host rect, .qca-overlay-host path { cursor:pointer; }
  .qca-page-placeholder { color:#999; display:flex; align-items:center;
                          justify-content:center; font-size:12px; }
  .qca-status { font-size:12px; color:#666; }
  @media (prefers-reduced-motion: reduce) { #qca-scroll { scroll-behavior:auto; } }
</style>
</head>
<body>
<div id="qca-viewer" data-doc-id="__DOC_ID__">
  <div class="qca-toolbar" role="toolbar" aria-label="Dokumentnavigation">
    <button id="qca-prev" title="Vorherige Seite" aria-label="Vorherige Seite">◀</button>
    <input id="qca-page-input" type="number" min="1" value="__INITIAL_PAGE__"
           aria-label="Seite">
    <span class="qca-status">/ <span id="qca-page-count">?</span></span>
    <select id="qca-label-select" aria-label="Seitenlabel">__NAV_OPTIONS__</select>
    <span id="qca-label" aria-live="polite"></span>
    <button id="qca-next" title="Naechste Seite" aria-label="Naechste Seite">▶</button>
    <span class="qca-spacer"></span>
    __SEARCH_BLOCK__
    <button id="qca-zoom-out" aria-label="Herauszoomen">−</button>
    <button id="qca-zoom-fit" aria-label="An Breite anpassen">Breite</button>
    <button id="qca-zoom-in" aria-label="Hineinzoomen">+</button>
  </div>
  <div class="qca-body">
    <aside id="qca-thumbs" aria-label="Seitenminiaturen"></aside>
    <main id="qca-scroll" tabindex="0" aria-label="Dokumentinhalt"></main>
  </div>
  __OVERLAY_HOSTS__
</div>

<script type="module">
import * as pdfjsLib from "__LIB_URL__";
pdfjsLib.GlobalWorkerOptions.workerSrc = "__WORKER_URL__";

const DOC_ID = document.getElementById("qca-viewer").dataset.docId;
const SHOW_THUMBS = "__SHOW_THUMBS__" === "1";
const PDF_URI = "__PDF_DATA__";
const INITIAL_PAGE = parseInt("__INITIAL_PAGE__", 10) || 1;
let overlays = JSON.parse(document.getElementById("qca-overlays")?.textContent || "{}");
let pdfDoc = null;
let scale = 1.0;
let fitWidth = true;
let currentPage = INITIAL_PAGE;
let pageSizes = [];
const rendered = new Set();
let SEQ = 0;

const scrollEl = document.getElementById("qca-scroll");
const thumbsEl = document.getElementById("qca-thumbs");
const pageInput = document.getElementById("qca-page-input");
const labelSelect = document.getElementById("qca-label-select");
const labelEl = document.getElementById("qca-label");

function emit(type, detail) {
  window.parent.postMessage({ source: "qca-viewer", type, docId: DOC_ID, seq: ++SEQ, ...detail }, "*");
}

// Kommandos des Component-Hosts (App -> Viewer), ohne iframe-Neuaufbau:
// Overlays ersetzen und zur Seite springen. Der Host hat den Rueckkanal
// (page-change) und schickt nur, wenn sich die Seite wirklich aendert.
window.addEventListener("message", (ev) => {
  const data = ev.data;
  if (!data || data.source !== "qca-host") return;
  if (data.type === "set-overlays") {
    overlays = data.overlays || {};
    document.querySelectorAll(".qca-overlay-host").forEach((h) => h.remove());
    rendered.forEach((p) => {
      const div = pageDiv(p);
      if (div) attachOverlay(p, div);
    });
  } else if (data.type === "goto-page") {
    gotoPage(parseInt(data.page, 10) || 1, false);
  }
});

async function load() {
  pdfDoc = await pdfjsLib.getDocument({ url: PDF_URI }).promise;
  document.getElementById("qca-page-count").textContent = pdfDoc.numPages;
  for (let p = 1; p <= pdfDoc.numPages; p++) {
    const page = await pdfDoc.getPage(p);
    const vp = page.getViewport({ scale: 1 });
    pageSizes.push({ width: vp.width, height: vp.height });
  }
  buildPlaceholders();
  buildThumbnails();
  observePages();
  await gotoPage(INITIAL_PAGE, false);
  emit("ready", { pages: pdfDoc.numPages });
}

function buildPlaceholders() {
  const w = pageSizes[0].width, h = pageSizes[0].height;
  for (let p = 1; p <= pdfDoc.numPages; p++) {
    const div = document.createElement("div");
    div.className = "qca-page qca-page-placeholder";
    div.dataset.page = p;
    div.style.width = (w * scale) + "px";
    div.style.height = (h * scale) + "px";
    div.textContent = "Seite " + p;
    scrollEl.appendChild(div);
  }
}

function pageDiv(p) { return scrollEl.querySelector('.qca-page[data-page="' + p + '"]'); }

function attachOverlay(p, div) {
  const svg = overlays[String(p)];
  if (!svg) return;
  let host = div.querySelector(".qca-overlay-host");
  if (!host) { host = document.createElement("div"); host.className = "qca-overlay-host"; div.appendChild(host); }
  host.innerHTML = svg;
  host.querySelectorAll("[data-coding-id]").forEach(el => {
    el.addEventListener("click", ev => {
      ev.stopPropagation();
      const codingId = el.getAttribute("data-coding-id");
      setActiveCoding(codingId);
      emit("coding-click", { codingId, page: p });
    });
  });
}

async function renderPage(p) {
  if (rendered.has(p)) return;
  const page = await pdfDoc.getPage(p);
  const viewport = page.getViewport({ scale });
  const div = pageDiv(p);
  div.classList.remove("qca-page-placeholder");
  div.textContent = "";
  div.style.width = viewport.width + "px";
  div.style.height = viewport.height + "px";

  const canvas = document.createElement("canvas");
  const ratio = window.devicePixelRatio || 1;
  canvas.width = viewport.width * ratio;
  canvas.height = viewport.height * ratio;
  canvas.style.width = viewport.width + "px";
  canvas.style.height = viewport.height + "px";
  div.appendChild(canvas);

  const ctx = canvas.getContext("2d");
  await page.render({
    canvasContext: ctx,
    viewport,
    transform: ratio !== 1 ? [ratio, 0, 0, ratio, 0, 0] : null
  }).promise;

  try {
    const textContent = await page.getTextContent();
    const textLayer = document.createElement("div");
    textLayer.className = "qca-text-layer";
    div.appendChild(textLayer);
    pdfjsLib.renderTextLayer({
      textContentSource: textContent,
      container: textLayer,
      viewport,
      textDivs: []
    });
  } catch (err) { /* Textlayer ist optional */ }

  attachOverlay(p, div);
  rendered.add(p);
}

function observePages() {
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        renderPage(parseInt(entry.target.dataset.page, 10));
      }
    });
  }, { root: scrollEl, rootMargin: "200px 0px" });
  scrollEl.querySelectorAll(".qca-page").forEach(el => io.observe(el));
}

function buildThumbnails() {
  if (!SHOW_THUMBS) { thumbsEl.hidden = true; return; }
  for (let p = 1; p <= pdfDoc.numPages; p++) {
    const wrap = document.createElement("div");
    wrap.className = "thumb"; wrap.dataset.page = p;
    const canvas = document.createElement("canvas");
    wrap.appendChild(canvas);
    wrap.addEventListener("click", () => gotoPage(p));
    thumbsEl.appendChild(wrap);
    renderThumb(p, canvas);
  }
}

async function renderThumb(p, canvas) {
  const page = await pdfDoc.getPage(p);
  const vp = page.getViewport({ scale: 0.2 });
  canvas.width = vp.width; canvas.height = vp.height;
  await page.render({ canvasContext: canvas.getContext("2d"), viewport: vp }).promise;
}

function updateActiveThumb(p) {
  thumbsEl.querySelectorAll(".thumb").forEach(el =>
    el.classList.toggle("active", parseInt(el.dataset.page, 10) === p));
}

async function gotoPage(p, smooth = true) {
  p = Math.max(1, Math.min(p, pdfDoc.numPages));
  const div = pageDiv(p);
  if (!div) return;
  div.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
  currentPage = p;
  pageInput.value = p;
  updateActiveThumb(p);
  updateLabel(p);
  emit("page-change", { page: p });
}

function updateLabel(p) {
  const opt = labelSelect.querySelector('option[value="' + p + '"]');
  const text = opt ? opt.textContent : "";
  labelEl.textContent = text ? "· " + text : "";
}

function setScale(newScale) {
  scale = Math.max(0.25, Math.min(newScale, 4));
  fitWidth = false;
  pageSizes.forEach((s, i) => {
    const div = pageDiv(i + 1);
    if (div) { div.style.width = (s.width * scale) + "px"; div.style.height = (s.height * scale) + "px"; }
  });
  rendered.clear();
  scrollEl.querySelectorAll(".qca-page").forEach(div => {
    div.textContent = "Seite " + div.dataset.page;
    div.classList.add("qca-page-placeholder");
    const host = div.querySelector(".qca-overlay-host");
    if (host) host.remove();
  });
  observePages();
  gotoPage(currentPage, false);
}

function fitToWidth() {
  const avail = scrollEl.clientWidth - 32;
  setScale(avail / pageSizes[0].width);
  fitWidth = true;
}

async function search(term) {
  term = (term || "").trim().toLowerCase();
  if (!term || !pdfDoc) return;
  for (let p = 1; p <= pdfDoc.numPages; p++) {
    const page = await pdfDoc.getPage(p);
    const content = await page.getTextContent();
    const text = content.items.map(i => i.str).join(" ").toLowerCase();
    if (text.includes(term)) { await gotoPage(p); emit("search-hit", { page: p, term }); return; }
  }
  emit("search-hit", { page: null, term });
}

function setActiveCoding(codingId) {
  document.querySelectorAll(".qca-overlay-host [data-coding-id]").forEach(el => {
    el.style.filter = el.getAttribute("data-coding-id") === codingId ? "brightness(1.15)" : "";
  });
}

function scrollToQuad(rect) {
  // rect = [x0, y0, x1, y1] im PDF-Koordinatensystem der aktuellen Seite
  if (!rect) return;
  const div = pageDiv(currentPage);
  if (!div) return;
  const top = div.offsetTop + rect[1] * scale;
  scrollEl.scrollTo({ top: Math.max(0, top - 80), behavior: "smooth" });
}

window.qcaViewer = {
  gotoPage, next: () => gotoPage(currentPage + 1), prev: () => gotoPage(currentPage - 1),
  setActiveCoding, scrollToQuad, search, setScale, fitToWidth,
  get page() { return currentPage; }
};

document.getElementById("qca-prev").addEventListener("click", () => gotoPage(currentPage - 1));
document.getElementById("qca-next").addEventListener("click", () => gotoPage(currentPage + 1));
pageInput.addEventListener("change", e => gotoPage(parseInt(e.target.value, 10) || 1));
labelSelect.addEventListener("change", e => gotoPage(parseInt(e.target.value, 10) || 1));
document.getElementById("qca-zoom-in").addEventListener("click", () => setScale(scale * 1.2));
document.getElementById("qca-zoom-out").addEventListener("click", () => setScale(scale / 1.2));
document.getElementById("qca-zoom-fit").addEventListener("click", fitToWidth);
const searchInput = document.getElementById("qca-search");
if (searchInput) {
  searchInput.addEventListener("change", e => search(e.target.value));
  searchInput.addEventListener("keydown", e => { if (e.key === "Enter") search(e.target.value); });
}
scrollEl.addEventListener("keydown", e => {
  if (e.key === "ArrowRight" || e.key === "PageDown") { e.preventDefault(); gotoPage(currentPage + 1); }
  if (e.key === "ArrowLeft" || e.key === "PageUp") { e.preventDefault(); gotoPage(currentPage - 1); }
});

load().catch(err => {
  scrollEl.textContent = "PDF konnte nicht geladen werden: " + err;
  emit("error", { message: String(err) });
});
</script>
</body>
</html>
"""


__all__ = [
    "ViewerConfig",
    "build_viewer_html",
    "render_pdf_viewer",
    "pdf_to_data_uri",
    "DEFAULT_HEIGHT",
    "DEFAULT_PDFJS_VERSION",
]