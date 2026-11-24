"""
Viewer Bridge
=============
Reine Logikschicht zwischen PDF.js-Viewer und Streamlit-Session
(siehe .kiro/specs/pdf-preview-annotation, Anforderungen 6.3, 6.7, 7.1).

Ohne Browser-Abhaengigkeit testbar:
  - ViewerEvent.parse: entgegennimmt postMessage-Payloads des Viewers
  - ViewerState + apply_event: Zustandsubergang (Seite, aktive Kodierung)
  - build_command: erzeugt JS-Aufrufe an window.qcaViewer (App -> Viewer)
"""

import json
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

VIEWER_SOURCE = "qca-viewer"

EVENT_PAGE_CHANGE = "page-change"
EVENT_CODING_CLICK = "coding-click"
EVENT_SEARCH_HIT = "search-hit"
EVENT_READY = "ready"
EVENT_ERROR = "error"

_KNOWN_EVENTS = {
    EVENT_PAGE_CHANGE,
    EVENT_CODING_CLICK,
    EVENT_SEARCH_HIT,
    EVENT_READY,
    EVENT_ERROR,
}


@dataclass
class ViewerEvent:
    """Aus einem postMessage-Payload geparstes Viewer-Ereignis."""

    type: str
    doc_id: Optional[str] = None
    page: Optional[int] = None
    coding_id: Optional[str] = None
    term: Optional[str] = None
    message: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    @classmethod
    def parse(cls, payload: Any) -> Optional["ViewerEvent"]:
        """Parst dict/JSON-String. Liefert None, wenn es kein Viewer-Event ist."""
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(payload, dict):
            return None
        if payload.get("source") != VIEWER_SOURCE:
            return None
        event_type = payload.get("type")
        if event_type not in _KNOWN_EVENTS:
            return None

        page = payload.get("page")
        page = int(page) if isinstance(page, (int, float)) else None
        return cls(
            type=event_type,
            doc_id=payload.get("docId"),
            page=page,
            coding_id=payload.get("codingId"),
            term=payload.get("term"),
            message=payload.get("message"),
            raw=payload,
        )


@dataclass
class ViewerState:
    """Persistierbarer Viewer-Zustand (Session-State)."""

    current_page: int = 1
    active_coding_id: Optional[str] = None
    ready: bool = False
    last_search: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_page": self.current_page,
            "active_coding_id": self.active_coding_id,
            "ready": self.ready,
            "last_search": self.last_search,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ViewerState":
        return cls(
            current_page=data.get("current_page", 1),
            active_coding_id=data.get("active_coding_id"),
            ready=data.get("ready", False),
            last_search=data.get("last_search"),
            error=data.get("error"),
        )


def apply_event(state: ViewerState, event: ViewerEvent) -> ViewerState:
    """Wendet ein Viewer-Ereignis auf den Zustand an (unveraenderlich)."""
    if event.type == EVENT_PAGE_CHANGE and event.page:
        return replace(state, current_page=event.page)
    if event.type == EVENT_CODING_CLICK:
        new = replace(state, active_coding_id=event.coding_id)
        if event.page:
            new = replace(new, current_page=event.page)
        return new
    if event.type == EVENT_SEARCH_HIT:
        return replace(state, last_search={"page": event.page, "term": event.term})
    if event.type == EVENT_READY:
        return replace(state, ready=True)
    if event.type == EVENT_ERROR:
        return replace(state, error=event.message)
    return state


def apply_events(state: ViewerState, payloads) -> ViewerState:
    """Wendet mehrere Payloads an (ignoriert Fremd-Nachrichten)."""
    for payload in payloads:
        event = ViewerEvent.parse(payload)
        if event is not None:
            state = apply_event(state, event)
    return state


# --------------------------------------------------------------------------
# App -> Viewer: JS-Kommandos
# --------------------------------------------------------------------------

def build_command(name: str, **kwargs) -> str:
    """Erzeugt einen JS-Aufruf an window.qcaViewer."""
    if name == "goto_page":
        page = int(kwargs["page"])
        return f"window.qcaViewer && window.qcaViewer.gotoPage({page});"
    if name == "set_active_coding":
        coding_id = json.dumps(kwargs.get("coding_id"))
        return f"window.qcaViewer && window.qcaViewer.setActiveCoding({coding_id});"
    if name == "scroll_to_quad":
        rect = json.dumps(list(kwargs["rect"]))
        return f"window.qcaViewer && window.qcaViewer.scrollToQuad({rect});"
    raise ValueError(f"unbekanntes Kommando: {name}")


def goto_page_command(page: int) -> str:
    return build_command("goto_page", page=page)


__all__ = [
    "ViewerEvent",
    "ViewerState",
    "apply_event",
    "apply_events",
    "build_command",
    "goto_page_command",
    "VIEWER_SOURCE",
]