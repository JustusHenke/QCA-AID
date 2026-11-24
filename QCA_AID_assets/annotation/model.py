"""
Annotation Data Models
======================
Datenmodelle fuer Vorschau und Annotation (siehe
.kiro/specs/pdf-preview-annotation).

Verwendet dataclasses im Stil der uebrigen Modelle
(QCA_AID_app/webapp_models/*.py), inklusive ``to_dict`` / ``from_dict`` und
einer ``validate()``-Methode fuer die Invarianten aus Anforderung 7.

Kernmodelle:
  - Anchor:        Verankerung einer Kodierung (TextQuoteSelector + Quads)
  - Coding:        Eine Kodierung mit Invarianten
  - DocumentIndex: Strukturiertes Canonical-PDF (Seiten, Marker, Offsets)
  - RenderOptions: Darstellungsoptionen fuer das Overlay
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Erlaubte Werte (bewusst als Konstanten gehalten, damit Modell und Tests
# dieselbe Quelle verwenden).
CODING_STATUS = ("proposed", "accepted", "rejected")
CODING_CREATED_BY = ("ai", "human")
RENDER_MODES = ("highlight", "underline", "margin", "bracket")
BLEND_MODES = ("multiply", "normal")


@dataclass
class Anchor:
    """Verankerung einer Kodierung im Canonical_PDF.

    Persistiert werden ein TextQuoteSelector (quote/prefix/suffix) und die
    geometrischen Quads (je Textzeile, [x0, y0, x1, y1]).
    """

    doc_id: str
    page: int  # 1-basiert
    quote: str
    prefix: str = ""
    suffix: str = ""
    page_label: Optional[str] = None
    quads: List[List[float]] = field(default_factory=list)
    anchored: bool = False

    def validate(self) -> Tuple[bool, List[str]]:
        """Validiert den Anker.

        Returns:
            Tuple[bool, List[str]]: (is_valid, errors)
        """
        errors: List[str] = []
        if not self.doc_id:
            errors.append("doc_id darf nicht leer sein")
        if not isinstance(self.page, int) or self.page < 1:
            errors.append("page muss >= 1 sein")

        # Ein als 'anchored' markierter Anker braucht quote und gueltige Quads.
        if self.anchored:
            if not self.quote or not self.quote.strip():
                errors.append("anchored=True verlangt einen nicht-leeren quote")
            if not self.quads:
                errors.append("anchored=True verlangt mindestens ein Quad")
            for i, quad in enumerate(self.quads):
                if len(quad) != 4:
                    errors.append(f"Quad {i} muss genau 4 Zahlen enthalten")
                elif not all(isinstance(v, (int, float)) for v in quad):
                    errors.append(f"Quad {i} enthaelt Nicht-Zahlen")
                elif not (quad[0] <= quad[2] and quad[1] <= quad[3]):
                    errors.append(f"Quad {i} ist invertiert (x0>x1 oder y0>y1)")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "page": self.page,
            "page_label": self.page_label,
            "quote": self.quote,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "quads": self.quads,
            "anchored": self.anchored,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Anchor":
        return cls(
            doc_id=data["doc_id"],
            page=data["page"],
            quote=data.get("quote", ""),
            prefix=data.get("prefix", ""),
            suffix=data.get("suffix", ""),
            page_label=data.get("page_label"),
            quads=data.get("quads", []),
            anchored=data.get("anchored", False),
        )


@dataclass
class Coding:
    """Eine Kodierung eines Textsegments."""

    segment_id: str
    doc_id: str
    category: str
    subcategories: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    justification: Optional[str] = None
    created_by: str = "ai"
    status: str = "proposed"
    inference: bool = False
    page: Optional[int] = None
    anchor: Optional[Anchor] = None
    #: Wortlaut-Auszuege aus der Analyse-XLSX (Spalte "Textanker") -
    #: Quelle fuer Stufe 1 der Anker-Kette (review_service.anchor_codings).
    text_anchors: List[str] = field(default_factory=list)
    #: Aufgeloeste Anker-Regionen; Primaerregion ist anchors[0] == anchor.
    anchors: List[Anchor] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Haelt Primaer-Anker und Anker-Liste konsistent (E4).

        - Nur ``anchor`` gesetzt (alte Aufrufstellen): ``anchors`` wird
          daraus initialisiert.
        - Nur ``anchors`` gesetzt: ``anchor`` zeigt auf die erste Region
          (Primaerseite).
        - Beide gesetzt: ``anchors[0]`` gewinnt.
        """
        if self.anchors:
            if self.anchor is not self.anchors[0]:
                self.anchor = self.anchors[0]
        elif self.anchor is not None:
            self.anchors = [self.anchor]

    def validate(self, require_anchor: bool = True) -> Tuple[bool, List[str]]:
        """Validiert Kodierung und Invarianten.

        Args:
            require_anchor: Wenn True, ist ein vorhandener und gueltiger Anker
                Pflicht (Anforderung 7.4). Waehrend der Pipeline kann vor der
                Anker-Aufloesung mit False validiert werden.

        Returns:
            Tuple[bool, List[str]]: (is_valid, errors)
        """
        errors: List[str] = []
        if not self.segment_id:
            errors.append("segment_id darf nicht leer sein")
        if not self.doc_id:
            errors.append("doc_id darf nicht leer sein")
        if not self.category or not self.category.strip():
            errors.append("category darf nicht leer sein")
        if self.status not in CODING_STATUS:
            errors.append(
                f"status '{self.status}' ungueltig, erlaubt: {CODING_STATUS}"
            )
        if self.created_by not in CODING_CREATED_BY:
            errors.append(
                f"created_by '{self.created_by}' ungueltig, erlaubt: {CODING_CREATED_BY}"
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            errors.append("confidence muss zwischen 0.0 und 1.0 liegen")

        # Anforderung 7.5: Inferenz-Kodierung braucht eine Begruendung.
        if self.inference and not (self.justification and self.justification.strip()):
            errors.append("Inferenz-Kodierung verlangt eine Begruendung")

        # Anforderung 7.4: Kodierung ohne (gueltigen) Anker ist ungueltig.
        if require_anchor:
            if self.anchor is None:
                errors.append("Pflicht-Anker fehlt")
            else:
                anchor_ok, anchor_errors = self.anchor.validate()
                if not anchor_ok:
                    errors.extend(f"Anker: {e}" for e in anchor_errors)
                if not self.anchor.anchored:
                    errors.append("Anker ist nicht aufgeloest (unanchored)")
                for i, extra in enumerate(self.anchors[1:], start=2):
                    extra_ok, extra_errors = extra.validate()
                    if not extra_ok:
                        errors.extend(f"Zusatzanker {i}: {e}" for e in extra_errors)
                    if not extra.anchored:
                        errors.append(
                            f"Zusatzanker {i} ist nicht aufgeloest (unanchored)"
                        )

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "doc_id": self.doc_id,
            "category": self.category,
            "subcategories": self.subcategories,
            "confidence": self.confidence,
            "justification": self.justification,
            "created_by": self.created_by,
            "status": self.status,
            "inference": self.inference,
            "page": self.page,
            "anchor": self.anchor.to_dict() if self.anchor else None,
            "text_anchors": list(self.text_anchors),
            "anchors": [a.to_dict() for a in self.anchors],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Coding":
        anchor_data = data.get("anchor")
        return cls(
            segment_id=data["segment_id"],
            doc_id=data["doc_id"],
            category=data.get("category", ""),
            subcategories=data.get("subcategories", []),
            confidence=data.get("confidence"),
            justification=data.get("justification"),
            created_by=data.get("created_by", "ai"),
            status=data.get("status", "proposed"),
            inference=data.get("inference", False),
            page=data.get("page"),
            anchor=Anchor.from_dict(anchor_data) if anchor_data else None,
            text_anchors=list(data.get("text_anchors", [])),
            anchors=[
                Anchor.from_dict(a) for a in data.get("anchors", []) if a
            ],
        )


@dataclass
class TextChunk:
    """Ein Text-Chunk mit Seitenzuordnung (fuer seitenbewusstes Chunking)."""

    chunk_index: int
    segment_id: str
    doc_id: str
    text: str
    start: int
    end: int
    page_start: int
    page_end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "segment_id": self.segment_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextChunk":
        return cls(
            chunk_index=data["chunk_index"],
            segment_id=data["segment_id"],
            doc_id=data["doc_id"],
            text=data["text"],
            start=data["start"],
            end=data["end"],
            page_start=data["page_start"],
            page_end=data["page_end"],
        )


@dataclass
class PageText:
    """Text einer Seite mit Offsets im Gesamttext."""

    page: int
    text: str
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return {"page": self.page, "text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageText":
        return cls(page=data["page"], text=data["text"], start=data["start"], end=data["end"])


@dataclass
class Marker:
    """Seitenmarker <<PAGE N>>: Offset-Bereich einer Seite im Gesamttext."""

    page: int
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return {"page": self.page, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Marker":
        return cls(page=data["page"], start=data["start"], end=data["end"])


@dataclass
class DocumentIndex:
    """Strukturartefakt zum Canonical_PDF (document.json)."""

    doc_id: str
    page_count: int
    pages: List[PageText] = field(default_factory=list)
    page_labels: Optional[List[str]] = None
    page_markers: List[Marker] = field(default_factory=list)

    def validate(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if not self.doc_id:
            errors.append("doc_id darf nicht leer sein")
        if self.page_count < 0:
            errors.append("page_count darf nicht negativ sein")
        if len(self.pages) != self.page_count:
            errors.append(
                f"pages ({len(self.pages)}) muss page_count ({self.page_count}) entsprechen"
            )
        if self.page_labels is not None and len(self.page_labels) != self.page_count:
            errors.append("page_labels muss entweder None oder page_count lang sein")
        for marker in self.page_markers:
            if marker.start > marker.end:
                errors.append(f"Marker Seite {marker.page}: start > end")
            if marker.page < 1 or marker.page > self.page_count:
                errors.append(f"Marker Seite {marker.page} liegt ausserhalb 1..{self.page_count}")
        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "page_count": self.page_count,
            "page_labels": self.page_labels,
            "pages": [p.to_dict() for p in self.pages],
            "page_markers": [m.to_dict() for m in self.page_markers],
        }

    def to_marked_text(self, separator: str = "\n") -> str:
        """Rendert den Text mit <<PAGE n>>-Markern (fuer Chunking/LLM)."""
        parts = []
        for page in self.pages:
            parts.append(
                f"<<PAGE {page.page}>>{separator}{page.text}{separator}<<END PAGE {page.page}>>"
            )
        return separator.join(parts)

    def page_for_offset(self, offset: int) -> Optional[int]:
        """Liefert die 1-basierte Seite fuer einen Offset im Gesamttext."""
        for marker in self.page_markers:
            if marker.start <= offset <= marker.end:
                return marker.page
        return None

    def full_text(self, separator: str = "\n") -> str:
        """Rekonstruiert den Gesamttext passend zu den Offsets."""
        return separator.join(page.text for page in self.pages)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentIndex":
        return cls(
            doc_id=data["doc_id"],
            page_count=data["page_count"],
            pages=[PageText.from_dict(p) for p in data.get("pages", [])],
            page_labels=data.get("page_labels"),
            page_markers=[Marker.from_dict(m) for m in data.get("page_markers", [])],
        )


@dataclass
class RenderOptions:
    """Darstellungsoptionen fuer das Annotation-Overlay."""

    mode: str = "highlight"
    active_coding_id: Optional[str] = None
    opacity: float = 0.4
    blend: str = "multiply"
    #: Mehrfachkodierungen derselben Textstelle im Textmarker-Modus zusaetzlich
    #: schraffieren (statt nur als Randstreifen zu zeigen).
    hatch_shared: bool = True
    #: Index der aktiv hervorgehobenen Region innerhalb der aktiven
    #: Kodierung (Spec text-anker-export, R13). None = alle Regionen
    #: der aktiven Kodierung voll deckend.
    active_anchor_index: Optional[int] = None

    def validate(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if self.mode not in RENDER_MODES:
            errors.append(f"mode '{self.mode}' ungueltig, erlaubt: {RENDER_MODES}")
        if not (0.0 <= self.opacity <= 1.0):
            errors.append("opacity muss zwischen 0.0 und 1.0 liegen")
        if self.blend not in BLEND_MODES:
            errors.append(f"blend '{self.blend}' ungueltig, erlaubt: {BLEND_MODES}")
        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "active_coding_id": self.active_coding_id,
            "opacity": self.opacity,
            "blend": self.blend,
            "hatch_shared": self.hatch_shared,
            "active_anchor_index": self.active_anchor_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RenderOptions":
        return cls(
            mode=data.get("mode", "highlight"),
            active_coding_id=data.get("active_coding_id"),
            opacity=data.get("opacity", 0.4),
            blend=data.get("blend", "multiply"),
            hatch_shared=data.get("hatch_shared", True),
            active_anchor_index=data.get("active_anchor_index"),
        )


__all__ = [
    "Anchor",
    "Coding",
    "TextChunk",
    "PageText",
    "Marker",
    "DocumentIndex",
    "RenderOptions",
    "CODING_STATUS",
    "CODING_CREATED_BY",
    "RENDER_MODES",
    "BLEND_MODES",
]
