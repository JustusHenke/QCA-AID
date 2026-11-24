"""
Annotation Package
==================
Datenmodelle, Anker-Auflösung und Rendering fuer die Dokumenten-Vorschau und
-annotation in QCA-AID.

Module:
  - model: Datenmodelle (Anchor, Coding, DocumentIndex, RenderOptions)
  - anchor: AnchorResolver (TextQuoteSelector + zeilenweise Quads)
"""

from .model import (
    Anchor,
    Coding,
    DocumentIndex,
    Marker,
    PageText,
    RenderOptions,
)
from .anchor import AnchorResolver
from .renderer import OverlayRenderer, DEFAULT_PALETTE
from .export_pdf import PdfBakingExporter
from .export_data import build_rows, export_csv, export_xlsx, export as export_data

__all__ = [
    "Anchor",
    "Coding",
    "DocumentIndex",
    "Marker",
    "PageText",
    "TextChunk",
    "RenderOptions",
    "AnchorResolver",
    "OverlayRenderer",
    "DEFAULT_PALETTE",
    "PdfBakingExporter",
    "build_rows",
    "export_csv",
    "export_xlsx",
    "export_data",
]