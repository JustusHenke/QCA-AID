"""
Export Package

Generates output files and provides interactive review tools.

Features:
- PDF annotation with color-coded highlights
- Manual review GUI for coding discrepancies
- Category-specific review workflows
- Visual representation of competing codings
- Decision recording and consensus building

Exports:
  - PDFAnnotator: Creates annotated PDFs from codings
  - ManualReviewGUI: Simple category review interface
  - ManualReviewComponent: Advanced review with discrepancy handling
"""

from .pdf_annotator import PDFAnnotator
from .review import ManualReviewGUI, ManualReviewComponent

__all__ = [
    'PDFAnnotator',
    'ManualReviewGUI',
    'ManualReviewComponent',
]
