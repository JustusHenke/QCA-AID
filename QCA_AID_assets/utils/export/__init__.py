"""
Export Package

Generates output files and provides interactive review tools.

Features:
- PDF annotation with color-coded highlights
- Manual review GUI for coding discrepancies
- Category-specific review workflows
- Visual representation of competing codings
- Decision recording and consensus building
- Text sanitization and formatting for Excel export
- Color palette generation

Exports:
  - PDFAnnotator: Creates annotated PDFs from codings
  - ManualReviewGUI: Simple category review interface
  - ManualReviewComponent: Advanced review with discrepancy handling
  - sanitize_text_for_excel: Clean text for Excel export
  - generate_pastel_colors: Create color palettes
  - format_confidence: Format confidence values
"""

try:
    from .pdf_annotator import PDFAnnotator
except ImportError:
    PDFAnnotator = None

from .review import ManualReviewGUI, ManualReviewComponent
from .helpers import sanitize_text_for_excel, generate_pastel_colors, format_confidence

__all__ = [
    'PDFAnnotator',
    'ManualReviewGUI',
    'ManualReviewComponent',
    'sanitize_text_for_excel',
    'generate_pastel_colors',
    'format_confidence',
]
