"""
I/O Package

Document reading and safe application shutdown handling.

Features:
- Multi-format document support (TXT, DOCX, PDF)
- Text extraction and metadata handling
- Problematic character cleaning
- ESC-key handling for safe shutdown
- Intermediate result export on abort

Exports:
  - DocumentReader: Multi-format document parser
  - EscapeHandler: Signal handling for graceful shutdown
"""

from .document_reader import DocumentReader
from .escape_handler import EscapeHandler

__all__ = [
    'DocumentReader',
    'EscapeHandler',
]
