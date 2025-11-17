"""
Dialog Package

Tkinter GUI components for user interactions during analysis.

Features:
- Multi-select listbox with Ctrl+Click support
- Manual multiple coding dialog
- Category confirmation dialogs
- User choice collection

All components are 100% Tkinter-based with no external dependencies.

Exports:
  - MultiSelectListbox: Custom multi-select widget
  - ManualMultipleCodingDialog: Coding confirmation dialog
"""

from .widgets import MultiSelectListbox
from .multiple_coding import ManualMultipleCodingDialog

__all__ = [
    'MultiSelectListbox',
    'ManualMultipleCodingDialog',
]
