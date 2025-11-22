"""
QCA-AID Utils Package

Refactored utility modules split from the monolithic QCA_Utils.py.
Provides LLM providers, configuration loading, token tracking, GUI dialogs,
document I/O, export functionality, and system utilities.

Subpackages:
  - llm: LLM provider implementations (OpenAI, Mistral)
  - config: Configuration loading and validation
  - tracking: Token usage and cost tracking
  - dialog: Tkinter GUI components
  - export: PDF annotation and manual review
  - io: Document reading and I/O handling
  - system: System utilities (Tkinter patches, input handling)

Common Utilities:
  - common: Shared constants, enums, and helper functions
"""

__version__ = "0.10.1"
__author__ = "QCA-AID Team"

# Import system utilities for convenient access
from .system import patch_tkinter_for_threaded_exit, get_input_with_timeout

__all__ = [
    'patch_tkinter_for_threaded_exit',
    'get_input_with_timeout',
]
