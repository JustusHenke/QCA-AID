"""
QCA-AID Utils Package

Refactored utility modules split from the monolithic QCA_Utils.py.
Provides LLM providers, configuration loading, token tracking, GUI dialogs,
document I/O, and export functionality.

Subpackages:
  - llm: LLM provider implementations (OpenAI, Mistral)
  - config: Configuration loading and validation
  - tracking: Token usage and cost tracking
  - dialog: Tkinter GUI components
  - export: PDF annotation and manual review
  - io: Document reading and I/O handling
"""

__version__ = "2.0.0"
__author__ = "QCA-AID Team"

# Import and re-export main classes for convenient access
# This will be populated as modules are migrated
