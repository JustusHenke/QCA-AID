"""
Configuration Package

Loads and manages QCA-AID configuration from Excel codebook files.
Handles:
- Category definitions and hierarchies
- Coding rules and exclusions
- Multi-coder settings
- Model and API configuration
- Temperature settings per coder

No LLM calls, no GUI code, no I/O except Excel reading.

Exports:
  - ConfigLoader: Main configuration loader class
"""

from .loader import ConfigLoader

__all__ = [
    'ConfigLoader',
]
