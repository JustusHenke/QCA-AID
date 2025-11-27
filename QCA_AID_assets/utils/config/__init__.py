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
  - ConfigLoader: Main configuration loader class for QCA-AID
  - ExplorerConfigLoader: Configuration loader class for QCA-AID-Explorer
  - ConfigConverter: Converts between XLSX and JSON formats
  - ConfigSynchronizer: Synchronizes XLSX and JSON configurations
"""

from .loader import ConfigLoader
from .explorer_loader import ExplorerConfigLoader
from .converter import ConfigConverter
from .synchronizer import ConfigSynchronizer

__all__ = [
    'ConfigLoader',
    'ExplorerConfigLoader',
    'ConfigConverter',
    'ConfigSynchronizer',
]
