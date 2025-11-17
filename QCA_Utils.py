"""
QCA_Utils.py - Backward Compatibility Layer

This module provides backward compatibility by re-exporting all classes and functions
from the new modular utils package structure. All imports from the old monolithic
QCA_Utils.py continue to work without modification.

Migration Path:
  Old: from QCA_Utils import TokenTracker
  New: from QCA_AID_assets.utils.tracking import TokenTracker
  Both work seamlessly with this compatibility layer.
"""

# ============================================================================
# Tracking Module - Token tracking and cost calculation
# ============================================================================
from QCA_AID_assets.utils.tracking.token_tracker import TokenTracker
from QCA_AID_assets.utils.tracking.token_counter import TokenCounter

# ============================================================================
# LLM Module - Language Model Providers and Response Handling
# ============================================================================
from QCA_AID_assets.utils.llm.response import LLMResponse
from QCA_AID_assets.utils.llm.base import LLMProvider
from QCA_AID_assets.utils.llm.openai_provider import OpenAIProvider
from QCA_AID_assets.utils.llm.mistral_provider import MistralProvider
from QCA_AID_assets.utils.llm.factory import LLMProviderFactory

# ============================================================================
# Configuration Module - Config loading and management
# ============================================================================
from QCA_AID_assets.utils.config.loader import ConfigLoader

# ============================================================================
# Dialog Module - GUI components for manual coding
# ============================================================================
from QCA_AID_assets.utils.dialog.widgets import MultiSelectListbox
from QCA_AID_assets.utils.dialog.multiple_coding import ManualMultipleCodingDialog

# ============================================================================
# Export Module - Results export and review functionality
# ============================================================================
try:
    from QCA_AID_assets.utils.export.pdf_annotator import PDFAnnotator
    from QCA_AID_assets.utils.export.review import ManualReviewGUI, ManualReviewComponent
except ImportError:
    # fuzzywuzzy may not be installed
    PDFAnnotator = None
    ManualReviewGUI = None
    ManualReviewComponent = None

# ============================================================================
# I/O Module - Document reading and escape handler
# ============================================================================
from QCA_AID_assets.utils.io.document_reader import DocumentReader
from QCA_AID_assets.utils.io.escape_handler import EscapeHandler

# ============================================================================
# Common Module - Shared enums, constants, and utilities
# ============================================================================
from QCA_AID_assets.utils.common import (
    AnalysisMode,
    ModelProvider,
    DocumentFormat,
    ModelInfo,
    TokenUsage,
    get_model_family,
    clean_text,
    ensure_dir,
    detect_document_format,
    format_tokens,
    format_cost,
)

# ============================================================================
# Module Re-exports for convenience
# ============================================================================
__all__ = [
    # Tracking
    'TokenTracker',
    'TokenCounter',
    # LLM
    'LLMResponse',
    'LLMProvider',
    'OpenAIProvider',
    'MistralProvider',
    'LLMProviderFactory',
    # Configuration
    'ConfigLoader',
    # Dialog/GUI
    'MultiSelectListbox',
    'ManualMultipleCodingDialog',
    # Export
    'PDFAnnotator',
    'ManualReviewGUI',
    'ManualReviewComponent',
    # I/O
    'DocumentReader',
    'EscapeHandler',
    # Common
    'AnalysisMode',
    'ModelProvider',
    'DocumentFormat',
    'ModelInfo',
    'TokenUsage',
    'get_model_family',
    'clean_text',
    'ensure_dir',
    'detect_document_format',
    'format_tokens',
    'format_cost',
]
