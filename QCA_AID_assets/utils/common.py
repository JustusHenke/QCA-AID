"""
Common Utilities for QCA-AID Utils Package

Shared constants, types, and helper functions used across multiple modules.
"""

from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import os


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class AnalysisMode(str, Enum):
    """Supported analysis modes"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    GROUNDED = "grounded"


class ModelProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    MISTRAL = "mistral"
    CLAUDE = "claude"


class DocumentFormat(str, Enum):
    """Supported document formats"""
    TEXT = "txt"
    DOCX = "docx"
    PDF = "pdf"


# File extensions mapping
DOCUMENT_EXTENSIONS = {
    ".txt": DocumentFormat.TEXT,
    ".docx": DocumentFormat.DOCX,
    ".pdf": DocumentFormat.PDF,
}

SUPPORTED_EXTENSIONS = set(DOCUMENT_EXTENSIONS.keys())

# Default configuration paths
DEFAULT_CONFIG_FILE = "QCA-AID-Codebook.xlsx"
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"

# API timeout and retry settings
DEFAULT_API_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2  # exponential backoff multiplier

# Token limits by model family
TOKEN_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-5": 200000,
    "gpt-5-nano": 200000,
    "gpt-5-mini": 200000,
    "mistral-large": 32000,
    "mistral-medium": 32000,
    "claude-3": 200000,
}

# Default temperatures per component
DEFAULT_TEMPERATURES = {
    "deductive_coder": 0.3,
    "inductive_coder": 0.2,
    "relevance_checker": 0.3,
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ModelInfo:
    """Information about an LLM model"""
    name: str
    provider: ModelProvider
    max_tokens: int
    cost_per_1m_input: float  # in dollars
    cost_per_1m_output: float  # in dollars
    supports_json_mode: bool = True
    supports_temperature: Optional[bool] = None  # None = unknown


@dataclass
class TokenUsage:
    """Token usage statistics for a single request"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    cost: float


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_family(model_name: str) -> str:
    """
    Extract model family from model name.
    
    Examples:
        "gpt-4o-mini" -> "gpt-4o"
        "gpt-5-nano" -> "gpt-5"
        "mistral-large-latest" -> "mistral-large"
    """
    if "gpt-4o" in model_name:
        return "gpt-4o"
    elif "gpt-5" in model_name:
        return "gpt-5"
    elif "gpt-4" in model_name:
        return "gpt-4"
    elif "mistral" in model_name:
        if "large" in model_name:
            return "mistral-large"
        elif "medium" in model_name:
            return "mistral-medium"
        else:
            return "mistral"
    elif "claude" in model_name:
        return "claude-3"
    else:
        return model_name


def clean_text(text: str) -> str:
    """
    Remove problematic characters and normalize whitespace.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text safe for processing
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove other problematic characters
    problematic_chars = ['\x01', '\x02', '\x03', '\x04', '\x05']
    for char in problematic_chars:
        text = text.replace(char, '')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Absolute path to directory
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def detect_document_format(file_path: str) -> Optional[DocumentFormat]:
    """
    Detect document format from file extension.
    
    Args:
        file_path: Path to document file
        
    Returns:
        DocumentFormat enum or None if unsupported
    """
    _, ext = os.path.splitext(file_path.lower())
    return DOCUMENT_EXTENSIONS.get(ext)


def format_cost(cost: float) -> str:
    """Format cost as USD currency string"""
    return f"${cost:.6f}"


def format_tokens(tokens: int) -> str:
    """Format token count with thousands separator"""
    return f"{tokens:,}"


# ============================================================================
# VERSION INFO
# ============================================================================

UTILS_VERSION = "2.0.0"
REFACTORING_DATE = "2025-11-17"
ORIGINAL_LINES = 3954
TARGET_LINES = 2000  # Estimated after refactoring


def create_filter_string(filters: Dict[str, str]) -> str:
    """
    Erstellt eine String-Repräsentation der Filter für Dateinamen.
    
    Diese Funktion wird verwendet, um aus einem Dictionary von Filtern
    einen kompakten String zu erstellen, der in Dateinamen verwendet werden kann.
    Leere Filter-Werte werden automatisch übersprungen.
    Lange Werte (z.B. mehrere Kategorien) werden gekürzt, um Pfadlängenbeschränkungen zu vermeiden.
    
    Args:
        filters: Dictionary mit Filter-Parametern (z.B. {'Hauptkategorie': 'Kategorie1', 'Dokument': 'Doc1'})
        
    Returns:
        String-Repräsentation der Filter im Format "key1-value1_key2-value2"
        
    Examples:
        >>> create_filter_string({'Hauptkategorie': 'Kategorie1', 'Dokument': 'Doc1'})
        'Hauptkategorie-Kategorie1_Dokument-Doc1'
        
        >>> create_filter_string({'Hauptkategorie': 'Kategorie1', 'Dokument': ''})
        'Hauptkategorie-Kategorie1'
        
        >>> create_filter_string({'Hauptkategorie': 'Kat1, Kat2, Kat3'})
        'Hauptkategorie-3cats'
    """
    parts = []
    for k, v in filters.items():
        if not v:
            continue
        
        # Shorten long values (e.g., multiple categories)
        if ',' in str(v):
            # Multiple values - count them
            values = [x.strip() for x in str(v).split(',') if x.strip()]
            if len(values) > 1:
                # Use count instead of full list
                v = f"{len(values)}items"
        
        # Limit individual value length to avoid path issues
        v_str = str(v)
        if len(v_str) > 50:
            v_str = v_str[:47] + "..."
        
        # Sanitize for filename (remove invalid characters)
        v_str = v_str.replace('/', '-').replace('\\', '-').replace(':', '-')
        
        parts.append(f"{k}-{v_str}")
    
    return '_'.join(parts)
