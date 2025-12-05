"""
Core-Module für QCA-AID
=======================
Enthält fundamentale Datenmodelle, Validatoren und Konfiguration.
"""

from .data_models import CategoryDefinition, CodingResult, CategoryChange
from .validators import CategoryValidator
from .config import (
    FORSCHUNGSFRAGE, KODIERREGELN,
    VALIDATION_THRESHOLDS, CONFIG
)

__all__ = [
    'CategoryDefinition',
    'CodingResult', 
    'CategoryChange',
    'CategoryValidator',
    'FORSCHUNGSFRAGE',
    'KODIERREGELN',
    'VALIDATION_THRESHOLDS',
    'CONFIG'
]
