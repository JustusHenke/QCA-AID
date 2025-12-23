"""
Core-Module für QCA-AID
=======================
Enthält fundamentale Datenmodelle, Validatoren und Konfiguration.
Extended with Dynamic Cache System data models.
"""

from .data_models import CategoryDefinition, CodingResult, CategoryChange, ExtendedCodingResult
from .validators import CategoryValidator
from .config import (
    FORSCHUNGSFRAGE, KODIERREGELN,
    VALIDATION_THRESHOLDS, CONFIG
)

__all__ = [
    'CategoryDefinition',
    'CodingResult', 
    'CategoryChange',
    'ExtendedCodingResult',
    'CategoryValidator',
    'FORSCHUNGSFRAGE',
    'KODIERREGELN',
    'VALIDATION_THRESHOLDS',
    'CONFIG'
]
