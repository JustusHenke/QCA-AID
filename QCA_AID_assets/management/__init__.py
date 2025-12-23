"""
Management-Module für QCA-AID
==============================
Enthält Klassen für Kategorien-Management und Versionierung.
"""

from .category_manager import CategoryManager
from .category_revision import CategoryRevisionManager

__all__ = [
    'CategoryManager',
    'CategoryRevisionManager'
]
