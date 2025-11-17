"""
Management-Module für QCA-AID
==============================
Enthält Klassen für Kategorien-Management und Versionierung.
"""

from .category_manager import CategoryManager
from .category_revision import CategoryRevisionManager
from .development_history import DevelopmentHistory

__all__ = [
    'CategoryManager',
    'CategoryRevisionManager',
    'DevelopmentHistory'
]
