"""
Webapp UI Components for QCA-AID Streamlit Application
Includes Microsoft Fluent UI Design System implementation
"""

__version__ = "0.1.0"

# Export Fluent UI modules
from .fluent_styles import (
    FluentColors,
    FluentTypography,
    FluentSpacing,
    FluentShadows,
    FluentBorders,
    get_fluent_css,
    get_fluent_component_styles
)

__all__ = [
    # Fluent Styles
    "FluentColors",
    "FluentTypography",
    "FluentSpacing",
    "FluentShadows",
    "FluentBorders",
    "get_fluent_css",
    "get_fluent_component_styles",
]
