#!/usr/bin/env python3
"""
Microsoft Fluent UI Design System für QCA-AID Webapp - SAFE VERSION
===================================================================
Sichere Implementierung von Fluent UI Design-Prinzipien.
Ändert NUR Farben und Font, KEINE Layout-Eigenschaften (Padding, Border-Width, Height).

Basierend auf detaillierter Analyse in STREAMLIT_FLUENT_ANALYSIS.md

Quellen:
- https://fluent2.microsoft.design/layout
- https://storybooks.fluentui.dev/react/?path=/docs/theme-colors--docs
"""

from typing import Dict, Any


class FluentColors:
    """Fluent UI Color Palette"""
    
    # Brand Colors
    BRAND_PRIMARY = "#0078D4"  # Fluent Blue
    BRAND_SECONDARY = "#106EBE"
    BRAND_TERTIARY = "#005A9E"
    
    # Neutral Colors (Light Theme)
    NEUTRAL_BACKGROUND = "#FFFFFF"
    NEUTRAL_BACKGROUND_SECONDARY = "#F5F5F5"
    NEUTRAL_BACKGROUND_TERTIARY = "#EBEBEB"
    
    NEUTRAL_FOREGROUND = "#242424"
    NEUTRAL_FOREGROUND_SECONDARY = "#605E5C"
    NEUTRAL_FOREGROUND_TERTIARY = "#8A8886"
    
    # Stroke Colors
    NEUTRAL_STROKE = "#E1DFDD"
    NEUTRAL_STROKE_ACCESSIBLE = "#C8C6C4"
    
    # Semantic Colors
    SUCCESS = "#107C10"
    WARNING = "#F7630C"
    ERROR = "#D13438"
    INFO = "#0078D4"
    
    # Hover & Active States
    HOVER_BACKGROUND = "#F3F2F1"
    ACTIVE_BACKGROUND = "#EDEBE9"


class FluentTypography:
    """Fluent UI Typography System"""
    
    # Font Family (Segoe UI with fallbacks)
    FONT_FAMILY = "'Segoe UI', 'Segoe UI Web', -apple-system, BlinkMacSystemFont, 'Roboto', 'Helvetica Neue', sans-serif"


class FluentSpacing:
    """Fluent UI Spacing System (4px Grid)"""
    
    XXS = "4px"
    XS = "8px"
    S = "12px"
    M = "16px"
    L = "20px"
    XL = "24px"
    XXL = "32px"


class FluentShadows:
    """Fluent UI Shadow/Elevation System"""
    
    SHADOW_2 = "0 0 2px rgba(0,0,0,0.12), 0 2px 4px rgba(0,0,0,0.14)"
    SHADOW_4 = "0 0 2px rgba(0,0,0,0.12), 0 4px 8px rgba(0,0,0,0.14)"
    SHADOW_8 = "0 0 2px rgba(0,0,0,0.12), 0 8px 16px rgba(0,0,0,0.14)"


class FluentBorders:
    """Fluent UI Border System"""
    
    RADIUS_MEDIUM = "4px"
    RADIUS_LARGE = "8px"
    WIDTH_THIN = "1px"


def get_fluent_css() -> str:
    """
    Generiert SICHERES CSS für Fluent UI Design in Streamlit.
    Ändert NUR Farben und Font, KEINE Layout-Eigenschaften.
    
    Returns:
        CSS-String für st.markdown(unsafe_allow_html=True)
    """
    return f"""
    <style>
    /* ===== FLUENT UI SAFE STYLES ===== */
    /* Ändert NUR Farben und Font, KEINE Größen/Padding/Borders */
    
    /* Import Segoe UI Font */
    @import url('https://fonts.cdnfonts.com/css/segoe-ui-4');
    
    /* ===== TYPOGRAPHY ===== */
    
    /* Global Font Family */
    html, body, [class*="css"], .stApp {{
        font-family: {FluentTypography.FONT_FAMILY};
    }}
    
    /* ===== COLORS ===== */
    
    /* Primary Button Color */
    button[kind="primary"],
    .stButton > button[kind="primary"] {{
        background-color: {FluentColors.BRAND_PRIMARY};
        border-color: {FluentColors.BRAND_PRIMARY};
    }}
    
    button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:hover {{
        background-color: {FluentColors.BRAND_SECONDARY};
        border-color: {FluentColors.BRAND_SECONDARY};
    }}
    
    button[kind="primary"]:active,
    .stButton > button[kind="primary"]:active {{
        background-color: {FluentColors.BRAND_TERTIARY};
        border-color: {FluentColors.BRAND_TERTIARY};
    }}
    
    /* Secondary Button Hover */
    .stButton > button:hover:not([kind="primary"]) {{
        background-color: {FluentColors.HOVER_BACKGROUND};
    }}
    
    /* Input Border Colors */
    input, textarea, select {{
        border-color: {FluentColors.NEUTRAL_STROKE};
    }}
    
    input:focus, textarea:focus, select:focus {{
        border-color: {FluentColors.BRAND_PRIMARY};
        outline-color: {FluentColors.BRAND_PRIMARY};
    }}
    
    /* Select Box Border */
    .stSelectbox > div > div {{
        border-color: {FluentColors.NEUTRAL_STROKE};
    }}
    
    /* Tab Active Color */
    .stTabs [aria-selected="true"] {{
        border-bottom-color: {FluentColors.BRAND_PRIMARY};
        color: {FluentColors.BRAND_PRIMARY};
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {FluentColors.HOVER_BACKGROUND};
    }}
    
    /* Sidebar Background */
    [data-testid="stSidebar"] {{
        background-color: {FluentColors.NEUTRAL_BACKGROUND_SECONDARY};
    }}
    
    /* Expander Header Background */
    .streamlit-expanderHeader {{
        background-color: {FluentColors.NEUTRAL_BACKGROUND_SECONDARY};
    }}
    
    .streamlit-expanderHeader:hover {{
        background-color: {FluentColors.HOVER_BACKGROUND};
    }}
    
    /* ===== SCROLLBAR ===== */
    
    ::-webkit-scrollbar {{
        width: 12px;
        height: 12px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {FluentColors.NEUTRAL_BACKGROUND_SECONDARY};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {FluentColors.NEUTRAL_STROKE_ACCESSIBLE};
        border-radius: {FluentBorders.RADIUS_LARGE};
        border: 2px solid {FluentColors.NEUTRAL_BACKGROUND_SECONDARY};
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {FluentColors.NEUTRAL_FOREGROUND_TERTIARY};
    }}
    
    /* ===== LOADING SPINNER ===== */
    
    .stSpinner > div {{
        border-top-color: {FluentColors.BRAND_PRIMARY};
    }}
    
    /* ===== CUSTOM STYLES ===== */
    /* Füge hier deine eigenen CSS-Styles hinzu */
    
    /* Beispiel: st-emotion Klassen stylen */
    /* [class*="st-emotion"] {{
        /* Deine Styles hier */
    /* }} */
    
    /* Beispiel: Spezifische Komponente */
    /* .st-emotion-cache-xyz {{
        background-color: #F5F5F5;
        border-radius: 4px;
    }} */

    .st-emotion-cache-zy6yx3 {{
        width: 100%;
        padding: 3rem;
        max-width: initial;
        min-width: auto;
    }}
    .st-emotion-cache-1046a32 {{
        font-family: "Source Sans", sans-serif;
        font-size: 1rem;
        color: inherit;
        max-width: 100%;
        overflow-wrap: break-word;
    }}
    .st-emotion-cache-1diky36 {{
        font-size: 1rem;
        color: rgb(36, 36, 36);
        padding-bottom: 0.25rem;
    }}
    
    </style>
    """


def get_fluent_component_styles() -> Dict[str, Any]:
    """
    Gibt Fluent UI Styles als Dictionary zurück für programmatische Verwendung.
    
    Returns:
        Dictionary mit Style-Definitionen
    """
    return {
        "colors": {
            "primary": FluentColors.BRAND_PRIMARY,
            "secondary": FluentColors.BRAND_SECONDARY,
            "background": FluentColors.NEUTRAL_BACKGROUND,
            "background_secondary": FluentColors.NEUTRAL_BACKGROUND_SECONDARY,
            "foreground": FluentColors.NEUTRAL_FOREGROUND,
            "foreground_secondary": FluentColors.NEUTRAL_FOREGROUND_SECONDARY,
            "success": FluentColors.SUCCESS,
            "warning": FluentColors.WARNING,
            "error": FluentColors.ERROR,
            "info": FluentColors.INFO,
        },
        "spacing": {
            "xxs": FluentSpacing.XXS,
            "xs": FluentSpacing.XS,
            "s": FluentSpacing.S,
            "m": FluentSpacing.M,
            "l": FluentSpacing.L,
            "xl": FluentSpacing.XL,
            "xxl": FluentSpacing.XXL,
        },
        "typography": {
            "font_family": FluentTypography.FONT_FAMILY,
        },
        "borders": {
            "radius": FluentBorders.RADIUS_MEDIUM,
            "width": FluentBorders.WIDTH_THIN,
        },
        "shadows": {
            "small": FluentShadows.SHADOW_2,
            "medium": FluentShadows.SHADOW_4,
            "large": FluentShadows.SHADOW_8,
        }
    }
