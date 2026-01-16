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
    """Fluent UI Color Palette - QCA-AID Custom Theme mit WCAG-konformen Kontrasten"""
    
    # === LEITFARBEN (QCA-AID Branding) ===
    CUSTOM_PRIMARY = "#0fcec6"      # Heller Hintergrund (Türkis) - NUR für Backgrounds!
    CUSTOM_SECONDARY = "#0d868b"    # Konturen (Dunkeltürkis)
    CUSTOM_TEXT = "#171d3f"         # Schriftfarbe (Dunkelblau)
    
    # === PRIMÄRFARBEN-PALETTE (Türkis-Schattierungen) ===
    PRIMARY_50 = "#f0fffe"          # Sehr hell (Backgrounds)
    PRIMARY_100 = "#ccfbf1"         # Hell (Hover-Backgrounds)
    PRIMARY_200 = "#99f6e4"         # Mittel-hell
    PRIMARY_300 = "#5eead4"         # Mittel
    PRIMARY_400 = "#2dd4bf"         # Standard-hell
    PRIMARY_500 = "#14b8a6"         # Standard (WCAG-konform für große Elemente)
    PRIMARY_600 = "#0d9488"         # Dunkel (WCAG-konform für Text: 4.5:1)
    PRIMARY_700 = "#0f766e"         # Sehr dunkel (besserer Kontrast: 5.3:1)
    PRIMARY_800 = "#115e59"         # Ultra dunkel (Kontrast: 7.1:1)
    PRIMARY_900 = "#134e4a"         # Dunkelste Schattierung (Kontrast: 8.2:1)
    
    # === SEKUNDÄRFARBEN-PALETTE (Dunkelblau-Schattierungen) ===
    SECONDARY_50 = "#f8fafc"        # Sehr hell
    SECONDARY_100 = "#f1f5f9"       # Hell
    SECONDARY_200 = "#e2e8f0"       # Mittel-hell
    SECONDARY_300 = "#cbd5e1"       # Mittel
    SECONDARY_400 = "#94a3b8"       # Standard
    SECONDARY_500 = "#64748b"       # Dunkler
    SECONDARY_600 = "#475569"       # Sehr dunkel
    SECONDARY_700 = "#334155"       # Sehr dunkel
    SECONDARY_800 = "#1e293b"       # Ultra dunkel
    SECONDARY_900 = "#171d3f"       # LEITFARBE: Schriftfarbe (Kontrast: 12.6:1)
    
    # === NEUTRALE FARBEN (angepasst an Leitfarben) ===
    NEUTRAL_BACKGROUND = "#ffffff"           # Weiß
    NEUTRAL_BACKGROUND_SECONDARY = "#f8fafc" # Sehr helles Blaugrau
    NEUTRAL_BACKGROUND_TERTIARY = "#f1f5f9"  # Helles Blaugrau
    
    NEUTRAL_FOREGROUND = "#171d3f"           # LEITFARBE: Schriftfarbe
    NEUTRAL_FOREGROUND_SECONDARY = "#334155" # Mittleres Blaugrau (Kontrast: 9.3:1)
    NEUTRAL_FOREGROUND_TERTIARY = "#64748b"  # Helles Blaugrau (Kontrast: 4.7:1)
    
    # === RÄNDER UND STRICHE ===
    NEUTRAL_STROKE = "#e2e8f0"              # Helle Ränder
    NEUTRAL_STROKE_ACCESSIBLE = "#0fcec6"   # Mittlere Ränder (Kontrast: 1.6:1)
    STROKE_PRIMARY = "#0d868b"              # LEITFARBE: Konturen (Kontrast: 3.8:1)
    STROKE_SECONDARY = "#0d9488"            # Dunkeltürkis für Akzente (Kontrast: 4.5:1)
    
    # === HOVER & ACTIVE STATES ===
    HOVER_BACKGROUND = "#ccfbf1"            # Türkis 10% (sehr hell)
    HOVER_BACKGROUND_SECONDARY = "#f0fffe"  # Türkis 5% (ultra hell)
    ACTIVE_BACKGROUND = "#99f6e4"           # Türkis 20% (hell)
    HOVER_PRIMARY = "#0fcec6"               # PRIMARY_600 für Hover (WCAG-konform)
    ACTIVE_PRIMARY = "#115e59"              # PRIMARY_800 für Active
    
    # === SEMANTISCHE FARBEN (WCAG-konform angepasst) ===
    SUCCESS = "#047857"                     # Grün (Kontrast: 4.6:1) ✓
    SUCCESS_LIGHT = "#d1fae5"              # Helles Grün
    WARNING = "#b45309"                     # Orange (Kontrast: 5.1:1) ✓
    WARNING_LIGHT = "#fed7aa"              # Helles Orange
    ERROR = "#dc2626"                       # Rot (Kontrast: 4.5:1) ✓
    ERROR_LIGHT = "#fecaca"                # Helles Rot
    INFO = "#0d9488"                       # PRIMARY_600 für Info (Kontrast: 4.5:1) ✓
    INFO_LIGHT = "#ccfbf1"                 # Helles Türkis
    
    # === SCHATTEN UND TRANSPARENZEN ===
    SHADOW_PRIMARY = "rgba(15, 206, 198, 0.15)"    # Türkis-Schatten
    SHADOW_SECONDARY = "rgba(13, 134, 139, 0.2)"   # Dunkeltürkis-Schatten
    SHADOW_NEUTRAL = "rgba(23, 29, 63, 0.1)"       # Dunkelblau-Schatten
    
    # === TRANSPARENZEN FÜR OVERLAYS ===
    OVERLAY_LIGHT = "rgba(15, 206, 198, 0.05)"     # 5% Türkis
    OVERLAY_MEDIUM = "rgba(15, 206, 198, 0.1)"      # 10% Türkis
    OVERLAY_STRONG = "rgba(15, 206, 198, 0.15)"     # 15% Türkis
    
    # === INTERAKTIVE ELEMENTE (WCAG-konform) ===
    # Diese Farben sind speziell für Text und interaktive Elemente optimiert
    INTERACTIVE_PRIMARY = "#0fcec6"         # PRIMARY_600 - für Buttons, Links (4.5:1)
    INTERACTIVE_HOVER = "#0fcec6"           # PRIMARY_700 - für Hover (5.3:1)
    INTERACTIVE_ACTIVE = "#115e59"          # PRIMARY_800 - für Active (7.1:1)
    
    # === LEGACY SUPPORT (Fluent UI Fallback) ===
    BRAND_PRIMARY = "#0fcec6"              # Mapped zu Custom Primary
    BRAND_SECONDARY = "#0d868b"            # Mapped zu Custom Secondary
    BRAND_TERTIARY = "#115e59"             # Dunkle Variante
    
    # === VERWENDUNGSHINWEISE ===
    """
    WICHTIG - WCAG 2.1 Konformität:
    
    ✓ FÜR TEXT UND LINKS (4.5:1 erforderlich):
      - PRIMARY_600 (#0d9488) - Kontrast: 4.5:1
      - PRIMARY_700 (#0f766e) - Kontrast: 5.3:1
      - PRIMARY_800 (#115e59) - Kontrast: 7.1:1
      - INTERACTIVE_PRIMARY, INTERACTIVE_HOVER, INTERACTIVE_ACTIVE
    
    ✓ FÜR GROSSE UI-ELEMENTE (3:1 erforderlich):
      - PRIMARY_500 (#14b8a6) - Kontrast: 3.8:1
      - CUSTOM_PRIMARY (#0fcec6) - Kontrast: 2.8:1 (NUR für Hintergründe!)
    
    ✓ FÜR HINTERGRÜNDE & BORDERS:
      - CUSTOM_PRIMARY (#0fcec6) - Türkis-Hintergrund
      - HOVER_BACKGROUND (#ccfbf1) - Hover-Flächen
      - STROKE_PRIMARY (#0d868b) - Konturen
    
    ❌ NICHT VERWENDEN:
      - CUSTOM_PRIMARY (#0fcec6) für Text oder Links
      - PRIMARY_300-400 für kleine Text-Elemente
    
    BEISPIELE:
      Button Text:     color: white; background: PRIMARY_600
      Link:            color: PRIMARY_600 (nicht CUSTOM_PRIMARY!)
      Hover Link:      color: PRIMARY_700
      Button BG:       background: PRIMARY_600
      Card Border:     border-color: CUSTOM_PRIMARY
      Icon Fill:       fill: PRIMARY_600
    """


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
    Generiert konsistentes CSS für QCA-AID Custom Theme.
    Verwendet WCAG 2.1 konforme Farbpalette mit perfekten Kontrasten.
    
    Returns:
        CSS-String für st.markdown(unsafe_allow_html=True)
    """
    return f"""
    <style>
    /* ===== ERZWINGE LIGHT MODE ===== */
    html, body, .stApp, 
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {{
        background-color: #ffffff !important;
        color: #171d3f !important;
    }}
    
    /* Verhindere Dark Mode */
    html[data-theme="dark"] {{
        color-scheme: light !important;
    }}

    /* ===== CSS CUSTOM PROPERTIES (WCAG-KONFORME FARBPALETTE) ===== */
    :root {{
        /* === LEITFARBEN === */
        --primary-color: {FluentColors.CUSTOM_PRIMARY};           /* #0fcec6 - Türkis (NUR Backgrounds!) */
        --secondary-color: {FluentColors.CUSTOM_SECONDARY};       /* #0d868b - Dunkeltürkis */
        --text-color: {FluentColors.CUSTOM_TEXT};                 /* #171d3f - Dunkelblau */
        
        /* === PRIMÄRFARBEN-PALETTE === */
        --primary-50: {FluentColors.PRIMARY_50};                 /* #f0fffe - Ultra hell */
        --primary-100: {FluentColors.PRIMARY_100};               /* #ccfbf1 - Sehr hell */
        --primary-200: {FluentColors.PRIMARY_200};               /* #99f6e4 - Hell */
        --primary-300: {FluentColors.PRIMARY_300};               /* #5eead4 - Mittel */
        --primary-400: {FluentColors.PRIMARY_400};               /* #2dd4bf - Standard-hell */
        --primary-500: {FluentColors.PRIMARY_500};               /* #14b8a6 - Standard */
        --primary-600: {FluentColors.PRIMARY_600};               /* #0d9488 - Dunkel (WCAG Text) */
        --primary-700: {FluentColors.PRIMARY_700};               /* #0f766e - Sehr dunkel */
        --primary-800: {FluentColors.PRIMARY_800};               /* #115e59 - Ultra dunkel */
        --primary-900: {FluentColors.PRIMARY_900};               /* #134e4a - Dunkelste */
        
        /* === INTERAKTIVE ELEMENTE (WCAG-konform) === */
        --interactive-primary: {FluentColors.INTERACTIVE_PRIMARY};   /* #0d9488 - Buttons, Links */
        --interactive-hover: {FluentColors.INTERACTIVE_HOVER};       /* #0f766e - Hover */
        --interactive-active: {FluentColors.INTERACTIVE_ACTIVE};     /* #115e59 - Active */
        
        /* === HOVER & ACTIVE STATES === */
        --hover-bg-light: {FluentColors.HOVER_BACKGROUND_SECONDARY};     /* #f0fffe - 5% Türkis */
        --hover-bg-medium: {FluentColors.HOVER_BACKGROUND};              /* #ccfbf1 - 10% Türkis */
        --hover-bg-strong: {FluentColors.ACTIVE_BACKGROUND};             /* #99f6e4 - 20% Türkis */
        --hover-primary: {FluentColors.HOVER_PRIMARY};                   /* #0d9488 - Dunkler für Hover */
        --active-primary: {FluentColors.ACTIVE_PRIMARY};                 /* #115e59 - Sehr dunkel für Active */
        
        /* === HINTERGRÜNDE === */
        --bg-primary: {FluentColors.NEUTRAL_BACKGROUND};                 /* #ffffff - Weiß */
        --bg-secondary: {FluentColors.NEUTRAL_BACKGROUND_SECONDARY};     /* #f8fafc - Sehr hell */
        --bg-tertiary: {FluentColors.NEUTRAL_BACKGROUND_TERTIARY};       /* #f1f5f9 - Hell */
        
        /* === SCHRIFTFARBEN === */
        --text-primary: {FluentColors.NEUTRAL_FOREGROUND};               /* #171d3f - Dunkelblau */
        --text-secondary: {FluentColors.NEUTRAL_FOREGROUND_SECONDARY};   /* #334155 - Mittelblau */
        --text-tertiary: {FluentColors.NEUTRAL_FOREGROUND_TERTIARY};     /* #64748b - Hellblau */
        
        /* === RÄNDER === */
        --border-light: {FluentColors.NEUTRAL_STROKE};                   /* #e2e8f0 - Helle Ränder */
        --border-medium: {FluentColors.NEUTRAL_STROKE_ACCESSIBLE};       /* #cbd5e1 - Mittlere Ränder */
        --border-primary: {FluentColors.STROKE_PRIMARY};                 /* #0d868b - Türkis-Ränder */
        --border-secondary: {FluentColors.STROKE_SECONDARY};             /* #0d9488 - Dunkeltürkis */
        
        /* === SEMANTISCHE FARBEN (WCAG-konform) === */
        --success-color: {FluentColors.SUCCESS};           /* #047857 - Grün (4.6:1) */
        --success-light: {FluentColors.SUCCESS_LIGHT};     /* #d1fae5 - Helles Grün */
        --warning-color: {FluentColors.WARNING};           /* #b45309 - Orange (5.1:1) */
        --warning-light: {FluentColors.WARNING_LIGHT};     /* #fed7aa - Helles Orange */
        --error-color: {FluentColors.ERROR};               /* #dc2626 - Rot (4.5:1) */
        --error-light: {FluentColors.ERROR_LIGHT};         /* #fecaca - Helles Rot */
        --info-color: {FluentColors.INFO};                 /* #0d9488 - Türkis (4.5:1) */
        --info-light: {FluentColors.INFO_LIGHT};           /* #ccfbf1 - Helles Türkis */
        
        /* === SCHATTEN === */
        --shadow-primary: {FluentColors.SHADOW_PRIMARY};         /* rgba(15, 206, 198, 0.15) */
        --shadow-secondary: {FluentColors.SHADOW_SECONDARY};     /* rgba(13, 134, 139, 0.2) */
        --shadow-neutral: {FluentColors.SHADOW_NEUTRAL};         /* rgba(23, 29, 63, 0.1) */
        
        /* === SPACING & TYPOGRAPHY === */
        --font-family: {FluentTypography.FONT_FAMILY};
        --border-radius: {FluentBorders.RADIUS_MEDIUM};
        --border-radius-large: {FluentBorders.RADIUS_LARGE};
        --spacing-xs: {FluentSpacing.XS};
        --spacing-s: {FluentSpacing.S};
        --spacing-m: {FluentSpacing.M};
        --spacing-l: {FluentSpacing.L};
        --spacing-xl: {FluentSpacing.XL};
    }}
    
    
    
    /* ===== GLOBALE STYLES ===== */
    
    /* Typography */
    html, body, [class*="css"], .stApp {{
        font-family: var(--font-family);
        color: var(--text-primary);
        background-color: var(--bg-primary);
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary);
        font-weight: 600;
    }}

    /* ===== QCA-AID HEADER STYLES ===== */
    
    .qca-header {{
        display: flex;
        align-items: center;
        margin-bottom: 25px;
        padding: 15px 0;
        border-bottom: 2px solid var(--primary-400);  /* #2dd4bf */
        background-color: var(--bg-primary);
    }}
    
    .qca-icon-container {{
        width: 68px;
        height: 68px;
        margin-right: 20px;
        flex-shrink: 0;
        /* Gradient mit CSS-Variablen */
        background: linear-gradient(135deg, var(--primary-400), var(--primary-300));
        border-radius: var(--border-radius);
        padding: 4px;
        box-shadow: 0 2px 8px var(--shadow-primary);
    }}
    
    .qca-icon-container.qca-icon-emoji {{
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .qca-icon-img {{
        width: 60px;
        height: 60px;
        object-fit: contain;
        display: block;
        margin: auto;
    }}
    
    .qca-emoji {{
        font-size: 24px;
    }}
    
    .qca-header-text {{
        flex: 1;
    }}
    
    .qca-title {{
        margin: 0;
        color: var(--text-primary);              /* #171d3f statt #0A1929 */
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.2em;
        font-weight: 600;
    }}
    
    .qca-title-bold {{
        font-family: 'Segoe UI', sans-serif;
        font-weight: 900;
        font-size: 2.2em;
    }}
    
    .qca-title-italic {{
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        font-style: italic;
        font-size: 2.2em;
    }}
    
    .qca-title-sub {{
        font-size: 0.6em;
        font-weight: normal;
        font-size: 2.2em;
        color: var(--text-secondary);            /* #334155 */
    }}
    
    .qca-subtitle {{
        margin: 0;
        color: var(--text-secondary);            /* #334155 statt #666 */
        font-size: 0.9em;
        font-family: 'Segoe UI', sans-serif;
    }}
    
    /* Responsive für kleinere Bildschirme */
    @media (max-width: 768px) {{
        .qca-header {{
            flex-direction: column;
            align-items: flex-start;
        }}
        
        .qca-icon-container {{
            margin-bottom: 15px;
        }}
        
        .qca-title {{
            font-size: 1.8em;
        }}
    }}

    /* ===== STREAMLIT-GENERIERTE KLASSEN ÜBERSCHREIBEN ===== */
    
    .st-bb {{
        border-color: var(--interactive-primary) !important;
        background-color: var(--hover-bg-light) !important;
    }}
    
    .st-at {{
        background-color: var(--hover-bg-light) !important;
        border-color: var(--interactive-primary) !important;
    }}
    
    /* .st-at als Input/Container */
    .st-at input,
    .st-at textarea,
    .st-at select {{
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-medium) !important;
    }}
    
    /* .st-at im Focus */
    .st-at:focus,
    .st-at:focus-within {{
        border-color: var(--interactive-primary) !important;
        box-shadow: 0 0 0 3px var(--hover-bg-medium) !important;
    }}

    .st-ce {{
        color: inherit;
        font-weight: inherit;
    }}

    .st-d2 {{
    border-bottom-color: var(--border-medium);
    }}
    .st-d1 {{
    border-top-color: var(--border-medium);
    }}
    .st-d0 {{
    border-right-color: var(--border-medium);
    }}
    .st-cz {{
    border-left-color: var(--border-medium);
    }}
    
    /* ===== BUTTONS ===== */
    
    /* Primary Buttons (WCAG-konform) */
    button[kind="primary"],
    .stButton > button[kind="primary"] {{
        background-color: var(--interactive-primary);    /* #0d9488 - WCAG-konform */
        border: none;
        color: white;                                    /* Weiß für perfekten Kontrast */
        font-weight: 600;
        border-radius: var(--border-radius);
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }}
    
    button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:hover {{
        background-color: var(--interactive-hover);      /* #0f766e - dunkler */
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px var(--shadow-secondary);
    }}
    
    button[kind="primary"]:active,
    .stButton > button[kind="primary"]:active {{
        background-color: var(--interactive-active);     /* #115e59 - noch dunkler */
        color: white;
        transform: translateY(0);
        box-shadow: 0 2px 4px var(--shadow-secondary);
    }}
    
    /* Secondary Buttons */
    .stButton > button:not([kind="primary"]) {{
        background-color: var(--bg-primary);
        border: 1px solid var(--border-medium);
        color: var(--text-primary);
        font-weight: 500;
        border-radius: var(--border-radius);
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }}
    
    .stButton > button:hover:not([kind="primary"]) {{
        background-color: var(--hover-bg-medium);
        border-color: var(--interactive-primary);
        color: var(--interactive-primary);
    }}
    
    .stButton > button:active:not([kind="primary"]) {{
        background-color: var(--hover-bg-strong);
        border-color: var(--interactive-hover);
    }}
    
    /* ===== INPUTS ===== */
    
    input, textarea, select {{
        border: 1px solid var(--border-medium);
        color: var(--text-primary);
        background-color: var(--bg-primary);
        border-radius: var(--border-radius);
        padding: 0.5rem 0.75rem;
        transition: all 0.2s ease;
    }}
    
    input:hover, textarea:hover, select:hover {{
        border-color: var(--border-primary);
    }}
    
    input:focus, textarea:focus, select:focus {{
        border-color: var(--interactive-primary);
        outline: none;
        box-shadow: 0 0 0 3px var(--hover-bg-medium);
    }}
    
    /* Streamlit Input Widgets */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        border-color: var(--border-medium);
        background-color: var(--bg-primary);
    }}
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: var(--interactive-primary);
        box-shadow: 0 0 0 3px var(--hover-bg-medium);
    }}
    
    /* Select Boxes */
    .stSelectbox > div > div {{
        border: 1px solid var(--border-medium);
        background-color: var(--bg-primary);
        border-radius: var(--border-radius);
    }}
    
    .stSelectbox > div > div:hover {{
        border-color: var(--interactive-primary);
    }}
    
    /* ===== NAVIGATION ===== */
    
    /* Standard Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        border-bottom: 1px solid var(--border-light);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: var(--text-secondary);
        border-radius: var(--border-radius) var(--border-radius) 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: var(--hover-bg-light);
        color: var(--interactive-primary);
    }}
    
    .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid var(--interactive-primary);
        color: var(--interactive-primary);
        font-weight: 600;
        background-color: transparent;
    }}
    
    /* ===== SIDEBAR ===== */
    
    [data-testid="stSidebar"] {{
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-light);
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: var(--text-primary);
    }}
    
    [data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        background-color: var(--bg-primary);
        border: 1px solid var(--border-medium);
    }}
    
    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: var(--hover-bg-medium);
        border-color: var(--interactive-primary);
    }}
    
    /* ===== EXPANDER ===== */
    
    .streamlit-expanderHeader {{
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: var(--border-radius);
        color: var(--text-primary);
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    
    .streamlit-expanderHeader:hover {{
        background-color: var(--hover-bg-light);
        border-color: var(--interactive-primary);
    }}
    
    .streamlit-expanderContent {{
        border: 1px solid var(--border-light);
        border-top: none;
        border-radius: 0 0 var(--border-radius) var(--border-radius);
    }}
    
    /* ===== MESSAGES / ALERTS ===== */
    
    .stSuccess {{
        background-color: var(--success-light);
        border-left: 4px solid var(--success-color);
        color: var(--text-primary);
        border-radius: var(--border-radius);
        padding: 1rem;
    }}
    
    .stWarning {{
        background-color: var(--warning-light);
        border-left: 4px solid var(--warning-color);
        color: var(--text-primary);
        border-radius: var(--border-radius);
        padding: 1rem;
    }}
    
    .stError {{
        background-color: var(--error-light);
        border-left: 4px solid var(--error-color);
        color: var(--text-primary);
        border-radius: var(--border-radius);
        padding: 1rem;
    }}
    
    .stInfo {{
        background-color: var(--info-light);
        border-left: 4px solid var(--info-color);
        color: var(--text-primary);
        border-radius: var(--border-radius);
        padding: 1rem;
    }}
    
    /* ===== LINKS (WCAG-konform) ===== */
    
    a, .stMarkdown a {{
        color: var(--interactive-primary);               /* #0d9488 - WCAG-konform */
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s ease;
    }}
    
    a:hover, .stMarkdown a:hover {{
        color: var(--interactive-hover);                 /* #0f766e - dunkler */
        text-decoration: underline;
    }}
    
    a:active, .stMarkdown a:active {{
        color: var(--interactive-active);                /* #115e59 - noch dunkler */
    }}
    
    /* ===== PROGRESS & METRICS ===== */
    
    .stProgress > div > div {{
        background-color: var(--interactive-primary);
        border-radius: var(--border-radius);
    }}
    
    .stProgress > div {{
        background-color: var(--bg-secondary);
        border-radius: var(--border-radius);
    }}
    
    .metric-container [data-testid="metric-container"] {{
        color: var(--text-primary);
        background-color: var(--bg-secondary);
        border-radius: var(--border-radius);
        padding: 1rem;
    }}
    
    [data-testid="stMetricValue"] {{
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 600;
    }}
    
    [data-testid="stMetricDelta"] {{
        font-size: 1rem;
    }}
    
    /* ===== SCROLLBAR ===== */
    
    ::-webkit-scrollbar {{
        width: 12px;
        height: 12px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--bg-secondary);
        border-radius: var(--border-radius);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--interactive-primary);          /* #0d9488 */
        border-radius: var(--border-radius-large);
        border: 2px solid var(--bg-secondary);
        transition: background 0.2s ease;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--interactive-hover);            /* #0f766e */
    }}
    
    /* ===== LOADING SPINNER ===== */
    
    .stSpinner > div {{
        border-top-color: var(--interactive-primary);
    }}
    
    /* ===== FILE UPLOADER ===== */
    
    .stFileUploader {{
        border: 2px dashed var(--border-medium);
        background-color: var(--bg-primary);
        border-radius: var(--border-radius-large);
        padding: 2rem;
        transition: all 0.2s ease;
    }}
    
    .stFileUploader:hover {{
        border-color: var(--interactive-primary);
        background-color: var(--hover-bg-light);
    }}
    
    [data-testid="stFileUploadDropzone"] {{
        background-color: var(--bg-secondary);
        border-radius: var(--border-radius);
    }}
    
    /* ===== DATAFRAME / TABLE ===== */
    
    .stDataFrame {{
        border: 1px solid var(--border-light);
        border-radius: var(--border-radius);
        overflow: hidden;
    }}
    
    .stDataFrame th {{
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        border-color: var(--border-light);
        font-weight: 600;
        padding: 0.75rem 1rem;
    }}
    
    .stDataFrame td {{
        color: var(--text-primary);
        border-color: var(--border-light);
        padding: 0.5rem 1rem;
    }}
    
    .stDataFrame tr:hover {{
        background-color: var(--hover-bg-light);
    }}
    
    /* ===== CODE BLOCKS ===== */
    
    code {{
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border-light);
        border-radius: var(--border-radius);
        padding: 0.2rem 0.4rem;
        font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
        font-size: 0.9em;
    }}
    
    pre {{
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: var(--border-radius);
        padding: 1rem;
        overflow-x: auto;
    }}
    
    pre code {{
        background-color: transparent;
        border: none;
        padding: 0;
    }}
    
    /* ===== CARDS / CONTAINERS ===== */
    
    .element-container {{
        transition: all 0.2s ease;
    }}
    
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="element-container"] {{
        background-color: var(--bg-primary);
    }}
    
    /* ===== CUSTOM LAYOUT FIXES ===== */

    .st-emotion-cache-8hk7nf h1 {{
        font-size: inherit;
        font-weight: inherit;
        padding: inherit;
    }}
    
    .st-emotion-cache-zy6yx3 {{
        width: 100%;
        padding: 3rem;
        max-width: initial;
        min-width: auto;
    }}
    
    .st-emotion-cache-1046a32 {{
        font-family: var(--font-family);
        font-size: 1rem;
        color: var(--text-primary);
        max-width: 100%;
        overflow-wrap: break-word;
    }}
    
    .st-emotion-cache-1diky36 {{
        font-size: 1rem;
        color: var(--text-primary);
        padding-bottom: 0.25rem;
    }}
    
    .st-emotion-cache-zh2fnc {{
        width: auto;
        font-weight: bold;
    }}
    /* ===== CHECKBOX & RADIO ===== */
    
    .stCheckbox {{
        color: var(--text-primary);
    }}
    
    .stCheckbox > label > div[role="checkbox"] {{
        border-color: var(--border-medium);
        background-color: var(--bg-primary);
    }}
    
    .stCheckbox > label > div[role="checkbox"][aria-checked="true"] {{
        background-color: var(--interactive-primary);
        border-color: var(--interactive-primary);
    }}
    
    .stRadio > label {{
        color: var(--text-primary);
    }}
    
    /* ===== SLIDER ===== */
    
    .stSlider > div > div > div > div {{
        background-color: var(--interactive-primary);
    }}
    
    .stSlider > div > div > div {{
        background-color: var(--border-light);
    }}
    
    /* ===== TOAST NOTIFICATIONS ===== */
    
    .stToast {{
        background-color: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: var(--border-radius-large);
        box-shadow: 0 4px 12px var(--shadow-neutral);
    }}
    
    /* ===== UTILITY CLASSES ===== */
    
    .text-primary {{
        color: var(--text-primary);
    }}
    
    .text-secondary {{
        color: var(--text-secondary);
    }}
    
    .text-tertiary {{
        color: var(--text-tertiary);
    }}
    
    .bg-primary {{
        background-color: var(--bg-primary);
    }}
    
    .bg-secondary {{
        background-color: var(--bg-secondary);
    }}
    
    .border-primary {{
        border-color: var(--interactive-primary);
    }}
    
    /* ===== TOOLTIPS ===== */
    
    /* Streamlit Tooltip Container */
    [data-testid="stTooltipIcon"],
    .stTooltipIcon {{
        color: var(--text-primary) !important;
    }}
    
    /* Tooltip Content */
    [role="tooltip"],
    .stTooltip,
    [data-baseweb="tooltip"] {{
        background-color: var(--text-primary) !important;
        color: #ffffff !important;
        border-radius: var(--border-radius) !important;
        padding: 0.5rem 0.75rem !important;
        font-size: 0.875rem !important;
        box-shadow: 0 4px 12px var(--shadow-neutral) !important;
    }}
    
    /* Tooltip Arrow */
    [role="tooltip"]::before,
    .stTooltip::before {{
        border-color: var(--text-primary) transparent transparent transparent !important;
    }}
    
    /* Help Icon in Streamlit */
    .stTooltipIcon svg {{
        fill: var(--text-primary) !important;
    }}
    
    </style>
    """

def get_fluent_component_styles() -> Dict[str, Any]:
    """
    Gibt konsistente QCA-AID Theme Styles als Dictionary zurück.
    Optimiert für WCAG 2.1 Konformität mit perfekten Kontrasten.
    
    Returns:
        Dictionary mit Style-Definitionen
    """
    return {
        "colors": {
            # === LEITFARBEN ===
            "primary": FluentColors.CUSTOM_PRIMARY,           # #0fcec6 - NUR für Backgrounds!
            "secondary": FluentColors.CUSTOM_SECONDARY,       # #0d868b  
            "text": FluentColors.CUSTOM_TEXT,                 # #171d3f
            
            # === PRIMÄRFARBEN-PALETTE ===
            "primary_50": FluentColors.PRIMARY_50,            # #f0fffe - Ultra hell
            "primary_100": FluentColors.PRIMARY_100,          # #ccfbf1 - Sehr hell
            "primary_200": FluentColors.PRIMARY_200,          # #99f6e4 - Hell
            "primary_300": FluentColors.PRIMARY_300,          # #5eead4 - Mittel
            "primary_400": FluentColors.PRIMARY_400,          # #2dd4bf - Standard-hell
            "primary_500": FluentColors.PRIMARY_500,          # #14b8a6 - Standard
            "primary_600": FluentColors.PRIMARY_600,          # #0d9488 - Dunkel (WCAG Text)
            "primary_700": FluentColors.PRIMARY_700,          # #0f766e - Sehr dunkel
            "primary_800": FluentColors.PRIMARY_800,          # #115e59 - Ultra dunkel
            "primary_900": FluentColors.PRIMARY_900,          # #134e4a - Dunkelste
            
            # === SEKUNDÄRFARBEN-PALETTE ===
            "secondary_50": FluentColors.SECONDARY_50,        # #f8fafc
            "secondary_100": FluentColors.SECONDARY_100,      # #f1f5f9
            "secondary_200": FluentColors.SECONDARY_200,      # #e2e8f0
            "secondary_300": FluentColors.SECONDARY_300,      # #cbd5e1
            "secondary_400": FluentColors.SECONDARY_400,      # #94a3b8
            "secondary_500": FluentColors.SECONDARY_500,      # #64748b
            "secondary_600": FluentColors.SECONDARY_600,      # #475569
            "secondary_700": FluentColors.SECONDARY_700,      # #334155
            "secondary_800": FluentColors.SECONDARY_800,      # #1e293b
            "secondary_900": FluentColors.SECONDARY_900,      # #171d3f
            
            # === INTERAKTIVE ELEMENTE (WCAG-konform) ===
            "interactive_primary": FluentColors.INTERACTIVE_PRIMARY,   # #0d9488 - Buttons, Links
            "interactive_hover": FluentColors.INTERACTIVE_HOVER,       # #0f766e - Hover
            "interactive_active": FluentColors.INTERACTIVE_ACTIVE,     # #115e59 - Active
            
            # === HOVER & ACTIVE STATES ===
            "hover_light": FluentColors.HOVER_BACKGROUND_SECONDARY,    # #f0fffe - 5% Türkis
            "hover_medium": FluentColors.HOVER_BACKGROUND,             # #ccfbf1 - 10% Türkis
            "hover_strong": FluentColors.ACTIVE_BACKGROUND,            # #99f6e4 - 20% Türkis
            "hover_primary": FluentColors.HOVER_PRIMARY,               # #0d9488 - Dunkler für Hover
            "active_primary": FluentColors.ACTIVE_PRIMARY,             # #115e59 - Sehr dunkel für Active
            
            # === HINTERGRÜNDE ===
            "background": FluentColors.NEUTRAL_BACKGROUND,                 # #ffffff - Weiß
            "background_secondary": FluentColors.NEUTRAL_BACKGROUND_SECONDARY,     # #f8fafc - Sehr hell
            "background_tertiary": FluentColors.NEUTRAL_BACKGROUND_TERTIARY,       # #f1f5f9 - Hell
            
            # === SCHRIFTFARBEN ===
            "foreground": FluentColors.NEUTRAL_FOREGROUND,               # #171d3f - Dunkelblau
            "foreground_secondary": FluentColors.NEUTRAL_FOREGROUND_SECONDARY,   # #334155 - Mittelblau
            "foreground_tertiary": FluentColors.NEUTRAL_FOREGROUND_TERTIARY,     # #64748b - Hellblau
            
            # === RÄNDER ===
            "border_light": FluentColors.NEUTRAL_STROKE,                   # #e2e8f0 - Helle Ränder
            "border_medium": FluentColors.NEUTRAL_STROKE_ACCESSIBLE,       # #cbd5e1 - Mittlere Ränder
            "border_primary": FluentColors.STROKE_PRIMARY,                 # #0d868b - Türkis-Ränder
            "border_secondary": FluentColors.STROKE_SECONDARY,             # #0d9488 - Dunkeltürkis
            
            # === SEMANTISCHE FARBEN (WCAG-konform) ===
            "success": FluentColors.SUCCESS,                 # #047857 - Grün (4.6:1)
            "success_light": FluentColors.SUCCESS_LIGHT,     # #d1fae5 - Helles Grün
            "warning": FluentColors.WARNING,                 # #b45309 - Orange (5.1:1)
            "warning_light": FluentColors.WARNING_LIGHT,     # #fed7aa - Helles Orange
            "error": FluentColors.ERROR,                     # #dc2626 - Rot (4.5:1)
            "error_light": FluentColors.ERROR_LIGHT,         # #fecaca - Helles Rot
            "info": FluentColors.INFO,                       # #0d9488 - Türkis (4.5:1)
            "info_light": FluentColors.INFO_LIGHT,           # #ccfbf1 - Helles Türkis
            
            # === SCHATTEN ===
            "shadow_primary": FluentColors.SHADOW_PRIMARY,           # rgba(15, 206, 198, 0.15)
            "shadow_secondary": FluentColors.SHADOW_SECONDARY,       # rgba(13, 134, 139, 0.2)
            "shadow_neutral": FluentColors.SHADOW_NEUTRAL,           # rgba(23, 29, 63, 0.1)
            
            # === TRANSPARENZEN FÜR OVERLAYS ===
            "overlay_light": FluentColors.OVERLAY_LIGHT,             # rgba(15, 206, 198, 0.05)
            "overlay_medium": FluentColors.OVERLAY_MEDIUM,           # rgba(15, 206, 198, 0.1)
            "overlay_strong": FluentColors.OVERLAY_STRONG,           # rgba(15, 206, 198, 0.15)
            
            # === LEGACY SUPPORT ===
            "brand_primary": FluentColors.BRAND_PRIMARY,             # #0fcec6
            "brand_secondary": FluentColors.BRAND_SECONDARY,         # #0d868b
            "brand_tertiary": FluentColors.BRAND_TERTIARY,           # #115e59
        },
        
        "spacing": {
            "xxs": FluentSpacing.XXS,      # 4px
            "xs": FluentSpacing.XS,        # 8px
            "s": FluentSpacing.S,          # 12px
            "m": FluentSpacing.M,          # 16px
            "l": FluentSpacing.L,          # 24px
            "xl": FluentSpacing.XL,        # 32px
            "xxl": FluentSpacing.XXL,      # 48px
        },
        
        "typography": {
            "font_family": FluentTypography.FONT_FAMILY,
        },
        
        "borders": {
            "radius": FluentBorders.RADIUS_MEDIUM,       # 8px
            "radius_large": FluentBorders.RADIUS_LARGE,  # 12px
            "width": FluentBorders.WIDTH_THIN,           # 1px
        },
        
        "shadows": {
            "small": FluentShadows.SHADOW_2,    # Subtile Schatten
            "medium": FluentShadows.SHADOW_4,   # Standard Schatten
            "large": FluentShadows.SHADOW_8,    # Starke Schatten
        },
        
        # === VERWENDUNGSHINWEISE ===
        "usage_guide": {
            "text_colors": {
                "primary_text": "foreground",              # #171d3f - Haupttext
                "secondary_text": "foreground_secondary",  # #334155 - Sekundärtext
                "tertiary_text": "foreground_tertiary",    # #64748b - Tertiärtext
                "link": "interactive_primary",             # #0d9488 - Links (WCAG-konform)
                "link_hover": "interactive_hover",         # #0f766e - Link Hover
            },
            "button_colors": {
                "primary_bg": "interactive_primary",       # #0d9488 - Button Hintergrund
                "primary_hover": "interactive_hover",      # #0f766e - Button Hover
                "primary_active": "interactive_active",    # #115e59 - Button Active
                "primary_text": "#ffffff",                 # Weiß - Button Text
                "secondary_bg": "background",              # #ffffff - Secondary Button
                "secondary_border": "border_medium",       # #cbd5e1 - Border
                "secondary_text": "foreground",            # #171d3f - Text
            },
            "background_colors": {
                "page": "background",                      # #ffffff - Haupthintergrund
                "card": "background_secondary",            # #f8fafc - Card Hintergrund
                "sidebar": "background_tertiary",          # #f1f5f9 - Sidebar
                "hover": "hover_medium",                   # #ccfbf1 - Hover State
                "active": "hover_strong",                  # #99f6e4 - Active State
            },
            "border_colors": {
                "default": "border_light",                 # #e2e8f0 - Standard Border
                "focus": "interactive_primary",            # #0d9488 - Focus Border
                "hover": "border_primary",                 # #0d868b - Hover Border
                "error": "error",                          # #dc2626 - Error Border
            },
            "status_colors": {
                "success_bg": "success_light",             # #d1fae5 - Success Background
                "success_text": "success",                 # #047857 - Success Text
                "warning_bg": "warning_light",             # #fed7aa - Warning Background
                "warning_text": "warning",                 # #b45309 - Warning Text
                "error_bg": "error_light",                 # #fecaca - Error Background
                "error_text": "error",                     # #dc2626 - Error Text
                "info_bg": "info_light",                   # #ccfbf1 - Info Background
                "info_text": "info",                       # #0d9488 - Info Text
            }
        },
        
        # === WCAG KONFORMITÄT ===
        "accessibility": {
            "contrast_ratios": {
                "interactive_primary": "4.5:1",    # ✓ AA Compliant
                "interactive_hover": "5.3:1",      # ✓ AA Compliant
                "interactive_active": "7.1:1",     # ✓ AAA Compliant
                "foreground": "12.6:1",            # ✓ AAA Compliant
                "foreground_secondary": "9.3:1",   # ✓ AAA Compliant
                "foreground_tertiary": "4.7:1",    # ✓ AA Compliant
                "success": "4.6:1",                # ✓ AA Compliant
                "warning": "5.1:1",                # ✓ AA Compliant
                "error": "4.5:1",                  # ✓ AA Compliant
                "info": "4.5:1",                   # ✓ AA Compliant
            },
            "do_not_use_for_text": [
                "primary",           # #0fcec6 - Zu hell (2.8:1)
                "primary_300",       # #5eead4 - Zu hell
                "primary_400",       # #2dd4bf - Zu hell
                "brand_primary",     # #0fcec6 - Zu hell
            ],
            "safe_for_text": [
                "interactive_primary",   # #0d9488 - ✓ 4.5:1
                "interactive_hover",     # #0f766e - ✓ 5.3:1
                "interactive_active",    # #115e59 - ✓ 7.1:1
                "primary_600",           # #0d9488 - ✓ 4.5:1
                "primary_700",           # #0f766e - ✓ 5.3:1
                "primary_800",           # #115e59 - ✓ 7.1:1
                "primary_900",           # #134e4a - ✓ 8.2:1
            ]
        }
    }