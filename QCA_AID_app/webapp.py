#!/usr/bin/env python3
"""
QCA-AID Webapp
==============
Streamlit-basierte Webanwendung zur Verwaltung von QCA-AID und QCA-AID Explorer.

Läuft ausschließlich auf localhost und bietet eine intuitive Benutzeroberfläche
für Konfiguration, Codebook-Verwaltung, Analyse und Ergebnisexploration.

Design System
-------------
Implementiert Microsoft Fluent UI Design-Prinzipien:
- Farben: Fluent Blue (#0078D4) als Primärfarbe, Neutral Palette
- Typografie: Segoe UI Font-Familie mit klarer Hierarchie
- Spacing: 4px Grid-System (4, 8, 12, 16, 20, 24, 32, 40, 48px)
- Shadows: Subtile Schatten für visuelle Tiefe
- Borders: 4px Border Radius für Konsistenz

Siehe: webapp_components/FLUENT_UI_GUIDE.md für Details

Requirements: 1.1, 1.2, 1.3, 1.4, 10.1, 10.2, 10.3
"""

import sys
from pathlib import Path

# Fix für Unicode-Encoding auf Windows-Konsolen
if sys.platform == 'win32':
    try:
        # Setze stdout und stderr auf UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        # Fallback für ältere Python-Versionen
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add parent directory to path to access QCA_AID_assets
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import streamlit as st
from typing import Optional

# Performance monitoring
from webapp_logic.performance_monitor import get_performance_monitor, checkpoint, render_performance_debug

# Import webapp components
from webapp_components.config_ui import render_config_tab
from webapp_components.codebook_ui import render_codebook_tab
from webapp_components.analysis_ui import render_analysis_tab
from webapp_components.explorer_ui import render_explorer_tab
from webapp_components.fluent_styles import get_fluent_css

# Import webapp logic
from webapp_logic.config_manager import ConfigManager
from webapp_logic.file_manager import FileManager
from webapp_logic.project_manager import ProjectManager
from webapp_models.config_data import ConfigData


def lazy_load_tab_data(tab_name: str):
    """
    PERFORMANCE OPTIMIZATION: Lädt Tab-spezifische Daten erst bei Bedarf.
    
    Args:
        tab_name: Name des Tabs ("Konfiguration", "Codebook", "Analyse", "Explorer")
    """
    project_root = st.session_state.project_manager.get_root_directory()
    
    if tab_name == "Konfiguration":
        # Config Manager und Daten laden
        if 'config_manager' not in st.session_state:
            st.session_state.config_manager = ConfigManager(project_dir=str(project_root))
        
        if 'file_manager' not in st.session_state:
            st.session_state.file_manager = FileManager(base_dir=str(project_root))
        
        if 'config_data' not in st.session_state:
            config_manager = st.session_state.config_manager
            success, config_data, errors = config_manager.load_config()
            
            if success and config_data:
                st.session_state.config_data = config_data
                st.session_state.config_loaded_from = "auto"
            else:
                st.session_state.config_data = config_manager.get_default_config()
                st.session_state.config_loaded_from = "default"
        
        # Config tab state
        st.session_state.setdefault('config_modified', False)
    
    elif tab_name == "Codebook":
        # Codebook Daten laden
        if 'codebook_data' not in st.session_state:
            from webapp_logic.codebook_manager import CodebookManager
            codebook_manager = CodebookManager(project_root)
            
            success, codebook_data, errors = codebook_manager.load_codebook()
            
            if success and codebook_data:
                st.session_state.codebook_data = codebook_data
                st.session_state.codebook_loaded_from = "file"
            else:
                st.session_state.codebook_data = None
                st.session_state.codebook_loaded_from = "none"
        
        st.session_state.setdefault('codebook_modified', False)
    
    elif tab_name == "Analyse":
        # Analysis state
        st.session_state.setdefault('analysis_running', False)
        st.session_state.setdefault('analysis_logs', [])
    
    elif tab_name == "Explorer":
        # Explorer Manager und Daten laden
        if 'explorer_config_manager' not in st.session_state:
            from webapp_logic.explorer_config_manager import ExplorerConfigManager
            st.session_state.explorer_config_manager = ExplorerConfigManager(str(project_root))
        
        if 'explorer_config_data' not in st.session_state:
            from webapp_models.explorer_config_data import ExplorerConfigData
            
            manager = st.session_state.explorer_config_manager
            success, config_data, errors = manager.load_config()
            
            if success and config_data:
                if not config_data.base_config.get('script_dir'):
                    config_data.base_config['script_dir'] = str(project_root)
                st.session_state.explorer_config_data = config_data
                st.session_state.explorer_config_loaded_from = "file"
            else:
                st.session_state.explorer_config_data = ExplorerConfigData.create_default()
                st.session_state.explorer_config_data.base_config['script_dir'] = str(project_root)
                st.session_state.explorer_config_loaded_from = "default"
                if errors:
                    st.session_state.explorer_config_load_errors = errors
        
        # Explorer state
        st.session_state.setdefault('explorer_config', None)
        st.session_state.setdefault('selected_output_file', None)
        st.session_state.setdefault('explorer_active', False)
        st.session_state.setdefault('explorer_file_path', None)
        st.session_state.setdefault('show_add_analysis_dialog', False)
        st.session_state.setdefault('explorer_config_modified', False)


def initialize_session_state():
    """
    OPTIMIERTE Session State Initialisierung für bessere Startup-Performance.
    
    Lädt nur kritische Komponenten beim ersten Start, andere werden lazy geladen.
    
    Requirement 1.4: WHEN ein Benutzer zwischen Reitern wechselt 
                    THEN das System SHALL den aktuellen Zustand der Eingaben beibehalten
    Requirement 10.2: WHEN die Webapp startet 
                     THEN das System SHALL die zuletzt verwendete Konfiguration automatisch laden
    Requirement 7.2: WHEN die Webapp neu gestartet wird 
                    THEN das System SHALL das zuletzt verwendete Projekt-Root-Verzeichnis laden
    """
    # PERFORMANCE: Verhindere mehrfache Initialisierung
    if st.session_state.get('initialization_complete', False):
        return
    
    # PERFORMANCE: Nur kritische Manager beim Startup initialisieren
    if 'project_manager' not in st.session_state:
        project_root = Path(__file__).resolve().parent.parent
        st.session_state.project_manager = ProjectManager(root_dir=project_root)
        st.session_state.project_manager.load_settings()
    
    # Track project root für Manager-Reinitialisierung
    project_root = st.session_state.project_manager.get_root_directory()
    if 'current_project_root' not in st.session_state:
        st.session_state.current_project_root = project_root
    
    # CRITICAL: Config-Daten werden beim Start geladen (für Sidebar)
    if 'config_data' not in st.session_state:
        lazy_load_tab_data("Konfiguration")
    
    # PERFORMANCE: Basis UI-State minimal initialisieren
    st.session_state.setdefault('active_tab', "config")  # Neue primäre Variable
    st.session_state.setdefault('active_tab_key', "config")  # Kompatibilität
    st.session_state.setdefault('startup_complete', False)
    
    # CRITICAL: Pagination states (werden in mehreren Tabs benötigt)
    st.session_state.setdefault('input_files_page', 1)
    st.session_state.setdefault('output_files_page', 1)
    st.session_state.setdefault('results_files_page', 1)
    st.session_state.setdefault('files_per_page', 10)
    
    # PERFORMANCE: Markiere Initialisierung als abgeschlossen
    st.session_state.initialization_complete = True
    
    # LAZY LOADING: Andere Manager und Daten werden erst bei Bedarf geladen
    # Siehe lazy_load_tab_data() Funktion


def switch_to_tab(tab_key: str):
    """
    Hilfsfunktion zum Wechseln zu einem bestimmten Tab.
    
    Args:
        tab_key: Tab-Schlüssel ("config", "codebook", "analysis", "explorer")
    """
    st.session_state.active_tab_key = tab_key
    # Setze auch den Tab-Namen für Kompatibilität
    tab_names = ["Konfiguration", "Codebook", "Analyse", "Explorer"]
    tab_keys = ["config", "codebook", "analysis", "explorer"]
    if tab_key in tab_keys:
        st.session_state.active_tab = tab_names[tab_keys.index(tab_key)]
    st.rerun()


def rerun_with_tab_state():
    """
    Führt st.rerun() aus und behält den aktuellen Tab-State bei.
    Tab-State wird automatisch über Session State beibehalten.
    """
    st.rerun()


def render_navigation():
    """Tab-Navigation mit WCAG-konformen Farben."""
    
    TABS = [
        {"id": "config",   "label": "⚙️ Konfiguration"},
        {"id": "codebook", "label": "📚 Codebook"},
        {"id": "analysis", "label": "🔬 Analyse"},
        {"id": "explorer", "label": "📊 Explorer"},
    ]

    st.markdown("""
    <style>
    /* ===== TAB-NAVIGATION (WCAG-KONFORM) ===== */
    
    /* Gesamte Tab-Leiste */
    div[role="radiogroup"] {
        display: flex;
        gap: 0.25rem;
        border-bottom: 2px solid var(--border-light);
        padding-bottom: 0px;
        background-color: var(--bg-primary) !important;
    }
    div[role="radiogroup"] p { font-weight: bold; }

    /* Einzelner Tab */
    label[data-baseweb="radio"] {
        display: flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-bottom: 3px solid transparent;
        cursor: pointer;
        font-weight: 500;
        color: var(--text-secondary);
        background: transparent;
        border-radius: var(--border-radius) var(--border-radius) 0 0;
        transition: all 0.2s ease;
    }

    /* Radio-Kreis vollständig entfernen */
    label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }

    /* Hover State */
    label[data-baseweb="radio"]:hover {
        background-color: var(--hover-bg-medium);
        color: var(--interactive-primary);
    }

    /* AKTIVER TAB (WCAG-KONFORM) */
    label[data-baseweb="radio"]:has(input:checked) {
        background-color: var(--interactive-primary);   /* #0d9488 - Dunkel */
        border-bottom: 3px solid var(--interactive-active); /* #115e59 */
        font-weight: 600;
        margin-bottom: -2px;
        color: white !important;                                   /* Weiß für perfekten Kontrast */
        box-shadow: 0 2px 8px var(--shadow-secondary);
    }
    
    /* Aktiver Tab Hover */
    label[data-baseweb="radio"]:has(input:checked):hover {
        background-color: var(--interactive-hover);     /* #0f766e - Dunkler */
        color: white;
        box-shadow: 0 4px 12px var(--shadow-secondary);
    }
    
    /* Verhindere Dark Mode Überschreibung */
    html[data-theme="dark"] label[data-baseweb="radio"]:has(input:checked) {
        background-color: var(--interactive-primary) !important;
        color: white !important;
    }

    </style>
    """, unsafe_allow_html=True)

    labels = [t["label"] for t in TABS]
    ids    = [t["id"]    for t in TABS]

    if "active_tab" not in st.session_state or st.session_state.active_tab not in ids:
        st.session_state.active_tab = ids[0]
        st.session_state.active_tab_key = ids[0]

    active_index = ids.index(st.session_state.active_tab)

    selected_label = st.radio(
        "Navigation Tabs",
        labels,
        index=active_index,
        horizontal=True,
        label_visibility="collapsed",
    )

    new_active_id = ids[labels.index(selected_label)]

    if new_active_id != st.session_state.active_tab:
        st.session_state.active_tab = new_active_id
        st.session_state.active_tab_key = new_active_id
        st.rerun()

    return labels, ids


def render_placeholder_tab(tab_name: str):
    """
    Rendert Platzhalter für noch nicht implementierte Tabs.
    
    Args:
        tab_name: Name des Tabs
    """
    st.info(f"🚧 Der {tab_name}-Tab wird in einer späteren Aufgabe implementiert.")
    st.markdown(f"""
    Dieser Tab wird folgende Funktionen bieten:
    
    **{tab_name}**
    - Wird in den kommenden Tasks implementiert
    - Siehe tasks.md für Details
    """)


def render_config_info():
    """
    Zeigt Informationen über die geladene Konfiguration und Projekt-Root an.
    
    Requirement 1.2: Projekt-Root-Verzeichnis prominent anzeigen
    """
    # Show project root
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Projekt")
    project_manager = st.session_state.project_manager
    project_root = project_manager.get_root_directory()
    
    # Truncate path if too long for sidebar
    root_str = str(project_root)
    if len(root_str) > 30:
        # Show last part of path
        root_display = "..." + root_str[-27:]
    else:
        root_display = root_str
    
    st.sidebar.text(f"Root: {root_display}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Workflow-Status")
    
    # Determine workflow status
    config = st.session_state.config_data
    config_manager = st.session_state.config_manager
    
    # Check if config exists
    config_exists = config_manager.json_path.exists() or config_manager.xlsx_path.exists()
    
    # Check if codebook exists and has categories
    from webapp_logic.codebook_manager import CodebookManager
    codebook_manager = CodebookManager(st.session_state.project_manager.get_root_directory())
    success, codebook_data, errors = codebook_manager.load_codebook()
    has_categories = success and codebook_data and len(codebook_data.deduktive_kategorien) > 0
    
    # Check if input files exist
    input_dir = Path(st.session_state.project_manager.get_root_directory()) / config.data_dir
    has_input_files = input_dir.exists() and any(input_dir.glob('*.txt')) or any(input_dir.glob('*.pdf')) or any(input_dir.glob('*.docx'))
    
    # Check if analysis has been run
    output_dir = Path(st.session_state.project_manager.get_root_directory()) / config.output_dir
    has_results = output_dir.exists() and any(output_dir.glob('QCA-AID_Analysis_*.xlsx'))
    
    # Display workflow steps with status
    steps = [
        ("1️⃣ Konfiguration", config_exists, "Konfiguration laden/erstellen"),
        ("2️⃣ Codebook", has_categories, "Kategorien definieren"),
        ("3️⃣ Eingabedateien", has_input_files, "Dokumente bereitstellen"),
        ("4️⃣ Analyse", has_results, "Analyse durchführen")
    ]
    
    for step_name, is_complete, description in steps:
        if is_complete:
            st.sidebar.success(f"✅ {step_name}")
        else:
            st.sidebar.warning(f"⏳ {step_name}")
            st.sidebar.caption(f"   → {description}")
    
    st.sidebar.markdown("---")
    
    # Show config status
    if st.session_state.config_loaded_from == "auto":
        st.sidebar.info("📄 Konfiguration automatisch geladen")
    elif st.session_state.config_loaded_from == "default":
        st.sidebar.info("📄 Standard-Konfiguration aktiv")
    
    if st.session_state.config_modified:
        st.sidebar.error("💾 Ungespeicherte Änderungen!")
    
    # Show basic config info
    with st.sidebar.expander("ℹ️ Aktuelle Einstellungen", expanded=False):
        st.text(f"Modell: {config.model_provider}")
        st.text(f"Name: {config.model_name}")
        st.text(f"Modus: {config.analysis_mode}")
        st.text(f"Coder: {len(config.coder_settings)}")


def main():
    """Haupteinstiegspunkt mit erzwungenem Light Mode."""
    
    # Performance monitoring
    monitor = get_performance_monitor()
    monitor.enable_monitoring()
    
    # ✅ WICHTIG: Page Config VOR allen anderen st.* Aufrufen!
    st.set_page_config(
        page_title="QCA-AID Webapp",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
        # ✅ Erzwinge Light Mode Theme
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "QCA-AID Webapp - Läuft ausschließlich auf localhost"
        }
    )
    checkpoint("Page Config")
    
    # ✅ KRITISCH: CSS als allererstes laden (überschreibt Streamlit Theme)
    st.markdown(get_fluent_css(), unsafe_allow_html=True)
    checkpoint("CSS Loading")
    
    # ✅ ZUSÄTZLICH: Body-Styles erzwingen
    st.markdown("""
    <style>
    /* Erzwinge Light Mode für gesamte App */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #ffffff !important;
        color: #171d3f !important;
    }
    
    /* Sidebar immer hell */
    [data-testid="stSidebar"] {
        background-color: var(--hover-bg-light) !important;
        color: var(--text-color)!important;
    }
    
    /* Verhindere Dark Mode Klassen */
    html[data-theme="dark"] .stApp,
    html[data-theme="dark"] [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        color: #171d3f !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # PERFORMANCE OPTIMIZATION: Cached icon loading
    @st.cache_data
    def load_app_icon():
        """Cached icon loading mit optimierter Pfad-Suche."""
        # Nur die wahrscheinlichsten Pfade prüfen
        icon_paths = [
            Path(__file__).resolve().parent.parent / "qca_aid_icon.png",
            Path.cwd() / "qca_aid_icon.png"
        ]
        
        for icon_path in icon_paths:
            if icon_path.exists():
                try:
                    import base64
                    with open(icon_path, 'rb') as f:
                        png_data = f.read()
                    return base64.b64encode(png_data).decode()
                except Exception:
                    continue
        return None
    
    png_base64 = load_app_icon()
    checkpoint("Icon Loading")
    
    if png_base64:
        # Display PNG icon in header with enhanced styling
        st.markdown(f"""
        <div class="qca-header">
            <div class="qca-icon-container">
                <img src="data:image/png;base64,{png_base64}" 
                    class="qca-icon-img"
                    alt="QCA-AID Icon">
            </div>
            <div class="qca-header-text">
                <h1 class="qca-title">
                    <span class="qca-title-bold">QCA</span><span class="qca-title-italic">-AID</span>&nbsp;&nbsp;
                    <span class="qca-title-sub">Webapp</span>
                </h1>
                <p class="qca-subtitle">Qualitative Content Analysis with AI-Discovery</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="qca-header">
            <div class="qca-icon-container qca-icon-emoji">
                <span class="qca-emoji">🔬</span>
            </div>
            <div class="qca-header-text">
                <h1 class="qca-title">
                    <span class="qca-title-bold">QCA</span><span class="qca-title-italic">-AID</span>
                    <span class="qca-title-sub">Webapp</span>
                </h1>
                <p class="qca-subtitle">Qualitative Content Analysis with AI-Discovery</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Apply Fluent UI Design System (bei jedem Rerun für konsistente Farben)
    st.markdown(get_fluent_css(), unsafe_allow_html=True)
    checkpoint("CSS Loading")
    
    # Initialize session state
    initialize_session_state()
    checkpoint("Session State Init")
    
    # Sidebar
    with st.sidebar:
        # Custom styled logo
        # st.markdown("""
        # <div style='text-align: center; margin-bottom: 10px;'>
        #     <span style='font-family: Arial, sans-serif; font-size: 2.5em; font-weight: 900;'>QCA</span><span style='font-family: Arial, sans-serif; font-size: 2.5em; font-weight: 600; font-style: italic;'>-AID</span>
        # </div>
        # """, unsafe_allow_html=True)
        # st.markdown("<p style='text-align: center; font-size: 0.9em; margin-top: -10px;'><strong>Qualitative Content Analysis with AI</strong></p>", unsafe_allow_html=True)
        # st.markdown("---")
        
        st.markdown("""
        ### Willkommen!
        
        QCA-AID Webapp ermöglicht die Verwaltung von:
        - ⚙️ Konfigurationseinstellungen
        - 📚 Codebook und Kategorien
        - 🔬 Analysen und Fortschritt
        - 📊 Ergebnisexploration
        """)
        
        # Show config info
        render_config_info()
        
        # Performance report
        # render_performance_debug()
        
        st.markdown("---")
        st.caption("Läuft auf localhost:8501")
        
        # Version info with update check
        from webapp_logic.version_checker import VersionChecker
        VersionChecker.render_version_info()
        
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-QCA--AID-blue?logo=github)](https://github.com/JustusHenke/QCA-AID)", unsafe_allow_html=True)
    
    # Main content area with styled title
    # st.markdown("""
    # <h1 style='margin-bottom: 0;'>
    #     <span style='font-family: Arial, sans-serif; font-weight: 900;'>QCA</span><span style='font-family: Arial, sans-serif; font-weight: 600; font-style: italic;'>-AID</span> 
    #     <span style='font-size: 0.6em; font-weight: normal;'>Webapp</span>
    # </h1>
    # """, unsafe_allow_html=True)
    st.markdown("Verwalten Sie Ihre qualitative Inhaltsanalyse mit KI-Unterstützung")
    
    # Render navigation tabs
    tab_names, tab_keys = render_navigation()
    checkpoint("Tab Navigation")
    
    # Render active tab content based on session state
    active_tab = st.session_state.active_tab
    
    if active_tab == "config":
        lazy_load_tab_data("Konfiguration")
        render_config_tab()
    
    elif active_tab == "codebook":
        # Lazy load beim ersten Zugriff auf Tab
        if 'codebook_data' not in st.session_state:
            lazy_load_tab_data("Codebook")
        render_codebook_tab()
    
    elif active_tab == "analysis":
        # Analysis Tab benötigt keine zusätzlichen Daten (nutzt config_data)
        render_analysis_tab()
    
    elif active_tab == "explorer":
        # Lazy load beim ersten Zugriff auf Tab
        if 'explorer_config_data' not in st.session_state:
            lazy_load_tab_data("Explorer")
        render_explorer_tab()
    
    else:
        # Fallback für unbekannte Tabs
        st.error(f"Unbekannter Tab: {active_tab}")
        st.info("Verfügbare Tabs: config, codebook, analysis, explorer")
    
    checkpoint("Tab Rendering Complete")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        QCA-AID Webapp | Läuft ausschließlich auf localhost | 
        Alle Daten werden lokal gespeichert
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
