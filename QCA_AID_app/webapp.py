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


def initialize_session_state():
    """
    Initialisiert Streamlit Session State für Zustandsverwaltung.
    
    Performance Optimization: Optimized session state initialization to avoid
    redundant operations and minimize memory usage.
    
    Requirement 1.4: WHEN ein Benutzer zwischen Reitern wechselt 
                    THEN das System SHALL den aktuellen Zustand der Eingaben beibehalten
    Requirement 10.2: WHEN die Webapp startet 
                     THEN das System SHALL die zuletzt verwendete Konfiguration automatisch laden
    Requirement 7.2: WHEN die Webapp neu gestartet wird 
                    THEN das System SHALL das zuletzt verwendete Projekt-Root-Verzeichnis laden
    """
    # Initialize project manager first (needed by other managers)
    if 'project_manager' not in st.session_state:
        # Set project root to parent directory of QCA_AID_app
        # This ensures the root is \QCA-AID and not \QCA-AID\QCA_AID_app
        # IMPORTANT: resolve() must be called BEFORE .parent to get absolute path
        project_root = Path(__file__).resolve().parent.parent
        
        st.session_state.project_manager = ProjectManager(root_dir=project_root)
        # Try to load saved project settings
        st.session_state.project_manager.load_settings()
    
    # Get project root from project manager
    project_root = st.session_state.project_manager.get_root_directory()
    
    # Track project root changes and reinitialize managers if needed
    # This ensures all managers use the current project root
    if 'current_project_root' not in st.session_state:
        st.session_state.current_project_root = project_root
    
    # If project root changed, reinitialize managers with new root
    if st.session_state.current_project_root != project_root:
        st.session_state.current_project_root = project_root
        # Force reinitialization of managers
        if 'config_manager' in st.session_state:
            del st.session_state.config_manager
        if 'file_manager' in st.session_state:
            del st.session_state.file_manager
    
    # Initialize managers if not already present (singleton pattern)
    # Pass project root to ensure all managers use the same base directory
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager(project_dir=str(project_root))
    
    if 'file_manager' not in st.session_state:
        st.session_state.file_manager = FileManager(base_dir=str(project_root))
    
    # Initialize configuration state with lazy loading
    if 'config_data' not in st.session_state:
        # Try to load last used configuration
        config_manager = st.session_state.config_manager
        
        # Attempt to load from JSON first, then XLSX, then defaults
        success, config_data, errors = config_manager.load_config()
        
        if success and config_data:
            st.session_state.config_data = config_data
            st.session_state.config_loaded_from = "auto"
        else:
            # Load default configuration
            st.session_state.config_data = config_manager.get_default_config()
            st.session_state.config_loaded_from = "default"
    
    # Initialize UI state for each tab (use setdefault for cleaner code)
    st.session_state.setdefault('active_tab', "Konfiguration")
    
    # Config tab state
    st.session_state.setdefault('config_modified', False)
    
    # Codebook tab state - load default codebook if not present
    if 'codebook_data' not in st.session_state:
        from webapp_logic.codebook_manager import CodebookManager
        codebook_manager = CodebookManager(project_root)
        
        # Try to load existing codebook
        success, codebook_data, errors = codebook_manager.load_codebook()
        
        if success and codebook_data:
            st.session_state.codebook_data = codebook_data
            st.session_state.codebook_loaded_from = "file"
        else:
            # No codebook found - this is OK, user can create one
            st.session_state.codebook_data = None
            st.session_state.codebook_loaded_from = "none"
    
    st.session_state.setdefault('codebook_modified', False)
    
    # Analysis tab state
    st.session_state.setdefault('analysis_running', False)
    st.session_state.setdefault('analysis_logs', [])
    
    # Explorer tab state
    # Initialize explorer_config_manager if not present
    # Requirement 1.2: Initialize explorer_config_manager in session state
    if 'explorer_config_manager' not in st.session_state:
        from webapp_logic.explorer_config_manager import ExplorerConfigManager
        st.session_state.explorer_config_manager = ExplorerConfigManager(str(project_root))
    
    # Initialize explorer_config_data if not present
    # Requirement 2.3, 7.1, 7.3, 7.4: Initialize explorer_config_data in session state
    if 'explorer_config_data' not in st.session_state:
        from webapp_models.explorer_config_data import ExplorerConfigData
        
        # Try to load existing config
        manager = st.session_state.explorer_config_manager
        success, config_data, errors = manager.load_config()
        
        if success and config_data:
            # Ensure script_dir is set to project root
            if not config_data.base_config.get('script_dir'):
                config_data.base_config['script_dir'] = str(project_root)
            st.session_state.explorer_config_data = config_data
            st.session_state.explorer_config_loaded_from = "file"
        else:
            # Create default config with project root
            st.session_state.explorer_config_data = ExplorerConfigData.create_default()
            st.session_state.explorer_config_data.base_config['script_dir'] = str(project_root)
            st.session_state.explorer_config_loaded_from = "default"
            # Store errors for potential display
            if errors:
                st.session_state.explorer_config_load_errors = errors
    
    # Legacy explorer config (for backward compatibility with old UI code)
    st.session_state.setdefault('explorer_config', None)
    st.session_state.setdefault('selected_output_file', None)
    
    # Explorer view state
    st.session_state.setdefault('explorer_active', False)
    st.session_state.setdefault('explorer_file_path', None)
    st.session_state.setdefault('show_add_analysis_dialog', False)
    st.session_state.setdefault('explorer_config_modified', False)
    
    # Pagination state (for performance optimization)
    st.session_state.setdefault('input_files_page', 1)
    st.session_state.setdefault('output_files_page', 1)
    st.session_state.setdefault('results_files_page', 1)
    st.session_state.setdefault('files_per_page', 10)


def render_navigation():
    """
    Rendert Tab-Navigation für die Webapp.
    
    Requirement 1.3: WHEN die Webapp läuft 
                    THEN das System SHALL eine Navigation mit Reitern für 
                    Konfiguration, Codebook, Analyse und Explorer anzeigen
    """
    # Create tabs
    tabs = st.tabs([
        "⚙️ Konfiguration",
        "📚 Codebook", 
        "🔬 Analyse",
        "📊 Explorer"
    ])
    
    return tabs


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
    """
    Haupteinstiegspunkt der Webapp.
    
    Requirement 1.1: WHEN die Webapp gestartet wird 
                    THEN das System SHALL eine lokale Streamlit-Anwendung über localhost öffnen
    Requirement 1.2: WHEN die Webapp läuft 
                    THEN das System SHALL ausschließlich auf localhost lauschen
    Requirement 10.1: WHEN ein Benutzer "streamlit run webapp.py" ausführt 
                     THEN das System SHALL die Webapp im Standard-Browser über localhost öffnen
    Requirement 10.3: WHEN keine vorherige Konfiguration existiert 
                     THEN das System SHALL Standard-Werte aus config.py laden
    """
    # Configure Streamlit page
    st.set_page_config(
        page_title="QCA-AID Webapp",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply Fluent UI Design System
    st.markdown(get_fluent_css(), unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
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
        
        st.markdown("---")
        st.caption("Läuft auf localhost:8501")
        
        # Version info with update check
        from webapp_logic.version_checker import VersionChecker
        VersionChecker.render_version_info()
        
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-QCA--AID-blue?logo=github)](https://github.com/JustusHenke/QCA-AID)", unsafe_allow_html=True)
    
    # Main content area with styled title
    st.markdown("""
    <h1 style='margin-bottom: 0;'>
        <span style='font-family: Arial, sans-serif; font-weight: 900;'>QCA</span><span style='font-family: Arial, sans-serif; font-weight: 600; font-style: italic;'>-AID</span> 
        <span style='font-size: 0.6em; font-weight: normal;'>Webapp</span>
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("Verwalten Sie Ihre qualitative Inhaltsanalyse mit KI-Unterstützung")
    
    # Render navigation tabs
    tabs = render_navigation()
    
    # Render each tab
    with tabs[0]:  # Konfiguration
        render_config_tab()
    
    with tabs[1]:  # Codebook
        render_codebook_tab()
    
    with tabs[2]:  # Analyse
        render_analysis_tab()
    
    with tabs[3]:  # Explorer
        render_explorer_tab()
    
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
