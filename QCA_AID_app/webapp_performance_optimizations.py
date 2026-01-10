#!/usr/bin/env python3
"""
Performance Optimizations für QCA-AID Webapp
============================================
Sammlung von Optimierungen zur Reduzierung der Startup-Zeit.
"""

import streamlit as st
from pathlib import Path
from typing import Optional
import functools
import time


# 1. LAZY LOADING DECORATOR
def lazy_import(func):
    """Decorator für lazy loading von Modulen"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# 2. CACHED ICON LOADING
@st.cache_data
def load_app_icon() -> Optional[str]:
    """
    Cached icon loading mit optimierter Pfad-Suche.
    Reduziert Dateisystem-Zugriffe beim Startup.
    """
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


# 3. MINIMAL SESSION STATE INIT
def initialize_minimal_session_state():
    """
    Minimale Session State Initialisierung.
    Lädt nur das Nötigste beim ersten Start.
    """
    # Nur kritische Manager initialisieren
    if 'project_manager' not in st.session_state:
        from webapp_logic.project_manager import ProjectManager
        project_root = Path(__file__).resolve().parent.parent
        st.session_state.project_manager = ProjectManager(root_dir=project_root)
    
    # UI State minimal initialisieren
    st.session_state.setdefault('active_tab', "Konfiguration")
    st.session_state.setdefault('startup_complete', False)


# 4. LAZY TAB LOADING
def lazy_load_tab_content(tab_name: str):
    """
    Lädt Tab-Inhalte erst wenn sie benötigt werden.
    """
    if tab_name == "Konfiguration" and 'config_manager' not in st.session_state:
        from webapp_logic.config_manager import ConfigManager
        project_root = st.session_state.project_manager.get_root_directory()
        st.session_state.config_manager = ConfigManager(project_dir=str(project_root))
        
        # Config laden
        success, config_data, errors = st.session_state.config_manager.load_config()
        if success and config_data:
            st.session_state.config_data = config_data
        else:
            st.session_state.config_data = st.session_state.config_manager.get_default_config()
    
    elif tab_name == "Codebook" and 'codebook_data' not in st.session_state:
        from webapp_logic.codebook_manager import CodebookManager
        project_root = st.session_state.project_manager.get_root_directory()
        codebook_manager = CodebookManager(project_root)
        
        success, codebook_data, errors = codebook_manager.load_codebook()
        st.session_state.codebook_data = codebook_data if success else None
    
    # Weitere Tabs analog...


# 5. PERFORMANCE MONITORING
class PerformanceMonitor:
    """Überwacht Startup-Performance"""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
    
    def checkpoint(self, name: str):
        """Setzt einen Performance-Checkpoint"""
        self.checkpoints[name] = time.time() - self.start_time
    
    def report(self):
        """Zeigt Performance-Report"""
        if st.sidebar.checkbox("🔍 Performance Debug", value=False):
            st.sidebar.markdown("### ⏱️ Startup Times")
            for name, duration in self.checkpoints.items():
                st.sidebar.text(f"{name}: {duration:.2f}s")


# 6. OPTIMIZED MAIN FUNCTION
def optimized_main():
    """
    Optimierte Hauptfunktion mit Performance-Monitoring.
    """
    monitor = PerformanceMonitor()
    
    # Page config (schnell)
    st.set_page_config(
        page_title="QCA-AID Webapp",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    monitor.checkpoint("Page Config")
    
    # Minimal session state
    initialize_minimal_session_state()
    monitor.checkpoint("Session State")
    
    # Cached icon loading
    icon_base64 = load_app_icon()
    monitor.checkpoint("Icon Loading")
    
    # Header mit cached icon
    if icon_base64:
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 25px;">
            <img src="data:image/png;base64,{icon_base64}" 
                 style="width: 48px; height: 48px; margin-right: 15px;" 
                 alt="QCA-AID Icon">
            <h1 style="margin: 0;">QCA-AID Webapp</h1>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.title("🔬 QCA-AID Webapp")
    
    monitor.checkpoint("Header Render")
    
    # CSS nur einmal laden
    if 'css_loaded' not in st.session_state:
        from webapp_components.fluent_styles import get_fluent_css
        st.markdown(get_fluent_css(), unsafe_allow_html=True)
        st.session_state.css_loaded = True
    
    monitor.checkpoint("CSS Loading")
    
    # Tabs mit lazy loading
    tabs = st.tabs([
        "⚙️ Konfiguration",
        "📚 Codebook", 
        "🔬 Analyse",
        "📊 Explorer"
    ])
    
    # Nur aktiven Tab laden
    tab_names = ["Konfiguration", "Codebook", "Analyse", "Explorer"]
    
    for i, (tab, tab_name) in enumerate(zip(tabs, tab_names)):
        with tab:
            if st.session_state.get('active_tab') == tab_name or i == 0:  # Ersten Tab immer laden
                lazy_load_tab_content(tab_name)
                
                if tab_name == "Konfiguration":
                    from webapp_components.config_ui import render_config_tab
                    render_config_tab()
                elif tab_name == "Codebook":
                    from webapp_components.codebook_ui import render_codebook_tab
                    render_codebook_tab()
                elif tab_name == "Analyse":
                    from webapp_components.analysis_ui import render_analysis_tab
                    render_analysis_tab()
                elif tab_name == "Explorer":
                    from webapp_components.explorer_ui import render_explorer_tab
                    render_explorer_tab()
            else:
                st.info(f"Tab '{tab_name}' wird geladen wenn ausgewählt...")
    
    monitor.checkpoint("Tab Rendering")
    
    # Performance report
    monitor.report()


if __name__ == "__main__":
    optimized_main()