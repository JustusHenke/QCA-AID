"""
Configuration UI Component
==========================
Streamlit UI component for managing QCA-AID configuration.

Requirements: 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4, 9.1, 9.2, 9.3, 9.4
"""

import streamlit as st
from pathlib import Path
from typing import Optional

from webapp_models.config_data import ConfigData, CoderSetting
from webapp_logic.file_browser_service import FileBrowserService
from webapp_logic.ui_helpers import (
    truncate_path_for_display,
    show_file_operation_success,
    show_file_operation_error,
    show_real_time_path_validation
)


def render_config_tab():
    """
    Rendert Konfigurationsreiter als Hauptlayout.
    
    Requirements:
    - 1.2: Projekt-Root-Verzeichnis prominent anzeigen
    - 1.3: Verzeichnis-Browser-Dialog für Projekt-Root
    - 2.1: Laden von XLSX und JSON Konfigurationen
    - 2.2: Standard-Pfad anzeigen wenn Eingabefeld leer
    - 2.3: Speichern mit Format-Auswahl
    - 2.4: Vollständigen Pfad im Eingabefeld anzeigen
    - 2.5: Automatische Format-Erkennung
    - 3.1-3.5: UI-Elemente für alle CONFIG-Parameter
    - 6.1-6.4: CODER_SETTINGS Verwaltung
    - 9.1-9.4: ATTRIBUTE_LABELS Verwaltung
    """
    st.header("⚙️ Konfiguration")
    st.markdown("Verwalten Sie alle Einstellungen für QCA-AID")
    
    # Get config from session state
    config = st.session_state.config_data
    
    # Project root management section
    render_project_root_section()
    
    st.markdown("---")
    
    # File operations section
    render_file_operations()
    
    st.markdown("---")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Modell-Einstellungen")
        render_model_settings()
        
        st.markdown("---")
        
        st.subheader("📊 Chunk-Einstellungen")
        render_chunk_settings()
        
        st.markdown("---")
        
        st.subheader("🔧 Analyse-Einstellungen")
        render_analysis_settings()
    
    with col2:
        st.subheader("👥 Coder-Einstellungen")
        render_coder_settings()
        
        st.markdown("---")
        
        st.subheader("🏷️ Attribut-Labels")
        render_attribute_labels()
    
    # Show validation status
    st.markdown("---")
    render_validation_status()


def render_project_root_section():
    """
    Rendert Projekt-Root-Verzeichnis Verwaltung.
    
    Requirement 1.2: WHEN ein Benutzer im Config-Tab ist 
                    THEN das System SHALL das aktuelle Projekt-Root-Verzeichnis prominent anzeigen
    Requirement 1.3: WHEN ein Benutzer auf "Projekt-Verzeichnis ändern" klickt 
                    THEN das System SHALL einen Verzeichnis-Browser-Dialog öffnen
    """
    st.subheader("📁 Projekt-Verzeichnis")
    
    # Get project manager from session state
    project_manager = st.session_state.project_manager
    current_root = project_manager.get_root_directory()
    
    # Display current project root prominently
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"**Aktuelles Projekt-Verzeichnis:**\n\n`{current_root}`")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("📁 Ändern", use_container_width=True, help="Projekt-Verzeichnis ändern"):
            # Open directory browser dialog
            new_root = FileBrowserService.open_directory_dialog(
                title="Projekt-Verzeichnis auswählen",
                initial_dir=current_root
            )
            
            if new_root:
                # Update project root
                success = project_manager.set_root_directory(new_root)
                if success:
                    st.success(f"✅ Projekt-Verzeichnis geändert zu: {new_root}")
                    # Save settings
                    project_manager.save_settings()
                    st.rerun()
                else:
                    st.error(f"❌ Fehler beim Setzen des Projekt-Verzeichnisses")


def render_file_operations():
    """
    Rendert Laden/Speichern Buttons mit Format-Auswahl und File Browser.
    
    Requirement 2.1: WHEN ein Benutzer auf "Konfiguration laden" klickt 
                    THEN das System SHALL den aktuellen Dateipfad im Eingabefeld anzeigen
    Requirement 2.2: WHEN das Eingabefeld leer ist 
                    THEN das System SHALL den Standard-Pfad anzeigen
    Requirement 2.3: WHEN ein Benutzer auf den Datei-Browser-Button klickt 
                    THEN das System SHALL einen nativen Datei-Auswahl-Dialog öffnen
    Requirement 2.4: WHEN ein Benutzer eine Datei im Dialog auswählt 
                    THEN das System SHALL den vollständigen Pfad im Eingabefeld anzeigen
    Requirement 2.5: WHEN ein Benutzer eine Datei auswählt 
                    THEN das System SHALL automatisch das passende Format erkennen
    """
    # Show current config file info
    config_manager = st.session_state.config_manager
    source = st.session_state.get('config_loaded_from', 'default')
    
    if source == 'user':
        # Try to determine which file was loaded
        project_manager = st.session_state.project_manager
        project_root = project_manager.get_root_directory()
        
        # Check for standard files
        from pathlib import Path
        json_path = Path(project_root) / "QCA-AID-Codebook.json"
        xlsx_path = Path(project_root) / "QCA-AID-Codebook.xlsx"
        
        if json_path.exists():
            st.caption(f"📁 Aktuelle Konfiguration: `{json_path.name}`")
        elif xlsx_path.exists():
            st.caption(f"📁 Aktuelle Konfiguration: `{xlsx_path.name}`")
        else:
            st.caption("📁 Konfiguration aus Datei geladen")
    else:
        st.caption("📁 Standard-Konfiguration (keine Datei geladen)")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**Dateioperationen**")
    
    with col2:
        # Load button
        if st.button("📂 Konfiguration laden", use_container_width=True):
            st.session_state.show_load_dialog = True
    
    with col3:
        # Save button
        if st.button("💾 Speichern", use_container_width=True):
            st.session_state.show_save_dialog = True
    
    # Load dialog
    if st.session_state.get('show_load_dialog', False):
        with st.expander("📂 Konfiguration laden", expanded=True):
            # Get project manager for default paths
            project_manager = st.session_state.project_manager
            
            # Initialize selected file path in session state if not present
            if 'selected_config_load_path' not in st.session_state:
                st.session_state.selected_config_load_path = ""
            
            # File path input with browser button
            col_path, col_browse = st.columns([4, 1])
            
            with col_path:
                # Show current path or default
                current_path = st.session_state.selected_config_load_path
                default_json = "QCA-AID-Codebook.json"
                default_xlsx = "QCA-AID-Codebook.xlsx"
                
                placeholder_text = f"Standard: {default_json} oder {default_xlsx}"
                
                # Requirement 8.3: Truncate long paths with ellipsis
                display_value = current_path
                help_text = "Leer lassen um Standard-Datei zu laden"
                
                if current_path and len(current_path) > 50:
                    display_value = truncate_path_for_display(current_path, 50)
                    # Show full path in tooltip
                    help_text = f"Vollständiger Pfad: {current_path}\n\n{help_text}"
                
                # Use a dynamic key to force widget recreation when file is selected
                widget_key = f"config_load_path_input_{hash(st.session_state.selected_config_load_path)}"
                
                file_path = st.text_input(
                    "Dateipfad:",
                    value=st.session_state.selected_config_load_path,
                    placeholder=placeholder_text,
                    help=help_text,
                    key=widget_key
                )
                
                # Update session state when user types in the field
                st.session_state.selected_config_load_path = file_path
                
                # Requirement 8.4: Real-time path validation
                if file_path and file_path.strip():
                    path_resolver = project_manager.path_resolver
                    show_real_time_path_validation(file_path, path_resolver, check_writable=False)
            
            with col_browse:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                # Requirement 8.1, 8.2: Browser button with icon and tooltip
                if st.button("📁", key="browse_config_load", help="Datei durchsuchen"):
                    # Open file browser
                    selected_file = FileBrowserService.open_file_dialog(
                        title="Konfigurationsdatei auswählen",
                        initial_dir=project_manager.get_root_directory(),
                        file_types=[
                            ("JSON files", "*.json"),
                            ("Excel files", "*.xlsx"),
                            ("All files", "*.*")
                        ]
                    )
                    
                    if selected_file:
                        # Update session state - don't modify widget state directly
                        st.session_state.selected_config_load_path = str(selected_file)
                        # Force rerun to update the text input with new value
                        st.rerun()
            
            # Auto-detect format from file extension
            detected_format = None
            if file_path and file_path.strip():
                file_path_lower = file_path.strip().lower()
                if file_path_lower.endswith('.json'):
                    detected_format = 'json'
                elif file_path_lower.endswith('.xlsx'):
                    detected_format = 'xlsx'
            
            # Track previous format to detect changes
            if 'previous_load_format' not in st.session_state:
                st.session_state.previous_load_format = detected_format or 'json'
            
            # Format selection with auto-detection
            if detected_format:
                st.info(f"📋 Format automatisch erkannt: **{detected_format.upper()}**")
                load_format = detected_format
                st.session_state.previous_load_format = detected_format
            else:
                # Only show radio buttons if format cannot be auto-detected
                load_format = st.radio(
                    "Format wählen:",
                    options=['json', 'xlsx'],
                    format_func=lambda x: 'JSON' if x == 'json' else 'Excel (XLSX)',
                    horizontal=True,
                    key="load_format_radio"
                )
                
                # Update file extension if format changed manually
                from pathlib import Path
                if file_path and load_format != st.session_state.previous_load_format:
                    path_obj = Path(file_path.strip())
                    old_ext = f".{st.session_state.previous_load_format}"
                    if path_obj.suffix.lower() == old_ext:
                        # Update extension to match new format
                        new_path = str(path_obj.with_suffix(f".{load_format}"))
                        st.session_state.selected_config_load_path = new_path
                        st.rerun()
                
                st.session_state.previous_load_format = load_format
            
            col_load1, col_load2 = st.columns(2)
            
            with col_load1:
                if st.button("Laden", key="load_config_btn", use_container_width=True):
                    config_manager = st.session_state.config_manager
                    
                    # Use the file path from session state (which contains the selected file)
                    selected_path = st.session_state.selected_config_load_path.strip()
                    
                    # Determine actual file path
                    if selected_path:
                        actual_path = selected_path
                        # Load configuration with specific file path
                        success, config_data, errors = config_manager.load_config(
                            file_path=selected_path,
                            format=load_format
                        )
                    else:
                        # Use default filename
                        actual_path = f"QCA-AID-Codebook.{load_format}"
                        success, config_data, errors = config_manager.load_config(
                            format=load_format
                        )
                    
                    if success:
                        st.session_state.config_data = config_data
                        st.session_state.config_modified = False
                        st.session_state.config_loaded_from = "user"
                        
                        # Speichere den aktuell geladenen Dateinamen für zukünftige Speicher-Operationen
                        from pathlib import Path
                        loaded_path = Path(actual_path)
                        st.session_state.current_config_filename = loaded_path.name
                        
                        # Requirement 8.5: Success confirmation with file info
                        from pathlib import Path
                        loaded_path = Path(actual_path)
                        additional_info = {
                            "Kategorien": len(config_data.coder_settings),
                            "Attribute": len(config_data.attribute_labels)
                        }
                        show_file_operation_success("geladen", loaded_path, additional_info)
                        
                        # Clear selected path
                        st.session_state.selected_config_load_path = ""
                        st.session_state.show_load_dialog = False
                        st.rerun()
                    else:
                        # Enhanced error display
                        from pathlib import Path
                        show_file_operation_error(
                            "Laden",
                            Path(actual_path),
                            "\n".join(errors),
                            ["Überprüfen Sie das Dateiformat", "Stellen Sie sicher, dass die Datei existiert"]
                        )
            
            with col_load2:
                if st.button("Abbrechen", key="cancel_load_config", use_container_width=True):
                    # Clear selected path
                    st.session_state.selected_config_load_path = ""
                    st.session_state.show_load_dialog = False
                    st.rerun()
    
    # Save dialog
    if st.session_state.get('show_save_dialog', False):
        with st.expander("💾 Konfiguration speichern", expanded=True):
            # Get project manager for default paths
            project_manager = st.session_state.project_manager
            
            # Initialize selected file path in session state if not present
            if 'selected_config_save_path' not in st.session_state:
                st.session_state.selected_config_save_path = ""
            
            # Track previous format to detect changes
            if 'previous_save_format' not in st.session_state:
                st.session_state.previous_save_format = 'json'
            
            # Format selection first (to determine default filename)
            save_format = st.radio(
                "Format wählen:",
                options=['json', 'xlsx'],
                format_func=lambda x: 'JSON (empfohlen)' if x == 'json' else 'Excel (XLSX)',
                horizontal=True,
                help="JSON ist das empfohlene Format für neue Konfigurationen",
                key="save_format_radio"
            )
            
            # File path input with browser button
            col_path, col_browse = st.columns([4, 1])
            
            with col_path:
                # Show current path or default filename
                current_path = st.session_state.selected_config_save_path
                
                # Bestimme Standard-Dateiname basierend auf aktuell geladener Datei
                current_config_filename = st.session_state.get('current_config_filename', f"QCA-AID-Codebook.{save_format}")
                if current_config_filename:
                    # Verwende den Namen der aktuell geladenen Datei, aber mit der gewählten Erweiterung
                    from pathlib import Path
                    base_name = Path(current_config_filename).stem  # Name ohne Erweiterung
                    default_filename = f"{base_name}.{save_format}"
                else:
                    default_filename = f"QCA-AID-Codebook.{save_format}"
                
                # Update file extension if format changed
                from pathlib import Path
                if current_path and save_format != st.session_state.previous_save_format:
                    path_obj = Path(current_path)
                    # Check if current extension matches old format
                    old_ext = f".{st.session_state.previous_save_format}"
                    if path_obj.suffix.lower() == old_ext:
                        # Update extension to match new format
                        display_path = str(path_obj.with_suffix(f".{save_format}"))
                        # Update session state to reflect the change
                        st.session_state.selected_config_save_path = display_path
                    else:
                        # Extension doesn't match - keep as is
                        display_path = current_path
                elif current_path:
                    # Format hasn't changed - use current path
                    display_path = current_path
                else:
                    # No path selected - use default
                    display_path = default_filename
                
                # Update previous format tracker
                st.session_state.previous_save_format = save_format
                
                placeholder_text = f"Standard: {default_filename}"
                
                # Requirement 8.3: Truncate long paths with ellipsis
                help_text = "Standard-Dateiname wird verwendet, wenn kein vollständiger Pfad angegeben wird"
                
                if display_path and len(display_path) > 50:
                    display_value = truncate_path_for_display(display_path, 50)
                    # Show full path in tooltip
                    help_text = f"Vollständiger Pfad: {display_path}\n\n{help_text}"
                else:
                    display_value = display_path
                
                # Use a dynamic key to force widget recreation when file is selected
                widget_key = f"config_save_path_input_{hash(st.session_state.selected_config_save_path)}"
                
                file_path = st.text_input(
                    "Dateipfad:",
                    value=st.session_state.selected_config_save_path,
                    placeholder=placeholder_text,
                    help=help_text,
                    key=widget_key
                )
                
                # Update session state when user types in the field
                st.session_state.selected_config_save_path = file_path
                
                # Requirement 8.4: Real-time path validation
                if file_path and file_path.strip():
                    path_resolver = project_manager.path_resolver
                    show_real_time_path_validation(file_path, path_resolver, check_writable=True)
            
            with col_browse:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                # Requirement 8.1, 8.2: Browser button with icon and tooltip
                if st.button("📁", key="browse_config_save", help="Speicherort wählen"):
                    # Open save file dialog
                    selected_file = FileBrowserService.save_file_dialog(
                        title="Konfiguration speichern",
                        initial_dir=project_manager.get_root_directory(),
                        default_filename=default_filename,
                        file_types=[
                            ("JSON files", "*.json") if save_format == 'json' else ("Excel files", "*.xlsx"),
                            ("All files", "*.*")
                        ]
                    )
                    
                    if selected_file:
                        # Update session state - don't modify widget state directly
                        st.session_state.selected_config_save_path = str(selected_file)
                        # Force rerun to update the text input with new value
                        st.rerun()
            
            col_save1, col_save2 = st.columns(2)
            
            with col_save1:
                if st.button("Speichern", key="save_config_btn", use_container_width=True):
                    config_manager = st.session_state.config_manager
                    config = st.session_state.config_data
                    
                    # Use the file path from session state (which contains the selected file)
                    selected_path = st.session_state.selected_config_save_path.strip()
                    
                    # Determine actual file path
                    if selected_path:
                        actual_path = selected_path
                        # Save configuration with specific file path
                        success, errors = config_manager.save_config(
                            config=config,
                            file_path=selected_path,
                            format=save_format
                        )
                    else:
                        # Use default filename
                        actual_path = f"QCA-AID-Codebook.{save_format}"
                        success, errors = config_manager.save_config(
                            config=config,
                            format=save_format
                        )
                    
                    if success:
                        # Reload config from file to ensure exact match with saved values
                        # This prevents widgets from triggering config_modified on rerun
                        reload_success, reloaded_config, reload_errors = config_manager.load_config(
                            file_path=actual_path,
                            format=save_format
                        )
                        
                        if reload_success and reloaded_config:
                            st.session_state.config_data = reloaded_config
                            st.session_state.config_modified = False
                        else:
                            # If reload fails, keep modified flag but show warning
                            st.warning("⚠️ Konfiguration gespeichert, aber Reload fehlgeschlagen. Bitte laden Sie die Datei manuell neu.")
                        
                        # Requirement 8.5: Success confirmation with file info
                        from pathlib import Path
                        saved_path = Path(actual_path)
                        additional_info = {
                            "Kategorien": len(config.coder_settings),
                            "Attribute": len(config.attribute_labels)
                        }
                        show_file_operation_success("gespeichert", saved_path, additional_info)
                        
                        # Clear selected path
                        st.session_state.selected_config_save_path = ""
                        st.session_state.show_save_dialog = False
                        st.rerun()
                    else:
                        # Enhanced error display
                        from pathlib import Path
                        show_file_operation_error(
                            "Speichern",
                            Path(actual_path),
                            "\n".join(errors),
                            ["Überprüfen Sie die Schreibrechte", "Stellen Sie sicher, dass das Verzeichnis existiert"]
                        )
            
            with col_save2:
                if st.button("Abbrechen", key="cancel_save_config", use_container_width=True):
                    # Clear selected path
                    st.session_state.selected_config_save_path = ""
                    st.session_state.show_save_dialog = False
                    st.rerun()


def _load_model_pricing(provider_display_name: str, model_id: str) -> Optional[tuple]:
    """
    Lädt Preisinformationen für ein Modell aus den JSON-Konfigurationsdateien.
    
    Args:
        provider_display_name: Anzeigename des Providers (z.B. 'OpenAI')
        model_id: ID des Modells (z.B. 'gpt-4o-mini')
    
    Returns:
        Tuple (input_cost, output_cost) oder None wenn nicht gefunden
    """
    import json
    from pathlib import Path
    
    # Map display names to config file names
    # For Mistral, we'll search in OpenRouter since Mistral models are there
    provider_file_map = {
        'OpenAI': ['openai.json'],
        'Anthropic': ['anthropic.json'],
        'Mistral': ['openrouter.json'],  # Mistral models are in OpenRouter
        'OpenRouter': ['openrouter.json'],
        'Local (LM Studio/Ollama)': []  # No pricing for local models
    }
    
    config_files = provider_file_map.get(provider_display_name, [])
    if not config_files:
        # No config files for this provider (e.g., local models)
        return None
    
    # Get path to config directory
    try:
        # IMPORTANT: Use the module directory, not the project directory
        # The config files are part of the QCA-AID installation, not the user's project
        module_root = Path(__file__).parent.parent.parent
        config_dir = module_root / 'QCA_AID_assets' / 'utils' / 'llm' / 'configs'
        
        # Try each config file
        for config_file in config_files:
            config_path = config_dir / config_file
            
            if not config_path.exists():
                continue
            
            # Load JSON
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Find model in models list
            models = config_data.get('models', [])
            for model in models:
                model_model_id = model.get('id', '')
                
                # Direct match
                if model_id == model_model_id:
                    input_cost = model.get('cost_per_1m_in')
                    output_cost = model.get('cost_per_1m_out')
                    
                    if input_cost is not None and output_cost is not None:
                        return (float(input_cost), float(output_cost))
                
                # Fuzzy matching for Mistral and OpenRouter
                # e.g. 'mistral-large-latest' matches 'mistralai/mistral-large'
                # e.g. 'deepseek/deepseek-v3.2' matches 'deepseek/deepseek-v3.2-exp'
                if provider_display_name in ['Mistral', 'OpenRouter']:
                    # Remove provider prefix from stored ID
                    stored_id_without_prefix = model_id.split('/')[-1]
                    config_id_without_prefix = model_model_id.split('/')[-1]
                    
                    # Check if one is a substring of the other or they share a common base
                    if (stored_id_without_prefix in config_id_without_prefix or 
                        config_id_without_prefix in stored_id_without_prefix):
                        input_cost = model.get('cost_per_1m_in')
                        output_cost = model.get('cost_per_1m_out')
                        
                        if input_cost is not None and output_cost is not None:
                            return (float(input_cost), float(output_cost))
        
        return None
        
    except Exception as e:
        # Silently fail and return None
        return None


def check_api_key_for_provider(provider_display_name: str, provider_display_map: dict):
    """
    Prüft ob ein API-Key für den ausgewählten Provider vorhanden ist.
    
    Args:
        provider_display_name: Anzeigename des Providers (z.B. 'OpenAI')
        provider_display_map: Mapping von Provider-IDs zu Anzeigenamen
    """
    import os
    
    # Map display name to provider ID
    reverse_map = {v: k for k, v in provider_display_map.items()}
    provider_id = reverse_map.get(provider_display_name, provider_display_name.lower())
    
    # API-Key-Variablen für jeden Provider
    api_key_map = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'mistral': 'MISTRAL_API_KEY',
        'openrouter': 'OPENROUTER_API_KEY',
        'local': None  # Lokale Modelle benötigen keinen API-Key
    }
    
    # Prüfe ob Provider einen API-Key benötigt
    api_key_var = api_key_map.get(provider_id)
    
    if api_key_var is None:
        # Lokale Modelle - keine Warnung
        if provider_id == 'local':
            st.info("ℹ️ Lokale Modelle benötigen keinen API-Key. Stellen Sie sicher, dass LM Studio (Port 1234) oder Ollama (Port 11434) läuft.")
        return
    
    # Prüfe ob API-Key gesetzt ist
    api_key = os.getenv(api_key_var)
    
    if not api_key:
        # API-Key fehlt - zeige Warnung mit Anleitung
        st.warning(f"⚠️ **API-Key für {provider_display_name} nicht gefunden**")
        
        with st.expander("📖 Anleitung: API-Key einrichten", expanded=True):
            st.markdown(f"""
            ### So richten Sie Ihren {provider_display_name} API-Key ein:
            
            #### Option 1: .env Datei (Empfohlen)
            
            Erstellen Sie eine `.env` Datei an **einem** der folgenden Orte:
            
            1. **Im QCA-AID-Projektverzeichnis** (empfohlen)
               - Pfad: `QCA-AID/.env`
            
            2. **In Ihrem Home-Verzeichnis**
               - Pfad: `~/.environ.env` (Windows: `C:\\Users\\IhrName\\.environ.env`)
            
            Fügen Sie folgende Zeile hinzu:
            ```
            {api_key_var}=ihr-api-key-hier
            ```
            
            Speichern Sie die Datei und starten Sie die Webapp neu.
            
            #### Option 2: Systemumgebungsvariable
            
            **Windows (PowerShell/CMD):**
            ```powershell
            setx {api_key_var} "ihr-api-key-hier"
            ```
            ⚠️ **Wichtig:** Nach `setx` müssen Sie die Webapp **neu starten** (Terminal schließen und neu öffnen)
            
            **Windows (GUI):**
            1. Suchen Sie nach "Umgebungsvariablen" im Startmenü
            2. Klicken Sie auf "Umgebungsvariablen bearbeiten"
            3. Unter "Benutzervariablen" klicken Sie auf "Neu"
            4. Name: `{api_key_var}`
            5. Wert: `ihr-api-key-hier`
            6. Starten Sie die Webapp neu
            
            **Linux/Mac:**
            ```bash
            # Temporär (nur für aktuelle Session):
            export {api_key_var}="ihr-api-key-hier"
            
            # Permanent (in ~/.bashrc oder ~/.zshrc):
            echo 'export {api_key_var}="ihr-api-key-hier"' >> ~/.bashrc
            source ~/.bashrc
            ```
            
            #### API-Key erhalten:
            """)
            
            # Provider-spezifische Links
            if provider_id == 'openai':
                st.markdown("- Besuchen Sie: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)")
            elif provider_id == 'anthropic':
                st.markdown("- Besuchen Sie: [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)")
            elif provider_id == 'mistral':
                st.markdown("- Besuchen Sie: [https://console.mistral.ai/api-keys/](https://console.mistral.ai/api-keys/)")
            elif provider_id == 'openrouter':
                st.markdown("- Besuchen Sie: [https://openrouter.ai/keys](https://openrouter.ai/keys)")
            
            st.markdown("""
            ---
            **Wichtig:** 
            - Fügen Sie `.env` zu Ihrer `.gitignore` hinzu, um API-Keys nicht versehentlich zu veröffentlichen
            - Teilen Sie Ihre API-Keys niemals öffentlich
            """)
    else:
        # API-Key gefunden - zeige Bestätigung
        # Zeige nur die ersten und letzten 4 Zeichen
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
        st.success(f"✅ API-Key für {provider_display_name} gefunden: `{masked_key}`")


def render_model_settings():
    """
    Rendert Modell-Einstellungen mit Dropdowns.
    
    Requirement 3.1: WHEN der Konfigurationsreiter angezeigt wird 
                    THEN das System SHALL Dropdown-Menüs für MODEL_PROVIDER und MODEL_NAME anzeigen
    
    Uses LLMProviderManager to dynamically load available models from all providers.
    """
    config = st.session_state.config_data
    
    # Map provider IDs to display names (define at function scope)
    provider_display_map = {
        'openai': 'OpenAI',
        'anthropic': 'Anthropic',
        'mistral': 'Mistral',
        'openrouter': 'OpenRouter',
        'local': 'Local (LM Studio/Ollama)'
    }
    
    # Initialize provider manager if not already done
    if 'llm_provider_manager' not in st.session_state:
        try:
            import asyncio
            import sys
            from pathlib import Path
            
            # Add parent directory to path
            parent_dir = Path(__file__).parent.parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from QCA_AID_assets.utils.llm.provider_manager import LLMProviderManager
            from QCA_AID_assets.core.config import get_provider_manager_config
            
            # Get config
            pm_config = get_provider_manager_config()
            
            # Create provider manager
            manager = LLMProviderManager(
                cache_dir=pm_config['CACHE_DIR'],
                fallback_dir=pm_config['FALLBACK_DIR'],
                config_dir=pm_config['CONFIG_DIR']
            )
            
            # Initialize manager (this loads all providers and models)
            asyncio.run(manager.initialize(force_refresh=False))
            
            st.session_state.llm_provider_manager = manager
            st.session_state.llm_models_loaded = True
        except Exception as e:
            st.warning(f"⚠️ Konnte Provider Manager nicht laden: {e}")
            st.session_state.llm_provider_manager = None
            st.session_state.llm_models_loaded = False
    
    # Get available providers and models
    if st.session_state.get('llm_provider_manager') and st.session_state.get('llm_models_loaded'):
        manager = st.session_state.llm_provider_manager
        
        try:
            # Get all supported providers
            all_providers = manager.get_supported_providers()
            
            # Always add 'local' to the list if not present
            if 'local' not in all_providers:
                all_providers.append('local')
            
            # Create provider options with display names
            provider_options = [provider_display_map.get(p, p.title()) for p in all_providers]
            
            # Map current config provider to display name
            current_provider_display = provider_display_map.get(config.model_provider.lower(), config.model_provider)
            current_provider_idx = provider_options.index(current_provider_display) if current_provider_display in provider_options else 0
            
        except Exception as e:
            st.error(f"Fehler beim Laden der Provider: {e}")
            # Fallback to hardcoded options
            provider_options = ['OpenAI', 'Anthropic', 'Mistral', 'OpenRouter', 'Local (LM Studio/Ollama)']
            current_provider_idx = 0
    else:
        # Fallback to hardcoded options if manager not available
        provider_options = ['OpenAI', 'Anthropic', 'Mistral', 'OpenRouter', 'Local (LM Studio/Ollama)']
        current_provider_idx = provider_options.index(config.model_provider) if config.model_provider in provider_options else 0
    
    new_provider = st.selectbox(
        "Modell-Anbieter",
        options=provider_options,
        index=current_provider_idx,
        help="Wählen Sie den LLM-Anbieter"
    )
    
    # Map display name back to provider ID for comparison
    reverse_map = {v: k for k, v in provider_display_map.items()}
    new_provider_id = reverse_map.get(new_provider, new_provider)
    
    if new_provider_id != config.model_provider:
        config.model_provider = new_provider_id
        st.session_state.config_modified = True
    
    # API-Key-Prüfung für den ausgewählten Provider
    check_api_key_for_provider(new_provider, provider_display_map)
    
    # Model Name (depends on provider)
    # Special handling for local models
    if new_provider == 'Local (LM Studio/Ollama)':
        # Show button to detect local models
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Erkennen", help="Sucht nach laufenden LM Studio oder Ollama Servern", use_container_width=True):
                with st.spinner("Suche nach lokalen Modellen..."):
                    try:
                        import asyncio
                        from QCA_AID_assets.utils.llm.local_detector import LocalDetector
                        
                        detector = LocalDetector()
                        local_models = asyncio.run(detector.detect_all())
                        
                        if local_models:
                            # Extract model IDs
                            model_options = []
                            for model in local_models:
                                model_id = model.get('id') or model.get('name') or model.get('model')
                                if model_id:
                                    # Filter out embedding models
                                    if 'embed' not in model_id.lower():
                                        model_options.append(model_id)
                            
                            if model_options:
                                st.success(f"✅ {len(model_options)} Modell(e) gefunden!")
                                st.session_state.local_models = model_options
                                st.rerun()
                            else:
                                st.warning("⚠️ Keine Chat-Modelle gefunden (nur Embedding-Modelle)")
                        else:
                            st.warning("⚠️ Keine lokalen Modelle gefunden. Stellen Sie sicher, dass LM Studio oder Ollama läuft.")
                    except Exception as e:
                        st.error(f"❌ Fehler: {e}")
        
        # Use cached local models if available
        if 'local_models' in st.session_state and st.session_state.local_models:
            model_options = st.session_state.local_models
            current_model_idx = model_options.index(config.model_name) if config.model_name in model_options else 0
        else:
            model_options = ['Bitte auf "Erkennen" klicken']
            current_model_idx = 0
            st.info("ℹ️ **Lokale Modelle verwenden:**\n\n"
                   "1. Starten Sie LM Studio (Port 1234) oder Ollama (Port 11434)\n"
                   "2. Laden Sie ein Modell in LM Studio/Ollama\n"
                   "3. Klicken Sie auf '🔄 Erkennen'\n"
                   "4. Wählen Sie ein erkanntes Modell aus")
    
    elif st.session_state.get('llm_provider_manager') and st.session_state.get('llm_models_loaded'):
        manager = st.session_state.llm_provider_manager
        
        try:
            # Get provider ID from display name
            reverse_map = {v: k for k, v in provider_display_map.items()}
            provider_id = reverse_map.get(new_provider, new_provider.lower())
            
            # Get models for selected provider
            provider_models = manager.get_models_by_provider(provider_id)
            
            if provider_models:
                # Create model options from model IDs
                model_options = [model.model_id for model in provider_models]
                
                # Try to find current model in options
                current_model_idx = model_options.index(config.model_name) if config.model_name in model_options else 0
            else:
                # No models found for this provider
                model_options = [config.model_name] if config.model_name else ['No models available']
                current_model_idx = 0
                
        except Exception as e:
            st.error(f"Fehler beim Laden der Modelle: {e}")
            # Fallback to current model
            model_options = [config.model_name] if config.model_name else ['gpt-4o-mini']
            current_model_idx = 0
    else:
        # Fallback to hardcoded options if manager not available
        if new_provider == 'OpenAI':
            model_options = [
                'gpt-4o',
                'gpt-4o-mini',
                'gpt-4-turbo',
                'gpt-4',
                'gpt-3.5-turbo'
            ]
        elif new_provider == 'Anthropic':
            model_options = [
                'claude-sonnet-4-5-20250929',
                'claude-3-5-sonnet-20241022',
                'claude-3-opus-20240229'
            ]
        elif new_provider == 'Mistral':
            model_options = [
                'mistral-large-latest',
                'mistral-medium-latest',
                'mistral-small-latest',
                'open-mistral-7b'
            ]
        else:  # OpenRouter, Local, or other
            model_options = [config.model_name] if config.model_name else ['mistral/mistral-large']
        
        current_model_idx = model_options.index(config.model_name) if config.model_name in model_options else 0
    
    new_model = st.selectbox(
        "Modell-Name",
        options=model_options,
        index=current_model_idx,
        help="Wählen Sie das spezifische Modell"
    )
    
    # Show pricing information for selected model
    # Load pricing from JSON config files
    pricing_info = _load_model_pricing(new_provider, new_model)
    
    if pricing_info:
        input_cost, output_cost = pricing_info
        # Show pricing info as caption for subtle display (escape $ to avoid LaTeX)
        st.caption(f"💰 Kosten: \\${input_cost:.2f} / 1M Input-Tokens · \\${output_cost:.2f} / 1M Output-Tokens")
    elif 'local' in new_provider.lower() or 'ollama' in new_model.lower() or 'lm-studio' in new_model.lower():
        st.caption("💰 Kostenlos (lokales Modell)")
    
    if new_model != config.model_name:
        config.model_name = new_model
        st.session_state.config_modified = True
    
    # Directories with validation
    from pathlib import Path
    project_manager = st.session_state.project_manager
    project_root = Path(project_manager.get_root_directory())
    
    new_data_dir = st.text_input(
        "Eingabeverzeichnis",
        value=config.data_dir,
        help="Verzeichnis mit Eingabedateien (relativ zum Projektverzeichnis oder absoluter Pfad)"
    )
    
    if new_data_dir != config.data_dir:
        config.data_dir = new_data_dir
        st.session_state.config_modified = True
    
    # Validate input directory
    if new_data_dir:
        input_path = Path(new_data_dir)
        
        # If relative path, resolve against project root
        if not input_path.is_absolute():
            input_path = project_root / input_path
        
        if not input_path.exists():
            st.warning(f"⚠️ Verzeichnis existiert nicht und wird bei Bedarf erstellt: `{input_path}`")
        elif not input_path.is_dir():
            st.error(f"❌ Pfad ist kein Verzeichnis: `{input_path}`")
        else:
            st.success(f"✅ Verzeichnis gefunden: `{input_path}`")
    else:
        st.error("❌ Eingabeverzeichnis darf nicht leer sein")
    
    new_output_dir = st.text_input(
        "Ausgabeverzeichnis",
        value=config.output_dir,
        help="Verzeichnis für Analyseergebnisse (relativ zum Projektverzeichnis oder absoluter Pfad)"
    )
    
    if new_output_dir != config.output_dir:
        config.output_dir = new_output_dir
        st.session_state.config_modified = True
        
        # Sync output_dir to Explorer config if it exists
        if 'explorer_config_data' in st.session_state:
            st.session_state.explorer_config_data.base_config['output_dir'] = new_output_dir
    
    # Validate output directory
    if new_output_dir:
        output_path = Path(new_output_dir)
        
        # If relative path, resolve against project root
        if not output_path.is_absolute():
            output_path = project_root / output_path
        
        if not output_path.exists():
            st.warning(f"⚠️ Verzeichnis existiert nicht und wird bei Bedarf erstellt: `{output_path}`")
        elif not output_path.is_dir():
            st.error(f"❌ Pfad ist kein Verzeichnis: `{output_path}`")
        else:
            st.success(f"✅ Verzeichnis gefunden: `{output_path}`")
    else:
        st.error("❌ Ausgabeverzeichnis darf nicht leer sein")


def render_chunk_settings():
    """
    Rendert Chunk-Einstellungen mit Zahleneingaben.
    
    Requirement 3.2: WHEN der Konfigurationsreiter angezeigt wird 
                    THEN das System SHALL Zahleneingabefelder mit Validierung für 
                    CHUNK_SIZE, CHUNK_OVERLAP und BATCH_SIZE anzeigen
    """
    config = st.session_state.config_data
    
    # Chunk Size
    new_chunk_size = st.number_input(
        "Chunk-Größe",
        min_value=100,
        max_value=10000,
        value=config.chunk_size,
        step=100,
        help="Größe der Textchunks in Zeichen"
    )
    
    if new_chunk_size != config.chunk_size:
        config.chunk_size = new_chunk_size
        st.session_state.config_modified = True
    
    # Chunk Overlap
    new_chunk_overlap = st.number_input(
        "Chunk-Überlappung",
        min_value=0,
        max_value=min(1000, new_chunk_size - 1),
        value=min(config.chunk_overlap, new_chunk_size - 1),
        step=10,
        help="Überlappung zwischen Chunks in Zeichen"
    )
    
    if new_chunk_overlap != config.chunk_overlap:
        config.chunk_overlap = new_chunk_overlap
        st.session_state.config_modified = True
    
    # Validate chunk overlap
    if new_chunk_overlap >= new_chunk_size:
        st.error("⚠️ Chunk-Überlappung muss kleiner als Chunk-Größe sein")
    
    # Batch Size
    new_batch_size = st.number_input(
        "Batch-Größe",
        min_value=1,
        max_value=100,
        value=config.batch_size,
        step=1,
        help="Anzahl paralleler API-Anfragen"
    )
    
    if new_batch_size != config.batch_size:
        config.batch_size = new_batch_size
        st.session_state.config_modified = True


def render_analysis_settings():
    """
    Rendert Analyse-Einstellungen mit Checkboxen und Dropdowns.
    
    Requirement 3.3: WHEN der Konfigurationsreiter angezeigt wird 
                    THEN das System SHALL Checkboxen für boolesche Parameter anzeigen
    Requirement 3.4: WHEN der Konfigurationsreiter angezeigt wird 
                    THEN das System SHALL Dropdown-Menüs für ANALYSIS_MODE und REVIEW_MODE anzeigen
    """
    config = st.session_state.config_data
    
    # Analysis Mode
    analysis_mode_options = ['deductive', 'inductive', 'abductive', 'grounded']
    current_mode_idx = analysis_mode_options.index(config.analysis_mode) if config.analysis_mode in analysis_mode_options else 0
    
    new_analysis_mode = st.selectbox(
        "Analyse-Modus",
        options=analysis_mode_options,
        index=current_mode_idx,
        help="Wählen Sie den Analyse-Modus"
    )
    
    if new_analysis_mode != config.analysis_mode:
        config.analysis_mode = new_analysis_mode
        st.session_state.config_modified = True
    
    # Review Mode
    review_mode_options = ['auto', 'manual', 'consensus', 'majority']
    current_review_idx = review_mode_options.index(config.review_mode) if config.review_mode in review_mode_options else 0
    
    new_review_mode = st.selectbox(
        "Review-Modus",
        options=review_mode_options,
        index=current_review_idx,
        help="Wählen Sie den Review-Modus"
    )
    
    if new_review_mode != config.review_mode:
        config.review_mode = new_review_mode
        st.session_state.config_modified = True
    
    # Boolean settings
    # new_enable_optimization = st.checkbox(
    #     "🚀 Neue effiziente Kodiermethode verwenden",
    #     value=config.enable_optimization,
    #     help="Aktiviert die optimierte Analyse mit Batching und Caching. Reduziert API-Calls um 50-73% und verbessert die Effizienz erheblich. (Empfohlen: Aktiviert)"
    # )
    
    # if new_enable_optimization != config.enable_optimization:
    #     config.enable_optimization = new_enable_optimization
    #     st.session_state.config_modified = True
    
    # if new_enable_optimization:
    #     st.info("ℹ️ Optimierte Methode aktiviert: Batching und Caching werden verwendet, um API-Calls zu reduzieren und die Effizienz zu verbessern.")
    # else:
    #     st.warning("⚠️ Optimierte Methode deaktiviert: Es wird die Standard-Analyse verwendet (mehr API-Calls, höhere Kosten).")
    
    new_code_with_context = st.checkbox(
        "Mit Kontext kodieren",
        value=config.code_with_context,
        help="Umgebenden Kontext bei der Kodierung berücksichtigen"
    )
    
    if new_code_with_context != config.code_with_context:
        config.code_with_context = new_code_with_context
        st.session_state.config_modified = True
    
    new_multiple_codings = st.checkbox(
        "Mehrfachkodierungen erlauben",
        value=config.multiple_codings,
        help="Erlaube mehrere Kategorien pro Textsegment"
    )
    
    if new_multiple_codings != config.multiple_codings:
        config.multiple_codings = new_multiple_codings
        st.session_state.config_modified = True
    
    # Multiple coding threshold (only if multiple codings enabled)
    if new_multiple_codings:
        new_threshold = st.slider(
            "Mehrfachkodierungs-Schwellwert",
            min_value=0.0,
            max_value=1.0,
            value=config.multiple_coding_threshold,
            step=0.05,
            help="Konfidenz-Schwellwert für Mehrfachkodierungen"
        )
        
        if new_threshold != config.multiple_coding_threshold:
            config.multiple_coding_threshold = new_threshold
            st.session_state.config_modified = True
    
    # PDF Export settings
    new_export_pdfs = st.checkbox(
        "Annotierte PDFs exportieren",
        value=config.export_annotated_pdfs,
        help="Erstelle annotierte PDF-Versionen der Eingabedateien"
    )
    
    if new_export_pdfs != config.export_annotated_pdfs:
        config.export_annotated_pdfs = new_export_pdfs
        st.session_state.config_modified = True
    
    if new_export_pdfs:
        new_pdf_threshold = st.slider(
            "PDF-Annotations-Schwellwert",
            min_value=0.0,
            max_value=1.0,
            value=config.pdf_annotation_fuzzy_threshold,
            step=0.05,
            help="Fuzzy-Matching-Schwellwert für PDF-Annotationen"
        )
        
        if new_pdf_threshold != config.pdf_annotation_fuzzy_threshold:
            config.pdf_annotation_fuzzy_threshold = new_pdf_threshold
            st.session_state.config_modified = True
    
    # Relevance Threshold setting
    st.markdown("---")
    
    # Erklärung der LLM-Schwelle
    st.markdown("### 🎯 Relevanz-Schwellwert")
    st.info("""
    **Wie funktioniert die Relevanz-Bewertung:**
    - Das LLM wendet automatisch eine ~0.3-0.4 Schwelle an (basierend auf Training)
    - **0.3 (Standard)**: Verwendet LLM-Entscheidungen wie sie sind
    - **Höhere Werte (0.4-1.0)**: Strengere Filterung, weniger Segmente
    - **Niedrigere Werte (0.0-0.2)**: Inkludiert auch vom LLM verworfene Segmente
    """)
    
    new_relevance_threshold = st.slider(
        "Relevanz-Schwellwert",
        min_value=0.0,
        max_value=1.0,
        value=config.relevance_threshold,
        step=0.05,
        help="0.3 = LLM-Standard | Höher = strenger | Niedriger = inkludiert LLM-verworfene Segmente"
    )
    
    # Warnung bei niedrigen Werten
    if new_relevance_threshold < 0.3:
        st.warning(f"⚠️ Wert unter 0.3: Inkludiert Segmente die das LLM als nicht relevant eingestuft hat (Confidence {new_relevance_threshold:.1f}-0.3)")
    elif new_relevance_threshold > 0.5:
        st.info(f"ℹ️ Wert über 0.5: Sehr strenge Filterung, nur hochrelevante Segmente (Confidence ≥{new_relevance_threshold:.1f})")
    
    if new_relevance_threshold != config.relevance_threshold:
        config.relevance_threshold = new_relevance_threshold
        st.session_state.config_modified = True
    
    # Info about relevance threshold
    if new_relevance_threshold > 0.0:
        st.info(f"ℹ️ Nur Segmente mit Relevanz-Konfidenz ≥ {new_relevance_threshold:.2f} werden analysiert.")
    else:
        st.info("ℹ️ Alle vom LLM als relevant identifizierten Segmente werden analysiert (empfohlen).")


def render_coder_settings():
    """
    Rendert Coder-Einstellungen mit dynamischer Liste.
    
    Requirement 6.1: WHEN der Konfigurationsreiter angezeigt wird 
                    THEN das System SHALL eine Liste der konfigurierten Coder anzeigen
    Requirement 6.2: WHEN ein Benutzer auf "Coder hinzuFügen" klickt 
                    THEN das System SHALL ein neues Coder-Konfigurationsformular anzeigen
    Requirement 6.3: WHEN ein Coder konfiguriert wird 
                    THEN das System SHALL Eingabefelder für temperature und coder_id anzeigen
    Requirement 6.4: WHEN ein Benutzer auf "Coder entfernen" klickt 
                    THEN das System SHALL den ausgewählten Coder aus der Liste entfernen
    """
    config = st.session_state.config_data
    
    st.markdown("Konfigurieren Sie mehrere KI-Coder mit unterschiedlichen Temperaturen")
    
    # Manual Coding Option
    st.markdown("---")
    manual_enabled = st.checkbox(
        "🖐️ Manuelles Kodieren aktivieren",
        value=config.manual_coding_enabled,
        key="manual_coding_enabled_checkbox",
        help="Aktiviert die manuelle Kodierung durch einen menschlichen Kodierer zusätzlich zu den KI-Codern"
    )
    
    if manual_enabled != config.manual_coding_enabled:
        config.manual_coding_enabled = manual_enabled
        st.session_state.config_modified = True
    
    if manual_enabled:
        st.info("ℹ️ Bei aktivierter manueller Kodierung wird nach der automatischen Kodierung ein Fenster geöffnet, in dem Sie jedes Segment manuell kodieren können.")
    
    st.markdown("---")
    st.markdown("**KI-Coder:**")
    
    # Display existing coders
    for i, coder in enumerate(config.coder_settings):
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                new_temp = st.number_input(
                    f"Temperatur",
                    min_value=0.0,
                    max_value=2.0,
                    value=coder.temperature,
                    step=0.1,
                    key=f"coder_temp_{i}",
                    help="Kreativität des Coders (0.0 = deterministisch, 2.0 = sehr kreativ)"
                )
                
                if new_temp != coder.temperature:
                    coder.temperature = new_temp
                    st.session_state.config_modified = True
            
            with col2:
                new_id = st.text_input(
                    f"Coder-ID",
                    value=coder.coder_id,
                    key=f"coder_id_{i}",
                    help="Eindeutige Kennung für diesen Coder"
                )
                
                if new_id != coder.coder_id:
                    coder.coder_id = new_id
                    st.session_state.config_modified = True
                
                # Validate coder ID uniqueness
                if new_id:
                    other_ids = [c.coder_id for j, c in enumerate(config.coder_settings) if j != i]
                    if new_id in other_ids:
                        st.error(f"❌ Coder-ID '{new_id}' ist nicht eindeutig")
                else:
                    st.error("❌ Coder-ID darf nicht leer sein")
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                if st.button("🗑️", key=f"remove_coder_{i}", help="Coder entfernen"):
                    if len(config.coder_settings) > 1:
                        config.coder_settings.pop(i)
                        st.session_state.config_modified = True
                        st.rerun()
                    else:
                        st.error("Mindestens ein Coder erforderlich")
            
            st.markdown("---")
    
    # Add new coder button
    if st.button("➕ Coder hinzuFügen", use_container_width=True):
        # Generate new coder ID
        existing_ids = [c.coder_id for c in config.coder_settings]
        new_id = f"auto_{len(config.coder_settings) + 1}"
        counter = len(config.coder_settings) + 1
        while new_id in existing_ids:
            counter += 1
            new_id = f"auto_{counter}"
        
        # Add new coder with default temperature
        new_coder = CoderSetting(temperature=0.3, coder_id=new_id)
        config.coder_settings.append(new_coder)
        st.session_state.config_modified = True
        st.rerun()


def render_attribute_labels():
    """
    Rendert Attribut-Labels mit Add/Remove Funktionalität.
    
    Requirement 9.1: WHEN der Konfigurationsreiter angezeigt wird 
                    THEN das System SHALL eine Liste der ATTRIBUTE_LABELS anzeigen
    Requirement 9.2: WHEN ein Benutzer auf "Attribut hinzuFügen" klickt 
                    THEN das System SHALL Eingabefelder für Schlüssel und Bezeichnung anzeigen
    Requirement 9.3: WHEN ein Benutzer ein Attribut bearbeitet 
                    THEN das System SHALL die Änderungen sofort in der Vorschau anzeigen
    Requirement 9.4: WHEN ein Benutzer ein Attribut entfernt 
                    THEN das System SHALL eine Bestätigung anfordern
    """
    config = st.session_state.config_data
    
    st.markdown("Definieren Sie Metadaten-Attribute für Dateinamen-Extraktion")
    
    # Display existing attributes
    attributes_to_remove = []
    
    for key, label in config.attribute_labels.items():
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            new_key = st.text_input(
                "Schlüssel",
                value=key,
                key=f"attr_key_{key}",
                help="Interner Schlüssel (z.B. 'attribut1')"
            )
        
        with col2:
            new_label = st.text_input(
                "Bezeichnung",
                value=label,
                key=f"attr_label_{key}",
                help="Anzeigename (z.B. 'Quelle')"
            )
            
            # Update if changed
            if new_label != label:
                config.attribute_labels[key] = new_label
                st.session_state.config_modified = True
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("🗑️", key=f"remove_attr_{key}", help="Attribut entfernen"):
                # Mark for removal (will be processed after loop)
                attributes_to_remove.append(key)
        
        # Handle key change (need to update dict)
        if new_key != key and new_key not in config.attribute_labels:
            config.attribute_labels[new_key] = config.attribute_labels.pop(key)
            st.session_state.config_modified = True
            st.rerun()
        
        st.markdown("---")
    
    # Process removals
    for key in attributes_to_remove:
        if key in config.attribute_labels:
            del config.attribute_labels[key]
            st.session_state.config_modified = True
            st.rerun()
    
    # Add new attribute section
    with st.expander("➕ Neues Attribut hinzuFügen"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_attr_key = st.text_input(
                "Neuer Schlüssel",
                value="",
                key="new_attr_key",
                help="z.B. 'quelle', 'jahr', 'kategorie'"
            )
        
        with col2:
            new_attr_label = st.text_input(
                "Neue Bezeichnung",
                value="",
                key="new_attr_label",
                help="z.B. 'Quelle', 'Jahr', 'Kategorie'"
            )
        
        if st.button("Attribut hinzuFügen", use_container_width=True):
            if new_attr_key and new_attr_label:
                if new_attr_key not in config.attribute_labels:
                    config.attribute_labels[new_attr_key] = new_attr_label
                    st.session_state.config_modified = True
                    st.success(f"✅ Attribut '{new_attr_key}' hinzugefügt")
                    st.rerun()
                else:
                    st.error(f"❌ Attribut '{new_attr_key}' existiert bereits")
            else:
                st.error("❌ Bitte beide Felder ausfüllen")
    
    # Preview section
    if config.attribute_labels:
        st.markdown("**Vorschau:**")
        preview_text = " | ".join([f"{k}: {v}" for k, v in config.attribute_labels.items()])
        st.code(preview_text, language=None)


def render_validation_status():
    """
    Zeigt Validierungsstatus der Konfiguration an.
    
    Requirement 3.5: WHEN ein Benutzer einen ungültigen Wert eingibt 
                    THEN das System SHALL eine Inline-Validierungsmeldung anzeigen
    """
    config = st.session_state.config_data
    
    # Validate configuration
    is_valid, errors = config.validate()
    
    if is_valid:
        st.success("✅ Konfiguration ist gültig")
    else:
        st.error("❌ Konfiguration enthält Fehler:")
        for error in errors:
            st.error(f"  • {error}")
    
    # Show modification status
    if st.session_state.config_modified:
        st.warning("⚠️ Ungespeicherte Änderungen vorhanden")
