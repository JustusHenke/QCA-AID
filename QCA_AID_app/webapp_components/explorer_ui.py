"""
Explorer UI Component
=====================
Streamlit UI component for managing QCA-AID Explorer configuration and output files.

Requirements: 1.2, 7.1, 7.2, 7.3, 7.4, 11.1, 11.2, 11.3, 11.4, 11.5
"""

import streamlit as st
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd

from webapp_models.file_info import ExplorerConfig
from webapp_models.explorer_config_data import ExplorerConfigData, AnalysisConfig
from webapp_logic.explorer_config_manager import ExplorerConfigManager
from webapp_logic.explorer_analysis_runner import ExplorerAnalysisRunner
from webapp_components.config_ui import check_api_key_for_provider


def render_explorer_model_settings():
    """
    Rendert Modell-Einstellungen für Explorer (Post-hoc-Analyse).
    
    Implementiert bidirektionale Synchronisation zwischen config_data und 
    explorer_config_data.base_config.
    
    Requirements: 1.1, 1.2, 2.1, 3.1, 3.2
    """
    import asyncio
    import sys
    from pathlib import Path
    
    # Get config from session state
    config = st.session_state.config_data
    explorer_config = st.session_state.explorer_config_data
    
    # Requirement 3.1: Read values from explorer_config_data.base_config with fallback to config_data
    # Requirement 2.1: Use config_data as default when base_config is empty
    current_provider = explorer_config.base_config.get('provider') or config.model_provider
    current_model = explorer_config.base_config.get('model') or config.model_name
    
    # Map provider IDs to display names
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
            
            # Initialize manager
            asyncio.run(manager.initialize(force_refresh=False))
            
            st.session_state.llm_provider_manager = manager
            st.session_state.llm_models_loaded = True
        except Exception as e:
            st.warning(f"⚠️ Konnte Provider Manager nicht laden: {e}")
            st.session_state.llm_provider_manager = None
            st.session_state.llm_models_loaded = False
    
    # Create two columns for provider and model
    col1, col2 = st.columns(2)
    
    with col1:
        # Get available providers
        if st.session_state.get('llm_provider_manager') and st.session_state.get('llm_models_loaded'):
            manager = st.session_state.llm_provider_manager
            try:
                all_providers = manager.get_supported_providers()
                provider_options = [provider_display_map.get(p, p.title()) for p in all_providers]
                current_provider_display = provider_display_map.get(current_provider.lower(), current_provider)
                current_provider_idx = provider_options.index(current_provider_display) if current_provider_display in provider_options else 0
            except Exception:
                provider_options = ['OpenAI', 'Anthropic', 'Mistral', 'OpenRouter']
                current_provider_idx = 0
        else:
            provider_options = ['OpenAI', 'Anthropic', 'Mistral', 'OpenRouter']
            current_provider_display = provider_display_map.get(current_provider, current_provider)
            current_provider_idx = provider_options.index(current_provider_display) if current_provider_display in provider_options else 0
        
        new_provider = st.selectbox(
            "Modell-Anbieter",
            options=provider_options,
            index=current_provider_idx,
            help="Wählen Sie den LLM-Anbieter für Post-hoc-Analysen",
            key="explorer_model_provider"
        )
        
        # Requirement 3.2: Update both config_data AND explorer_config_data.base_config
        # Requirement 1.1: Store provider value in explorer_config_data.base_config['provider']
        reverse_map = {v: k for k, v in provider_display_map.items()}
        new_provider_id = reverse_map.get(new_provider, new_provider.lower())
        
        if new_provider_id != current_provider:
            # Bidirectional update
            config.model_provider = new_provider_id
            explorer_config.base_config['provider'] = new_provider_id
            st.session_state.config_modified = True
            st.session_state.explorer_config_modified = True
    
    with col2:
        # Get models for selected provider
        if st.session_state.get('llm_provider_manager') and st.session_state.get('llm_models_loaded'):
            manager = st.session_state.llm_provider_manager
            try:
                reverse_map = {v: k for k, v in provider_display_map.items()}
                provider_id = reverse_map.get(new_provider, new_provider.lower())
                provider_models = manager.get_models_by_provider(provider_id)
                
                if provider_models:
                    model_options = [model.model_id for model in provider_models]
                    current_model_idx = model_options.index(current_model) if current_model in model_options else 0
                else:
                    model_options = [current_model] if current_model else ['No models available']
                    current_model_idx = 0
            except Exception:
                model_options = [current_model] if current_model else ['gpt-4o-mini']
                current_model_idx = 0
        else:
            # Fallback
            if new_provider == 'OpenAI':
                model_options = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']
            elif new_provider == 'Anthropic':
                model_options = ['claude-sonnet-4-5-20250929', 'claude-3-5-sonnet-20241022']
            elif new_provider == 'Mistral':
                model_options = ['mistral-large-latest', 'mistral-medium-latest']
            else:
                model_options = [current_model] if current_model else ['gpt-4o-mini']
            
            current_model_idx = model_options.index(current_model) if current_model in model_options else 0
        
        new_model = st.selectbox(
            "Modell-Name",
            options=model_options,
            index=current_model_idx,
            help="Wählen Sie das spezifische Modell",
            key="explorer_model_name"
        )
        
        # Requirement 3.2: Update both config_data AND explorer_config_data.base_config
        # Requirement 1.2: Store model value in explorer_config_data.base_config['model']
        if new_model != current_model:
            # Bidirectional update
            config.model_name = new_model
            explorer_config.base_config['model'] = new_model
            st.session_state.config_modified = True
            st.session_state.explorer_config_modified = True
    
    # API-Key-Prüfung
    check_api_key_for_provider(new_provider, provider_display_map)
    
    st.markdown("---")
    
    # Keyword-Harmonisierung Option
    st.markdown("**🔧 Erweiterte Optionen**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get current value from base_config
        current_clean_keywords = explorer_config.base_config.get('clean_keywords', True)
        if isinstance(current_clean_keywords, str):
            current_clean_keywords = current_clean_keywords.lower() in ('true', 'ja', 'yes', '1')
        
        new_clean_keywords = st.checkbox(
            "Keyword-Harmonisierung aktivieren",
            value=current_clean_keywords,
            key="explorer_clean_keywords",
            help="Harmonisiert ähnliche Schlüsselwörter automatisch (z.B. 'Nachhaltigkeit' und 'nachhaltig')"
        )
        
        if new_clean_keywords != current_clean_keywords:
            explorer_config.base_config['clean_keywords'] = new_clean_keywords
            st.session_state.explorer_config_modified = True
    
    with col2:
        # Similarity threshold for keyword harmonization
        current_threshold = explorer_config.base_config.get('similarity_threshold', 0.7)
        if isinstance(current_threshold, str):
            try:
                current_threshold = float(current_threshold)
            except ValueError:
                current_threshold = 0.7
        
        new_threshold = st.slider(
            "Ähnlichkeitsschwelle",
            min_value=0.5,
            max_value=0.95,
            value=float(current_threshold),
            step=0.05,
            key="explorer_similarity_threshold",
            help="Schwellenwert für die Keyword-Harmonisierung (höher = strenger)",
            disabled=not new_clean_keywords
        )
        
        if new_threshold != current_threshold:
            explorer_config.base_config['similarity_threshold'] = new_threshold
            st.session_state.explorer_config_modified = True


def render_explorer_tab():
    """
    Rendert Explorer-Reiter als Hauptlayout.
    
    Requirements:
    - 7.1-7.4: Output-Dateien anzeigen und verwalten
    - 11.1-11.5: Explorer-Konfiguration verwalten
    """
    st.header("📊 Explorer")
    st.markdown("Analysieren und visualisieren Sie Ihre QCA-AID Ergebnisse")
    
    # Check if explorer is active
    if st.session_state.get('explorer_active', False) and st.session_state.get('explorer_file_path'):
        render_explorer_view()
        return
    
    # Configuration view
    render_explorer_config_view()


def render_config_file_controls():
    """
    Renders load/save controls for explorer configuration.
    
    Requirements: 1.1, 1.5
    """
    st.subheader("📁 Konfigurationsdatei")
    
    # Show default config path
    manager = st.session_state.explorer_config_manager
    default_json_path = manager.json_path
    st.caption(f"Standard-Pfad: `{default_json_path}`")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        # Show current config source
        source = st.session_state.get('explorer_config_loaded_from', 'default')
        if source == 'file':
            st.success("✅ Konfiguration aus Datei geladen")
        else:
            st.info("ℹ️ Standard-Konfiguration (keine Datei geladen)")
    
    with col2:
        if st.button("📂 Laden", use_container_width=True):
            manager = st.session_state.explorer_config_manager
            success, loaded_config_data, errors = manager.load_config()
            
            if success and loaded_config_data:
                st.session_state.explorer_config_data = loaded_config_data
                st.session_state.explorer_config_loaded_from = "file"
                st.session_state.explorer_config_modified = False
                
                # Requirement 1.4: Copy LLM settings from base_config to config_data after loading
                # Requirement 2.4: Use LLM settings from loaded config
                # Requirement 3.5: Transfer provider and model from base_config to config_data
                try:
                    config_data = st.session_state.config_data
                    base_config = loaded_config_data.base_config
                    
                    # Copy provider with validation and fallback
                    if 'provider' in base_config and base_config['provider']:
                        # Validate provider
                        valid_providers = ['openai', 'anthropic', 'mistral', 'openrouter', 'local']
                        provider = base_config['provider'].lower()
                        if provider in valid_providers:
                            config_data.model_provider = provider
                        else:
                            # Invalid provider - use fallback
                            st.warning(f"⚠️ Ungültiger Provider '{provider}' in Config - verwende Standard")
                    
                    # Copy model with validation
                    if 'model' in base_config and base_config['model']:
                        config_data.model_name = base_config['model']
                    
                except Exception as e:
                    # Error handling for missing or invalid values
                    st.warning(f"⚠️ Fehler beim Synchronisieren der LLM-Einstellungen: {e}")
                
                st.success("✅ Konfiguration erfolgreich geladen")
                st.rerun()
            else:
                st.error(f"❌ Fehler beim Laden: {', '.join(errors)}")
    
    with col3:
        if st.button("💾 Speichern (JSON)", use_container_width=True):
            manager = st.session_state.explorer_config_manager
            explorer_config_data = st.session_state.explorer_config_data
            config_data = st.session_state.config_data
            
            # Requirement 3.3: Copy LLM settings from config_data to base_config before saving
            explorer_config_data.base_config['provider'] = config_data.model_provider
            explorer_config_data.base_config['model'] = config_data.model_name
            
            success, errors = manager.save_config(explorer_config_data, format='json')
            
            if success:
                st.session_state.explorer_config_modified = False
                st.success("✅ Als JSON gespeichert")
                st.rerun()
            else:
                st.error(f"❌ Fehler beim Speichern: {', '.join(errors)}")
    
    with col4:
        if st.button("💾 Speichern (XLSX)", use_container_width=True):
            manager = st.session_state.explorer_config_manager
            explorer_config_data = st.session_state.explorer_config_data
            config_data = st.session_state.config_data
            
            # Requirement 3.3: Copy LLM settings from config_data to base_config before saving
            explorer_config_data.base_config['provider'] = config_data.model_provider
            explorer_config_data.base_config['model'] = config_data.model_name
            
            success, errors = manager.save_config(explorer_config_data, format='xlsx')
            
            if success:
                st.session_state.explorer_config_modified = False
                st.success("✅ Als XLSX gespeichert")
                st.rerun()
            else:
                st.error(f"❌ Fehler beim Speichern: {', '.join(errors)}")
    
    st.markdown("---")


def render_explorer_config_view():
    """
    Rendert die Konfigurations-Ansicht mit Tabbed Interface.
    
    Requirements: 1.2, 7.1, 7.2, 7.3, 7.4, 2.1, 2.5
    """
    # Session state is initialized in webapp.py initialize_session_state()
    # Display any load errors from initialization
    if st.session_state.get('explorer_config_load_errors'):
        st.error(f"⚠️ Fehler beim Laden der Konfiguration: {', '.join(st.session_state.explorer_config_load_errors)}")
        st.info("💡 Die Konfigurationsdatei hat möglicherweise ein altes Format. Erstellen Sie eine neue Konfiguration oder laden Sie eine gültige Datei.")
        # Clear errors after displaying
        del st.session_state.explorer_config_load_errors
    
    # Requirement 2.1: Initialize base_config with values from config_data when base_config is empty
    # Requirement 2.5: Use config_data as fallback when no Explorer-Config exists
    config_data = st.session_state.config_data
    explorer_config_data = st.session_state.explorer_config_data
    
    # Ensure provider and model are always present in base_config
    if not explorer_config_data.base_config.get('provider'):
        explorer_config_data.base_config['provider'] = config_data.model_provider
    
    if not explorer_config_data.base_config.get('model'):
        explorer_config_data.base_config['model'] = config_data.model_name
    
    # Add load/save buttons at the top
    render_config_file_controls()
    
    # Workflow hint
    config = st.session_state.config_data
    project_root = st.session_state.project_manager.get_root_directory()
    output_dir = Path(project_root) / config.output_dir
    has_results = output_dir.exists() and any(output_dir.glob('QCA-AID_Analysis_*.xlsx'))
    
    if not has_results:
        st.warning("⚠️ **Keine Analyseergebnisse gefunden.** Führen Sie zuerst eine Analyse im **Analyse**-Tab durch.")
    
    # Step 0: Model Settings for Post-hoc Analysis
    st.subheader("🤖 Modell-Einstellungen")
    st.markdown("Wählen Sie das LLM-Modell für die Post-hoc-Analyse (Zusammenfassungen, Sentiment-Analyse, etc.)")
    render_explorer_model_settings()
    
    st.markdown("---")
    
    # Step 1: Select Analysis File
    st.subheader("1️⃣ Analysedatei auswählen")
    selected_file = render_file_selector()
    
    if not selected_file:
        st.info("💡 Wählen Sie eine Analysedatei aus, um fortzufahren.")
        return
    
    st.markdown("---")
    
    # Step 2: Configure Analyses with Tabbed Interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("2️⃣ Analysen konfigurieren")
    
    with col2:
        # Show unsaved changes indicator
        if st.session_state.get('explorer_config_modified', False):
            st.warning("⚠️ Ungespeicherte Änderungen")
    
    render_explorer_config_tabs()
    
    st.markdown("---")
    
    # Step 3: Start Explorer
    st.subheader("3️⃣ Explorer starten")
    render_explorer_controls(selected_file)


def render_explorer_config_tabs():
    """
    Renders tabbed interface for analysis configurations.
    
    Requirement 1.2: WHEN multiple analysis configurations exist 
                    THEN the system SHALL display each analysis in a separate tab
    Requirements: 7.1, 7.2, 7.3, 7.4
    """
    explorer_config = st.session_state.explorer_config_data
    
    # Add Analysis button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**{len(explorer_config.analysis_configs)} Analyse(n) konfiguriert**")
    
    with col2:
        if st.button("➕ Analyse hinzufügen", use_container_width=True, key="add_analysis_btn"):
            handle_add_analysis()
    
    # If no analyses, show message
    if not explorer_config.analysis_configs:
        st.info("💡 Keine Analysen konfiguriert. Klicken Sie auf 'Analyse hinzufügen', um zu beginnen.")
        return
    
    # Create tabs for each analysis
    tab_labels = []
    for i, analysis in enumerate(explorer_config.analysis_configs):
        # Add visual indicator for active/inactive
        status_icon = "✅" if analysis.active else "⏸️"
        tab_labels.append(f"{status_icon} {analysis.name}")
    
    tabs = st.tabs(tab_labels)
    
    # Render each analysis tab
    for i, (tab, analysis) in enumerate(zip(tabs, explorer_config.analysis_configs)):
        with tab:
            render_analysis_tab(analysis, i)


def render_analysis_tab(analysis: AnalysisConfig, index: int):
    """
    Renders single analysis configuration tab.
    
    Args:
        analysis: AnalysisConfig to render
        index: Index of the analysis in the configuration list
    
    Requirements: 1.2, 2.4, 7.3, 7.4
    """
    # Visual styling based on active state
    # Requirement 2.4: WHEN displaying analysis tabs THEN the system SHALL visually indicate 
    # which analyses are active and which are inactive
    if not analysis.active:
        st.markdown("""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; border-left: 4px solid #ff6b6b;">
            <p style="color: #666; margin: 0;">⏸️ <strong>Diese Analyse ist deaktiviert</strong> - Sie wird beim Ausführen übersprungen</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
    else:
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 4px solid #4caf50;">
            <p style="color: #2e7d32; margin: 0;">✅ <strong>Diese Analyse ist aktiv</strong> - Sie wird beim Ausführen durchgeführt</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
    
    # Header with controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Active/Inactive toggle
        # Requirement 2.3: WHEN the user toggles an analysis active state in the UI 
        # THEN the system SHALL update the 'active' parameter in the configuration
        new_active = st.checkbox(
            "Analyse aktiv",
            value=analysis.active,
            key=f"active_{index}",
            help="Deaktivierte Analysen werden übersprungen"
        )
        
        if new_active != analysis.active:
            handle_update_analysis(index, {'active': new_active})
    
    with col2:
        # Move up/down buttons for reordering
        if index > 0:
            if st.button("⬆️ Nach oben", key=f"move_up_{index}", use_container_width=True):
                handle_reorder_analysis(index, index - 1)
        
    with col3:
        # Remove button
        if st.button("🗑️ Entfernen", key=f"remove_{index}", use_container_width=True, type="secondary"):
            handle_remove_analysis(index)
    
    st.markdown("---")
    
    # Analysis name
    new_name = st.text_input(
        "Analysename:",
        value=analysis.name,
        key=f"name_{index}"
    )
    
    if new_name != analysis.name:
        handle_update_analysis(index, {'name': new_name})
    
    # Analysis type (read-only for now, shown for reference)
    st.markdown(f"**Analysetyp:** `{analysis.analysis_type}`")
    
    # Display help text for this analysis type
    render_analysis_help_text(analysis.analysis_type)
    
    st.markdown("---")
    
    # Render filters (common to all analyses)
    st.markdown("### 🔍 Filter")
    render_filter_controls(analysis, index)
    
    st.markdown("---")
    
    # Render analysis-specific parameters
    st.markdown("### ⚙️ Parameter")
    
    if analysis.analysis_type == 'netzwerk':
        render_network_parameters(analysis, index)
    elif analysis.analysis_type == 'heatmap':
        render_heatmap_parameters(analysis, index)
    elif analysis.analysis_type in ('summary_paraphrase', 'summary_reasoning', 'custom_summary'):
        render_summary_parameters(analysis, index)
    elif analysis.analysis_type == 'sentiment_analysis':
        render_sentiment_parameters(analysis, index)
    else:
        st.info(f"Keine spezifischen Parameter für Analysetyp '{analysis.analysis_type}'")


def render_analysis_help_text(analysis_type: str):
    """
    Renders explanatory text for analysis type.
    
    Args:
        analysis_type: Type of analysis
    
    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
    """
    help_texts = {
        'netzwerk': """
        **📊 Netzwerkanalyse**
        
        Erstellt ein Netzwerkdiagramm, das die Beziehungen zwischen Kategorien visualisiert.
        Knoten repräsentieren Kategorien, Kanten zeigen Co-Occurrences.
        
        **Parameter:**
        - **Node Size Factor**: Skalierungsfaktor für Knotengröße
        - **Layout Iterations**: Anzahl der Iterationen für Layout-Algorithmus
        - **Gravity**: Anziehungskraft zwischen Knoten
        - **Scaling**: Gesamtskalierung des Graphen
        """,
        'heatmap': """
        **🔥 Heatmap-Analyse**
        
        Erstellt eine Heatmap zur Visualisierung von Häufigkeiten oder Werten über zwei Dimensionen.
        
        **Parameter:**
        - **X/Y/Z Attribute**: Spalten für X-Achse, Y-Achse und Werte
        - **Colormap**: Farbschema für die Heatmap
        - **Figure Size**: Größe der Grafik
        - **Annotations**: Werte in Zellen anzeigen
        """,
        'summary_paraphrase': """
        **📝 Zusammenfassung (Paraphrase)**
        
        Erstellt eine KI-generierte Zusammenfassung der Paraphrasen.
        Verwendet einen vordefinierten Prompt für Paraphrasierung.
        
        **Parameter:**
        - **Text Column**: Spalte mit zu zusammenfassendem Text
        - **Prompt Template**: Optional eigener Prompt (leer = Standard)
        """,
        'summary_reasoning': """
        **🧠 Zusammenfassung (Reasoning)**
        
        Erstellt eine KI-generierte Zusammenfassung der Reasoning-Texte.
        Verwendet einen vordefinierten Prompt für Reasoning-Analyse.
        
        **Parameter:**
        - **Text Column**: Spalte mit zu zusammenfassendem Text
        - **Prompt Template**: Optional eigener Prompt (leer = Standard)
        """,
        'custom_summary': """
        **✏️ Benutzerdefinierte Zusammenfassung**
        
        Erstellt eine KI-generierte Zusammenfassung mit eigenem Prompt.
        Ermöglicht vollständige Kontrolle über die Zusammenfassungslogik.
        
        **Parameter:**
        - **Text Column**: Spalte mit zu zusammenfassendem Text
        - **Prompt Template**: Eigener Prompt (erforderlich)
        """,
        'sentiment_analysis': """
        **😊 Sentiment-Analyse**
        
        Analysiert die Stimmung in Texten und erstellt Visualisierungen.
        Kategorisiert Texte nach Sentiment-Kategorien.
        
        **Parameter:**
        - **Text Column**: Spalte mit zu analysierendem Text
        - **Sentiment Categories**: Liste der Sentiment-Kategorien
        - **Temperature**: LLM-Temperatur für Analyse
        - **Chart Title**: Titel für Diagramm
        """
    }
    
    help_text = help_texts.get(analysis_type, "Keine Hilfe verfügbar für diesen Analysetyp.")
    st.info(help_text)


def render_filter_controls(analysis: AnalysisConfig, index: int):
    """
    Renders filter controls (common to all analyses).
    
    Args:
        analysis: AnalysisConfig to render filters for
        index: Index of the analysis
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
    """
    st.markdown("Filtern Sie die Daten vor der Analyse:")
    
    filters = analysis.filters.copy()
    updated = False
    
    # Document filter
    new_dokument = st.text_input(
        "Dokument:",
        value=filters.get('Dokument') or '',
        key=f"filter_dokument_{index}",
        help="Leer lassen für alle Dokumente"
    )
    if (new_dokument or None) != filters.get('Dokument'):
        filters['Dokument'] = new_dokument if new_dokument else None
        updated = True
    
    # Main category filter
    new_hauptkat = st.text_input(
        "Hauptkategorie:",
        value=filters.get('Hauptkategorie') or '',
        key=f"filter_hauptkat_{index}",
        help="Leer lassen für alle Hauptkategorien"
    )
    if (new_hauptkat or None) != filters.get('Hauptkategorie'):
        filters['Hauptkategorie'] = new_hauptkat if new_hauptkat else None
        updated = True
    
    # Subcategories filter
    new_subkat = st.text_input(
        "Subkategorien:",
        value=filters.get('Subkategorien') or '',
        key=f"filter_subkat_{index}",
        help="Leer lassen für alle Subkategorien"
    )
    if (new_subkat or None) != filters.get('Subkategorien'):
        filters['Subkategorien'] = new_subkat if new_subkat else None
        updated = True
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Attribute 1 filter
        new_attr1 = st.text_input(
            "Attribut 1:",
            value=filters.get('Attribut_1') or '',
            key=f"filter_attr1_{index}",
            help="Leer lassen für alle Werte"
        )
        if (new_attr1 or None) != filters.get('Attribut_1'):
            filters['Attribut_1'] = new_attr1 if new_attr1 else None
            updated = True
    
    with col2:
        # Attribute 2 filter
        new_attr2 = st.text_input(
            "Attribut 2:",
            value=filters.get('Attribut_2') or '',
            key=f"filter_attr2_{index}",
            help="Leer lassen für alle Werte"
        )
        if (new_attr2 or None) != filters.get('Attribut_2'):
            filters['Attribut_2'] = new_attr2 if new_attr2 else None
            updated = True
    
    if updated:
        handle_update_analysis(index, {'filters': filters})


def render_network_parameters(analysis: AnalysisConfig, index: int):
    """
    Renders network-specific parameters.
    
    Args:
        analysis: AnalysisConfig to render parameters for
        index: Index of the analysis
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
    """
    params = analysis.params.copy()
    updated = False
    
    # Node size factor
    new_node_size = st.number_input(
        "Node Size Factor:",
        min_value=0.1,
        max_value=10.0,
        value=float(params.get('node_size_factor', 1.0)),
        step=0.1,
        key=f"node_size_{index}",
        help="Skalierungsfaktor für Knotengröße"
    )
    if new_node_size != params.get('node_size_factor'):
        params['node_size_factor'] = new_node_size
        updated = True
    
    # Layout iterations
    new_iterations = st.number_input(
        "Layout Iterations:",
        min_value=1,
        max_value=500,
        value=int(params.get('layout_iterations', 50)),
        step=10,
        key=f"iterations_{index}",
        help="Anzahl der Iterationen für Layout-Berechnung"
    )
    if new_iterations != params.get('layout_iterations'):
        params['layout_iterations'] = new_iterations
        updated = True
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gravity
        new_gravity = st.number_input(
            "Gravity:",
            min_value=-1.0,
            max_value=1.0,
            value=float(params.get('gravity', 0.1)),
            step=0.05,
            key=f"gravity_{index}",
            help="Anziehungskraft zwischen Knoten"
        )
        if new_gravity != params.get('gravity'):
            params['gravity'] = new_gravity
            updated = True
    
    with col2:
        # Scaling
        new_scaling = st.number_input(
            "Scaling:",
            min_value=0.1,
            max_value=10.0,
            value=float(params.get('scaling', 1.0)),
            step=0.1,
            key=f"scaling_{index}",
            help="Gesamtskalierung des Graphen"
        )
        if new_scaling != params.get('scaling'):
            params['scaling'] = new_scaling
            updated = True
    
    if updated:
        handle_update_analysis(index, {'params': params})


def render_heatmap_parameters(analysis: AnalysisConfig, index: int):
    """
    Renders heatmap-specific parameters.
    
    Args:
        analysis: AnalysisConfig to render parameters for
        index: Index of the analysis
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10
    """
    params = analysis.params.copy()
    updated = False
    
    # X, Y, Z attributes
    new_x = st.text_input(
        "X-Achse Attribut:",
        value=params.get('x_attribute', 'Hauptkategorie'),
        key=f"x_attr_{index}",
        help="Spaltenname für X-Achse"
    )
    if new_x != params.get('x_attribute'):
        params['x_attribute'] = new_x
        updated = True
    
    new_y = st.text_input(
        "Y-Achse Attribut:",
        value=params.get('y_attribute', 'Subkategorien'),
        key=f"y_attr_{index}",
        help="Spaltenname für Y-Achse"
    )
    if new_y != params.get('y_attribute'):
        params['y_attribute'] = new_y
        updated = True
    
    new_z = st.text_input(
        "Werte Attribut:",
        value=params.get('z_attribute', 'count'),
        key=f"z_attr_{index}",
        help="Spaltenname für Werte (z.B. 'count' für Häufigkeiten)"
    )
    if new_z != params.get('z_attribute'):
        params['z_attribute'] = new_z
        updated = True
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Colormap
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'RdYlGn']
        current_cmap = params.get('cmap', 'viridis')
        cmap_index = colormaps.index(current_cmap) if current_cmap in colormaps else 0
        
        new_cmap = st.selectbox(
            "Farbschema:",
            options=colormaps,
            index=cmap_index,
            key=f"cmap_{index}"
        )
        if new_cmap != params.get('cmap'):
            params['cmap'] = new_cmap
            updated = True
    
    with col2:
        # Annotations
        new_annot = st.checkbox(
            "Werte anzeigen",
            value=params.get('annot', True),
            key=f"annot_{index}",
            help="Werte in Heatmap-Zellen anzeigen"
        )
        if new_annot != params.get('annot'):
            params['annot'] = new_annot
            updated = True
    
    # Figure size
    figsize = params.get('figsize', [10, 8])
    col1, col2 = st.columns(2)
    
    with col1:
        new_width = st.number_input(
            "Breite:",
            min_value=5,
            max_value=30,
            value=int(figsize[0]),
            key=f"width_{index}"
        )
    
    with col2:
        new_height = st.number_input(
            "Höhe:",
            min_value=5,
            max_value=30,
            value=int(figsize[1]),
            key=f"height_{index}"
        )
    
    if [new_width, new_height] != figsize:
        params['figsize'] = [new_width, new_height]
        updated = True
    
    # Format string
    new_fmt = st.text_input(
        "Zahlenformat:",
        value=params.get('fmt', '.2f'),
        key=f"fmt_{index}",
        help="Format für Zahlendarstellung (z.B. '.2f' für 2 Dezimalstellen)"
    )
    if new_fmt != params.get('fmt'):
        params['fmt'] = new_fmt
        updated = True
    
    if updated:
        handle_update_analysis(index, {'params': params})


def render_summary_parameters(analysis: AnalysisConfig, index: int):
    """
    Renders summary-specific parameters.
    
    Args:
        analysis: AnalysisConfig to render parameters for
        index: Index of the analysis
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 5.12
    """
    params = analysis.params.copy()
    updated = False
    
    # Text column
    new_text_col = st.text_input(
        "Text-Spalte:",
        value=params.get('text_column', 'Paraphrase'),
        key=f"text_col_{index}",
        help="Spalte mit zu zusammenfassendem Text"
    )
    if new_text_col != params.get('text_column'):
        params['text_column'] = new_text_col
        updated = True
    
    # Prompt template
    current_prompt = params.get('prompt_template', '')
    
    if analysis.analysis_type == 'custom_summary':
        st.markdown("**Eigener Prompt (erforderlich):**")
        new_prompt = st.text_area(
            "Prompt Template:",
            value=current_prompt,
            height=150,
            key=f"prompt_{index}",
            help="Definieren Sie Ihren eigenen Prompt für die Zusammenfassung"
        )
    else:
        st.markdown("**Eigener Prompt (optional):**")
        new_prompt = st.text_area(
            "Prompt Template:",
            value=current_prompt,
            height=150,
            key=f"prompt_{index}",
            help="Leer lassen für Standard-Prompt, oder eigenen Prompt eingeben"
        )
    
    if new_prompt != current_prompt:
        params['prompt_template'] = new_prompt
        updated = True
    
    if updated:
        handle_update_analysis(index, {'params': params})


def render_sentiment_parameters(analysis: AnalysisConfig, index: int):
    """
    Renders sentiment-specific parameters.
    
    Args:
        analysis: AnalysisConfig to render parameters for
        index: Index of the analysis
    
    Requirements: 6.1, 6.2, 6.3, 6.4
    """
    params = analysis.params.copy()
    updated = False
    
    # Text column
    new_text_col = st.text_input(
        "Text-Spalte:",
        value=params.get('text_column', 'Paraphrase'),
        key=f"sent_text_col_{index}",
        help="Spalte mit zu analysierendem Text"
    )
    if new_text_col != params.get('text_column'):
        params['text_column'] = new_text_col
        updated = True
    
    # Chart title
    new_title = st.text_input(
        "Diagramm-Titel:",
        value=params.get('chart_title', 'Sentiment Analysis'),
        key=f"chart_title_{index}"
    )
    if new_title != params.get('chart_title'):
        params['chart_title'] = new_title
        updated = True
    
    # Temperature
    new_temp = st.slider(
        "LLM Temperature:",
        min_value=0.0,
        max_value=2.0,
        value=float(params.get('temperature', 0.3)),
        step=0.1,
        key=f"sent_temp_{index}",
        help="Temperatur für LLM-Analyse (niedriger = konsistenter)"
    )
    if new_temp != params.get('temperature'):
        params['temperature'] = new_temp
        updated = True
    
    # Sentiment categories (simplified - show as text for now)
    categories = params.get('sentiment_categories', ['positive', 'neutral', 'negative'])
    categories_str = ', '.join(categories)
    
    new_categories_str = st.text_input(
        "Sentiment-Kategorien (kommagetrennt):",
        value=categories_str,
        key=f"sent_cats_{index}",
        help="Liste der Sentiment-Kategorien, z.B. 'positive, neutral, negative'"
    )
    
    if new_categories_str != categories_str:
        new_categories = [cat.strip() for cat in new_categories_str.split(',') if cat.strip()]
        params['sentiment_categories'] = new_categories
        updated = True
    
    # Figure size
    figsize = params.get('figsize', [10, 6])
    col1, col2 = st.columns(2)
    
    with col1:
        new_width = st.number_input(
            "Diagramm-Breite:",
            min_value=5,
            max_value=30,
            value=int(figsize[0]),
            key=f"sent_width_{index}"
        )
    
    with col2:
        new_height = st.number_input(
            "Diagramm-Höhe:",
            min_value=5,
            max_value=30,
            value=int(figsize[1]),
            key=f"sent_height_{index}"
        )
    
    if [new_width, new_height] != figsize:
        params['figsize'] = [new_width, new_height]
        updated = True
    
    if updated:
        handle_update_analysis(index, {'params': params})


def handle_add_analysis():
    """
    Handles adding new analysis.
    
    Requirement 7.1: WHEN the user clicks an "Add Analysis" button 
                    THEN the system SHALL create a new analysis configuration with default values
    """
    # Show dialog to select analysis type
    st.session_state.show_add_analysis_dialog = True


def handle_remove_analysis(index: int):
    """
    Handles removing analysis.
    
    Args:
        index: Index of analysis to remove
    
    Requirement 7.3: WHEN the user clicks a "Remove Analysis" button 
                    THEN the system SHALL delete that analysis configuration
    Requirement 2.3: Update session state when configuration changes
    """
    manager = st.session_state.explorer_config_manager
    config = st.session_state.explorer_config_data
    
    success, new_config, errors = manager.remove_analysis(config, index)
    
    if success and new_config:
        # Update session state
        st.session_state.explorer_config_data = new_config
        # Mark as modified
        st.session_state.explorer_config_modified = True
        st.success(f"✅ Analyse entfernt!")
        st.rerun()
    else:
        st.error(f"❌ Fehler beim Entfernen: {', '.join(errors)}")


def handle_update_analysis(index: int, updates: Dict[str, Any]):
    """
    Handles updating analysis configuration.
    
    Requirement 2.3: Update session state when configuration changes
    
    Args:
        index: Index of analysis to update
        updates: Dictionary of updates to apply
    """
    manager = st.session_state.explorer_config_manager
    config = st.session_state.explorer_config_data
    
    success, new_config, errors = manager.update_analysis(config, index, updates)
    
    if success and new_config:
        # Update session state
        st.session_state.explorer_config_data = new_config
        # Mark as modified for potential auto-save
        st.session_state.explorer_config_modified = True
    else:
        st.error(f"❌ Fehler beim Aktualisieren: {', '.join(errors)}")


def handle_reorder_analysis(from_index: int, to_index: int):
    """
    Handles reordering analysis (swap two analyses).
    
    Args:
        from_index: Current index
        to_index: Target index
    
    Requirement 7.4: WHEN the user reorders analysis tabs 
                    THEN the system SHALL update the configuration order accordingly
    Requirement 2.3: Update session state when configuration changes
    """
    config = st.session_state.explorer_config_data
    
    # Create new order by swapping
    new_order = list(range(len(config.analysis_configs)))
    new_order[from_index], new_order[to_index] = new_order[to_index], new_order[from_index]
    
    manager = st.session_state.explorer_config_manager
    success, new_config, errors = manager.reorder_analyses(config, new_order)
    
    if success and new_config:
        # Update session state
        st.session_state.explorer_config_data = new_config
        # Mark as modified
        st.session_state.explorer_config_modified = True
        st.rerun()
    else:
        st.error(f"❌ Fehler beim Neuordnen: {', '.join(errors)}")


def render_explorer_view():
    """Rendert die aktive Explorer-Ansicht mit Datenvisualisierung"""
    file_path = st.session_state.explorer_file_path
    
    # Header with back button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.subheader(f"📊 Analyse: {Path(file_path).name}")
    
    with col2:
        if st.button("◀ Zurück zur Konfiguration", key="back_to_config"):
            st.session_state.explorer_active = False
            st.rerun()
    
    st.markdown("---")
    
    # Load and display data
    try:
        import pandas as pd
        
        # Read Excel file
        with st.spinner("Lade Analysedaten..."):
            # Read main results sheet
            df = pd.read_excel(file_path, sheet_name='Kodierungsergebnisse')
        
        st.success(f"✅ {len(df)} Kodierungen geladen")
        
        # Apply filters
        filters = st.session_state.get('explorer_filters', {})
        min_confidence = filters.get('min_confidence', 0.0)
        
        if min_confidence > 0:
            if 'Konfidenz' in df.columns:
                df = df[df['Konfidenz'] >= min_confidence]
                st.info(f"🔍 Filter angewendet: Minimale Konfidenz {min_confidence:.2f} → {len(df)} Kodierungen")
        
        # Display visualizations
        config = st.session_state.explorer_config
        
        if 'category_distribution' in config.enabled_charts:
            st.subheader("📊 Kategorieverteilung")
            if 'Hauptkategorie' in df.columns:
                category_counts = df['Hauptkategorie'].value_counts()
                st.bar_chart(category_counts)
            else:
                st.warning("Spalte 'Hauptkategorie' nicht gefunden")
        
        if 'confidence_histogram' in config.enabled_charts:
            st.subheader("📈 Konfidenz-Verteilung")
            if 'Konfidenz' in df.columns:
                # Create histogram using pandas
                hist_data = df['Konfidenz'].value_counts(bins=20, sort=False).sort_index()
                st.bar_chart(hist_data)
            else:
                st.warning("Spalte 'Konfidenz' nicht gefunden")
        
        # Show statistics
        if st.session_state.get('show_statistics', True):
            st.subheader("📊 Statistiken")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Gesamt Kodierungen", len(df))
            
            with col2:
                if 'Hauptkategorie' in df.columns:
                    st.metric("Kategorien", df['Hauptkategorie'].nunique())
            
            with col3:
                if 'Dokument' in df.columns:
                    st.metric("Dokumente", df['Dokument'].nunique())
        
        # Data table
        st.subheader("📋 Kodierungsdaten")
        
        # Column selection
        if len(df.columns) > 10:
            display_cols = st.multiselect(
                "Anzuzeigende Spalten:",
                options=df.columns.tolist(),
                default=df.columns.tolist()[:5],
                key="display_columns"
            )
            if display_cols:
                st.dataframe(df[display_cols], use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
        
        # Export options
        st.markdown("---")
        st.subheader("💾 Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download filtered data
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Gefilterte Daten als CSV",
                data=csv,
                file_name=f"filtered_{Path(file_path).stem}.csv",
                mime="text/csv",
                key="download_csv"
            )
        
        with col2:
            # Download as JSON
            json_str = df.to_json(orient='records', force_ascii=False, indent=2)
            st.download_button(
                label="📥 Gefilterte Daten als JSON",
                data=json_str,
                file_name=f"filtered_{Path(file_path).stem}.json",
                mime="application/json",
                key="download_json"
            )
        
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Daten: {str(e)}")
        import traceback
        with st.expander("🔍 Fehlerdetails"):
            st.code(traceback.format_exc())


def render_file_selector():
    """
    Rendert Dateiauswahl für Analyseergebnisse.
    Returns: Pfad zur ausgewählten Datei oder None
    """
    config = st.session_state.config_data
    file_manager = st.session_state.file_manager
    project_root = st.session_state.project_manager.get_root_directory()
    
    output_dir = config.output_dir
    if not Path(output_dir).is_absolute():
        output_dir = str(project_root / output_dir)
    
    try:
        file_manager.ensure_directory(output_dir)
        files = file_manager.list_files(directory=output_dir, extensions=['.xlsx'])
        
        if not files:
            return None
        
        # Filter for analysis files
        analysis_files = [f for f in files if 'QCA-AID_Analysis' in f.name]
        
        if not analysis_files:
            st.warning("Keine QCA-AID Analysedateien gefunden")
            return None
        
        # Sort by modification time (newest first)
        analysis_files.sort(key=lambda x: x.modified, reverse=True)
        
        # Create selection
        file_options = {f"{f.name} ({f.format_size()}, {f.format_date()})": f.path for f in analysis_files}
        
        selected = st.selectbox(
            "Wählen Sie eine Analysedatei:",
            options=list(file_options.keys()),
            key="selected_analysis_file"
        )
        
        if selected:
            return file_options[selected]
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Dateien: {str(e)}")
    
    return None


def render_visualization_config():
    """Rendert Visualisierungs-Konfiguration"""
    if st.session_state.explorer_config is None:
        st.session_state.explorer_config = ExplorerConfig.create_default()
    
    config = st.session_state.explorer_config
    
    available_charts = [
        ('category_distribution', '📊 Kategorieverteilung'),
        ('confidence_histogram', '📈 Konfidenz-Histogramm'),
        ('coder_agreement', '🤝 Coder-Übereinstimmung'),
        ('temporal_analysis', '⏱️ Zeitliche Analyse'),
    ]
    
    for chart_id, chart_label in available_charts:
        is_enabled = chart_id in config.enabled_charts
        new_state = st.checkbox(chart_label, value=is_enabled, key=f"chart_{chart_id}")
        
        if new_state and chart_id not in config.enabled_charts:
            config.enabled_charts.append(chart_id)
        elif not new_state and chart_id in config.enabled_charts:
            config.enabled_charts.remove(chart_id)


def render_filter_config():
    """Rendert Filter-Konfiguration"""
    st.markdown("**Kategorien filtern:**")
    
    # Initialize filter state
    if 'explorer_filters' not in st.session_state:
        st.session_state.explorer_filters = {
            'categories': [],
            'min_confidence': 0.0,
            'documents': []
        }
    
    # Category filter
    filter_categories = st.multiselect(
        "Nur diese Kategorien anzeigen:",
        options=["Alle Kategorien"],  # TODO: Load from codebook
        default=[],
        key="filter_categories",
        help="Leer lassen für alle Kategorien"
    )
    
    # Confidence filter
    min_confidence = st.slider(
        "Minimale Konfidenz:",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key="filter_confidence",
        help="Nur Kodierungen mit mindestens dieser Konfidenz anzeigen"
    )
    
    st.session_state.explorer_filters['min_confidence'] = min_confidence
    
    # Show statistics
    show_stats = st.checkbox(
        "Statistiken anzeigen",
        value=True,
        key="show_statistics"
    )


def render_advanced_config():
    """Rendert erweiterte Konfiguration"""
    if st.session_state.explorer_config is None:
        st.session_state.explorer_config = ExplorerConfig.create_default()
    
    config = st.session_state.explorer_config
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Color scheme
        color_schemes = ['default', 'dark', 'light', 'colorblind']
        current_idx = color_schemes.index(config.color_scheme) if config.color_scheme in color_schemes else 0
        
        new_scheme = st.selectbox(
            "Farbschema:",
            options=color_schemes,
            index=current_idx,
            format_func=lambda x: {
                'default': 'Standard',
                'dark': 'Dunkel',
                'light': 'Hell',
                'colorblind': 'Farbenblind-freundlich'
            }.get(x, x),
            key="color_scheme_select"
        )
        config.color_scheme = new_scheme
    
    with col2:
        # Export format
        export_formats = ['xlsx', 'csv', 'json', 'html']
        current_idx = export_formats.index(config.export_format) if config.export_format in export_formats else 0
        
        new_format = st.selectbox(
            "Export-Format:",
            options=export_formats,
            index=current_idx,
            format_func=lambda x: x.upper(),
            key="export_format_select"
        )
        config.export_format = new_format
    
    # Custom prompt
    st.markdown("**🤖 KI-Analyse Prompt (Optional):**")
    custom_prompt = st.text_area(
        "Eigener Prompt für KI-gestützte Analyse:",
        value="",
        height=100,
        placeholder="Z.B.: Analysiere die Häufigkeit von Kategorien und identifiziere Muster...",
        key="custom_analysis_prompt",
        help="Leer lassen für Standard-Analyse"
    )
    
    if 'custom_prompt' not in st.session_state:
        st.session_state.custom_prompt = ""
    st.session_state.custom_prompt = custom_prompt


def render_explorer_controls(selected_file: str):
    """
    Rendert Explorer-Steuerung mit Start-Button.
    
    Requirements: 1.5, 7.5
    """
    # Handle add analysis dialog
    if st.session_state.get('show_add_analysis_dialog', False):
        render_add_analysis_dialog()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.info(f"📄 Ausgewählte Datei: {Path(selected_file).name}")
    
    with col2:
        # Save config button
        if st.button("💾 Config speichern", use_container_width=True, key="save_explorer_config_btn"):
            save_explorer_config_new()
    
    with col3:
        # Start explorer button
        if st.button("▶️ Explorer starten", use_container_width=True, type="primary", key="start_explorer_btn"):
            start_explorer_analysis(selected_file)
    
    # Display log outside of columns (full width)
    if st.session_state.get('explorer_analysis_log'):
        st.markdown("---")
        log_data = st.session_state.explorer_analysis_log
        st.text_area("📋 Analyse-Log", log_data['log_text'], height=300, key="explorer_log_display")
        
        if log_data.get('success'):
            st.success("✅ Alle Analysen erfolgreich abgeschlossen!")
            if log_data.get('output_dir'):
                st.info(f"📁 Ergebnisse gespeichert in: {log_data['output_dir']}")
        else:
            st.warning("⚠️ Analysen abgeschlossen mit Warnungen. Siehe Log für Details.")


def render_add_analysis_dialog():
    """
    Renders dialog for adding new analysis.
    
    Requirement 7.1, 7.2: Add analysis with type selection
    """
    st.markdown("---")
    st.subheader("➕ Neue Analyse hinzufügen")
    
    # Analysis type selection
    analysis_types = {
        'netzwerk': '📊 Netzwerkanalyse',
        'heatmap': '🔥 Heatmap',
        'summary_paraphrase': '📝 Zusammenfassung (Paraphrase)',
        'summary_reasoning': '🧠 Zusammenfassung (Reasoning)',
        'custom_summary': '✏️ Benutzerdefinierte Zusammenfassung',
        'sentiment_analysis': '😊 Sentiment-Analyse'
    }
    
    selected_type = st.selectbox(
        "Analysetyp auswählen:",
        options=list(analysis_types.keys()),
        format_func=lambda x: analysis_types[x],
        key="new_analysis_type"
    )
    
    # Analysis name
    default_name = analysis_types[selected_type].split(' ', 1)[1]  # Remove emoji
    analysis_name = st.text_input(
        "Analysename:",
        value=default_name,
        key="new_analysis_name"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Hinzufügen", use_container_width=True, type="primary"):
            # Create new analysis
            manager = st.session_state.explorer_config_manager
            new_analysis = manager.add_analysis(selected_type, analysis_name)
            
            # Add to config
            config = st.session_state.explorer_config_data
            config.analysis_configs.append(new_analysis)
            
            # Update session state
            st.session_state.explorer_config_data = config
            # Mark as modified
            st.session_state.explorer_config_modified = True
            
            # Close dialog
            st.session_state.show_add_analysis_dialog = False
            st.success(f"✅ Analyse '{analysis_name}' hinzugefügt!")
            st.rerun()
    
    with col2:
        if st.button("❌ Abbrechen", use_container_width=True):
            st.session_state.show_add_analysis_dialog = False
            st.rerun()
    
    st.markdown("---")


def save_explorer_config_new():
    """
    Saves explorer configuration using ExplorerConfigManager.
    
    Requirement 1.3: WHEN the user saves the Explorer-Config 
                    THEN the system SHALL transfer current LLM settings from config_data to base_config
    Requirement 1.5: WHEN the user saves the configuration 
                    THEN the system SHALL persist all analysis configurations to the config file
    Requirement 2.3: Update session state when configuration changes
    Requirement 3.3: WHEN save_explorer_config_new() is called 
                    THEN the system SHALL copy LLM settings from config_data to explorer_config_data.base_config
    Requirement 3.4: WHEN the Explorer-Config is saved 
                    THEN the system SHALL persist provider and model in base_config
    """
    manager = st.session_state.explorer_config_manager
    config = st.session_state.explorer_config_data
    config_data = st.session_state.config_data
    
    # Requirement 3.3: Copy LLM settings from config_data to base_config before saving
    # Requirement 1.3: Transfer current LLM settings
    config.base_config['provider'] = config_data.model_provider
    config.base_config['model'] = config_data.model_name
    
    # Save to both JSON and XLSX
    # Requirement 3.4: Ensure values are persisted in both formats
    success_json, errors_json = manager.save_config(config, format='json')
    success_xlsx, errors_xlsx = manager.save_config(config, format='xlsx')
    
    if success_json and success_xlsx:
        st.success("✅ Konfiguration erfolgreich gespeichert (JSON & XLSX)!")
        # Clear modified flag
        st.session_state.explorer_config_modified = False
    elif success_json:
        st.warning(f"⚠️ JSON gespeichert, aber XLSX-Fehler: {', '.join(errors_xlsx)}")
        # Partial success - clear modified flag
        st.session_state.explorer_config_modified = False
    elif success_xlsx:
        st.warning(f"⚠️ XLSX gespeichert, aber JSON-Fehler: {', '.join(errors_json)}")
        # Partial success - clear modified flag
        st.session_state.explorer_config_modified = False
    else:
        st.error(f"❌ Fehler beim Speichern: JSON: {', '.join(errors_json)}, XLSX: {', '.join(errors_xlsx)}")


def start_explorer_analysis(file_path: str):
    """
    Startet Explorer-Analyse mit konfigurierten Analysen.
    
    Requirements:
    - 1.3: Apply filters to data before running specific analysis
    - 1.4: Use configured parameters when executing analysis
    - 2.1: Execute analysis when active parameter is true
    - 2.2: Skip analysis when active parameter is false
    """
    # Get explorer config
    config_data = st.session_state.explorer_config_data
    
    # Validate that we have at least one active analysis
    active_analyses = [ac for ac in config_data.analysis_configs if ac.active]
    
    if not active_analyses:
        st.warning("⚠️ Keine aktiven Analysen konfiguriert. Bitte aktivieren Sie mindestens eine Analyse.")
        return
    
    # Show progress
    with st.spinner("🔄 Führe Analysen durch..."):
        progress_placeholder = st.empty()
        
        try:
            # Create analysis runner
            runner = ExplorerAnalysisRunner(file_path, config_data)
            
            # Initialize
            progress_placeholder.info("Initialisiere Analyzer...")
            success, init_messages = asyncio.run(runner.initialize())
            
            if not success:
                st.error("❌ Fehler bei Initialisierung:")
                for msg in init_messages:
                    st.error(f"  - {msg}")
                # Store error in session state
                st.session_state.explorer_analysis_log = {
                    'log_text': "\n".join(init_messages),
                    'success': False,
                    'output_dir': None
                }
                st.rerun()
                return
            
            # Run all analyses
            progress_placeholder.info("Führe Analysen durch...")
            success, analysis_messages = asyncio.run(runner.run_all_analyses())
            
            # Combine all messages
            log_text = "\n".join(init_messages + analysis_messages)
            
            # Store log in session state for display outside columns
            st.session_state.explorer_analysis_log = {
                'log_text': log_text,
                'success': success,
                'output_dir': str(runner.analyzer.output_dir) if success else None
            }
            
            if success:
                # Store results in session state for viewing
                st.session_state.explorer_active = True
                st.session_state.explorer_file_path = file_path
                st.session_state.explorer_output_dir = str(runner.analyzer.output_dir)
            
            # Trigger rerun to display log outside columns
            st.rerun()
                
        except Exception as e:
            error_msg = f"❌ Fehler bei Analyse-Ausführung: {str(e)}"
            st.error(error_msg)
            import traceback
            error_trace = traceback.format_exc()
            
            # Store error in session state
            st.session_state.explorer_analysis_log = {
                'log_text': f"{error_msg}\n\n{error_trace}",
                'success': False,
                'output_dir': None
            }
            
            with st.expander("🔍 Fehlerdetails"):
                st.code(error_trace)


def render_output_files():
    """
    Zeigt verfügbare Output-XLSX-Dateien an.
    
    Requirement 7.1: WHEN der Explorer-Reiter angezeigt wird 
                    THEN das System SHALL alle XLSX-Dateien im OUTPUT_DIR auflisten
    Requirement 7.2: WHEN Output-Dateien angezeigt werden 
                    THEN das System SHALL Dateiname, Größe und Änderungsdatum anzeigen
    Requirement 7.4: WHEN keine Output-Dateien vorhanden sind 
                    THEN das System SHALL eine Meldung "Keine Analyseergebnisse gefunden" anzeigen
    Requirement 7.5: WHEN der OUTPUT_DIR nicht existiert 
                    THEN das System SHALL den Ordner automatisch erstellen
    """
    config = st.session_state.config_data
    output_dir = config.output_dir
    file_manager = st.session_state.file_manager
    
    # Get project root from project manager
    project_manager = st.session_state.project_manager
    project_root = project_manager.get_root_directory()
    
    # Resolve output_dir relative to project root if it's a relative path
    if not Path(output_dir).is_absolute():
        output_dir = str(project_root / output_dir)
    
    # Ensure output directory exists
    try:
        file_manager.ensure_directory(output_dir)
    except Exception as e:
        st.error(f"❌ Fehler beim Erstellen des Output-Verzeichnisses: {str(e)}")
        return
    
    # List XLSX files
    try:
        files = file_manager.list_files(
            directory=output_dir,
            extensions=['.xlsx']
        )
        
        if not files:
            st.info("📭 Keine Analyseergebnisse gefunden")
            st.markdown("""
            Führen Sie eine Analyse im **Analyse**-Tab durch, 
            um Ergebnisse zu generieren.
            """)
            return
        
        # Performance Optimization: Pagination for long file lists
        total_files = len(files)
        st.markdown(f"**{total_files} Datei(en) gefunden**")
        
        # Pagination controls
        if total_files > st.session_state.files_per_page:
            # Get current page
            current_page = st.session_state.get('output_files_page', 1)
            
            # Paginate files
            paginated_files, total_pages, _ = file_manager.paginate_files(
                files,
                page=current_page,
                page_size=st.session_state.files_per_page
            )
            
            # Pagination UI
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("◀ Zurück", disabled=(current_page <= 1), key="output_prev"):
                    st.session_state.output_files_page = max(1, current_page - 1)
                    st.rerun()
            
            with col2:
                st.markdown(f"<div style='text-align: center'>Seite {current_page} von {total_pages}</div>", 
                           unsafe_allow_html=True)
            
            with col3:
                if st.button("Weiter ▶", disabled=(current_page >= total_pages), key="output_next"):
                    st.session_state.output_files_page = min(total_pages, current_page + 1)
                    st.rerun()
            
            st.markdown("---")
            
            # Use paginated files
            files = paginated_files
        
        for file_info in files:
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # File name and metadata
                    st.markdown(f"**{file_info.name}**")
                    st.caption(f"📏 {file_info.format_size()} | 📅 {file_info.format_date()}")
                
                with col2:
                    # Select button
                    if st.button(
                        "👁️ Vorschau", 
                        key=f"select_{file_info.path}",
                        use_container_width=True
                    ):
                        st.session_state.selected_output_file = file_info.path
                        st.rerun()
                
                st.markdown("---")
        
    except Exception as e:
        st.error(f"❌ Fehler beim Auflisten der Dateien: {str(e)}")


def render_explorer_config():
    """
    Rendert Explorer-Konfiguration mit Checkboxen.
    
    Requirement 11.1: WHEN der Explorer-Reiter angezeigt wird 
                     THEN das System SHALL eine Schaltfläche "Explorer-Config laden" anzeigen
    Requirement 11.2: WHEN eine Explorer-Config geladen wird 
                     THEN das System SHALL alle Visualisierungseinstellungen in UI-Elemente übertragen
    Requirement 11.3: WHEN der Explorer-Reiter angezeigt wird 
                     THEN das System SHALL Checkboxen für aktivierte/deaktivierte Diagrammtypen anzeigen
    Requirement 11.5: WHEN Explorer-Einstellungen gespeichert werden 
                     THEN das System SHALL die Datei als QCA-AID-Explorer-Config.json speichern
    """
    # Initialize explorer config if not present
    if st.session_state.explorer_config is None:
        st.session_state.explorer_config = ExplorerConfig.create_default()
    
    config = st.session_state.explorer_config
    
    # File operations
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📂 Config laden", use_container_width=True):
            load_explorer_config()
    
    with col2:
        if st.button("💾 Config speichern", use_container_width=True):
            save_explorer_config()
    
    st.markdown("---")
    
    # Chart type checkboxes
    st.markdown("**Aktivierte Diagrammtypen:**")
    
    available_charts = [
        ('category_distribution', '📊 Kategorieverteilung'),
        ('confidence_histogram', '📈 Konfidenz-Histogramm'),
        ('coder_agreement', '🤝 Coder-Übereinstimmung'),
        ('temporal_analysis', '⏱️ Zeitliche Analyse'),
        ('network_graph', '🕸️ Netzwerk-Graph'),
        ('heatmap', '🔥 Heatmap')
    ]
    
    for chart_id, chart_label in available_charts:
        is_enabled = chart_id in config.enabled_charts
        
        new_state = st.checkbox(
            chart_label,
            value=is_enabled,
            key=f"chart_{chart_id}"
        )
        
        # Update config
        if new_state and chart_id not in config.enabled_charts:
            config.enabled_charts.append(chart_id)
        elif not new_state and chart_id in config.enabled_charts:
            config.enabled_charts.remove(chart_id)
    
    st.markdown("---")
    
    # Other settings
    st.markdown("**Weitere Einstellungen:**")
    
    # Color scheme
    color_schemes = ['default', 'dark', 'light', 'colorblind']
    current_scheme_idx = color_schemes.index(config.color_scheme) if config.color_scheme in color_schemes else 0
    
    new_scheme = st.selectbox(
        "Farbschema",
        options=color_schemes,
        index=current_scheme_idx,
        format_func=lambda x: {
            'default': 'Standard',
            'dark': 'Dunkel',
            'light': 'Hell',
            'colorblind': 'Farbenblind-freundlich'
        }.get(x, x)
    )
    
    if new_scheme != config.color_scheme:
        config.color_scheme = new_scheme
    
    # Show statistics
    new_show_stats = st.checkbox(
        "Statistiken anzeigen",
        value=config.show_statistics
    )
    
    if new_show_stats != config.show_statistics:
        config.show_statistics = new_show_stats
    
    # Export format
    export_formats = ['xlsx', 'csv', 'json', 'html']
    current_format_idx = export_formats.index(config.export_format) if config.export_format in export_formats else 0
    
    new_format = st.selectbox(
        "Export-Format",
        options=export_formats,
        index=current_format_idx,
        format_func=lambda x: x.upper()
    )
    
    if new_format != config.export_format:
        config.export_format = new_format
    
    # Preview
    st.markdown("---")
    render_config_preview()


def render_file_preview(file_path: str):
    """
    Zeigt Vorschau einer Output-Datei mit Sheet-Übersicht.
    
    Requirement 7.3: WHEN ein Benutzer eine Output-Datei auswählt 
                    THEN das System SHALL eine Vorschau der enthaltenen Sheets anzeigen
    Requirement 11.4: WHEN ein Benutzer Explorer-Einstellungen ändert 
                     THEN das System SHALL eine Vorschau der Konfiguration anzeigen
    """
    st.subheader("📄 Dateivorschau")
    
    try:
        # Get file info
        file_manager = st.session_state.file_manager
        file_info = file_manager.get_file_info(file_path)
        
        # Display file metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dateiname", file_info.name)
        
        with col2:
            st.metric("Größe", file_info.format_size())
        
        with col3:
            st.metric("Geändert", file_info.format_date())
        
        st.markdown("---")
        
        # Read Excel file to get sheet names
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            st.markdown(f"**{len(sheet_names)} Sheet(s) gefunden:**")
            
            # Display sheets in expandable sections
            for sheet_name in sheet_names:
                with st.expander(f"📋 {sheet_name}"):
                    try:
                        # Read first few rows
                        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)
                        
                        st.markdown(f"**Spalten:** {', '.join(df.columns.tolist())}")
                        st.markdown(f"**Zeilen (gesamt):** {len(pd.read_excel(file_path, sheet_name=sheet_name))}")
                        
                        st.markdown("**Vorschau (erste 5 Zeilen):**")
                        st.dataframe(df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Fehler beim Lesen des Sheets: {str(e)}")
            
            # Clear selection button
            if st.button("❌ Vorschau schließen", use_container_width=True):
                st.session_state.selected_output_file = None
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Fehler beim Lesen der Excel-Datei: {str(e)}")
            
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Dateiinformationen: {str(e)}")


def render_config_preview():
    """
    Zeigt Vorschau der aktuellen Explorer-Konfiguration.
    
    Requirement 11.4: WHEN ein Benutzer Explorer-Einstellungen ändert 
                     THEN das System SHALL eine Vorschau der Konfiguration anzeigen
    """
    config = st.session_state.explorer_config
    
    st.markdown("**Konfigurationsvorschau:**")
    
    # Create preview dict
    preview_dict = config.to_dict()
    
    # Display as formatted JSON
    st.code(json.dumps(preview_dict, indent=2, ensure_ascii=False), language='json')
    
    # Validation status
    is_valid, errors = config.validate()
    
    if is_valid:
        st.success("✅ Konfiguration ist gültig")
    else:
        st.error("❌ Konfiguration enthält Fehler:")
        for error in errors:
            st.error(f"  • {error}")


def load_explorer_config():
    """
    Lädt Explorer-Konfiguration aus Datei.
    
    Requirement 11.2: WHEN eine Explorer-Config geladen wird 
                     THEN das System SHALL alle Visualisierungseinstellungen in UI-Elemente übertragen
    """
    config_path = Path("QCA-AID-Explorer-Config.json")
    
    try:
        if not config_path.exists():
            st.warning("⚠️ Keine Explorer-Config gefunden. Verwende Standard-Konfiguration.")
            st.session_state.explorer_config = ExplorerConfig.create_default()
            return
        
        # Load JSON
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Create ExplorerConfig from dict
        st.session_state.explorer_config = ExplorerConfig.from_dict(config_data)
        
        st.success("✅ Explorer-Konfiguration erfolgreich geladen!")
        st.rerun()
        
    except json.JSONDecodeError as e:
        st.error(f"❌ Fehler beim Parsen der JSON-Datei: {str(e)}")
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Konfiguration: {str(e)}")


def save_explorer_config():
    """
    Speichert Explorer-Konfiguration in Datei.
    
    Requirement 11.5: WHEN Explorer-Einstellungen gespeichert werden 
                     THEN das System SHALL die Datei als QCA-AID-Explorer-Config.json speichern
    """
    config = st.session_state.explorer_config
    config_path = Path("QCA-AID-Explorer-Config.json")
    
    try:
        # Validate before saving
        is_valid, errors = config.validate()
        
        if not is_valid:
            st.error("❌ Konfiguration ist ungültig:")
            for error in errors:
                st.error(f"  • {error}")
            return
        
        # Convert to dict
        config_dict = config.to_dict()
        
        # Save as JSON
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        st.success(f"✅ Explorer-Konfiguration gespeichert: {config_path}")
        
    except Exception as e:
        st.error(f"❌ Fehler beim Speichern der Konfiguration: {str(e)}")
