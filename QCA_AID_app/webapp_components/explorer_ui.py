"""
Explorer UI Component
=====================
Streamlit UI component for managing QCA-AID Explorer configuration and output files.

Requirements: 1.2, 7.1, 7.2, 7.3, 7.4, 11.1, 11.2, 11.3, 11.4, 11.5
"""

import streamlit as st
import json
import asyncio
import os
import subprocess
import platform
from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd

from webapp_models.file_info import ExplorerConfig
from webapp_models.explorer_config_data import ExplorerConfigData, AnalysisConfig
from webapp_logic.explorer_config_manager import ExplorerConfigManager
from webapp_logic.explorer_analysis_runner import ExplorerAnalysisRunner
from webapp_components.config_ui import check_api_key_for_provider
from webapp_components.pagination import render_pagination_controls
from webapp_components.dialog_helpers import (
    build_start_summary,
    confirm_explorer_start,
    confirm_removal,
)


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
    
    # FIX: Explorer hat sein eigenes LLM-Modell (unabhängig von Config UI)
    # Wenn noch nicht gesetzt, verwendet der Explorer seine Standard-Defaults (nicht config_data)
    current_provider = explorer_config.base_config.get('provider')
    current_model = explorer_config.base_config.get('model')
    
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
    
    # R14.1: Trennung transparent machen - der Explorer nutzt bewusst ein eigenes
    # Modell für Post-hoc-Analysen (Zusammenfassungen, Sentiment), nicht das Modell
    # der Hauptanalyse. Ohne Hinweis wirkt das wie ein Synchronisationsfehler.
    st.caption(
        "ℹ️ Der Explorer verwendet ein **eigenes Modell** für Post-hoc-Analysen "
        "(Zusammenfassungen, Sentiment) - unabhängig vom Modell der Hauptanalyse."
    )
    
    # R14.2: Abweichung sichtbar machen + bequeme Übernahme
    main_provider = getattr(config, "model_provider", None)
    main_model = getattr(config, "model_name", None)
    if (
        main_provider
        and main_model
        and (new_provider_id != main_provider or new_model != main_model)
    ):
        col_main1, col_main2 = st.columns([2, 1])
        with col_main1:
            st.info(f"🤖 Hauptanalyse nutzt: `{main_provider} / {main_model}`")
        with col_main2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "📋 Von Hauptanalyse übernehmen",
                use_container_width=True,
                key="explorer_model_from_main",
            ):
                # R14.3: Kopiert in explorer_config_data.base_config; setzt beide
                # Modified-Flags (Bidirektional-Sync-Semantik bleibt unverändert)
                explorer_config.base_config["provider"] = main_provider
                explorer_config.base_config["model"] = main_model
                st.session_state.explorer_config_modified = True
                st.rerun()
    
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
            help="Harmonisiert ähnliche Schlüsselwörter automatisch, um Variationen zusammenzufassen. Beispiel: 'Nachhaltigkeit', 'nachhaltig' und 'Sustainability' werden als ein Konzept behandelt. Dies verbessert die Netzwerk-Visualisierungen und Analysen."
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
            help="Bestimmt, wie ähnlich Schlüsselwörter sein müssen, um harmonisiert zu werden. Höhere Werte (z.B. 0.85) = nur sehr ähnliche Wörter werden zusammengefasst. Niedrigere Werte (z.B. 0.65) = auch weniger ähnliche Wörter werden harmonisiert. Empfohlen: 0.70-0.75",
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
                    base_config = loaded_config_data.base_config
                    
                    # FIX: Validiere nur Provider/Modell im Explorer-Config, nicht in config_data
                    # Explorer hat sein eigenes LLM-Modell
                    if 'provider' in base_config and base_config['provider']:
                        # Validate provider
                        valid_providers = ['openai', 'anthropic', 'mistral', 'openrouter', 'local']
                        provider = base_config['provider'].lower()
                        if provider not in valid_providers:
                            st.warning(f"⚠️ Ungültiger Provider '{provider}' in Explorer-Config")
                    
                except Exception as e:
                    # Error handling for missing or invalid values
                    st.warning(f"⚠️ Fehler beim Validieren der Explorer-Einstellungen: {e}")
                
                st.success("✅ Konfiguration erfolgreich geladen")
                st.rerun()
            else:
                st.error(f"❌ Fehler beim Laden: {', '.join(errors)}")
    
    with col3:
        # R3.1: Genau EIN dominierender Speichern-Button; Format via Dropdown (R3.3)
        if st.button("💾 Speichern", type="primary", use_container_width=True, key="explorer_save_btn"):
            save_explorer_config_data(format=st.session_state.get("explorer_save_format", "both"))
    
    with col4:
        # R3.3: Format-Auswahl - Standard speichert beide Formate (R3.2)
        st.selectbox(
            "Format",
            options=["both", "json", "xlsx"],
            format_func={"both": "Beide (Standard)", "json": "Nur JSON", "xlsx": "Nur XLSX"}.get,
            key="explorer_save_format",
        )
    
    st.markdown("---")


def _load_categories_for_file(explorer_config_manager, selected_file: str):
    """Laedt Kategorien aus der gewaehlten Analysedatei (bei Dateiwechsel).

    Extrahiert aus dem bisherigen Flow von render_explorer_config_view;
    Verhalten und Meldungen bleiben unveraendert.
    """
    current_file = st.session_state.get('explorer_current_file')
    if current_file == selected_file:
        return explorer_config_manager.get_category_loader()

    try:
        from webapp_logic.category_loader import CategoryLoader
        explorer_config_manager.category_loader = CategoryLoader(selected_file)
        st.session_state.explorer_current_file = selected_file

        if explorer_config_manager.category_loader.is_loaded:
            stats = explorer_config_manager.category_loader.get_statistics()
            if stats['total_main_categories'] == 0:
                st.warning("⚠️ Kategorien-Sheet gefunden, aber keine Kategorien darin. Verwende Freitext-Eingabe.")
        else:
            st.warning("⚠️ Kein Kategorien-Sheet in der Analysedatei gefunden. Verwende Freitext-Eingabe.")
    except Exception as e:
        # Failed to load categories - that's OK, we'll use text input
        explorer_config_manager.category_loader = None
        st.session_state.explorer_current_file = selected_file
        st.warning(f"⚠️ Fehler beim Laden der Kategorien: {str(e)}. Verwende Freitext-Eingabe.")

    return explorer_config_manager.get_category_loader()


def render_explorer_config_view():
    """
    Rendert die Konfigurations-Ansicht im Kompaktmodus (R9).

    Kompaktbereich oben (Datei-Auswahl, Summary-Zeile, Start-Button),
    danach Modell-/Analysen-/Optionen-Bereiche als eingeklappte Expander.
    Bestehende Render-Funktionen und Widget-Keys bleiben unveraendert.

    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 6.1-6.5, 1.2, 7.1-7.4, 2.1, 2.5
    """
    # Session state is initialized in webapp.py initialize_session_state()
    # Display any load errors from initialization
    if st.session_state.get('explorer_config_load_errors'):
        st.error(f"⚠️ Fehler beim Laden der Konfiguration: {', '.join(st.session_state.explorer_config_load_errors)}")
        st.info("📁 Die Konfigurationsdatei hat möglicherweise ein altes Format. Erstellen Sie eine neue Konfiguration oder laden Sie eine gültige Datei.")
        # Clear errors after displaying
        del st.session_state.explorer_config_load_errors

    # FIX: Explorer hat sein eigenes LLM-Modell
    # Fallback zu config_data wurde entfernt - Explorer nutzt sein eigenes Default-Modell
    explorer_config_data = st.session_state.explorer_config_data

    # =============================================================
    # Kompaktbereich (R9.1): Datei-Auswahl, Summary-Zeile, Start
    # =============================================================
    st.subheader("🚀 Explorer starten")
    selected_file = render_file_selector()

    summary = build_start_summary(explorer_config_data, selected_file)

    col_file, col_analyses, col_model = st.columns(3)
    with col_file:
        st.caption(f"📄 Datei: {summary['file_name'] or '– keine gewählt –'}")
    with col_analyses:
        analyses_text = f"🔬 Analysen: {summary['active_count']} aktiv"
        if summary['inactive_count']:
            analyses_text += f", {summary['inactive_count']} inaktiv"
        st.caption(analyses_text)
    with col_model:
        st.caption(f"🤖 Modell: {summary['provider'] or '–'} / {summary['model'] or '–'}")

    # R9.5: Jump-Link zum Analyse-Tab, wenn keine Ergebnisse vorliegen
    if selected_file is None:
        st.warning("⚠️ **Keine Analyseergebnisse gefunden.** Führen Sie zuerst eine Analyse im Analyse-Tab durch.")
        if st.button("→ Zum Analyse-Tab", key="explorer_jump_to_analysis", type="primary"):
            st.session_state.active_tab_key = "analysis"
            st.session_state.active_tab = "Analyse"
            st.rerun()

    # R9.3: Start-Button - disabled mit Tooltip, wenn keine Datei gewaehlt
    start_disabled = selected_file is None
    if st.button(
        "▶️ Explorer starten",
        type="primary",
        key="start_explorer_btn",
        use_container_width=True,
        disabled=start_disabled,
        help="Bitte zuerst eine Analysedatei auswählen." if start_disabled else None,
    ):
        # R6: Start nur nach Bestaetigung (Zusammenfassung + Kosten-Hinweis)
        confirm_explorer_start(
            summary,
            on_confirm=lambda f=selected_file: start_explorer_analysis(f),
        )

    st.markdown("---")

    # =============================================================
    # Details als eingeklappte Expander (R9.1)
    # =============================================================
    with st.expander("🤖 Modell-Einstellungen", expanded=False):
        st.markdown("Wählen Sie das LLM-Modell für die Post-hoc-Analyse (Zusammenfassungen, Sentiment-Analyse, etc.)")
        render_explorer_model_settings()

    with st.expander("🔬 Analysen konfigurieren", expanded=False):
        # Kategorie-Informationen (bisher nach der Dateiauswahl)
        explorer_config_manager = st.session_state.get('explorer_config_manager')
        category_loader = None
        if selected_file and explorer_config_manager:
            category_loader = _load_categories_for_file(explorer_config_manager, selected_file)
        else:
            st.info("📁 Wählen Sie oben eine Analysedatei aus, um Filter und Analysen zu konfigurieren.")

        if category_loader and category_loader.is_loaded:
            stats = category_loader.get_statistics()
            if stats['total_main_categories'] > 0:
                with st.expander("📊 Verfügbare Kategorien", expanded=False):
                    from webapp_components.smart_filter_controls import render_category_statistics
                    render_category_statistics(category_loader)
            else:
                st.info("ℹ️ **Keine Kategorien verfügbar** - Filter verwenden Freitext-Eingabe. Das Kategorien-Sheet ist leer.")
        elif selected_file:
            st.info("ℹ️ **Keine Kategorien verfügbar** - Filter verwenden Freitext-Eingabe. Stellen Sie sicher, dass Ihre Analysedatei ein 'Kategorien'-Sheet enthält.")

        # Unsaved changes indicator
        if st.session_state.get('explorer_config_modified', False):
            st.warning("⚠️ Ungespeicherte Änderungen")

        render_explorer_config_tabs()

        # Zeige Filter-Validierung falls Kategorien verfǬgbar
        if category_loader and category_loader.is_loaded:
            st.markdown("---")
            from webapp_components.smart_filter_controls import render_filter_validation_summary
            render_filter_validation_summary(explorer_config_data.analysis_configs, category_loader)

    with st.expander("🔧 Optionen & Speichern", expanded=False):
        render_config_file_controls()

    # Fehler-Log einer fehlgeschlagenen Analyse (falls vorhanden)
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
            # R5.1: Analysen hinzufuegen als echtes Modal (@st.dialog)
            render_add_analysis_dialog()
    
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
        # Placeholder for layout consistency
        st.empty()
        
    with col3:
        # Remove button
        if st.button("🗑️ Entfernen", key=f"remove_{index}", use_container_width=True, type="secondary"):
            # R7.2: Entfernen nur nach Bestaetigung (Confirm-Dialog)
            confirm_removal(
                f"Analyse '{analysis.name}'",
                on_confirm=lambda idx=index: handle_remove_analysis(idx),
            )
    
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
    
    # Hole CategoryLoader vom ExplorerConfigManager
    explorer_config_manager = st.session_state.get('explorer_config_manager')
    category_loader = explorer_config_manager.get_category_loader() if explorer_config_manager else None
    
    # Verwende intelligente Filter-Controls
    from webapp_components.smart_filter_controls import render_smart_filter_controls
    filter_updated = render_smart_filter_controls(analysis, index, category_loader)
    
    # Prüfe auf Filter-Updates über Session State
    filter_update_key = f'filter_update_{index}'
    if filter_update_key in st.session_state:
        updated_filters = st.session_state[filter_update_key]
        handle_update_analysis(index, {'filters': updated_filters})
        del st.session_state[filter_update_key]  # Cleanup
    
    st.markdown("---")
    
    # Render analysis-specific parameters
    st.markdown("### ⚙️ Parameter")
    
    if analysis.analysis_type == 'netzwerk':
        render_network_parameters(analysis, index)
    elif analysis.analysis_type == 'heatmap':
        render_heatmap_parameters(analysis, index)
    elif analysis.analysis_type == 'sunburst':
        render_sunburst_parameters(analysis, index)
    elif analysis.analysis_type == 'treemap':
        render_treemap_parameters(analysis, index)
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
        'sunburst': """
        **☀️ Sunburst-Diagramm**
        
        Erstellt ein interaktives Sunburst-Diagramm zur Visualisierung hierarchischer Daten.
        Zeigt die Beziehungen zwischen Hauptkategorien, Subkategorien und Schlüsselwörtern.
        
        **Ausgabe:**
        - Zwei HTML-Dateien werden erstellt:
          1. Standard-Version (nur Labels)
          2. Version mit Werten in den Beschriftungen (_with_values.html)
        - Excel-Datei mit den zugrunde liegenden Daten
        
        **Keine Parameter erforderlich** - Die Visualisierung wird automatisch aus den gefilterten Daten erstellt.
        """,
        'treemap': """
        **🗺️ Treemap-Diagramm**
        
        Erstellt ein interaktives Treemap-Diagramm zur Visualisierung hierarchischer Daten.
        Zeigt die Beziehungen zwischen Hauptkategorien, Subkategorien und Schlüsselwörtern als verschachtelte Rechtecke.
        
        **Ausgabe:**
        - Zwei HTML-Dateien werden erstellt:
          1. Standard-Version (nur Labels)
          2. Version mit Werten in den Beschriftungen (_with_values.html)
        - Excel-Datei mit den zugrunde liegenden Daten
        
        **Keine Parameter erforderlich** - Die Visualisierung wird automatisch aus den gefilterten Daten erstellt.
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
    
    # Estimate network size based on filters
    try:
        explorer_config_manager = st.session_state.get('explorer_config_manager')
        category_loader = explorer_config_manager.get_category_loader() if explorer_config_manager else None
        
        if category_loader:
            # Get the selected file path
            selected_file = st.session_state.get('selected_analysis_file')
            if selected_file:
                import pandas as pd
                
                # Load data and apply filters
                df = pd.read_excel(category_loader.excel_path, sheet_name='Kodierungsergebnisse')
                
                # Apply filters
                filters = analysis.filters
                if filters.get('Dokument'):
                    df = df[df['Dokument'] == filters['Dokument']]
                if filters.get('Hauptkategorie'):
                    # Support multiple main categories (comma-separated)
                    main_cats = [s.strip() for s in filters['Hauptkategorie'].split(',') if s.strip()]
                    if main_cats:
                        df = df[df['Hauptkategorie'].isin(main_cats)]
                if filters.get('Subkategorien'):
                    # Support multiple subcategories (comma-separated)
                    subcats = [s.strip() for s in filters['Subkategorien'].split(',') if s.strip()]
                    if subcats:
                        # Check if Subkategorie column exists, otherwise try Subkategorien
                        if 'Subkategorie' in df.columns:
                            df = df[df['Subkategorie'].isin(subcats)]
                        elif 'Subkategorien' in df.columns:
                            # For comma-separated subcategories in a single column
                            df = df[df['Subkategorien'].fillna('').str.split(',').apply(
                                lambda x: any(subcat in [item.strip() for item in x] for subcat in subcats)
                            )]
                
                # Estimate nodes and edges
                num_records = len(df)
                num_main_categories = df['Hauptkategorie'].nunique() if 'Hauptkategorie' in df.columns else 0
                num_subcategories = 0
                if 'Subkategorie' in df.columns:
                    num_subcategories = df['Subkategorie'].nunique()
                elif 'Subkategorien' in df.columns:
                    # Count unique subcategories from comma-separated values
                    all_subcats = set()
                    for subcats_str in df['Subkategorien'].dropna():
                        all_subcats.update([s.strip() for s in str(subcats_str).split(',') if s.strip()])
                    num_subcategories = len(all_subcats)
                
                num_keywords = 0
                if 'Schlüsselwort' in df.columns:
                    num_keywords = df['Schlüsselwort'].nunique()
                elif 'Schlüsselwörter' in df.columns:
                    # Count unique keywords from comma-separated values
                    all_keywords = set()
                    for keywords_str in df['Schlüsselwörter'].dropna():
                        all_keywords.update([k.strip() for k in str(keywords_str).split(',') if k.strip()])
                    num_keywords = len(all_keywords)
                
                # Estimate total nodes (categories + subcategories + keywords)
                estimated_nodes = num_main_categories + num_subcategories + num_keywords
                
                # Estimate edges (rough approximation based on co-occurrences)
                # Typically: edges ≈ 1.5 * nodes for sparse networks, up to nodes² for dense networks
                # We use a conservative estimate
                estimated_edges = min(num_records, estimated_nodes * 3)
                
                # Display estimation
                st.info(f"""
📊 **Geschätzte Netzwerkgröße** (basierend auf Filtern):
- **{num_records} Kodierungen** gefiltert
- **~{estimated_nodes} Knoten** ({num_main_categories} Haupt-, {num_subcategories} Sub-, {num_keywords} Keywords)
- **~{estimated_edges} Kanten** (geschätzt)

💡 **Empfohlene Parameter für diese Größe:**
""")
                
                # Provide recommendations based on size
                if estimated_nodes < 20:
                    st.success("✓ Kleines Netzwerk - Standardwerte sind optimal")
                elif estimated_nodes < 50:
                    st.warning("⚠️ Mittleres Netzwerk - Erwägen Sie: Node Size 0.5-1.0, Iterations 50-100")
                else:
                    st.error("⚠️ Großes Netzwerk - Empfohlen: Node Size 0.3-0.7, Iterations 100-200, Scaling 0.5-1.0")
                
    except Exception as e:
        # Silently fail if estimation doesn't work
        pass
    
    # Node size factor
    new_node_size = st.number_input(
        "Node Size Factor:",
        min_value=0.1,
        max_value=10.0,
        value=float(params.get('node_size_factor', 1.0)),
        step=0.1,
        key=f"node_size_{index}",
        help="""Skalierungsfaktor für Knotengröße. 
        
Empfehlungen:
• Kleine Netzwerke (< 20 Knoten): 1.0 - 2.0
• Mittlere Netzwerke (20-50 Knoten): 0.5 - 1.0
• Große Netzwerke (> 50 Knoten): 0.3 - 0.7

Höhere Werte = größere Knoten (besser lesbar, aber mehr Überlappung)"""
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
        help="""Anzahl der Iterationen für Layout-Berechnung (Force-Directed Algorithm).

Empfehlungen:
• Kleine Netzwerke (< 20 Knoten): 30-50 Iterationen
• Mittlere Netzwerke (20-50 Knoten): 50-100 Iterationen
• Große Netzwerke (> 50 Knoten): 100-200 Iterationen
• Sehr dichte Netzwerke: 200-300 Iterationen

Mehr Iterationen = besseres Layout, aber längere Berechnungszeit"""
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
            help="""Anziehungskraft zum Zentrum des Graphen.

Empfehlungen:
• Lockere Netzwerke (wenige Kanten): 0.1 - 0.3
• Dichte Netzwerke (viele Kanten): 0.0 - 0.1
• Sehr dichte Netzwerke: -0.1 - 0.0 (abstoßend)

Positive Werte = Knoten werden zum Zentrum gezogen
Negative Werte = Knoten werden vom Zentrum abgestoßen
0 = keine Gravitation"""
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
            help="""Gesamtskalierung des Graphen (Abstand zwischen Knoten).

Empfehlungen:
• Kleine Netzwerke (< 20 Knoten): 1.0 - 2.0
• Mittlere Netzwerke (20-50 Knoten): 0.8 - 1.5
• Große Netzwerke (> 50 Knoten): 0.5 - 1.0
• Sehr große Netzwerke (> 100 Knoten): 0.3 - 0.7

Höhere Werte = mehr Abstand zwischen Knoten (weniger Überlappung)
Niedrigere Werte = kompakteres Layout"""
        )
        if new_scaling != params.get('scaling'):
            params['scaling'] = new_scaling
            updated = True
    
    # Label offset
    new_label_offset = st.slider(
        "Label-Abstand:",
        min_value=0.5,
        max_value=3.0,
        value=float(params.get('label_offset', 1.0)),
        step=0.1,
        key=f"label_offset_{index}",
        help="""Abstand der Labels von den Knoten.

Empfehlungen:
• 0.5-0.8: Labels näher an Knoten (kompakter, mehr Überlappung)
• 1.0: Standard-Abstand (ausgewogen)
• 1.5-2.0: Labels weiter entfernt (weniger Überlappung, mehr Platz)
• 2.5-3.0: Maximaler Abstand (für sehr dichte Netzwerke)

Höhere Werte = weniger Label-Überlappung, aber mehr Platz benötigt"""
    )
    if new_label_offset != params.get('label_offset'):
        params['label_offset'] = new_label_offset
        updated = True
    
    # Exclude "nicht kodiert" checkbox
    st.markdown("---")
    new_exclude = st.checkbox(
        "❌ 'Nicht kodiert' ausschließen",
        value=params.get('exclude_not_coded', True),
        key=f"exclude_not_coded_{index}",
        help="Wenn aktiviert, werden Einträge mit 'Nicht kodiert' oder 'Kein Kodierkonsens' automatisch aus der Visualisierung ausgeschlossen."
    )
    if new_exclude != params.get('exclude_not_coded', True):
        params['exclude_not_coded'] = new_exclude
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
    
    # Show recommendations for heatmap analyses
    with st.expander("💡 Empfohlene Heatmap-Analysen", expanded=False):
        st.markdown("""
        **Sinnvolle Kombinationen für Ihre QCA-Daten:**
        
        1. **Kategorien × Dokumente** (Häufigkeitsverteilung)
           - X: Hauptkategorie, Y: Dokument, Z: count
           - Zeigt, welche Kategorien in welchen Dokumenten vorkommen
        
        2. **Hauptkategorien × Subkategorien** (Hierarchie-Übersicht)
           - X: Hauptkategorie, Y: Subkategorie, Z: count
           - Zeigt die Verteilung der Subkategorien pro Hauptkategorie
        
        3. **Kategorien × Attribut 1** (z.B. demografische Verteilung)
           - X: Hauptkategorie, Y: Attribut1, Z: count
           - Zeigt Verteilung nach Geschlecht, Alter, etc.
        
        4. **Kategorien × Attribut 2**
           - X: Hauptkategorie, Y: Attribut2, Z: count
        
        5. **Attribut 1 × Attribut 2** (Kreuztabelle)
           - X: Attribut1, Y: Attribut2, Z: count
           - Zeigt Zusammenhänge zwischen zwei Attributen
        
        6. **Kategorien × Coder** (Intercoder-Vergleich)
           - X: Hauptkategorie, Y: Coder_ID, Z: count
           - Zeigt Kodierungsverteilung pro Coder
        
        **Tipp:** Verwenden Sie Filter, um die Analyse auf bestimmte Dokumente oder Kategorien einzugrenzen!
        """)
    
    # Get available columns from the selected file
    available_columns = []
    try:
        explorer_config_manager = st.session_state.get('explorer_config_manager')
        category_loader = explorer_config_manager.get_category_loader() if explorer_config_manager else None
        
        if category_loader:
            import pandas as pd
            df = pd.read_excel(category_loader.excel_path, sheet_name='Kodierungsergebnisse')
            available_columns = df.columns.tolist()
    except:
        pass
    
    # Common heatmap columns
    common_columns = ['Hauptkategorie', 'Subkategorie', 'Schlüsselwort', 'Dokument', 'Coder_ID', 'count']
    
    # Merge available and common columns
    if available_columns:
        column_options = available_columns
    else:
        column_options = common_columns
    
    # X, Y, Z attributes with dropdowns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # X-Achse
        current_x = params.get('x_attribute', 'Hauptkategorie')
        x_index = column_options.index(current_x) if current_x in column_options else 0
        
        new_x = st.selectbox(
            "X-Achse Attribut:",
            options=column_options,
            index=x_index,
            key=f"x_attr_{index}",
            help="Spalte für X-Achse der Heatmap"
        )
        if new_x != params.get('x_attribute'):
            params['x_attribute'] = new_x
            updated = True
    
    with col2:
        # Y-Achse
        current_y = params.get('y_attribute', 'Subkategorie')
        y_index = column_options.index(current_y) if current_y in column_options else (1 if len(column_options) > 1 else 0)
        
        new_y = st.selectbox(
            "Y-Achse Attribut:",
            options=column_options,
            index=y_index,
            key=f"y_attr_{index}",
            help="Spalte für Y-Achse der Heatmap"
        )
        if new_y != params.get('y_attribute'):
            params['y_attribute'] = new_y
            updated = True
    
    with col3:
        # Z-Werte (mit 'count' als spezielle Option)
        z_options = ['count'] + [col for col in column_options if col != 'count']
        current_z = params.get('z_attribute', 'count')
        z_index = z_options.index(current_z) if current_z in z_options else 0
        
        new_z = st.selectbox(
            "Werte Attribut:",
            options=z_options,
            index=z_index,
            key=f"z_attr_{index}",
            help="Werte für Heatmap-Zellen. 'count' = Häufigkeiten zählen"
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
    
    # Format string with dropdown
    format_options = {
        '.0f': 'Ganzzahl (z.B. 42)',
        '.1f': '1 Dezimalstelle (z.B. 42.5)',
        '.2f': '2 Dezimalstellen (z.B. 42.50)',
        '.3f': '3 Dezimalstellen (z.B. 42.500)',
        'd': 'Integer (z.B. 42)',
        '.0%': 'Prozent ohne Dezimalen (z.B. 42%)',
        '.1%': 'Prozent mit 1 Dezimale (z.B. 42.5%)',
        '.2%': 'Prozent mit 2 Dezimalen (z.B. 42.50%)'
    }
    
    current_fmt = params.get('fmt', '.2f')
    fmt_keys = list(format_options.keys())
    fmt_index = fmt_keys.index(current_fmt) if current_fmt in fmt_keys else 2  # Default to '.2f'
    
    new_fmt = st.selectbox(
        "Zahlenformat:",
        options=fmt_keys,
        index=fmt_index,
        format_func=lambda x: format_options[x],
        key=f"fmt_{index}",
        help="Format für die Darstellung der Werte in den Heatmap-Zellen"
    )
    if new_fmt != params.get('fmt'):
        params['fmt'] = new_fmt
        updated = True
    
    # Exclude "nicht kodiert" checkbox
    st.markdown("---")
    new_exclude = st.checkbox(
        "❌ 'Nicht kodiert' ausschließen",
        value=params.get('exclude_not_coded', True),
        key=f"exclude_not_coded_heatmap_{index}",
        help="Wenn aktiviert, werden Einträge mit 'Nicht kodiert' oder 'Kein Kodierkonsens' automatisch aus der Visualisierung ausgeschlossen."
    )
    if new_exclude != params.get('exclude_not_coded', True):
        params['exclude_not_coded'] = new_exclude
        updated = True
    
    if updated:
        handle_update_analysis(index, {'params': params})


def render_sunburst_parameters(analysis: AnalysisConfig, index: int):
    """
    Renders sunburst-specific parameters.
    
    Args:
        analysis: AnalysisConfig to render parameters for
        index: Index of the analysis
    """
    params = analysis.params.copy()
    updated = False
    
    st.markdown("""
    **📊 Sunburst-Visualisierung**
    
    Erstellt eine kreisförmige hierarchische Darstellung mit konzentrischen Ringen.
    Ausgabe als hochauflösende PNG-Datei.
    """)
    
    with st.expander("🎨 Erweiterte Parameter", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # DPI
            new_dpi = st.number_input(
                "Auflösung (DPI):",
                min_value=72,
                max_value=600,
                value=int(params.get('dpi', 300)),
                step=50,
                key=f"sunburst_dpi_{index}",
                help="""Auflösung der PNG-Datei in Dots Per Inch.

**Empfehlungen:**
• 72-150 DPI: Bildschirmanzeige, Web
• 300 DPI: Druckqualität (Standard)
• 600 DPI: Hochauflösender Druck

Höhere Werte = größere Dateien, aber schärfere Bilder."""
            )
            if new_dpi != params.get('dpi'):
                params['dpi'] = new_dpi
                updated = True
            
            # Font size
            new_font_size = st.number_input(
                "Schriftgröße Labels:",
                min_value=4,
                max_value=20,
                value=int(params.get('font_size', 8)),
                key=f"sunburst_font_{index}",
                help="""Schriftgröße für die Labels in den Ringen.

**Empfehlungen:**
• 6-8: Viele kleine Labels
• 8-10: Standard (gut lesbar)
• 10-14: Wenige große Labels
• 14+: Sehr große, prominente Labels

Bei vielen Knoten kleinere Schrift verwenden."""
            )
            if new_font_size != params.get('font_size'):
                params['font_size'] = new_font_size
                updated = True
            
            # Max label length
            new_max_label = st.number_input(
                "Max. Label-Länge:",
                min_value=5,
                max_value=50,
                value=int(params.get('max_label_length', 15)),
                key=f"sunburst_maxlabel_{index}",
                help="""Maximale Zeichenanzahl für Labels.

Längere Labels werden abgeschnitten und mit '...' versehen.

**Empfehlungen:**
• 10-15: Kompakte Darstellung
• 15-20: Standard (gut lesbar)
• 20-30: Ausführliche Labels
• 30+: Sehr lange Labels (nur bei wenigen Knoten)"""
            )
            if new_max_label != params.get('max_label_length'):
                params['max_label_length'] = new_max_label
                updated = True
        
        with col2:
            # Ring width
            new_ring_width = st.number_input(
                "Ring-Breite:",
                min_value=0.5,
                max_value=3.0,
                value=float(params.get('ring_width', 1.5)),
                step=0.1,
                key=f"sunburst_ring_{index}",
                help="""Breite der konzentrischen Ringe (Hierarchieebenen).

**Empfehlungen:**
• 1.0-1.3: Schmale Ringe (viele Ebenen)
• 1.5: Standard (ausgewogen)
• 1.8-2.5: Breite Ringe (wenige Ebenen)
• 2.5+: Sehr breite Ringe (max. 2-3 Ebenen)

Kleinere Werte = mehr Platz für tiefe Hierarchien."""
            )
            if new_ring_width != params.get('ring_width'):
                params['ring_width'] = new_ring_width
                updated = True
            
            # Color scheme
            color_schemes = ['Set3', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'tab10', 'tab20']
            new_color_scheme = st.selectbox(
                "Farbschema:",
                options=color_schemes,
                index=color_schemes.index(params.get('color_scheme', 'Set3')) if params.get('color_scheme', 'Set3') in color_schemes else 0,
                key=f"sunburst_colors_{index}",
                help="""Matplotlib Farbschema für Hauptkategorien.

**Verfügbare Schemata:**
• Set3: Pastellfarben, gut unterscheidbar (Standard)
• Pastel1/2: Sanfte Pastelltöne
• Set1/2: Kräftige, gesättigte Farben
• tab10: 10 unterschiedliche Farben
• tab20: 20 unterschiedliche Farben (viele Kategorien)

Unterkategorien werden automatisch heller eingefärbt."""
            )
            if new_color_scheme != params.get('color_scheme'):
                params['color_scheme'] = new_color_scheme
                updated = True
            
            # Show values
            new_show_values = st.checkbox(
                "Werte anzeigen",
                value=params.get('show_values', True),
                key=f"sunburst_values_{index}",
                help="""Anzahl der Kodierungen in Labels anzeigen.

**Aktiviert:** "Label (5)" - zeigt Anzahl
**Deaktiviert:** "Label" - nur Text

Nützlich um die Häufigkeit von Codes zu sehen."""
            )
            if new_show_values != params.get('show_values'):
                params['show_values'] = new_show_values
                updated = True
        
        # Label background settings
        st.markdown("**Label-Hintergrund:**")
        col3, col4 = st.columns(2)
        
        with col3:
            # Label background color
            new_label_bg = st.color_picker(
                "Hintergrundfarbe:",
                value=params.get('label_bg_color', '#FFFFFF'),
                key=f"sunburst_labelbg_{index}",
                help="""Hintergrundfarbe für Label-Boxen (HEX-Wert).

**Empfehlungen:**
• #FFFFFF (Weiß): Standard, gute Lesbarkeit
• #000000 (Schwarz): Für helle Schrift
• Transparent (Alpha 0): Kein Hintergrund
• Beliebige Farbe: Für spezielle Designs

Kombinieren Sie mit Transparenz für beste Ergebnisse."""
            )
            if new_label_bg != params.get('label_bg_color'):
                params['label_bg_color'] = new_label_bg
                updated = True
        
        with col4:
            # Label background alpha
            new_label_bg_alpha = st.slider(
                "Transparenz:",
                min_value=0.0,
                max_value=1.0,
                value=float(params.get('label_bg_alpha', 0.7)),
                step=0.05,
                key=f"sunburst_labelalpha_{index}",
                help="""Transparenz des Label-Hintergrunds.

**Werte:**
• 0.0: Vollständig durchsichtig (kein Hintergrund)
• 0.5-0.7: Halbtransparent (Standard, gute Balance)
• 1.0: Vollständig deckend

Niedrigere Werte zeigen mehr von den Farben darunter."""
            )
            if new_label_bg_alpha != params.get('label_bg_alpha'):
                params['label_bg_alpha'] = new_label_bg_alpha
                updated = True
    
    # Exclude "nicht kodiert" checkbox
    st.markdown("---")
    new_exclude = st.checkbox(
        "❌ 'Nicht kodiert' ausschließen",
        value=params.get('exclude_not_coded', True),
        key=f"exclude_not_coded_sunburst_{index}",
        help="Wenn aktiviert, werden Einträge mit 'Nicht kodiert' oder 'Kein Kodierkonsens' automatisch aus der Visualisierung ausgeschlossen."
    )
    if new_exclude != params.get('exclude_not_coded', True):
        params['exclude_not_coded'] = new_exclude
        updated = True
    
    if updated:
        handle_update_analysis(index, {'params': params})


def render_treemap_parameters(analysis: AnalysisConfig, index: int):
    """
    Renders treemap-specific parameters.
    
    Args:
        analysis: AnalysisConfig to render parameters for
        index: Index of the analysis
    """
    params = analysis.params.copy()
    updated = False
    
    st.markdown("""
    **📊 Treemap-Visualisierung**
    
    Erstellt eine rechteckige hierarchische Darstellung mit optimiertem Layout.
    Ausgabe als hochauflösende PNG-Dateien (Übersicht + Detail pro Kategorie).
    """)
    
    with st.expander("🎨 Erweiterte Parameter", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # DPI
            new_dpi = st.number_input(
                "Auflösung (DPI):",
                min_value=72,
                max_value=600,
                value=int(params.get('dpi', 300)),
                step=50,
                key=f"treemap_dpi_{index}",
                help="""Auflösung der PNG-Dateien in Dots Per Inch.

**Empfehlungen:**
• 72-150 DPI: Bildschirmanzeige, Web
• 300 DPI: Druckqualität (Standard)
• 600 DPI: Hochauflösender Druck

Gilt für Übersicht und Detail-Ansichten."""
            )
            if new_dpi != params.get('dpi'):
                params['dpi'] = new_dpi
                updated = True
            
            # Font size
            new_font_size = st.number_input(
                "Schriftgröße:",
                min_value=6,
                max_value=20,
                value=int(params.get('font_size', 10)),
                key=f"treemap_font_{index}",
                help="""Schriftgröße für Labels in der Hauptansicht.

**Empfehlungen:**
• 8-10: Viele Kategorien
• 10-12: Standard (gut lesbar)
• 12-16: Wenige Kategorien
• 16+: Sehr große, prominente Labels"""
            )
            if new_font_size != params.get('font_size'):
                params['font_size'] = new_font_size
                updated = True
            
            # Detail font size
            new_detail_font = st.number_input(
                "Schriftgröße Detail:",
                min_value=6,
                max_value=20,
                value=int(params.get('detail_font_size', 9)),
                key=f"treemap_detailfont_{index}",
                help="""Schriftgröße für Labels in der Detail-Ansicht.

**Empfehlungen:**
• 7-9: Viele Subkategorien
• 9-11: Standard (gut lesbar)
• 11-14: Wenige Subkategorien

Detail-Ansicht zeigt Subkategorien pro Hauptkategorie."""
            )
            if new_detail_font != params.get('detail_font_size'):
                params['detail_font_size'] = new_detail_font
                updated = True
        
        with col2:
            # Color scheme
            color_schemes = ['Set3', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'tab10', 'tab20']
            new_color_scheme = st.selectbox(
                "Farbschema:",
                options=color_schemes,
                index=color_schemes.index(params.get('color_scheme', 'Set3')) if params.get('color_scheme', 'Set3') in color_schemes else 0,
                key=f"treemap_colors_{index}",
                help="""Matplotlib Farbschema für Hauptansicht.

**Verfügbare Schemata:**
• Set3: Pastellfarben, gut unterscheidbar (Standard)
• Pastel1/2: Sanfte Pastelltöne
• Set1/2: Kräftige, gesättigte Farben
• tab10: 10 unterschiedliche Farben
• tab20: 20 unterschiedliche Farben (viele Kategorien)"""
            )
            if new_color_scheme != params.get('color_scheme'):
                params['color_scheme'] = new_color_scheme
                updated = True
            
            # Detail color scheme
            detail_schemes = ['Pastel1', 'Pastel2', 'Set3', 'Set1', 'Set2']
            new_detail_scheme = st.selectbox(
                "Farbschema Detail:",
                options=detail_schemes,
                index=detail_schemes.index(params.get('detail_color_scheme', 'Pastel1')) if params.get('detail_color_scheme', 'Pastel1') in detail_schemes else 0,
                key=f"treemap_detailcolors_{index}",
                help="""Farbschema für Detail-Ansicht (Subkategorien).

**Empfehlungen:**
• Pastel1/2: Sanfte Farben (Standard)
• Set3: Pastellfarben mit guter Unterscheidbarkeit
• Set1/2: Kräftige Farben für Kontrast

Wird für Balkendiagramme der Subkategorien verwendet."""
            )
            if new_detail_scheme != params.get('detail_color_scheme'):
                params['detail_color_scheme'] = new_detail_scheme
                updated = True
            
            # Show values
            new_show_values = st.checkbox(
                "Werte anzeigen",
                value=params.get('show_values', True),
                key=f"treemap_values_{index}",
                help="""Anzahl der Kodierungen in Labels anzeigen.

**Aktiviert:** "Label (5)" - zeigt Anzahl
**Deaktiviert:** "Label" - nur Text

Nützlich um die Häufigkeit von Codes zu sehen.
Gilt für Hauptansicht und Detail-Ansicht."""
            )
            if new_show_values != params.get('show_values'):
                params['show_values'] = new_show_values
                updated = True
    
    # Exclude "nicht kodiert" checkbox
    st.markdown("---")
    new_exclude = st.checkbox(
        "❌ 'Nicht kodiert' ausschließen",
        value=params.get('exclude_not_coded', True),
        key=f"exclude_not_coded_treemap_{index}",
        help="Wenn aktiviert, werden Einträge mit 'Nicht kodiert' oder 'Kein Kodierkonsens' automatisch aus der Visualisierung ausgeschlossen."
    )
    if new_exclude != params.get('exclude_not_coded', True):
        params['exclude_not_coded'] = new_exclude
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
    
    # Exclude "nicht kodiert" checkbox - default FALSE for LLM analyses
    st.markdown("---")
    new_exclude = st.checkbox(
        "❌ 'Nicht kodiert' ausschließen",
        value=params.get('exclude_not_coded', False),
        key=f"exclude_not_coded_summary_{index}",
        help="Wenn aktiviert, werden Einträge mit 'Nicht kodiert' oder 'Kein Kodierkonsens' aus der Analyse ausgeschlossen. Für LLM-Analysen standardmäßig deaktiviert, da auch nicht-kodierte Texte relevant sein können."
    )
    if new_exclude != params.get('exclude_not_coded', False):
        params['exclude_not_coded'] = new_exclude
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
        
        # If 'active' status changed, rerun to update tab labels
        if 'active' in updates:
            st.rerun()
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
    output_dir = st.session_state.get('explorer_output_dir')
    
    # Header with back button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.subheader(f"📊 Analyse: {Path(file_path).name}")
    
    with col2:
        if st.button("◀ Zurück zur Konfiguration", key="back_to_config"):
            st.session_state.explorer_active = False
            st.rerun()
    
    st.markdown("---")
    
    # Show success message with output directory and log
    if output_dir:
        st.success("✅ Alle Analysen erfolgreich abgeschlossen!")
        
        # Action buttons for output directory
        col_info, col_btn = st.columns([3, 1])
        
        with col_info:
            st.info(f"📁 Ergebnisse gespeichert in: {output_dir}")
        
        with col_btn:
            # Open folder button
            if st.button("📂 Ordner öffnen", key="open_explorer_results_folder"):
                try:
                    if platform.system() == 'Windows':
                        os.startfile(output_dir)
                    elif platform.system() == 'Darwin':  # macOS
                        subprocess.run(['open', output_dir])
                    else:  # Linux
                        subprocess.run(['xdg-open', output_dir])
                    
                    st.success("✅ Ordner geöffnet")
                except Exception as e:
                    st.error(f"❌ Fehler: {str(e)}")
        
        # Show analysis log in expander
        if st.session_state.get('explorer_analysis_log'):
            log_data = st.session_state.explorer_analysis_log
            with st.expander("📋 Analyse-Log anzeigen", expanded=False):
                st.text_area("Log", log_data['log_text'], height=300, key="explorer_log_display_success", label_visibility="collapsed")
        
        # R9.1: Shortcut fuer eine neue Analyse (zurueck zur Konfiguration)
        if st.button("➕ Neue Analyse starten", key="new_analysis_shortcut", type="primary"):
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
        
        # Show statistics FIRST (before charts to prevent auto-scroll)
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
        
        st.markdown("---")
        
        # Display visualizations in expanders to prevent auto-scroll
        config = st.session_state.explorer_config_data
        
        if 'category_distribution' in config.enabled_charts:
            with st.expander("📊 Kategorieverteilung", expanded=True):
                if 'Hauptkategorie' in df.columns:
                    category_counts = df['Hauptkategorie'].value_counts()
                    st.bar_chart(category_counts)
                else:
                    st.warning("Spalte 'Hauptkategorie' nicht gefunden")
        
        if 'confidence_histogram' in config.enabled_charts:
            with st.expander("📈 Konfidenz-Verteilung", expanded=True):
                st.caption("Die Konfidenz gibt an, wie sicher das LLM-Modell bei der Zuordnung einer Kodierung war (0 = unsicher, 1 = sehr sicher).")
                
                if 'Konfidenz' in df.columns:
                    # Create histogram with better binning
                    import numpy as np
                    
                    # Create bins from 0 to 1 with 0.05 steps (20 bins)
                    bins = np.linspace(0, 1, 21)
                    bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
                    
                    # Cut data into bins
                    df['Konfidenz_Bin'] = pd.cut(df['Konfidenz'], bins=bins, labels=bin_labels, include_lowest=True)
                    
                    # Count values per bin
                    hist_data = df['Konfidenz_Bin'].value_counts().sort_index()
                    
                    # Create a more readable chart
                    st.bar_chart(hist_data)
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Durchschnitt", f"{df['Konfidenz'].mean():.3f}")
                    with col2:
                        st.metric("Median", f"{df['Konfidenz'].median():.3f}")
                    with col3:
                        st.metric("Std. Abweichung", f"{df['Konfidenz'].std():.3f}")
                else:
                    st.warning("Spalte 'Konfidenz' nicht gefunden")
        
        st.markdown("---")
        
        # Data table - in expander to prevent auto-scroll
        with st.expander("📋 Kodierungsdaten anzeigen", expanded=False):
            st.markdown("**Datentabelle**")
            
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

        # R2.3: Handoff - soeben erzeugte Analysedatei vorbelegen.
        # Stiller Fallback (R2.4): ohne Handoff (oder wenn die Datei nicht mehr
        # in der Liste ist) gilt Index 0 = aktuellste Datei.
        default_index = 0
        handoff_file = st.session_state.pop("handoff_analysis_file", None)
        if handoff_file:
            for i, (label, path) in enumerate(file_options.items()):
                if str(path) == str(handoff_file):
                    default_index = i
                    # Widget-State (falls vorhanden) auf die Handoff-Datei setzen
                    st.session_state["selected_analysis_file"] = label
                    break

        selected = st.selectbox(
            "Wählen Sie eine Analysedatei:",
            options=list(file_options.keys()),
            index=default_index,
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
    Zeigt das Fehler-Log einer fehlgeschlagenen Explorer-Analyse
    (Start-Button und Datei-Info sind in den Kompaktbereich gerueckt, R9).
    
    Requirements: 1.5, 7.5
    """
    # R9: Der Start-Button ist in den Kompaktbereich gerueckt (R9.1) und
    # oeffnet dort den Bestaetigungsdialog (R6); die ausgewaehlte Datei
    # zeigt die Summary-Zeile im Kompaktbereich an.
    # Hier bleibt nur die Anzeige des Fehler-Logs.
    # Display log ONLY if analysis failed (not on success - success goes to explorer view)
    if st.session_state.get('explorer_analysis_log'):
        log_data = st.session_state.explorer_analysis_log
        
        # Only show log here if analysis failed
        if not log_data.get('success'):
            st.markdown("---")
            st.error("❌ Analyse fehlgeschlagen")
            st.text_area("📋 Analyse-Log", log_data['log_text'], height=300, key="explorer_log_display_error")


@st.dialog("➕ Neue Analyse hinzufügen")
def render_add_analysis_dialog():
    """
    Rendert "Analyse hinzufügen" als echtes Modal (@st.dialog, R5).

    Requirement 5.1, 5.2, 5.3: Analyse im Modal konfigurieren und hinzufuegen
    """
    # Analysis type selection
    analysis_types = {
        'netzwerk': '📊 Netzwerkanalyse',
        'heatmap': '🔥 Heatmap',
        'sunburst': '☀️ Sunburst-Diagramm',
        'treemap': '🗺️ Treemap-Diagramm',
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
            
            # @st.dialog: st.rerun() schliesst das Modal automatisch
            st.success(f"✅ Analyse '{analysis_name}' hinzugefügt!")
            st.rerun()
    
    with col2:
        if st.button("❌ Abbrechen", use_container_width=True):
            st.rerun()
    
    st.markdown("---")


def save_explorer_config_data(format: str = "both"):
    """
    Speichert die Explorer-Konfiguration über ExplorerConfigManager (R3.2-R3.4).
    
    Args:
        format: 'both' (Standard: JSON + XLSX), 'json' oder 'xlsx'
    
    Verhalten:
    - Success: EINE Success-Meldung, `explorer_config_modified` wird gelöscht
    - Partial Success (R3.4): EINE gebündelte Warning-Box mit dem fehlgeschlagenen
      Format und dessen Fehlern; Modified-Flag wird gelöscht ( mind. ein Format persistiert)
    - Beide fehlgeschlagen: Error-Box mit beiden Fehlern, Modified-Flag bleibt
    
    Requirement 1.5: WHEN the user saves the configuration 
                    THEN the system SHALL persist all analysis configurations to the config file
    Requirement 3.4: WHEN the Explorer-Config is saved 
                    THEN the system SHALL persist provider and model in base_config
    
    FIX: Explorer speichert sein eigenes, unabhängiges LLM-Modell
         (nicht vom Config UI-Modell überschrieben) - die Werte wurden bereits
         über die UI-Controls in explorer_config_data.base_config gesetzt.
    """
    manager = st.session_state.explorer_config_manager
    config = st.session_state.explorer_config_data
    
    success_json, errors_json = (True, [])
    success_xlsx, errors_xlsx = (True, [])
    if format in ("both", "json"):
        success_json, errors_json = manager.save_config(config, format='json')
    if format in ("both", "xlsx"):
        success_xlsx, errors_xlsx = manager.save_config(config, format='xlsx')
    
    if success_json and success_xlsx:
        st.session_state.explorer_config_modified = False
        label = {"both": " (JSON & XLSX)", "json": " (JSON)", "xlsx": " (XLSX)"}.get(format, "")
        st.success(f"✅ Konfiguration gespeichert{label}")
    elif success_json or success_xlsx:
        # R3.4: Partial Success in EINER gebündelten Warning-Box
        saved_label = "JSON" if success_json else "XLSX"
        failed_label = "XLSX" if success_json else "JSON"
        failed_errors = errors_xlsx if success_json else errors_json
        st.session_state.explorer_config_modified = False
        st.warning(
            f"⚠️ {saved_label} gespeichert, aber {failed_label} fehlgeschlagen: "
            f"{', '.join(failed_errors)}"
        )
    else:
        st.error(
            f"❌ Fehler beim Speichern: JSON: {', '.join(errors_json)}, "
            f"XLSX: {', '.join(errors_xlsx)}"
        )


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
            
            # Trigger rerun to display results view (which starts at top)
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
        
        # Action buttons at the top (appear once for all files)
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            # Open file location button - opens the output directory
            if st.button("📂 Ordner öffnen", key="open_explorer_folder_top"):
                import subprocess
                import platform
                
                try:
                    if platform.system() == 'Windows':
                        os.startfile(output_dir)
                    elif platform.system() == 'Darwin':  # macOS
                        subprocess.run(['open', output_dir])
                    else:  # Linux
                        subprocess.run(['xdg-open', output_dir])
                    
                    st.success("✅ Ordner geöffnet")
                except Exception as e:
                    st.error(f"❌ Fehler: {str(e)}")
        
        with col_btn2:
            # Show directory path button
            if st.button("📋 Pfad kopieren", key="copy_explorer_path_top"):
                st.code(output_dir, language=None)
        
        st.markdown("---")
        
        # R13: Gemeinsame Paginierung (siehe pagination.py)
        if total_files > st.session_state.files_per_page:
            page, page_size, total_pages = render_pagination_controls(total_files, "output_files")

            paginated_files, total_pages, _ = file_manager.paginate_files(
                files, page=page, page_size=page_size
            )

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
