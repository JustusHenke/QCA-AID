"""
QCA-AID Explorer Main Module
=============================

This module contains the main function for QCA-AID Explorer.
It orchestrates the analysis pipeline for exploring qualitative coding data.

The main function:
- Loads configuration from Excel or JSON files
- Initializes the LLM provider
- Creates a QCAAnalyzer instance
- Performs selective keyword harmonization
- Executes all configured analyses (network, heatmap, summary, sentiment)
"""

import os
import asyncio
from typing import Dict, Any

from QCA_AID_assets.utils.config.explorer_loader import ExplorerConfigLoader
from QCA_AID_assets.utils.llm.factory import LLMProviderFactory
from QCA_AID_assets.analysis.qca_analyzer import QCAAnalyzer
from QCA_AID_assets.utils.prompts import get_default_prompts
from QCA_AID_assets.utils.common import create_filter_string


async def main():
    """
    Hauptfunktion für QCA-AID Explorer.
    
    Diese Funktion orchestriert den gesamten Analyse-Workflow:
    1. Lädt die Konfiguration aus einer Excel- oder JSON-Datei
    2. Initialisiert den LLM-Provider (OpenAI, Mistral, etc.)
    3. Erstellt einen QCAAnalyzer mit den geladenen Daten
    4. Führt selektive Keyword-Harmonisierung durch (falls aktiviert)
    5. Führt alle konfigurierten Analysen durch:
       - Netzwerk-Visualisierungen
       - Heatmaps
       - Zusammenfassungen (Paraphrasen, Begründungen, benutzerdefiniert)
       - Sentiment-Analysen
    
    Die Konfiguration wird aus der Datei 'QCA-AID-Explorer-Config.xlsx' 
    im gleichen Verzeichnis wie das Skript geladen.
    
    Raises:
        FileNotFoundError: Wenn die Konfigurationsdatei nicht gefunden wird
        ValueError: Wenn keine Explorationsdatei in der Konfiguration angegeben ist
    """
    # Import version information
    try:
        from .__version__ import __version__, __version_date__
    except ImportError:
        __version__ = "0.11.1"
        __version_date__ = "2025-12-01"
    
    print(f"\n=== QCA-AID Explorer ===")
    print(f"Version {__version__} ({__version_date__})")
    print("Konfiguration über Excel-Datei")
    
    # Pfad zur Konfigurations-Excel-Datei
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up one level since we're now in QCA_AID_assets/
    SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
    CONFIG_FILE = "QCA-AID-Explorer-Config.xlsx"
    CONFIG_PATH = os.path.join(SCRIPT_DIR, CONFIG_FILE)
    
    # Lade Konfiguration
    config_loader = ExplorerConfigLoader(CONFIG_PATH)
    base_config = config_loader.get_base_config()
    analysis_configs = config_loader.get_analysis_configs()
    
    # Validiere Konfiguration gegen verfügbare Kategorien
    if config_loader.get_category_loader():
        print("\n=== Konfigurationsvalidierung ===")
        all_valid = True
        
        for config in analysis_configs:
            validation_errors = config_loader.validate_analysis_filters(config)
            if validation_errors:
                all_valid = False
                for error in validation_errors:
                    print(f"⚠️  {error}")
        
        if all_valid:
            print("✓ Alle Filter sind gültig")
        else:
            print("\n❌ Validierungsfehler gefunden!")
            print("Verwenden Sie den QCA-AID-Config-Builder.py um die Konfiguration zu korrigieren.")
            
            continue_anyway = input("Trotzdem fortfahren? (j/n): ").lower().startswith('j')
            if not continue_anyway:
                print("Analyse abgebrochen.")
                return
        print("=" * 40)
    
    # Extrahiere Basis-Parameter
    PROVIDER_NAME = base_config.get('provider', 'openai')
    MODEL_NAME = base_config.get('model', 'gpt-4o-mini')
    TEMPERATURE = float(base_config.get('temperature', 0.7))
    SCRIPT_DIR = base_config.get('script_dir') or SCRIPT_DIR
    OUTPUT_DIR = base_config.get('output_dir', 'output')
    EXPLORE_FILE = base_config.get('explore_file', '')
    CLEAN_KEYWORDS = str(base_config.get('clean_keywords', 'True')).lower() == 'true'
    SIMILARITY_THRESHOLD = float(base_config.get('similarity_threshold', 0.7))
    
    # Prüfe, ob die Explorationsdatei angegeben wurde
    if not EXPLORE_FILE:
        print("Fehler: Keine Explorationsdatei in der Konfiguration angegeben.")
        return
    
    # Pfadkonfiguration
    EXCEL_PATH = os.path.join(SCRIPT_DIR, OUTPUT_DIR, EXPLORE_FILE)
    print(f"\nLese Excel-Datei: {EXCEL_PATH}")
    
    # Initialize LLM provider
    print("\nInitialisiere LLM Provider...")
    llm_provider = LLMProviderFactory.create_provider(PROVIDER_NAME)
    
    # Initialize analyzer
    analyzer = QCAAnalyzer(EXCEL_PATH, llm_provider, base_config)
    print(f"verfügbare Spalten: {', '.join(analyzer.columns)}")
    
    # Standardprompts laden
    default_prompts = get_default_prompts()
    
    # Führe selektive Keyword-Harmonisierung durch
    harmonization_performed = analyzer.perform_selective_harmonization(
        analysis_configs, 
        CLEAN_KEYWORDS, 
        SIMILARITY_THRESHOLD
    )
    
    # Analysekonfigurationen durchlaufen
    print(f"\nFühre {len(analysis_configs)} Auswertungen durch...")
    
    for config_idx, analysis_config in enumerate(analysis_configs, 1):
        analysis_name = analysis_config['name']

        # Prüfe, ob die Analyse aktiviert ist
        is_active = analysis_config.get('params', {}).get('active', True)
        if isinstance(is_active, str):
            is_active = is_active.lower() in ('true', 'ja', 'yes', '1')
        else:
            is_active = bool(is_active)
        
        if not is_active:
            print(f"\n--- Überspringe deaktivierte Auswertung: {analysis_name} ---")
            continue
        
        print(f"\n--- Auswertung {config_idx}/{len(analysis_configs)}: {analysis_name} ---")
        
        # Extrahiere Filter und Parameter
        filters = analysis_config['filters']
        params = analysis_config['params']
        
        # Füge analysis_type zu params hinzu, falls nicht vorhanden
        if 'analysis_type' not in params:
            params['analysis_type'] = analysis_config.get('analysis_type', '')
        
        # Filtere die Daten
        filtered_df = analyzer.filter_data(filters)
        
        # Prüfe, ob nach dem Filtern Daten vorhanden sind
        if filtered_df.empty:
            print(f"⚠️  WARNUNG: Keine Daten nach Filterung für Analyse '{analysis_name}'")
            print(f"   Angewendete Filter: {filters}")
            print(f"   Diese Analyse wird übersprungen.")
            print(f"   Bitte überprüfen Sie die Filter-Einstellungen.\n")
            continue
        
        print(f"✓ {len(filtered_df)} Datensätze nach Filterung gefunden")
        
        # Erzeuge Filterstring für Dateinamen
        filter_str = create_filter_string(filters)
        
        # Füge Harmonisierungsinfo nur hinzu, wenn sie tatsächlich durchgeführt wurde
        # UND für diese spezifische Analyse relevant ist
        analysis_type = params.get('analysis_type', '').lower()
        if harmonization_performed and analyzer.needs_keyword_harmonization(analysis_type, params):
            filter_str = f"harmonized_{filter_str}" if filter_str else "harmonized"
        
        # Füge Analysenamen zum Filterstring hinzu
        output_prefix = f"{analysis_name}_{filter_str}" if filter_str else analysis_name
        
        if analysis_type == 'netzwerk':
            # Netzwerkvisualisierung erstellen
            print("\nErstelle Netzwerk-Visualisierung...")
            analyzer.create_network_graph(
                filtered_df,
                f"Code-Network_{output_prefix}",
                params
            )
            
        elif analysis_type == 'heatmap':
            # Heatmap erstellen
            print("\nErstelle Heatmap...")
            analyzer.create_heatmap(
                filtered_df,
                f"Heatmap_{output_prefix}",
                params
            )
            
        elif analysis_type == 'sunburst':
            # Sunburst-Visualisierung erstellen
            print("\nErstelle Sunburst-Visualisierung...")
            analyzer.create_sunburst(
                filtered_df,
                f"Sunburst_{output_prefix}",
                params
            )
            
        elif analysis_type == 'treemap':
            # Treemap-Visualisierung erstellen
            print("\nErstelle Treemap-Visualisierung...")
            analyzer.create_treemap(
                filtered_df,
                f"Treemap_{output_prefix}",
                params
            )
            
        elif analysis_type == 'summary_paraphrase':
            # Paraphrasenzusammenfassung erstellen
            print("\nErstelle Paraphrasen-Zusammenfassung...")
            # Verwende benutzerdefinierten Prompt, falls vorhanden
            prompt_template = params.get('prompt_template', default_prompts.get('paraphrase', None))
            
            await analyzer.create_custom_summary(
                filtered_df,
                prompt_template,
                f"Summary_Paraphrase_{output_prefix}",
                MODEL_NAME,
                TEMPERATURE,
                filters,
                params
            )
            
        elif analysis_type == 'summary_reasoning':
            # Begründungszusammenfassung erstellen
            print("\nErstelle Begründungs-Zusammenfassung...")
            # Verwende benutzerdefinierten Prompt, falls vorhanden
            prompt_template = params.get('prompt_template', default_prompts.get('reasoning', None))
            
            await analyzer.create_custom_summary(
                filtered_df,
                prompt_template,
                f"Summary_Begruendung_{output_prefix}",
                MODEL_NAME,
                TEMPERATURE,
                filters,
                params
            )
            
        elif analysis_type == 'custom_summary':
            # Benutzerdefinierte Zusammenfassung erstellen
            print("\nErstelle benutzerdefinierte Zusammenfassung...")
            # Muss einen benutzerdefinierten Prompt haben
            prompt_template = params.get('prompt_template', '')
            if not prompt_template:
                print("Warnung: Kein Prompt-Template für benutzerdefinierte Zusammenfassung angegeben. Überspringe.")
                continue
                
            await analyzer.create_custom_summary(
                filtered_df,
                prompt_template,
                f"Summary_Custom_{output_prefix}",
                MODEL_NAME,
                TEMPERATURE,
                filters,
                params
            )

        elif analysis_type == 'sentiment_analysis':
            # Sentiment-Analyse erstellen
            print("\nFühre Sentiment-Analyse durch...")
            await analyzer.create_sentiment_analysis(
                filtered_df,
                f"Sentiment_{output_prefix}",
                params
            )  
        else:
            print(f"Warnung: Unbekannter Analysetyp '{analysis_type}'. Überspringe.")
    
    print("\n=== QCA Analyse abgeschlossen ===\n")
