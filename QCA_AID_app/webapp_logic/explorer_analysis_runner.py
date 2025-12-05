"""
Explorer Analysis Runner
========================
Manages QCA-AID Explorer analysis execution with configuration support.

This module integrates the ExplorerConfigData with the QCAAnalyzer to execute
analyses based on the configured parameters, filters, and active states.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from QCA_AID_assets.analysis.qca_analyzer import QCAAnalyzer
from QCA_AID_assets.utils.llm.factory import LLMProviderFactory
from QCA_AID_assets.utils.prompts import get_default_prompts
from QCA_AID_assets.utils.common import create_filter_string
from webapp_models.explorer_config_data import ExplorerConfigData, AnalysisConfig


class ExplorerAnalysisRunner:
    """
    Manages execution of QCA-AID Explorer analyses based on configuration.
    
    This class integrates with ExplorerConfigData to:
    - Execute only active analyses
    - Apply filters before analysis execution
    - Pass parameters to analysis functions
    - Handle errors and provide progress feedback
    
    Requirements:
    - 1.3: Apply filters to data before running specific analysis
    - 1.4: Use configured parameters when executing analysis
    - 2.1: Execute analysis when active parameter is true
    - 2.2: Skip analysis when active parameter is false
    """
    
    def __init__(self, excel_path: str, config_data: ExplorerConfigData):
        """
        Initialize the Explorer Analysis Runner.
        
        Args:
            excel_path: Path to the Excel file with coded segments
            config_data: ExplorerConfigData with base config and analysis configs
        """
        self.excel_path = excel_path
        self.config_data = config_data
        self.analyzer: Optional[QCAAnalyzer] = None
        self.llm_provider = None
        self.default_prompts = get_default_prompts()
        
    async def initialize(self) -> Tuple[bool, List[str]]:
        """
        Initialize the LLM provider and QCAAnalyzer.
        
        Returns:
            Tuple[bool, List[str]]: (success, messages)
        """
        messages = []
        
        try:
            # Initialize LLM provider from base_config (Requirement 1.5)
            provider_name = self.config_data.base_config.get('provider', 'openai')
            model_name = self.config_data.base_config.get('model', 'gpt-4o-mini')
            messages.append(f"Initialisiere LLM Provider: {provider_name} mit Modell: {model_name}")
            
            self.llm_provider = LLMProviderFactory.create_provider(provider_name, model_name=model_name)
            
            # Initialize analyzer
            messages.append(f"Lade Excel-Datei: {self.excel_path}")
            self.analyzer = QCAAnalyzer(
                self.excel_path, 
                self.llm_provider, 
                self.config_data.base_config
            )
            
            messages.append(f"Verfügbare Spalten: {', '.join(self.analyzer.columns)}")
            
            return True, messages
            
        except FileNotFoundError as e:
            error_msg = f"Datei nicht gefunden: {str(e)}"
            messages.append(error_msg)
            return False, messages
        except Exception as e:
            error_msg = f"Fehler bei Initialisierung: {str(e)}"
            messages.append(error_msg)
            return False, messages
    
    async def run_all_analyses(self) -> Tuple[bool, List[str]]:
        """
        Execute all configured analyses.
        
        This method:
        - Checks active state of each analysis (Requirement 2.1, 2.2)
        - Applies filters before execution (Requirement 1.3)
        - Passes parameters to analysis functions (Requirement 1.4)
        
        Returns:
            Tuple[bool, List[str]]: (success, messages with progress and errors)
        """
        if not self.analyzer:
            return False, ["Analyzer nicht initialisiert. Bitte initialize() aufrufen."]
        
        messages = []
        
        # Perform selective keyword harmonization
        clean_keywords = self.config_data.base_config.get('clean_keywords', True)
        if isinstance(clean_keywords, str):
            clean_keywords = clean_keywords.lower() in ('true', 'ja', 'yes', '1')
        
        similarity_threshold = float(self.config_data.base_config.get('similarity_threshold', 0.7))
        
        harmonization_performed = self.analyzer.perform_selective_harmonization(
            [self._convert_analysis_config_to_dict(ac) for ac in self.config_data.analysis_configs],
            clean_keywords,
            similarity_threshold
        )
        
        if harmonization_performed:
            messages.append("Keyword-Harmonisierung durchgeführt")
        
        # Count active analyses
        active_analyses = [ac for ac in self.config_data.analysis_configs if ac.active]
        total_analyses = len(active_analyses)
        
        messages.append(f"Führe {total_analyses} aktive Auswertungen durch...")
        
        # Execute each analysis
        for idx, analysis_config in enumerate(self.config_data.analysis_configs, 1):
            # Requirement 2.2: Skip inactive analyses
            if not analysis_config.active:
                messages.append(f"Überspringe deaktivierte Auswertung: {analysis_config.name}")
                continue
            
            # Requirement 2.1: Execute active analyses
            messages.append(f"--- Auswertung {idx}/{len(self.config_data.analysis_configs)}: {analysis_config.name} ---")
            
            try:
                success, analysis_messages = await self._execute_single_analysis(analysis_config)
                messages.extend(analysis_messages)
                
                if not success:
                    messages.append(f"⚠️ Fehler bei Analyse '{analysis_config.name}'")
                else:
                    messages.append(f"✓ Analyse '{analysis_config.name}' erfolgreich abgeschlossen")
                    
            except Exception as e:
                error_msg = f"Fehler bei Analyse '{analysis_config.name}': {str(e)}"
                messages.append(error_msg)
        
        messages.append("=== Alle Analysen abgeschlossen ===")
        return True, messages
    
    async def _execute_single_analysis(self, analysis_config: AnalysisConfig) -> Tuple[bool, List[str]]:
        """
        Execute a single analysis configuration.
        
        Args:
            analysis_config: The analysis configuration to execute
            
        Returns:
            Tuple[bool, List[str]]: (success, messages)
        """
        messages = []
        
        # Requirement 1.3: Apply filters before analysis execution
        messages.append(f"Wende Filter an für '{analysis_config.name}'...")
        filtered_df = self.analyzer.filter_data(analysis_config.filters)
        
        # Check if filtering resulted in empty dataset
        if filtered_df.empty:
            messages.append(f"⚠️ WARNUNG: Keine Daten nach Filterung für Analyse '{analysis_config.name}'")
            messages.append(f"   Angewendete Filter: {analysis_config.filters}")
            messages.append(f"   Diese Analyse wird übersprungen.")
            return False, messages
        
        messages.append(f"✓ {len(filtered_df)} Datensätze nach Filterung gefunden")
        
        # Generate filter string for output filename
        filter_str = create_filter_string(analysis_config.filters)
        
        # Add harmonization info if relevant
        if self.analyzer.needs_keyword_harmonization(analysis_config.analysis_type, analysis_config.params):
            if self.analyzer.keyword_mappings:
                filter_str = f"harmonized_{filter_str}" if filter_str else "harmonized"
        
        # Create output prefix
        output_prefix = f"{analysis_config.name}_{filter_str}" if filter_str else analysis_config.name
        
        # Requirement 1.4: Pass parameters to analysis functions
        # Execute based on analysis type
        try:
            if analysis_config.analysis_type == 'netzwerk':
                messages.append("Erstelle Netzwerk-Visualisierung...")
                self.analyzer.create_network_graph(
                    filtered_df,
                    f"Code-Network_{output_prefix}",
                    analysis_config.params
                )
                
            elif analysis_config.analysis_type == 'heatmap':
                messages.append("Erstelle Heatmap...")
                self.analyzer.create_heatmap(
                    filtered_df,
                    f"Heatmap_{output_prefix}",
                    analysis_config.params
                )
                
            elif analysis_config.analysis_type == 'sunburst':
                messages.append("Erstelle Sunburst-Visualisierung...")
                self.analyzer.create_sunburst(
                    filtered_df,
                    f"Sunburst_{output_prefix}",
                    analysis_config.params
                )
                
            elif analysis_config.analysis_type == 'treemap':
                messages.append("Erstelle Treemap-Visualisierung...")
                self.analyzer.create_treemap(
                    filtered_df,
                    f"Treemap_{output_prefix}",
                    analysis_config.params
                )
                
            elif analysis_config.analysis_type == 'summary_paraphrase':
                messages.append("Erstelle Paraphrasen-Zusammenfassung...")
                prompt_template = analysis_config.params.get(
                    'prompt_template', 
                    self.default_prompts.get('paraphrase', None)
                )
                
                await self.analyzer.create_custom_summary(
                    filtered_df,
                    prompt_template,
                    f"Summary_Paraphrase_{output_prefix}",
                    self.config_data.base_config.get('model', 'gpt-4o-mini'),
                    float(self.config_data.base_config.get('temperature', 0.7)),
                    analysis_config.filters,
                    analysis_config.params
                )
                
            elif analysis_config.analysis_type == 'summary_reasoning':
                messages.append("Erstelle Begründungs-Zusammenfassung...")
                prompt_template = analysis_config.params.get(
                    'prompt_template',
                    self.default_prompts.get('reasoning', None)
                )
                
                await self.analyzer.create_custom_summary(
                    filtered_df,
                    prompt_template,
                    f"Summary_Begruendung_{output_prefix}",
                    self.config_data.base_config.get('model', 'gpt-4o-mini'),
                    float(self.config_data.base_config.get('temperature', 0.7)),
                    analysis_config.filters,
                    analysis_config.params
                )
                
            elif analysis_config.analysis_type == 'custom_summary':
                messages.append("Erstelle benutzerdefinierte Zusammenfassung...")
                prompt_template = analysis_config.params.get('prompt_template', '')
                
                if not prompt_template:
                    messages.append("⚠️ Kein Prompt-Template für benutzerdefinierte Zusammenfassung angegeben")
                    return False, messages
                
                await self.analyzer.create_custom_summary(
                    filtered_df,
                    prompt_template,
                    f"Summary_Custom_{output_prefix}",
                    self.config_data.base_config.get('model', 'gpt-4o-mini'),
                    float(self.config_data.base_config.get('temperature', 0.7)),
                    analysis_config.filters,
                    analysis_config.params
                )
                
            elif analysis_config.analysis_type == 'sentiment_analysis':
                messages.append("Führe Sentiment-Analyse durch...")
                await self.analyzer.create_sentiment_analysis(
                    filtered_df,
                    f"Sentiment_{output_prefix}",
                    analysis_config.params
                )
                
            else:
                messages.append(f"⚠️ Unbekannter Analysetyp '{analysis_config.analysis_type}'")
                return False, messages
            
            return True, messages
            
        except Exception as e:
            error_msg = f"Fehler bei Ausführung: {str(e)}"
            messages.append(error_msg)
            return False, messages
    
    def _convert_analysis_config_to_dict(self, analysis_config: AnalysisConfig) -> Dict[str, Any]:
        """
        Convert AnalysisConfig to dictionary format expected by QCAAnalyzer.
        
        Args:
            analysis_config: AnalysisConfig instance
            
        Returns:
            Dictionary with 'name', 'filters', and 'params' keys
        """
        return {
            'name': analysis_config.name,
            'filters': analysis_config.filters,
            'params': {
                **analysis_config.params,
                'active': analysis_config.active,
                'analysis_type': analysis_config.analysis_type
            }
        }
