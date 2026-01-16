"""
QCA Analyzer Module
===================

This module contains the QCAAnalyzer class for analyzing qualitative coding data.
It provides functionality for filtering data, harmonizing keywords, and creating
various visualizations and analyses.
"""

# Standard library imports
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

# Local imports
from QCA_AID_assets.utils.llm.base import LLMProvider
from QCA_AID_assets.utils.llm.response import LLMResponse
from QCA_AID_assets.utils.visualization.layout import create_forceatlas_like_layout


class QCAAnalyzer:
    """
    Analyzer for qualitative coding data with support for network visualization,
    heatmaps, custom summaries, and sentiment analysis.
    
    This class provides comprehensive analysis capabilities for coded qualitative data,
    including keyword harmonization, data filtering, and various visualization methods.
    
    Attributes:
        df (pd.DataFrame): The main dataframe containing coded segments
        llm_provider (LLMProvider): Instance of LLM provider for text generation
        keyword_mappings (Dict[str, str]): Mapping of harmonized keywords
        output_dir (Path): Directory for output files
        columns (List[str]): List of column names in the dataframe
    """
    
    def __init__(self, excel_path: str, llm_provider: LLMProvider, config: Dict[str, Any]):
        """
        Initialize the QCA Analyzer.
        
        Args:
            excel_path: Path to the Excel file with coded segments
            llm_provider: Instance of LLMProvider for text generation
            config: Dictionary with base configuration including output directory
        """
        # Read specifically from 'Kodierungsergebnisse' sheet
        self.df = pd.read_excel(excel_path, sheet_name='Kodierungsergebnisse')
        self.llm_provider = llm_provider
        self.keyword_mappings = {}
        
        # Normalize column names to handle encoding issues
        # Map common variations to standard names
        column_mapping = {}
        for col in self.df.columns:
            col_lower = col.lower()
            if 'schlüssel' in col_lower or 'schlussel' in col_lower:
                column_mapping[col] = 'Schlüsselwörter'
            elif 'begründ' in col_lower or 'begrund' in col_lower:
                column_mapping[col] = 'Begründung'
        
        # Rename columns if needed
        if column_mapping:
            self.df.rename(columns=column_mapping, inplace=True)
            print(f"Spaltennamen normalisiert: {column_mapping}")
        
        # Get the input filename without extension
        input_filename = Path(excel_path).stem

        # Set output directory from config or use default
        # If script_dir is provided in config, use it; otherwise determine from current location
        if 'script_dir' in config and config['script_dir']:
            script_dir = config['script_dir']
        else:
            # We're in QCA_AID_assets/analysis/, so go up two levels to get to root
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        output_dir = config.get('output_dir', 'output')
        self.base_output_dir = Path(script_dir) / output_dir
        self.base_output_dir.mkdir(exist_ok=True)

        # Create analysis-specific subdirectory
        self.output_dir = self.base_output_dir / input_filename
        self.output_dir.mkdir(exist_ok=True)
        
        # Store column names for easy access
        self.columns = list(self.df.columns)

    def filter_data(self, filters: Dict[str, str]) -> pd.DataFrame:
        """
        Filter the dataframe based on provided column-value pairs.
        
        Supports both column names and generic attribute designations (Attribut_1, Attribut_2, etc.).
        
        Args:
            filters: Dictionary with column-value pairs for filtering
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = self.df.copy()
        
        # Create mapping from generic attribute names to actual column names
        # Columns B, C, D correspond to Attribut_1, Attribut_2, Attribut_3
        attribute_mapping = {
            'Attribut_1': self.df.columns[1] if len(self.df.columns) > 1 else None,  # Column B (Index 1)
            'Attribut_2': self.df.columns[2] if len(self.df.columns) > 2 else None,  # Column C (Index 2)
            'Attribut_3': self.df.columns[3] if len(self.df.columns) > 3 else None,  # Column D (Index 3)
        }
        
        # Check if any generic attributes are used in filters
        uses_generic_attributes = any(col in attribute_mapping for col in filters.keys())
        
        # Show mapping only if generic attributes are actually used
        if uses_generic_attributes:
            print("\nSpalten-Mapping:")
            for attr, col in attribute_mapping.items():
                if col:
                    print(f"  {attr} → {col}")
        
        # Check which filters can be applied
        applicable_filters = {}
        for col, value in filters.items():
            # Skip empty filters
            if not value or pd.isna(value):
                continue
                
            # Check if it's a generic attribute
            actual_col = col
            if col in attribute_mapping and attribute_mapping[col]:
                actual_col = attribute_mapping[col]
                if uses_generic_attributes:
                    print(f"Filter '{col}' wird auf Spalte '{actual_col}' angewendet")
                
            # Check if column exists
            if actual_col not in self.df.columns:
                print(f"Warnung: Spalte '{actual_col}' (von Filter '{col}') nicht in den Daten gefunden. Filter wird übersprungen.")
                continue
                
            applicable_filters[actual_col] = value
        
        if not applicable_filters:
            print("Keine gültigen Filter gefunden. Verwende ungefilterte Daten.")
            return filtered_df
        
        # Apply valid filters
        for col, value in applicable_filters.items():
            # Special handling depending on column type
            if col in ['Subkategorien', 'Subkategorie']:
                # Special handling for subcategories (comma-separated lists)
                # Support multiple subcategories in filter (comma-separated)
                filter_subcats = [v.strip() for v in str(value).split(',') if v.strip()]
                if filter_subcats:
                    filtered_df = filtered_df[filtered_df[col].fillna('').str.split(',').apply(
                        lambda x: any(subcat in [item.strip() for item in x] for subcat in filter_subcats)
                    )]
            elif col in ['Hauptkategorie']:
                # Special handling for main categories - support multiple categories (comma-separated)
                filter_categories = [v.strip() for v in str(value).split(',') if v.strip()]
                if filter_categories:
                    filtered_df = filtered_df[filtered_df[col].fillna('').astype(str).isin(filter_categories)]
            else:
                # Convert both sides to string for robust comparison
                value_str = str(value).strip()
                
                # Check different matching strategies
                if value_str == '*' or value_str.lower() == 'alle':
                    # If filter value is '*' or 'alle', skip
                    continue
                elif '*' in value_str:
                    # Wildcard matching (e.g., "Text*" for text beginning)
                    pattern = value_str.replace('*', '.*')
                    filtered_df = filtered_df[filtered_df[col].fillna('').astype(str).str.match(pattern, case=False)]
                else:
                    # Exact matching - convert both sides to string
                    filtered_df = filtered_df[filtered_df[col].astype(str) == value_str]
        
        # Debug output
        if applicable_filters:
            print(f"\nFilter angewendet:")
            for col, value in applicable_filters.items():
                print(f"- {col}: {value}")
        print(f"Anzahl der ursprünglichen Zeilen: {len(self.df)}")
        print(f"Anzahl der gefilterten Zeilen: {len(filtered_df)}")
        
        # Warning for empty result
        if len(filtered_df) == 0:
            print("WARNUNG: Die Filterung ergab keine Ergebnisse!")
            # Show available values for filter columns
            print("\nverfügbare Werte in den Filter-Spalten:")
            for col in applicable_filters.keys():
                if col in self.df.columns:
                    unique_values = self.df[col].value_counts().head(10)
                    print(f"\n{col} (Top 10):")
                    for val, count in unique_values.items():
                        print(f"  - {val}: {count} Vorkommen")
        
        return filtered_df

    def get_column_mapping(self) -> Dict[str, str]:
        """
        Get the mapping from generic attribute names to actual column names.
        
        Returns:
            Dictionary with attribute-to-column mapping
        """
        return {
            'Attribut_1': self.df.columns[1] if len(self.df.columns) > 1 else None,
            'Attribut_2': self.df.columns[2] if len(self.df.columns) > 2 else None,
            'Attribut_3': self.df.columns[3] if len(self.df.columns) > 3 else None,
        }

    def print_available_filters(self):
        """
        Display all available columns and their values that can be used as filters.
        """
        print("\n=== verfügbare Filter ===")
        
        # Show generic attributes
        attribute_mapping = self.get_column_mapping()
        print("\nGenerische Attribute:")
        for attr, col in attribute_mapping.items():
            if col:
                print(f"  {attr} → {col}")
                # Show first 5 unique values
                unique_values = self.df[col].value_counts().head(5)
                for val, count in unique_values.items():
                    print(f"    - {val} ({count} Vorkommen)")
        
        # Show standard columns
        standard_columns = ['Hauptkategorie', 'Subkategorien', 'Schlüsselwörter', 'Dokument']
        print("\nStandard-Spalten:")
        for col in standard_columns:
            if col in self.df.columns:
                print(f"  {col}")
                unique_values = self.df[col].value_counts().head(5)
                for val, count in unique_values.items():
                    # For subcategories show only first 50 characters
                    val_display = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
                    print(f"    - {val_display} ({count} Vorkommen)")
        
        # Show other columns
        other_columns = [col for col in self.df.columns if col not in standard_columns and col not in attribute_mapping.values()]
        if other_columns:
            print("\nWeitere Spalten:")
            for col in other_columns:
                print(f"  {col}")
    
    def _filter_not_coded(self, df: pd.DataFrame, exclude: bool = True) -> pd.DataFrame:
        """
        Helper method to filter out "nicht kodiert" entries from a dataframe.
        
        Args:
            df: DataFrame to filter
            exclude: If True, exclude "nicht kodiert" entries; if False, return df unchanged
            
        Returns:
            Filtered DataFrame
        """
        if not exclude:
            return df
        
        original_count = len(df)
        
        # Robust filtering that handles NaN values
        if 'Hauptkategorie' in df.columns:
            # Create a mask that excludes "nicht kodiert" and "kein kodierkonsens"
            # Handle NaN values by filling them with empty string for comparison
            mask = ~df['Hauptkategorie'].fillna('').str.lower().isin(['nicht kodiert', 'kein kodierkonsens'])
            df = df[mask].copy()
            
            if len(df) < original_count:
                excluded_count = original_count - len(df)
                print(f"Hinweis: {excluded_count} 'Nicht kodiert' Einträge wurden automatisch ausgeschlossen")
        else:
            print("Warnung: Spalte 'Hauptkategorie' nicht gefunden - kann 'Nicht kodiert' nicht filtern")
        
        return df

    async def generate_summary(self, 
                             text: str, 
                             prompt_template: str, 
                             model: str,
                             temperature: float = 0.7,
                             **kwargs) -> str:
        """
        Generate summary using configured LLM provider.
        
        Args:
            text: Text to summarize
            prompt_template: Template for the prompt
            model: Name of the LLM model to use
            temperature: Temperature parameter for generation
            **kwargs: Additional keyword arguments for prompt formatting
            
        Returns:
            Generated summary text
        """
        # Format filters if present
        if 'filters' in kwargs:
            kwargs['filters'] = self._format_filters_for_prompt(kwargs['filters'])

        format_args = {'text': text, **kwargs}
        prompt = prompt_template.format(**format_args)
        
        messages = [
            {"role": "system", "content": "Sie sind ein hilfreicher Forschungsassistent, der sich mit der Analyse qualitativer Daten auskennt."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.llm_provider.create_completion(
            messages=messages,
            model=model,
            temperature=temperature
        )
        
        llm_response = LLMResponse(response)
        return llm_response.content

    def save_text_summary(self, summary: str, filename: str):
        """
        Save summary to text file.
        
        Args:
            summary: Summary text to save
            filename: Name of the output file
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)

    def needs_keyword_harmonization(self, analysis_type: str, params: Dict[str, Any] = None) -> bool:
        """
        Determine if keyword harmonization is required for a specific analysis type.
        
        Args:
            analysis_type: The type of analysis
            params: Additional analysis parameters
            
        Returns:
            True if harmonization is needed, False otherwise
        """
        # List of analysis types that need keyword harmonization
        keyword_dependent_analyses = [
            'netzwerk',           # Network visualization uses keywords
            'network',            # Alternative spelling
            'keyword_analysis',   # Future keyword analysis
            'keyword_cloud',      # Future word cloud
            'sentiment_keywords', # Sentiment analysis with keywords
        ]
        
        # Check analysis type
        if analysis_type.lower() in keyword_dependent_analyses:
            return True
        
        # Check parameters for special cases
        if params:
            # If keywords are explicitly mentioned in parameters
            if params.get('use_keywords', False):
                return True
            
            # If analysis uses keywords in any form
            if any('keyword' in str(key).lower() or 'schlüsselwört' in str(key).lower() 
                for key in params.keys()):
                return True
        
        return False

    def perform_selective_harmonization(self, analysis_configs: List[Dict[str, Any]], 
                                    clean_keywords: bool, 
                                    similarity_threshold: float) -> bool:
        """
        Perform keyword harmonization only if at least one analysis requires it.
        
        Args:
            analysis_configs: List of analysis configurations
            clean_keywords: Whether harmonization is globally enabled
            similarity_threshold: Threshold for similarity matching
            
        Returns:
            True if harmonization was performed, False otherwise
        """
        if not clean_keywords:
            print("Keyword-Harmonisierung ist global deaktiviert.")
            return False
        
        # Check if any active analysis needs harmonization
        needs_harmonization = False
        analyses_requiring_harmonization = []
        
        for config in analysis_configs:
            # Check if analysis is active
            is_active = config.get('params', {}).get('active', True)
            if isinstance(is_active, str):
                is_active = is_active.lower() in ('true', 'ja', 'yes', '1')
            
            if not is_active:
                continue
            
            analysis_type = config.get('params', {}).get('analysis_type', '')
            analysis_name = config.get('name', 'Unbekannt')
            
            if self.needs_keyword_harmonization(analysis_type, config.get('params', {})):
                needs_harmonization = True
                analyses_requiring_harmonization.append(analysis_name)
        
        if needs_harmonization:
            print(f"\nKeyword-Harmonisierung wird durchgeführt für folgende Analysen:")
            for analysis in analyses_requiring_harmonization:
                print(f"  - {analysis}")
            
            print("\nStarte Keyword-Harmonisierung...")
            # Update the main DataFrame with harmonized keywords
            self.df = self.harmonize_keywords(similarity_threshold=similarity_threshold)
            self.validate_keyword_mapping()
            print("Keyword-Harmonisierung abgeschlossen.")
            return True
        else:
            print("\nKeyword-Harmonisierung wird übersprungen (von keiner aktiven Analyse benötigt).")
            return False
    
    def harmonize_keywords(self, similarity_threshold: float = 0.60) -> pd.DataFrame:
        """
        Harmonize keywords using fuzzy matching while preserving hierarchical relationships.
        
        Args:
            similarity_threshold: Threshold for fuzzy matching (default: 0.60)
                
        Returns:
            DataFrame with harmonized keywords
        """
        def get_similarity(a: str, b: str) -> float:
            """Calculate string similarity with enhanced rules for academic terms."""
            if not isinstance(a, str) or not isinstance(b, str):
                return 0.0
                    
            # Normalize strings
            a = a.lower().strip()
            b = b.lower().strip()
                
            # Base similarity with SequenceMatcher
            base_similarity = SequenceMatcher(None, a, b).ratio()
            
            return min(1.0, base_similarity)

        def find_most_common_variant(keywords: list) -> dict:
            """Find the most common variant with improved academic context handling."""
            keyword_mapping = {}
            processed = set()
                
            # Flatten and clean keywords
            flat_keywords = []
            for kw_list in keywords:
                if pd.notna(kw_list):
                    flat_keywords.extend([k.strip() for k in str(kw_list).split(',') if k.strip()])
                
            # Convert to frequency series and sort
            keyword_freq = pd.Series(flat_keywords).value_counts()
            
            # Define preferred forms for certain terms
            preferred_forms = {
                # Add preferred forms here if needed
            }
            
            print("\nAnalyzing keyword similarities...")
            
            # Store potential matches for diagnostics
            potential_matches = []
            
            for keyword in keyword_freq.index:
                if keyword in processed:
                    continue
                
                similar_keywords = []
                
                # Find similar keywords
                for other_keyword in keyword_freq.index:
                    if other_keyword != keyword and other_keyword not in processed:
                        similarity = get_similarity(keyword, other_keyword)
                        if similarity >= similarity_threshold:
                            similar_keywords.append(other_keyword)
                            processed.add(other_keyword)
                            potential_matches.append((keyword, other_keyword, similarity))
                
                if similar_keywords:
                    # Check if any preferred form exists
                    normalized_key = keyword.lower()
                    preferred_form = next(
                        (pref for key, pref in preferred_forms.items() 
                        if key in normalized_key),
                        None
                    )
                    
                    if preferred_form:
                        canonical = preferred_form
                    else:
                        # Use most frequent variant
                        canonical = max(similar_keywords + [keyword], 
                                    key=lambda x: keyword_freq[x])
                    
                    for kw in similar_keywords + [keyword]:
                        keyword_mapping[kw] = canonical
            
            # Print diagnostic information
            if potential_matches:
                print("\nPotential keyword matches found (showing pairs with similarity >= threshold):")
                for kw1, kw2, sim in sorted(potential_matches, key=lambda x: x[2], reverse=True):
                    print(f"  '{kw1}' ↔ '{kw2}' (similarity: {sim:.3f})")
            else:
                print("\nNo keyword pairs met the similarity threshold.")
                
            # Store mappings in class instance
            self.keyword_mappings.update(keyword_mapping)
            return keyword_mapping

        # Create a copy of the DataFrame
        harmonized_df = self.df.copy()
            
        # Collect all keywords from the entire DataFrame
        all_keywords = []
        for kw_list in harmonized_df['Schlüsselwörter'].dropna():
            all_keywords.extend([k.strip() for k in str(kw_list).split(',') if k.strip()])

        # Generate global keyword mappings
        keyword_mapping = find_most_common_variant(all_keywords)
        self.keyword_mappings.update(keyword_mapping)

        # Apply mappings to all rows
        def replace_keywords(x):
            """
            Replace keywords in a cell according to the keyword mapping.
            
            Args:
                x: Cell value containing comma-separated keywords
                
            Returns:
                String with mapped keywords, duplicates removed
            """
            if pd.isna(x):
                return x
            original = [k.strip() for k in str(x).split(',')]
            mapped = [keyword_mapping.get(k, k) for k in original]
            seen = set()
            return ','.join([k for k in mapped if not (k in seen or seen.add(k))])

        harmonized_df['Schlüsselwörter'] = harmonized_df['Schlüsselwörter'].apply(replace_keywords)

        # Update main DataFrame
        self.df = harmonized_df
        return self.df
    
    def validate_keyword_mapping(self):
        """Validate keyword mappings and print diagnostics."""
        mapping = defaultdict(set)
        
        for kw in self.df['Schlüsselwörter'].dropna():
            for keyword in str(kw).split(','):
                cleaned = keyword.strip().lower()
                mapping[cleaned].add(keyword.strip())
        
        print("\nSchlüsselwort-Zuordnungen:")
        for canonical, variants in mapping.items():
            if len(variants) > 1:
                print(f"Kanonisch: {canonical}")
                print(f"Varianten: {', '.join(variants)}")
                print("------")

    def create_network_graph(self, filtered_df: pd.DataFrame, output_filename: str, params: Dict[str, Any] = None):
        """
        Create network graph visualization using harmonized keywords.
        
        Creates a hierarchical network visualization showing relationships between
        main categories, subcategories, and keywords.
        
        Args:
            filtered_df: Filtered DataFrame to visualize
            output_filename: Base name for output files
            params: Optional parameters for customization (node_size_factor, layout_iterations, etc.)
        """
        print("Erstelle Netzwerkgraph...")
        
        # Check if filtered_df is empty
        if filtered_df.empty:
            print("WARNUNG: Keine Daten nach Filterung vorhanden!")
            print("Der Graph kann nicht erstellt werden.")
            print("Bitte überprüfen Sie die Filter-Einstellungen in der Konfiguration.")
            return
        
        # Use parameters from configuration or default values
        if params is None:
            params = {}
        
        # Filter out "nicht kodiert" entries (default: True for visualizations)
        exclude_not_coded = params.get('exclude_not_coded', True)
        filtered_df = self._filter_not_coded(filtered_df, exclude_not_coded)
        
        # Check again if filtered_df is empty after removing "nicht kodiert"
        if filtered_df.empty:
            print("WARNUNG: Keine kodierten Daten vorhanden (nur 'Nicht kodiert' Einträge)!")
            print("Der Graph kann nicht erstellt werden.")
            return
        
        # Auto-adjust parameters based on data characteristics
        params = self._adjust_network_parameters(filtered_df, params)
        
        # Extract parameters with fallback values
        node_size_factor = float(params.get('node_size_factor', 10))  # Default value 10
        iterations = int(params.get('layout_iterations', 100))
        gravity = float(params.get('gravity', 0.05))
        scaling = float(params.get('scaling', 2.0))
        label_offset = float(params.get('label_offset', 1.0))  # New parameter for label distance
        scaling = float(params.get('scaling', 2.0))
        
        # Optional: set background color from parameter
        bg_color = params.get('bg_color', 'white')  # Default: white background instead of pink
        
        G = nx.DiGraph()
        
        # Debug: Print mappings
        print("\nAktive Keyword-Mappings:")
        for original, harmonized in self.keyword_mappings.items():
            if original != harmonized:
                print(f"  '{original}' → '{harmonized}'")
        
        # Count occurrences for node sizing
        category_counts = {}
        subcategory_counts = {}
        keyword_counts = {}
        
        # Color mapping for main categories with default color
        unique_main_categories = filtered_df['Hauptkategorie'].unique()
        if len(unique_main_categories) > 0:
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_main_categories)))
            main_category_colors = dict(zip(unique_main_categories, colors))
            default_color = colors[0]  # Use first color as default
        else:
            # If no main categories, use a default color
            default_color = plt.cm.Set3(0)
            main_category_colors = {}
        
        nodes_data = []
        edges_data = []
        
        # Process each row
        for _, row in filtered_df.iterrows():
            if pd.isna(row['Hauptkategorie']):
                continue
                    
            main_category = str(row['Hauptkategorie'])
            category_counts[main_category] = category_counts.get(main_category, 0) + 1
            
            if main_category not in [node[0] for node in nodes_data]:
                G.add_node(main_category, node_type='main')
                nodes_data.append([main_category, 'main', category_counts[main_category]])

            # Handle subcategories
            if pd.notna(row['Subkategorien']):
                sub_categories = [cat.strip() for cat in str(row['Subkategorien']).split(',') if cat.strip()]
                
                for sub_cat in sub_categories:
                    subcategory_counts[sub_cat] = subcategory_counts.get(sub_cat, 0) + 1
                    
                    if sub_cat not in [node[0] for node in nodes_data]:
                        G.add_node(sub_cat, node_type='sub')
                        nodes_data.append([sub_cat, 'sub', subcategory_counts[sub_cat]])
                    
                    if not G.has_edge(main_category, sub_cat):
                        G.add_edge(main_category, sub_cat)
                        edges_data.append([main_category, sub_cat, 'main_to_sub'])
                    
                    # Handle keywords
                    if pd.notna(row['Schlüsselwörter']):
                        keywords = [kw.strip() for kw in str(row['Schlüsselwörter']).split(',') if kw.strip()]
                        
                        for keyword in keywords:
                            # Use harmonized version of keyword if available
                            harmonized_keyword = self.keyword_mappings.get(keyword, keyword)
                            
                            # Update counts using harmonized keyword
                            keyword_counts[harmonized_keyword] = keyword_counts.get(harmonized_keyword, 0) + 1
                            
                            # Add node using harmonized keyword
                            if harmonized_keyword not in [node[0] for node in nodes_data]:
                                G.add_node(harmonized_keyword, node_type='keyword')
                                nodes_data.append([harmonized_keyword, 'keyword', keyword_counts[harmonized_keyword]])
                            
                            # Add edge using harmonized keyword
                            if not G.has_edge(sub_cat, harmonized_keyword):
                                G.add_edge(sub_cat, harmonized_keyword)
                                edges_data.append([sub_cat, harmonized_keyword, 'sub_to_keyword'])
        
        # Export network data
        print("\nExportiere Netzwerkdaten...")
        nodes_df = pd.DataFrame(nodes_data, columns=['Node', 'Type', 'Count'])
        edges_df = pd.DataFrame(edges_data, columns=['Source', 'Target', 'Type'])
        
        # Print statistics
        print(f"\nStatistiken:")
        print(f"Anzahl Knoten: {len(nodes_df)}")
        print(f"Anzahl Kanten: {len(edges_df)}")
        print("\nKnotentypen Verteilung:")
        print(nodes_df['Type'].value_counts())
        
        # Export to Excel with error handling
        safe_filename = output_filename.replace('/', '_').replace('\\', '_').replace(':', '_')
        excel_output_path = self.output_dir / f"{safe_filename}_network_data.xlsx"
        
        try:
            with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
                nodes_df.to_excel(writer, sheet_name='Nodes', index=False)
                edges_df.to_excel(writer, sheet_name='Edges', index=False)
            print(f"✓ Netzwerkdaten exportiert nach: {excel_output_path}")
        except Exception as e:
            print(f"❌ Fehler beim Exportieren der Netzwerkdaten: {str(e)}")
            # Continue with visualization even if Excel export fails
        
        # Create visualization
        print("\nErstelle Visualisierung...")
        plt.figure(figsize=(24, 20), facecolor=bg_color)

        # Use ForceAtlas2-like layout
        print("\nBerechne Knotenpositionen mit ForceAtlas2-ähnlichem Layout...")
        
        # Create initial positions for better convergence
        initial_pos = {}
        
        # Position main categories in center
        main_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'main']
        num_main = len(main_nodes)
        for i, node in enumerate(main_nodes):
            if num_main > 1:
                angle = 2 * np.pi * i / num_main
                initial_pos[node] = (0.2 * np.cos(angle), 0.2 * np.sin(angle))
            else:
                initial_pos[node] = (0, 0)
        
        # Position subcategories in middle ring
        sub_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'sub']
        num_sub = len(sub_nodes)
        for i, node in enumerate(sub_nodes):
            if num_sub > 1:
                angle = 2 * np.pi * i / num_sub
                initial_pos[node] = (0.5 * np.cos(angle), 0.5 * np.sin(angle))
            else:
                initial_pos[node] = (0.5, 0)
        
        # Position keywords in outer ring
        keyword_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'keyword']
        num_kw = len(keyword_nodes)
        for i, node in enumerate(keyword_nodes):
            if num_kw > 1:
                angle = 2 * np.pi * i / num_kw
                initial_pos[node] = (1.0 * np.cos(angle), 1.0 * np.sin(angle))
            else:
                initial_pos[node] = (1.0, 0)
        
        # Calculate layout with our ForceAtlas2-like function
        pos = create_forceatlas_like_layout(
            G,
            iterations=iterations,
            gravity=gravity,
            scaling=scaling
        )
    
        # Define label placement function
        def get_smart_label_pos(node_pos, node_type, node_size):
            """Calculate intelligent label position based on position in plot and node size."""
            x, y = node_pos
            
            # Calculate node radius from node size
            node_radius = np.sqrt(node_size/np.pi) / 1000
            base_offset = max(0.02, node_radius * 1.2) * label_offset  # Apply label_offset multiplier
            
            if node_type == 'main':
                # Main categories: labels above
                return (x, y + base_offset), 'center'
            elif node_type == 'sub':
                # Subcategories: left or right depending on x-position
                if x >= 0:
                    return (x + base_offset, y), 'left'
                else:
                    return (x - base_offset, y), 'right'
            else:
                # Keywords with smaller distance
                offset = base_offset * 0.5  # Slightly smaller distance for keywords
                # Determine quadrants
                if x >= 0 and y >= 0:  # Quadrant 1 (top right)
                    return (x + offset, y), 'left'
                elif x < 0 and y >= 0:  # Quadrant 2 (top left)
                    return (x - offset, y), 'right'
                elif x < 0 and y < 0:  # Quadrant 3 (bottom left)
                    return (x - offset, y), 'right'
                else:  # Quadrant 4 (bottom right)
                    return (x + offset, y), 'left'

        # Calculate label positions and alignments
        label_pos = {}
        label_alignments = {}
        
        # NODE SIZES
        main_base_size = 500 
        sub_base_size = 300
        keyword_base_size = 200  

        for node, node_pos in pos.items():
            node_type = G.nodes[node]['node_type']
            # Calculate node size for this node
            if node_type == 'main':
                node_size = main_base_size * node_size_factor * (category_counts.get(node, 1) / max(category_counts.values()))
            elif node_type == 'sub':
                node_size = sub_base_size * node_size_factor * (subcategory_counts.get(node, 1) / max(subcategory_counts.values()))
            else:
                node_size = keyword_base_size * node_size_factor * (keyword_counts.get(node, 1) / max(keyword_counts.values()))
            
            label_position, alignment = get_smart_label_pos(node_pos, node_type, node_size)
            label_pos[node] = label_position
            label_alignments[node] = alignment
        
        # Draw edges with adjusted style
        nx.draw_networkx_edges(G, pos,
                            edge_color='gray',
                            alpha=0.4,
                            arrows=True,
                            arrowsize=10,  # Smaller arrows
                            width=0.5,    # Thinner lines
                            connectionstyle="arc3,rad=0.1")  # Reduced arc for better layout
        
        # Draw nodes
        for node_type, counts, base_size, shape in [
            ('main', category_counts, main_base_size, 'o'),
            ('sub', subcategory_counts, sub_base_size, 's'),
            ('keyword', keyword_counts, keyword_base_size, 'd')
        ]:
            node_list = [n for n, d in G.nodes(data=True) if d['node_type'] == node_type]
            
            if node_list:
                # Calculate node size with scaling factor and normalization
                node_sizes = [base_size * node_size_factor * (counts.get(node, 1) / max(counts.values()))
                            for node in node_list]
                
                if node_type == 'main':
                    node_colors = [main_category_colors.get(node, default_color) for node in node_list]
                else:
                    node_colors = '#2E8B57' if node_type == 'sub' else '#FF6B6B'  # SeaGreen for subcategories, Coral Red for keywords
                
                nx.draw_networkx_nodes(G, pos,
                                    nodelist=node_list,
                                    node_color=node_colors,
                                    node_size=node_sizes,
                                    node_shape=shape,
                                    alpha=0.8)
        
        # Draw node labels without frames for cleaner appearance
        for node_type in ['main', 'sub', 'keyword']:
            nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == node_type]
            for node in nodes:
                label_position = label_pos[node]
                alignment = label_alignments[node]
                
                # Adjusted text properties by node type
                if node_type == 'main':
                    fontweight = 'bold'
                    fontsize = 12
                    alpha = 0.9
                elif node_type == 'sub':
                    fontweight = 'normal'
                    fontsize = 12
                    alpha = 0.9
                else:
                    fontweight = 'light'
                    fontsize = 12
                    alpha = 0.9
                        
                plt.text(label_position[0], label_position[1], 
                        node,
                        horizontalalignment=alignment,
                        fontsize=fontsize,
                        fontweight=fontweight,
                        bbox=dict(facecolor='white',
                                edgecolor='none',
                                alpha=alpha,
                                pad=0.3))
        
        plt.title("Code Network Analysis (ForceAtlas2-Style Layout)", pad=20, fontsize=16)
        plt.axis('off')
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=default_color,
                    markersize=8, label='Hauptkategorien'),
            plt.Line2D([0], [0], marker='s', color='w',
                    markerfacecolor='lightgreen', markersize=8,
                    label='Subkategorien'),
            plt.Line2D([0], [0], marker='d', color='w',
                    markerfacecolor='lightpink', markersize=8,
                    label='Schlüsselwörter')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Save
        print(f"Speichere Netzwerk-Visualisierung...")
        
        # Sanitize filename to avoid issues
        safe_filename = output_filename.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        output_path = self.output_dir / f"{safe_filename}.pdf"
        svg_output_path = self.output_dir / f"{safe_filename}.svg"
        
        try:
            plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
            print(f"✓ PDF gespeichert: {output_path}")
            
            # Additionally save as SVG for better editability
            plt.savefig(svg_output_path, format='svg', bbox_inches='tight')
            print(f"✓ SVG gespeichert: {svg_output_path}")
            
            plt.close()
            print(f"Netzwerk-Visualisierung erfolgreich erstellt")
        except Exception as e:
            plt.close()
            print(f"❌ Fehler beim Speichern der Netzwerk-Visualisierung: {str(e)}")
            raise

    def create_heatmap(self, filtered_df: pd.DataFrame, output_filename: str, params: Dict[str, Any] = None):
        """
        Create a heatmap of codes along document attributes.
        
        Args:
            filtered_df: Filtered DataFrame
            output_filename: Output filename
            params: Parameters for heatmap creation (x_attribute, y_attribute, cmap, etc.)
        """
        print("\nErstelle Heatmap der Codes entlang der Dokumentattribute...")
        
        # Use parameters from configuration or default values
        if params is None:
            params = {}
        
        # Filter out "nicht kodiert" entries (default: True for visualizations)
        exclude_not_coded = params.get('exclude_not_coded', True)
        filtered_df = self._filter_not_coded(filtered_df, exclude_not_coded)
        
        if filtered_df.empty:
            print("WARNUNG: Keine kodierten Daten vorhanden!")
            return
        
        # Extract parameters with fallback values
        x_attribute = params.get('x_attribute', 'Dokument')
        y_attribute = params.get('y_attribute', 'Hauptkategorie')
        z_attribute = params.get('z_attribute', 'count')  # 'count' or 'percentage'
        use_subcodes = params.get('use_subcodes', True)
        cmap = params.get('cmap', 'YlGnBu')
        
        # Ensure figsize is a tuple
        figsize_param = params.get('figsize', (14, 10))
        
        # Default figsize
        default_figsize = (14, 10)
        figsize = default_figsize
        
        if isinstance(figsize_param, (int, float)):
            # If only a single value, use square dimension
            figsize = (float(figsize_param), float(figsize_param))
        elif isinstance(figsize_param, str):
            # If a string, try to parse
            try:
                # Remove all whitespace
                clean_param = figsize_param.strip()
                
                # Handle German decimal comma notation
                if ',' in clean_param and 'x' not in clean_param:
                    # Check if it's a German decimal number
                    parts = clean_param.split(',')
                    if len(parts) == 2:
                        try:
                            # If second part is a number and short, consider it a German decimal
                            float(parts[1])
                            if len(parts[1]) <= 2:  # Typically 1-2 decimal places
                                print(f"Warnung: '{figsize_param}' scheint eine Dezimalzahl zu sein, nicht ein Tupel.")
                                print(f"Verwende Default-Figsize: {default_figsize}")
                                figsize = default_figsize
                            else:
                                # Otherwise treat as tuple separator
                                width, height = float(parts[0]), float(parts[1])
                                figsize = (width, height)
                        except ValueError:
                            # If second part is not a number, treat as tuple separator
                            width, height = float(parts[0]), float(parts[1])
                            figsize = (width, height)
                    else:
                        # Multiple commas, treat as tuple separator
                        print(f"Warnung: Mehrere Kommas in '{figsize_param}'. Verwende erstes als Separator.")
                        width, height = float(parts[0]), float(parts[1])
                        figsize = (width, height)
                elif 'x' in clean_param:
                    # Format "14x10"
                    width, height = map(float, clean_param.split('x'))
                    figsize = (width, height)
                else:
                    # Fallback
                    print(f"Warnung: Konnte figsize '{figsize_param}' nicht parsen. Verwende Default.")
                    figsize = default_figsize
            except Exception as e:
                print(f"Warnung: Fehler beim Parsen von figsize '{figsize_param}': {str(e)}")
                print(f"Verwende Default-Figsize: {default_figsize}")
                figsize = default_figsize
        else:
            # Should already be a tuple or similar
            try:
                # Ensure it's really a tuple with two values
                if len(figsize_param) == 2:
                    figsize = (float(figsize_param[0]), float(figsize_param[1]))
                else:
                    print(f"Warnung: figsize hat falsche Anzahl von Werten: {figsize_param}")
                    figsize = default_figsize
            except:
                print(f"Warnung: figsize ist kein gültiges Tupel: {figsize_param}")
                figsize = default_figsize
        
        # Parameters for annotations
        annot_param = params.get('annot', True)
        if isinstance(annot_param, str):
            # Convert string to boolean
            annot = annot_param.lower() in ('true', 'ja', 'yes', '1')
        else:
            annot = bool(annot_param)
        
        fmt = params.get('fmt', '.0f')
        
        # Check if required attributes are present
        if x_attribute not in filtered_df.columns:
            print(f"Warnung: Attribut '{x_attribute}' nicht in Daten vorhanden.")
            return
        
        if y_attribute not in filtered_df.columns and not (y_attribute == 'Subcodes' and 'Subkategorien' in filtered_df.columns):
            print(f"Warnung: Attribut '{y_attribute}' nicht in Daten vorhanden.")
            return
        
        # Special handling for subcodes if enabled
        if use_subcodes or y_attribute == 'Subcodes':
            print("Verwende Subkategorien für die Heatmap...")
            
            # Check if subcategories are available
            if 'Subkategorien' not in filtered_df.columns:
                print("Warnung: Spalte 'Subkategorien' nicht gefunden. Kann keine Subcode-Heatmap erstellen.")
                return
                
            # Create a copy of the DataFrame for processing
            working_df = filtered_df.copy()
            
            # Explode the Subkategorien column (split comma-separated values into separate rows)
            working_df['Subcodes'] = working_df['Subkategorien'].str.split(',')
            working_df = working_df.explode('Subcodes')
            
            # Clean up the subcodes (remove whitespace)
            working_df['Subcodes'] = working_df['Subcodes'].str.strip()
            
            # Remove empty subcodes
            working_df = working_df[working_df['Subcodes'].notna() & (working_df['Subcodes'] != '')]
            
            # Override y_attribute to use our new column
            y_attribute = 'Subcodes'
            
            # Use the processed DataFrame
            heatmap_df = working_df
        else:
            # Use the original DataFrame
            heatmap_df = filtered_df
        
        # Create pivot table for heatmap
        # Count occurrences of y_attribute for each x_attribute
        pivot_df = pd.crosstab(heatmap_df[y_attribute], heatmap_df[x_attribute])
        
        # Debug: Show pivot table dimensions
        print(f"Pivot-Tabelle Dimensionen: {pivot_df.shape} (Zeilen: {pivot_df.shape[0]}, Spalten: {pivot_df.shape[1]})")
        
        # Adjust figsize based on pivot table size
        # If there are many categories, increase height
        if pivot_df.shape[0] > 10:
            # Calculate appropriate height based on number of rows
            # At least 10, plus 0.5 for each additional row over 10
            adjusted_height = 10 + (pivot_df.shape[0] - 10) * 0.5
            figsize = (figsize[0], adjusted_height)
            print(f"Viele Kategorien entdeckt, passe Figsize an: {figsize}")
        
        # Convert to percentage per column if desired
        if z_attribute == 'percentage':
            pivot_df = pivot_df.apply(lambda x: x / x.sum() * 100, axis=0)
            fmt = '.1f'  # One decimal place for percentages
        
        # Debug output
        print(f"Verwende figsize: {figsize}")
        
        # Create heatmap
        plt.figure(figsize=figsize)
        
        # Create heatmap with or without annotations
        if annot:
            ax = sns.heatmap(pivot_df, 
                        annot=True,  # Explicitly set True
                        fmt=fmt, 
                        cmap=cmap,
                        linewidths=0.5,
                        cbar_kws={'label': 'Anzahl' if z_attribute == 'count' else 'Prozent (%)'})
        else:
            ax = sns.heatmap(pivot_df, 
                        annot=False,  # Explicitly set False
                        cmap=cmap,
                        linewidths=0.5,
                        cbar_kws={'label': 'Anzahl' if z_attribute == 'count' else 'Prozent (%)'})
        
        # Set title and labels
        plt.title(f"Verteilung: {y_attribute} nach {x_attribute}", fontsize=14)
        plt.xlabel(x_attribute, fontsize=12)
        plt.ylabel(y_attribute, fontsize=12)
        
        # Improve readability of axis labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"{output_filename}.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        
        # Additionally save as PNG for easier use
        png_output_path = self.output_dir / f"{output_filename}.png"
        plt.savefig(png_output_path, format='png', bbox_inches='tight', dpi=300)
        
        # Export data as Excel
        excel_output_path = self.output_dir / f"{output_filename}_data.xlsx"
        with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
            pivot_df.to_excel(writer, sheet_name='Heatmap')
        
        plt.close()
        print(f"Heatmap erfolgreich erstellt und gespeichert unter: {output_path}")
        print(f"Daten exportiert nach: {excel_output_path}")

    def _adjust_network_parameters(self, filtered_df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-adjust network visualization parameters based on data characteristics.
        
        Args:
            filtered_df: Filtered DataFrame
            params: Original parameters
            
        Returns:
            Dict[str, Any]: Adjusted parameters
        """
        adjusted = params.copy()
        
        # Count node types
        main_cats = filtered_df['Hauptkategorie'].nunique()
        total_rows = len(filtered_df)
        
        # Adjust node size factor based on data size
        if total_rows > 100:
            # For large datasets, reduce node sizes to prevent overcrowding
            adjusted['node_size_factor'] = max(2.0, params.get('node_size_factor', 10) * 0.5)
            print(f"ℹ️ Automatische Anpassung: node_size_factor reduziert auf {adjusted['node_size_factor']:.1f} (großer Datensatz)")
        
        # Adjust layout iterations based on complexity
        expected_nodes = main_cats * 3  # Rough estimate
        if expected_nodes > 50:
            adjusted['layout_iterations'] = min(200, params.get('layout_iterations', 100) * 1.5)
            print(f"ℹ️ Automatische Anpassung: layout_iterations erhöht auf {int(adjusted['layout_iterations'])} (komplexe Struktur)")
        
        # Adjust gravity for better separation
        if main_cats > 5:
            adjusted['gravity'] = max(0.01, params.get('gravity', 0.05) * 0.7)
            print(f"ℹ️ Automatische Anpassung: gravity reduziert auf {adjusted['gravity']:.3f} (viele Hauptkategorien)")
        
        return adjusted

    def create_sunburst(self, filtered_df: pd.DataFrame, output_filename: str, params: Dict[str, Any] = None):
        """
        Create a static sunburst-style visualization using matplotlib.
        
        Creates a circular hierarchical visualization showing relationships between
        main categories, subcategories, and keywords.
        
        Args:
            filtered_df: Filtered DataFrame to visualize
            output_filename: Base name for output files
            params: Optional parameters for customization:
                - figure_size: Tuple (width, height) in inches, default (16, 16)
                - dpi: Resolution in dots per inch, default 300
                - font_size: Font size for labels, default 8
                - title_font_size: Font size for title, default 16
                - max_label_length: Maximum characters for labels, default 15
                - ring_width: Width of each ring level, default 1.5
                - color_scheme: Matplotlib colormap name, default 'Set3'
                - show_values: Show count values in labels, default True
                - label_alpha: Transparency of label backgrounds, default 0.7
                - label_bg_color: Background color for labels (HEX), default 'white'
                - label_bg_alpha: Transparency of label background (0-1), default 0.7
                - exclude_not_coded: Exclude "nicht kodiert" entries, default True
        """
        print("Erstelle statische Sunburst-Visualisierung...")
        
        # Use parameters from configuration or default values
        if params is None:
            params = {}
        
        # Filter out "nicht kodiert" entries (default: True for visualizations)
        exclude_not_coded = params.get('exclude_not_coded', True)
        filtered_df = self._filter_not_coded(filtered_df, exclude_not_coded)
        
        if filtered_df.empty:
            print("WARNUNG: Keine Daten nach Filterung vorhanden!")
            return
        
        # Extract parameters with defaults
        figure_size = params.get('figure_size', (16, 16))
        dpi = int(params.get('dpi', 300))
        font_size = int(params.get('font_size', 8))
        title_font_size = int(params.get('title_font_size', 16))
        max_label_length = int(params.get('max_label_length', 15))
        ring_width = float(params.get('ring_width', 1.5))
        color_scheme = params.get('color_scheme', 'Set3')
        show_values = params.get('show_values', True)
        label_alpha = float(params.get('label_alpha', 0.7))
        label_bg_color = params.get('label_bg_color', 'white')
        label_bg_alpha = float(params.get('label_bg_alpha', 0.7))
        max_label_length = int(params.get('max_label_length', 15))
        ring_width = float(params.get('ring_width', 1.5))
        color_scheme = params.get('color_scheme', 'Set3')
        show_values = params.get('show_values', True)
        label_alpha = float(params.get('label_alpha', 0.7))
        
        # Prepare hierarchical data
        hierarchy_data = []
        
        for _, row in filtered_df.iterrows():
            if pd.isna(row['Hauptkategorie']):
                continue
            
            main_cat = str(row['Hauptkategorie']).strip()
            hierarchy_data.append({'labels': main_cat, 'parents': '', 'values': 1})
            
            if pd.notna(row['Subkategorien']):
                sub_cats = [cat.strip() for cat in str(row['Subkategorien']).split(',') if cat.strip()]
                
                for sub_cat in sub_cats:
                    hierarchy_data.append({'labels': sub_cat, 'parents': main_cat, 'values': 1})
                    
                    if pd.notna(row['Schlüsselwörter']):
                        keywords = [kw.strip() for kw in str(row['Schlüsselwörter']).split(',') if kw.strip()]
                        
                        for keyword in keywords:
                            harmonized_keyword = self.keyword_mappings.get(keyword, keyword)
                            hierarchy_data.append({'labels': harmonized_keyword, 'parents': sub_cat, 'values': 1})
        
        if not hierarchy_data:
            print("WARNUNG: Keine Sunburst-Daten erstellt!")
            return
        
        sunburst_df = pd.DataFrame(hierarchy_data)
        sunburst_df = sunburst_df.groupby(['labels', 'parents'], as_index=False)['values'].sum()
        
        print(f"📊 Sunburst-Daten: {len(sunburst_df)} eindeutige Knoten")
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=figure_size, subplot_kw=dict(aspect="equal"))
        
        # Build hierarchy tree
        from collections import defaultdict
        children = defaultdict(list)
        node_values = {}
        
        for _, row in sunburst_df.iterrows():
            label = row['labels']
            parent = row['parents']
            value = row['values']
            node_values[label] = value
            if parent:
                children[parent].append(label)
        
        # Find root nodes
        roots = sunburst_df[sunburst_df['parents'] == '']['labels'].tolist()
        
        # Color palette
        try:
            cmap = plt.cm.get_cmap(color_scheme)
            colors = cmap(np.linspace(0, 1, len(roots)))
        except Exception as e:
            print(f"⚠️ Warnung: Farbschema '{color_scheme}' nicht verfügbar, verwende 'Set3'")
            # Fallback to Set3 if color scheme is invalid
            colors = plt.cm.Set3(np.linspace(0, 1, len(roots)))
        
        root_colors = dict(zip(roots, colors))
        
        print(f"Zeichne {len(roots)} Hauptkategorien...")
        
        # Maximum recursion depth to prevent infinite loops
        MAX_DEPTH = 10
        
        # Draw concentric rings with recursion protection
        def draw_ring(nodes, inner_radius, outer_radius, start_angle, end_angle, parent_color, level, path=None):
            if not nodes:
                return
            
            # Initialize path on first call (tracks current branch only)
            if path is None:
                path = []
            
            # Prevent infinite recursion
            if level >= MAX_DEPTH:
                print(f"⚠️ Warnung: Maximale Hierarchietiefe ({MAX_DEPTH}) erreicht")
                return
            
            angle_per_node = (end_angle - start_angle) / len(nodes)
            
            for i, node in enumerate(nodes):
                # Check for circular references in current path only
                if node in path:
                    print(f"⚠️ Warnung: Zirkuläre Referenz erkannt bei '{node}' im Pfad {' -> '.join(path + [node])}")
                    continue
                
                node_start = start_angle + i * angle_per_node
                node_end = node_start + angle_per_node
                
                # Determine color
                if level == 0:  # Root level
                    color = root_colors.get(node, [0.5, 0.5, 0.5, 0.8])
                else:
                    # Lighter shade of parent color
                    color = [min(1.0, c + 0.15 * level) for c in parent_color[:3]] + [0.8]
                
                # Draw wedge
                theta = np.linspace(node_start, node_end, 100)
                x_inner = inner_radius * np.cos(theta)
                y_inner = inner_radius * np.sin(theta)
                x_outer = outer_radius * np.cos(theta)
                y_outer = outer_radius * np.sin(theta)
                
                verts = list(zip(x_inner, y_inner)) + list(zip(x_outer[::-1], y_outer[::-1]))
                poly = plt.Polygon(verts, facecolor=color, edgecolor='white', linewidth=2)
                ax.add_patch(poly)
                
                # Add label
                mid_angle = (node_start + node_end) / 2
                mid_radius = (inner_radius + outer_radius) / 2
                label_x = mid_radius * np.cos(mid_angle)
                label_y = mid_radius * np.sin(mid_angle)
                
                # Truncate long labels
                display_label = node if len(node) <= max_label_length else node[:max_label_length-3] + '...'
                value = node_values.get(node, 0)
                
                # Format label text
                if show_values:
                    label_text = f'{display_label}\n({value})'
                else:
                    label_text = display_label
                
                rotation = np.degrees(mid_angle)
                if rotation > 90 and rotation < 270:
                    rotation = rotation - 180
                
                try:
                    ax.text(label_x, label_y, label_text, 
                           ha='center', va='center', fontsize=font_size, rotation=rotation,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=label_bg_color, alpha=label_bg_alpha, edgecolor='none'))
                except Exception:
                    # Skip problematic labels but continue
                    pass
                
                # Recursively draw children
                if node in children:
                    # Add current node to path for circular reference detection
                    new_path = path + [node]
                    draw_ring(children[node], outer_radius, outer_radius + ring_width, 
                             node_start, node_end, color, level + 1, new_path)
        
        try:
            # Start drawing from roots
            draw_ring(roots, 0, 2, 0, 2 * np.pi, None, 0)
            
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.axis('off')
            ax.set_title('Hierarchische Code-Struktur (Sunburst)', fontsize=title_font_size, pad=20)
            
            print("Speichere Visualisierung...")
            # Save figure
            output_path = self.output_dir / f"{output_filename}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Sunburst-Visualisierung gespeichert: {output_path}")
            
        except Exception as e:
            plt.close()
            print(f"❌ Fehler beim Erstellen der Sunburst-Visualisierung: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # Export data
        excel_output_path = self.output_dir / f"{output_filename}_data.xlsx"
        sunburst_df.to_excel(excel_output_path, index=False)
        print(f"✅ Daten exportiert: {excel_output_path}")
        plt.close()
        print(f"✅ Sunburst-Visualisierung gespeichert: {output_path}")
        
        # Export data
        excel_output_path = self.output_dir / f"{output_filename}_data.xlsx"
        sunburst_df.to_excel(excel_output_path, index=False)
        print(f"✅ Daten exportiert: {excel_output_path}")

    def create_treemap(self, filtered_df: pd.DataFrame, output_filename: str, params: Dict[str, Any] = None):
        """
        Create a static treemap visualization using matplotlib and squarify.
        
        Creates a rectangular hierarchical visualization showing relationships between
        main categories, subcategories, and keywords.
        
        Args:
            filtered_df: Filtered DataFrame to visualize
            output_filename: Base name for output files
            params: Optional parameters for customization:
                - figure_size: Tuple (width, height) in inches, default (20, 12)
                - detail_figure_height: Height per category for detail view, default 4
                - dpi: Resolution in dots per inch, default 300
                - font_size: Font size for labels, default 10
                - detail_font_size: Font size for detail labels, default 9
                - title_font_size: Font size for title, default 16
                - color_scheme: Matplotlib colormap name, default 'Set3'
                - detail_color_scheme: Colormap for detail view, default 'Pastel1'
                - show_values: Show count values in labels, default True
                - alpha: Transparency of rectangles, default 0.8
                - exclude_not_coded: Exclude "nicht kodiert" entries, default True
        """
        print("Erstelle statische Treemap-Visualisierung...")
        
        # Use parameters from configuration or default values
        if params is None:
            params = {}
        
        # Use parameters from configuration or default values
        if params is None:
            params = {}
        
        # Filter out "nicht kodiert" entries (default: True for visualizations)
        exclude_not_coded = params.get('exclude_not_coded', True)
        filtered_df = self._filter_not_coded(filtered_df, exclude_not_coded)
        
        if filtered_df.empty:
            print("WARNUNG: Keine Daten nach Filterung vorhanden!")
            return
        
        # Extract parameters with defaults
        figure_size = params.get('figure_size', (20, 12))
        detail_figure_height = int(params.get('detail_figure_height', 4))
        dpi = int(params.get('dpi', 300))
        font_size = int(params.get('font_size', 10))
        detail_font_size = int(params.get('detail_font_size', 9))
        title_font_size = int(params.get('title_font_size', 16))
        color_scheme = params.get('color_scheme', 'Set3')
        detail_color_scheme = params.get('detail_color_scheme', 'Pastel1')
        show_values = params.get('show_values', True)
        alpha = float(params.get('alpha', 0.8))
        
        # Try to import squarify
        try:
            import squarify
        except ImportError:
            print("⚠️ squarify nicht installiert. Installiere mit: pip install squarify")
            print("Erstelle alternative Visualisierung...")
            squarify = None
        
        # Prepare hierarchical data
        value_counts = {}
        
        for _, row in filtered_df.iterrows():
            if pd.isna(row['Hauptkategorie']):
                continue
            
            main_cat = str(row['Hauptkategorie']).strip()
            key_main = (main_cat, '')
            value_counts[key_main] = value_counts.get(key_main, 0) + 1
            
            if pd.notna(row['Subkategorien']):
                sub_cats = [cat.strip() for cat in str(row['Subkategorien']).split(',') if cat.strip()]
                
                for sub_cat in sub_cats:
                    key_sub = (sub_cat, main_cat)
                    value_counts[key_sub] = value_counts.get(key_sub, 0) + 1
                    
                    if pd.notna(row['Schlüsselwörter']):
                        keywords = [kw.strip() for kw in str(row['Schlüsselwörter']).split(',') if kw.strip()]
                        
                        for keyword in keywords:
                            harmonized_keyword = self.keyword_mappings.get(keyword, keyword)
                            key_kw = (harmonized_keyword, sub_cat)
                            value_counts[key_kw] = value_counts.get(key_kw, 0) + 1
        
        treemap_data = []
        for (label, parent), count in value_counts.items():
            treemap_data.append({'labels': label, 'parents': parent, 'values': count})
        
        if not treemap_data:
            print("WARNUNG: Keine Treemap-Daten erstellt!")
            return
        
        treemap_df = pd.DataFrame(treemap_data)
        print(f"📊 Treemap-Daten: {len(treemap_df)} Knoten")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Get root nodes (main categories)
        roots = treemap_df[treemap_df['parents'] == '']
        
        # Get color scheme
        try:
            cmap = plt.cm.get_cmap(color_scheme)
            colors = cmap(np.linspace(0, 1, len(roots)))
        except:
            colors = plt.cm.Set3(np.linspace(0, 1, len(roots)))
        
        if squarify:
            # Use squarify for better layout
            sizes = roots['values'].tolist()
            if show_values:
                labels = [f"{row['labels']}\n({row['values']})" for _, row in roots.iterrows()]
            else:
                labels = [row['labels'] for _, row in roots.iterrows()]
            
            squarify.plot(sizes=sizes, label=labels, color=colors, alpha=alpha, 
                         text_kwargs={'fontsize': font_size, 'weight': 'bold'}, ax=ax)
            
            ax.set_title('Hierarchische Code-Struktur (Treemap)', fontsize=title_font_size, pad=20)
            ax.axis('off')
        else:
            # Fallback: Simple bar chart
            roots_sorted = roots.sort_values('values', ascending=True)
            
            y_pos = np.arange(len(roots_sorted))
            ax.barh(y_pos, roots_sorted['values'], color=colors, alpha=alpha)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(roots_sorted['labels'])
            ax.set_xlabel('Anzahl', fontsize=font_size)
            ax.set_title('Hierarchische Code-Struktur (Balkendiagramm)', fontsize=title_font_size)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            if show_values:
                for i, (_, row) in enumerate(roots_sorted.iterrows()):
                    ax.text(row['values'], i, f" {row['values']}", 
                           va='center', fontsize=font_size, weight='bold')
        
        # Save figure
        output_path = self.output_dir / f"{output_filename}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Treemap-Visualisierung gespeichert: {output_path}")
        
        # Create detailed subcategory visualization
        fig, axes = plt.subplots(len(roots), 1, figsize=(16, detail_figure_height * len(roots)))
        if len(roots) == 1:
            axes = [axes]
        
        # Get detail color scheme
        try:
            detail_cmap = plt.cm.get_cmap(detail_color_scheme)
        except:
            detail_cmap = plt.cm.Pastel1
        
        for idx, (_, root) in enumerate(roots.iterrows()):
            main_cat = root['labels']
            subcats = treemap_df[treemap_df['parents'] == main_cat]
            
            if len(subcats) > 0:
                subcats_sorted = subcats.sort_values('values', ascending=True)
                detail_colors = detail_cmap(np.linspace(0, 1, len(subcats_sorted)))
                
                y_pos = np.arange(len(subcats_sorted))
                axes[idx].barh(y_pos, subcats_sorted['values'], color=detail_colors, alpha=alpha)
                axes[idx].set_yticks(y_pos)
                axes[idx].set_yticklabels(subcats_sorted['labels'], fontsize=detail_font_size)
                axes[idx].set_xlabel('Anzahl', fontsize=detail_font_size)
                axes[idx].set_title(f'{main_cat} - Subkategorien', fontsize=font_size, weight='bold')
                axes[idx].grid(axis='x', alpha=0.3)
                
                # Add value labels
                if show_values:
                    for i, (_, row) in enumerate(subcats_sorted.iterrows()):
                        axes[idx].text(row['values'], i, f" {row['values']}", 
                                      va='center', fontsize=detail_font_size)
        
        output_path_detail = self.output_dir / f"{output_filename}_detail.png"
        plt.tight_layout()
        plt.savefig(output_path_detail, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Detaillierte Visualisierung gespeichert: {output_path_detail}")
        
        # Export data
        excel_output_path = self.output_dir / f"{output_filename}_data.xlsx"
        treemap_df.to_excel(excel_output_path, index=False)
        print(f"✅ Daten exportiert: {excel_output_path}")

    async def create_custom_summary(self, 
                            filtered_df: pd.DataFrame, 
                            prompt_template: str, 
                            output_filename: str,
                            model: str,
                            temperature: float = 0.7,
                            filters: Dict[str, str] = None,
                            params: Dict[str, Any] = None):
        """
        Create a custom summary of filtered data using an LLM.
        
        Args:
            filtered_df: Filtered DataFrame
            prompt_template: Template for the prompt
            output_filename: Output filename
            model: Name of the LLM model
            temperature: Temperature parameter for the LLM
            filters: Filters applied to the data
            params: Additional parameters from configuration
            
        Returns:
            Generated summary text or None if error occurred
        """
        print("\nErstelle benutzerdefinierte Zusammenfassung...")
        
        # Ensure params exists
        if params is None:
            params = {}
        
        if len(filtered_df) == 0:
            print("Warnung: Keine Daten für die Zusammenfassung vorhanden!")
            return
        
        # Check if prompt_template is None or empty
        if not prompt_template or not prompt_template.strip():
            print("Warnung: Kein Prompt-Template angegeben! Verwende Standard-Prompt.")
            prompt_template = """
    Bitte analysieren Sie die folgenden Textsegmente:

    {text}

    Erstellen Sie eine umfassende Zusammenfassung der Hauptthemen und Konzepte.
            """
        
        # Collect text for summary
        text_segments = []
        
        # Use text column specified in params if available
        text_column = params.get('text_column', None)
        
        # If no column explicitly specified, try to find suitable columns
        if not text_column or text_column not in filtered_df.columns:
            if not text_column:
                print("Keine Textspalte in der Konfiguration angegeben. Suche nach passenden Spalten...")
            else:
                print(f"Angegebene Textspalte '{text_column}' nicht gefunden. Suche nach Alternativen...")
            
            # Default search priorities depending on analysis type (from params)
            analysis_type = params.get('analysis_type', '').lower()
            
            if analysis_type == 'summary_reasoning':
                # For reasoning analysis prefer columns with justifications
                preferred_columns = ['Begründung', 'Reasoning', 'Rationale', 'Kodierungsbegründung']
            else:
                # For other analyses prefer paraphrases/text segments
                preferred_columns = ['Paraphrase', 'Text', 'Textsegment', 'Segment', 'Kodiertext']
                
            # Extend with general text columns as fallback
            preferred_columns.extend(['Inhalt', 'Content', 'Kommentar', 'Comment'])
            
            # Search in order of preferred columns
            for column_name in preferred_columns:
                if column_name in filtered_df.columns:
                    text_column = column_name
                    break
                    
            if not text_column:
                # Fallback: Use first column that has "text", "inhalt", or "content" in name
                for col in filtered_df.columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['text', 'inhalt', 'content', 'comment', 'grund']):
                        text_column = col
                        break
        
            if not text_column:
                # Last fallback: Use any column that's not main category or subcategory
                for col in filtered_df.columns:
                    if col not in ['Hauptkategorie', 'Subkategorien', 'Schlüsselwörter']:
                        text_column = col
                        break
        
        if not text_column:
            print("Fehler: Keine geeignete Spalte für Textdaten gefunden!")
            return
            
        print(f"Verwende Spalte '{text_column}' für die Textzusammenfassung")
        
        # Extract text segments
        for _, row in filtered_df.iterrows():
            if pd.notna(row[text_column]):
                # Add main category and subcategories if available
                segment_info = []
                
                if 'Hauptkategorie' in row and pd.notna(row['Hauptkategorie']):
                    segment_info.append(f"Hauptkategorie: {row['Hauptkategorie']}")
                    
                if 'Subkategorien' in row and pd.notna(row['Subkategorien']):
                    segment_info.append(f"Subkategorien: {row['Subkategorien']}")
                    
                if 'Schlüsselwörter' in row and pd.notna(row['Schlüsselwörter']):
                    segment_info.append(f"Schlüsselwörter: {row['Schlüsselwörter']}")
                    
                # Add document info if available
                if 'Dokument' in row and pd.notna(row['Dokument']):
                    segment_info.append(f"Dokument: {row['Dokument']}")
                    
                # Format segment with metadata
                if segment_info:
                    segment_header = " | ".join(segment_info)
                    text_segments.append(f"[{segment_header}]\n{row[text_column]}\n")
                else:
                    text_segments.append(f"{row[text_column]}\n")
        
        # Combine all text segments
        combined_text = "\n".join(text_segments)
        
        # Prepare prompt
        format_args = {'text': combined_text}
        
        # Add filter info if present
        if filters:
            filter_str = ", ".join([f"{k}: {v}" for k, v in filters.items() if v])
            format_args['filters'] = filter_str
        
        # Format prompt with data
        try:
            prompt = prompt_template.format(**format_args)
        except KeyError as e:
            print(f"Warnung: Fehler beim Formatieren des Prompts: {str(e)}")
            # Try simpler formatting
            try:
                prompt = prompt_template.replace("{text}", combined_text)
                if 'filters' in format_args:
                    prompt = prompt.replace("{filters}", format_args['filters'])
            except Exception as e:
                print(f"Fehler bei der Fallback-Formatierung: {str(e)}")
                # Absolute fallback: Use raw text with simple prompt
                prompt = f"Bitte analysieren Sie die folgenden Textsegmente und erstellen Sie eine Zusammenfassung:\n\n{combined_text}"
        
        # Execute LLM request
        print(f"Sende Anfrage an LLM ({model})...")
        system_prompt = "Sie sind ein hilfreicher Forschungsassistent, der sich mit der Analyse qualitativer Daten auskennt."
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_provider.create_completion(
                messages=messages,
                model=model,
                temperature=temperature
            )
            
            llm_response = LLMResponse(response)
            summary = llm_response.content
            
            # Save summary
            output_path = self.output_dir / f"{output_filename}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
                
            print(f"Zusammenfassung erfolgreich erstellt und gespeichert unter: {output_path}")
            
            # Additionally save used prompt for documentation
            prompt_path = self.output_dir / f"{output_filename}_prompt.txt"
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(f"System:\n{system_prompt}\n\nUser:\n{prompt}")
                
            return summary
            
        except Exception as e:
            print(f"Fehler bei der LLM-Anfrage: {str(e)}")
            # Save error message
            error_path = self.output_dir / f"{output_filename}_error.txt"
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"Fehler bei der LLM-Anfrage: {str(e)}\n\nPrompt:\n{prompt}")
            return None

    async def create_sentiment_analysis(
        self, 
        filtered_df: pd.DataFrame, 
        output_filename: str, 
        params: Dict[str, Any] = None
    ):
        """
        Perform LLM-based sentiment analysis of filtered data.
        
        Creates both an Excel table and a bubble chart visualization of sentiment analysis results.
        
        Args:
            filtered_df: Filtered DataFrame
            output_filename: Output filename
            params: Parameters for sentiment analysis (model, temperature, sentiment_categories, etc.)
            
        Returns:
            DataFrame with sentiment analysis results or None if error occurred
        """
        print("\nFühre Sentiment-Analyse durch...")
        
        # Use parameters from configuration or default values
        if params is None:
            params = {}
        
        # Extract parameters with fallback values
        model = params.get('model', 'gpt-4o-mini')  # Specify default directly
        temperature = float(params.get('temperature', 0.3))  # Lower for more consistent results
        
        # Get sentiment categories from parameters
        # Format should be: "Category1, Category2, Category3" or ["Category1", "Category2", ...]
        sentiment_categories_raw = params.get('sentiment_categories', "Positiv, Negativ, Neutral")
        if isinstance(sentiment_categories_raw, str):
            sentiment_categories = [cat.strip() for cat in sentiment_categories_raw.split(',')]
        elif isinstance(sentiment_categories_raw, list):
            sentiment_categories = sentiment_categories_raw
        else:
            print(f"Warnung: Unerwartetes Format für sentiment_categories: {type(sentiment_categories_raw)}")
            sentiment_categories = ["Positiv", "Negativ", "Neutral"]
        
        # Show sentiment categories
        print(f"Verwende Sentiment-Kategorien: {', '.join(sentiment_categories)}")
        
        # Colors for visualization from parameters or defaults
        color_mapping_raw = params.get('color_mapping', None)
        
        # Default color scheme
        default_colors = {
            "Positiv": "#4CAF50",  # Green
            "Negativ": "#F44336",  # Red
            "Neutral": "#9E9E9E",  # Gray
            "Kritisch": "#FF5722",  # Orange
            "Befürwortend": "#2196F3",  # Blue
            "Ambivalent": "#9C27B0",  # Purple
        }
        
        # Create color mapping based on specified categories
        color_mapping = {}
        if color_mapping_raw and isinstance(color_mapping_raw, str):
            # Try to parse JSON string
            try:
                color_mapping = json.loads(color_mapping_raw)
            except json.JSONDecodeError:
                print(f"Warnung: Ungültiges JSON-Format für color_mapping: {color_mapping_raw}")
                # Create automatic mapping
        elif color_mapping_raw and isinstance(color_mapping_raw, dict):
            # Use specified mapping
            color_mapping = color_mapping_raw
        
        # If no valid mapping created, create automatic one
        if not color_mapping:
            # Create automatic mapping based on default colors
            for i, category in enumerate(sentiment_categories):
                if category in default_colors:
                    color_mapping[category] = default_colors[category]
                else:
                    # Random color for unknown categories
                    color_mapping[category] = f"#{random.randint(0, 255):02X}{random.randint(0, 255):02X}{random.randint(0, 255):02X}"
        
        # Determine text column for analysis
        text_column = params.get('text_column', None)
        
        if not text_column or text_column not in filtered_df.columns:
            # Search for suitable columns
            preferred_columns = ['Text', 'Paraphrase', 'Textsegment', 'Segment', 'Kodiertext', 'Begründung']
            for column_name in preferred_columns:
                if column_name in filtered_df.columns:
                    text_column = column_name
                    break
            
            if not text_column:
                # Fallback: Use first column containing "text"
                for col in filtered_df.columns:
                    if 'text' in col.lower():
                        text_column = col
                        break
        
        if not text_column:
            print("Fehler: Keine geeignete Spalte für Textdaten gefunden!")
            return
            
        print(f"Verwende Spalte '{text_column}' für die Sentiment-Analyse")
        
        # ADJUSTED PROMPT that now also requests keywords
        default_prompt = """
        Du bist ein Experte für qualitative Textanalyse und Sentimentbewertung.
        
        Analysiere den folgenden Text und klassifiziere ihn anhand des Sentiments in eine der folgenden Kategorien: {sentiment_categories}
        
        Achte bei deiner Analyse besonders auf:
        1. Explizite Bewertungen und Emotionsausdrücke
        2. Implizite Wertungen (durch Wortwahl, Vergleiche etc.)
        3. Den Gesamtkontext des Textes
        
        Text:
        ---
        {text}
        ---
        
        Antworte mit einem JSON-Objekt im folgenden Format:
        {{
            "sentiment": "Kategorie", // Eine der vorgegebenen Kategorien
            "keywords": ["wort1", "wort2", "wort3"], // maximal drei Schlüsselwörter, die für das Sentiment im Text entscheidend sind
            "explanation": "Kurze Begründung" // Kurze Erklärung (1-2 Sätze)
        }}
        """
        
        prompt_template = params.get('prompt_template', default_prompt)
        
        # Perform analysis for each text segment
        results = []
        total_segments = len(filtered_df)
        
        print(f"Analysiere {total_segments} Textsegmente...")
        
        # Collect all texts and metadata
        analysis_tasks = []
        for idx, row in filtered_df.iterrows():
            if pd.isna(row[text_column]):
                continue
                
            # Extract text and metadata
            text = row[text_column]
            
            # Collect metadata
            metadata = {}
            for col in filtered_df.columns:
                if col != text_column and not pd.isna(row[col]):
                    metadata[col] = row[col]
            
            # Format prompt with data
            formatted_prompt = prompt_template.format(
                sentiment_categories=", ".join(sentiment_categories),
                text=text
            )
            
            # Create task with text and metadata
            analysis_tasks.append({
                'text': text,
                'prompt': formatted_prompt,
                'metadata': metadata
            })
        
        # Perform analysis asynchronously
        sentiment_results = []
        
        # Asynchronous batch processing for faster processing
        async def process_batch(batch, batch_idx, total_batches):
            """
            Process a batch of sentiment analysis tasks asynchronously.
            
            Args:
                batch: List of tasks to process
                batch_idx: Index of the current batch
                total_batches: Total number of batches
                
            Returns:
                List of sentiment analysis results for the batch
            """
            batch_results = []
            for i, task in enumerate(batch):
                try:
                    messages = [
                        {"role": "system", "content": "Du bist ein hilfreicher Forschungsassistent, der sich mit der Analyse qualitativer Daten auskennt. Antworte im JSON-Format."},
                        {"role": "user", "content": task['prompt']}
                    ]
                    
                    response = await self.llm_provider.create_completion(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        response_format={"type": "json_object"}  # Force JSON response
                    )
                    
                    llm_response = LLMResponse(response)
                    response_text = llm_response.content.strip()
                    
                    # Parse JSON response
                    try:
                        response_json = json.loads(response_text)
                        sentiment = response_json.get('sentiment', '')
                        keywords = response_json.get('keywords', [])
                        explanation = response_json.get('explanation', '')
                        
                        # Normalize sentiment
                        sentiment = sentiment.strip().rstrip('.')
                        
                        # Check if response is in allowed categories
                        if sentiment not in sentiment_categories:
                            # Try to find most similar category
                            for category in sentiment_categories:
                                if category.lower() in sentiment.lower():
                                    print(f"Korrigiere Sentiment '{sentiment}' zu '{category}'")
                                    sentiment = category
                                    break
                            else:
                                # If still not found, use "Neutral" or first category
                                if "Neutral" in sentiment_categories:
                                    sentiment = "Neutral"
                                else:
                                    sentiment = sentiment_categories[0]
                                    print(f"Warnung: Sentiment '{sentiment}' wurde auf '{sentiment_categories[0]}' zurückgesetzt")
                        
                        # Save result with metadata
                        result = {
                            'text': task['text'],
                            'sentiment': sentiment,
                            'keywords': keywords,
                            'explanation': explanation,
                            **task['metadata']
                        }
                        
                        batch_results.append(result)
                        
                    except json.JSONDecodeError:
                        print(f"Fehler beim Parsen der JSON-Antwort: {response_text[:100]}...")
                        # Try to extract simpler response
                        sentiment = None
                        
                        # Search for one of the sentiment categories in text
                        for category in sentiment_categories:
                            if category in response_text:
                                sentiment = category
                                break
                        
                        if not sentiment:
                            # Fallback: Use "Neutral" or first category
                            sentiment = "Neutral" if "Neutral" in sentiment_categories else sentiment_categories[0]
                        
                        # Create minimal result
                        result = {
                            'text': task['text'],
                            'sentiment': sentiment,
                            'keywords': ["Error", "Parsing", "Failed"],
                            'explanation': "Failed to parse JSON response",
                            **task['metadata']
                        }
                        batch_results.append(result)
                    
                    # Show progress
                    overall_progress = ((batch_idx * len(batch) + i + 1) / total_segments) * 100
                    if (i + 1) % 5 == 0 or i == len(batch) - 1:
                        print(f"Fortschritt: {overall_progress:.1f}% - Batch {batch_idx+1}/{total_batches}, Element {i+1}/{len(batch)}")
                    
                except Exception as e:
                    print(f"Fehler bei der Sentiment-Analyse: {str(e)}")
                    # Add entry with error status
                    batch_results.append({
                        'text': task['text'],
                        'sentiment': "Fehler",
                        'keywords': ["Error"],
                        'explanation': str(e),
                        **task['metadata']
                    })
            
            return batch_results
        
        # Create batches for asynchronous processing
        batch_size = 5  # Adjust as needed
        batches = [analysis_tasks[i:i+batch_size] for i in range(0, len(analysis_tasks), batch_size)]
        
        # Process all batches
        for batch_idx, batch in enumerate(batches):
            batch_results = await process_batch(batch, batch_idx, len(batches))
            sentiment_results.extend(batch_results)
        
        if not sentiment_results:
            print("Keine Ergebnisse von der Sentiment-Analyse erhalten.")
            return
        
        # Create DataFrame with results
        results_df = pd.DataFrame(sentiment_results)
        
        # Count sentiment frequencies
        sentiment_counts = Counter(results_df['sentiment'])
        
        print("\nSentiment-Verteilung:")
        for sentiment, count in sentiment_counts.most_common():
            percentage = (count / len(results_df)) * 100
            print(f"- {sentiment}: {count} ({percentage:.1f}%)")
        
        # Collect all keywords with their sentiments
        keyword_sentiments = {}
        all_keywords = []
        
        for _, row in results_df.iterrows():
            if 'keywords' in row and isinstance(row['keywords'], list):
                for keyword in row['keywords']:
                    if isinstance(keyword, str) and keyword.strip():
                        clean_keyword = keyword.strip().lower()
                        all_keywords.append(clean_keyword)
                        keyword_sentiments[clean_keyword] = row['sentiment']
        
        # Count keyword frequencies
        keyword_counts = Counter(all_keywords)
        
        print("\nTop 10 Schlüsselwörter:")
        for keyword, count in keyword_counts.most_common(10):
            print(f"- {keyword}: {count}")
        
        # Create result file
        output_path = self.output_dir / f"{output_filename}_results.xlsx"
        
        # Save results to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Save detailed results
            results_df.to_excel(writer, sheet_name='Detailergebnisse', index=False)
            
            # Create and save sentiment summary
            summary_data = []
            for sentiment, count in sentiment_counts.most_common():
                percentage = (count / len(results_df)) * 100
                summary_data.append({
                    'Sentiment': sentiment,
                    'Anzahl': count,
                    'Prozent': percentage
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Sentiment-Zusammenfassung', index=False)
            
            # Create and save keyword summary
            keyword_data = []
            for keyword, count in keyword_counts.most_common(30):  # Top 30
                keyword_data.append({
                    'Schlüsselwort': keyword,
                    'Sentiment': keyword_sentiments.get(keyword, 'Unbekannt'),
                    'Anzahl': count,
                    'Prozent': (count / len(all_keywords)) * 100
                })
            
            keyword_df = pd.DataFrame(keyword_data)
            keyword_df.to_excel(writer, sheet_name='Schlüsselwort-Zusammenfassung', index=False)
            
            # Create crosstabs for interesting dimensions
            for dimension in params.get('crosstab_dimensions', []):
                if dimension in results_df.columns:
                    try:
                        # Create crosstab
                        crosstab = pd.crosstab(
                            results_df[dimension], 
                            results_df['sentiment'],
                            normalize='index'  # Percentage distribution per row
                        ) * 100  # Convert to percent
                        
                        # Save to Excel
                        crosstab.to_excel(writer, sheet_name=f'Kreuztabelle_{dimension[:20]}')
                    except Exception as e:
                        print(f"Fehler bei Kreuztabelle für {dimension}: {str(e)}")
        
        print(f"Ergebnisse gespeichert: {output_path}")
        
        # Create visualization (Bubble Chart) with keywords as bubbles
        self._create_keyword_bubble_chart(
            keyword_data=keyword_data[:30],  # Use top 30 keywords
            output_filename=output_filename,
            color_mapping=color_mapping,
            params=params
        )
        
        return results_df

    def _create_keyword_bubble_chart(
        self,
        keyword_data: List[Dict[str, Any]],
        output_filename: str,
        color_mapping: Dict[str, str],
        params: Dict[str, Any] = None
    ):
        """
        Create a bubble chart visualization of keywords with circlify for optimal layout, colored by sentiment.
        
        Args:
            keyword_data: List of dictionaries with keyword information
            output_filename: Output filename
            color_mapping: Mapping of sentiment categories to colors
            params: Additional parameters for visualization
        """
        import circlify  # New dependency
        
        if params is None:
            params = {}
        
        # Extract parameters
        title = params.get('chart_title', 'Sentiment-Analyse: Schlüsselwörter')
        
        # Process figsize
        figsize_param = params.get('figsize', (14, 10))
        default_figsize = (14, 10)
        
        # Try to extract figsize from various formats
        try:
            if isinstance(figsize_param, (int, float)):
                figsize = (float(figsize_param), float(figsize_param))
            elif isinstance(figsize_param, str):
                if 'x' in figsize_param:
                    width, height = map(float, figsize_param.split('x'))
                    figsize = (width, height)
                else:
                    figsize = default_figsize
            elif isinstance(figsize_param, (list, tuple)) and len(figsize_param) == 2:
                figsize = (float(figsize_param[0]), float(figsize_param[1]))
            else:
                figsize = default_figsize
        except Exception as e:
            print(f"Warnung: Fehler beim Parsen von figsize '{figsize_param}': {str(e)}")
            figsize = default_figsize
        
        # Check if data is available
        if not keyword_data:
            print(f"Warnung: Keine Schlüsselwort-Daten für Bubble-Chart vorhanden.")
            # Create empty visualization with note
            plt.figure(figsize=(10, 6), facecolor='white')
            plt.text(0.5, 0.5, "Keine Schlüsselwörter gefunden", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            
            # Save empty visualization
            output_path = self.output_dir / f"{output_filename}_keyword_bubble_chart.png"
            plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Leeres Keyword-Bubble-Chart erstellt: {output_path}")
            return
        
        # Transform data into format needed by circlify
        # Each element needs a 'datum' for size and an 'id' for identification
        elements = [
            {
                'datum': item['Anzahl'],  # Use 'Anzahl' as size value
                'id': i,                  # Index as ID
                'sentiment': item['Sentiment']  # Also store sentiment
            }
            for i, item in enumerate(keyword_data)
        ]
        
        # Calculate circle layout with circlify
        circles = circlify.circlify(
            elements,
            show_enclosure=False,  # No encompassing outer circle
            target_enclosure=circlify.Circle(x=0, y=0, r=1),  # Target area (normalized)
            datum_field='datum',  # Field for size
            id_field='id'        # Field for ID
        )
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Turn off axes
        plt.axis('off')
        
        # Calculate bounds for diagram
        lim = max(
            max(
                abs(circle.x) + circle.r,
                abs(circle.y) + circle.r,
            )
            for circle in circles
        )
        plt.xlim(-lim * 1.05, lim * 1.05)  # Leave some space at edges
        plt.ylim(-lim * 1.05, lim * 1.05)
        
        # Draw circles
        legend_handles = {}
        for circle in circles:
            # Extract circle parameters
            x, y, r = circle.x, circle.y, circle.r
            
            # Get original data via ID
            original_index = circle.ex['id']
            original_data = keyword_data[original_index]
            
            # Get relevant data
            keyword = original_data['Schlüsselwort']
            count = original_data['Anzahl']
            sentiment = original_data['Sentiment']
            
            # Determine color based on sentiment
            color = color_mapping.get(sentiment, "#CCCCCC")  # Gray as fallback
            
            # Draw circle
            circle_patch = plt.Circle(
                (x, y), r,
                facecolor=color,
                edgecolor='white',
                linewidth=1,
                alpha=0.8  # Slightly transparent for better readability
            )
            ax.add_patch(circle_patch)
            
            # Dynamic font size based on circle radius
            font_size_keyword = max(8, r * 30)  # At least 8pt, scales with radius
            font_size_count = max(6, r * 25)   # Slightly smaller for number
            
            # Slight offset for better placement
            offset_y = r * 0.2
            
            # Draw keyword text
            ax.text(
                x, y + offset_y, keyword,
                ha='center', va='center',
                fontsize=font_size_keyword,
                fontweight='bold',
                color='black'
            )
            
            # Draw count
            ax.text(
                x, y - offset_y, f"({count})",
                ha='center', va='center',
                fontsize=font_size_count,
                color='black'
            )
            
            # Add to legend if not already present
            if sentiment not in legend_handles:
                legend_handles[sentiment] = circle_patch
        
        # Add title
        plt.title(title, fontsize=16, pad=20)
        
        # Add legend
        if legend_handles:
            plt.legend(
                handles=list(legend_handles.values()),
                labels=list(legend_handles.keys()),
                loc='upper right',
                fontsize=10
            )
        
        # Add overall information
        total_keywords = len(keyword_data)
        total_text = f"Top {total_keywords} Schlüsselwörter"
        plt.figtext(0.5, 0.02, total_text, ha='center', fontsize=12)
        
        # Add date
        date_text = f"Erstellt: {datetime.now().strftime('%d.%m.%Y')}"
        plt.figtext(0.95, 0.02, date_text, ha='right', fontsize=10)
        
        # Save visualization
        output_path = self.output_dir / f"{output_filename}_keyword_bubble_chart.png"
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
        
        # Additionally as PDF for better quality
        pdf_output_path = self.output_dir / f"{output_filename}_keyword_bubble_chart.pdf"
        plt.savefig(pdf_output_path, format='pdf', bbox_inches='tight')
        
        plt.close()
        print(f"Keyword-Bubble-Chart mit circlify erstellt: {output_path}")
