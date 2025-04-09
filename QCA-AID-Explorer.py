"""
QCA-AID Explorer 
========================================================================
Version: 0.4
--------
Neue features in version 0.4 (2025-04-07):
- Konfiguration über Excel-Datei "QCA-AID-Explorer-Config.xlsx"
- Heatmap-Visualisierung von Codes entlang von Dokumentattributen
- Mehrere Analysetypen konfigurierbar (Netzwerk, Heatmap, verschiedene Zusammenfassungen)
- Anpassbare Parameter für jede Analyse
- Verbessertes ForceAtlas2-Layout für Netzwerkvisualisierungen
- SVG-Export für bessere Editierbarkeit

QCA-AID Explorer ist ein Tool zur Analyse von qualitativen Kodierungsdaten.
Es ermöglicht die Visualisierung von Kodierungsnetzwerken mit Hauptkategorien,
Subkategorien und Schlüsselwörtern sowie die automatisierte Zusammenfassung
von kodierten Textsegmenten mit Hilfe von LLM-Modellen.

Das ForceAtlas2-Layout bietet eine verbesserte Netzwerkvisualisierung,
bei der zusammengehörige Elemente näher beieinander platziert werden
und die hierarchische Struktur der Daten besser sichtbar wird.

Folgende Pakete sollten vor der ersten Nutzung installiert werden:
pip install networkx reportlab scikit-learn pandas openpyxl matplotlib seaborn

"""

"""
QCA-AID Explorer 
========================================================================
Version:
--------
0.4 (2025-04-07)
New in this version
- Configuration through Excel file "QCA-AID-Explorer-Config.xlsx"
- Heatmap visualization of codes along document attributes
- Multiple analysis types configurable from Excel
- Customizable parameters for each analysis

QCA-AID Explorer ist ein Tool zur Analyse von qualitativen Kodierungsdaten.
Es ermöglicht die Visualisierung von Kodierungsnetzwerken mit Hauptkategorien,
Subkategorien und Schlüsselwörtern sowie die automatisierte Zusammenfassung
von kodierten Textsegmenten mit Hilfe von LLM-Modellen.

Folgende Pakete sollten vor der ersten Nutzung installiert werden:
pip install networkx reportlab scikit-learn pandas openpyxl matplotlib seaborn
"""

import pandas as pd
from difflib import SequenceMatcher
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import asyncio
import json
from datetime import datetime
import os
import httpx
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import openai
from mistralai import Mistral
from docx import Document
from reportlab.pdfgen import canvas
import numpy as np
from sklearn.manifold import MDS
from scipy.sparse import csgraph
import seaborn as sns
import re

class LLMProvider(ABC):
    """Abstrakte Basisklasse für LLM Provider"""
    
    @abstractmethod
    async def create_completion(self, 
                              messages: List[Dict[str, str]], 
                              model: str,
                              temperature: float = 0.7,
                              response_format: Optional[Dict] = None) -> Any:
        """Generiert eine Completion mit dem konfigurierten LLM"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI spezifische Implementation"""
    
    def __init__(self):
        """Initialisiert den OpenAI Client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API Key nicht gefunden")
            
            # Erstelle einen expliziten httpx Client ohne Proxy Konfiguration
            http_client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=60.0
            )
            
            # Initialisiere den OpenAI Client mit unserem sauberen httpx Client
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                http_client=http_client
            )
            
            print("OpenAI Client erfolgreich initialisiert")
            
        except Exception as e:
            print(f"Fehler bei OpenAI Client Initialisierung: {str(e)}")
            raise

    async def create_completion(self,
                              messages: List[Dict[str, str]],
                              model: str,
                              temperature: float = 0.7, 
                              response_format: Optional[Dict] = None) -> Any:
        """Erzeugt eine Completion mit OpenAI"""
        try:
            # Direkter API-Aufruf ohne Context Manager
            return await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format
            )
        except Exception as e:
            print(f"Fehler bei OpenAI API Call: {str(e)}")
            raise

class MistralProvider(LLMProvider):
    """Mistral spezifische Implementation"""
    
    def __init__(self):
        """Initialisiert den Mistral Client"""
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API Key nicht gefunden")

    async def create_completion(self,
                              messages: List[Dict[str, str]],
                              model: str,
                              temperature: float = 0.7,
                              response_format: Optional[Dict] = None) -> Any:
        """Erzeugt eine Completion mit Mistral"""
        try:
            async with Mistral(api_key=self.api_key) as client:
                return await client.chat.complete_async(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format,
                    stream=False
                )
        except Exception as e:
            print(f"Fehler bei Mistral API Call: {str(e)}")
            raise

class LLMProviderFactory:
    """Factory Klasse zur Erstellung des konfigurierten LLM Providers"""
    
    @staticmethod
    def create_provider(provider_name: str) -> LLMProvider:
        """Erstellt die passende Provider Instanz"""
        try:
            # Lade Environment Variablen
            env_path = os.path.join(os.path.expanduser("~"), '.environ.env')
            load_dotenv(env_path)
            
            print(f"\nInitialisiere LLM Provider: {provider_name}")
            
            if provider_name.lower() == "openai":
                return OpenAIProvider()
                
            elif provider_name.lower() == "mistral":
                return MistralProvider()
                
            else:
                raise ValueError(f"Ungültiger Provider Name: {provider_name}")
                
        except Exception as e:
            print(f"Fehler bei Provider-Erstellung: {str(e)}")
            raise

class LLMResponse:
    """Wrapper für Provider-spezifische Responses"""
    
    def __init__(self, provider_response: Any):
        self.raw_response = provider_response
        
    @property  
    def content(self) -> str:
        """Extrahiert den Content aus der Provider Response"""
        try:
            if hasattr(self.raw_response, "choices"):
                # OpenAI Format
                return self.raw_response.choices[0].message.content
            else:
                # Mistral Format 
                return self.raw_response.choices[0].message.content
        except Exception as e:
            print(f"Fehler beim Extrahieren des Response Contents: {str(e)}")
            return ""

class ConfigLoader:
    """Lädt die Konfiguration aus einer Excel-Datei"""
    
    def __init__(self, config_path: str):
        """
        Initialisiert den ConfigLoader
        
        Args:
            config_path: Pfad zur Excel-Konfigurationsdatei
        """
        self.config_path = config_path
        self.base_config = {}
        self.analysis_configs = []
        self._load_config()
        
    def _load_config(self):
        """Lädt die Konfiguration aus der Excel-Datei"""
        try:
            print(f"\nLade Konfiguration aus: {self.config_path}")
            
            # Prüfe, ob die Datei existiert
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {self.config_path}")
            
            # Lese das Basis-Sheet
            base_df = pd.read_excel(self.config_path, sheet_name='Basis')
            
            # Konvertiere zu Dictionary
            self.base_config = {}
            for _, row in base_df.iterrows():
                param_name = str(row['Parameter'])
                param_value = row['Wert']
                
                # Leere Werte als None behandeln
                if pd.isna(param_value):
                    param_value = None
                    
                self.base_config[param_name] = param_value
            
            print("Basis-Konfiguration geladen:")
            for key, value in self.base_config.items():
                print(f"  {key}: {value}")
            
            # Lese alle anderen Sheets für Auswertungskonfigurationen
            excel = pd.ExcelFile(self.config_path)
            sheet_names = excel.sheet_names
            
            # Überspringe das Basis-Sheet
            for sheet_name in sheet_names:
                if sheet_name.lower() != 'basis':
                    analysis_df = pd.read_excel(self.config_path, sheet_name=sheet_name)
                    
                    # Extrahiere die Parameter für diese Auswertung
                    analysis_config = {'name': sheet_name}
                    filter_params = {}
                    other_params = {}
                    
                    for _, row in analysis_df.iterrows():
                        param_name = str(row['Parameter'])
                        param_value = row['Wert']
                        
                        # Leere Werte als None behandeln
                        if pd.isna(param_value):
                            param_value = None
                        
                        # Unterscheide zwischen Filter-Parametern und anderen Parametern
                        if param_name.startswith('filter_'):
                            # Entferne 'filter_' Präfix und speichere als Filter
                            filter_name = param_name[7:]  # Länge von 'filter_' ist 7
                            filter_params[filter_name] = param_value
                        else:
                            other_params[param_name] = param_value
                    
                    analysis_config['filters'] = filter_params
                    analysis_config['params'] = other_params
                    self.analysis_configs.append(analysis_config)
            
            print(f"\n{len(self.analysis_configs)} Auswertungskonfigurationen gefunden:")
            for config in self.analysis_configs:
                print(f"  - {config['name']}")
                
        except Exception as e:
            print(f"Fehler beim Laden der Konfiguration: {str(e)}")
            raise

    def get_base_config(self) -> Dict[str, Any]:
        """Gibt die Basis-Konfiguration zurück"""
        return self.base_config
    
    def get_analysis_configs(self) -> List[Dict[str, Any]]:
        """Gibt die Auswertungskonfigurationen zurück"""
        return self.analysis_configs

class QCAAnalyzer:
    def __init__(self, excel_path: str, llm_provider: LLMProvider, config: Dict[str, Any]):
        """
        Initialize the QCA Analyzer
        
        Args:
            excel_path: Path to the Excel file with coded segments
            llm_provider: Instance of LLMProvider
            config: Dictionary with base configuration
        """
        # Read specifically from 'Kodierte_Segmente' sheet
        self.df = pd.read_excel(excel_path, sheet_name='Kodierte_Segmente')
        self.llm_provider = llm_provider
        self.keyword_mappings = {}
        
        # Get the input filename without extension
        input_filename = Path(excel_path).stem

        # Set output directory from config or use default
        script_dir = config.get('script_dir') or os.path.dirname(os.path.abspath(__file__))
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
        Filter the dataframe based on provided column-value pairs
        
        Args:
            filters: Dictionary with column-value pairs
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = self.df.copy()
        
        for col, value in filters.items():
            if not value:  # Skip empty filters
                continue
                
            if col == "Subkategorien":
                # Spezielle Behandlung für Subkategorien
                # Prüfe ob der gesuchte Wert in der kommagetrennten Liste vorkommt
                filtered_df = filtered_df[filtered_df[col].fillna('').str.split(',').apply(
                    lambda x: value.strip() in [item.strip() for item in x]
                )]
            else:
                # Standardfilterung für andere Spalten
                filtered_df = filtered_df[filtered_df[col] == value]
        
        # Debug-Ausgabe
        print(f"\nFilter angewendet:")
        for col, value in filters.items():
            if value:
                print(f"- {col}: {value}")
        print(f"Anzahl der gefilterten Zeilen: {len(filtered_df)}")
        
        return filtered_df

    async def generate_summary(self, 
                             text: str, 
                             prompt_template: str, 
                             model: str,
                             temperature: float = 0.7,
                             **kwargs) -> str:
        """Generate summary using configured LLM provider"""

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
        """Save summary to text file"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)

    def harmonize_keywords(self, similarity_threshold: float = 0.60) -> pd.DataFrame:
        """
        Harmonize keywords using fuzzy matching while preserving hierarchical relationships.
        
        Args:
            similarity_threshold: Threshold for fuzzy matching (default: 0.60)
                
        Returns:
            DataFrame with harmonized keywords
        """
        def get_similarity(a: str, b: str) -> float:
            """Calculate string similarity with enhanced rules for academic terms"""
            if not isinstance(a, str) or not isinstance(b, str):
                return 0.0
                    
            # Normalize strings
            a = a.lower().strip()
            b = b.lower().strip()
                
            # Base similarity with SequenceMatcher
            base_similarity = SequenceMatcher(None, a, b).ratio()
            
            return min(1.0, base_similarity)

        def find_most_common_variant(keywords: list) -> dict:
            """Find the most common variant with improved academic context handling"""
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
                # private hochschule': 'Private Hochschule',
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
            
        # Sammle alle Schlüsselwörter aus dem gesamten DataFrame
        all_keywords = []
        for kw_list in harmonized_df['Schlüsselwörter'].dropna():
            all_keywords.extend([k.strip() for k in str(kw_list).split(',') if k.strip()])

        # Generiere globale Keyword-Mappings
        keyword_mapping = find_most_common_variant(all_keywords)
        self.keyword_mappings.update(keyword_mapping)

        # Wende Mappings auf alle Zeilen an
        def replace_keywords(x):
            if pd.isna(x):
                return x
            original = [k.strip() for k in str(x).split(',')]
            mapped = [keyword_mapping.get(k, k) for k in original]
            seen = set()
            return ','.join([k for k in mapped if not (k in seen or seen.add(k))])

        harmonized_df['Schlüsselwörter'] = harmonized_df['Schlüsselwörter'].apply(replace_keywords)

        # Aktualisiere den Haupt-DataFrame
        self.df = harmonized_df
        return self.df
    
    def validate_keyword_mapping(self):
        """Validiert die Schlüsselwort-Zuordnungen"""
        from collections import defaultdict
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
        """Create network graph using harmonized keywords"""
        
        print("Erstelle Netzwerkgraph...")
        
        # Verwende Parameter aus der Konfiguration oder Default-Werte
        if params is None:
            params = {}
        
        # Extrahiere Parameter mit Fallback-Werten
        node_size_factor = float(params.get('node_size_factor', 10))  # Default-Wert 10
        iterations = int(params.get('layout_iterations', 100))
        gravity = float(params.get('gravity', 0.05))
        scaling = float(params.get('scaling', 2.0))
        
        # Optional: setze Hintergrundfarbe aus Parameter
        bg_color = params.get('bg_color', 'white')  # Default: weißer Hintergrund statt pink
        
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
        
        # Export to Excel
        excel_output_path = self.output_dir / f"{output_filename}_network_data.xlsx"
        with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
            nodes_df.to_excel(writer, sheet_name='Nodes', index=False)
            edges_df.to_excel(writer, sheet_name='Edges', index=False)
        print(f"Netzwerkdaten exportiert nach: {excel_output_path}")
        
        # Create visualization
        print("\nErstelle Visualisierung...")
        plt.figure(figsize=(24, 20), facecolor=bg_color)

        # --- NEUER CODE: Verwendung des ForceAtlas2-ähnlichen Layouts ---
        print("\nBerechne Knotenpositionen mit ForceAtlas2-ähnlichem Layout...")
        
        # Erstelle initiale Positionen für bessere Konvergenz
        initial_pos = {}
        
        # Positioniere Hauptkategorien im Zentrum
        main_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'main']
        num_main = len(main_nodes)
        for i, node in enumerate(main_nodes):
            if num_main > 1:
                angle = 2 * np.pi * i / num_main
                initial_pos[node] = (0.2 * np.cos(angle), 0.2 * np.sin(angle))
            else:
                initial_pos[node] = (0, 0)
        
        # Positioniere Subkategorien in einem mittleren Ring
        sub_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'sub']
        num_sub = len(sub_nodes)
        for i, node in enumerate(sub_nodes):
            if num_sub > 1:
                angle = 2 * np.pi * i / num_sub
                initial_pos[node] = (0.5 * np.cos(angle), 0.5 * np.sin(angle))
            else:
                initial_pos[node] = (0.5, 0)
        
        # Positioniere Keywords in einem äußeren Ring
        keyword_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'keyword']
        num_kw = len(keyword_nodes)
        for i, node in enumerate(keyword_nodes):
            if num_kw > 1:
                angle = 2 * np.pi * i / num_kw
                initial_pos[node] = (1.0 * np.cos(angle), 1.0 * np.sin(angle))
            else:
                initial_pos[node] = (1.0, 0)
        
        # Berechne das Layout mit unserer ForceAtlas2-ähnlichen Funktion
        # Verwende Parameter aus der Konfiguration
        pos = create_forceatlas_like_layout(
            G,
            iterations=iterations,
            gravity=gravity,
            scaling=scaling
        )
    
        # Define label placement function
        def get_smart_label_pos(node_pos, node_type, node_size):
            """Berechne intelligente Label-Position basierend auf Position im Plot und Knotengröße"""
            x, y = node_pos
            
            # Berechne den Radius des Knotens aus der Knotengröße
            node_radius = np.sqrt(node_size/np.pi) / 1000
            base_offset = max(0.02, node_radius * 1.2)  # Kleinerer Offset für bessere Lesbarkeit
            
            if node_type == 'main':
                # Hauptkategorien: Labels oberhalb
                return (x, y + base_offset), 'center'
            elif node_type == 'sub':
                # Subkategorien: Links oder rechts, je nach x-Position
                if x >= 0:
                    return (x + base_offset, y), 'left'
                else:
                    return (x - base_offset, y), 'right'
            else:
                # Keywords mit kleinerem Abstand
                offset = base_offset * 0.5  # Etwas kleinerer Abstand für Keywords
                # Bestimme Quadranten
                if x >= 0 and y >= 0:  # Quadrant 1 (oben rechts)
                    return (x + offset, y), 'left'
                elif x < 0 and y >= 0:  # Quadrant 2 (oben links)
                    return (x - offset, y), 'right'
                elif x < 0 and y < 0:  # Quadrant 3 (unten links)
                    return (x - offset, y), 'right'
                else:  # Quadrant 4 (unten rechts)
                    return (x + offset, y), 'left'

        # Berechne Label-Positionen und Ausrichtungen
        label_pos = {}
        label_alignments = {}
        
        # KNOTENGRÖSSEN
        main_base_size = 500 
        sub_base_size = 300
        keyword_base_size = 200  

        for node, node_pos in pos.items():
            node_type = G.nodes[node]['node_type']
            # Berechne Knotengröße für diesen Node - MIT DRASTISCH KLEINEREN BASISWERTEN
            if node_type == 'main':
                node_size = main_base_size * node_size_factor * (category_counts.get(node, 1) / max(category_counts.values()))
            elif node_type == 'sub':
                node_size = sub_base_size * node_size_factor * (subcategory_counts.get(node, 1) / max(subcategory_counts.values()))
            else:
                node_size = keyword_base_size * node_size_factor * (keyword_counts.get(node, 1) / max(keyword_counts.values()))
            
            label_position, alignment = get_smart_label_pos(node_pos, node_type, node_size)
            label_pos[node] = label_position
            label_alignments[node] = alignment
        
        # Zeichne Kanten mit angepasstem Stil
        nx.draw_networkx_edges(G, pos,
                            edge_color='gray',
                            alpha=0.4,
                            arrows=True,
                            arrowsize=10,  # Kleinere Pfeile
                            width=0.5,    # Dünnere Linien
                            connectionstyle="arc3,rad=0.1")  # Reduzierter Bogen für besseres Layout
        
        # Zeichne Knoten mit EXTREM KLEINEN BASISGRÖSSEN
        for node_type, counts, base_size, shape in [
            ('main', category_counts, main_base_size, 'o'),
            ('sub', subcategory_counts, sub_base_size, 's'),
            ('keyword', keyword_counts, keyword_base_size, 'd')
        ]:
            node_list = [n for n, d in G.nodes(data=True) if d['node_type'] == node_type]
            
            if node_list:
                # Berechne die Knotengröße mit Skalierungsfaktor und Normalisierung
                node_sizes = [base_size * node_size_factor * (counts.get(node, 1) / max(counts.values()))
                            for node in node_list]
                
                if node_type == 'main':
                    node_colors = [main_category_colors.get(node, default_color) for node in node_list]
                else:
                    node_colors = 'lightgreen' if node_type == 'sub' else 'lightpink'
                
                nx.draw_networkx_nodes(G, pos,
                                    nodelist=node_list,
                                    node_color=node_colors,
                                    node_size=node_sizes,
                                    node_shape=shape,
                                    alpha=0.8)
        
        # Zeichne nur die Knoten-Labels ohne Rahmen für ein aufgeräumteres Aussehen
        # Knoten-Labels direkt zeichnen ohne Hintergrundboxen
        for node_type in ['main', 'sub', 'keyword']:
            nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == node_type]
            for node in nodes:
                label_position = label_pos[node]
                alignment = label_alignments[node]
                
                # Angepasste Texteigenschaften nach Knotentyp
                if node_type == 'main':
                    fontweight = 'bold'
                    fontsize = 12  # Kleinere Schrift
                    alpha = 0.9
                elif node_type == 'sub':
                    fontweight = 'normal'
                    fontsize = 12 # Kleinere Schrift
                    alpha = 0.9
                else:
                    fontweight = 'light'
                    fontsize = 12 # Kleinere Schrift
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
        
        # Legende
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
        
        # Speichern
        print(f"Speichere Netzwerk-Visualisierung...")
        output_path = self.output_dir / f"{output_filename}.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        
        # Zusätzlich als SVG speichern für bessere Bearbeitbarkeit
        svg_output_path = self.output_dir / f"{output_filename}.svg"
        plt.savefig(svg_output_path, format='svg', bbox_inches='tight')
        
        plt.close()
        print(f"Netzwerk-Visualisierung erfolgreich erstellt")

    def create_heatmap(self, filtered_df: pd.DataFrame, output_filename: str, params: Dict[str, Any] = None):
        """
        Erstellt eine Heatmap der Codes entlang der Dokumentattribute
        
        Args:
            filtered_df: Gefilterte DataFrame
            output_filename: Ausgabedateiname
            params: Parameter für die Heatmap-Erstellung
        """
        print("\nErstelle Heatmap der Codes entlang der Dokumentattribute...")
        
        # Verwende Parameter aus der Konfiguration oder Default-Werte
        if params is None:
            params = {}
        
        # Extrahiere Parameter mit Fallback-Werten
        x_attribute = params.get('x_attribute', 'Dokument')
        y_attribute = params.get('y_attribute', 'Hauptkategorie')
        z_attribute = params.get('z_attribute', 'count')  # 'count' oder 'percentage'
        use_subcodes = params.get('use_subcodes', True)
        cmap = params.get('cmap', 'YlGnBu')
        
        # Sicherstellen, dass figsize ein Tuple ist
        figsize_param = params.get('figsize', (14, 10))
        
        # Standard-Figsize
        default_figsize = (14, 10)
        figsize = default_figsize
        
        if isinstance(figsize_param, (int, float)):
            # Falls nur ein einzelner Wert, verwende quadratische Dimension
            figsize = (float(figsize_param), float(figsize_param))
        elif isinstance(figsize_param, str):
            # Falls ein String, versuche zu parsen
            try:
                # Entferne alle Leerzeichen
                clean_param = figsize_param.strip()
                
                # Deutsche Dezimalkomma-Notation abfangen
                if ',' in clean_param and 'x' not in clean_param:
                    # Prüfe, ob es sich um eine deutsche Dezimalzahl handelt
                    parts = clean_param.split(',')
                    if len(parts) == 2:
                        try:
                            # Wenn der zweite Teil eine Zahl ist und kurz, betrachte es als dt. Dezimalzahl
                            float(parts[1])
                            if len(parts[1]) <= 2:  # Typischerweise 1-2 Nachkommastellen
                                print(f"Warnung: '{figsize_param}' scheint eine Dezimalzahl zu sein, nicht ein Tupel.")
                                print(f"Verwende Default-Figsize: {default_figsize}")
                                figsize = default_figsize
                            else:
                                # Sonst behandle es als Tupel-Separator
                                width, height = float(parts[0]), float(parts[1])
                                figsize = (width, height)
                        except ValueError:
                            # Wenn der zweite Teil keine Zahl ist, behandle es als Tupel-Separator
                            width, height = float(parts[0]), float(parts[1])
                            figsize = (width, height)
                    else:
                        # Mehrere Kommas, behandle als Tupel-Separator
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
            # Sollte bereits ein Tupel oder ähnliches sein
            try:
                # Sicherstellen, dass es wirklich ein Tupel mit zwei Werten ist
                if len(figsize_param) == 2:
                    figsize = (float(figsize_param[0]), float(figsize_param[1]))
                else:
                    print(f"Warnung: figsize hat falsche Anzahl von Werten: {figsize_param}")
                    figsize = default_figsize
            except:
                print(f"Warnung: figsize ist kein gültiges Tupel: {figsize_param}")
                figsize = default_figsize
        
        # Parameter für Annotationen
        annot_param = params.get('annot', True)
        if isinstance(annot_param, str):
            # Konvertiere String zu Boolean
            annot = annot_param.lower() in ('true', 'ja', 'yes', '1')
        else:
            annot = bool(annot_param)
        
        fmt = params.get('fmt', '.0f')
        
        # Prüfe, ob die benötigten Attribute vorhanden sind
        if x_attribute not in filtered_df.columns:
            print(f"Warnung: Attribut '{x_attribute}' nicht in Daten vorhanden.")
            return
        
        if y_attribute not in filtered_df.columns and not (y_attribute == 'Subcodes' and 'Subkategorien' in filtered_df.columns):
            print(f"Warnung: Attribut '{y_attribute}' nicht in Daten vorhanden.")
            return
        
        # Spezielle Behandlung für Subcodes, wenn aktiviert
        if use_subcodes or y_attribute == 'Subcodes':
            print("Verwende Subkategorien für die Heatmap...")
            
            # Prüfe, ob Subkategorien verfügbar sind
            if 'Subkategorien' not in filtered_df.columns:
                print("Warnung: Spalte 'Subkategorien' nicht gefunden. Kann keine Subcode-Heatmap erstellen.")
                return
                
            # Erstelle eine kopie des DataFrames zur Bearbeitung
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
        
        # Erzeuge Pivot-Tabelle für die Heatmap
        # Zähle Vorkommen von y_attribute für jedes x_attribute
        pivot_df = pd.crosstab(heatmap_df[y_attribute], heatmap_df[x_attribute])
        
        # Debug: Zeige Dimensionen der Pivot-Tabelle
        print(f"Pivot-Tabelle Dimensionen: {pivot_df.shape} (Zeilen: {pivot_df.shape[0]}, Spalten: {pivot_df.shape[1]})")
        
        # Passe die Figsize basierend auf der Größe der Pivot-Tabelle an
        # Wenn es viele Kategorien gibt, vergrößere die Höhe
        if pivot_df.shape[0] > 10:
            # Berechne eine angemessene Höhe basierend auf der Anzahl der Zeilen
            # Mindestens 10, plus 0.5 für jede zusätzliche Zeile über 10
            adjusted_height = 10 + (pivot_df.shape[0] - 10) * 0.5
            figsize = (figsize[0], adjusted_height)
            print(f"Viele Kategorien entdeckt, passe Figsize an: {figsize}")
        
        # Konvertiere zu Prozent pro Spalte, falls gewünscht
        if z_attribute == 'percentage':
            pivot_df = pivot_df.apply(lambda x: x / x.sum() * 100, axis=0)
            fmt = '.1f'  # Eine Dezimalstelle für Prozentangaben
        
        # Debug-Ausgabe
        print(f"Verwende figsize: {figsize}")
        
        # Erzeuge Heatmap
        plt.figure(figsize=figsize)
        
        # Erzeuge Heatmap mit oder ohne Annotationen
        if annot:
            ax = sns.heatmap(pivot_df, 
                        annot=True,  # Explizit True setzen
                        fmt=fmt, 
                        cmap=cmap,
                        linewidths=0.5,
                        cbar_kws={'label': 'Anzahl' if z_attribute == 'count' else 'Prozent (%)'})
        else:
            ax = sns.heatmap(pivot_df, 
                        annot=False,  # Explizit False setzen
                        cmap=cmap,
                        linewidths=0.5,
                        cbar_kws={'label': 'Anzahl' if z_attribute == 'count' else 'Prozent (%)'})
        
        # Setze Titel und Labels
        plt.title(f"Verteilung: {y_attribute} nach {x_attribute}", fontsize=14)
        plt.xlabel(x_attribute, fontsize=12)
        plt.ylabel(y_attribute, fontsize=12)
        
        # Verbessere Lesbarkeit der Achsenbeschriftungen
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Anpassen des Layouts
        plt.tight_layout()
        
        # Speichern
        output_path = self.output_dir / f"{output_filename}.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        
        # Zusätzlich als PNG speichern für einfachere Verwendung
        png_output_path = self.output_dir / f"{output_filename}.png"
        plt.savefig(png_output_path, format='png', bbox_inches='tight', dpi=300)
        
        # Daten als Excel exportieren
        excel_output_path = self.output_dir / f"{output_filename}_data.xlsx"
        with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
            pivot_df.to_excel(writer, sheet_name='Heatmap')
        
        plt.close()
        print(f"Heatmap erfolgreich erstellt und gespeichert unter: {output_path}")
        print(f"Daten exportiert nach: {excel_output_path}")
    
    async def create_custom_summary(self, 
                            filtered_df: pd.DataFrame, 
                            prompt_template: str, 
                            output_filename: str,
                            model: str,
                            temperature: float = 0.7,
                            filters: Dict[str, str] = None,
                            params: Dict[str, Any] = None):
        """
        Erstellt eine benutzerdefinierte Zusammenfassung der gefilterten Daten
        unter Verwendung eines LLM.
        
        Args:
            filtered_df: Gefilterte DataFrame
            prompt_template: Vorlage für den Prompt
            output_filename: Ausgabedateiname
            model: Name des LLM-Modells
            temperature: Temperature-Parameter für das LLM
            filters: Filter, die auf die Daten angewendet wurden
            params: Zusätzliche Parameter aus der Konfiguration
        """
        print("\nErstelle benutzerdefinierte Zusammenfassung...")
        
        # Sicherstellen, dass params existiert
        if params is None:
            params = {}
        
        if len(filtered_df) == 0:
            print("Warnung: Keine Daten für die Zusammenfassung vorhanden!")
            return
        
        # Prüfen ob prompt_template None ist
        if prompt_template is None:
            print("Warnung: Kein Prompt-Template angegeben! Verwende Standard-Prompt.")
            prompt_template = """
    Bitte analysieren Sie die folgenden Textsegmente:

    {text}

    Erstellen Sie eine umfassende Zusammenfassung der Hauptthemen und Konzepte.
            """
        
        # Sammle den Text für die Zusammenfassung
        text_segments = []
        
        # Verwende die in params angegebene Textspalte, wenn vorhanden
        text_column = params.get('text_column', None)
        
        # Wenn keine Spalte explizit angegeben wurde, versuche passende Spalten zu finden
        if not text_column or text_column not in filtered_df.columns:
            if not text_column:
                print("Keine Textspalte in der Konfiguration angegeben. Suche nach passenden Spalten...")
            else:
                print(f"Angegebene Textspalte '{text_column}' nicht gefunden. Suche nach Alternativen...")
            
            # Standard-Suchprioritäten je nach Analysetyp (aus params)
            analysis_type = params.get('analysis_type', '').lower()
            
            if analysis_type == 'summary_reasoning':
                # Für Begründungsanalyse bevorzuge Spalten mit Begründungen
                preferred_columns = ['Begründung', 'Reasoning', 'Rationale', 'Kodierungsbegründung']
            else:
                # Für andere Analysen bevorzuge Paraphrasen/Textsegmente
                preferred_columns = ['Paraphrase', 'Text', 'Textsegment', 'Segment', 'Kodiertext']
                
            # Erweitere um allgemeine Textspalten als Fallback
            preferred_columns.extend(['Inhalt', 'Content', 'Kommentar', 'Comment'])
            
            # Suche in der Reihenfolge der bevorzugten Spalten
            for column_name in preferred_columns:
                if column_name in filtered_df.columns:
                    text_column = column_name
                    break
                    
            if not text_column:
                # Fallback: Verwende die erste Spalte, die "text", "inhalt", oder "content" im Namen hat
                for col in filtered_df.columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['text', 'inhalt', 'content', 'comment', 'grund']):
                        text_column = col
                        break
        
            if not text_column:
                # Letzter Fallback: Verwende irgendeine Spalte, die nicht Hauptkategorie oder Subkategorie ist
                for col in filtered_df.columns:
                    if col not in ['Hauptkategorie', 'Subkategorien', 'Schlüsselwörter']:
                        text_column = col
                        break
        
        if not text_column:
            print("Fehler: Keine geeignete Spalte für Textdaten gefunden!")
            return
            
        print(f"Verwende Spalte '{text_column}' für die Textzusammenfassung")
        
        # Extrahiere die Textsegmente
        for _, row in filtered_df.iterrows():
            if pd.notna(row[text_column]):
                # Füge Hauptkategorie und Subkategorien hinzu, wenn verfügbar
                segment_info = []
                
                if 'Hauptkategorie' in row and pd.notna(row['Hauptkategorie']):
                    segment_info.append(f"Hauptkategorie: {row['Hauptkategorie']}")
                    
                if 'Subkategorien' in row and pd.notna(row['Subkategorien']):
                    segment_info.append(f"Subkategorien: {row['Subkategorien']}")
                    
                if 'Schlüsselwörter' in row and pd.notna(row['Schlüsselwörter']):
                    segment_info.append(f"Schlüsselwörter: {row['Schlüsselwörter']}")
                    
                # Füge Dokumentinfo hinzu, wenn verfügbar
                if 'Dokument' in row and pd.notna(row['Dokument']):
                    segment_info.append(f"Dokument: {row['Dokument']}")
                    
                # Formatiere das Segment mit den Metadaten
                if segment_info:
                    segment_header = " | ".join(segment_info)
                    text_segments.append(f"[{segment_header}]\n{row[text_column]}\n")
                else:
                    text_segments.append(f"{row[text_column]}\n")
        
        # Kombiniere alle Textsegmente
        combined_text = "\n".join(text_segments)
        
        # Bereite den Prompt vor
        format_args = {'text': combined_text}
        
        # Füge Filter-Info hinzu, wenn vorhanden
        if filters:
            filter_str = ", ".join([f"{k}: {v}" for k, v in filters.items() if v])
            format_args['filters'] = filter_str
        
        # Formatiere den Prompt mit den Daten
        try:
            prompt = prompt_template.format(**format_args)
        except KeyError as e:
            print(f"Warnung: Fehler beim Formatieren des Prompts: {str(e)}")
            # Versuche eine einfachere Formatierung
            try:
                prompt = prompt_template.replace("{text}", combined_text)
                if 'filters' in format_args:
                    prompt = prompt.replace("{filters}", format_args['filters'])
            except Exception as e:
                print(f"Fehler bei der Fallback-Formatierung: {str(e)}")
                # Absoluter Fallback: Verwende den Rohtext mit einem einfachen Prompt
                prompt = f"Bitte analysieren Sie die folgenden Textsegmente und erstellen Sie eine Zusammenfassung:\n\n{combined_text}"
        
        # Führe die LLM-Anfrage durch
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
            
            # Speichere die Zusammenfassung
            output_path = self.output_dir / f"{output_filename}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
                
            print(f"Zusammenfassung erfolgreich erstellt und gespeichert unter: {output_path}")
            
            # Speichere zusätzlich den verwendeten Prompt zur Dokumentation
            prompt_path = self.output_dir / f"{output_filename}_prompt.txt"
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(f"System:\n{system_prompt}\n\nUser:\n{prompt}")
                
            return summary
            
        except Exception as e:
            print(f"Fehler bei der LLM-Anfrage: {str(e)}")
            # Speichere die Fehlermeldung
            error_path = self.output_dir / f"{output_filename}_error.txt"
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"Fehler bei der LLM-Anfrage: {str(e)}\n\nPrompt:\n{prompt}")
            return None

def create_forceatlas_like_layout(G, iterations=100, gravity=0.01, scaling=10.0):
    """Erzeugt ein ForceAtlas2-ähnliches Layout mit NetworkX und scikit-learn
    
    Args:
        G: NetworkX Graph
        iterations: Anzahl Iterationen
        gravity: Stärke der Anziehung zum Zentrum
        scaling: Skalierungsfaktor für Knotenabstände
        
    Returns:
        Dictionary mit Knotenpositionen
    """
    # Imports innerhalb der Funktion platzieren
    import numpy as np
    from sklearn.manifold import MDS
    
    print("Berechne ForceAtlas-ähnliches Layout...")
    
    # Wichtig: Erstelle eine ungerichtete Kopie des Graphen für die Distanzberechnung
    # Dies stellt sicher, dass die Distanzmatrix symmetrisch ist
    G_undirected = G.to_undirected()
    
    # Sonderbehandlung für hierarchische Strukturen
    # Unterschiedliche Gewichtung je nach Knotentyp
    edges = list(G_undirected.edges())
    for src, tgt in edges:
        src_type = G.nodes[src]['node_type']
        tgt_type = G.nodes[tgt]['node_type']
        
        # Gleiche Typen näher zusammen, unterschiedliche Typen weiter weg
        if src_type == tgt_type:
            if src_type == 'main':
                weight = 0.5  # Hauptkategorien etwas näher zusammen
            elif src_type == 'sub':
                weight = 1.0  # Subkategorien normal
            else:
                weight = 2.0  # Keywords weiter auseinander
        else:
            # Verbindungen zwischen verschiedenen Typen
            if (src_type == 'main' and tgt_type == 'sub') or (src_type == 'sub' and tgt_type == 'main'):
                weight = 1.0  # Haupt-zu-Sub: normal
            else:
                weight = 2.0  # Andere Verbindungen: weiter auseinander
        
        # Gewicht direkt im ungerichteten Graphen setzen
        G_undirected[src][tgt]['weight'] = weight
    
    # Verwende NetworkX's integrierte Methode für kürzeste Pfade
    # Dies ist robuster als csgraph für gerichtete Graphen
    try:
        # Berechne alle kürzesten Pfade mit dem ungerichteten Graphen
        distances = dict(nx.all_pairs_shortest_path_length(G_undirected))
        
        # Konvertiere zu Matrix
        nodes = list(G.nodes())
        n = len(nodes)
        distance_matrix = np.zeros((n, n))
        
        # Erstelle eine Lookup-Tabelle für Node-Indizes
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        # Befülle die Distanzmatrix - garantiert symmetrisch
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i == j:
                    # Diagonale mit Nullen
                    distance_matrix[i, j] = 0
                else:
                    try:
                        # Verwende vorberechnete kürzeste Pfade
                        distance_matrix[i, j] = distances[node1][node2]
                    except KeyError:
                        # Für unverbundene Knoten: maximale Distanz
                        distance_matrix[i, j] = n * 2
        
        # Für MDS wird eine symmetrische Matrix benötigt
        # Dies sollte bereits gewährleistet sein, aber wir prüfen zur Sicherheit
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # Verwende MDS (Multidimensional Scaling) als Ersatz für ForceAtlas2
        mds = MDS(
            n_components=2,            # 2D-Layout
            dissimilarity='precomputed',  # Wir liefern Distanzmatrix
            random_state=42,           # Für Reproduzierbarkeit
            n_init=1,                  # Eine Initialisierung
            max_iter=iterations,       # Iterationen
            normalized_stress=False,   # Deaktiviere normalisiertem Stress für metrisches MDS
            n_jobs=1                   # Single-threaded für Stabilität
        )
                
        # Wende MDS auf die Distanzmatrix an
        pos_array = mds.fit_transform(distance_matrix)
        
        # Skalierungen hinzufügen
        pos_array *= scaling
        
        # Schwache Anziehung zum Zentrum (Gravitation)
        # Dies ist eine vereinfachte Version der Gravitation in ForceAtlas2
        if gravity > 0:
            center = np.mean(pos_array, axis=0)
            for i in range(pos_array.shape[0]):
                direction = center - pos_array[i]
                distance = np.linalg.norm(direction)
                if distance > 0:  # Vermeiden der Division durch Null
                    # Je weiter weg, desto stärker die Anziehung
                    force = gravity * distance
                    pos_array[i] += direction / distance * force
        
        # Konvertiere zurück zum Dictionary
        pos = {nodes[i]: (pos_array[i, 0], pos_array[i, 1]) for i in range(n)}
        
        # Optionale weitere Anpassungen:
        # 1. Knoten nach Typ gruppieren
        node_by_type = {'main': [], 'sub': [], 'keyword': []}
        for node in G.nodes():
            node_type = G.nodes[node]['node_type']
            node_by_type[node_type].append(node)
        
        # 2. Leichte Anziehung innerhalb der gleichen Typen
        for node_type, nodes_of_type in node_by_type.items():
            if len(nodes_of_type) > 1:
                # Finde den Schwerpunkt dieser Gruppe
                centroid = np.mean([pos[node] for node in nodes_of_type], axis=0)
                
                # Anziehungskoeffizient je nach Typ
                attraction = 0.1 if node_type == 'main' else 0.05 if node_type == 'sub' else 0.01
                
                # Bewege alle Knoten etwas in Richtung ihres Zentroids
                for node in nodes_of_type:
                    curr_pos = np.array(pos[node])
                    direction = centroid - curr_pos
                    # Mische aktuelle Position mit einer leichten Anziehung zum Zentroid
                    new_pos = curr_pos + direction * attraction
                    pos[node] = (new_pos[0], new_pos[1])
        
        print("ForceAtlas-ähnliches Layout berechnet.")
        return pos
        
    except Exception as e:
        print(f"Fehler bei ForceAtlas-Layout: {str(e)}")
        print("Falle zurück auf Spring-Layout als Alternative...")
        
        # Import auch hier für den Fallback benötigt
        import numpy as np
        
        # Erstelle initiale Positionen für bessere Konvergenz
        initial_pos = {}
        
        # Positioniere Hauptkategorien im Zentrum
        main_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'main']
        num_main = len(main_nodes)
        for i, node in enumerate(main_nodes):
            if num_main > 1:
                angle = 2 * np.pi * i / num_main
                initial_pos[node] = (0.2 * np.cos(angle), 0.2 * np.sin(angle))
            else:
                initial_pos[node] = (0, 0)
        
        # Positioniere Subkategorien in einem mittleren Ring
        sub_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'sub']
        num_sub = len(sub_nodes)
        for i, node in enumerate(sub_nodes):
            if num_sub > 1:
                angle = 2 * np.pi * i / num_sub
                initial_pos[node] = (0.5 * np.cos(angle), 0.5 * np.sin(angle))
            else:
                initial_pos[node] = (0.5, 0)
        
        # Positioniere Keywords in einem äußeren Ring
        keyword_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'keyword']
        num_kw = len(keyword_nodes)
        for i, node in enumerate(keyword_nodes):
            if num_kw > 1:
                angle = 2 * np.pi * i / num_kw
                initial_pos[node] = (1.0 * np.cos(angle), 1.0 * np.sin(angle))
            else:
                initial_pos[node] = (1.0, 0)
        
        # Verwende das Spring-Layout als Fallback mit gewichteten Kanten
        # für eine bessere hierarchische Darstellung
        pos = nx.spring_layout(
            G,
            pos=initial_pos,
            k=1.5/np.sqrt(len(G.nodes())),
            iterations=100,
            weight='weight',
            scale=scaling
        )
        
        return pos

def create_filter_string(filters: Dict[str, str]) -> str:
        """Create a string representation of the filters for filenames"""
        return '_'.join(f"{k}-{v}" for k, v in filters.items() if v)


def get_default_prompts() -> Dict[str, str]:
    """Gibt die Standard-Prompts für verschiedene Analysetypen zurück"""
    prompts = {
        "paraphrase": """
Bitte analysieren Sie die folgenden paraphrasierten Textabschnitte und erstellen Sie einen thematischen Überblick:

{text}

Bitte geben Sie an:
1. Zusammanfassung identifizierter Hauptthemen. Ergänze dies anschließend mit konkreten Beispieln
2. Zusammanfassung wichtiger Muster und Beziehungen. Ergänze dies anschließend mit konkreten Beispieln
3. Zusammenfassung der Ergebnisse
        """,
        
        "reasoning": """
Bitte analysieren Sie die folgenden Begründungen für die Inklusion von Textsegementen in diesen Kategorien: {filters} 

Erstellen Sie eine umfassende Zusammenfassung:

{text}

Bitte geben Sie an:
1. Identifizierte Hauptargumente, die zur Inklusion in die Kategorien führten
2. Gemeinsame Muster in der Argumentation
3. Zusammenfassung der wichtigsten Begründungen
        """
    }
    return prompts

# ------------------------------------ #
# ---      HAUPTFUNKTION           --- #
# ------------------------------------ #
async def main():
    print("\n=== QCA-AID Explorer v0.4 Start ===")
    print("Konfiguration über Excel-Datei")
    
    # Pfad zur Konfigurations-Excel-Datei
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_FILE = "QCA-AID-Explorer-Config.xlsx"
    CONFIG_PATH = os.path.join(SCRIPT_DIR, CONFIG_FILE)
    
    # Lade Konfiguration
    config_loader = ConfigLoader(CONFIG_PATH)
    base_config = config_loader.get_base_config()
    analysis_configs = config_loader.get_analysis_configs()
    
    # Extrahiere Basis-Parameter
    PROVIDER_NAME = base_config.get('provider', 'openai')
    MODEL_NAME = base_config.get('model', 'gpt-4o-mini')
    TEMPERATURE = float(base_config.get('temperature', 0.7))
    SCRIPT_DIR = base_config.get('script_dir') or os.path.dirname(os.path.abspath(__file__))
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
    print(f"Verfügbare Spalten: {', '.join(analyzer.columns)}")
    
    # Standardprompts laden
    default_prompts = get_default_prompts()
    
    # Schlüsselwörter harmonisieren, falls aktiviert
    if CLEAN_KEYWORDS:
        print("\nStarte Keyword-Harmonisierung...")
        # Update the main DataFrame with harmonized keywords
        analyzer.df = analyzer.harmonize_keywords(similarity_threshold=SIMILARITY_THRESHOLD)
        analyzer.validate_keyword_mapping()
        print("Keyword-Harmonisierung abgeschlossen.")
    
    # Analysekonfigurationen durchlaufen
    print(f"\nFühre {len(analysis_configs)} Auswertungen durch...")
    
    for config_idx, analysis_config in enumerate(analysis_configs, 1):
        analysis_name = analysis_config['name']
        print(f"\n--- Auswertung {config_idx}/{len(analysis_configs)}: {analysis_name} ---")
        
        # Extrahiere Filter und Parameter
        filters = analysis_config['filters']
        params = analysis_config['params']
        
        # Füge analysis_type zu params hinzu, falls nicht vorhanden
        if 'analysis_type' not in params:
            params['analysis_type'] = analysis_config.get('analysis_type', '')
        
        # Filtere die Daten
        filtered_df = analyzer.filter_data(filters)
        
        # Erzeuge Filterstring für Dateinamen
        filter_str = create_filter_string(filters)
        
        # Schlüsselwort-Harmonisierungsinfo zum Filterstring hinzufügen, falls aktiviert
        if CLEAN_KEYWORDS:
            filter_str = f"harmonized_{filter_str}" if filter_str else "harmonized"
        
        # Füge Analysenamen zum Filterstring hinzu
        output_prefix = f"{analysis_name}_{filter_str}" if filter_str else analysis_name
        
        # Bestimme den Analysetyp
        analysis_type = params.get('analysis_type', '').lower()
        
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
            
        elif analysis_type == 'summary_paraphrase':
            # Paraphrasenzusammenfassung erstellen
            print("\nErstelle Paraphrasen-Zusammenfassung...")
            # Verwende benutzerdefinierten Prompt, falls vorhanden
            prompt_template = params.get('prompt_template', default_prompts.get('paraphrase', None))
            
            # ÜBERGEBE AUCH DIE PARAMETER
            await analyzer.create_custom_summary(
                filtered_df,
                prompt_template,
                f"Summary_Paraphrase_{output_prefix}",
                MODEL_NAME,
                TEMPERATURE,
                filters,
                params  # Übergebe die Parameter an die Methode
            )
            
        elif analysis_type == 'summary_reasoning':
            # Begründungszusammenfassung erstellen
            print("\nErstelle Begründungs-Zusammenfassung...")
            # Verwende benutzerdefinierten Prompt, falls vorhanden
            prompt_template = params.get('prompt_template', default_prompts.get('reasoning', None))
            
            # ÜBERGEBE AUCH DIE PARAMETER
            await analyzer.create_custom_summary(
                filtered_df,
                prompt_template,
                f"Summary_Begruendung_{output_prefix}",
                MODEL_NAME,
                TEMPERATURE,
                filters,
                params  # Übergebe die Parameter an die Methode
            )
            
        elif analysis_type == 'custom_summary':
            # Benutzerdefinierte Zusammenfassung erstellen
            print("\nErstelle benutzerdefinierte Zusammenfassung...")
            # Muss einen benutzerdefinierten Prompt haben
            prompt_template = params.get('prompt_template', '')
            if not prompt_template:
                print("Warnung: Kein Prompt-Template für benutzerdefinierte Zusammenfassung angegeben. Überspringe.")
                continue
                
            # ÜBERGEBE AUCH DIE PARAMETER
            await analyzer.create_custom_summary(
                filtered_df,
                prompt_template,
                f"Summary_Custom_{output_prefix}",
                MODEL_NAME,
                TEMPERATURE,
                filters,
                params  # Übergebe die Parameter an die Methode
            )
            
        else:
            print(f"Warnung: Unbekannter Analysetyp '{analysis_type}'. Überspringe.")
    
    print("\n=== QCA Analyse abgeschlossen ===\n")

if __name__ == "__main__":
    try:
        # Windows-spezifische Event Loop Policy setzen
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Hauptprogramm ausführen
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer beendet")
    except Exception as e:
        print(f"Fehler im Hauptprogramm: {str(e)}")
        raise