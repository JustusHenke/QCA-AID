"""
QCA-AID Explorer 
========================================================================
Version:
--------
0.3 (2025-02-24)
New in this version
- ForceAtlas2 layout for network graphs
- Enhanced visualization with hierarchical node placement
- SVG export for better editability
- Improved node and label styling
- Fallback to spring layout for stability

QCA-AID Explorer ist ein Tool zur Analyse von qualitativen Kodierungsdaten.
Es ermöglicht die Visualisierung von Kodierungsnetzwerken mit Hauptkategorien,
Subkategorien und Schlüsselwörtern sowie die automatisierte Zusammenfassung
von kodierten Textsegmenten mit Hilfe von LLM-Modellen.

Das ForceAtlas2-Layout bietet eine verbesserte Netzwerkvisualisierung,
bei der zusammengehörige Elemente näher beieinander platziert werden
und die hierarchische Struktur der Daten besser sichtbar wird.
"""

import pandas as pd
from difflib import SequenceMatcher
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional
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

class QCAAnalyzer:
    def __init__(self, excel_path: str, llm_provider: LLMProvider):
        """
        Initialize the QCA Analyzer
        
        Args:
            excel_path: Path to the Excel file
            llm_provider: Instance of LLMProvider
        """
        # Read specifically from 'Kodierte_Segmente' sheet
        self.df = pd.read_excel(excel_path, sheet_name='Kodierte_Segmente')
        self.llm_provider = llm_provider
        self.keyword_mappings = {}
        
        # Get the input filename without extension
        input_filename = Path(excel_path).stem

        # Set output directory relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_output_dir = Path(script_dir) / 'output'
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

    def create_network_graph(self, filtered_df: pd.DataFrame, output_filename: str):
        """Create network graph using harmonized keywords"""
        print("Erstelle Netzwerkgraph...")
        
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
        plt.figure(figsize=(24, 20))

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
        # Sie können diese Parameter anpassen, um das Layout zu optimieren
        pos = create_forceatlas_like_layout(
            G,
            iterations=100,    # Mehr Iterationen = besseres Layout, aber langsamer
            gravity=0.05,      # Stärke der Anziehung zum Zentrum (0.01-0.1 empfohlen)
            scaling=2.0        # Skalierungsfaktor (1.0-10.0 empfohlen)
        )
        # --- ENDE NEUER CODE ---

        # Define label placement function
        def get_smart_label_pos(node_pos, node_type, node_size):
            """Berechne intelligente Label-Position basierend auf Position im Plot und Knotengröße"""
            x, y = node_pos
            
            # Berechne den Radius des Knotens aus der Knotengröße
            node_radius = np.sqrt(node_size/np.pi) / 1000
            base_offset = max(0.05, node_radius * 1.2)  # Etwas größerer Offset für bessere Lesbarkeit
            
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
        for node, node_pos in pos.items():
            node_type = G.nodes[node]['node_type']
            # Berechne Knotengröße für diesen Node
            if node_type == 'main':
                node_size = 3000 * (category_counts.get(node, 1) / max(category_counts.values()))
            elif node_type == 'sub':
                node_size = 2000 * (subcategory_counts.get(node, 1) / max(subcategory_counts.values()))
            else:
                node_size = 1000 * (keyword_counts.get(node, 1) / max(keyword_counts.values()))
            
            label_position, alignment = get_smart_label_pos(node_pos, node_type, node_size)
            label_pos[node] = label_position
            label_alignments[node] = alignment
        
        # Zeichne Kanten mit angepasstem Stil
        nx.draw_networkx_edges(G, pos,
                            edge_color='gray',
                            alpha=0.4,
                            arrows=True,
                            arrowsize=10,
                            width=0.5,
                            connectionstyle="arc3,rad=0.1")  # Reduzierter Bogen für besseres Layout
        
        # Zeichne Knoten
        for node_type, counts, base_size, shape in [
            ('main', category_counts, 3000, 'o'),
            ('sub', subcategory_counts, 2000, 's'),
            ('keyword', keyword_counts, 1000, 'd')
        ]:
            node_list = [n for n, d in G.nodes(data=True) if d['node_type'] == node_type]
            
            if node_list:
                node_sizes = [base_size * (counts.get(node, 1) / max(counts.values()))
                            for node in node_list]
                
                if node_type == 'main':
                    node_colors = [main_category_colors[node] for node in node_list]
                else:
                    node_colors = 'lightgreen' if node_type == 'sub' else 'lightpink'
                
                nx.draw_networkx_nodes(G, pos,
                                    nodelist=node_list,
                                    node_color=node_colors,
                                    node_size=node_sizes,
                                    node_shape=shape,
                                    alpha=0.8)
        
        # Zeichne Labels für jeden Node-Typ separat mit verbesserten Textboxen
        for node_type in ['main', 'sub', 'keyword']:
            nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == node_type]
            for node in nodes:
                label_position = label_pos[node]
                alignment = label_alignments[node]
                
                # Angepasste Texteigenschaften nach Knotentyp
                if node_type == 'main':
                    fontweight = 'bold'
                    fontsize = 10
                    alpha = 0.85
                elif node_type == 'sub':
                    fontweight = 'medium'
                    fontsize = 9
                    alpha = 0.8
                else:
                    fontweight = 'normal'
                    fontsize = 8
                    alpha = 0.75
                    
                plt.text(label_position[0], label_position[1], 
                        node,
                        horizontalalignment=alignment,
                        fontsize=fontsize,
                        fontweight=fontweight,
                        bbox=dict(facecolor='white',
                                edgecolor='none',
                                alpha=alpha,
                                pad=0.5))
        
        plt.title("Code Network Analysis (ForceAtlas2-Style Layout)", pad=20, fontsize=16)
        plt.axis('off')
        
        # Legende
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=default_color,
                    markersize=10, label='Hauptkategorien'),
            plt.Line2D([0], [0], marker='s', color='w',
                    markerfacecolor='lightgreen', markersize=10,
                    label='Subkategorien'),
            plt.Line2D([0], [0], marker='d', color='w',
                    markerfacecolor='lightpink', markersize=10,
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

    def _format_filters_for_prompt(self, filters: Dict[str, str]) -> str:
        """
        Formatiert Filter-Dictionary in einen lesbaren Text
        
        Args:
            filters: Dictionary mit Spaltenname-Wert Paaren
            
        Returns:
            Formatierter Text der aktiven Filter
        """
        active_filters = []
        for column, value in filters.items():
            if value:  # Nur aktive Filter einbeziehen
                # Sonderbehandlung für bestimmte Spaltennamen
                if column == "Hauptkategorie":
                    active_filters.append(f"der Hauptkategorie '{value}'")
                elif column == "Subkategorien":
                    active_filters.append(f"der Subkategorie '{value}'")
                elif column == "Dokument":
                    active_filters.append(f"aus dem Dokument '{value}'")
                else:
                    active_filters.append(f"mit {column} = '{value}'")
        
        if not active_filters:
            return "ohne spezifische Filterkriterien"
            
        if len(active_filters) == 1:
            return active_filters[0]
            
        # Für mehrere Filter: Alle außer dem letzten mit Komma, letzter mit "und"
        return ", ".join(active_filters[:-1]) + " und " + active_filters[-1]

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

# --- Helper functions --- #

def create_filter_string(filters: Dict[str, str]) -> str:
    """Create a string representation of the filters for filenames"""
    return '_'.join(f"{k}-{v}" for k, v in filters.items() if v)

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


# ------------------------------------ #
# ---      ANPASSUNGEN AB HIER     --- #
# ------------------------------------ #
async def main():
    print("\n=== QCA Analyse Start ===")
    # LLM Configuration
    PROVIDER_NAME = "openai"  # or "mistral"
    MODEL_NAME = "gpt-4o-mini"  # or "mistral-medium" etc.
    TEMPERATURE = 0.7
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = "output"

    # --- KONFIGURATIONSMÖGLICHKEITEN --- #
    CLEAN_KEYWORDS = True  # Auf True setzen, um die Harmonisierung von Schlüsselwörtern zu aktivieren
    SIMILARITY_THRESHOLD = 0.7  # Anpassen des Ähnlichkeitsschwellenwerts für den Schlüsselwortabgleich

    # --- HIER DATEINAMEN EINTRAGEN --- #
    # --- DATEI SCHLIESSEN VOR START DES SKRIPTS --- #
    EXPLORE_FILE = "QCA-AID_Analysis_20250224_201708.xlsx"  
    # -------------------------------------------------


    # Path Configuration
    EXCEL_PATH = os.path.join(SCRIPT_DIR, OUTPUT_DIR, EXPLORE_FILE)
    print(f"\nLese Excel-Datei: {EXCEL_PATH}")
    
     # Initialize LLM provider
    print("\nInitialisiere LLM Provider...")
    llm_provider = LLMProviderFactory.create_provider(PROVIDER_NAME)
    
    # Initialize analyzer
    analyzer = QCAAnalyzer(EXCEL_PATH, llm_provider)
    print(f"Verfügbare Spalten: {', '.join(analyzer.columns)}")
    
    # Get the column names dynamically
    second_column = analyzer.columns[1] if len(analyzer.columns) > 1 else None
    third_column = analyzer.columns[2] if len(analyzer.columns) > 2 else None
    

    # -------------------------------------------------
    # HIER FILTER DEFINIEREN (nach Bedarf anpassen)
    # -------------------------------------------------
    filters = {
        "Dokument": None,  # Optional: set to None if not filtering
        "Hauptkategorie": "Strukturelle Rahmenbedingungen",
        "Subkategorien" : "Finanzierung",
        second_column: None,  # Spalte mit "Attribut_1"
        third_column: None    # Spalte mit "Attribut_2"
    }
    # -------------------------------------------------

    # Schlüsselwörter harmonisieren, falls aktiviert
    if CLEAN_KEYWORDS:
        print("\nStarte Keyword-Harmonisierung...")
        # Update the main DataFrame with harmonized keywords
        analyzer.df = analyzer.harmonize_keywords(similarity_threshold=SIMILARITY_THRESHOLD)
        analyzer.validate_keyword_mapping()
        print("Keyword-Harmonisierung abgeschlossen.")

    # Filter data
    filtered_df = analyzer.filter_data(filters)
    
    # Create filter string for filenames
    filter_str = create_filter_string(filters)

    # Schlüsselwort-Harmonisierungsinfo zum Filterstring hinzufügen, falls aktiviert
    if CLEAN_KEYWORDS:
        filter_str = f"harmonized_{filter_str}" if filter_str else "harmonized"
    
    # 1. Paraphrasenzusammenfassung erstellen und speichern
    # -------------------------------------------------
    paraphrase_prompt = """
    Bitte analysieren Sie die folgenden paraphrasierten Textabschnitte und erstellen Sie einen thematischen Überblick:

    {text}

    Bitte geben Sie an:
    1. Zusammanfassung identifizierter Hauptthemen. Ergänze dies anschließend mit konkreten Beispieln
    2. Zusammanfassung wichtiger Muster und Beziehungen. Ergänze dies anschließend mit konkreten Beispieln
    3. Zusammenfassung der Ergebnisse
    """
    
    paraphrase_text = '\n'.join(filtered_df['Paraphrase'].dropna())
    print("Sende Anfrage an LLM...")
    paraphrase_summary = await analyzer.generate_summary(
        text=paraphrase_text,
        prompt_template=paraphrase_prompt,
        model=MODEL_NAME,
        temperature=TEMPERATURE
    )
    output_file = f"Summary_Paraphrase_{filter_str}.txt"
    analyzer.save_text_summary(
        paraphrase_summary,
        output_file
    )
    print(f"Zusammenfassung gespeichert in: {output_file}")
    
    # 2. Zusammenfassung der Argumentation generieren und speichern
    # -------------------------------------------------
    reasoning_prompt = """
    Bitte analysieren Sie die folgenden Begründungen für die Inklusion von Textsegementen in diesen Kategorien: {filters} 
    
    Erstellen Sie eine umfassende Zusammenfassung:

    {text}

    Bitte geben Sie an:
    1. Identifizierte Hauptargumente, die zur Inklusion in die Kategorien führten
    2. Gemeinsame Muster in der Argumentation
    3. Zusammenfassung der wichtigsten Begründungen
    """
    
    reasoning_text = '\n'.join(filtered_df['Begründung'].dropna())
    print("Sende Anfrage an LLM...")
    reasoning_summary = await analyzer.generate_summary(
        text=reasoning_text,
        prompt_template=reasoning_prompt,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        filters=filters
    )
    output_file = f"Summary_Begründung_{filter_str}.txt"
    analyzer.save_text_summary(
        reasoning_summary,
        output_file
    )
    print(f"Zusammenfassung gespeichert in: {output_file}")
    
    # 3. Netzwerkvisualisierung erstellen und speichern
    # -------------------------------------------------
    print("\nErstelle Netzwerk-Visualisierung...")
    output_file = f"Code-Network_{filter_str}.pdf"
    analyzer.create_network_graph(
        filtered_df,
        f"Code-Network_{filter_str}"
    )
    print(f"Netzwerk-Visualisierung gespeichert in: {output_file}")
    
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
