import pandas as pd
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
        """Filter the dataframe based on provided column-value pairs"""
        filtered_df = self.df.copy()
        for col, value in filters.items():
            if value:
                filtered_df = filtered_df[filtered_df[col] == value]
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
        """Create and save network graph visualization with improved parameters and export network data"""
        print("Erstelle Netzwerkgraph...")
        G = nx.DiGraph()
        
        # Count occurrences for node sizing
        category_counts = {}
        subcategory_counts = {}
        keyword_counts = {}
        
        # Color mapping for main categories
        unique_main_categories = filtered_df['Hauptkategorie'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_main_categories)))
        main_category_colors = dict(zip(unique_main_categories, colors))
        
        # Lists to store node and edge data for Excel export
        nodes_data = []
        edges_data = []
        
        # Process each row
        print("Verarbeite Datensätze...")
        total_rows = len(filtered_df)
        rows_processed = 0
        
        for _, row in filtered_df.iterrows():
            if pd.isna(row['Hauptkategorie']):
                continue
                
            main_category = str(row['Hauptkategorie'])
            category_counts[main_category] = category_counts.get(main_category, 0) + 1
            
            # Add main category node
            if main_category not in [node[0] for node in nodes_data]:
                G.add_node(main_category, node_type='main')  # Add node_type attribute here
                nodes_data.append([main_category, 'main', category_counts[main_category]])
            
            # Handle subcategories
            sub_categories = []
            if pd.notna(row['Subkategorien']):
                sub_categories = [cat.strip() for cat in str(row['Subkategorien']).split(',') if cat.strip()]
                for sub_cat in sub_categories:
                    subcategory_counts[sub_cat] = subcategory_counts.get(sub_cat, 0) + 1
                    
                    # Add subcategory node if not exists
                    if sub_cat not in [node[0] for node in nodes_data]:
                        G.add_node(sub_cat, node_type='sub')  # Add node_type attribute here
                        nodes_data.append([sub_cat, 'sub', subcategory_counts[sub_cat]])
                    G.add_edge(main_category, sub_cat)
                    edges_data.append([main_category, sub_cat, 'main_to_sub'])
            
            # Handle keywords
            keywords = []
            if pd.notna(row['Schlüsselwörter']):
                keywords = [kw.strip() for kw in str(row['Schlüsselwörter']).split(',') if kw.strip()]
                for kw in keywords:
                    keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
                    
                    # Add keyword nodes and connect to subcategories
                    for sub_cat in sub_categories:
                        if kw not in [node[0] for node in nodes_data]:
                            G.add_node(kw, node_type='keyword')  # Add node_type attribute here
                            nodes_data.append([kw, 'keyword', keyword_counts[kw]])
                        G.add_edge(sub_cat, kw)
                        edges_data.append([sub_cat, kw, 'sub_to_keyword'])
            
            rows_processed += 1
            if rows_processed % max(1, total_rows // 5) == 0:
                print(f"Fortschritt: {rows_processed}/{total_rows} Datensätze verarbeitet")
        
        # Export network data to Excel
        print("\nExportiere Netzwerkdaten nach Excel...")
        nodes_df = pd.DataFrame(nodes_data, columns=['Node', 'Type', 'Count'])
        edges_df = pd.DataFrame(edges_data, columns=['Source', 'Target', 'Type'])
        
        excel_output_path = self.output_dir / f"{output_filename}_network_data.xlsx"
        with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
            nodes_df.to_excel(writer, sheet_name='Nodes', index=False)
            edges_df.to_excel(writer, sheet_name='Edges', index=False)
        print(f"Netzwerkdaten exportiert nach: {excel_output_path}")
        
        # Starte die Visualisierung
        print("\nErstelle Visualisierung...")
        plt.figure(figsize=(24, 20))
        
        # Sammle Knoten nach Typ
        main_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'main']
        sub_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'sub']
        keyword_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'keyword']
        
        # Berechne Positionen für Hauptkategorien im Zentrum
        num_main = len(main_nodes)
        main_radius = 0.3
        main_pos = {}
        for i, node in enumerate(main_nodes):
            angle = 2 * np.pi * i / num_main
            main_pos[node] = (main_radius * np.cos(angle), main_radius * np.sin(angle))
        
        # Berechne Positionen für Subkategorien in mittlerem Ring
        sub_radius = 0.6
        sub_pos = {}
        for i, node in enumerate(sub_nodes):
            main_neighbors = [n for n in G.predecessors(node) if n in main_nodes]
            if main_neighbors:
                main_x, main_y = main_pos[main_neighbors[0]]
                angle = 2 * np.pi * i / len(sub_nodes)
                offset_x = sub_radius * np.cos(angle)
                offset_y = sub_radius * np.sin(angle)
                sub_pos[node] = (0.7 * offset_x + 0.3 * main_x, 
                            0.7 * offset_y + 0.3 * main_y)
            else:
                angle = 2 * np.pi * i / len(sub_nodes)
                sub_pos[node] = (sub_radius * np.cos(angle), 
                            sub_radius * np.sin(angle))
        
        # Berechne Positionen für Keywords im äußeren Ring
        keyword_radius = 1.0
        keyword_pos = {}
        for i, node in enumerate(keyword_nodes):
            sub_neighbors = [n for n in G.predecessors(node) if n in sub_nodes]
            if sub_neighbors:
                sub_x, sub_y = sub_pos[sub_neighbors[0]]
                angle = 2 * np.pi * i / len(keyword_nodes)
                offset_x = keyword_radius * np.cos(angle)
                offset_y = keyword_radius * np.sin(angle)
                keyword_pos[node] = (0.7 * offset_x + 0.3 * sub_x,
                                0.7 * offset_y + 0.3 * sub_y)
            else:
                angle = 2 * np.pi * i / len(keyword_nodes)
                keyword_pos[node] = (keyword_radius * np.cos(angle),
                                keyword_radius * np.sin(angle))
        
        # Kombiniere alle Positionen
        pos = {**main_pos, **sub_pos, **keyword_pos}

        def get_smart_label_pos(node_pos, node_type, node_size):
            """Berechne intelligente Label-Position basierend auf Position im Plot und Knotengröße"""
            x, y = node_pos
            
            # Berechne den Radius des Knotens aus der Knotengröße
            # Die Formel sqrt(node_size/pi) gibt uns den ungefähren Radius in Punkten
            node_radius = np.sqrt(node_size/np.pi) / 1000  # Skalierungsfaktor für bessere Anpassung
            # print(f"node_radius: {node_radius}")
            # Basis-Offset ist nun proportional zur Knotengröße
            base_offset = max(0.01, node_radius * 1)  # Mindestens 0.3 Einheiten Abstand
            
            if node_type == 'main':
                # Hauptkategorien: Labels immer nach rechts mit größerem Abstand
                return (x, y + base_offset), 'center'
            elif node_type == 'sub':
                # Subkategorien: Links oder rechts, je nach x-Position
                if x >= 0:
                    return (x + base_offset, y), 'left'
                else:
                    return (x - base_offset, y), 'right'
            else:
                # Keywords mit kleinerem Abstand
                offset = base_offset * 0.3  # Etwas kleinerer Abstand für Keywords
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
        
        # Zeichne Kanten
        nx.draw_networkx_edges(G, pos,
                            edge_color='gray',
                            alpha=0.3,
                            arrows=True,
                            arrowsize=10,
                            width=0.5,
                            connectionstyle="arc3,rad=0.2")
        
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
                                    alpha=0.7)
        
        # Zeichne Labels für jeden Node-Typ separat
        for node_type in ['main', 'sub', 'keyword']:
            nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == node_type]
            for node in nodes:
                label_position = label_pos[node]
                alignment = label_alignments[node]
                plt.text(label_position[0], label_position[1], 
                        node,
                        horizontalalignment=alignment,
                        fontsize=8,
                        fontweight='bold',
                        bbox=dict(facecolor='white',
                                edgecolor='none',
                                alpha=0.7,
                                pad=0.5))
        
        plt.title("Code Network Analysis", pad=20, fontsize=16)
        plt.axis('off')
        
        # Legende
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=list(main_category_colors.values())[0],
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

def create_filter_string(filters: Dict[str, str]) -> str:
    """Create a string representation of the filters for filenames"""
    return '_'.join(f"{k}-{v}" for k, v in filters.items() if v)


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

    # --- HIER DATEINAMEN EINTRAGEN --- #
    # --- DATEI SCHLIESSEN VOR START DES SKRIPTS --- #
    EXPLORE_FILE = "QCA-AID_Analysis_20250222_192824.xlsx"  
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
        "Subkategorien" : None,
        second_column: None,  # Spalte mit "Attribut_1"
        third_column: None    # Spalte mit "Attribut_2"
    }
    # -------------------------------------------------


    # Filter data
    filtered_df = analyzer.filter_data(filters)
    
    # Create filter string for filenames
    filter_str = create_filter_string(filters)
    
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
    # paraphrase_summary = await analyzer.generate_summary(
    #     text=paraphrase_text,
    #     prompt_template=paraphrase_prompt,
    #     model=MODEL_NAME,
    #     temperature=TEMPERATURE
    # )
    # output_file = f"Summary_Paraphrase_{filter_str}.txt"
    # analyzer.save_text_summary(
    #     paraphrase_summary,
    #     output_file
    # )
    # print(f"Zusammenfassung gespeichert in: {output_file}")
    
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
    # reasoning_summary = await analyzer.generate_summary(
    #     text=reasoning_text,
    #     prompt_template=reasoning_prompt,
    #     model=MODEL_NAME,
    #     temperature=TEMPERATURE,
    #     filters=filters
    # )
    # output_file = f"Summary_Begründung_{filter_str}.txt"
    # analyzer.save_text_summary(
    #     reasoning_summary,
    #     output_file
    # )
    # print(f"Zusammenfassung gespeichert in: {output_file}")
    
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
