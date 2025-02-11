"""
QCA-AID: Qualitative Content Analysis with AI Support - Deductive Coding
========================================================================

A Python implementation of Mayring's Qualitative Content Analysis methodology,
enhanced with AI capabilities through the OpenAI API.

Version:
--------
0.9 (2025-02-12)

Description:
-----------
This script provides a framework for conducting qualitative content analysis
following Mayring's approach, combining both deductive and inductive category
development. It supports automated coding through AI while maintaining
methodological rigor and transparency.

ATTENTION: 
Be aware that this script is still under development and not all features are available. Also keep
in mind that AI-Assistance is not perfect and the results depend on the quality of the input data.
Use the script at your own risk!
Feedback is welcome!

Key Features:
------------
- Automated text preprocessing and chunking
- Deductive category application
- Inductive category development
- Multi-coder support (AI and human)
- Intercoder reliability calculation
- Comprehensive analysis export
- Detailed documentation of the coding process

Requirements:
------------
- Python 3.8+
- OpenAI API key
- Required packages: see requirements.txt

Usage:
------
1. Place interview transcripts in the 'transkripte' directory
2. Configure .env file with OpenAI API key
3. Adjust CONFIG settings if needed
4. Run the script
5. Results will be exported to the 'output' directory

File Naming Convention:
---------------------
Interview files should follow the pattern: attribute1_attribute2_whatever_you_want.extension
Example: university-type_position_2024-01-01.txt

Author:
-------
Justus Henke 
Institut für Hochschulforschung Halle-Wittenberg (HoF)
Contact: justus.henke@hof.uni-halle.de

License:
--------
MIT License
Copyright (c) 2025 Justus Henke


Repository:
----------
https://github.com/JustusHenke/QCA-AID

Citation:
--------
If you use this software in your research, please cite:
Henke, J. (2025). QCA-AID: Qualitative Content Analysis with AI Support [Computer software].
https://github.com/JustusHenke/QCA-AID 

"""

# ============================
# 1. Import notwendiger Module
# ============================
import os        # Dateisystemzugriff
import re        # Reguläre Ausdrücke für deduktives Codieren
import openai    # OpenAI API-Integration
import json      # Export/Import von Daten (z.B. CSV/JSON)
import pandas as pd  # Zum Aggregieren und Visualisieren der Ergebnisse
import logging   # Protokollierung
import markdown  # Für Markdown-Konvertierung
from datetime import datetime  # Für Datum und Zeit
from dotenv import load_dotenv  # Für das Laden von Umgebungsvariablen
import asyncio  # Für asynchrone Programmierung
import spacy
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING, Tuple
from typing import List, Tuple, Set
from dataclasses import dataclass
from openai import OpenAI
import sys
import docx
import PyPDF2
import itertools 
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
from collections import Counter
from difflib import SequenceMatcher
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import load_workbook
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Dict, List, Optional
import platform

# ============================
# 2. Globale Variablen
# ============================

# Definition der Forschungsfrage
FORSCHUNGSFRAGE = "Wie gestaltet sich [Phänomen] im Kontext von [Setting] und welche [Aspekt] lassen sich dabei identifizieren?"

# Allgemeine Kodierregeln
KODIERREGELN = {
    "general": [
        "Kodiere nur manifeste, nicht latente Inhalte",
        "Berücksichtige den Kontext der Aussage",
        "Bei Unsicherheit dokumentiere die Gründe",
        "Kodiere vollständige Sinneinheiten",
        "Prüfe Überschneidungen zwischen Kategorien"
    ],
    "format": [
        "Markiere relevante Textstellen",
        "Dokumentiere Begründung der Zuordnung",
        "Gib Konfidenzwert (1-5) an",
        "Notiere eventuelle Querverbindungen zu anderen Kategorien"
    ]
}

# Deduktive Kategorien mit integrierten spezifischen Kodierregeln
DEDUKTIVE_KATEGORIEN = {
    "Akteure": {
        "definition": "Erfasst alle handelnden Personen, Gruppen oder Institutionen sowie deren Rollen, Beziehungen und Interaktionen",
        "rules": "Codiere Aussagen zu: Individuen, Gruppen, Organisationen, Netzwerken",
        "subcategories": {
            "Individuelle_Akteure": "Einzelpersonen und deren Eigenschaften",
            "Kollektive_Akteure": "Gruppen, Organisationen, Institutionen",
            "Beziehungen": "Interaktionen, Hierarchien, Netzwerke",
            "Rollen": "Formelle und informelle Positionen"
        },
        "examples": {
            "Die Projektleiterin hat die Entscheidung eigenständig getroffen",
            "Die Arbeitsgruppe trifft sich wöchentlich zur Abstimmung"    ,
            "Als Vermittler zwischen den Parteien konnte er den Konflikt lösen",
            "Die beteiligten Organisationen haben eine Kooperationsvereinbarung unterzeichnet"
        }
    },
    "Kontextfaktoren": {
        "definition": "Umfasst die strukturellen, zeitlichen und räumlichen Rahmenbedingungen des untersuchten Phänomens",
        "subcategories": {
            "Strukturell": "Organisatorische und institutionelle Bedingungen",
            "Zeitlich": "Historische Entwicklung, Zeitpunkte, Perioden",
            "Räumlich": "Geografische und sozialräumliche Aspekte",
            "Kulturell": "Normen, Werte, Traditionen"
        }
    },
    "Kontextfaktoren": {
        "definition": "Umfasst die strukturellen, zeitlichen und räumlichen Rahmenbedingungen des untersuchten Phänomens",
        "subcategories": {
            "Strukturell": "Organisatorische und institutionelle Bedingungen",
            "Zeitlich": "Historische Entwicklung, Zeitpunkte, Perioden",
            "Räumlich": "Geografische und sozialräumliche Aspekte",
            "Kulturell": "Normen, Werte, Traditionen"
        }
    },
    "Prozesse": {
        "definition": "Erfasst Abläufe, Entwicklungen und Veränderungen über Zeit",
        "subcategories": {
            "Entscheidungsprozesse": "Formelle und informelle Entscheidungsfindung",
            "Entwicklungsprozesse": "Veränderungen und Transformationen",
            "Interaktionsprozesse": "Kommunikation und Austausch",
            "Konfliktprozesse": "Aushandlungen und Konflikte"
        }
    },
    "Ressourcen": {
        "definition": "Materielle und immaterielle Mittel und Kapazitäten",
        "subcategories": {
            "Materiell": "Finanzielle und physische Ressourcen",
            "Immateriell": "Wissen, Kompetenzen, soziales Kapital",
            "Zugang": "Verfügbarkeit und Verteilung",
            "Nutzung": "Einsatz und Verwertung"
        }
    },
    "Strategien": {
        "definition": "Handlungsmuster und -konzepte zur Zielerreichung",
        "subcategories": {
            "Formell": "Offizielle Strategien und Pläne",
            "Informell": "Ungeschriebene Praktiken",
            "Adaptiv": "Anpassungsstrategien",
            "Innovativ": "Neue Lösungsansätze"
        }
    },
    "Outcomes": {
        "definition": "Ergebnisse, Wirkungen und Folgen von Handlungen und Prozessen",
        "subcategories": {
            "Intendiert": "Beabsichtigte Wirkungen",
            "Nicht_intendiert": "Unbeabsichtigte Folgen",
            "Kurzfristig": "Unmittelbare Effekte",
            "Langfristig": "Nachhaltige Wirkungen"
        }
    },
    "Herausforderungen": {
        "definition": "Probleme, Hindernisse und Spannungsfelder",
        "subcategories": {
            "Strukturell": "Systemische Barrieren",
            "Prozessual": "Ablaufbezogene Schwierigkeiten",
            "Individuell": "Persönliche Herausforderungen",
            "Kontextuell": "Umfeldbezogene Probleme"
        }
    },
    "Legitimation": {
        "definition": "Begründungen, Rechtfertigungen und Deutungsmuster",
        "subcategories": {
            "Normativ": "Wertbasierte Begründungen",
            "Pragmatisch": "Praktische Rechtfertigungen",
            "Kognitiv": "Wissensbasierte Erklärungen",
            "Emotional": "Gefühlsbezogene Deutungen"
        }
    }
}

# ------------------------
# Konfigurationskonstanten
# ------------------------
CONFIG = {
    'MODEL_NAME': 'gpt-4o-mini',
    'DATA_DIR': os.path.join(os.getcwd(), 'transkripte'),
    'OUTPUT_DIR': os.path.join(os.getcwd(), 'output'),
    'CHUNK_SIZE': 800,
    'CHUNK_OVERLAP': 80,
    'ATTRIBUTE_LABELS': {
        'attribut1': 'Hochschulprofil',
        'attribut2': 'Akteur'
    },
    'CODER_SETTINGS': [
        {
            'temperature': 0.3,
            'coder_id': 'auto_1'
        },
        {
            'temperature': 0.5,
            'coder_id': 'auto_2'
        }
    ]
}

# Stelle sicher, dass die Verzeichnisse existieren
os.makedirs(CONFIG['DATA_DIR'], exist_ok=True)
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# Lade Umgebungsvariablen
env_path = os.path.join(os.path.expanduser("~"), '.environ.env')
load_dotenv(env_path)


# ============================
# 3. Klassen und Funktionen
# ============================

@dataclass
class CategoryDefinition:
    """Datenklasse für eine Kategorie im Kodiersystem"""
    name: str
    definition: str
    examples: List[str]
    rules: List[str]  # Explizit als List[str] definiert
    subcategories: Dict[str, str]
    added_date: str
    modified_date: str

@dataclass
class CodingResult:
    """Datenklasse für ein Kodierungsergebnis"""
    category: str
    subcategories: List[str]
    justification: str
    confidence: Dict[str, Union[float, List[str]]]
    text_references: List[str]
    uncertainties: Optional[List[str]] = None

@dataclass
class CategoryChange:
    """Dokumentiert eine Änderung an einer Kategorie"""
    category_name: str
    change_type: str  # 'add', 'modify', 'delete', 'merge', 'split'
    description: str
    timestamp: str
    old_value: Optional[dict] = None
    new_value: Optional[dict] = None
    affected_codings: List[str] = None
    justification: str = ""


# --- Klasse: ConfigLoader ---
class ConfigLoader:
    def __init__(self, script_dir):
        self.script_dir = script_dir
        self.excel_path = os.path.join(script_dir, "QCA-AID-Codebook.xlsx")
        self.config = {
            'FORSCHUNGSFRAGE': "",
            'KODIERREGELN': {},
            'DEDUKTIVE_KATEGORIEN': {},
            'CONFIG': {}
        }


    def load_config(self):
        print(f"Versuche Konfiguration zu laden von: {self.excel_path}")
        if not os.path.exists(self.excel_path):
            print(f"Excel-Datei nicht gefunden: {self.excel_path}")
            return False

        try:
            wb = load_workbook(self.excel_path, read_only=True, data_only=True)
            print(f"Excel-Datei erfolgreich geladen. Verfügbare Sheets: {wb.sheetnames}")
            
            self._load_research_question(wb)
            self._load_coding_rules(wb)
            self._load_deduktive_kategorien(wb)
            self._load_config(wb)

            return True
        except Exception as e:
            print(f"Fehler beim Lesen der Excel-Datei: {str(e)}")
            return False

    def _load_research_question(self, wb):
        if 'FORSCHUNGSFRAGE' in wb.sheetnames:
            sheet = wb['FORSCHUNGSFRAGE']
            value = sheet['B1'].value
            # print(f"Geladene Forschungsfrage: {value}")  # Debug-Ausgabe
            self.config['FORSCHUNGSFRAGE'] = value


    def _load_coding_rules(self, wb):
        if 'KODIERREGELN' in wb.sheetnames:
            df = pd.read_excel(self.excel_path, sheet_name='KODIERREGELN', header=0)
            # print(f"Geladene Kodierregeln: {df}")  # Debug-Ausgabe
            self.config['KODIERREGELN'] = {
                'general': df['Allgemeine Kodierregeln'].dropna().tolist(),
                'format': df['Formatregeln'].dropna().tolist()
            }

    def _load_deduktive_kategorien(self, wb):
        if 'DEDUKTIVE_KATEGORIEN' in wb.sheetnames:
            df = pd.read_excel(self.excel_path, sheet_name='DEDUKTIVE_KATEGORIEN')
            # print(f"Geladene deduktive Kategorien:\n{df.head()}")  # Debug-Ausgabe
            self.config['DEDUKTIVE_KATEGORIEN'] = {}
            
            current_category = None
            for _, row in df.iterrows():
                key = row['Key']
                sub_key = row['Sub-Key']
                sub_sub_key = row['Sub-Sub-Key']
                value = row['Value']

                if pd.notna(key):  # Neue Hauptkategorie
                    current_category = key
                    self.config['DEDUKTIVE_KATEGORIEN'][current_category] = {
                        'definition': '',
                        'rules': [],
                        'examples': [],
                        'subcategories': {}
                    }

                if current_category is not None:
                    if sub_key == 'definition':
                        self.config['DEDUKTIVE_KATEGORIEN'][current_category]['definition'] = value
                    elif sub_key == 'rules':
                        self.config['DEDUKTIVE_KATEGORIEN'][current_category]['rules'].append(value)
                    elif sub_key == 'examples':
                        self.config['DEDUKTIVE_KATEGORIEN'][current_category]['examples'].append(value)
                    elif sub_key == 'subcategories' and pd.notna(sub_sub_key):
                        self.config['DEDUKTIVE_KATEGORIEN'][current_category]['subcategories'][sub_sub_key] = value

    def _load_config(self, wb):
        if 'CONFIG' in wb.sheetnames:
            df = pd.read_excel(self.excel_path, sheet_name='CONFIG')
            config = {}
            
            for _, row in df.iterrows():
                key = row['Key']
                sub_key = row['Sub-Key']
                sub_sub_key = row['Sub-Sub-Key']
                value = row['Value']

                if key not in config:
                    config[key] = value if pd.isna(sub_key) else {}

                if not pd.isna(sub_key):
                    if sub_key.startswith('['):  # Für Listen wie CODER_SETTINGS
                        if not isinstance(config[key], list):
                            config[key] = []
                        index = int(sub_key.strip('[]'))
                        while len(config[key]) <= index:
                            config[key].append({})
                        if pd.isna(sub_sub_key):
                            config[key][index] = value
                        else:
                            config[key][index][sub_sub_key] = value
                    else:  # Für verschachtelte Dicts wie ATTRIBUTE_LABELS
                        if not isinstance(config[key], dict):
                            config[key] = {}
                        if pd.isna(sub_sub_key):
                            config[key][sub_key] = value
                        else:
                            if sub_key not in config[key]:
                                config[key][sub_key] = {}
                            config[key][sub_key][sub_sub_key] = value

            self.config['CONFIG'] = self._sanitize_config(config)

    def _sanitize_config(self, config):
        """
        Bereinigt und validiert die Konfigurationswerte.
        Überschreibt Standardwerte mit Werten aus dem Codebook.
        
        Args:
            config: Dictionary mit rohen Konfigurationswerten
            
        Returns:
            dict: Bereinigtes Konfigurations-Dictionary
        """
        try:
            sanitized = {}
            for key, value in config.items():
                # Verzeichnispfade relativ zum aktuellen Arbeitsverzeichnis
                if key in ['DATA_DIR', 'OUTPUT_DIR']:
                    sanitized[key] = os.path.join(os.getcwd(), str(value))
                
                # Numerische Werte für Chunking
                elif key in ['CHUNK_SIZE', 'CHUNK_OVERLAP']:
                    try:
                        # Konvertiere zu Integer und stelle sicher, dass die Werte positiv sind
                        sanitized[key] = max(1, int(value))
                        print(f"Übernehme {key} aus Codebook: {sanitized[key]}")
                    except (ValueError, TypeError):
                        # Wenn Konvertierung fehlschlägt, behalte Standardwert
                        default_value = CONFIG[key]
                        print(f"Warnung: Ungültiger Wert für {key}, verwende Standard: {default_value}")
                        sanitized[key] = default_value
                
                # Coder-Einstellungen mit Typkonvertierung
                elif key == 'CODER_SETTINGS':
                    sanitized[key] = [
                        {
                            'temperature': float(coder['temperature']) 
                                if isinstance(coder.get('temperature'), (int, float, str)) 
                                else 0.7,
                            'coder_id': str(coder.get('coder_id', f'auto_{i}'))
                        }
                        for i, coder in enumerate(value)
                    ]
                
                # Alle anderen Werte unverändert übernehmen
                else:
                    sanitized[key] = value
                    
            # Stelle sicher, dass CHUNK_OVERLAP kleiner als CHUNK_SIZE ist
            if 'CHUNK_SIZE' in sanitized and 'CHUNK_OVERLAP' in sanitized:
                if sanitized['CHUNK_OVERLAP'] >= sanitized['CHUNK_SIZE']:
                    print(f"Warnung: CHUNK_OVERLAP ({sanitized['CHUNK_OVERLAP']}) muss kleiner sein als CHUNK_SIZE ({sanitized['CHUNK_SIZE']})")
                    sanitized['CHUNK_OVERLAP'] = sanitized['CHUNK_SIZE'] // 10  # 10% als Standardwert
                    
            return sanitized
            
        except Exception as e:
            print(f"Fehler bei der Konfigurationsbereinigung: {str(e)}")
            import traceback
            traceback.print_exc()
            # Verwende im Fehlerfall die Standard-Konfiguration
            return CONFIG

    def get_config(self):
        return self.config

    def update_script_globals(self, globals_dict):
        for key, value in self.config.items():
            if key in globals_dict:
                if isinstance(globals_dict[key], dict) and isinstance(value, dict):
                    globals_dict[key].clear()  # Löschen Sie zuerst den bestehenden Inhalt
                    globals_dict[key].update(value)
                else:
                    globals_dict[key] = value
        
        if 'DEDUKTIVE_KATEGORIEN' in self.config:
            global DEDUKTIVE_KATEGORIEN
            DEDUKTIVE_KATEGORIEN = self.config['DEDUKTIVE_KATEGORIEN']
        
        print("DEDUKTIVE_KATEGORIEN nach der Aktualisierung:")
        print(DEDUKTIVE_KATEGORIEN)

# --- Klasse: MaterialLoader ---
# Aufgabe: Laden und Vorbereiten des Analysematerials (Textdokumente, Transkripte)
# WICHTIG: Lange Texte werden mittels Chunking in überschaubare Segmente zerlegt.
class MaterialLoader:
    """Lädt und verarbeitet Interviewdokumente."""
    
    def __init__(self, data_dir: str = CONFIG['DATA_DIR'], 
                 chunk_size: int = CONFIG['CHUNK_SIZE'], 
                 chunk_overlap: int = CONFIG['CHUNK_OVERLAP']):
        """
        Initialisiert den MaterialLoader.
        
        Args:
            data_dir (str): Verzeichnis mit den Dokumenten
            chunk_size (int): Ungefähre Anzahl der Zeichen pro Chunk
            chunk_overlap (int): Überlappung zwischen Chunks in Zeichen
        """
        self.data_dir: str = data_dir
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        
        # Lade das deutsche Sprachmodell für spaCy
        try:
            import spacy
            self.nlp = spacy.load("de_core_news_sm")
        except Exception as e:
            print("Bitte installieren Sie das deutsche Sprachmodell:")
            print("python -m spacy download de_core_news_sm")
            raise e

    def chunk_text(self, text: str) -> List[str]:
        """
        Teilt Text in überlappende Chunks basierend auf Satzgrenzen.
        
        Args:
            text (str): Zu teilender Text
            
        Returns:
            List[str]: Liste der Text-Chunks
        """
        # Debug-Ausgabe
        print(f"\nChunking-Parameter:")
        print(f"- Chunk Size: {self.chunk_size}")
        print(f"- Chunk Overlap: {self.chunk_overlap}")
        print(f"- Gesamtlänge Text: {len(text)} Zeichen")
        
        # Verarbeite den Text mit spaCy
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_text = sentence.text.strip()
            sentence_length = len(sentence_text)
            
            # Wenn der aktuelle Chunk mit diesem Satz zu groß würde
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Speichere aktuellen Chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                print(f"- Neuer Chunk erstellt: {len(chunk_text)} Zeichen")
                
                # Starte neuen Chunk mit Überlappung
                # Berechne wie viele Sätze wir für die Überlappung behalten
                overlap_length = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    overlap_length += len(sent)
                    if overlap_length > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, sent)
                
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
            
            # Füge den Satz zum aktuellen Chunk hinzu
            current_chunk.append(sentence_text)
            current_length += sentence_length
        
        # Letzten Chunk hinzufügen, falls vorhanden
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            print(f"- Letzter Chunk: {len(chunk_text)} Zeichen")
        
        print(f"\nChunking Ergebnis:")
        print(f"- Anzahl Chunks: {len(chunks)}")
        print(f"- Durchschnittliche Chunk-Länge: {sum(len(c) for c in chunks)/len(chunks):.0f} Zeichen")
        
        return chunks

    def preprocess_text(self, text: str) -> str:
        """
        Bereinigt und normalisiert Textdaten.
        
        Args:
            text (str): Zu bereinigender Text
            
        Returns:
            str: Bereinigter Text
        """
        if not text:
            return ""
            
        # Entferne überflüssige Whitespaces
        text = ' '.join(text.split())
        
        # Ersetze verschiedene Anführungszeichen durch einheitliche
        text = text.replace('"', '"').replace('"', '"')
        
        # Ersetze verschiedene Bindestriche durch einheitliche
        text = text.replace('–', '-').replace('—', '-')
        
        # Entferne spezielle Steuerzeichen
        text = ''.join(char for char in text if ord(char) >= 32)
        
        return text

    def load_documents(self) -> dict:
        """
        Sammelt und lädt alle unterstützten Dateien aus dem Verzeichnis.
        
        Returns:
            dict: Dictionary mit Dateinamen als Schlüssel und Dokumenteninhalt als Wert
        """
        documents = {}
        supported_extensions = {'.txt', '.docx', '.doc', '.pdf'}
        
        try:
            # Prüfe ob Verzeichnis existiert
            if not os.path.exists(self.data_dir):
                print(f"Warnung: Verzeichnis {self.data_dir} existiert nicht")
                os.makedirs(self.data_dir)
                return documents

            # Durchsuche Verzeichnis nach Dateien
            for file in os.listdir(self.data_dir):
                file_path = os.path.join(self.data_dir, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if not os.path.isfile(file_path):
                    continue
                    
                print(f"Gefunden: {file} ({file_ext})")
                
                try:
                    if file_ext == '.txt':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    elif file_ext == '.docx':
                        from docx import Document
                        doc = Document(file_path)
                        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                    elif file_ext == '.doc':
                        # Benötigt antiword oder ähnliches Tool
                        import subprocess
                        text = subprocess.check_output(['antiword', file_path]).decode('utf-8')
                    elif file_ext == '.pdf':
                        import PyPDF2
                        with open(file_path, 'rb') as f:
                            pdf = PyPDF2.PdfReader(f)
                            text = '\n'.join([page.extract_text() for page in pdf.pages])
                    else:
                        print(f"Überspringe nicht unterstützte Datei: {file}")
                        continue
                        
                    # Bereinige und speichere Text
                    text = self.preprocess_text(text)
                    if text.strip():  # Nur nicht-leere Texte speichern
                        documents[file] = text
                        print(f"Erfolgreich geladen: {file} ({len(text)} Zeichen)")
                    else:
                        print(f"Warnung: Leerer Text in {file}")
                        
                except Exception as e:
                    print(f"Fehler beim Laden von {file}: {str(e)}")
                    continue
                    
            if not documents:
                print(f"\nKeine Dokumente gefunden in {self.data_dir}")
                print(f"Unterstützte Formate: {', '.join(supported_extensions)}")
            
        except Exception as e:
            print(f"Fehler beim Durchsuchen des Verzeichnisses: {str(e)}")
            
        return documents

# --- Klasse: CategoryManager ---
class CategoryManager:
    """Verwaltet das Speichern und Laden von Kategorien"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.categories_file = os.path.join(output_dir, 'category_system.json')
        
    def save_categories(self, categories: Dict[str, CategoryDefinition]) -> None:
        """Speichert das aktuelle Kategoriensystem als JSON"""
        try:
            # Konvertiere CategoryDefinition Objekte in serialisierbares Format
            categories_dict = {
                name: {
                    'definition': cat.definition,
                    'examples': cat.examples,
                    'rules': cat.rules,
                    'subcategories': cat.subcategories,
                    'timestamp': datetime.now().isoformat()
                } for name, cat in categories.items()
            }
            
            # Metadata hinzufügen
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'categories': categories_dict
            }
            
            with open(self.categories_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            print(f"\nKategoriensystem gespeichert in: {self.categories_file}")
            
        except Exception as e:
            print(f"Fehler beim Speichern des Kategoriensystems: {str(e)}")

    def load_categories(self) -> Optional[Dict[str, CategoryDefinition]]:
        """
        Lädt gespeichertes Kategoriensystem falls vorhanden.
        
        Returns:
            Optional[Dict[str, CategoryDefinition]]: Dictionary mit Kategorienamen und deren Definitionen,
            oder None wenn kein gespeichertes System verwendet werden soll
        """
        try:
            if os.path.exists(self.categories_file):
                with open(self.categories_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Zeige Informationen zum gefundenen Kategoriensystem
                timestamp = datetime.fromisoformat(data['timestamp'])
                print(f"\nGefundenes Kategoriensystem vom {timestamp.strftime('%d.%m.%Y %H:%M')}")
                print(f"Enthält {len(data['categories'])} Kategorien:")
                for cat_name in data['categories'].keys():
                    print(f"- {cat_name}")
                
                # Frage Benutzer
                while True:
                    answer = input("\nMöchten Sie dieses Kategoriensystem verwenden? (j/n): ").lower()
                    if answer in ['j', 'n']:
                        break
                    print("Bitte antworten Sie mit 'j' oder 'n'")
                
                if answer == 'j':
                    # Konvertiere zurück zu CategoryDefinition Objekten
                    categories = {}
                    for name, cat_data in data['categories'].items():
                        # Stelle sicher, dass die Zeitstempel existieren
                        if 'timestamp' in cat_data:
                            added_date = modified_date = cat_data['timestamp'].split('T')[0]
                        else:
                            # Verwende aktuelle Zeit für fehlende Zeitstempel
                            current_date = datetime.now().strftime("%Y-%m-%d")
                            added_date = modified_date = current_date

                        categories[name] = CategoryDefinition(
                            name=name,
                            definition=cat_data['definition'],
                            examples=cat_data.get('examples', []),
                            rules=cat_data.get('rules', []),
                            subcategories=cat_data.get('subcategories', {}),
                            added_date=cat_data.get('added_date', added_date),
                            modified_date=cat_data.get('modified_date', modified_date)
                        )
                    
                    print(f"\nKategoriensystem erfolgreich geladen.")
                    return categories
                
            return None
                
        except Exception as e:
            print(f"Fehler beim Laden des Kategoriensystems: {str(e)}")
            print("Folgende Details könnten hilfreich sein:")
            import traceback
            traceback.print_exc()
            return None

# Klasse: CategoryMerger
# ----------------------
class CategoryMerger:
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.merge_log = []

    def _calculate_semantic_similarity(self, name1: str, name2: str) -> float:
        """Berechnet die semantische Ähnlichkeit zwischen zwei Kategorienamen"""
        norm1 = self._normalize_category_name(name1)
        norm2 = self._normalize_category_name(name2)
        
        # Basis-Ähnlichkeit durch Fuzzy String Matching
        base_similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Zusätzliche Gewichtung für gemeinsame Wortstämme
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        common_words = words1.intersection(words2)
        word_similarity = len(common_words) / max(len(words1), len(words2))
        
        return (base_similarity + word_similarity) / 2

    def _normalize_category_name(self, name: str) -> str:
        """Normalisiert Kategorienamen für besseren Vergleich"""
        # Zu Kleinbuchstaben und entferne Sonderzeichen
        name = re.sub(r'[^\w\s]', '', name.lower())
        # Entferne Stoppwörter
        stop_words = {'und', 'oder', 'der', 'die', 'das', 'in', 'im', 'für', 'bei'}
        return ' '.join(word for word in name.split() if word not in stop_words)

    def suggest_merges(self, categories: Dict[str, CategoryDefinition]) -> List[Tuple[str, str, float]]:
        """Identifiziert ähnliche Kategorien basierend auf semantischer Ähnlichkeit."""
        similar_pairs = []
        category_names = list(categories.keys())
        
        for i in range(len(category_names)):
            for j in range(i + 1, len(category_names)):
                name1, name2 = category_names[i], category_names[j]
                similarity = self._calculate_semantic_similarity(name1, name2)
                
                if similarity >= self.similarity_threshold:
                    similar_pairs.append((name1, name2, similarity))
        
        return similar_pairs

    def merge_categories(self, categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """Führt ähnliche Kategorien zusammen und dokumentiert Änderungen."""
        similar_pairs = self.suggest_merges(categories)
        merged_categories = categories.copy()
        
        for cat1, cat2, sim in similar_pairs:
            # Implementiere hier die Logik zum Zusammenführen der Kategorien
            merged_name = f"{cat1}_{cat2}"
            merged_def = self._merge_definitions(categories[cat1].definition, categories[cat2].definition)
            merged_examples = list(set(categories[cat1].examples + categories[cat2].examples))
            merged_subcats = {**categories[cat1].subcategories, **categories[cat2].subcategories}
            
            merged_categories[merged_name] = CategoryDefinition(
                name=merged_name,
                definition=merged_def,
                examples=merged_examples,
                rules=list(set(categories[cat1].rules + categories[cat2].rules)),
                subcategories=merged_subcats,
                added_date=datetime.now().strftime("%Y-%m-%d"),
                modified_date=datetime.now().strftime("%Y-%m-%d")
            )
            
            del merged_categories[cat1]
            del merged_categories[cat2]
            
            self.merge_log.append({
                'merged': [cat1, cat2],
                'new_category': merged_name,
                'similarity': sim,
                'timestamp': datetime.now().isoformat()
            })
        
        return merged_categories

    def _merge_definitions(self, def1: str, def2: str) -> str:
        """Kombiniert zwei Kategoriendefinitionen"""
        # Einfache Implementierung - kann verfeinert werden
        return f"{def1}\n\nZusätzlich: {def2}"

    def save_merge_log(self, output_dir: str):
        """Speichert das Zusammenführungsprotokoll."""
        if self.merge_log:
            merge_log_path = os.path.join(output_dir, 
                f'category_merges_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(merge_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.merge_log, f, ensure_ascii=False, indent=2)
            print(f"Zusammenführungsprotokoll gespeichert unter: {merge_log_path}")


# --- Klasse: Analyzer ---
# Aufgabe: Festlegung der Analyseeinheiten (Auswertungs-, Kodiereinheit, Kontexteinheit)
class Analyzer:
    """
    Legt fest, welche Einheiten in der Analyse von Interviewdaten verwendet werden.
    Implementiert die Festlegung der Analyseeinheiten nach Mayring für die deduktive Kategorienanwendung.
    """
    
    def __init__(self, chunked_docs: dict = None):
        """
        Initialisiert den Analyzer mit den zu analysierenden Dokumenten.
        
        Args:
            chunked_docs (dict): Dictionary mit den in Chunks zerlegten Dokumenten
        """
        self.chunked_docs = chunked_docs or {}
        self.coded_data = {}  # Speichert die kodierten Daten
        self.current_position = {
            'doc_name': None,
            'chunk_index': 0
        }
        
        # Definition der Analyseeinheiten
        self.units = {
            'coding_unit': {
                'type': 'statement',  # Einzelne Aussage als kleinste Einheit
                'description': 'Inhaltlich zusammenhängende Aussage zu einem Thema'
            },
            'context_unit': {
                'type': 'chunk',  # Textabschnitt als Kontexteinheit
                'description': 'Umgebender Textabschnitt zur Interpretation der Aussage'
            },
            'analysis_unit': {
                'type': 'interview',  # Interview als Auswertungseinheit
                'description': 'Vollständiges Interview als Basis der Analyse'
            }
        }
        
    def set_current_document(self, doc_name: str) -> bool:
        """
        Setzt das aktuelle Dokument für die Analyse.
        
        Args:
            doc_name (str): Name des zu analysierenden Dokuments
            
        Returns:
            bool: True wenn erfolgreich, False wenn Dokument nicht gefunden
        """
        if doc_name in self.chunked_docs:
            self.current_position['doc_name'] = doc_name
            self.current_position['chunk_index'] = 0
            return True
        return False
        
    def get_current_chunk(self) -> tuple:
        """
        Gibt den aktuellen Chunk und seinen Kontext zurück.
        
        Returns:
            tuple: (chunk_text, context_info) oder (None, None) wenn kein Chunk verfügbar
        """
        doc_name = self.current_position['doc_name']
        chunk_index = self.current_position['chunk_index']
        
        if not doc_name or doc_name not in self.chunked_docs:
            return None, None
            
        chunks = self.chunked_docs[doc_name]
        if chunk_index >= len(chunks):
            return None, None
            
        chunk_text = chunks[chunk_index]
        
        # Erstelle Kontextinformationen
        context_info = {
            'document': doc_name,
            'chunk_index': chunk_index,
            'total_chunks': len(chunks),
            'previous_chunk': chunks[chunk_index - 1] if chunk_index > 0 else None,
            'next_chunk': chunks[chunk_index + 1] if chunk_index < len(chunks) - 1 else None
        }
        
        return chunk_text, context_info
        
    def next_chunk(self) -> bool:
        """
        Geht zum nächsten Chunk im aktuellen Dokument.
        
        Returns:
            bool: True wenn erfolgreich, False wenn kein weiterer Chunk verfügbar
        """
        doc_name = self.current_position['doc_name']
        if not doc_name or doc_name not in self.chunked_docs:
            return False
            
        next_index = self.current_position['chunk_index'] + 1
        if next_index < len(self.chunked_docs[doc_name]):
            self.current_position['chunk_index'] = next_index
            return True
        return False
        
    def add_coding(self, chunk_id: str, coding_data: dict) -> bool:
        """
        Fügt eine Kodierung für einen Chunk hinzu.
        
        Args:
            chunk_id (str): ID des Chunks
            coding_data (dict): Kodierungsdaten (Kategorie, Subkategorien, etc.)
            
        Returns:
            bool: True wenn erfolgreich, False bei Fehler
        """
        try:
            self.coded_data[chunk_id] = coding_data
            return True
        except Exception as e:
            print(f"Fehler beim Hinzufügen der Kodierung: {str(e)}")
            return False
            
    def get_coded_data(self) -> dict:
        """
        Gibt alle kodierten Daten zurück.
        
        Returns:
            dict: Dictionary mit allen Kodierungen
        """
        return self.coded_data
        
    def get_document_codings(self, doc_name: str) -> dict:
        """
        Gibt alle Kodierungen für ein bestimmtes Dokument zurück.
        
        Args:
            doc_name (str): Name des Dokuments
            
        Returns:
            dict: Kodierungen für das angegebene Dokument
        """
        return {
            chunk_id: coding 
            for chunk_id, coding in self.coded_data.items() 
            if chunk_id.startswith(doc_name)
        }

# --- Klasse: IntegratedAnalysisManager ---
# Aufgabe: Integriert die verschiedenen Analysephasen in einem zusammenhängenden Prozess
class IntegratedAnalysisManager:
    def __init__(self, config: Dict):
        self.inductive_coder = InductiveCoder(config['MODEL_NAME'])
        self.deductive_coders = [
            DeductiveCoder(config['MODEL_NAME'], coder_config['temperature'], coder_config['coder_id'])
            for coder_config in config['CODER_SETTINGS']
        ]
        self.category_merger = CategoryMerger(similarity_threshold=0.8)
        self.processed_segments = set()
        self.coding_results = []
        
        # Attribute für Sättigungsprüfung
        self._iteration_count = 1
        self._stable_iterations = 0
        self.MAX_ITERATIONS = 10
        self.MIN_MATERIAL_PERCENTAGE = 70
        self.STABILITY_THRESHOLD = 3
        
        # Performance Tracking
        self.start_time = datetime.now()
        self.performance_metrics = {
            'total_segments': 0,
            'processed_segments': 0,
            'new_categories': 0,
            'modified_categories': 0
        }

        print("\nAnalyse-Manager initialisiert:")
        print(f"- Max Iterationen: {self.MAX_ITERATIONS}")
        print(f"- Min Material %: {self.MIN_MATERIAL_PERCENTAGE}%")
        print(f"- Stabilitätsschwelle: {self.STABILITY_THRESHOLD} Durchläufe")
        
    async def analyze_material(self, chunks: Dict[str, List[str]], initial_categories: Dict):
        """Integrierter Analyseprozess mit Sättigungsprüfung"""
        current_categories = initial_categories.copy()
        
        # Berechne Batch-Größe mit Minimum von 1
        total_chunks = len(chunks)
        batch_size = max(1, total_chunks // 5)  # Mindestens 1 Chunk pro Batch
        
        print(f"\nAnalyse-Setup:")
        print(f"- Gesamt Chunks: {total_chunks}")
        print(f"- Batch-Größe: {batch_size}")
        
        # Segmente in Batches organisieren
        all_segments = []
        for doc_name, doc_chunks in chunks.items():
            for chunk_idx, chunk in enumerate(doc_chunks):
                all_segments.append((f"{doc_name}_chunk_{chunk_idx}", chunk))
        
        total_segments = len(all_segments)
        print(f"- Gesamt Segmente: {total_segments}")
        
        # Verarbeite Batches
        for batch_start in range(0, total_segments, batch_size):
            batch = all_segments[batch_start:batch_start + batch_size]
            print(f"\nVerarbeite Batch {batch_start//batch_size + 1} "
                f"(Segmente {batch_start+1} bis {min(batch_start+batch_size, total_segments)})")
            
            # 1. Induktive Analyse des Batches
            new_categories = await self.inductive_coder.develop_category_system(
                [chunk for _, chunk in batch]
            )
            
            # 2. Kategorien zusammenführen
            merged_categories = self.category_merger.merge_categories({
                **current_categories,
                **new_categories
            })
            
            # 3. Kodierung des Batches mit allen Kodierern
            batch_results = []
            for segment_id, chunk in batch:
                chunk_results = []
                for coder in self.deductive_coders:
                    result = await coder.code_chunk(chunk, merged_categories)
                    if result:
                        # Füge segment_id zum Ergebnis hinzu
                        result_dict = {
                            'segment_id': segment_id,
                            'coder_id': coder.coder_id,
                            'category': result.category,
                            'subcategories': result.subcategories,
                            'confidence': result.confidence,
                            'justification': result.justification
                        }
                        chunk_results.append(result_dict)
                batch_results.extend(chunk_results)
            
            # 4. Sättigungsprüfung
            category_changes = self._compare_category_systems(
                current_categories, 
                merged_categories
            )
            
            current_categories = merged_categories
            self.coding_results.extend(batch_results)
            
            material_percentage = (batch_start + len(batch)) / total_segments * 100
            if self._check_saturation(category_changes, material_percentage):
                print(f"\nSättigung erreicht bei {material_percentage:.1f}% des Materials")
                break
        
        return current_categories, self.coding_results

    def _compare_category_systems(self, old_cats: Dict, new_cats: Dict) -> List[Dict]:
        """Identifiziert Änderungen zwischen zwei Kategoriensystemen"""
        changes = []
        # Implementation...
        return changes

    def _check_saturation(self, changes: List[Dict], material_percentage: float) -> bool:
        """
        Verbesserte Sättigungsprüfung, die verschiedene Qualitätskriterien berücksichtigt.
        
        Args:
            changes: Liste der Kategorienänderungen im aktuellen Batch
            material_percentage: Prozentsatz des analysierten Materials
            
        Returns:
            bool: True wenn Sättigung erreicht ist
        """
        try:
            # 1. Grundlegende Abbruchkriterien prüfen
            if self._iteration_count >= self.MAX_ITERATIONS:
                print(f"\nSättigung erreicht: Maximale Anzahl von {self.MAX_ITERATIONS} Durchläufen erreicht")
                return True

            # 2. Minimale Materialmenge prüfen
            if material_percentage < self.MIN_MATERIAL_PERCENTAGE:
                print(f"\nFortsetzung: Erst {material_percentage:.1f}% des Materials analysiert "
                        f"(Minimum: {self.MIN_MATERIAL_PERCENTAGE}%)")
                return False

            # 3. Analyse der Änderungen
            change_metrics = self._analyze_changes(changes)
            
            # 4. Qualitative Sättigungsprüfung
            saturation_indicators = {
                'new_categories': change_metrics['new_categories'] == 0,
                'significant_modifications': change_metrics['significant_modifications'] == 0,
                'subcategory_changes': change_metrics['subcategory_changes'] <= 1,
                'definition_refinements': change_metrics['definition_refinements'] <= 2
            }
            
            # 5. Stabilität prüfen
            current_stable = all(saturation_indicators.values())
            if current_stable:
                self._stable_iterations += 1
                print(f"\nStabiler Durchlauf {self._stable_iterations}/{self.STABILITY_THRESHOLD}")
                
                if self._stable_iterations >= self.STABILITY_THRESHOLD:
                    print(f"\nSättigung erreicht nach {self._stable_iterations} stabilen Durchläufen")
                    self._document_saturation_metrics(change_metrics, material_percentage)
                    return True
            else:
                self._stable_iterations = 0
                print("\nNeue Änderungen erkannt - Stabilitätszähler zurückgesetzt")

            # 6. Detaillierte Statusausgabe
            print(f"\nSättigungsanalyse Durchlauf {self._iteration_count}:")
            print(f"- Material analysiert: {material_percentage:.1f}%")
            print(f"- Stabile Durchläufe: {self._stable_iterations}/{self.STABILITY_THRESHOLD}")
            print("- Änderungsmetriken:")
            for metric, value in change_metrics.items():
                print(f"  • {metric}: {value}")
            
            self._iteration_count += 1
            return False

        except Exception as e:
            print(f"Fehler in der Sättigungsprüfung: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _analyze_changes(self, changes: List[Dict]) -> Dict[str, int]:
        """
        Analysiert die Änderungen im Detail.
        
        Args:
            changes: Liste der Kategorienänderungen
            
        Returns:
            Dict mit Änderungsmetriken
        """
        metrics = {
            'new_categories': 0,
            'significant_modifications': 0,
            'subcategory_changes': 0,
            'definition_refinements': 0
        }
        
        for change in changes:
            change_type = change.get('change_type')
            if change_type == 'add':
                metrics['new_categories'] += 1
            elif change_type == 'modify':
                old_val = change.get('old_value', {})
                new_val = change.get('new_value', {})
                
                # Prüfe Definition
                if old_val.get('definition') != new_val.get('definition'):
                    metrics['definition_refinements'] += 1
                
                # Prüfe Subkategorien
                old_subs = set(old_val.get('subcategories', {}).keys())
                new_subs = set(new_val.get('subcategories', {}).keys())
                if old_subs != new_subs:
                    metrics['subcategory_changes'] += 1
                
                # Prüfe auf signifikante Änderungen
                if self._is_significant_modification(old_val, new_val):
                    metrics['significant_modifications'] += 1
        
        return metrics

    def _is_significant_modification(self, old_val: Dict, new_val: Dict) -> bool:
        """
        Bestimmt ob eine Änderung als signifikant einzustufen ist.
        """
        if not old_val or not new_val:
            return True
            
        # Definition hat sich substanziell geändert
        if old_val.get('definition') != new_val.get('definition'):
            old_words = set(old_val.get('definition', '').split())
            new_words = set(new_val.get('definition', '').split())
            changes = len(old_words.symmetric_difference(new_words))
            if changes > len(old_words) * 0.3:  # Mehr als 30% Änderung
                return True
                
        # Subkategorienstruktur hat sich wesentlich verändert
        old_subs = set(old_val.get('subcategories', {}).keys())
        new_subs = set(new_val.get('subcategories', {}).keys())
        if len(old_subs.symmetric_difference(new_subs)) > 2:
            return True
            
        return False

    def _document_saturation_metrics(self, metrics: Dict[str, int], material_percentage: float):
        """
        Dokumentiert die finalen Sättigungsmetriken.
        """
        print("\n=== Finale Sättigungsanalyse ===")
        print(f"Material analysiert: {material_percentage:.1f}%")
        print(f"Durchläufe: {self._iteration_count}")
        print(f"Stabile Durchläufe: {self._stable_iterations}")
        print("\nÄnderungsmetriken im letzten Durchlauf:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value}")
        

# --- Klasse: DeductiveCategoryBuilder ---
# Aufgabe: Ableiten deduktiver Kategorien basierend auf theoretischem Vorwissen
class DeductiveCategoryBuilder:
    """
    Baut ein initiales, theoriebasiertes Kategoriensystem auf.
    """
    def load_theoretical_categories(self) -> Dict[str, CategoryDefinition]:
        """
        Lädt die vordefinierten deduktiven Kategorien.
        
        Returns:
            Dict[str, CategoryDefinition]: Dictionary mit Kategorienamen und deren Definitionen
        """
        categories = {}
        today = datetime.now().strftime("%Y-%m-%d")

        print("DEDUKTIVE_KATEGORIEN in load_theoretical_categories:")
        print(DEDUKTIVE_KATEGORIEN)
        
        try:
            # Konvertiere DEDUKTIVE_KATEGORIEN in CategoryDefinition-Objekte
            for name, data in DEDUKTIVE_KATEGORIEN.items():
                # Stelle sicher, dass rules als Liste vorhanden ist
                rules = data.get("rules", [])
                if not isinstance(rules, list):
                    rules = [str(rules)] if rules else []
                
                categories[name] = CategoryDefinition(
                    name=name,
                    definition=data.get("definition", ""),
                    examples=data.get("examples", []),
                    rules=rules,  # Übergebe die validierte rules Liste
                    subcategories=data.get("subcategories", {}),
                    added_date=today,
                    modified_date=today
                )
                
            return categories
            
        except Exception as e:
            print(f"Fehler beim Laden der Kategorien: {str(e)}")
            print("Aktuelle Kategorie:", name)
            print("Kategorie-Daten:", json.dumps(data, indent=2, ensure_ascii=False))
            raise

# --- Klasse: CodingGuide ---
# Aufgabe: Erstellung und Dokumentation des Kodierleitfadens
class CodingGuide:
    """
    Erstellt und verwaltet einen Kodierleitfaden für die deduktive Kategorienanwendung.
    Der Leitfaden enthält Hauptkategorien mit Definitionen, Ankerbeispielen und Kodierregeln.
    """
    
    def __init__(self):
        """
        Initialisiert den CodingGuide mit einer leeren Kategorienstruktur.
        """
        self.categories = {}
        self.last_modified = None
        self.version = "1.0"
        
    def add_category(self, name: str, definition: str, examples: list = None, 
                    rules: list = None, subcategories: dict = None) -> bool:
        """
        Fügt eine neue Kategorie zum Kodierleitfaden hinzu.
        
        Args:
            name (str): Name der Kategorie
            definition (str): Definition der Kategorie
            examples (list): Liste von Ankerbeispielen
            rules (list): Liste von Kodierregeln
            subcategories (dict): Dictionary mit Subkategorien
            
        Returns:
            bool: True wenn erfolgreich, False bei Fehler
        """
        try:
            self.categories[name] = {
                'definition': definition,
                'examples': examples or [],
                'subcategories': subcategories or {},
                'added_date': datetime.now().strftime("%Y-%m-%d"),
                'modified_date': datetime.now().strftime("%Y-%m-%d")
            }
            self.last_modified = datetime.now()
            return True
        except Exception as e:
            print(f"Fehler beim Hinzufügen der Kategorie: {str(e)}")
            return False
            
    def update_category(self, name: str, updates: dict) -> bool:
        """
        Aktualisiert eine bestehende Kategorie.
        
        Args:
            name (str): Name der Kategorie
            updates (dict): Zu aktualisierende Felder und Werte
            
        Returns:
            bool: True wenn erfolgreich, False bei Fehler
        """
        if name not in self.categories:
            return False
            
        try:
            for key, value in updates.items():
                if key in self.categories[name]:
                    self.categories[name][key] = value
            
            self.categories[name]['modified_date'] = datetime.now().strftime("%Y-%m-%d")
            self.last_modified = datetime.now()
            return True
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Kategorie: {str(e)}")
            return False
            
    def generate_coding_guide(self, output_path: str = None, format: str = 'markdown') -> str:
        """
        Generiert den Kodierleitfaden im gewünschten Format.
        
        Args:
            output_path (str): Pfad für die Ausgabedatei (optional)
            format (str): Ausgabeformat ('markdown' oder 'html')
            
        Returns:
            str: Generierter Kodierleitfaden als String
        """
        try:
            # Header erstellen
            content = [
                "# Kodierleitfaden für die qualitative Inhaltsanalyse",
                f"Version: {self.version}",
                f"Letzte Aktualisierung: {self.last_modified.strftime('%Y-%m-%d %H:%M')}\n",
                "## Kategorienübersicht\n"
            ]
            
            # Kategorien sortiert ausgeben
            for cat_name in sorted(self.categories.keys()):
                cat = self.categories[cat_name]
                content.extend([
                    f"### {cat_name}",
                    f"\n**Definition:**\n{cat['definition']}",
                    "\n**Ankerbeispiele:**"
                ])
                
                # Ankerbeispiele
                for example in cat['examples']:
                    content.append(f"- {example}")
                    
                        
                # Subkategorien
                if cat['subcategories']:
                    content.append("\n**Subkategorien:**")
                    for subcat, subdef in cat['subcategories'].items():
                        content.append(f"- {subcat}: {subdef}")
                
                content.append("\n---\n")
            
            # Zusammenführen und Formatierung
            guide_content = '\n'.join(content)
            
            # HTML Konvertierung falls gewünscht
            if format == 'html':
                guide_content = markdown.markdown(guide_content)
            
            # In Datei speichern falls Pfad angegeben
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(guide_content)
                print(f"Kodierleitfaden gespeichert unter: {output_path}")
            
            return guide_content
            
        except Exception as e:
            print(f"Fehler bei der Generierung des Kodierleitfadens: {str(e)}")
            return ""
            
    def export_to_excel(self, output_path: str) -> bool:
        try:
            data = []
            for cat_name, cat in self.categories.items():
                # Ensure rules exists with a default empty list
                rules = cat.get('rules', [])
                if isinstance(rules, str):
                    rules = [rules]
                elif not isinstance(rules, list):
                    rules = []
                    
                row = {
                    'Kategorie': cat_name,
                    'Typ': 'Hauptkategorie',
                    'Definition': cat.get('definition', ''),
                    'Ankerbeispiele': '\n'.join(cat.get('examples', [])),
                    'Kodierregeln': '\n'.join(rules),
                    'Letzte Änderung': cat.get('modified_date', '')
                }
                data.append(row)
                
                # Handle subcategories
                for subcat, subdef in cat.get('subcategories', {}).items():
                    row = {
                        'Kategorie': subcat,
                        'Typ': 'Subkategorie',
                        'Definition': subdef,
                        'Übergeordnete Kategorie': cat_name,
                        'Letzte Änderung': cat.get('modified_date', '')
                    }
                    data.append(row)
            
            # Create DataFrame and export
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False)
            print(f"Kodierleitfaden als Excel exportiert: {output_path}")
            return True
        
        except Exception as e:
            print(f"Fehler beim Excel-Export: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# --- Klasse: DeductiveCoder ---
# Aufgabe: Automatisches deduktives Codieren von Text-Chunks anhand des Leitfadens
class DeductiveCoder:
    """
    Ordnet Text-Chunks automatisch deduktive Kategorien zu basierend auf dem Kodierleitfaden.
    Nutzt GPT-4-Mini für die qualitative Inhaltsanalyse nach Mayring.
    """
    
    def __init__(self, model_name: str, temperature: str, coder_id: str):
        """
        Initializes the DeductiveCoder with configuration for the GPT model.
        
        Args:
            model_name (str): Name of the GPT model to use
            temperature (float): Controls randomness in the model's output 
                - Lower values (e.g., 0.3) make output more focused and deterministic
                - Higher values (e.g., 0.7) make output more creative and diverse
            coder_id (str): Unique identifier for this coder instance
        """
        self.model_name = model_name
        self.temperature = temperature
        self.coder_id = coder_id
        
        # Lade API Key aus .env Datei
        env_path = os.path.join(os.path.expanduser("~"), '.renviron.env')
        load_dotenv(env_path)
        
        # Initialisiere OpenAI Client
        self.client = OpenAI()
        
        # Prüfe API Key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(f"OPENAI_API_KEY nicht in {env_path} gefunden")

    async def _check_deductive_categories(self, chunk: str, deductive_categories: Dict) -> Optional[CodingResult]:
        """
        Prüft, ob der Chunk zu einer deduktiven Kategorie passt und reichert diese ggf. mit induktiven 
        Subkategorien an.
        """
        try:
            prompt = f"""
            Analysiere den Text in zwei Schritten:

            1. DEDUKTIVE HAUPTKATEGORIEN:
            Prüfe zunächst, ob der Text zu einer dieser deduktiven Hauptkategorien passt:
            {json.dumps(deductive_categories, indent=2, ensure_ascii=False)}

            2. INDUKTIVE SUBKATEGORIEN:
            Falls der Text einer deduktiven Hauptkategorie zugeordnet werden kann:
            - Prüfe die bestehenden Subkategorien
            - Identifiziere zusätzliche, neue thematische Aspekte
            - Schlage neue Subkategorien vor, wenn wichtige Aspekte nicht abgedeckt sind

            TEXT:
            {chunk}

            FORSCHUNGSFRAGE:
            {FORSCHUNGSFRAGE}

            Antworte nur mit einem JSON-Objekt:
            {{
                "matches_deductive": true/false,
                "category": "Name der deduktiven Hauptkategorie (oder leer)",
                "existing_subcategories": ["Liste", "bestehender", "Subkategorien"],
                "new_subcategories": ["Liste", "neuer", "Subkategorien"],
                "confidence": {{
                    "category": 0.9,
                    "subcategories": 0.8
                }},
                "justification": "Begründung der Zuordnung und neuer Subkategorien"
            }}

            WICHTIG:
            - Neue Subkategorien nur vorschlagen, wenn sie:
            1. Einen neuen, relevanten Aspekt erfassen
            2. Sich klar von bestehenden Subkategorien unterscheiden
            3. Zur Forschungsfrage beitragen
            - Subkategorien müssen spezifisch und präzise sein
            - Verwende ausschließlich deutsche Begriffe
            """

            input_tokens = estimate_tokens(prompt + chunk)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            output_tokens = estimate_tokens(response.choices[0].message.content)
            token_counter.add_tokens(input_tokens, output_tokens)
            
            if result.get('matches_deductive', False) and result.get('category'):
                # Kombiniere bestehende und neue Subkategorien
                all_subcategories = (
                    result.get('existing_subcategories', []) +
                    result.get('new_subcategories', [])
                )
                
                # Erstelle erweiterte Begründung
                justification = result.get('justification', '')
                if result.get('new_subcategories'):
                    justification += f"\nNeue Subkategorien: {', '.join(result['new_subcategories'])}"

                return CodingResult(
                    category=result['category'],
                    subcategories=all_subcategories,
                    justification=justification,
                    confidence=result.get('confidence', {'category': 0.0, 'subcategories': 0.0}),
                    text_references=[chunk[:100]],
                    uncertainties=[]
                )
            return None

        except Exception as e:
            print(f"Fehler bei der Prüfung deduktiver Kategorien: {str(e)}")
            return None

    
    async def code_chunk(self, chunk: str, categories: Dict[str, CategoryDefinition]) -> Optional[CodingResult]:
        try:
            # Prüfe zuerst die Relevanz des Chunks
            is_relevant = await self._check_relevance(chunk)
            if not is_relevant:
                return CodingResult(
                    category="Nicht kodiert",
                    subcategories=[],
                    justification="Segment hat keinen hinreichenden Bezug zur Forschungsfrage",
                    confidence={'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                    text_references=[chunk[:100]],
                    uncertainties=[]
                )
            
            # Prüfe zuerst deduktive Kategorien
            deductive_match = await self._check_deductive_categories(chunk, DEDUKTIVE_KATEGORIEN)
            if deductive_match:
                return deductive_match
            
            # Konvertiere CategoryDefinition in serialisierbares Dict
            categories_dict = {
                name: {
                    'definition': cat.definition,
                    'examples': cat.examples,
                    'rules': cat.rules,
                    'subcategories': cat.subcategories
                } for name, cat in categories.items()
            }

            prompt = f"""
            Analysiere folgenden Text nach dem Kategoriensystem von Mayring.
            Gib deine Antwort ausschließlich als valides JSON-Objekt zurück.

            TEXT:
            {chunk}

            KATEGORIENSYSTEM:
            {json.dumps(categories_dict, indent=2, ensure_ascii=False)}

            KODIERREGELN:
            {json.dumps(KODIERREGELN, indent=2, ensure_ascii=False)}

            Deine Antwort MUSS exakt diesem JSON-Format folgen:
            {{
                "category": "Name der Hauptkategorie",
                "subcategories": ["Liste", "der", "Subkategorien"],
                "justification": "Begründung der Zuordnung",
                "confidence": {{
                    "total": 0.85,
                    "category": 0.9,
                    "subcategories": 0.8
                }},
                "text_references": ["Relevante", "Textstellen"],
                "uncertainties": ["Optional: Unsicherheiten", "bei der Kodierung"]
            }}
            """

            input_tokens = estimate_tokens(prompt + chunk)

            # Synchroner API-Aufruf
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            output_tokens = estimate_tokens(response.choices[0].message.content)
            token_counter.add_tokens(input_tokens, output_tokens)

            return self._validate_coding(result)

        except Exception as e:
            print(f"Fehler bei der deduktiven Kodierung: {str(e)}")
            print(f"Chunk: {chunk[:100]}...")
            return None

    async def _check_relevance(self, chunk: str) -> bool:
        """
        Prüft die Relevanz eines Chunks für die Forschungsfrage und Kategorien.
        """
        try:
            prompt = f"""
            Analysiere den folgenden Text im Hinblick auf seine Relevanz für die Forschungsfrage:
            "{FORSCHUNGSFRAGE}"
            
            TEXT:
            {chunk}
            
            Prüfe:
            1. Bezieht sich der Text auf die Forschungsfrage
            2. Behandelt er Aspekte die zum Kontext der Forschungsfrage passen?
            3. Enthält er konkrete, analysierbare Aussagen?

            Antworte nur mit einem JSON-Objekt:
            {{
                "is_relevant": true/false,
                "justification": "Begründung der Entscheidung"
            }}
            """

            input_tokens = estimate_tokens(prompt + chunk)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}  # Wichtig: Erzwingt JSON-Antwort
            )

            # Extrahiere den JSON-String aus der Response
            result = json.loads(response.choices[0].message.content)
            
            output_tokens = estimate_tokens(response.choices[0].message.content)
            token_counter.add_tokens(input_tokens, output_tokens)

            if not isinstance(result, dict) or 'is_relevant' not in result:
                print(f"Ungültiges Response-Format: {result}")
                return True  # Im Zweifelsfall als relevant markieren
                
            return result['is_relevant']

        except Exception as e:
            print(f"Fehler bei der Relevanzprüfung: {str(e)}")
            print(f"Response war: {response.choices[0].message.content if response else 'Keine Response'}")
            return True  # Im Zweifelsfall als relevant markieren

    
    def _validate_coding(self, result: dict) -> Optional[CodingResult]:
        """Validiert und konvertiert das API-Ergebnis in ein CodingResult-Objekt"""
        try:
            return CodingResult(
                category=result.get('category', ''),
                subcategories=result.get('subcategories', []),
                justification=result.get('justification', ''),
                confidence=result.get('confidence', {'total': 0, 'category': 0, 'subcategories': 0}),
                text_references=result.get('text_references', []),
                uncertainties=result.get('uncertainties', [])
            )
        except Exception as e:
            print(f"Fehler bei der Validierung des Kodierungsergebnisses: {str(e)}")
            return None

# ---
# --- Klasse: InductiveCoder ---
# Aufgabe: Ergänzung deduktiver Kategorien durch induktive Kategorien mittels OpenAI API
class InductiveCoder:
    """
    Implements inductive category development according to Mayring's qualitative content analysis.
    
    This class handles the systematic development of categories from the text material,
    including multiple feedback loops, reliability checks, and quality criteria validation.
    The process follows Mayring's step-by-step model for inductive category formation.
    """
    
    def __init__(self, model_name: str, output_dir: str = None):
        """
        Initializes the InductiveCoder with OpenAI API access.
        
        Args:
            model_name (str): Name of the GPT model to use
            output_dir (str): Directory for saving reliability reports
        """
        # Load environment variables for API access
        load_dotenv(os.path.join(os.path.expanduser("~"), '.renviron.env'))
        self.client = openai.AsyncOpenAI()
        self.model_name = model_name

        # Set output directory
        self.output_dir = output_dir or CONFIG['OUTPUT_DIR']
        
        # Initialize category tracking
        self.category_development_history = []
        self.processed_material_percentage = 0
        
        # Define quality thresholds
        self.RELIABILITY_THRESHOLD = 0.80
        self.SIMILARITY_THRESHOLD = 0.70
        self.MIN_CATEGORY_SUPPORT = 3  # Minimum text segments per category
        
        # System context for the AI analysis
        self.system_context = """
        Sie sind ein Experte für qualitative Inhaltsanalyse nach Mayring.
        Ihre Aufgabe ist es, sinnvolle Kategorien in Interviewtexten zu identifizieren, indem Sie:
        1. das Material Zeile für Zeile durcharbeiten
        2. Kategorien induktiv aus dem Inhalt ableiten
        3. Beibehaltung klarer Kategoriendefinitionen und Beispiele
        4. Sicherstellen, dass die Kategorien eindeutig und gut definiert sind
        5. Schaffung eines systematischen und nachvollziehbaren Kategoriesystems
        """

    def _validate_category(self, category: dict) -> bool:
        """
        Validiert die Vollständigkeit einer Kategorie und initialisiert fehlende Standardwerte.
        
        Args:
            category: Die zu validierende Kategorie
            
        Returns:
            bool: True wenn die Kategorie nach Standardwert-Initialisierung valide ist
        """
        try:
            if not isinstance(category, dict):
                print(f"Warnung: Kategorie ist kein Dictionary: {category}")
                return False

            # Definiere Standardwerte für fehlende Felder
            defaults = {
                'name': 'Neue Kategorie',
                'definition': 'Zu definierende Kategorie',
                'example': '',
                'existing_subcategories': [],
                'new_subcategories': [],
                'justification': 'Induktiv aus dem Material entwickelt',
                'confidence': {'category': 0.7, 'subcategories': 0.7}
            }

            # Füge fehlende Felder mit Standardwerten hinzu
            missing_fields = []
            for field, default_value in defaults.items():
                if field not in category or not category[field]:
                    category[field] = default_value
                    missing_fields.append(field)

            if missing_fields:
                name = category.get('name', 'UNBENANNTE KATEGORIE')
                print(f"Info: Standardwerte für {name} ergänzt: {', '.join(missing_fields)}")

            # Validiere Datentypen
            type_checks = {
                'name': str,
                'definition': str,
                'example': str,
                'existing_subcategories': list,
                'new_subcategories': list,
                'justification': str,
                'confidence': dict
            }

            for field, expected_type in type_checks.items():
                if not isinstance(category[field], expected_type):
                    print(f"Warnung: Falscher Datentyp für {field} in {category['name']}")
                    return False

            return True

        except Exception as e:
            print(f"Fehler bei der Kategorievalidierung: {str(e)}")
            return False
    
    def _validate_category_system(self, categories: Dict[str, CategoryDefinition]) -> bool:
        """
        Validates the coherence and quality of the entire category system.
        
        Checks:
        - Mutual exclusiveness of categories
        - Completeness of the system
        - Consistency of abstraction levels
        - Quality of definitions and examples
        
        Args:
            categories: Category system to validate
            
        Returns:
            bool: True if system meets all quality criteria
        """
        try:
            validation_results = []
            
            # Check category definitions
            for name, category in categories.items():
                # Definition quality
                if len(category.definition.split()) < 15:  # Mindestens 15 Wörter
                    print(f"Warnung: Definition für '{name}' zu kurz")
                    validation_results.append(False)
                
                # Example quality
                if len(category.examples) < 2:  # Mindestens 2 Beispiele
                    print(f"Warnung: Zu wenige Beispiele für '{name}'")
                    validation_results.append(False)
                
                # Rules presence
                if not category.rules:
                    print(f"Warnung: Keine Kodierregeln für '{name}'")
                    validation_results.append(False)
            
            # Check category relationships
            for name1, cat1 in categories.items():
                for name2, cat2 in categories.items():
                    if name1 >= name2:
                        continue
                    
                    # Check for overlapping definitions
                    similarity = self._calculate_text_similarity(
                        cat1.definition,
                        cat2.definition
                    )
                    if similarity > 0.7:  # 70% Ähnlichkeitsschwelle
                        print(f"Warnung: Hohe Ähnlichkeit zwischen '{name1}' und '{name2}'")
                        validation_results.append(False)
            
            # Final validation result
            if not validation_results:
                return True  # Wenn keine Prüfungen fehlgeschlagen sind
            
            return all(validation_results)  # True nur wenn alle Prüfungen bestanden wurden
            
        except Exception as e:
            print(f"Fehler bei der Kategoriesystem-Validierung: {str(e)}")
            return False

    def _is_valid_category(self, category: dict) -> bool:
        required_fields = ['name', 'definition', 'example', 'existing_subcategories', 'new_subcategories', 'justification', 'confidence']
        
        if not isinstance(category, dict):
            print(f"Warnung: Kategorie ist kein Dictionary: {category}")
            return False

        missing_fields = [field for field in required_fields if field not in category or not category[field]]
        
        if missing_fields:
            print(f"Warnung: Fehlende Felder in Kategorie: {', '.join(missing_fields)}")
            return False
        
        return True


    def _integrate_new_category(self, new_category: dict, existing_categories: Dict[str, CategoryDefinition]) -> Tuple[Dict[str, CategoryDefinition], Optional[CategoryChange]]:
        if new_category['name'] in existing_categories:
            # Update existing category
            existing_cat = existing_categories[new_category['name']]
            updated_cat = CategoryDefinition(
                name=existing_cat.name,
                definition=existing_cat.definition,
                examples=list(set(existing_cat.examples + [new_category['example']])),
                rules=existing_cat.rules,
                subcategories={**existing_cat.subcategories, **dict.fromkeys(new_category['new_subcategories'])},
                added_date=existing_cat.added_date,
                modified_date=datetime.now().strftime("%Y-%m-%d")
            )
            existing_categories[new_category['name']] = updated_cat
            change = CategoryChange(
                category_name=new_category['name'],
                change_type='modify',
                description=f"Updated category {new_category['name']}",
                timestamp=datetime.now().isoformat(),
                old_value=existing_cat.__dict__,
                new_value=updated_cat.__dict__,
                justification=new_category['justification']
            )
        else:
            # Add new category
            new_cat = CategoryDefinition(
                name=new_category['name'],
                definition=new_category['definition'],
                examples=[new_category['example']],
                rules=[],  # You might want to add rules here
                subcategories=dict.fromkeys(new_category['new_subcategories']),
                added_date=datetime.now().strftime("%Y-%m-%d"),
                modified_date=datetime.now().strftime("%Y-%m-%d")
            )
            existing_categories[new_category['name']] = new_cat
            change = CategoryChange(
                category_name=new_category['name'],
                change_type='add',
                description=f"Added new category {new_category['name']}",
                timestamp=datetime.now().isoformat(),
                new_value=new_cat.__dict__,
                justification=new_category['justification']
            )
        
        return existing_categories, change


    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Berechnet die Ähnlichkeit zwischen zwei Texten.
        
        Args:
            text1: Erster Text
            text2: Zweiter Text
            
        Returns:
            float: Ähnlichkeitswert zwischen 0 und 1
        """
        try:
            # Texte in Wortmengen umwandeln
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            # Jaccard-Ähnlichkeit berechnen
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union == 0:
                return 0.0
                
            return intersection / union
            
        except Exception as e:
            print(f"Fehler bei der Textähnlichkeitsberechnung: {str(e)}")
            return 0.0

    async def develop_category_system(self, material: List[str]) -> Dict[str, CategoryDefinition]:
        """
        Develops a category system from the material following Mayring's steps.
        
        Args:
            material: List of text segments to analyze
            
        Returns:
            Dict[str, CategoryDefinition]: Developed category system
        """
        try:
            # 1. Initial category development (first 10% of material)
            initial_sample = material[:int(len(material) * 0.1)]
            initial_categories = await self._initial_categorization(initial_sample)
            
            # 2. First revision loop (up to 50% of material)
            mid_sample = material[:int(len(material) * 0.5)]
            revised_categories = await self._revision_loop(
                initial_categories,
                mid_sample,
                "First revision - 50% of material"
            )
            
            # 3. Second revision loop (remaining material)
            final_categories = await self._revision_loop(
                revised_categories,
                material,
                "Final revision - complete material"
            )
            
            # 4. Quality checks and documentation
            if self._validate_category_system(final_categories):
                self._document_category_development(
                    "Final category system validated",
                    final_categories
                )
                return final_categories
            else:
                raise ValueError("Final category system failed validation")
                
        except Exception as e:
            print(f"Error in category system development: {str(e)}")
            raise

    async def _initial_categorization(self, material: List[str]) -> Dict[str, CategoryDefinition]:
        """
        Performs the initial category development on a sample of the material.
        
        Args:
            material: Initial text segments for analysis
            
        Returns:
            Dict[str, CategoryDefinition]: Initial category system
        """
        categories = {}
        
        for segment in material:
            try:
                # Use AI to suggest categories for the segment
                suggested_categories = await self._extract_categories_from_segment(segment)
                
                # Process and integrate each suggested category
                for category in suggested_categories:
                    if self._is_valid_category(category):
                        if category['name'] in categories:
                            # Update existing category
                            categories[category['name']] = self._merge_category_definitions(
                                categories[category['name']],
                                category
                            )
                        else:
                            # Add new category
                            categories[category['name']] = CategoryDefinition(
                                name=category['name'],
                                definition=category['definition'],
                                examples=[category['example']],
                                rules=self._generate_KODIERREGELN(category),
                                subcategories={},
                                added_date=datetime.now().strftime("%Y-%m-%d"),
                                modified_date=datetime.now().strftime("%Y-%m-%d")
                            )
                            
                self._document_category_development(
                    f"Added categories from segment: {segment[:50]}...",
                    categories
                )
                
            except Exception as e:
                print(f"Error processing segment: {str(e)}")
                continue
                
        return categories

    async def _analyze_segment_coverage(self, segment: str, categories: Dict[str, CategoryDefinition]) -> float:
        """
        Analysiert, wie gut ein Segment durch das bestehende Kategoriensystem abgedeckt wird.
        """
        try:
            prompt = f"""
            Analysiere, wie gut das folgende Textsegment durch das bestehende Kategoriensystem 
            abgedeckt wird. Berücksichtige dabei:
            1. Inhaltliche Passung zu bestehenden Kategorien
            2. Vollständigkeit der Abdeckung
            3. Präzision der Kategorisierung

            TEXT:
            {segment}

            KATEGORIENSYSTEM:
            {json.dumps({name: cat.__dict__ for name, cat in categories.items()}, indent=2, ensure_ascii=False)}

            Antworte nur mit einem JSON-Objekt:
            {{
                "coverage_score": 0.8,
                "covered_aspects": ["Liste der abgedeckten Aspekte"],
                "uncovered_aspects": ["Liste der nicht abgedeckten Aspekte"],
                "justification": "Begründung der Bewertung"
            }}
            """

            input_tokens = estimate_tokens(prompt + segment)

            # Hier das wichtige await hinzufügen
            response =  await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_context},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            # Hier das Ergebnis direkt aus der Response extrahieren
            result = json.loads(response.choices[0].message.content)

            output_tokens = estimate_tokens(response.choices[0].message.content)
            token_counter.add_tokens(input_tokens, output_tokens)

            return result.get('coverage_score', 0.0)

        except Exception as e:
            print(f"Error in coverage analysis: {str(e)}")
            return 0.0

    async def _revision_loop(
        self,
        current_categories: Dict[str, CategoryDefinition],
        material: List[str],
        revision_phase: str
    ) -> Dict[str, CategoryDefinition]:
        revised_categories = current_categories.copy()
        category_changes = []
        segments_processed = 0
        
        for segment in material:
            segments_processed += 1
            self.processed_material_percentage = (segments_processed / len(material)) * 100
            
            coverage = await self._analyze_segment_coverage(segment, revised_categories)
            
            if coverage < self.RELIABILITY_THRESHOLD:
                new_categories = await self._extract_categories_from_segment(segment)
                
                for category in new_categories:
                    if self._is_valid_category(category):
                        revised_categories, change = self._integrate_new_category(category, revised_categories)
                        if change:
                            category_changes.append(change)
            
            if self._check_category_saturation(category_changes):
                self._document_category_development(
                    f"Category saturation reached at {self.processed_material_percentage:.1f}%",
                    revised_categories
                )
                break
        
        revised_categories = self._optimize_category_system(revised_categories)
        
        return revised_categories


    async def _extract_categories_from_segment(self, segment: str) -> List[dict]:
        """
        Extrahiert potenzielle Kategorien aus einem Textsegment mit Fokus auf deutsche Kategorienamen.
        """
        try:
            prompt = f"""
            Analysiere das folgende Textsegment nach Mayrings qualitativer Inhaltsanalyse.
            Fokussiere dabei auf die induktive Kategorienbildung und die Relevanz für die Forschungsfrage:
            "{FORSCHUNGSFRAGE}"

            WICHTIG - KATEGORIENSTRUKTUR UND HIERARCHIE:
            1. Unterscheide klar zwischen Haupt- und Subkategorien:

            HAUPTKATEGORIEN müssen:
            - übergeordnete Themenkomplexe abdecken
            - mehrere Subkategorien bündeln
            - sich gegenseitig ausschließen
            - maximal 4-6 Hauptkategorien insgesamt
            Beispiele: "Forschungsstrukturen", "Personalentwicklung"

            SUBKATEGORIEN müssen:
            - spezifische Aspekte einer Hauptkategorie beschreiben
            - sich eindeutig zuordnen lassen
            - konkrete Phänomene erfassen
            - 2-5 Subkategorien pro Hauptkategorie
            Beispiele: "Forschungsprofil", "Beruflicher Werdegang"

            WICHTIG - DEUTSCHE KATEGORIENBILDUNG:
            1. Kategorienamen MÜSSEN auf Deutsch sein - KEINE AUSNAHMEN!
            2. Übersetze englische Konzepte IMMER ins Deutsche
            3. Nutze etablierte deutsche Fachbegriffe
            4. Vermeide englische Wörter, Anglizismen, Fachbegriffe oder Syntax
            5. Kategorienname sollte prägnant sein (2-4 Wörter)

            WICHTIGE REGELN FÜR NEUE KATEGORIEN:
            1. VORRANG DEDUKTIVER KATEGORIEN:
            - Prüfe IMMER ZUERST, ob der Inhalt zu einer bestehenden deduktiven Kategorie passt
            - Neue Hauptkategorien NUR wenn keine bestehende Kategorie passt
            - Bevorzuge das Hinzufügen von Subkategorien zu bestehenden Hauptkategorien

            2. HIERARCHIE BEACHTEN:
            - Maximal 8-10 Hauptkategorien insgesamt
            - Neue Hauptkategorien müssen sich deutlich von bestehenden unterscheiden
            - Ähnliche Konzepte als Subkategorien einordnen

            TEXT:
            {segment}

            BESTEHENDES KATEGORIENSYSTEM:
            {json.dumps(DEDUKTIVE_KATEGORIEN, indent=2, ensure_ascii=False)}

            Prüfe bei jedem neuen Konzept:
            1. Passt es in eine bestehende Hauptkategorie?
            2. Könnte es als Subkategorie eingeordnet werden?
            3. Ist eine neue Hauptkategorie WIRKLICH notwendig?

            Antworte nur mit einem JSON-Array von NEUEN Kategorien:
            [
                {{
                    "name": "Name der neuen Hauptkategorie (ZWINGEND DEUTSCH!)",
                    "definition": "Präzise Definition der Hauptkategorie",
                    "example": "Relevante Textstelle als Beispiel",
                    "existing_subcategories": [],
                    "new_subcategories": [
                        "Subkategorie 1 (präzise, spezifisch)",
                        "Subkategorie 2 (präzise, spezifisch)"
                    ],
                    "justification": "Prägnante Paraphrase der relevanten Textaspekte und ihre Bedeutung für die Kategorienzuordnung",
                    "confidence": {{
                        "category": 0.9,
                        "subcategories": 0.8
                    }}
                }}
            ]

            WICHTIG FÜR DIE BEGRÜNDUNG (justification):
            - Knappe Paraphrase der relevanten Textinhalte ohne einleitende Phrasen wie "Der Text beschreibt..."
            - Fokus auf spezifische Informationen, die für die Kategorienzuordnung und die Forschungsfrage relevant sind
            - Vermeide Redundanzen und allgemeine Beschreibungen
            - Erkläre, warum eine neue Hauptkategorie nötig ist oder warum bestehende Kategorien nicht ausreichen

            WICHTIG: 
            1. Erstelle nur dann neue Hauptkategorien, wenn der Inhalt nicht durch das bestehende System abgedeckt wird!
            2. Prüfe immer, ob neue Aspekte nicht als Subkategorien in bestehende Hauptkategorien eingeordnet werden können!
            3. Kategorienamen MÜSSEN auf Deutsch sein!
            4. Stelle sicher, dass Hauptkategorien sich nicht überschneiden!
            5. Bei Unsicherheit IMMER die deutsche Variante wählen!
            """

            input_tokens = estimate_tokens(prompt + segment)

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. "
                    "Deine Aufgabe ist es, AUSSCHLIESSLICH DEUTSCHE Kategoriennamen zu erstellen und "
                    "eine klare Hierarchie zwischen Haupt- und Subkategorien zu gewährleisten."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            output_tokens = estimate_tokens(response.choices[0].message.content)
        
            token_counter.add_tokens(input_tokens, output_tokens)
            
            # Stelle sicher, dass das Ergebnis eine Liste ist
            if isinstance(result, dict):
                result = [result]
            elif not isinstance(result, list):
                print(f"Unerwartetes Response-Format: {type(result)}")
                return []

            # Zusätzliche Validierung der Kategorienamen
            validated_categories = []
            for category in result:
                # Prüfe auf englische Wörter im Kategorienamen
                name = category.get('name', '')
                english_indicators = {'and', 'of', 'in', 'the', 'research', 'higher', 
                                'education', 'private', 'state', 'involvement', 
                                'sector', 'institutions', 'quality', 'management',
                                'development', 'framework', 'impact', 'challenge'}
                
                words = set(name.lower().split())
                if any(word in english_indicators for word in words):
                    print(f"Überspringe Kategorie mit englischem Namen: {name}")
                    continue
                    
                validated_categories.append(category)

            return self._validate_categories(validated_categories)

        except Exception as e:
            print(f"Error in category extraction: {str(e)}")
            if 'response' in locals():
                print(f"Response was: {response.choices[0].message.content}")
            return []


    def _validate_categories(self, categories: List[dict]) -> List[dict]:
        """
        Validiert die extrahierten Kategorien anhand der Qualitätskriterien von Mayring.
        """
        valid_categories = []
        required_fields = {
            'name', 'definition', 'existing_subcategories', 
            'new_subcategories', 'justification', 'example', 'confidence'
        }
        
        for category in categories:
            try:
                category_name = category.get('name', 'UNBENANNTE KATEGORIE')
            
                # Prüfe ob alle erforderlichen Felder vorhanden sind
                missing_fields = [field for field in required_fields if field not in category]
                if missing_fields:
                    print(f"Warnung: Fehlende Felder in Kategorie '{category_name}': {', '.join(missing_fields)}")
                    continue
                    
                # Prüfe ob Subkategorien vorhanden sind
                has_subcategories = (
                    len(category['existing_subcategories']) > 0 or 
                    len(category['new_subcategories']) > 0
                )
                if not has_subcategories:
                    print(f"Warnung: Keine Subkategorien für {category['name']}")
                    continue
                    
                # Weitere Qualitätsprüfungen...
                valid_categories.append(category)
                
            except Exception as e:
                print(f"Fehler bei Kategorienvalidierung von '{category_name}': {str(e)}")
                continue
            
        return valid_categories

    def _optimize_category_system(self, categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        Optimizes the category system by merging similar categories and adjusting abstraction levels.
        
        Args:
            categories: Current category system
            
        Returns:
            Dict[str, CategoryDefinition]: Optimized category system
        """
        optimized = categories.copy()
        
        # Get all possible pairs of category names using itertools.combinations
        # This ensures we compare each pair exactly once
        category_pairs = itertools.combinations(list(optimized.keys()), 2)
        
        # Check each pair of categories for similarity
        for cat1_name, cat2_name in category_pairs:
            # Calculate similarity between the two categories
            similarity = self._calculate_category_similarity(
                optimized[cat1_name],
                optimized[cat2_name]
            )
            
            # If categories are similar enough to merge
            if similarity > self.SIMILARITY_THRESHOLD:
                # Create a merged category
                merged = self._merge_categories(
                    optimized[cat1_name], 
                    optimized[cat2_name]
                )
                # Add the merged category and remove the original ones
                optimized[merged.name] = merged
                del optimized[cat1_name]
                del optimized[cat2_name]
        
        # After merging, adjust abstraction levels of remaining categories
        for name, category in list(optimized.items()):  # Use list() to avoid runtime modification issues
            if len(category.examples) > self.MIN_CATEGORY_SUPPORT:
                optimized[name] = self._adjust_abstraction_level(category)
        
        return optimized

    def _document_category_development(self, 
                                    action: str, 
                                    categories: Dict[str, CategoryDefinition]) -> None:
        """
        Documents the category development process.
        
        Args:
            action: Description of the development step
            categories: Current state of the category system
        """
        self.category_development_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'processed_percentage': self.processed_material_percentage,
            'num_categories': len(categories),
            'categories': {name: cat.__dict__ for name, cat in categories.items()}
        })

    def generate_development_report(self) -> str:
        """
        Generates a detailed report of the category development process.
        
        Returns:
            str: Formatted report of the category development process
        """
        report = ["# Category Development Report\n"]
        
        for entry in self.category_development_history:
            report.extend([
                f"## {entry['timestamp']}",
                f"Action: {entry['action']}",
                f"Material processed: {entry['processed_percentage']:.1f}%",
                f"Number of categories: {entry['num_categories']}\n",
                "### Categories:"
            ])
            
            for name, cat in entry['categories'].items():
                report.extend([
                    f"#### {name}",
                    f"Definition: {cat['definition']}",
                    f"Examples: {', '.join(cat['examples'])}",
                    "---\n"
                ])
                
        return '\n'.join(report)

    def _calculate_reliability(self, codings: List[Dict]) -> float:
        """
        Berechnet die Intercoder-Reliabilität mit Krippendorffs Alpha.
        Optimiert die Berechnung durch einmalige Iteration über die Daten.
        
        Args:
            codings: Liste der Kodierungen mit Format:
                {
                    'segment_id': str,
                    'coder_id': str,
                    'category': str,
                    'confidence': float,
                    'subcategories': List[str]
                }
                
        Returns:
            float: Krippendorffs Alpha (-1 bis 1)
        """
        try:
            # Sammle alle benötigten Statistiken in einem Durchlauf
            statistics = {
                'segment_codings': defaultdict(dict),  # {segment_id: {coder_id: category}}
                'category_frequencies': defaultdict(int),  # {category: count}
                'total_pairs': 0,
                'agreements': 0,
                'coders': set(),
                'segments': set()
            }
            
            # Ein Durchlauf für alle Statistiken
            for coding in codings:
                segment_id = coding['segment_id']
                coder_id = coding['coder_id']
                category = coding['category']
                
                # Aktualisiere Segment-Kodierungen
                statistics['segment_codings'][segment_id][coder_id] = category
                
                # Aktualisiere Kategorienhäufigkeiten
                statistics['category_frequencies'][category] += 1
                
                # Sammle eindeutige Codierer und Segmente
                statistics['coders'].add(coder_id)
                statistics['segments'].add(segment_id)
            
            # Berechne Übereinstimmungen und Paare
            for segment_codes in statistics['segment_codings'].values():
                coders = list(segment_codes.keys())
                # Vergleiche jedes Codiererpaar
                for i in range(len(coders)):
                    for j in range(i + 1, len(coders)):
                        statistics['total_pairs'] += 1
                        if segment_codes[coders[i]] == segment_codes[coders[j]]:
                            statistics['agreements'] += 1

            # Berechne erwartete Zufallsübereinstimmung
            total_codings = sum(statistics['category_frequencies'].values())
            expected_agreement = 0
            
            if total_codings > 1:  # Verhindere Division durch Null
                for count in statistics['category_frequencies'].values():
                    expected_agreement += (count / total_codings) ** 2

            # Berechne Krippendorffs Alpha
            observed_agreement = statistics['agreements'] / statistics['total_pairs'] if statistics['total_pairs'] > 0 else 0
            alpha = (observed_agreement - expected_agreement) / (1 - expected_agreement) if expected_agreement != 1 else 1

            # Dokumentiere die Ergebnisse
            self._document_reliability_results(
                alpha=alpha,
                total_segments=len(statistics['segments']),
                total_coders=len(statistics['coders']),
                category_frequencies=dict(statistics['category_frequencies'])
            )

            return alpha

        except Exception as e:
            print(f"Fehler bei der Reliabilitätsberechnung: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _document_reliability_results(
        self, 
        alpha: float, 
        total_segments: int, 
        total_coders: int, 
        category_frequencies: dict
    ) -> str:
        """
        Generiert einen detaillierten Bericht über die Intercoder-Reliabilität.

        Args:
            alpha: Krippendorffs Alpha Koeffizient
            total_segments: Gesamtzahl der analysierten Segmente
            total_coders: Gesamtzahl der Kodierer
            category_frequencies: Häufigkeiten der Kategorien
            
        Returns:
            str: Formatierter Bericht als Markdown-Text
        """
        try:
            # Bestimme das Reliabilitätsniveau basierend auf Alpha
            reliability_level = (
                "Excellent" if alpha > 0.8 else
                "Acceptable" if alpha > 0.667 else
                "Poor"
            )
            
            # Erstelle den Bericht
            report = [
                "# Intercoder Reliability Analysis Report",
                f"\n## Overview",
                f"- Number of text segments: {total_segments}",
                f"- Number of coders: {total_coders}",
                f"- Krippendorff's Alpha: {alpha:.3f}",
                f"- Reliability Assessment: {reliability_level}",
                "\n## Category Usage",
                "| Category | Frequency |",
                "|----------|-----------|"
            ]
            
            # Füge Kategorienhäufigkeiten hinzu
            for category, frequency in sorted(category_frequencies.items(), key=lambda x: x[1], reverse=True):
                report.append(f"| {category} | {frequency} |")
            
            # Füge Empfehlungen hinzu
            report.extend([
                "\n## Recommendations",
                "Based on the reliability analysis, the following actions are suggested:"
            ])
            
            if alpha < 0.667:
                report.extend([
                    "1. Review and clarify category definitions",
                    "2. Provide additional coder training",
                    "3. Consider merging similar categories",
                    "4. Add more explicit coding rules"
                ])
            elif alpha < 0.8:
                report.extend([
                    "1. Review cases of disagreement",
                    "2. Refine coding guidelines for ambiguous cases",
                    "3. Consider additional coder calibration"
                ])
            else:
                report.extend([
                    "1. Continue with current coding approach",
                    "2. Document successful coding practices",
                    "3. Consider using this category system as a template for future analyses"
                ])
            
            # Füge detaillierte Analyse hinzu
            report.extend([
                "\n## Detailed Analysis",
                "### Interpretation of Krippendorff's Alpha",
                "- **> 0.800**: Excellent reliability",
                "- **0.667 - 0.800**: Acceptable reliability",
                "- **< 0.667**: Poor reliability",
                "\n### Category Usage Analysis",
                "- Most frequently used category: " + max(category_frequencies.items(), key=lambda x: x[1])[0],
                "- Least frequently used category: " + min(category_frequencies.items(), key=lambda x: x[1])[0],
                "- Number of categories with single use: " + str(sum(1 for x in category_frequencies.values() if x == 1)),
                "\n### Coder Performance",
                "- Average segments per coder: " + f"{total_segments/total_coders:.1f}" if total_coders > 0 else "N/A"
            ])
            
            # Füge Zeitstempel hinzu
            report.append(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return '\n'.join(report)
            
        except Exception as e:
            print(f"Error generating reliability report: {str(e)}")
            import traceback
            traceback.print_exc()
            return "# Reliability Report\n\nError generating report"


    def _check_category_saturation(self, category_changes: List[Dict]) -> bool:
        """
        Überprüft, ob das Kategoriensystem gesättigt ist.
        
        Args:
            category_changes: Liste der Kategorienänderungen
            
        Returns:
            bool: True wenn Sättigung erreicht ist, sonst False
        """
        try:
            # Konstanten für Abbruchkriterien
            MAX_ITERATIONS = 10  # Maximale Anzahl der Durchläufe
            MIN_MATERIAL_PERCENTAGE = 70  # Minimaler Prozentsatz des Materials
            STABILITY_THRESHOLD = 3  # Anzahl der Durchläufe ohne signifikante Änderungen

            # Prüfe ob maximale Iterationen erreicht
            if hasattr(self, '_iteration_count'):
                self._iteration_count += 1
            else:
                self._iteration_count = 1

            if self._iteration_count >= MAX_ITERATIONS:
                print(f"\nSättigung erreicht: Maximale Anzahl von {MAX_ITERATIONS} Durchläufen erreicht")
                return True

            # Prüfe ob genügend Material analysiert wurde
            if self.processed_material_percentage >= MIN_MATERIAL_PERCENTAGE:
                if not category_changes:
                    if hasattr(self, '_stable_iterations'):
                        self._stable_iterations += 1
                    else:
                        self._stable_iterations = 1

                    if self._stable_iterations >= STABILITY_THRESHOLD:
                        print(f"\nSättigung erreicht: {STABILITY_THRESHOLD} stabile Durchläufe bei {self.processed_material_percentage:.1f}% des Materials")
                        return True
                else:
                    self._stable_iterations = 0

            # Prüfe Art der Änderungen
            significant_changes = False
            for change in category_changes:
                # Neue Kategorien oder substanzielle Änderungen
                if change.get('change_type') in ['add', 'modify']:
                    if change.get('change_type') == 'modify':
                        # Bei Modifikationen: Prüfe ob es sich um wesentliche Änderungen handelt
                        old_val = change.get('old_value', {})
                        new_val = change.get('new_value', {})
                        
                        # Prüfe auf wesentliche Änderungen in Definition oder Struktur
                        if (old_val.get('definition') != new_val.get('definition') or
                            old_val.get('subcategories') != new_val.get('subcategories')):
                            significant_changes = True
                            break
                    else:  # Bei neuen Kategorien
                        significant_changes = True
                        break

            # Logging des Fortschritts
            print(f"\nSättigungsprüfung:")
            print(f"- Durchlauf: {self._iteration_count}/{MAX_ITERATIONS}")
            print(f"- Material analysiert: {self.processed_material_percentage:.1f}%")
            print(f"- Stabile Durchläufe: {getattr(self, '_stable_iterations', 0)}/{STABILITY_THRESHOLD}")
            print(f"- Signifikante Änderungen: {'Ja' if significant_changes else 'Nein'}")

            return not significant_changes and self.processed_material_percentage >= MIN_MATERIAL_PERCENTAGE
                
        except Exception as e:
            print(f"Fehler bei der Sättigungsprüfung: {str(e)}")
            return False


    def _merge_categories(self, 
                        cat1: CategoryDefinition, 
                        cat2: CategoryDefinition) -> CategoryDefinition:
        """
        Merges two similar categories.
        
        Args:
            cat1: First category
            cat2: Second category
            
        Returns:
            CategoryDefinition: Merged category
        """
        return CategoryDefinition(
            name=f"{cat1.name}_{cat2.name}",
            definition=self._merge_definitions(cat1.definition, cat2.definition),
            examples=list(set(cat1.examples + cat2.examples)),
            rules=list(set(cat1.rules + cat2.rules)),
            subcategories={**cat1.subcategories, **cat2.subcategories},
            added_date=datetime.now().strftime("%Y-%m-%d"),
            modified_date=datetime.now().strftime("%Y-%m-%d")
        )

    def _adjust_abstraction_level(self, 
                                category: CategoryDefinition) -> CategoryDefinition:
        """
        Adjusts the abstraction level of a category.
        
        Args:
            category: Category to adjust
            
        Returns:
            CategoryDefinition: Adjusted category
        """
        # Implementation of abstraction level adjustment
        pass

# --- Klasse: ManualCoder ---
class ManualCoder:
    def __init__(self, coder_id: str):
        self.coder_id = coder_id
        self.root = None
        self.text_chunk = None
        self.category_listbox = None
        self.categories = {}
        self.current_coding = None
        self._is_processing = False
        
    async def code_chunk(self, chunk: str, categories: Dict[str, CategoryDefinition]) -> Optional[CodingResult]:
        """Nutzt das asyncio Event Loop, um tkinter korrekt zu starten"""
        try:
            self.categories = categories
            self.current_coding = None
            
            # Erstelle und starte das Tkinter-Fenster im Hauptthread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_tk_window, chunk)
            
            # Nach dem Beenden des Fensters
            if self.root:
                try:
                    self.root.destroy()
                    self.root = None
                except:
                    pass
                
            return self.current_coding
            
        except Exception as e:
            print(f"Fehler in code_chunk: {str(e)}")
            if self.root:
                try:
                    self.root.destroy()
                    self.root = None
                except:
                    pass
            return None

    def _run_tk_window(self, chunk: str):
        """Führt das Tkinter-Fenster im Hauptthread aus"""
        try:
            if self.root is not None:
                try:
                    self.root.quit()
                    self.root.destroy()
                except:
                    pass
            
            self.root = tk.Tk()
            self.root.title(f"Manueller Coder - {self.coder_id}")
            self.root.geometry("800x600")
            
            # Protokoll für das Schließen des Fensters
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
            # GUI erstellen...
            self.text_chunk = tk.Text(self.root, height=10, wrap=tk.WORD)
            self.text_chunk.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            self.text_chunk.insert(tk.END, chunk)
            
            category_frame = ttk.Frame(self.root)
            category_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            self.category_listbox = tk.Listbox(category_frame)
            self.category_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            scrollbar = ttk.Scrollbar(category_frame, orient=tk.VERTICAL, command=self.category_listbox.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.category_listbox.config(yscrollcommand=scrollbar.set)
            
            button_frame = ttk.Frame(self.root)
            button_frame.pack(padx=10, pady=10)
            
            ttk.Button(button_frame, text="Kodieren", command=self._safe_code_selection).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Neue Hauptkategorie", command=self._safe_new_main_category).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Neue Subkategorie", command=self._safe_new_sub_category).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Überspringen", command=self._safe_skip_chunk).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Abbrechen", command=self._safe_abort_coding).pack(side=tk.LEFT, padx=5)
            
            self.update_category_list()
            
            # Fenster in den Vordergrund bringen
            self.root.lift()  # Hebt das Fenster über andere
            self.root.attributes('-topmost', True)  # Hält es im Vordergrund
            self.root.attributes('-topmost', False)  # Erlaubt anderen Fenstern wieder in den Vordergrund zu kommen
            self.root.focus_force()  # Erzwingt den Fokus
            
            # Plattformspezifische Anpassungen
            if platform.system() == "Darwin":  # macOS
                self.root.createcommand('tk::mac::RaiseWindow', self.root.lift)
            
            # Update des Fensters erzwingen
            self.root.update()
            
            # Starte MainLoop
            self.root.mainloop()
            
        except Exception as e:
            print(f"Fehler beim Erstellen des Tkinter-Fensters: {str(e)}")
            self.current_coding = None
            return

    def update_category_list(self):
        """Aktualisiert die Liste der Kategorien in der GUI"""
        if not self.category_listbox:
            return
            
        self.category_listbox.delete(0, tk.END)
        for i, (cat_name, cat_def) in enumerate(self.categories.items(), 1):
            # Hauptkategorie
            self.category_listbox.insert(tk.END, f"{i}. {cat_name}")
            # Subkategorien
            for j, sub in enumerate(cat_def.subcategories.keys(), 1):
                self.category_listbox.insert(tk.END, f"   {i}.{j} {sub}")

        # Scrolle zum Anfang der Liste
        self.category_listbox.see(0)

    def on_closing(self):
        """Sicheres Schließen des Fensters"""
        try:
            if messagebox.askokcancel("Beenden", "Möchten Sie das Kodieren wirklich beenden?"):
                self.current_coding = None
                self._is_processing = False
                if self.root:
                    try:
                        self.root.quit()
                    except:
                        pass
        except:
            pass

    def _safe_code_selection(self):
        """Thread-sichere Kodierungsauswahl"""
        if not self._is_processing:
            try:
                selection = self.category_listbox.curselection()
                if not selection:
                    messagebox.showwarning("Warnung", "Bitte wählen Sie eine Kategorie aus.")
                    return
                
                index = selection[0]
                category = self.category_listbox.get(index)
                
                if '.' in category:
                    # Für Subkategorien
                    parts = category.split('.')
                    if len(parts) >= 2:
                        main_cat = parts[0].split('. ', 1)[-1] if '. ' in parts[0] else parts[0]
                        sub_cat = parts[-1].strip()
                    else:
                        raise ValueError("Ungültiges Subkategorie-Format")
                else:
                    # Für Hauptkategorien
                    main_cat = category.split('. ', 1)[-1] if '. ' in category else category
                    sub_cat = None

                self.current_coding = CodingResult(
                    category=main_cat,
                    subcategories=[sub_cat] if sub_cat else [],
                    justification="Manuelle Kodierung",
                    confidence={'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                    text_references=[self.text_chunk.get("1.0", tk.END)[:100]]
                )
                
                self._is_processing = False
                self.root.quit()
                
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler bei der Kategorieauswahl: {str(e)}")
                print(f"Fehler bei der Kategorieauswahl: {str(e)}")

    def _safe_abort_coding(self):
        """Thread-sicheres Abbrechen"""
        if not self._is_processing:
            if messagebox.askyesno("Abbrechen", 
                "Möchten Sie wirklich das manuelle Kodieren komplett abbrechen?"):
                self.current_coding = "ABORT_ALL"
                self._is_processing = False
                # Sicheres Beenden des Fensters
                try:
                    self.root.quit()
                    self.root.destroy()
                    self.root = None  # Wichtig: Referenz löschen
                except:
                    pass

    def _safe_skip_chunk(self):
        """Thread-sicheres Überspringen"""
        if not self._is_processing:
            self.current_coding = CodingResult(
                category="Nicht kodiert",
                subcategories=[],
                justification="Chunk übersprungen",
                confidence={'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                text_references=[self.text_chunk.get("1.0", tk.END)[:100]]
            )
            self._is_processing = False
            self.root.quit()

    def _safe_new_main_category(self):
        """Thread-sichere neue Hauptkategorie"""
        if not self._is_processing:
            new_cat = simpledialog.askstring("Neue Hauptkategorie", 
                "Geben Sie den Namen der neuen Hauptkategorie ein:")
            if new_cat:
                if new_cat in self.categories:
                    messagebox.showwarning("Warnung", "Diese Kategorie existiert bereits.")
                    return
                self.categories[new_cat] = CategoryDefinition(
                    name=new_cat,
                    definition="",
                    examples=[],
                    rules=[],
                    subcategories={},
                    added_date=datetime.now().strftime("%Y-%m-%d"),
                    modified_date=datetime.now().strftime("%Y-%m-%d")
                )
                self.update_category_list()

    def _safe_new_sub_category(self):
        """Thread-sichere neue Subkategorie"""
        if not self._is_processing:
            main_cat = simpledialog.askstring("Hauptkategorie", "Geben Sie die Nummer der Hauptkategorie ein:")
            if main_cat and main_cat.isdigit():
                main_cat = int(main_cat)
                if 1 <= main_cat <= len(self.categories):
                    new_sub = simpledialog.askstring("Neue Subkategorie", 
                        "Geben Sie den Namen der neuen Subkategorie ein:")
                    if new_sub:
                        cat_name = list(self.categories.keys())[main_cat-1]
                        if new_sub in self.categories[cat_name].subcategories:
                            messagebox.showwarning("Warnung", "Diese Subkategorie existiert bereits.")
                            return
                        self.categories[cat_name].subcategories[new_sub] = ""
                        self.update_category_list()
                else:
                    messagebox.showwarning("Warnung", "Ungültige Hauptkategorienummer.")
            else:
                messagebox.showwarning("Warnung", "Bitte geben Sie eine gültige Nummer ein.")


# --- Klasse: ResultsExporter ---
# Aufgabe: Export der kodierten Daten und des finalen Kategoriensystems
class ResultsExporter:
    """
    Exports the results of the analysis in various formats (JSON, CSV, Excel, Markdown).
    Supports the documentation of coded chunks and the category system.
    """
    
    def __init__(self, output_dir: str, attribute_labels: Dict[str, str], inductive_coder: InductiveCoder = None):
        self.output_dir = output_dir
        self.attribute_labels = attribute_labels
        self.inductive_coder = inductive_coder
        os.makedirs(output_dir, exist_ok=True)

    def _get_consensus_coding(self, segment_codes: List[Dict]) -> Optional[Dict]:
        """
        Determines the consensus coding for a segment based on multiple codings.
        
        Args:
            segment_codes (List[Dict]): List of coding results for a single segment
                
        Returns:
            Optional[Dict]: The consensus coding or None if no consensus can be reached
        """
        if not segment_codes:
            return None

        # Count occurrences of each category
        category_counts = Counter(coding['category'] for coding in segment_codes)
        
        # Find the most common category
        most_common_category, count = category_counts.most_common(1)[0]
        
        # Check if there's a clear consensus (more than 50% agreement)
        if count > len(segment_codes) / 2:
            # Find the coding with the highest confidence for this category
            matching_codings = [
                coding for coding in segment_codes 
                if coding['category'] == most_common_category
            ]
            
            # Sort by confidence and take the one with highest confidence
            consensus_coding = max(
                matching_codings,
                key=lambda x: float(x['confidence']) if isinstance(x['confidence'], (int, float)) 
                else float(x['confidence'].get('total', 0))
            )
            
            return consensus_coding
        else:
            # No clear consensus
            return None
    
    def _get_majority_coding(self, segment_codes: List[Dict]) -> Optional[Dict]:
        # Implementierung ähnlich wie _get_consensus_coding, aber mit einfacher Mehrheit
        pass

    def _get_manual_priority_coding(self, segment_codes: List[Dict]) -> Optional[Dict]:
        # Implementierung für manuelle Priorisierung
        pass
    
    def export_merge_analysis(self, original_categories: Dict[str, CategoryDefinition], 
                            merged_categories: Dict[str, CategoryDefinition],
                            merge_log: List[Dict]):
        """Exportiert eine detaillierte Analyse der Kategorienzusammenführungen."""
        analysis_path = os.path.join(self.output_dir, f'merge_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write("# Analyse der Kategorienzusammenführungen\n\n")
            
            f.write("## Übersicht\n")
            f.write(f"- Ursprüngliche Kategorien: {len(original_categories)}\n")
            f.write(f"- Zusammengeführte Kategorien: {len(merged_categories)}\n")
            f.write(f"- Anzahl der Zusammenführungen: {len(merge_log)}\n\n")
            
            f.write("## Detaillierte Zusammenführungen\n")
            for merge in merge_log:
                f.write(f"### {merge['new_category']}\n")
                f.write(f"- Zusammengeführt aus: {', '.join(merge['merged'])}\n")
                f.write(f"- Ähnlichkeit: {merge['similarity']:.2f}\n")
                f.write(f"- Zeitpunkt: {merge['timestamp']}\n\n")
                
                f.write("#### Ursprüngliche Definitionen:\n")
                for cat in merge['merged']:
                    f.write(f"- {cat}: {original_categories[cat].definition}\n")
                f.write("\n")
                
                f.write("#### Neue Definition:\n")
                f.write(f"{merged_categories[merge['new_category']].definition}\n\n")
            
            f.write("## Statistiken\n")
            f.write(f"- Durchschnittliche Ähnlichkeit bei Zusammenführungen: {sum(m['similarity'] for m in merge_log) / len(merge_log):.2f}\n")
            f.write(f"- Kategorienreduktion: {(1 - len(merged_categories) / len(original_categories)) * 100:.1f}%\n")
        
        print(f"Merge-Analyse exportiert nach: {analysis_path}")
        pass
    
    def _prepare_coding_for_export(self, coding: dict, chunk: str, chunk_id: int, doc_name: str) -> dict:
        """Bereitet eine Kodierung für den Export vor."""
        attribut1, attribut2 = self._extract_metadata(doc_name)
        export_data = {
            'Dokument': doc_name,
            self.attribute_labels['attribut1']: attribut1,
            self.attribute_labels['attribut2']: attribut2,
            'Chunk_Nr': chunk_id,
            'Text': chunk,
            'Kodiert': 'Nein' if coding['category'] == "Nicht kodiert" else 'Ja',
            'Hauptkategorie': coding.get('category', ''),
            'Kategorietyp': coding.get('Kategorietyp', 'unbekannt'),
            'Subkategorien': ', '.join(coding.get('subcategories', [])),
            'Begründung': coding.get('justification', ''),
            'Konfidenz': coding.get('confidence', 0),
            'Mehrfachkodierung': 'Ja' if len(coding.get('subcategories', [])) > 1 else 'Nein'
        }
        return export_data

    def _extract_metadata(self, filename: str) -> tuple:
        """Extrahiert Metadaten aus dem Dateinamen"""
        from pathlib import Path
        tokens = Path(filename).stem.split("_")
        attribut1 = tokens[0] if len(tokens) >= 1 else ""
        attribut2 = tokens[1] if len(tokens) >= 2 else ""
        return attribut1, attribut2

    def _export_frequency_analysis(
            self, writer, df_coded: pd.DataFrame, df_pivot_main: pd.DataFrame, 
            df_pivot_subcats: pd.DataFrame, df_pivot_attr1: pd.DataFrame, 
            df_pivot_attr2: pd.DataFrame, attribut1_label: str, attribut2_label: str
        ) -> None:
        """
        Exportiert die Häufigkeitsanalysen mit Randsummen und bedingter Formatierung.
        """
        try:
            if 'Häufigkeitsanalysen' not in writer.sheets:
                writer.book.create_sheet('Häufigkeitsanalysen')
            
            worksheet = writer.sheets['Häufigkeitsanalysen']
            start_row = 1

            # 1. Hauptkategorien nach Dokumenten
            worksheet.cell(row=start_row, column=1, value="1. Hauptkategorien nach Dokumenten")
            start_row += 1

            # Füge Randsummen hinzu
            df_main_with_totals = df_pivot_main.copy()
            df_main_with_totals['Summe'] = df_main_with_totals.sum(axis=1)
            df_main_with_totals.loc['Summe'] = df_main_with_totals.sum()

            # Export der Hauptkategorien
            for r_idx, row in enumerate(dataframe_to_rows(df_main_with_totals, index=True), start_row):
                for c_idx, value in enumerate(row, 1):
                    cell = worksheet.cell(row=r_idx, column=c_idx, value=value)
                    if r_idx == len(df_main_with_totals) + start_row - 1 or c_idx == len(row):
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')

            # Bedingte Formatierung für Hauptkategorien
            data_range = f"B{start_row+1}:{get_column_letter(len(df_pivot_main.columns)+1)}{start_row+len(df_pivot_main)}"
            self._add_color_scale(worksheet, data_range)

            # 2. Subkategorien nach Dokumenten
            if not df_pivot_subcats.empty:
                start_row += len(df_main_with_totals) + 3
                worksheet.cell(row=start_row, column=1, value="2. Subkategorien nach Dokumenten")
                start_row += 1

                # Pivot-Tabelle für Subkategorien erstellen
                df_sub_pivot = pd.pivot_table(
                    df_pivot_subcats,
                    index=['Subkategorie'],
                    columns=[attribut1_label, attribut2_label],
                    values='Chunk_Nr',
                    aggfunc='count',
                    fill_value=0
                )

                # Randsummen hinzufügen
                df_sub_with_totals = df_sub_pivot.copy()
                df_sub_with_totals['Summe'] = df_sub_with_totals.sum(axis=1)
                df_sub_with_totals.loc['Summe'] = df_sub_with_totals.sum()

                for r_idx, row in enumerate(dataframe_to_rows(df_sub_with_totals, index=True), start_row):
                    for c_idx, value in enumerate(row, 1):
                        cell = worksheet.cell(row=r_idx, column=c_idx, value=value)
                        if r_idx == len(df_sub_with_totals) + start_row - 1 or c_idx == len(row):
                            cell.font = Font(bold=True)
                            cell.fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')

                data_range = f"B{start_row+1}:{get_column_letter(len(df_sub_pivot.columns)+1)}{start_row+len(df_sub_pivot)}"
                self._add_color_scale(worksheet, data_range)

            # 3. Attribut 1 Analyse
            start_row += (len(df_sub_with_totals) if 'df_sub_with_totals' in locals() else 0) + 3
            worksheet.cell(row=start_row, column=1, value=f"3. Analyse nach {attribut1_label}")
            start_row += 1

            # Randsummen für Attribut 1
            df_attr1_with_totals = df_pivot_attr1.copy()
            df_attr1_with_totals.loc['Summe'] = df_attr1_with_totals.sum()

            for r_idx, row in enumerate(dataframe_to_rows(df_attr1_with_totals, index=True), start_row):
                for c_idx, value in enumerate(row, 1):
                    cell = worksheet.cell(row=r_idx, column=c_idx, value=value)
                    if r_idx == len(df_attr1_with_totals) + start_row - 1:
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')

            # 4. Attribut 2 Analyse
            start_row += len(df_attr1_with_totals) + 3
            worksheet.cell(row=start_row, column=1, value=f"4. Analyse nach {attribut2_label}")
            start_row += 1

            # Randsummen für Attribut 2
            df_attr2_with_totals = df_pivot_attr2.copy()
            df_attr2_with_totals.loc['Summe'] = df_attr2_with_totals.sum()

            for r_idx, row in enumerate(dataframe_to_rows(df_attr2_with_totals, index=True), start_row):
                for c_idx, value in enumerate(row, 1):
                    cell = worksheet.cell(row=r_idx, column=c_idx, value=value)
                    if r_idx == len(df_attr2_with_totals) + start_row - 1:
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')

            # 5. Kreuztabelle der Attribute
            start_row += len(df_attr2_with_totals) + 3
            worksheet.cell(row=start_row, column=1, value="5. Kreuztabelle der Attribute")
            start_row += 1

            # Erstelle Kreuztabelle
            df_cross = pd.crosstab(df_coded[attribut1_label], df_coded[attribut2_label])
            
            # Randsummen für Kreuztabelle
            df_cross_with_totals = df_cross.copy()
            df_cross_with_totals['Summe'] = df_cross_with_totals.sum(axis=1)
            df_cross_with_totals.loc['Summe'] = df_cross_with_totals.sum()

            for r_idx, row in enumerate(dataframe_to_rows(df_cross_with_totals, index=True), start_row):
                for c_idx, value in enumerate(row, 1):
                    cell = worksheet.cell(row=r_idx, column=c_idx, value=value)
                    if r_idx == len(df_cross_with_totals) + start_row - 1 or c_idx == len(row):
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')

            data_range = f"B{start_row+1}:{get_column_letter(len(df_cross.columns)+1)}{start_row+len(df_cross)}"
            self._add_color_scale(worksheet, data_range)

            # Allgemeine Formatierung
            self._format_frequency_worksheet(worksheet)

        except Exception as e:
            print(f"Fehler beim Export der Häufigkeitsanalysen: {str(e)}")
            import traceback
            traceback.print_exc()
    def _add_color_scale(self, worksheet, range_string: str) -> None:
        """Fügt Farbskala für einen Bereich hinzu."""
        color_scale_rule = ColorScaleRule(
            start_type='min',
            start_color='FFFFFF',
            mid_type='percentile',
            mid_value=50,
            mid_color='FFD966',
            end_type='max',
            end_color='FF8C00'
        )
        worksheet.conditional_formatting.add(range_string, color_scale_rule)

    def _format_frequency_worksheet(self, worksheet) -> None:
        """Formatiert das Häufigkeitsanalysen-Worksheet."""
        try:
            # Definiere Stile
            header_font = Font(bold=True)
            centered_alignment = Alignment(horizontal='center', vertical='center')
            border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )

            # Setze Spaltenbreiten
            worksheet.column_dimensions['A'].width = 40  # Kategorienamen
            for col in range(1, worksheet.max_column + 1):
                col_letter = get_column_letter(col)
                if col > 1:
                    worksheet.column_dimensions[col_letter].width = 15

            # Formatiere Überschriften und Zellen
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.border = border
                    
                    # Formatiere Überschriften
                    if cell.row == 1 or cell.column == 1:
                        cell.font = header_font
                    
                    # Zentriere Zahlenwerte
                    if isinstance(cell.value, (int, float)):
                        cell.alignment = centered_alignment

        except Exception as e:
            print(f"Warnung: Formatierung des Häufigkeitsanalysen-Worksheets fehlgeschlagen: {str(e)}")

    def _export_reliability_report(self, writer, reliability: float, total_segments: int, 
                                   total_coders: int, category_frequencies: dict):
        """
        Exportiert den Reliability Report als zusätzliches Excel-Sheet,
        formatiert wie in der _document_reliability_results() Methode.
        """
        if 'Reliability Report' not in writer.sheets:
            writer.book.create_sheet('Reliability Report')
    
        worksheet = writer.sheets['Reliability Report']
        current_row = 1

        # Generiere den Report-Inhalt
        if self.inductive_coder:
            report_content = self.inductive_coder._document_reliability_results(
                alpha=reliability,
                total_segments=total_segments,
                total_coders=total_coders,
                category_frequencies=category_frequencies
            )
        else:
            # Fallback, falls kein inductive_coder verfügbar ist
            report_content = self._generate_fallback_reliability_report(
                reliability, total_segments, total_coders, category_frequencies
            )
        
        # Füge den Inhalt zum Worksheet hinzu
        for line in report_content.split('\n'):
            if line.startswith('# '):
                worksheet.cell(row=current_row, column=1, value=line[2:])
                current_row += 2  # Zusätzliche Leerzeile nach dem Titel
            elif line.startswith('## '):
                worksheet.cell(row=current_row, column=1, value=line[3:])
                current_row += 1
            elif line.startswith('- '):
                key, value = line[2:].split(': ', 1)
                worksheet.cell(row=current_row, column=1, value=key)
                worksheet.cell(row=current_row, column=2, value=value)
                current_row += 1
            elif '|' in line:  # Tabelle
                if '---' not in line:  # Überspringen der Trennzeile
                    cells = line.split('|')
                    for col, cell in enumerate(cells[1:-1], start=1):  # Erste und letzte Zelle ignorieren
                        worksheet.cell(row=current_row, column=col, value=cell.strip())
                    current_row += 1
            elif line.strip():  # Alle anderen nicht-leeren Zeilen
                worksheet.cell(row=current_row, column=1, value=line.strip())
                current_row += 1
            else:  # Leerzeile
                current_row += 1

        # Formatierung
        self._format_reliability_worksheet(worksheet)


    def _format_reliability_worksheet(self, worksheet):
        """
        Formatiert das Reliability Report Worksheet.
        """
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

        # Definiere Stile
        title_font = Font(bold=True, size=14)
        header_font = Font(bold=True, size=12)
        normal_font = Font(size=11)
        border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))

        # Setze Spaltenbreiten
        worksheet.column_dimensions['A'].width = 40
        worksheet.column_dimensions['B'].width = 20

        # Formatiere Zellen
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical='top')
                cell.font = normal_font
                if cell.row == 1:  # Titel
                    cell.font = title_font
                elif cell.column == 1 and cell.value and ':' not in str(cell.value):  # Überschriften
                    cell.font = header_font
                
                # Füge Rahmen zu Tabellenzellen hinzu
                if '|' in str(worksheet.cell(row=1, column=cell.column).value):
                    cell.border = border

        # Zentriere Tabellenkopf
        for cell in worksheet[2]:
            if cell.value:
                cell.alignment = Alignment(horizontal='center', vertical='center')

        # Füge einen schmalen Rahmen um den gesamten Bericht hinzu
        thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
        for row in worksheet.iter_rows():
            for cell in row:
                if cell.value:
                    cell.border = thin_border

    
    async def export_results(
        self,
        codings: List[Dict],
        reliability: float,
        categories: Dict[str, CategoryDefinition],
        chunks: Dict[str, List[str]],
        revision_manager: 'CategoryRevisionManager',
        export_mode: str = "all",
        original_categories: Dict[str, CategoryDefinition] = None,
        merge_log: List[Dict] = None,
        inductive_coder: 'InductiveCoder' = None 
        ) -> None:
        """
        Exportiert die Analyseergebnisse in eine Excel-Datei mit verschiedenen Auswertungen.
        """

        # Wenn inductive_coder als Parameter übergeben wurde, aktualisieren Sie das Attribut
        if inductive_coder:
            self.inductive_coder = inductive_coder

        try:
            # Erstelle Zeitstempel für den Dateinamen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"QCA-AID_Analysis_{timestamp}.xlsx"
            filepath = os.path.join(self.output_dir, filename)
            
            # Hole die Bezeichnungen für die Attribute
            attribut1_label = self.attribute_labels['attribut1']
            attribut2_label = self.attribute_labels['attribut2']

            # Berechne benötigte Statistiken
            total_segments = len(codings)
            coders = list(set(c['coder_id'] for c in codings))
            total_coders = len(coders)
            category_frequencies = Counter(c['category'] for c in codings if 'category' in c)
            
            if original_categories and merge_log:
                self.export_merge_analysis(original_categories, categories, merge_log)

            # Stelle sicher, dass codings eine Liste ist
            if not isinstance(codings, list):
                codings = list(codings)

            # Kombiniere deduktive und induktive Kategorien
            all_categories = {**DEDUKTIVE_KATEGORIEN, **categories}
            
            # Sammle alle Kodierungen (deduktiv und induktiv)
            enriched_codings = []
            for coding in codings:
                filename = coding['segment_id'].split('_chunk_')[0]
                attribut1, attribut2 = self._extract_metadata(filename)
                
                # Prüfe ob die Kategorie deduktiv oder induktiv ist
                category = coding['category']
                if category in DEDUKTIVE_KATEGORIEN:
                    category_type = 'deduktiv'
                else:
                    category_type = 'induktiv'
                
                enriched_coding = coding.copy()
                enriched_coding.update({
                    attribut1_label: attribut1,
                    attribut2_label: attribut2,
                    'Datei': filename,
                    'Kategorietyp': category_type  # Füge Information über Kategorietyp hinzu
                })
                enriched_codings.append(enriched_coding)

            # Gruppierung nach Segmenten für Konsensus-Entscheidungen
            segment_codings = defaultdict(list)
            for coding in enriched_codings:
                segment_codings[coding['segment_id']].append(coding)

            # Erstelle die Revisionshistorie als DataFrame
            revisions_data = []
            for change in revision_manager.changes:
                revisions_data.append({
                    'Datum': datetime.fromisoformat(change.timestamp).strftime('%Y-%m-%d %H:%M'),
                    'Kategorie': change.category_name,
                    'Art der Änderung': change.change_type,
                    'Beschreibung': change.description,
                    'Begründung': change.justification,
                    'Betroffene Kodierungen': ', '.join(change.affected_codings) if change.affected_codings else ''
                })
            df_revisions = pd.DataFrame(revisions_data)

            # Erstelle DataFrame für detaillierte Ergebnisse
            export_data = []
            for segment_id, segment_codes in segment_codings.items():
                chunk_text = chunks[segment_id.split('_chunk_')[0]][int(segment_id.split('_chunk_')[1])]
                chunk_id = int(segment_id.split('_chunk_')[1])
                doc_name = segment_id.split('_chunk_')[0]

                if export_mode == "all":
                    for coding in segment_codes:
                        export_data.append(self._prepare_coding_for_export(coding, chunk_text, chunk_id, doc_name))
                else:
                    if export_mode == "consensus":
                        final_coding = self._get_consensus_coding(segment_codes)
                    elif export_mode == "majority":
                        final_coding = self._get_majority_coding(segment_codes)
                    elif export_mode == "manual":
                        final_coding = self._get_manual_priority_coding(segment_codes)
                    
                    if final_coding:
                        export_data.append(self._prepare_coding_for_export(final_coding, chunk_text, chunk_id, doc_name))
                
            # Erstelle die verschiedenen DataFrames für die Analyse
            df_details = pd.DataFrame(export_data)
            df_coded = df_details[df_details['Kodiert'] == 'Ja']

            # Erstelle verschiedene Analysen
            # 1. Hauptkategorien nach Attributen
            df_pivot_main = pd.pivot_table(
                df_coded,
                index=['Hauptkategorie'],
                columns=[attribut1_label, attribut2_label],
                values='Chunk_Nr',
                aggfunc='count',
                fill_value=0
            )

            # 2. Subkategorien-Analyse
            subcats_split = []
            for _, row in df_coded.iterrows():
                subcats = row['Subkategorien'].split(', ') if row['Subkategorien'] else []
                for subcat in subcats:
                    if subcat:
                        new_row = row.copy()
                        new_row['Subkategorie'] = subcat.strip()
                        subcats_split.append(new_row)
            
            df_subcats = pd.DataFrame(subcats_split)

            # 3. Häufigkeit der Attribute
            df_pivot_attr1 = pd.pivot_table(
                df_coded,
                index=[attribut1_label],
                values='Chunk_Nr',
                aggfunc='count',
                fill_value=0
            ).rename(columns={'Chunk_Nr': 'Anzahl Kodierungen'})

            df_pivot_attr2 = pd.pivot_table(
                df_coded,
                index=[attribut2_label],
                values='Chunk_Nr',
                aggfunc='count',
                fill_value=0
            ).rename(columns={'Chunk_Nr': 'Anzahl Kodierungen'})

            # Exportiere alle Ergebnisse nach Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Arbeitsblatt 1: Detaillierte Kodierungen
                df_details.to_excel(writer, sheet_name='Kodierte_Segmente', index=False)
                self._format_worksheet(writer.sheets['Kodierte_Segmente'])

                # Arbeitsblatt 2: Häufigkeitsanalysen
                self._export_frequency_analysis(
                    writer=writer,
                    df_coded=df_coded,
                    df_pivot_main=df_pivot_main,
                    df_pivot_subcats=df_subcats,
                    df_pivot_attr1=df_pivot_attr1,
                    df_pivot_attr2=df_pivot_attr2,
                    attribut1_label=attribut1_label,
                    attribut2_label=attribut2_label
                )

                # Arbeitsblatt 3: Revisionshistorie
                if not df_revisions.empty:
                    df_revisions.to_excel(writer, sheet_name='Revisionshistorie', index=False)
                    self._format_revision_worksheet(writer.sheets['Revisionshistorie'])

                # Arbeitsblatt 4: Intercoder-Analyse
                self._export_intercoder_analysis(writer, segment_codings, reliability)

                # Arbeitsblatt 5: Reliabilitätsanalyse
                if inductive_coder:
                    self._export_reliability_report(
                        writer, 
                        reliability,
                        total_segments,
                        total_coders,
                        category_frequencies
                    )

            print(f"\nErgebnisse erfolgreich exportiert nach: {filepath}")

        except Exception as e:
            print(f"Fehler beim Excel-Export: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _format_worksheet(self, worksheet) -> None:
        """Formatiert das Detail-Worksheet"""
        try:
            column_widths = {
                'A': 30,  # Dokument
                'B': 15,  # Attribut1
                'C': 15,  # Attribut2
                'D': 5,   # Chunk_Nr
                'E': 30,  # Text
                'F': 5,   # Kodiert
                'G': 20,  # Hauptkategorie
                'H': 15,  # Kategorietyp
                'I': 40,  # Subkategorien
                'J': 40,  # Begründung
                'K': 15,  # Konfidenz
                'L': 15   # Mehrfachkodierung
            }
            
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width
        except Exception as e:
            print(f"Warnung: Formatierung fehlgeschlagen: {str(e)}")

    def _format_revision_worksheet(self, worksheet) -> None:
        """Formatiert das Revisions-Worksheet"""
        try:
            column_widths = {
                'A': 20,  # Datum
                'B': 25,  # Kategorie
                'C': 15,  # Art der Änderung
                'D': 50,  # Beschreibung
                'E': 50,  # Begründung
                'F': 40   # Betroffene Kodierungen
            }
            
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width
                
            # Überschriften formatieren
            from openpyxl.styles import Font, PatternFill
            header_font = Font(bold=True)
            header_fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')
            
            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                
        except Exception as e:
            print(f"Warnung: Revisions-Formatierung fehlgeschlagen: {str(e)}")
    
    def _export_intercoder_analysis(self, writer, segment_codings: Dict[str, List[Dict]], reliability: float):
        """
        Exportiert die Intercoder-Analyse für Haupt- und Subkategorien.
        
        Args:
            writer: Excel Writer Objekt
            segment_codings: Dictionary mit Kodierungen pro Segment
            reliability: Berechnete Reliabilität
        """
        try:
            if 'Intercoderanalyse' not in writer.sheets:
                writer.book.create_sheet('Intercoderanalyse')
            
            worksheet = writer.sheets['Intercoderanalyse']
            current_row = 1

            # 1. Überschrift und Gesamtreliabilität
            worksheet.cell(row=current_row, column=1, value="Intercoderanalyse")
            current_row += 2
            worksheet.cell(row=current_row, column=1, value="Krippendorffs Alpha (Hauptkategorien):")
            worksheet.cell(row=current_row, column=2, value=round(reliability, 3))
            current_row += 2

            # 2. Separate Analyse für Haupt- und Subkategorien
            worksheet.cell(row=current_row, column=1, value="Übereinstimmungsanalyse")
            current_row += 2

            # 2.1 Hauptkategorien-Analyse
            worksheet.cell(row=current_row, column=1, value="A. Hauptkategorien")
            current_row += 1

            headers = [
                'Segment_ID', 
                'Text',
                'Anzahl Codierer', 
                'Übereinstimmungsgrad', 
                'Hauptkategorien', 
                'Begründungen'
            ]
            for col, header in enumerate(headers, 1):
                worksheet.cell(row=current_row, column=col, value=header)
            current_row += 1

            # Analyse für Hauptkategorien
            for segment_id, codings in segment_codings.items():
                categories = [c['category'] for c in codings]
                category_agreement = len(set(categories)) == 1
                
                # Verbesserte Text-Extraktion
                text_chunk = codings[0].get('text', '')
                if text_chunk:
                    text_chunk = text_chunk[:200] + ("..." if len(text_chunk) > 200 else "")
                else:
                    text_chunk = "Text nicht verfügbar"
                
                row_data = [
                    segment_id,
                    text_chunk,
                    len(codings),
                    "Vollständig" if category_agreement else "Keine Übereinstimmung",
                    ' | '.join(set(categories)),
                    '\n'.join([c.get('justification', '')[:100] + '...' for c in codings])
                ]

                for col, value in enumerate(row_data, 1):
                    worksheet.cell(row=current_row, column=col, value=value)
                current_row += 1

            # 2.2 Subkategorien-Analyse
            current_row += 2
            worksheet.cell(row=current_row, column=1, value="B. Subkategorien")
            current_row += 1

            headers = [
                'Segment_ID', 
                'Text',
                'Anzahl Codierer', 
                'Übereinstimmungsgrad', 
                'Subkategorien', 
                'Begründungen'
            ]
            for col, header in enumerate(headers, 1):
                worksheet.cell(row=current_row, column=col, value=header)
            current_row += 1

            # Analyse für Subkategorien
            for segment_id, codings in segment_codings.items():
                subcategories = [set(c.get('subcategories', [])) for c in codings]
                
                # Verbesserte Text-Extraktion
                text_chunk = codings[0].get('text', '')
                if text_chunk:
                    text_chunk = text_chunk[:200] + ("..." if len(text_chunk) > 200 else "")
                else:
                    text_chunk = "Text nicht verfügbar"
                
                # Berechne Übereinstimmungsgrad für Subkategorien
                if subcategories:
                    all_equal = all(s == subcategories[0] for s in subcategories)
                    partial_overlap = any(s1 & s2 for s1, s2 in itertools.combinations(subcategories, 2))
                    
                    if all_equal:
                        agreement = "Vollständig"
                    elif partial_overlap:
                        agreement = "Teilweise"
                    else:
                        agreement = "Keine Übereinstimmung"
                else:
                    agreement = "Keine Subkategorien"

                row_data = [
                    segment_id,
                    text_chunk,
                    len(codings),
                    agreement,
                    ' | '.join(set.union(*subcategories) if subcategories else set()),
                    '\n'.join([c.get('justification', '')[:100] + '...' for c in codings])
                ]

                for col, value in enumerate(row_data, 1):
                    worksheet.cell(row=current_row, column=col, value=value)
                current_row += 1

            # 3. Übereinstimmungsmatrix für Codierer
            current_row += 2
            worksheet.cell(row=current_row, column=1, value="C. Codierer-Vergleichsmatrix")
            current_row += 1

            # Erstelle separate Matrizen für Haupt- und Subkategorien
            for analysis_type in ['Hauptkategorien', 'Subkategorien']:
                current_row += 1
                worksheet.cell(row=current_row, column=1, value=f"Matrix für {analysis_type}")
                current_row += 1

                coders = sorted(list({coding['coder_id'] for codings in segment_codings.values() for coding in codings}))
                
                # Schreibe Spaltenüberschriften
                for col, coder in enumerate(coders, 2):
                    worksheet.cell(row=current_row, column=col, value=coder)
                current_row += 1

                # Fülle Matrix
                for coder1 in coders:
                    worksheet.cell(row=current_row, column=1, value=coder1)
                    
                    for col, coder2 in enumerate(coders, 2):
                        if coder1 == coder2:
                            agreement = 1.0
                        else:
                            # Berechne Übereinstimmung
                            agreements = 0
                            total = 0
                            for codings in segment_codings.values():
                                coding1 = next((c for c in codings if c['coder_id'] == coder1), None)
                                coding2 = next((c for c in codings if c['coder_id'] == coder2), None)
                                
                                if coding1 and coding2:
                                    total += 1
                                    if analysis_type == 'Hauptkategorien':
                                        if coding1['category'] == coding2['category']:
                                            agreements += 1
                                    else:  # Subkategorien
                                        subcats1 = set(coding1.get('subcategories', []))
                                        subcats2 = set(coding2.get('subcategories', []))
                                        if subcats1 == subcats2:
                                            agreements += 1
                            
                            agreement = agreements / total if total > 0 else 0
                        
                        cell = worksheet.cell(row=current_row, column=col, value=round(agreement, 2))
                        # Formatiere Zelle als Prozentsatz
                        cell.number_format = '0%'
                    
                    current_row += 1
                current_row += 2

            # 4. Zusammenfassende Statistiken
            current_row += 2
            worksheet.cell(row=current_row, column=1, value="D. Zusammenfassende Statistiken")
            current_row += 1

            # Berechne Statistiken
            total_segments = len(segment_codings)
            total_coders = len({coding['coder_id'] for codings in segment_codings.values() for coding in codings})
            
            stats = [
                ('Anzahl analysierter Segmente', total_segments),
                ('Anzahl Codierer', total_coders),
                ('Durchschnittliche Übereinstimmung Hauptkategorien', reliability),
            ]

            for stat_name, stat_value in stats:
                worksheet.cell(row=current_row, column=1, value=stat_name)
                cell = worksheet.cell(row=current_row, column=2, value=stat_value)
                if isinstance(stat_value, float):
                    cell.number_format = '0.00%'
                current_row += 1

            # Formatierung
            self._format_intercoder_worksheet(worksheet)

        except Exception as e:
            print(f"Fehler beim Export der Intercoder-Analyse: {str(e)}")
            import traceback
            traceback.print_exc()

    def _format_intercoder_worksheet(self, worksheet) -> None:
        """
        Formatiert das Intercoder-Worksheet für bessere Lesbarkeit.
        
        Args:
            worksheet: Das zu formatierende Worksheet-Objekt
        """
        try:
            # Importiere Styling-Klassen
            from openpyxl.styles import Font, PatternFill, Alignment
            
            # Definiere Spaltenbreiten
            column_widths = {
                'A': 40,  # Segment_ID
                'B': 60,  # Text
                'C': 15,  # Anzahl Codierer
                'D': 20,  # Übereinstimmungsgrad
                'E': 40,  # Haupt-/Subkategorien
                'F': 60   # Begründungen
            }
            
            # Setze Spaltenbreiten
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width

            # Definiere Stile
            header_font = Font(bold=True)
            header_fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')
            wrapped_alignment = Alignment(wrap_text=True, vertical='top')
            
            # Formatiere alle Zellen
            for row in worksheet.iter_rows():
                for cell in row:
                    # Grundformatierung für alle Zellen
                    cell.alignment = wrapped_alignment
                    
                    # Spezielle Formatierung für Überschriften
                    if (cell.row == 1 or  # Hauptüberschrift
                        (cell.value and isinstance(cell.value, str) and 
                         (cell.value.startswith(('A.', 'B.', 'C.', 'D.')) or  # Abschnittsüberschriften
                          cell.value in ['Segment_ID', 'Text', 'Anzahl Codierer', 'Übereinstimmungsgrad', 
                                       'Hauptkategorien', 'Subkategorien', 'Begründungen']))):  # Spaltenüberschriften
                        cell.font = header_font
                        cell.fill = header_fill

            # Zusätzliche Formatierung für die Übereinstimmungsmatrix
            matrix_start = None
            for row_idx, row in enumerate(worksheet.iter_rows(), 1):
                for cell in row:
                    if cell.value == "C. Codierer-Vergleichsmatrix":
                        matrix_start = row_idx
                        break
                if matrix_start:
                    break

            if matrix_start:
                for row in worksheet.iter_rows(min_row=matrix_start):
                    for cell in row:
                        if isinstance(cell.value, (int, float)):
                            cell.number_format = '0.00%'

        except Exception as e:
            print(f"Warnung: Formatierung des Intercoder-Worksheets fehlgeschlagen: {str(e)}")
            import traceback
            traceback.print_exc()


# --- Klasse: CategoryRevisionManager ---
# Aufgabe: Verwaltung der iterativen Kategorienrevision
class CategoryRevisionManager:
    """
    Manages the iterative refinement of categories during qualitative content analysis.
    
    This class implements Mayring's approach to category revision, ensuring systematic
    documentation of changes, validation of the category system, and maintenance of
    methodological quality standards throughout the analysis process.
    """
    
    def __init__(self, output_dir: str):
        """
        Initializes the CategoryRevisionManager with necessary tracking mechanisms.
        
        Args:
            output_dir (str): Directory for storing revision documentation
        """
        self.output_dir = output_dir
        self.changes: List[CategoryChange] = []
        self.revision_log_path = os.path.join(output_dir, "category_revisions.json")
        
        # Define quality thresholds for category system
        self.SIMILARITY_THRESHOLD = 0.7
        self.MIN_EXAMPLES_PER_CATEGORY = 3
        self.MIN_DEFINITION_WORDS = 15
        
        # Load existing revision history if available
        self._load_revision_history()

    def revise_category_system(self, 
                             categories: Dict[str, CategoryDefinition],
                             coded_segments: List[CodingResult],
                             material_percentage: float) -> Dict[str, CategoryDefinition]:
        """
        Implements a comprehensive category revision process following Mayring's methodology.
        
        This method systematically reviews and refines the category system based on:
        - Category distinctiveness
        - Definition clarity
        - Coding reliability
        - Category saturation
        - Abstraction level appropriateness
        
        Args:
            categories: Current category system
            coded_segments: Previously coded text segments
            material_percentage: Percentage of material processed
            
        Returns:
            Dict[str, CategoryDefinition]: Revised category system
        """
        try:
            print(f"\nStarting category system revision at {material_percentage:.1f}% of material")
            revised_categories = categories.copy()
            
            # Step 1: Analyze current category usage and effectiveness
            category_statistics = self._analyze_category_usage(
                categories,
                coded_segments
            )
            
            # Step 2: Check for problematic categories
            problematic_categories = self._identify_problematic_categories(
                category_statistics,
                revised_categories
            )
            
            if problematic_categories:
                print(f"\nFound {len(problematic_categories)} categories requiring attention:")
                for cat_name, issues in problematic_categories.items():
                    print(f"- {cat_name}: {', '.join(issues)}")
            
            # Step 3: Apply necessary revisions
            for cat_name, issues in problematic_categories.items():
                revised_categories = self._apply_category_revisions(
                    cat_name,
                    issues,
                    revised_categories,
                    category_statistics
                )
            
            # Step 4: Check category system coherence
            if not self._validate_category_system(revised_categories):
                print("Warning: Category system validation revealed issues")
                self._document_validation_issues(revised_categories)
            
            # Step 5: Update revision history
            self._document_revision(
                original=categories,
                revised=revised_categories,
                material_percentage=material_percentage
            )
            
            return revised_categories
            
        except Exception as e:
            print(f"Error during category revision: {str(e)}")
            import traceback
            traceback.print_exc()
            return categories

    def _analyze_category_usage(self, 
                              categories: Dict[str, CategoryDefinition],
                              coded_segments: List[CodingResult]) -> Dict[str, dict]:
        """
        Analyzes how categories are being used in the coding process.
        
        Creates detailed statistics about category usage, including:
        - Frequency of application
        - Consistency of usage
        - Overlap with other categories
        - Quality of coding justifications
        
        Args:
            categories: Current category system
            coded_segments: Previously coded segments
            
        Returns:
            Dict[str, dict]: Statistics for each category
        """
        statistics = {}
        
        for category_name in categories:
            # Initialize statistics for this category
            statistics[category_name] = {
                'frequency': 0,
                'coding_confidence': [],
                'overlapping_categories': set(),
                'justification_quality': [],
                'example_segments': []
            }
        
        # Analyze each coded segment
        for coding in coded_segments:
            if coding.category in statistics:
                stats = statistics[coding.category]
                
                # Update frequency
                stats['frequency'] += 1
                
                # Track confidence scores
                stats['coding_confidence'].append(
                    coding.confidence.get('total', 0)
                )
                
                # Store justification
                stats['justification_quality'].append(
                    len(coding.justification.split())
                )
                
                # Track example segments
                stats['example_segments'].append(
                    coding.text_references[0] if coding.text_references else ''
                )
                
                # Note overlapping categories
                if len(coding.subcategories) > 0:
                    stats['overlapping_categories'].update(coding.subcategories)
        
        # Calculate aggregate statistics
        for category_name, stats in statistics.items():
            if stats['frequency'] > 0:
                stats['avg_confidence'] = sum(stats['coding_confidence']) / len(stats['coding_confidence'])
                stats['avg_justification_length'] = sum(stats['justification_quality']) / len(stats['justification_quality'])
            else:
                stats['avg_confidence'] = 0
                stats['avg_justification_length'] = 0
        
        return statistics

    def validate_category_changes(self, categories: Dict[str, CategoryDefinition]) -> bool:
        """
        Validates the methodological integrity of categories and their changes according to Mayring's criteria.
        
        This method implements a comprehensive validation process that checks:
        1. Category completeness and definition quality
        2. Mutual exclusivity between categories
        3. Logical consistency of the revision history
        4. Proper documentation of changes
        
        Args:
            categories: Current category system to validate
            
        Returns:
            bool: True if the category system meets all quality criteria
        """
        try:
            print("\nValidating category system...")
            validation_issues = []

            # 1. Check basic category requirements
            for name, category in categories.items():
                # Validate definition completeness
                if len(category.definition.split()) < self.MIN_DEFINITION_WORDS:
                    validation_issues.append(
                        f"Category '{name}' has insufficient definition length"
                    )

                # Validate example presence
                if len(category.examples) < self.MIN_EXAMPLES_PER_CATEGORY:
                    validation_issues.append(
                        f"Category '{name}' needs more examples (has {len(category.examples)}, needs {self.MIN_EXAMPLES_PER_CATEGORY})"
                    )

                # Validate coding rules
                if not category.rules:
                    validation_issues.append(
                        f"Category '{name}' lacks coding rules"
                    )

            # 2. Check for category overlap
            for cat1_name, cat1 in categories.items():
                for cat2_name, cat2 in categories.items():
                    if cat1_name >= cat2_name:  # Skip self-comparison and duplicates
                        continue
                        
                    similarity = self._calculate_text_similarity(
                        cat1.definition,
                        cat2.definition
                    )
                    if similarity > self.SIMILARITY_THRESHOLD:
                        validation_issues.append(
                            f"High similarity ({similarity:.2f}) between '{cat1_name}' and '{cat2_name}'"
                        )

            # 3. Validate revision history consistency
            if self.changes:
                # Check chronological order
                timestamps = [change.timestamp for change in self.changes]
                if timestamps != sorted(timestamps):
                    validation_issues.append(
                        "Revision history is not in chronological order"
                    )

                # Track category lifecycle
                active_categories = set()
                for change in self.changes:
                    if change.change_type == 'add':
                        if change.category_name in active_categories:
                            validation_issues.append(
                                f"Category '{change.category_name}' added multiple times"
                            )
                        active_categories.add(change.category_name)
                    elif change.change_type == 'delete':
                        if change.category_name not in active_categories:
                            validation_issues.append(
                                f"Attempt to delete non-existent category '{change.category_name}'"
                            )
                        active_categories.remove(change.category_name)

                # Verify all current categories have proper history
                for category_name in categories:
                    if category_name not in active_categories:
                        validation_issues.append(
                            f"Category '{category_name}' lacks proper revision history"
                        )

            # Report validation results
            if validation_issues:
                print("\nCategory system validation found issues:")
                for issue in validation_issues:
                    print(f"- {issue}")
                return False

            print("Category system validation successful")
            return True

        except Exception as e:
            print(f"Error during category validation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates the similarity between two text strings using a simple but effective approach.
        
        This method implements a combination of:
        1. Word overlap ratio
        2. Character-based similarity
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Convert texts to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity for words
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union

    def _identify_problematic_categories(self,
                                       statistics: Dict[str, dict],
                                       categories: Dict[str, CategoryDefinition]) -> Dict[str, List[str]]:
        """
        Identifies categories that need revision based on usage statistics.
        
        Args:
            statistics: Usage statistics for each category
            categories: Current category system
            
        Returns:
            Dict[str, List[str]]: Categories and their identified issues
        """
        problems = {}
        
        for cat_name, stats in statistics.items():
            category = categories.get(cat_name)
            if not category:
                continue
                
            issues = []
            
            # Check usage frequency
            if stats['frequency'] < self.MIN_EXAMPLES_PER_CATEGORY:
                issues.append("insufficient_usage")
            
            # Check coding confidence
            if stats.get('avg_confidence', 0) < 0.7:
                issues.append("low_confidence")
            
            # Check definition clarity
            if len(category.definition.split()) < self.MIN_DEFINITION_WORDS:
                issues.append("unclear_definition")
            
            # Check for overlapping categories
            if len(stats['overlapping_categories']) > len(categories) * 0.3:
                issues.append("high_overlap")
            
            # Check justification quality
            if stats.get('avg_justification_length', 0) < 10:
                issues.append("poor_justifications")
            
            if issues:
                problems[cat_name] = issues
        
        return problems

    def _apply_category_revisions(self,
                                category_name: str,
                                issues: List[str],
                                categories: Dict[str, CategoryDefinition],
                                statistics: Dict[str, dict]) -> Dict[str, CategoryDefinition]:
        """
        Applies appropriate revisions to problematic categories.
        
        Args:
            category_name: Name of the category to revise
            issues: List of identified issues
            categories: Current category system
            statistics: Usage statistics for categories
            
        Returns:
            Dict[str, CategoryDefinition]: Updated category system
        """
        revised = categories.copy()
        category = revised[category_name]
        
        for issue in issues:
            if issue == "insufficient_usage":
                # Consider merging with similar category
                similar_category = self._find_similar_category(
                    category,
                    revised,
                    statistics
                )
                if similar_category:
                    revised = self._merge_categories(
                        category_name,
                        similar_category,
                        revised
                    )
                    
            elif issue == "unclear_definition":
                # Enhance category definition
                enhanced_definition = self._enhance_category_definition(
                    category,
                    statistics[category_name]['example_segments']
                )
                if enhanced_definition:
                    revised[category_name] = category._replace(
                        definition=enhanced_definition
                    )
                    
            elif issue == "high_overlap":
                # Clarify category boundaries
                revised = self._clarify_category_boundaries(
                    category_name,
                    statistics[category_name]['overlapping_categories'],
                    revised
                )
        
        return revised

    def _validate_category(self, category: Dict) -> bool:
        """
        Validates the completeness of a category.

        Args:
            category: Category dictionary

        Returns:
            bool: True if all required fields are present
        """
        required_fields = {
            'name': str,
            'definition': str,
            'example': str,
            'existing_subcategories': list,
            'new_subcategories': list,
            'justification': str,
            'confidence': dict
        }
        
        try:
            for field, field_type in required_fields.items():
                if field not in category:
                    print(f"Warnung: Fehlendes Feld '{field}' in Kategorie")
                    return False
                if not isinstance(category[field], field_type):
                    print(f"Warnung: Falscher Datentyp für Feld '{field}' in Kategorie")
                    return False
                    
            # Spezielle Prüfung für confidence
            if not all(k in category['confidence'] for k in ['category', 'subcategories']):
                print("Warnung: Unvollständige confidence-Werte in Kategorie")
                return False
                
            return True
            
        except Exception as e:
            print(f"Fehler bei der Kategorievalidierung: {str(e)}")
            return False

    def _validate_categories(self, categories: List[dict]) -> List[dict]:
        """
        Validates the extracted categories using Mayring's quality criteria.
        """
        valid_categories = []
        required_fields = {
            'name': str,
            'definition': str,
            'example': str,
            'existing_subcategories': list,
            'new_subcategories': list,
            'justification': str,
            'confidence': dict
        }
        
        for category in categories:
            try:
                # Zeige Kategorienamen in der Warnung
                category_name = category.get('name', 'UNBENANNTE KATEGORIE')
                
                # Prüfe ob alle erforderlichen Felder vorhanden sind
                missing_fields = [field for field in required_fields if field not in category]
                if missing_fields:
                    print(f"Warnung: Fehlende Felder in Kategorie '{category_name}': {', '.join(missing_fields)}")
                    continue
                
                # Prüfe Datentypen
                invalid_types = []
                for field, expected_type in required_fields.items():
                    if not isinstance(category[field], expected_type):
                        actual_type = type(category[field]).__name__
                        invalid_types.append(f"{field} (ist {actual_type}, erwartet {expected_type.__name__})")
                
                if invalid_types:
                    print(f"Warnung: Ungültige Datentypen in Kategorie '{category_name}': {', '.join(invalid_types)}")
                    continue
                
                # Prüfe ob Subkategorien vorhanden sind
                has_subcategories = (
                    len(category['existing_subcategories']) > 0 or 
                    len(category['new_subcategories']) > 0
                )
                if not has_subcategories:
                    print(f"Warnung: Keine Subkategorien für Kategorie '{category_name}'")
                    continue
                
                # Weitere Qualitätsprüfungen...
                valid_categories.append(category)
                
            except Exception as e:
                print(f"Fehler bei Kategorienvalidierung von '{category.get('name', 'UNBENANNTE KATEGORIE')}': {str(e)}")
                continue
        
        # Zusammenfassung
        if len(valid_categories) < len(categories):
            print(f"\nValidierungszusammenfassung:")
            print(f"- Eingereichte Kategorien: {len(categories)}")
            print(f"- Valide Kategorien: {len(valid_categories)}")
            print(f"- Zurückgewiesene Kategorien: {len(categories) - len(valid_categories)}")
        
        return valid_categories


    def _validate_category_system(self, categories: Dict[str, CategoryDefinition]) -> bool:
        """
        Validates the coherence and quality of the entire category system.
        
        Checks:
        - Mutual exclusiveness of categories
        - Completeness of the system
        - Consistency of abstraction levels
        - Quality of definitions and examples
        
        Args:
            categories: Category system to validate
            
        Returns:
            bool: True if system meets all quality criteria
        """
        validation_results = []
        
        # Check category definitions
        for category in categories.values():
            # Definition quality
            definition_valid = len(category.definition.split()) >= self.MIN_DEFINITION_WORDS
            validation_results.append(definition_valid)
            
            # Example quality
            examples_valid = len(category.examples) >= self.MIN_EXAMPLES_PER_CATEGORY
            validation_results.append(examples_valid)
            
            # Rules presence
            rules_valid = len(category.rules) >= 2
            validation_results.append(rules_valid)
        
        # Check category relationships
        for cat1, cat2 in itertools.combinations(categories.values(), 2):
            # Check for overlapping definitions
            distinctiveness = self._calculate_text_similarity(
                cat1.definition,
                cat2.definition
            ) < self.SIMILARITY_THRESHOLD
            validation_results.append(distinctiveness)
        
        return all(validation_results)

    def _load_revision_history(self) -> None:
        """
        Loads the existing revision history from the JSON file if available.
        This method is called during initialization to restore previous category changes.
        """
        try:
            if os.path.exists(self.revision_log_path):
                with open(self.revision_log_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    # Convert the loaded dictionary data back into CategoryChange objects
                    self.changes = [
                        CategoryChange(
                            category_name=change['category_name'],
                            change_type=change['change_type'],
                            description=change['description'],
                            timestamp=change['timestamp'],
                            old_value=change.get('old_value'),
                            new_value=change.get('new_value'),
                            affected_codings=change.get('affected_codings'),
                            justification=change.get('justification', '')
                        )
                        for change in history
                    ]
                print(f"Loaded revision history: {len(self.changes)} changes")
            else:
                print("No existing revision history found - starting fresh")
                self.changes = []
                
        except Exception as e:
            print(f"Error loading revision history: {str(e)}")
            print("Starting with empty revision history")
            self.changes = []

    def _document_revision(self,
                     original: Dict[str, CategoryDefinition],
                     revised: Dict[str, CategoryDefinition],
                     material_percentage: float) -> None:
        """
        Documents the changes to the category system in detail.

        Args:
            original: Original category system
            revised: Revised category system
            material_percentage: Percentage of material processed
        """
        timestamp = datetime.now().isoformat()
        
        # Track category changes
        for cat_name in set(list(original.keys()) + list(revised.keys())):
            if cat_name not in original:
                # Neue Kategorie
                self.changes.append(CategoryChange(
                    category_name=cat_name,
                    change_type='add',
                    description=f"Neue Kategorie hinzugefügt bei {material_percentage:.1f}% des Materials",
                    timestamp=timestamp,
                    new_value={
                        'definition': revised[cat_name].definition,
                        'examples': revised[cat_name].examples,
                        'rules': revised[cat_name].rules,
                        'subcategories': revised[cat_name].subcategories
                    },
                    justification="Kategorie aus Datenanalyse entwickelt"
                ))
            elif cat_name not in revised:
                # Gelöschte Kategorie
                self.changes.append(CategoryChange(
                    category_name=cat_name,
                    change_type='delete',
                    description=f"Kategorie entfernt bei {material_percentage:.1f}% des Materials",
                    timestamp=timestamp,
                    old_value={
                        'definition': original[cat_name].definition,
                        'examples': original[cat_name].examples,
                        'rules': original[cat_name].rules,
                        'subcategories': original[cat_name].subcategories
                    },
                    justification="Kategorie erwies sich als unzureichend"
                ))
            elif original[cat_name] != revised[cat_name]:
                # Modifizierte Kategorie - detaillierter Vergleich
                changes_desc = []
                old_val = {}
                new_val = {}
                
                # Vergleiche Definition
                if original[cat_name].definition != revised[cat_name].definition:
                    changes_desc.append("Definition geändert")
                    old_val['definition'] = original[cat_name].definition
                    new_val['definition'] = revised[cat_name].definition
                
                # Vergleiche Beispiele
                if set(original[cat_name].examples) != set(revised[cat_name].examples):
                    added = set(revised[cat_name].examples) - set(original[cat_name].examples)
                    removed = set(original[cat_name].examples) - set(revised[cat_name].examples)
                    if added:
                        changes_desc.append(f"Neue Beispiele hinzugefügt: {', '.join(added)}")
                    if removed:
                        changes_desc.append(f"Beispiele entfernt: {', '.join(removed)}")
                    old_val['examples'] = original[cat_name].examples
                    new_val['examples'] = revised[cat_name].examples
                
                # Vergleiche Regeln
                if set(original[cat_name].rules) != set(revised[cat_name].rules):
                    changes_desc.append("Kodierregeln aktualisiert")
                    old_val['rules'] = original[cat_name].rules
                    new_val['rules'] = revised[cat_name].rules
                
                # Vergleiche Subkategorien
                orig_subs = original[cat_name].subcategories
                rev_subs = revised[cat_name].subcategories
                if orig_subs != rev_subs:
                    added_subs = set(rev_subs.keys()) - set(orig_subs.keys())
                    removed_subs = set(orig_subs.keys()) - set(rev_subs.keys())
                    modified_subs = {k for k in orig_subs.keys() & rev_subs.keys() 
                                if orig_subs[k] != rev_subs[k]}
                    
                    if added_subs:
                        changes_desc.append(f"Neue Subkategorien: {', '.join(added_subs)}")
                    if removed_subs:
                        changes_desc.append(f"Entfernte Subkategorien: {', '.join(removed_subs)}")
                    if modified_subs:
                        changes_desc.append(f"Modifizierte Subkategorien: {', '.join(modified_subs)}")
                    
                    old_val['subcategories'] = orig_subs
                    new_val['subcategories'] = rev_subs
                
                # Füge Änderung hinzu wenn Unterschiede gefunden wurden
                if changes_desc:
                    self.changes.append(CategoryChange(
                        category_name=cat_name,
                        change_type='modify',
                        description=f"Änderungen bei {material_percentage:.1f}% des Materials: " + "; ".join(changes_desc),
                        timestamp=timestamp,
                        old_value=old_val,
                        new_value=new_val,
                        justification=f"Präzisierung basierend auf Analyseergebnissen: {'; '.join(changes_desc)}"
                    ))
        
        # Speichere Revisionshistorie
        self._save_revision_history()

    
    def _export_intercoder_analysis(self, writer, segment_codings: Dict[str, List[Dict]], reliability: float):
        """
        Exports the intercoder analysis to a separate Excel sheet.
        """
        if 'Intercoderanalyse' not in writer.sheets:
            writer.book.create_sheet('Intercoderanalyse')
        
        worksheet = writer.sheets['Intercoderanalyse']
        current_row = 1

        # 1. Überschrift und Gesamtreliabilität
        worksheet.cell(row=current_row, column=1, value="Intercoderanalyse")
        current_row += 2
        worksheet.cell(row=current_row, column=1, value="Krippendorffs Alpha:")
        worksheet.cell(row=current_row, column=2, value=round(reliability, 3))
        current_row += 2

        # 2. Übereinstimmungsanalyse pro Segment
        worksheet.cell(row=current_row, column=1, value="Detaillierte Segmentanalyse")
        current_row += 1

        headers = ['Segment_ID', 'Anzahl Codierer', 'Übereinstimmungsgrad', 'Hauptkategorien', 'Subkategorien', 'Begründungen']
        for col, header in enumerate(headers, 1):
            worksheet.cell(row=current_row, column=col, value=header)
        current_row += 1

        # Analyse für jedes Segment
        for segment_id, codings in segment_codings.items():
            # Zähle verschiedene Kategorien
            categories = [c['category'] for c in codings]
            subcategories = [', '.join(c.get('subcategories', [])) for c in codings]
            
            # Berechne Übereinstimmungsgrad
            category_agreement = len(set(categories)) == 1
            subcat_agreement = len(set(subcategories)) == 1
            
            if category_agreement and subcat_agreement:
                agreement = "Vollständig"
            elif category_agreement:
                agreement = "Nur Hauptkategorie"
            else:
                agreement = "Keine Übereinstimmung"

            # Fülle Zeile
            row_data = [
                segment_id,
                len(codings),
                agreement,
                ' | '.join(set(categories)),
                ' | '.join(set(subcategories)),
                '\n'.join([c.get('justification', '')[:100] + '...' for c in codings])
            ]

            for col, value in enumerate(row_data, 1):
                worksheet.cell(row=current_row, column=col, value=value)
            current_row += 1

        # 3. Codierer-Vergleichsmatrix
        current_row += 2
        worksheet.cell(row=current_row, column=1, value="Codierer-Vergleichsmatrix")
        current_row += 1

        # Erstelle Liste aller Codierer
        coders = sorted(list({coding['coder_id'] for codings in segment_codings.values() for coding in codings}))
        
        # Schreibe Spaltenüberschriften
        for col, coder in enumerate(coders, 2):
            worksheet.cell(row=current_row, column=col, value=coder)
        current_row += 1

        # Fülle Matrix
        agreement_matrix = {}
        for coder1 in coders:
            agreement_matrix[coder1] = {}
            worksheet.cell(row=current_row, column=1, value=coder1)
            
            for col, coder2 in enumerate(coders, 2):
                if coder1 == coder2:
                    agreement_matrix[coder1][coder2] = 1.0
                else:
                    # Berechne Übereinstimmung zwischen coder1 und coder2
                    agreements = 0
                    total = 0
                    for codings in segment_codings.values():
                        coding1 = next((c for c in codings if c['coder_id'] == coder1), None)
                        coding2 = next((c for c in codings if c['coder_id'] == coder2), None)
                        
                        if coding1 and coding2:
                            total += 1
                            if coding1['category'] == coding2['category']:
                                agreements += 1
                    
                    agreement = agreements / total if total > 0 else 0
                    agreement_matrix[coder1][coder2] = agreement
                    
                worksheet.cell(row=current_row, column=col, value=round(agreement_matrix[coder1][coder2], 2))
            current_row += 1

        # 4. Kategorienspezifische Analyse
        current_row += 2
        worksheet.cell(row=current_row, column=1, value="Kategorienspezifische Übereinstimmung")
        current_row += 1

        # Sammle alle verwendeten Kategorien
        all_categories = set()
        for codings in segment_codings.values():
            for coding in codings:
                all_categories.add(coding['category'])

        # Schreibe Überschriften
        headers = ['Kategorie', 'Verwendungshäufigkeit', 'Übereinstimmungsgrad', 'Hauptcodierer']
        for col, header in enumerate(headers, 1):
            worksheet.cell(row=current_row, column=col, value=header)
        current_row += 1

        # Analysiere jede Kategorie
        for category in sorted(all_categories):
            category_usage = 0
            category_agreements = 0
            coders_using = defaultdict(int)
            
            for codings in segment_codings.values():
                category_codings = [c for c in codings if c['category'] == category]
                if category_codings:
                    category_usage += 1
                    if len(category_codings) == len(codings):  # Alle Codierer einig
                        category_agreements += 1
                    for coding in category_codings:
                        coders_using[coding['coder_id']] += 1

            # Finde Hauptcodierer
            main_coders = sorted(coders_using.items(), key=lambda x: x[1], reverse=True)
            main_coders_str = ', '.join([f"{coder}: {count}" for coder, count in main_coders[:3]])

            # Schreibe Zeile
            row_data = [
                category,
                category_usage,
                f"{(category_agreements/category_usage*100):.1f}%" if category_usage > 0 else "0%",
                main_coders_str
            ]

            for col, value in enumerate(row_data, 1):
                worksheet.cell(row=current_row, column=col, value=value)
            current_row += 1

        # Formatierung
        self._format_intercoder_worksheet(worksheet)

    def _format_intercoder_worksheet(self, worksheet) -> None:
        """Formatiert das Intercoder-Worksheet"""
        try:
            # Spaltenbreiten
            column_widths = {
                'A': 40,  # Segment_ID/Kategorie
                'B': 15,  # Anzahl/Häufigkeit
                'C': 20,  # Übereinstimmung
                'D': 40,  # Kategorien/Codierer
                'E': 40,  # Subkategorien
                'F': 60   # Begründungen
            }
            
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width

            # Überschriften formatieren
            from openpyxl.styles import Font, PatternFill, Alignment
            header_font = Font(bold=True)
            header_fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')
            
            for row in worksheet.iter_rows():
                first_cell = row[0]
                if first_cell.value and isinstance(first_cell.value, str):
                    if first_cell.value.endswith(':') or first_cell.value.isupper():
                        for cell in row:
                            cell.font = header_font
                            cell.fill = header_fill

            # Zellausrichtung
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
                    
        except Exception as e:
            print(f"Warnung: Intercoder-Formatierung fehlgeschlagen: {str(e)}")


# --- Klasse: DocumentReader ---
# Aufgabe: Laden und Vorbereiten des Analysematerials (Textdokumente, Transkripte)

class DocumentReader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Define the core supported formats we want to process
        self.supported_formats = {'.docx', '.pdf', '.txt'}
        os.makedirs(data_dir, exist_ok=True)
        
        print("\nInitialisiere DocumentReader:")
        print(f"Verzeichnis: {os.path.abspath(data_dir)}")
        print(f"Unterstützte Formate: {', '.join(self.supported_formats)}")

    async def read_documents(self) -> Dict[str, str]:
        documents = {}
        try:
            all_files = os.listdir(self.data_dir)
            
            print("\nDateianalyse:")
            def is_supported_file(filename: str) -> bool:
                # Exclude backup and temporary files
                if any(ext in filename.lower() for ext in ['.bak', '.bkk', '.tmp', '~']):
                    return False
                # Get the file extension
                extension = os.path.splitext(filename)[1].lower()
                return extension in self.supported_formats

            supported_files = [f for f in all_files if is_supported_file(f)]
            
            print(f"\nGefundene Dateien:")
            for file in all_files:
                status = "✓" if is_supported_file(file) else "✗"
                print(f"{status} {file}")
            
            print(f"\nVerarbeite Dateien:")
            for filename in supported_files:
                try:
                    filepath = os.path.join(self.data_dir, filename)
                    extension = os.path.splitext(filename)[1].lower()
                    
                    print(f"\nLese: {filename}")
                    
                    if extension == '.docx':
                        content = self._read_docx(filepath)
                    elif extension == '.pdf':
                        content = self._read_pdf(filepath)
                    elif extension == '.txt':
                        content = self._read_txt(filepath)
                    else:
                        print(f"⚠ Nicht unterstütztes Format: {extension}")
                        continue
                    
                    if content and content.strip():
                        documents[filename] = content
                        print(f"✓ Erfolgreich eingelesen: {len(content)} Zeichen")
                    else:
                        print(f"⚠ Keine Textinhalte gefunden")
                
                except Exception as e:
                    print(f"✗ Fehler bei {filename}: {str(e)}")
                    print("Details:")
                    import traceback
                    traceback.print_exc()
                    continue

            print(f"\nVerarbeitungsstatistik:")
            print(f"- Dateien im Verzeichnis: {len(all_files)}")
            print(f"- Unterstützte Dateien: {len(supported_files)}")
            print(f"- Erfolgreich eingelesen: {len(documents)}")
            
            return documents

        except Exception as e:
            print(f"Fehler beim Einlesen der Dokumente: {str(e)}")
            import traceback
            traceback.print_exc()
            return documents

    def _read_txt(self, filepath: str) -> str:
        """Liest eine Textdatei ein."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def _read_docx(self, filepath: str) -> str:
        """
        Liest eine DOCX-Datei ein und extrahiert den Text mit ausführlicher Diagnose.
        """
        try:
            from docx import Document
            print(f"\nDetailierte Analyse von: {os.path.basename(filepath)}")
            
            # Öffne das Dokument mit zusätzlicher Fehlerbehandlung
            try:
                doc = Document(filepath)
            except Exception as e:
                print(f"  Fehler beim Öffnen der Datei: {str(e)}")
                print("  Versuche alternative Öffnungsmethode...")
                # Manchmal hilft es, die Datei zuerst zu kopieren
                import shutil
                temp_path = filepath + '.temp'
                shutil.copy2(filepath, temp_path)
                doc = Document(temp_path)
                os.remove(temp_path)

            # Sammle Dokumentinformationen
            paragraphs = []
            print("\nDokumentanalyse:")
            print(f"  Gefundene Paragraphen: {len(doc.paragraphs)}")
            
            # Verarbeite jeden Paragraphen einzeln
            for i, paragraph in enumerate(doc.paragraphs):
                text = paragraph.text.strip()
                if text:
                    print(f"  Paragraph {i+1}: {len(text)} Zeichen")
                    paragraphs.append(text)
                else:
                    print(f"  Paragraph {i+1}: Leer")

            # Wenn Paragraphen gefunden wurden
            if paragraphs:
                full_text = '\n'.join(paragraphs)
                print(f"\nErgebnis:")
                print(f"  ✓ {len(paragraphs)} Textparagraphen extrahiert")
                print(f"  ✓ Gesamtlänge: {len(full_text)} Zeichen")
                return full_text
            
            # Wenn keine Paragraphen gefunden wurden, suche in anderen Bereichen
            print("\nSuche nach alternativen Textinhalten:")
            
            # Prüfe Tabellen
            table_texts = []
            for i, table in enumerate(doc.tables):
                print(f"  Prüfe Tabelle {i+1}:")
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            table_texts.append(cell.text.strip())
                            print(f"    Zelleninhalt gefunden: {len(cell.text)} Zeichen")
            
            if table_texts:
                full_text = '\n'.join(table_texts)
                print(f"\nErgebnis:")
                print(f"  ✓ {len(table_texts)} Tabelleneinträge extrahiert")
                print(f"  ✓ Gesamtlänge: {len(full_text)} Zeichen")
                return full_text
                
            print("\n✗ Keine Textinhalte im Dokument gefunden")
            return ""
                
        except ImportError:
            print("\n✗ python-docx nicht installiert.")
            print("  Bitte installieren Sie das Paket mit:")
            print("  pip install python-docx")
            raise
        except Exception as e:
            print(f"\n✗ Unerwarteter Fehler beim DOCX-Lesen:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _read_pdf(self, filepath: str) -> str:
        """Liest eine PDF-Datei ein und extrahiert den Text."""
        try:
            import PyPDF2
            text_content = []
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    if text := page.extract_text():
                        text_content.append(text)
            return '\n'.join(text_content)
        except ImportError:
            print("PyPDF2 nicht installiert. Bitte installieren Sie: pip install PyPDF2")
            raise
        except Exception as e:
            print(f"Fehler beim PDF-Lesen: {str(e)}")
            raise
    def _extract_metadata(self, filename: str) -> Tuple[str, str]:
        """
        Extrahiert Metadaten aus dem Dateinamen.
        Erwartet Format: attribut1_attribut2.extension
        
        Args:
            filename (str): Name der Datei
            
        Returns:
            Tuple[str, str]: (attribut1, attribut2)
        """
        try:
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split('_')
            if len(parts) >= 2:
                return parts[0], parts[1]
            else:
                return name_without_ext, ""
        except Exception as e:
            print(f"Fehler beim Extrahieren der Metadaten aus {filename}: {str(e)}")
            return filename, ""

# --- Klasse: TokenCounter ---
class TokenCounter:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def add_tokens(self, input_tokens, output_tokens):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def get_report(self):
        return f"Gesamte Token-Nutzung:\n" \
               f"Input Tokens: {self.input_tokens}\n" \
               f"Output Tokens: {self.output_tokens}\n" \
               f"Gesamt Tokens: {self.input_tokens + self.output_tokens}"

token_counter = TokenCounter()


# --- Hilfsfunktionen ---

def estimate_tokens(text: str) -> int:
    return len(text.split())

def get_input_with_timeout(prompt: str, timeout: int = 10) -> str:
    """
    Fragt nach Benutzereingabe mit Timeout.
    
    Args:
        prompt: Anzuzeigender Text
        timeout: Timeout in Sekunden
        
    Returns:
        str: Benutzereingabe oder 'n' bei Timeout
    """
    import threading
    import time
    
    answer = {'value': 'n'}  # Default-Wert bei Timeout
    
    def get_input():
        answer['value'] = input(prompt).lower()
    
    # Starte Input-Thread
    thread = threading.Thread(target=get_input)
    thread.daemon = True
    thread.start()
    
    # Warte auf Antwort oder Timeout
    thread.join(timeout)
    
    if thread.is_alive():
        # Thread läuft noch (Timeout erreicht)
        print(f"\nKeine Eingabe innerhalb von {timeout} Sekunden - fahre automatisch fort...")
        return 'n'
    
    return answer['value']

async def perform_manual_coding(chunks, categories, manual_coders):
    """Führt die manuelle Kodierung durch"""
    manual_codings = []
    for document_name, document_chunks in chunks.items():
        for chunk_id, chunk in enumerate(document_chunks):
            print(f"\nManuelles Codieren: Dokument {document_name}, Chunk {chunk_id + 1}/{len(document_chunks)}")
            
            for manual_coder in manual_coders:
                try:
                    coding_result = await manual_coder.code_chunk(chunk, categories)
                    if coding_result == "ABORT_ALL":
                        print("Manuelles Kodieren wurde vom Benutzer abgebrochen.")
                        return manual_codings
                        
                    if coding_result:
                        coding_entry = {
                            'segment_id': f"{document_name}_chunk_{chunk_id}",
                            'coder_id': manual_coder.coder_id,
                            'category': coding_result.category,
                            'subcategories': coding_result.subcategories,
                            'confidence': coding_result.confidence.get('total', 0),
                            'justification': coding_result.justification
                        }
                        manual_codings.append(coding_entry)
                        print(f"✓ Manuelle Kodierung erfolgreich: {coding_entry['category']}")
                    else:
                        print("⚠ Manuelle Kodierung übersprungen")
                        
                except Exception as e:
                    print(f"Fehler bei manuellem Kodierer {manual_coder.coder_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue  # Fahre mit dem nächsten Chunk fort
                    
                # Kurze Pause zwischen den Chunks
                await asyncio.sleep(0.5)
    
    return manual_codings


# ============================ 
# 5. Hauptprogramm
# ============================ 

# Aufgabe: Zusammenführung aller Komponenten, Steuerung des gesamten Analyseprozesses
async def main() -> None:
    try:
        print("=== Qualitative Inhaltsanalyse nach Mayring ===")

        # 0. Konfiguration laden
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_loader = ConfigLoader(script_dir)
        
        if config_loader.load_config():
            config_loader.update_script_globals(globals())
            print("\nKonfiguration erfolgreich geladen")
        else:
            print("Verwende Standard-Konfiguration")

        # 1. Kategoriensystem initialisieren
        print("\n1. Initialisiere Kategoriensystem...")
        category_builder = DeductiveCategoryBuilder()
        initial_categories = category_builder.load_theoretical_categories()
        
        # Revision Manager initialisieren
        revision_manager = CategoryRevisionManager(CONFIG['OUTPUT_DIR'])
        
        for category_name in initial_categories.keys():
            revision_manager.changes.append(CategoryChange(
                category_name=category_name,
                change_type='add',
                description="Erste deduktive Kategorie",
                timestamp=datetime.now().isoformat(),
                justification="Teil des ursprünglichen deduktiven Kategoriensystems"
            ))

        # 2. Dokumente einlesen
        print("\n2. Lese Dokumente ein...")
        reader = DocumentReader(CONFIG['DATA_DIR'])
        documents = await reader.read_documents()

        if not documents:
            print("\nKeine Dokumente zum Analysieren gefunden.")
            return

        # 3. Kodierer konfigurieren
        auto_coders = []
        manual_coders = []

        for config in CONFIG['CODER_SETTINGS']:
            coder = DeductiveCoder(
                model_name=CONFIG['MODEL_NAME'],
                temperature=config["temperature"],
                coder_id=config["coder_id"]
            )
            auto_coders.append(coder)

        # Optional: Manuellen Kodierer hinzufügen
        print("\nAutomatische Fortführung in 10 Sekunden...")
        include_manual = get_input_with_timeout("\nMöchten Sie manuell kodieren? (j/n): ") == 'j'
        if include_manual:
            manual_coder = ManualCoder(coder_id="human_1")
            manual_coders.append(manual_coder)

        # 4. Chunks erstellen
        loader = MaterialLoader()
        chunks = {}
        for doc_name, doc_text in documents.items():
            chunks[doc_name] = loader.chunk_text(doc_text)

        
        # 5. Manuelle Codierung (falls gewünscht)
        manual_codings = []
        if include_manual:
            print("\n5. Starte manuelle Codierung...")
            manual_coding_result = await perform_manual_coding(
                chunks=chunks, 
                categories=initial_categories,  # Verwende initiale Kategorien
                manual_coders=manual_coders
            )
            if manual_coding_result == "ABORT_ALL":
                print("Manuelle Codierung wurde abgebrochen. Beende das Programm.")
                return
            manual_codings = manual_coding_result
            print(f"Manuelle Codierung abgeschlossen: {len(manual_codings)} Codierungen")

        # 6. Integrierte Analyse starten
        print("\n5. Starte integrierte Analyse...")
        analysis_manager = IntegratedAnalysisManager(CONFIG)
        
        # Führe integrierte Analyse durch
        final_categories, all_codings = await analysis_manager.analyze_material(
            chunks=chunks,
            initial_categories=initial_categories
        )

        # Kombiniere alle Kodierungen
        all_codings.extend(manual_codings)

        # 7. Berechne Intercoder-Reliabilität
        print("\n7. Berechne Intercoder-Reliabilität...")
        reliability_calculator = InductiveCoder(
            model_name=CONFIG['MODEL_NAME'],
            output_dir=CONFIG['OUTPUT_DIR']
        )
        reliability = reliability_calculator._calculate_reliability(all_codings)

        print(f"\nIntercoder-Reliabilität (Krippendorffs Alpha): {reliability:.3f}")

        # 8. Exportiere Ergebnisse
        print("\n8. Exportiere Ergebnisse...")
        exporter = ResultsExporter(
            output_dir=CONFIG['OUTPUT_DIR'],
            attribute_labels=CONFIG['ATTRIBUTE_LABELS']
        )
        
        await exporter.export_results(
            codings=all_codings,
            reliability=reliability,
            categories=final_categories,
            chunks=chunks,
            revision_manager=revision_manager,
            export_mode="consensus",
            original_categories=initial_categories,
            merge_log=analysis_manager.category_merger.merge_log,
            inductive_coder=reliability_calculator
        )

        print("\nAnalyse abgeschlossen.")
        print("\n" + token_counter.get_report())

    except Exception as e:
        print(f"Fehler in der Hauptausführung: {str(e)}")
        import traceback
        traceback.print_exc()

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
