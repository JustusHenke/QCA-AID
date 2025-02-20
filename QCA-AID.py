"""
QCA-AID: Qualitative Content Analysis with AI Support - Deductive Coding
========================================================================

A Python implementation of Mayring's Qualitative Content Analysis methodology,
enhanced with AI capabilities through the OpenAI API.

Version:
--------
0.9.7 (2025-02-20)

New:
- Switch between OpenAI and Mistral using CONFIG parameter 'MODEL_PROVIDER'
- Standard model for Openai is 'GPT-4o-mini', for Mistral 'mistral-small'

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
1. Place interview transcripts in the 'output' directory
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
from openai import AsyncOpenAI
import httpx
from mistralai import Mistral
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
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
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl import load_workbook
from openpyxl import Workbook
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Dict, List, Optional
import platform
import time
import statistics
import traceback 

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

VALIDATION_THRESHOLDS = {
    'MIN_DEFINITION_WORDS': 15,
    'MIN_EXAMPLES': 2,
    'SIMILARITY_THRESHOLD': 0.7,
    'MIN_SUBCATEGORIES': 2,
    'MAX_NAME_LENGTH': 50,
    'MIN_NAME_LENGTH': 3
}

ENGLISH_WORDS = {
    'research', 'development', 'management', 
    'system', 'process', 'analysis'
}

VALIDATION_MESSAGES = {
    'short_definition': "Definition zu kurz (min. {min_words} Wörter)",
    'few_examples': "Zu wenige Beispiele (min. {min_examples})",
    'english_terms': "Name enthält englische Begriffe",
    'name_length': "Name muss zwischen {min_len} und {max_len} Zeichen sein"
}

# ------------------------
# Konfigurationskonstanten
# ------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    'MODEL_PROVIDER': 'OpenAI',
    'MODEL_NAME': 'gpt-4o-mini',
    'DATA_DIR': os.path.join(SCRIPT_DIR, 'input'),
    'OUTPUT_DIR': os.path.join(SCRIPT_DIR, 'output'),
    'CHUNK_SIZE': 800,
    'CHUNK_OVERLAP': 80,
    'BATCH_SIZE': 5,
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
# 2. LLM-Provider konfigurieren
# ============================





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
        """
        Erzeugt eine Completion mit Mistral
        
        Args:
            messages: Liste von Nachrichten im Mistral Format
            model: Name des zu verwendenden Modells 
            temperature: Kreativität der Antworten (0.0-1.0)
            response_format: Optional dict für JSON response etc.
            
        Returns:
            Mistral Chat Completion Response
        """
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
        """
        Erstellt die passende Provider Instanz
        
        Args:
            provider_name: Name des Providers ("openai" oder "mistral")
            
        Returns:
            LLMProvider Instanz
        
        Raises:
            ValueError: Wenn ungültiger Provider Name
        """
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

# Hilfsklasse für einheitliche Response Verarbeitung            
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

# ============================
# 3. Klassen und Funktionen
# ============================

@dataclass
class CategoryDefinition:
    """Datenklasse für eine Kategorie im Kodiersystem"""
    name: str
    definition: str
    examples: List[str]
    rules: List[str]
    subcategories: Dict[str, str]
    added_date: str
    modified_date: str

    def replace(self, **changes) -> 'CategoryDefinition':
        """
        Erstellt eine neue Instanz mit aktualisierten Werten.
        Ähnlich wie _replace bei namedtuples.
        """
        new_values = {
            'name': self.name,
            'definition': self.definition,
            'examples': self.examples.copy(),
            'rules': self.rules.copy(),
            'subcategories': self.subcategories.copy(),
            'added_date': self.added_date,
            'modified_date': datetime.now().strftime("%Y-%m-%d")
        }
        new_values.update(changes)
        return CategoryDefinition(**new_values)

    def update_examples(self, new_examples: List[str]) -> None:
        """Fügt neue Beispiele hinzu ohne Duplikate."""
        self.examples = list(set(self.examples + new_examples))
        self.modified_date = datetime.now().strftime("%Y-%m-%d")

    def update_rules(self, new_rules: List[str]) -> None:
        """Fügt neue Kodierregeln hinzu ohne Duplikate."""
        self.rules = list(set(self.rules + new_rules))
        self.modified_date = datetime.now().strftime("%Y-%m-%d")

    def add_subcategories(self, new_subcats: Dict[str, str]) -> None:
        """Fügt neue Subkategorien hinzu."""
        self.subcategories.update(new_subcats)
        self.modified_date = datetime.now().strftime("%Y-%m-%d")

    def to_dict(self) -> Dict:
        """Konvertiert die Kategorie in ein Dictionary."""
        return {
            'name': self.name,
            'definition': self.definition,
            'examples': self.examples,
            'rules': self.rules,
            'subcategories': self.subcategories,
            'added_date': self.added_date,
            'modified_date': self.modified_date
        }

@dataclass(frozen=True)  # Macht die Klasse immutable und hashable
class CodingResult:
    """Datenklasse für ein Kodierungsergebnis"""
    category: str
    subcategories: Tuple[str, ...]  # Änderung von List zu Tuple für Hashability
    justification: str
    confidence: Dict[str, Union[float, Tuple[str, ...]]]  # Ändere List zu Tuple
    text_references: Tuple[str, ...]  # Änderung von List zu Tuple
    uncertainties: Optional[Tuple[str, ...]] = None  # Änderung von List zu Tuple
    paraphrase: str = ""
    keywords: str = "" 

    def __post_init__(self):
        # Konvertiere Listen zu Tupeln, falls nötig
        object.__setattr__(self, 'subcategories', tuple(self.subcategories) if isinstance(self.subcategories, list) else self.subcategories)
        object.__setattr__(self, 'text_references', tuple(self.text_references) if isinstance(self.text_references, list) else self.text_references)
        if self.uncertainties is not None:
            object.__setattr__(self, 'uncertainties', tuple(self.uncertainties) if isinstance(self.uncertainties, list) else self.uncertainties)
        
        # Konvertiere confidence Listen zu Tupeln
        if isinstance(self.confidence, dict):
            new_confidence = {}
            for k, v in self.confidence.items():
                if isinstance(v, list):
                    new_confidence[k] = tuple(v)
                else:
                    new_confidence[k] = v
            object.__setattr__(self, 'confidence', new_confidence)

    def to_dict(self) -> Dict:
        """Konvertiert das CodingResult in ein Dictionary"""
        return {
            'category': self.category,
            'subcategories': list(self.subcategories),  # Zurück zu Liste für JSON-Serialisierung
            'justification': self.justification,
            'confidence': self.confidence,
            'text_references': list(self.text_references),  # Zurück zu Liste
            'uncertainties': list(self.uncertainties) if self.uncertainties else None,
            'paraphrase': self.paraphrase ,
            'keywords': self.keywords         }

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
        
    def load_codebook(self):
        print(f"Versuche Konfiguration zu laden von: {self.excel_path}")
        if not os.path.exists(self.excel_path):
            print(f"Excel-Datei nicht gefunden: {self.excel_path}")
            return False

        try:
            # Öffne die Excel-Datei mit ausführlicher Fehlerbehandlung
            print("\nÖffne Excel-Datei...")
            wb = load_workbook(self.excel_path, read_only=True, data_only=True)
            print(f"Excel-Datei erfolgreich geladen. Verfügbare Sheets: {wb.sheetnames}")
            
            # Prüfe DEDUKTIVE_KATEGORIEN Sheet
            if 'DEDUKTIVE_KATEGORIEN' in wb.sheetnames:
                print("\nLese DEDUKTIVE_KATEGORIEN Sheet...")
                sheet = wb['DEDUKTIVE_KATEGORIEN']
            
            
            # Lade die verschiedenen Komponenten
            self._load_research_question(wb)
            self._load_coding_rules(wb)
            self._load_config(wb)
            
            # Debug-Ausgabe vor dem Laden der Kategorien
            print("\nStarte Laden der deduktiven Kategorien...")
            kategorien = self._load_deduktive_kategorien(wb)
            
            # Prüfe das Ergebnis
            if kategorien:
                print("\nGeladene Kategorien:")
                for name, data in kategorien.items():
                    print(f"\n{name}:")
                    print(f"- Definition: {len(data['definition'])} Zeichen")
                    print(f"- Beispiele: {len(data['examples'])}")
                    print(f"- Regeln: {len(data['rules'])}")
                    print(f"- Subkategorien: {len(data['subcategories'])}")
                
                # Speichere in Config
                self.config['DEDUKTIVE_KATEGORIEN'] = kategorien
                print("\nKategorien erfolgreich in Config gespeichert")
                return True
            else:
                print("\nKeine Kategorien geladen!")
                return False

        except Exception as e:
            print(f"Fehler beim Lesen der Excel-Datei: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()
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
        try:
            if 'DEDUKTIVE_KATEGORIEN' not in wb.sheetnames:
                print("Warnung: Sheet 'DEDUKTIVE_KATEGORIEN' nicht gefunden")
                return {}

            print("\nLade deduktive Kategorien...")
            sheet = wb['DEDUKTIVE_KATEGORIEN']
            
            # Initialisiere Kategorien
            kategorien = {}
            current_category = None
            
            # Hole Header-Zeile
            headers = []
            for cell in sheet[1]:
                headers.append(cell.value)
            print(f"Gefundene Spalten: {headers}")
            
            # Indizes für Spalten finden
            key_idx = headers.index('Key') if 'Key' in headers else None
            sub_key_idx = headers.index('Sub-Key') if 'Sub-Key' in headers else None
            sub_sub_key_idx = headers.index('Sub-Sub-Key') if 'Sub-Sub-Key' in headers else None
            value_idx = headers.index('Value') if 'Value' in headers else None
            
            if None in [key_idx, sub_key_idx, value_idx]:
                print("Fehler: Erforderliche Spalten fehlen!")
                return {}
                
            # Verarbeite Zeilen
            for row_idx, row in enumerate(sheet.iter_rows(min_row=2), 2):
                try:
                    key = row[key_idx].value
                    sub_key = row[sub_key_idx].value if row[sub_key_idx].value else None
                    sub_sub_key = row[sub_sub_key_idx].value if sub_sub_key_idx is not None and row[sub_sub_key_idx].value else None
                    value = row[value_idx].value if row[value_idx].value else None
                    
                    
                    # Neue Hauptkategorie
                    if key and isinstance(key, str):
                        key = key.strip()
                        if key not in kategorien:
                            print(f"\nNeue Hauptkategorie: {key}")
                            current_category = key
                            kategorien[key] = {
                                'definition': '',
                                'rules': [],
                                'examples': [],
                                'subcategories': {}
                            }
                    
                    # Verarbeite Unterkategorien und Werte
                    if current_category and sub_key:
                        sub_key = sub_key.strip()
                        if isinstance(value, str):
                            value = value.strip()
                            
                        if sub_key == 'definition':
                            kategorien[current_category]['definition'] = value
                            print(f"  Definition hinzugefügt: {len(value)} Zeichen")
                            
                        elif sub_key == 'rules':
                            if value:
                                kategorien[current_category]['rules'].append(value)
                                print(f"  Regel hinzugefügt: {value[:50]}...")
                                
                        elif sub_key == 'examples':
                            if value:
                                kategorien[current_category]['examples'].append(value)
                                print(f"  Beispiel hinzugefügt: {value[:50]}...")
                                
                        elif sub_key == 'subcategories' and sub_sub_key:
                            kategorien[current_category]['subcategories'][sub_sub_key] = value
                            print(f"  Subkategorie hinzugefügt: {sub_sub_key}")
                                
                except Exception as e:
                    print(f"Fehler in Zeile {row_idx}: {str(e)}")
                    continue

            # Validierung der geladenen Daten
            print("\nValidiere geladene Kategorien:")
            for name, kat in kategorien.items():
                print(f"\nKategorie: {name}")
                print(f"- Definition: {len(kat['definition'])} Zeichen")
                if not kat['definition']:
                    print("  WARNUNG: Keine Definition!")
                print(f"- Regeln: {len(kat['rules'])}")
                print(f"- Beispiele: {len(kat['examples'])}")
                print(f"- Subkategorien: {len(kat['subcategories'])}")
                for sub_name, sub_def in kat['subcategories'].items():
                    print(f"  • {sub_name}: {sub_def[:50]}...")

            # Ergebnis
            if kategorien:
                print(f"\nErfolgreich {len(kategorien)} Kategorien geladen")
                return kategorien
            else:
                print("\nKeine Kategorien gefunden!")
                return {}

        except Exception as e:
            print(f"Fehler beim Laden der Kategorien: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()
            return {}
        
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
                            if key == 'BATCH_SIZE' or sub_key == 'BATCH_SIZE':
                                try:
                                    value = int(value)
                                    print(f"BATCH_SIZE aus Codebook geladen: {value}")
                                except (ValueError, TypeError):
                                    value = 5  # Standardwert
                                    print(f"Warnung: Ungültiger BATCH_SIZE Wert, verwende Standard: {value}")
                            config[key][sub_key] = value
                        else:
                            if sub_key not in config[key]:
                                config[key][sub_key] = {}
                            config[key][sub_key][sub_sub_key] = value

            self.config['CONFIG'] = self._sanitize_config(config)
            return True  # Explizite Rückgabe von True
        return False

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
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Stelle sicher, dass OUTPUT_DIR immer gesetzt wird
            sanitized['OUTPUT_DIR'] = os.path.join(script_dir, 'output')
            os.makedirs(sanitized['OUTPUT_DIR'], exist_ok=True)
            print(f"Standard-Ausgabeverzeichnis gesichert: {sanitized['OUTPUT_DIR']}")
            
            for key, value in config.items():
                # Verzeichnispfade relativ zum aktuellen Arbeitsverzeichnis
                if key in ['DATA_DIR', 'OUTPUT_DIR']:
                    sanitized[key] = os.path.join(script_dir, str(value))
                    # Stelle sicher, dass Verzeichnis existiert
                    os.makedirs(sanitized[key], exist_ok=True)
                    print(f"Verzeichnis gesichert: {sanitized[key]}")
                
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
                                else 0.3,
                            'coder_id': str(coder.get('coder_id', f'auto_{i}'))
                        }
                        for i, coder in enumerate(value)
                    ]
                
                # Alle anderen Werte unverändert übernehmen
                else:
                    sanitized[key] = value
           
            if 'BATCH_SIZE' in config:
                try:
                    batch_size = int(config['BATCH_SIZE'])
                    if batch_size < 1:
                        print("Warnung: BATCH_SIZE muss mindestens 1 sein")
                        batch_size = 5
                    elif batch_size > 20:
                        print("Warnung: BATCH_SIZE > 20 könnte Performance-Probleme verursachen")
                    sanitized['BATCH_SIZE'] = batch_size
                    print(f"Finale BATCH_SIZE: {batch_size}")
                except (ValueError, TypeError):
                    print("Warnung: Ungültiger BATCH_SIZE Wert")
                    sanitized['BATCH_SIZE'] = 5
            else:
                print("BATCH_SIZE nicht in Codebook gefunden, verwende Standard: 5")
                sanitized['BATCH_SIZE'] = 5

            for key, value in config.items():
                if key == 'BATCH_SIZE':
                    continue

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
        
    def update_script_globals(self, globals_dict):
        """
        Aktualisiert die globalen Variablen mit den Werten aus der Config.
        """
        try:
            print("\nAktualisiere globale Variablen...")
            
            # Update DEDUKTIVE_KATEGORIEN
            if 'DEDUKTIVE_KATEGORIEN' in self.config:
                deduktive_kat = self.config['DEDUKTIVE_KATEGORIEN']
                if deduktive_kat and isinstance(deduktive_kat, dict):
                    globals_dict['DEDUKTIVE_KATEGORIEN'] = deduktive_kat
                    print(f"\nDEDUKTIVE_KATEGORIEN aktualisiert:")
                    for name in deduktive_kat.keys():
                        print(f"- {name}")
                else:
                    print("Warnung: Keine gültigen DEDUKTIVE_KATEGORIEN in Config")
            
            # Update andere Konfigurationswerte
            for key, value in self.config.items():
                if key != 'DEDUKTIVE_KATEGORIEN' and key in globals_dict:
                    if isinstance(value, dict) and isinstance(globals_dict[key], dict):
                        globals_dict[key].clear()
                        globals_dict[key].update(value)
                        print(f"Dict {key} aktualisiert")
                    else:
                        globals_dict[key] = value
                        print(f"Variable {key} aktualisiert")

            # Validiere Update
            if 'DEDUKTIVE_KATEGORIEN' in globals_dict:
                kat_count = len(globals_dict['DEDUKTIVE_KATEGORIEN'])
                print(f"\nFinale Validierung: {kat_count} Kategorien in globalem Namespace")
                if kat_count == 0:
                    print("Warnung: Keine Kategorien im globalen Namespace!")
            else:
                print("Fehler: DEDUKTIVE_KATEGORIEN nicht im globalen Namespace!")

        except Exception as e:
            print(f"Fehler beim Update der globalen Variablen: {str(e)}")
            import traceback
            traceback.print_exc()

    def _load_validation_config(self, wb):
        """Lädt die Validierungskonfiguration aus dem Codebook."""
        if 'CONFIG' not in wb.sheetnames:
            return {}
            
        validation_config = {
            'thresholds': {},
            'english_words': set(),
            'messages': {}
        }
        
        df = pd.read_excel(self.excel_path, sheet_name='CONFIG')
        validation_rows = df[df['Key'] == 'VALIDATION']
        
        for _, row in validation_rows.iterrows():
            sub_key = row['Sub-Key']
            sub_sub_key = row['Sub-Sub-Key']
            value = row['Value']
            
            if sub_key == 'ENGLISH_WORDS':
                if value == 'true':
                    validation_config['english_words'].add(sub_sub_key)
            elif sub_key == 'MESSAGES':
                validation_config['messages'][sub_sub_key] = value
            else:
                # Numerische Schwellenwerte
                try:
                    validation_config['thresholds'][sub_key] = float(value)
                except (ValueError, TypeError):
                    print(f"Warnung: Ungültiger Schwellenwert für {sub_key}: {value}")
        
        return validation_config

    def get_config(self):
        return self.config

    
        

# --- Klasse: MaterialLoader ---
# Aufgabe: Laden und Vorbereiten des Analysematerials (Textdokumente, output)
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
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
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

# --- Klasse: DevelopmentHistory ---
# Aufgabe: Dokumentation der Kategorienentwicklung
class DevelopmentHistory:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.history_file = os.path.join(output_dir, "development_history.json")
        
        # Various aspects of history
        self.category_changes = []     # Category changes
        self.coding_history = []       # Coding results
        self.batch_history = []        # Batch processing
        self.saturation_checks = []    # Saturation checks
        self.validation_history = []   # System validations
        self.analysis_events = []      # Analysis events
        self.errors = []              # Error events
        self.category_development = [] # Category development tracking
        self.analysis_start = None    # Analysis start information
        
        # Performance tracking
        self.performance_metrics = {
            'batch_times': [],
            'coding_times': [],
            'processing_rates': []
        }

    def log_category_development(self, phase: str, **metrics) -> None:
        """
        Logs the progress of category development.
        
        Args:
            phase: Current phase of development
            **metrics: Additional metrics to log
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            **metrics
        }
        self.category_development.append(entry)
        self._save_history()

    def log_batch_processing(self, batch_size: int, processing_time: float, results: dict) -> None:
        """Logs information about batch processing."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': batch_size,
            'processing_time': processing_time,
            'results': results
        }
        self.batch_history.append(entry)
        self._save_history()

    def log_saturation_check(self, material_percentage: float, result: str, metrics: Optional[dict]) -> None:
        """Logs the results of a saturation check."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'material_percentage': material_percentage,
            'result': result,
            'metrics': metrics
        }
        self.saturation_checks.append(entry)
        self._save_history()

    def log_saturation_reached(self, material_percentage: float, metrics: dict) -> None:
        """Logs when saturation is reached."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'saturation_reached',
            'material_percentage': material_percentage,
            'metrics': metrics
        }
        self.analysis_events.append(entry)
        self._save_history()

    def log_analysis_completion(self, final_categories: dict, total_time: float, total_codings: int) -> None:
        """Logs the completion of the analysis."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'analysis_complete',
            'total_time': total_time,
            'total_categories': len(final_categories),
            'total_codings': total_codings
        }
        self.analysis_events.append(entry)
        self._save_history()

    def log_analysis_start(self, total_segments: int, total_categories: int) -> None:
        """
        Logs the start of analysis with initial metrics.
        
        Args:
            total_segments: Total number of text segments to analyze
            total_categories: Initial number of categories
        """
        self.analysis_start = {
            'timestamp': datetime.now().isoformat(),
            'total_segments': total_segments,
            'total_categories': total_categories,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.analysis_events.append({
            'event': 'analysis_start',
            **self.analysis_start
        })
        self._save_history()

    def log_error(self, error_type: str, error_message: str) -> None:
        """Logs an error event."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'error',
            'type': error_type,
            'message': error_message
        }
        self.errors.append(entry)
        self._save_history()

    def _save_history(self) -> None:
        """Saves the current history to the JSON file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'category_changes': self.category_changes,
                    'coding_history': self.coding_history,
                    'batch_history': self.batch_history,
                    'saturation_checks': self.saturation_checks,
                    'validation_history': self.validation_history,
                    'analysis_events': self.analysis_events,
                    'errors': self.errors,
                    'category_development': self.category_development,
                    'performance_metrics': self.performance_metrics,
                'analysis_start': self.analysis_start
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving history: {str(e)}")

    def generate_development_report(self) -> str:
        """
        Generates a comprehensive report of the development process.
        
        Returns:
            str: Formatted report
        """
        report = ["=== Category Development Report ===\n"]
        
        # Phase Analysis
        phases = {}
        for entry in self.category_development:
            phase = entry['phase']
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(entry)
        
        for phase, entries in phases.items():
            report.append(f"\nPhase: {phase}")
            report.append("-" * (len(phase) + 7))
            
            # Summarize metrics for this phase
            metrics_summary = {}
            for entry in entries:
                for key, value in entry.items():
                    if key not in ['timestamp', 'phase']:
                        if key not in metrics_summary:
                            metrics_summary[key] = []
                        metrics_summary[key].append(value)
            
            # Report metrics
            for key, values in metrics_summary.items():
                if all(isinstance(v, (int, float)) for v in values):
                    avg = sum(values) / len(values)
                    report.append(f"{key}: avg = {avg:.2f}")
                else:
                    report.append(f"{key}: {len(values)} entries")
        
        # Error Analysis
        if self.errors:
            report.append("\nErrors:")
            report.append("-------")
            for error in self.errors:
                report.append(f"- {error['type']}: {error['message']}")
        
        return "\n".join(report)


class CategoryValidator:
    """
    Zentrale Klasse für alle Kategorievalidierungen mit Caching der Ergebnisse.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialisiert den Validator mit Konfiguration.
        
        Args:
            config: Konfigurationsdictionary aus dem Codebook (optional)
        """
        try:
            # Standardwerte
            validation_entries = {}
            
            # Hole VALIDATION-Einträge aus der Config, falls vorhanden
            if config and 'CONFIG' in config:
                # Durchsuche CONFIG nach VALIDATION-Einträgen
                for row in config['CONFIG']:
                    if isinstance(row, dict) and row.get('Key') == 'VALIDATION':
                        sub_key = row.get('Sub-Key')
                        value = row.get('Value')
                        if sub_key and value is not None:
                            validation_entries[sub_key] = value

            # Setze Schwellenwerte mit Fallback-Werten
            self.MIN_DEFINITION_WORDS = self._get_numeric_value(validation_entries, 'MIN_DEFINITION_WORDS', 15)
            self.MIN_EXAMPLES = self._get_numeric_value(validation_entries, 'MIN_EXAMPLES', 2)
            self.SIMILARITY_THRESHOLD = self._get_numeric_value(validation_entries, 'SIMILARITY_THRESHOLD', 0.7)
            self.MIN_SUBCATEGORIES = self._get_numeric_value(validation_entries, 'MIN_SUBCATEGORIES', 2)
            self.MAX_NAME_LENGTH = self._get_numeric_value(validation_entries, 'MAX_NAME_LENGTH', 50)
            self.MIN_NAME_LENGTH = self._get_numeric_value(validation_entries, 'MIN_NAME_LENGTH', 3)
            
            # Hole verbotene englische Begriffe
            self.ENGLISH_WORDS = set()
            if config and 'CONFIG' in config:
                for row in config['CONFIG']:
                    if (isinstance(row, dict) and 
                        row.get('Key') == 'VALIDATION' and 
                        row.get('Sub-Key') == 'ENGLISH_WORDS' and 
                        row.get('Sub-Sub-Key')):
                        self.ENGLISH_WORDS.add(row['Sub-Sub-Key'].lower())
            
            if not self.ENGLISH_WORDS:  # Fallback wenn keine Begriffe in Config
                self.ENGLISH_WORDS = {'research', 'development', 'management', 
                                    'system', 'process', 'analysis'}
            
            # Hole Fehlermeldungen
            self.messages = {}
            if config and 'CONFIG' in config:
                for row in config['CONFIG']:
                    if (isinstance(row, dict) and
                        row.get('Key') == 'VALIDATION' and 
                        row.get('Sub-Key') == 'MESSAGES' and 
                        row.get('Sub-Sub-Key')):
                        self.messages[row['Sub-Sub-Key']] = row['Value']
            
            # Cache und Statistiken
            self.validation_cache = {}
            self.similarity_cache = {} 
            self.validation_stats = {
                'cache_hits': 0,
                'cache_misses': 0,
                'total_validations': 0,
                'similarity_calculations': 0
            }
            
            print("\nKategorie-Validator initialisiert:")
            print(f"- Min. Wörter Definition: {self.MIN_DEFINITION_WORDS}")
            print(f"- Min. Beispiele: {self.MIN_EXAMPLES}")
            print(f"- Ähnlichkeitsschwelle: {self.SIMILARITY_THRESHOLD}")
            print(f"- Verbotene Begriffe: {len(self.ENGLISH_WORDS)}")

        except Exception as e:
            print(f"Fehler bei Validator-Initialisierung: {str(e)}")
            print("Details:")
            traceback.print_exc()
            print("\nVerwende Standard-Schwellenwerte")
            # Setze Standard-Schwellenwerte
            self.MIN_DEFINITION_WORDS = 15
            self.MIN_EXAMPLES = 2
            self.SIMILARITY_THRESHOLD = 0.7
            self.MIN_SUBCATEGORIES = 2
            self.MAX_NAME_LENGTH = 50
            self.MIN_NAME_LENGTH = 3
            self.ENGLISH_WORDS = {'research', 'development', 'management', 'system', 'process', 'analysis'}
            self.messages = {}
            self.validation_cache = {}
            self.validation_stats = {'cache_hits': 0, 'cache_misses': 0, 'total_validations': 0}

    def _get_numeric_value(self, entries: Dict, key: str, default: float) -> float:
        """Hilft beim sicheren Extrahieren numerischer Werte."""
        try:
            value = entries.get(key)
            if value is not None:
                return float(value)
        except (ValueError, TypeError):
            print(f"Warnung: Ungültiger Wert für {key}, verwende Standard: {default}")
        return default

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Berechnet die Ähnlichkeit zwischen zwei Texten."""
        cache_key = f"{hash(text1)}_{hash(text2)}"
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
            
        # Konvertiere Texte zu Sets von Wörtern
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Berechne Jaccard-Ähnlichkeit
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Cache das Ergebnis
        self.similarity_cache[cache_key] = similarity
        return similarity

    def validate_category(self, category: CategoryDefinition) -> Tuple[bool, List[str]]:
        """
        Validiert eine einzelne Kategorie und cached das Ergebnis.
        
        Args:
            category: Zu validierende Kategorie
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        # Generiere Cache-Key
        cache_key = (
            category.name,
            category.definition,
            tuple(category.examples),
            tuple(category.rules),
            tuple(sorted(category.subcategories.items()))
        )
        
        # Prüfe Cache
        if cache_key in self.validation_cache:
            self.validation_stats['cache_hits'] += 1
            return self.validation_cache[cache_key]
            
        self.validation_stats['cache_misses'] += 1
        self.validation_stats['total_validations'] += 1
        
        issues = []
        
        # 1. Name-Validierung
        if len(category.name) < 3:
            issues.append("Name zu kurz (min. 3 Zeichen)")
        if len(category.name) > 50:
            issues.append("Name zu lang (max. 50 Zeichen)")
            
        # Prüfe auf englische Wörter
        english_words = {'research', 'development', 'management', 'system', 'process', 'analysis'}
        if any(word.lower() in english_words for word in category.name.split()):
            issues.append("Name enthält englische Begriffe")
        
        # 2. Definition-Validierung
        word_count = len(category.definition.split())
        if word_count < self.MIN_DEFINITION_WORDS:
            issues.append(f"Definition zu kurz ({word_count} Wörter, min. {self.MIN_DEFINITION_WORDS})")
            
        # 3. Beispiel-Validierung
        if len(category.examples) < self.MIN_EXAMPLES:
            issues.append(f"Zu wenige Beispiele ({len(category.examples)}, min. {self.MIN_EXAMPLES})")
            
        # 4. Regeln-Validierung
        if not category.rules:
            issues.append("Keine Kodierregeln definiert")
            
        # 5. Subkategorien-Validierung
        if len(category.subcategories) < self.MIN_SUBCATEGORIES:
            issues.append(f"Zu wenige Subkategorien ({len(category.subcategories)}, min. {self.MIN_SUBCATEGORIES})")
            
        # Cache und Return
        result = (len(issues) == 0, issues)
        self.validation_cache[cache_key] = result
        return result

    def validate_category_system(self, 
                               categories: Dict[str, CategoryDefinition]) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validiert das gesamte Kategoriensystem.
        
        Args:
            categories: Zu validierendes Kategoriensystem
            
        Returns:
            Tuple[bool, Dict[str, List[str]]]: (is_valid, {category_name: issues})
        """
        if not categories:
            return False, {"system": ["Leeres Kategoriensystem"]}
            
        system_issues = {}
        
        # 1. Validiere einzelne Kategorien
        for name, category in categories.items():
            is_valid, issues = self.validate_category(category)
            if not is_valid:
                system_issues[name] = issues

        # 2. Prüfe auf Überlappungen zwischen Kategorien
        for name1, cat1 in categories.items():
            for name2, cat2 in categories.items():
                if name1 >= name2:
                    continue
                    
                similarity = self._calculate_text_similarity(
                    cat1.definition,
                    cat2.definition
                )
                
                if similarity > self.SIMILARITY_THRESHOLD:
                    issue = f"Hohe Ähnlichkeit ({similarity:.2f}) mit {name2}"
                    if name1 in system_issues:
                        system_issues[name1].append(issue)
                    else:
                        system_issues[name1] = [issue]

        # 3. Prüfe Hierarchie und Struktur
        has_root_categories = any(not cat.subcategories for cat in categories.values())
        if not has_root_categories:
            system_issues["system"] = system_issues.get("system", []) + ["Keine Hauptkategorien gefunden"]
        
        # Validierungsergebnis
        is_valid = len(system_issues) == 0
        return is_valid, system_issues

    def _auto_enhance_category(self, category: CategoryDefinition) -> CategoryDefinition:
        """Versucht automatisch, eine unvollständige Kategorie zu verbessern."""
        try:
            enhanced = category

            # 1. Generiere fehlende Beispiele falls nötig
            if len(enhanced.examples) < self.MIN_EXAMPLES:
                # Extrahiere potenzielle Beispiele aus der Definition
                sentences = enhanced.definition.split('.')
                potential_examples = [s.strip() for s in sentences if 'z.B.' in s or 'beispielsweise' in s]
                
                # Füge gefundene Beispiele hinzu
                if potential_examples:
                    enhanced = enhanced.replace(
                        examples=list(set(enhanced.examples + potential_examples))
                    )

            # 2. Generiere grundlegende Kodierregeln falls keine vorhanden
            if not enhanced.rules:
                enhanced = enhanced.replace(rules=[
                    f"Kodiere Textstellen, die sich auf {enhanced.name} beziehen",
                    "Berücksichtige den Kontext der Aussage",
                    "Bei Unsicherheit dokumentiere die Gründe"
                ])

            # 3. Generiere Subkategorien aus der Definition falls keine vorhanden
            if len(enhanced.subcategories) < self.MIN_SUBCATEGORIES:
                # Suche nach Aufzählungen in der Definition
                potential_subcats = {}
                
                # Suche nach "wie" oder "beispielsweise" Aufzählungen
                for marker in ['wie', 'beispielsweise', 'etwa', 'insbesondere']:
                    if marker in enhanced.definition.lower():
                        parts = enhanced.definition.split(marker)[1].split('.')
                        if parts:
                            items = parts[0].split(',')
                            for item in items:
                                # Teile auch bei "und" oder "oder"
                                subitems = [x.strip() for x in item.split(' und ')]
                                subitems.extend([x.strip() for x in item.split(' oder ')])
                                
                                for subitem in subitems:
                                    # Bereinige und normalisiere
                                    cleaned = subitem.strip().strip('.')
                                    if len(cleaned) > 3 and not any(x in cleaned.lower() for x in ['z.b', 'etc', 'usw']):
                                        # Erstelle aussagekräftige Definition
                                        subcat_def = f"Aspekte bezüglich {cleaned} im Kontext von {enhanced.name}"
                                        potential_subcats[cleaned] = subcat_def

                if potential_subcats:
                    enhanced = enhanced.replace(
                        subcategories={**enhanced.subcategories, **potential_subcats}
                    )
                else:
                    # Fallback: Erstelle grundlegende Subkategorien
                    basic_subcats = {
                        f"Strukturelle {enhanced.name}": f"Strukturelle Aspekte von {enhanced.name}",
                        f"Prozessuale {enhanced.name}": f"Prozessbezogene Aspekte von {enhanced.name}",
                        f"Personelle {enhanced.name}": f"Personalbezogene Aspekte von {enhanced.name}"
                    }
                    enhanced = enhanced.replace(
                        subcategories={**enhanced.subcategories, **basic_subcats}
                    )

            print(f"\nKategorie '{enhanced.name}' automatisch verbessert:")
            if len(enhanced.examples) > len(category.examples):
                print(f"- {len(enhanced.examples) - len(category.examples)} neue Beispiele")
            if len(enhanced.rules) > len(category.rules):
                print(f"- {len(enhanced.rules) - len(category.rules)} neue Kodierregeln")
            if len(enhanced.subcategories) > len(category.subcategories):
                print(f"- {len(enhanced.subcategories) - len(category.subcategories)} neue Subkategorien")

            return enhanced

        except Exception as e:
            print(f"Fehler bei automatischer Kategorienverbesserung: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return category
        
    def get_validation_stats(self) -> Dict:
        """Gibt Statistiken zur Validierung zurück."""
        if self.validation_stats['total_validations'] > 0:
            cache_hit_rate = (
                self.validation_stats['cache_hits'] / 
                self.validation_stats['total_validations']
            )
        else:
            cache_hit_rate = 0.0
            
        return {
            'cache_hit_rate': cache_hit_rate,
            'total_validations': self.validation_stats['total_validations'],
            'cache_size': len(self.validation_cache)
        }
    
    def validate_category(self, category: CategoryDefinition) -> Tuple[bool, List[str]]:
        """Validiert eine einzelne Kategorie und cached das Ergebnis."""
        # Generiere Cache-Key
        cache_key = (
            category.name,
            category.definition,
            tuple(category.examples),
            tuple(category.rules),
            tuple(sorted(category.subcategories.items()))
        )
        
        # Prüfe Cache
        if cache_key in self.validation_cache:
            self.validation_stats['cache_hits'] += 1
            return self.validation_cache[cache_key]
            
        self.validation_stats['cache_misses'] += 1
        self.validation_stats['total_validations'] += 1
        
        # Versuche zuerst die Kategorie zu verbessern
        enhanced_category = self._auto_enhance_category(category)
        
        issues = []
        
        # Validiere die verbesserte Kategorie
        if len(enhanced_category.name) < self.MIN_NAME_LENGTH:
            issues.append(f"Name zu kurz (min. {self.MIN_NAME_LENGTH} Zeichen)")
        if len(enhanced_category.name) > self.MAX_NAME_LENGTH:
            issues.append(f"Name zu lang (max. {self.MAX_NAME_LENGTH} Zeichen)")
            
        if len(enhanced_category.definition.split()) < self.MIN_DEFINITION_WORDS:
            issues.append(f"Definition zu kurz ({len(enhanced_category.definition.split())} Wörter, min. {self.MIN_DEFINITION_WORDS})")
            
        if len(enhanced_category.examples) < self.MIN_EXAMPLES:
            issues.append(f"Zu wenige Beispiele ({len(enhanced_category.examples)}, min. {self.MIN_EXAMPLES})")
            
        if not enhanced_category.rules:
            issues.append("Keine Kodierregeln definiert")
            
        if len(enhanced_category.subcategories) < self.MIN_SUBCATEGORIES:
            issues.append(f"Zu wenige Subkategorien ({len(enhanced_category.subcategories)}, min. {self.MIN_SUBCATEGORIES})")
        
        # Cache und Return
        result = (len(issues) == 0, issues)
        self.validation_cache[cache_key] = result
        return result
    
    def clear_cache(self):
        """Leert den Validierungs-Cache."""
        self.validation_cache.clear()
        self.similarity_cache.clear()
        print("Validierungs-Cache geleert")

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
    
    def save_codebook(self, 
                        categories: Dict[str, CategoryDefinition], 
                        filename: str = "codebook_inductive.json") -> None:
            """Speichert das vollständige Codebook inkl. deduktiver und induktiver Kategorien"""
            try:
                codebook_data = {
                    "metadata": {
                        "created": datetime.now().isoformat(),
                        "version": "2.0",
                        "total_categories": len(categories),
                        "research_question": FORSCHUNGSFRAGE
                    },
                    "categories": {}
                }
                
                for name, category in categories.items():
                    codebook_data["categories"][name] = {
                        "definition": category.definition,
                        # Wandle examples in eine Liste um, falls es ein Set ist
                        "examples": list(category.examples) if isinstance(category.examples, set) else category.examples,
                        # Wandle rules in eine Liste um, falls es ein Set ist
                        "rules": list(category.rules) if isinstance(category.rules, set) else category.rules,
                        # Wandle subcategories in ein Dictionary um, falls nötig
                        "subcategories": dict(category.subcategories) if isinstance(category.subcategories, set) else category.subcategories,
                        "development_type": "deductive" if name in DEDUKTIVE_KATEGORIEN else "inductive",
                        "added_date": category.added_date,
                        "last_modified": category.modified_date
                    }
                
                output_path = os.path.join(self.output_dir, filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(codebook_data, f, indent=2, ensure_ascii=False)
                    
                print(f"\nInduktiv erweitertes Codebook gespeichert unter: {output_path}")
                
            except Exception as e:
                print(f"Fehler beim Speichern des Codebooks: {str(e)}")
                # Zusätzliche Fehlerdiagnose
                import traceback
                traceback.print_exc()

class CategoryCleaner:
    """Helferklasse zum Bereinigen problematischer Kategorien."""
    
    def __init__(self, validator: CategoryValidator):
        self.validator = validator

    async def clean_categories(self, 
                             categories: Dict[str, CategoryDefinition],
                             issues: Dict[str, List[str]]) -> Dict[str, CategoryDefinition]:
        """
        Versucht Validierungsprobleme automatisch zu beheben.
        
        Args:
            categories: Problematische Kategorien
            issues: Gefundene Probleme pro Kategorie
            
        Returns:
            Dict[str, CategoryDefinition]: Bereinigte Kategorien
        """
        cleaned = categories.copy()
        
        for category_name, category_issues in issues.items():
            if category_name == "system":
                continue  # Systemweite Probleme werden separat behandelt
                
            category = cleaned.get(category_name)
            if not category:
                continue
                
            # Behandle verschiedene Problemtypen
            for issue in category_issues:
                if "Definition zu kurz" in issue:
                    cleaned[category_name] = await self._enhance_definition(category)
                elif "englische Begriffe" in issue:
                    cleaned[category_name] = self._translate_category(category)
                elif "Hohe Ähnlichkeit" in issue:
                    # Extrahiere Namen der ähnlichen Kategorie
                    similar_to = issue.split("mit ")[-1]
                    if similar_to in cleaned:
                        cleaned = self._merge_similar_categories(
                            cleaned, 
                            category_name, 
                            similar_to
                        )
                # Weitere Problemtypen hier behandeln...
                
        return cleaned

    async def _enhance_definition(self, category: CategoryDefinition) -> CategoryDefinition:
        """Verbessert zu kurze Definitionen."""
        # Hier würde die Logik zur Definition-Verbesserung kommen
        # Könnte z.B. einen API-Call an GPT machen
        return category

    def _translate_category(self, category: CategoryDefinition) -> CategoryDefinition:
        """Übersetzt englische Begriffe ins Deutsche."""
        # Hier würde die Übersetzungslogik kommen
        return category

    def _merge_similar_categories(self,
                                categories: Dict[str, CategoryDefinition],
                                cat1: str,
                                cat2: str) -> Dict[str, CategoryDefinition]:
        """Führt ähnliche Kategorien zusammen."""
        # Hier würde die Merge-Logik kommen
        return categories
    
# Klasse: CategoryOptimizer
# ----------------------
class CategoryOptimizer:
    """
    Optionale Klasse für die Optimierung des Kategoriensystems durch:
    - Identifikation ähnlicher Kategorien
    - Vorschläge für mögliche Zusammenführungen
    - Automatische Deduplizierung wenn gewünscht
    """
    
    def __init__(self, config: Dict = None):
        # Konfiguration für Ähnlichkeitsschwellen
        if config and 'validation_config' in config:
            validation_config = config['validation_config']
            thresholds = validation_config.get('thresholds', {})
            self.similarity_threshold = thresholds.get('SIMILARITY_THRESHOLD', 0.8)
        else:
            self.similarity_threshold = 0.8
        
         # Initialisiere LLM Provider
        provider_name = CONFIG.get('MODEL_PROVIDER', 'openai')
        try:
            self.llm_provider = LLMProviderFactory.create_provider(provider_name)
            print(f"LLM Provider '{provider_name}' für Kategorienoptimierung initialisiert")
        except Exception as e:
            print(f"Fehler bei Provider-Initialisierung: {str(e)}")
            raise
            
        self.optimization_log = []
        
    def suggest_optimizations(self, categories: Dict[str, CategoryDefinition]) -> List[Dict]:
        """
        Analysiert das Kategoriensystem und schlägt mögliche Optimierungen vor.
        
        Returns:
            List[Dict]: Liste von Optimierungsvorschlägen mit:
            - type: 'merge' | 'split' | 'reorganize'
            - categories: Betroffene Kategorien
            - similarity: Ähnlichkeitsmaß
            - recommendation: Textuelle Empfehlung
        """
        suggestions = []
        
        # 1. Suche nach ähnlichen Kategorien
        for name1, cat1 in categories.items():
            for name2, cat2 in categories.items():
                if name1 >= name2:
                    continue
                    
                similarity = self._calculate_semantic_similarity(
                    cat1.definition,
                    cat2.definition
                )
                
                if similarity >= self.similarity_threshold:
                    suggestions.append({
                        'type': 'merge',
                        'categories': [name1, name2],
                        'similarity': similarity,
                        'recommendation': f"Kategorien '{name1}' und '{name2}' sind sehr ähnlich "
                                       f"(Übereinstimmung: {similarity:.2f}). Zusammenführung empfohlen."
                    })
        
        # 2. Suche nach zu großen Kategorien
        for name, category in categories.items():
            if len(category.subcategories) > 10:
                suggestions.append({
                    'type': 'split',
                    'categories': [name],
                    'subcategory_count': len(category.subcategories),
                    'recommendation': f"Kategorie '{name}' hat sehr viele Subkategorien "
                                   f"({len(category.subcategories)}). Aufteilung empfohlen."
                })
        
        return suggestions
        
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Berechnet die semantische Ähnlichkeit zwischen zwei Texten."""
        # Konvertiere zu Sets von Wörtern
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Berechne Jaccard-Ähnlichkeit
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def apply_optimization(self, 
                         categories: Dict[str, CategoryDefinition],
                         optimization: Dict) -> Dict[str, CategoryDefinition]:
        """
        Wendet eine spezifische Optimierung auf das Kategoriensystem an.
        
        Args:
            categories: Aktuelles Kategoriensystem
            optimization: Durchzuführende Optimierung
            
        Returns:
            Dict[str, CategoryDefinition]: Optimiertes Kategoriensystem
        """
        optimized = categories.copy()
        
        try:
            if optimization['type'] == 'merge':
                cat1, cat2 = optimization['categories']
                
                # Erstelle neue zusammengeführte Kategorie
                merged_name = f"{cat1}_{cat2}"
                merged_def = f"{categories[cat1].definition}\n\nZusätzlich: {categories[cat2].definition}"
                merged_examples = list(set(categories[cat1].examples + categories[cat2].examples))
                merged_subcats = {**categories[cat1].subcategories, **categories[cat2].subcategories}
                
                optimized[merged_name] = CategoryDefinition(
                    name=merged_name,
                    definition=merged_def,
                    examples=merged_examples,
                    rules=list(set(categories[cat1].rules + categories[cat2].rules)),
                    subcategories=merged_subcats,
                    added_date=min(categories[cat1].added_date, categories[cat2].added_date),
                    modified_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                # Entferne ursprüngliche Kategorien
                del optimized[cat1]
                del optimized[cat2]
                
                # Dokumentiere Optimierung
                self.optimization_log.append({
                    'type': 'merge',
                    'original_categories': [cat1, cat2],
                    'result_category': merged_name,
                    'timestamp': datetime.now().isoformat()
                })
                
            # Weitere Optimierungstypen hier implementieren...
            
        except Exception as e:
            print(f"Fehler bei Optimierung: {str(e)}")
            
        return optimized

# --- Klasse: RelevanceChecker ---
# Aufgabe: Zentrale Klasse für Relevanzprüfungen mit Caching und Batch-Verarbeitung
class RelevanceChecker:
    """
    Zentrale Klasse für Relevanzprüfungen mit Caching und Batch-Verarbeitung.
    Reduziert API-Calls durch Zusammenfassung mehrerer Segmente.
    """
    
    def __init__(self, model_name: str, batch_size: int = 5):
        self.model_name = model_name
        self.batch_size = batch_size

        # Hole Provider aus CONFIG
        provider_name = CONFIG.get('MODEL_PROVIDER', 'openai')  # Fallback zu OpenAI
        try:
            # Wir lassen den LLMProvider sich selbst um die Client-Initialisierung kümmern
            self.llm_provider = LLMProviderFactory.create_provider(provider_name)
        except Exception as e:
            print(f"Fehler bei Provider-Initialisierung: {str(e)}")
            raise
        
        # Cache für Relevanzprüfungen
        self.relevance_cache = {}
        self.relevance_details = {}
        
        # Tracking-Metriken
        self.total_segments = 0
        self.relevant_segments = 0
        self.api_calls = 0

    async def check_relevance_batch(self, segments: List[Tuple[str, str]]) -> Dict[str, bool]:
        """
        Prüft die Relevanz mehrerer Segmente in einem API-Call.
        
        Args:
            segments: Liste von (segment_id, text) Tupeln
            
        Returns:
            Dict[str, bool]: Mapping von segment_id zu Relevanz
        """
        # Filtere bereits gecachte Segmente
        uncached_segments = [
            (sid, text) for sid, text in segments 
            if sid not in self.relevance_cache
        ]
        
        if not uncached_segments:
            return {sid: self.relevance_cache[sid] for sid, _ in segments}

        try:
            # Erstelle formatierten Text für Batch-Verarbeitung
            segments_text = "\n\n=== SEGMENT BREAK ===\n\n".join(
                f"SEGMENT {i + 1}:\n{text}" 
                for i, (_, text) in enumerate(uncached_segments)
            )

            prompt = f"""
            Analysiere die Relevanz der folgenden Textsegmente für die Forschungsfrage:
            "{FORSCHUNGSFRAGE}"

            WICHTIG: Berücksichtige bei der Relevanzprüfung die spezifischen Merkmale verschiedener Textsorten:

            1. INHALTLICHE RELEVANZ - Nach Textsorte:

            INTERVIEWS/GESPRÄCHE:
            - Direkte Erfahrungsberichte und Schilderungen
            - Persönliche Einschätzungen und Bewertungen
            - Konkrete Beispiele aus der Praxis
            - Auch implizites Erfahrungswissen beachten

            DOKUMENTE/BERICHTE:
            - Faktische Informationen und Beschreibungen
            - Formale Regelungen und Vorgaben
            - Dokumentierte Abläufe und Prozesse
            - Strukturelle Rahmenbedingungen

            PROTOKOLLE/NOTIZEN:
            - Beobachtete Handlungen und Interaktionen
            - Situationsbeschreibungen
            - Dokumentierte Entscheidungen
            - Kontextinformationen zu Ereignissen

            2. QUALITÄTSKRITERIEN:

            AUSSAGEKRAFT:
            - Enthält das Segment konkrete, gehaltvolle Information?
            - Geht es über Allgemeinplätze hinaus?
            - Ist die Information präzise genug für die Analyse?

            ANWENDBARKEIT:
            - Lässt sich die Information auf die Forschungsfrage beziehen?
            - Ermöglicht sie Erkenntnisse zu den Kernaspekten?
            - Trägt sie zum Verständnis relevanter Zusammenhänge bei?

            KONTEXTBEDEUTUNG:
            - Ist der Kontext wichtig für das Verständnis?
            - Hilft er bei der Interpretation anderer Informationen?
            - Ist er notwendig für die Einordnung der Aussagen?

            TEXTSEGMENTE:
            {segments_text}

            Antworte NUR mit einem JSON-Objekt:
            {{
                "segment_results": [
                    {{
                        "segment_number": 1,
                        "is_relevant": true/false,
                        "confidence": 0.0-1.0,
                        "text_type": "interview|dokument|protokoll|andere",
                        "key_aspects": ["konkrete", "für", "die", "Forschungsfrage", "relevante", "Aspekte"],
                        "justification": "Begründung der Relevanz unter Berücksichtigung der Textsorte"
                    }},
                    ...
                ]
            }}

            WICHTIGE HINWEISE:
            - Berücksichtige die spezifischen Merkmale der jeweiligen Textsorte
            - Beachte unterschiedliche Formen relevanter Information (explizit/implizit)
            - Prüfe den Informationsgehalt im Kontext der Textsorte
            - Bewerte die Relevanz entsprechend der textsortentypischen Merkmale
            - Bei Unsicherheit (confidence < 0.7) als nicht relevant markieren
            - Dokumentiere die Begründung mit Bezug zur Textsorte
            """

            input_tokens = estimate_tokens(prompt)

            # Ein API-Call für alle Segmente
            self.api_calls += 1
            response = await self.llm_provider.create_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Verarbeite Response mit Wrapper
            llm_response = LLMResponse(response)
            results = json.loads(llm_response.content)
            
            output_tokens = estimate_tokens(response.choices[0].message.content)
            token_counter.add_tokens(input_tokens, output_tokens)

            # Verarbeite Ergebnisse und aktualisiere Cache
            relevance_results = {}
            for i, (segment_id, _) in enumerate(uncached_segments):
                segment_result = results['segment_results'][i]
                is_relevant = segment_result['is_relevant']
                
                # Cache-Aktualisierung
                self.relevance_cache[segment_id] = is_relevant
                self.relevance_details[segment_id] = {
                    'confidence': segment_result['confidence'],
                    'key_aspects': segment_result['key_aspects']
                }
                
                relevance_results[segment_id] = is_relevant
                
                # Tracking-Aktualisierung
                self.total_segments += 1
                if is_relevant:
                    self.relevant_segments += 1

            # Kombiniere Cache und neue Ergebnisse
            return {
                sid: self.relevance_cache[sid] 
                for sid, _ in segments
            }

        except Exception as e:
            print(f"Fehler bei Batch-Relevanzprüfung: {str(e)}")
            # Fallback: Markiere alle als relevant bei Fehler
            return {sid: True for sid, _ in segments}

    def get_relevance_details(self, segment_id: str) -> Optional[Dict]:
        """Gibt detaillierte Relevanzinformationen für ein Segment zurück."""
        return self.relevance_details.get(segment_id)

    def get_statistics(self) -> Dict:
        """Gibt Statistiken zur Relevanzprüfung zurück."""
        return {
            'total_segments': self.total_segments,
            'relevant_segments': self.relevant_segments,
            'relevance_rate': self.relevant_segments / max(1, self.total_segments),
            'api_calls': self.api_calls,
            'cache_size': len(self.relevance_cache)
        }

    
# --- Klasse: IntegratedAnalysisManager ---
# Aufgabe: Integriert die verschiedenen Analysephasen in einem zusammenhängenden Prozess
class IntegratedAnalysisManager:
    def __init__(self, config: Dict):
        # Bestehende Initialisierung
        self.config = config
        self.history = DevelopmentHistory(config['OUTPUT_DIR'])

        # Zentrale Validierung hinzufügen
        self.validator = CategoryValidator(config)
        self.cleaner = CategoryCleaner(self.validator)

        # Initialize merge handling with config
        self.category_optimizer = CategoryOptimizer(config)

        # Batch Size aus Config
        self.batch_size = config.get('BATCH_SIZE', 5) 
        
        # Zentrale Relevanzprüfung
        self.relevance_checker = RelevanceChecker(
            model_name=config['MODEL_NAME'],
            batch_size=self.batch_size
        )
    
        self.saturation_checker = SaturationChecker(config, self.history)
        self.inductive_coder = InductiveCoder(
            model_name=config['MODEL_NAME'],
            history=self.history,
            output_dir=config['OUTPUT_DIR']
        )

        self.deductive_coders = [
            DeductiveCoder(
                config['MODEL_NAME'], 
                coder_config['temperature'],
                coder_config['coder_id']
            )
            for coder_config in config['CODER_SETTINGS']
        ]
        
        # Tracking-Variablen
        self.processed_segments = set()
        self.coding_results = []
        self.analysis_log = [] 
        self.performance_metrics = {
            'batch_processing_times': [],
            'coding_times': [],
            'category_changes': []
        }

    async def _get_next_batch(self, 
                           segments: List[Tuple[str, str]], 
                           batch_size: float) -> List[Tuple[str, str]]:
        """
        Bestimmt den nächsten zu analysierenden Batch.
        
        Args:
            segments: Liste aller Segmente
            batch_size_percentage: Batch-Größe als Prozentsatz
            
        Returns:
            List[Tuple[str, str]]: Nächster Batch von Segmenten
        """
        remaining_segments = [
            seg for seg in segments 
            if seg[0] not in self.processed_segments
        ]
        
        if not remaining_segments:
            return []
            
        batch_size = max(1, batch_size)
        return remaining_segments[:batch_size]
    
    
    async def _process_batch_inductively(self, 
                                    batch: List[Tuple[str, str]], 
                                    current_categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        Entwickelt neue Kategorien aus relevanten Textsegmenten.
        """
        try:
            # Prüfe Relevanz für ganzen Batch auf einmal
            relevance_results = await self.relevance_checker.check_relevance_batch(batch)

            # Filtere relevante Segmente
            relevant_segments = [
                text for (segment_id, text) in batch 
                if relevance_results.get(segment_id, False)
            ]

            if not relevant_segments:
                print("   ℹ️ Keine relevanten Segmente für induktive Kategorienentwicklung")
                return {}

            print(f"\nEntwickle Kategorien aus {len(relevant_segments)} relevanten Segmenten")
            
            
            # Induktive Kategorienentwicklung
            new_categories = await self.inductive_coder.develop_category_system(relevant_segments)
            
            if new_categories:
                print("\nNeue/aktualisierte Kategorien gefunden:")
                
                for cat_name, category in new_categories.items():
                    print(f"\n🆕 Neue Hauptkategorie: {cat_name}")
                    print(f"   Definition: {category.definition[:100]}...")
                    # if category.subcategories:
                    #     print(f"   Subkategorien:")
                    #     for sub_name in category.subcategories:
                    #         print(f"   - {sub_name}")
            else:
                print("   ℹ️ Keine neuen Kategorien in diesem Batch identifiziert")
            
            return new_categories
            
        except Exception as e:
            print(f"Fehler bei induktiver Kategorienentwicklung: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return {}
        
    async def _code_batch_deductively(self,
                                    batch: List[Tuple[str, str]],
                                    categories: Dict[str, CategoryDefinition]) -> List[Dict]:
        """Führt die deduktive Kodierung mit optimierter Relevanzprüfung durch."""
        batch_results = []
        batch_metrics = {
            'new_aspects': [],
            'category_coverage': {},
            'coding_confidence': []
        }
        
        # Prüfe Relevanz für ganzen Batch
        relevance_results = await self.relevance_checker.check_relevance_batch(batch)
        
        for segment_id, text in batch:
            print(f"\n--------------------------------------------------------")
            print(f"🔎 Verarbeite Segment {segment_id}")
            print(f"--------------------------------------------------------\n")
            # Nutze gespeicherte Relevanzprüfung
            if not relevance_results.get(segment_id, False):
                print(f"Segment wurde als nicht relevant markiert - wird übersprungen")
                
                # Erstelle "Nicht kodiert" Ergebnis
                for coder in self.deductive_coders:
                    result = {
                        'segment_id': segment_id,
                        'coder_id': coder.coder_id,
                        'category': "Nicht kodiert",
                        'subcategories': [],
                        'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                        'justification': "Nicht relevant für Forschungsfrage",
                        'text': text
                    }
                    batch_results.append(result)
                continue

            # Verarbeite relevante Segmente mit allen Codierern
            print(f"Kodiere relevantes Segment mit {len(self.deductive_coders)} Codierern")
            for coder in self.deductive_coders:
                try:
                    coding = await coder.code_chunk(text, categories)
                    
                    if coding and isinstance(coding, CodingResult):
                        result = {
                            'segment_id': segment_id,
                            'coder_id': coder.coder_id,
                            'category': coding.category,
                            'subcategories': list(coding.subcategories),
                            'confidence': coding.confidence,
                            'justification': coding.justification,
                            'text': text,
                            'paraphrase': coding.paraphrase,
                            'keywords': coding.keywords
                        }
                        print(f"  ✓ Kodierung von {coder.coder_id}: {result['category']}")
                        batch_results.append(result)
                    else:
                        print(f"  ✗ Keine gültige Kodierung von {coder.coder_id}")
                        
                except Exception as e:
                    print(f"  ✗ Fehler bei Kodierer {coder.coder_id}: {str(e)}")
                    continue

            self.processed_segments.add(segment_id)

        #  Aktualisiere Sättigungsmetriken
        saturation_metrics = {
            'new_aspects_found': len(batch_metrics['new_aspects']) > 0,
            'categories_sufficient': len(batch_metrics['category_coverage']) >= len(categories) * 0.8,
            'theoretical_coverage': len(batch_metrics['category_coverage']) / len(categories),
            'avg_confidence': sum(batch_metrics['coding_confidence']) / len(batch_metrics['coding_confidence']) if batch_metrics['coding_confidence'] else 0
        }
        
        # Füge Metriken zum SaturationChecker hinzu
        self.saturation_checker.add_saturation_metrics(saturation_metrics)

        return batch_results

    async def analyze_material(self, 
                                chunks: Dict[str, List[str]], 
                                initial_categories: Dict,
                                skip_inductive: bool = False,
                                batch_size: Optional[int] = None) -> Tuple[Dict, List]:
        """Verbesserte Hauptanalyse mit expliziter Kategorienintegration."""
        try:
            self.start_time = datetime.now()
            print(f"\nAnalyse gestartet um {self.start_time.strftime('%H:%M:%S')}")
            
            # Wichtig: Kopie des initialen Systems erstellen
            current_categories = initial_categories.copy()
            all_segments = self._prepare_segments(chunks)
            total_segments = len(all_segments)
            
            # Reset Tracking-Variablen
            self.coding_results = []
            self.processed_segments = set()
            
            if batch_size is None:
                batch_size = CONFIG.get('BATCH_SIZE', 5) 
            total_batches = 0
            
            print(f"Verarbeite {total_segments} Segmente mit Batch-Größe {batch_size}...")
            self.history.log_analysis_start(total_segments, len(initial_categories))
            
            while True:
                batch = await self._get_next_batch(all_segments, batch_size)
                if not batch:
                    break
                    
                total_batches += 1
                print(f"\nBatch {total_batches}: {len(batch)} Segmente")

                # Korrigierte Berechnung des Material-Prozentsatzes
                material_percentage = (len(self.processed_segments) / total_segments) * 100
                print(f"Verarbeiteter Materialanteil: {material_percentage:.1f}%")
                
                batch_start = time.time()
                
                try:
                    # Relevanzprüfung...
                    relevance_results = await self.relevance_checker.check_relevance_batch(batch)
                    self.processed_segments.update(sid for sid, _ in batch)
                    
                    # Relevante Segmente für induktive Analyse
                    relevant_batch = [
                        (segment_id, text) for segment_id, text in batch 
                        if relevance_results.get(segment_id, False)
                    ]
                    
                    print(f"Relevanzprüfung: {len(relevant_batch)} von {len(batch)} Segmenten relevant")
                    
                    # Induktive Analyse wenn nicht übersprungen
                    if not skip_inductive and relevant_batch:
                        print(f"\nStarte induktive Kategorienentwicklung für {len(relevant_batch)} relevante Segmente...")
                        
                        new_categories = await self._process_batch_inductively(
                            relevant_batch, 
                            current_categories
                        )
                        
                        if new_categories:
                            print("\nNeue/aktualisierte Kategorien gefunden:")
                            
                            # Kategorien integrieren
                            current_categories = self._merge_category_systems(
                                current_categories,
                                new_categories
                            )
                            
                            # Debug-Ausgabe des aktualisierten Systems
                            # print("\nAktuelles Kategoriensystem:")
                            # for name, cat in current_categories.items():
                            #     print(f"\n- {name}:")
                            #     print(f"  Definition: {cat.definition[:100]}...")
                            #     if cat.subcategories:
                            #         print("  Subkategorien:")
                            #         for sub_name in cat.subcategories:
                            #             print(f"    • {sub_name}")
                            
                            # Aktualisiere ALLE Kodierer mit dem neuen System
                            for coder in self.deductive_coders:
                                await coder.update_category_system(current_categories)
                            print(f"\nAlle Kodierer mit aktuellem System ({len(current_categories)} Kategorien) aktualisiert")
                    
                    # Deduktive Kodierung für alle Segmente
                    print("\nStarte deduktive Kodierung...")
                    batch_results = await self._code_batch_deductively(batch, current_categories)
                    self.coding_results.extend(batch_results)
                    
                    # Performance-Tracking
                    batch_time = time.time() - batch_start
                    self.performance_metrics['batch_processing_times'].append(batch_time)
                    avg_time_per_segment = batch_time / len(batch)
                    self.performance_metrics['coding_times'].append(avg_time_per_segment)
                    
                    # Fortschritt
                    material_percentage = (len(self.processed_segments) / total_segments) * 100
                    print(f"\nFortschritt: {material_percentage:.1f}%")
                    print(f"Kodierungen gesamt: {len(self.coding_results)}")
                    print(f"Durchschnittliche Zeit pro Segment: {avg_time_per_segment:.2f}s")
                    
                    # Relevanzstatistiken
                    relevance_stats = self.relevance_checker.get_statistics()
                    print("\nRelevanzstatistiken:")
                    print(f"- Relevante Segmente: {relevance_stats['relevant_segments']}/{relevance_stats['total_segments']}")
                    print(f"- Relevanzrate: {relevance_stats['relevance_rate']*100:.1f}%")
                    print(f"- API-Calls: {relevance_stats['api_calls']}")
                    
                    # Sättigungsprüfung nur bei aktivierter induktiver Analyse
                    if not skip_inductive:
                        # Hole Sättigungsmetriken vom SaturationChecker
                        is_saturated, saturation_metrics = self.saturation_checker.check_saturation(
                            current_categories=current_categories,
                            coded_segments=self.coding_results,
                            material_percentage=material_percentage
                        )
                        
                        if is_saturated:
                            print("\nSättigung erreicht!")
                            if saturation_metrics:
                                print("Sättigungsmetriken:")
                                for key, value in saturation_metrics.items():
                                    print(f"- {key}: {value}")
                            break

                    # Status-Update für History
                    self._log_iteration_status(
                        material_percentage=material_percentage,
                        saturation_metrics=saturation_metrics if not skip_inductive else None,
                        num_results=len(batch_results)
                    )
                    
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von Batch {total_batches}: {str(e)}")
                    print("Details:")
                    traceback.print_exc()
                    continue
            
            # Finaler Analysebericht
            final_metrics = self._finalize_analysis(
                final_categories=current_categories,
                initial_categories=initial_categories
            )
            
            # Abschluss
            self.end_time = datetime.now()
            processing_time = (self.end_time - self.start_time).total_seconds()
            
            print(f"\nAnalyse abgeschlossen:")
            print(f"- {len(self.processed_segments)} Segmente verarbeitet")
            print(f"- {len(self.coding_results)} Kodierungen erstellt")
            print(f"- {processing_time:.1f} Sekunden Verarbeitungszeit")
            print(f"- Durchschnittliche Zeit pro Segment: {processing_time/total_segments:.2f}s")
            
            if not skip_inductive:
                print(f"- Kategorienentwicklung:")
                print(f"  • Initial: {len(initial_categories)} Kategorien")
                print(f"  • Final: {len(current_categories)} Kategorien")
                print(f"  • Neu entwickelt: {len(current_categories) - len(initial_categories)} Kategorien")
            
            # Dokumentiere Abschluss
            self.history.log_analysis_completion(
                final_categories=current_categories,
                total_time=processing_time,
                total_codings=len(self.coding_results)
            )
            
            # Abschließende Relevanzstatistiken
            final_stats = self.relevance_checker.get_statistics()
            print("\nFinale Relevanzstatistiken:")
            print(f"- API-Calls gespart: {total_segments - final_stats['api_calls']}")
            print(f"- Cache-Nutzung: {final_stats['cache_size']} Einträge")
            
            return current_categories, self.coding_results
                
        except Exception as e:
            self.end_time = datetime.now()
            print(f"Fehler in der Analyse: {str(e)}")
            print("Details:")
            traceback.print_exc()
            raise
    
    def _merge_category_systems(self, 
                            current: Dict[str, CategoryDefinition], 
                            new: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        Führt bestehendes und neues Kategoriensystem zusammen.
        
        Args:
            current: Bestehendes Kategoriensystem
            new: Neue Kategorien
            
        Returns:
            Dict[str, CategoryDefinition]: Zusammengeführtes System
        """
        merged = current.copy()
        
        for name, category in new.items():
            if name not in merged:
                # Komplett neue Kategorie
                merged[name] = category
                print(f"\n🆕 Neue Hauptkategorie hinzugefügt: {name}")
                print(f"   Definition: {category.definition[:100]}...")
                if category.subcategories:
                    print("   Subkategorien:")
                    for sub_name in category.subcategories.keys():
                        print(f"   - {sub_name}")
            else:
                # Bestehende Kategorie aktualisieren
                existing = merged[name]
                
                # Sammle Änderungen für Debug-Ausgabe
                changes = []
                
                # Prüfe auf neue/geänderte Definition
                new_definition = category.definition
                if len(new_definition) > len(existing.definition):
                    changes.append("Definition aktualisiert")
                else:
                    new_definition = existing.definition
                
                # Kombiniere Beispiele
                new_examples = list(set(existing.examples) | set(category.examples))
                if len(new_examples) > len(existing.examples):
                    changes.append(f"{len(new_examples) - len(existing.examples)} neue Beispiele")
                
                # Kombiniere Regeln
                new_rules = list(set(existing.rules) | set(category.rules))
                if len(new_rules) > len(existing.rules):
                    changes.append(f"{len(new_rules) - len(existing.rules)} neue Regeln")
                
                # Neue Subkategorien
                new_subcats = {**existing.subcategories, **category.subcategories}
                if len(new_subcats) > len(existing.subcategories):
                    changes.append(f"{len(new_subcats) - len(existing.subcategories)} neue Subkategorien")
                
                # Erstelle aktualisierte Kategorie
                merged[name] = CategoryDefinition(
                    name=name,
                    definition=new_definition,
                    examples=new_examples,
                    rules=new_rules,
                    subcategories=new_subcats,
                    added_date=existing.added_date,
                    modified_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                if changes:
                    print(f"\n📝 Kategorie '{name}' aktualisiert:")
                    for change in changes:
                        print(f"   - {change}")
        
        return merged

    

    def _prepare_segments(self, chunks: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """
        Bereitet die Segmente für die Analyse vor.
        
        Args:
            chunks: Dictionary mit Dokumenten-Chunks
            
        Returns:
            List[Tuple[str, str]]: Liste von (segment_id, text) Tupeln
        """
        segments = []
        for doc_name, doc_chunks in chunks.items():
            for chunk_idx, chunk in enumerate(doc_chunks):
                segment_id = f"{doc_name}_chunk_{chunk_idx}"
                segments.append((segment_id, chunk))
        return segments


    async def _merge_categories(self,
                            current_cats: Dict,
                            new_cats: Dict) -> Dict:
        """
        Führt bestehende und neue Kategorien zusammen.
        
        Args:
            current_cats: Bestehendes Kategoriensystem
            new_cats: Neue Kategorien
            
        Returns:
            Dict: Zusammengeführtes Kategoriensystem
        """
        try:
            # Basiskopie der aktuellen Kategorien
            merged = current_cats.copy()
            
            # Verarbeite neue Kategorien
            for name, category in new_cats.items():
                if name in merged:
                    # Update bestehende Kategorie
                    current_cat = merged[name]
                    merged[name] = CategoryDefinition(
                        name=name,
                        definition=current_cat.definition,
                        examples=list(set(current_cat.examples + category.examples)),
                        rules=list(set(current_cat.rules + category.rules)),
                        subcategories={**current_cat.subcategories, **category.subcategories},
                        added_date=current_cat.added_date,
                        modified_date=datetime.now().strftime("%Y-%m-%d")
                    )
                else:
                    # Füge neue Kategorie hinzu
                    merged[name] = category
            
            return merged
            
        except Exception as e:
            print(f"Fehler beim Zusammenführen der Kategorien: {str(e)}")
            return current_cats

    def _find_similar_category(self, 
                                category: CategoryDefinition,
                                existing_categories: Dict[str, CategoryDefinition]) -> Optional[str]:
        """
        Findet ähnliche existierende Kategorien basierend auf Namen und Definition.
        
        Args:
            category: Zu prüfende Kategorie
            existing_categories: Bestehendes Kategoriensystem
            
        Returns:
            Optional[str]: Name der ähnlichsten Kategorie oder None
        """
        try:
            best_match = None
            highest_similarity = 0.0
            
            for existing_name, existing_cat in existing_categories.items():
                # Berechne Ähnlichkeit basierend auf verschiedenen Faktoren
                
                # 1. Name-Ähnlichkeit (gewichtet: 0.3)
                name_similarity = self._calculate_text_similarity(
                    category.name.lower(),
                    existing_name.lower()
                ) * 0.3
                
                # 2. Definitions-Ähnlichkeit (gewichtet: 0.5)
                definition_similarity = self._calculate_text_similarity(
                    category.definition,
                    existing_cat.definition
                ) * 0.5
                
                # 3. Subkategorien-Überlappung (gewichtet: 0.2)
                subcats1 = set(category.subcategories.keys())
                subcats2 = set(existing_cat.subcategories.keys())
                if subcats1 and subcats2:
                    subcat_overlap = len(subcats1 & subcats2) / len(subcats1 | subcats2)
                else:
                    subcat_overlap = 0
                subcat_similarity = subcat_overlap * 0.2
                
                # Gesamtähnlichkeit
                total_similarity = name_similarity + definition_similarity + subcat_similarity
                
                # Debug-Ausgabe für hohe Ähnlichkeiten
                if total_similarity > 0.5:
                    print(f"\nÄhnlichkeitsprüfung für '{category.name}' und '{existing_name}':")
                    print(f"- Name-Ähnlichkeit: {name_similarity:.2f}")
                    print(f"- Definitions-Ähnlichkeit: {definition_similarity:.2f}")
                    print(f"- Subkategorien-Überlappung: {subcat_similarity:.2f}")
                    print(f"- Gesamt: {total_similarity:.2f}")
                
                # Update beste Übereinstimmung
                if total_similarity > highest_similarity:
                    highest_similarity = total_similarity
                    best_match = existing_name
            
            # Nur zurückgeben wenn Ähnlichkeit hoch genug
            if highest_similarity > 0.7:  # Schwellenwert für Ähnlichkeit
                print(f"\n⚠ Hohe Ähnlichkeit ({highest_similarity:.2f}) gefunden:")
                print(f"- Neue Kategorie: {category.name}")
                print(f"- Existierende Kategorie: {best_match}")
                return best_match
                
            return None
            
        except Exception as e:
            print(f"Fehler bei Ähnlichkeitsprüfung: {str(e)}")
            return None

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Berechnet die Ähnlichkeit zwischen zwei Texten mit Caching."""
        cache_key = f"{hash(text1)}_{hash(text2)}"
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
            
        # Konvertiere Texte zu Sets von Wörtern
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Berechne Jaccard-Ähnlichkeit
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Cache das Ergebnis
        self.similarity_cache[cache_key] = similarity
        self.validation_stats['similarity_calculations'] += 1
        
        return similarity

    def _auto_enhance_category(self, category: CategoryDefinition) -> CategoryDefinition:
        """Versucht automatisch, eine unvollständige Kategorie zu verbessern."""
        try:
            enhanced = category

            # 1. Generiere fehlende Beispiele falls nötig
            if len(enhanced.examples) < self.MIN_EXAMPLES:
                # Extrahiere potenzielle Beispiele aus der Definition
                sentences = enhanced.definition.split('.')
                potential_examples = [s.strip() for s in sentences if 'z.B.' in s or 'beispielsweise' in s]
                
                # Füge gefundene Beispiele hinzu
                if potential_examples:
                    enhanced = enhanced._replace(
                        examples=list(set(enhanced.examples + potential_examples))
                    )

            # 2. Generiere grundlegende Kodierregeln falls keine vorhanden
            if not enhanced.rules:
                enhanced = enhanced._replace(rules=[
                    f"Kodiere Textstellen, die sich auf {enhanced.name} beziehen",
                    "Berücksichtige den Kontext der Aussage",
                    "Bei Unsicherheit dokumentiere die Gründe"
                ])

            # 3. Generiere Subkategorien aus der Definition falls keine vorhanden
            if len(enhanced.subcategories) < self.MIN_SUBCATEGORIES:
                # Suche nach Aufzählungen in der Definition
                if 'wie' in enhanced.definition:
                    parts = enhanced.definition.split('wie')[1].split(',')
                    potential_subcats = []
                    for part in parts:
                        if 'und' in part:
                            potential_subcats.extend(part.split('und'))
                        else:
                            potential_subcats.append(part)
                    
                    # Bereinige und füge Subkategorien hinzu
                    cleaned_subcats = {
                        subcat.strip().strip('.'): ""
                        for subcat in potential_subcats
                        if len(subcat.strip()) > 3
                    }
                    
                    if cleaned_subcats:
                        enhanced = enhanced._replace(
                            subcategories={**enhanced.subcategories, **cleaned_subcats}
                        )

            return enhanced

        except Exception as e:
            print(f"Fehler bei automatischer Kategorienverbesserung: {str(e)}")
            return category
        
    def _normalize_text(self, text: str) -> str:
        """
        Normalisiert Text für Vergleiche.
        
        Args:
            text: Zu normalisierender Text
            
        Returns:
            str: Normalisierter Text
        """
        # Zu Kleinbuchstaben
        text = text.lower()
        
        # Entferne Sonderzeichen
        text = re.sub(r'[^\w\s]', '', text)
        
        # Entferne Stoppwörter
        stop_words = {'und', 'oder', 'der', 'die', 'das', 'in', 'im', 'für', 'bei'}
        words = text.split()
        words = [w for w in words if w not in stop_words]
        
        return ' '.join(words)

    def _merge_category_definitions(self, 
                                original: CategoryDefinition,
                                new: CategoryDefinition) -> CategoryDefinition:
        """
        Führt zwei Kategoriendefinitionen zusammen.
        
        Args:
            original: Ursprüngliche Kategorie
            new: Neue Kategorie
            
        Returns:
            CategoryDefinition: Zusammengeführte Kategorie
        """
        try:
            # Kombiniere Definitionen
            combined_def = f"{original.definition}\n\nErgänzung: {new.definition}"
            
            # Kombiniere Beispiele
            combined_examples = list(set(original.examples + new.examples))
            
            # Kombiniere Subkategorien
            combined_subcats = {**original.subcategories, **new.subcategories}
            
            # Erstelle neue CategoryDefinition
            return CategoryDefinition(
                name=original.name,  # Behalte ursprünglichen Namen
                definition=combined_def,
                examples=combined_examples,
                rules=original.rules,  # Behalte ursprüngliche Regeln
                subcategories=combined_subcats,
                added_date=original.added_date,
                modified_date=datetime.now().strftime("%Y-%m-%d")
            )
            
        except Exception as e:
            print(f"Fehler beim Zusammenführen der Kategorien: {str(e)}")
            return original

    def _validate_and_integrate_categories(self, 
                                         existing_categories: Dict[str, CategoryDefinition],
                                         new_categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """Verbesserte Validierung und Integration von Kategorien."""
        try:
            # Validiere neue Kategorien einzeln
            valid_new_categories = {}
            for name, category in new_categories.items():
                is_valid, issues = self.validator.validate_category(category)
                if is_valid:
                    valid_new_categories[name] = category
                else:
                    print(f"\nKategorie '{name}' nicht valide:")
                    for issue in issues:
                        print(f"- {issue}")

            # Integriere valide neue Kategorien
            integrated = existing_categories.copy()
            integrated.update(valid_new_categories)

            # Validiere Gesamtsystem
            is_valid, system_issues = self.validator.validate_category_system(integrated)
            if not is_valid:
                print("\nProbleme nach Integration:")
                for category, issues in system_issues.items():
                    print(f"\n{category}:")
                    for issue in issues:
                        print(f"- {issue}")
                
                # Hier könnte zusätzliche Bereinigungslogik folgen...

            # Zeige Validierungsstatistiken
            stats = self.validator.get_validation_stats()
            print("\nValidierungsstatistiken:")
            print(f"- Cache-Trefferrate: {stats['cache_hit_rate']*100:.1f}%")
            print(f"- Validierungen gesamt: {stats['total_validations']}")

            return integrated

        except Exception as e:
            print(f"Fehler bei Kategorienintegration: {str(e)}")
            return existing_categories

    def _prepare_segments(self, chunks: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """
        Bereitet die Segmente für die Analyse vor.
        
        Args:
            chunks: Dictionary mit Dokumenten-Chunks
            
        Returns:
            List[Tuple[str, str]]: Liste von (segment_id, text) Tupeln
        """
        segments = []
        for doc_name, doc_chunks in chunks.items():
            for chunk_idx, chunk in enumerate(doc_chunks):
                segment_id = f"{doc_name}_chunk_{chunk_idx}"
                segments.append((segment_id, chunk))
        return segments

    def _update_tracking(self,
                        batch_results: List[Dict],
                        processing_time: float,
                        batch_size: int) -> None:
        """
        Aktualisiert die Performance-Metriken.
        
        Args:
            batch_results: Ergebnisse des Batches
            processing_time: Verarbeitungszeit
            batch_size: Größe des Batches
        """
        self.coding_results.extend(batch_results)
        self.performance_metrics['batch_processing_times'].append(processing_time)
        
        # Berechne durchschnittliche Zeit pro Segment
        avg_time_per_segment = processing_time / batch_size
        self.performance_metrics['coding_times'].append(avg_time_per_segment)

    def _log_iteration_status(self,
                            material_percentage: float,
                            saturation_metrics: Dict,
                            num_results: int) -> None:
        """
        Protokolliert den Status der aktuellen Iteration.
        
        Args:
            material_percentage: Prozentsatz des verarbeiteten Materials
            saturation_metrics: Metriken der Sättigungsprüfung
            num_results: Anzahl der Kodierungen
        """
        try:
            # Erstelle Status-Dictionary
            status = {
                'timestamp': datetime.now().isoformat(),
                'material_processed': material_percentage,
                'saturation_metrics': saturation_metrics,
                'results_count': num_results,
                'processing_time': self.performance_metrics['batch_processing_times'][-1] if self.performance_metrics['batch_processing_times'] else 0
            }
            
            # Füge Status zum Log hinzu
            self.analysis_log.append(status)
            
            # Debug-Ausgabe für wichtige Metriken
            print("\nIterations-Status:")
            print(f"- Material verarbeitet: {material_percentage:.1f}%")
            print(f"- Neue Kodierungen: {num_results}")
            print(f"- Verarbeitungszeit: {status['processing_time']:.2f}s")
            if saturation_metrics:
                print("- Sättigungsmetriken:")
                for key, value in saturation_metrics.items():
                    print(f"  • {key}: {value}")
        except Exception as e:
            print(f"Warnung: Fehler beim Logging des Iterationsstatus: {str(e)}")
            # Fehler beim Logging sollte die Hauptanalyse nicht unterbrechen

    def _finalize_analysis(self,
                          final_categories: Dict,
                          initial_categories: Dict) -> Tuple[Dict, List]:
        """
        Schließt die Analyse ab und bereitet die Ergebnisse vor.
        
        Args:
            final_categories: Finales Kategoriensystem
            initial_categories: Initiales Kategoriensystem
            
        Returns:
            Tuple[Dict, List]: (Finales Kategoriensystem, Kodierungsergebnisse)
        """
        # Berechne finale Statistiken
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        avg_time_per_segment = total_time / len(self.processed_segments)
        
        print("\nAnalyse abgeschlossen:")
        print(f"- Gesamtzeit: {total_time:.2f}s")
        print(f"- Durchschnittliche Zeit pro Segment: {avg_time_per_segment:.2f}s")
        print(f"- Verarbeitete Segmente: {len(self.processed_segments)}")
        print(f"- Finale Kategorien: {len(final_categories)}")
        print(f"- Gesamtanzahl Kodierungen: {len(self.coding_results)}")
        
        return final_categories, self.coding_results

    def get_analysis_report(self) -> Dict:
        """Erstellt einen detaillierten Analysebericht."""
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_segments': len(self.processed_segments),
            'total_codings': len(self.coding_results),
            'performance_metrics': {
                'avg_batch_time': (
                    sum(self.performance_metrics['batch_processing_times']) / 
                    len(self.performance_metrics['batch_processing_times'])
                ) if self.performance_metrics['batch_processing_times'] else 0,
                'total_batches': len(self.performance_metrics['batch_processing_times']),
                'coding_distribution': self._get_coding_distribution()
            },
            'coding_summary': self._get_coding_summary()
        }
    
    def _get_coding_distribution(self) -> Dict:
        """Analysiert die Verteilung der Kodierungen."""
        if not self.coding_results:
            return {}
            
        coders = Counter(coding['coder_id'] for coding in self.coding_results)
        categories = Counter(coding['category'] for coding in self.coding_results)
        
        return {
            'coders': dict(coders),
            'categories': dict(categories)
        }
    
    def _get_coding_summary(self) -> Dict:
        """Erstellt eine Zusammenfassung der Kodierungen."""
        if not self.coding_results:
            return {}
            
        return {
            'total_segments_coded': len(set(coding['segment_id'] for coding in self.coding_results)),
            'avg_codings_per_segment': len(self.coding_results) / len(self.processed_segments) if self.processed_segments else 0,
            'unique_categories': len(set(coding['category'] for coding in self.coding_results))
        }

    def get_progress_report(self) -> Dict:
        """
        Erstellt einen detaillierten Fortschrittsbericht für die laufende Analyse.
        
        Returns:
            Dict: Fortschrittsbericht mit aktuellen Metriken und Status
        """
        current_time = datetime.now()
        elapsed_time = (current_time - self.start_time).total_seconds()
        
        # Berechne durchschnittliche Verarbeitungszeiten
        avg_batch_time = statistics.mean(self.performance_metrics['batch_processing_times']) if self.performance_metrics['batch_processing_times'] else 0
        avg_coding_time = statistics.mean(self.performance_metrics['coding_times']) if self.performance_metrics['coding_times'] else 0
        
        # Hole Sättigungsmetriken vom SaturationChecker
        saturation_stats = {
            'material_processed': self.saturation_checker.processed_percentage,
            'stable_iterations': self.saturation_checker.stable_iterations,
            'current_batch_size': self.saturation_checker.current_batch_size
        }
        
        # Berechne Verarbeitungsstatistiken
        total_segments = len(self.processed_segments)
        segments_per_hour = (total_segments / elapsed_time) * 3600 if elapsed_time > 0 else 0
        
        return {
            'progress': {
                'processed_segments': total_segments,
                'total_codings': len(self.coding_results),
                'segments_per_hour': round(segments_per_hour, 2),
                'elapsed_time': round(elapsed_time, 2)
            },
            'performance': {
                'avg_batch_processing_time': round(avg_batch_time, 2),
                'avg_coding_time': round(avg_coding_time, 2),
                'last_batch_time': round(self.performance_metrics['batch_processing_times'][-1], 2) if self.performance_metrics['batch_processing_times'] else 0
            },
            'saturation': saturation_stats,
            'status': {
                'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'last_update': self.analysis_log[-1]['timestamp'] if self.analysis_log else None
            }
        }

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



# --- Klasse: DeductiveCoder ---
# Aufgabe: Automatisches deduktives Codieren von Text-Chunks anhand des Leitfadens
class DeductiveCoder:
    """
    Ordnet Text-Chunks automatisch deduktive Kategorien zu basierend auf dem Kodierleitfaden.
    Nutzt GPT-4-Mini für die qualitative Inhaltsanalyse nach Mayring.
    """
    
    def __init__(self, model_name: str, temperature: str, coder_id: str, skip_inductive: bool = False):
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
        self.temperature = float(temperature)
        self.coder_id = coder_id
        self.skip_inductive = skip_inductive
        self.current_categories = {}
           
        # Hole Provider aus CONFIG
        provider_name = CONFIG.get('MODEL_PROVIDER', 'openai')  # Fallback zu OpenAI
        try:
            # Wir lassen den LLMProvider sich selbst um die Client-Initialisierung kümmern
            self.llm_provider = LLMProviderFactory.create_provider(provider_name)
            print(f"LLM Provider '{provider_name}' für Kodierer {coder_id} initialisiert")
        except Exception as e:
            print(f"Fehler bei Provider-Initialisierung für {coder_id}: {str(e)}")
            raise

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

            response = await self.llm_provider.create_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                    {"role": "user", "content": prompt}
                ],
                temperature=float(self.temperature),
                response_format={"type": "json_object"}
            )
            
            # Verarbeite Response mit Wrapper
            llm_response = LLMResponse(response)
            result = json.loads(llm_response.content)

            print(f"  ✓ Kodierung erstellt: {result.get('category', 'Keine Kategorie')} "
                  f"(Konfidenz: {result.get('confidence', {}).get('total', 0):.2f})")


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
                    subcategories=tuple(result.get('subcategories', [])),  # Als Tuple
                    justification=result.get('justification', ''),
                    confidence=result.get('confidence', {'total': 0.0, 'category': 0.0, 'subcategories': 0.0}),
                    text_references=tuple([chunk[:100]]),  # Als Tuple
                    uncertainties=None
                )
            return None

        except Exception as e:
            print(f"Fehler bei der Prüfung deduktiver Kategorien: {str(e)}")
            return None

    async def update_category_system(self, categories: Dict[str, CategoryDefinition]) -> bool:
        """
        Aktualisiert das Kategoriensystem des Kodierers.
        
        Args:
            categories: Neues/aktualisiertes Kategoriensystem
            
        Returns:
            bool: True wenn Update erfolgreich
        """
        try:
            # Konvertiere CategoryDefinition in serialisierbares Dict
            categories_dict = {
                name: {
                    'definition': cat.definition,
                    'examples': list(cat.examples) if isinstance(cat.examples, set) else cat.examples,
                    'rules': list(cat.rules) if isinstance(cat.rules, set) else cat.rules,
                    'subcategories': dict(cat.subcategories) if isinstance(cat.subcategories, set) else cat.subcategories
                } for name, cat in categories.items()
            }

            # Aktualisiere das Kontextwissen des Kodierers
            prompt = f"""
            Das Kategoriensystem wurde aktualisiert. Neue Zusammensetzung:
            {json.dumps(categories_dict, indent=2, ensure_ascii=False)}
            
            Berücksichtige bei der Kodierung:
            1. Sowohl ursprüngliche als auch neue Kategorien verwenden
            2. Auf Überschneidungen zwischen Kategorien achten
            3. Subkategorien der neuen Kategorien einbeziehen
            """

            input_tokens = estimate_tokens(prompt)

            response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            # Zähle Tokens
            output_tokens = estimate_tokens(response.choices[0].message.content)
            token_counter.add_tokens(input_tokens, output_tokens)

            if response.choices[0].message.content:

                self.current_categories = categories

                print(f"Kategoriensystem für Kodierer {self.coder_id} aktualisiert")
                print(f"- {len(categories)} Kategorien verfügbar")
                return True
            
            return False

        except Exception as e:
            print(f"Fehler beim Update des Kategoriensystems für {self.coder_id}: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return False

    async def code_chunk(self, chunk: str, categories: Dict[str, CategoryDefinition] = None) -> Optional[CodingResult]:
        """
        Kodiert einen Text-Chunk basierend auf dem aktuellen Kategoriensystem.
        
        Args:
            chunk: Zu kodierender Text
            categories: Optional übergebenes Kategoriensystem (wird nur verwendet wenn kein aktuelles System existiert)
            
        Returns:
            Optional[CodingResult]: Kodierungsergebnis oder None bei Fehler
        """
        try:
            # Verwende das interne Kategoriensystem wenn vorhanden, sonst das übergebene
            current_categories = self.current_categories or categories
            
            if not current_categories:
                print(f"Fehler: Kein Kategoriensystem für Kodierer {self.coder_id} verfügbar")
                return None

            print(f"\nDeduktiver Kodierer 🧐 **{self.coder_id}** verarbeitet Chunk...")
            
            # Erstelle formatierte Kategorienübersicht mit Definitionen und Beispielen
            categories_overview = []
            for name, cat in current_categories.items():  # Verwende current_categories statt categories
                category_info = {
                    'name': name,
                    'definition': cat.definition,
                    'examples': list(cat.examples) if isinstance(cat.examples, set) else cat.examples,
                    'rules': list(cat.rules) if isinstance(cat.rules, set) else cat.rules,
                    'subcategories': {}
                }
                
                # Füge Subkategorien mit Definitionen hinzu
                for sub_name, sub_def in cat.subcategories.items():
                    category_info['subcategories'][sub_name] = sub_def
                    
                categories_overview.append(category_info)
            
            prompt = f"""
            Analysiere folgenden Text im Kontext der Forschungsfrage:
            "{FORSCHUNGSFRAGE}"
            
            TEXT:
            {chunk}

            KATEGORIENSYSTEM:
            Vergleiche den Text sorgfältig mit den folgenden Kategorien und ihren Beispielen:

            {json.dumps(categories_overview, indent=2, ensure_ascii=False)}

            KODIERREGELN:
            {json.dumps(KODIERREGELN, indent=2, ensure_ascii=False)}

            WICHTIG: 
            1. KATEGORIENVERGLEICH:
            - Vergleiche den Text mit JEDER Kategoriendefinition und deren Beispielen
            - Prüfe ob der Text ähnliche Aspekte wie die Beispiele aufweist
            - Prüfe ob der Text der Definition der Kategorie entspricht
            
            2. SUBKATEGORIENVERGLEICH:
            - Falls eine vorhandene Hauptkategorie passt, vergleiche mit den  Subkategoriendefinitionen
            - Wähle nur vorhandene Subkategorien, die wirklich zum Text passen
            
            3. ENTSCHEIDUNGSREGELN:
            - Der Text könnte für die Forschungsfrage relevant sein, aber zu keiner Kategorie passen
            - In diesem Fall gib "Keine passende Kategorie" zurück
            - Kodiere nur dann mit einer Kategorie, wenn Text UND Beispiele ähnlich sind
            - Erzwinge keine Zuordnung wenn keine Kategorie wirklich passt
            - Die Zuordnung muss durch Bezug auf Definition UND Beispiele begründbar sein

            Erstelle:
            1. Eine prägnante Paraphrase des Texts (max. 40 Wörter)
            2. Schlüsselwörter (2-3 zentrale Begriffe)
            3. Eine begründete Kategorienzuordnung oder "Keine passende Kategorie"

            Antworte ausschließlich mit einem JSON-Objekt:
            {{
                "paraphrase": "Deine prägnante Paraphrase hier",
                "keywords": "Deine Schlüsselwörter hier",
                "category": "Name der Hauptkategorie oder 'Keine passende Kategorie'",
                "subcategories": ["Liste", "existierender", "Subkategorien"],
                "justification": "Begründung mit Bezug auf Definitionen und Beispiele",
                "confidence": {{
                    "total": 0.85,
                    "category": 0.9,
                    "subcategories": 0.8
                }},
                "text_references": ["Relevante", "Textstellen"],
                "definition_matches": ["Welche Aspekte der Definition passen"],
                "example_matches": ["Welche Beispiele ähnlich sind"]
            }}
            """

            try:
                input_tokens = estimate_tokens(prompt + chunk)

                response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                # Verarbeite Response mit Wrapper
                llm_response = LLMResponse(response)
                result = json.loads(llm_response.content)

                output_tokens = estimate_tokens(response.choices[0].message.content)
                token_counter.add_tokens(input_tokens, output_tokens)
                
                if result and isinstance(result, dict):
                    if result.get('category'):
                        print(f"  ✓ Kodierung erstellt: {result['category']} "
                            f"(Konfidenz: {result.get('confidence', {}).get('total', 0):.2f})")
                        
                        # Debug-Ausgaben
                        print("\n👨‍⚖️  Kodierungsbegründung:")
                        if definition_matches := result.get('definition_matches', []):
                            print("  Passende Definitionsaspekte:")
                            for match in definition_matches:
                                print(f"  - {match}")
                        
                        if example_matches := result.get('example_matches', []):
                            print("  Ähnliche Beispiele:")
                            for match in example_matches:
                                print(f"  - {match}")
                        
                        paraphrase = result.get('paraphrase', '')
                        if paraphrase:
                            print(f"\nParaphrase: {paraphrase}")
                        
                        # Wenn "Keine passende Kategorie" zurückgegeben wurde
                        if result['category'] == "Keine passende Kategorie":
                            print("\nℹ Text als relevant erkannt, aber keine passende Kategorie gefunden")
                            print(f"Begründung: {result.get('justification', 'Keine Begründung angegeben')}")
                    
                    return CodingResult(
                        category=result.get('category', ''),
                        subcategories=tuple(result.get('subcategories', [])),
                        justification=result.get('justification', ''),
                        confidence=result.get('confidence', {'total': 0.0, 'category': 0.0, 'subcategories': 0.0}),
                        text_references=tuple([chunk[:100]]),
                        uncertainties=None,
                        paraphrase=result.get('paraphrase', ''),
                        keywords=result.get('keywords', '')
                    )
                else:
                    print("  ✗ Keine passende Kategorie gefunden")
                    return None
                
            except Exception as e:
                print(f"Fehler bei API Call: {str(e)}")
                return None          

        except Exception as e:
            print(f"Fehler bei der Kodierung durch {self.coder_id}: {str(e)}")
            return None

    async def _check_relevance(self, chunk: str) -> bool:
        """
        Prüft die Relevanz eines Chunks für die Forschungsfrage.
        
        Args:
            chunk: Zu prüfender Text
            
        Returns:
            bool: True wenn der Text relevant ist
        """
        try:
            prompt = f"""
            Analysiere sorgfältig die Relevanz des folgenden Texts für die Forschungsfrage:
            "{FORSCHUNGSFRAGE}"
            
            TEXT:
            {chunk}
            
            Prüfe systematisch:
            1. Inhaltlicher Bezug: Behandelt der Text explizit Aspekte der Forschungsfrage?
            2. Aussagekraft: Enthält der Text konkrete, analysierbare Aussagen?
            3. Substanz: Geht der Text über oberflächliche/beiläufige Erwähnungen hinaus?
            4. Kontext: Ist der Bezug zur Forschungsfrage eindeutig und nicht nur implizit?

            Antworte NUR mit einem JSON-Objekt:
            {{
                "is_relevant": true/false,
                "confidence": 0.0-1.0,
                "justification": "Kurze Begründung der Entscheidung",
                "key_aspects": ["Liste", "relevanter", "Aspekte"]
            }}
            """

            input_tokens = estimate_tokens(prompt + chunk)
            
            response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
            # Verarbeite Response mit Wrapper
            llm_response = LLMResponse(response)
            result = json.loads(llm_response.content)
            
            output_tokens = estimate_tokens(response.choices[0].message.content)
            token_counter.add_tokens(input_tokens, output_tokens)

            # Detaillierte Ausgabe der Relevanzprüfung
            if result.get('is_relevant'):
                print(f"✓ Relevanz bestätigt (Konfidenz: {result.get('confidence', 0):.2f})")
                if result.get('key_aspects'):
                    print("  Relevante Aspekte:")

                    for aspect in result['key_aspects']:
                        print(f"  - {aspect}")
            else:
                print(f"❌ Nicht relevant: {result.get('justification', 'Keine Begründung')}")

            return result.get('is_relevant', False)

        except Exception as e:
            print(f"Fehler bei der Relevanzprüfung: {str(e)}")
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
    Implementiert die induktive Kategorienentwicklung nach Mayring.
    Optimiert für Performance und Integration mit SaturationChecker.
    """
    
    def __init__(self, model_name: str, history: DevelopmentHistory, output_dir: str = None, config: dict = None):
        self.model_name = model_name
        self.output_dir = output_dir or CONFIG['OUTPUT_DIR']
        
        # Initialisiere LLM Provider
        provider_name = CONFIG.get('MODEL_PROVIDER', 'openai')
        try:
            self.llm_provider = LLMProviderFactory.create_provider(provider_name)
            print(f"\nLLM Provider '{provider_name}' für induktive Kodierung initialisiert")
        except Exception as e:
            print(f"Fehler bei Provider-Initialisierung: {str(e)}")
            raise
        
        # Performance-Optimierung
        self.category_cache = {}  # Cache für häufig verwendete Kategorien
        self.batch_results = []   # Speichert Batch-Ergebnisse für Analyse
        self.analysis_cache = {}  # Cache für Kategorienanalysen
        
        # Qualitätsschwellen
        validation_config = CONFIG.get('validation_config', {})
        thresholds = validation_config.get('thresholds', {})
        self.MIN_CONFIDENCE = thresholds.get('MIN_CONFIDENCE', 0.6)
        self.MIN_EXAMPLES = thresholds.get('MIN_EXAMPLES', 2)
        self.MIN_DEFINITION_WORDS = thresholds.get('MIN_DEFINITION_WORDS', 15)

        # Batch processing configuration
        self.BATCH_SIZE = CONFIG.get('BATCH_SIZE', 5)  # Number of segments to process at once
        self.MAX_RETRIES = 3 # Maximum number of retries for failed API calls
        
        # Tracking
        self.history = history
        self.development_history = []
        self.last_analysis_time = None

         # Verwende zentrale Validierung
        self.validator = CategoryValidator(config)

        # Initialisiere SaturationChecker
        self.saturation_checker = SaturationChecker(config, history)
        
        # System-Prompt-Cache
        self._cached_system_prompt = None

        print("\nInduktiveCoder initialisiert:")
        print(f"- Model: {model_name}")
        print(f"- Batch-Größe: {self.BATCH_SIZE}")
        print(f"- Max Retries: {self.MAX_RETRIES}")
        print(f"- Cache aktiviert: Ja")

    def _create_batches(self, segments: List[str], batch_size: int = None) -> List[List[str]]:
        """
        Creates batches of segments for processing.
        
        Args:
            segments: List of text segments to process
            batch_size: Optional custom batch size (defaults to self.BATCH_SIZE)
            
        Returns:
            List[List[str]]: List of segment batches
        """
        batch_size = self.BATCH_SIZE
            
        return [
            segments[i:i + batch_size] 
            for i in range(0, len(segments), batch_size)
        ]
       
    async def develop_category_system(self, segments: List[str]) -> Dict[str, CategoryDefinition]:
        """Entwickelt induktiv neue Kategorien mit inkrementeller Erweiterung."""
        try:
            # Voranalyse und Filterung der Segmente
            relevant_segments = await self._prefilter_segments(segments)
            
            # Erstelle Batches für Analyse
            batches = self._create_batches(relevant_segments)
            
            # Initialisiere Dict für das erweiterte Kategoriensystem
            extended_categories = {}
            
            for batch_idx, batch in enumerate(batches):
                print(f"\nAnalysiere Batch {batch_idx + 1}/{len(batches)}...")

                # Berechne Material-Prozentsatz
                material_percentage = ((batch_idx + 1) * len(batch) / len(segments)) * 100
                
                # Übergebe aktuelles System an Batch-Analyse
                batch_analysis = await self.analyze_category_batch(
                    category=extended_categories,  # Wichtig: Übergebe bisheriges System
                    segments=batch,
                    material_percentage=material_percentage 
                )
                
                if batch_analysis:
                    # 1. Aktualisiere bestehende Kategorien
                    if 'existing_categories' in batch_analysis:
                        for cat_name, updates in batch_analysis['existing_categories'].items():
                            if cat_name in extended_categories:
                                current_cat = extended_categories[cat_name]
                                
                                # Aktualisiere Definition falls vorhanden
                                if 'refinements' in updates:
                                    refinements = updates['refinements']
                                    if refinements['confidence'] > 0.7:  # Nur bei hoher Konfidenz
                                        current_cat = current_cat._replace(
                                            definition=refinements['definition'],
                                            modified_date=datetime.now().strftime("%Y-%m-%d")
                                        )
                                
                                # Füge neue Subkategorien hinzu
                                if 'new_subcategories' in updates:
                                    new_subcats = {
                                        sub['name']: sub['definition']
                                        for sub in updates['new_subcategories']
                                        if sub['confidence'] > 0.7  # Nur bei hoher Konfidenz
                                    }
                                    current_cat = current_cat._replace(
                                        subcategories={**current_cat.subcategories, **new_subcats},
                                        modified_date=datetime.now().strftime("%Y-%m-%d")
                                    )
                                    
                                extended_categories[cat_name] = current_cat
                                print(f"✓ Kategorie '{cat_name}' aktualisiert")
                    
                    # 2. Füge neue Kategorien hinzu
                    if 'new_categories' in batch_analysis:
                        for new_cat in batch_analysis['new_categories']:
                            if new_cat['confidence'] < 0.7:  # Prüfe Konfidenz
                                continue
                                
                            cat_name = new_cat['name']
                            if cat_name not in extended_categories:  # Nur wenn wirklich neu
                                # Erstelle neue CategoryDefinition
                                extended_categories[cat_name] = CategoryDefinition(
                                    name=cat_name,
                                    definition=new_cat['definition'],
                                    examples=new_cat.get('evidence', []),
                                    rules=[],  # Regeln werden später entwickelt
                                    subcategories={
                                        sub['name']: sub['definition']
                                        for sub in new_cat.get('subcategories', [])
                                    },
                                    added_date=datetime.now().strftime("%Y-%m-%d"),
                                    modified_date=datetime.now().strftime("%Y-%m-%d")
                                )
                                print(f"✓ Neue Kategorie erstellt: '{cat_name}'")
                    
                    # Prüfe Sättigung
                    is_saturated, metrics = self.saturation_checker.check_saturation(
                        current_categories=extended_categories,
                        coded_segments=self.batch_results,
                        material_percentage=material_percentage
                    )
                    
                    if is_saturated:
                        print(f"\nSättigung erreicht bei {material_percentage:.1f}% des Materials")
                        print(f"Begründung: {metrics.get('justification', 'Keine Begründung verfügbar')}")
                        break

                    # Dokumentiere Entwicklung
                    self.history.log_category_development(
                        phase=f"batch_{batch_idx + 1}",
                        new_categories=len(batch_analysis.get('new_categories', [])),
                        modified_categories=len(batch_analysis.get('existing_categories', {}))
                    )
                    
                    print(f"\nZwischenstand nach Batch {batch_idx + 1}:")
                    print(f"- Kategorien gesamt: {len(extended_categories)}")
                    print(f"- Davon neu in diesem Batch: {len(batch_analysis.get('new_categories', []))}")
                    print(f"- Aktualisiert in diesem Batch: {len(batch_analysis.get('existing_categories', {}))}")
            
            return extended_categories
                
        except Exception as e:
            print(f"Fehler bei Kategorienentwicklung: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return {}

    async def analyze_category_batch(self, 
                                category: Dict[str, CategoryDefinition], 
                                segments: List[str],
                                material_percentage: float) -> Dict[str, Any]:  
        """
        Verbesserte Batch-Analyse mit Berücksichtigung des aktuellen Kategoriensystems.
        
        Args:
            category: Aktuelles Kategoriensystem
            segments: Liste der Textsegmente
            material_percentage: Prozentsatz des verarbeiteten Materials
            
        Returns:
            Dict[str, Any]: Analyseergebnisse einschließlich Sättigungsmetriken
        """
        try:
            # Cache-Key erstellen
            cache_key = (
                frozenset(category.items()) if isinstance(category, dict) else str(category),
                tuple(segments)
            )
            
            # Prüfe Cache
            if cache_key in self.analysis_cache:
                print("Nutze gecachte Analyse")
                return self.analysis_cache[cache_key]

            # Erstelle formatierten Text der aktuellen Kategorien
            current_categories_text = ""
            if category:
                current_categories_text = "Aktuelle Kategorien:\n"
                for name, cat in category.items():
                    current_categories_text += f"\n{name}:\n"
                    current_categories_text += f"- Definition: {cat.definition}\n"
                    if cat.subcategories:
                        current_categories_text += "- Subkategorien:\n"
                        for sub_name, sub_def in cat.subcategories.items():
                            current_categories_text += f"  • {sub_name}: {sub_def}\n"

            # Definiere JSON-Schema außerhalb des f-strings
            json_schema = '''{
                "existing_categories": {
                    "kategorie_name": {
                        "refinements": {
                            "definition": "Erweiterte Definition",
                            "justification": "Begründung",
                            "confidence": 0.0-1.0
                        },
                        "new_subcategories": [
                            {
                                "name": "Name",
                                "definition": "Definition",
                                "evidence": ["Textbelege"],
                                "confidence": 0.0-1.0
                            }
                        ]
                    }
                },
                "new_categories": [
                    {
                        "name": "Name",
                        "definition": "Definition",
                        "subcategories": [
                            {
                                "name": "Name",
                                "definition": "Definition"
                            }
                        ],
                        "evidence": ["Textbelege"],
                        "confidence": 0.0-1.0,
                        "justification": "Begründung"
                    }
                ],
                "saturation_metrics": {
                    "new_aspects_found": true/false,
                    "categories_sufficient": true/false,
                    "theoretical_coverage": 0.0-1.0,
                    "justification": "Begründung der Sättigungseinschätzung"
                }
            }'''
            prompt = f"""
            Führe eine vollständige Kategorienanalyse basierend auf den Textsegmenten durch.
            Berücksichtige dabei das bestehende Kategoriensystem und erweitere es.
            
            {current_categories_text}
            
            TEXTSEGMENTE:
            {json.dumps(segments, indent=2, ensure_ascii=False)}
            
            FORSCHUNGSFRAGE:
            {FORSCHUNGSFRAGE}
            
            Analysiere systematisch:
            1. BESTEHENDE KATEGORIEN
            - Prüfe ob neue Aspekte zu bestehenden Kategorien passen
            - Schlage Erweiterungen/Verfeinerungen vor
            - Identifiziere neue Subkategorien für bestehende Hauptkategorien
            
            2. NEUE KATEGORIEN
            - Identifiziere gänzlich neue Aspekte
            - Entwickle neue Hauptkategorien wenn nötig
            - Stelle Trennschärfe zu bestehenden Kategorien sicher
            
            3. BEGRÜNDUNG UND EVIDENZ
            - Belege alle Vorschläge mit Textstellen
            - Dokumentiere Entscheidungsgründe
            - Gib Konfidenzwerte für Vorschläge an

            4. SÄTTIGUNGSANALYSE
            - Bewerte ob neue inhaltliche Aspekte gefunden wurden
            - Prüfe ob bestehende Kategorien ausreichen
            - Beurteile ob weitere Kategorienentwicklung nötig ist

            Antworte NUR mit einem JSON-Objekt:
            {json_schema}
            """

            input_tokens = estimate_tokens(prompt)

            response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
            # Verarbeite Response mit Wrapper
            llm_response = LLMResponse(response)
            result = json.loads(llm_response.content)
            
            output_tokens = estimate_tokens(response.choices[0].message.content)
            token_counter.add_tokens(input_tokens, output_tokens)

            # Cache das Ergebnis
            self.analysis_cache[cache_key] = result
            
            # Debug-Ausgaben
            print("\nAnalyseergebnisse:")

            # Sättigungsmetriken extrahieren und speichern
            if 'saturation_metrics' in result:
                # Erweiterte Sättigungsanalyse
                category_usage = {
                    cat_name: {
                        'usage_count': 0,
                        'confidence_scores': [],
                        'subcategory_usage': defaultdict(int),
                        'last_used_batch': 0
                    } for cat_name in category.keys()
                }
                
                # Analysiere Kategoriennutzung im aktuellen Batch
                for coding in self.batch_results:
                    if coding['category'] in category_usage:
                        cat_stats = category_usage[coding['category']]
                        cat_stats['usage_count'] += 1
                        cat_stats['confidence_scores'].append(coding['confidence']['total'])
                        cat_stats['last_used_batch'] = len(self.batch_results)
                        
                        # Subkategorien-Nutzung
                        for subcat in coding.get('subcategories', []):
                            cat_stats['subcategory_usage'][subcat] += 1

                # Berechne erweiterte theoretische Abdeckung
                if len(category) > 0:  # Prüfe ob überhaupt Kategorien vorhanden sind
                    theoretical_coverage = {
                        'category_coverage': len([c for c in category_usage.values() if c['usage_count'] > 0]) / len(category),
                        'usage_stability': sum(1 for c in category_usage.values() if c['usage_count'] >= 3) / len(category),
                        'subcategory_coverage': sum(1 for c in category_usage.values() if len(c['subcategory_usage']) > 0) / len(category),
                        'avg_confidence': statistics.mean([score for c in category_usage.values() for score in c['confidence_scores']]) if any(c['confidence_scores'] for c in category_usage.values()) else 0
                    }
                else:
                    # Fallback-Werte wenn keine Kategorien vorhanden
                    theoretical_coverage = {
                        'category_coverage': 0.0,
                        'usage_stability': 0.0,
                        'subcategory_coverage': 0.0,
                        'avg_confidence': 0.0
                    }
                # Gewichtete Gesamtabdeckung
                total_coverage = (
                    theoretical_coverage['category_coverage'] * 0.4 +
                    theoretical_coverage['usage_stability'] * 0.3 +
                    theoretical_coverage['subcategory_coverage'] * 0.2 +
                    theoretical_coverage['avg_confidence'] * 0.1
                )

                # Aktualisiere Sättigungsmetriken im Ergebnis
                result['saturation_metrics'].update({
                    'new_aspects_found': any(c['usage_count'] == 1 for c in category_usage.values()),
                    'categories_sufficient': total_coverage >= 0.8,
                    'theoretical_coverage': total_coverage,
                    'coverage_details': theoretical_coverage,
                    'justification': f"Theoretische Abdeckung: {total_coverage:.2f} (Kategorien: {theoretical_coverage['category_coverage']:.2f}, Stabilität: {theoretical_coverage['usage_stability']:.2f}, Subkategorien: {theoretical_coverage['subcategory_coverage']:.2f}, Konfidenz: {theoretical_coverage['avg_confidence']:.2f})"
                })

                # Übergebe Metriken an SaturationChecker
                self.saturation_checker.add_saturation_metrics(result['saturation_metrics'])
            
                saturation_info = result['saturation_metrics']

                # Speichere Metriken in der History
                self.history.log_saturation_check(
                    material_percentage=material_percentage,
                    result="saturated" if saturation_info['categories_sufficient'] else "not_saturated",
                    metrics=saturation_info
                )
                
                # Gebe Sättigungsinfo in Konsole aus
                if saturation_info['categories_sufficient']:
                    print("\nSättigungsanalyse:")
                    print(f"- Neue Aspekte gefunden: {'Ja' if saturation_info['new_aspects_found'] else 'Nein'}")
                    print(f"- Theoretische Abdeckung: {total_coverage:.2f}")
                    print(f"- Details:")
                    print(f"  • Kategorienutzung: {theoretical_coverage['category_coverage']:.2f}")
                    print(f"  • Nutzungsstabilität: {theoretical_coverage['usage_stability']:.2f}")
                    print(f"  • Subkategorien: {theoretical_coverage['subcategory_coverage']:.2f}")
                    print(f"  • Durchschn. Konfidenz: {theoretical_coverage['avg_confidence']:.2f}")
                    print(f"- Begründung: {saturation_info['justification']}")

            # Zeige Erweiterungen bestehender Kategorien
            if 'existing_categories' in result:
                for cat_name, updates in result['existing_categories'].items():
                    print(f"\nAktualisierung für '{cat_name}':")
                    if 'refinements' in updates:
                        print(f"- Definition erweitert (Konfidenz: {updates['refinements']['confidence']:.2f})")
                    if 'new_subcategories' in updates:
                        print("- Neue Subkategorien:")
                        for sub in updates['new_subcategories']:
                            print(f"  • {sub['name']} (Konfidenz: {sub['confidence']:.2f})")
                            
            # Zeige neue Kategorien
            if 'new_categories' in result:
                print("\nNeue Hauptkategorien:")
                for new_cat in result['new_categories']:
                    print(f"- {new_cat['name']} (Konfidenz: {new_cat['confidence']:.2f})")
                    if 'subcategories' in new_cat:
                        print("  Subkategorien:")
                        for sub in new_cat['subcategories']:
                            print(f"  • {sub['name']}")

            return result

        except Exception as e:
            print(f"Fehler bei Kategorienanalyse: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return None
        
    async def _prefilter_segments(self, segments: List[str]) -> List[str]:
        """
        Filtert Segmente nach Relevanz für Kategorienentwicklung.
        Optimiert durch Parallelverarbeitung und Caching.
        """
        async def check_segment(segment: str) -> Tuple[str, float]:
            cache_key = hash(segment)
            if cache_key in self.category_cache:
                return segment, self.category_cache[cache_key]
            
            relevance = await self._assess_segment_relevance(segment)
            self.category_cache[cache_key] = relevance
            return segment, relevance
        
        # Parallele Relevanzprüfung
        tasks = [check_segment(seg) for seg in segments]
        results = await asyncio.gather(*tasks)
        
        # Filter relevante Segmente
        return [seg for seg, relevance in results if relevance > self.MIN_CONFIDENCE]

    async def _assess_segment_relevance(self, segment: str) -> float:
        """
        Bewertet die Relevanz eines Segments für die Kategorienentwicklung.
        """
        prompt = f"""
        Bewerte die Relevanz des folgenden Textsegments für die Kategorienentwicklung.
        Berücksichtige:
        1. Bezug zur Forschungsfrage: {FORSCHUNGSFRAGE}
        2. Informationsgehalt
        3. Abstraktionsniveau
        
        Text: {segment}
        
        Antworte nur mit einem JSON-Objekt:
        {{
            "relevance_score": 0.8,  // 0-1
            "reasoning": "Kurze Begründung"
        }}
        """
        
        try:
            input_tokens = estimate_tokens(prompt)

            response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
            # Verarbeite Response mit Wrapper
            llm_response = LLMResponse(response)
            result = json.loads(llm_response.content)

            output_tokens = estimate_tokens(response.choices[0].message.content)
            token_counter.add_tokens(input_tokens, output_tokens)

            return float(result.get('relevance_score', 0))
            
        except Exception as e:
            print(f"Error in relevance assessment: {str(e)}")
            return 0.0

    async def _extract_categories_from_batch(self, batch: List[str]) -> List[Dict]:
        """
        Extrahiert Kategorien aus einem Batch von Segmenten.
        
        Args:
            batch: Zu analysierende Textsegmente
            
        Returns:
            List[Dict]: Extrahierte Kategorien
        """
        async def process_segment(segment: str) -> List[Dict]:
            prompt = self._get_category_extraction_prompt(segment)
            
           

            try:
                print("\nSende Anfrage an API...")

                input_tokens = estimate_tokens(prompt)
                
                response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                
                # Verarbeite Response mit Wrapper
                llm_response = LLMResponse(response)
                raw_response = json.loads(llm_response.content)
                

                output_tokens = estimate_tokens(response.choices[0].message.content)
                token_counter.add_tokens(output_tokens)
                print("\nAPI-Antwort erhalten:")
                print(raw_response)

                try:
                    result = json.loads(llm_response)
                    print("\nJSON erfolgreich geparst:")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                    
                    # Suche nach Kategorien in verschiedenen Formaten
                    categories = []
                    
                    # Fall 1: Direkte Liste von Kategorien
                    if isinstance(result, list):
                        categories = result
                        
                    # Fall 2: Verschachtelte Liste unter "new_categories"
                    elif isinstance(result, dict) and 'new_categories' in result:
                        categories = result['new_categories']
                        
                    # Fall 3: Einzelne Kategorie als Objekt
                    elif isinstance(result, dict) and 'name' in result:
                        categories = [result]
                        
                    # Fall 4: Andere verschachtelte Arrays suchen
                    elif isinstance(result, dict):
                        for key, value in result.items():
                            if isinstance(value, list) and value and isinstance(value[0], dict):
                                if 'name' in value[0]:  # Prüfe ob es wie eine Kategorie aussieht
                                    categories = value
                                    print(f"Kategorien unter Schlüssel '{key}' gefunden")
                                    break
                    
                    if not categories:
                        print("Keine Kategorien in der Antwort gefunden")
                        return []
                    
                    # Validiere jede gefundene Kategorie
                    validated_categories = []
                    for cat in categories:
                        if isinstance(cat, dict):
                            # Stelle sicher, dass alle erforderlichen Felder vorhanden sind
                            required_fields = {
                                'name': str,
                                'definition': str,
                                'example': str,
                                'new_subcategories': list,
                                'justification': str,
                                'confidence': dict
                            }
                            
                            valid = True
                            # Erstelle eine Kopie der Kategorie für Modifikationen
                            processed_cat = cat.copy()
                            
                            for field, field_type in required_fields.items():
                                if field not in processed_cat:
                                    print(f"Fehlendes Feld '{field}' - Versuche Standard")
                                    # Setze Standardwerte für fehlende Felder
                                    if field == 'example':
                                        processed_cat[field] = ""
                                    elif field == 'new_subcategories':
                                        processed_cat[field] = []
                                    elif field == 'confidence':
                                        processed_cat[field] = {'category': 0.7, 'subcategories': 0.7}
                                    else:
                                        print(f"Kritisches Feld '{field}' fehlt")
                                        valid = False
                                        break
                                elif not isinstance(processed_cat[field], field_type):
                                    print(f"Falscher Typ für Feld '{field}': {type(processed_cat[field])} statt {field_type}")
                                    valid = False
                                    break
                            
                            if valid:
                                validated_categories.append(processed_cat)
                                print(f"✓ Kategorie '{processed_cat['name']}' validiert")
                    
                    return validated_categories
                    
                except json.JSONDecodeError as e:
                    print(f"Fehler beim JSON-Parsing: {str(e)}")
                    print("Ungültige Antwort:", raw_response)
                    return []
                    
            except Exception as e:
                print(f"Fehler bei API-Anfrage: {str(e)}")
                print("Details:")
                traceback.print_exc()
                return []
        
        # Parallele Verarbeitung der Segmente
        print(f"\nVerarbeite {len(batch)} Segmente...")
        tasks = [process_segment(seg) for seg in batch]
        try:
            results = await asyncio.gather(*tasks)
            
            # Kombiniere und bereinige Ergebnisse
            all_categories = []
            for categories in results:
                if categories and isinstance(categories, list):
                    all_categories.extend(categories)
            
            if all_categories:
                print(f"\n✓ {len(all_categories)} neue Kategorien gefunden")
                for cat in all_categories:
                    print(f"\n🆕 Neue Kategorie: {cat['name']}")
                    print(f"   Definition: {cat['definition'][:100]}...")
                    if cat['new_subcategories']:
                        print(f"   Subkategorien:")
                        for sub in cat['new_subcategories']:
                            print(f"   - {sub}")
            else:
                print("\n⚠ Keine neuen Kategorien gefunden")
                
            return all_categories
            
        except Exception as e:
            print(f"Fehler bei Batch-Verarbeitung: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return []
        
        # Parallele Verarbeitung der Segmente
        print(f"\nVerarbeite {len(batch)} Segmente...")
        tasks = [process_segment(seg) for seg in batch]
        try:
            results = await asyncio.gather(*tasks)
            
            # Kombiniere und bereinige Ergebnisse
            all_categories = []
            for categories in results:
                if categories and isinstance(categories, list):
                    all_categories.extend(categories)
            
            if all_categories:
                print(f"\n✓ {len(all_categories)} neue Kategorien gefunden")
            else:
                print("\n⚠ Keine neuen Kategorien gefunden")
                
            return all_categories
            
        except Exception as e:
            print(f"Fehler bei Batch-Verarbeitung: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return []

    def _get_system_prompt(self) -> str:
        """
        Systemkontext für die induktive Kategorienentwicklung.
        """
        return """Du bist ein Experte für qualitative Inhaltsanalyse nach Mayring.
        Deine Aufgabe ist die systematische Entwicklung von Kategorien aus Textmaterial.
        
        Zentrale Prinzipien:
        1. Nähe zum Material - Kategorien direkt aus dem Text ableiten
        2. Präzise Definitionen - klar, verständlich, abgrenzbar
        3. Angemessenes Abstraktionsniveau - nicht zu spezifisch, nicht zu allgemein
        4. Systematische Subkategorienbildung - erschöpfend aber trennscharf
        5. Regelgeleitetes Vorgehen - nachvollziehbare Begründungen"""

    def _get_category_extraction_prompt(self, segment: str) -> str:
        """
        Detaillierter Prompt für die Kategorienentwicklung mit Fokus auf 
        hierarchische Einordnung und Vermeidung zu spezifischer Hauptkategorien.
        """
        return f"""Analysiere das folgende Textsegment für die Entwicklung induktiver Kategorien.
        Forschungsfrage: "{FORSCHUNGSFRAGE}"

        WICHTIG - HIERARCHISCHE EINORDNUNG:
        
        1. PRÜFE ZUERST EINORDNUNG IN BESTEHENDE HAUPTKATEGORIEN
        Aktuelle Hauptkategorien:
        {json.dumps(DEDUKTIVE_KATEGORIEN, indent=2, ensure_ascii=False)}

        2. ENTSCHEIDUNGSREGELN FÜR KATEGORIENEBENE:

        HAUPTKATEGORIEN (nur wenn wirklich nötig):
        - Beschreiben übergeordnete Themenkomplexe oder Handlungsfelder
        - Umfassen mehrere verwandte Aspekte unter einem konzeptionellen Dach
        - Sind abstrakt genug für verschiedene Unterthemen
        
        GUTE BEISPIELE für Hauptkategorien:
        - "Wissenschaftlicher Nachwuchs" (übergeordneter Themenkomplex)
        - "Wissenschaftliches Personal" (breites Handlungsfeld)
        - "Berufungsstrategien" (strategische Ebene)
        
        SCHLECHTE BEISPIELE für Hauptkategorien:
        - "Promotion in Unternehmen" (→ besser als Subkategorie unter "Wissenschaftlicher Nachwuchs")
        - "Rolle wissenschaftlicher Mitarbeiter" (→ besser als Subkategorie unter "Wissenschaftliches Personal")
        - "Forschungsinteresse bei Berufungen" (→ besser als Subkategorie unter "Berufungsstrategien")

        SUBKATEGORIEN (bevorzugt):
        - Beschreiben spezifische Aspekte einer Hauptkategorie
        - Konkretisieren einzelne Phänomene oder Handlungen
        - Sind empirisch direkt beobachtbar

        3. ENTWICKLUNGSSCHRITTE:
        a) Prüfe ZUERST, ob der Inhalt als Subkategorie in bestehende Hauptkategorien passt
        b) Entwickle neue Subkategorien für bestehende Hauptkategorien
        c) NUR WENN NÖTIG: Schlage neue Hauptkategorie vor, wenn wirklich neuer Themenkomplex

        TEXT:
        {segment}

        Antworte ausschließlich mit einem JSON-Objekt:
        {{
            "categorization_type": "subcategory_extension" | "new_main_category",
            "analysis": {{
                "existing_main_category": "Name der passenden Hauptkategorie oder null",
                "justification": "Begründung der hierarchischen Einordnung"
            }},
            "suggested_changes": {{
                "new_subcategories": [
                    {{
                        "name": "Name der Subkategorie",
                        "definition": "Definition",
                        "example": "Textstelle als Beleg"
                    }}
                ],
                "new_main_category": null | {{  // nur wenn wirklich nötig
                    "name": "Name der neuen Hauptkategorie",
                    "definition": "Definition",
                    "justification": "Ausführliche Begründung, warum neue Hauptkategorie nötig",
                    "initial_subcategories": [
                        {{
                            "name": "Name der Subkategorie",
                            "definition": "Definition"
                        }}
                    ]
                }}
            }},
            "confidence": {{
                "hierarchy_level": 0.0-1.0,
                "categorization": 0.0-1.0
            }}
        }}

        WICHTIG:
        - Bevorzuge IMMER die Entwicklung von Subkategorien
        - Neue Hauptkategorien nur bei wirklich übergeordneten Themenkomplexen
        - Prüfe genau die konzeptionelle Ebene der Kategorisierung
        - Achte auf angemessenes Abstraktionsniveau
        """



    def _get_definition_enhancement_prompt(self, category: Dict) -> str:
        """
        Prompt zur Verbesserung unzureichender Definitionen.
        """
        return f"""Erweitere die folgende Kategoriendefinition zu einer vollständigen Definition.

        KATEGORIE: {category['name']}
        AKTUELLE DEFINITION: {category['definition']}
        BEISPIEL: {category.get('example', '')}
        
        ANFORDERUNGEN:
        1. Zentrale Merkmale klar benennen
        2. Anwendungsbereich definieren
        3. Abgrenzung zu ähnlichen Konzepten
        4. Mindestens drei vollständige Sätze
        5. Fachsprachlich präzise

        BEISPIEL GUTER DEFINITION:
        "Qualitätssicherungsprozesse umfassen alle systematischen Verfahren und Maßnahmen zur 
        Überprüfung und Gewährleistung definierter Qualitätsstandards in der Hochschule. Sie 
        beinhalten sowohl interne Evaluationen und Audits als auch externe Begutachtungen und 
        Akkreditierungen. Im Gegensatz zum allgemeinen Qualitätsmanagement fokussieren sie 
        sich auf die konkrete Durchführung und Dokumentation von qualitätssichernden 
        Maßnahmen."

        Antworte nur mit der erweiterten Definition."""

    def _get_subcategory_generation_prompt(self, category: Dict) -> str:
        """
        Prompt zur Generierung fehlender Subkategorien.
        """
        return f"""Entwickle passende Subkategorien für die folgende Hauptkategorie.

        HAUPTKATEGORIE: {category['name']}
        DEFINITION: {category['definition']}
        BEISPIEL: {category.get('example', '')}

        ANFORDERUNGEN AN SUBKATEGORIEN:
        1. 2-4 logisch zusammengehörende Aspekte
        2. Eindeutig der Hauptkategorie zuordenbar
        3. Sich gegenseitig ausschließend
        4. Mit kurzer Erläuterung (1 Satz)
        5. Relevant für die Forschungsfrage: {FORSCHUNGSFRAGE}

        BEISPIEL GUTER SUBKATEGORIEN:
        "Qualitätssicherungsprozesse":
        - "Interne Evaluation": Systematische Selbstbewertung durch die Hochschule
        - "Externe Begutachtung": Qualitätsprüfung durch unabhängige Gutachter
        - "Akkreditierung": Formale Anerkennung durch Akkreditierungsagenturen

        Antworte nur mit einem JSON-Array von Subkategorien:
        ["Subkategorie 1: Erläuterung", "Subkategorie 2: Erläuterung", ...]"""

    def _validate_basic_requirements(self, category: dict) -> bool:
        """
        Grundlegende Validierung einer Kategorie.
        
        Args:
            category: Zu validierende Kategorie
            
        Returns:
            bool: True wenn Grundanforderungen erfüllt
        """
        try:
            # Zeige Kategorienamen in der Warnung
            category_name = category.get('name', 'UNBENANNTE KATEGORIE')
            
            # Pflichtfelder prüfen
            required_fields = {
                'name': str,
                'definition': str
            }
            
            for field, field_type in required_fields.items():
                if field not in category:
                    print(f"Fehlendes Pflichtfeld '{field}' in Kategorie '{category_name}'")
                    return False
                if not isinstance(category[field], field_type):
                    print(f"Falscher Datentyp für '{field}' in Kategorie '{category_name}': "
                        f"Erwartet {field_type.__name__}, erhalten {type(category[field]).__name__}")
                    return False
            
            # Name validieren
            name = category['name'].strip()
            if len(name) < 3:
                print(f"Kategoriename '{name}' zu kurz (min. 3 Zeichen)")
                return False
            if len(name) > 50:
                print(f"Kategoriename '{name}' zu lang (max. 50 Zeichen)")
                return False
                
            # Prüfe auf englische Wörter im Namen
            english_indicators = {'research', 'development', 'management', 
                                'system', 'process', 'analysis'}
            if any(word.lower() in english_indicators 
                for word in name.split()):
                print(f"Englische Begriffe im Namen '{name}' gefunden")
                return False
                
            # Optionale Felder initialisieren
            if 'example' not in category:
                print(f"Initialisiere fehlendes Beispiel für '{category_name}'")
                category['example'] = ""
                
            if 'new_subcategories' not in category:
                print(f"Initialisiere fehlende Subkategorien für '{category_name}'")
                category['new_subcategories'] = []
                
            if 'confidence' not in category:
                print(f"Initialisiere fehlende Konfidenzwerte für '{category_name}'")
                category['confidence'] = {
                    'category': 0.7,
                    'subcategories': 0.7
                }
            
            print(f"✓ Grundanforderungen erfüllt für Kategorie '{category_name}'")
            return True
            
        except Exception as e:
            print(f"Fehler bei Grundvalidierung: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return False

    def _validate_categories(self, categories: List[dict]) -> List[dict]:
        """
        Validiert extrahierte Kategorien mit zentraler Validierung.
        """
        valid_categories = []
        
        for category in categories:
            try:
                # Konvertiere dict zu CategoryDefinition
                cat_def = CategoryDefinition(
                    name=category.get('name', ''),
                    definition=category.get('definition', ''),
                    examples=category.get('examples', []),
                    rules=category.get('rules', []),
                    subcategories=category.get('subcategories', {}),
                    added_date=datetime.now().strftime("%Y-%m-%d"),
                    modified_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                # Nutze zentrale Validierung
                is_valid, issues = self.validator.validate_category(cat_def)
                
                if is_valid:
                    valid_categories.append(category)
                    print(f"✓ Kategorie '{category.get('name', '')}' validiert")
                else:
                    print(f"\nValidierungsprobleme für '{category.get('name', '')}':")
                    for issue in issues:
                        print(f"- {issue}")
                    
                    # Versuche Kategorie zu verbessern
                    if 'definition zu kurz' in ' '.join(issues):
                        enhanced_def = self._enhance_category_definition(category)
                        if enhanced_def:
                            category['definition'] = enhanced_def
                            valid_categories.append(category)
                            print(f"✓ Kategorie nach Verbesserung validiert")
                            
            except Exception as e:
                print(f"Fehler bei Validierung: {str(e)}")
                continue
                
        return valid_categories

    def _validate_category(self, category: Dict) -> bool:
        """
        Validiert eine einzelne Kategorie nach definierten Kriterien.
        
        Args:
            category: Zu validierende Kategorie
            
        Returns:
            bool: True wenn die Kategorie valide ist
        """
        try:
            # 1. Prüfe ob alle erforderlichen Felder vorhanden sind
            required_fields = {
                'name': str,
                'definition': str,
                'example': str,
                'new_subcategories': list,
                'existing_subcategories': list,
                'justification': str,
                'confidence': dict
            }
            
            for field, expected_type in required_fields.items():
                if field not in category:
                    print(f"Warnung: Fehlendes Feld '{field}' in Kategorie")
                    return False
                if not isinstance(category[field], expected_type):
                    print(f"Warnung: Falscher Datentyp für '{field}' in Kategorie")
                    return False
            
            # 2. Prüfe Inhaltsqualität
            name = category['name']
            definition = category['definition']
            example = category['example']
            
            # Name-Validierung
            if len(name) < 3 or len(name) > 50:
                print(f"Warnung: Ungültige Namenslänge für '{name}'")
                return False
            
            # Definition-Validierung
            if len(definition.split()) < self.MIN_DEFINITION_WORDS:
                print(f"Warnung: Definition zu kurz für '{name}'")
                return False
            
            # Beispiel-Validierung
            if len(example) < 10:
                print(f"Warnung: Beispiel zu kurz für '{name}'")
                return False
            
            # 3. Prüfe Subkategorien
            subcats = category['new_subcategories'] + category['existing_subcategories']
            if not subcats:
                print(f"Warnung: Keine Subkategorien für '{name}'")
                return False
            
            # 4. Prüfe Confidence-Werte
            confidence = category['confidence']
            required_confidence = {'category', 'subcategories'}
            if not all(key in confidence for key in required_confidence):
                print(f"Warnung: Unvollständige Confidence-Werte für '{name}'")
                return False
            
            # 5. Prüfe auf englische Wörter im Namen
            english_indicators = {'research', 'development',
                             'management', 'system', 'process', 'analysis'}
            name_words = set(name.lower().split())
            if name_words & english_indicators:
                print(f"Warnung: Englische Wörter in Kategoriename '{name}'")
                return False
            
            return True
            
        except Exception as e:
            print(f"Fehler bei der Kategorievalidierung: {str(e)}")
            return False

    def _assess_definition_quality(self, category: Dict) -> float:
        """
        Bewertet die Qualität einer Kategoriendefinition.
        """
        definition = category.get('definition', '')
        
        # Grundlegende Qualitätskriterien
        criteria = {
            'length': len(definition.split()) >= self.MIN_DEFINITION_WORDS,
            'structure': all(x in definition.lower() for x in ['ist', 'umfasst', 'bezeichnet']),
            'specificity': len(set(definition.split())) / len(definition.split()) > 0.5
        }
        
        return sum(criteria.values()) / len(criteria)

    def _integrate_categories(self, 
                            existing: Dict[str, CategoryDefinition],
                            new_categories: List[Dict]) -> Dict[str, CategoryDefinition]:
        """
        Integriert neue Kategorien in das bestehende System.
        """
        integrated = existing.copy()
        
        for category in new_categories:
            name = category['name']
            
            if name in integrated:
                # Update bestehende Kategorie
                current = integrated[name]
                integrated[name] = self._merge_category_definitions(current, category)
            else:
                # Füge neue Kategorie hinzu
                integrated[name] = CategoryDefinition(
                    name=name,
                    definition=category['definition'],
                    examples=[category['example']],
                    rules=[],  # Regeln werden später entwickelt
                    subcategories=dict.fromkeys(category['subcategories']),
                    added_date=datetime.now().strftime("%Y-%m-%d"),
                    modified_date=datetime.now().strftime("%Y-%m-%d")
                )
        
        return integrated

    def _remove_redundant_categories(self, categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        Entfernt redundante Kategorien basierend auf Ähnlichkeitsanalyse.
        
        Args:
            categories: Dictionary mit Kategorien
            
        Returns:
            Dict[str, CategoryDefinition]: Bereinigtes Kategoriensystem
        """
        try:
            # Wenn zu wenige Kategorien vorhanden, keine Bereinigung nötig
            if len(categories) <= 1:
                return categories
                
            unique_categories = {}
            redundant_pairs = []
            
            # Vergleiche jede Kategorie mit jeder anderen
            for name1, cat1 in categories.items():
                is_redundant = False
                
                for name2, cat2 in categories.items():
                    if name1 >= name2:  # Überspringe Selbstvergleich und Duplikate
                        continue
                        
                    # Berechne verschiedene Ähnlichkeitsmetriken
                    name_similarity = self._calculate_text_similarity(name1, name2)
                    def_similarity = self._calculate_text_similarity(cat1.definition, cat2.definition)
                    
                    # Prüfe Subkategorien-Überlappung
                    sub_overlap = len(set(cat1.subcategories.keys()) & set(cat2.subcategories.keys()))
                    sub_similarity = sub_overlap / max(len(cat1.subcategories), len(cat2.subcategories)) if cat1.subcategories else 0
                    
                    # Gewichtete Gesamtähnlichkeit
                    total_similarity = (name_similarity * 0.3 + 
                                     def_similarity * 0.4 + 
                                     sub_similarity * 0.3)
                    
                    # Wenn Kategorien sehr ähnlich sind
                    if total_similarity > 0.8:
                        redundant_pairs.append((name1, name2, total_similarity))
                        is_redundant = True
                        
                        # Dokumentiere die Redundanz
                        print(f"\nRedundante Kategorien gefunden:")
                        print(f"- {name1} und {name2}")
                        print(f"- Ähnlichkeit: {total_similarity:.2f}")
                        print(f"  • Name: {name_similarity:.2f}")
                        print(f"  • Definition: {def_similarity:.2f}")
                        print(f"  • Subkategorien: {sub_similarity:.2f}")
                
                # Behalte nur nicht-redundante Kategorien
                if not is_redundant:
                    unique_categories[name1] = cat1
            
            # Wenn redundante Paare gefunden wurden, merge diese
            if redundant_pairs:
                print(f"\nFüge {len(redundant_pairs)} redundante Kategorienpaare zusammen...")
                for name1, name2, sim in redundant_pairs:
                    if name1 in categories and name2 in categories:
                        merged = self._merge_redundant_categories(
                            categories[name1],
                            categories[name2],
                            sim
                        )
                        unique_categories[merged.name] = merged
            
            return unique_categories
            
        except Exception as e:
            print(f"Fehler beim Entfernen redundanter Kategorien: {str(e)}")
            return categories

    def _validate_category_system(self, categories: Dict[str, CategoryDefinition]) -> bool:
        """Validiert das gesamte Kategoriensystem mit zentraler Validierung."""
        is_valid, issues = self.validator.validate_category_system(categories)
        
        if not is_valid:
            print("\nProbleme im Kategoriensystem:")
            for category, category_issues in issues.items():
                print(f"\n{category}:")
                for issue in category_issues:
                    print(f"- {issue}")
                    
        # Zeige Validierungsstatistiken
        stats = self.validator.get_validation_stats()
        print("\nValidierungsstatistiken:")
        print(f"- Cache-Trefferrate: {stats['cache_hit_rate']*100:.1f}%")
        print(f"- Validierungen gesamt: {stats['total_validations']}")
        print(f"- Cache-Größe: {stats['cache_size']} Einträge")
        
        return is_valid
        
    def _optimize_category_system(self, categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        Optimiert das Kategoriensystem für Konsistenz und Effizienz.
        
        Args:
            categories: Zu optimierendes Kategoriensystem
            
        Returns:
            Dict[str, CategoryDefinition]: Optimiertes Kategoriensystem
        """
        try:
            print("\nOptimiere Kategoriensystem...")
            optimized = {}
            
            # 1. Entferne redundante Kategorien
            unique_categories = self._remove_redundant_categories(categories)
            
            # 2. Optimiere Hierarchie und Struktur
            for name, category in unique_categories.items():
                try:
                    # 2.1 Validiere und optimiere Subkategorien
                    valid_subs = self._validate_subcategories(category.subcategories)
                    
                    # 2.2 Optimiere Definition
                    improved_definition = self._improve_definition(category.definition)
                    
                    # 2.3 Optimiere Beispiele
                    curated_examples = self._curate_examples(category.examples)
                    
                    # 2.4 Generiere Kodierregeln falls nötig
                    if not category.rules:
                        rules = self._generate_coding_rules(category)
                    else:
                        rules = category.rules
                    
                    # 2.5 Erstelle optimierte Kategorie
                    optimized[name] = CategoryDefinition(
                        name=name,
                        definition=improved_definition,
                        examples=curated_examples,
                        rules=rules,
                        subcategories=valid_subs,
                        added_date=category.added_date,
                        modified_date=datetime.now().strftime("%Y-%m-%d")
                    )
                    
                except Exception as e:
                    print(f"Warnung: Fehler bei der Optimierung von '{name}': {str(e)}")
                    # Bei Fehler: Behalte ursprüngliche Kategorie
                    optimized[name] = category
            
            # 3. Validiere finales System
            if self._validate_category_system(optimized):
                print(f"Optimierung abgeschlossen: {len(optimized)} Kategorien")
            else:
                print("Warnung: Optimiertes System erfüllt nicht alle Qualitätskriterien")
            
            return optimized
            
        except Exception as e:
            print(f"Fehler bei der Systemoptimierung: {str(e)}")
            return categories

    def _validate_subcategories(self, subcategories: Dict[str, str]) -> Dict[str, str]:
        """
        Validiert und bereinigt Subkategorien.
        
        Args:
            subcategories: Dictionary mit Subkategorien
            
        Returns:
            Dict[str, str]: Bereinigte Subkategorien
        """
        valid_subcats = {}
        
        for name, definition in subcategories.items():
            # Grundlegende Validierung
            if not name or not isinstance(name, str):
                continue
                
            # Bereinige Namen
            clean_name = name.strip()
            if len(clean_name) < 3:
                continue
                
            # Prüfe auf englische Wörter
            english_indicators = {'and', 'of', 'the', 'for'}
            if any(word in clean_name.lower().split() for word in english_indicators):
                continue
                
            # Füge valide Subkategorie hinzu
            valid_subcats[clean_name] = definition.strip() if definition else ""
            
        return valid_subcats

    def _improve_definition(self, definition: str) -> str:
        """
        Verbessert eine Kategoriendefinition.
        
        Args:
            definition: Ursprüngliche Definition
            
        Returns:
            str: Verbesserte Definition
        """
        if not definition:
            return ""
            
        # Entferne überflüssige Whitespaces
        improved = " ".join(definition.split())
        
        # Stelle sicher, dass die Definition mit Großbuchstaben beginnt
        improved = improved[0].upper() + improved[1:]
        
        # Füge Punkt am Ende hinzu falls nötig
        if not improved.endswith('.'):
            improved += '.'
            
        return improved

    def _curate_examples(self, examples: List[str]) -> List[str]:
        """
        Bereinigt und optimiert Beispiele.
        
        Args:
            examples: Liste von Beispielen
            
        Returns:
            List[str]: Bereinigte Beispiele
        """
        curated = []
        
        for example in examples:
            if not example or not isinstance(example, str):
                continue
                
            # Bereinige Text
            cleaned = " ".join(example.split())
            
            # Prüfe Mindestlänge
            if len(cleaned) < 10:
                continue
                
            curated.append(cleaned)
            
        return curated[:self.MIN_EXAMPLES]  # Begrenzt auf minimale Anzahl

    def _generate_coding_rules(self, category: CategoryDefinition) -> List[str]:
        """
        Generiert Kodierregeln für eine Kategorie.
        
        Args:
            category: Kategorie für die Regeln generiert werden sollen
            
        Returns:
            List[str]: Generierte Kodierregeln
        """
        rules = [
            f"Kodiere Textstellen, die sich auf {category.name} beziehen",
            f"Berücksichtige den Kontext der Aussage",
            f"Bei Unsicherheit dokumentiere die Gründe"
        ]
        
        # Füge kategorienspezifische Regeln hinzu
        if category.subcategories:
            rules.append(f"Prüfe Zuordnung zu Subkategorien: {', '.join(category.subcategories.keys())}")
        
        return rules

    def _merge_redundant_categories(self, 
                                  cat1: CategoryDefinition, 
                                  cat2: CategoryDefinition,
                                  similarity: float) -> CategoryDefinition:
        """
        Führt zwei redundante Kategorien zusammen.
        
        Args:
            cat1: Erste Kategorie
            cat2: Zweite Kategorie
            similarity: Ähnlichkeitswert der Kategorien
            
        Returns:
            CategoryDefinition: Zusammengeführte Kategorie
        """
        # Wähle den kürzeren Namen oder kombiniere bei geringer Namenssimilarität
        name_sim = self._calculate_text_similarity(cat1.name, cat2.name)
        if name_sim > 0.7:
            name = min([cat1.name, cat2.name], key=len)
        else:
            name = f"{cat1.name}_{cat2.name}"
            
        # Kombiniere Definitionen
        definition = self._merge_definitions(cat1.definition, cat2.definition)
        
        # Vereinige Beispiele und entferne Duplikate
        examples = list(set(cat1.examples + cat2.examples))
        
        # Vereinige Regeln
        rules = list(set(cat1.rules + cat2.rules))
        
        # Kombiniere Subkategorien
        subcategories = {**cat1.subcategories, **cat2.subcategories}
        
        # Erstelle neue zusammengeführte Kategorie
        return CategoryDefinition(
            name=name,
            definition=definition,
            examples=examples,
            rules=rules,
            subcategories=subcategories,
            added_date=min(cat1.added_date, cat2.added_date),
            modified_date=datetime.now().strftime("%Y-%m-%d")
        )

    def _merge_definitions(self, def1: str, def2: str) -> str:
        """
        Führt zwei Kategoriendefinitionen intelligent zusammen.
        
        Args:
            def1: Erste Definition
            def2: Zweite Definition
            
        Returns:
            str: Kombinierte Definition
        """
        # Wenn eine Definition leer ist, verwende die andere
        if not def1:
            return def2
        if not def2:
            return def1
            
        # Teile Definitionen in Sätze
        sentences1 = set(sent.strip() for sent in def1.split('.') if sent.strip())
        sentences2 = set(sent.strip() for sent in def2.split('.') if sent.strip())
        
        # Finde einzigartige Aspekte
        unique_sentences = sentences1 | sentences2
        
        # Kombiniere zu einer neuen Definition
        combined = '. '.join(sorted(unique_sentences))
        if not combined.endswith('.'):
            combined += '.'
            
        return combined

    def _track_development(self, 
                          categories: Dict[str, CategoryDefinition],
                          batch_size: int) -> None:
        """
        Dokumentiert die Kategorienentwicklung für Analyse und Reporting.
        """
        timestamp = datetime.now().isoformat()
        
        entry = {
            'timestamp': timestamp,
            'num_categories': len(categories),
            'batch_size': batch_size,
            'categories': {
                name: cat.__dict__ for name, cat in categories.items()
            }
        }
        
        self.development_history.append(entry)

    def _finalize_categories(self, categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        Führt finale Optimierungen und Qualitätsprüfungen durch.
        """
        # 1. Entferne unterentwickelte Kategorien
        finalized = {
            name: cat for name, cat in categories.items()
            if len(cat.examples) >= self.MIN_EXAMPLES
        }
        
        # 2. Entwickle Kodierregeln
        for name, category in finalized.items():
            if not category.rules:
                category.rules = self._generate_coding_rules(category)
        
        # 3. Überprüfe Gesamtqualität
        if not self._validate_category_system(finalized):
            print("Warning: Final category system may have quality issues")
            
        return finalized

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
                "- > 0.800: Excellent reliability",
                "- 0.667 - 0.800: Acceptable reliability",
                "- < 0.667: Poor reliability",
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
        
    def _log_performance(self, 
                        num_segments: int,
                        num_categories: int,
                        processing_time: float) -> None:
        """
        Protokolliert Performance-Metriken.
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'segments_processed': num_segments,
            'categories_developed': num_categories,
            'processing_time': processing_time,
            'segments_per_second': num_segments / processing_time
        }
        
        print("\nPerformance Metrics:")
        print(f"- Segments processed: {num_segments}")
        print(f"- Categories developed: {num_categories}")
        print(f"- Processing time: {processing_time:.2f}s")
        print(f"- Segments/second: {metrics['segments_per_second']:.2f}")
        
        # Speichere Metriken
        self.last_analysis_time = processing_time
        self.batch_results.append(metrics)

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
        self.current_categories = {}
        
    async def code_chunk(self, chunk: str, categories: Optional[Dict[str, CategoryDefinition]]) -> Optional[CodingResult]:
        """Nutzt das asyncio Event Loop, um tkinter korrekt zu starten"""
        try:
            self.categories = self.current_categories or categories
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
        # Speichere die Original-Kategorienamen für spätere Referenz
        self.category_map = {}  # Dictionary zum Mapping von Listbox-Index zu echtem Kategorienamen
        
        current_index = 0
        for cat_name, cat_def in self.categories.items():
            # Hauptkategorie
            display_text = f"{current_index + 1}. {cat_name}"
            self.category_listbox.insert(tk.END, display_text)
            self.category_map[current_index] = {'type': 'main', 'name': cat_name}
            current_index += 1
            
            # Subkategorien
            for sub_name in cat_def.subcategories.keys():
                display_text = f"   {current_index + 1}. {sub_name}"
                self.category_listbox.insert(tk.END, display_text)
                self.category_map[current_index] = {
                    'type': 'sub',
                    'main_category': cat_name,
                    'name': sub_name
                }
                current_index += 1

        # Scrolle zum Anfang der Liste
        self.category_listbox.see(0)
        print("Kategorieliste aktualisiert:")
        for idx, mapping in self.category_map.items():
            if mapping['type'] == 'main':
                print(f"Index {idx}: Hauptkategorie '{mapping['name']}'")
            else:
                print(f"Index {idx}: Subkategorie '{mapping['name']}' von '{mapping['main_category']}'")

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
        """Thread-sichere Kodierungsauswahl mit korrekter Kategoriezuordnung"""
        if not self._is_processing:
            try:
                selection = self.category_listbox.curselection()
                if not selection:
                    messagebox.showwarning("Warnung", "Bitte wählen Sie eine Kategorie aus.")
                    return
                
                index = selection[0]
                
                # Hole die tatsächliche Kategorie aus dem Mapping
                if index not in self.category_map:
                    messagebox.showerror("Fehler", "Ungültiger Kategorieindex")
                    return
                    
                category_info = self.category_map[index]
                print(f"Debug - Ausgewählte Kategorie: {category_info}")  # Debug-Ausgabe
                
                if category_info['type'] == 'main':
                    main_cat = category_info['name']
                    sub_cat = None
                else:
                    main_cat = category_info['main_category']
                    sub_cat = category_info['name']
                
                # Verifiziere die Kategorien
                if main_cat not in self.categories:
                    messagebox.showerror("Fehler", 
                        f"Hauptkategorie '{main_cat}' nicht gefunden.\n"
                        f"Verfügbare Kategorien: {', '.join(self.categories.keys())}")
                    return
                    
                if sub_cat and sub_cat not in self.categories[main_cat].subcategories:
                    messagebox.showerror("Fehler", 
                        f"Subkategorie '{sub_cat}' nicht in '{main_cat}' gefunden.\n"
                        f"Verfügbare Subkategorien: {', '.join(self.categories[main_cat].subcategories.keys())}")
                    return

                # Erstelle bereinigte Kodierung
                self.current_coding = CodingResult(
                    category=main_cat,
                    subcategories=(sub_cat,) if sub_cat else tuple(),
                    justification="Manuelle Kodierung",
                    confidence={'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                    text_references=(self.text_chunk.get("1.0", tk.END)[:100],)
                )
                
                print(f"Manuelle Kodierung erstellt:")
                print(f"- Hauptkategorie: {main_cat}")
                if sub_cat:
                    print(f"- Subkategorie: {sub_cat}")
                
                self._is_processing = False
                self.root.quit()
                
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler bei der Kategorieauswahl: {str(e)}")
                print(f"Fehler bei der Kategorieauswahl: {str(e)}")
                print("Details:")
                import traceback
                traceback.print_exc()
                

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


# --- Klasse: SaturationChecker ---
class SaturationChecker:
    """
    Zentrale Klasse für die Sättigungsprüfung nach Mayring.
    Verarbeitet die Sättigungsmetriken aus der induktiven Kategorienentwicklung.
    """
    
    def __init__(self, config: dict, history: DevelopmentHistory):
        """
        Initialisiert den SaturationChecker mit Konfiguration und History-Tracking.
        
        Args:
            config: Konfigurationsdictionary
            history: DevelopmentHistory Instanz für Logging
        """
        self._history = history

         # Robustere Konfigurationsverarbeitung
        if config is None:
            config = {}
        
        # Konfigurierbare Schwellenwerte
        validation_config = config.get('validation_config', {})
        thresholds = validation_config.get('thresholds', {})
        
        self.MIN_MATERIAL_PERCENTAGE = thresholds.get('MIN_MATERIAL_PERCENTAGE', 70)
        self.STABILITY_THRESHOLD = thresholds.get('STABILITY_THRESHOLD', 3)
        
        # Tracking-Variablen
        self.processed_percentage = 0
        self.stable_iterations = 0
        self.current_batch_size = 0
        self.saturation_history = []

        self.category_usage_history = []
        self.new_category_iterations = 0
        self.max_stable_iterations = self.STABILITY_THRESHOLD
        self.theoretical_coverage_threshold = 0.8
        
        print("\nInitialisierung von SaturationChecker:")
        print(f"- Stabilitätsschwelle: {self.STABILITY_THRESHOLD} Iterationen")
        print(f"- Minimale Materialmenge: {self.MIN_MATERIAL_PERCENTAGE}%")

    def check_saturation(self, 
                        current_categories: Dict[str, CategoryDefinition],
                        coded_segments: List[CodingResult],
                        material_percentage: float) -> Tuple[bool, Dict]:
        """
        Verbesserte Sättigungsprüfung mit dynamischerer Bewertung
        """
        try:
            # 1. Kategoriennutzung analysieren
            category_usage = self._analyze_category_usage(current_categories, coded_segments)
            
            # 2. Theoretische Abdeckung berechnen
            theoretical_coverage = self._calculate_theoretical_coverage(category_usage, current_categories)
            
            # 3. Neue Kategorien-Dynamik prüfen
            is_category_stable = self._check_category_stability(category_usage)
            
            # 4. Gesamte Sättigungsbewertung
            is_saturated = (
                material_percentage >= self.MIN_MATERIAL_PERCENTAGE and
                theoretical_coverage >= self.theoretical_coverage_threshold and
                is_category_stable
            )
            
            # 5. Detaillierte Metriken
            detailed_metrics = {
                'material_processed': material_percentage,
                'theoretical_coverage': theoretical_coverage,
                'stable_iterations': self.stable_iterations,
                'categories_sufficient': is_category_stable,
                'justification': self._generate_saturation_justification(
                    is_saturated, 
                    material_percentage, 
                    theoretical_coverage,
                    is_category_stable
                )
            }
            
            # 6. Dokumentation
            self._history.log_saturation_check(
                material_percentage=material_percentage,
                result="saturated" if is_saturated else "not_saturated",
                metrics=detailed_metrics
            )
            
            return is_saturated, detailed_metrics
            
        except Exception as e:
            print(f"Fehler bei Sättigungsprüfung: {str(e)}")
            return False, {}
            
    def _generate_saturation_status(self,
                                  is_saturated: bool,
                                  material_percentage: float,
                                  theoretical_coverage: float,
                                  stable_iterations: int) -> str:
        """Generiert eine aussagekräftige Statusmeldung."""
        if is_saturated:
            return (
                f"Sättigung erreicht bei {material_percentage:.1f}% des Materials. "
                f"Theoretische Abdeckung: {theoretical_coverage:.2f}. "
                f"Stabil seit {stable_iterations} Iterationen."
            )
        else:
            reasons = []
            if material_percentage < self.MIN_MATERIAL_PERCENTAGE:
                reasons.append(f"Mindestmaterialmenge ({self.MIN_MATERIAL_PERCENTAGE}%) nicht erreicht")
            if theoretical_coverage <= 0.8:
                reasons.append(f"Theoretische Abdeckung ({theoretical_coverage:.2f}) unzureichend")
            if stable_iterations < self.STABILITY_THRESHOLD:
                reasons.append(f"Stabilitätskriterium ({stable_iterations}/{self.STABILITY_THRESHOLD} Iterationen) nicht erfüllt")
                
            return "Keine Sättigung: " + "; ".join(reasons)
            
    def _analyze_category_usage(self, 
                             categories: Dict[str, CategoryDefinition], 
                             coded_segments: List[Union[Dict, CodingResult]]) -> Dict[str, int]:
        """
        Analysiert die Nutzung von Kategorien in den kodierten Segmenten.
        Unterstützt sowohl CodingResult-Objekte als auch Dictionaries.
        
        Returns:
            Dict mit Kategorien und ihrer Häufigkeit
        """
        category_usage = {cat: 0 for cat in categories.keys()}
        
        for segment in coded_segments:
            # Extrahiere Kategorie basierend auf Objekttyp
            category = None
            if hasattr(segment, 'category'):
                category = segment.category
            elif isinstance(segment, dict):
                category = segment.get('category')
            
            if category and category in category_usage:
                category_usage[category] += 1
        
        return category_usage

    def _calculate_theoretical_coverage(self, 
                                   category_usage: Dict[str, int], 
                                   categories: Dict[str, CategoryDefinition]) -> float:
        """
        Berechnet die theoretische Abdeckung basierend auf Kategoriennutzung.
        
        Args:
            category_usage: Häufigkeit der Kategorien
            categories: Gesamtes Kategoriensystem
        
        Returns:
            float: Theoretische Abdeckung zwischen 0 und 1
        """
        # Kategorien mit mindestens einer Kodierung
        used_categories = sum(1 for count in category_usage.values() if count > 0)
        total_categories = len(categories)
        
        # Verhindere Division durch Null
        if total_categories == 0:
            return 0.0
        
        theoretical_coverage = used_categories / total_categories
        return min(1.0, theoretical_coverage)

    def _check_category_stability(self, category_usage: Dict[str, int]) -> bool:
        """
        Prüft die Stabilität der Kategorien.
        
        Args:
            category_usage: Häufigkeit der Kategorien
        
        Returns:
            bool: Sind die Kategorien stabil?
        """
        # Neue Kategorien zählen
        new_categories = sum(1 for count in category_usage.values() if count > 0)
        
        # Tracke neue Kategorien
        if new_categories > len(self.category_usage_history):
            self.new_category_iterations = 0
            self.category_usage_history.append(new_categories)
        else:
            self.new_category_iterations += 1
        
        # Prüfe Stabilität
        is_stable = (
            self.new_category_iterations >= self.max_stable_iterations and
            len(set(self.category_usage_history[-self.max_stable_iterations:])) == 1
        )
        
        return is_stable

    def _generate_saturation_justification(self, 
                                          is_saturated: bool,
                                          material_percentage: float,
                                          theoretical_coverage: float,
                                          categories_stable: bool) -> str:
        """
        Generiert eine Begründung für den Sättigungsstatus.
        """
        if is_saturated:
            return (
                f"Sättigung erreicht bei {material_percentage:.1f}% des Materials. "
                f"Theoretische Abdeckung: {theoretical_coverage:.2f}. "
                f"Kategoriesystem stabil."
            )
        else:
            reasons = []
            if material_percentage < self.MIN_MATERIAL_PERCENTAGE:
                reasons.append(f"Mindestmaterialmenge ({self.MIN_MATERIAL_PERCENTAGE}%) nicht erreicht")
            if theoretical_coverage < self.theoretical_coverage_threshold:
                reasons.append(f"Theoretische Abdeckung ({theoretical_coverage:.2f}) unzureichend")
            if not categories_stable:
                reasons.append("Kategoriesystem noch nicht stabil")
                
            return "Keine Sättigung: " + "; ".join(reasons)

    def add_saturation_metrics(self, metrics: Dict) -> None:
        """
        Fügt neue Sättigungsmetriken zur Historie hinzu.
        
        Args:
            metrics: Sättigungsmetriken aus der Kategorienentwicklung
        """
        self.saturation_history.append(metrics)



# --- Klasse: ResultsExporter ---
# Aufgabe: Export der kodierten Daten und des finalen Kategoriensystems
class ResultsExporter:
    """
    Exports the results of the analysis in various formats (JSON, CSV, Excel, Markdown).
    Supports the documentation of coded chunks and the category system.
    """
    
    def __init__(self, output_dir: str, attribute_labels: Dict[str, str], inductive_coder: InductiveCoder = None,
                 analysis_manager: 'IntegratedAnalysisManager' = None):
        self.output_dir = output_dir
        self.attribute_labels = attribute_labels
        self.inductive_coder = inductive_coder
        self.analysis_manager = analysis_manager
        self.category_colors = {}
        os.makedirs(output_dir, exist_ok=True)

    def _get_consensus_coding(self, segment_codes: List[Dict]) -> Optional[Dict]:
        """
        Ermittelt die Konsens-Kodierung für ein Segment basierend auf einem mehrstufigen Prozess.
        Berücksichtigt Hauptkategorien, Subkategorien und verschiedene Qualitätskriterien.
        
        Args:
            segment_codes: Liste der Kodierungen für ein Segment von verschiedenen Kodierern
                
        Returns:
            Optional[Dict]: Konsens-Kodierung oder None wenn kein Konsens erreicht
        """
        if not segment_codes:
            return None

        # 1. Analyse der Hauptkategorien
        category_counts = Counter(coding['category'] for coding in segment_codes)
        total_coders = len(segment_codes)
        
        # Finde häufigste Hauptkategorie(n)
        max_count = max(category_counts.values())
        majority_categories = [
            category for category, count in category_counts.items()
            if count == max_count
        ]
        
        # Prüfe ob es eine klare Mehrheit gibt (>50%)
        if max_count <= total_coders / 2:
            print(f"Keine Mehrheit für Hauptkategorie gefunden: {dict(category_counts)}")
            return None

        # 2. Wenn es mehrere gleichhäufige Hauptkategorien gibt, verwende Tie-Breaking
        if len(majority_categories) > 1:
            print(f"Gleichstand zwischen Kategorien: {majority_categories}")
            # Sammle alle Kodierungen für die Mehrheitskategorien
            candidate_codings = [
                coding for coding in segment_codes
                if coding['category'] in majority_categories
            ]
            
            # Wähle basierend auf höchster durchschnittlicher Konfidenz
            majority_category = max(
                majority_categories,
                key=lambda cat: sum(
                    float(coding['confidence'].get('total', 0))
                    for coding in candidate_codings
                    if coding['category'] == cat
                ) / len([c for c in candidate_codings if c['category'] == cat])
            )
            print(f"Kategorie '{majority_category}' durch höchste Konfidenz gewählt")
        else:
            majority_category = majority_categories[0]

        # 3. Analyse der Subkategorien für die Mehrheitskategorie
        matching_codings = [
            coding for coding in segment_codes
            if coding['category'] == majority_category
        ]
        
        # Sammle alle verwendeten Subkategorien
        all_subcategories = []
        for coding in matching_codings:
            subcats = coding.get('subcategories', [])
            if isinstance(subcats, (list, tuple)):
                all_subcategories.extend(subcats)
        
        # Zähle Häufigkeit der Subkategorien
        subcat_counts = Counter(all_subcategories)
        
        # Wähle Subkategorien die von mindestens 50% der Kodierer verwendet wurden
        consensus_subcats = [
            subcat for subcat, count in subcat_counts.items()
            if count >= len(matching_codings) / 2
        ]
        
        # 4. Wähle die beste Basiskodierung aus
        base_coding = max(
            matching_codings,
            key=lambda x: self._calculate_coding_quality(x, consensus_subcats)
        )
        
        # 5. Erstelle finale Konsens-Kodierung
        consensus_coding = base_coding.copy()
        consensus_coding['subcategories'] = consensus_subcats
        
        # Dokumentiere den Konsensprozess
        consensus_coding['consensus_info'] = {
            'total_coders': total_coders,
            'category_agreement': max_count / total_coders,
            'subcategory_agreement': len(consensus_subcats) / len(all_subcategories) if all_subcategories else 1.0,
            'source_codings': len(matching_codings)
        }
        
        print(f"\nKonsens-Kodierung erstellt:")
        print(f"- Hauptkategorie: {consensus_coding['category']} ({max_count}/{total_coders} Kodierer)")
        print(f"- Subkategorien: {len(consensus_subcats)} im Konsens")
        print(f"- Übereinstimmung: {(max_count/total_coders)*100:.1f}%")
        
        return consensus_coding

    
    def _get_majority_coding(self, segment_codes: List[Dict]) -> Optional[Dict]:
        # Implementierung ähnlich wie _get_consensus_coding, aber mit einfacher Mehrheit
        pass

    def _get_manual_priority_coding(self, segment_codes: List[Dict]) -> Optional[Dict]:
        # Implementierung für manuelle Priorisierung
        pass

    def _calculate_coding_quality(self, coding: Dict, consensus_subcats: List[str]) -> float:
        """
        Berechnet einen Qualitätsscore für eine Kodierung.
        Berücksichtigt mehrere Faktoren:
        - Konfidenz der Kodierung
        - Übereinstimmung mit Konsens-Subkategorien
        - Qualität der Begründung

        Args:
            coding: Einzelne Kodierung
            consensus_subcats: Liste der Konsens-Subkategorien

        Returns:
            float: Qualitätsscore zwischen 0 und 1
        """
        try:
            # Hole Konfidenzwert (gesamt oder Hauptkategorie)
            if isinstance(coding.get('confidence'), dict):
                confidence = float(coding['confidence'].get('total', 0))
            else:
                confidence = float(coding.get('confidence', 0))

            # Berechne Übereinstimmung mit Konsens-Subkategorien
            coding_subcats = set(coding.get('subcategories', []))
            consensus_subcats_set = set(consensus_subcats)
            if consensus_subcats_set:
                subcat_overlap = len(coding_subcats & consensus_subcats_set) / len(consensus_subcats_set)
            else:
                subcat_overlap = 1.0  # Volle Punktzahl wenn keine Konsens-Subkategorien

            # Bewerte Qualität der Begründung
            justification = coding.get('justification', '')
            if isinstance(justification, str):
                justification_score = min(len(justification.split()) / 20, 1.0)  # Max bei 20 Wörtern
            else:
                justification_score = 0.0  # Keine Begründung vorhanden oder ungültiger Typ

            # Gewichtete Kombination der Faktoren
            quality_score = (
                confidence * 0.5 +          # 50% Konfidenz
                subcat_overlap * 0.3 +      # 30% Subkategorien-Übereinstimmung
                justification_score * 0.2   # 20% Begründungsqualität
            )

            return quality_score

        except Exception as e:
            print(f"Fehler bei der Berechnung der Codierungsqualität: {str(e)}")
            return 0.0  # Rückgabe eines neutralen Scores im Fehlerfall
    
    def export_optimization_analysis(self, 
                                original_categories: Dict[str, CategoryDefinition],
                                optimized_categories: Dict[str, CategoryDefinition],
                                optimization_log: List[Dict]):
        """Exportiert eine detaillierte Analyse der Kategorienoptimierungen."""
        
        analysis_path = os.path.join(self.output_dir, 
                                    f'category_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write("# Analyse der Kategorienoptimierungen\n\n")
            
            f.write("## Übersicht\n")
            f.write(f"- Ursprüngliche Kategorien: {len(original_categories)}\n")
            f.write(f"- Optimierte Kategorien: {len(optimized_categories)}\n")
            f.write(f"- Anzahl der Optimierungen: {len(optimization_log)}\n\n")
            
            f.write("## Detaillierte Optimierungen\n")
            for entry in optimization_log:
                if entry['type'] == 'merge':
                    f.write(f"\n### Zusammenführung zu: {entry['result_category']}\n")
                    f.write(f"- Ursprüngliche Kategorien: {', '.join(entry['original_categories'])}\n")
                    f.write(f"- Zeitpunkt: {entry['timestamp']}\n\n")
                    
                    f.write("#### Ursprüngliche Definitionen:\n")
                    for cat in entry['original_categories']:
                        if cat in original_categories:
                            f.write(f"- {cat}: {original_categories[cat].definition}\n")
                    f.write("\n")
                    
                    if entry['result_category'] in optimized_categories:
                        f.write("#### Neue Definition:\n")
                        f.write(f"{optimized_categories[entry['result_category']].definition}\n\n")
                        
                # Weitere Optimierungstypen hier...
            
            f.write("\n## Statistiken\n")
            f.write(f"- Kategorienreduktion: {(1 - len(optimized_categories) / len(original_categories)) * 100:.1f}%\n")
            
            # Zähle Optimierungstypen
            optimization_types = Counter(entry['type'] for entry in optimization_log)
            f.write("\nOptimierungstypen:\n")
            for opt_type, count in optimization_types.items():
                f.write(f"- {opt_type}: {count}\n")
        
        print(f"Optimierungsanalyse exportiert nach: {analysis_path}")

    # def export_merge_analysis(self, original_categories: Dict[str, CategoryDefinition], 
    #                         merged_categories: Dict[str, CategoryDefinition],
    #                         merge_log: List[Dict]):
    #     """Exportiert eine detaillierte Analyse der Kategorienzusammenführungen."""
    #     analysis_path = os.path.join(self.output_dir, f'merge_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
        
    #     with open(analysis_path, 'w', encoding='utf-8') as f:
    #         f.write("# Analyse der Kategorienzusammenführungen\n\n")
            
    #         f.write("## Übersicht\n")
    #         f.write(f"- Ursprüngliche Kategorien: {len(original_categories)}\n")
    #         f.write(f"- Zusammengeführte Kategorien: {len(merged_categories)}\n")
    #         f.write(f"- Anzahl der Zusammenführungen: {len(merge_log)}\n\n")
            
    #         f.write("## Detaillierte Zusammenführungen\n")
    #         for merge in merge_log:
    #             f.write(f"### {merge['new_category']}\n")
    #             f.write(f"- Zusammengeführt aus: {', '.join(merge['merged'])}\n")
    #             f.write(f"- Ähnlichkeit: {merge['similarity']:.2f}\n")
    #             f.write(f"- Zeitpunkt: {merge['timestamp']}\n\n")
                
    #             f.write("#### Ursprüngliche Definitionen:\n")
    #             for cat in merge['merged']:
    #                 f.write(f"- {cat}: {original_categories[cat].definition}\n")
    #             f.write("\n")
                
    #             f.write("#### Neue Definition:\n")
    #             f.write(f"{merged_categories[merge['new_category']].definition}\n\n")
            
    #         f.write("## Statistiken\n")
    #         f.write(f"- Durchschnittliche Ähnlichkeit bei Zusammenführungen: {sum(m['similarity'] for m in merge_log) / len(merge_log):.2f}\n")
    #         f.write(f"- Kategorienreduktion: {(1 - len(merged_categories) / len(original_categories)) * 100:.1f}%\n")
        
    #     print(f"Merge-Analyse exportiert nach: {analysis_path}")
    #     pass
    
    def _prepare_coding_for_export(self, coding: dict, chunk: str, chunk_id: int, doc_name: str) -> dict:
        """
        Bereitet eine Kodierung für den Export vor.
        """
        try:
            # Extrahiere Attribute aus dem Dateinamen
            attribut1, attribut2 = self._extract_metadata(doc_name)
            
            # Prüfe ob eine gültige Kategorie vorhanden ist
            category = coding.get('category', '')
            
            # Bestimme den Kategorietyp (deduktiv/induktiv)
            # Wichtig: Vergleiche mit den ursprünglichen deduktiven Kategorien
            if category in DEDUKTIVE_KATEGORIEN:
                kategorie_typ = "deduktiv"
            else:
                kategorie_typ = "induktiv"
                print(f"Induktive Kategorie gefunden: {category}")  # Debug-Ausgabe
                
            # Setze Kodiert-Status basierend auf Kategorie
            is_coded = 'Ja' if category and category != "Nicht kodiert" else 'Nein'
            
            raw_keywords = coding.get('keywords', '')
            if isinstance(raw_keywords, list):
                formatted_keywords = [kw.strip() for kw in raw_keywords]  # Leerzeichen entfernen
            else:
                formatted_keywords = raw_keywords.replace("[", "").replace("]", "").replace("'", "").split(",")
                formatted_keywords = [kw.strip() for kw in formatted_keywords]

            # Umwandlung der Liste in einen String
            formatted_keywords_list = ", ".join(formatted_keywords)

            # Export-Dictionary mit allen erforderlichen Feldern
            export_data = {
                'Dokument': doc_name,
                self.attribute_labels['attribut1']: attribut1,
                self.attribute_labels['attribut2']: attribut2,
                'Chunk_Nr': chunk_id,
                'Text': chunk,
                'Paraphrase': coding.get('paraphrase', ''), 
                'Kodiert': is_coded,
                'Hauptkategorie': category,
                'Kategorietyp': kategorie_typ,  # Hier wird der korrekte Typ gesetzt
                'Subkategorien': ', '.join(coding.get('subcategories', [])),
                'Schlüsselwörter': formatted_keywords_list,
                'Begründung': coding.get('justification', ''),
                'Konfidenz': self._format_confidence(coding.get('confidence', {})),
                'Mehrfachkodierung': 'Ja' if len(coding.get('subcategories', [])) > 1 else 'Nein'
            }
            
            return export_data
            
        except Exception as e:
            print(f"Fehler bei der Exportvorbereitung für Chunk {chunk_id}: {str(e)}")
            return {
                'Dokument': doc_name,
                'Chunk_Nr': chunk_id,
                'Text': chunk,
                'Paraphrase': '',
                'Kodiert': 'Nein',
                'Hauptkategorie': 'Fehler bei Verarbeitung',
                'Kategorietyp': 'unbekannt',
                'Begründung': f'Fehler: {str(e)}'
            }

    def _format_confidence(self, confidence: dict) -> str:
        """Formatiert die Konfidenz-Werte"""
        if isinstance(confidence, dict):
            return (
                f"Kategorie: {confidence.get('category', 0):.2f}\n"
                f"Subkategorien: {confidence.get('subcategories', 0):.2f}"
            )
        elif isinstance(confidence, (int, float)):
            return f"{float(confidence):.2f}"
        else:
            return "0.00"

    def _validate_export_data(self, export_data: List[dict]) -> bool:
        """
        Validiert die zu exportierenden Daten.
        
        Args:
            export_data: Liste der aufbereiteten Export-Daten
            
        Returns:
            bool: True wenn Daten valide sind
        """
        required_columns = {
            'Dokument', 'Chunk_Nr', 'Text', 'Kodiert', 
            'Hauptkategorie', 'Kategorietyp', 'Subkategorien', 
            'Begründung', 'Konfidenz', 'Mehrfachkodierung'
        }
        
        try:
            if not export_data:
                print("Warnung: Keine Daten zum Exportieren vorhanden")
                return False
                
            # Prüfe ob alle erforderlichen Spalten vorhanden sind
            for entry in export_data:
                missing_columns = required_columns - set(entry.keys())
                if missing_columns:
                    print(f"Warnung: Fehlende Spalten in Eintrag: {missing_columns}")
                    return False
                    
                # Prüfe Kodiert-Status
                if entry['Kodiert'] not in {'Ja', 'Nein'}:
                    print(f"Warnung: Ungültiger Kodiert-Status: {entry['Kodiert']}")
                    return False
                    
            print("Validierung der Export-Daten erfolgreich")
            return True
            
        except Exception as e:
            print(f"Fehler bei der Validierung der Export-Daten: {str(e)}")
            return False
            
    def _extract_metadata(self, filename: str) -> tuple:
        """Extrahiert Metadaten aus dem Dateinamen"""
        from pathlib import Path
        tokens = Path(filename).stem.split("_")
        attribut1 = tokens[0] if len(tokens) >= 1 else ""
        attribut2 = tokens[1] if len(tokens) >= 2 else ""
        return attribut1, attribut2

    def _initialize_category_colors(self, df: pd.DataFrame) -> None:
        """
        Initialisiert die Farbzuordnung für alle Kategorien einmalig.
        
        Args:
            df: DataFrame mit einer 'Hauptkategorie' Spalte
        """
        if not self.category_colors:  # Nur initialisieren wenn noch nicht geschehen
            # Hole alle eindeutigen Hauptkategorien außer 'Nicht kodiert'
            categories = sorted([cat for cat in df['Hauptkategorie'].unique() 
                              if cat != 'Nicht kodiert'])
            
            # Generiere Pastellfarben
            colors = self._generate_pastel_colors(len(categories))
            
            # Erstelle Mapping in alphabetischer Reihenfolge
            self.category_colors = {
                category: color for category, color in zip(categories, colors)
            }
            
            # Füge 'Nicht kodiert' mit grauer Farbe hinzu
            if 'Nicht kodiert' in df['Hauptkategorie'].unique():
                self.category_colors['Nicht kodiert'] = 'CCCCCC'
            
            print("\nFarbzuordnung initialisiert:")
            for cat, color in self.category_colors.items():
                print(f"- {cat}: {color}")


    def _generate_pastel_colors(self, num_colors):
        """
        Generiert eine Palette mit Pastellfarben.
        
        Args:
            num_colors (int): Anzahl der benötigten Farben
        
        Returns:
            List[str]: Liste von Hex-Farbcodes in Pastelltönen
        """
        import colorsys
        
        pastel_colors = []
        for i in range(num_colors):
            # Wähle Hue gleichmäßig über Farbkreis
            hue = i / num_colors
            # Konvertiere HSV zu RGB mit hoher Helligkeit und Sättigung
            rgb = colorsys.hsv_to_rgb(hue, 0.4, 0.95)
            # Konvertiere RGB zu Hex
            hex_color = 'FF{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255), 
                int(rgb[1] * 255), 
                int(rgb[2] * 255)
            )
            pastel_colors.append(hex_color)
        
        return pastel_colors

    
                    
    def _export_frequency_analysis(self, writer, df_coded: pd.DataFrame, attribut1_label: str, attribut2_label: str) -> None:
        try:
            # Hole alle Datensätze, auch "Nicht kodiert"
            df_all = df_coded.copy()
            
            # Hole eindeutige Hauptkategorien, inkl. "Nicht kodiert"
            main_categories = df_all['Hauptkategorie'].unique()
            category_colors = {cat: color for cat, color in zip(main_categories, self._generate_pastel_colors(len(main_categories)))}

            if 'Häufigkeitsanalysen' not in writer.sheets:
                writer.book.create_sheet('Häufigkeitsanalysen')
            
            worksheet = writer.sheets['Häufigkeitsanalysen']
            worksheet.delete_rows(1, worksheet.max_row)  # Bestehende Daten löschen

            current_row = 1
            
            # Formatierungsstile
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            header_font = Font(bold=True)
            title_font = Font(bold=True, size=12)
            total_font = Font(bold=True)
            
            # Rahmenlinien definieren
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )

            # 1. Hauptkategorien nach Dokumenten
            cell = worksheet.cell(row=current_row, column=1, value="1. Verteilung der Hauptkategorien")
            cell.font = title_font
            current_row += 2

            # Pivot-Tabelle für Hauptkategorien, inkl. "Nicht kodiert"
            pivot_main = pd.pivot_table(
                df_all,
                index=['Hauptkategorie'],
                columns='Dokument',
                values='Chunk_Nr',
                aggfunc='count',
                margins=True,
                margins_name='Gesamt',
                fill_value=0
            )

            # Header mit Rahmen
            header_row = current_row
            headers = ['Hauptkategorie'] + list(pivot_main.columns)
            for col, header in enumerate(headers, 1):
                cell = worksheet.cell(row=header_row, column=col, value=str(header))
                cell.font = header_font
                cell.border = thin_border
            current_row += 1

            # Daten mit Rahmen und Farbkodierung
            for idx, row in pivot_main.iterrows():
                is_total = idx == 'Gesamt'
                
                # Hauptkategorie-Zelle
                cell = worksheet.cell(row=current_row, column=1, value=str(idx))
                cell.border = thin_border
                
                # Farbkodierung für Hauptkategorien
                if not is_total and idx in self.category_colors:
                    color = self.category_colors[idx]
                    cell.fill = PatternFill(start_color=color, end_color=color, fill_type='solid')
                                
                if is_total:
                    cell.font = total_font
                
                # Datenzellen
                for col, value in enumerate(row, 2):
                    cell = worksheet.cell(row=current_row, column=col, value=value)
                    cell.border = thin_border
                    
                    if is_total or col == len(row) + 2:  # Randsummen
                        cell.font = total_font
                
                current_row += 1

            current_row += 2

            # 2. Subkategorien-Hierarchie (nur für kodierte Segmente)
            cell = worksheet.cell(row=current_row, column=1, value="2. Subkategorien nach Hauptkategorien")
            cell.font = title_font
            current_row += 2

            # Filtere "Nicht kodiert" für Subkategorien-Analyse aus
            df_sub = df_all[df_all['Hauptkategorie'] != "Nicht kodiert"].copy()
            df_sub['Subkategorie'] = df_sub['Subkategorien'].str.split(', ')
            df_sub = df_sub.explode('Subkategorie')
            
            # Erstelle Pivot-Tabelle mit korrekten Spalten-Labels
            pivot_sub = pd.pivot_table(
                df_sub,
                index=['Hauptkategorie', 'Subkategorie'],
                columns='Dokument',
                values='Chunk_Nr',
                aggfunc='count',
                margins=True,
                margins_name='Gesamt',
                fill_value=0
            )

            # Formatierte Spaltenbezeichnungen
            formatted_columns = []
            for col in pivot_sub.columns:
                if isinstance(col, tuple):
                    # Extrahiere die Metadaten aus dem Dateinamen
                    col_parts = [str(part) for part in col if part and part != '']
                    formatted_columns.append(' - '.join(col_parts))
                else:
                    formatted_columns.append(str(col))

            # Header mit Rahmen
            header_row = current_row
            headers = ['Hauptkategorie', 'Subkategorie'] + formatted_columns
            for col, header in enumerate(headers, 1):
                cell = worksheet.cell(row=header_row, column=col, value=header)
                cell.font = header_font
                cell.border = thin_border
            current_row += 1

            # Daten mit hierarchischer Struktur und Formatierung
            current_main_cat = None
            for index, row in pivot_sub.iterrows():
                is_total = isinstance(index, str) and index == 'Gesamt'
                
                if isinstance(index, tuple):
                    main_cat, sub_cat = index
                    # Hauptkategorie-Zeile
                    worksheet.cell(row=current_row, column=1, value=main_cat).border = thin_border
                    cell = worksheet.cell(row=current_row, column=2, value=sub_cat)
                    cell.border = thin_border
                    
                    # Datenzellen
                    for col, value in enumerate(row, 3):
                        cell = worksheet.cell(row=current_row, column=col, value=value)
                        cell.border = thin_border
                else:
                    # Randsummen-Zeile
                    cell = worksheet.cell(row=current_row, column=1, value=index)
                    cell.font = total_font
                    cell.border = thin_border
                    
                    # Leere Zelle für Subkategorie-Spalte
                    worksheet.cell(row=current_row, column=2, value='').border = thin_border
                    
                    # Datenzellen für Randsummen
                    for col, value in enumerate(row, 3):
                        cell = worksheet.cell(row=current_row, column=col, value=value)
                        cell.font = total_font
                        cell.border = thin_border
                
                current_row += 1


            # 3. Attribut-Analysen
            cell = worksheet.cell(row=current_row, column=1, value="3. Verteilung nach Attributen")
            cell.font = title_font
            current_row += 2

            # 3.1 Attribut 1
            cell = worksheet.cell(row=current_row, column=1, value=f"3.1 Verteilung nach {attribut1_label}")
            cell.font = title_font
            current_row += 1

            # Analyse für Attribut 1
            attr1_counts = df_coded[attribut1_label].value_counts()
            attr1_counts['Gesamt'] = attr1_counts.sum()

            # Header
            headers = [attribut1_label, 'Anzahl']
            for col, header in enumerate(headers, 1):
                cell = worksheet.cell(row=current_row, column=col, value=header)
                cell.font = header_font
                cell.border = thin_border
            current_row += 1

            # Daten für Attribut 1
            for idx, value in attr1_counts.items():
                # Wert-Zelle
                cell = worksheet.cell(row=current_row, column=1, value=idx)
                cell.border = thin_border
                if idx == 'Gesamt':
                    cell.font = total_font
                
                # Anzahl-Zelle
                cell = worksheet.cell(row=current_row, column=2, value=value)
                cell.border = thin_border
                if idx == 'Gesamt':
                    cell.font = total_font
                
                current_row += 1

            current_row += 2

            # 3.2 Attribut 2
            cell = worksheet.cell(row=current_row, column=1, value=f"3.2 Verteilung nach {attribut2_label}")
            cell.font = title_font
            current_row += 1

            # Analyse für Attribut 2
            attr2_counts = df_coded[attribut2_label].value_counts()
            attr2_counts['Gesamt'] = attr2_counts.sum()

            # Header
            headers = [attribut2_label, 'Anzahl']
            for col, header in enumerate(headers, 1):
                cell = worksheet.cell(row=current_row, column=col, value=header)
                cell.font = header_font
                cell.border = thin_border
            current_row += 1

            # Daten für Attribut 2
            for idx, value in attr2_counts.items():
                # Wert-Zelle
                cell = worksheet.cell(row=current_row, column=1, value=idx)
                cell.border = thin_border
                if idx == 'Gesamt':
                    cell.font = total_font
                
                # Anzahl-Zelle
                cell = worksheet.cell(row=current_row, column=2, value=value)
                cell.border = thin_border
                if idx == 'Gesamt':
                    cell.font = total_font
                
                current_row += 1

            current_row += 2

            # 3.3 Kreuztabelle der Attribute
            cell = worksheet.cell(row=current_row, column=1, value="3.3 Kreuztabelle der Attribute")
            cell.font = title_font
            current_row += 1

            # Erstelle Kreuztabelle
            cross_tab = pd.crosstab(
                df_coded[attribut1_label], 
                df_coded[attribut2_label],
                margins=True,
                margins_name='Gesamt'
            )

            # Header für Kreuztabelle
            headers = [attribut1_label] + list(cross_tab.columns)
            for col, header in enumerate(headers, 1):
                cell = worksheet.cell(row=current_row, column=col, value=header)
                cell.font = header_font
                cell.border = thin_border
            current_row += 1

            # Daten für Kreuztabelle
            for idx, row in cross_tab.iterrows():
                # Index-Zelle
                cell = worksheet.cell(row=current_row, column=1, value=idx)
                cell.border = thin_border
                if idx == 'Gesamt':
                    cell.font = total_font
                
                # Datenzellen
                for col, value in enumerate(row, 2):
                    cell = worksheet.cell(row=current_row, column=col, value=value)
                    cell.border = thin_border
                    # Fettdruck für Randsummen (letzte Zeile oder letzte Spalte)
                    if idx == 'Gesamt' or col == len(row) + 2:
                        cell.font = total_font
                
                current_row += 1

            # Passe Spaltenbreiten an
            for col in worksheet.columns:
                max_length = 0
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                    worksheet.column_dimensions[col[0].column_letter].width = min(max_length + 2, 40)

            print("Häufigkeitsanalysen erfolgreich exportiert")
            
        except Exception as e:
            print(f"Fehler bei Häufigkeitsanalysen: {str(e)}")
            import traceback
            traceback.print_exc()

    

    
        
    

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


    def _format_reliability_worksheet(self, worksheet) -> None:
        """
        Formatiert das Reliability Report Worksheet und entfernt Markdown-Formatierungen.
        
        Args:
            worksheet: Das zu formatierende Worksheet-Objekt
        """
        try:
            # Importiere Styling-Klassen
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            
            # Definiere Stile
            title_font = Font(bold=True, size=14)
            header_font = Font(bold=True, size=12)
            normal_font = Font(size=11)
            header_fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')
            border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )

            # Setze Spaltenbreiten
            worksheet.column_dimensions['A'].width = 40
            worksheet.column_dimensions['B'].width = 20

            # Formatiere Zellen und entferne Markdown
            for row in worksheet.iter_rows():
                for cell in row:
                    if cell.value and isinstance(cell.value, str):
                        # Entferne Markdown-Formatierungen
                        value = cell.value
                        # Entferne Überschriften-Markierungen
                        if value.startswith('# '):
                            value = value.replace('# ', '')
                            cell.font = title_font
                        elif value.startswith('## '):
                            value = value.replace('## ', '')
                            cell.font = header_font
                        elif value.startswith('### '):
                            value = value.replace('### ', '')
                            cell.font = header_font
                        
                        # Entferne Aufzählungszeichen
                        if value.startswith('- '):
                            value = value.replace('- ', '')
                        elif value.startswith('* '):
                            value = value.replace('* ', '')
                        
                        # Aktualisiere Zellenwert
                        cell.value = value
                        
                        # Grundformatierung
                        cell.alignment = Alignment(wrap_text=True, vertical='top')
                        if not cell.font:
                            cell.font = normal_font

                        # Formatiere Überschriften
                        if row[0].row == 1:  # Erste Zeile
                            cell.font = title_font
                        elif cell.column == 1 and value and ':' not in value:
                            cell.font = header_font
                            cell.fill = header_fill

                        # Rahmen für alle nicht-leeren Zellen
                        if value:
                            cell.border = border

                        # Spezielle Formatierung für Tabellenzellen
                        if '|' in str(worksheet.cell(row=1, column=cell.column).value):
                            cell.alignment = Alignment(horizontal='center', vertical='center')
                            cell.border = border

        except Exception as e:
            print(f"Warnung: Formatierung des Reliability-Worksheets fehlgeschlagen: {str(e)}")
            import traceback
            traceback.print_exc()

    async def export_results(self,
                        codings: List[Dict],
                        reliability: float,
                        categories: Dict[str, CategoryDefinition],
                        chunks: Dict[str, List[str]],
                        revision_manager: 'CategoryRevisionManager',
                        export_mode: str = "consensus",
                        original_categories: Dict[str, CategoryDefinition] = None,
                        inductive_coder: 'InductiveCoder' = None) -> None:
        """
        Exportiert die Analyseergebnisse mit Konsensfindung zwischen Kodierern.
        """
        try:
            # Wenn inductive_coder als Parameter übergeben wurde, aktualisiere das Attribut
            if inductive_coder:
                self.inductive_coder = inductive_coder

            # Erstelle Zeitstempel für den Dateinamen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"QCA-AID_Analysis_{timestamp}.xlsx"
            filepath = os.path.join(self.output_dir, filename)
            
            # Hole die Bezeichnungen für die Attribute
            attribut1_label = self.attribute_labels['attribut1']
            attribut2_label = self.attribute_labels['attribut2']

            # Berechne Statistiken
            total_segments = len(codings)
            coders = list(set(c.get('coder_id', 'unknown') for c in codings))
            total_coders = len(coders)
            category_frequencies = Counter(c.get('category', 'unknown') for c in codings)
            
            print(f"\nVerarbeite {len(codings)} Kodierungen...")
            print(f"Gefunden: {total_segments} Segmente, {total_coders} Kodierer")

            # Gruppiere Kodierungen nach Segmenten
            segment_codings = {}
            for coding in codings:
                segment_id = coding.get('segment_id')
                if segment_id:
                    if segment_id not in segment_codings:
                        segment_codings[segment_id] = []
                    segment_codings[segment_id].append(coding)

            # Erstelle Konsens-Kodierungen
            consensus_codings = []
            for segment_id, segment_codes in segment_codings.items():
                if export_mode == "consensus":
                    consensus = self._get_consensus_coding(segment_codes)
                elif export_mode == "majority":
                    consensus = self._get_majority_coding(segment_codes)
                elif export_mode == "manual_priority":
                    consensus = self._get_manual_priority_coding(segment_codes)
                else:
                    raise ValueError(f"Ungültiger export_mode: {export_mode}")

                if consensus:
                    consensus_codings.append(consensus)

            print(f"Konsens-Kodierungen erstellt: {len(consensus_codings)}")
            
            # Bereite Export-Daten vor
            export_data = []
            for coding in consensus_codings:
                segment_id = coding.get('segment_id', '')
                if not segment_id:
                    continue

                try:
                    doc_name = segment_id.split('_chunk_')[0]
                    chunk_id = int(segment_id.split('_chunk_')[1])
                    chunk_text = chunks[doc_name][chunk_id]
                    export_entry = self._prepare_coding_for_export(coding, chunk_text, chunk_id, doc_name)
                    export_data.append(export_entry)
                except Exception as e:
                    print(f"Fehler bei Verarbeitung von Segment {segment_id}: {str(e)}")
                    continue

            # Validiere Export-Daten
            if not self._validate_export_data(export_data):
                print("Fehler: Keine validen Export-Daten vorhanden")
                return

            # Erstelle DataFrames
            df_details = pd.DataFrame(export_data)
            df_coded = df_details[df_details['Kodiert'] == 'Ja'].copy()

            # Initialisiere Farbzuordnung einmalig für alle Sheets
            self._initialize_category_colors(df_details)

            print(f"DataFrames erstellt: {len(df_details)} Gesamt, {len(df_coded)} Kodiert")

            # Exportiere nach Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Erstelle ein leeres Workbook
                if not hasattr(writer, 'book') or writer.book is None:
                    writer.book = Workbook()
                
                # 1. Kodierte Segmente
                print("\nExportiere kodierte Segmente...")
                df_details.to_excel(writer, sheet_name='Kodierte_Segmente', index=False)
                self._format_worksheet(writer.sheets['Kodierte_Segmente'], as_table=True)
                
                # 2. Häufigkeitsanalysen nur wenn kodierte Daten vorhanden
                if not df_coded.empty:
                    print("\nExportiere Häufigkeitsanalysen...")
                    self._export_frequency_analysis(writer, df_coded, attribut1_label, attribut2_label)
                                        
                # 3. Exportiere weitere Analysen
                if revision_manager and hasattr(revision_manager, 'changes'):
                    print("\nExportiere Revisionshistorie...")
                    revision_manager._export_revision_history(writer, revision_manager.changes)
                
                # 4. Exportiere Intercoderanalyse
                if segment_codings:
                    print("\nExportiere Intercoderanalyse...")
                    self._export_intercoder_analysis(
                        writer, 
                        segment_codings,
                        reliability
                    )

                # 5. Exportiere Reliabilitätsbericht
                if inductive_coder:
                    print("\nExportiere Reliabilitätsbericht...")
                    self._export_reliability_report(
                        writer,
                        reliability=reliability,
                        total_segments=len(segment_codings),
                        total_coders=total_coders,
                        category_frequencies=category_frequencies
                    )

                # 6. Exportiere Kategorienentwicklung wenn vorhanden
                if original_categories and hasattr(self.analysis_manager, 'category_optimizer') and self.analysis_manager.category_optimizer.optimization_log:
                    print("\nExportiere Kategorienentwicklung...")
                    self.export_optimization_analysis(
                        original_categories=original_categories,
                        optimized_categories=categories,
                        optimization_log=self.analysis_manager.category_optimizer.optimization_log
                    )

                # Stelle sicher, dass mindestens ein Sheet sichtbar ist
                if len(writer.book.sheetnames) == 0:
                    writer.book.create_sheet('Leeres_Sheet')
                
                # Setze alle Sheets auf sichtbar
                for sheet in writer.book.sheetnames:
                    writer.book[sheet].sheet_state = 'visible'

                print(f"\nErgebnisse erfolgreich exportiert nach: {filepath}")
                print(f"- {len(consensus_codings)} Konsens-Kodierungen")
                print(f"- {len(segment_codings)} Segmente analysiert")
                print(f"- Reliabilität: {reliability:.3f}")

        except Exception as e:
            print(f"Fehler beim Excel-Export: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _format_worksheet(self, worksheet, as_table: bool = False) -> None:
        """Formatiert das Detail-Worksheet"""
        try:
            # Prüfe ob Daten vorhanden sind
            if worksheet.max_row < 2:
                print(f"Warnung: Worksheet '{worksheet.title}' enthält keine Daten")
                return

            # Spaltenbreiten definieren
            column_widths = {
                'A': 30, 'B': 15, 'C': 15, 'D': 5, 'E': 40, 
                'F': 40, 'G': 5, 'H': 20, 'I': 15, 'J': 40, 
                'K': 40, 'L': 40, 'M': 15, 'N': 15
            }

            # Hole alle Zeilen als DataFrame für Farbzuordnung
            data = []
            headers = []
            for idx, row in enumerate(worksheet.iter_rows(values_only=True), 1):
                if idx == 1:
                    headers = list(row)
                else:
                    data.append(row)
            
            df = pd.DataFrame(data, columns=headers)
            
            # Setze Spaltenbreiten
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width

            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter
            from openpyxl.worksheet.table import Table, TableStyleInfo
            
            # Definiere Styles
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            header_font = Font(bold=True)
            header_fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')
            
            # Eindeutige Hauptkategorien extrahieren
            main_categories = set(worksheet.cell(row=row, column=8).value 
                                for row in range(2, worksheet.max_row + 1) 
                                if worksheet.cell(row=row, column=8).value)

            # Header formatieren
            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(wrap_text=True, vertical='center', horizontal='center')
                cell.border = thin_border

            # Daten formatieren
            for row in worksheet.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=False, vertical='top')
                    cell.border = thin_border

                    # Farbkodierung für Hauptkategorien (8. Spalte)
                    if cell.column == 8 and cell.value in self.category_colors:
                        cell.fill = PatternFill(
                            start_color=self.category_colors[cell.value], 
                            end_color=self.category_colors[cell.value], 
                            fill_type='solid'
                        )

            # Excel-Tabelle erstellen wenn gewünscht
            if as_table:
                try:
                    # Entferne vorhandene Tabellen sicher
                    table_names = list(worksheet.tables.keys()).copy()
                    for table_name in table_names:
                        del worksheet.tables[table_name]
                    
                    # Sichere Bestimmung der letzten Spalte und Zeile
                    last_col_index = worksheet.max_column
                    last_col_letter = get_column_letter(last_col_index)
                    last_row = worksheet.max_row
                    
                    # Generiere eindeutigen Tabellennamen
                    safe_table_name = f"Table_{worksheet.title.replace(' ', '_')}"
                    
                    # Tabellenverweis generieren
                    table_ref = f"A1:{last_col_letter}{last_row}"
                    
                    # AutoFilter aktivieren
                    # worksheet.auto_filter.ref = table_ref
                    
                    # Neue Tabelle mit sicherer Namensgebung
                    tab = Table(displayName=safe_table_name, ref=table_ref)
                    
                    # Tabellenstil definieren mit Fallback
                    style = TableStyleInfo(
                        name="TableStyleMedium2",
                        showFirstColumn=False,
                        showLastColumn=False,
                        showRowStripes=True,
                        showColumnStripes=False
                    )
                    tab.tableStyleInfo = style
                    
                    # Tabelle zum Worksheet hinzufügen
                    worksheet.add_table(tab)
                    
                    print(f"Tabelle '{safe_table_name}' erfolgreich erstellt")
                    
                except Exception as table_error:
                    print(f"Warnung bei Tabellenerstellung: {str(table_error)}")
                    # Fallback: Nur Formatierung ohne Tabelle
                    print("Tabellenerstellung übersprungen - nur Formatierung angewendet")

            print(f"Worksheet '{worksheet.title}' erfolgreich formatiert")
            
        except Exception as e:
            print(f"Fehler bei der Formatierung von {worksheet.title}: {str(e)}")
            import traceback
            traceback.print_exc()

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

                # Debug-Ausgabe
                # print(f"Segment ID: {segment_id}, Categories: {categories}, Text Chunk: {text_chunk}")

                # Überprüfen Sie die Struktur der 'justification'-Felder
                justifications = []
                for c in codings:
                    justification = c.get('justification', '')
                    if isinstance(justification, dict):
                        # Wenn 'justification' ein Dictionary ist, nehmen Sie einen bestimmten Schlüssel
                        justification_text = justification.get('text', '')
                    elif isinstance(justification, str):
                        # Wenn 'justification' ein String ist, verwenden Sie ihn direkt
                        justification_text = justification
                    else:
                        # Wenn 'justification' ein unerwarteter Typ ist, setzen Sie ihn auf einen leeren String
                        justification_text = ''
                    justifications.append(justification_text)

                # print(f"Justifications: {justifications}")

                row_data = [
                    segment_id,
                    text_chunk,
                    len(codings),
                    "Vollständig" if category_agreement else "Keine Übereinstimmung",
                    ' | '.join(set(categories)),
                    '\n'.join([j[:100] + '...' for j in justifications])
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
                    '\n'.join([j[:100] + '...' for j in justifications])
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
    
    def __init__(self, output_dir: str, config: Dict):
        """
        Initializes the CategoryRevisionManager with necessary tracking mechanisms.
        
        Args:
            output_dir (str): Directory for storing revision documentation
        """
        self.config = config
        self.output_dir = output_dir
        self.changes: List[CategoryChange] = []
        self.revision_log_path = os.path.join(output_dir, "category_revisions.json")
        
        # Define quality thresholds for category system
        self.validator = CategoryValidator(config)

        # Load existing revision history if available
        self._load_revision_history()

    def revise_category_system(self, 
                                categories: Dict[str, CategoryDefinition],
                                coded_segments: List[CodingResult],
                                material_percentage: float) -> Dict[str, CategoryDefinition]:
        """
        Implementiert eine umfassende Kategorienrevision nach Mayring.
        
        Args:
            categories: Aktuelles Kategoriensystem
            coded_segments: Bisher kodierte Textsegmente
            material_percentage: Prozentsatz des verarbeiteten Materials
            
        Returns:
            Dict[str, CategoryDefinition]: Überarbeitetes Kategoriensystem
        """
        try:
            print(f"\nStarte Kategorienrevision bei {material_percentage:.1f}% des Materials")
            revised_categories = categories.copy()
            
            # 1. Analyse des aktuellen Kategoriensystems
            print("\nAnalysiere aktuelles Kategoriensystem...")
            is_valid, system_issues = self.validator.validate_category_system(revised_categories)
            
            if not is_valid:
                print("\nProbleme im Kategoriensystem gefunden:")
                for category, issues in system_issues.items():
                    print(f"\n{category}:")
                    for issue in issues:
                        print(f"- {issue}")
                
                # Versuche automatische Bereinigung
                print("\nStarte automatische Bereinigung...")
                revised_categories =  self.cleaner.clean_categories(
                    revised_categories, 
                    system_issues
                )
                
                # Prüfe Ergebnis der Bereinigung
                is_valid, remaining_issues = self.validator.validate_category_system(revised_categories)
                if not is_valid:
                    print("\nNach Bereinigung verbleibende Probleme:")
                    for category, issues in remaining_issues.items():
                        print(f"\n{category}:")
                        for issue in issues:
                            print(f"- {issue}")
            
            # 2. Analyse der Kategoriennutzung
            print("\nAnalysiere Kategoriennutzung...")
            usage_metrics = self._analyze_category_usage(coded_segments)
            
            # Identifiziere problematische Kategorien basierend auf Nutzung
            problematic_categories = set()
            for category_name, metrics in usage_metrics.items():
                if metrics['frequency'] < 2:  # Wenig genutzte Kategorien
                    problematic_categories.add(category_name)
                    print(f"- '{category_name}': Selten verwendet ({metrics['frequency']} mal)")
                elif metrics['avg_confidence'] < 0.7:  # Niedrige Kodierungssicherheit
                    problematic_categories.add(category_name)
                    print(f"- '{category_name}': Niedrige Kodiersicherheit ({metrics['avg_confidence']:.2f})")
            
            # 3. Kategorienanpassungen
            if problematic_categories:
                print(f"\nBearbeite {len(problematic_categories)} problematische Kategorien...")
                
                for category_name in problematic_categories:
                    # Dokumentiere den Ausgangszustand
                    old_category = revised_categories[category_name]
                    
                    if category_name in revised_categories:
                        metrics = usage_metrics[category_name]
                        
                        # Entscheidung über Maßnahmen
                        if metrics['frequency'] < 2:
                            # Prüfe auf ähnliche Kategorien
                            similar_category = self._find_similar_category(
                                revised_categories[category_name],
                                revised_categories,
                                usage_metrics
                            )
                            if similar_category:
                                # Kategorien zusammenführen
                                revised_categories = self._merge_categories(
                                    category_name,
                                    similar_category,
                                    revised_categories
                                )
                                
                                # Dokumentiere Zusammenführung
                                self.changes.append(CategoryChange(
                                    category_name=category_name,
                                    change_type='merge',
                                    description=f"Mit '{similar_category}' zusammengeführt bei {material_percentage:.1f}% des Materials",
                                    timestamp=datetime.now().isoformat(),
                                    old_value=old_category.__dict__,
                                    justification=f"Selten verwendet ({metrics['frequency']} mal) und Ähnlichkeit zu '{similar_category}'"
                                ))
                            else:
                                # Kategorie entfernen
                                del revised_categories[category_name]
                                
                                # Dokumentiere Löschung
                                self.changes.append(CategoryChange(
                                    category_name=category_name,
                                    change_type='delete',
                                    description=f"Kategorie entfernt bei {material_percentage:.1f}% des Materials",
                                    timestamp=datetime.now().isoformat(),
                                    old_value=old_category.__dict__,
                                    justification=f"Zu selten verwendet ({metrics['frequency']} mal)"
                                ))
                                
                        elif metrics['avg_confidence'] < 0.7:
                            # Verbessere Kategoriendefinition
                            enhanced_definition = self._enhance_category_definition(
                                revised_categories[category_name],
                                metrics['example_segments']
                            )
                            
                            if enhanced_definition:
                                # Aktualisiere Kategorie
                                revised_categories[category_name] = revised_categories[category_name]._replace(
                                    definition=enhanced_definition,
                                    modified_date=datetime.now().strftime("%Y-%m-%d")
                                )
                                
                                # Dokumentiere Verbesserung
                                self.changes.append(CategoryChange(
                                    category_name=category_name,
                                    change_type='modify',
                                    description=f"Definition verbessert bei {material_percentage:.1f}% des Materials",
                                    timestamp=datetime.now().isoformat(),
                                    old_value=old_category.__dict__,
                                    new_value=revised_categories[category_name].__dict__,
                                    justification=f"Niedrige Kodiersicherheit ({metrics['avg_confidence']:.2f})"
                                ))
            
            # 4. Finale Validierung
            final_valid, final_issues = self.validator.validate_category_system(revised_categories)
            
            # Zeige Revisionsstatistiken
            print("\nRevisionsstatistiken:")
            print(f"- Kategorien ursprünglich: {len(categories)}")
            print(f"- Kategorien final: {len(revised_categories)}")
            print(f"- Problematische Kategorien behandelt: {len(problematic_categories)}")
            print(f"- System valide nach Revision: {'Ja' if final_valid else 'Nein'}")
            
            # Validierungsstatistiken
            val_stats = self.validator.get_validation_stats()
            print("\nValidierungsstatistiken:")
            print(f"- Cache-Trefferrate: {val_stats['cache_hit_rate']*100:.1f}%")
            print(f"- Validierungen gesamt: {val_stats['total_validations']}")
            
            return revised_categories
            
        except Exception as e:
            print(f"Fehler bei der Kategorienrevision: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return categories  # Im Fehlerfall ursprüngliche Kategorien zurückgeben

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
        Validiert die methodologische Integrität der Kategorien.
        Nutzt die zentrale Validierungsklasse.
        """
        is_valid, issues = self.validator.validate_category_system(categories)
        
        if not is_valid:
            print("\nValidierungsprobleme gefunden:")
            for category, category_issues in issues.items():
                print(f"\n{category}:")
                for issue in category_issues:
                    print(f"- {issue}")
                    
            # Dokumentiere Validierungsprobleme
            self._document_validation_issues(issues)
            
        return is_valid

    def _document_validation_issues(self, issues: Dict[str, List[str]]):
        """Dokumentiert gefundene Validierungsprobleme."""
        timestamp = datetime.now().isoformat()
        
        for category, category_issues in issues.items():
            self.changes.append(CategoryChange(
                category_name=category,
                change_type='validation_issue',
                description=f"Validierungsprobleme gefunden: {', '.join(category_issues)}",
                timestamp=timestamp,
                justification="Automatische Validierung"
            ))
            
        self._save_revision_history()
        
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
                
                # Grundlegende Validierung
                if not self._validate_basic_requirements(category):
                    continue
                
                # Prüfe Definition
                definition = category.get('definition', '')
                if len(definition.split()) < self.MIN_DEFINITION_WORDS:
                    print(f"Warnung: Definition zu kurz für '{category_name}' "
                          f"({len(definition.split())} Wörter)")
                    # Versuche Definition zu erweitern statt sie abzulehnen
                    enhanced_def = self._enhance_category_definition(category)
                    if enhanced_def:
                        category['definition'] = enhanced_def
                    else:
                        continue
                
                # Prüfe Subkategorien
                subcats = (category.get('existing_subcategories', []) + 
                          category.get('new_subcategories', []))
                if not subcats:
                    print(f"Warnung: Keine Subkategorien für '{category_name}'")
                    # Versuche Subkategorien zu generieren
                    if self._generate_subcategories(category):
                        subcats = (category.get('existing_subcategories', []) + 
                                 category.get('new_subcategories', []))
                    else:
                        continue
                
                # Prüfe Konfidenz
                confidence = category.get('confidence', {})
                if not all(0 <= confidence.get(k, 0) <= 1 for k in ['category', 'subcategories']):
                    print(f"Warnung: Ungültige Konfidenzwerte für '{category_name}'")
                    # Setze Standard-Konfidenzwerte
                    category['confidence'] = {'category': 0.7, 'subcategories': 0.7}
                
                valid_categories.append(category)
                print(f"✓ Kategorie '{category_name}' validiert")
                
            except Exception as e:
                print(f"Fehler bei Validierung von '{category_name}': {str(e)}")
                continue
        
        return valid_categories

    async def _enhance_category_definition(self, category: dict) -> Optional[str]:
        """
        Erweitert eine zu kurze Kategoriendefinition.
        Nutzt den _get_definition_enhancement_prompt.
        
        Args:
            category: Kategorie mit unzureichender Definition
            
        Returns:
            Optional[str]: Erweiterte Definition oder None bei Fehler
        """
        try:
            # Hole den spezialisierten Prompt für Definitionsverbesserung
            prompt = self._get_definition_enhancement_prompt(category)
            
            input_tokens = estimate_tokens(prompt)

            response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                

            enhanced_def = response.choices[0].message.content.strip()
            
            output_tokens = estimate_tokens(response.choices[0].message.content)
            token_counter.add_tokens(input_tokens, output_tokens)

            # Prüfe ob die neue Definition besser ist
            if len(enhanced_def.split()) >= self.MIN_DEFINITION_WORDS:
                print(f"Definition erfolgreich erweitert von {len(category['definition'].split())} "
                    f"auf {len(enhanced_def.split())} Wörter")
                return enhanced_def
            else:
                print(f"Warnung: Erweiterte Definition immer noch zu kurz")
                return None

        except Exception as e:
            print(f"Fehler bei Definitionserweiterung: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return None

    async def _generate_subcategories(self, category: dict) -> bool:
        """
        Generiert Subkategorien für eine Kategorie ohne solche.
        Nutzt den _get_subcategory_generation_prompt.
        
        Args:
            category: Kategorie ohne Subkategorien
            
        Returns:
            bool: True wenn Subkategorien erfolgreich generiert wurden
        """
        try:
            # Hole den spezialisierten Prompt für Subkategoriengenerierung
            prompt = self._get_subcategory_generation_prompt(category)
            
            input_tokens = estimate_tokens(prompt)

            response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    response_format={"type": "json_object"}
                )
                
            # Verarbeite Response mit Wrapper
            llm_response = LLMResponse(response)
            result = json.loads(llm_response.content)
            
            output_tokens = estimate_tokens(response.choices[0].message.content)
            token_counter.add_tokens(input_tokens, output_tokens)

            # Stelle sicher dass wir eine Liste haben
            if isinstance(result, str):
                subcats = [result]
            elif isinstance(result, list):
                subcats = result
            else:
                print(f"Warnung: Unerwartetes Format für Subkategorien: {type(result)}")
                return False

            # Validiere die generierten Subkategorien
            if len(subcats) >= 2:
                # Füge die neuen Subkategorien zur Kategorie hinzu
                category['new_subcategories'] = subcats
                print(f"Subkategorien erfolgreich generiert:")
                for subcat in subcats:
                    print(f"- {subcat}")
                return True
            else:
                print(f"Warnung: Zu wenige Subkategorien generiert ({len(subcats)})")
                return False

        except json.JSONDecodeError:
            print("Fehler: Ungültiges JSON in der Antwort")
            return False
        except Exception as e:
            print(f"Fehler bei Subkategoriengenerierung: {str(e)}")
            print("Details:")
            traceback.print_exc()
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


    def _export_revision_history(self, writer, changes: List['CategoryChange']) -> None:
        """
        Exportiert die Revisionshistorie in ein separates Excel-Sheet.
        
        Args:
            writer: Excel Writer Objekt
            changes: Liste der Kategorieänderungen
        """
        try:
            # Erstelle DataFrame für Revisionshistorie
            revision_data = []
            for change in changes:
                # Erstelle lesbares Änderungsdatum
                change_date = datetime.fromisoformat(change.timestamp)
                
                # Bereite die betroffenen Kodierungen auf
                affected_codings = (
                    ', '.join(change.affected_codings)
                    if change.affected_codings
                    else 'Keine'
                )
                
                # Sammle Änderungsdetails
                if change.old_value and change.new_value:
                    details = []
                    for key in set(change.old_value.keys()) | set(change.new_value.keys()):
                        old = change.old_value.get(key, 'Nicht vorhanden')
                        new = change.new_value.get(key, 'Nicht vorhanden')
                        if old != new:
                            details.append(f"{key}: {old} → {new}")
                    details_str = '\n'.join(details)
                else:
                    details_str = ''

                revision_data.append({
                    'Datum': change_date.strftime('%Y-%m-%d %H:%M'),
                    'Kategorie': change.category_name,
                    'Art der Änderung': change.change_type,
                    'Beschreibung': change.description,
                    'Details': details_str,
                    'Begründung': change.justification,
                    'Betroffene Kodierungen': affected_codings
                })
            
            if revision_data:
                df_revisions = pd.DataFrame(revision_data)
                sheet_name = 'Revisionshistorie'
                
                # Erstelle Sheet falls nicht vorhanden
                if sheet_name not in writer.sheets:
                    writer.book.create_sheet(sheet_name)
                    
                # Exportiere Daten
                df_revisions.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Formatiere Worksheet
                worksheet = writer.sheets[sheet_name]
                
                # Setze Spaltenbreiten
                column_widths = {
                    'A': 20,  # Datum
                    'B': 25,  # Kategorie
                    'C': 15,  # Art der Änderung
                    'D': 50,  # Beschreibung
                    'E': 50,  # Details
                    'F': 50,  # Begründung
                    'G': 40   # Betroffene Kodierungen
                }
                
                for col, width in column_widths.items():
                    worksheet.column_dimensions[col].width = width
                    
                # Formatiere Überschriften
                for cell in worksheet[1]:
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')
                
                # Aktiviere Zeilenumbruch für lange Texte
                for row in worksheet.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical='top')
                
                print(f"Revisionshistorie mit {len(revision_data)} Einträgen exportiert")
                
            else:
                print("Keine Revisionshistorie zum Exportieren vorhanden")
                
        except Exception as e:
            print(f"Warnung: Fehler beim Export der Revisionshistorie: {str(e)}")
            import traceback
            traceback.print_exc()


# --- Klasse: DocumentReader ---
# Aufgabe: Laden und Vorbereiten des Analysematerials (Textdokumente, output)

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

    def add_tokens(self, input_tokens: int, output_tokens: int = 0):
        """
        Zählt Input- und Output-Tokens.
        
        Args:
            input_tokens: Anzahl der Input-Tokens
            output_tokens: Anzahl der Output-Tokens (optional, Standard 0)
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def get_report(self):
        return f"Gesamte Token-Nutzung:\n" \
               f"Input Tokens: {self.input_tokens}\n" \
               f"Output Tokens: {self.output_tokens}\n" \
               f"Gesamt Tokens: {self.input_tokens + self.output_tokens}"

token_counter = TokenCounter()



# --- Hilfsfunktionen ---

# Hilfsfunktion zur Token-Schätzung
def estimate_tokens(text: str) -> int:
    """Schätzt die Anzahl der Tokens in einem Text."""
    return len(text.split())

def get_input_with_timeout(prompt: str, timeout: int = 30) -> str:
    """
    Fragt nach Benutzereingabe mit Timeout.
    
    Args:
        prompt: Anzuzeigender Text
        timeout: Timeout in Sekunden
        
    Returns:
        str: Benutzereingabe oder 'n' bei Timeout
    """
    import threading
    import sys
    import time
    from threading import Event
    
    # Plattformspezifische Imports
    if sys.platform == 'win32':
        import msvcrt
    else:
        import select

    answer = {'value': None}
    stop_event = Event()
    
    def input_thread():
        try:
            # Zeige Countdown
            remaining_time = timeout
            while remaining_time > 0 and not stop_event.is_set():
                sys.stdout.write(f'\r{prompt} ({remaining_time}s): ')
                sys.stdout.flush()
                
                # Plattformspezifische Eingabeprüfung
                if sys.platform == 'win32':
                    if msvcrt.kbhit():
                        answer['value'] = msvcrt.getche().decode().strip().lower()
                        sys.stdout.write('\n')
                        stop_event.set()
                        return
                else:
                    if select.select([sys.stdin], [], [], 1)[0]:
                        answer['value'] = sys.stdin.readline().strip().lower()
                        stop_event.set()
                        return
                
                time.sleep(1)
                remaining_time -= 1
            
            # Bei Timeout
            if not stop_event.is_set():
                sys.stdout.write('\n')
                sys.stdout.flush()
                
        except (KeyboardInterrupt, EOFError):
            stop_event.set()
    
    # Starte Input-Thread
    thread = threading.Thread(target=input_thread)
    thread.daemon = True
    thread.start()
    
    # Warte auf Antwort oder Timeout
    thread.join(timeout)
    stop_event.set()
    
    if answer['value'] is None:
        print(f"\nKeine Eingabe innerhalb von {timeout} Sekunden - verwende 'n'")
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

        # 1. Konfiguration laden
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_loader = ConfigLoader(script_dir)
        
        if config_loader.load_codebook():
            config = config_loader.get_config() 
            config_loader.update_script_globals(globals())
            print("\nKonfiguration erfolgreich geladen")
        else:
            print("Verwende Standard-Konfiguration")
            config = CONFIG

        # 2. Kategoriensystem initialisieren
        print("\n1. Initialisiere Kategoriensystem...")
        category_builder = DeductiveCategoryBuilder()
        initial_categories = category_builder.load_theoretical_categories()
        
        # 3. Manager und History initialisieren
        development_history = DevelopmentHistory(CONFIG['OUTPUT_DIR'])
        revision_manager = CategoryRevisionManager(
            output_dir=CONFIG['OUTPUT_DIR'],
            config=config
        )
        
        # Initiale Kategorien dokumentieren
        for category_name in initial_categories.keys():
            revision_manager.changes.append(CategoryChange(
                category_name=category_name,
                change_type='add',
                description="Initiale deduktive Kategorie",
                timestamp=datetime.now().isoformat(),
                justification="Teil des ursprünglichen deduktiven Kategoriensystems"
            ))

        # 4. Dokumente einlesen
        print("\n2. Lese Dokumente ein...")
        reader = DocumentReader(CONFIG['DATA_DIR'])
        documents = await reader.read_documents()

        if not documents:
            print("\nKeine Dokumente zum Analysieren gefunden.")
            return

        # 4b. Abfrage zur induktiven Kodierung
        print("\n3. Induktive Kodierung konfigurieren...")

        # Prüfe ob ein induktives Codebook existiert
        codebook_path = os.path.join(CONFIG['OUTPUT_DIR'], "codebook_inductive.json")
        skip_inductive = False

        if os.path.exists(codebook_path):
            print("\nGespeichertes induktives Codebook gefunden.")
            print("Automatische Fortführung in 10 Sekunden...")
            
            use_saved = get_input_with_timeout(
                "\nMöchten Sie das gespeicherte erweiterte Kodesystem laden? (j/N)",
                timeout=10
            )
            
            if use_saved.lower() == 'j':
                try:
                    with open(codebook_path, 'r', encoding='utf-8') as f:
                        saved_categories = json.load(f)
                        
                    if 'categories' in saved_categories:
                        # Konvertiere JSON zurück in CategoryDefinition Objekte
                        for name, cat_data in saved_categories['categories'].items():
                            initial_categories[name] = CategoryDefinition(
                                name=name,
                                definition=cat_data['definition'],
                                examples=cat_data.get('examples', []),
                                rules=cat_data.get('rules', []),
                                subcategories=cat_data.get('subcategories', {}),
                                added_date=cat_data.get('added_date', datetime.now().strftime("%Y-%m-%d")),
                                modified_date=cat_data.get('modified_date', datetime.now().strftime("%Y-%m-%d"))
                            )
                        print(f"\n✓ {len(saved_categories['categories'])} Kategorien aus Codebook geladen")
                        skip_inductive = True
                    else:
                        print("\nWarnung: Ungültiges Codebook-Format")
                        
                except Exception as e:
                    print(f"\nFehler beim Laden des Codebooks: {str(e)}")
                    print("Fahre mit Standard-Kategorien fort")

        if not skip_inductive:
            print("\nAutomatische Fortführung in 10 Sekunden...")
            user_input = get_input_with_timeout(
                "\nMöchten Sie die induktive Kodierung überspringen und nur deduktiv arbeiten? (j/n)",
                timeout=10
            )
            
            if user_input.lower() == 'j':
                skip_inductive = True
                print("\nℹ Induktive Kodierung wird übersprungen - Nur deduktive Kategorien werden verwendet")
            else:
                print("\nℹ Vollständige Analyse mit deduktiven und induktiven Kategorien")

        # 5. Kodierer konfigurieren
        print("\n4. Konfiguriere Kodierer...")
        # Automatische Kodierer
        auto_coders = [
            DeductiveCoder(
                model_name=CONFIG['MODEL_NAME'],
                temperature=coder_config['temperature'],
                coder_id=coder_config['coder_id']
            )
            for coder_config in CONFIG['CODER_SETTINGS']
        ]

        # Manuelle Kodierung konfigurieren
        print("\nKonfiguriere manuelle Kodierung...")
        print("Sie haben 10 Sekunden Zeit für die Eingabe.")
        print("Drücken Sie 'j' für manuelle Kodierung oder 'n' zum Überspringen.")

        manual_coders = []
        user_input = get_input_with_timeout(
            "\nMöchten Sie manuell kodieren? (j/N)",
            timeout=10
        )
        
        if user_input.lower() == 'j':
            manual_coders.append(ManualCoder(coder_id="human_1"))
            print("\n✓ Manueller Kodierer wurde hinzugefügt")
        else:
            print("\nℹ Keine manuelle Kodierung - nur automatische Kodierung wird durchgeführt")

        # 6. Material vorbereiten
        print("\n5. Bereite Material vor...")
        loader = MaterialLoader(
            data_dir=CONFIG['DATA_DIR'],
            chunk_size=CONFIG['CHUNK_SIZE'],
            chunk_overlap=CONFIG['CHUNK_OVERLAP']
        )
        chunks = {}
        for doc_name, doc_text in documents.items():
            chunks[doc_name] = loader.chunk_text(doc_text)
            print(f"- {doc_name}: {len(chunks[doc_name])} Chunks erstellt")

        # 7. Manuelle Kodierung durchführen
        manual_codings = []
        if manual_coders:
            print("\n6. Starte manuelle Kodierung...")
            manual_coding_result = await perform_manual_coding(
                chunks=chunks, 
                categories=initial_categories,
                manual_coders=manual_coders
            )
            if manual_coding_result == "ABORT_ALL":
                print("Manuelle Kodierung abgebrochen. Beende Programm.")
                return
            manual_codings = manual_coding_result
            print(f"Manuelle Kodierung abgeschlossen: {len(manual_codings)} Kodierungen")

        # 8. Integrierte Analyse starten
        print("\n7. Starte integrierte Analyse...")
        analysis_manager = IntegratedAnalysisManager(CONFIG)

        # Initialisiere Fortschrittsüberwachung
        progress_task = asyncio.create_task(
            monitor_progress(analysis_manager)
        )

        try:
            # Starte die Hauptanalyse
            final_categories, coding_results = await analysis_manager.analyze_material(
                chunks=chunks,
                initial_categories=initial_categories,
                skip_inductive=skip_inductive
            )

            # Beende Fortschrittsüberwachung
            progress_task.cancel()
            await progress_task

            # Kombiniere alle Kodierungen
            all_codings = []
            
            # Füge automatische Kodierungen hinzu
            if coding_results and len(coding_results) > 0:
                print(f"\nFüge {len(coding_results)} automatische Kodierungen hinzu")
                for coding in coding_results:
                    if isinstance(coding, dict) and 'segment_id' in coding:
                        all_codings.append(coding)
                    else:
                        print(f"Überspringe ungültige Kodierung: {coding}")
            
            # Füge manuelle Kodierungen hinzu
            if manual_codings and len(manual_codings) > 0:
                print(f"Füge {len(manual_codings)} manuelle Kodierungen hinzu")
                all_codings.extend(manual_codings)
            
            print(f"\nGesamtzahl Kodierungen: {len(all_codings)}")

            # 9. Berechne Intercoder-Reliabilität
            if all_codings:
                print("\n8. Berechne Intercoder-Reliabilität...")
                reliability_calculator = InductiveCoder(
                    model_name=CONFIG['MODEL_NAME'],
                    history=development_history,
                    output_dir=CONFIG['OUTPUT_DIR']
                )
                reliability = reliability_calculator._calculate_reliability(all_codings)
                print(f"Reliabilität (Krippendorffs Alpha): {reliability:.3f}")
            else:
                print("\nKeine Kodierungen für Reliabilitätsberechnung")
                reliability = 0.0

            # 10. Speichere induktiv erweitertes Codebook
            if final_categories:
                category_manager = CategoryManager(CONFIG['OUTPUT_DIR'])
                category_manager.save_codebook(
                    categories=final_categories,
                    filename="codebook_inductive.json"
                )

            # 11. Export der Ergebnisse
            print("\n9. Exportiere Ergebnisse...")
            if all_codings:
                exporter = ResultsExporter(
                    output_dir=CONFIG['OUTPUT_DIR'],
                    attribute_labels=CONFIG['ATTRIBUTE_LABELS'],
                    analysis_manager=analysis_manager,
                    inductive_coder=reliability_calculator
                )
                
                await exporter.export_results(
                    codings=all_codings,
                    reliability=reliability,
                    categories=final_categories,
                    chunks=chunks,
                    revision_manager=revision_manager,
                    export_mode="consensus",
                    original_categories=initial_categories,
                    inductive_coder=reliability_calculator
                )
                print("Export erfolgreich abgeschlossen")
            else:
                print("Keine Kodierungen zum Exportieren vorhanden")

            # 12. Zeige finale Statistiken
            print("\nAnalyse abgeschlossen:")
            print(analysis_manager.get_analysis_report())
            
            # Token-Statistiken
            print("\nToken-Nutzung:")
            print(token_counter.get_report())
            
            # Relevanz-Statistiken
            relevance_stats = analysis_manager.relevance_checker.get_statistics()
            print("\nRelevanz-Statistiken:")
            print(f"- Segmente analysiert: {relevance_stats['total_segments']}")
            print(f"- Relevante Segmente: {relevance_stats['relevant_segments']}")
            print(f"- Relevanzrate: {relevance_stats['relevance_rate']*100:.1f}%")
            print(f"- API-Calls gespart: {relevance_stats['total_segments'] - relevance_stats['api_calls']}")
            print(f"- Cache-Nutzung: {relevance_stats['cache_size']} Einträge")

        except asyncio.CancelledError:
            print("\nAnalyse wurde abgebrochen.")
        finally:
            # Stelle sicher, dass die Fortschrittsüberwachung beendet wird
            if not progress_task.done():
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        print(f"Fehler in der Hauptausführung: {str(e)}")
        traceback.print_exc()

async def monitor_progress(analysis_manager: IntegratedAnalysisManager):
    """
    Überwacht und zeigt den Analysefortschritt an.
    """
    try:
        while True:
            progress = analysis_manager.get_progress_report()
            
            # Formatiere Fortschrittsanzeige
            print("\n--- Analysefortschritt ---")
            print(f"Verarbeitet: {progress['progress']['processed_segments']} Segmente")
            print(f"Geschwindigkeit: {progress['progress']['segments_per_hour']:.1f} Segmente/Stunde")
            print("------------------------")
            
            await asyncio.sleep(30)  # Update alle 30 Sekunden
            
    except asyncio.CancelledError:
        print("\nFortschrittsüberwachung beendet.")

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
