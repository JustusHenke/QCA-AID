"""
QCA-AID: Qualitative Content Analysis with AI Support - Deductive Coding
========================================================================

A Python implementation of Mayring's Qualitative Content Analysis methodology,
enhanced with AI capabilities through the OpenAI API.

Version:
--------
0.9.14 (2025-05-26)

New in 0.9.14
- Implementierung der Mehrfachkodierung von Textsegmenten für mehrere Hauptkategorien
- Neue CONFIG-Parameter: MULTIPLE_CODINGS (default: True) und MULTIPLE_CODING_THRESHOLD (default: 0.7)
- Erweiterte Relevanzprüfung erkennt Segmente mit Bezug zu mehreren Hauptkategorien (>=70% Relevanz)
- Fokussierte Kodierung: Segmente werden gezielt für jede relevante Hauptkategorie kodiert
- Export-Erweiterung: Mehrfach kodierte Segmente erscheinen pro Hauptkategorie separat in der Outputtabelle
- Neue Export-Felder: Mehrfachkodierung_Instanz, Kategorie_Fokus, Fokus_verwendet
- Eindeutige Chunk-IDs mit Instanz-Suffix bei Mehrfachkodierung (z.B. "DOC-5-1", "DOC-5-2")
- Effiziente Batch-Verarbeitung und Caching für Mehrfachkodierungs-Prüfungen
- Konfigurierbare Deaktivierung der Mehrfachkodierung für Einzelkodierung


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
# 1. Lokale Imports
# ============================


# lokale Imports
from QCA_Utils import (
    TokenCounter, estimate_tokens, get_input_with_timeout, _calculate_multiple_coding_stats, 
    _patch_tkinter_for_threaded_exit, ConfigLoader,
    LLMProvider, LLMProviderFactory, LLMResponse, MistralProvider, OpenAIProvider,
    _sanitize_text_for_excel, _generate_pastel_colors, _format_confidence
)
from QCA_Prompts import QCAPrompts  # Prompt Bibliothek

# Instanziierung des globalen Token-Counters
token_counter = TokenCounter()

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
    'CODE_WITH_CONTEXT': False,
    'MULTIPLE_CODINGS': True, 
    'MULTIPLE_CODING_THRESHOLD': 0.7,  # Schwellenwert für zusätzliche Relevanz
    'ANALYSIS_MODE': 'deductive',
    'REVIEW_MODE': 'consensus',
    'ATTRIBUTE_LABELS': {
        'attribut1': 'Attribut1',
        'attribut2': 'Attribut2',
        'attribut3': 'Attribut3'  
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
                # print(f"- Neuer Chunk erstellt: {len(chunk_text)} Zeichen")
                
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
            # print(f"- Letzter Chunk: {len(chunk_text)} Zeichen")
        
        print(f"\nChunking Ergebnis:")
        print(f"- Anzahl Chunks: {len(chunks)}")
        print(f"- Durchschnittliche Chunk-Länge: {sum(len(c) for c in chunks)/len(chunks):.0f} Zeichen")
        
        return chunks

    def clean_problematic_characters(self, text: str) -> str:
        """
        Bereinigt Text von problematischen Zeichen, die später beim Excel-Export
        zu Fehlern führen könnten.
        
        Args:
            text (str): Zu bereinigender Text
            
        Returns:
            str: Bereinigter Text
        """
        if not text:
            return ""
            
        # Entferne problematische Steuerzeichen
        import re
        cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFE\uFFFF]', '', text)
        
        # Ersetze bekannte problematische Sonderzeichen
        problematic_chars = ['☺', '☻', '♥', '♦', '♣', '♠']
        for char in problematic_chars:
            cleaned_text = cleaned_text.replace(char, '')
        
        return cleaned_text

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
        
        # Entferne spezielle Steuerzeichen und problematische Zeichen
        text = self.clean_problematic_characters(text)
        
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
    
    def save_codebook(self, categories: Dict[str, CategoryDefinition], filename: str = "codebook_inductive.json") -> None:
        """Speichert das vollständige Codebook inkl. deduktiver, induktiver und grounded Kategorien"""
        try:
            codebook_data = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "2.0",
                    "total_categories": len(categories),
                    "research_question": FORSCHUNGSFRAGE,
                    "analysis_mode": CONFIG.get('ANALYSIS_MODE', 'deductive')  # Speichere den Analysemodus
                },
                "categories": {}
            }
            
            for name, category in categories.items():
                # Bestimme den Kategorietyp je nach Analysemodus
                if name in DEDUKTIVE_KATEGORIEN:
                    development_type = "deductive"
                elif CONFIG.get('ANALYSIS_MODE') == 'grounded':
                    development_type = "grounded"  # Neue Markierung für grounded Kategorien
                else:
                    development_type = "inductive"
                    
                codebook_data["categories"][name] = {
                    "definition": category.definition,
                    # Wandle examples in eine Liste um, falls es ein Set ist
                    "examples": list(category.examples) if isinstance(category.examples, set) else category.examples,
                    # Wandle rules in eine Liste um, falls es ein Set ist
                    "rules": list(category.rules) if isinstance(category.rules, set) else category.rules,
                    # Wandle subcategories in ein Dictionary um, falls nötig
                    "subcategories": dict(category.subcategories) if isinstance(category.subcategories, set) else category.subcategories,
                    "development_type": development_type,
                    "added_date": category.added_date,
                    "last_modified": category.modified_date
                }
            
            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(codebook_data, f, indent=2, ensure_ascii=False)
                
            print(f"\nCodebook gespeichert unter: {output_path}")
            print(f"- Deduktive Kategorien: {sum(1 for c in codebook_data['categories'].values() if c['development_type'] == 'deductive')}")
            print(f"- Induktive Kategorien: {sum(1 for c in codebook_data['categories'].values() if c['development_type'] == 'inductive')}")
            print(f"- Grounded Kategorien: {sum(1 for c in codebook_data['categories'].values() if c['development_type'] == 'grounded')}")
            
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
        provider_name = CONFIG.get('MODEL_PROVIDER', 'openai').lower()
        try:
            self.llm_provider = LLMProviderFactory.create_provider(provider_name)
            print(f"🤖 LLM Provider '{provider_name}' für Kategorienoptimierung initialisiert")
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
        provider_name = CONFIG.get('MODEL_PROVIDER', 'openai').lower()  # Fallback zu OpenAI
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

        # Hole Ausschlussregeln aus KODIERREGELN
        self.exclusion_rules = KODIERREGELN.get('exclusion', [])
        print("\nRelevanceChecker initialisiert:")
        print(f"- {len(self.exclusion_rules)} Ausschlussregeln geladen")

        # Hole Mehrfachkodierungsparameter aus CONFIG
        self.multiple_codings_enabled = CONFIG.get('MULTIPLE_CODINGS', True)
        self.multiple_threshold = CONFIG.get('MULTIPLE_CODING_THRESHOLD', 0.7)

        # Cache für Mehrfachkodierungen
        self.multiple_coding_cache = {}
        
        print("\nRelevanceChecker initialisiert:")
        print(f"- {len(self.exclusion_rules)} Ausschlussregeln geladen")
        print(f"- Mehrfachkodierung: {'Aktiviert' if self.multiple_codings_enabled else 'Deaktiviert'}")
        if self.multiple_codings_enabled:
            print(f"- Mehrfachkodierung-Schwellenwert: {self.multiple_threshold}")
        
        # Prompt-Handler
        self.prompt_handler = QCAPrompts(
            forschungsfrage=FORSCHUNGSFRAGE,
            kodierregeln=KODIERREGELN,
            deduktive_kategorien=DEDUKTIVE_KATEGORIEN
        )

    
    async def check_multiple_category_relevance(self, segments: List[Tuple[str, str]], 
                                            categories: Dict[str, CategoryDefinition]) -> Dict[str, List[Dict]]:
        """
        Prüft ob Segmente für mehrere Hauptkategorien relevant sind.
        
        Args:
            segments: Liste von (segment_id, text) Tupeln
            categories: Verfügbare Hauptkategorien
            
        Returns:
            Dict mit segment_id als Key und Liste von relevanten Kategorien als Value
        """
        if not self.multiple_codings_enabled:
            return {}
            
        # Filtere bereits gecachte Segmente
        uncached_segments = [
            (sid, text) for sid, text in segments 
            if sid not in self.multiple_coding_cache
        ]
        
        if not uncached_segments:
            return {sid: self.multiple_coding_cache[sid] for sid, _ in segments}

        try:
            # Erstelle Kategorien-Kontext für den Prompt
            category_descriptions = []
            for cat_name, cat_def in categories.items():
                # Nur Hauptkategorien, nicht "Nicht kodiert" etc.
                if cat_name in ["Nicht kodiert", "Kein Kodierkonsens"]:
                    continue
                    
                category_descriptions.append({
                    'name': cat_name,
                    'definition': cat_def.definition[:200] + '...' if len(cat_def.definition) > 200 else cat_def.definition,
                    'examples': cat_def.examples[:2] if cat_def.examples else []
                })

            # Batch-Text für API
            segments_text = "\n\n=== SEGMENT BREAK ===\n\n".join(
                f"SEGMENT {i + 1}:\n{text}" 
                for i, (_, text) in enumerate(uncached_segments)
            )

            prompt = self.prompt_handler.get_multiple_category_relevance_prompt(
                segments_text=segments_text,
                category_descriptions=category_descriptions,
                multiple_threshold=self.multiple_threshold
            )
            

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
            
            llm_response = LLMResponse(response)
            results = json.loads(llm_response.content)
            
            output_tokens = estimate_tokens(response.choices[0].message.content)
            token_counter.add_tokens(input_tokens, output_tokens)

            # Verarbeite Ergebnisse und aktualisiere Cache
            multiple_coding_results = {}
            for i, (segment_id, _) in enumerate(uncached_segments):
                segment_result = results['segment_results'][i]
                
                # Filtere Kategorien nach Schwellenwert
                relevant_categories = [
                    cat for cat in segment_result['relevant_categories']
                    if cat['relevance_score'] >= self.multiple_threshold
                ]
                
                # Cache-Aktualisierung
                self.multiple_coding_cache[segment_id] = relevant_categories
                multiple_coding_results[segment_id] = relevant_categories
                
                # Debug-Ausgabe
                if len(relevant_categories) > 1:
                    print(f"\n🔄 Mehrfachkodierung identifiziert für Segment {segment_id}:")
                    for cat in relevant_categories:
                        print(f"  - {cat['category']}: {cat['relevance_score']:.2f} ({cat['justification'][:60]}...)")
                    print(f"  Begründung: {segment_result.get('multiple_coding_justification', '')}")

            # Kombiniere Cache und neue Ergebnisse
            return {
                sid: self.multiple_coding_cache[sid] 
                for sid, _ in segments
            }

        except Exception as e:
            print(f"Fehler bei Mehrfachkodierungs-Prüfung: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
        
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

            # Formatiere Ausschlussregeln für den Prompt
            exclusion_rules_text = "\n".join(f"- {rule}" for rule in self.exclusion_rules)

            prompt = self.prompt_handler.get_relevance_check_prompt(
                segments_text=segments_text,
                exclusion_rules=self.exclusion_rules
            )
            

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
            try:
                results = json.loads(llm_response.content)
            except json.JSONDecodeError as e:
                print(f"Fehler beim Parsen von JSON: {str(e)}")
                results = {}  # Setze ein leeres Dictionary als Fallback
            
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
                    'key_aspects': segment_result['key_aspects'],
                    'justification': segment_result['justification']
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

        # Konfigurationsparameter
        self.use_context = config.get('CODE_WITH_CONTEXT', True)
        print(f"\nKontextuelle Kodierung: {'Aktiviert' if self.use_context else 'Deaktiviert'}")

        # Dictionary für die Verwaltung der Document-Summaries
        self.document_summaries = {}

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
        try:
            # Prüfe Relevanz für ganzen Batch auf einmal
            relevance_results = await self.relevance_checker.check_relevance_batch(batch)

            # Filtere relevante Segmente
            relevant_segments = [
                text for (segment_id, text) in batch 
                if relevance_results.get(segment_id, False)
            ]

            if not relevant_segments:
                print("   ℹ️ Keine relevanten Segmente für Kategorienentwicklung")
                return {}

            print(f"\nEntwickle Kategorien aus {len(relevant_segments)} relevanten Segmenten")
            
            # Hole den aktuellen Analyse-Modus
            analysis_mode = CONFIG.get('ANALYSIS_MODE', 'full')
            # analysis_mode = self.analysis_mode
            print(f"Aktiver Analyse-Modus: {analysis_mode}")
            
            # Induktive Kategorienentwicklung
            new_categories = await self.inductive_coder.develop_category_system(
                relevant_segments,
                current_categories)
            
            # Extrahiere den Analysemodus aus der Config
            analysis_mode = CONFIG.get('ANALYSIS_MODE', 'full')
            
            # Erstelle Dictionary für das erweiterte Kategoriensystem
            extended_categories = current_categories.copy()
            
            if new_categories:
                # Verarbeite je nach Analysemodus
                if analysis_mode == 'full':
                    # Im vollen Modus: Neue Kategorien hinzufügen und bestehende aktualisieren
                    for cat_name, category in new_categories.items():
                        if cat_name in current_categories:
                            # Bestehende Kategorie aktualisieren
                            current_cat = current_categories[cat_name]
                            
                            # Neue Subkategorien hinzufügen
                            merged_subcats = {**current_cat.subcategories, **category.subcategories}
                            
                            # Definition aktualisieren wenn die neue aussagekräftiger ist
                            new_definition = category.definition
                            if len(new_definition) > len(current_cat.definition):
                                print(f"✓ Definition für '{cat_name}' aktualisiert")
                            else:
                                new_definition = current_cat.definition
                            
                            # Kombiniere Beispiele
                            merged_examples = list(set(current_cat.examples + category.examples))
                            
                            # Erstelle aktualisierte Kategorie
                            extended_categories[cat_name] = CategoryDefinition(
                                name=cat_name,
                                definition=new_definition,
                                examples=merged_examples,
                                rules=current_cat.rules,
                                subcategories=merged_subcats,
                                added_date=current_cat.added_date,
                                modified_date=datetime.now().strftime("%Y-%m-%d")
                            )
                            
                            print(f"✓ Kategorie '{cat_name}' aktualisiert")
                            if len(merged_subcats) > len(current_cat.subcategories):
                                print(f"  - {len(merged_subcats) - len(current_cat.subcategories)} neue Subkategorien")
                        else:
                            # Neue Kategorie hinzufügen
                            extended_categories[cat_name] = category
                            print(f"🆕 Neue Hauptkategorie hinzugefügt: {cat_name}")
                            print(f"  - Definition: {category.definition[:100]}...")
                            print(f"  - Subkategorien: {len(category.subcategories)}")
                
                elif analysis_mode == 'abductive':
                    # Im abduktiven Modus: KEINE neuen Hauptkategorien, NUR Subkategorien zu bestehenden hinzufügen
                    print("\nVerarbeite im abduktiven Modus - nur Subkategorien werden erweitert")
                    
                    for cat_name, category in new_categories.items():
                        if cat_name in current_categories:
                            # Bestehende Kategorie aktualisieren - NUR Subkategorien
                            current_cat = current_categories[cat_name]
                            
                            # Neue Subkategorien zählen
                            new_subcats = {}
                            for sub_name, sub_def in category.subcategories.items():
                                if sub_name not in current_cat.subcategories:
                                    new_subcats[sub_name] = sub_def
                            
                            if new_subcats:
                                # Erstelle aktualisierte Kategorie mit neuen Subkategorien
                                extended_categories[cat_name] = CategoryDefinition(
                                    name=cat_name,
                                    definition=current_cat.definition,  # Behalte ursprüngliche Definition
                                    examples=current_cat.examples,      # Behalte ursprüngliche Beispiele
                                    rules=current_cat.rules,            # Behalte ursprüngliche Regeln
                                    subcategories={**current_cat.subcategories, **new_subcats},
                                    added_date=current_cat.added_date,
                                    modified_date=datetime.now().strftime("%Y-%m-%d")
                                )
                                
                                print(f"✓ Kategorie '{cat_name}' abduktiv erweitert")
                                print(f"  - {len(new_subcats)} neue Subkategorien:")
                                for sub_name in new_subcats.keys():
                                    print(f"    • {sub_name}")
                        else:
                            print(f"⚠️ Achtung: Kategorie '{cat_name}' nicht im Hauptkategoriensystem gefunden.")
                            print(f"    Die abduktiv erzeugten Subkategorien werden ignoriert.")
                            print(f"    Verfügbare Hauptkategorien: {', '.join(current_categories.keys())}")
                
                else:  # deduktiver Modus
                    # Im deduktiven Modus: Keine Änderungen am Kategoriensystem
                    print("\nDeduktiver Modus - keine Kategorienentwicklung")
                    return {}
                
                return extended_categories
            else:
                print("   ℹ️ Keine neuen Kategorien in diesem Batch identifiziert")
                return {}
            
        except Exception as e:
            print(f"Fehler bei Kategorienentwicklung: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return {}

    async def _code_batch_deductively(self,
                                    batch: List[Tuple[str, str]],
                                    categories: Dict[str, CategoryDefinition]) -> List[Dict]:
        """Führt die deduktive Kodierung mit optimierter Relevanzprüfung und Mehrfachkodierung durch."""
        batch_results = []
        batch_metrics = {
            'new_aspects': [],
            'category_coverage': {},
            'coding_confidence': []
        }
        
        # 1. Standard-Relevanzprüfung für ganzen Batch
        relevance_results = await self.relevance_checker.check_relevance_batch(batch)
        
        # 2. Mehrfachkodierungs-Prüfung (wenn aktiviert)
        multiple_coding_results = {}
        if CONFIG.get('MULTIPLE_CODINGS', True):
            relevant_segments = [
                (segment_id, text) for segment_id, text in batch
                if relevance_results.get(segment_id, False)
            ]
            
            if relevant_segments:
                print(f"  🔄 Prüfe {len(relevant_segments)} relevante Segmente auf Mehrfachkodierung...")
                multiple_coding_results = await self.relevance_checker.check_multiple_category_relevance(
                    relevant_segments, categories
                )
        
        for segment_id, text in batch:
            print(f"\n--------------------------------------------------------")
            print(f"🔎 Verarbeite Segment {segment_id}")
            print(f"--------------------------------------------------------\n")
            
            # Nutze gespeicherte Relevanzprüfung
            if not relevance_results.get(segment_id, False):
                print(f"Segment wurde als nicht relevant markiert - wird übersprungen")
                
                # Erstelle "Nicht kodiert" Ergebnis für alle Kodierer
                for coder in self.deductive_coders:
                    result = {
                        'segment_id': segment_id,
                        'coder_id': coder.coder_id,
                        'category': "Nicht kodiert",
                        'subcategories': [],
                        'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                        'justification': "Nicht relevant für Forschungsfrage",
                        'text': text,
                        'multiple_coding_instance': 1,
                        'total_coding_instances': 1,
                        'target_category': '',
                        'category_focus_used': False
                    }
                    batch_results.append(result)
                continue

            # Bestimme Anzahl der Kodierungen für dieses Segment
            coding_instances = []
            multiple_categories = multiple_coding_results.get(segment_id, [])
            
            if len(multiple_categories) > 1:
                # Mehrfachkodierung
                print(f"  🔄 Mehrfachkodierung aktiviert: {len(multiple_categories)} Kategorien")
                for i, category_info in enumerate(multiple_categories, 1):
                    coding_instances.append({
                        'instance': i,
                        'total_instances': len(multiple_categories),
                        'target_category': category_info['category'],
                        'category_context': category_info
                    })
                    print(f"    {i}. {category_info['category']} (Relevanz: {category_info['relevance_score']:.2f})")
            else:
                # Standardkodierung
                coding_instances.append({
                    'instance': 1,
                    'total_instances': 1,
                    'target_category': '',
                    'category_context': None
                })

            # Verarbeite relevante Segmente mit allen Kodierern
            print(f"Kodiere relevantes Segment mit {len(self.deductive_coders)} Kodierern")
            
            # Kodiere für jede Instanz
            for instance_info in coding_instances:
                if instance_info['total_instances'] > 1:
                    print(f"\n  📝 Kodierungsinstanz {instance_info['instance']}/{instance_info['total_instances']}")
                    print(f"      Fokus-Kategorie: {instance_info['target_category']}")
                
                for coder in self.deductive_coders:
                    try:
                        # Kodierung mit optionalem Kategorie-Fokus
                        if instance_info['target_category']:
                            # Mehrfachkodierung mit Fokus
                            coding = await coder.code_chunk_with_focus(
                                text, categories, 
                                focus_category=instance_info['target_category'],
                                focus_context=instance_info['category_context']
                            )
                        else:
                            # Standard-Kodierung
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
                                'keywords': coding.keywords,
                                'multiple_coding_instance': instance_info['instance'],
                                'total_coding_instances': instance_info['total_instances'],
                                'target_category': instance_info['target_category'],
                                'category_focus_used': bool(instance_info['target_category'])
                            }
                            
                            batch_results.append(result)
                            
                            # Log für Mehrfachkodierung
                            if instance_info['total_instances'] > 1:
                                print(f"        ✓ {coder.coder_id}: {coding.category}")
                            
                        else:
                            print(f"  ✗ Keine gültige Kodierung von {coder.coder_id}")
                            
                    except Exception as e:
                        print(f"  ✗ Fehler bei Kodierer {coder.coder_id}: {str(e)}")
                        continue

            self.processed_segments.add(segment_id)

        # Aktualisiere Sättigungsmetriken (unverändert)
        saturation_metrics = {
            'new_aspects_found': len(batch_metrics['new_aspects']) > 0,
            'categories_sufficient': len(batch_metrics['category_coverage']) >= len(categories) * 0.8,
            'theoretical_coverage': len(batch_metrics['category_coverage']) / len(categories),
            'avg_confidence': sum(batch_metrics['coding_confidence']) / len(batch_metrics['coding_confidence']) if batch_metrics['coding_confidence'] else 0
        }
        
        self.saturation_checker.add_saturation_metrics(saturation_metrics)

        return batch_results

    async def _code_with_category_focus(self, coder, text, categories, instance_info):
        """Kodiert mit optionalem Fokus auf bestimmte Kategorie"""
        
        if instance_info['target_category']:
            # Mehrfachkodierung mit Kategorie-Fokus
            return await coder.code_chunk_with_focus(
                text, categories, 
                focus_category=instance_info['target_category'],
                focus_context=instance_info['category_context']
            )
        else:
            # Standard-Kodierung
            return await coder.code_chunk(text, categories)

    async def _code_batch_with_context(self, batch: List[Tuple[str, str]], categories: Dict[str, CategoryDefinition]) -> List[Dict]:
        """
        Kodiert einen Batch sequentiell mit progressivem Dokumentkontext und Mehrfachkodierung.
        """
        batch_results = []
        
        # Prüfe Mehrfachkodierungs-Möglichkeiten für den ganzen Batch
        multiple_coding_results = {}
        if CONFIG.get('MULTIPLE_CODINGS', True):
            # Relevanzprüfung für ganzen Batch
            relevance_results = await self.relevance_checker.check_relevance_batch(batch)
            relevant_segments = [
                (segment_id, text) for segment_id, text in batch
                if relevance_results.get(segment_id, False)
            ]
            
            if relevant_segments:
                print(f"  🔄 Prüfe {len(relevant_segments)} relevante Segmente auf Mehrfachkodierung...")
                multiple_coding_results = await self.relevance_checker.check_multiple_category_relevance(
                    relevant_segments, categories
                )
        
        # Sequentielle Verarbeitung um Kontext aufzubauen
        for segment_id, text in batch:
            # Extrahiere Dokumentnamen und Chunk-ID
            doc_name, chunk_id = self._extract_doc_and_chunk_id(segment_id)
            position = f"Chunk {chunk_id}"
            
            # Hole aktuelles Summary oder initialisiere es
            current_summary = self.document_summaries.get(doc_name, "")
            
            # Segment-Informationen
            segment_info = {
                'doc_name': doc_name,
                'chunk_id': chunk_id,
                'position': position
            }
            
            # Kontext-Daten vorbereiten
            context_data = {
                'current_summary': current_summary,
                'segment_info': segment_info
            }
            
            print(f"\n🔍 Verarbeite Segment {segment_id} mit Kontext")
            
            # Prüfe Relevanz
            relevance_result = await self.relevance_checker.check_relevance_batch([(segment_id, text)])
            is_relevant = relevance_result.get(segment_id, False)
            
            if not is_relevant:
                print(f"  ↪ Segment als nicht relevant markiert - wird übersprungen")
                
                # Erstelle "Nicht kodiert" Ergebnis für alle Kodierer
                for coder in self.deductive_coders:
                    result = {
                        'segment_id': segment_id,
                        'coder_id': coder.coder_id,
                        'category': "Nicht kodiert",
                        'subcategories': [],
                        'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                        'justification': "Nicht relevant für Forschungsfrage",
                        'text': text,
                        'context_summary': current_summary,
                        'multiple_coding_instance': 1,
                        'total_coding_instances': 1,
                        'target_category': '',
                        'category_focus_used': False
                    }
                    batch_results.append(result)
                continue
            
            # Bestimme Kodierungsinstanzen
            coding_instances = []
            multiple_categories = multiple_coding_results.get(segment_id, [])
            
            if len(multiple_categories) > 1:
                print(f"  🔄 Mehrfachkodierung mit Kontext: {len(multiple_categories)} Kategorien")
                for i, category_info in enumerate(multiple_categories, 1):
                    coding_instances.append({
                        'instance': i,
                        'total_instances': len(multiple_categories),
                        'target_category': category_info['category'],
                        'category_context': category_info
                    })
            else:
                coding_instances.append({
                    'instance': 1,
                    'total_instances': 1,
                    'target_category': '',
                    'category_context': None
                })
            
            # Verarbeite relevante Segmente mit Kontext für ALLE Kodierer und Instanzen
            updated_summary = current_summary
            
            for instance_info in coding_instances:
                if instance_info['total_instances'] > 1:
                    print(f"\n    📝 Kontext-Kodierung {instance_info['instance']}/{instance_info['total_instances']}")
                    print(f"        Fokus: {instance_info['target_category']}")
            
                for coder_index, coder in enumerate(self.deductive_coders):
                    try:
                        # Bestimme ob Summary aktualisiert werden soll (nur beim ersten Kodierer der ersten Instanz)
                        should_update_summary = (coder_index == 0 and instance_info['instance'] == 1)
                        
                        if instance_info['target_category']:
                            # Mehrfachkodierung mit Fokus und Kontext
                            combined_result = await coder.code_chunk_with_focus_and_context(
                                text, categories, 
                                focus_category=instance_info['target_category'],
                                focus_context=instance_info['category_context'],
                                current_summary=updated_summary if should_update_summary else current_summary,
                                segment_info=segment_info,
                                update_summary=should_update_summary
                            )
                        else:
                            # Standard Kontext-Kodierung
                            combined_result = await coder.code_chunk_with_progressive_context(
                                text, 
                                categories, 
                                updated_summary if should_update_summary else current_summary,
                                segment_info
                            )
                        
                        if combined_result:
                            # Extrahiere Kodierungsergebnis und ggf. aktualisiertes Summary
                            coding_result = combined_result.get('coding_result', {})
                            
                            # Summary nur beim ersten Kodierer der ersten Instanz aktualisieren
                            if should_update_summary:
                                updated_summary = combined_result.get('updated_summary', current_summary)
                                self.document_summaries[doc_name] = updated_summary
                                print(f"🔄 Summary aktualisiert: {len(updated_summary.split())} Wörter")
                            
                            # Erstelle Kodierungseintrag
                            coding_entry = {
                                'segment_id': segment_id,
                                'coder_id': coder.coder_id,
                                'category': coding_result.get('category', ''),
                                'subcategories': coding_result.get('subcategories', []),
                                'justification': coding_result.get('justification', ''),
                                'confidence': coding_result.get('confidence', {}),
                                'text': text,
                                'paraphrase': coding_result.get('paraphrase', ''),
                                'keywords': coding_result.get('keywords', ''),
                                'context_summary': updated_summary,
                                'context_influence': coding_result.get('context_influence', ''),
                                'multiple_coding_instance': instance_info['instance'],
                                'total_coding_instances': instance_info['total_instances'],
                                'target_category': instance_info['target_category'],
                                'category_focus_used': bool(instance_info['target_category'])
                            }
                            
                            batch_results.append(coding_entry)
                            
                            if instance_info['total_instances'] > 1:
                                print(f"        ✓ {coder.coder_id}: {coding_entry['category']}")
                            else:
                                print(f"  ✓ Kodierer {coder.coder_id}: {coding_entry['category']}")
                        else:
                            print(f"  ⚠ Keine Kodierung von {coder.coder_id} erhalten")
                            
                    except Exception as e:
                        print(f"  ⚠ Fehler bei {coder.coder_id}: {str(e)}")
                        continue
        
        return batch_results

    def _extract_doc_and_chunk_id(self, segment_id: str) -> Tuple[str, str]:
        """Extrahiert Dokumentname und Chunk-ID aus segment_id."""
        parts = segment_id.split('_chunk_')
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            return segment_id, "unknown"

    def _extract_segment_sort_key(self, segment_id: str) -> Tuple[str, int]:
        """Erstellt einen Sortierschlüssel für die richtige Chunk-Reihenfolge."""
        try:
            doc_name, chunk_id = self._extract_doc_and_chunk_id(segment_id)
            return (doc_name, int(chunk_id) if chunk_id.isdigit() else 0)
        except Exception:
            return (segment_id, 0)
    
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

            # Kontext-Feature aktiviert?
            use_context = CONFIG.get('CODE_WITH_CONTEXT', False)

            # Wenn Kontext verwendet wird, Segmente sortieren
            if use_context:
                print("\n🔄 Kontextuelle Kodierung aktiviert")
                print("Sortiere Segmente für sequentielle Verarbeitung...")
                all_segments.sort(key=lambda x: self._extract_segment_sort_key(x[0]))
                # Reset document_summaries
                self.document_summaries = {}
            else:
                print("\n🔄 Standardkodierung ohne Kontext aktiviert")
            
            # Reset Tracking-Variablen
            self.coding_results = []
            self.processed_segments = set()
            
            # Batch-Größe festlegen
            if batch_size is None:
                batch_size = CONFIG.get('BATCH_SIZE', 5)
            total_batches = 0
            
            total_segments = len(all_segments)
            print(f"Verarbeite {total_segments} Segmente mit Batch-Größe {batch_size}...")
            self.history.log_analysis_start(total_segments, len(initial_categories))

            analysis_mode = CONFIG.get('ANALYSIS_MODE', 'full')
            print(f"\nAnalyse-Modus: {analysis_mode}")
            
            # Spezielle Behandlung für 'grounded' Modus
            grounded_subcodes = {}  # Dictionary für gesammelte Subcodes im grounded Modus
            if analysis_mode == 'grounded':
                print("\nVerarbeitung im 'grounded' Modus:")
                print("1. Sammeln von Subcodes ohne Hauptkategorien")
                print("2. Fortlaufende Verwendung der gesammelten Subcodes durch den deduktiven Kodierer")
                print("3. Nach Abschluss aller Segmente: Generierung von Hauptkategorien")
                
                # Ersetze initiale Kategorien durch ein leeres Set im grounded Modus
                if not skip_inductive:
                    print("\n⚠️ Im grounded Modus werden die deduktiven Kategorien aus dem Codebook nicht verwendet!")
                    current_categories = {}  # Leeres Kategoriensystem zu Beginn
                
                # Initialisiere Sammler für Subcodes
                if not hasattr(self.inductive_coder, 'collected_subcodes'):
                    self.inductive_coder.collected_subcodes = []
                    self.inductive_coder.segment_analyses = []

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
                    
                    # Induktive oder Grounded Analyse wenn nicht übersprungen
                    if not skip_inductive and relevant_batch:
                        if analysis_mode == 'grounded':
                            print(f"\nStarte Grounded-Analyse für {len(relevant_batch)} relevante Segmente...")
                            
                            # Extrahiere nur die Texte für die Grounded-Analyse
                            relevant_texts = [text for _, text in relevant_batch]
                            
                            # Verwende die Methode für Grounded-Analyse
                            grounded_analysis = await self.inductive_coder.analyze_grounded_batch(
                                segments=relevant_texts,
                                material_percentage=material_percentage
                            )
                            
                            if grounded_analysis and 'segment_analyses' in grounded_analysis:
                                # Neue Subcodes aus diesem Batch extrahieren
                                new_batch_subcodes = []
                                for segment_analysis in grounded_analysis['segment_analyses']:
                                    for subcode in segment_analysis.get('subcodes', []):
                                        new_batch_subcodes.append(subcode)
                                
                                # Speichere für die Hauptkategoriengenerierung am Ende
                                self.inductive_coder.collected_subcodes.extend(new_batch_subcodes)
                                self.inductive_coder.segment_analyses.append(grounded_analysis)
                                
                                # Erstelle/aktualisiere Kategorien-Dictionary für den deduktiven Kodierer
                                for subcode in new_batch_subcodes:
                                    subcode_name = subcode.get('name', '')
                                    if subcode_name and subcode_name not in grounded_subcodes:
                                        subcode_definition = subcode.get('definition', '')
                                        subcode_keywords = subcode.get('keywords', [])
                                        subcode_evidence = subcode.get('evidence', [])
                                        
                                        # Erstelle eine minimale CategoryDefinition
                                        grounded_subcodes[subcode_name] = CategoryDefinition(
                                            name=subcode_name,
                                            definition=subcode_definition,
                                            examples=subcode_evidence,
                                            rules=[f"Identifizierte Keywords: {', '.join(subcode_keywords)}"],
                                            subcategories={},  # Keine Subkategorien für Subcodes
                                            added_date=datetime.now().strftime("%Y-%m-%d"),
                                            modified_date=datetime.now().strftime("%Y-%m-%d")
                                        )
                                
                                print(f"\nGrounded-Analyse für Batch abgeschlossen:")
                                print(f"- {len(new_batch_subcodes)} neue Subcodes in diesem Batch identifiziert")
                                print(f"- Gesamtzahl Subcodes: {len(grounded_subcodes)}")
                                
                                # Aktualisiere das aktuelle Kategoriensystem für die Kodierung
                                current_categories = grounded_subcodes.copy()
                                
                                # Aktualisiere ALLE Kodierer mit dem neuen System
                                for coder in self.deductive_coders:
                                    await coder.update_category_system(current_categories)
                                print(f"\nAlle Kodierer mit aktuellem Subcode-System ({len(current_categories)} Subcodes) aktualisiert")
                        else:
                            # Standard induktive Kategorienentwicklung für andere Modi
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

                                # Aktualisiere ALLE Kodierer mit dem neuen System
                                for coder in self.deductive_coders:
                                    await coder.update_category_system(current_categories)
                                print(f"\nAlle Kodierer mit aktuellem System ({len(current_categories)} Kategorien) aktualisiert")
                    
                    # Deduktive Kodierung für alle Segmente
                    print("\nStarte deduktive Kodierung...")

                    if use_context:
                        # Kontextuelle Kodierung sequentiell für jeden Chunk
                        batch_results = await self._code_batch_with_context(batch, current_categories)
                    else:
                        # Klassische Kodierung ohne Kontext
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
                    
                    # Status-Update für History
                    self._log_iteration_status(
                        material_percentage=material_percentage,
                        saturation_metrics=None,  # Im grounded Modus verwenden wir keine Sättigungsmetriken
                        num_results=len(batch_results)
                    )
                    
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von Batch {total_batches}: {str(e)}")
                    print("Details:")
                    traceback.print_exc()
                    continue
            
            # Nach Abschluss aller Segmente im 'grounded' Modus
            # müssen wir die Hauptkategorien generieren lassen
            if analysis_mode == 'grounded' and len(self.processed_segments) >= total_segments * 0.9:
                print("\nAlle Segmente verarbeitet. Generiere Hauptkategorien im 'grounded' Modus...")
                
                # Konvertiere das Dictionary zurück in eine Liste von Subcodes
                subcodes_for_generation = []
                for subcode_name, category in grounded_subcodes.items():
                    # Extrahiere Keywords aus den Rules
                    keywords = []
                    for rule in category.rules:
                        if "Identifizierte Keywords:" in rule:
                            keywords_str = rule.replace("Identifizierte Keywords:", "").strip()
                            keywords = [kw.strip() for kw in keywords_str.split(',')]
                            break
                    
                    subcodes_for_generation.append({
                        'name': subcode_name,
                        'definition': category.definition,
                        'keywords': keywords,
                        'evidence': category.examples,
                        'confidence': 0.8  # Default-Konfidenz
                    })
                
                # Setze die Subcodes für die Hauptkategoriengenerierung
                self.inductive_coder.collected_subcodes = subcodes_for_generation
                
                # Generiere Hauptkategorien (leeres Dict als initial_categories)
                main_categories = await self.inductive_coder._generate_main_categories_from_subcodes({})
                
                # Speichere die Zuordnung von Subcodes zu Hauptkategorien für das spätere Matching
                subcode_to_main_category = {}
                for main_name, main_category in main_categories.items():
                    for subcode_name in main_category.subcategories.keys():
                        subcode_to_main_category[subcode_name] = main_name
                
                print("\n🔄 Ordne bestehende Kodierungen den neuen Hauptkategorien zu...")
                
                # Aktualisiere alle Kodierungen mit Hauptkategorienzuordnungen
                updated_codings = []
                for coding in self.coding_results:
                    updated_coding = coding.copy()
                    
                    # Wenn der Code ein Subcode ist, ordne die entsprechende Hauptkategorie zu
                    subcode = coding.get('category', '')
                    if subcode in subcode_to_main_category:
                        main_category = subcode_to_main_category[subcode]
                        # Füge Hauptkategorie hinzu und verschiebe den bisherigen Code zu den Subcodes
                        updated_coding['main_category'] = main_category
                        if subcode not in updated_coding.get('subcategories', []):
                            updated_coding['subcategories'] = list(updated_coding.get('subcategories', [])) + [subcode]
                    else:
                        # Wenn kein Matching gefunden wurde, behalte den ursprünglichen Code als Hauptkategorie
                        updated_coding['main_category'] = subcode
                    
                    updated_codings.append(updated_coding)
                
                # Ersetze die Kodierungen mit den aktualisierten
                self.coding_results = updated_codings
                
                print(f"✅ Kodierungen aktualisiert mit Hauptkategorie-Zuordnungen")
                
                # Finales Kategoriensystem
                current_categories = main_categories
                
                # Aktualisiere ALLE Kodierer mit dem neuen System
                for coder in self.deductive_coders:
                    await coder.update_category_system(current_categories)
                print(f"\nAlle Kodierer mit finalem 'grounded' System ({len(current_categories)} Hauptkategorien) aktualisiert")
            
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
                if analysis_mode == 'grounded':
                    print(f"- Grounded Kategorienentwicklung:")
                    print(f"  • Initial: 0 Kategorien (im grounded Modus wird ohne initiale Kategorien begonnen)")
                    print(f"  • Gesammelte Subcodes: {len(grounded_subcodes)}")
                    print(f"  • Generierte Hauptkategorien: {len(current_categories)}")
                else:
                    print(f"- Kategorienentwicklung:")
                    print(f"  • Initial: {len(initial_categories)} Kategorien")
                    print(f"  • Final: {len(current_categories)} Kategorien")
                    print(f"  • Neu entwickelt: {len(current_categories) - len(initial_categories)} Kategorien")
            
            # Bei kontextueller Kodierung: Zeige Zusammenfassung der Dokument-Summaries
            if use_context and self.document_summaries:
                print("\nDocument-Summaries:")
                for doc_name, summary in self.document_summaries.items():
                    print(f"\n📄 {doc_name}:")
                    print(f"  {summary}")

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
            
            # Finalisiere Kategoriensystem mit ausführlicher Ausgabe
            print("\nFinalisiere Kategoriensystem...")
            print(f"- Kategorien vor Speicherung: {len(current_categories)}")
            for cat_name, category in current_categories.items():
                subcat_count = len(category.subcategories)
                print(f"  • {cat_name}: {subcat_count} Subkategorien")
                # Zeige alle Subkategorien für bessere Nachverfolgbarkeit
                if subcat_count > 0:
                    print(f"    Subkategorien:")
                    for subcat_name in category.subcategories.keys():
                        print(f"      - {subcat_name}")
                        
            total_subcats = sum(len(cat.subcategories) for cat in current_categories.values())
            print(f"- Insgesamt {total_subcats} Subkategorien")

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
        provider_name = CONFIG.get('MODEL_PROVIDER', 'openai').lower()  # Fallback zu OpenAI
        try:
            # Wir lassen den LLMProvider sich selbst um die Client-Initialisierung kümmern
            self.llm_provider = LLMProviderFactory.create_provider(provider_name)
            print(f"🤖 LLM Provider '{provider_name}' für Kodierer {coder_id} initialisiert")
        except Exception as e:
            print(f"Fehler bei Provider-Initialisierung für {coder_id}: {str(e)}")
            raise
        
        # Prompt-Handler initialisieren
        self.prompt_handler = QCAPrompts(
            forschungsfrage=FORSCHUNGSFRAGE,
            kodierregeln=KODIERREGELN,
            deduktive_kategorien=DEDUKTIVE_KATEGORIEN

        )

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

    async def code_chunk_with_context_switch(self, 
                                         chunk: str, 
                                         categories: Dict[str, CategoryDefinition],
                                         use_context: bool = True,
                                         context_data: Dict = None) -> Dict:
        """
        Wrapper-Methode, die basierend auf dem use_context Parameter
        zwischen code_chunk und code_chunk_with_progressive_context wechselt.
        
        Args:
            chunk: Zu kodierender Text
            categories: Kategoriensystem
            use_context: Ob Kontext verwendet werden soll
            context_data: Kontextdaten (current_summary und segment_info)
            
        Returns:
            Dict: Kodierungsergebnis (und ggf. aktualisiertes Summary)
        """
        if use_context and context_data:
            # Mit progressivem Kontext kodieren
            return await self.code_chunk_with_progressive_context(
                chunk, 
                categories, 
                context_data.get('current_summary', ''),
                context_data.get('segment_info', {})
            )
        else:
            # Klassische Kodierung ohne Kontext
            result = await self.code_chunk(chunk, categories)
            
            # Wandle CodingResult in Dictionary um, wenn nötig
            if result and isinstance(result, CodingResult):
                return {
                    'coding_result': result.to_dict(),
                    'updated_summary': context_data.get('current_summary', '') if context_data else ''
                }
            elif result and isinstance(result, dict):
                return {
                    'coding_result': result,
                    'updated_summary': context_data.get('current_summary', '') if context_data else ''
                }
            else:
                return None
            
    async def code_chunk(self, chunk: str, categories: Optional[Dict[str, CategoryDefinition]] = None, is_last_segment: bool = False) -> Optional[CodingResult]:
        """
        Kodiert einen Text-Chunk basierend auf dem aktuellen Kategoriensystem.
        
        Args:
            chunk: Zu kodierender Text
            categories: Optional übergebenes Kategoriensystem (wird nur verwendet wenn kein aktuelles System existiert)
            is_last_segment: Gibt an, ob dies das letzte zu kodierende Segment ist
            
        Returns:
            Optional[CodingResult]: Kodierungsergebnis oder None bei Fehler
        """
        try:
            # Speichere Information, ob letztes Segment
            self.is_last_segment = is_last_segment

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
            
            prompt = self.prompt_handler.get_deductive_coding_prompt(
                chunk=chunk,
                categories_overview=categories_overview
            )


            try:
                input_tokens = estimate_tokens(prompt)

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

                        # Verarbeite Paraphrase
                        paraphrase = result.get('paraphrase', '')
                        if paraphrase:
                            print(f"\n🗒️  Paraphrase: {paraphrase}")


                        print(f"\n  ✓ Kodierung von {self.coder_id}: 🏷️  {result.get('category', '')}")
                        print(f"  ✓ Subkategorien von {self.coder_id}: 🏷️  {', '.join(result.get('subcategories', []))}")
                        print(f"  ✓ Keywords von {self.coder_id}: 🏷️  {result.get('keywords', '')}")

                        # Debug-Ausgaben
                        print("\n👨‍⚖️  Kodierungsbegründung:")
                        
                        # Verarbeite Begründung
                        justification = result.get('justification', '')
                        if isinstance(justification, dict):
                            # Formatiere Dictionary-Begründung
                            for key, value in justification.items():
                                print(f"  {key}: {value}")
                        elif justification:
                            print(f"  {justification}")
                                               
                                               
                        # Zeige Definition-Matches wenn vorhanden
                        definition_matches = result.get('definition_matches', [])
                        if isinstance(definition_matches, list) and definition_matches:
                            print("\n  Passende Definitionsaspekte:")
                            for match in definition_matches:
                                print(f"  - {match}")
                                
                        # Zeige Konfidenzdetails
                        confidence = result.get('confidence', {})
                        if isinstance(confidence, dict) and confidence:
                            print("\n  Konfidenzwerte:")
                            for key, value in confidence.items():
                                if isinstance(value, (int, float)):
                                    print(f"  - {key}: {value:.2f}")
                    
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

    async def code_chunk_with_progressive_context(self, 
                                          chunk: str, 
                                          categories: Dict[str, CategoryDefinition],
                                          current_summary: str,
                                          segment_info: Dict) -> Dict:
        """
        Kodiert einen Text-Chunk und aktualisiert gleichzeitig das Dokument-Summary
        mit einem dreistufigen Reifungsmodell.
        
        Args:
            chunk: Zu kodierender Text
            categories: Kategoriensystem
            current_summary: Aktuelles Dokument-Summary
            segment_info: Zusätzliche Informationen über das Segment
            
        Returns:
            Dict: Enthält sowohl Kodierungsergebnis als auch aktualisiertes Summary
        """
        try:
            # Verwende das interne Kategoriensystem wenn vorhanden, sonst das übergebene
            current_categories = self.current_categories or categories
            
            if not current_categories:
                print(f"Fehler: Kein Kategoriensystem für Kodierer {self.coder_id} verfügbar")
                return None

            print(f"\nDeduktiver Kodierer 🧐 **{self.coder_id}** verarbeitet Chunk mit progressivem Kontext...")
            
            # Erstelle formatierte Kategorienübersicht
            categories_overview = []
            for name, cat in current_categories.items():
                category_info = {
                    'name': name,
                    'definition': cat.definition,
                    'examples': list(cat.examples) if isinstance(cat.examples, set) else cat.examples,
                    'rules': list(cat.rules) if isinstance(cat.rules, set) else cat.rules,
                    'subcategories': {}
                }
                
                # Füge Subkategorien hinzu
                for sub_name, sub_def in cat.subcategories.items():
                    category_info['subcategories'][sub_name] = sub_def
                    
                categories_overview.append(category_info)
            
            # Position im Dokument und Fortschritt berechnen
            position_info = f"Segment: {segment_info.get('position', '')}"
            doc_name = segment_info.get('doc_name', 'Unbekanntes Dokument')
            
            # Berechne die relative Position im Dokument (für das Reifungsmodell)
            chunk_id = 0
            total_chunks = 1
            if 'position' in segment_info:
                try:
                    # Extrahiere Chunk-Nummer aus "Chunk X"
                    chunk_id = int(segment_info['position'].split()[-1])
                    
                    # Schätze Gesamtanzahl der Chunks (basierend auf bisherigen Chunks)
                    # Alternative: Tatsächliche Anzahl übergeben, falls verfügbar
                    total_chunks = max(chunk_id * 1.5, 20)  # Schätzung
                    
                    document_progress = chunk_id / total_chunks
                    print(f"Dokumentfortschritt: ca. {document_progress:.1%}")
                except (ValueError, IndexError):
                    document_progress = 0.5  # Fallback
            else:
                document_progress = 0.5  # Fallback
                
            # Bestimme die aktuelle Reifephase basierend auf dem Fortschritt
            if document_progress < 0.3:
                reifephase = "PHASE 1 (Sammlung)"
                max_aenderung = "50%"
            elif document_progress < 0.7:
                reifephase = "PHASE 2 (Konsolidierung)"
                max_aenderung = "30%"
            else:
                reifephase = "PHASE 3 (Präzisierung)"
                max_aenderung = "10%"
                
            print(f"Summary-Reifephase: {reifephase}, max. Änderung: {max_aenderung}")
            
            # Angepasster Prompt basierend auf dem dreistufigen Reifungsmodell
            # Verbesserter summary_update_prompt für die _code_chunk_with_progressive_context Methode

            summary_update_prompt = f"""
            ## AUFGABE 2: SUMMARY-UPDATE ({reifephase}, {int(document_progress*100)}%)

            """

            # Robustere Phasen-spezifische Anweisungen
            if document_progress < 0.3:
                summary_update_prompt += """
            SAMMLUNG (0-30%) - STRUKTURIERTER AUFBAU:
            - SCHLÜSSELINFORMATIONEN: Beginne mit einer LISTE wichtigster Konzepte im Telegrammstil
            - FORMAT: "Thema1: Kernaussage; Thema2: Kernaussage" 
            - SPEICHERSTRUKTUR: Speichere alle Informationen in KATEGORIEN (z.B. Akteure, Prozesse, Faktoren)
            - KEINE EINLEITUNGEN oder narrative Elemente, NUR Fakten und Verbindungen
            - BEHALTE IMMER: Bereits dokumentierte Schlüsselkonzepte müssen bestehen bleiben
            """
            elif document_progress < 0.7:
                summary_update_prompt += """
            KONSOLIDIERUNG (30-70%) - HIERARCHISCHE ORGANISATION:
            - SCHLÜSSELINFORMATIONEN BEWAHREN: Alle bisherigen Hauptkategorien beibehalten
            - NEUE STRUKTUR: Als hierarchische Liste mit Kategorien und Unterpunkten organisieren
            - KOMPRIMIEREN: Details aus gleichen Themenbereichen zusammenführen
            - PRIORITÄTSFORMAT: "Kategorie: Hauptpunkt1; Hauptpunkt2 → Detail"
            - STATT LÖSCHEN: Verwandte Inhalte zusammenfassen, aber KEINE Kategorien eliminieren
            """
            else:
                summary_update_prompt += """
            PRÄZISIERUNG (70-100%) - VERDICHTUNG MIT THESAURUS:
            - THESAURUS-METHODE: Jede Kategorie braucht genau 1-2 Sätze im Telegrammstil
            - HAUPTKONZEPTE STABIL HALTEN: Alle identifizierten Kategorien müssen enthalten bleiben
            - ABSTRAHIEREN: Einzelinformationen innerhalb einer Kategorie verdichten
            - STABILITÄTSPRINZIP: Einmal erkannte wichtige Zusammenhänge dürfen nicht verloren gehen
            - PRIORITÄTSORDNUNG: Wichtigste Informationen IMMER am Anfang jeder Kategorie
            """

            # Allgemeine Kriterien für Stabilität und Komprimierung
            summary_update_prompt += """

            INFORMATIONSERHALTUNGS-SYSTEM:
            - MAXIMUM 80 WÖRTER - Komprimiere alte statt neue Informationen zu verwerfen
            - KATEGORIEBASIERT: Jedes Summary muss immer in 3-5 klare Themenkategorien strukturiert sein
            - SCHLÜSSELPRINZIP: Bilde das Summary als INFORMATIONALE HIERARCHIE:
            1. Stufe: Immer stabile Themenkategorien
            2. Stufe: Zentrale Aussagen zu jeder Kategorie
            3. Stufe: Ergänzende Details (diese können komprimiert werden)
            - STABILITÄTSGARANTIE: Neue Iteration darf niemals vorherige Kategorie-Level-1-Information verlieren
            - KOMPRIMIERUNGSSTRATEGIE: Bei Platzmangel Details (Stufe 3) zusammenfassen statt zu entfernen
            - FORMAT: "Kategorie1: Hauptpunkt; Hauptpunkt. Kategorie2: Hauptpunkt; Detail." (mit Doppelpunkten)
            - GRUNDREGEL: Neue Informationen ergänzen bestehende Kategorien statt sie zu ersetzen
            """
            
            # Prompt mit erweiterter Aufgabe für Summary-Update
            prompt = self.prompt_handler.get_progressive_context_prompt(
                chunk=chunk,
                categories_overview=categories_overview,
                current_summary=current_summary,
                position_info=position_info,
                summary_update_prompt=summary_update_prompt
            )
            
            # API-Call
            input_tokens = estimate_tokens(prompt)

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
            
            # Extrahiere relevante Teile
            if result and isinstance(result, dict):
                coding_result = result.get('coding_result', {})
                updated_summary = result.get('updated_summary', current_summary)
                
                # Prüfe Wortlimit beim Summary
                if len(updated_summary.split()) > 80:  # Etwas Spielraum über 70
                    words = updated_summary.split()
                    updated_summary = ' '.join(words[:70])
                    print(f"⚠️ Summary wurde gekürzt: {len(words)} → 70 Wörter")
                
                # Analyse der Veränderungen
                if current_summary:
                    # Berechne Prozent der Änderung
                    old_words = set(current_summary.lower().split())
                    new_words = set(updated_summary.lower().split())
                    
                    if old_words:
                        # Jaccard-Distanz als Maß für Veränderung
                        unchanged = len(old_words.intersection(new_words))
                        total = len(old_words.union(new_words))
                        change_percent = (1 - (unchanged / total)) * 100
                        
                        print(f"Summary Änderung: {change_percent:.1f}% (Ziel: max. {max_aenderung})")
                
                if coding_result:
                    paraphrase = coding_result.get('paraphrase', '')
                    if paraphrase:
                        print(f"\n🗒️  Paraphrase: {paraphrase}")
                    print(f"  ✓ Kodierung von {self.coder_id}: 🏷️  {coding_result.get('category', '')}")
                    print(f"  ✓ Subkategorien von {self.coder_id}: 🏷️  {', '.join(coding_result.get('subcategories', []))}")
                    print(f"  ✓ Keywords von {self.coder_id}: 🏷️  {coding_result.get('keywords', '')}")
                    print(f"\n📝 Summary für {doc_name} aktualisiert ({len(updated_summary.split())} Wörter):")
                    print(f"{updated_summary[:1000]}..." if len(updated_summary) > 100 else f"📄 {updated_summary}")
                    
                    # Kombiniertes Ergebnis zurückgeben
                    return {
                        'coding_result': coding_result,
                        'updated_summary': updated_summary
                    }
                else:
                    print(f"  ✗ Keine gültige Kodierung erhalten")
                    return None
            else:
                print("  ✗ Keine gültige Antwort erhalten")
                return None
                
        except Exception as e:
            print(f"Fehler bei der Kodierung durch {self.coder_id}: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()
            return None

    async def code_chunk_with_focus(self, chunk: str, categories: Dict[str, CategoryDefinition], 
                                focus_category: str, focus_context: Dict) -> Optional[CodingResult]:
        """
        Kodiert einen Text-Chunk mit Fokus auf eine bestimmte Kategorie (für Mehrfachkodierung).
        
        Args:
            chunk: Zu kodierender Text
            categories: Kategoriensystem  
            focus_category: Kategorie auf die fokussiert werden soll
            focus_context: Kontext zur fokussierten Kategorie mit 'justification', 'text_aspects', 'relevance_score'
            
        Returns:
            Optional[CodingResult]: Kodierungsergebnis mit Fokus-Kennzeichnung
        """
        try:
            # Verwende das interne Kategoriensystem wenn vorhanden, sonst das übergebene
            current_categories = self.current_categories or categories
            
            if not current_categories:
                print(f"Fehler: Kein Kategoriensystem für Kodierer {self.coder_id} verfügbar")
                return None

            print(f"    🎯 Fokuskodierung für Kategorie: {focus_category} (Relevanz: {focus_context.get('relevance_score', 0):.2f})")
            
            # Erstelle formatierte Kategorienübersicht mit Fokus-Hervorhebung
            categories_overview = []
            for name, cat in current_categories.items():
                # Hebe Fokus-Kategorie hervor
                display_name = name
                if name == focus_category:
                    display_name = name
                
                category_info = {
                    'name': display_name,
                    'definition': cat.definition,
                    'examples': list(cat.examples) if isinstance(cat.examples, set) else cat.examples,
                    'rules': list(cat.rules) if isinstance(cat.rules, set) else cat.rules,
                    'subcategories': {}
                }
                
                # Füge Subkategorien hinzu
                for sub_name, sub_def in cat.subcategories.items():
                    category_info['subcategories'][sub_name] = sub_def
                    
                categories_overview.append(category_info)

            # Erstelle fokussierten Prompt
            prompt = self.prompt_handler.get_focus_coding_prompt(
                chunk=chunk,
                categories_overview=categories_overview,
                focus_category=focus_category,
                focus_context=focus_context
            )
            
            try:
                input_tokens = estimate_tokens(prompt)

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
                        # Verarbeite Paraphrase
                        paraphrase = result.get('paraphrase', '')
                        if paraphrase:
                            print(f"      🗒️  Fokus-Paraphrase: {paraphrase}")

                        # Dokumentiere Fokus-Adherence
                        focus_adherence = result.get('focus_adherence', {})
                        followed_focus = focus_adherence.get('followed_focus', True)
                        focus_icon = "🎯" if followed_focus else "🔄"
                        
                        print(f"      {focus_icon} Fokuskodierung von {self.coder_id}: 🏷️  {result.get('category', '')}")
                        print(f"      ✓ Subkategorien: 🏷️  {', '.join(result.get('subcategories', []))}")
                        print(f"      ✓ Keywords: 🏷️  {result.get('keywords', '')}")
                        
                        if not followed_focus:
                            deviation_reason = focus_adherence.get('deviation_reason', 'Nicht angegeben')
                            print(f"      ⚠️ Fokus-Abweichung: {deviation_reason}")

                        # Debug-Ausgaben für Fokus-Details
                        if focus_adherence:
                            focus_score = focus_adherence.get('focus_category_score', 0)
                            chosen_score = focus_adherence.get('chosen_category_score', 0)
                            print(f"      📊 Fokus-Score: {focus_score:.2f}, Gewählt-Score: {chosen_score:.2f}")

                        # Erweiterte Begründung mit Fokus-Kennzeichnung
                        original_justification = result.get('justification', '')
                        focus_prefix = f"[FOKUS: {focus_category}, Adherence: {'Ja' if followed_focus else 'Nein'}] "
                        enhanced_justification = focus_prefix + original_justification
                    
                        return CodingResult(
                            category=result.get('category', ''),
                            subcategories=tuple(result.get('subcategories', [])),
                            justification=enhanced_justification,
                            confidence=result.get('confidence', {'total': 0.0, 'category': 0.0, 'subcategories': 0.0}),
                            text_references=tuple([chunk[:100]]),
                            uncertainties=None,
                            paraphrase=result.get('paraphrase', ''),
                            keywords=result.get('keywords', '')
                        )
                    else:
                        print("      ✗ Keine passende Kategorie gefunden")
                        return None
                    
            except Exception as e:
                print(f"Fehler bei API Call für fokussierte Kodierung: {str(e)}")
                return None          

        except Exception as e:
            print(f"Fehler bei der fokussierten Kodierung durch {self.coder_id}: {str(e)}")
            return None

    async def code_chunk_with_focus_and_context(self, 
                                            chunk: str, 
                                            categories: Dict[str, CategoryDefinition],
                                            focus_category: str,
                                            focus_context: Dict,
                                            current_summary: str,
                                            segment_info: Dict,
                                            update_summary: bool = False) -> Dict:
        """
        Kodiert einen Text-Chunk mit Fokus auf eine Kategorie UND progressivem Kontext.
        Kombiniert die Funktionalität von code_chunk_with_focus und code_chunk_with_progressive_context.
        
        Args:
            chunk: Zu kodierender Text
            categories: Kategoriensystem
            focus_category: Kategorie auf die fokussiert werden soll
            focus_context: Kontext zur fokussierten Kategorie
            current_summary: Aktuelles Dokument-Summary
            segment_info: Zusätzliche Informationen über das Segment
            update_summary: Ob das Summary aktualisiert werden soll
            
        Returns:
            Dict: Enthält sowohl Kodierungsergebnis als auch aktualisiertes Summary
        """
        try:
            # Verwende das interne Kategoriensystem wenn vorhanden, sonst das übergebene
            current_categories = self.current_categories or categories
            
            if not current_categories:
                print(f"Fehler: Kein Kategoriensystem für Kodierer {self.coder_id} verfügbar")
                return None

            print(f"      🎯 Fokus-Kontext-Kodierung für: {focus_category}")
            
            # Erstelle formatierte Kategorienübersicht mit Fokus-Hervorhebung
            categories_overview = []
            for name, cat in current_categories.items():
                # Hebe Fokus-Kategorie hervor
                display_name = name
                if name == focus_category:
                    display_name = name
                
                category_info = {
                    'name': display_name,
                    'definition': cat.definition,
                    'examples': list(cat.examples) if isinstance(cat.examples, set) else cat.examples,
                    'rules': list(cat.rules) if isinstance(cat.rules, set) else cat.rules,
                    'subcategories': {}
                }
                
                # Füge Subkategorien hinzu
                for sub_name, sub_def in cat.subcategories.items():
                    category_info['subcategories'][sub_name] = sub_def
                    
                categories_overview.append(category_info)

            # Position im Dokument und Fortschritt berechnen
            position_info = f"Segment: {segment_info.get('position', '')}"
            doc_name = segment_info.get('doc_name', 'Unbekanntes Dokument')
            
            # Summary-Update-Anweisungen (gleiche Logik wie in code_chunk_with_progressive_context)
            summary_update_prompt = ""
            if update_summary:
                summary_update_prompt = f"""
                ## AUFGABE 2: SUMMARY-UPDATE

                INFORMATIONSERHALTUNGS-SYSTEM:
                - MAXIMUM 80 WÖRTER - Komprimiere alte statt neue Informationen zu verwerfen
                - KATEGORIEBASIERT: Jedes Summary muss immer in 3-5 klare Themenkategorien strukturiert sein
                - SCHLÜSSELPRINZIP: Bilde das Summary als INFORMATIONALE HIERARCHIE:
                1. Stufe: Immer stabile Themenkategorien
                2. Stufe: Zentrale Aussagen zu jeder Kategorie
                3. Stufe: Ergänzende Details (diese können komprimiert werden)
                - STABILITÄTSGARANTIE: Neue Iteration darf niemals vorherige Kategorie-Level-1-Information verlieren
                - KOMPRIMIERUNGSSTRATEGIE: Bei Platzmangel Details (Stufe 3) zusammenfassen statt zu entfernen
                - FORMAT: "Kategorie1: Hauptpunkt; Hauptpunkt. Kategorie2: Hauptpunkt; Detail." (mit Doppelpunkten)
                - GRUNDREGEL: Neue Informationen ergänzen bestehende Kategorien statt sie zu ersetzen
                """

            # Erstelle fokussierten Prompt mit Kontext
            prompt = self.prompt_handler.get_focus_context_coding_prompt(
                chunk=chunk,
                categories_overview=categories_overview,
                focus_category=focus_category,
                focus_context=focus_context,
                current_summary=current_summary,
                position_info=position_info,
                summary_update_prompt=summary_update_prompt,
                update_summary=update_summary
            )
            

            # API-Call
            input_tokens = estimate_tokens(prompt)

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
            
            # Extrahiere relevante Teile
            if result and isinstance(result, dict):
                coding_result = result.get('coding_result', {})
                
                # Summary nur aktualisieren wenn angefordert
                if update_summary:
                    updated_summary = result.get('updated_summary', current_summary)
                    
                    # Prüfe Wortlimit beim Summary
                    if len(updated_summary.split()) > 80:
                        words = updated_summary.split()
                        updated_summary = ' '.join(words[:70])
                        print(f"        ⚠️ Summary wurde gekürzt: {len(words)} → 70 Wörter")
                else:
                    updated_summary = current_summary
                
                if coding_result:
                    paraphrase = coding_result.get('paraphrase', '')
                    if paraphrase:
                        print(f"        🗒️  Fokus-Kontext-Paraphrase: {paraphrase}")

                    # Dokumentiere Fokus-Adherence
                    focus_adherence = coding_result.get('focus_adherence', {})
                    followed_focus = focus_adherence.get('followed_focus', True)
                    focus_icon = "🎯" if followed_focus else "🔄"
                    
                    print(f"        {focus_icon} Fokus-Kontext-Kodierung von {self.coder_id}: 🏷️  {coding_result.get('category', '')}")
                    print(f"        ✓ Subkategorien: 🏷️  {', '.join(coding_result.get('subcategories', []))}")
                    print(f"        ✓ Keywords: 🏷️  {coding_result.get('keywords', '')}")
                    
                    if not followed_focus:
                        deviation_reason = focus_adherence.get('deviation_reason', 'Nicht angegeben')
                        print(f"        ⚠️ Fokus-Abweichung: {deviation_reason}")

                    if update_summary:
                        print(f"        📝 Summary aktualisiert ({len(updated_summary.split())} Wörter)")
                    
                    # Kombiniertes Ergebnis zurückgeben
                    return {
                        'coding_result': coding_result,
                        'updated_summary': updated_summary
                    }
                else:
                    print(f"        ✗ Keine gültige Kodierung erhalten")
                    return None
            else:
                print("        ✗ Keine gültige Antwort erhalten")
                return None
                
        except Exception as e:
            print(f"Fehler bei der fokussierten Kontext-Kodierung durch {self.coder_id}: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()
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
        provider_name = CONFIG.get('MODEL_PROVIDER', 'openai').lower()
        try:
            self.llm_provider = LLMProviderFactory.create_provider(provider_name)
            print(f"\n🤖 LLM Provider '{provider_name}' für induktive Kodierung initialisiert")
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
        

        # Speichere den Analysemodus aus CONFIG
        self.analysis_mode = CONFIG.get('ANALYSIS_MODE', 'full')
        print(f"\n🔍 InductiveCoder verwendet Analysemodus: {self.analysis_mode}")

        # Tracking
        self.history = history
        self.development_history = []
        self.last_analysis_time = None

         # Verwende zentrale Validierung
        self.validator = CategoryValidator(config)

        # Initialisiere SaturationChecker
        self.saturation_checker = SaturationChecker(config, history)

        # Prompt-Handler
        self.prompt_handler = QCAPrompts(
            forschungsfrage=FORSCHUNGSFRAGE,
            kodierregeln=KODIERREGELN,
            deduktive_kategorien=DEDUKTIVE_KATEGORIEN
        )
        
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
       
    async def develop_category_system(self, segments: List[str], initial_categories: Dict[str, CategoryDefinition] = None) -> Dict[str, CategoryDefinition]:

        """Entwickelt induktiv neue Kategorien mit inkrementeller Erweiterung."""
        try:
            # Voranalyse und Filterung der Segmente
            relevant_segments = await self._prefilter_segments(segments)
            
            # Erstelle Batches für Analyse
            batches = self._create_batches(relevant_segments)
            
            # Initialisiere Dict für das erweiterte Kategoriensystem
            # extended_categories = {}
            # Initialisiere Dict für das erweiterte Kategoriensystem mit den übergebenen Kategorien
            extended_categories = initial_categories.copy() if initial_categories else {}
            
            # Hole den konfigurierten Analyse-Modus
            #analysis_mode = self.analysis_mode
            analysis_mode = CONFIG.get('ANALYSIS_MODE', 'full')
            print(f"\nAnalyse-Modus: {analysis_mode}")

            # Anpassung für 'grounded' Modus
            if analysis_mode == 'grounded':
                # Im 'grounded' Modus speichern wir alle identifizierten Subkategorien und Keywords
                # ohne sie sofort Hauptkategorien zuzuordnen
                self.collected_subcodes = []
                self.collected_keywords = []
                self.segment_analyses = []
                
                # Alternative Verarbeitung für 'grounded' Modus
                for batch_idx, batch in enumerate(batches):
                    print(f"\nAnalysiere Batch {batch_idx + 1}/{len(batches)} im 'grounded' Modus...")
                    
                    # Berechne Material-Prozentsatz
                    material_percentage = ((batch_idx + 1) * len(batch) / len(segments)) * 100
                    
                    # Grounded Analyse des Batches
                    batch_analysis = await self.analyze_grounded_batch(
                        segments=batch,
                        material_percentage=material_percentage
                    )
                    
                    if batch_analysis:
                        # Sammle Subkategorien und Keywords
                        self.collected_subcodes.extend(batch_analysis.get('subcodes', []))
                        self.collected_keywords.extend(batch_analysis.get('keywords', []))
                        self.segment_analyses.append(batch_analysis)
                
                # Nach Abschluss aller Batches: Generiere Hauptkategorien aus gesammelten Daten
                if len(segments) == len(self.processed_segments):
                    print("\nVerarbeitung aller Segmente abgeschlossen. Generiere Hauptkategorien...")
                    extended_categories = await self._generate_main_categories_from_subcodes(initial_categories)
                    return extended_categories
                
                # Falls noch nicht alle Segmente verarbeitet wurden, geben wir das initiale System zurück
                return initial_categories or {}
            
            for batch_idx, batch in enumerate(batches):
                print(f"\nAnalysiere Batch {batch_idx + 1}/{len(batches)}...")

                # Berechne Material-Prozentsatz
                material_percentage = ((batch_idx + 1) * len(batch) / len(segments)) * 100
                
                # Übergebe aktuelles System an Batch-Analyse
                batch_analysis = await self.analyze_category_batch(
                    category=extended_categories,  # Wichtig: Übergebe bisheriges System
                    segments=batch,
                    material_percentage=material_percentage,
                    analysis_mode=analysis_mode  # Füge den Analyse-Modus hier hinzu
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
                                        current_cat = CategoryDefinition(  # Neuer Code
                                            name=current_cat.name,
                                            definition=refinements['definition'],
                                            examples=current_cat.examples,
                                            rules=current_cat.rules,
                                            subcategories=current_cat.subcategories,
                                            added_date=current_cat.added_date,
                                            modified_date=datetime.now().strftime("%Y-%m-%d")
                                        )
                                
                                # Verbesserte Verarbeitung von Subkategorien, besonders wichtig für den abduktiven Modus
                                if 'new_subcategories' in updates:
                                    new_subcats = {}
                                    for sub in updates['new_subcategories']:
                                        # Erhöhte Schwelle für den abduktiven Modus, da dieser sich auf Subkategorien konzentriert
                                        confidence_threshold = 0.7 if analysis_mode != 'abductive' else 0.6
                                        
                                        if sub['confidence'] > confidence_threshold:
                                            new_subcats[sub['name']] = sub['definition']
                                            # Für abduktiven Modus: Ausführlichere Logging
                                            if analysis_mode == 'abductive':
                                                print(f"   ✓ Neue Subkategorie im abduktiven Modus: {sub['name']} (Konfidenz: {sub['confidence']:.2f})")
                                    
                                    if new_subcats:
                                        current_cat = CategoryDefinition(
                                            name=current_cat.name,
                                            definition=current_cat.definition,
                                            examples=current_cat.examples,
                                            rules=current_cat.rules,
                                            subcategories={**current_cat.subcategories, **new_subcats},
                                            added_date=current_cat.added_date,
                                            modified_date=datetime.now().strftime("%Y-%m-%d")
                                        )
                                        extended_categories[cat_name] = current_cat
                                        # Für abduktiven Modus: Spezifisches Logging
                                        # if analysis_mode == 'abductive':
                                            # print(f"✓ Kategorie '{cat_name}' im abduktiven Modus mit {len(new_subcats)} neuen Subkategorien aktualisiert")
                                        
                    # 2. Füge neue Kategorien hinzu (nur im 'full' Modus)
                    if analysis_mode == 'full' and 'new_categories' in batch_analysis:
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
                    
                    print(f"\n----------------------------------------------------------------------")
                    print(f"\nZwischenstand nach Batch {batch_idx + 1}:")
                    print(f"- Kategorien gesamt: {len(extended_categories)}")
                    print(f"- Davon neu in diesem Batch: {len(batch_analysis.get('new_categories', []))}")
                    print(f"- Aktualisiert in diesem Batch: {len(batch_analysis.get('existing_categories', {}))}")
                    print(f"\n----------------------------------------------------------------------")

            
            return extended_categories
                
        except Exception as e:
            print(f"Fehler bei Kategorienentwicklung: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return {}
    
    async def analyze_category_batch(self, 
                                category: Dict[str, CategoryDefinition], 
                                segments: List[str],
                                material_percentage: float,
                                analysis_mode: str = 'full') -> Dict[str, Any]:  
        """
        Verbesserte Batch-Analyse mit Berücksichtigung des aktuellen Kategoriensystems und Analyse-Modus.
        Spezielle Unterstützung für den abduktiven Modus hinzugefügt.
        
        Args:
            category: Aktuelles Kategoriensystem
            segments: Liste der Textsegmente
            material_percentage: Prozentsatz des verarbeiteten Materials
            analysis_mode: Analyse-Modus ('deductive', 'full', 'abductive')
            
        Returns:
            Dict[str, Any]: Analyseergebnisse einschließlich Sättigungsmetriken
        """
        try:
            # Cache-Key erstellen
            if isinstance(category, dict):
                # Wenn es ein Dict von CategoryDefinition-Objekten ist, wandle es in einen String um
                if all(isinstance(v, CategoryDefinition) for v in category.values()):
                    cache_key = (
                        frozenset((k, str(v)) for k, v in category.items()),
                        tuple(segments),
                        analysis_mode
                    )
                else:
                    cache_key = (
                        frozenset(category.items()),
                        tuple(segments),
                        analysis_mode
                    )
            else:
                cache_key = (
                    str(category),
                    tuple(segments),
                    analysis_mode
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

            # Definiere modusbasierte Anweisungen
            mode_instructions = ""
            if analysis_mode == 'abductive':
                mode_instructions = """
                BESONDERE ANWEISUNGEN FÜR DEN ABDUKTIVEN MODUS:
                - KEINE NEUEN HAUPTKATEGORIEN entwickeln
                - Du DARFST NUR die folgenden existierenden Hauptkategorien mit Subkategorien erweitern: {', '.join(category.keys())}
                - Konzentriere dich AUSSCHLIESSLICH auf die Verfeinerung des bestehenden Systems
                - Prüfe JEDE bestehende Hauptkategorie auf mögliche neue Subkategorien
                - Subkategorien sollen differenzierend, präzise und klar definiert sein
                - Bei JEDER Hauptkategorie mindestens eine mögliche neue Subkategorie in Betracht ziehen
                - Setze niedrigere Konfidenzschwelle für Subkategorien als für Hauptkategorien
                """
            elif analysis_mode == 'full':
                mode_instructions = """
                ANWEISUNGEN FÜR DEN VOLLEN INDUKTIVEN MODUS:
                - Sowohl neue Hauptkategorien als auch neue Subkategorien entwickeln
                - Bestehende Kategorien verfeinern und erweitern
                - Vollständig neue Kategorien nur bei wirklich neuen Phänomenen hinzufügen
                """
            elif analysis_mode == 'deductive':
                # Bei rein deduktiver Analyse kein neues Schema benötigt, da keine Kategorien entwickelt werden
                return {"existing_categories": {}, "new_categories": []}

            # Definiere JSON-Schema abhängig vom Analyse-Modus
            if analysis_mode == 'abductive':
                # Modifiziertes Schema für abduktiven Modus - keine neuen Hauptkategorien
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
                    "saturation_metrics": {
                        "new_aspects_found": true/false,
                        "categories_sufficient": true/false,
                        "theoretical_coverage": 0.0-1.0,
                        "justification": "Begründung der Sättigungseinschätzung"
                    }
                }'''
                
            else:  # 'full' Modus (Standard)
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

            # Anpassung des Prompt-Texts abhängig vom Analyse-Modus
            prompt = f"""
            Führe eine vollständige Kategorienanalyse basierend auf den Textsegmenten durch.
            Berücksichtige dabei das bestehende Kategoriensystem und erweitere es.
            
            {current_categories_text}
            
            TEXTSEGMENTE:
            {json.dumps(segments, indent=2, ensure_ascii=False)}
            
            FORSCHUNGSFRAGE:
            {FORSCHUNGSFRAGE}
            
            {mode_instructions}
            
            """
            
            # Ergänze den Prompt je nach Analyse-Modus
            if analysis_mode == 'abductive':
                prompt += """
            Analysiere systematisch mit FOKUS AUF SUBKATEGORIEN:
            1. BESTEHENDE HAUPTKATEGORIEN
            - Prüfe jede einzelne Hauptkategorie auf mögliche neue Subkategorien
            - Schlage konkrete, differenzierende neue Subkategorien vor
            - Belege jeden Vorschlag mit konkreten Textstellen
            - WICHTIG: Wende eine niedrigere Konfidenzschwelle für Subkategorien an (ab 0.6 verwertbar)
            - Gib für jeden Vorschlag eine präzise Definition
            
            2. VERMEIDUNG NEUER HAUPTKATEGORIEN
            - Statt neue Hauptkategorien vorzuschlagen, ordne Phänomene als Subkategorien ein
            - Sofern ein Phänomen nicht als Subkategorie passt, explizit erläutern warum
            
            3. BEGRÜNDUNG UND EVIDENZ
            - Belege alle Vorschläge mit konkreten Textstellen
            - Dokumentiere Entscheidungsgründe transparent
            - Gib Konfidenzwerte zwischen 0 und 1 für jeden Vorschlag an

            4. SÄTTIGUNGSANALYSE
            - Bewerte, ob neue inhaltliche Aspekte gefunden wurden
            - Prüfe, ob bestehende Haupt- und Subkategorien ausreichen
            - Beurteile, ob weitere Subkategorienentwicklung nötig ist
                """
            else:  # full mode
                prompt += """
            Analysiere systematisch:
            1. BESTEHENDE KATEGORIEN
            - Prüfe ob neue Aspekte zu bestehenden Kategorien passen
            - Schlage Erweiterungen/Verfeinerungen vor
            - Identifiziere neue Subkategorien für bestehende Hauptkategorien

            2. NEUE KATEGORIEN
            - Identifiziere gänzlich neue Aspekte
            - die für die Beantwortung der Forschungsfrage UNBEDINGT NOTWENDIG sind
            - identifizierten Phänomene fallen NICHT unter bestehende Kategorien 
            - Entwickle neue Hauptkategorien nur wenn nötig
            - Prüfe bei jedem Vorschlag kritisch: Trägt diese neue Kategorie wesentlich zur Beantwortung der Forschungsfrage bei oder handelt es sich um einen Aspekt, 
            der als Subkategorie bestehender Hauptkategorien besser aufgehoben wäre?
            - Stelle Trennschärfe zu bestehenden Kategorien sicher

            3. BEGRÜNDUNG UND EVIDENZ
            - Belege alle Vorschläge mit Textstellen
            - Dokumentiere Entscheidungsgründe
            - Gib Konfidenzwerte für Vorschläge an

            4. SÄTTIGUNGSANALYSE
            - Bewerte ob neue inhaltliche Aspekte gefunden wurden
            - Prüfe ob bestehende Haupt- und Sub-Kategorien ausreichen
            - Beurteile ob weitere Kategorienentwicklung nötig ist
                """

            prompt += f"""
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
            
            # Debug-Ausgaben basierend auf Analyse-Modus
            print(f"\n{analysis_mode.upper()} Mode Analyseergebnisse:")

            # Gemeinsame Debug-Ausgaben für alle Modi
            if 'existing_categories' in result:
                updates_count = len(result['existing_categories'])
                print(f"\n{updates_count} bestehende Kategorien analysiert")
                
                if updates_count > 0:
                    for cat_name, updates in result['existing_categories'].items():
                        print(f"\nAktualisierung für '{cat_name}':")
                        
                        if 'refinements' in updates:
                            confidence = updates['refinements'].get('confidence', 0)
                            print(f"- Definition erweitert (Konfidenz: {confidence:.2f})")
                        
                        if 'new_subcategories' in updates:
                            subcats = updates['new_subcategories']
                            print(f"- {len(subcats)} neue Subkategorien gefunden:")
                            
                            # Ausführlichere Ausgabe für den abduktiven Modus
                            if analysis_mode == 'abductive':
                                for sub in subcats:
                                    print(f"  • {sub['name']} (Konfidenz: {sub['confidence']:.2f})")
                                    print(f"    Definition: {sub['definition'][:100]}...")
                                    
                                    # Niedrigere Schwelle für abduktiven Modus hervorheben
                                    if 0.6 <= sub['confidence'] < 0.7:
                                        print(f"    ⚠️ Konfidenz unter Standardschwelle, aber akzeptabel im abduktiven Modus")
                            else:
                                for sub in subcats:
                                    print(f"  • {sub['name']} (Konfidenz: {sub['confidence']:.2f})")
            
            # Nur im vollen Modus: Ausgabe neuer Hauptkategorien
            if analysis_mode == 'full' and 'new_categories' in result:
                print("\nNeue Hauptkategorien:")
                for new_cat in result['new_categories']:
                    print(f"- {new_cat['name']} (Konfidenz: {new_cat['confidence']:.2f})")
                    if 'subcategories' in new_cat:
                        print("  Subkategorien:")
                        for sub in new_cat['subcategories']:
                            print(f"  • {sub['name']}")
            
            # Für abduktiven Modus: Hinweis dass keine neuen Hauptkategorien entwickelt werden
            elif analysis_mode == 'abductive' and 'new_categories' in result:
                print("\nHinweis: Im abduktiven Modus werden keine neuen Hauptkategorien hinzugefügt,")
                print("selbst wenn potenzielle neue Kategorien identifiziert wurden.")

            # Sättigungsmetriken extrahieren und speichern wenn vorhanden
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
            
                # Speichere Metriken in der History
                self.history.log_saturation_check(
                    material_percentage=material_percentage,
                    result="saturated" if result['saturation_metrics']['categories_sufficient'] else "not_saturated",
                    metrics=result['saturation_metrics']
                )
                
                # Gebe Sättigungsinfo in Konsole aus
                saturation_info = result['saturation_metrics']
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

            return result

        except Exception as e:
            print(f"Fehler bei Kategorienanalyse: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return None

    async def analyze_grounded_batch(self, segments: List[str], material_percentage: float) -> Dict[str, Any]:
        """
        Analysiert einen Batch von Segmenten im 'grounded' Modus.
        Extrahiert Subcodes und Keywords ohne direkte Zuordnung zu Hauptkategorien.
        Sorgt für angemessenen Abstand zwischen Keywords und Subcodes.
        
        Args:
            segments: Liste der Textsegmente
            material_percentage: Prozentsatz des verarbeiteten Materials
            
        Returns:
            Dict[str, Any]: Analyseergebnisse mit Subcodes und Keywords
        """
        try:
            # Cache-Key erstellen
            cache_key = (
                tuple(segments),
                'grounded'
            )
            
            # Prüfe Cache
            if cache_key in self.analysis_cache:
                print("Nutze gecachte Analyse")
                return self.analysis_cache[cache_key]

            # Bestehende Subcodes sammeln
            existing_subcodes = []
            if hasattr(self, 'collected_subcodes'):
                existing_subcodes = [sc.get('name', '') for sc in self.collected_subcodes if isinstance(sc, dict)]
            
            # Definiere JSON-Schema für den grounded Modus
            json_schema = '''{
                "segment_analyses": [
                    {
                        "segment_text": "Textsegment",
                        "subcodes": [
                            {
                                "name": "Subcode-Name",
                                "definition": "Definition des Subcodes",
                                "evidence": ["Textbelege"],
                                "keywords": ["Schlüsselwörter des Subcodes"],
                                "confidence": 0.0-1.0
                            }
                        ],
                        "memo": "Analytische Notizen zum Segment"
                    }
                ],
                "abstraction_quality": {
                    "keyword_subcode_distinction": 0.0-1.0,
                    "comment": "Bewertung der Abstraktionshierarchie"
                },
                "saturation_metrics": {
                    "new_aspects_found": true/false,
                    "coverage": 0.0-1.0,
                    "justification": "Begründung"
                }
            }'''

            # Verbesserter Prompt mit Fokus auf die Abstraktionshierarchie
            prompt = self.prompt_handler.get_grounded_analysis_prompt(
                segments=segments,
                existing_subcodes=existing_subcodes,
                json_schema=json_schema
            )
            
            input_tokens = estimate_tokens(prompt)

            # API-Call
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
            
            # Bewertung der Abstraktionsqualität
            abstraction_quality = result.get('abstraction_quality', {})
            if abstraction_quality and 'keyword_subcode_distinction' in abstraction_quality:
                quality_score = abstraction_quality['keyword_subcode_distinction']
                quality_comment = abstraction_quality.get('comment', '')
                print(f"\nAbstraktionsqualität: {quality_score:.2f}/1.0")
                print(f"Kommentar: {quality_comment}")
            
            # Debug-Ausgabe und verbesserte Fortschrittsanzeige
            segment_count = len(result.get('segment_analyses', []))
            
            # Zähle Subcodes und ihre Keywords
            subcode_count = 0
            keyword_count = 0
            new_subcodes = []
            
            for analysis in result.get('segment_analyses', []):
                subcodes = analysis.get('subcodes', [])
                subcode_count += len(subcodes)
                
                for subcode in subcodes:
                    new_subcodes.append(subcode)
                    keyword_count += len(subcode.get('keywords', []))
                    
                    # Zeige Abstraktionsbeispiele für besseres Monitoring
                    keywords = subcode.get('keywords', [])
                    if keywords and len(keywords) > 0:
                        print(f"\nAbstraktionsbeispiel:")
                        print(f"Keywords: {', '.join(keywords[:3])}" + ("..." if len(keywords) > 3 else ""))
                        print(f"Subcode: {subcode.get('name', '')}")
            
            # Erweiterte Fortschrittsanzeige
            print(f"\nGrounded Analyse für {segment_count} Segmente abgeschlossen:")
            print(f"- {subcode_count} neue Subcodes identifiziert")
            print(f"- {keyword_count} Keywords mit Subcodes verknüpft")
            print(f"- Material-Fortschritt: {material_percentage:.1f}%")
            
            # Progress Bar für Gesamtfortschritt der Subcode-Sammlung
            if hasattr(self, 'collected_subcodes'):
                total_collected = len(self.collected_subcodes) + subcode_count
                # Einfache ASCII Progress Bar
                bar_length = 30
                filled_length = int(bar_length * material_percentage / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"\nGesamtfortschritt Grounded-Analyse:")
                print(f"[{bar}] {material_percentage:.1f}%")
                print(f"Bisher gesammelt: {total_collected} Subcodes mit ihren Keywords")
            
            return result
            
        except Exception as e:
            print(f"Fehler bei Grounded-Analyse: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return {}
    
    async def _generate_main_categories_from_subcodes(self, initial_categories: Dict[str, CategoryDefinition] = None) -> Dict[str, CategoryDefinition]:
        """
        Generiert Hauptkategorien aus den gesammelten Subcodes und ihren Keywords.
        Wird nach Abschluss der Verarbeitung aller Segmente im 'grounded' Modus aufgerufen.
        Verbessert für die Nachverfolgung der Subcode-Zuordnung zu Hauptkategorien.
        
        Args:
            initial_categories: Initiale Kategorien (bei grounded typischerweise leer)
            
        Returns:
            Dict[str, CategoryDefinition]: Generierte Hauptkategorien
        """
        try:
            # Prüfe, ob genügend Subcodes gesammelt wurden
            if not hasattr(self, 'collected_subcodes') or len(self.collected_subcodes) < 5:
                print(f"\n⚠️ Zu wenige Subcodes für Kategorienbildung. Mindestens 5 Subcodes benötigt.")
                return initial_categories or {}

            print(f"\n🔍 Generiere Hauptkategorien aus {len(self.collected_subcodes)} Subcodes mit ihren Keywords...")
            
            # Bereite die Daten für die Analyse vor - WICHTIG: Keywords bleiben mit Subcodes verknüpft
            subcodes_data = []
            for subcode in self.collected_subcodes:
                if isinstance(subcode, dict) and 'name' in subcode and 'definition' in subcode:
                    subcodes_data.append({
                        'name': subcode['name'],
                        'definition': subcode['definition'],
                        'keywords': subcode.get('keywords', []),
                        'evidence': subcode.get('evidence', []),
                        'confidence': subcode.get('confidence', 0.7)
                    })
            
            # Fortschrittsanzeige
            print("\n📊 Subcode-Statistik für Hauptkategorienbildung:")
            confidence_values = [s.get('confidence', 0) for s in subcodes_data]
            avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
            keyword_counts = [len(s.get('keywords', [])) for s in subcodes_data]
            avg_keywords = sum(keyword_counts) / len(keyword_counts) if keyword_counts else 0
            
            print(f"- {len(subcodes_data)} Subcodes mit durchschnittlich {avg_keywords:.1f} Keywords pro Subcode")
            print(f"- Durchschnittliche Konfidenz: {avg_confidence:.2f}")
            
            # Zeige die häufigsten Keywords zur Übersicht
            all_keywords = [kw for s in subcodes_data for kw in s.get('keywords', [])]
            keyword_counter = Counter(all_keywords)
            top_keywords = keyword_counter.most_common(10)
            
            print("\n🔑 Häufigste Keywords (Top 10):")
            for kw, count in top_keywords:
                print(f"- {kw}: {count}x")
            
            # Erstelle ein finales Kategoriensystem
            prompt = self.prompt_handler.get_main_categories_generation_prompt(
                subcodes_data=subcodes_data
            )
            
            # Zeige Fortschrittsanzeige
            print("\n⏳ Generiere Hauptkategorien aus den gesammelten Subcodes und Keywords...")
            print("Dies kann einen Moment dauern, da die Gruppierung komplex ist.")
            
            # Zeige "Animation" für Verarbeitung
            progress_chars = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
            for _ in range(3):
                for char in progress_chars:
                    sys.stdout.write(f"\r{char} Analysiere Subcodes und Keywords...   ")
                    sys.stdout.flush()
                    await asyncio.sleep(0.1)

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
            
            # Konvertiere das Ergebnis in CategoryDefinition-Objekte
            grounded_categories = {}
            
            print("\n✅ Hauptkategorien erfolgreich generiert:")
            print(f"- {len(result.get('main_categories', []))} Hauptkategorien erstellt")
            
            # Speichere das Mapping von Subcodes zu Hauptkategorien
            self.subcode_to_main_mapping = result.get('subcode_mappings', {})
            print(f"- {len(self.subcode_to_main_mapping)} Subcodes zu Hauptkategorien zugeordnet")
            
            # Meta-Analyse Informationen
            meta = result.get('meta_analysis', {})
            if meta:
                print("\n📊 Meta-Analyse des generierten Kategoriensystems:")
                print(f"- Theoretische Sättigung: {meta.get('theoretical_saturation', 0):.2f}")
                print(f"- Abdeckung: {meta.get('coverage', 0):.2f}")
                print(f"- Theoretische Dichte: {meta.get('theoretical_density', 0):.2f}")
                print(f"- Begründung: {meta.get('justification', '')}")
            
            for i, category in enumerate(result.get('main_categories', []), 1):
                name = category.get('name', '')
                definition = category.get('definition', '')
                examples = category.get('examples', [])
                rules = category.get('rules', [])
                char_keywords = category.get('characteristic_keywords', [])
                
                # Erstelle Subcategorien-Dictionary
                subcategories = {}
                subcodes = category.get('subcodes', [])
                for subcode in subcodes:
                    subcode_name = subcode.get('name', '')
                    subcode_definition = subcode.get('definition', '')
                    if subcode_name:
                        subcategories[subcode_name] = subcode_definition
                
                # Detaillierte Ausgabe für jede Hauptkategorie
                print(f"\n📁 {i}. Hauptkategorie: {name}")
                print(f"- Definition: {definition[:100]}..." if len(definition) > 100 else f"- Definition: {definition}")
                print(f"- Charakteristische Keywords: {', '.join(char_keywords)}")
                print(f"- {len(subcategories)} Subcodes zugeordnet")
                
                # Zeige die zugeordneten Subcodes
                if subcategories:
                    print("  Zugeordnete Subcodes:")
                    for j, (subname, subdefinition) in enumerate(subcategories.items(), 1):
                        if j <= 3 or len(subcategories) <= 5:  # Zeige maximal 3 Subcodes oder alle wenn weniger als 5
                            print(f"  {j}. {subname}")
                    if len(subcategories) > 5:
                        print(f"  ... und {len(subcategories) - 3} weitere")
                
                # Erstelle CategoryDefinition
                if name and definition:
                    grounded_categories[name] = CategoryDefinition(
                        name=name,
                        definition=definition,
                        examples=examples,
                        rules=rules,
                        subcategories=subcategories,
                        added_date=datetime.now().strftime("%Y-%m-%d"),
                        modified_date=datetime.now().strftime("%Y-%m-%d")
                    )
            
            # Prüfe ob alle Subcodes zugeordnet wurden
            all_subcodes = set(s.get('name', '') for s in subcodes_data)
            mapped_subcodes = set(self.subcode_to_main_mapping.keys())
            unmapped_subcodes = all_subcodes - mapped_subcodes
            
            if unmapped_subcodes:
                print(f"\n⚠️ {len(unmapped_subcodes)} Subcodes wurden nicht zugeordnet!")
                print("Nicht zugeordnete Subcodes:")
                for subcode in unmapped_subcodes:
                    print(f"- {subcode}")
                print("Diese Subcodes werden keiner Hauptkategorie zugeordnet.")
            
            # Kombiniere die generierten Kategorien mit den initialen, falls vorhanden
            if initial_categories:
                combined_categories = initial_categories.copy()
                for name, category in grounded_categories.items():
                    combined_categories[name] = category
                return combined_categories
            
            return grounded_categories
            
        except Exception as e:
            print(f"Fehler bei der Hauptkategoriengenerierung: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return initial_categories or {}
    
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
        prompt = self.prompt_handler.get_segment_relevance_assessment_prompt(segment)
                
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
                token_counter.add_tokens(input_tokens,output_tokens)
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
        return self.prompt_handler.get_category_extraction_prompt(segment)
        

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
        KORRIGIERT: Berücksichtigt Mehrfachkodierung korrekt - Kodierer stimmen überein,
        wenn sie dieselben Kategorien identifizieren (auch wenn in verschiedenen Instanzen).
        
        Args:
            codings: Liste der Kodierungen
                
        Returns:
            float: Krippendorffs Alpha (-1 bis 1)
        """
        try:
            print(f"\nBerechne Intercoder-Reliabilität für {len(codings)} Kodierungen...")
            
            # 1. FILTER: Nur ursprüngliche Kodierungen für Reliabilität verwenden
            original_codings = []
            review_count = 0
            consolidated_count = 0
            
            for coding in codings:
                # Überspringe manuelle Review-Entscheidungen
                if coding.get('manual_review', False):
                    review_count += 1
                    continue
                    
                # Überspringe konsolidierte Kodierungen
                if coding.get('consolidated_from_multiple', False):
                    consolidated_count += 1
                    continue
                    
                # Überspringe Kodierungen ohne echten Kodierer
                coder_id = coding.get('coder_id', '')
                if not coder_id or coder_id in ['consensus', 'majority', 'review']:
                    continue
                    
                original_codings.append(coding)
            
            print(f"Gefilterte Kodierungen:")
            print(f"- Ursprüngliche Kodierungen: {len(original_codings)}")
            print(f"- Review-Entscheidungen übersprungen: {review_count}")
            print(f"- Konsolidierte Kodierungen übersprungen: {consolidated_count}")
            
            if len(original_codings) < 2:
                print("Warnung: Weniger als 2 ursprüngliche Kodierungen - keine Reliabilität berechenbar")
                return 1.0
            
            # 2. GRUPPIERUNG: Nach BASIS-Segmenten (nicht nach Instanzen!)
            base_segment_codings = defaultdict(list)
            
            for coding in original_codings:
                segment_id = coding.get('segment_id', '')
                if not segment_id:
                    continue
                
                # Extrahiere Basis-Segment-ID (ohne Instanz-Suffix)
                base_segment_id = coding.get('Original_Chunk_ID', segment_id)
                if not base_segment_id:
                    # Fallback: Entferne mögliche Instanz-Suffixe
                    if '_inst_' in segment_id:
                        base_segment_id = segment_id.split('_inst_')[0]
                    elif segment_id.endswith('-1') or segment_id.endswith('-2'):
                        base_segment_id = segment_id.rsplit('-', 1)[0]
                    else:
                        base_segment_id = segment_id
                
                base_segment_codings[base_segment_id].append(coding)
            
            print(f"Basis-Segmente: {len(base_segment_codings)}")
            
            # 3. FÜR JEDES BASIS-SEGMENT: Sammle alle Kategorien pro Kodierer
            comparable_segments = []
            single_coder_segments = 0
            
            for base_segment_id, segment_codings in base_segment_codings.items():
                # Gruppiere nach Kodierern
                coder_categories = defaultdict(set)
                coder_subcategories = defaultdict(set)
                
                for coding in segment_codings:
                    coder_id = coding.get('coder_id', '')
                    category = coding.get('category', '')
                    subcats = coding.get('subcategories', [])
                    
                    if category and category not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                        coder_categories[coder_id].add(category)
                    
                    # Subkategorien sammeln
                    if isinstance(subcats, (list, tuple)):
                        coder_subcategories[coder_id].update(subcats)
                    elif subcats:
                        coder_subcategories[coder_id].add(str(subcats))
                
                # Nur Segmente mit mindestens 2 Kodierern berücksichtigen
                if len(coder_categories) < 2:
                    single_coder_segments += 1
                    continue
                
                comparable_segments.append({
                    'base_segment_id': base_segment_id,
                    'coder_categories': dict(coder_categories),
                    'coder_subcategories': dict(coder_subcategories),
                    'sample_text': segment_codings[0].get('text', '')[:200] + '...'
                })
            
            print(f"Vergleichbare Basis-Segmente: {len(comparable_segments)}")
            print(f"Einzelkodierer-Segmente übersprungen: {single_coder_segments}")
            
            if len(comparable_segments) == 0:
                print("Warnung: Keine vergleichbaren Segmente gefunden")
                return 1.0
            
            # 4. BERECHNE ÜBEREINSTIMMUNGEN
            # Für Hauptkategorien: Übereinstimmung wenn Kodierer dieselben Kategorien-Sets haben
            # Für Subkategorien: Übereinstimmung wenn Kodierer dieselben Subkategorien-Sets haben
            
            main_agreements = 0
            sub_agreements = 0
            sub_comparable = 0
            
            disagreement_examples = []
            
            for segment_data in comparable_segments:
                coder_categories = segment_data['coder_categories']
                coder_subcategories = segment_data['coder_subcategories']
                
                # HAUPTKATEGORIEN-VERGLEICH
                coders = list(coder_categories.keys())
                if len(coders) >= 2:
                    # Vergleiche alle Kodierer-Paare
                    segment_agreements = 0
                    segment_comparisons = 0
                    
                    for i in range(len(coders)):
                        for j in range(i + 1, len(coders)):
                            coder1, coder2 = coders[i], coders[j]
                            cats1 = coder_categories[coder1]
                            cats2 = coder_categories[coder2]
                            
                            segment_comparisons += 1
                            
                            # Übereinstimmung wenn beide dieselben Kategorien identifiziert haben
                            if cats1 == cats2:
                                segment_agreements += 1
                            else:
                                # Sammle Unstimmigkeiten für Debugging
                                if len(disagreement_examples) < 5:
                                    disagreement_examples.append({
                                        'segment': segment_data['base_segment_id'],
                                        'coder1': coder1,
                                        'coder2': coder2,
                                        'cats1': list(cats1),
                                        'cats2': list(cats2),
                                        'subcats1': list(coder_subcategories.get(coder1, set())),
                                        'subcats2': list(coder_subcategories.get(coder2, set())),
                                        'text': segment_data['sample_text']
                                    })
                    
                    # Segment gilt als übereinstimmend wenn alle Paare übereinstimmen
                    if segment_agreements == segment_comparisons:
                        main_agreements += 1
                
                # SUBKATEGORIEN-VERGLEICH
                # Nur analysieren wenn mindestens ein Kodierer Subkategorien hat
                if any(len(subcats) > 0 for subcats in coder_subcategories.values()):
                    sub_comparable += 1
                    
                    # Vergleiche Subkategorien-Sets
                    subcat_sets = list(coder_subcategories.values())
                    if len(set(frozenset(s) for s in subcat_sets)) == 1:
                        sub_agreements += 1
            
            # 5. BERECHNE ÜBEREINSTIMMUNGSRATEN
            main_agreement_rate = main_agreements / len(comparable_segments) if comparable_segments else 0
            sub_agreement_rate = sub_agreements / sub_comparable if sub_comparable > 0 else 1.0
            
            print(f"\nReliabilitäts-Details:")
            print(f"Hauptkategorien:")
            print(f"- Basis-Segmente analysiert: {len(comparable_segments)}")
            print(f"- Vollständige Übereinstimmungen: {main_agreements}")
            print(f"- Übereinstimmungsrate: {main_agreement_rate:.3f}")
            
            print(f"Subkategorien:")
            print(f"- Vergleichbare Segmente: {sub_comparable}")
            print(f"- Vollständige Übereinstimmungen: {sub_agreements}")
            print(f"- Übereinstimmungsrate: {sub_agreement_rate:.3f}")
            
            # Zeige Beispiele für Unstimmigkeiten
            if disagreement_examples:
                print(f"\nBeispiele für Hauptkategorien-Unstimmigkeiten:")
                for i, example in enumerate(disagreement_examples, 1):
                    print(f"{i}. Basis-Segment {example['segment']}:")
                    print(f"   {example['coder1']}: {example['cats1']}")
                    print(f"   {example['coder2']}: {example['cats2']}")
                    print(f"   Text: {example['text'][:100]}...")
            
            # 6. KRIPPENDORFFS ALPHA BERECHNUNG
            observed_agreement = main_agreement_rate
            
            # Sammle alle Kategorien für erwartete Zufallsübereinstimmung
            all_categories = []
            for segment_data in comparable_segments:
                for coder_cats in segment_data['coder_categories'].values():
                    all_categories.extend(list(coder_cats))
            
            # Berechne erwartete Zufallsübereinstimmung
            category_frequencies = Counter(all_categories)
            total_category_instances = len(all_categories)
            
            expected_agreement = 0
            if total_category_instances > 1:
                for count in category_frequencies.values():
                    prob = count / total_category_instances
                    expected_agreement += prob ** 2
            
            # Krippendorffs Alpha
            if expected_agreement >= 1.0:
                alpha = 1.0
            else:
                alpha = (observed_agreement - expected_agreement) / (1 - expected_agreement)
            
            print(f"\nKrippendorffs Alpha Berechnung:")
            print(f"- Beobachtete Übereinstimmung: {observed_agreement:.3f}")
            print(f"- Erwartete Zufallsübereinstimmung: {expected_agreement:.3f}")
            print(f"- Krippendorffs Alpha: {alpha:.3f}")
            
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
        self.is_last_segment = False  # Neue Variable zur Identifizierung des letzten Segments
        
    async def code_chunk(self, chunk: str, categories: Optional[Dict[str, CategoryDefinition]], is_last_segment: bool = False) -> Optional[CodingResult]:
        """
        Nutzt das asyncio Event Loop, um tkinter korrekt zu starten.
        
        Args:
            chunk: Zu kodierender Text
            categories: Kategoriensystem
            is_last_segment: Gibt an, ob dies das letzte zu kodierende Segment ist
            
        Returns:
            Optional[CodingResult]: Kodierungsergebnis oder None bei Fehler
        """
        try:
            self.categories = self.current_categories or categories
            self.current_coding = None
            self.is_last_segment = is_last_segment  # Speichere Information, ob letztes Segment
            
            # Erstelle und starte das Tkinter-Fenster im Hauptthread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_tk_window, chunk)
            
            # Nach dem Beenden des Fensters
            try:
                # Stelle sicher, dass das Fenster vollständig geschlossen wird
                if self.root:
                    self.root.destroy()
                    self.root = None
            except Exception as e:
                print(f"Warnung: Fehler beim Schließen des Fensters: {str(e)}")
                # Ignorieren, da wir das Fenster trotzdem schließen wollen
            
            # Wichtig: Text zum Kodierungsergebnis hinzufügen
            if self.current_coding:
                # Text für das CodingResult hinzufügen, falls es nicht bereits enthalten ist
                # Wir erstellen ein neues CodingResult mit dem Text
                text_references = list(self.current_coding.text_references)
                if chunk not in text_references:
                    text_references.append(chunk)
                
                enhanced_coding = CodingResult(
                    category=self.current_coding.category,
                    subcategories=self.current_coding.subcategories,
                    justification=self.current_coding.justification,
                    confidence=self.current_coding.confidence,
                    text_references=tuple(text_references),
                    uncertainties=self.current_coding.uncertainties,
                    paraphrase=self.current_coding.paraphrase,
                    keywords=self.current_coding.keywords
                )
                
                # Aktualisiere das current_coding Attribut
                self.current_coding = enhanced_coding
                
            # Debug-Ausgabe
            result_status = "Kodierung erstellt" if self.current_coding else "Keine Kodierung"
            print(f"ManualCoder Ergebnis: {result_status}")
            
            # Abschließende Bereinigung aller Tkinter-Ressourcen
            self._cleanup_tkinter_resources()
            
            return self.current_coding
            
        except Exception as e:
            print(f"Fehler in code_chunk: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()
            
            # Stelle sicher, dass das Fenster auch bei Fehlern geschlossen wird
            if self.root:
                try:
                    self.root.destroy()
                    self.root = None
                except:
                    pass
                    
            # Bereinigung bei Fehlern
            self._cleanup_tkinter_resources()
                
            return None

    def _run_tk_window(self, chunk: str):
        """Führt das Tkinter-Fenster im Hauptthread aus mit verbessertem Handling des letzten Segments"""
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
            # Header mit Fortschrittsinformation
            header_frame = ttk.Frame(self.root)
            header_frame.pack(padx=10, pady=5, fill=tk.X)
            
            # Wenn letztes Segment, zeige Hinweis an
            if self.is_last_segment:
                last_segment_label = ttk.Label(
                    header_frame, 
                    text="DIES IST DAS LETZTE SEGMENT",
                    font=('Arial', 10, 'bold'),
                    foreground='red'
                )
                last_segment_label.pack(side=tk.TOP, padx=5, pady=5)
            

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

            if self.is_last_segment:
                ttk.Button(
                    button_frame, 
                    text="Kodieren & Abschließen", 
                    command=self._safe_finish_coding
                ).pack(side=tk.LEFT, padx=5)
            else:
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

    def _cleanup_tkinter_resources(self):
        """Bereinigt alle Tkinter-Ressourcen"""
        try:
            # Entferne alle Tkinter-Variablen
            for attr_name in list(self.__dict__.keys()):
                if attr_name.startswith('_tk_var_'):
                    delattr(self, attr_name)
                    
            # Setze Fenster-Referenzen auf None
            self.text_chunk = None
            self.category_listbox = None
            
            # Sammle Garbage-Collection
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Warnung: Fehler bei der Bereinigung von Tkinter-Ressourcen: {str(e)}")

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
        """Sicheres Schließen des Fensters mit vollständiger Ressourcenfreigabe"""
        try:
            if messagebox.askokcancel("Beenden", "Möchten Sie das Kodieren wirklich beenden?"):
                self.current_coding = None
                self._is_processing = False
                
                # Alle Tkinter-Variablen explizit löschen
                if hasattr(self, 'root') and self.root:
                    for attr_name in dir(self):
                        attr = getattr(self, attr_name) 
                        # Prüfen, ob es sich um eine Tkinter-Variable handelt
                        if hasattr(attr, '_tk'):
                            delattr(self, attr_name)
                    
                    # Fenster schließen
                    try:
                        self.root.quit()
                        self.root.destroy()
                        self.root = None  # Referenz entfernen
                    except:
                        pass
        except:
            # Stelle sicher, dass Fenster auch bei Fehlern geschlossen wird
            if hasattr(self, 'root') and self.root:
                try:
                    self.root.quit()
                    self.root.destroy()
                    self.root = None
                except:
                    pass

    def _safe_code_selection(self):
        """Thread-sichere Kodierungsauswahl mit korrekter Kategoriezuordnung"""
        if not self._is_processing:
            try:
                # Flag setzen, um mehrfache Verarbeitung zu verhindern
                self._is_processing = True
                
                selection = self.category_listbox.curselection()
                if not selection:
                    messagebox.showwarning("Warnung", "Bitte wählen Sie eine Kategorie aus.")
                    self._is_processing = False
                    return
                
                index = selection[0]
                
                # Hole die tatsächliche Kategorie aus dem Mapping
                if index not in self.category_map:
                    messagebox.showerror("Fehler", "Ungültiger Kategorieindex")
                    self._is_processing = False
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
                    self._is_processing = False
                    return
                    
                if sub_cat and sub_cat not in self.categories[main_cat].subcategories:
                    messagebox.showerror("Fehler", 
                        f"Subkategorie '{sub_cat}' nicht in '{main_cat}' gefunden.\n"
                        f"Verfügbare Subkategorien: {', '.join(self.categories[main_cat].subcategories.keys())}")
                    self._is_processing = False
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
                    
                # Bei letztem Segment Hinweis anzeigen
                if self.is_last_segment:
                    messagebox.showinfo("Kodierung abgeschlossen", 
                                    "Die Kodierung des letzten Segments wurde abgeschlossen.\n"
                                    "Der manuelle Kodierungsprozess wird beendet.")
                
                # Setze das Flag zurück
                self._is_processing = False
                
                # Dann Fenster schließen - in dieser Reihenfolge!
                if self.root:
                    try:
                        # Fenster schließen (wichtig: destroy vor quit)
                        self.root.destroy()
                        self.root.quit()
                    except Exception as e:
                        print(f"Fehler beim Schließen des Fensters: {str(e)}")
                        
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler bei der Kategorieauswahl: {str(e)}")
                print(f"Fehler bei der Kategorieauswahl: {str(e)}")
                print("Details:")
                import traceback
                traceback.print_exc()
                
                # Setze das Flag zurück
                self._is_processing = False
                

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
            
            # Bei letztem Segment Hinweis anzeigen
            if self.is_last_segment:
                messagebox.showinfo("Kodierung abgeschlossen", 
                                   "Die Kodierung des letzten Segments wurde übersprungen.\n"
                                   "Der manuelle Kodierungsprozess wird beendet.")
            
            self.root.quit()

    def _safe_finish_coding(self):
        """Thread-sicherer Abschluss der Kodierung (nur für letztes Segment)"""
        if not self._is_processing and self.is_last_segment:
            if messagebox.askyesno("Segment kodieren und abschließen", 
                "Möchten Sie das aktuelle Segment kodieren und den manuellen Kodierungsprozess abschließen?"):
                
                # Zuerst die normale Kodierung durchführen
                selection = self.category_listbox.curselection()
                if not selection:
                    messagebox.showwarning("Warnung", "Bitte wählen Sie eine Kategorie aus.")
                    return
                
                # Die gleiche Logik wie in _safe_code_selection verwenden
                index = selection[0]
                
                # Hole die tatsächliche Kategorie aus dem Mapping
                if index not in self.category_map:
                    messagebox.showerror("Fehler", "Ungültiger Kategorieindex")
                    return
                    
                category_info = self.category_map[index]
                
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

                # Erstelle Kodierung mit der gewählten Kategorie
                self.current_coding = CodingResult(
                    category=main_cat,
                    subcategories=(sub_cat,) if sub_cat else tuple(),
                    justification="Manuelle Kodierung (Abschluss)",
                    confidence={'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                    text_references=(self.text_chunk.get("1.0", tk.END)[:100],)
                )
                
                self._is_processing = False
                
                messagebox.showinfo("Kodierung abgeschlossen", 
                                f"Das Segment wurde als '{main_cat}' kodiert.\n"
                                "Der manuelle Kodierungsprozess wird beendet.")
                
                # Explizit alle Ressourcen freigeben und Fenster schließen
                for attr_name in dir(self):
                    attr = getattr(self, attr_name)
                    # Prüfen, ob es sich um eine Tkinter-Variable handelt
                    if hasattr(attr, '_tk'):
                        delattr(self, attr_name)
                
                # Sicherstellen, dass das Fenster wirklich geschlossen wird
                try:
                    # Wichtig: Erst destroy aufrufen, dann quit
                    self.root.destroy()
                    self.root.quit()
                except:
                    pass
                
                # Sicherstellen, dass _is_processing zurückgesetzt wird
                self._is_processing = False

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


class ManualReviewComponent:
    """
    Komponente für die manuelle Überprüfung und Entscheidung bei Kodierungsunstimmigkeiten.
    Zeigt dem Benutzer Textstellen mit abweichenden Kodierungen und lässt ihn die finale Entscheidung treffen.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialisiert die Manual Review Komponente.
        
        Args:
            output_dir (str): Verzeichnis für Export-Dokumente
        """
        self.output_dir = output_dir
        self.root = None
        self.review_results = []
        self.current_segment = None
        self.current_codings = None
        self.current_index = 0
        self.total_segments = 0
        self._is_processing = False
        
        # Import tkinter innerhalb der Methode, um Abhängigkeiten zu reduzieren
        self.tk = None
        self.ttk = None
        
    async def review_discrepancies(self, segment_codings: dict) -> list:
        """
        Führt einen manuellen Review-Prozess für Segmente mit abweichenden Kodierungen durch.
        
        Args:
            segment_codings: Dictionary mit Segment-ID als Schlüssel und Liste von Kodierungen als Werte
            
        Returns:
            list: Liste der finalen Kodierungsentscheidungen
        """
        try:
            # Importiere tkinter bei Bedarf
            import tkinter as tk
            from tkinter import ttk
            self.tk = tk
            self.ttk = ttk
            
            print("\n=== Manuelle Überprüfung von Kodierungsunstimmigkeiten ===")
            
            # Identifiziere Segmente mit abweichenden Kodierungen
            discrepant_segments = self._identify_discrepancies(segment_codings)
            
            if not discrepant_segments:
                print("Keine Unstimmigkeiten gefunden. Manueller Review nicht erforderlich.")
                return []
                
            self.total_segments = len(discrepant_segments)
            print(f"\nGefunden: {self.total_segments} Segmente mit Kodierungsabweichungen")
            
            # Starte das Tkinter-Fenster für den manuellen Review
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_review_gui, discrepant_segments)
            
            print(f"\nManueller Review abgeschlossen: {len(self.review_results)} Entscheidungen getroffen")
            
            return self.review_results
            
        except Exception as e:
            print(f"Fehler beim manuellen Review: {str(e)}")
            import traceback
            traceback.print_exc()
            if self.root:
                try:
                    self.root.quit()
                    self.root.destroy()
                except:
                    pass
            return []
    
    def _identify_discrepancies(self, segment_codings: dict) -> list:
        """
        Identifiziert Segmente, bei denen verschiedene Kodierer zu unterschiedlichen Ergebnissen kommen.
        
        Args:
            segment_codings: Dictionary mit Segment-ID als Schlüssel und Liste von Kodierungen als Werte
            
        Returns:
            list: Liste von Tuples (segment_id, text, codings) mit Unstimmigkeiten
        """
        discrepancies = []
        
        for segment_id, codings in segment_codings.items():
            # Ignoriere Segmente mit nur einer Kodierung
            if len(codings) <= 1:
                continue
                
            # Prüfe auf Unstimmigkeiten in Hauptkategorien
            categories = set(coding.get('category', '') for coding in codings)
            
            # Bei REVIEW_MODE 'manual' auch automatische Kodierungen prüfen
            if len(categories) > 1:
                # Hole den Text des Segments
                text = codings[0].get('text', '')
                if not text:
                    # Alternative Textquelle, falls 'text' nicht direkt verfügbar
                    text = codings[0].get('text_references', [''])[0] if codings[0].get('text_references') else ''
                
                discrepancies.append((segment_id, text, codings))
                
        print(f"Unstimmigkeiten identifiziert: {len(discrepancies)}/{len(segment_codings)} Segmente")
        return discrepancies
    
    def _run_review_gui(self, discrepant_segments: list):
        """
        Führt die grafische Benutzeroberfläche für den manuellen Review aus.
        
        Args:
            discrepant_segments: Liste von Segmenten mit Unstimmigkeiten
        """
        if self.root is not None:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
                
        self.root = self.tk.Tk()
        self.root.title("QCA-AID Manueller Review")
        self.root.geometry("1000x700")
        
        # Protokoll für das Schließen des Fensters
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Hauptframe
        main_frame = self.ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=self.tk.BOTH, expand=True)
        
        # Fortschrittsanzeige
        progress_frame = self.ttk.Frame(main_frame)
        progress_frame.pack(padx=5, pady=5, fill=self.tk.X)
        
        self.ttk.Label(progress_frame, text="Fortschritt:").pack(side=self.tk.LEFT, padx=5)
        progress_var = self.tk.StringVar()
        progress_var.set(f"Segment 1/{self.total_segments}")
        progress_label = self.ttk.Label(progress_frame, textvariable=progress_var)
        progress_label.pack(side=self.tk.LEFT, padx=5)
        
        # Text-Frame
        text_frame = self.ttk.LabelFrame(main_frame, text="Textsegment")
        text_frame.pack(padx=5, pady=5, fill=self.tk.BOTH, expand=True)
        
        text_widget = self.tk.Text(text_frame, height=10, wrap=self.tk.WORD)
        text_widget.pack(padx=5, pady=5, fill=self.tk.BOTH, expand=True)
        text_widget.config(state=self.tk.DISABLED)
        
        # Kodierungen-Frame
        codings_frame = self.ttk.LabelFrame(main_frame, text="Konkurrierende Kodierungen")
        codings_frame.pack(padx=5, pady=5, fill=self.tk.BOTH, expand=True)
        
        codings_canvas = self.tk.Canvas(codings_frame)
        scrollbar = self.ttk.Scrollbar(codings_frame, orient=self.tk.VERTICAL, command=codings_canvas.yview)
        
        codings_scrollable = self.ttk.Frame(codings_canvas)
        codings_scrollable.bind(
            "<Configure>",
            lambda e: codings_canvas.configure(
                scrollregion=codings_canvas.bbox("all")
            )
        )
        
        codings_canvas.create_window((0, 0), window=codings_scrollable, anchor="nw")
        codings_canvas.configure(yscrollcommand=scrollbar.set)
        
        codings_canvas.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        
        # Button-Frame
        button_frame = self.ttk.Frame(main_frame)
        button_frame.pack(padx=5, pady=10, fill=self.tk.X)
        
        self.ttk.Button(
            button_frame, 
            text="Vorheriges", 
            command=lambda: self._navigate(-1, text_widget, codings_scrollable, discrepant_segments, progress_var)
        ).pack(side=self.tk.LEFT, padx=5)
        
        self.ttk.Button(
            button_frame, 
            text="Nächstes", 
            command=lambda: self._navigate(1, text_widget, codings_scrollable, discrepant_segments, progress_var)
        ).pack(side=self.tk.LEFT, padx=5)
        
        self.ttk.Button(
            button_frame, 
            text="Abbrechen", 
            command=self._on_closing
        ).pack(side=self.tk.RIGHT, padx=5)
        
        # Begründung eingeben
        justification_frame = self.ttk.LabelFrame(main_frame, text="Begründung für Ihre Entscheidung")
        justification_frame.pack(padx=5, pady=5, fill=self.tk.X)
        
        justification_text = self.tk.Text(justification_frame, height=3, wrap=self.tk.WORD)
        justification_text.pack(padx=5, pady=5, fill=self.tk.X)
        
        # Initialisiere mit dem ersten Segment
        if discrepant_segments:
            self.current_index = 0
            self._update_display(text_widget, codings_scrollable, discrepant_segments, justification_text, progress_var)
        
        # Starte MainLoop
        self.root.mainloop()
    
    def _update_display(self, text_widget, codings_frame, discrepant_segments, justification_text, progress_var):
        """
        Aktualisiert die Anzeige für das aktuelle Segment.
        """
        # Aktualisiere Fortschrittsanzeige
        progress_var.set(f"Segment {self.current_index + 1}/{self.total_segments}")
        
        # Hole aktuelles Segment und Kodierungen
        segment_id, text, codings = discrepant_segments[self.current_index]
        self.current_segment = segment_id
        self.current_codings = codings
        
        # Setze Text
        text_widget.config(state=self.tk.NORMAL)
        text_widget.delete(1.0, self.tk.END)
        text_widget.insert(self.tk.END, text)
        text_widget.config(state=self.tk.DISABLED)
        
        # Begründungsfeld leeren
        justification_text.delete(1.0, self.tk.END)
        
        # Lösche alte Kodierungsoptionen
        for widget in codings_frame.winfo_children():
            widget.destroy()
            
        # Anzeige-Variable für die ausgewählte Kodierung
        selection_var = self.tk.StringVar()
        
        # Erstelle Radiobuttons für jede Kodierung
        for i, coding in enumerate(codings):
            coder_id = coding.get('coder_id', 'Unbekannt')
            category = coding.get('category', 'Keine Kategorie')
            subcategories = coding.get('subcategories', [])
            if isinstance(subcategories, tuple):
                subcategories = list(subcategories)
            confidence = 0.0
            
            # Extrahiere Konfidenzwert
            if isinstance(coding.get('confidence'), dict):
                confidence = coding['confidence'].get('total', 0.0)
            elif isinstance(coding.get('confidence'), (int, float)):
                confidence = float(coding['confidence'])
                
            # Formatiere die Subkategorien
            subcats_text = ', '.join(subcategories) if subcategories else 'Keine'
            
            # Erstelle Label-Text
            is_human = 'human' in coder_id
            coder_prefix = "[Mensch]" if is_human else "[Auto]"
            radio_text = f"{coder_prefix} {coder_id}: {category} ({confidence:.2f})\nSubkategorien: {subcats_text}"
            
            # Radiobutton mit Rahmen für bessere Sichtbarkeit
            coding_frame = self.ttk.Frame(codings_frame, relief=self.tk.GROOVE, borderwidth=2)
            coding_frame.pack(padx=5, pady=5, fill=self.tk.X)
            
            radio = self.ttk.Radiobutton(
                coding_frame,
                text=radio_text,
                variable=selection_var,
                value=str(i),
                command=lambda idx=i, j_text=justification_text: self._select_coding(idx, j_text)
            )
            radio.pack(padx=5, pady=5, anchor=self.tk.W)
            
            # Begründung anzeigen wenn vorhanden
            justification = coding.get('justification', '')
            if justification:
                just_label = self.ttk.Label(
                    coding_frame, 
                    text=f"Begründung: {justification[:150]}..." if len(justification) > 150 else f"Begründung: {justification}",
                    wraplength=500
                )
                just_label.pack(padx=5, pady=5, anchor=self.tk.W)
        
        # Eigene Kodierung als Option
        custom_frame = self.ttk.Frame(codings_frame, relief=self.tk.GROOVE, borderwidth=2)
        custom_frame.pack(padx=5, pady=5, fill=self.tk.X)
        
        custom_radio = self.ttk.Radiobutton(
            custom_frame,
            text="Eigene Entscheidung eingeben",
            variable=selection_var,
            value="custom",
            command=lambda: self._create_custom_coding(justification_text)
        )
        custom_radio.pack(padx=5, pady=5, anchor=self.tk.W)
        
        # Standardmäßig menschliche Kodierung auswählen, falls vorhanden
        for i, coding in enumerate(codings):
            if 'human' in coding.get('coder_id', ''):
                selection_var.set(str(i))
                self._select_coding(i, justification_text)
                break
    
    def _select_coding(self, coding_index, justification_text):
        """
        Ausgewählte Kodierung für das aktuelle Segment speichern.
        """
        self.selected_coding_index = coding_index
        
        # Hole die ausgewählte Kodierung
        selected_coding = self.current_codings[coding_index]
        
        # Fülle Begründung mit Vorschlag
        existing_just = selected_coding.get('justification', '')
        if existing_just:
            justification_text.delete(1.0, self.tk.END)
            justification_text.insert(self.tk.END, f"Übernommen von {selected_coding.get('coder_id', 'Kodierer')}: {existing_just}")
    
    def _create_custom_coding(self, justification_text):
        """
        Erstellt ein benutzerdefiniertes Kodierungsfenster.
        """
        custom_window = self.tk.Toplevel(self.root)
        custom_window.title("Eigene Kodierung")
        custom_window.geometry("600x500")
        
        input_frame = self.ttk.Frame(custom_window)
        input_frame.pack(padx=10, pady=10, fill=self.tk.BOTH, expand=True)
        
        # Hauptkategorie
        self.ttk.Label(input_frame, text="Hauptkategorie:").grid(row=0, column=0, padx=5, pady=5, sticky=self.tk.W)
        category_entry = self.ttk.Entry(input_frame, width=30)
        category_entry.grid(row=0, column=1, padx=5, pady=5, sticky=self.tk.W+self.tk.E)
        
        # Subkategorien
        self.ttk.Label(input_frame, text="Subkategorien (mit Komma getrennt):").grid(row=1, column=0, padx=5, pady=5, sticky=self.tk.W)
        subcats_entry = self.ttk.Entry(input_frame, width=30)
        subcats_entry.grid(row=1, column=1, padx=5, pady=5, sticky=self.tk.W+self.tk.E)
        
        # Begründung
        self.ttk.Label(input_frame, text="Begründung:").grid(row=2, column=0, padx=5, pady=5, sticky=self.tk.W)
        just_text = self.tk.Text(input_frame, height=5, width=30)
        just_text.grid(row=2, column=1, padx=5, pady=5, sticky=self.tk.W+self.tk.E)
        
        # Buttons
        button_frame = self.ttk.Frame(input_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.ttk.Button(
            button_frame, 
            text="Übernehmen",
            command=lambda: self._apply_custom_coding(
                category_entry.get(),
                subcats_entry.get(),
                just_text.get(1.0, self.tk.END),
                justification_text,
                custom_window
            )
        ).pack(side=self.tk.LEFT, padx=5)
        
        self.ttk.Button(
            button_frame,
            text="Abbrechen",
            command=custom_window.destroy
        ).pack(side=self.tk.LEFT, padx=5)
    
    def _apply_custom_coding(self, category, subcategories, justification, main_just_text, window):
        """
        Übernimmt die benutzerdefinierte Kodierung.
        """
        # Erstelle eine benutzerdefinierte Kodierung
        self.custom_coding = {
            'category': category,
            'subcategories': [s.strip() for s in subcategories.split(',') if s.strip()],
            'justification': justification.strip(),
            'coder_id': 'human_review',
            'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0}
        }
        
        # Aktualisiere das Begründungsfeld im Hauptfenster
        main_just_text.delete(1.0, self.tk.END)
        main_just_text.insert(self.tk.END, f"Eigene Entscheidung: {justification.strip()}")
        
        # Schließe das Fenster
        window.destroy()
    
    def _navigate(self, direction, text_widget, codings_frame, discrepant_segments, progress_var):
        """
        Navigation zwischen den Segmenten und Speicherung der Entscheidung.
        """
        if self.current_segment is None or self.current_codings is None:
            return
            
        # Speichere aktuelle Entscheidung
        self._save_current_decision(text_widget)
        
        # Berechne neuen Index
        new_index = self.current_index + direction
        
        # Prüfe Grenzen
        if 0 <= new_index < len(discrepant_segments):
            self.current_index = new_index
            self._update_display(text_widget, codings_frame, discrepant_segments, text_widget, progress_var)
        elif new_index >= len(discrepant_segments):
            # Wenn wir am Ende angelangt sind, frage nach Abschluss
            if self.tk.messagebox.askyesno(
                "Review abschließen", 
                "Das war das letzte Segment. Möchten Sie den Review abschließen?"
            ):
                self.root.quit()
    
    def _save_current_decision(self, justification_text):
        """
        Speichert die aktuelle Entscheidung.
        """
        try:
            if hasattr(self, 'selected_coding_index'):
                # Normale Kodierungsentscheidung
                selected_coding = self.current_codings[self.selected_coding_index].copy()
                
                # Hole Begründung aus Textfeld
                justification = justification_text.get(1.0, self.tk.END).strip()
                
                # Aktualisiere die Kodierung
                selected_coding['segment_id'] = self.current_segment
                selected_coding['review_justification'] = justification
                selected_coding['manual_review'] = True
                selected_coding['review_date'] = datetime.now().isoformat()
                
                self.review_results.append(selected_coding)
                print(f"Entscheidung für Segment {self.current_segment} gespeichert: {selected_coding['category']}")
                
            elif hasattr(self, 'custom_coding'):
                # Benutzerdefinierte Kodierung
                custom = self.custom_coding.copy()
                custom['segment_id'] = self.current_segment
                custom['manual_review'] = True
                custom['review_date'] = datetime.now().isoformat()
                
                self.review_results.append(custom)
                print(f"Eigene Entscheidung für Segment {self.current_segment} gespeichert: {custom['category']}")
        
        except Exception as e:
            print(f"Fehler beim Speichern der Entscheidung: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _on_closing(self):
        """Sicheres Schließen des Fensters mit vollständiger Ressourcenfreigabe"""
        try:
            if hasattr(self, 'root') and self.root:
                if self.tk.messagebox.askokcancel(
                    "Review beenden", 
                    "Möchten Sie den Review-Prozess wirklich beenden?\nGetroffene Entscheidungen werden gespeichert."
                ):
                    # Speichere aktuelle Entscheidung falls vorhanden
                    if self.current_segment is not None:
                        justification_text = None
                        for widget in self.root.winfo_children():
                            if isinstance(widget, self.tk.Text):
                                justification_text = widget
                                break
                        
                        if justification_text:
                            self._save_current_decision(justification_text)
                    
                    # Alle Tkinter-Variablen explizit löschen
                    for attr_name in dir(self):
                        attr = getattr(self, attr_name)
                        # Prüfen, ob es sich um eine Tkinter-Variable handelt
                        if hasattr(attr, '_tk'):
                            delattr(self, attr_name)
                    
                    # Fenster schließen
                    self.root.quit()
                    self.root.destroy()
                    self.root = None  # Wichtig: Referenz entfernen
        except:
            # Stelle sicher, dass Fenster auch bei Fehlern geschlossen wird
            if hasattr(self, 'root') and self.root:
                try:
                    self.root.quit()
                    self.root.destroy()
                    self.root = None
                except:
                    pass

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
        self.relevance_checker = analysis_manager.relevance_checker if analysis_manager else None
        self.category_colors = {}
        os.makedirs(output_dir, exist_ok=True)

        # Importierte Funktionen als Instanzmethoden verfügbar machen
        self._sanitize_text_for_excel = _sanitize_text_for_excel
        self._generate_pastel_colors = _generate_pastel_colors
        self._format_confidence = _format_confidence

    def _get_consensus_coding(self, segment_codes: List[Dict]) -> Dict:
        """
        KORRIGIERT: Besseres Debugging für Mehrfachkodierung mit präziser Subkategorien-Zuordnung
        """
        if not segment_codes:
            return {}

        # Prüfe ob es echte Mehrfachkodierung gibt (verschiedene Hauptkategorien)
        categories = [coding['category'] for coding in segment_codes]
        unique_categories = list(set(categories))
        
        # print(f"DEBUG _get_consensus_coding: {len(segment_codes)} Kodierungen, Kategorien: {unique_categories}")
        
        # Wenn alle dieselbe Hauptkategorie haben, normale Konsensbildung
        if len(unique_categories) == 1:
            return self._get_single_consensus_coding(segment_codes)
        
        # Mehrfachkodierung: Erstelle präzises Kategorie-Subkategorie-Mapping
        # print(f"DEBUG: Mehrfachkodierung erkannt mit Kategorien: {unique_categories}")
        
        best_coding = None
        highest_confidence = 0
                
        for coding in segment_codes:
            category = coding.get('category', '')
            subcats = coding.get('subcategories', [])
            confidence = self._extract_confidence_value(coding)
                                   
            # Globale beste Kodierung
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_coding = coding

        # Konvertiere Sets zu Listen für JSON-Serialisierung

        
        if best_coding:
            # Sammle alle Kategorien für competing_categories
            all_categories = list(set(categories))
            
            # Erstelle erweiterte Kodierung
            consensus_coding = best_coding.copy()
            consensus_coding['multiple_coding_detected'] = True
            consensus_coding['all_categories'] = all_categories
            consensus_coding['category_distribution'] = {cat: categories.count(cat) for cat in all_categories}
            
            # VEREINFACHT: Verwende die Subkategorien der gewählten Kodierung direkt
            consensus_coding['subcategories'] = best_coding.get('subcategories', [])
            
            consensus_coding['justification'] = f"[Mehrfachkodierung erkannt: {', '.join(all_categories)}] " + consensus_coding.get('justification', '')
            
            # print(f"DEBUG: Consensus coding erstellt für '{consensus_coding.get('category', '')}'")
            
            return consensus_coding
        
        # Fallback: Erste Kodierung verwenden
        return segment_codes[0] if segment_codes else {}

    def _get_single_consensus_coding(self, segment_codes: List[Dict]) -> Optional[Dict]:
        """
        Ermittelt die Konsens-Kodierung für ein Segment basierend auf einem mehrstufigen Prozess.
        KORRIGIERT: Präzise Subkategorien-Zuordnung ohne Vermischung zwischen Hauptkategorien
        
        Args:
            segment_codes: Liste der Kodierungen für ein Segment von verschiedenen Kodierern
                
        Returns:
            Optional[Dict]: Konsens-Kodierung oder konfidenzbasierte Kodierung
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
            
            # Suche nach Kodierung mit höchster Konfidenz
            highest_confidence = -1
            best_coding = None
            
            for coding in segment_codes:
                confidence = self._extract_confidence_value(coding)
                
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_coding = coding
            
            # Minimalschwelle für Konfidenz (kann angepasst werden)
            confidence_threshold = 0.7
            
            if highest_confidence >= confidence_threshold:
                # Verwende die Kodierung mit der höchsten Konfidenz
                result_coding = best_coding.copy()
                
                # KORRIGIERT: Behalte nur Subkategorien der gewählten Hauptkategorie
                if 'subcategories' in best_coding:
                    result_coding['subcategories'] = best_coding['subcategories']
                
                # Füge Hinweis zur konfidenzbedingten Auswahl hinzu
                result_coding['justification'] = (f"[Konfidenzbasierte Auswahl: {highest_confidence:.2f}] " + 
                                                result_coding.get('justification', ''))
                
                # Dokumentiere den Konsensprozess
                result_coding['consensus_info'] = {
                    'total_coders': total_coders,
                    'category_distribution': dict(category_counts),
                    'max_agreement': max_count / total_coders,
                    'selection_type': 'confidence_based',
                    'confidence': highest_confidence
                }
                
                print(f"  Konfidenzbasierte Auswahl: '{result_coding['category']}' mit {len(result_coding.get('subcategories', []))} Subkategorien")
                return result_coding
            else:
                # Erstelle "Kein Kodierkonsens"-Eintrag
                base_coding = segment_codes[0].copy()
                base_coding['category'] = "Kein Kodierkonsens"
                base_coding['subcategories'] = []  # Keine Subkategorien bei fehlendem Konsens
                base_coding['justification'] = (f"Keine Mehrheit unter den Kodierern und keine Kodierung " +
                                            f"mit ausreichender Konfidenz (max: {highest_confidence:.2f}). " +
                                            f"Kategorien: {', '.join(category_counts.keys())}")
                
                # Dokumentiere den Konsensprozess
                base_coding['consensus_info'] = {
                    'total_coders': total_coders,
                    'category_distribution': dict(category_counts),
                    'max_agreement': max_count / total_coders,
                    'selection_type': 'no_consensus',
                    'max_confidence': highest_confidence
                }
                
                print(f"  Kein Konsens und unzureichende Konfidenz (max: {highest_confidence:.2f})")
                return base_coding

        # 2. Wenn es mehrere gleichhäufige Hauptkategorien gibt, verwende Tie-Breaking
        if len(majority_categories) > 1:
            print(f"Gleichstand zwischen Kategorien: {majority_categories}")
            # Sammle alle Kodierungen für die Mehrheitskategorien
            candidate_codings = [
                coding for coding in segment_codes
                if coding['category'] in majority_categories
            ]
            
            # Wähle basierend auf höchster durchschnittlicher Konfidenz
            highest_avg_confidence = -1
            selected_category = None
            
            for category in majority_categories:
                category_codings = [c for c in candidate_codings if c['category'] == category]
                total_confidence = 0.0
                
                for coding in category_codings:
                    confidence = self._extract_confidence_value(coding)
                    total_confidence += confidence
                    
                avg_confidence = total_confidence / len(category_codings) if category_codings else 0
                
                if avg_confidence > highest_avg_confidence:
                    highest_avg_confidence = avg_confidence
                    selected_category = category
                    
            majority_category = selected_category
            print(f"  Kategorie '{majority_category}' durch höchste Konfidenz ({highest_avg_confidence:.2f}) gewählt")
        else:
            majority_category = majority_categories[0]

        # 3. KORRIGIERT: Sammle nur Kodierungen für die gewählte Mehrheitskategorie
        matching_codings = [
            coding for coding in segment_codes
            if coding['category'] == majority_category
        ]
        
        # 4. PRÄZISE Subkategorien-Konsens NUR für die Mehrheitskategorie
        # print(f"DEBUG: Analysiere Subkategorien für Hauptkategorie '{majority_category}'")
        
        # Sammle Subkategorien gruppiert nach Kodierer
        coder_subcategories = {}
        
        for coding in matching_codings:
            coder_id = coding.get('coder_id', 'unknown')
            subcats = coding.get('subcategories', [])
            
            # Normalisiere Subkategorien-Datenstruktur
            if isinstance(subcats, (list, tuple)):
                coder_subcategories[coder_id] = set(subcats)
            elif isinstance(subcats, str) and subcats:
                # Falls als String übergeben, teile bei Komma
                subcats_list = [s.strip() for s in subcats.split(',') if s.strip()]
                coder_subcategories[coder_id] = set(subcats_list)
            else:
                coder_subcategories[coder_id] = set()
            
            print(f"  Kodierer {coder_id}: {list(coder_subcategories[coder_id])}")
        
        # Bestimme Konsens-Subkategorien (von mindestens 50% der Kodierer verwendet)
        all_subcategories = set()
        for subcat_set in coder_subcategories.values():
            all_subcategories.update(subcat_set)
        
        subcat_counts = {}
        for subcat in all_subcategories:
            count = sum(1 for subcat_set in coder_subcategories.values() if subcat in subcat_set)
            subcat_counts[subcat] = count
        
        # Wähle Subkategorien die von mindestens 50% der Kodierer verwendet wurden
        min_subcat_votes = len(matching_codings) / 2
        consensus_subcats = [
            subcat for subcat, count in subcat_counts.items()
            if count >= min_subcat_votes
        ]
        
        print(f"  Subkategorien-Konsens für '{majority_category}': {len(consensus_subcats)} von {len(all_subcategories)} einzigartige gefunden")
        for subcat, count in sorted(subcat_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= min_subcat_votes:
                print(f"    ✓ {subcat}: {count}/{len(matching_codings)} Kodierer")
            else:
                print(f"    ✗ {subcat}: {count}/{len(matching_codings)} Kodierer (nicht genug)")
        
        # 5. Wähle die beste Basiskodierung aus
        base_coding = max(
            matching_codings,
            key=lambda x: self._calculate_coding_quality(x, consensus_subcats)
        )
        
        # 6. Erstelle finale Konsens-Kodierung
        consensus_coding = base_coding.copy()
        consensus_coding['subcategories'] = consensus_subcats  # NUR Konsens-Subkategorien für diese Hauptkategorie
        
        # 7. Kombiniere Begründungen aller matching codings
        all_justifications = []
        for coding in matching_codings:
            justification = coding.get('justification', '')
            if justification and justification not in all_justifications:
                all_justifications.append(justification)
        
        if all_justifications:
            consensus_coding['justification'] = f"[Konsens aus {len(matching_codings)} Kodierern] " + " | ".join(all_justifications[:3])
        
        # Dokumentiere den Konsensprozess
        consensus_coding['consensus_info'] = {
            'total_coders': total_coders,
            'category_agreement': max_count / total_coders,
            'subcategory_agreement': len(consensus_subcats) / len(all_subcategories) if all_subcategories else 1.0,
            'source_codings': len(matching_codings),
            'selection_type': 'consensus',
            'subcategory_distribution': dict(subcat_counts)
        }
        
        print(f"\nKonsens-Kodierung erstellt:")
        print(f"- Hauptkategorie: {consensus_coding['category']} ({max_count}/{total_coders} Kodierer)")
        print(f"- Subkategorien: {len(consensus_subcats)} im Konsens: {', '.join(consensus_subcats)}")
        print(f"- Übereinstimmung: {(max_count/total_coders)*100:.1f}%")
        
        return consensus_coding

    def _get_majority_coding(self, segment_codes: List[Dict]) -> Optional[Dict]:
        """
        Ermittelt die Mehrheits-Kodierung für ein Segment basierend auf einfacher Mehrheit.
        KORRIGIERT: Subkategorien werden korrekt verarbeitet und zusammengeführt.
        
        Args:
            segment_codes: Liste der Kodierungen für ein Segment von verschiedenen Kodierern
                
        Returns:
            Optional[Dict]: Mehrheits-Kodierung oder konfidenzbasierte Kodierung
        """
        if not segment_codes:
            return None

        print(f"\nMehrheitsentscheidung für Segment mit {len(segment_codes)} Kodierungen...")

        # 1. Zähle Hauptkategorien
        category_counts = Counter(coding['category'] for coding in segment_codes)
        total_coders = len(segment_codes)
        
        # Finde häufigste Hauptkategorie(n)
        max_count = max(category_counts.values())
        majority_categories = [
            category for category, count in category_counts.items()
            if count == max_count
        ]
        
        print(f"  Kategorieverteilung: {dict(category_counts)}")
        print(f"  Häufigste Kategorie(n): {majority_categories} ({max_count}/{total_coders})")
        
        # 2. Bei eindeutiger Mehrheit
        if len(majority_categories) == 1:
            majority_category = majority_categories[0]
            print(f"  ✓ Eindeutige Mehrheit für: '{majority_category}'")
        else:
            # 3. Bei Gleichstand: Wähle nach höchster Konfidenz
            print(f"  Gleichstand zwischen {len(majority_categories)} Kategorien")
            
            # Sammle Kodierungen für die gleichstehenden Kategorien
            tied_codings = [
                coding for coding in segment_codes
                if coding['category'] in majority_categories
            ]
            
            # Finde die Kodierung mit der höchsten Konfidenz
            highest_confidence = -1
            best_coding = None
            
            for coding in tied_codings:
                # Extrahiere Konfidenzwert
                confidence = 0.0
                if isinstance(coding.get('confidence'), dict):
                    confidence = float(coding['confidence'].get('total', 0))
                    if confidence == 0:  # Fallback auf category-Konfidenz
                        confidence = float(coding['confidence'].get('category', 0))
                elif isinstance(coding.get('confidence'), (int, float)):
                    confidence = float(coding['confidence'])
                
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_coding = coding
                    majority_category = coding['category']
            
            print(f"  ✓ Tie-Breaking durch Konfidenz: '{majority_category}' (Konfidenz: {highest_confidence:.2f})")
        
        # 4. Sammle alle Kodierungen für die gewählte Mehrheitskategorie
        matching_codings = [
            coding for coding in segment_codes
            if coding['category'] == majority_category
        ]
        
        # 5. WICHTIG: Subkategorien-Mehrheitsentscheidung
        all_subcategories = []
        for coding in matching_codings:
            subcats = coding.get('subcategories', [])
            # Normalisiere Datenstruktur
            if isinstance(subcats, (list, tuple)):
                all_subcategories.extend(subcats)
            elif isinstance(subcats, str) and subcats:
                # Falls als String übergeben, teile bei Komma
                subcats_list = [s.strip() for s in subcats.split(',') if s.strip()]
                all_subcategories.extend(subcats_list)
        
        # Zähle Subkategorien
        subcat_counts = Counter(all_subcategories)
        
        # Wähle Subkategorien die von mindestens der Hälfte der Kodierer dieser Kategorie verwendet wurden
        min_subcat_votes = len(matching_codings) / 2
        majority_subcats = [
            subcat for subcat, count in subcat_counts.items()
            if count >= min_subcat_votes
        ]
        
        print(f"  Subkategorien-Mehrheit: {len(majority_subcats)} von {len(set(all_subcategories))} einzigartige")
        for subcat, count in subcat_counts.most_common():
            if count >= min_subcat_votes:
                print(f"    ✓ {subcat}: {count}/{len(matching_codings)} Kodierer")
            else:
                print(f"    ✗ {subcat}: {count}/{len(matching_codings)} Kodierer")
        
        # 6. Wähle die beste Basiskodierung (höchste Konfidenz unter den Mehrheitskodierungen)
        base_coding = max(
            matching_codings,
            key=lambda x: self._extract_confidence_value(x)
        )
        
        # 7. Erstelle finale Mehrheits-Kodierung
        majority_coding = base_coding.copy()
        majority_coding['subcategories'] = majority_subcats  # WICHTIG: Setze Mehrheits-Subkategorien
        
        # 8. Kombiniere Begründungen
        all_justifications = []
        for coding in matching_codings:
            justification = coding.get('justification', '')
            if justification and justification not in all_justifications:
                all_justifications.append(justification)  
        
        if all_justifications:
            majority_coding['justification'] = f"[Mehrheit aus {len(matching_codings)} Kodierern] " + " | ".join(all_justifications[:3])
        
        # Dokumentiere den Mehrheitsprozess
        majority_coding['consensus_info'] = {
            'total_coders': total_coders,
            'category_votes': max_count,
            'category_agreement': max_count / total_coders,
            'tied_categories': majority_categories if len(majority_categories) > 1 else [],
            'subcategory_agreement': len(majority_subcats) / len(set(all_subcategories)) if all_subcategories else 1.0,
            'selection_type': 'majority',
            'tie_broken_by_confidence': len(majority_categories) > 1,
            'subcategory_distribution': dict(subcat_counts)
        }
        
        # Aktualisiere Begründung
        if len(majority_categories) > 1:
            majority_coding['justification'] = (
                f"[Mehrheitsentscheidung mit Tie-Breaking] " + 
                majority_coding.get('justification', '')
            )
        else:
            majority_coding['justification'] = (
                f"[Mehrheitsentscheidung: {max_count}/{total_coders}] " + 
                majority_coding.get('justification', '')
            )
        
        print(f"  ✓ Mehrheits-Kodierung erstellt: '{majority_category}' mit {len(majority_subcats)} Subkategorien: {', '.join(majority_subcats)}")
        
        return majority_coding
    
    def _create_category_specific_codings(self, segment_codes: List[Dict], segment_id: str) -> List[Dict]:
        """
        Erstellt kategorie-spezifische Kodierungen mit korrekter Subkategorien-Zuordnung
        KORRIGIERT: Präzise Zuordnung von Subkategorien zu ihren ursprünglichen Hauptkategorien
        """
        # Gruppiere Kodierungen nach Hauptkategorien
        category_groups = {}
        
        # print(f"DEBUG: Erstelle kategorie-spezifische Kodierungen für {segment_id}")
        # print(f"Input: {len(segment_codes)} Kodierungen")
        
        for coding in segment_codes:
            main_cat = coding.get('category', '')
            if main_cat and main_cat not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                if main_cat not in category_groups:
                    category_groups[main_cat] = []
                category_groups[main_cat].append(coding)
        
        # print(f"DEBUG: Kategorie-Gruppen für {segment_id}: {list(category_groups.keys())}")
        
        # Erstelle für jede Hauptkategorie eine konsolidierte Kodierung
        result_codings = []
        
        for i, (main_cat, codings_for_cat) in enumerate(category_groups.items(), 1):
            # print(f"DEBUG: Verarbeite Hauptkategorie '{main_cat}' mit {len(codings_for_cat)} Kodierungen")
            
            # Wähle die beste Kodierung für diese Kategorie als Basis
            best_coding = max(codings_for_cat, key=lambda x: self._extract_confidence_value(x))
            
            # VEREINFACHT: Sammle Subkategorien basierend auf der Hauptkategorie
            category_specific_subcats = set()

            for coding in codings_for_cat:
                # Hole alle Subkategorien aus Kodierungen für diese Hauptkategorie
                subcats = coding.get('subcategories', [])
                
                # Normalisiere Subkategorien-Format
                if isinstance(subcats, (list, tuple)):
                    category_specific_subcats.update(subcats)
                elif isinstance(subcats, str) and subcats:
                    category_specific_subcats.update([s.strip() for s in subcats.split(',') if s.strip()])
                
                print(f"  Kodierung für '{main_cat}': {len(subcats)} Subkategorien gefunden")

            # Konvertiere zu Liste für JSON-Serialisierung
            unique_subcats = list(category_specific_subcats)
            
            # KORRIGIERT: Erstelle konsolidierte Kodierung mit präzisen Subkategorien
            consolidated_coding = best_coding.copy()
            consolidated_coding['category'] = main_cat
            consolidated_coding['subcategories'] = unique_subcats  # NUR kategorie-spezifische Subkategorien
            consolidated_coding['multiple_coding_instance'] = i
            consolidated_coding['total_coding_instances'] = len(category_groups)
            consolidated_coding['target_category'] = main_cat
            consolidated_coding['category_focus_used'] = True
            
            # Erweiterte Begründung für Mehrfachkodierung
            original_justification = consolidated_coding.get('justification', '')
            consolidated_coding['justification'] = f"[Mehrfachkodierung - Kategorie {i}/{len(category_groups)}] {original_justification}"
            
            print(f"  Kategorie '{main_cat}': {len(unique_subcats)} finale Subkategorien -> {unique_subcats}")
            
            result_codings.append(consolidated_coding)
        
        # print(f"DEBUG: Erstellt {len(result_codings)} kategorie-spezifische Kodierungen für {segment_id}")
        return result_codings
    
   
   
    # Zusätzliche Methode für ResultsExporter Klasse
    def debug_export_process(self, codings: List[Dict]) -> None:
        """
        Öffentliche Debug-Methode für Export-Prozess
        Kann vor dem eigentlichen Export aufgerufen werden
        """
        print(f"\n🔍 STARTE EXPORT-DEBUG für {len(codings)} Kodierungen")
        self._debug_export_preparation(codings)
        
        # Zusätzliche Checks
        segments_with_issues = []
        
        for coding in codings:
            segment_id = coding.get('segment_id', '')
            category = coding.get('category', '')
            subcats = coding.get('subcategories', [])
            
            # Prüfe auf leere Subkategorien bei kategorisierten Segmenten
            if category and category not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                if not subcats or (isinstance(subcats, list) and len(subcats) == 0):
                    segments_with_issues.append({
                        'segment_id': segment_id,
                        'category': category,
                        'issue': 'Keine Subkategorien trotz Kategorisierung'
                    })
        
        if segments_with_issues:
            print(f"\n⚠ GEFUNDENE PROBLEME: {len(segments_with_issues)} Segmente mit fehlenden Subkategorien")
            for issue in segments_with_issues[:3]:
                print(f"  - {issue['segment_id']}: {issue['category']} -> {issue['issue']}")
            if len(segments_with_issues) > 3:
                print(f"  ... und {len(segments_with_issues) - 3} weitere")
        else:
            print(f"\n✅ Keine offensichtlichen Subkategorien-Probleme gefunden")
        
        print(f"\n🔍 EXPORT-DEBUG ABGESCHLOSSEN")

    def _extract_confidence_value(self, coding: Dict) -> float:
        """
        Hilfsmethode zum Extrahieren des Konfidenzwerts aus einer Kodierung.
        
        Args:
            coding: Kodierung mit Konfidenzinformation
            
        Returns:
            float: Konfidenzwert zwischen 0 und 1
        """
        try:
            if isinstance(coding.get('confidence'), dict):
                confidence = float(coding['confidence'].get('total', 0))
                if confidence == 0:  # Fallback auf category-Konfidenz
                    confidence = float(coding['confidence'].get('category', 0))
            elif isinstance(coding.get('confidence'), (int, float)):
                confidence = float(coding['confidence'])
            else:
                confidence = 0.0
            return confidence
        except (ValueError, TypeError):
            return 0.0

    def _get_manual_priority_coding(self, segment_codes: List[Dict]) -> Optional[Dict]:
        """
        Bevorzugt manuelle Kodierungen vor automatischen Kodierungen.
        KORRIGIERT: Subkategorien werden korrekt verarbeitet und zusammengeführt.
        
        Args:
            segment_codes: Liste der Kodierungen für ein Segment von verschiedenen Kodierern
                
        Returns:
            Optional[Dict]: Priorisierte Kodierung mit korrekten Subkategorien
        """
        if not segment_codes:
            return None

        print(f"\nManuelle Priorisierung für Segment mit {len(segment_codes)} Kodierungen...")

        # 1. Trenne manuelle und automatische Kodierungen
        manual_codings = []
        auto_codings = []
        
        for coding in segment_codes:
            coder_id = coding.get('coder_id', '')
            # Erkenne manuelle Kodierungen anhand der coder_id
            if 'human' in coder_id.lower() or 'manual' in coder_id.lower() or coding.get('manual_coding', False):
                manual_codings.append(coding)
            else:
                auto_codings.append(coding)
        
        print(f"  Gefunden: {len(manual_codings)} manuelle, {len(auto_codings)} automatische Kodierungen")
        
        # 2. Wenn manuelle Kodierungen vorhanden sind, bevorzuge diese
        if manual_codings:
            print("  ✓ Verwende manuelle Kodierungen mit Priorität")
            
            if len(manual_codings) == 1:
                # Nur eine manuelle Kodierung - verwende diese direkt
                selected_coding = manual_codings[0].copy()
                
                # WICHTIG: Subkategorien beibehalten!
                if 'subcategories' in manual_codings[0]:
                    selected_coding['subcategories'] = manual_codings[0]['subcategories']
                
                selected_coding['consensus_info'] = {
                    'total_coders': len(segment_codes),
                    'manual_coders': len(manual_codings),
                    'auto_coders': len(auto_codings),
                    'selection_type': 'single_manual',
                    'priority_reason': 'Einzige manuelle Kodierung verfügbar'
                }
                print(f"    Einzige manuelle Kodierung: '{selected_coding['category']}' mit {len(selected_coding.get('subcategories', []))} Subkategorien")
                
            else:
                # Mehrere manuelle Kodierungen - suche Konsens unter diesen
                print(f"    Suche Konsens unter {len(manual_codings)} manuellen Kodierungen")
                
                # Prüfe ob alle dieselbe Hauptkategorie haben
                manual_categories = [c['category'] for c in manual_codings]
                if len(set(manual_categories)) == 1:
                    # Alle haben dieselbe Hauptkategorie - konsolidiere Subkategorien
                    main_category = manual_categories[0]
                    
                    # Sammle alle Subkategorien von manuellen Kodierungen
                    all_manual_subcats = []
                    for coding in manual_codings:
                        subcats = coding.get('subcategories', [])
                        if isinstance(subcats, (list, tuple)):
                            all_manual_subcats.extend(subcats)
                        elif isinstance(subcats, str) and subcats:
                            subcats_list = [s.strip() for s in subcats.split(',') if s.strip()]
                            all_manual_subcats.extend(subcats_list)
                    
                    # Finde Konsens-Subkategorien (mindestens von der Hälfte verwendet)
                    subcat_counts = Counter(all_manual_subcats)
                    min_votes = len(manual_codings) / 2
                    consensus_subcats = [
                        subcat for subcat, count in subcat_counts.items()
                        if count >= min_votes
                    ]
                    
                    # Wähle beste manuelle Kodierung als Basis
                    selected_coding = max(
                        manual_codings,
                        key=lambda x: self._extract_confidence_value(x)
                    ).copy()
                    
                    # Setze konsolidierte Subkategorien
                    selected_coding['subcategories'] = consensus_subcats
                    
                    # Kombiniere Begründungen
                    manual_justifications = [c.get('justification', '')[:100] for c in manual_codings if c.get('justification', '')]
                    if manual_justifications:
                        selected_coding['justification'] = f"[Konsens aus {len(manual_codings)} manuellen Kodierungen] " + " | ".join(manual_justifications[:3])
                    
                    selected_coding['consensus_info'] = {
                        'total_coders': len(segment_codes),
                        'manual_coders': len(manual_codings),
                        'auto_coders': len(auto_codings),
                        'selection_type': 'manual_consensus',
                        'priority_reason': 'Konsens unter manuellen Kodierungen',
                        'subcategory_distribution': dict(subcat_counts)
                    }
                    print(f"    Konsens bei manuellen Kodierungen: '{selected_coding['category']}' mit {len(consensus_subcats)} Subkategorien: {', '.join(consensus_subcats)}")
                else:
                    # Verschiedene Hauptkategorien - wähle nach Konfidenz
                    print("    Verschiedene Hauptkategorien - wähle nach Konfidenz")
                    selected_coding = max(
                        manual_codings,
                        key=lambda x: self._extract_confidence_value(x)
                    ).copy()
                    
                    # WICHTIG: Subkategorien der gewählten Kodierung beibehalten
                    if 'subcategories' in selected_coding:
                        original_subcats = selected_coding['subcategories']
                        selected_coding['subcategories'] = original_subcats
                    
                    selected_coding['consensus_info'] = {
                        'total_coders': len(segment_codes),
                        'manual_coders': len(manual_codings),
                        'auto_coders': len(auto_codings),
                        'selection_type': 'manual_confidence',
                        'priority_reason': 'Höchste Konfidenz unter manuellen Kodierungen (verschiedene Hauptkategorien)'
                    }
                    print(f"    Manuelle Kodierung nach Konfidenz: '{selected_coding['category']}' mit {len(selected_coding.get('subcategories', []))} Subkategorien")
        
        else:
            # 3. Keine manuellen Kodierungen - verwende automatische mit Konsens
            print("  Keine manuellen Kodierungen - verwende automatische Kodierungen")
            
            # Verwende die bestehende Konsens-Logik für automatische Kodierungen
            consensus_coding = self._get_consensus_coding(auto_codings)
            
            if consensus_coding:
                selected_coding = consensus_coding.copy()
                # Aktualisiere consensus_info
                selected_coding['consensus_info'] = selected_coding.get('consensus_info', {})
                selected_coding['consensus_info'].update({
                    'total_coders': len(segment_codes),
                    'manual_coders': 0,
                    'auto_coders': len(auto_codings),
                    'selection_type': 'auto_consensus',
                    'priority_reason': 'Keine manuellen Kodierungen verfügbar - automatischer Konsens'
                })
                print(f"    Automatischer Konsens: '{selected_coding['category']}' mit {len(selected_coding.get('subcategories', []))} Subkategorien")
            else:
                # Fallback: Wähle automatische Kodierung mit höchster Konfidenz
                selected_coding = max(
                    auto_codings,
                    key=lambda x: self._extract_confidence_value(x)
                ).copy()
                
                # WICHTIG: Subkategorien beibehalten
                if 'subcategories' in selected_coding:
                    original_subcats = selected_coding['subcategories']
                    selected_coding['subcategories'] = original_subcats
                
                selected_coding['consensus_info'] = {
                    'total_coders': len(segment_codes),
                    'manual_coders': 0,
                    'auto_coders': len(auto_codings),
                    'selection_type': 'auto_confidence',
                    'priority_reason': 'Kein automatischer Konsens - höchste Konfidenz'
                }
                print(f"    Automatische Kodierung nach Konfidenz: '{selected_coding['category']}' mit {len(selected_coding.get('subcategories', []))} Subkategorien")
        
        # 4. Aktualisiere Begründung mit Prioritätsinformation
        priority_info = selected_coding['consensus_info']['priority_reason']
        selected_coding['justification'] = (
            f"[Manuelle Priorisierung: {priority_info}] " + 
            selected_coding.get('justification', '')
        )
        
        return selected_coding


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
  
    def _prepare_coding_for_export(self, coding: dict, chunk: str, chunk_id: int, doc_name: str) -> dict:
        """
        Bereitet eine Kodierung für den Export vor.
        KORRIGIERT: Nutzt präzises Kategorie-Subkategorie-Mapping bei Mehrfachkodierung
        """
        try:
            # Extrahiere Attribute aus dem Dateinamen
            attribut1, attribut2, attribut3 = self._extract_metadata(doc_name)
            
            # Erstelle eindeutigen Präfix für Chunk-Nr
            chunk_prefix = ""
            if attribut1 and attribut2:
                chunk_prefix = (attribut1[:2] + attribut2[:2] + attribut3[:2]).upper()
            else:
                chunk_prefix = doc_name[:5].upper()
            
            # Prüfe ob eine gültige Kategorie vorhanden ist
            category = coding.get('category', '')
            
            # KORRIGIERT: Nutze Kategorie-Subkategorie-Mapping bei Mehrfachkodierung
            subcategories = coding.get('subcategories', [])
            
            # Prüfe auf Hauptkategorie im grounded Modus
            main_category = coding.get('main_category', '')
            if main_category and main_category != category:
                if CONFIG.get('ANALYSIS_MODE') == 'grounded':
                    display_category = main_category
                    if category and category not in subcategories:
                        subcategories = list(subcategories) + [category]
                else:
                    display_category = category
            else:
                display_category = category
            
            # Verbesserte Subkategorien-Verarbeitung mit Bereinigung
            subcats_text = ""
            
            if subcategories:
                if isinstance(subcategories, str):
                    # String: Direkt verwenden nach Bereinigung
                    subcats_text = subcategories.strip()
                elif isinstance(subcategories, (list, tuple)):
                    # Liste/Tupel: Bereinige und verbinde
                    clean_subcats = []
                    for subcat in subcategories:
                        if subcat and str(subcat).strip():
                            clean_text = str(subcat).strip()
                            # Entferne verschiedene Arten von Klammern und Anführungszeichen
                            clean_text = clean_text.replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                            if clean_text:
                                clean_subcats.append(clean_text)
                    subcats_text = ', '.join(clean_subcats)
                elif isinstance(subcategories, dict):
                    # Dict: Verwende Schlüssel (falls es ein Dict von Subkategorien ist)
                    clean_subcats = []
                    for key in subcategories.keys():
                        clean_key = str(key).strip()
                        if clean_key:
                            clean_subcats.append(clean_key)
                    subcats_text = ', '.join(clean_subcats)
                else:
                    # Andere Typen: String-Konversion mit Bereinigung
                    subcats_text = str(subcategories).strip()
                    subcats_text = subcats_text.replace('[', '').replace(']', '').replace("'", "").replace('"', '')
            
            # Zusätzliche Bereinigung für den Export
            subcats_text = subcats_text.replace('[', '').replace(']', '').replace("'", "")
                       
            # Bestimme den Kategorietyp und Kodiertstatus
            if display_category == "Kein Kodierkonsens":
                kategorie_typ = "unkodiert"
                is_coded = 'Nein'
            elif display_category == "Nicht kodiert":
                kategorie_typ = "unkodiert"
                is_coded = 'Nein'
            else:
                if display_category in DEDUKTIVE_KATEGORIEN:
                    kategorie_typ = "deduktiv"
                elif CONFIG.get('ANALYSIS_MODE') == 'grounded':
                    kategorie_typ = "grounded"
                else:
                    kategorie_typ = "induktiv"
                is_coded = 'Ja'
                        
            # Formatiere Keywords
            raw_keywords = coding.get('keywords', '')
            if isinstance(raw_keywords, list):
                formatted_keywords = [kw.strip() for kw in raw_keywords]
            else:
                formatted_keywords = raw_keywords.replace("[", "").replace("]", "").replace("'", "").split(",")
                formatted_keywords = [kw.strip() for kw in formatted_keywords if kw.strip()]
            
            # Formatierung der Begründung (unverändert)
            justification = coding.get('justification', '')
            
            # Entferne nur die Review-Prefixes, aber behalte die volle Begründung
            if justification.startswith('[Konsens'):
                parts = justification.split('] ', 1)
                if len(parts) > 1:
                    remaining_text = parts[1]
                    if ' | ' in remaining_text:
                        split_parts = remaining_text.split(' | ')
                        if len(split_parts) > 1 and split_parts[0].strip() == split_parts[1].strip():
                            justification = split_parts[0].strip()
                        else:
                            justification = split_parts[0].strip()
                    else:
                        justification = remaining_text.strip()
            elif justification.startswith('[Mehrheit'):
                parts = justification.split('] ', 1)
                if len(parts) > 1:
                    remaining_text = parts[1]
                    if ' | ' in remaining_text:
                        split_parts = remaining_text.split(' | ')
                        justification = split_parts[0].strip()
                    else:
                        justification = remaining_text.strip()
            elif justification.startswith('[Manuelle Priorisierung'):
                parts = justification.split('] ', 1)
                if len(parts) > 1:
                    justification = parts[1].strip()
            elif justification.startswith('[Konfidenzbasierte Auswahl'):
                parts = justification.split('] ', 1)
                if len(parts) > 1:
                    justification = parts[1].strip()
            
            # Export-Dictionary mit allen erforderlichen Feldern
            export_data = {
                'Dokument': self._sanitize_text_for_excel(doc_name),
                self.attribute_labels['attribut1']: self._sanitize_text_for_excel(attribut1),
                self.attribute_labels['attribut2']: self._sanitize_text_for_excel(attribut2),
            }
            
            # Füge attribut3 hinzu, wenn es definiert ist
            if 'attribut3' in self.attribute_labels and self.attribute_labels['attribut3']:
                export_data[self.attribute_labels['attribut3']] = self._sanitize_text_for_excel(attribut3)
            

            # Erstelle eindeutige Chunk-ID mit Mehrfachkodierungs-Suffix
            if coding.get('total_coding_instances', 1) > 1:
                unique_chunk_id = f"{chunk_prefix}-{chunk_id}-{coding.get('multiple_coding_instance', 1)}"
                mehrfachkodierung_status = 'Ja'
            else:
                unique_chunk_id = f"{chunk_prefix}-{chunk_id}"
                mehrfachkodierung_status = 'Nein'
            

            # Rest der Daten in der gewünschten Reihenfolge
            additional_fields = {
                'Chunk_Nr': unique_chunk_id,
                'Text': self._sanitize_text_for_excel(chunk),
                'Paraphrase': self._sanitize_text_for_excel(coding.get('paraphrase', '')),
                'Kodiert': is_coded,
                'Hauptkategorie': self._sanitize_text_for_excel(display_category),
                'Kategorietyp': kategorie_typ,
                'Subkategorien': self._sanitize_text_for_excel(subcats_text), 
                'Schlüsselwörter': self._sanitize_text_for_excel(', '.join(formatted_keywords)),
                'Begründung': self._sanitize_text_for_excel(justification),
                'Konfidenz': self._sanitize_text_for_excel(self._format_confidence(coding.get('confidence', {}))),
                'Mehrfachkodierung': mehrfachkodierung_status, 
                # Neue Felder für Mehrfachkodierung:
                'Mehrfachkodierung_Instanz': coding.get('multiple_coding_instance', 1),
                'Mehrfachkodierung_Gesamt': coding.get('total_coding_instances', 1),
                'Fokus_Kategorie': self._sanitize_text_for_excel(coding.get('target_category', '')),
                'Fokus_verwendet': 'Ja' if coding.get('category_focus_used', False) else 'Nein',
                'Original_Chunk_ID': f"{chunk_prefix}-{chunk_id}"
            }

            export_data.update(additional_fields)

            # Nur Kontext-bezogene Felder hinzufügen, wenn vorhanden
            if 'context_summary' in coding and coding['context_summary']:
                export_data['Progressive_Context'] = self._sanitize_text_for_excel(coding.get('context_summary', ''))
            
            if 'context_influence' in coding and coding['context_influence']:
                export_data['Context_Influence'] = self._sanitize_text_for_excel(coding.get('context_influence', ''))
            
            return export_data
                
        except Exception as e:
            print(f"Fehler bei der Exportvorbereitung für Chunk {chunk_id}: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()
            return {
                'Dokument': self._sanitize_text_for_excel(doc_name),
                'Chunk_Nr': f"{doc_name[:5].upper()}-{chunk_id}",
                'Text': self._sanitize_text_for_excel(chunk),
                'Paraphrase': '',
                'Kodiert': 'Nein',
                'Hauptkategorie': 'Fehler bei Verarbeitung',
                'Kategorietyp': 'unbekannt',
                'Begründung': self._sanitize_text_for_excel(f'Fehler: {str(e)}'),
                'Subkategorien': '',
                'Konfidenz': '',
                'Mehrfachkodierung': 'Nein'
            }
    
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
        attribut3 = tokens[2] if len(tokens) >= 3 else "" 
        return attribut1, attribut2, attribut3

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

            current_row += 2

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

            # 3.3 Attribut 3 (nur wenn definiert)
            attribut3_label = self.attribute_labels.get('attribut3', '')
            if attribut3_label and attribut3_label in df_coded.columns:
                cell = worksheet.cell(row=current_row, column=1, value=f"3.3 Verteilung nach {attribut3_label}")
                cell.font = title_font
                current_row += 1

                # Analyse für Attribut 3
                attr3_counts = df_coded[attribut3_label].value_counts()
                attr3_counts['Gesamt'] = attr3_counts.sum()

                # Header
                headers = [attribut3_label, 'Anzahl']
                for col, header in enumerate(headers, 1):
                    cell = worksheet.cell(row=current_row, column=col, value=header)
                    cell.font = header_font
                    cell.border = thin_border
                current_row += 1

                # Daten für Attribut 3
                for idx, value in attr3_counts.items():
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

            # 3.4 Kreuztabelle der Attribute
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

            # Füge erweitere Kreuztabelle für Attribut 3 hinzu, wenn vorhanden
            if attribut3_label and attribut3_label in df_coded.columns:
                # Erstelle zusätzliche Kreuztabelle für Attribut 1 und 3
                cross_tab_1_3 = pd.crosstab(
                    df_coded[attribut1_label], 
                    df_coded[attribut3_label],
                    margins=True,
                    margins_name='Gesamt'
                )

                # Header für Kreuztabelle 1-3
                headers = [attribut1_label] + list(cross_tab_1_3.columns)
                for col, header in enumerate(headers, 1):
                    cell = worksheet.cell(row=current_row, column=col, value=header)
                    cell.font = header_font
                    cell.border = thin_border
                current_row += 1

                # Daten für Kreuztabelle 1-3
                for idx, row in cross_tab_1_3.iterrows():
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
                    
                current_row += 2
                
                # Erstelle zusätzliche Kreuztabelle für Attribut 2 und 3
                cross_tab_2_3 = pd.crosstab(
                    df_coded[attribut2_label], 
                    df_coded[attribut3_label],
                    margins=True,
                    margins_name='Gesamt'
                )

                # Header für Kreuztabelle 2-3
                headers = [attribut2_label] + list(cross_tab_2_3.columns)
                for col, header in enumerate(headers, 1):
                    cell = worksheet.cell(row=current_row, column=col, value=header)
                    cell.font = header_font
                    cell.border = thin_border
                current_row += 1

                # Daten für Kreuztabelle 2-3
                for idx, row in cross_tab_2_3.iterrows():
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
                    
                current_row += 2
            
            # Passe Spaltenbreiten an
            for col in worksheet.columns:
                max_length = 0
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                    worksheet.column_dimensions[col[0].column_letter].width = min(max_length + 2, 20)


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
                        inductive_coder: 'InductiveCoder' = None,
                        document_summaries: Dict[str, str] = None) -> None:
        """
        Exportiert die Analyseergebnisse mit Konsensfindung zwischen Kodierern.
        KORRIGIERT: Führt Review-Prozess durch bevor exportiert wird.
        """
        try:
            # Wenn inductive_coder als Parameter übergeben wurde, aktualisiere das Attribut
            if inductive_coder:
                self.inductive_coder = inductive_coder

            # WICHTIG: Speichere chunks als Instanzvariable für _prepare_coding_for_export
            self.chunks = chunks

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

            # ===== NEUER CODE: REVIEW-PROZESS VOR EXPORT =====
            print(f"\nStarte {export_mode}-Review für Kodierungsentscheidungen...")
            
            # Gruppiere Kodierungen nach Segmenten für Review
            segment_codings = {}
            for coding in codings:
                segment_id = coding.get('segment_id')
                if segment_id:
                    if segment_id not in segment_codings:
                        segment_codings[segment_id] = []
                    segment_codings[segment_id].append(coding)
            
            print(f"Gefunden: {len(segment_codings)} einzigartige Segmente")
            
            # Führe Review-Prozess durch
            reviewed_codings = []
            all_reviewed_codings = []

            for segment_id, segment_codes in segment_codings.items():
                if len(segment_codes) == 1:
                    # Nur eine Kodierung für dieses Segment
                    reviewed_codings.append(segment_codes[0])
                    continue
                
                # Mehrere Kodierungen - führe Review durch
                if export_mode == "consensus":
                    final_coding = self._get_consensus_coding(segment_codes)
                elif export_mode == "majority":
                    final_coding = self._get_majority_coding(segment_codes)
                elif export_mode == "manual_priority":
                    final_coding = self._get_manual_priority_coding(segment_codes)
                
                # Fallback wenn kein Review-Ergebnis
                if not final_coding:
                    final_coding = segment_codes[0]
                    final_coding['category'] = "Kein Kodierkonsens"
                
                # KORRIGIERT: Behandle Mehrfachkodierung mit korrekter Subkategorien-Zuordnung
                if final_coding.get('multiple_coding_detected', False):
                    # Verwende die neue präzise Methode
                    category_specific_codings = self._create_category_specific_codings(segment_codes, segment_id)
                    all_reviewed_codings.extend(category_specific_codings)
                else:
                    # Normale Kodierung ohne Mehrfachkodierung
                    all_reviewed_codings.append(final_coding)

            # Verwende reviewed_codings statt codings für den Export
            export_data = []
            for coding in all_reviewed_codings:
                segment_id = coding.get('segment_id', '')
                # Überspringe Kodierungen ohne segment_id
                if not segment_id:
                    continue

                try:
                    doc_name = segment_id.split('_chunk_')[0]
                    chunk_id = int(segment_id.split('_chunk_')[1])

                    # NEUE PRÜFUNG: Sicherstellen, dass der Dokumentname im chunks Dictionary existiert
                    if doc_name not in chunks:
                         print(f"Warnung: Dokumentname '{doc_name}' aus Segment-ID '{segment_id}' nicht in geladenen Chunks gefunden. Überspringe Export für diese Kodierung.")
                         continue
                    
                    # Stelle sicher, dass der chunk_id im chunks Dictionary für das Dokument existiert
                    if chunk_id >= len(chunks[doc_name]):
                         print(f"Warnung: Chunk {segment_id} nicht in den geladenen Chunks für Dokument '{doc_name}' gefunden. Überspringe Export für diese Kodierung.")
                         continue

                    chunk_text = chunks[doc_name][chunk_id]
                    export_entry = self._prepare_coding_for_export(coding, chunk_text, chunk_id, doc_name)
                    export_data.append(export_entry)

                except Exception as e:
                    print(f"Fehler bei Verarbeitung von Segment {segment_id} für Export: {str(e)}")
                    # Details zum Fehler ausgeben
                    import traceback
                    traceback.print_exc()
                    continue

            # Validiere Export-Daten
            if not self._validate_export_data(export_data):
                 print("Warnung: Export-Daten enthalten möglicherweise Fehler oder sind unvollständig nach der Aufbereitung.")
                 if not export_data:
                      print("Fehler: Keine Export-Daten nach Aufbereitung vorhanden.")
                      return

            # Erstelle DataFrames mit zusätzlicher Bereinigung für Zeilen und Spalten
            try:
                # Bereinige Spaltennamen für DataFrame
                sanitized_export_data = []
                for entry in export_data:
                    sanitized_entry = {}
                    for key, value in entry.items():
                        # Bereinige auch die Schlüssel (falls nötig)
                        sanitized_key = self._sanitize_text_for_excel(key)
                        sanitized_entry[sanitized_key] = value
                    sanitized_export_data.append(sanitized_entry)

                print(f"Export-Daten nach Review bereinigt: {len(sanitized_export_data)} Einträge")

                # Verwende ALLE aufbereiteten und reviewten Export-Einträge für den df_details
                df_details = pd.DataFrame(sanitized_export_data)
                # Filtere für df_coded nur die Einträge, die erfolgreich kodiert wurden
                df_coded = df_details[df_details['Kodiert'].isin(['Ja', 'Teilweise'])].copy()

                print(f"DataFrames erstellt: {len(df_details)} Gesamt, {len(df_coded)} Kodiert")

            except Exception as e:
                print(f"Fehler bei der Erstellung des DataFrame: {str(e)}")
                print("Details:")
                traceback.print_exc()
                return

            # Initialisiere Farbzuordnung einmalig für alle Sheets
            self._initialize_category_colors(df_details)

            print(f"Finale DataFrames erstellt: {len(df_details)} Gesamt, {len(df_coded)} Kodiert")

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
                
                # 4. Exportiere Intercoderanalyse (mit ursprünglichen Kodierungen vor Review)
                if segment_codings:
                    print("\nExportiere Intercoderanalyse...")
                    self._export_intercoder_analysis(
                        writer, 
                        segment_codings,  # Verwende ursprüngliche Kodierungen für Intercoder-Analyse
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

                # 7. Exportiere progressive Summaries
                if document_summaries:
                    self._export_progressive_summaries(writer, document_summaries)

                # 8. Exportiere Review-Statistiken
                review_stats = self._calculate_review_statistics(all_reviewed_codings, export_mode)
                self._export_review_statistics(writer, review_stats, export_mode)

                # Stelle sicher, dass mindestens ein Sheet sichtbar ist
                if len(writer.book.sheetnames) == 0:
                    writer.book.create_sheet('Leeres_Sheet')
                
                # Setze alle Sheets auf sichtbar
                for sheet in writer.book.sheetnames:
                    writer.book[sheet].sheet_state = 'visible'

                print(f"\nErgebnisse erfolgreich exportiert nach: {filepath}")
                print(f"- {len(segment_codings)} Segmente vor Review")
                print(f"- {len(reviewed_codings)} finale Kodierungen nach {export_mode}-Review")
                print(f"- Reliabilität: {reliability:.3f}")

        except Exception as e:
            print(f"Fehler beim Excel-Export: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _calculate_review_statistics(self, codings: List[Dict], export_mode: str) -> Dict[str, int]:
        """
        Berechnet Statistiken des Review-Prozesses.
        
        Args:
            codings: Liste der finalen Kodierungen nach Review
            export_mode: Verwendeter Review-Modus
            
        Returns:
            Dict[str, int]: Statistiken des Review-Prozesses
        """
        stats = {
            'consensus_found': 0,
            'majority_found': 0,
            'manual_priority': 0,
            'no_consensus': 0,
            'single_coding': 0,
            'multiple_coding_consolidated': 0
        }
        
        for coding in codings:
            # Bestimme den Typ der Kodierung basierend auf verfügbaren Informationen
            if coding.get('manual_review', False):
                stats['manual_priority'] += 1
            elif coding.get('consolidated_from_multiple', False):
                stats['multiple_coding_consolidated'] += 1
            elif coding.get('consensus_info', {}).get('selection_type') == 'consensus':
                stats['consensus_found'] += 1
            elif coding.get('consensus_info', {}).get('selection_type') == 'majority':
                stats['majority_found'] += 1
            elif coding.get('consensus_info', {}).get('selection_type') == 'no_consensus':
                stats['no_consensus'] += 1
            elif coding.get('category') == 'Kein Kodierkonsens':
                stats['no_consensus'] += 1
            else:
                stats['single_coding'] += 1
        
        return stats

    def _export_review_statistics(self, writer, review_stats: Dict, export_mode: str):
        """
        Exportiert Statistiken des Review-Prozesses in ein separates Excel-Sheet.
        
        Args:
            writer: Excel Writer Objekt
            review_stats: Statistiken des Review-Prozesses
            export_mode: Verwendeter Review-Modus
        """
        try:
            if 'Review_Statistiken' not in writer.sheets:
                writer.book.create_sheet('Review_Statistiken')
                
            worksheet = writer.sheets['Review_Statistiken']
            current_row = 1
            
            # Titel
            worksheet.cell(row=current_row, column=1, value=f"Review-Statistiken ({export_mode}-Modus)")
            worksheet.cell(row=current_row, column=1).font = Font(bold=True, size=14)
            current_row += 2
            
            # Statistiken
            worksheet.cell(row=current_row, column=1, value="Kategorie")
            worksheet.cell(row=current_row, column=2, value="Anzahl")
            worksheet.cell(row=current_row, column=1).font = Font(bold=True)
            worksheet.cell(row=current_row, column=2).font = Font(bold=True)
            current_row += 1
            
            total_reviewed = sum(review_stats.values())
            for stat_name, count in review_stats.items():
                if count > 0:
                    # Übersetze Statistik-Namen
                    german_names = {
                        'consensus_found': 'Konsens gefunden',
                        'majority_found': 'Mehrheit gefunden', 
                        'manual_priority': 'Manuelle Priorität',
                        'no_consensus': 'Kein Konsens',
                        'single_coding': 'Einzelkodierung'
                    }
                    
                    display_name = german_names.get(stat_name, stat_name)
                    worksheet.cell(row=current_row, column=1, value=display_name)
                    worksheet.cell(row=current_row, column=2, value=count)
                    
                    # Prozentangabe
                    if total_reviewed > 0:
                        percentage = (count / total_reviewed) * 100
                        worksheet.cell(row=current_row, column=3, value=f"{percentage:.1f}%")
                    
                    current_row += 1
            
            # Gesamtsumme
            worksheet.cell(row=current_row, column=1, value="Gesamt")
            worksheet.cell(row=current_row, column=2, value=total_reviewed)
            worksheet.cell(row=current_row, column=3, value="100.0%")
            worksheet.cell(row=current_row, column=1).font = Font(bold=True)
            worksheet.cell(row=current_row, column=2).font = Font(bold=True)
            worksheet.cell(row=current_row, column=3).font = Font(bold=True)
            
            # Spaltenbreiten anpassen
            worksheet.column_dimensions['A'].width = 20
            worksheet.column_dimensions['B'].width = 10
            worksheet.column_dimensions['C'].width = 10
            
            print("Review-Statistiken erfolgreich exportiert")
            
        except Exception as e:
            print(f"Fehler beim Export der Review-Statistiken: {str(e)}")
            import traceback
            traceback.print_exc()
    def _export_progressive_summaries(self, writer, document_summaries):
        """
        Exportiert die finalen Document-Summaries mit verbesserter Fehlerbehandlung.
        """
        try:
            if 'Progressive_Summaries' not in writer.sheets:
                writer.book.create_sheet('Progressive_Summaries')
                
            worksheet = writer.sheets['Progressive_Summaries']
            
            # Eventuell vorhandene Daten löschen
            if worksheet.max_row > 0:
                worksheet.delete_rows(1, worksheet.max_row)
            
            # Prüfe ob Daten vorhanden sind
            if not document_summaries:
                # Dummy-Zeile einfügen, um leeres Blatt zu vermeiden
                worksheet.cell(row=1, column=1, value="Keine progressiven Summaries verfügbar")
                return
            
            # Erstelle Daten für Export
            summary_data = []
            for doc_name, summary in document_summaries.items():
                summary_data.append({
                    'Dokument': doc_name,
                    'Finales Summary': summary,
                    'Wortanzahl': len(summary.split())
                })
                
            # Erstelle DataFrame
            if summary_data:
                df = pd.DataFrame(summary_data)
                
                # Exportiere Daten ohne Tabellenformatierung
                for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
                    for c_idx, value in enumerate(row, 1):
                        worksheet.cell(row=r_idx, column=c_idx, value=value)
                
                # Einfache Formatierung ohne Tabellen-Definition
                for cell in worksheet[1]:
                    cell.font = Font(bold=True)
                    
                # Spaltenbreiten anpassen
                worksheet.column_dimensions['A'].width = 25
                worksheet.column_dimensions['B'].width = 80
                worksheet.column_dimensions['C'].width = 15
                
                print("Progressive Summaries erfolgreich exportiert")
            else:
                print("Keine Daten für Progressive Summaries verfügbar")
        
        except Exception as e:
            print(f"Fehler beim Export der progressiven Summaries: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()

    def _format_worksheet(self, worksheet, as_table: bool = False) -> None:
        """
        Formatiert das Detail-Worksheet mit flexibler Farbkodierung und adaptiven Spaltenbreiten
        für eine variable Anzahl von Attributen.
        """
        try:
            # Prüfe ob Daten vorhanden sind
            if worksheet.max_row < 2:
                print(f"Warnung: Worksheet '{worksheet.title}' enthält keine Daten")
                return

            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter
            from openpyxl.worksheet.table import Table, TableStyleInfo
            
            # Hole alle Zeilen als DataFrame für Farbzuordnung
            data = []
            headers = []
            for idx, row in enumerate(worksheet.iter_rows(values_only=True), 1):
                if idx == 1:
                    headers = list(row)
                else:
                    data.append(row)
            
            df = pd.DataFrame(data, columns=headers)
            
            # Zähle die Anzahl der Attributspalten (attribut1, attribut2, attribut3, ...)
            attribut_count = 0
            for header in headers:
                if header in self.attribute_labels.values() and header:
                    attribut_count += 1
            
            print(f"Erkannte Attributspalten: {attribut_count}")
            
            # Definiere Standardbreiten für verschiedene Spaltentypen
            width_defaults = {
                'dokument': 30,       # Dokument
                'attribut': 15,       # Attributspalte
                'chunk_nr': 10,       # Chunk_Nr
                'text': 40,           # Text, Paraphrase, etc.
                'kategorie': 20,      # Hauptkategorie
                'typ': 15,            # Kategorietyp
                'boolean': 5,         # Ja/Nein-Spalten
                'medium': 25,         # Mittellange Textfelder
                'large': 40,          # Lange Textfelder
                'default': 15         # Standardbreite
            }
            
            # Bestimme dynamisch, welche Spalte welche Breite erhält,
            # basierend auf den Spaltenüberschriften
            col_widths = {}
            
            for idx, header in enumerate(headers, 1):
                col_letter = get_column_letter(idx)
                
                # Dokumentspalte
                if header == 'Dokument':
                    col_widths[col_letter] = width_defaults['dokument']
                # Attributspalten
                elif header in self.attribute_labels.values() and header:
                    col_widths[col_letter] = width_defaults['attribut']
                # Chunk-Nummer
                elif header == 'Chunk_Nr':
                    col_widths[col_letter] = width_defaults['chunk_nr']
                # Text und ähnliche lange Felder
                elif header in ['Text', 'Paraphrase', 'Begründung', 'Textstellen', 
                            'Definition_Übereinstimmungen', 
                            'Progressive_Context', 'Context_Influence']:
                    col_widths[col_letter] = width_defaults['text']
                # Hauptkategorie
                elif header == 'Hauptkategorie':
                    col_widths[col_letter] = width_defaults['kategorie']
                # Kategorietyp
                elif header == 'Kategorietyp':
                    col_widths[col_letter] = width_defaults['typ']
                # Ja/Nein-Spalten
                elif header in ['Kodiert', 'Mehrfachkodierung']:
                    col_widths[col_letter] = width_defaults['boolean']
                # Mittellange Textfelder
                elif header in ['Subkategorien', 'Schlüsselwörter', 'Konfidenz', 'Konsenstyp']:
                    col_widths[col_letter] = width_defaults['medium']
                # Defaultwert für alle anderen
                else:
                    col_widths[col_letter] = width_defaults['default']
            
            # Setze die berechneten Spaltenbreiten
            for col_letter, width in col_widths.items():
                worksheet.column_dimensions[col_letter].width = width

            # Definiere Styles
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            header_font = Font(bold=True)
            header_fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')
            
            # Finde den Index der Hauptkategorie-Spalte dynamisch
            hauptkategorie_idx = None
            for idx, header in enumerate(headers, 1):
                if header == 'Hauptkategorie':
                    hauptkategorie_idx = idx
                    break

            # Wenn Hauptkategorie-Spalte nicht gefunden wurde, benutze Fallback-Methode
            if hauptkategorie_idx is None:
                print("Warnung: Spalte 'Hauptkategorie' nicht gefunden")
                # Versuche alternative Methode, falls der Header-Name unterschiedlich ist
                for idx, val in enumerate(worksheet[1], 1):
                    if val.value and 'kategorie' in str(val.value).lower():
                        hauptkategorie_idx = idx
                        break
            
            # Extrahiere eindeutige Hauptkategorien wenn möglich
            main_categories = set()
            if hauptkategorie_idx:
                for row in range(2, worksheet.max_row + 1):
                    category = worksheet.cell(row=row, column=hauptkategorie_idx).value
                    if category and category != "Nicht kodiert" and category != "Kein Kodierkonsens":
                        main_categories.add(category)
            
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

                    # Farbkodierung für Hauptkategorien mit flexibler Spaltenposition
                    if hauptkategorie_idx and cell.column == hauptkategorie_idx and cell.value in self.category_colors:
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

            print(f"Worksheet '{worksheet.title}' erfolgreich formatiert" + 
                (f" mit Farbkodierung für Hauptkategorien (Spalte {hauptkategorie_idx})" if hauptkategorie_idx else ""))
            
        except Exception as e:
            print(f"Fehler bei der Formatierung von {worksheet.title}: {str(e)}")
            import traceback
            traceback.print_exc()

    def _export_intercoder_analysis(self, writer, segment_codings: Dict[str, List[Dict]], reliability: float):
        """
        Exportiert die Intercoder-Analyse mit korrekter Behandlung von Mehrfachkodierung.
        ERWEITERT: Jetzt auch mit Subkategorien-Analyse.
        """
        try:
            if 'Intercoderanalyse' not in writer.sheets:
                writer.book.create_sheet('Intercoderanalyse')

            worksheet = writer.sheets['Intercoderanalyse']
            current_row = 1

            # 1. Überschrift und Gesamtreliabilität
            worksheet.cell(row=current_row, column=1, value="Intercoderanalyse (Mehrfachkodierung berücksichtigt)")
            worksheet.cell(row=current_row, column=1).font = Font(bold=True, size=14)
            current_row += 2
            
            worksheet.cell(row=current_row, column=1, value="Krippendorffs Alpha (Hauptkategorien):")
            worksheet.cell(row=current_row, column=2, value=round(reliability, 3))
            current_row += 2

            # 2. FILTER UND GRUPPIERUNG: Nach Basis-Segmenten
            print("\nBereite Intercoder-Analyse vor (Mehrfachkodierung)...")
            
            # Filtere ursprüngliche Kodierungen
            original_segment_codings = {}
            filtered_count = 0
            
            for segment_id, codings in segment_codings.items():
                original_codings = []
                
                for coding in codings:
                    if (not coding.get('manual_review', False) and 
                        not coding.get('consolidated_from_multiple', False) and
                        coding.get('coder_id', '') not in ['consensus', 'majority', 'review']):
                        original_codings.append(coding)
                    else:
                        filtered_count += 1
                
                if len(original_codings) >= 2:
                    original_segment_codings[segment_id] = original_codings
            
            # Gruppiere nach Basis-Segmenten
            base_segment_groups = defaultdict(list)
            
            for segment_id, codings in original_segment_codings.items():
                # Bestimme Basis-Segment-ID
                base_segment_id = None
                if codings:
                    base_segment_id = codings[0].get('Original_Chunk_ID', segment_id)
                    if not base_segment_id:
                        if '_inst_' in segment_id:
                            base_segment_id = segment_id.split('_inst_')[0]
                        elif segment_id.endswith('-1') or segment_id.endswith('-2'):
                            base_segment_id = segment_id.rsplit('-', 1)[0]
                        else:
                            base_segment_id = segment_id
                
                if base_segment_id:
                    base_segment_groups[base_segment_id].extend(codings)
            
            print(f"Basis-Segmente für Intercoder-Analyse: {len(base_segment_groups)}")
            print(f"Gefilterte Kodierungen: {filtered_count}")

            # 3. HAUPTKATEGORIEN-ANALYSE
            worksheet.cell(row=current_row, column=1, value="A. Basis-Segmente Hauptkategorien-Übereinstimmung")
            worksheet.cell(row=current_row, column=1).font = Font(bold=True)
            current_row += 1

            headers = [
                'Basis_Segment_ID',
                'Text (Auszug)', 
                'Kodierer',
                'Identifizierte Kategorien',
                'Übereinstimmung',
                'Details'
            ]
            
            # Header formatieren
            for col, header in enumerate(headers, 1):
                cell = worksheet.cell(row=current_row, column=col, value=header)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')
            current_row += 1

            # Analysiere jedes Basis-Segment für Hauptkategorien
            agreement_count = 0
            total_base_segments = 0
            
            for base_segment_id, all_codings in base_segment_groups.items():
                # Gruppiere nach Kodierern
                coder_categories = defaultdict(set)
                coder_details = defaultdict(list)
                
                for coding in all_codings:
                    coder_id = coding.get('coder_id', 'Unbekannt')
                    category = coding.get('category', '')
                    subcats = coding.get('subcategories', [])
                    
                    if category and category not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                        coder_categories[coder_id].add(category)
                        
                        # Details für Anzeige
                        if isinstance(subcats, (list, tuple)):
                            subcats_str = ', '.join(subcats) if subcats else '(keine)'
                        else:
                            subcats_str = str(subcats) if subcats else '(keine)'
                        
                        instance = coding.get('multiple_coding_instance', 1)
                        detail = f"Inst.{instance}: {category} [{subcats_str}]"
                        coder_details[coder_id].append(detail)
                
                # Nur Basis-Segmente mit mindestens 2 Kodierern
                if len(coder_categories) < 2:
                    continue
                    
                total_base_segments += 1
                
                # Bestimme Übereinstimmung: Alle Kodierer müssen dieselben Kategorien-Sets haben
                category_sets = list(coder_categories.values())
                all_identical = len(set(frozenset(s) for s in category_sets)) == 1
                
                if all_identical:
                    agreement = "✓ Vollständig"
                    agreement_count += 1
                else:
                    # Prüfe partielle Übereinstimmung
                    intersection = set.intersection(*category_sets) if category_sets else set()
                    if intersection:
                        agreement = f"◐ Teilweise ({len(intersection)} gemeinsam)"
                    else:
                        agreement = "✗ Keine Übereinstimmung"
                
                # Sammle alle identifizierten Kategorien
                all_categories = set()
                for cat_set in category_sets:
                    all_categories.update(cat_set)
                
                # Extrahiere Beispieltext
                text_sample = all_codings[0].get('text', '')[:200] + "..." if len(all_codings[0].get('text', '')) > 200 else all_codings[0].get('text', 'Text nicht verfügbar')
                
                # Formatiere Kodierer-Details
                coders_list = sorted(coder_categories.keys())
                details_text = []
                for coder in coders_list:
                    categories = ', '.join(sorted(coder_categories[coder]))
                    details = '; '.join(coder_details[coder])
                    details_text.append(f"{coder}: [{categories}] - {details}")
                
                # Zeile einfügen
                row_data = [
                    self._sanitize_text_for_excel(base_segment_id),
                    self._sanitize_text_for_excel(text_sample),
                    ', '.join(coders_list),
                    self._sanitize_text_for_excel(' | '.join(sorted(all_categories))),
                    agreement,
                    self._sanitize_text_for_excel('\n'.join(details_text))
                ]
                
                for col, value in enumerate(row_data, 1):
                    cell = worksheet.cell(row=current_row, column=col, value=value)
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
                    
                    # Farbkodierung basierend auf Übereinstimmung
                    if col == 5:  # Übereinstimmungs-Spalte
                        if agreement.startswith("✓"):
                            cell.fill = PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid')  # Hellgrün
                        elif agreement.startswith("◐"):
                            cell.fill = PatternFill(start_color='FFFF90', end_color='FFFF90', fill_type='solid')  # Hellgelb
                        else:
                            cell.fill = PatternFill(start_color='FFB6C1', end_color='FFB6C1', fill_type='solid')  # Hellrot
                
                current_row += 1

            # Statistik Hauptkategorien
            current_row += 1
            worksheet.cell(row=current_row, column=1, value="Hauptkategorien-Statistik (Basis-Segmente):")
            worksheet.cell(row=current_row, column=1).font = Font(bold=True)
            current_row += 1
            
            worksheet.cell(row=current_row, column=1, value="Basis-Segmente analysiert:")
            worksheet.cell(row=current_row, column=2, value=total_base_segments)
            current_row += 1
            
            worksheet.cell(row=current_row, column=1, value="Vollständige Übereinstimmung:")
            worksheet.cell(row=current_row, column=2, value=f"{agreement_count}/{total_base_segments}")
            worksheet.cell(row=current_row, column=3, value=f"{(agreement_count/total_base_segments)*100:.1f}%" if total_base_segments > 0 else "0%")
            current_row += 2

            # 4. NEU: SUBKATEGORIEN-ANALYSE
            worksheet.cell(row=current_row, column=1, value="B. Subkategorien-Übereinstimmung")
            worksheet.cell(row=current_row, column=1).font = Font(bold=True)
            current_row += 2

            # Header für Subkategorien-Analyse
            subcat_headers = [
                'Basis_Segment_ID',
                'Hauptkategorie', 
                'Kodierer',
                'Subkategorien',
                'Übereinstimmung',
                'Details'
            ]
            
            for col, header in enumerate(subcat_headers, 1):
                cell = worksheet.cell(row=current_row, column=col, value=header)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')
            current_row += 1

            # Analysiere Subkategorien für jedes Basis-Segment
            subcat_agreement_count = 0
            subcat_total_segments = 0
            
            for base_segment_id, all_codings in base_segment_groups.items():
                # Gruppiere nach Hauptkategorien
                main_categories = set()
                for coding in all_codings:
                    category = coding.get('category', '')
                    if category and category not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                        main_categories.add(category)
                
                # Für jede Hauptkategorie analysiere Subkategorien-Übereinstimmung
                for main_cat in main_categories:
                    # Sammle alle Kodierungen für diese Hauptkategorie
                    cat_codings = [c for c in all_codings if c.get('category') == main_cat]
                    
                    if len(cat_codings) < 2:
                        continue
                    
                    # Gruppiere Subkategorien nach Kodierern
                    coder_subcats = defaultdict(set)
                    for coding in cat_codings:
                        coder_id = coding.get('coder_id', 'Unbekannt')
                        subcats = coding.get('subcategories', [])
                        
                        if isinstance(subcats, (list, tuple)):
                            coder_subcats[coder_id].update(subcats)
                        elif isinstance(subcats, str) and subcats:
                            subcat_list = [s.strip() for s in subcats.split(',') if s.strip()]
                            coder_subcats[coder_id].update(subcat_list)
                    
                    if len(coder_subcats) < 2:
                        continue
                    
                    subcat_total_segments += 1
                    
                    # Bestimme Subkategorien-Übereinstimmung
                    subcat_sets = list(coder_subcats.values())
                    subcat_identical = len(set(frozenset(s) for s in subcat_sets)) == 1
                    
                    if subcat_identical:
                        subcat_agreement = "✓ Vollständig"
                        subcat_agreement_count += 1
                    else:
                        # Prüfe partielle Übereinstimmung
                        subcat_intersection = set.intersection(*subcat_sets) if subcat_sets else set()
                        if subcat_intersection:
                            subcat_agreement = f"◐ Teilweise ({len(subcat_intersection)} gemeinsam)"
                        else:
                            subcat_agreement = "✗ Keine Übereinstimmung"
                    
                    # Sammle alle Subkategorien
                    all_subcats = set()
                    for subcat_set in subcat_sets:
                        all_subcats.update(subcat_set)
                    
                    # Formatiere Details
                    coders_list = sorted(coder_subcats.keys())
                    subcat_details = []
                    for coder in coders_list:
                        subcats = ', '.join(sorted(coder_subcats[coder])) if coder_subcats[coder] else '(keine)'
                        subcat_details.append(f"{coder}: [{subcats}]")
                    
                    # Zeile einfügen
                    subcat_row_data = [
                        self._sanitize_text_for_excel(base_segment_id),
                        self._sanitize_text_for_excel(main_cat),
                        ', '.join(coders_list),
                        self._sanitize_text_for_excel(', '.join(sorted(all_subcats))),
                        subcat_agreement,
                        self._sanitize_text_for_excel('\n'.join(subcat_details))
                    ]
                    
                    for col, value in enumerate(subcat_row_data, 1):
                        cell = worksheet.cell(row=current_row, column=col, value=value)
                        cell.alignment = Alignment(wrap_text=True, vertical='top')
                        
                        # Farbkodierung basierend auf Übereinstimmung
                        if col == 5:  # Übereinstimmungs-Spalte
                            if subcat_agreement.startswith("✓"):
                                cell.fill = PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid')
                            elif subcat_agreement.startswith("◐"):
                                cell.fill = PatternFill(start_color='FFFF90', end_color='FFFF90', fill_type='solid')
                            else:
                                cell.fill = PatternFill(start_color='FFB6C1', end_color='FFB6C1', fill_type='solid')
                    
                    current_row += 1

            # Statistik Subkategorien
            current_row += 1
            worksheet.cell(row=current_row, column=1, value="Subkategorien-Statistik:")
            worksheet.cell(row=current_row, column=1).font = Font(bold=True)
            current_row += 1
            
            worksheet.cell(row=current_row, column=1, value="Hauptkategorie-Instanzen analysiert:")
            worksheet.cell(row=current_row, column=2, value=subcat_total_segments)
            current_row += 1
            
            worksheet.cell(row=current_row, column=1, value="Vollständige Subkat-Übereinstimmung:")
            worksheet.cell(row=current_row, column=2, value=f"{subcat_agreement_count}/{subcat_total_segments}")
            worksheet.cell(row=current_row, column=3, value=f"{(subcat_agreement_count/subcat_total_segments)*100:.1f}%" if subcat_total_segments > 0 else "0%")
            current_row += 2

            # 5. Kodierer-Übereinstimmungsmatrix (bestehend)
            worksheet.cell(row=current_row, column=1, value="C. Kodierer-Übereinstimmungsmatrix (Basis-Segmente)")
            worksheet.cell(row=current_row, column=1).font = Font(bold=True)
            current_row += 2

            # Extrahiere alle Kodierer
            all_coders = set()
            for codings in base_segment_groups.values():
                for coding in codings:
                    all_coders.add(coding.get('coder_id', 'Unbekannt'))
            
            coders = sorted(list(all_coders))
            
            # Matrix-Header
            worksheet.cell(row=current_row, column=1, value="Kodierer")
            for col, coder in enumerate(coders, 2):
                cell = worksheet.cell(row=current_row, column=col, value=coder)
                cell.font = Font(bold=True)
            current_row += 1

            # Berechne paarweise Übereinstimmungen basierend auf Basis-Segmenten
            for row_idx, coder1 in enumerate(coders):
                worksheet.cell(row=current_row, column=1, value=coder1)
                
                for col_idx, coder2 in enumerate(coders, 2):
                    if coder1 == coder2:
                        agreement_value = 1.0
                    else:
                        # Berechne Übereinstimmung zwischen coder1 und coder2 auf Basis-Segment-Ebene
                        common_base_segments = 0
                        agreements = 0
                        
                        for base_segment_id, all_codings in base_segment_groups.items():
                            # Sammle Kategorien beider Kodierer für dieses Basis-Segment
                            coder1_categories = set()
                            coder2_categories = set()
                            
                            for coding in all_codings:
                                coder_id = coding.get('coder_id', '')
                                category = coding.get('category', '')
                                
                                if category and category not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                                    if coder_id == coder1:
                                        coder1_categories.add(category)
                                    elif coder_id == coder2:
                                        coder2_categories.add(category)
                            
                            # Beide Kodierer müssen mindestens eine Kategorie haben
                            if coder1_categories and coder2_categories:
                                common_base_segments += 1
                                # Übereinstimmung wenn beide dieselben Kategorien-Sets haben
                                if coder1_categories == coder2_categories:
                                    agreements += 1
                        
                        agreement_value = agreements / common_base_segments if common_base_segments > 0 else 0.0
                    
                    cell = worksheet.cell(row=current_row, column=col_idx, value=agreement_value)
                    cell.number_format = '0.00'
                    
                    # Farbkodierung
                    if agreement_value >= 0.8:
                        cell.fill = PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid')
                    elif agreement_value >= 0.6:
                        cell.fill = PatternFill(start_color='FFFF90', end_color='FFFF90', fill_type='solid')
                    else:
                        cell.fill = PatternFill(start_color='FFB6C1', end_color='FFB6C1', fill_type='solid')
                
                current_row += 1

            # Erklärung hinzufügen
            current_row += 2
            worksheet.cell(row=current_row, column=1, value="Erklärung:")
            worksheet.cell(row=current_row, column=1).font = Font(bold=True)
            current_row += 1
            
            explanation_text = (
                "Diese Analyse berücksichtigt Mehrfachkodierung korrekt und analysiert sowohl:\n"
                "- Hauptkategorien-Übereinstimmung auf Basis-Segment-Ebene\n"
                "- Subkategorien-Übereinstimmung für jede Hauptkategorie\n"
                "- Kodierer stimmen überein, wenn sie dieselben Kategorien/Subkategorien identifizieren\n"
                "- Auch wenn Kategorien in verschiedenen Mehrfachkodierungs-Instanzen auftreten"
            )
            
            cell = worksheet.cell(row=current_row, column=1, value=explanation_text)
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            worksheet.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=6)

            # Spaltenbreiten anpassen
            worksheet.column_dimensions['A'].width = 30
            worksheet.column_dimensions['B'].width = 40
            worksheet.column_dimensions['C'].width = 20
            worksheet.column_dimensions['D'].width = 35
            worksheet.column_dimensions['E'].width = 20
            worksheet.column_dimensions['F'].width = 50

            print("Erweiterte Intercoder-Analyse (mit Subkategorien) erfolgreich exportiert")
            
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
    
    def _export_review_decisions(self, writer, review_decisions: List[Dict]):
        """
        Exportiert die manuellen Review-Entscheidungen in ein separates Excel-Sheet.
        Verbessert zur korrekten Erfassung aller Entscheidungen.
        
        Args:
            writer: Excel Writer Objekt
            review_decisions: Liste der manuellen Review-Entscheidungen
        """
        try:
            # Prüfen, ob Review-Entscheidungen vorhanden sind
            if not review_decisions:
                print("Keine manuellen Review-Entscheidungen zum Exportieren vorhanden")
                return
                
            print(f"\nExportiere {len(review_decisions)} manuelle Review-Entscheidungen...")
            
            # Erstelle Worksheet für Review-Entscheidungen falls es noch nicht existiert
            if 'Manuelle_Entscheidungen' not in writer.sheets:
                writer.book.create_sheet('Manuelle_Entscheidungen')
            
            worksheet = writer.sheets['Manuelle_Entscheidungen']
            
            # Lösche eventuell bestehende Daten
            if worksheet.max_row > 0:
                worksheet.delete_rows(1, worksheet.max_row)
            
            # Erstelle Daten für den Export
            review_data = []
            for decision in review_decisions:
                # Extrahiere wesentliche Informationen
                segment_id = decision.get('segment_id', '')
                text = decision.get('text', '')
                
                # Extrahiere Dokumentnamen und Metadaten
                doc_name, chunk_id = self._extract_doc_and_chunk_id(segment_id)
                attribut1, attribut2, attribut3 = self._extract_metadata(doc_name)
                
                # Extrahiere Kategorie und Subkategorien
                category = decision.get('category', '')
                subcategories = decision.get('subcategories', [])
                if isinstance(subcategories, tuple):
                    subcategories = list(subcategories)
                subcats_text = ', '.join(subcategories) if subcategories else ''
                
                # Extrahiere Kodierer und Review-Information
                coder_id = decision.get('coder_id', 'Unbekannt')
                original_coder = coder_id
                review_date = decision.get('review_date', '')
                review_justification = decision.get('review_justification', '')
                
                # Sammle Informationen über konkurrierende Kodierungen, falls verfügbar
                competing_codings = decision.get('competing_codings', [])
                competing_text = ""
                if competing_codings:
                    competing_lines = []
                    for comp_coding in competing_codings:
                        comp_coder = comp_coding.get('coder_id', 'Unbekannt')
                        comp_cat = comp_coding.get('category', '')
                        competing_lines.append(f"{comp_coder}: {comp_cat}")
                    competing_text = '; '.join(competing_lines)
                
                # Erstelle einen Eintrag für die Tabelle
                review_data.append({
                    'Dokument': self._sanitize_text_for_excel(doc_name),
                    self.attribute_labels['attribut1']: self._sanitize_text_for_excel(attribut1),
                    self.attribute_labels['attribut2']: self._sanitize_text_for_excel(attribut2),
                    'Chunk_Nr': f"{chunk_id}",
                    'Text': self._sanitize_text_for_excel(text[:500] + ('...' if len(text) > 500 else '')),
                    'Gewählte_Kategorie': self._sanitize_text_for_excel(category),
                    'Gewählte_Subkategorien': self._sanitize_text_for_excel(subcats_text),
                    'Ursprünglicher_Kodierer': self._sanitize_text_for_excel(original_coder),
                    'Review_Datum': review_date,
                    'Review_Begründung': self._sanitize_text_for_excel(review_justification),
                    'Konkurrierende_Kodierungen': self._sanitize_text_for_excel(competing_text)
                })
                
                # Füge attribut3 hinzu, wenn es in den Labels definiert und nicht leer ist
                if 'attribut3' in self.attribute_labels and self.attribute_labels['attribut3']:
                    review_data[-1][self.attribute_labels['attribut3']] = self._sanitize_text_for_excel(attribut3)
            
            # Erstelle DataFrame und exportiere in Excel
            if review_data:
                # Erstelle DataFrame
                df_review = pd.DataFrame(review_data)
                
                # Exportiere in das Excel-Sheet
                for r_idx, row in enumerate(dataframe_to_rows(df_review, index=False, header=True), 1):
                    for c_idx, value in enumerate(row, 1):
                        worksheet.cell(row=r_idx, column=c_idx, value=value)
                
                # Formatiere das Worksheet
                self._format_review_worksheet(worksheet)
                
                print(f"Manuelle Review-Entscheidungen exportiert: {len(review_data)} Einträge")
            else:
                # Wenn keine Daten vorhanden sind, füge mindestens einen Hinweis ein
                worksheet.cell(row=1, column=1, value="Keine manuellen Review-Entscheidungen vorhanden")
                
        except Exception as e:
            print(f"Fehler beim Export der Review-Entscheidungen: {str(e)}")
            import traceback
            traceback.print_exc()

    def _format_review_worksheet(self, worksheet) -> None:
        """
        Formatiert das Review-Entscheidungen Worksheet für bessere Lesbarkeit.
        
        Args:
            worksheet: Das zu formatierende Worksheet-Objekt
        """
        try:
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            
            # Definiere Spaltenbreiten
            column_widths = {
                'A': 30,  # Dokument
                'B': 15,  # Attribut1
                'C': 15,  # Attribut2
                'D': 10,  # Chunk_Nr
                'E': 50,  # Text
                'F': 25,  # Gewählte_Kategorie
                'G': 25,  # Gewählte_Subkategorien
                'H': 15,  # Ursprünglicher_Kodierer
                'I': 20,  # Review_Datum
                'J': 40,  # Review_Begründung
                'K': 40   # Konkurrierende_Kodierungen
            }
            
            # Setze Spaltenbreiten
            for i, width in enumerate(column_widths.values(), 1):
                col_letter = get_column_letter(i)
                worksheet.column_dimensions[col_letter].width = width
            
            # Formatiere Überschriften
            header_font = Font(bold=True)
            header_fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')
            
            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(wrap_text=True, vertical='center')
            
            # Formatiere Datenzeilen
            for row in worksheet.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
                    
            # Automatische Filterung aktivieren
            if worksheet.max_row > 1:
                worksheet.auto_filter.ref = f"A1:{get_column_letter(worksheet.max_column)}{worksheet.max_row}"
                
        except Exception as e:
            print(f"Warnung: Formatierung des Review-Worksheets fehlgeschlagen: {str(e)}")

    def _extract_doc_and_chunk_id(self, segment_id: str) -> tuple:
        """
        Extrahiert Dokumentnamen und Chunk-ID aus einer Segment-ID.
        
        Args:
            segment_id: Segment-ID im Format "dokument_chunk_X"
                
        Returns:
            tuple: (doc_name, chunk_id)
        """
        parts = segment_id.split('_chunk_')
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            # Fallback für ungültige Segment-IDs
            return segment_id, "unknown"



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

        # Prompt-Handler hinzufügen
        self.prompt_handler = QCAPrompts(
            forschungsfrage=FORSCHUNGSFRAGE,
            kodierregeln=KODIERREGELN,
            deduktive_kategorien=DEDUKTIVE_KATEGORIEN
        )


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
            prompt = self.prompt_handler.get_definition_enhancement_prompt(category)
            
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
            # prompt = self._get_subcategory_generation_prompt(category)
            prompt = self.prompt_handler.get_subcategory_generation_prompt(category)
            
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

    def clean_problematic_characters(self, text: str) -> str:
        """
        Bereinigt Text von problematischen Zeichen, die später beim Excel-Export
        zu Fehlern führen könnten.
        
        Args:
            text (str): Zu bereinigender Text
            
        Returns:
            str: Bereinigter Text
        """
        if not text:
            return ""
            
        # Entferne problematische Steuerzeichen
        import re
        cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFE\uFFFF]', '', text)
        
        # Ersetze bekannte problematische Sonderzeichen
        problematic_chars = ['☺', '☻', '♥', '♦', '♣', '♠']
        for char in problematic_chars:
            cleaned_text = cleaned_text.replace(char, '')
        
        return cleaned_text

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
        Enthält zusätzliche Bereinigung für problematische Zeichen.
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
                    # Hier bereinigen wir Steuerzeichen und problematische Zeichen
                    clean_text = self.clean_problematic_characters(text)
                    paragraphs.append(clean_text)
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
                        cell_text = cell.text.strip()
                        if cell_text:
                            # Auch hier bereinigen
                            clean_text = self.clean_problematic_characters(cell_text)
                            table_texts.append(clean_text)
                            print(f"    Zelleninhalt gefunden: {len(cell_text)} Zeichen")
            
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
        """
        Liest eine PDF-Datei ein und extrahiert den Text mit verbesserter Fehlerbehandlung.
        
        Args:
            filepath: Pfad zur PDF-Datei
                
        Returns:
            str: Extrahierter und bereinigter Text
        """
        try:
            import PyPDF2
            print(f"\nLese PDF: {os.path.basename(filepath)}")
            
            text_content = []
            with open(filepath, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    print(f"  Gefundene Seiten: {total_pages}")
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                # Bereinige den Text von problematischen Zeichen
                                cleaned_text = self.clean_problematic_characters(page_text)
                                text_content.append(cleaned_text)
                                print(f"  Seite {page_num}/{total_pages}: {len(cleaned_text)} Zeichen extrahiert")
                            else:
                                print(f"  Seite {page_num}/{total_pages}: Kein Text gefunden")
                                
                        except Exception as page_error:
                            print(f"  Fehler bei Seite {page_num}: {str(page_error)}")
                            # Versuche es mit alternativer Methode
                            try:
                                # Fallback: Extrahiere einzelne Textfragmente
                                print("  Versuche alternative Extraktionsmethode...")
                                if hasattr(page, 'extract_text'):
                                    fragments = []
                                    for obj in page.get_text_extraction_elements():
                                        if hasattr(obj, 'get_text'):
                                            fragments.append(obj.get_text())
                                    if fragments:
                                        fallback_text = ' '.join(fragments)
                                        cleaned_text = self.clean_problematic_characters(fallback_text)
                                        text_content.append(cleaned_text)
                                        print(f"  Alternative Methode erfolgreich: {len(cleaned_text)} Zeichen")
                            except:
                                print("  Alternative Methode fehlgeschlagen")
                                continue
                    
                except PyPDF2.errors.PdfReadError as pdf_error:
                    print(f"  PDF Lesefehler: {str(pdf_error)}")
                    print("  Versuche Fallback-Methode...")
                    
                    # Fallback-Methode, wenn direkte Lesemethode fehlschlägt
                    try:
                        from pdf2image import convert_from_path
                        from pytesseract import image_to_string
                        
                        print("  Verwende OCR-Fallback via pdf2image und pytesseract")
                        # Konvertiere PDF-Seiten zu Bildern
                        images = convert_from_path(filepath)
                        
                        for i, image in enumerate(images):
                            try:
                                # Extrahiere Text über OCR
                                ocr_text = image_to_string(image, lang='deu')
                                if ocr_text:
                                    # Bereinige den Text
                                    cleaned_text = self.clean_problematic_characters(ocr_text)
                                    text_content.append(cleaned_text)
                                    print(f"  OCR Seite {i+1}: {len(cleaned_text)} Zeichen extrahiert")
                                else:
                                    print(f"  OCR Seite {i+1}: Kein Text gefunden")
                            except Exception as ocr_error:
                                print(f"  OCR-Fehler bei Seite {i+1}: {str(ocr_error)}")
                                continue
                    
                    except ImportError:
                        print("  OCR-Fallback nicht verfügbar. Bitte installieren Sie pdf2image und pytesseract")
                    
            # Zusammenfassen des extrahierten Textes
            if text_content:
                full_text = '\n'.join(text_content)
                print(f"\nErgebnis:")
                print(f"  ✓ {len(text_content)} Textabschnitte extrahiert")
                print(f"  ✓ Gesamtlänge: {len(full_text)} Zeichen")
                return full_text
            else:
                print("\n✗ Kein Text aus PDF extrahiert")
                return ""
                
        except ImportError:
            print("PyPDF2 nicht installiert. Bitte installieren Sie: pip install PyPDF2")
            raise
        except Exception as e:
            print(f"Fehler beim PDF-Lesen: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""  # Leerer String im Fehlerfall, damit der Rest funktioniert
    
    def _extract_metadata(self, filename: str) -> Tuple[str, str, str]:
        """
        Extrahiert Metadaten aus dem Dateinamen.
        Erwartet Format: attribut1_attribut2_attribut3.extension
        
        Args:
            filename (str): Name der Datei
            
        Returns:
            Tuple[str, str, str]: (attribut1, attribut2, attribut3)
        """
        try:
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split('_')
            
            # Extrahiere bis zu drei Attribute, wenn verfügbar
            attribut1 = parts[0] if len(parts) >= 1 else ""
            attribut2 = parts[1] if len(parts) >= 2 else ""
            attribut3 = parts[2] if len(parts) >= 3 else ""
            
            return attribut1, attribut2, attribut3
        except Exception as e:
            print(f"Fehler beim Extrahieren der Metadaten aus {filename}: {str(e)}")
            return filename, "", ""





# --- Hilfsfunktionen ---


async def perform_manual_coding(chunks, categories, manual_coders):
    """
    Führt die manuelle Kodierung durch und bereitet Ergebnisse für späteren Review vor.
    Stellt sicher, dass der Prozess nach dem letzten Segment sauber beendet wird.
    
    Args:
        chunks: Dictionary mit Chunks für jedes Dokument
        categories: Kategoriensystem
        manual_coders: Liste der manuellen Kodierer
        
    Returns:
        list: Manuelle Kodierungen
    """
    manual_codings = []
    total_segments = sum(len(chunks[doc]) for doc in chunks)
    processed_segments = 0
    
    # Erstelle eine flache Liste aller zu kodierenden Segmente
    all_segments = []
    for document_name, document_chunks in chunks.items():
        for chunk_id, chunk in enumerate(document_chunks):
            all_segments.append((document_name, chunk_id, chunk))
    
    print(f"\nManuelles Kodieren: Insgesamt {total_segments} Segmente zu kodieren")
    
    try:
        # Verarbeite alle Segmente
        for idx, (document_name, chunk_id, chunk) in enumerate(all_segments):
            processed_segments += 1
            progress_percentage = (processed_segments / total_segments) * 100
            
            print(f"\nManuelles Codieren: Dokument {document_name}, "
                  f"Chunk {chunk_id + 1}/{len(chunks[document_name])} "
                  f"(Gesamt: {processed_segments}/{total_segments}, {progress_percentage:.1f}%)")
            
            # Prüfe, ob es das letzte Segment ist
            last_segment = (processed_segments == total_segments)
            
            for coder_idx, manual_coder in enumerate(manual_coders):
                try:
                    # Informiere den Benutzer über den Fortschritt
                    if last_segment:
                        print(f"Dies ist das letzte zu kodierende Segment!")
                    
                    # Übergabe des last_segment Parameters an die code_chunk Methode
                    coding_result = await manual_coder.code_chunk(chunk, categories, is_last_segment=last_segment)
                    
                    if coding_result == "ABORT_ALL":
                        print("Manuelles Kodieren wurde vom Benutzer abgebrochen.")
                        
                        # Schließe alle verbliebenen GUI-Fenster
                        for coder in manual_coders:
                            if hasattr(coder, 'root') and coder.root:
                                try:
                                    coder.root.quit()
                                    coder.root.destroy()
                                except:
                                    pass
                        
                        return manual_codings
                        
                    if coding_result:
                        # Erstelle ein detailliertes Dictionary für die spätere Verarbeitung
                        coding_entry = {
                            'segment_id': f"{document_name}_chunk_{chunk_id}",
                            'coder_id': manual_coder.coder_id,
                            'category': coding_result.category,
                            'subcategories': coding_result.subcategories,
                            'confidence': coding_result.confidence,
                            'justification': coding_result.justification,
                            'text': chunk,  # Wichtig: Den vollständigen Text speichern!
                            'document_name': document_name,
                            'chunk_id': chunk_id,
                            'manual_coding': True,  # Markierung für manuelle Kodierung
                            'coding_date': datetime.now().isoformat()
                        }
                        
                        # Füge weitere CodingResult-Attribute hinzu, falls vorhanden
                        if hasattr(coding_result, 'paraphrase') and coding_result.paraphrase:
                            coding_entry['paraphrase'] = coding_result.paraphrase
                            
                        if hasattr(coding_result, 'keywords') and coding_result.keywords:
                            coding_entry['keywords'] = coding_result.keywords
                            
                        if hasattr(coding_result, 'text_references') and coding_result.text_references:
                            coding_entry['text_references'] = list(coding_result.text_references)
                            
                        if hasattr(coding_result, 'uncertainties') and coding_result.uncertainties:
                            coding_entry['uncertainties'] = list(coding_result.uncertainties)
                        
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
    
        print("\n✅ Manueller Kodierungsprozess abgeschlossen")
        print(f"- {len(manual_codings)}/{total_segments} Segmente erfolgreich kodiert")
        
        # Sicherstellen, dass alle Fenster geschlossen sind
        for coder in manual_coders:
            if hasattr(coder, 'root') and coder.root:
                try:
                    coder.root.quit()
                    coder.root.destroy()
                    coder.root = None
                except:
                    pass
    
    except Exception as e:
        print(f"Fehler im manuellen Kodierungsprozess: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Versuche, alle Fenster zu schließen, selbst im Fehlerfall
        for coder in manual_coders:
            if hasattr(coder, 'root') and coder.root:
                try:
                    coder.root.quit()
                    coder.root.destroy()
                    coder.root = None
                except:
                    pass
    
    return manual_codings

async def perform_manual_review(segment_codings, output_dir):
    """
    Führt den manuellen Review für Segmente mit Kodierungsunstimmigkeiten durch.
    KORRIGIERT: Führt auch zu Zusammenführung identischer Hauptcodes pro Chunk.
    
    Args:
        segment_codings: Dictionary mit Segment-ID und zugehörigen Kodierungen
        output_dir: Verzeichnis für Exportdaten
        
    Returns:
        list: Liste der finalen Review-Entscheidungen (eine pro Segment)
    """
    review_component = ManualReviewComponent(output_dir)
    
    # Führe den manuellen Review durch
    raw_review_decisions = await review_component.review_discrepancies(segment_codings)
    
    # NEUE LOGIK: Führe Hauptcode-Zusammenführung durch
    print(f"\nVerarbeite {len(raw_review_decisions)} manuelle Review-Entscheidungen...")
    
    # Gruppiere Review-Entscheidungen nach Segment-ID
    segment_decisions = {}
    for decision in raw_review_decisions:
        segment_id = decision.get('segment_id', '')
        if segment_id:
            if segment_id not in segment_decisions:
                segment_decisions[segment_id] = []
            segment_decisions[segment_id].append(decision)
    
    # Erstelle finale Review-Entscheidungen (eine pro Segment)
    final_review_decisions = []
    consolidation_stats = {
        'segments_with_multiple_decisions': 0,
        'segments_consolidated': 0,
        'total_decisions_before': len(raw_review_decisions),
        'total_decisions_after': 0
    }
    
    for segment_id, decisions in segment_decisions.items():
        if len(decisions) > 1:
            consolidation_stats['segments_with_multiple_decisions'] += 1
            
            # Prüfe ob alle Entscheidungen dieselbe Hauptkategorie haben
            main_categories = [d.get('category', '') for d in decisions]
            unique_categories = set(main_categories)
            
            if len(unique_categories) == 1:
                # Alle haben dieselbe Hauptkategorie - konsolidiere Subkategorien
                main_category = list(unique_categories)[0]
                print(f"  Konsolidiere {len(decisions)} Entscheidungen für Segment {segment_id} (Hauptkategorie: {main_category})")
                
                # Sammle alle Subkategorien
                all_subcategories = []
                all_justifications = []
                highest_confidence_decision = None
                max_confidence = 0
                
                for decision in decisions:
                    # Subkategorien sammeln
                    subcats = decision.get('subcategories', [])
                    if isinstance(subcats, (list, tuple)):
                        all_subcategories.extend(subcats)
                    
                    # Begründungen sammeln
                    justification = decision.get('justification', '')
                    if justification:
                        all_justifications.append(justification)
                    
                    # Höchste Konfidenz finden
                    confidence = decision.get('confidence', {})
                    if isinstance(confidence, dict):
                        total_conf = confidence.get('total', 0)
                        if total_conf > max_confidence:
                            max_confidence = total_conf
                            highest_confidence_decision = decision
                
                # Erstelle konsolidierte Entscheidung
                if highest_confidence_decision:
                    consolidated_decision = highest_confidence_decision.copy()
                    
                    # Aktualisiere mit konsolidierten Daten
                    consolidated_decision['subcategories'] = list(set(all_subcategories))  # Entferne Duplikate
                    consolidated_decision['justification'] = f"[Konsolidiert aus {len(decisions)} manuellen Entscheidungen] " + "; ".join(set(all_justifications))
                    consolidated_decision['manual_review'] = True
                    consolidated_decision['consolidated_from_multiple'] = True
                    consolidated_decision['original_decision_count'] = len(decisions)
                    
                    # Hole ursprüngliche Kodierungen für Kontext
                    original_codings = segment_codings.get(segment_id, [])
                    consolidated_decision['competing_codings'] = original_codings
                    
                    # Extrahiere Text aus ursprünglichen Kodierungen
                    if original_codings:
                        consolidated_decision['text'] = original_codings[0].get('text', '')
                    
                    final_review_decisions.append(consolidated_decision)
                    consolidation_stats['segments_consolidated'] += 1
                    
                    print(f"    ✓ Konsolidiert zu: {main_category} mit {len(consolidated_decision['subcategories'])} Subkategorien")
                    
            else:
                # Verschiedene Hauptkategorien - das sollte eigentlich nicht passieren bei manuellem Review
                print(f"  Warnung: Verschiedene Hauptkategorien für Segment {segment_id}: {unique_categories}")
                # Nimm die erste Entscheidung als Fallback
                decision = decisions[0]
                decision['competing_codings'] = segment_codings.get(segment_id, [])
                if segment_codings.get(segment_id):
                    decision['text'] = segment_codings[segment_id][0].get('text', '')
                final_review_decisions.append(decision)
        else:
            # Nur eine Entscheidung für dieses Segment
            decision = decisions[0]
            # Erweitere die Entscheidung um ursprüngliche Kodierungen
            original_codings = segment_codings.get(segment_id, [])
            decision['competing_codings'] = original_codings
            decision['manual_review'] = True
            
            # Extrahiere den Text aus einer der ursprünglichen Kodierungen
            if original_codings:
                decision['text'] = original_codings[0].get('text', '')
            
            final_review_decisions.append(decision)
    
    # Aktualisiere Statistiken
    consolidation_stats['total_decisions_after'] = len(final_review_decisions)
    
    # Zeige Konsolidierungsstatistiken
    if consolidation_stats['segments_consolidated'] > 0:
        print(f"\nKonsolidierungsstatistiken:")
        print(f"- Segmente mit mehreren Entscheidungen: {consolidation_stats['segments_with_multiple_decisions']}")
        print(f"- Davon konsolidiert: {consolidation_stats['segments_consolidated']}")
        print(f"- Entscheidungen vor Konsolidierung: {consolidation_stats['total_decisions_before']}")
        print(f"- Finale Entscheidungen: {consolidation_stats['total_decisions_after']}")
    
    return final_review_decisions


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

        # Mehrfachkodierungs-Konfiguration anzeigen
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    MEHRFACHKODIERUNG                         ║
╠══════════════════════════════════════════════════════════════╣
║ Status: {'✓ AKTIVIERT' if CONFIG.get('MULTIPLE_CODINGS', True) else '✗ DEAKTIVIERT'}                                   ║
║ Schwellenwert: {CONFIG.get('MULTIPLE_CODING_THRESHOLD', 0.6):.1%} Relevanz                        ║
║ Verhalten: Segmente werden mehrfach kodiert wenn sie         ║
║           >= {CONFIG.get('MULTIPLE_CODING_THRESHOLD', 0.6):.0%} Relevanz für verschiedene Hauptkategorien   ║
║           haben                                              ║
╚══════════════════════════════════════════════════════════════╝""")
       

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
            default_mode = CONFIG['ANALYSIS_MODE']
            print("\nAktueller Analysemodus aus Codebook: {default_mode}")
            print("Sie haben 10 Sekunden Zeit für die Eingabe.")
            print("Optionen:")
            print("1 = full (volle induktive Analyse)")
            print("2 = abductive (nur Subkategorien entwickeln)")
            print("3 = deductive (nur deduktiv)")
            print("4 = grounded (Subkategorien sammeln, später Hauptkategorien generieren)")

            analysis_mode = get_input_with_timeout(
                f"\nWelchen Analysemodus möchten Sie verwenden? [1/2/3/4] (Standard: {CONFIG['ANALYSIS_MODE']})", 
                timeout=10
            )

            # Mapping von Zahlen zu Modi
            mode_mapping = {
                '1': 'full',
                '2': 'abductive',
                '3': 'deductive',
                '4': 'grounded'
            }

            # Verarbeite Zahlen oder direkte Modusangaben, behalte Default wenn leere oder ungültige Eingabe
            if analysis_mode:  # Nur wenn etwas eingegeben wurde
                if analysis_mode in mode_mapping:
                    CONFIG['ANALYSIS_MODE'] = mode_mapping[analysis_mode]
                elif analysis_mode.lower() in mode_mapping.values():
                    CONFIG['ANALYSIS_MODE'] = analysis_mode.lower()
                else:
                    print(f"\nUngültiger Modus '{analysis_mode}'. Verwende Default-Modus '{default_mode}'.")
                    # Keine Änderung an CONFIG['ANALYSIS_MODE'], Default bleibt bestehen
            else:
                print(f"Keine Eingabe. Verwende Default-Modus '{default_mode}'.")

            # Bestimme, ob induktive Analyse übersprungen wird
            skip_inductive = CONFIG['ANALYSIS_MODE'] == 'deductive'

            print(f"\nAnalysemodus: {CONFIG['ANALYSIS_MODE']} {'(Skip induktiv)' if skip_inductive else ''}")

            # Bei Modus 'grounded' zusätzliche Informationen anzeigen
            if CONFIG['ANALYSIS_MODE'] == 'grounded':
                print("""
            Grounded Theory Modus ausgewählt:
            - Zunächst werden Subcodes und Keywords gesammelt, ohne Hauptkategorien zu bilden
            - Erst nach Abschluss aller Segmente werden die Hauptkategorien generiert
            - Die Subcodes werden anhand ihrer Keywords zu thematisch zusammenhängenden Hauptkategorien gruppiert
            - Im Export werden diese als 'grounded' (statt 'induktiv') markiert
            """)

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
            
            # Verwende die verbesserte perform_manual_coding Funktion
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
            
            # Stelle sicher, dass alle Kodierer-Fenster geschlossen sind
            for coder in manual_coders:
                if hasattr(coder, 'root') and coder.root:
                    try:
                        coder.root.quit()
                        coder.root.destroy()
                        coder.root = None
                    except:
                        pass


        # 8. Integrierte Analyse starten
        print("\n7. Starte integrierte Analyse...")

        # Zeige Kontext-Modus an
        print(f"\nKodierungsmodus: {'Mit progressivem Kontext' if CONFIG.get('CODE_WITH_CONTEXT', True) else 'Ohne Kontext'}")
        
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

            # NEU: Vorbereitung für manuellen Review
            # Gruppiere Kodierungen nach Segmenten für Review
            segment_codings = {}
            for coding in all_codings:
                segment_id = coding.get('segment_id')
                if segment_id:
                    if segment_id not in segment_codings:
                        segment_codings[segment_id] = []
                    segment_codings[segment_id].append(coding)

            
            review_mode = CONFIG.get('REVIEW_MODE', 'consensus')
            print(f"\nKonfigurierter Review-Modus: {review_mode}")

            # Bestimme ob manuelles Review durchgeführt werden soll
            should_perform_manual_review = False

            if manual_coders and review_mode == 'manual':
                # Manuelle Kodierer vorhanden UND manueller Review explizit konfiguriert
                should_perform_manual_review = True
                print("Manueller Review aktiviert: Manuelle Kodierer vorhanden und REVIEW_MODE = 'manual'")
            elif not manual_coders and review_mode == 'manual':
                # Nur automatische Kodierer, aber manueller Review explizit konfiguriert
                should_perform_manual_review = True
                print("Manueller Review aktiviert: REVIEW_MODE = 'manual' für automatische Kodierungen")
            elif manual_coders and review_mode in ['consensus', 'majority', 'auto']:
                # Manuelle Kodierer vorhanden, aber anderer Review-Modus konfiguriert
                should_perform_manual_review = False
                print(f"Kein manueller Review: Manuelle Kodierer vorhanden, aber REVIEW_MODE = '{review_mode}'")
            else:
                # Alle anderen Fälle: Verwende konfigurierten Modus
                should_perform_manual_review = False
                print(f"Verwende {review_mode}-Modus für Kodierungsentscheidungen")

            # Führe manuellen Review durch wenn konfiguriert
            if should_perform_manual_review:
                print("\nStarte manuelles Review für Kodierungsunstimmigkeiten...")
                review_decisions = await perform_manual_review(segment_codings, CONFIG['OUTPUT_DIR'])
                
                if review_decisions:
                    print(f"\n{len(review_decisions)} manuelle Review-Entscheidungen getroffen")
                    
                    # Entferne alte Kodierungen für Segmente mit Review-Entscheidung
                    reviewed_segments = set(decision.get('segment_id', '') for decision in review_decisions)
                    all_codings = [coding for coding in all_codings if coding.get('segment_id', '') not in reviewed_segments]
                    
                    # Füge Review-Entscheidungen hinzu
                    all_codings.extend(review_decisions)
                    print(f"Aktualisierte Gesamtzahl Kodierungen: {len(all_codings)}")
                else:
                    print("Keine Review-Entscheidungen getroffen - verwende automatische Konsensbildung")
            else:
                print(f"\nVerwende {review_mode}-Modus für Kodierungsentscheidungen im ResultsExporter")
                # Die bestehende Consensus/Majority-Logik wird im ResultsExporter verwendet
            
            

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
            # Hier die Zusammenfassung der finalen Kategorien vor dem Speichern:
            print("\nFinales Kategoriensystem komplett:")
            print(f"- Insgesamt {len(final_categories)} Hauptkategorien")
            print(f"- Davon {len(final_categories) - len(initial_categories)} neu entwickelt")
            
            # Zähle Subkategorien für zusammenfassende Statistik
            total_subcats = sum(len(cat.subcategories) for cat in final_categories.values())
            print(f"- Insgesamt {total_subcats} Subkategorien")
            
            # 10. Speichere induktiv erweitertes Codebook
            if final_categories:
                category_manager = CategoryManager(CONFIG['OUTPUT_DIR'])
                category_manager.save_codebook(
                    categories=final_categories,
                    filename="codebook_inductive.json"
                )
                print(f"\nCodebook erfolgreich gespeichert mit {len(final_categories)} Hauptkategorien und {total_subcats} Subkategorien")

            # 11. Export der Ergebnisse
            print("\n9. Exportiere Ergebnisse...")
            if all_codings:
                exporter = ResultsExporter(
                    output_dir=CONFIG['OUTPUT_DIR'],
                    attribute_labels=CONFIG['ATTRIBUTE_LABELS'],
                    analysis_manager=analysis_manager,
                    inductive_coder=reliability_calculator
                )
                
                # Exportiere Ergebnisse mit Document-Summaries, wenn vorhanden
                summary_arg = analysis_manager.document_summaries if CONFIG.get('CODE_WITH_CONTEXT', True) else None

                # Bestimme den Export-Modus basierend auf REVIEW_MODE
                export_mode = CONFIG.get('REVIEW_MODE', 'consensus')

                # Validiere und mappe den Export-Modus
                if export_mode == 'auto':
                    export_mode = 'consensus'  # 'auto' ist ein Alias für 'consensus'
                elif export_mode not in ['consensus', 'majority', 'manual_priority']:
                    print(f"Warnung: Unbekannter REVIEW_MODE '{export_mode}', verwende 'consensus'")
                    export_mode = 'consensus'

                print(f"Export wird mit Modus '{export_mode}' durchgeführt")

                await exporter.export_results(
                    codings=all_codings,
                    reliability=reliability,
                    categories=final_categories,
                    chunks=chunks,
                    revision_manager=revision_manager,
                    export_mode=export_mode,
                    original_categories=initial_categories,
                    inductive_coder=reliability_calculator,
                    document_summaries=summary_arg
                )

                # Ausgabe der finalen Summaries, wenn vorhanden
                if CONFIG.get('CODE_WITH_CONTEXT', True) and analysis_manager.document_summaries:
                    print("\nFinale Document-Summaries:")
                    for doc_name, summary in analysis_manager.document_summaries.items():
                        print(f"\n📄 {doc_name}:")
                        print(f"  {summary}")

                print("Export erfolgreich abgeschlossen")
            else:
                print("Keine Kodierungen zum Exportieren vorhanden")

            # 12. Zeige finale Statistiken
            print("\nAnalyse abgeschlossen:")
            print(analysis_manager.get_analysis_report())

            if CONFIG.get('MULTIPLE_CODINGS', True) and all_codings:
                multiple_coding_stats = _calculate_multiple_coding_stats(all_codings)
                print(f"""
                    Mehrfachkodierungs-Statistiken:
                    - Segmente mit Mehrfachkodierung: {multiple_coding_stats['segments_with_multiple']}
                    - Durchschnittliche Kodierungen pro Segment: {multiple_coding_stats['avg_codings_per_segment']:.2f}
                    - Häufigste Kategorie-Kombinationen: {', '.join(multiple_coding_stats['top_combinations'][:3])}
                    - Fokus-Adherence Rate: {multiple_coding_stats['focus_adherence_rate']:.1%}""")
            
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

_patch_tkinter_for_threaded_exit()

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
