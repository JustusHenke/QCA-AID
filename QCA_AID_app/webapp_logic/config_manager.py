"""
Configuration Manager
=====================
Manages loading, saving, and validation of QCA-AID configuration.
Integrates with existing ConfigLoader and ConfigSynchronizer.
"""

import sys
import os
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import streamlit as st
from functools import lru_cache
import hashlib
import json

# Add parent directory to path to access QCA_AID_assets
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import existing QCA-AID components
from QCA_AID_assets.core.config import CONFIG as DEFAULT_CONFIG
from QCA_AID_assets.utils.config.loader import ConfigLoader
from QCA_AID_assets.utils.config.converter import ConfigConverter

# Import webapp models
from webapp_models.config_data import ConfigData, CoderSetting


class ConfigManager:
    """
    Verwaltet QCA-AID Konfiguration für die Webapp.
    
    Responsibilities:
    - Lädt und speichert Konfigurationen (XLSX und JSON)
    - Validiert Konfigurationswerte
    - Nutzt bestehende ConfigLoader und ConfigSynchronizer
    - Verwaltet Konfigurationszustand
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """
    
    def __init__(self, project_dir: Optional[str] = None):
        """
        Initialisiert ConfigManager.
        
        Args:
            project_dir: Projektverzeichnis (Standard: aktuelles Verzeichnis)
        """
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.xlsx_path = self.project_dir / "QCA-AID-Codebook.xlsx"
        self.json_path = self.project_dir / "QCA-AID-Codebook.json"
    
    @st.cache_data(ttl=300, show_spinner=False)
    def _load_config_cached(_self, file_path: str, file_hash: str, format: str) -> Tuple[bool, Optional[Dict], List[str]]:
        """
        Cached config loading helper.
        
        Performance Optimization: Caches loaded configurations for 5 minutes (300s).
        Uses file hash to invalidate cache when file changes.
        
        Args:
            file_path: Path to config file
            file_hash: Hash of file content for cache invalidation
            format: Format ('xlsx' or 'json')
            
        Returns:
            Tuple[bool, Optional[Dict], List[str]]: (success, config_dict, errors)
        """
        if format == 'json':
            return _self._load_from_json(file_path)
        elif format == 'xlsx':
            return _self._load_from_xlsx(file_path)
        else:
            return False, None, [f"Ungültiges Format: {format}"]
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        Computes hash of file for cache invalidation.
        
        Args:
            file_path: Path to file
            
        Returns:
            str: MD5 hash of file content
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            # Return timestamp as fallback
            return str(Path(file_path).stat().st_mtime)
    
    def load_config(self, file_path: Optional[str] = None, format: Optional[str] = None) -> Tuple[bool, Optional[ConfigData], List[str]]:
        """
        Lädt Konfiguration aus XLSX oder JSON mit Caching.
        
        Requirement 2.1: WHEN ein Benutzer auf "Konfiguration laden" klickt 
                        THEN das System SHALL sowohl XLSX als auch JSON Dateien akzeptieren
        Requirement 2.2: WHEN eine Konfigurationsdatei geladen wird 
                        THEN das System SHALL alle Einstellungen in die entsprechenden UI-Elemente übertragen
        
        Performance Optimization: Uses caching to avoid re-loading unchanged files.
        
        Args:
            file_path: Pfad zur Konfigurationsdatei (optional)
            format: Format ('xlsx' oder 'json', wird automatisch erkannt wenn None)
            
        Returns:
            Tuple[bool, Optional[ConfigData], List[str]]: 
                (success, config_data, error_messages)
        """
        errors = []
        
        try:
            # Bestimme Dateipfad und Format
            if file_path:
                file_path = Path(file_path)
                if not file_path.exists():
                    return False, None, [f"Datei nicht gefunden: {file_path}"]
                
                # Erkenne Format aus Dateiendung wenn nicht angegeben
                if format is None:
                    if file_path.suffix.lower() in ['.xlsx', '.xls']:
                        format = 'xlsx'
                    elif file_path.suffix.lower() == '.json':
                        format = 'json'
                    else:
                        return False, None, [f"Unbekanntes Dateiformat: {file_path.suffix}"]
            else:
                # Verwende Standard-Pfade
                if self.json_path.exists():
                    file_path = self.json_path
                    format = 'json'
                elif self.xlsx_path.exists():
                    file_path = self.xlsx_path
                    format = 'xlsx'
                else:
                    return False, None, ["Keine Konfigurationsdatei gefunden"]
            
            # Lade basierend auf Format mit Caching
            file_hash = self._get_file_hash(str(file_path))
            success, config_dict, load_errors = self._load_config_cached(
                str(file_path), 
                file_hash, 
                format
            )
            
            if not success:
                return False, None, load_errors
            
            # Konvertiere zu ConfigData
            try:
                config_data = self._dict_to_config_data(config_dict)
                
                # Validiere Konfiguration
                is_valid, validation_errors = config_data.validate()
                if not is_valid:
                    errors.extend(validation_errors)
                    # Gebe trotzdem config_data zurück, damit UI Werte anzeigen kann
                    return False, config_data, errors
                
                return True, config_data, []
                
            except Exception as e:
                return False, None, [f"Fehler beim Konvertieren der Konfiguration: {str(e)}"]
                
        except Exception as e:
            return False, None, [f"Fehler beim Laden der Konfiguration: {str(e)}"]
    
    def save_config(self, config: ConfigData, file_path: Optional[str] = None, format: str = 'json') -> Tuple[bool, List[str]]:
        """
        Speichert Konfiguration in XLSX oder JSON.
        
        Requirement 2.3: WHEN ein Benutzer auf "Konfiguration speichern" klickt 
                        THEN das System SHALL ein Dropdown-Menü mit Optionen für XLSX und JSON anzeigen
        Requirement 2.4: WHEN eine Konfiguration gespeichert wird 
                        THEN das System SHALL die Datei im gewählten Format im Projektverzeichnis speichern
        
        Args:
            config: ConfigData Objekt zum Speichern
            file_path: Pfad zur Zieldatei (optional, verwendet Standard-Pfad)
            format: Format ('xlsx' oder 'json')
            
        Returns:
            Tuple[bool, List[str]]: (success, error_messages)
        """
        errors = []
        
        try:
            # Validiere Konfiguration vor dem Speichern
            is_valid, validation_errors = config.validate()
            if not is_valid:
                return False, validation_errors
            
            # Bestimme Dateipfad
            if file_path:
                file_path = Path(file_path)
                # Wenn relativer Pfad (inkl. nur Dateiname), verwende Projektverzeichnis als Basis
                if not file_path.is_absolute():
                    file_path = self.project_dir / file_path
            else:
                if format == 'json':
                    file_path = self.json_path
                elif format == 'xlsx':
                    file_path = self.xlsx_path
                else:
                    return False, [f"Ungültiges Format: {format}"]
            
            # Konvertiere ConfigData zu Dictionary
            config_dict = config.to_dict()
            
            # Speichere basierend auf Format
            if format == 'json':
                success, save_errors = self._save_to_json(config_dict, str(file_path))
            elif format == 'xlsx':
                success, save_errors = self._save_to_xlsx(config_dict, str(file_path))
            else:
                return False, [f"Ungültiges Format: {format}"]
            
            if not success:
                return False, save_errors
            
            return True, []
            
        except Exception as e:
            return False, [f"Fehler beim Speichern der Konfiguration: {str(e)}"]
    
    def validate_config(self, config: ConfigData) -> Tuple[bool, List[str]]:
        """
        Validiert Konfiguration und gibt detaillierte Fehlermeldungen zurück.
        
        Requirement 2.5: WHEN eine ungültige Konfigurationsdatei geladen wird 
                        THEN das System SHALL eine Fehlermeldung mit Details anzeigen
        
        Args:
            config: ConfigData Objekt zum Validieren
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        return config.validate()
    
    def get_default_config(self) -> ConfigData:
        """
        Gibt Standard-Konfiguration aus config.py zurück.
        
        Returns:
            ConfigData: Standard-Konfiguration
        """
        # Erstelle ConfigData aus DEFAULT_CONFIG
        config_dict = {
            'model_provider': DEFAULT_CONFIG.get('MODEL_PROVIDER', 'OpenAI'),
            'model_name': DEFAULT_CONFIG.get('MODEL_NAME', 'gpt-4o-mini'),
            'data_dir': DEFAULT_CONFIG.get('DATA_DIR', 'input'),
            'output_dir': DEFAULT_CONFIG.get('OUTPUT_DIR', 'output'),
            'chunk_size': DEFAULT_CONFIG.get('CHUNK_SIZE', 1200),
            'chunk_overlap': DEFAULT_CONFIG.get('CHUNK_OVERLAP', 50),
            'batch_size': DEFAULT_CONFIG.get('BATCH_SIZE', 8),
            'code_with_context': DEFAULT_CONFIG.get('CODE_WITH_CONTEXT', False),
            'multiple_codings': DEFAULT_CONFIG.get('MULTIPLE_CODINGS', True),
            'multiple_coding_threshold': DEFAULT_CONFIG.get('MULTIPLE_CODING_THRESHOLD', 0.85),
            'analysis_mode': DEFAULT_CONFIG.get('ANALYSIS_MODE', 'deductive'),
            'review_mode': DEFAULT_CONFIG.get('REVIEW_MODE', 'consensus'),
            'attribute_labels': DEFAULT_CONFIG.get('ATTRIBUTE_LABELS', {
                'attribut1': 'Attribut1',
                'attribut2': 'Attribut2',
                'attribut3': 'Attribut3'
            }),
            'coder_settings': DEFAULT_CONFIG.get('CODER_SETTINGS', [
                {'temperature': 0.3, 'coder_id': 'auto_1'},
                {'temperature': 0.5, 'coder_id': 'auto_2'}
            ]),
            'export_annotated_pdfs': DEFAULT_CONFIG.get('EXPORT_ANNOTATED_PDFS', True),
            'pdf_annotation_fuzzy_threshold': DEFAULT_CONFIG.get('PDF_ANNOTATION_FUZZY_THRESHOLD', 0.85)
        }
        
        return ConfigData.from_dict(config_dict)
    
    # Private helper methods
    
    def _load_from_json(self, json_path: str) -> Tuple[bool, Optional[Dict], List[str]]:
        """
        Lädt Konfiguration aus JSON-Datei.
        
        Args:
            json_path: Pfad zur JSON-Datei
            
        Returns:
            Tuple[bool, Optional[Dict], List[str]]: (success, config_dict, errors)
        """
        try:
            # Verwende ConfigConverter zum Laden
            json_data = ConfigConverter.load_json(json_path)
            
            # Extrahiere config-Sektion
            if 'config' in json_data:
                config_dict = json_data['config']
            else:
                # Fallback: Verwende gesamte JSON-Daten
                config_dict = json_data
            
            return True, config_dict, []
            
        except FileNotFoundError as e:
            return False, None, [f"JSON-Datei nicht gefunden: {str(e)}"]
        except ValueError as e:
            return False, None, [f"Ungültiges JSON-Format: {str(e)}"]
        except Exception as e:
            return False, None, [f"Fehler beim Laden der JSON-Datei: {str(e)}"]
    
    def _load_from_xlsx(self, xlsx_path: str) -> Tuple[bool, Optional[Dict], List[str]]:
        """
        Lädt Konfiguration aus XLSX-Datei.
        
        Args:
            xlsx_path: Pfad zur XLSX-Datei
            
        Returns:
            Tuple[bool, Optional[Dict], List[str]]: (success, config_dict, errors)
        """
        try:
            # Verwende ConfigLoader zum Laden
            global_config = {}
            
            # Bestimme Verzeichnis und Dateiname
            xlsx_path_obj = Path(xlsx_path)
            script_dir = str(xlsx_path_obj.parent)
            filename = xlsx_path_obj.name
            
            # Erstelle ConfigLoader mit dem Verzeichnis
            loader = ConfigLoader(script_dir, global_config)
            
            # Überschreibe den hardcoded Excel-Pfad mit dem tatsächlichen Pfad
            loader.excel_path = str(xlsx_path_obj)
            
            success = loader.load_codebook()
            
            if not success:
                return False, None, [f"Fehler beim Laden der XLSX-Datei: {filename}"]
            
            # Debug: Zeige geladene Konfigurationswerte
            print(f"[DEBUG] Geladene Konfiguration aus {filename}:")
            for key, value in global_config.items():
                print(f"  {key}: {value}")
            
            # Extrahiere relevante Konfigurationswerte
            # Priorität: Kleinbuchstaben-Keys (aktuelle Werte) vor Großbuchstaben-Keys (Standard-Werte)
            config_dict = {
                'model_provider': global_config.get('model_provider', global_config.get('MODEL_PROVIDER', 'OpenAI')),
                'model_name': global_config.get('model_name', global_config.get('MODEL_NAME', 'gpt-4o-mini')),
                'data_dir': global_config.get('data_dir', global_config.get('DATA_DIR', 'input')),
                'output_dir': global_config.get('output_dir', global_config.get('OUTPUT_DIR', 'output')),
                'chunk_size': int(global_config.get('chunk_size', global_config.get('CHUNK_SIZE', 1200))),
                'chunk_overlap': int(global_config.get('chunk_overlap', global_config.get('CHUNK_OVERLAP', 50))),
                'batch_size': int(global_config.get('batch_size', global_config.get('BATCH_SIZE', 8))),
                'code_with_context': self._convert_to_bool(global_config.get('code_with_context', global_config.get('CODE_WITH_CONTEXT', False))),
                'multiple_codings': self._convert_to_bool(global_config.get('multiple_codings', global_config.get('MULTIPLE_CODINGS', True))),
                'multiple_coding_threshold': float(global_config.get('multiple_coding_threshold', global_config.get('MULTIPLE_CODING_THRESHOLD', 0.85))),
                'analysis_mode': global_config.get('analysis_mode', global_config.get('ANALYSIS_MODE', 'deductive')),
                'review_mode': global_config.get('review_mode', global_config.get('REVIEW_MODE', 'consensus')),
                'attribute_labels': global_config.get('attribute_labels', global_config.get('ATTRIBUTE_LABELS', {})),
                'coder_settings': global_config.get('coder_settings', global_config.get('CODER_SETTINGS', [])),
                'export_annotated_pdfs': self._convert_to_bool(global_config.get('export_annotated_pdfs', global_config.get('EXPORT_ANNOTATED_PDFS', True))),
                'pdf_annotation_fuzzy_threshold': float(global_config.get('pdf_annotation_fuzzy_threshold', global_config.get('PDF_ANNOTATION_FUZZY_THRESHOLD', 0.85)))
            }
            
            # Debug: Zeige welche Attribut-Labels verwendet werden
            print(f"[DEBUG] Attribut-Labels Auswahl:")
            print(f"  attribute_labels (Kleinbuchstaben): {global_config.get('attribute_labels', 'NICHT GEFUNDEN')}")
            print(f"  ATTRIBUTE_LABELS (Großbuchstaben): {global_config.get('ATTRIBUTE_LABELS', 'NICHT GEFUNDEN')}")
            print(f"  Verwendete Werte: {config_dict['attribute_labels']}")
            
            return True, config_dict, []
            
        except FileNotFoundError as e:
            return False, None, [f"XLSX-Datei nicht gefunden: {str(e)}"]
        except Exception as e:
            return False, None, [f"Fehler beim Laden der XLSX-Datei: {str(e)}"]
    
    def _convert_to_bool(self, value) -> bool:
        """
        Konvertiert verschiedene Werte zu Boolean.
        
        Args:
            value: Wert zum Konvertieren
            
        Returns:
            bool: Konvertierter Boolean-Wert
        """
        if isinstance(value, bool):
            return value
        elif isinstance(value, (int, float)):
            return bool(value) and value != 0
        elif isinstance(value, str):
            return value.lower() in ['true', '1', 'yes', 'ja', 'on', 'wahr']
        else:
            return bool(value)
    
    def _save_to_json(self, config_dict: Dict, json_path: str) -> Tuple[bool, List[str]]:
        """
        Speichert Konfiguration in JSON-Datei.
        
        Args:
            config_dict: Konfiguration als Dictionary
            json_path: Pfad zur Ziel-JSON-Datei
            
        Returns:
            Tuple[bool, List[str]]: (success, errors)
        """
        try:
            # Erstelle vollständige JSON-Struktur
            json_data = {
                'forschungsfrage': '',  # Wird separat verwaltet
                'kodierregeln': {
                    'general': [],
                    'format': [],
                    'exclusion': []
                },
                'deduktive_kategorien': {},  # Wird separat verwaltet
                'config': config_dict
            }
            
            # Wenn JSON bereits existiert, behalte forschungsfrage und kategorien
            if Path(json_path).exists():
                try:
                    existing_data = ConfigConverter.load_json(json_path)
                    json_data['forschungsfrage'] = existing_data.get('forschungsfrage', '')
                    json_data['kodierregeln'] = existing_data.get('kodierregeln', json_data['kodierregeln'])
                    json_data['deduktive_kategorien'] = existing_data.get('deduktive_kategorien', {})
                except:
                    pass  # Verwende leere Werte wenn Laden fehlschlägt
            
            # Speichere mit ConfigConverter
            ConfigConverter.save_json(json_data, json_path)
            
            return True, []
            
        except IOError as e:
            return False, [f"Fehler beim Schreiben der JSON-Datei: {str(e)}"]
        except Exception as e:
            return False, [f"Fehler beim Speichern der JSON-Datei: {str(e)}"]
    
    def _save_to_xlsx(self, config_dict: Dict, xlsx_path: str) -> Tuple[bool, List[str]]:
        """
        Speichert Konfiguration in XLSX-Datei.
        
        Args:
            config_dict: Konfiguration als Dictionary
            xlsx_path: Pfad zur Ziel-XLSX-Datei
            
        Returns:
            Tuple[bool, List[str]]: (success, errors)
        """
        try:
            # Erstelle vollständige JSON-Struktur für Konvertierung
            json_data = {
                'forschungsfrage': '',
                'kodierregeln': {
                    'general': [],
                    'format': [],
                    'exclusion': []
                },
                'deduktive_kategorien': {},
                'config': config_dict
            }
            
            # Wenn XLSX bereits existiert, lade existierende Daten
            if Path(xlsx_path).exists():
                try:
                    existing_data = ConfigConverter.qca_aid_xlsx_to_json(xlsx_path)
                    json_data['forschungsfrage'] = existing_data.get('forschungsfrage', '')
                    json_data['kodierregeln'] = existing_data.get('kodierregeln', json_data['kodierregeln'])
                    json_data['deduktive_kategorien'] = existing_data.get('deduktive_kategorien', {})
                except:
                    pass  # Verwende leere Werte wenn Laden fehlschlägt
            
            # Konvertiere zu XLSX mit ConfigConverter
            ConfigConverter.qca_aid_json_to_xlsx(json_data, xlsx_path)
            
            return True, []
            
        except ValueError as e:
            return False, [f"Fehler beim Konvertieren zu XLSX: {str(e)}"]
        except Exception as e:
            return False, [f"Fehler beim Speichern der XLSX-Datei: {str(e)}"]
    
    def _dict_to_config_data(self, config_dict: Dict) -> ConfigData:
        """
        Konvertiert Dictionary zu ConfigData Objekt.
        
        Args:
            config_dict: Konfiguration als Dictionary
            
        Returns:
            ConfigData: Konfigurationsobjekt
        """
        # Verwende from_dict Methode von ConfigData
        return ConfigData.from_dict(config_dict)
