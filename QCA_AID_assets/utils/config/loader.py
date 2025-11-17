"""
Configuration Loader

Loads and manages QCA-AID configuration from Excel codebook files.
Handles category definitions, coding rules, settings validation, and sanitization.
"""

import os
import pandas as pd
from typing import Dict, Optional, Any
from openpyxl import load_workbook


class ConfigLoader:
    """
    Lädt und verwaltet die QCA-AID Konfiguration aus Excel-Codebüchern.
    
    Handles:
    - Loading research question from Excel
    - Loading coding rules
    - Loading deductive category definitions
    - Loading general configuration settings
    - Validation and sanitization of all loaded values
    - Boolean parameter handling
    - Directory path management
    
    The loader stores all values in a global_config dictionary which is
    updated throughout the loading process.
    """
    
    def __init__(self, script_dir: str, global_config: Dict) -> None:
        """
        Initialisiert den ConfigLoader.
        
        Args:
            script_dir: Directory where QCA-AID-Codebook.xlsx is located
            global_config: Dictionary to store loaded configuration (passed by reference)
        """
        self.script_dir = script_dir
        self.excel_path = os.path.join(script_dir, "QCA-AID-Codebook.xlsx")
        self.global_config = global_config  # Reference zum globalen CONFIG-Objekt
    
    def load_codebook(self) -> bool:
        """
        Lädt die komplette Konfiguration aus dem Excel-Codebook.
        
        Attempts to load with read_only mode for better compatibility.
        If Excel file is locked (PermissionError), tries alternative loading strategy.
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        print(f"Versuche Konfiguration zu laden von: {self.excel_path}")
        if not os.path.exists(self.excel_path):
            print(f"Excel-Datei nicht gefunden: {self.excel_path}")
            return False

        try:
            # FIX: Erweiterte Optionen für das Öffnen der Excel-Datei
            print("\nÖffne Excel-Datei...")
            wb = load_workbook(
                self.excel_path,
                read_only=True,
                data_only=True,
                keep_vba=False  # Verhindert Probleme mit VBA-Code
            )
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
                # Speichere in Config
                self.global_config['DEDUKTIVE_KATEGORIEN'] = kategorien
                print("\nKategorien erfolgreich in Config gespeichert")
                return True
            else:
                print("\nKeine Kategorien geladen!")
                return False

        except PermissionError:
            # Spezifische Behandlung für geöffnete Dateien
            print("\n⚠️ Datei ist bereits geöffnet (z.B. in Excel).")
            print("Versuche alternative Lademethode...")
            try:
                # Versuche mit keep_links=False für bessere Kompatibilität
                wb = load_workbook(
                    self.excel_path,
                    read_only=True,
                    data_only=True,
                    keep_vba=False,
                    keep_links=False  # Entfernt externe Links
                )
                
                # Lade die verschiedenen Komponenten
                self._load_research_question(wb)
                self._load_coding_rules(wb)
                self._load_config(wb)
                kategorien = self._load_deduktive_kategorien(wb)
                
                if kategorien:
                    self.global_config['DEDUKTIVE_KATEGORIEN'] = kategorien
                    print("\n[OK] Konfiguration trotz geöffneter Datei erfolgreich geladen!")
                    return True
                else:
                    print("\n[ERROR] Keine Kategorien geladen, auch bei alternativer Methode")
                    return False
                    
            except Exception as alt_e:
                print(f"\n[ERROR] Auch alternative Lademethode fehlgeschlagen: {str(alt_e)}")
                print("\nLösung: Schließen Sie die Excel-Datei und versuchen Sie erneut,")
                print("oder deaktivieren Sie das automatische Speichern in OneDrive.")
                return False

        except Exception as e:
            print(f"Fehler beim Lesen der Excel-Datei: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_research_question(self, wb) -> None:
        """Lädt die Forschungsfrage aus dem Excel-Codebook."""
        if 'FORSCHUNGSFRAGE' in wb.sheetnames:
            sheet = wb['FORSCHUNGSFRAGE']
            value = sheet['B1'].value
            self.global_config['FORSCHUNGSFRAGE'] = value

    def _load_coding_rules(self, wb) -> None:
        """Lädt Kodierregeln aus dem Excel-Codebook."""
        if 'KODIERREGELN' in wb.sheetnames:
            df = pd.read_excel(self.excel_path, sheet_name='KODIERREGELN', header=0)
            
            # Initialisiere Regelkategorien
            rules = {
                'general': [],       # Allgemeine Kodierregeln
                'format': [],        # Formatregeln
                'exclusion': []      # Ausschlussregeln
            }
            
            # Verarbeite jede Spalte
            for column in df.columns:
                rules_list = df[column].dropna().tolist()
                
                if 'Allgemeine' in column:
                    rules['general'].extend(rules_list)
                elif 'Format' in column:
                    rules['format'].extend(rules_list)
                elif 'Ausschluss' in column:
                    rules['exclusion'].extend(rules_list)
            
            print("\nKodierregeln geladen:")
            print(f"- Allgemeine Regeln: {len(rules['general'])}")
            print(f"- Formatregeln: {len(rules['format'])}")
            print(f"- Ausschlussregeln: {len(rules['exclusion'])}")
            
            self.global_config['KODIERREGELN'] = rules

    def _load_deduktive_kategorien(self, wb) -> Dict:
        """
        Lädt deduktive Kategorien mit Hierarchie aus dem Excel-Codebook.
        
        Supports multi-level hierarchy:
        - Key: Hauptkategorie
        - Sub-Key: definition, rules, examples, subcategories
        - Sub-Sub-Key: Unterkategorie-Namen
        
        Returns:
            Dictionary with category structure or empty dict on error
        """
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
                            
                        elif sub_key == 'rules':
                            if value:
                                kategorien[current_category]['rules'].append(value)
                                
                        elif sub_key == 'examples':
                            if value:
                                kategorien[current_category]['examples'].append(value)
                                
                        elif sub_key == 'subcategories' and sub_sub_key:
                            kategorien[current_category]['subcategories'][sub_sub_key] = value
                                
                except Exception as e:
                    print(f"Fehler in Zeile {row_idx}: {str(e)}")
                    continue

            # Ergebnis
            if kategorien:
                print(f"Erfolgreich {len(kategorien)} Kategorien geladen")
                return kategorien
            else:
                print("Keine Kategorien gefunden!")
                return {}

        except Exception as e:
            print(f"Fehler beim Laden der Kategorien: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()
            return {}
    
    def _load_config(self, wb) -> bool:
        """
        Lädt allgemeine Konfigurationswerte aus dem CONFIG Sheet.
        
        Handles:
        - Nested dictionary structures (Key -> Sub-Key -> Sub-Sub-Key)
        - List structures for CODER_SETTINGS
        - Boolean parameter conversion
        - Analysis mode and review mode validation
        
        Returns:
            bool: True if config loaded, False if CONFIG sheet not found
        """
        if 'CONFIG' not in wb.sheetnames:
            return False
            
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
                else:  # Für verschachtelte Dicts
                    if not isinstance(config[key], dict):
                        config[key] = {}
                    if pd.isna(sub_sub_key):
                        # Extended Boolean handling für alle relevanten Parameter
                        if sub_key in ['EXPORT_ANNOTATED_PDFS', 'CODE_WITH_CONTEXT', 'SAVE_PROGRESS', 'PARALLEL_PROCESSING', 'MULTIPLE_CODINGS']:
                            if isinstance(value, str):
                                config[key][sub_key] = value.lower() in ['true', '1', 'yes', 'ja', 'on', 'wahr']
                            elif isinstance(value, (int, float)):
                                config[key][sub_key] = bool(value) and value != 0
                            else:
                                config[key][sub_key] = bool(value)
                            print(f"[CONFIG] {sub_key} geladen: {config[key][sub_key]} (ursprünglich: '{value}')")
                        elif sub_key in ['BATCH_SIZE', 'CHUNK_SIZE', 'CHUNK_OVERLAP', 'MAX_RETRIES'] or key == 'BATCH_SIZE':
                            try:
                                config[key][sub_key] = int(value)
                                print(f"[CONFIG] {sub_key} geladen: {config[key][sub_key]}")
                            except (ValueError, TypeError):
                                default_values = {'BATCH_SIZE': 5, 'CHUNK_SIZE': 2000, 'CHUNK_OVERLAP': 200, 'MAX_RETRIES': 3}
                                config[key][sub_key] = default_values.get(sub_key, 5)
                                print(f"[CONFIG] Ungültiger {sub_key} Wert '{value}', verwende Standard: {config[key][sub_key]}")
                        elif sub_key in ['MULTIPLE_CODING_THRESHOLD', 'SIMILARITY_THRESHOLD', 'PDF_ANNOTATION_FUZZY_THRESHOLD']:
                            try:
                                config[key][sub_key] = float(value)
                                print(f"[CONFIG] {sub_key} geladen: {config[key][sub_key]}")
                            except (ValueError, TypeError):
                                default_values = {'MULTIPLE_CODING_THRESHOLD': 0.85, 'SIMILARITY_THRESHOLD': 0.7, 'PDF_ANNOTATION_FUZZY_THRESHOLD': 0.85}
                                config[key][sub_key] = default_values.get(sub_key, 0.85)
                                print(f"[CONFIG] Ungültiger {sub_key} Wert '{value}', verwende Standard: {config[key][sub_key]}")
                        else:
                            config[key][sub_key] = value
                    else:
                        if sub_key not in config[key]:
                            config[key][sub_key] = {}
                        config[key][sub_key][sub_sub_key] = value

            # Top-level Boolean parameters
            if key == 'EXPORT_ANNOTATED_PDFS' and pd.isna(sub_key):
                if isinstance(value, str):
                    config['EXPORT_ANNOTATED_PDFS'] = value.lower() in ['true', '1', 'yes', 'ja', 'on', 'wahr']
                elif isinstance(value, (int, float)):
                    config['EXPORT_ANNOTATED_PDFS'] = bool(value) and value != 0
                else:
                    config['EXPORT_ANNOTATED_PDFS'] = bool(value)
                print(f"[CONFIG] EXPORT_ANNOTATED_PDFS geladen: {config['EXPORT_ANNOTATED_PDFS']}")

        # Validierung: ANALYSIS_MODE
        if 'ANALYSIS_MODE' in config:
            valid_modes = {'full', 'abductive', 'deductive', 'inductive', 'grounded'}
            if config['ANALYSIS_MODE'] not in valid_modes:
                print(f"[CONFIG] Warnung: Ungültiger ANALYSIS_MODE '{config['ANALYSIS_MODE']}'. Verwende 'deductive'.")
                config['ANALYSIS_MODE'] = 'deductive'
            else:
                print(f"[CONFIG] ANALYSIS_MODE geladen: {config['ANALYSIS_MODE']}")
        else:
            config['ANALYSIS_MODE'] = 'deductive'
            print(f"[CONFIG] ANALYSIS_MODE nicht gefunden, verwende Standard: deductive")

        # Validierung: REVIEW_MODE
        if 'REVIEW_MODE' in config:
            valid_modes = {'auto', 'manual', 'consensus', 'majority'}
            if config['REVIEW_MODE'] not in valid_modes:
                print(f"[CONFIG] Warnung: Ungültiger REVIEW_MODE '{config['REVIEW_MODE']}'. Verwende 'consensus'.")
                config['REVIEW_MODE'] = 'consensus'
            else:
                print(f"[CONFIG] REVIEW_MODE geladen: {config['REVIEW_MODE']}")
        else:
            config['REVIEW_MODE'] = 'consensus'
            print(f"[CONFIG] REVIEW_MODE nicht gefunden, verwende Standard: consensus")

        # Stelle sicher, dass ATTRIBUTE_LABELS vorhanden ist
        if 'ATTRIBUTE_LABELS' not in config:
            config['ATTRIBUTE_LABELS'] = {'attribut1': 'Attribut1', 'attribut2': 'Attribut2', 'attribut3': 'Attribut3'}
        elif isinstance(config.get('ATTRIBUTE_LABELS'), dict) and 'attribut3' not in config['ATTRIBUTE_LABELS']:
            config['ATTRIBUTE_LABELS']['attribut3'] = 'Attribut3'

        # Extract top-level booleans from nested structure
        for key in ['SETTINGS', 'OPTIONS', 'GENERAL']:
            if key in config and isinstance(config[key], dict):
                for param_name in ['CODE_WITH_CONTEXT', 'MULTIPLE_CODINGS', 'PARALLEL_PROCESSING', 'SAVE_PROGRESS']:
                    if param_name in config[key]:
                        config[param_name] = config[key][param_name]
                        print(f"[CONFIG] Extrahierte {param_name} aus {key}: {config[param_name]}")
                # Also extract threshold values
                for threshold_name in ['MULTIPLE_CODING_THRESHOLD', 'SIMILARITY_THRESHOLD', 'PDF_ANNOTATION_FUZZY_THRESHOLD']:
                    if threshold_name in config[key]:
                        config[threshold_name] = config[key][threshold_name]
                        print(f"[CONFIG] Extrahierte {threshold_name} aus {key}: {config[threshold_name]}")

        
        self._sanitize_config(config)
        
        # Update global_config with values that weren't specially handled in _sanitize_config
        # Only add keys that aren't already set by _sanitize_config
        specially_handled_keys = {
            'DATA_DIR', 'OUTPUT_DIR', 'INPUT_DIR',
            'CHUNK_SIZE', 'CHUNK_OVERLAP',
            'CODER_SETTINGS',
            'CODE_WITH_CONTEXT',
            'MULTIPLE_CODINGS',
            'MULTIPLE_CODING_THRESHOLD',
            'SIMILARITY_THRESHOLD',
            'PDF_ANNOTATION_FUZZY_THRESHOLD',
        }
        
        for key, value in config.items():
            if key not in specially_handled_keys and key not in self.global_config:
                self.global_config[key] = value
        
        return True

    def _sanitize_config(self, loaded_config: Dict) -> None:
        """
        Bereinigt und validiert Konfigurationswerte.
        
        Handles:
        - Directory path resolution and creation
        - Numeric value conversion and validation
        - CODER_SETTINGS type conversion
        - Parameter constraint checking
        - Chunk size validation
        
        Args:
            loaded_config: Dictionary with raw configuration values from codebook
        """
        try:
            # Navigate to root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))  # utils/config
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # QCA-AID root
            
            # Update OUTPUT_DIR and create if needed
            self.global_config['OUTPUT_DIR'] = os.path.join(root_dir, 'output')
            os.makedirs(self.global_config['OUTPUT_DIR'], exist_ok=True)
            print(f"[SANITIZE] Ausgabeverzeichnis: {self.global_config['OUTPUT_DIR']}")
            
            for key, value in loaded_config.items():
                # Directory paths relative to root
                if key in ['DATA_DIR', 'OUTPUT_DIR', 'INPUT_DIR']:
                    self.global_config[key] = os.path.join(root_dir, str(value).lstrip('/\\'))
                    os.makedirs(self.global_config[key], exist_ok=True)
                    print(f"[SANITIZE] Verzeichnis: {self.global_config[key]}")
                
                # Numeric values for chunking
                elif key in ['CHUNK_SIZE', 'CHUNK_OVERLAP']:
                    try:
                        self.global_config[key] = max(1, int(value))
                        print(f"[SANITIZE] {key} = {self.global_config[key]}")
                    except (ValueError, TypeError):
                        print(f"[SANITIZE] Ungültiger {key}, verwende Standard")
                        # Use default from common.py constants
                        self.global_config[key] = value
                
                # Coder settings with type conversion
                elif key == 'CODER_SETTINGS':
                    self.global_config[key] = [
                        {
                            'temperature': float(coder['temperature'])
                                if isinstance(coder.get('temperature'), (int, float, str))
                                else 0.3,
                            'coder_id': str(coder.get('coder_id', f'auto_{i}'))
                        }
                        for i, coder in enumerate(value)
                    ]
                
                # All other values pass through unchanged
                else:
                    self.global_config[key] = value

            # CODE_WITH_CONTEXT handling
            if 'CODE_WITH_CONTEXT' in loaded_config:
                value = loaded_config['CODE_WITH_CONTEXT']
                if isinstance(value, str):
                    self.global_config['CODE_WITH_CONTEXT'] = value.lower() in ('true', 'ja', 'yes', '1')
                else:
                    self.global_config['CODE_WITH_CONTEXT'] = bool(value)
                print(f"[SANITIZE] CODE_WITH_CONTEXT = {self.global_config['CODE_WITH_CONTEXT']}")
            else:
                self.global_config['CODE_WITH_CONTEXT'] = False

            # MULTIPLE_CODINGS handling
            if 'MULTIPLE_CODINGS' in loaded_config:
                value = loaded_config['MULTIPLE_CODINGS']
                if isinstance(value, str):
                    self.global_config['MULTIPLE_CODINGS'] = value.lower() in ('true', 'ja', 'yes', '1')
                else:
                    self.global_config['MULTIPLE_CODINGS'] = bool(value)
                print(f"[SANITIZE] MULTIPLE_CODINGS = {self.global_config['MULTIPLE_CODINGS']}")
            else:
                self.global_config['MULTIPLE_CODINGS'] = True
            
            # MULTIPLE_CODING_THRESHOLD handling
            if 'MULTIPLE_CODING_THRESHOLD' in loaded_config:
                try:
                    threshold = float(loaded_config['MULTIPLE_CODING_THRESHOLD'])
                    if 0.0 <= threshold <= 1.0:
                        self.global_config['MULTIPLE_CODING_THRESHOLD'] = threshold
                        print(f"[SANITIZE] MULTIPLE_CODING_THRESHOLD = {threshold}")
                    else:
                        print(f"[SANITIZE] Warnung: MULTIPLE_CODING_THRESHOLD außerhalb Bereich, verwende 0.6")
                        self.global_config['MULTIPLE_CODING_THRESHOLD'] = 0.6
                except (ValueError, TypeError):
                    print(f"[SANITIZE] Ungültiger MULTIPLE_CODING_THRESHOLD, verwende 0.6")
                    self.global_config['MULTIPLE_CODING_THRESHOLD'] = 0.6
            else:
                self.global_config['MULTIPLE_CODING_THRESHOLD'] = 0.6

            # BATCH_SIZE handling
            if 'BATCH_SIZE' in loaded_config:
                try:
                    batch_size = int(loaded_config['BATCH_SIZE'])
                    if batch_size < 1:
                        print("[SANITIZE] Warnung: BATCH_SIZE muss >= 1 sein")
                        batch_size = 5
                    elif batch_size > 20:
                        print("[SANITIZE] Warnung: BATCH_SIZE > 20 könnte Performance-Probleme verursachen")
                    self.global_config['BATCH_SIZE'] = batch_size
                    print(f"[SANITIZE] BATCH_SIZE = {batch_size}")
                except (ValueError, TypeError):
                    print("[SANITIZE] Ungültiger BATCH_SIZE, verwende 5")
                    self.global_config['BATCH_SIZE'] = 5
            else:
                self.global_config['BATCH_SIZE'] = 5

            # Ensure CHUNK_OVERLAP < CHUNK_SIZE
            if 'CHUNK_SIZE' in self.global_config and 'CHUNK_OVERLAP' in self.global_config:
                if self.global_config['CHUNK_OVERLAP'] >= self.global_config['CHUNK_SIZE']:
                    print(f"[SANITIZE] Warnung: CHUNK_OVERLAP >= CHUNK_SIZE")
                    self.global_config['CHUNK_OVERLAP'] = max(1, self.global_config['CHUNK_SIZE'] // 10)
                    
        except Exception as e:
            print(f"[SANITIZE] Fehler: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_config(self) -> Dict:
        """
        Returns the global configuration dictionary.
        
        Returns:
            Dictionary containing all loaded and sanitized configuration
        """
        return self.global_config
