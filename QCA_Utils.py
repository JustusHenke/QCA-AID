import re
import asyncio
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import os
import json
import pandas as pd
from openpyxl import load_workbook
from datetime import datetime

CONFIG = { } # hier leer, in Hauptskript integriert 

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



# Hilfsfunktion zur Token-Schätzung
def estimate_tokens(text: str) -> int:
    """
    Schätzt die Anzahl der Tokens in einem Text mit verbesserter Genauigkeit.
    Berücksichtigt verschiedene Zeichentypen, nicht nur Wortgrenzen.
    
    Args:
        text: Zu schätzender Text
        
    Returns:
        int: Geschätzte Tokenanzahl
    """
    if not text:
        return 0
        
    # Grundlegende Schätzung: 1 Token ≈ 4 Zeichen für englischen Text
    # Für deutsche Texte (mit längeren Wörtern) etwas anpassen: 1 Token ≈ 4.5 Zeichen
    char_per_token = 4.5
    
    # Anzahl der Sonderzeichen, die oft eigene Tokens bilden
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    
    # Anzahl der Wörter
    words = len(text.split())
    
    # Gewichtete Berechnung
    estimated_tokens = int(
        (len(text) / char_per_token) * 0.7 +  # Zeichenbasierte Schätzung (70% Gewichtung)
        (words + special_chars) * 0.3          # Wort- und Sonderzeichenbasierte Schätzung (30% Gewichtung)
    )
    
    return max(1, estimated_tokens)  # Mindestens 1 Token

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

def _calculate_multiple_coding_stats(all_codings: List[Dict]) -> Dict:
    """
    Berechnet Statistiken zur Mehrfachkodierung.
    
    Args:
        all_codings: Liste aller Kodierungen
        
    Returns:
        Dict: Statistiken zur Mehrfachkodierung
    """
    from collections import defaultdict, Counter
    
    segment_counts = defaultdict(int)
    focus_adherence = []
    category_combinations = []
    
    for coding in all_codings:
        segment_id = coding.get('segment_id', '')
        segment_counts[segment_id] += 1
        
        # Focus adherence tracking
        if coding.get('category_focus_used', False):
            focus_adherence.append(coding.get('target_category', '') == coding.get('category', ''))
        
        # Kategorie-Kombinationen sammeln
        if coding.get('total_coding_instances', 1) > 1:
            # Sammle alle Kategorien für dieses Segment
            segment_categories = [c.get('category', '') for c in all_codings 
                                if c.get('segment_id', '') == segment_id]
            if len(segment_categories) > 1:
                category_combinations.append(' + '.join(sorted(set(segment_categories))))
    
    segments_with_multiple = len([count for count in segment_counts.values() if count > 1])
    total_segments = len(segment_counts)
    total_codings = len(all_codings)
    
    combination_counter = Counter(category_combinations)
    
    return {
        'segments_with_multiple': segments_with_multiple,
        'total_segments': total_segments,
        'avg_codings_per_segment': total_codings / total_segments if total_segments > 0 else 0,
        'top_combinations': [combo for combo, _ in combination_counter.most_common(5)],
        'focus_adherence_rate': sum(focus_adherence) / len(focus_adherence) if focus_adherence else 0
    }

def _patch_tkinter_for_threaded_exit():
    """
    Patcht die Tkinter Variable.__del__ Methode, um den RuntimeError beim Beenden zu vermeiden.
    """
    import tkinter
    
    # Originale __del__ Methode speichern
    original_del = tkinter.Variable.__del__
    
    # Neue __del__ Methode definieren, die Ausnahmen abfängt
    def safe_del(self):
        try:
            # Nur aufrufen, wenn _tk existiert und es sich um ein valides Tkinter-Objekt handelt
            if hasattr(self, '_tk') and self._tk:
                original_del(self)
        except (RuntimeError, TypeError, AttributeError):
            # Diese Ausnahmen stillschweigend ignorieren
            pass
    
    # Die ursprüngliche Methode ersetzen
    tkinter.Variable.__del__ = safe_del
    print("Tkinter für sicheres Beenden gepatcht.")



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
        """Lädt Kodierregeln aus dem Excel-Codebook."""
        if 'KODIERREGELN' in wb.sheetnames:
            df = pd.read_excel(self.excel_path, sheet_name='KODIERREGELN', header=0)
            
            # Initialisiere Regelkategorien
            rules = {
                'general': [],       # Allgemeine Kodierregeln
                'format': [],        # Formatregeln
                'exclusion': []      # Neue Kategorie für Ausschlussregeln
            }
            
            # Verarbeite jede Spalte
            for column in df.columns:
                rules_list = df[column].dropna().tolist()
                
                if 'Allgemeine' in column:
                    rules['general'].extend(rules_list)
                elif 'Format' in column:
                    rules['format'].extend(rules_list)
                elif 'Ausschluss' in column:  # Neue Spalte für Ausschlussregeln
                    rules['exclusion'].extend(rules_list)
            
            print("\nKodierregeln geladen:")
            print(f"- Allgemeine Regeln: {len(rules['general'])}")
            print(f"- Formatregeln: {len(rules['format'])}")
            print(f"- Ausschlussregeln: {len(rules['exclusion'])}")
            
            self.config['KODIERREGELN'] = rules

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
            # print(f"Gefundene Spalten: {headers}")
            
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
            # for name, kat in kategorien.items():
            #     print(f"\nKategorie: {name}")
            #     print(f"- Definition: {len(kat['definition'])} Zeichen")
            #     if not kat['definition']:
            #         print("  WARNUNG: Keine Definition!")
            #     print(f"- Regeln: {len(kat['rules'])}")
            #     print(f"- Beispiele: {len(kat['examples'])}")
            #     print(f"- Subkategorien: {len(kat['subcategories'])}")
            #     for sub_name, sub_def in kat['subcategories'].items():
            #         print(f"  • {sub_name}: {sub_def[:50]}...")

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

            # Prüfe auf ANALYSIS_MODE in der Konfiguration
            if 'ANALYSIS_MODE' in config:
                valid_modes = {'full', 'abductive', 'deductive'}
                if config['ANALYSIS_MODE'] not in valid_modes:
                    print(f"Warnung: Ungültiger ANALYSIS_MODE '{config['ANALYSIS_MODE']}' im Codebook. Verwende 'deductive'.")
                    config['ANALYSIS_MODE'] = 'deductive'
                else:
                    print(f"ANALYSIS_MODE aus Codebook geladen: {config['ANALYSIS_MODE']}")
            else:
                config['ANALYSIS_MODE'] = 'deductive'  # Standardwert
                print(f"ANALYSIS_MODE nicht im Codebook gefunden, verwende Standard: {config['ANALYSIS_MODE']}")

            # Prüfe auf REVIEW_MODE in der Konfiguration
            if 'REVIEW_MODE' in config:
                valid_modes = {'auto', 'manual', 'consensus', 'majority'}
                if config['REVIEW_MODE'] not in valid_modes:
                    print(f"Warnung: Ungültiger REVIEW_MODE '{config['REVIEW_MODE']}' im Codebook. Verwende 'auto'.")
                    config['REVIEW_MODE'] = 'consensus'
                else:
                    print(f"REVIEW_MODE aus Codebook geladen: {config['REVIEW_MODE']}")
            else:
                config['REVIEW_MODE'] = 'consensus'  # Standardwert
                print(f"REVIEW_MODE nicht im Codebook gefunden, verwende Standard: {config['REVIEW_MODE']}")


            # Stelle sicher, dass ATTRIBUTE_LABELS vorhanden ist
            if 'ATTRIBUTE_LABELS' not in config:
                config['ATTRIBUTE_LABELS'] = {'attribut1': 'Attribut1', 'attribut2': 'Attribut2', 'attribut3': 'Attribut3'}
            elif 'attribut3' not in config['ATTRIBUTE_LABELS']:
                # Füge attribut3 hinzu wenn noch nicht vorhanden
                config['ATTRIBUTE_LABELS']['attribut3'] = 'Attribut3'
            
            # Debug für attribut3
            if config['ATTRIBUTE_LABELS']['attribut3']:
                print(f"Drittes Attribut-Label geladen: {config['ATTRIBUTE_LABELS']['attribut3']}")

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

            # Verarbeitung des CODE_WITH_CONTEXT Parameters
            if 'CODE_WITH_CONTEXT' in config:
                # Konvertiere zu Boolean
                value = config['CODE_WITH_CONTEXT']
                if isinstance(value, str):
                    sanitized['CODE_WITH_CONTEXT'] = value.lower() in ('true', 'ja', 'yes', '1')
                else:
                    sanitized['CODE_WITH_CONTEXT'] = bool(value)
                print(f"Übernehme CODE_WITH_CONTEXT aus Codebook: {sanitized['CODE_WITH_CONTEXT']}")
            else:
                # Standardwert setzen
                sanitized['CODE_WITH_CONTEXT'] = True
                print(f"CODE_WITH_CONTEXT nicht in Codebook gefunden, verwende Standard: {sanitized['CODE_WITH_CONTEXT']}")

            # Verarbeitung des MULTIPLE_CODINGS Parameters
            if 'MULTIPLE_CODINGS' in config:
                # Konvertiere zu Boolean
                value = config['MULTIPLE_CODINGS']
                if isinstance(value, str):
                    sanitized['MULTIPLE_CODINGS'] = value.lower() in ('true', 'ja', 'yes', '1')
                else:
                    sanitized['MULTIPLE_CODINGS'] = bool(value)
                print(f"Übernehme MULTIPLE_CODINGS aus Codebook: {sanitized['MULTIPLE_CODINGS']}")
            else:
                # Standardwert setzen
                sanitized['MULTIPLE_CODINGS'] = True
                print(f"MULTIPLE_CODINGS nicht in Codebook gefunden, verwende Standard: {sanitized['MULTIPLE_CODINGS']}")
            
            # Verarbeitung des MULTIPLE_CODING_THRESHOLD Parameters
            if 'MULTIPLE_CODING_THRESHOLD' in config:
                try:
                    threshold = float(config['MULTIPLE_CODING_THRESHOLD'])
                    if 0.0 <= threshold <= 1.0:
                        sanitized['MULTIPLE_CODING_THRESHOLD'] = threshold
                        print(f"Übernehme MULTIPLE_CODING_THRESHOLD aus Codebook: {sanitized['MULTIPLE_CODING_THRESHOLD']}")
                    else:
                        print(f"Warnung: MULTIPLE_CODING_THRESHOLD muss zwischen 0.0 und 1.0 liegen, verwende Standard: 0.6")
                        sanitized['MULTIPLE_CODING_THRESHOLD'] = 0.6
                except (ValueError, TypeError):
                    print(f"Warnung: Ungültiger MULTIPLE_CODING_THRESHOLD Wert, verwende Standard: 0.6")
                    sanitized['MULTIPLE_CODING_THRESHOLD'] = 0.6
            else:
                sanitized['MULTIPLE_CODING_THRESHOLD'] = 0.6
                print(f"MULTIPLE_CODING_THRESHOLD nicht in Codebook gefunden, verwende Standard: {sanitized['MULTIPLE_CODING_THRESHOLD']}")

            # Verarbeitung des BATCH_SIZE Parameters
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

    
        