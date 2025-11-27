"""
Explorer Configuration Loader

Loads and manages QCA-AID-Explorer configuration from Excel or JSON files.
Handles analysis configurations, filters, and parameters for the Explorer tool.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from .converter import ConfigConverter
from .synchronizer import ConfigSynchronizer


class ExplorerConfigLoader:
    """
    Lädt die Konfiguration für QCA-AID-Explorer aus einer Excel-Datei oder JSON-Datei.
    
    Diese Klasse ist speziell für QCA-AID-Explorer und lädt Analysekonfigurationen,
    Filter und Parameter für verschiedene Auswertungstypen (Netzwerk, Heatmap, etc.).
    
    Attributes:
        config_path: Pfad zur Konfigurationsdatei (.xlsx oder .json)
        base_config: Dictionary mit Basis-Konfigurationsparametern
        analysis_configs: Liste von Analysekonfigurationen
    """
    
    def __init__(self, config_path: str):
        """
        Initialisiert den ExplorerConfigLoader.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei (.xlsx oder .json)
        """
        self.config_path = config_path
        self.base_config = {}
        self.analysis_configs = []
        
        # Synchronisation vor dem Laden
        self._sync_configs()
        
        # Lade Konfiguration
        self._load_config()
    
    def _sync_configs(self) -> None:
        """
        Führt Synchronisation zwischen XLSX und JSON durch falls nötig.
        
        Bestimmt automatisch die Pfade für XLSX und JSON basierend auf config_path
        und synchronisiert die Dateien falls beide existieren.
        """
        print("\n=== Konfigurationssynchronisation ===")
        
        # Bestimme Pfade basierend auf config_path
        config_path = Path(self.config_path)
        
        # Wenn config_path auf .json zeigt, leite XLSX-Pfad ab
        if config_path.suffix.lower() == '.json':
            xlsx_path = config_path.with_suffix('.xlsx')
            json_path = config_path
        else:
            # Ansonsten ist es XLSX, leite JSON-Pfad ab
            xlsx_path = config_path
            json_path = config_path.with_suffix('.json')
        
        print(f"XLSX-Pfad: {xlsx_path}")
        print(f"JSON-Pfad: {json_path}")
        
        # Führe Synchronisation durch
        try:
            print("Starte Synchronisationsprozess...")
            synchronizer = ConfigSynchronizer(str(xlsx_path), str(json_path))
            result = synchronizer.sync()
            
            print(f"Synchronisation abgeschlossen. Verwende: {result.upper()}")
            
            # Aktualisiere config_path basierend auf Synchronisationsergebnis
            if result == 'json' and json_path.exists():
                self.config_path = str(json_path)
            elif xlsx_path.exists():
                self.config_path = str(xlsx_path)
            
            print("=== Synchronisation erfolgreich ===\n")
                
        except FileNotFoundError as e:
            # Wenn beide Dateien fehlen, wird der Fehler später beim Laden geworfen
            print("Keine Konfigurationsdateien gefunden. Synchronisation übersprungen.\n")
            pass
        except Exception as e:
            print(f"Warnung: Fehler bei Synchronisation: {str(e)}")
            print("Fahre mit ursprünglichem Pfad fort.\n")
            # Fahre mit ursprünglichem Pfad fort
    
    def _load_config(self):
        """
        Lädt die Konfiguration aus JSON oder XLSX-Datei.
        
        Bevorzugt JSON wenn vorhanden, fällt zurück auf XLSX.
        
        Raises:
            FileNotFoundError: Wenn keine Konfigurationsdatei gefunden wurde
            Exception: Bei Fehlern beim Laden der Konfiguration
        """
        try:
            config_path = Path(self.config_path)
            
            # Bestimme JSON-Pfad
            if config_path.suffix.lower() == '.json':
                json_path = config_path
                xlsx_path = config_path.with_suffix('.xlsx')
            else:
                xlsx_path = config_path
                json_path = config_path.with_suffix('.json')
            
            # Bevorzuge JSON wenn vorhanden
            if json_path.exists():
                try:
                    print(f"\nLade Konfiguration aus JSON: {json_path}")
                    self._load_from_json(str(json_path))
                    return
                except Exception as e:
                    print(f"Warnung: Fehler beim Laden der JSON-Datei: {str(e)}")
                    print(f"Fallback auf XLSX-Datei...")
            
            # Fallback auf XLSX
            if not xlsx_path.exists():
                raise FileNotFoundError(
                    f"Konfigurationsdatei nicht gefunden:\n"
                    f"  JSON: {json_path}\n"
                    f"  XLSX: {xlsx_path}"
                )
            
            print(f"\nLade Konfiguration aus XLSX: {xlsx_path}")
            self._load_from_xlsx(str(xlsx_path))
                
        except Exception as e:
            print(f"Fehler beim Laden der Konfiguration: {str(e)}")
            raise
    
    def _load_from_json(self, json_path: str) -> None:
        """
        Lädt Konfiguration aus JSON-Datei.
        
        Args:
            json_path: Pfad zur JSON-Datei
            
        Raises:
            FileNotFoundError: Wenn JSON-Datei nicht existiert
            ValueError: Wenn JSON-Struktur ungültig ist
        """
        from QCA_AID_assets.core.validators import ConfigValidator
        
        config_data = ConfigConverter.load_json(json_path)
        
        # Validiere JSON-Format (UTF-8, Einrückung, erforderliche Keys)
        is_valid_format, format_errors = ConfigValidator.validate_json_format(json_path)
        if not is_valid_format:
            print(f"Warnung: JSON-Format-Probleme erkannt:")
            for error in format_errors:
                print(f"  - {error}")
        
        # Validiere Struktur und Datentypen
        is_valid_structure, structure_errors = ConfigValidator.validate_json_config(config_data)
        if not is_valid_structure:
            error_msg = "JSON-Konfiguration ist ungültig:\n" + "\n".join(f"  - {e}" for e in structure_errors)
            raise ValueError(error_msg)
        
        # Validiere Struktur (Backward-Kompatibilität)
        if 'base_config' not in config_data or 'analysis_configs' not in config_data:
            raise ValueError("JSON muss 'base_config' und 'analysis_configs' enthalten")
        
        # Lade base_config
        self.base_config = config_data['base_config']
        
        print("Basis-Konfiguration geladen:")
        for key, value in self.base_config.items():
            print(f"  {key}: {value}")
        
        # Lade analysis_configs
        self.analysis_configs = config_data['analysis_configs']
        
        print(f"\n{len(self.analysis_configs)} Auswertungskonfigurationen gefunden:")
        for config in self.analysis_configs:
            active_status = config['params'].get('active', True)
            status_text = "aktiviert" if active_status else "deaktiviert"
            print(f"  - {config['name']} ({status_text})")
    
    def _load_from_xlsx(self, xlsx_path: str) -> None:
        """
        Lädt Konfiguration aus XLSX-Datei.
        
        Liest das Basis-Sheet für grundlegende Parameter und alle weiteren Sheets
        für Analysekonfigurationen.
        
        Args:
            xlsx_path: Pfad zur XLSX-Datei
            
        Raises:
            FileNotFoundError: Wenn XLSX-Datei nicht existiert
        """
        # Prüfe, ob die Datei existiert
        if not os.path.exists(xlsx_path):
            raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {xlsx_path}")
        
        # Lese das Basis-Sheet
        base_df = pd.read_excel(xlsx_path, sheet_name='Basis')
        
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
        excel = pd.ExcelFile(xlsx_path)
        sheet_names = excel.sheet_names
        
        # Überspringe das Basis-Sheet
        for sheet_name in sheet_names:
            if sheet_name.lower() != 'basis':
                analysis_df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
                
                # Extrahiere die Parameter für diese Auswertung
                analysis_config = {'name': sheet_name}
                filter_params = {}
                other_params = {}
                
                for _, row in analysis_df.iterrows():
                    param_name = str(row['Parameter'])
                    param_value = row['Wert']

                    # Spezielle Behandlung für active/enabled Parameter
                    if param_name.lower() == 'active' or param_name.lower() == 'enabled':
                        if pd.isna(param_value):
                            # Wenn kein Wert angegeben, default auf True
                            param_value = True
                        elif isinstance(param_value, str):
                            param_value = param_value.lower() in ('true', 'ja', 'yes', '1')
                        else:
                            param_value = bool(param_value)
                            
                        # Speichere standardisiert als 'active'
                        other_params['active'] = param_value
                        continue
                    
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
                
                # Stelle sicher, dass 'active' immer existiert
                if 'active' not in other_params:
                    other_params['active'] = True  # Default: aktiviert
                
                analysis_config['filters'] = filter_params
                analysis_config['params'] = other_params
                self.analysis_configs.append(analysis_config)
        
        print(f"\n{len(self.analysis_configs)} Auswertungskonfigurationen gefunden:")
        for config in self.analysis_configs:
            active_status = config['params'].get('active', True)
            status_text = "aktiviert" if active_status else "deaktiviert"
            print(f"  - {config['name']} ({status_text})")

    def get_base_config(self) -> Dict[str, Any]:
        """
        Gibt die Basis-Konfiguration zurück.
        
        Returns:
            Dictionary mit Basis-Konfigurationsparametern
        """
        return self.base_config
    
    def get_analysis_configs(self) -> List[Dict[str, Any]]:
        """
        Gibt die Auswertungskonfigurationen zurück.
        
        Returns:
            Liste von Dictionaries mit Analysekonfigurationen
        """
        return self.analysis_configs
