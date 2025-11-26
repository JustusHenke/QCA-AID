"""
Configuration Synchronizer

Manages synchronization between Excel (XLSX) and JSON configuration files.
Detects differences and prompts user to resolve conflicts.
"""

import os
from typing import Dict, Any, List, Tuple
from pathlib import Path
from .converter import ConfigConverter


class ConfigSynchronizer:
    """Verwaltet Synchronisation zwischen XLSX und JSON Konfigurationsdateien"""
    
    def __init__(self, xlsx_path: str, json_path: str):
        """
        Initialisiert ConfigSynchronizer mit Pfaden
        
        Args:
            xlsx_path: Pfad zur Excel-Konfigurationsdatei
            json_path: Pfad zur JSON-Konfigurationsdatei
        """
        self.xlsx_path = Path(xlsx_path)
        self.json_path = Path(json_path)
    
    def sync(self) -> str:
        """
        Führt Synchronisation zwischen XLSX und JSON durch
        
        Returns:
            'json' wenn JSON verwendet werden soll, 'xlsx' wenn XLSX verwendet werden soll
            
        Raises:
            FileNotFoundError: Wenn beide Dateien fehlen
        """
        xlsx_exists = self.xlsx_path.exists()
        json_exists = self.json_path.exists()
        
        # Fall 1: Beide Dateien fehlen
        if not xlsx_exists and not json_exists:
            raise FileNotFoundError(
                f"Weder XLSX noch JSON gefunden:\n"
                f"  XLSX: {self.xlsx_path}\n"
                f"  JSON: {self.json_path}"
            )
        
        # Fall 2: Nur XLSX existiert - erstelle JSON
        if xlsx_exists and not json_exists:
            print(f"JSON-Datei nicht gefunden. Erstelle aus XLSX: {self.json_path}")
            self._update_from_xlsx()
            return 'json'
        
        # Fall 3: Nur JSON existiert - verwende JSON
        if json_exists and not xlsx_exists:
            print(f"Warnung: XLSX-Datei nicht gefunden: {self.xlsx_path}")
            print(f"Verwende JSON-Datei: {self.json_path}")
            return 'json'
        
        # Fall 4: Beide existieren - prüfe auf Differenzen
        try:
            xlsx_data = ConfigConverter.xlsx_to_json(str(self.xlsx_path))
            json_data = ConfigConverter.load_json(str(self.json_path))
            
            differences = self._detect_differences(xlsx_data, json_data)
            
            if not differences:
                # Keine Differenzen - verwende JSON
                return 'json'
            
            # Differenzen gefunden - frage Benutzer
            choice = self._prompt_user_choice(differences)
            
            if choice == 'xlsx':
                self._update_from_xlsx()
                return 'json'
            elif choice == 'json':
                self._update_from_json()
                return 'json'
            else:
                # Abbruch - verwende JSON
                print("Synchronisation abgebrochen. Verwende JSON-Datei.")
                return 'json'
                
        except Exception as e:
            print(f"Fehler bei Synchronisation: {str(e)}")
            print("Verwende JSON-Datei.")
            return 'json'
    
    def _detect_differences(self, xlsx_data: Dict[str, Any], json_data: Dict[str, Any]) -> List[str]:
        """
        Erkennt Differenzen zwischen XLSX und JSON Daten
        
        Args:
            xlsx_data: Daten aus XLSX-Datei
            json_data: Daten aus JSON-Datei
            
        Returns:
            Liste von Differenz-Beschreibungen
        """
        import math
        
        def values_are_different(v1, v2) -> bool:
            """Prüft ob zwei Werte unterschiedlich sind, mit Toleranz für Floats"""
            # Excel hat ~15 signifikante Stellen Präzision für Floats
            if isinstance(v1, float) and isinstance(v2, float):
                return not math.isclose(v1, v2, rel_tol=1e-14)
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Vergleiche int und float mit Toleranz
                return not math.isclose(float(v1), float(v2), rel_tol=1e-14)
            else:
                return v1 != v2
        
        differences = []
        
        # Prüfe base_config
        xlsx_base = xlsx_data.get('base_config', {})
        json_base = json_data.get('base_config', {})
        
        # Alle Keys aus beiden Configs
        all_base_keys = set(xlsx_base.keys()) | set(json_base.keys())
        
        for key in sorted(all_base_keys):
            xlsx_value = xlsx_base.get(key)
            json_value = json_base.get(key)
            
            if values_are_different(xlsx_value, json_value):
                differences.append(
                    f"base_config.{key}: XLSX={repr(xlsx_value)} vs JSON={repr(json_value)}"
                )
        
        # Prüfe analysis_configs
        xlsx_analyses = xlsx_data.get('analysis_configs', [])
        json_analyses = json_data.get('analysis_configs', [])
        
        # Erstelle Mapping nach Namen
        xlsx_by_name = {a['name']: a for a in xlsx_analyses}
        json_by_name = {a['name']: a for a in json_analyses}
        
        all_analysis_names = set(xlsx_by_name.keys()) | set(json_by_name.keys())
        
        for name in sorted(all_analysis_names):
            xlsx_analysis = xlsx_by_name.get(name)
            json_analysis = json_by_name.get(name)
            
            # Analyse existiert nur in einer Datei
            if xlsx_analysis is None:
                differences.append(f"analysis_configs[{name}]: Nur in JSON vorhanden")
                continue
            if json_analysis is None:
                differences.append(f"analysis_configs[{name}]: Nur in XLSX vorhanden")
                continue
            
            # Prüfe filters
            xlsx_filters = xlsx_analysis.get('filters', {})
            json_filters = json_analysis.get('filters', {})
            all_filter_keys = set(xlsx_filters.keys()) | set(json_filters.keys())
            
            for filter_key in sorted(all_filter_keys):
                xlsx_filter_value = xlsx_filters.get(filter_key)
                json_filter_value = json_filters.get(filter_key)
                
                if values_are_different(xlsx_filter_value, json_filter_value):
                    differences.append(
                        f"analysis_configs[{name}].filters.{filter_key}: "
                        f"XLSX={repr(xlsx_filter_value)} vs JSON={repr(json_filter_value)}"
                    )
            
            # Prüfe params
            xlsx_params = xlsx_analysis.get('params', {})
            json_params = json_analysis.get('params', {})
            all_param_keys = set(xlsx_params.keys()) | set(json_params.keys())
            
            for param_key in sorted(all_param_keys):
                xlsx_param_value = xlsx_params.get(param_key)
                json_param_value = json_params.get(param_key)
                
                if values_are_different(xlsx_param_value, json_param_value):
                    differences.append(
                        f"analysis_configs[{name}].params.{param_key}: "
                        f"XLSX={repr(xlsx_param_value)} vs JSON={repr(json_param_value)}"
                    )
        
        return differences
    
    def _prompt_user_choice(self, differences: List[str]) -> str:
        """
        Fragt Benutzer welche Version aktueller ist
        
        Args:
            differences: Liste von Differenzen
            
        Returns:
            'xlsx', 'json', oder 'abort'
        """
        print("\n" + "="*70)
        print("KONFIGURATIONSDIFFERENZEN ERKANNT")
        print("="*70)
        print("\nFolgende Unterschiede wurden zwischen XLSX und JSON gefunden:\n")
        
        for i, diff in enumerate(differences, 1):
            print(f"  {i}. {diff}")
        
        print("\n" + "="*70)
        print("Welche Datei enthält die aktuellere Version?")
        print("  1 - XLSX ist aktueller (JSON wird aktualisiert)")
        print("  2 - JSON ist aktueller (XLSX wird aktualisiert)")
        print("  q - Abbrechen (keine Änderungen)")
        print("="*70)
        
        while True:
            choice = input("\nIhre Wahl (1/2/q): ").strip().lower()
            
            if choice == '1':
                return 'xlsx'
            elif choice == '2':
                return 'json'
            elif choice == 'q':
                return 'abort'
            else:
                print("Ungültige Eingabe. Bitte wählen Sie 1, 2 oder q.")
    
    def _update_from_xlsx(self) -> None:
        """
        Aktualisiert JSON-Datei aus XLSX-Datei
        
        Raises:
            FileNotFoundError: Wenn XLSX nicht existiert
            ValueError: Wenn XLSX ungültig ist
        """
        print(f"Aktualisiere JSON aus XLSX...")
        xlsx_data = ConfigConverter.xlsx_to_json(str(self.xlsx_path))
        ConfigConverter.save_json(xlsx_data, str(self.json_path))
        print(f"JSON-Datei aktualisiert: {self.json_path}")
    
    def _update_from_json(self) -> None:
        """
        Aktualisiert XLSX-Datei aus JSON-Datei
        
        Raises:
            FileNotFoundError: Wenn JSON nicht existiert
            ValueError: Wenn JSON ungültig ist
        """
        print(f"Aktualisiere XLSX aus JSON...")
        json_data = ConfigConverter.load_json(str(self.json_path))
        ConfigConverter.json_to_xlsx(json_data, str(self.xlsx_path))
        print(f"XLSX-Datei aktualisiert: {self.xlsx_path}")
