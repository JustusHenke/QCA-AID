"""
Configuration Converter

Converts between Excel (XLSX) and JSON configuration formats for QCA-AID Explorer.
Handles bidirectional conversion while preserving data structure.
"""

import json
import os
from typing import Dict, Any
import pandas as pd
from openpyxl import Workbook, load_workbook


class ConfigConverter:
    """Konvertiert zwischen XLSX und JSON Konfigurationsformaten"""
    
    @staticmethod
    def xlsx_to_json(xlsx_path: str) -> Dict[str, Any]:
        """
        Konvertiert Excel-Konfigurationsdatei zu JSON-Struktur
        
        Args:
            xlsx_path: Pfad zur Excel-Datei
            
        Returns:
            Dictionary mit 'base_config' und 'analysis_configs'
            
        Raises:
            FileNotFoundError: Wenn Excel-Datei nicht existiert
            ValueError: Wenn Excel-Struktur ungültig ist
        """
        if not os.path.exists(xlsx_path):
            raise FileNotFoundError(f"Excel-Datei nicht gefunden: {xlsx_path}")
        
        try:
            # Lese Basis-Sheet
            # keep_default_na=False verhindert dass Strings wie "null", "NA" als NaN interpretiert werden
            base_df = pd.read_excel(xlsx_path, sheet_name='Basis', keep_default_na=False)
            base_config = {}
            
            # Definiere erwartete Typen für base_config Parameter
            base_config_types = {
                'clean_keywords': bool,
                'temperature': float,
                'similarity_threshold': float
            }
            
            for _, row in base_df.iterrows():
                param_name = str(row['Parameter'])
                param_value = row['Wert']
                
                # Mit keep_default_na=False sind leere Zellen bereits leere Strings
                # Aber wir müssen noch NaN-Werte behandeln die durch andere Wege entstehen können
                if pd.isna(param_value):
                    param_value = ''
                
                # Konvertiere zu erwartetem Typ falls bekannt
                if param_name in base_config_types and param_value != '':
                    expected_type = base_config_types[param_name]
                    if expected_type == bool:
                        # Konvertiere 0/1 zurück zu bool
                        if isinstance(param_value, (int, float)):
                            param_value = bool(param_value)
                    elif expected_type == float:
                        # Stelle sicher dass Floats als Float bleiben
                        if isinstance(param_value, (int, float)):
                            param_value = float(param_value)
                    
                base_config[param_name] = param_value
            
            # Lese alle anderen Sheets für Analyse-Konfigurationen
            excel = pd.ExcelFile(xlsx_path)
            sheet_names = excel.sheet_names
            analysis_configs = []
            
            for sheet_name in sheet_names:
                if sheet_name.lower() != 'basis':
                    # keep_default_na=False verhindert dass Strings wie "null", "NA" als NaN interpretiert werden
                    analysis_df = pd.read_excel(xlsx_path, sheet_name=sheet_name, keep_default_na=False)
                    
                    analysis_config = {'name': sheet_name}
                    filter_params = {}
                    other_params = {}
                    
                    for _, row in analysis_df.iterrows():
                        param_name = str(row['Parameter'])
                        param_value = row['Wert']
                        
                        # Spezielle Behandlung für active/enabled Parameter
                        if param_name.lower() == 'active' or param_name.lower() == 'enabled':
                            if pd.isna(param_value):
                                param_value = True
                            elif isinstance(param_value, str):
                                param_value = param_value.lower() in ('true', 'ja', 'yes', '1')
                            else:
                                param_value = bool(param_value)
                            other_params['active'] = param_value
                            continue
                        
                        # Nur NaN/NaT als None behandeln, nicht leere Strings
                        if pd.isna(param_value):
                            param_value = None
                        elif isinstance(param_value, str) and param_value == '':
                            # Leere Strings bleiben leere Strings
                            param_value = ''
                        
                        # Filter-Parameter vs. andere Parameter
                        if param_name.startswith('filter_'):
                            filter_name = param_name[7:]
                            filter_params[filter_name] = param_value
                        else:
                            other_params[param_name] = param_value
                    
                    # Stelle sicher, dass 'active' existiert
                    if 'active' not in other_params:
                        other_params['active'] = True
                    
                    analysis_config['filters'] = filter_params
                    analysis_config['params'] = other_params
                    analysis_configs.append(analysis_config)
            
            return {
                'base_config': base_config,
                'analysis_configs': analysis_configs
            }
            
        except Exception as e:
            raise ValueError(f"Fehler beim Lesen der Excel-Datei: {str(e)}")
    
    @staticmethod
    def json_to_xlsx(json_data: Dict[str, Any], xlsx_path: str) -> None:
        """
        Schreibt JSON-Struktur in Excel-Datei
        
        Args:
            json_data: Dictionary mit 'base_config' und 'analysis_configs'
            xlsx_path: Pfad zur Ziel-Excel-Datei
            
        Raises:
            ValueError: Wenn JSON-Struktur ungültig ist
        """
        if 'base_config' not in json_data or 'analysis_configs' not in json_data:
            raise ValueError("JSON muss 'base_config' und 'analysis_configs' enthalten")
        
        try:
            wb = Workbook()
            
            # Erstelle Basis-Sheet
            ws_basis = wb.active
            ws_basis.title = 'Basis'
            ws_basis.append(['Parameter', 'Wert'])
            
            for param_name, param_value in json_data['base_config'].items():
                # Konvertiere None zu leerem String für Excel
                if param_value is None:
                    param_value = ''
                ws_basis.append([param_name, param_value])
            
            # Erstelle Sheets für Analyse-Konfigurationen
            for analysis_config in json_data['analysis_configs']:
                sheet_name = analysis_config['name']
                ws = wb.create_sheet(title=sheet_name)
                ws.append(['Parameter', 'Wert'])
                
                # Schreibe Filter-Parameter
                filters = analysis_config.get('filters', {})
                for filter_name, filter_value in filters.items():
                    # Konvertiere None zu leerem String für Excel
                    if filter_value is None:
                        filter_value = ''
                    ws.append([f'filter_{filter_name}', filter_value])
                
                # Schreibe andere Parameter
                params = analysis_config.get('params', {})
                for param_name, param_value in params.items():
                    # Konvertiere None zu leerem String für Excel
                    if param_value is None:
                        param_value = ''
                    ws.append([param_name, param_value])
            
            # Speichere Workbook
            wb.save(xlsx_path)
            wb.close()
            
        except Exception as e:
            raise ValueError(f"Fehler beim Schreiben der Excel-Datei: {str(e)}")
    
    @staticmethod
    def save_json(json_data: Dict[str, Any], json_path: str) -> None:
        """
        Speichert JSON-Daten mit UTF-8 Encoding und Formatierung
        
        Args:
            json_data: Dictionary zum Speichern
            json_path: Pfad zur Ziel-JSON-Datei
            
        Raises:
            IOError: Wenn Datei nicht geschrieben werden kann
        """
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise IOError(f"Fehler beim Schreiben der JSON-Datei: {str(e)}")
    
    @staticmethod
    def load_json(json_path: str) -> Dict[str, Any]:
        """
        Lädt JSON-Datei mit Fehlerbehandlung
        
        Args:
            json_path: Pfad zur JSON-Datei
            
        Returns:
            Dictionary mit JSON-Daten
            
        Raises:
            FileNotFoundError: Wenn JSON-Datei nicht existiert
            ValueError: Wenn JSON ungültig ist
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON-Datei nicht gefunden: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ungültiges JSON-Format: {str(e)}")
        except Exception as e:
            raise IOError(f"Fehler beim Lesen der JSON-Datei: {str(e)}")
