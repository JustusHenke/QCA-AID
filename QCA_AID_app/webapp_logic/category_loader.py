"""
Category Loader for QCA-AID Explorer Webapp
===========================================

This module loads category information from the "Kategorien" sheet of the Excel file
to provide intelligent dropdown options for filtering in the Explorer UI.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class CategoryLoader:
    """
    Lädt Kategorieinformationen aus dem "Kategorien"-Sheet einer Excel-Datei.
    
    Diese Klasse extrahiert Hauptkategorien und ihre zugehörigen Subkategorien
    aus der Excel-Datei, um intelligente Dropdown-Filter zu ermöglichen.
    
    Attributes:
        categories_df (pd.DataFrame): DataFrame mit Kategoriendaten
        main_categories (List[str]): Liste aller Hauptkategorien
        category_mapping (Dict[str, List[str]]): Mapping von Hauptkategorien zu Subkategorien
    """
    
    def __init__(self, excel_path: str):
        """
        Initialisiert den CategoryLoader.
        
        Args:
            excel_path: Pfad zur Excel-Datei mit dem "Kategorien"-Sheet
            
        Raises:
            FileNotFoundError: Wenn die Excel-Datei nicht existiert
            ValueError: Wenn das "Kategorien"-Sheet nicht gefunden wird
        """
        self.excel_path = excel_path
        self.categories_df = None
        self.main_categories = []
        self.category_mapping = {}
        self.is_loaded = False
        
        # Additional filter values from Kodierungsergebnisse sheet
        self.documents = []
        self.attribut1_values = []
        self.attribut2_values = []
        
        self._load_categories()
        self._load_filter_values()
    
    def _load_categories(self) -> None:
        """
        Lädt die Kategorien aus dem "Kategorien"-Sheet.
        
        Erwartet folgende Spaltenstruktur:
        - Hauptkategorie: Name der Hauptkategorie
        - Typ: Typ der Kategorie (z.B. "Grounded")
        - Definition: Beschreibung der Kategorie
        - Anzahl_Subkategorien: Anzahl der Subkategorien
        - Subkategorien: Komma-getrennte Liste der Subkategorien
        
        Raises:
            FileNotFoundError: Wenn die Excel-Datei nicht existiert
            ValueError: Wenn das "Kategorien"-Sheet nicht gefunden wird oder die Struktur ungültig ist
        """
        if not Path(self.excel_path).exists():
            raise FileNotFoundError(f"Excel-Datei nicht gefunden: {self.excel_path}")
        
        try:
            # Versuche das "Kategorien"-Sheet zu laden
            self.categories_df = pd.read_excel(self.excel_path, sheet_name='Kategorien')
            self.is_loaded = True
        except ValueError as e:
            if "Worksheet named 'Kategorien' not found" in str(e):
                # Versuche alternative Sheetnamen
                try:
                    excel_file = pd.ExcelFile(self.excel_path)
                    sheet_names = excel_file.sheet_names
                    
                    # Suche nach ähnlichen Sheetnamen
                    category_sheets = [name for name in sheet_names 
                                     if 'kategori' in name.lower() or 'category' in name.lower()]
                    
                    if category_sheets:
                        sheet_name = category_sheets[0]
                        self.categories_df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
                        self.is_loaded = True
                    else:
                        # Kein Kategorien-Sheet gefunden - das ist OK, wir verwenden Fallback
                        self.is_loaded = False
                        return
                        
                except Exception:
                    # Fehler beim Laden - verwende Fallback
                    self.is_loaded = False
                    return
            else:
                # Anderer Fehler - verwende Fallback
                self.is_loaded = False
                return
        
        if not self.is_loaded:
            return
        
        # Validiere die Spaltenstruktur
        required_columns = ['Hauptkategorie', 'Subkategorien']
        missing_columns = [col for col in required_columns if col not in self.categories_df.columns]
        
        if missing_columns:
            # Spalten fehlen - verwende Fallback
            self.is_loaded = False
            return
        
        # Extrahiere Hauptkategorien und Subkategorien
        self._extract_categories()
    
    def _extract_categories(self) -> None:
        """
        Extrahiert Hauptkategorien und Subkategorien aus dem DataFrame.
        
        Erstellt eine Mapping-Struktur von Hauptkategorien zu ihren Subkategorien.
        """
        self.main_categories = []
        self.category_mapping = {}
        
        for _, row in self.categories_df.iterrows():
            main_category = str(row['Hauptkategorie']).strip()
            
            # Überspringe leere Hauptkategorien
            if pd.isna(row['Hauptkategorie']) or main_category == '' or main_category == 'nan':
                continue
            
            # Füge Hauptkategorie zur Liste hinzu (falls noch nicht vorhanden)
            if main_category not in self.main_categories:
                self.main_categories.append(main_category)
            
            # Extrahiere Subkategorien
            subcategories_str = row['Subkategorien']
            subcategories = []
            
            if pd.notna(subcategories_str) and str(subcategories_str).strip():
                # Teile Subkategorien an Kommas und bereinige sie
                subcategories = [
                    sub.strip() 
                    for sub in str(subcategories_str).split(',') 
                    if sub.strip()
                ]
            
            # Speichere Mapping (überschreibt bei Duplikaten)
            self.category_mapping[main_category] = subcategories
        
        # Sortiere Hauptkategorien alphabetisch
        self.main_categories.sort()
    
    def get_main_categories(self) -> List[str]:
        """
        Gibt die Liste aller Hauptkategorien zurück.
        Filtert automatisch "nicht kodiert" heraus.
        
        Returns:
            Liste der Hauptkategorien, alphabetisch sortiert, ohne "nicht kodiert"
        """
        if not self.is_loaded:
            return []
        # Filter out "nicht kodiert" (case-insensitive)
        return [cat for cat in self.main_categories if cat.lower() != "nicht kodiert"]
    
    def get_subcategories(self, main_category: str) -> List[str]:
        """
        Gibt die Subkategorien für eine bestimmte Hauptkategorie zurück.
        Filtert automatisch "nicht kodiert" heraus.
        
        Args:
            main_category: Name der Hauptkategorie
            
        Returns:
            Liste der Subkategorien für die angegebene Hauptkategorie, ohne "nicht kodiert"
            Leere Liste, wenn die Hauptkategorie nicht existiert
        """
        if not self.is_loaded:
            return []
        subcats = self.category_mapping.get(main_category, [])
        # Filter out "nicht kodiert" (case-insensitive)
        return [sub for sub in subcats if sub.lower() != "nicht kodiert"]
    
    def get_all_subcategories(self) -> List[str]:
        """
        Gibt alle Subkategorien aus allen Hauptkategorien zurück.
        Filtert automatisch "nicht kodiert" heraus.
        
        Returns:
            Liste aller Subkategorien, alphabetisch sortiert, ohne Duplikate und ohne "nicht kodiert"
        """
        if not self.is_loaded:
            return []
        
        all_subcategories = set()
        for subcategories in self.category_mapping.values():
            all_subcategories.update(subcategories)
        
        # Filter out "nicht kodiert" (case-insensitive)
        filtered = [sub for sub in all_subcategories if sub.lower() != "nicht kodiert"]
        return sorted(filtered)
    
    def get_category_mapping(self) -> Dict[str, List[str]]:
        """
        Gibt das vollständige Mapping von Hauptkategorien zu Subkategorien zurück.
        
        Returns:
            Dictionary mit Hauptkategorien als Keys und Listen von Subkategorien als Values
        """
        if not self.is_loaded:
            return {}
        return self.category_mapping.copy()
    
    def get_category_info(self, main_category: str) -> Optional[Dict[str, Any]]:
        """
        Gibt detaillierte Informationen zu einer Hauptkategorie zurück.
        
        Args:
            main_category: Name der Hauptkategorie
            
        Returns:
            Dictionary mit Kategorieinformationen oder None wenn nicht gefunden
            Enthält: name, typ, definition, anzahl_subkategorien, subkategorien
        """
        if not self.is_loaded:
            return None
        
        # Finde die Zeile für diese Hauptkategorie
        category_row = self.categories_df[
            self.categories_df['Hauptkategorie'].str.strip() == main_category
        ]
        
        if category_row.empty:
            return None
        
        row = category_row.iloc[0]
        
        return {
            'name': main_category,
            'typ': row.get('Typ', ''),
            'definition': row.get('Definition', ''),
            'anzahl_subkategorien': row.get('Anzahl_Subkategorien', 0),
            'subkategorien': self.get_subcategories(main_category)
        }
    
    def validate_filter_values(self, main_category: Optional[str] = None, 
                             subcategories: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validiert Filter-Werte gegen die verfügbaren Kategorien.
        Unterstützt komma-getrennte Listen für Hauptkategorien.
        
        Args:
            main_category: Zu validierende Hauptkategorie(n) - kann komma-getrennt sein
            subcategories: Zu validierende Subkategorien
            
        Returns:
            Tuple aus (is_valid, error_messages)
        """
        if not self.is_loaded:
            return True, []  # Keine Validierung möglich ohne geladene Kategorien
        
        errors = []
        
        # Validiere Hauptkategorie(n)
        if main_category is not None:
            # Parse komma-getrennte Hauptkategorien
            main_categories_list = [cat.strip() for cat in main_category.split(',') if cat.strip()]
            
            invalid_main_cats = []
            for cat in main_categories_list:
                if cat not in self.main_categories:
                    invalid_main_cats.append(cat)
            
            if invalid_main_cats:
                errors.append(f"Unbekannte Hauptkategorie(n): {', '.join(invalid_main_cats)}")
                errors.append(f"Verfügbare Hauptkategorien: {', '.join(self.main_categories)}")
        
        # Validiere Subkategorien
        if subcategories is not None:
            all_subcategories = self.get_all_subcategories()
            
            invalid_subcats = []
            for subcategory in subcategories:
                if subcategory not in all_subcategories:
                    invalid_subcats.append(subcategory)
            
            if invalid_subcats:
                errors.append(f"Unbekannte Subkategorie(n): {', '.join(invalid_subcats)}")
            
            # Wenn Hauptkategorien angegeben sind, prüfe ob Subkategorien dazu passen
            if main_category is not None:
                main_categories_list = [cat.strip() for cat in main_category.split(',') if cat.strip()]
                valid_main_cats = [cat for cat in main_categories_list if cat in self.main_categories]
                
                if valid_main_cats:
                    # Sammle alle gültigen Subkategorien für die ausgewählten Hauptkategorien
                    valid_subcategories = set()
                    for main_cat in valid_main_cats:
                        valid_subcategories.update(self.get_subcategories(main_cat))
                    
                    invalid_subcategories = [
                        sub for sub in subcategories 
                        if sub not in valid_subcategories
                    ]
                    
                    if invalid_subcategories:
                        errors.append(
                            f"Subkategorien passen nicht zu den gewählten Hauptkategorien: "
                            f"{', '.join(invalid_subcategories)}"
                        )
                        errors.append(
                            f"Gültige Subkategorien für '{', '.join(valid_main_cats)}': "
                            f"{', '.join(sorted(valid_subcategories))}"
                        )
        
        return len(errors) == 0, errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über die geladenen Kategorien zurück.
        
        Returns:
            Dictionary mit Statistiken
        """
        if not self.is_loaded:
            return {
                'total_main_categories': 0,
                'total_subcategories': 0,
                'average_subcategories_per_main': 0,
                'max_subcategories': 0,
                'min_subcategories': 0,
                'categories_with_most_subcategories': [],
                'categories_with_least_subcategories': [],
                'main_categories': []
            }
        
        total_subcategories = sum(len(subs) for subs in self.category_mapping.values())
        avg_subcategories = total_subcategories / len(self.main_categories) if self.main_categories else 0
        
        # Finde Kategorien mit den meisten/wenigsten Subkategorien
        if self.category_mapping:
            max_subs = max(len(subs) for subs in self.category_mapping.values())
            min_subs = min(len(subs) for subs in self.category_mapping.values())
            
            categories_with_max = [
                cat for cat, subs in self.category_mapping.items() 
                if len(subs) == max_subs
            ]
            categories_with_min = [
                cat for cat, subs in self.category_mapping.items() 
                if len(subs) == min_subs
            ]
        else:
            max_subs = min_subs = 0
            categories_with_max = categories_with_min = []
        
        return {
            'total_main_categories': len(self.main_categories),
            'total_subcategories': total_subcategories,
            'average_subcategories_per_main': round(avg_subcategories, 2),
            'max_subcategories': max_subs,
            'min_subcategories': min_subs,
            'categories_with_most_subcategories': categories_with_max,
            'categories_with_least_subcategories': categories_with_min,
            'main_categories': self.main_categories
        }
    
    def _load_filter_values(self) -> None:
        """
        Lädt verfügbare Werte für Filter aus dem Kodierungsergebnisse-Sheet.
        
        Extrahiert eindeutige Werte für:
        - Dokument
        - Attribut1 (über ATTRIBUT1_LABEL aus Konfiguration)
        - Attribut2 (über ATTRIBUT2_LABEL aus Konfiguration)
        """
        try:
            # Lade zuerst die Attribut-Labels aus dem Konfiguration-Sheet
            config_df = pd.read_excel(self.excel_path, sheet_name='Konfiguration', header=None)
            
            # Finde die Labels für Attribut1 und Attribut2
            attribut1_label = None
            attribut2_label = None
            
            for idx, row in config_df.iterrows():
                if row[0] == 'ATTRIBUT1_LABEL':
                    attribut1_label = str(row[1]) if pd.notna(row[1]) else None
                elif row[0] == 'ATTRIBUT2_LABEL':
                    attribut2_label = str(row[1]) if pd.notna(row[1]) else None
            
            # Lade Kodierungsergebnisse-Sheet
            df = pd.read_excel(self.excel_path, sheet_name='Kodierungsergebnisse')
            
            # Extrahiere eindeutige Dokumente
            if 'Dokument' in df.columns:
                self.documents = sorted([str(doc) for doc in df['Dokument'].dropna().unique()])
            
            # Extrahiere eindeutige Attribut1-Werte (mit dynamischem Spaltennamen)
            if attribut1_label and attribut1_label in df.columns:
                self.attribut1_values = sorted([str(val) for val in df[attribut1_label].dropna().unique()])
            
            # Extrahiere eindeutige Attribut2-Werte (mit dynamischem Spaltennamen)
            if attribut2_label and attribut2_label in df.columns:
                self.attribut2_values = sorted([str(val) for val in df[attribut2_label].dropna().unique()])
                
        except Exception as e:
            # Fehler beim Laden - verwende leere Listen
            self.documents = []
            self.attribut1_values = []
            self.attribut2_values = []
    
    def get_documents(self) -> List[str]:
        """
        Gibt die Liste aller verfügbaren Dokumente zurück.
        
        Returns:
            Liste der Dokumentnamen, alphabetisch sortiert
        """
        return self.documents.copy()
    
    def get_attribut1_values(self) -> List[str]:
        """
        Gibt die Liste aller verfügbaren Attribut1-Werte zurück.
        
        Returns:
            Liste der Attribut1-Werte, alphabetisch sortiert
        """
        return self.attribut1_values.copy()
    
    def get_attribut2_values(self) -> List[str]:
        """
        Gibt die Liste aller verfügbaren Attribut2-Werte zurück.
        
        Returns:
            Liste der Attribut2-Werte, alphabetisch sortiert
        """
        return self.attribut2_values.copy()