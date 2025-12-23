"""
Codebook Manager
================
Manages QCA-AID codebook operations including categories, rules, and validation.
"""

import sys
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

# Add parent directory to path to access QCA_AID_assets
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import webapp models
from QCA_AID_app.webapp_models.codebook_data import CodebookData, CategoryData

# Import existing QCA-AID components
from QCA_AID_assets.utils.config.converter import ConfigConverter


class CodebookManager:
    """
    Verwaltet QCA-AID Codebook für die Webapp.
    
    Responsibilities:
    - Verwaltet Codebook-Datenstruktur
    - Fügt/Entfernt Kategorien und Subkategorien hinzu
    - Validiert Kategoriedefinitionen
    - Generiert JSON-Vorschau
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    
    def __init__(self, project_dir: Optional[str] = None):
        """
        Initialisiert CodebookManager.
        
        Args:
            project_dir: Projektverzeichnis (Standard: aktuelles Verzeichnis)
        """
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.xlsx_path = self.project_dir / "QCA-AID-Codebook.xlsx"
        self.json_path = self.project_dir / "QCA-AID-Codebook.json"
        self.codebook: Optional[CodebookData] = None
    
    def load_codebook(self, file_path: Optional[str] = None, format: Optional[str] = None) -> Tuple[bool, Optional[CodebookData], List[str]]:
        """
        Lädt Codebook aus XLSX oder JSON.
        
        Args:
            file_path: Pfad zur Codebook-Datei (optional)
            format: Format ('xlsx' oder 'json', wird automatisch erkannt wenn None)
            
        Returns:
            Tuple[bool, Optional[CodebookData], List[str]]: 
                (success, codebook_data, error_messages)
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
                    return False, None, ["Keine Codebook-Datei gefunden"]
            
            # Lade basierend auf Format
            if format == 'json':
                success, codebook_dict, load_errors = self._load_from_json(str(file_path))
            elif format == 'xlsx':
                success, codebook_dict, load_errors = self._load_from_xlsx(str(file_path))
            else:
                return False, None, [f"Ungültiges Format: {format}"]
            
            if not success:
                return False, None, load_errors
            
            # Konvertiere zu CodebookData
            try:
                codebook_data = CodebookData.from_dict(codebook_dict)
                self.codebook = codebook_data
                
                # Validiere Codebook
                is_valid, validation_errors = codebook_data.validate()
                if not is_valid:
                    errors.extend(validation_errors)
                    # Gebe trotzdem codebook_data zurück, damit UI Werte anzeigen kann
                    return False, codebook_data, errors
                
                return True, codebook_data, []
                
            except Exception as e:
                return False, None, [f"Fehler beim Konvertieren des Codebooks: {str(e)}"]
                
        except Exception as e:
            return False, None, [f"Fehler beim Laden des Codebooks: {str(e)}"]
    
    def save_codebook(self, codebook: CodebookData, file_path: Optional[str] = None, format: str = 'json') -> Tuple[bool, List[str]]:
        """
        Speichert Codebook in XLSX oder JSON.
        
        Args:
            codebook: CodebookData Objekt zum Speichern
            file_path: Pfad zur Zieldatei (optional, verwendet Standard-Pfad)
            format: Format ('xlsx' oder 'json')
            
        Returns:
            Tuple[bool, List[str]]: (success, error_messages)
        """
        errors = []
        
        try:
            # Ensure all categories are CategoryData instances before validation
            for cat_name, category in list(codebook.deduktive_kategorien.items()):
                # Check by type name instead of isinstance (handles import conflicts)
                cat_type_name = type(category).__name__
                
                if cat_type_name == 'CategoryData':
                    # It's already a CategoryData (even if isinstance fails due to import issues)
                    continue
                elif isinstance(category, dict):
                    # Convert dict to CategoryData
                    if 'name' not in category:
                        category['name'] = cat_name
                    codebook.deduktive_kategorien[cat_name] = CategoryData.from_dict(category)
                else:
                    # Unknown type - return error with debug info
                    return False, [f"Category '{cat_name}' has unexpected type: {cat_type_name}. Expected CategoryData or dict."]
            
            # Validiere Codebook vor dem Speichern
            is_valid, validation_errors = codebook.validate()
            if not is_valid:
                return False, validation_errors
            
            # Bestimme Dateipfad
            if file_path:
                file_path = Path(file_path)
            else:
                if format == 'json':
                    file_path = self.json_path
                elif format == 'xlsx':
                    file_path = self.xlsx_path
                else:
                    return False, [f"Ungültiges Format: {format}"]
            
            # Konvertiere CodebookData zu Dictionary
            codebook_dict = codebook.to_dict()
            
            # Speichere basierend auf Format
            if format == 'json':
                success, save_errors = self._save_to_json(codebook_dict, str(file_path))
            elif format == 'xlsx':
                success, save_errors = self._save_to_xlsx(codebook_dict, str(file_path))
            else:
                return False, [f"Ungültiges Format: {format}"]
            
            if not success:
                return False, save_errors
            
            # Update internal state
            self.codebook = codebook
            
            return True, []
            
        except Exception as e:
            return False, [f"Fehler beim Speichern des Codebooks: {str(e)}"]
    
    def add_category(self, name: str, definition: str, rules: List[str], 
                     examples: List[str], subcategories: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Fügt neue Kategorie hinzu.
        
        Requirement 4.3: WHEN der Codebook-Reiter angezeigt wird 
                        THEN das System SHALL eine Schaltfläche zum HinzuFügen neuer Hauptkategorien anzeigen
        
        Args:
            name: Kategoriename
            definition: Kategoriedefinition
            rules: Liste von Regeln
            examples: Liste von Beispielen
            subcategories: Dictionary von Subkategorien
            
        Returns:
            Tuple[bool, List[str]]: (success, error_messages)
        """
        errors = []
        
        try:
            # Stelle sicher dass Codebook geladen ist
            if self.codebook is None:
                # Erstelle neues leeres Codebook
                self.codebook = CodebookData(
                    forschungsfrage='',
                    kodierregeln={'general': [], 'format': [], 'exclusion': []},
                    deduktive_kategorien={}
                )
            
            # Prüfe ob Kategorie bereits existiert
            if name in self.codebook.deduktive_kategorien:
                return False, [f"Kategorie '{name}' existiert bereits"]
            
            # Erstelle neue Kategorie
            new_category = CategoryData(
                name=name,
                definition=definition,
                rules=rules,
                examples=examples,
                subcategories=subcategories,
                added_date=datetime.now().strftime("%Y-%m-%d"),
                modified_date=datetime.now().strftime("%Y-%m-%d")
            )
            
            # Validiere Kategorie
            is_valid, validation_errors = self.validate_category(new_category)
            if not is_valid:
                return False, validation_errors
            
            # Füge Kategorie hinzu
            self.codebook.deduktive_kategorien[name] = new_category
            
            return True, []
            
        except Exception as e:
            return False, [f"Fehler beim HinzuFügen der Kategorie: {str(e)}"]
    
    def remove_category(self, name: str) -> Tuple[bool, List[str]]:
        """
        Entfernt Kategorie mit Konsistenzprüfung.
        
        Args:
            name: Name der zu entfernenden Kategorie
            
        Returns:
            Tuple[bool, List[str]]: (success, error_messages)
        """
        errors = []
        
        try:
            # Stelle sicher dass Codebook geladen ist
            if self.codebook is None:
                return False, ["Kein Codebook geladen"]
            
            # Prüfe ob Kategorie existiert
            if name not in self.codebook.deduktive_kategorien:
                return False, [f"Kategorie '{name}' nicht gefunden"]
            
            # Entferne Kategorie
            del self.codebook.deduktive_kategorien[name]
            
            # Prüfe Konsistenz: Mindestens eine Kategorie muss vorhanden sein
            if len(self.codebook.deduktive_kategorien) == 0:
                return False, ["Mindestens eine Kategorie muss vorhanden sein"]
            
            return True, []
            
        except Exception as e:
            return False, [f"Fehler beim Entfernen der Kategorie: {str(e)}"]
    
    def update_category(self, name: str, **kwargs) -> Tuple[bool, List[str]]:
        """
        Aktualisiert Kategorie für alle Felder.
        
        Requirement 4.2: WHEN eine Kategorie erweitert wird 
                        THEN das System SHALL Eingabefelder für Definition, Regeln und Beispiele anzeigen
        
        Args:
            name: Name der zu aktualisierenden Kategorie
            **kwargs: Felder zum Aktualisieren (definition, rules, examples, subcategories)
            
        Returns:
            Tuple[bool, List[str]]: (success, error_messages)
        """
        errors = []
        
        try:
            # Stelle sicher dass Codebook geladen ist
            if self.codebook is None:
                return False, ["Kein Codebook geladen"]
            
            # Prüfe ob Kategorie existiert
            if name not in self.codebook.deduktive_kategorien:
                return False, [f"Kategorie '{name}' nicht gefunden"]
            
            # Hole aktuelle Kategorie
            category = self.codebook.deduktive_kategorien[name]
            
            # Aktualisiere Felder
            if 'definition' in kwargs:
                category.definition = kwargs['definition']
            if 'rules' in kwargs:
                category.rules = kwargs['rules']
            if 'examples' in kwargs:
                category.examples = kwargs['examples']
            if 'subcategories' in kwargs:
                category.subcategories = kwargs['subcategories']
            
            # Update modified date
            category.modified_date = datetime.now().strftime("%Y-%m-%d")
            
            # Validiere aktualisierte Kategorie
            is_valid, validation_errors = self.validate_category(category)
            if not is_valid:
                return False, validation_errors
            
            return True, []
            
        except Exception as e:
            return False, [f"Fehler beim Aktualisieren der Kategorie: {str(e)}"]
    
    def validate_category(self, category: CategoryData) -> Tuple[bool, List[str]]:
        """
        Validiert Kategorie mit INCOSE-Regeln.
        
        Args:
            category: CategoryData Objekt zum Validieren
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        return category.validate()
    
    def to_json_preview(self) -> str:
        """
        Generiert JSON-Vorschau des aktuellen Codebooks.
        
        Requirement 4.5: WHEN Änderungen am Codebook vorgenommen werden 
                        THEN das System SHALL eine Vorschau der JSON-Struktur anzeigen
        
        Returns:
            str: Formatierte JSON-Vorschau
        """
        try:
            if self.codebook is None:
                return json.dumps({
                    'forschungsfrage': '',
                    'kodierregeln': {'general': [], 'format': [], 'exclusion': []},
                    'deduktive_kategorien': {}
                }, ensure_ascii=False, indent=2)
            
            # Konvertiere zu Dictionary
            codebook_dict = self.codebook.to_dict()
            
            # Formatiere als JSON
            return json.dumps(codebook_dict, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return f"Fehler beim Generieren der JSON-Vorschau: {str(e)}"
    
    def get_category(self, name: str) -> Optional[CategoryData]:
        """
        Gibt Kategorie zurück.
        
        Args:
            name: Name der Kategorie
            
        Returns:
            Optional[CategoryData]: Kategorie oder None wenn nicht gefunden
        """
        if self.codebook is None:
            return None
        
        return self.codebook.deduktive_kategorien.get(name)
    
    def list_categories(self) -> List[str]:
        """
        Gibt Liste aller Kategorienamen zurück.
        
        Returns:
            List[str]: Liste von Kategorienamen
        """
        if self.codebook is None:
            return []
        
        return list(self.codebook.deduktive_kategorien.keys())
    
    # Private helper methods
    
    def _load_from_json(self, json_path: str) -> Tuple[bool, Optional[Dict], List[str]]:
        """
        Lädt Codebook aus JSON-Datei.
        
        Args:
            json_path: Pfad zur JSON-Datei
            
        Returns:
            Tuple[bool, Optional[Dict], List[str]]: (success, codebook_dict, errors)
        """
        try:
            # Verwende ConfigConverter zum Laden
            json_data = ConfigConverter.load_json(json_path)
            
            return True, json_data, []
            
        except FileNotFoundError as e:
            return False, None, [f"JSON-Datei nicht gefunden: {str(e)}"]
        except ValueError as e:
            return False, None, [f"Ungültiges JSON-Format: {str(e)}"]
        except Exception as e:
            return False, None, [f"Fehler beim Laden der JSON-Datei: {str(e)}"]
    
    def _load_from_xlsx(self, xlsx_path: str) -> Tuple[bool, Optional[Dict], List[str]]:
        """
        Lädt Codebook aus XLSX-Datei.
        
        Args:
            xlsx_path: Pfad zur XLSX-Datei
            
        Returns:
            Tuple[bool, Optional[Dict], List[str]]: (success, codebook_dict, errors)
        """
        try:
            # Verwende ConfigConverter zum Laden
            codebook_dict = ConfigConverter.qca_aid_xlsx_to_json(xlsx_path)
            
            return True, codebook_dict, []
            
        except FileNotFoundError as e:
            return False, None, [f"XLSX-Datei nicht gefunden: {str(e)}"]
        except Exception as e:
            return False, None, [f"Fehler beim Laden der XLSX-Datei: {str(e)}"]
    
    def _save_to_json(self, codebook_dict: Dict, json_path: str) -> Tuple[bool, List[str]]:
        """
        Speichert Codebook in JSON-Datei.
        
        Args:
            codebook_dict: Codebook als Dictionary
            json_path: Pfad zur Ziel-JSON-Datei
            
        Returns:
            Tuple[bool, List[str]]: (success, errors)
        """
        try:
            # Erstelle vollständige JSON-Struktur
            json_data = codebook_dict.copy()
            
            # Wenn JSON bereits existiert, behalte config-Sektion
            if Path(json_path).exists():
                try:
                    existing_data = ConfigConverter.load_json(json_path)
                    if 'config' in existing_data:
                        json_data['config'] = existing_data['config']
                except:
                    pass  # Verwende Daten ohne config wenn Laden fehlschlägt
            
            # Speichere mit ConfigConverter
            ConfigConverter.save_json(json_data, json_path)
            
            return True, []
            
        except IOError as e:
            return False, [f"Fehler beim Schreiben der JSON-Datei: {str(e)}"]
        except Exception as e:
            return False, [f"Fehler beim Speichern der JSON-Datei: {str(e)}"]
    
    def _save_to_xlsx(self, codebook_dict: Dict, xlsx_path: str) -> Tuple[bool, List[str]]:
        """
        Speichert Codebook in XLSX-Datei.
        
        Args:
            codebook_dict: Codebook als Dictionary
            xlsx_path: Pfad zur Ziel-XLSX-Datei
            
        Returns:
            Tuple[bool, List[str]]: (success, errors)
        """
        try:
            # Erstelle vollständige JSON-Struktur für Konvertierung
            json_data = codebook_dict.copy()
            
            # Wenn XLSX bereits existiert, lade existierende config
            if Path(xlsx_path).exists():
                try:
                    existing_data = ConfigConverter.qca_aid_xlsx_to_json(xlsx_path)
                    if 'config' in existing_data:
                        json_data['config'] = existing_data['config']
                except:
                    pass  # Verwende Daten ohne config wenn Laden fehlschlägt
            
            # Stelle sicher dass config existiert (minimal)
            if 'config' not in json_data:
                json_data['config'] = {}
            
            # Konvertiere zu XLSX mit ConfigConverter
            ConfigConverter.qca_aid_json_to_xlsx(json_data, xlsx_path)
            
            return True, []
            
        except ValueError as e:
            return False, [f"Fehler beim Konvertieren zu XLSX: {str(e)}"]
        except Exception as e:
            return False, [f"Fehler beim Speichern der XLSX-Datei: {str(e)}"]
