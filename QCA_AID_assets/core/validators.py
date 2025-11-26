"""
Validatoren für QCA-AID
=======================
Enthält Klassen zur Validierung von Kategorien und Kodierungen.
"""

from typing import Dict, List, Any, Set, Tuple


class CategoryValidator:
    """
    Utility-Klasse für alle Kategorie- und Subkategorie-Validierungen
    Mit robuster Kategorie-Struktur-Erkennung
    """
    
    @staticmethod
    def validate_subcategories_for_category(subcategories: List[str], 
                                           main_category: str, 
                                           categories_dict: Dict[str, Any],
                                           warn_only: bool = True) -> List[str]:
        """
        Robuste Subkategorien-Validierung mit verbesserter Fehlerbehandlung
        
        Args:
            subcategories: Liste der zu validierenden Subkategorien
            main_category: Name der Hauptkategorie
            categories_dict: Dictionary mit Kategorie-Definitionen
            warn_only: Wenn True, nur warnen statt entfernen
            
        Returns:
            List[str]: Validierte Subkategorien-Liste
        """
        if not subcategories:
            return []
            
        if not categories_dict:
            print(f"[WARN] KRITISCH: Kein categories_dict verfuegbar fuer Validierung!")
            return [] if not warn_only else subcategories
            
        if main_category not in categories_dict:
            if warn_only:
                print(f"[WARN] WARNUNG: Hauptkategorie '{main_category}' nicht im Kategoriensystem gefunden")
                print(f"   Verfügbare Kategorien: {list(categories_dict.keys())[:5]}...")
                return subcategories
            else:
                print(f"   [FIX] Hauptkategorie '{main_category}' ungueltig - alle Subkategorien entfernt")
                return []
        
        # Verbesserte Extraktion der gültigen Subkategorien
        main_cat_def = categories_dict[main_category]
        valid_subcats = CategoryValidator._extract_valid_subcategories(main_cat_def)
        
        if not valid_subcats:
            if warn_only:
                print(f"⚠️ WARNUNG: Keine Subkategorien definiert für '{main_category}'")
                return subcategories
            else:
                print(f"[FIX] Keine gültigen Subkategorien für '{main_category}' - alle entfernt")
                return []
        
        # Validiere jede Subkategorie einzeln
        validated = []
        invalid = []
        
        for subcat in subcategories:
            if subcat in valid_subcats:
                validated.append(subcat)
            else:
                invalid.append(subcat)
        
        # Detailliertes Logging
        if invalid:
            if warn_only:
                print(f"⚠️ WARNUNG: {len(invalid)} ungültige Subkategorien für '{main_category}': {invalid}")
                print(f"   Gültige Subkategorien: {sorted(list(valid_subcats))}")
                return subcategories  # Behalte alle
            else:
                print(f"🔧 FIX: Entfernt {len(invalid)} ungültige Subkategorien für '{main_category}': {invalid}")
                print(f"   Behalten: {validated}")
                print(f"   Gültige Optionen: {sorted(list(valid_subcats))}")
                return validated
        
        return subcategories
    
    @staticmethod
    def _extract_valid_subcategories(category_def: Any) -> Set[str]:
        """
        Robuste Extraktion von Subkategorien aus verschiedenen Datenstrukturen
        
        Args:
            category_def: Kategorie-Definition (kann verschiedene Typen sein)
            
        Returns:
            Set[str]: Set der gültigen Subkategorien-Namen
        """
        valid_subcats = set()
        
        try:
            # Versuche verschiedene mögliche Strukturen
            
            # 1. Object mit subcategories-Attribut (häufigster Fall)
            if hasattr(category_def, 'subcategories'):
                subcats = category_def.subcategories
                
                if isinstance(subcats, dict):
                    # Dict mit Subkategorie-Name -> Definition
                    valid_subcats.update(subcats.keys())
                elif isinstance(subcats, (list, set, tuple)):
                    # Liste/Set von Subkategorie-Namen
                    valid_subcats.update(str(sub) for sub in subcats)
                elif isinstance(subcats, str):
                    # Einzelner String
                    valid_subcats.add(subcats)
                    
            # 2. Dictionary-Struktur direkt
            elif isinstance(category_def, dict):
                if 'subcategories' in category_def:
                    sub_def = category_def['subcategories']
                    if isinstance(sub_def, dict):
                        valid_subcats.update(sub_def.keys())
                    elif isinstance(sub_def, (list, set)):
                        valid_subcats.update(str(sub) for sub in sub_def)
                        
            # 3. Liste von Subkategorien direkt
            elif isinstance(category_def, (list, set, tuple)):
                valid_subcats.update(str(sub) for sub in category_def)
                
        except Exception as e:
            print(f"⚠️ FEHLER bei Subkategorien-Extraktion: {str(e)}")
            print(f"   Kategorie-Definition Typ: {type(category_def)}")
            if hasattr(category_def, '__dict__'):
                print(f"   Verfügbare Attribute: {list(category_def.__dict__.keys())}")
        
        return valid_subcats
    
    @staticmethod
    def validate_coding_consistency(coding_result: Dict[str, Any], 
                                  categories_dict: Dict[str, Any],
                                  fix_inconsistencies: bool = True) -> Dict[str, Any]:
        """
        Verbesserte Konsistenz-Validierung für einzelne Kodierungen
        """
        if not coding_result:
            return coding_result
        
        main_category = coding_result.get('category', '')
        subcategories = coding_result.get('subcategories', [])
        
        # Hauptkategorie validieren
        if main_category and main_category not in categories_dict:
            print(f"⚠️ INKONSISTENZ: Hauptkategorie '{main_category}' existiert nicht im Kategoriensystem")
            if fix_inconsistencies:
                coding_result['category'] = 'Nicht kodiert'
                coding_result['subcategories'] = []
                coding_result['justification'] = f"[KORRIGIERT] Ungültige Kategorie '{main_category}' → 'Nicht kodiert'"
                return coding_result
        
        # Subkategorien validieren - nur wenn Hauptkategorie gültig
        if main_category and main_category in categories_dict and subcategories:
            validated_subcats = CategoryValidator.validate_subcategories_for_category(
                subcategories, main_category, categories_dict, warn_only=not fix_inconsistencies
            )
            
            if fix_inconsistencies:
                removed_subcats = set(subcategories) - set(validated_subcats)
                if removed_subcats:
                    coding_result['subcategories'] = validated_subcats
                    original_justification = coding_result.get('justification', '')
                    coding_result['justification'] = f"{original_justification} [FIX: Entfernt ungültige Subkategorien: {list(removed_subcats)}]"
        
        return coding_result
    
    @staticmethod
    def validate_multiple_codings(codings: List[Dict[str, Any]], 
                                categories_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validiert eine Liste von Kodierungen (z.B. bei Mehrfachkodierung)
        
        Args:
            codings: Liste von Kodierungsergebnissen
            categories_dict: Verfügbares Kategoriensystem
            
        Returns:
            List[Dict[str, Any]]: Liste validierter Kodierungen
        """
        validated_codings = []
        
        for coding in codings:
            validated_coding = CategoryValidator.validate_coding_consistency(
                coding, categories_dict, fix_inconsistencies=True
            )
            validated_codings.append(validated_coding)
        
        return validated_codings
    
    @staticmethod
    def get_category_statistics(categories_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Erstellt Statistiken über das Kategoriensystem
        
        Args:
            categories_dict: Kategoriensystem-Dictionary
            
        Returns:
            Dict[str, Any]: Statistiken über Kategorien und Subkategorien
        """
        stats = {
            'total_main_categories': len(categories_dict),
            'categories_with_subcategories': 0,
            'total_subcategories': 0,
            'subcategories_per_category': {},
            'average_subcategories': 0.0,
            'category_details': {}
        }
        
        total_subcats = 0
        
        for cat_name, cat_def in categories_dict.items():
            valid_subcats = CategoryValidator._extract_valid_subcategories(cat_def)
            subcat_count = len(valid_subcats)
            
            if subcat_count > 0:
                stats['categories_with_subcategories'] += 1
                total_subcats += subcat_count
            
            stats['subcategories_per_category'][cat_name] = subcat_count
            stats['category_details'][cat_name] = {
                'subcategory_count': subcat_count,
                'subcategories': sorted(list(valid_subcats))
            }
        
        stats['total_subcategories'] = total_subcats
        stats['average_subcategories'] = total_subcats / max(1, len(categories_dict))
        
        return stats
    
    @staticmethod
    def find_category_conflicts(codings: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Findet Konflikte zwischen verschiedenen Kodierungen
        
        Args:
            codings: Liste von Kodierungsergebnissen für das gleiche Segment
            
        Returns:
            Dict[str, List[str]]: Dictionary mit Konflikt-Typen und Details
        """
        conflicts = {
            'main_category_conflicts': [],
            'subcategory_conflicts': [],
            'confidence_discrepancies': []
        }
        
        if len(codings) <= 1:
            return conflicts
        
        # Hauptkategorien-Konflikte
        main_categories = [coding.get('category', '') for coding in codings]
        if len(set(main_categories)) > 1:
            conflicts['main_category_conflicts'] = list(set(main_categories))
        
        # Subkategorien-Konflikte (nur bei gleicher Hauptkategorie)
        if len(set(main_categories)) == 1:
            subcategory_sets = []
            for coding in codings:
                subcats = set(coding.get('subcategories', []))
                subcategory_sets.append(subcats)
            
            if len(set(frozenset(s) for s in subcategory_sets)) > 1:
                conflicts['subcategory_conflicts'] = [list(s) for s in subcategory_sets]
        
        # Konfidenz-Diskrepanzen
        confidences = []
        for coding in codings:
            conf = coding.get('confidence', {})
            if isinstance(conf, dict):
                total_conf = conf.get('total', 0.0)
            else:
                total_conf = float(conf) if conf else 0.0
            confidences.append(total_conf)
        
        if confidences and (max(confidences) - min(confidences)) > 0.3:
            conflicts['confidence_discrepancies'] = confidences
        
        return conflicts


class ConfigValidator:
    """
    Validator für JSON-Konfigurationsdateien
    
    Validiert die Struktur und Datentypen von JSON-Konfigurationen
    für QCA-AID Explorer.
    """
    
    @staticmethod
    def validate_json_config(config_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validiert JSON-Konfigurationsstruktur und Datentypen
        
        Args:
            config_data: Dictionary mit Konfigurationsdaten
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
                - is_valid: True wenn Konfiguration gültig ist
                - error_messages: Liste von Fehlermeldungen (leer wenn gültig)
        """
        errors = []
        
        # Prüfe erforderliche Top-Level Keys
        if 'base_config' not in config_data:
            errors.append("Erforderlicher Key 'base_config' fehlt")
        
        if 'analysis_configs' not in config_data:
            errors.append("Erforderlicher Key 'analysis_configs' fehlt")
        
        # Wenn Top-Level Keys fehlen, können wir nicht weitermachen
        if errors:
            return False, errors
        
        # Validiere base_config
        base_config = config_data['base_config']
        if not isinstance(base_config, dict):
            errors.append(f"'base_config' muss ein Dictionary sein, ist aber {type(base_config).__name__}")
        else:
            # Validiere Datentypen in base_config
            base_config_errors = ConfigValidator._validate_base_config_types(base_config)
            errors.extend(base_config_errors)
        
        # Validiere analysis_configs
        analysis_configs = config_data['analysis_configs']
        if not isinstance(analysis_configs, list):
            errors.append(f"'analysis_configs' muss eine Liste sein, ist aber {type(analysis_configs).__name__}")
        else:
            # Validiere jede Analyse-Konfiguration
            for idx, analysis_config in enumerate(analysis_configs):
                analysis_errors = ConfigValidator._validate_analysis_config(analysis_config, idx)
                errors.extend(analysis_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_base_config_types(base_config: Dict[str, Any]) -> List[str]:
        """
        Validiert Datentypen in base_config
        
        Args:
            base_config: Dictionary mit Basis-Konfiguration
            
        Returns:
            Liste von Fehlermeldungen
        """
        errors = []
        
        # Definiere erwartete Typen für bekannte Parameter
        expected_types = {
            'provider': str,
            'model': str,
            'temperature': (int, float),
            'script_dir': str,
            'output_dir': str,
            'explore_file': str,
            'clean_keywords': bool,
            'similarity_threshold': (int, float)
        }
        
        for param_name, param_value in base_config.items():
            if param_name in expected_types:
                expected_type = expected_types[param_name]
                
                # Erlaube None oder leere Strings für String-Parameter
                if expected_type == str and (param_value is None or param_value == ''):
                    continue
                
                # Prüfe Typ
                if not isinstance(param_value, expected_type):
                    if isinstance(expected_type, tuple):
                        type_names = ' oder '.join(t.__name__ for t in expected_type)
                        errors.append(
                            f"base_config['{param_name}'] sollte {type_names} sein, "
                            f"ist aber {type(param_value).__name__}"
                        )
                    else:
                        errors.append(
                            f"base_config['{param_name}'] sollte {expected_type.__name__} sein, "
                            f"ist aber {type(param_value).__name__}"
                        )
                
                # Zusätzliche Bereichsprüfungen
                if param_name == 'temperature' and isinstance(param_value, (int, float)):
                    if param_value < 0.0 or param_value > 2.0:
                        errors.append(
                            f"base_config['temperature'] sollte zwischen 0.0 und 2.0 liegen, "
                            f"ist aber {param_value}"
                        )
                
                if param_name == 'similarity_threshold' and isinstance(param_value, (int, float)):
                    if param_value < 0.0 or param_value > 1.0:
                        errors.append(
                            f"base_config['similarity_threshold'] sollte zwischen 0.0 und 1.0 liegen, "
                            f"ist aber {param_value}"
                        )
        
        return errors
    
    @staticmethod
    def _validate_analysis_config(analysis_config: Dict[str, Any], index: int) -> List[str]:
        """
        Validiert eine einzelne Analyse-Konfiguration
        
        Args:
            analysis_config: Dictionary mit Analyse-Konfiguration
            index: Index in der analysis_configs Liste
            
        Returns:
            Liste von Fehlermeldungen
        """
        errors = []
        
        if not isinstance(analysis_config, dict):
            errors.append(
                f"analysis_configs[{index}] muss ein Dictionary sein, "
                f"ist aber {type(analysis_config).__name__}"
            )
            return errors
        
        # Prüfe erforderliche Keys
        if 'name' not in analysis_config:
            errors.append(f"analysis_configs[{index}] fehlt erforderlicher Key 'name'")
        elif not isinstance(analysis_config['name'], str):
            errors.append(
                f"analysis_configs[{index}]['name'] sollte String sein, "
                f"ist aber {type(analysis_config['name']).__name__}"
            )
        
        # Prüfe filters (optional, aber wenn vorhanden muss es ein Dict sein)
        if 'filters' in analysis_config:
            if not isinstance(analysis_config['filters'], dict):
                errors.append(
                    f"analysis_configs[{index}]['filters'] sollte Dictionary sein, "
                    f"ist aber {type(analysis_config['filters']).__name__}"
                )
        
        # Prüfe params (optional, aber wenn vorhanden muss es ein Dict sein)
        if 'params' in analysis_config:
            if not isinstance(analysis_config['params'], dict):
                errors.append(
                    f"analysis_configs[{index}]['params'] sollte Dictionary sein, "
                    f"ist aber {type(analysis_config['params']).__name__}"
                )
            else:
                # Prüfe 'active' Parameter wenn vorhanden
                params = analysis_config['params']
                if 'active' in params and not isinstance(params['active'], bool):
                    errors.append(
                        f"analysis_configs[{index}]['params']['active'] sollte Boolean sein, "
                        f"ist aber {type(params['active']).__name__}"
                    )
        
        return errors
    
    @staticmethod
    def validate_json_format(json_path: str) -> Tuple[bool, List[str]]:
        """
        Validiert JSON-Datei auf korrekte Formatierung
        
        Prüft:
        - UTF-8 Encoding
        - Einrückung (indent)
        - Erforderliche Top-Level Keys
        
        Args:
            json_path: Pfad zur JSON-Datei
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        import json
        import os
        
        errors = []
        
        if not os.path.exists(json_path):
            errors.append(f"JSON-Datei nicht gefunden: {json_path}")
            return False, errors
        
        try:
            # Prüfe UTF-8 Encoding
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content)
            
            # Prüfe erforderliche Keys
            if 'base_config' not in data:
                errors.append("JSON fehlt erforderlicher Key 'base_config'")
            
            if 'analysis_configs' not in data:
                errors.append("JSON fehlt erforderlicher Key 'analysis_configs'")
            
            # Prüfe Einrückung (indent)
            # Wenn die Datei eingerückt ist, sollte sie Newlines und Spaces enthalten
            if '\n' not in content:
                errors.append("JSON ist nicht formatiert (keine Zeilenumbrüche)")
            elif '  ' not in content and '\t' not in content:
                errors.append("JSON ist nicht eingerückt")
            
        except UnicodeDecodeError:
            errors.append("JSON-Datei verwendet nicht UTF-8 Encoding")
        except json.JSONDecodeError as e:
            errors.append(f"Ungültiges JSON-Format: {str(e)}")
        except Exception as e:
            errors.append(f"Fehler beim Lesen der JSON-Datei: {str(e)}")
        
        return len(errors) == 0, errors
