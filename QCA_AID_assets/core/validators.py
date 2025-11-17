"""
Validatoren für QCA-AID
=======================
Enthält Klassen zur Validierung von Kategorien und Kodierungen.
"""

from typing import Dict, List, Any, Set


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
