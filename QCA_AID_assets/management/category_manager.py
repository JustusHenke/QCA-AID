"""
Kategorien-Manager für QCA-AID
===============================
Verwaltet das Kategoriensystem und dessen Änderungen.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..core.data_models import CategoryDefinition, CategoryChange
from ..core.validators import CategoryValidator
from ..core.config import CONFIG, FORSCHUNGSFRAGE, KODIERREGELN


class CategoryManager:
    """Verwaltet das Speichern und Laden von Kategorien"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.categories_file = os.path.join(output_dir, 'category_system.json')
        
    def save_categories(self, categories: Dict[str, CategoryDefinition]) -> None:
        """Speichert das aktuelle Kategoriensystem als JSON"""
        try:
            # Konvertiere CategoryDefinition Objekte in serialisierbares Format
            categories_dict = {
                name: {
                    'definition': cat.definition,
                    'examples': cat.examples,
                    'rules': cat.rules,
                    'subcategories': cat.subcategories,
                    'timestamp': datetime.now().isoformat()
                } for name, cat in categories.items()
            }
            
            # Metadata hinzuFügen
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'categories': categories_dict
            }
            
            with open(self.categories_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            print(f"\nKategoriensystem gespeichert in: {self.categories_file}")
            
        except Exception as e:
            print(f"Fehler beim Speichern des Kategoriensystems: {str(e)}")

    def load_categories(self) -> Optional[Dict[str, CategoryDefinition]]:
        """
        LÄdt gespeichertes Kategoriensystem falls vorhanden.
        
        Returns:
            Optional[Dict[str, CategoryDefinition]]: Dictionary mit Kategorienamen und deren Definitionen,
            oder None wenn kein gespeichertes System verwendet werden soll
        """
        try:
            if os.path.exists(self.categories_file):
                with open(self.categories_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Zeige Informationen zum gefundenen Kategoriensystem
                timestamp = datetime.fromisoformat(data['timestamp'])
                print(f"\nGefundenes Kategoriensystem vom {timestamp.strftime('%d.%m.%Y %H:%M')}")
                print(f"EnthÄlt {len(data['categories'])} Kategorien:")
                for cat_name in data['categories'].keys():
                    print(f"- {cat_name}")
                
                # Frage Benutzer
                while True:
                    answer = input("\nMÖchten Sie dieses Kategoriensystem verwenden? (j/n): ").lower()
                    if answer in ['j', 'n']:
                        break
                    print("Bitte antworten Sie mit 'j' oder 'n'")
                
                if answer == 'j':
                    # Konvertiere zurÜck zu CategoryDefinition Objekten
                    categories = {}
                    for name, cat_data in data['categories'].items():
                        # Stelle sicher, dass die Zeitstempel existieren
                        if 'timestamp' in cat_data:
                            added_date = modified_date = cat_data['timestamp'].split('T')[0]
                        else:
                            # Verwende aktuelle Zeit fuer fehlende Zeitstempel
                            current_date = datetime.now().strftime("%Y-%m-%d")
                            added_date = modified_date = current_date

                        categories[name] = CategoryDefinition(
                            name=name,
                            definition=cat_data['definition'],
                            examples=cat_data.get('examples', []),
                            rules=cat_data.get('rules', []),
                            subcategories=cat_data.get('subcategories', {}),
                            added_date=cat_data.get('added_date', added_date),
                            modified_date=cat_data.get('modified_date', modified_date)
                        )
                    
                    print(f"\nKategoriensystem erfolgreich geladen.")
                    return categories
                
            return None
                
        except Exception as e:
            print(f"Fehler beim Laden des Kategoriensystems: {str(e)}")
            print("Folgende Details kÖnnten hilfreich sein:")
            import traceback
            traceback.print_exc()
            return None
    
    def save_codebook(self, categories: Dict[str, CategoryDefinition], filename: str = "codebook_inductive.json", research_question: Optional[str] = None) -> None:
        """Speichert das vollständige Codebook inkl. deduktiver, induktiver und grounded Kategorien"""
        try:
            # Use provided research question or fall back to config/default
            actual_research_question = research_question or CONFIG.get('FORSCHUNGSFRAGE', FORSCHUNGSFRAGE)
            
            codebook_data = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "2.0",
                    "total_categories": len(categories),
                    "research_question": actual_research_question,
                    "analysis_mode": CONFIG.get('ANALYSIS_MODE', 'deductive')  # Speichere den Analysemodus
                },
                "categories": {}
            }
            
            for name, category in categories.items():
                # Bestimme den Kategorietyp je nach Analysemodus und Subkategorien-Entwicklung
                if name in CONFIG.get('DEDUKTIVE_KATEGORIEN', {}):
                    # Check if this deductive category was extended with abductive subcategories
                    has_abductive_subcategories = False
                    if hasattr(category, 'subcategories') and category.subcategories:
                        # Check if any subcategories were added recently (indicating abductive development)
                        original_deductive_cat = CONFIG.get('DEDUKTIVE_KATEGORIEN', {}).get(name)
                        if original_deductive_cat:
                            # Handle both CategoryDefinition objects and dictionary format
                            if hasattr(original_deductive_cat, 'subcategories'):
                                original_deductive_subcats = original_deductive_cat.subcategories or {}
                            elif isinstance(original_deductive_cat, dict):
                                original_deductive_subcats = original_deductive_cat.get('subcategories', {})
                            else:
                                original_deductive_subcats = {}
                        else:
                            original_deductive_subcats = {}
                            
                        for subcat_name, subcat_obj in category.subcategories.items():
                            # If subcategory is not in original deductive definition, it was added abductively
                            if subcat_name not in original_deductive_subcats:
                                has_abductive_subcategories = True
                                break
                    
                    if has_abductive_subcategories:
                        development_type = "abductive"  # Mark as abductive if extended
                        print(f"   🔄 Kategorie '{name}' als 'abductive' markiert (erweitert mit neuen Subkategorien)")
                    else:
                        development_type = "deductive"
                elif CONFIG.get('ANALYSIS_MODE') == 'grounded':
                    development_type = "grounded"  # Neue Markierung fuer grounded Kategorien
                else:
                    development_type = "inductive"
                    
                # Convert subcategories properly - they contain CategoryDefinition objects
                subcategories_dict = {}
                if hasattr(category, 'subcategories') and category.subcategories:
                    for subcat_name, subcat_obj in category.subcategories.items():
                        if isinstance(subcat_obj, CategoryDefinition):
                            # Convert CategoryDefinition to dict
                            subcategories_dict[subcat_name] = {
                                "definition": subcat_obj.definition,
                                "examples": list(subcat_obj.examples) if isinstance(subcat_obj.examples, set) else subcat_obj.examples,
                                "rules": list(subcat_obj.rules) if isinstance(subcat_obj.rules, set) else subcat_obj.rules,
                                "added_date": subcat_obj.added_date,
                                "last_modified": subcat_obj.modified_date
                            }
                        else:
                            # Handle string subcategories (legacy format)
                            subcategories_dict[subcat_name] = str(subcat_obj)
                
                codebook_data["categories"][name] = {
                    "definition": category.definition,
                    # Wandle examples in eine Liste um, falls es ein Set ist
                    "examples": list(category.examples) if isinstance(category.examples, set) else category.examples,
                    # Wandle rules in eine Liste um, falls es ein Set ist
                    "rules": list(category.rules) if isinstance(category.rules, set) else category.rules,
                    # Use properly converted subcategories
                    "subcategories": subcategories_dict,
                    "development_type": development_type,
                    "added_date": category.added_date,
                    "last_modified": category.modified_date
                }
            
            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(codebook_data, f, indent=2, ensure_ascii=False)
                
            print(f"\nCodebook gespeichert unter: {output_path}")
            print(f"- Deduktive Kategorien: {sum(1 for c in codebook_data['categories'].values() if c['development_type'] == 'deductive')}")
            print(f"- Abduktive Kategorien: {sum(1 for c in codebook_data['categories'].values() if c['development_type'] == 'abductive')}")
            print(f"- Induktive Kategorien: {sum(1 for c in codebook_data['categories'].values() if c['development_type'] == 'inductive')}")
            print(f"- Grounded Kategorien: {sum(1 for c in codebook_data['categories'].values() if c['development_type'] == 'grounded')}")
            
        except Exception as e:
            print(f"Fehler beim Speichern des Codebooks: {str(e)}")
            # Zusätzliche Fehlerdiagnose
            import traceback
            traceback.print_exc()


# --- Klasse: RelevanceChecker ---
# Aufgabe: Zentrale Klasse fuer Relevanzprüfungen mit Caching und Batch-Verarbeitung
