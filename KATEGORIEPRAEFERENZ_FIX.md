# Fix: Kategoriepräferenzen im deduktiven Modus

## Problem
- Im abduktiven Modus funktionierte die Kategoriepräferenz-Prüfung gut
- Im deduktiven Modus kam immer "Keine starken Kategoriepräferenzen (alle Scores < 0.6)"
- Unterschiedliche Datenstrukturen in den Modi führten zu Fehlern

## Ursache
In `QCA_Prompts.py` gab es eine fehlerhafte Fallback-Logik:

```python
# FEHLERHAFT:
category_descriptions[name] = {
    'definition': str(cat_def.get('definition', 'Keine Definition verfügbar')),
    'subcategories': []
}
```

**Problem**: Im deduktiven Modus ist `cat_def` ein String, nicht ein Dictionary. 
Der Aufruf `cat_def.get('definition', ...)` führte zu einem Fehler.

## Lösung
Robuste Typerkennung in beiden Prompt-Methoden:

```python
if hasattr(cat_def, 'definition') and hasattr(cat_def, 'subcategories'):
    # CategoryDefinition-Objekt (abduktive Analyse)
    category_descriptions[name] = {
        'definition': cat_def.definition,
        'subcategories': list(cat_def.subcategories.keys()) if cat_def.subcategories else []
    }
elif isinstance(cat_def, dict):
    # Dictionary-Format (bereits serialisiert)
    category_descriptions[name] = {
        'definition': cat_def.get('definition', 'Keine Definition verfügbar'),
        'subcategories': list(cat_def.get('subcategories', {}).keys()) if cat_def.get('subcategories') else []
    }
else:
    # String-Format (deduktive Analyse)
    category_descriptions[name] = {
        'definition': str(cat_def) if cat_def else 'Keine Definition verfügbar',
        'subcategories': []
    }
```

## Betroffene Methoden
- `get_category_preferences_prompt()`
- `get_relevance_with_category_preselection_prompt()`

## Ergebnis
- Kategoriepräferenzen funktionieren jetzt in beiden Modi korrekt
- Deduktive Analyse zeigt jetzt starke Kategoriepräferenzen (Scores ≥ 0.6)
- Robuste Behandlung aller Datenstrukturen (CategoryDefinition, Dict, String)