"""
Analysis Helper Functions

Statistical and analytical utilities for coding analysis.
"""

from collections import defaultdict, Counter
from typing import Dict, List, Any

# Platzhalter fuer Kodierungen ohne Kategorienamen (None/leer). Wird nur fuer
# die Anzeige der Kategorie-Kombinationen benoetigt.
EMPTY_CATEGORY_LABEL = '(ohne Kategorie)'


def _normalize_text(value: Any) -> str:
    """
    Wandelt einen beliebigen Wert in einen sicheren String um.

    Kodierungen koennen aus LLM-Antworten oder der Reliability-Database stammen
    und dort None als Kategorie/Coder-ID enthalten. Ohne Normalisierung
    scheitern String-Operationen wie sorted() mit
    "TypeError: '<' not supported between instances of 'NoneType' and 'str'".
    """
    if value is None:
        return ''
    return str(value).strip()


def _as_int(value: Any, default: int = 1) -> int:
    """Wandelt einen Wert robust in int um (Fallback: default)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def calculate_multiple_coding_stats(all_codings: List[Dict]) -> Dict[str, Any]:
    """
    Berechnung der Mehrfachkodierungs-Statistiken.
    
    Berücksichtigt sowohl:
    1. Mehrfachkodierung durch verschiedene Kodierer (deduktiver Modus)
    2. Echte Mehrfachkodierung (verschiedene Kategorien für gleichen Text)
    
    Args:
        all_codings: Liste aller Kodierungen
        
    Returns:
        Dict: Statistiken zur Mehrfachkodierung mit folgenden Schlüsseln:
            - segments_with_multiple: Hauptzähler für Mehrfachkodierungen
            - total_segments: Gesamtzahl Segmente
            - avg_codings_per_segment: Durchschnitt Kodierungen pro Segment
            - top_combinations: Top 5 Kategorie-Kombinationen
            - focus_adherence_rate: Rate der Fokus-Kategorien-Einhaltung
            - segments_with_multiple_coders: Segmente von mehreren Kodierern
            - segments_with_multiple_categories: Segmente mit mehreren Kategorien
            - segments_with_true_multiple_coding: Echte Mehrfachkodierungen
    """
    # Group by segment ID and analyze different types of multiple coding
    segment_codings = defaultdict(list)
    focus_adherence = []
    category_combinations = []
    
    for coding in all_codings:
        segment_id = _normalize_text(coding.get('segment_id', ''))
        segment_codings[segment_id].append(coding)
        
        # Track focus adherence
        if coding.get('category_focus_used', False):
            focus_adherence.append(
                _normalize_text(coding.get('target_category', ''))
                == _normalize_text(coding.get('category', ''))
            )
    
    # Analyze different types of multiple coding
    segments_with_multiple_coders = 0
    segments_with_multiple_categories = 0
    segments_with_true_multiple_coding = 0
    
    for segment_id, codings in segment_codings.items():
        if len(codings) > 1:
            # Different coders for same segment
            unique_coders = set(_normalize_text(c.get('coder_id', '')) for c in codings)
            if len(unique_coders) > 1:
                segments_with_multiple_coders += 1

            # Different categories for same segment
            unique_categories = set(
                _normalize_text(c.get('category', '')) for c in codings
            )
            if len(unique_categories) > 1:
                segments_with_multiple_categories += 1
                # Collect category combinations (None/leere Kategorien abfangen)
                category_combinations.append(
                    ' + '.join(
                        sorted(cat or EMPTY_CATEGORY_LABEL for cat in unique_categories)
                    )
                )
            
            # True multiple coding (different instances)
            multiple_instances = any(
                _as_int(c.get('total_coding_instances')) > 1 for c in codings
            )
            if multiple_instances:
                segments_with_true_multiple_coding += 1
    
    # Determine the dominant type of multiple coding for stats output
    if segments_with_multiple_coders > segments_with_true_multiple_coding:
        # Deductive mode: Count segments with multiple coders as multiple coding
        segments_with_multiple = segments_with_multiple_coders
    else:
        # True multiple coding mode: Count only true multiple codings
        segments_with_multiple = segments_with_true_multiple_coding
    
    total_segments = len(segment_codings)
    total_codings = len(all_codings)
    
    combination_counter = Counter(category_combinations)
    
    return {
        'segments_with_multiple': segments_with_multiple,
        'total_segments': total_segments,
        'avg_codings_per_segment': total_codings / total_segments if total_segments > 0 else 0,
        'top_combinations': [combo for combo, _ in combination_counter.most_common(5)],
        'focus_adherence_rate': sum(focus_adherence) / len(focus_adherence) if focus_adherence else 0,
        # Additional details for extended output (optional)
        'segments_with_multiple_coders': segments_with_multiple_coders,
        'segments_with_multiple_categories': segments_with_multiple_categories,
        'segments_with_true_multiple_coding': segments_with_true_multiple_coding
    }


__all__ = [
    'EMPTY_CATEGORY_LABEL',
    'calculate_multiple_coding_stats',
]
