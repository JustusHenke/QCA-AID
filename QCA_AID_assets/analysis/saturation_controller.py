"""
Sättigungs-Controller für QCA-AID
==================================
Kontrolliert die theoretische Sättigung bei induktiver Kategorienentwicklung.
"""

from typing import Dict, List, Set, Optional
from collections import defaultdict

from ..core.data_models import CategoryDefinition


class ImprovedSaturationController:
    """
    Verbesserte Sättigungskontrolle mit modusabhÄngigen Kriterien
    """
    
    def __init__(self, analysis_mode: str):
        self.analysis_mode = analysis_mode
        self.stability_counter = 0
        self.saturation_history = []
        
        # ModusabhÄngige Schwellenwerte
        if analysis_mode == 'inductive':
            self.min_batches = 5
            self.min_material = 0.7
            self.min_stability = 3
            self.min_theoretical = 0.8
        elif analysis_mode == 'abductive':
            self.min_batches = 3
            self.min_material = 0.6
            self.min_stability = 2
            self.min_theoretical = 0.7
        else:  # grounded
            self.min_batches = 4
            self.min_material = 0.8
            self.min_stability = 2
            self.min_theoretical = 0.75

    def assess_saturation(self, current_categories: Dict, material_percentage: float, 
                         batch_count: int, total_segments: int) -> Dict:
        """
        Umfassende Sättigungsbeurteilung
        """
        # Berechne theoretische Sättigung
        theoretical_saturation = self._calculate_theoretical_saturation(current_categories)
        
        # Berechne Kategorienqualität
        category_quality = self._assess_category_quality(current_categories)
        
        # Prüfe alle Kriterien
        criteria = {
            'min_batches': batch_count >= self.min_batches,
            'material_coverage': material_percentage >= (self.min_material * 100),
            'theoretical_saturation': theoretical_saturation >= self.min_theoretical,
            'category_quality': category_quality >= 0.7,
            'stability': self.stability_counter >= self.min_stability,
            'sufficient_categories': len(current_categories) >= 2
        }
        
        is_saturated = all(criteria.values())
        
        # Bestimme Sättigungsgrund
        if is_saturated:
            saturation_reason = "Alle Sättigungskriterien erfÜllt"
        else:
            missing = [k for k, v in criteria.items() if not v]
            saturation_reason = f"Fehlende Kriterien: {', '.join(missing)}"
        
        return {
            'is_saturated': is_saturated,
            'theoretical_saturation': theoretical_saturation,
            'material_coverage': material_percentage / 100,
            'stability_batches': self.stability_counter,
            'category_quality': category_quality,
            'saturation_reason': saturation_reason,
            'criteria_met': criteria
        }

    def _calculate_theoretical_saturation(self, categories: Dict) -> float:
        """
        Berechnet theoretische Sättigung
        """
        if not categories:
            return 0.0
        
        # Kategorienreife
        maturity_scores = []
        for cat in categories.values():
            score = 0
            # Definition
            if hasattr(cat, 'definition') and len(cat.definition.split()) >= 15:
                score += 0.4
            # Beispiele
            if hasattr(cat, 'examples') and len(cat.examples) >= 1:
                score += 0.3
            # Subkategorien
            if hasattr(cat, 'subcategories') and len(cat.subcategories) >= 1:
                score += 0.3
            maturity_scores.append(score)
        
        avg_maturity = sum(maturity_scores) / len(maturity_scores) if maturity_scores else 0
        
        # Anzahl-Faktor
        optimal_count = 8 if self.analysis_mode == 'inductive' else 6
        count_factor = min(len(categories) / optimal_count, 1.0)
        
        return (avg_maturity * 0.7) + (count_factor * 0.3)

    def _assess_category_quality(self, categories: Dict) -> float:
        """
        Bewertet Kategorienqualität
        """
        if not categories:
            return 0.0
        
        quality_scores = []
        for cat in categories.values():
            score = 0
            if hasattr(cat, 'definition') and len(cat.definition.split()) >= 10:
                score += 0.5
            if hasattr(cat, 'examples') and len(cat.examples) >= 1:
                score += 0.3
            if hasattr(cat, 'subcategories') and len(cat.subcategories) >= 1:
                score += 0.2
            quality_scores.append(score)
        
        return sum(quality_scores) / len(quality_scores)

    def increment_stability_counter(self):
        """ErhÖht StabilitÄtszÄhler"""
        self.stability_counter += 1

    def reset_stability_counter(self):
        """Setzt StabilitÄtszÄhler zurÜck"""
        self.stability_counter = 0

# --- Klasse: DeductiveCategoryBuilder ---
# Aufgabe: Ableiten deduktiver Kategorien basierend auf theoretischem Vorwissen
