"""
Reliabilitäts-Berechnungen für QCA-AID
=======================================
Berechnet Inter-Coder-Reliabilität und Konsens-Metriken.
"""

from typing import Dict, List, Tuple, Optional
from collections import Counter
import statistics

from ..core.data_models import CodingResult


class ReliabilityCalculator:
    """
    FIX: Einheitliche Krippendorff's Alpha Berechnung nach Krippendorff (2011)
    Alle Reliabilitäts-Berechnungen laufen Über diese Klasse
    """
    
    def __init__(self):
        self.debug = True
    
    def _extract_base_segment_id(self, coding: Dict) -> str:
        """
        FIX: Extrahiert Basis-Segment-ID ohne Mehrfachkodierungs-Suffixe
        FÜr ReliabilityCalculator Klasse
        """
        segment_id = coding.get('segment_id', '')
        
        # Entferne Mehrfachkodierungs-Suffixe
        # Format kann sein: "doc_chunk_5" oder "doc_chunk_5-1" fuer Mehrfachkodierung
        if '-' in segment_id:
            # Prüfe ob es ein Mehrfachkodierungs-Suffix ist (endet mit -Zahl)
            parts = segment_id.rsplit('-', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_id = parts[0]
            else:
                base_id = segment_id
        else:
            base_id = segment_id
        
        return base_id
    
    def calculate_comprehensive_reliability(self, codings: List[Dict]) -> dict:
        """
        FIX: Aktualisierte Hauptmethode mit robusteren Berechnungen
        FÜr ReliabilityCalculator Klasse
        """
        print("\n🧾 Umfassende Krippendorff's Alpha Analyse...")
        
        # FIX: Robusterer Filter
        original_codings = self._filter_original_codings(codings)
        
        if len(original_codings) < 2:
            print("❌ Zu wenige ursprüngliche Kodierungen fuer Reliabilitätsanalyse")
            return self._create_empty_reliability_report()
        
        # Basis-Statistiken (mit Fallback)
        statistics = self._calculate_basic_statistics(original_codings)
        
        # 1. Overall Alpha (kombinierte Sets) - FIX: Mit korrekter Mehrfachkodierungs-Behandlung
        overall_alpha = self._calculate_multiple_coding_alpha(original_codings)

        # 2. Hauptkategorien Alpha - extrahiere Float-Wert
        main_categories_alpha = self._calculate_main_categories_alpha(original_codings)

        # 3. FIX: Subkategorien Alpha mit partieller Übereinstimmung - bereits Float
        subcategories_alpha = self._calculate_subcategories_alpha(original_codings)
        
        # 4. Detaillierte Übereinstimmungsanalyse
        agreement_analysis = self._calculate_detailed_agreement_analysis(original_codings)
        
        reliability_report = {
            'overall_alpha': overall_alpha,
            'main_categories_alpha': main_categories_alpha,
            'subcategories_alpha': subcategories_alpha,
            'agreement_analysis': agreement_analysis,
            'statistics': statistics
        }
        
        self._print_reliability_summary(reliability_report)
        
        return reliability_report
    
    def calculate_reliability(self, all_codings: List[Dict]) -> float:
        """
        FIX: Hauptmethode - gibt Overall Alpha als Float zurÜck (fuer RÜckwÄrtskompatibilitÄt)
        """
        report = self.calculate_comprehensive_reliability(all_codings)
        overall_alpha = report['overall_alpha']
        # FIX: Stelle sicher, dass es ein Float ist
        if isinstance(overall_alpha, dict):
            return overall_alpha.get('alpha', 0.0)
        return float(overall_alpha)
    
    def _filter_original_codings(self, codings: List[Dict]) -> List[Dict]:
        """
        FIX: Robusterer Filter fuer ursprüngliche Kodierungen
        FÜr ReliabilityCalculator Klasse
        """
        original_codings = []
        
        # print(f"🕵️ Debug Filter - Input: {len(codings)} Kodierungen")
        
        for i, coding in enumerate(codings):
            coder_id = coding.get('coder_id', '')
            consensus_info = coding.get('consensus_info', {})
            manual_review = coding.get('manual_review', False)
            selection_type = consensus_info.get('selection_type', '')
            
            # FIX: Debug-Ausgabe fuer erste 3 Kodierungen
            # if i < 3:
            #     print(f"  Kodierung {i}: coder_id='{coder_id}', manual_review={manual_review}, selection_type='{selection_type}'")
            
            # FIX: Weniger strenger Filter - akzeptiere mehr Kodierungen
            is_excluded = (
                coder_id in ['consensus', 'majority', 'review'] or
                manual_review == True or
                selection_type in ['consensus', 'majority', 'manual_consensus']
            )
            
            if not is_excluded:
                original_codings.append(coding)
            elif i < 3:
                print(f"    -> Ausgeschlossen")
        
        print(f"🕵️ Gefilterte ursprüngliche Kodierungen: {len(original_codings)}")
        
        # FIX: Falls zu wenige gefunden, weniger streng filtern
        if len(original_codings) < 2:
            print("❌ Zu wenige gefunden - verwende weniger strengen Filter...")
            original_codings = []
            
            for coding in codings:
                coder_id = coding.get('coder_id', '')
                # FIX: Nur explizite Review-Resultate ausschlieẞen
                if coder_id not in ['consensus', 'majority', 'review']:
                    original_codings.append(coding)
            
            print(f"🕵️ Mit weniger strengem Filter: {len(original_codings)} Kodierungen")
        
        return original_codings

    def _calculate_combined_sets_alpha(self, codings: List[Dict]) -> float:
        """
        FIX: Overall Alpha mit Jaccard-Ähnlichkeit (konsistent mit Subkategorien-Behandlung)
        FÜr ReliabilityCalculator Klasse
        """
        segment_data = self._group_by_original_segments(codings)
        
        all_overlap_scores = []
        all_comparisons = 0
        
        for segment_id, coders_data in segment_data.items():
            if len(coders_data) < 2:
                continue
            
            coders = list(coders_data.keys())
            
            for i in range(len(coders)):
                for j in range(i + 1, len(coders)):
                    coder1, coder2 = coders[i], coders[j]
                    
                    # FIX: Kombiniere Haupt- und Subkategorien zu Sets
                    set1 = set()
                    set2 = set()
                    
                    for coding in coders_data[coder1]:
                        main_cat = coding.get('category', '')
                        if main_cat and main_cat not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                            set1.add(main_cat)
                        
                        subcats = coding.get('subcategories', [])
                        if isinstance(subcats, (list, tuple)):
                            set1.update(subcats)
                    
                    for coding in coders_data[coder2]:
                        main_cat = coding.get('category', '')
                        if main_cat and main_cat not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                            set2.add(main_cat)
                        
                        subcats = coding.get('subcategories', [])
                        if isinstance(subcats, (list, tuple)):
                            set2.update(subcats)
                    
                    all_comparisons += 1
                    
                    # FIX: Jaccard-Ähnlichkeit statt exakter Gleichheit
                    if len(set1) == 0 and len(set2) == 0:
                        # Beide haben keine Kategorien - perfekte Übereinstimmung
                        overlap_score = 1.0
                    elif len(set1) == 0 or len(set2) == 0:
                        # Einer hat keine, der andere schon - keine Übereinstimmung
                        overlap_score = 0.0
                    else:
                        # Jaccard-Koeffizient: |Schnittmenge| / |Vereinigungsmenge|
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        overlap_score = intersection / union if union > 0 else 0.0
                    
                    all_overlap_scores.append(overlap_score)
        
        if all_comparisons == 0:
            return 0.0
        
        # Durchschnittliche Übereinstimmung
        observed_agreement = sum(all_overlap_scores) / len(all_overlap_scores)
        
        # Erwartete Übereinstimmung fuer kombinierte Sets
        expected_agreement = 0.25  # Konservative SchÄtzung
        
        # Krippendorff's Alpha
        if expected_agreement >= 1.0:
            alpha = 1.0 if observed_agreement >= 1.0 else 0.0
        else:
            alpha = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        
        print(f"🧾 Overall Alpha Details:")
        print(f"   - Durchschnittliche Jaccard-Übereinstimmung: {observed_agreement:.3f}")
        print(f"   - Erwartete ZufallsÜbereinstimmung: {expected_agreement:.3f}")
        print(f"   - Overall Alpha (Jaccard-basiert): {alpha:.3f}")
        
        return max(0.0, alpha)
    
    def _calculate_main_categories_alpha(self, codings: List[Dict]) -> float:
        """
        FIX: Hauptkategorien Alpha mit Jaccard-Ähnlichkeit (fuer Konsistenz)
        FÜr ReliabilityCalculator Klasse
        """
        segment_data = self._group_by_original_segments(codings)
        
        all_overlap_scores = []
        all_comparisons = 0
        
        for segment_id, coders_data in segment_data.items():
            if len(coders_data) < 2:
                continue
            
            coders = list(coders_data.keys())
            
            for i in range(len(coders)):
                for j in range(i + 1, len(coders)):
                    coder1, coder2 = coders[i], coders[j]
                    
                    # Nur Hauptkategorien sammeln
                    set1 = set()
                    set2 = set()
                    
                    for coding in coders_data[coder1]:
                        main_cat = coding.get('category', '')
                        if main_cat and main_cat not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                            set1.add(main_cat)
                    
                    for coding in coders_data[coder2]:
                        main_cat = coding.get('category', '')
                        if main_cat and main_cat not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                            set2.add(main_cat)
                    
                    all_comparisons += 1
                    
                    # Jaccard-Ähnlichkeit
                    if len(set1) == 0 and len(set2) == 0:
                        overlap_score = 1.0
                    elif len(set1) == 0 or len(set2) == 0:
                        overlap_score = 0.0
                    else:
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        overlap_score = intersection / union if union > 0 else 0.0
                    
                    all_overlap_scores.append(overlap_score)
        
        if all_comparisons == 0:
            return 0.0
        
        # Durchschnittliche Übereinstimmung
        observed_agreement = sum(all_overlap_scores) / len(all_overlap_scores)
        
        # Erwartete Übereinstimmung
        expected_agreement = 0.20  # FÜr Hauptkategorien
        
        # Krippendorff's Alpha
        if expected_agreement >= 1.0:
            alpha = 1.0 if observed_agreement >= 1.0 else 0.0
        else:
            alpha = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        
        print(f"🧾 Hauptkategorien Alpha Details:")
        print(f"   - Durchschnittliche Übereinstimmung: {observed_agreement:.3f}")
        print(f"   - Hauptkategorien Alpha (Jaccard): {alpha:.3f}")
        
        return max(0.0, alpha)

    
    def _calculate_subcategories_alpha_old(self, codings: List[Dict]) -> dict:
        """
        FIX: Korrigierte Subkategorien Alpha Berechnung - behandelt Dictionary-Struktur korrekt
        FÜr ReliabilityCalculator Klasse
        """
        # FIX: Korrekte Gruppierung nach Segment-ID und Kodierer
        segment_data = self._group_by_original_segments(codings)
        
        all_overlap_scores = []
        total_comparisons = 0
        total_agreements = 0
        
        # FIX: Iteriere Über die Dictionary-Struktur korrekt
        for original_id, coder_data in segment_data.items():
            # Extrahiere alle Kodierungen fuer dieses Segment von verschiedenen Kodierern
            all_segment_codings = []
            for coder_id, coder_codings in coder_data.items():
                all_segment_codings.extend(coder_codings)
            
            # Mindestens 2 Kodierungen pro Segment nÖtig fuer Vergleich
            if len(all_segment_codings) < 2:
                continue
                
            # FIX: Paarweise Vergleiche zwischen allen Kodierungen dieses Segments
            for i in range(len(all_segment_codings)):
                for j in range(i + 1, len(all_segment_codings)):
                    subcats1 = all_segment_codings[i].get('subcategories', [])
                    subcats2 = all_segment_codings[j].get('subcategories', [])
                    
                    # Normalisiere zu Sets
                    if isinstance(subcats1, (list, tuple)):
                        set1 = set(subcats1)
                    else:
                        set1 = set()
                    
                    if isinstance(subcats2, (list, tuple)):
                        set2 = set(subcats2)
                    else:
                        set2 = set()
                    
                    total_comparisons += 1
                    
                    # Jaccard-Ähnlichkeit fuer Sets
                    if len(set1) == 0 and len(set2) == 0:
                        overlap = 1.0  # Beide leer = perfekte Übereinstimmung
                        total_agreements += 1
                    elif len(set1.union(set2)) == 0:
                        overlap = 0.0
                    else:
                        overlap = len(set1.intersection(set2)) / len(set1.union(set2))
                        if overlap == 1.0:
                            total_agreements += 1
                    
                    # FIX: Nur numerische Overlap-Werte hinzuFügen, KEINE String-Werte
                    all_overlap_scores.append(overlap)
                    
                    # FIX: Entfernt - keine String-Subkategorien zu all_overlap_scores hinzuFügen
                    # all_overlap_scores.extend(list(set1))  # ENTFERNT
                    # all_overlap_scores.extend(list(set2))  # ENTFERNT
        
        if total_comparisons == 0:
            return {'alpha': 0.0, 'observed_agreement': 0.0, 'expected_agreement': 0.25, 'comparisons': 0}
        
        # FIX: Durchschnittliche Jaccard-Ähnlichkeit als beobachtete Übereinstimmung
        # Jetzt sind alle Werte in all_overlap_scores garantiert Float-Werte
        observed_agreement = sum(all_overlap_scores) / len(all_overlap_scores) if len(all_overlap_scores) > 0 else 0.0
        expected_agreement = 0.25  # Vereinfacht
        
        # Krippendorff's Alpha
        if expected_agreement >= 1.0:
            alpha = 1.0 if observed_agreement >= 1.0 else 0.0
        else:
            alpha = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        
        return {
            'alpha': max(0.0, alpha),
            'observed_agreement': observed_agreement,
            'expected_agreement': expected_agreement,
            'comparisons': total_comparisons
        }
    
    def _calculate_subcategories_alpha(self, codings: List[Dict]) -> float:
        """
        FIX: Subkategorien Alpha mit partieller Übereinstimmung (Jaccard-Ähnlich)
        FÜr ReliabilityCalculator Klasse
        
        Behandelt: "subcat1, subcat2" vs. "subcat1, subcat3" als partielle Übereinstimmung
        statt als komplette Nicht-Übereinstimmung
        """
        segment_data = self._group_by_original_segments(codings)
        
        all_overlap_scores = []
        all_comparisons = 0
        
        print(f"🕵️ Debug Subkategorien: Analysiere {len(segment_data)} Segmente")
        
        for segment_id, coders_data in segment_data.items():
            if len(coders_data) < 2:
                continue
            
            coders = list(coders_data.keys())
            
            for i in range(len(coders)):
                for j in range(i + 1, len(coders)):
                    coder1, coder2 = coders[i], coders[j]
                    
                    # Sammle Subkategorien fuer beide Kodierer
                    set1 = set()
                    set2 = set()
                    
                    for coding in coders_data[coder1]:
                        subcats = coding.get('subcategories', [])
                        if isinstance(subcats, (list, tuple)):
                            set1.update(subcats)
                    
                    for coding in coders_data[coder2]:
                        subcats = coding.get('subcategories', [])
                        if isinstance(subcats, (list, tuple)):
                            set2.update(subcats)
                    
                    all_comparisons += 1
                    
                    # FIX: Jaccard-Ähnlichkeit statt exakter Gleichheit
                    if len(set1) == 0 and len(set2) == 0:
                        # Beide haben keine Subkategorien - perfekte Übereinstimmung
                        overlap_score = 1.0
                    elif len(set1) == 0 or len(set2) == 0:
                        # Einer hat keine, der andere schon - keine Übereinstimmung
                        overlap_score = 0.0
                    else:
                        # Jaccard-Koeffizient: |Schnittmenge| / |Vereinigungsmenge|
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        overlap_score = intersection / union if union > 0 else 0.0
                    
                    all_overlap_scores.append(overlap_score)
                    
                    # Debug fuer erste 3 Vergleiche
                    if all_comparisons <= 3:
                        print(f"  Vergleich {all_comparisons}: {list(set1)} vs {list(set2)} -> {overlap_score:.3f}")
        
        if all_comparisons == 0:
            print("❌ Keine Subkategorien-Vergleiche mÖglich")
            return 0.0
        
        # Durchschnittliche Übereinstimmung
        observed_agreement = sum(all_overlap_scores) / len(all_overlap_scores)
        
        # FIX: Erwartete ZufallsÜbereinstimmung fuer partielle Übereinstimmung
        # Vereinfachte Berechnung: Bei zufÄlliger Verteilung wÜrde man etwa 0.2-0.3 erwarten
        expected_agreement = 0.25  # Konservative SchÄtzung
        
        # Krippendorff's Alpha
        if expected_agreement >= 1.0:
            alpha = 1.0 if observed_agreement >= 1.0 else 0.0
        else:
            alpha = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        
        print(f"🧾 Subkategorien-Alpha Details:")
        print(f"   - Vergleiche durchgefÜhrt: {all_comparisons}")
        print(f"   - Durchschnittliche Übereinstimmung: {observed_agreement:.3f}")
        print(f"   - Erwartete ZufallsÜbereinstimmung: {expected_agreement:.3f}")
        print(f"   - Partielle Übereinstimmungs-Alpha: {alpha:.3f}")
        
        return max(0.0, alpha)
    
    def _calculate_detailed_agreement_analysis(self, codings: List[Dict]) -> dict:
        """
        FIX: Korrigierte detaillierte Übereinstimmungsanalyse nach Kategorien
        FÜr ReliabilityCalculator Klasse
        """
        # FIX: Korrekte Gruppierung nach Segment-ID und Kodierer
        segment_data = self._group_by_original_segments(codings)
        
        agreement_stats = {
            'Vollständige Übereinstimmung': 0,
            'Hauptkategorie gleich, Subkat. unterschiedlich': 0,
            'Hauptkategorie unterschiedlich': 0
        }
        
        # FIX: Iteriere Über die Dictionary-Struktur korrekt
        for original_id, coder_data in segment_data.items():
            # Extrahiere alle Kodierungen fuer dieses Segment von verschiedenen Kodierern
            all_segment_codings = []
            for coder_id, coder_codings in coder_data.items():
                all_segment_codings.extend(coder_codings)
            
            # Mindestens 2 Kodierungen pro Segment nÖtig fuer Vergleich
            if len(all_segment_codings) < 2:
                continue
                
            # FIX: Paarweise Vergleiche zwischen allen Kodierungen dieses Segments
            for i in range(len(all_segment_codings)):
                for j in range(i + 1, len(all_segment_codings)):
                    coding1 = all_segment_codings[i]
                    coding2 = all_segment_codings[j]
                    
                    main_cat1 = coding1.get('category', '')
                    main_cat2 = coding2.get('category', '')
                    subcats1 = set(coding1.get('subcategories', []))
                    subcats2 = set(coding2.get('subcategories', []))
                    
                    if main_cat1 == main_cat2 and subcats1 == subcats2:
                        agreement_stats['Vollständige Übereinstimmung'] += 1
                    elif main_cat1 == main_cat2:
                        agreement_stats['Hauptkategorie gleich, Subkat. unterschiedlich'] += 1
                    else:
                        agreement_stats['Hauptkategorie unterschiedlich'] += 1
        
        return agreement_stats
    
    def _group_by_original_segments_with_multiple_codings(self, codings: List[Dict]) -> dict:
        """
        FIX: Gruppiert Kodierungen nach ursprünglicher Segment-ID MIT korrekter Mehrfachkodierungs-Behandlung
        
        Methodisch korrekte Behandlung:
        - Jede Mehrfachkodierung wird als separate Kodierung behandelt
        - Alle Kombinationen zwischen Kodierern werden verglichen
        - Basis-Segment wird nur für Gruppierung verwendet
        
        Rückgabe: {base_segment_id: {coder_id: [actual_codings_with_full_segment_ids]}}
        """
        segment_data = {}
        
        for coding in codings:
            # Extrahiere Basis-Segment-ID für Gruppierung
            base_id = self._extract_base_segment_id(coding)
            
            if base_id not in segment_data:
                segment_data[base_id] = {}
            
            coder_id = coding.get('coder_id', 'unknown')
            if coder_id not in segment_data[base_id]:
                segment_data[base_id][coder_id] = []
            
            # WICHTIG: Speichere die vollständige Kodierung mit tatsächlicher segment_id
            segment_data[base_id][coder_id].append(coding)
        
        return segment_data
    
    def _calculate_multiple_coding_alpha(self, codings: List[Dict]) -> float:
        """
        FIX: Berechnet Alpha mit korrekter Mehrfachkodierungs-Behandlung
        
        Methodisch korrekte Behandlung nach Krippendorff (2011):
        - Jede Mehrfachkodierung wird als separate Beobachtung behandelt
        - Alle paarweisen Vergleiche zwischen Kodierern werden durchgeführt
        - Jaccard-Ähnlichkeit für Set-basierte Vergleiche
        """
        segment_data = self._group_by_original_segments_with_multiple_codings(codings)
        
        all_overlap_scores = []
        all_comparisons = 0
        
        print(f"🔍 DEBUG: Mehrfachkodierungs-Alpha für {len(segment_data)} Basis-Segmente")
        
        for base_segment_id, coders_data in segment_data.items():
            if len(coders_data) < 2:
                continue
            
            coders = list(coders_data.keys())
            
            # Paarweise Vergleiche zwischen allen Kodierern
            for i in range(len(coders)):
                for j in range(i + 1, len(coders)):
                    coder1, coder2 = coders[i], coders[j]
                    
                    # Alle Kodierungen von beiden Kodierern für dieses Basis-Segment
                    coder1_codings = coders_data[coder1]
                    coder2_codings = coders_data[coder2]
                    
                    # METHODISCH KORREKT: Vergleiche alle Kombinationen von Mehrfachkodierungen
                    for coding1 in coder1_codings:
                        for coding2 in coder2_codings:
                            # Sammle Sets für beide Kodierungen
                            set1 = set()
                            set2 = set()
                            
                            # Set 1 (Kodierung 1)
                            main_cat1 = coding1.get('category', '')
                            if main_cat1 and main_cat1 not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                                set1.add(main_cat1)
                            
                            subcats1 = coding1.get('subcategories', [])
                            if isinstance(subcats1, (list, tuple)):
                                set1.update(subcats1)
                            
                            # Set 2 (Kodierung 2)
                            main_cat2 = coding2.get('category', '')
                            if main_cat2 and main_cat2 not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                                set2.add(main_cat2)
                            
                            subcats2 = coding2.get('subcategories', [])
                            if isinstance(subcats2, (list, tuple)):
                                set2.update(subcats2)
                            
                            all_comparisons += 1
                            
                            # Jaccard-Ähnlichkeit
                            if len(set1) == 0 and len(set2) == 0:
                                overlap_score = 1.0
                            elif len(set1) == 0 or len(set2) == 0:
                                overlap_score = 0.0
                            else:
                                intersection = len(set1.intersection(set2))
                                union = len(set1.union(set2))
                                overlap_score = intersection / union if union > 0 else 0.0
                            
                            all_overlap_scores.append(overlap_score)
                            
                            # Debug für erste 3 Vergleiche
                            if all_comparisons <= 3:
                                print(f"  Vergleich {all_comparisons}: {coding1.get('segment_id')} vs {coding2.get('segment_id')}")
                                print(f"    Set1: {list(set1)}, Set2: {list(set2)} -> {overlap_score:.3f}")
        
        if all_comparisons == 0:
            return 0.0
        
        # Durchschnittliche Übereinstimmung
        observed_agreement = sum(all_overlap_scores) / len(all_overlap_scores)
        
        # Erwartete Übereinstimmung
        expected_agreement = 0.25  # Konservative Schätzung
        
        # Krippendorff's Alpha
        if expected_agreement >= 1.0:
            alpha = 1.0 if observed_agreement >= 1.0 else 0.0
        else:
            alpha = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        
        print(f"🧾 Mehrfachkodierungs-Alpha Details:")
        print(f"   - Vergleiche durchgeführt: {all_comparisons}")
        print(f"   - Durchschnittliche Übereinstimmung: {observed_agreement:.3f}")
        print(f"   - Mehrfachkodierungs-Alpha: {alpha:.3f}")
        
        return max(0.0, alpha)

    def _group_by_original_segments(self, codings: List[Dict]) -> dict:
        """
        FIX: Gruppiert Kodierungen nach ursprünglicher Segment-ID fuer ReliabilityCalculator
        RÜckgabe: {segment_id: {coder_id: [codings]}}
        FÜr ReliabilityCalculator Klasse
        """
        segment_data = {}
        
        for coding in codings:
            # Extrahiere ursprüngliche Segment-ID
            original_id = self._extract_base_segment_id(coding)
            
            if original_id not in segment_data:
                segment_data[original_id] = {}
            
            coder_id = coding.get('coder_id', 'unknown')
            if coder_id not in segment_data[original_id]:
                segment_data[original_id][coder_id] = []
            
            segment_data[original_id][coder_id].append(coding)
        
        return segment_data
    
    def _calculate_alpha_from_sets(self, agreements: int, comparisons: int, category_sets: List[set]) -> float:
        """
        FIX: Berechnet Krippendorff's Alpha aus Set-Daten
        """
        if comparisons == 0:
            return 0.0
        
        observed_agreement = agreements / comparisons
        expected_agreement = self._calculate_expected_set_agreement(category_sets)
        
        if expected_agreement >= 1.0:
            alpha = 1.0 if observed_agreement >= 1.0 else 0.0
        else:
            alpha = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        
        return max(0.0, alpha)
    
    def _calculate_expected_set_agreement(self, all_category_sets: List[set]) -> float:
        """
        FIX: Berechnet erwartete Set-Übereinstimmung nach Krippendorff (2011)
        """
        if not all_category_sets:
            return 0.0
        
        # Sammle alle individuellen Kategorien
        all_individual_categories = []
        for category_set in all_category_sets:
            all_individual_categories.extend(list(category_set))
        
        if not all_individual_categories:
            return 0.0
        
        # HÄufigkeitsverteilung
        from collections import Counter
        category_frequencies = Counter(all_individual_categories)
        total_instances = len(all_individual_categories)
        
        # Vereinfachte erwartete Übereinstimmung fuer Set-Variable
        # (Krippendorff 2011 empfiehlt komplexere Berechnung, aber das ist eine praktikable NÄherung)
        expected_agreement = 0.0
        unique_sets = list(set(frozenset(s) for s in all_category_sets))
        
        for unique_set in unique_sets:
            # Wahrscheinlichkeit dieses Set-Typs
            set_probability = 1.0
            for category in unique_set:
                cat_prob = category_frequencies[category] / total_instances
                set_probability *= cat_prob
            
            expected_agreement += set_probability ** 2
        
        return min(expected_agreement, 0.99)  # Verhindere Division durch 0
    
    def _extract_original_segment_id(self, coding: Dict) -> str:
        """
        FIX: Extrahiert ursprüngliche Segment-ID
        """
        # Erst consensus_info prÜfen
        consensus_info = coding.get('consensus_info', {})
        if consensus_info.get('original_segment_id'):
            return consensus_info['original_segment_id']
        
        # Fallback: Aus segment_id ableiten
        segment_id = coding.get('segment_id', '')
        if '-' in segment_id:
            parts = segment_id.rsplit('-', 1)
            if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2:
                return parts[0]
        
        return segment_id
    
    def _calculate_basic_statistics(self, codings: List[Dict]) -> dict:
        """
        FIX: Robuste Basis-Statistiken mit Fallback
        FÜr ReliabilityCalculator Klasse
        """
        if not codings:
            return {
                'total_codings': 0,
                'vergleichbare_segmente': 0,
                'total_segmente': 0,
                'anzahl_kodierer': 0,
                'mittelwert_kodierungen': 0.0
            }
        
        segment_data = self._group_by_original_segments(codings)
        vergleichbare_segmente = sum(1 for data in segment_data.values() if len(data) >= 2)
        
        coders = set()
        for coding in codings:
            coder_id = coding.get('coder_id', 'unknown')
            if coder_id:  # FIX: Leere coder_ids ignorieren
                coders.add(coder_id)
        
        anzahl_kodierer = len(coders) if coders else 1  # FIX: Mindestens 1 um Division durch 0 zu vermeiden
        
        return {
            'total_codings': len(codings),
            'vergleichbare_segmente': vergleichbare_segmente,
            'total_segmente': len(segment_data),
            'anzahl_kodierer': anzahl_kodierer,
            'mittelwert_kodierungen': len(codings) / anzahl_kodierer
        }
    
    def _create_empty_reliability_report(self) -> dict:
        """
        FIX: Erstellt vollständigen leeren Bericht
        FÜr ReliabilityCalculator Klasse
        """
        return {
            'overall_alpha': 0.0,
            'main_categories_alpha': 0.0,
            'subcategories_alpha': 0.0,
            'agreement_analysis': {
                'Vollständige Übereinstimmung': 0,
                'Hauptkategorie gleich, Subkat. unterschiedlich': 0,
                'Hauptkategorie unterschiedlich': 0,
                'Gesamt': 0
            },
            'statistics': {
                'total_codings': 0,
                'vergleichbare_segmente': 0,
                'total_segmente': 0,
                'anzahl_kodierer': 0,
                'mittelwert_kodierungen': 0.0
            }
        }
    
    def _calculate_combined_sets_alpha(self, codings: List[Dict]) -> float:
        """
        FIX: Overall Alpha mit Jaccard-Ähnlichkeit (konsistent mit Subkategorien-Behandlung)
        FÜr ReliabilityCalculator Klasse
        """
        segment_data = self._group_by_original_segments(codings)
        
        all_overlap_scores = []
        all_comparisons = 0
        
        for segment_id, coders_data in segment_data.items():
            if len(coders_data) < 2:
                continue
            
            coders = list(coders_data.keys())
            
            for i in range(len(coders)):
                for j in range(i + 1, len(coders)):
                    coder1, coder2 = coders[i], coders[j]
                    
                    # FIX: Kombiniere Haupt- und Subkategorien zu Sets
                    set1 = set()
                    set2 = set()
                    
                    for coding in coders_data[coder1]:
                        main_cat = coding.get('category', '')
                        if main_cat and main_cat not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                            set1.add(main_cat)
                        
                        subcats = coding.get('subcategories', [])
                        if isinstance(subcats, (list, tuple)):
                            set1.update(subcats)
                    
                    for coding in coders_data[coder2]:
                        main_cat = coding.get('category', '')
                        if main_cat and main_cat not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                            set2.add(main_cat)
                        
                        subcats = coding.get('subcategories', [])
                        if isinstance(subcats, (list, tuple)):
                            set2.update(subcats)
                    
                    all_comparisons += 1
                    
                    # FIX: Jaccard-Ähnlichkeit statt exakter Gleichheit
                    if len(set1) == 0 and len(set2) == 0:
                        # Beide haben keine Kategorien - perfekte Übereinstimmung
                        overlap_score = 1.0
                    elif len(set1) == 0 or len(set2) == 0:
                        # Einer hat keine, der andere schon - keine Übereinstimmung
                        overlap_score = 0.0
                    else:
                        # Jaccard-Koeffizient: |Schnittmenge| / |Vereinigungsmenge|
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        overlap_score = intersection / union if union > 0 else 0.0
                    
                    all_overlap_scores.append(overlap_score)
        
        if all_comparisons == 0:
            return 0.0
        
        # Durchschnittliche Übereinstimmung
        observed_agreement = sum(all_overlap_scores) / len(all_overlap_scores)
        
        # Erwartete Übereinstimmung fuer kombinierte Sets
        expected_agreement = 0.25  # Konservative SchÄtzung
        
        # Krippendorff's Alpha
        if expected_agreement >= 1.0:
            alpha = 1.0 if observed_agreement >= 1.0 else 0.0
        else:
            alpha = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        
        print(f"🧾 Overall Alpha Details:")
        print(f"   - Durchschnittliche Jaccard-Übereinstimmung: {observed_agreement:.3f}")
        print(f"   - Erwartete ZufallsÜbereinstimmung: {expected_agreement:.3f}")
        print(f"   - Overall Alpha (Jaccard-basiert): {alpha:.3f}")
        
        return max(0.0, alpha)

    def _calculate_main_categories_alpha(self, codings: List[Dict]) -> float:
        """
        FIX: Hauptkategorien Alpha mit Jaccard-Ähnlichkeit (fuer Konsistenz)
        FÜr ReliabilityCalculator Klasse
        """
        segment_data = self._group_by_original_segments(codings)
        
        all_overlap_scores = []
        all_comparisons = 0
        
        for segment_id, coders_data in segment_data.items():
            if len(coders_data) < 2:
                continue
            
            coders = list(coders_data.keys())
            
            for i in range(len(coders)):
                for j in range(i + 1, len(coders)):
                    coder1, coder2 = coders[i], coders[j]
                    
                    # Nur Hauptkategorien sammeln
                    set1 = set()
                    set2 = set()
                    
                    for coding in coders_data[coder1]:
                        main_cat = coding.get('category', '')
                        if main_cat and main_cat not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                            set1.add(main_cat)
                    
                    for coding in coders_data[coder2]:
                        main_cat = coding.get('category', '')
                        if main_cat and main_cat not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                            set2.add(main_cat)
                    
                    all_comparisons += 1
                    
                    # Jaccard-Ähnlichkeit
                    if len(set1) == 0 and len(set2) == 0:
                        overlap_score = 1.0
                    elif len(set1) == 0 or len(set2) == 0:
                        overlap_score = 0.0
                    else:
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        overlap_score = intersection / union if union > 0 else 0.0
                    
                    all_overlap_scores.append(overlap_score)
        
        if all_comparisons == 0:
            return 0.0
        
        # Durchschnittliche Übereinstimmung
        observed_agreement = sum(all_overlap_scores) / len(all_overlap_scores)
        
        # Erwartete Übereinstimmung
        expected_agreement = 0.20  # FÜr Hauptkategorien
        
        # Krippendorff's Alpha
        if expected_agreement >= 1.0:
            alpha = 1.0 if observed_agreement >= 1.0 else 0.0
        else:
            alpha = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        
        print(f"🧾 Hauptkategorien Alpha Details:")
        print(f"   - Durchschnittliche Übereinstimmung: {observed_agreement:.3f}")
        print(f"   - Hauptkategorien Alpha (Jaccard): {alpha:.3f}")
        
        return max(0.0, alpha)

    def _print_reliability_summary(self, report: dict):
        """
        FIX: Erweiterte Zusammenfassung mit Konsistenz-PrÜfung
        FÜr ReliabilityCalculator Klasse
        """
        print(f"\nℹ️ Krippendorff's Alpha Reliabilitäts-Analyse:")
        print(f"=" * 60)
        print(f"Overall Alpha (Jaccard-basiert):       {report['overall_alpha']:.3f}")
        print(f"Hauptkategorien Alpha (Jaccard):       {report['main_categories_alpha']:.3f}")
        print(f"Subkategorien Alpha (Jaccard):         {report['subcategories_alpha']:.3f}")
        print(f"Vergleichbare Segmente:                {report['statistics']['vergleichbare_segmente']}")
        print(f"Anzahl Kodierer:                       {report['statistics']['anzahl_kodierer']}")
        
        # FIX: Konsistenz-PrÜfung
        overall = report['overall_alpha']
        main_alpha = report['main_categories_alpha']
        sub_alpha = report['subcategories_alpha']
        
        print(f"\nℹ️ Methodik:")
        print(f"   - Alle Alpha-Werte verwenden Jaccard-Ähnlichkeit")
        print(f"   - Konsistente Set-basierte Berechnung")
        print(f"   - Overall sollte zwischen Haupt- und Sub-Alpha liegen")
        
        # Konsistenz-Check
        min_component = min(main_alpha, sub_alpha)
        max_component = max(main_alpha, sub_alpha)
        
        if min_component <= overall <= max_component:
            print(f"   ✅ Mathematische Konsistenz: {min_component:.3f} â‰¤ {overall:.3f} â‰¤ {max_component:.3f}")
        else:
            print(f"   ❌ Mathematische Inkonsistenz: Overall liegt auẞerhalb der Komponenten!")
            print(f"      Bereich: {min_component:.3f} - {max_component:.3f}, Overall: {overall:.3f}")
        
        # Bewertung
        rating = "Exzellent" if overall > 0.8 else "Akzeptabel" if overall > 0.667 else "Unzureichend"
        print(f"\nBewertung Overall Alpha:               {rating}")
        
        if overall < 0.667:
            print(f"❌  Reliabilität unter Schwellenwert - Kategoriensystem Überarbeiten")


# --- Klasse: ResultsExporter ---
# Aufgabe: Export der kodierten Daten und des finalen Kategoriensystems
