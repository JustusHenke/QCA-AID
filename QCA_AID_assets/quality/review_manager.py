"""
Review-Manager für QCA-AID
===========================
Verwaltet Review-Prozesse und Konsens-Findung bei Multi-Coder-Setups.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from ..core.data_models import CodingResult


class ReviewManager:
    """
    KORRIGIERT: Zentrale Verwaltung aller Review-Modi mit kategorie-zentrierter Mehrfachkodierungs-Behandlung
    """
    
    def __init__(self, output_dir: str, analysis_mode: str = None, timestamp: str = None):
        self.output_dir = output_dir
        self.analysis_mode = analysis_mode or 'deductive'
        self.timestamp = timestamp
        self.review_decisions = []  # Store all review decisions for logging
    
    def process_coding_review(self, all_codings: List[Dict], export_mode: str) -> List[Dict]:
        """
        Hauptfunktion fuer alle Review-Modi mit korrekter Mehrfachkodierungs-Behandlung
        
        Args:
            all_codings: Alle ursprünglichen Kodierungen
            export_mode: 'consensus', 'majority', 'manual', etc.
            
        Returns:
            Liste der finalen, reviewten Kodierungen
        """
        print(f"\n=== REVIEW-PROZESS ({export_mode.upper()}) ===")
        
        # 1. FRÜHE SEGMENTIERUNG: Erkenne Mehrfachkodierungen und erstelle kategorie-spezifische Segmente
        category_segments = self._create_category_specific_segments(all_codings)
        
        # 2. REVIEW-PROZESS: Wende gewÄhlten Modus auf kategorie-spezifische Segmente an
        if export_mode == 'manual':
            reviewed_codings = self._manual_review_process(category_segments)
        elif export_mode == 'majority':
            reviewed_codings = self._majority_review_process(category_segments)
        else:  # consensus (default)
            reviewed_codings = self._consensus_review_process(category_segments)
        
        print(f"Review abgeschlossen: {len(reviewed_codings)} finale Kodierungen")
        
        # Save review decisions to file
        self._save_review_decisions(export_mode)
        
        return reviewed_codings
    
    def _create_category_specific_segments(self, all_codings: List[Dict]) -> List[Dict]:
        """
        KERNFUNKTION: Erstelle kategorie-spezifische Segmente fuer korrekte Mehrfachkodierungs-Behandlung
        
        Verwandelt:
        - TEDFWI-1: [Akteure, Kontextfaktoren, Legitimation]
        
        In:
        - TEDFWI-1-01: [Akteure] (alle Akteure-Kodierungen fuer Segment TEDFWI-1)
        - TEDFWI-1-02: [Kontextfaktoren] (alle Kontextfaktoren-Kodierungen fuer Segment TEDFWI-1)  
        - TEDFWI-1-03: [Legitimation] (alle Legitimation-Kodierungen fuer Segment TEDFWI-1)
        """
        print("ℹ️ Erstelle kategorie-spezifische Segmente...")
        
        # Gruppiere nach ursprünglicher Segment-ID
        original_segments = defaultdict(list)
        for coding in all_codings:
            segment_id = coding.get('segment_id', '')
            if segment_id:
                # Extrahiere ursprüngliche Segment-ID (falls bereits erweitert)
                original_id = self._extract_original_segment_id(segment_id)
                original_segments[original_id].append(coding)
        
        # Erstelle kategorie-spezifische Segmente
        category_segments = []
        
        for original_id, codings in original_segments.items():
            # Identifiziere alle Hauptkategorien fuer dieses Segment
            categories = set()
            for coding in codings:
                category = coding.get('category', '')
                if category and category not in ['Nicht kodiert', 'Kein Kodierkonsens']:
                    categories.add(category)
            
            if len(categories) <= 1:
                # Einfachkodierung oder keine gültigen Kategorien
                if categories:
                    category_segments.append({
                        'segment_id': original_id,
                        'original_segment_id': original_id,
                        'target_category': list(categories)[0],
                        'codings': codings,
                        'is_multiple_coding': False,
                        'instance_info': {'instance_number': 1, 'total_instances': 1}
                    })
                else:
                    # Keine gültigen Kategorien - behalte ursprüngliches Segment
                    category_segments.append({
                        'segment_id': original_id,
                        'original_segment_id': original_id,
                        'target_category': None,
                        'codings': codings,
                        'is_multiple_coding': False,
                        'instance_info': {'instance_number': 1, 'total_instances': 1}
                    })
            else:
                # MEHRFACHKODIERUNG: Erstelle separate Segmente pro Kategorie
                sorted_categories = sorted(categories)  # Konsistente Sortierung fuer ID-Zuordnung
                total_instances = len(sorted_categories)
                
                for i, category in enumerate(sorted_categories, 1):
                    # FIX: Neue Segment-ID mit korrektem Mehrfachkodierungs-Suffix (einstellig, nicht zweistellig)
                    new_segment_id = f"{original_id}-{i}"  # Korrekt: -1, -2, -3 (nicht -01, -02, -03)
                    
                    # Filtere Kodierungen fuer diese spezifische Kategorie
                    category_codings = [
                        coding for coding in codings 
                        if coding.get('category', '') == category
                    ]
                    
                    category_segments.append({
                        'segment_id': new_segment_id,
                        'original_segment_id': original_id,
                        'target_category': category,
                        'codings': category_codings,
                        'is_multiple_coding': True,
                        'instance_info': {
                            'instance_number': i,
                            'total_instances': total_instances,
                            'category_rank': i,
                            'all_categories': sorted_categories
                        }
                    })
                
                print(f"  🧾 Mehrfachkodierung {original_id}: {len(sorted_categories)} Kategorien -> {len(sorted_categories)} Segmente")
        
        print(f"[OK] {len(category_segments)} kategorie-spezifische Segmente erstellt")
        return category_segments
    
    def _extract_original_segment_id(self, segment_id: str) -> str:
        """
        Extrahiert die ursprüngliche Segment-ID (entfernt Mehrfachkodierungs-Suffixe)
        
        Beispiele:
        - "Lei_Bera_02.pdf_chunk_3-1" -> "Lei_Bera_02.pdf_chunk_3" (interne ID)
        - "Lei_Bera_02.pdf_chunk_3" -> "Lei_Bera_02.pdf_chunk_3" (interne ID)
        """
        from ..core.segment_id_manager import SegmentIDManager
        
        # FIX: Prüfe ob es eine Display-ID ist - das sollte NICHT passieren!
        if segment_id and not '_chunk_' in segment_id and '-' in segment_id:
            print(f"   ❌ CRITICAL ERROR: Display-ID als segment_id im ReviewManager: {segment_id}")
            print(f"   ❌ Das ist ein Bug - Display-IDs sollen niemals intern verwendet werden!")
            print(f"   ❌ Quelle des Problems muss gefunden und behoben werden!")
            # Gebe die Display-ID zurück um den Fehler sichtbar zu machen
            return segment_id
        
        # Verwende SegmentIDManager für konsistente Extraktion
        return SegmentIDManager.extract_base_segment_id(segment_id)
    
    def _consensus_review_process(self, category_segments: List[Dict]) -> List[Dict]:
        """
        Consensus-Review fuer kategorie-spezifische Segmente
        """
        print("🕵️ Führe Consensus-Review durch...")
        reviewed_codings = []
        
        for segment in category_segments:
            codings = segment['codings']
            if len(codings) == 1:
                # Nur eine Kodierung - Übernehme direkt
                final_coding = codings[0].copy()
                
                # FIX: Preserve original internal segment_id, don't overwrite with category-specific ID
                # The segment['segment_id'] might be a category-specific ID like "TEDFWI-1-01"
                # but we need to preserve the original internal format for text lookup
                original_segment_id = segment.get('original_segment_id', segment['segment_id'])
                
                # For multiple coding, create proper multiple coding segment ID
                if segment.get('is_multiple_coding', False):
                    instance_number = segment.get('instance_info', {}).get('instance_number', 1)
                    final_coding['segment_id'] = f"{original_segment_id}-{instance_number}"
                else:
                    final_coding['segment_id'] = original_segment_id
                
                # Store metadata for export
                final_coding['original_segment_id'] = original_segment_id
                final_coding['is_multiple_coding'] = segment.get('is_multiple_coding', False)
                final_coding['multiple_coding_instance'] = segment.get('instance_info', {}).get('instance_number', 1)
                final_coding['total_coding_instances'] = segment.get('instance_info', {}).get('total_instances', 1)
                
                # Log the decision (no actual review needed)
                decision_reason = "Nur eine Kodierung verfügbar - keine Consensus-Entscheidung erforderlich"
                self._log_review_decision(segment, final_coding, codings, "single_coder", decision_reason)
                
                reviewed_codings.append(final_coding)
            else:
                # Mehrere Kodierungen - fÜhre Consensus durch
                consensus_coding = self._get_consensus_for_category_segment(segment)
                if consensus_coding:
                    reviewed_codings.append(consensus_coding)
        
        return reviewed_codings
    
    def _majority_review_process(self, category_segments: List[Dict]) -> List[Dict]:
        """
        Majority-Review fuer kategorie-spezifische Segmente
        """
        print("🕵️ Führe Majority-Review durch...")
        reviewed_codings = []
        
        for segment in category_segments:
            codings = segment['codings']
            if len(codings) == 1:
                # Nur eine Kodierung - Übernehme direkt
                final_coding = codings[0].copy()
                
                # FIX: Preserve original internal segment_id, don't overwrite with category-specific ID
                original_segment_id = segment.get('original_segment_id', segment['segment_id'])
                
                # For multiple coding, create proper multiple coding segment ID
                if segment.get('is_multiple_coding', False):
                    instance_number = segment.get('instance_info', {}).get('instance_number', 1)
                    final_coding['segment_id'] = f"{original_segment_id}-{instance_number}"
                else:
                    final_coding['segment_id'] = original_segment_id
                
                # Store metadata for export
                final_coding['original_segment_id'] = original_segment_id
                final_coding['is_multiple_coding'] = segment.get('is_multiple_coding', False)
                final_coding['multiple_coding_instance'] = segment.get('instance_info', {}).get('instance_number', 1)
                final_coding['total_coding_instances'] = segment.get('instance_info', {}).get('total_instances', 1)
                
                # Log the decision (no actual review needed)
                decision_reason = "Nur eine Kodierung verfügbar - keine Majority-Entscheidung erforderlich"
                self._log_review_decision(segment, final_coding, codings, "single_coder", decision_reason)
                
                reviewed_codings.append(final_coding)
            else:
                # Mehrere Kodierungen - fÜhre Majority durch
                majority_coding = self._get_majority_for_category_segment(segment)
                if majority_coding:
                    reviewed_codings.append(majority_coding)
        
        return reviewed_codings
    
    def _manual_review_process(self, category_segments: List[Dict]) -> List[Dict]:
        """
        FIX: Korrigierter manueller Review-Prozess ohne Event Loop Konflikt
        FÜr ReviewManager Klasse - verwendet bestehende Methoden und behÄlt Sortierreihenfolge bei
        """
        print("🕵️ Führe manuelles Review durch...")
        
        # Identifiziere Segmente, die Review benÖtigen
        segments_needing_review = []
        for segment in category_segments:
            if len(segment['codings']) > 1:
                # FIX: Verwende bestehende Methode zur Unstimmigkeits-PrÜfung
                if self._has_category_disagreement(segment):
                    segments_needing_review.append(segment)
        
        if not segments_needing_review:
            print("[OK] Kein manueller Review erforderlich - alle Segmente haben eindeutige Kodierungen")
            return self._consensus_review_process(category_segments)
        
        print(f"🎯 {len(segments_needing_review)} kategorie-spezifische Segmente benÖtigen Review:")
        for segment in segments_needing_review:
            category = segment.get('category', 'Unbekannt')
            segment_id = segment['segment_id']
            print(f"   {segment_id}: {category} (Teil {segment_id.split('-')[-1]} von {segment_id.rsplit('-', 1)[0]})")
        
        # FIX: Verwende asyncio.create_task() statt loop.run_until_complete()
        print("🕵️ Starte GUI-basiertes manuelles Review...")

        try:
            # Konvertiere segments_needing_review zu dem Format, das ManualReviewComponent erwartet
            segment_codings = {}
            for segment in segments_needing_review:
                segment_id = segment['segment_id']
                segment_codings[segment_id] = segment['codings']
            
            # Importiere und verwende ManualReviewComponent fuer echtes GUI
            from ..utils.export.review import ManualReviewGUI, ManualReviewComponent
            import asyncio
            
            # Erstelle ManualReviewComponent
            manual_review_component = ManualReviewComponent(self.output_dir)
            
            # FIX: Verwende asyncio.create_task() fuer bereits laufende Event Loop
            import concurrent.futures
            
            # Führe GUI-Review in separatem Thread aus
            def run_gui_review():
                try:
                    # Erstelle neue Event Loop fuer diesen Thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    
                    try:
                        # FIX: Verwende neue Methode, die Unstimmigkeits-PrÜfung Überspringt
                        review_decisions = new_loop.run_until_complete(
                            manual_review_component.review_discrepancies_direct(segment_codings, skip_discrepancy_check=True)
                        )
                        return review_decisions
                    finally:
                        new_loop.close()
                        
                except Exception as e:
                    print(f"⚠️ Fehler im GUI-Thread: {e}")
                    return None
            
            # Führe GUI-Review in separatem Thread aus
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_gui_review)
                review_decisions = future.result(timeout=300)  # 5 Minuten Timeout
            
            if review_decisions is None:
                raise Exception("GUI-Review fehlgeschlagen")
                
            print(f"[OK] GUI-Review abgeschlossen: {len(review_decisions)} Entscheidungen getroffen")
            
        except Exception as e:
            print(f"⚠️ Fehler beim GUI-Review: {e}")
            print("🔀 Verwende automatischen Consensus als Fallback")
            import traceback
            traceback.print_exc()
            
            # FIX: Fallback ohne problematische Coroutine-Aufrufe
            return self._consensus_review_process(category_segments)
        
        # FIX: Kombiniere Review-Entscheidungen mit Segmenten ohne Review IN KORREKTER REIHENFOLGE
        reviewed_codings = []
        review_decisions_dict = {decision['segment_id']: decision for decision in review_decisions}
        
        # Durchlaufe category_segments in ursprünglicher Reihenfolge
        for segment in category_segments:
            segment_id = segment['segment_id']
            
            if segment_id in review_decisions_dict:
                # Verwende manuelle Review-Entscheidung
                reviewed_codings.append(review_decisions_dict[segment_id])
            else:
                # Segment ohne Review - verwende bestehende Logik
                if len(segment['codings']) == 1:
                    final_coding = segment['codings'][0].copy()
                    
                    # FIX: Preserve original internal segment_id format
                    original_segment_id = segment.get('original_segment_id', segment['segment_id'])
                    
                    # For multiple coding, create proper multiple coding segment ID
                    if segment.get('is_multiple_coding', False):
                        instance_number = segment.get('instance_info', {}).get('instance_number', 1)
                        final_coding['segment_id'] = f"{original_segment_id}-{instance_number}"
                    else:
                        final_coding['segment_id'] = original_segment_id
                    
                    # Store metadata for export
                    final_coding['original_segment_id'] = original_segment_id
                    final_coding['is_multiple_coding'] = segment.get('is_multiple_coding', False)
                    final_coding['multiple_coding_instance'] = segment.get('instance_info', {}).get('instance_number', 1)
                    final_coding['total_coding_instances'] = segment.get('instance_info', {}).get('total_instances', 1)
                    
                    # Log the decision (no manual review needed)
                    decision_reason = "Nur eine Kodierung verfügbar - kein manueller Review erforderlich"
                    self._log_review_decision(segment, final_coding, segment['codings'], "single_coder", decision_reason)
                    
                    reviewed_codings.append(final_coding)
                else:
                    # FIX: Verwende bestehende Consensus-Methode fuer Fallback
                    consensus_coding = self._get_consensus_for_category_segment(segment)
                    if consensus_coding:
                        reviewed_codings.append(consensus_coding)
        
        return reviewed_codings
    
    def _has_category_disagreement(self, segment: Dict) -> bool:
        """
        PrÜft, ob es echte Unstimmigkeiten innerhalb einer Kategorie gibt
        
        Da alle Kodierungen bereits auf eine Kategorie gefiltert sind,
        prÜfen wir hauptsÄchlich Subkategorien-Unstimmigkeiten
        """
        codings = segment['codings']
        
        # Vergleiche Subkategorien
        subcategory_sets = []
        for coding in codings:
            subcats = set(coding.get('subcategories', []))
            subcategory_sets.append(subcats)
        
        # Prüfe auf Unterschiede in Subkategorien
        if len(set(frozenset(s) for s in subcategory_sets)) > 1:
            return True
        
        return False
    
    def _get_consensus_for_category_segment(self, segment: Dict) -> Optional[Dict]:
        """
        Ermittelt Consensus fuer ein kategorie-spezifisches Segment
        """
        codings = segment['codings']
        target_category = segment['target_category']
        
        if not codings:
            return None
        
        # Da alle Kodierungen bereits die gleiche Hauptkategorie haben,
        # konzentrieren wir uns auf Subkategorien-Consensus
        
        # Sammle alle Subkategorien fuer diese Kategorie
        all_subcategories = []
        for coding in codings:
            subcats = coding.get('subcategories', [])
            all_subcategories.extend(subcats)
        
        # ZÄhle Subkategorien-HÄufigkeiten
        from collections import Counter
        subcat_counts = Counter(all_subcategories)
        
        # Consensus-Subkategorien: Nur die, die von der Mehrheit gewÄhlt wurden
        total_coders = len(codings)
        consensus_threshold = total_coders // 2 + 1  # Mehr als die HÄlfte
        
        consensus_subcats = []
        for subcat, count in subcat_counts.items():
            if count >= consensus_threshold:
                consensus_subcats.append(subcat)
        
        # WÄhle beste Kodierung als Basis
        best_coding = max(codings, key=lambda x: self._extract_confidence_value(x))
        consensus_coding = best_coding.copy()
        
        # Aktualisiere mit Consensus-Informationen
        # FIX: Preserve original internal segment_id format
        original_segment_id = segment.get('original_segment_id', segment['segment_id'])
        
        # For multiple coding, create proper multiple coding segment ID
        if segment.get('is_multiple_coding', False):
            instance_number = segment.get('instance_info', {}).get('instance_number', 1)
            final_segment_id = f"{original_segment_id}-{instance_number}"
        else:
            final_segment_id = original_segment_id
        
        consensus_coding.update({
            'segment_id': final_segment_id,
            'original_segment_id': original_segment_id,
            'category': target_category,  # Bereits gefiltert
            'subcategories': consensus_subcats,
            'is_multiple_coding': segment.get('is_multiple_coding', False),
            'multiple_coding_instance': segment.get('instance_info', {}).get('instance_number', 1),
            'total_coding_instances': segment.get('instance_info', {}).get('total_instances', 1),
            'consensus_info': {
                'total_coders': total_coders,
                'selection_type': 'consensus',
                'subcat_consensus_threshold': consensus_threshold,
                'original_segment_id': original_segment_id,
                'is_multiple_coding_instance': segment['is_multiple_coding'],
                'instance_info': segment['instance_info']
            }
        })
        
        # Log the decision
        decision_reason = f"Consensus erreicht: {len(consensus_subcats)} Subkategorien erfüllen Schwellenwert ({consensus_threshold}/{total_coders}). Beste Kodierung nach Konfidenz gewählt."
        additional_info = {
            "consensus_threshold": consensus_threshold,
            "total_coders": total_coders,
            "consensus_subcategories": consensus_subcats,
            "subcategory_counts": dict(subcat_counts),
            "chosen_by_confidence": True
        }
        self._log_review_decision(segment, consensus_coding, codings, "consensus", decision_reason, additional_info)
        
        return consensus_coding
    
    def _get_majority_for_category_segment(self, segment: Dict) -> Optional[Dict]:
        """
        Ermittelt Majority fuer ein kategorie-spezifisches Segment
        """
        codings = segment['codings']
        target_category = segment['target_category']
        
        if not codings:
            return None
        
        # WÄhle beste Kodierung als Basis (hÖchste Konfidenz)
        best_coding = max(codings, key=lambda x: self._extract_confidence_value(x))
        majority_coding = best_coding.copy()
        
        # Aktualisiere mit Majority-Informationen
        # FIX: Preserve original internal segment_id format
        original_segment_id = segment.get('original_segment_id', segment['segment_id'])
        
        # For multiple coding, create proper multiple coding segment ID
        if segment.get('is_multiple_coding', False):
            instance_number = segment.get('instance_info', {}).get('instance_number', 1)
            final_segment_id = f"{original_segment_id}-{instance_number}"
        else:
            final_segment_id = original_segment_id
        
        majority_coding.update({
            'segment_id': final_segment_id,
            'original_segment_id': original_segment_id,
            'category': target_category,
            'is_multiple_coding': segment.get('is_multiple_coding', False),
            'multiple_coding_instance': segment.get('instance_info', {}).get('instance_number', 1),
            'total_coding_instances': segment.get('instance_info', {}).get('total_instances', 1),
            'consensus_info': {
                'total_coders': len(codings),
                'selection_type': 'majority',
                'confidence_based_selection': True,
                'original_segment_id': original_segment_id,
                'is_multiple_coding_instance': segment['is_multiple_coding'],
                'instance_info': segment['instance_info']
            }
        })
        
        # Log the decision
        confidence_scores = [(coding.get('coder_id'), self._extract_confidence_value(coding)) for coding in codings]
        confidence_scores.sort(key=lambda x: x[1], reverse=True)
        
        decision_reason = f"Majority-Voting: Kodierung mit höchster Konfidenz gewählt ({best_coding.get('coder_id')}: {self._extract_confidence_value(best_coding):.3f})"
        additional_info = {
            "total_coders": len(codings),
            "confidence_scores": confidence_scores,
            "selection_method": "highest_confidence"
        }
        self._log_review_decision(segment, majority_coding, codings, "majority", decision_reason, additional_info)
        
        return majority_coding
    
    def _extract_confidence_value(self, coding: Dict) -> float:
        """
        Extrahiert Konfidenzwert aus Kodierung
        """
        confidence = coding.get('confidence', {})
        if isinstance(confidence, dict):
            return confidence.get('total', 0.0)
        elif isinstance(confidence, (int, float)):
            return float(confidence)
        return 0.0
    
    def _save_review_decisions(self, review_mode: str) -> None:
        """
        Save review decisions to JSON file with timestamped filename matching Excel export.
        """
        if not self.review_decisions:
            print("   ℹ️ Keine Review-Entscheidungen zu speichern")
            return
        
        import os
        from datetime import datetime
        
        decisions_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "review_mode": review_mode,
                "total_decisions": len(self.review_decisions),
                "version": "1.0"
            },
            "decisions": self.review_decisions
        }
        
        # Generate timestamped filename matching Excel export pattern
        if self.timestamp:
            # Use provided timestamp (matches Excel export)
            filename = f"QCA-AID_Analysis_{self.analysis_mode}_{self.timestamp}_review_decisions.json"
        else:
            # Generate new timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"QCA-AID_Analysis_{self.analysis_mode}_{timestamp}_review_decisions.json"
        
        decisions_file = os.path.join(self.output_dir, filename)
        
        try:
            with open(decisions_file, 'w', encoding='utf-8') as f:
                json.dump(decisions_data, f, indent=2, ensure_ascii=False)
            
            print(f"   💾 Review-Entscheidungen gespeichert: {filename}")
            print(f"   📊 {len(self.review_decisions)} Entscheidungen dokumentiert")
            
            # Summary statistics
            decision_types = {}
            for decision in self.review_decisions:
                decision_type = decision.get('decision_type', 'unknown')
                decision_types[decision_type] = decision_types.get(decision_type, 0) + 1
            
            print(f"   📋 Entscheidungstypen: {dict(decision_types)}")
            
        except Exception as e:
            print(f"   ⚠️ Fehler beim Speichern der Review-Entscheidungen: {e}")
    
    def _log_review_decision(self, segment: Dict, chosen_coding: Dict, all_codings: List[Dict], 
                           decision_type: str, decision_reason: str, additional_info: Dict = None) -> None:
        """
        Log a review decision for transparency.
        
        Args:
            segment: The segment being reviewed
            chosen_coding: The coding that was chosen
            all_codings: All available codings for this segment
            decision_type: Type of decision (consensus, majority, confidence_based, manual)
            decision_reason: Detailed reason for the decision
            additional_info: Additional decision metadata
        """
        decision = {
            "segment_id": segment['segment_id'],
            "original_segment_id": segment.get('original_segment_id', segment['segment_id']),
            "target_category": segment.get('target_category'),
            "is_multiple_coding": segment.get('is_multiple_coding', False),
            "decision_type": decision_type,
            "decision_reason": decision_reason,
            "chosen_coder": chosen_coding.get('coder_id'),
            "chosen_category": chosen_coding.get('category'),
            "chosen_subcategories": chosen_coding.get('subcategories', []),
            "chosen_confidence": self._extract_confidence_value(chosen_coding),
            "alternatives": [],
            "decision_metadata": additional_info or {}
        }
        
        # Document all alternative codings
        for coding in all_codings:
            if coding.get('coder_id') != chosen_coding.get('coder_id'):
                alternative = {
                    "coder_id": coding.get('coder_id'),
                    "category": coding.get('category'),
                    "subcategories": coding.get('subcategories', []),
                    "confidence": self._extract_confidence_value(coding),
                    "justification": coding.get('justification', '')[:200] + '...' if len(coding.get('justification', '')) > 200 else coding.get('justification', '')
                }
                decision["alternatives"].append(alternative)
        
        self.review_decisions.append(decision)


# ============================
# KORRIGIERTE KRIPPENDORFF'S ALPHA BERECHNUNG
# ============================

