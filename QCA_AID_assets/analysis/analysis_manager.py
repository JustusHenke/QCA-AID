"""
Integrierter Analyse-Manager für QCA-AID
=========================================
Koordiniert die verschiedenen Analysephasen in einem zusammenhängenden Prozess.
"""

import json
import asyncio
import time
import traceback
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime

from ..core.config import CONFIG, FORSCHUNGSFRAGE, KODIERREGELN, DEDUKTIVE_KATEGORIEN
from ..core.data_models import CategoryDefinition, CodingResult
from ..core.validators import CategoryValidator
from ..management import DevelopmentHistory, CategoryManager, CategoryRevisionManager
from ..export.results_exporter import ResultsExporter
from .relevance_checker import RelevanceChecker
from .deductive_coding import DeductiveCoder
from .inductive_coding import InductiveCoder
from .manual_coding import ManualCoder
from .saturation_controller import ImprovedSaturationController
from ..utils.tracking.token_tracker import TokenTracker
from ..utils.io.escape_handler import EscapeHandler, add_escape_handler_to_manager
from ..utils.impact_analysis import analyze_multiple_coding_impact
from ..utils.validators import validate_category_specific_segments
from ..utils.llm.response import LLMResponse
from ..QCA_Prompts import QCAPrompts

# Globaler Token-Counter
token_counter = TokenTracker()


class IntegratedAnalysisManager:

    def __init__(self, config: Dict):
        # Bestehende Initialisierung
        self.config = config
        self.history = DevelopmentHistory(config['OUTPUT_DIR'])

        # FIX: model_name Attribut hinzufÜgen fuer RelevanceChecker
        self.model_name = config['MODEL_NAME']

        # Batch Size aus Config
        self.batch_size = config.get('BATCH_SIZE', 5) 

        # Prompt-Handler initialisieren
        self.prompt_handler = QCAPrompts(
            forschungsfrage=FORSCHUNGSFRAGE,
            kodierregeln=KODIERREGELN,
            deduktive_kategorien=DEDUKTIVE_KATEGORIEN

        )
        
        # Zentrale RelevanzprÜfung
        self.relevance_checker = RelevanceChecker(
            model_name=self.model_name,
            batch_size=self.batch_size,
            temperature=config.get('RELEVANCE_CHECK_TEMPERATURE', 0.3)
        )
        
        # KORREKTUR: Initialisiere den verbesserten InductiveCoder
        self.inductive_coder = InductiveCoder(
            model_name=config['MODEL_NAME'],
            temperature=config.get('INDUCTIVE_CODER_TEMPERATURE', 0.2),
            history=self.history,
            output_dir=config['OUTPUT_DIR'],
            config=config  # Übergebe config fuer verbesserte Initialisierung
        )

        self.deductive_coders = [
            DeductiveCoder(
                config['MODEL_NAME'], 
                coder_config['temperature'],
                coder_config['coder_id']
            )
            for coder_config in config['CODER_SETTINGS']
        ]
        
        # Tracking-Variablen (unverÄndert)
        self.processed_segments = set()
        self.coding_results = []
        self.analysis_log = [] 
        self.performance_metrics = {
            'batch_processing_times': [],
            'coding_times': [],
            'category_changes': []
        }

        # Konfigurationsparameter (unverÄndert)
        self.use_context = config.get('CODE_WITH_CONTEXT', False)
        print(f"\nKontextuelle Kodierung: {'Aktiviert' if self.use_context else 'Deaktiviert'}")

        # Dictionary fuer die Verwaltung der Document-Summaries (unverÄndert)
        self.document_summaries = {}

        # NEU: Grounded Mode Spezifische Variablen
        self.grounded_subcodes_collection = []  # Zentrale Sammlung aller Subcodes
        self.grounded_keywords_collection = []  # Zentrale Sammlung aller Keywords
        self.grounded_segment_analyses = []     # Zentrale Sammlung aller Segment-Analysen
        self.grounded_batch_history = []        # Historie der Batch-Ergebnisse
        self.grounded_saturation_counter = 0    # ZÄhler fuer Batches ohne neue Subcodes


        # NEU: Escape-Handler hinzufÜgen (unverÄndert)
        self.escape_handler = EscapeHandler(self)
        self._should_abort = False
        self._escape_abort_requested = False

        self.current_categories = {}  # FIX: HinzufÜgen

        print(f"\n🧑‍💼 IntegratedAnalysisManager initialisiert:")
        print(f"   - Analysemodus: {config.get('ANALYSIS_MODE', 'inductive')}")
        if config.get('ANALYSIS_MODE') == 'grounded':
            print(f"   - Grounded Mode: Subcode-Sammlung aktiviert")
            print(f"   - Hauptkategorien werden erst am Ende generiert")

    async def _get_next_batch(self, 
                           segments: List[Tuple[str, str]], 
                           batch_size: float) -> List[Tuple[str, str]]:
        """
        Bestimmt den nÄchsten zu analysierenden Batch.
        
        Args:
            segments: Liste aller Segmente
            batch_size_percentage: Batch-GrÖẞe als Prozentsatz
            
        Returns:
            List[Tuple[str, str]]: NÄchster Batch von Segmenten
        """
        remaining_segments = [
            seg for seg in segments 
            if seg[0] not in self.processed_segments
        ]
        
        if not remaining_segments:
            return []
            
        batch_size = max(1, batch_size)
        return remaining_segments[:batch_size]
    
    
    async def _process_batch_inductively(self, 
                                    batch: List[Tuple[str, str]], 
                                    current_categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        VEREINFACHT: Keine weitere RelevanzprÜfung mehr nÖtig
        """
        # Die Segmente sind bereits in analyze_material gefiltert worden
        relevant_segments = [text for _, text in batch]  # Einfach die Texte extrahieren
        
        if not relevant_segments:
            print("   ℹ️ Keine Segmente in diesem Batch")
            return {}

        print(f"\n🕵️ Entwickle Kategorien aus {len(relevant_segments)} Segmenten")
        
        analysis_mode = CONFIG.get('ANALYSIS_MODE', 'inductive')
        
        if analysis_mode == 'inductive':
            return await self._process_inductive_mode(relevant_segments, current_categories)
        elif analysis_mode == 'abductive':
            return await self._process_abductive_mode(relevant_segments, current_categories)
        elif analysis_mode == 'grounded':
            return await self._process_grounded_mode(relevant_segments, current_categories)
        else:
            return {}

    async def _process_inductive_mode(self, relevant_segments: List[str], 
                                    current_categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        INDUCTIVE MODE: VollstÄndige induktive Kategorienentwicklung (ehemals full mode)
        """
        print("ℹ️ INDUCTIVE MODE: VollstÄndige induktive Kategorienentwicklung")
        print("   - Entwickle eigenstÄndiges induktives Kategoriensystem")
        print("   - Deduktive Kategorien werden ignoriert")
        
        # KORRIGIERT: Übergebe bestehende induktive Kategorien als Basis
        new_categories = await self.inductive_coder.develop_category_system(
            relevant_segments,
            current_categories  # ✅ Bestehende induktive als Basis!
        )
        
        print(f"✅ INDUCTIVE MODE: {len(new_categories)} Kategorien entwickelt")
        if current_categories:
            print(f"   (zusÄtzlich zu {len(current_categories)} bereits bestehenden)")
        return new_categories

    async def _process_abductive_mode(self, relevant_segments: List[str], 
                                    current_categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        ABDUCTIVE MODE: Nur Subkategorien zu bestehenden Hauptkategorien
        """
        print("ℹ️ ABDUCTIVE MODE: Erweitere bestehende Kategorien um Subkategorien")
        
        if not current_categories:
            print("❌ ABDUCTIVE MODE: Keine bestehenden Kategorien zum Erweitern")
            return {}
        
        # Spezielle abduktive Analyse
        extended_categories = await self._analyze_for_subcategories(
            relevant_segments, 
            current_categories
        )
        
        print(f"✅ ABDUCTIVE MODE: {len(extended_categories)} Kategorien erweitert")
        return extended_categories

    
    async def _process_grounded_mode(self, relevant_segments: List[str], 
                                current_categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        KORRIGIERT: Diese Methode wird in normalen Batches NICHT aufgerufen im Grounded Mode
        """
        print("❌ WARNUNG: _process_grounded_mode sollte nicht in separatem Grounded Mode aufgerufen werden!")
        return {}
    
    async def _assess_grounded_saturation(self, batch_count: int, total_batches: int) -> bool:
        """
        KORRIGIERTE SÄttigungslogik fuer Grounded Mode.
        """
        try:
            # Berechne Material-Fortschritt
            if hasattr(self, 'chunks') and self.chunks:
                total_segments = sum(len(chunk_list) for chunk_list in self.chunks.values())
            elif hasattr(self, '_total_segments'):
                total_segments = self._total_segments
            else:
                total_segments = max(len(self.processed_segments) * 2, 20)
            
            material_percentage = (len(self.processed_segments) / total_segments) * 100 if total_segments > 0 else 0.0
            
            # KORRIGIERT: Verwende die richtige Sammlung
            if not hasattr(self, 'grounded_subcodes_collection'):
                self.grounded_subcodes_collection = []
            
            # Analyse der Subcode-Entwicklung
            subcode_diversity = len(set(sc['name'] for sc in self.grounded_subcodes_collection))
            
            if not hasattr(self, 'grounded_keywords_collection'):
                self.grounded_keywords_collection = []
            keyword_diversity = len(set(self.grounded_keywords_collection))
            
            # Berechne SÄttigungsmetriken
            avg_subcodes_per_batch = len(self.grounded_subcodes_collection) / max(batch_count, 1)
            
            # Kriterien fuer Grounded Mode SÄttigung
            criteria = {
                'min_batches': batch_count >= 3,  # Mindestens 3 Batches
                'material_coverage': material_percentage >= 70,  # 70% Material verarbeitet
                'subcodes_collected': len(self.grounded_subcodes_collection) >= 8,  # Min. 8 Subcodes
                'saturation_stability': self.grounded_saturation_counter >= 2,  # 2 Batches ohne neue
                'diversity_threshold': subcode_diversity >= 5,  # Mindestens 5 verschiedene Subcodes
                'keyword_richness': keyword_diversity >= 15,  # Mindestens 15 verschiedene Keywords
            }
            
            print(f"\n🕵️ Grounded Mode SÄttigungsprÜfung (Batch {batch_count}/{total_batches}):")
            print(f"🧾 Aktuelle Metriken:")
            print(f"   - Material-Fortschritt: {material_percentage:.1f}%")
            print(f"   - Gesammelte Subcodes: {len(self.grounded_subcodes_collection)}")
            print(f"   - Subcode-DiversitÄt: {subcode_diversity}")
            print(f"   - Keyword-DiversitÄt: {keyword_diversity}")
            print(f"   - SÄttigungs-Counter: {self.grounded_saturation_counter}")
            print(f"   - Ae˜ Subcodes/Batch: {avg_subcodes_per_batch:.1f}")
            
            print(f"\n🎯 SÄttigungskriterien:")
            for criterion, met in criteria.items():
                status = "✅" if met else "⚠️"
                print(f"   {status} {criterion}: {met}")
            
            # Bestimme SÄttigungsstatus
            critical_criteria = ['min_batches', 'subcodes_collected', 'saturation_stability']
            critical_met = all(criteria[crit] for crit in critical_criteria)
            
            # VollstÄndige SÄttigung: Alle Kriterien oder kritische + Material fast vollstÄndig
            full_saturation = all(criteria.values())
            partial_saturation = critical_met and (material_percentage >= 85 or criteria['material_coverage'])
            forced_saturation = material_percentage >= 100  # 100% Material = ZwangssÄttigung
            
            is_saturated = full_saturation or partial_saturation or forced_saturation
            
            if is_saturated:
                saturation_type = "VollstÄndig" if full_saturation else ("Partiell" if partial_saturation else "Material-bedingt")
                print(f"\n🎯 GROUNDED MODE SÄTTIGUNG erreicht ({saturation_type}):")
                print(f"   - Material: {material_percentage:.1f}% verarbeitet")
                print(f"   - Subcodes: {len(self.grounded_subcodes_collection)} gesammelt")
                print(f"   - SÄttigungs-Counter: {self.grounded_saturation_counter}")
            else:
                print(f"\nℹ️ SÄttigung noch nicht erreicht - setze Subcode-Sammlung fort")
                missing_criteria = [k for k, v in criteria.items() if not v]
                print(f"   - Fehlende Kriterien: {', '.join(missing_criteria)}")
            
            return is_saturated
            
        except Exception as e:
            print(f"⚠️ Fehler bei Grounded Mode SÄttigungsprÜfung: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback: Bei Fehler weiter sammeln, auẞer 100% Material erreicht
            return material_percentage >= 100
        
    async def _analyze_for_subcategories(self, relevant_segments: List[str], 
                                       current_categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        SPEZIELLE ABDUKTIVE ANALYSE: Nur Subkategorien entwickeln
        """
        segments_text = "\n\n=== SEGMENT BREAK ===\n\n".join(
            f"SEGMENT {i + 1}:\n{text}" 
            for i, text in enumerate(relevant_segments)
        )

        # Erstelle Kategorien-Kontext fuer abduktive Analyse
        categories_context = []
        for cat_name, cat_def in current_categories.items():
            categories_context.append({
                'name': cat_name,
                'definition': cat_def.definition[:200],
                'existing_subcategories': list(cat_def.subcategories.keys())
            })

        prompt = self.prompt_handler.get_analyze_for_subcategories_prompt(
                segments_text=segments_text,
                categories_context=categories_context
            )
        

        try:
            token_counter.start_request()
            
            response = await self.inductive_coder.llm_provider.create_completion(
                model=self.inductive_coder.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte fuer qualitative Inhaltsanalyse. Antworte auf deutsch."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.inductive_coder.temperature,
                response_format={"type": "json_object"}
            )
            
            llm_response = LLMResponse(response)
            result = json.loads(llm_response.content)
            
            
            token_counter.track_response(response, self.model_name)
            
            # Verarbeite Ergebnisse - erweitere bestehende Kategorien
            extended_categories = current_categories.copy()
            total_new_subcats = 0
            
            for main_cat_name, updates in result.get('extended_categories', {}).items():
                if main_cat_name in extended_categories:
                    current_cat = extended_categories[main_cat_name]
                    new_subcats = {}
                    
                    for sub_data in updates.get('new_subcategories', []):
                        if sub_data.get('confidence', 0) >= 0.7:  # Schwelle fuer Subkategorien
                            new_subcats[sub_data['name']] = sub_data['definition']
                            total_new_subcats += 1
                            print(f"✅ Neue Subkategorie: {main_cat_name} -> {sub_data['name']}")
                    
                    if new_subcats:
                        # Erweitere bestehende Kategorie
                        extended_categories[main_cat_name] = current_cat.replace(
                            subcategories={**current_cat.subcategories, **new_subcats},
                            modified_date=datetime.now().strftime("%Y-%m-%d")
                        )
            
            print(f"🧾 Abduktive Entwicklung: {total_new_subcats} neue Subkategorien")
            return extended_categories
            
        except Exception as e:
            print(f"Fehler bei abduktiver Analyse: {str(e)}")
            return current_categories

    async def _code_batch_deductively(self, 
                                     batch: List[Tuple[str, str]], 
                                     categories: Dict[str, CategoryDefinition],
                                     category_preselections: Dict[str, Dict] = None) -> List[Dict]:
        """
        Kodiert einen Batch parallel ohne progressive Kontext-FunktionalitÄt.
        FIX: Erweitert um Kategorie-Vorauswahl fuer deduktiven Modus
        BUGFIX: Verwendet separate, lockere RelevanzprÜfung fuer Kodierung.
        """
        print(f"\n🚀 PARALLEL-KODIERUNG: {len(batch)} Segmente gleichzeitig")
        start_time = time.time()
        
        # FIX: Standardwert fuer category_preselections
        if category_preselections is None:
            category_preselections = {}
        
        # FIX: Zeige Kategorie-Vorauswahl-Informationen
        if category_preselections:
            preselected_count = len([s for s in batch if s[0] in category_preselections])
            print(f"🎯 {preselected_count} Segmente haben Kategorie-PrÄferenzen")
            
            # Statistik der Kategorie-PrÄferenzen
            all_preferred = []
            for prefs in category_preselections.values():
                all_preferred.extend(prefs.get('preferred_categories', []))
            
            if all_preferred:
                from collections import Counter
                pref_stats = Counter(all_preferred)
                print(f"🎯 HÄufigste PrÄferenzen: {dict(pref_stats.most_common(3))}")
        
        print(f"\n🕵️ PrÜfe Kodierungs-Relevanz...")
        coding_relevance_results = await self.relevance_checker.check_relevance_batch(batch)
        
        # Debug-Ausgaben
        print(f"\n🕵️ Kodierungs-RelevanzprÜfung Ergebnisse:")
        relevant_count = sum(1 for is_relevant in coding_relevance_results.values() if is_relevant)
        print(f"   - Segmente geprÜft: {len(coding_relevance_results)}")
        print(f"   - Als kodierungsrelevant eingestuft: {relevant_count}")
        print(f"   - Als nicht kodierungsrelevant eingestuft: {len(coding_relevance_results) - relevant_count}")
        
        # 2. PARALLEL: Mehrfachkodierungs-PrÜfung (wenn aktiviert)
        multiple_coding_results = {}
        if CONFIG.get('MULTIPLE_CODINGS', True):
            coding_relevant_segments = [
                (segment_id, text) for segment_id, text in batch
                if coding_relevance_results.get(segment_id, True)
            ]
            
            if coding_relevant_segments:
                print(f"  ℹ️ PrÜfe {len(coding_relevant_segments)} kodierungsrelevante Segmente auf Mehrfachkodierung...")
                multiple_coding_results = await self.relevance_checker.check_multiple_category_relevance(
                    coding_relevant_segments, categories
                )
        
        # 3. PARALLEL: Kodierung aller Segmente
        async def code_single_segment_all_coders(segment_id: str, text: str) -> List[Dict]:
            """FIX: Kodiert ein einzelnes Segment mit gefilterten Kategorien basierend auf Vorauswahl."""
            
            # EARLY CHECK: Bestätige Kodierungsrelevanz VOR kategorie-Vorbereitung
            is_coding_relevant = coding_relevance_results.get(segment_id, True)  # Default: True

            # Zusätzliche einfache Heuristik fuer offensichtlich irrelevante Inhalte
            if len(text.strip()) < 20:
                is_coding_relevant = False
                print(f"   ❌ Segment {segment_id} zu kurz fuer Kodierung")
                
            text_lower = text.lower()
            exclusion_patterns = [
                'seite ', 'page ', 'copyright', '©', 'datum:', 'date:',
                'inhaltsverzeichnis', 'table of contents', 'literaturverzeichnis',
                'bibliography', 'anhang', 'appendix'
            ]
            
            is_metadata = any(pattern in text_lower for pattern in exclusion_patterns)
            if is_metadata and len(text) < 100:
                is_coding_relevant = False
                print(f"   ❌ Segment {segment_id} als Metadaten erkannt")
            
            # EARLY RETURN: Falls nicht kodierungsrelevant, direkt "Nicht kodiert" zurückgeben OHNE Kategorienvorbereitung
            if not is_coding_relevant:
                print(f"   ❌ Segment {segment_id} wird als 'Nicht kodiert' markiert")
                
                # FIX: Hole Begründung aus RelevanceChecker falls verfügbar
                relevance_details = self.relevance_checker.get_relevance_details(segment_id)
                justification = "Nicht relevant fuer Kodierung (zu kurz oder Metadaten)"
                if relevance_details:
                    if 'reasoning' in relevance_details and relevance_details['reasoning']:
                        justification = relevance_details['reasoning']
                    elif 'justification' in relevance_details and relevance_details['justification']:
                        justification = relevance_details['justification']
                
                # Spezifische Fallback-Begründungen
                if len(text.strip()) < 20:
                    justification = "Segment zu kurz fuer sinnvolle Kodierung"
                elif is_metadata:
                    justification = "Segment als Metadaten (z.B. Seitenzahl, Inhaltsverzeichnis) erkannt"
                
                # FIX: Holt preselection-Info für nicht-kodierte Segmente
                preselection = category_preselections.get(segment_id, {})
                preferred_cats = preselection.get('preferred_categories', [])
                
                not_coded_results = []
                for coder in self.deductive_coders:
                    result = {
                        'segment_id': segment_id,
                        'coder_id': coder.coder_id,
                        'category': "Nicht kodiert",
                        'subcategories': [],
                        'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                        'justification': justification,
                        'text': text,
                        'multiple_coding_instance': 1,
                        'total_coding_instances': 1,
                        'target_category': '',
                        'category_focus_used': False,
                        # FIX: Zusätzliche Kategorie-Vorauswahl-Info
                        'category_preselection_used': bool(preferred_cats),
                        'preferred_categories': preferred_cats,
                        'preselection_reasoning': preselection.get('reasoning', '')
                    }
                    not_coded_results.append(result)
                return not_coded_results
            
            # NOW: Bestimme effektive Kategorien fuer dieses Segment (nur wenn kodierungsrelevant)
            preselection = category_preselections.get(segment_id, {})
            preferred_cats = preselection.get('preferred_categories', [])
            
            if preferred_cats:
                # FIX: Verwende vollstÄndige CategoryDefinition-Objekte aus categories
                effective_categories = {
                    name: cat for name, cat in categories.items() 
                    if name in preferred_cats and isinstance(cat, CategoryDefinition)  # FIX: Validiere CategoryDefinition
                }
                
                if not effective_categories:
                    print(f"    ❌ Keine gÜltigen CategoryDefinition-Objekte in preferred_cats - verwende alle Kategorien")
                    effective_categories = categories
                else:
                    print(f"    🎯 Segment {segment_id}: Fokus auf {len(effective_categories)} Kategorien: {', '.join(preferred_cats)}")
                    
                    # FIX: Validiere, dass effective_categories vollstÄndige Definitionen hat
                    for name, cat in effective_categories.items():
                        if not hasattr(cat, 'subcategories'):
                            print(f"    ❌ KRITISCH: effective_categories['{name}'] fehlen Subkategorien - hole aus categories")
                            if name in categories:
                                effective_categories[name] = categories[name]
            else:
                # FIX: Fallback auf alle Kategorien wenn keine Vorauswahl
                effective_categories = categories
                print(f"    🔀 Segment {segment_id}: Standard-Kodierung mit allen {len(categories)} Kategorien")
            
            
            # Bestimme Kodierungsinstanzen (fuer Mehrfachkodierung)
            coding_instances = []
            multiple_categories = multiple_coding_results.get(segment_id, [])
            
            if len(multiple_categories) > 1:
                # Mehrfachkodierung
                for i, category_info in enumerate(multiple_categories, 1):
                    coding_instances.append({
                        'instance': i,
                        'total_instances': len(multiple_categories),
                        'target_category': category_info['category'],
                        'category_context': category_info
                    })
            else:
                # Standardkodierung
                coding_instances.append({
                    'instance': 1,
                    'total_instances': 1,
                    'target_category': '',
                    'category_context': None
                })
            
            # 🚀 PARALLEL: Alle Kodierer fuer alle Instanzen
            async def code_with_coder_and_instance(coder, instance_info):
                """FIX: Kodiert mit einem Kodierer unter Verwendung der vollstÄndigen CategoryDefinition-Objekte."""
                try:

                    # FIX: Bei Fokuskodierung die target_category zu effective_categories hinzufÜgen
                    enhanced_categories = effective_categories.copy()
                    target_cat = instance_info['target_category']
                    if target_cat:
                        if target_cat and target_cat not in enhanced_categories:
                            if target_cat in categories:  
                                enhanced_categories[target_cat] = categories[target_cat]  
                                print(f"    🎯 Fokuskategorie '{target_cat}' zu verfÜgbaren Kategorien hinzugefÜgt")
                            else:
                                print(f"    ❌ Fokuskategorie '{target_cat}' nicht in Kategorien vorhanden")

                    if target_cat:
                        coding = await coder.code_chunk_with_focus(
                            text, enhanced_categories,
                            focus_category=target_cat,
                            focus_context=instance_info['category_context']
                        )
                    else:
                        coding = await coder.code_chunk(text, enhanced_categories)


                    if coding and isinstance(coding, CodingResult):
                        main_category = coding.category
                        original_subcats = list(coding.subcategories)
                        
                        # FIX: Verwende enhanced_categories fuer Validierung
                        validated_subcats = original_subcats  # Fallback
                        validation_source = "keine"
                        
                        # 1. PrioritÄt: enhanced_categories (gefilterte + Fokuskategorien)
                        if main_category in enhanced_categories and hasattr(enhanced_categories[main_category], 'subcategories'):
                            try:
                                validated_subcats = CategoryValidator.validate_subcategories_for_category(
                                    original_subcats, main_category, enhanced_categories, warn_only=False
                                )
                                validation_source = "enhanced_categories"
                            except Exception as e:
                                print(f"    ⚠️ Validierung mit enhanced_categories fehlgeschlagen: {str(e)}")
                                # FIX: Fallback zu self.current_categories
                                if hasattr(self, 'current_categories') and main_category in self.current_categories:
                                    try:
                                        validated_subcats = CategoryValidator.validate_subcategories_for_category(
                                            original_subcats, main_category, self.current_categories, warn_only=False
                                        )
                                        validation_source = "self.current_categories_fallback"
                                    except Exception as e2:
                                        print(f"    ⚠️ Auch Fallback-Validierung fehlgeschlagen: {str(e2)}")
                        else:
                            # FIX: Informative Meldung bei nicht verfÜgbaren Kategorien
                            if main_category not in enhanced_categories:
                                print(f"    ℹ️ Kategorie '{main_category}' nicht in verfÜgbaren Kategorien")
                                print(f"    🎯 VerfÜgbare Kategorien: {list(enhanced_categories.keys())}")
                            elif not hasattr(enhanced_categories[main_category], 'subcategories'):
                                print(f"    ℹ️ Keine Subkategorie-Definitionen verfÜgbar fuer '{main_category}'")
                        
                        # FIX: Debug-Ausgabe nur bei wichtigen Ereignissen
                        if len(original_subcats) != len(validated_subcats):
                            removed = set(original_subcats) - set(validated_subcats)
                            print(f"    🔧 Subkategorien bereinigt: {len(original_subcats)} -> {len(validated_subcats)}")
                            if removed:
                                print(f"    🔧 Entfernt: {list(removed)} (Quelle: {validation_source})")
                        elif validation_source != "keine" and original_subcats:
                            print(f"    ✅ Alle {len(original_subcats)} Subkategorien gÜltig (Quelle: {validation_source})")
                        elif validation_source == "keine" and original_subcats:
                            print(f"    ℹ️ Subkategorien-Validierung Übersprungen fuer '{main_category}' (Quelle: {validation_source})")
                        
                        return {
                            'segment_id': segment_id,
                            'coder_id': coder.coder_id,
                            'category': coding.category,
                            'subcategories': validated_subcats,  # FIX: Immer validierte oder ursprÜngliche Subkategorien
                            'confidence': coding.confidence,
                            'justification': coding.justification,
                            'text': text,
                            'paraphrase': coding.paraphrase,
                            'keywords': coding.keywords,
                            'multiple_coding_instance': instance_info['instance'],
                            'total_coding_instances': instance_info['total_instances'],
                            'target_category': instance_info['target_category'],
                            'category_focus_used': bool(instance_info['target_category']),
                            # FIX: Verbesserte Debug-Informationen fuer enhanced_categories
                            'category_preselection_used': bool(preferred_cats),
                            'preferred_categories': preferred_cats,
                            'effective_categories_count': len(effective_categories),
                            'enhanced_categories_count': len(enhanced_categories),  # FIX: Neue Info
                            'preselection_reasoning': preselection.get('reasoning', ''),
                            'subcategories_validated': len(original_subcats) != len(validated_subcats),
                            'validation_source': validation_source,
                            'validation_successful': validation_source != "keine",
                            'category_in_enhanced': main_category in enhanced_categories,  # FIX: Verwende enhanced_categories
                            'enhanced_has_subcategories': main_category in enhanced_categories and hasattr(enhanced_categories[main_category], 'subcategories'),  # FIX: Neue Debug-Info
                            'focus_category_added': instance_info['target_category'] and instance_info['target_category'] not in effective_categories  # FIX: Neue Info
                        }
                        
                    else:
                        return None
                        
                except Exception as e:
                    print(f"    ❌ Kodierungsfehler {coder.coder_id}: {str(e)}")
                    return None
                
            # Erstelle Tasks fuer alle Kodierer Ae— alle Instanzen
            coding_tasks = []
            for instance_info in coding_instances:
                for coder in self.deductive_coders:
                    task = code_with_coder_and_instance(coder, instance_info)
                    coding_tasks.append(task)
            
            # FÜhre alle Kodierungen fuer dieses Segment parallel aus
            coding_results = await asyncio.gather(*coding_tasks, return_exceptions=True)
            
            # Sammle erfolgreiche Ergebnisse
            successful_codings = []
            for result in coding_results:
                if not isinstance(result, Exception) and result:
                    successful_codings.append(result)
            
            return successful_codings
        
        # 🚀 Erstelle Tasks fuer alle Segmente des Batches
        segment_tasks = [
            code_single_segment_all_coders(segment_id, text) 
            for segment_id, text in batch
        ]
        
        print(f"🚀 Starte parallele Kodierung von {len(segment_tasks)} Segmenten...")
        
        # 🚀 FÜhre alle Segment-Kodierungen parallel aus
        all_segment_results = await asyncio.gather(*segment_tasks, return_exceptions=True)
        
        # Sammle alle Ergebnisse
        batch_results = []
        successful_segments = 0
        error_count = 0
        preselection_used_count = 0
        validation_performed_count = 0
        
        for segment_result in all_segment_results:
            if isinstance(segment_result, Exception):
                print(f"❌ Segment-Fehler: {segment_result}")
                error_count += 1
                continue
                
            if segment_result:  # Liste von Kodierungen fuer dieses Segment
                batch_results.extend(segment_result)
                successful_segments += 1
                
                # FIX: Sammle Statistiken Über Kategorie-Vorauswahl-Nutzung
                for coding in segment_result:
                    if coding.get('category_preselection_used', False):
                        preselection_used_count += 1
                    if coding.get('subcategories_validated', False):
                        validation_performed_count += 1
        
        # Markiere verarbeitete Segmente
        for segment_id, text in batch:
            self.processed_segments.add(segment_id)
        
        processing_time = time.time() - start_time
        
        print(f"✅ PARALLEL-BATCH ABGESCHLOSSEN:")
        print(f"   âš¡ Zeit: {processing_time:.2f}s")
        if processing_time > 0:
            print(f"   🚀 Geschwindigkeit: {len(batch)/processing_time:.1f} Segmente/Sekunde")
        else:
            print(f"   🚀 Geschwindigkeit: {len(batch)} Segmente in <0.01s (sehr schnell)")
        print(f"   ✅ Erfolgreiche Segmente: {successful_segments}/{len(batch)}")
        print(f"   🧾 Gesamte Kodierungen: {len(batch_results)}")
        # FIX: ZusÄtzliche Statistiken fuer Kategorie-Vorauswahl
        if category_preselections:
            print(f"   🎯 Kategorie-Vorauswahl genutzt: {preselection_used_count} Kodierungen")
            print(f"   🔧 Subkategorie-Validierung durchgefÜhrt: {validation_performed_count} Kodierungen")
        if error_count > 0:
            print(f"   ❌ Fehler: {error_count}")
        
        return batch_results
    
    async def _code_with_category_focus(self, coder, text, categories, instance_info):
        """Kodiert mit optionalem Fokus auf bestimmte Kategorie"""
        
        if instance_info['target_category']:
            # Mehrfachkodierung mit Kategorie-Fokus
            return await coder.code_chunk_with_focus(
                text, categories, 
                focus_category=instance_info['target_category'],
                focus_context=instance_info['category_context']
            )
        else:
            # Standard-Kodierung
            return await coder.code_chunk(text, categories)

    async def _code_batch_with_context(self, batch: List[Tuple[str, str]], 
                                     categories: Dict[str, CategoryDefinition],
                                     category_preselections: Dict[str, Dict] = None) -> List[Dict]:
        """
        Kodiert einen Batch sequentiell mit progressivem Dokumentkontext und Mehrfachkodierung.
        FIX: Erweitert um category_preselections Parameter fuer gefilterte Kategorien
        """
        # FIX: Standardwert fuer category_preselections
        if category_preselections is None:
            category_preselections = {}
        
        batch_results = []
        
        # Debug-Info fuer Kategorie-PrÄferenzen
        if category_preselections:
            preselected_count = len([s for s in batch if s[0] in category_preselections])
            print(f"🎯 Kontext-Kodierung: {preselected_count} Segmente haben Kategorie-PrÄferenzen")
        
            # FIX: Erweiterte Statistik zur Kategorie-Vorauswahl wie im ohne-Kontext-Modus
            all_preferred = []
            for prefs in category_preselections.values():
                all_preferred.extend(prefs.get('preferred_categories', []))
            
            if all_preferred:
                from collections import Counter
                pref_stats = Counter(all_preferred)
                print(f"🎯 HÄufigste PrÄferenzen: {dict(pref_stats.most_common(3))}")
        else:
            print("🎯 Kontext-Kodierung: Keine Kategorie-PrÄferenzen Übertragen")

        # PrÜfe Mehrfachkodierungs-MÖglichkeiten fuer den ganzen Batch
        multiple_coding_results = {}
        if CONFIG.get('MULTIPLE_CODINGS', True):
            # RelevanzprÜfung fuer ganzen Batch
            relevance_results = await self.relevance_checker.check_relevance_batch(batch)
            relevant_segments = [
                (segment_id, text) for segment_id, text in batch
                if relevance_results.get(segment_id, False)
            ]
            
            if relevant_segments:
                print(f"  ℹ️ PrÜfe {len(relevant_segments)} relevante Segmente auf Mehrfachkodierung...")
                multiple_coding_results = await self.relevance_checker.check_multiple_category_relevance(
                    relevant_segments, categories
                )
        
        # Sequentielle Verarbeitung um Kontext aufzubauen
        for segment_id, text in batch:
            # Extrahiere Dokumentnamen und Chunk-ID
            doc_name, chunk_id = self._extract_doc_and_chunk_id(segment_id)
            position = f"Chunk {chunk_id}"
            
            # Hole aktuelles Summary oder initialisiere es
            current_summary = self.document_summaries.get(doc_name, "")
            
            # Segment-Informationen
            segment_info = {
                'doc_name': doc_name,
                'chunk_id': chunk_id,
                'position': position
            }
            
            # Kontext-Daten vorbereiten
            context_data = {
                'current_summary': current_summary,
                'segment_info': segment_info
            }
            
            # FIX: Bestimme gefilterte Kategorien fuer dieses Segment
            preselection = category_preselections.get(segment_id, {})
            preferred_cats = preselection.get('preferred_categories', [])
            
            if preferred_cats:
                # FIX: Gefilterte Kategorien fuer Kodierung verwenden
                filtered_categories = {
                    name: cat for name, cat in categories.items() 
                    if name in preferred_cats
                }
                print(f"\n🕵️ Verarbeite Segment {segment_id} mit Kontext (🎯 Fokus auf {len(filtered_categories)} Kategorien: {', '.join(preferred_cats)})")
                effective_categories = filtered_categories
            else:
                # FIX: Fallback auf alle Kategorien
                print(f"\n🕵️ Verarbeite Segment {segment_id} mit Kontext")
                effective_categories = categories
            
            # PrÜfe Relevanz
            relevance_result = await self.relevance_checker.check_relevance_batch([(segment_id, text)])
            is_relevant = relevance_result.get(segment_id, False)
            
            if not is_relevant:
                print(f"  🗑️ Segment als nicht relevant markiert - wird Übersprungen")
                
                # FIX: Hole spezifische BegrÜndung aus RelevanceChecker
                relevance_details = self.relevance_checker.get_relevance_details(segment_id)
                justification = "Nicht relevant fuer Forschungsfrage"
                if relevance_details:
                    if 'reasoning' in relevance_details and relevance_details['reasoning']:
                        justification = relevance_details['reasoning']
                    elif 'justification' in relevance_details and relevance_details['justification']:
                        justification = relevance_details['justification']
                # FIX: Ende
                
                # Erstelle "Nicht kodiert" Ergebnis fuer alle Kodierer
                for coder in self.deductive_coders:
                    result = {
                        'segment_id': segment_id,
                        'coder_id': coder.coder_id,
                        'category': "Nicht kodiert",
                        'subcategories': [],
                        'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                        'justification': justification,  # FIX: Verwende spezifische BegrÜndung
                        'text': text,
                        'context_summary': current_summary,
                        'multiple_coding_instance': 1,
                        'total_coding_instances': 1,
                        'target_category': '',
                        'category_focus_used': False,
                        # FIX: Neue Felder fuer Kategorie-PrÄferenzen
                        'category_preselection_used': bool(preferred_cats),
                        'preselected_categories': preferred_cats,
                        'category_filtering_applied': bool(preferred_cats)
                    }
                    batch_results.append(result)
                continue
            
            # Bestimme Kodierungsinstanzen
            coding_instances = []
            multiple_categories = multiple_coding_results.get(segment_id, [])
            
            if len(multiple_categories) > 1:
                print(f"  ℹ️ Mehrfachkodierung mit Kontext: {len(multiple_categories)} Kategorien")
                for i, category_info in enumerate(multiple_categories, 1):
                    coding_instances.append({
                        'instance': i,
                        'total_instances': len(multiple_categories),
                        'target_category': category_info['category'],
                        'category_context': category_info
                    })
            else:
                coding_instances.append({
                    'instance': 1,
                    'total_instances': 1,
                    'target_category': '',
                    'category_context': None
                })
            
            # Verarbeite relevante Segmente mit Kontext fuer ALLE Kodierer und Instanzen
            updated_summary = current_summary
            
            for instance_info in coding_instances:
                if instance_info['total_instances'] > 1:
                    print(f"\n    🔀 Kontext-Kodierung {instance_info['instance']}/{instance_info['total_instances']}")
                    print(f"        Fokus: {instance_info['target_category']}")
            
                for coder_index, coder in enumerate(self.deductive_coders):
                    try:
                        # Bestimme ob Summary aktualisiert werden soll (nur beim ersten Kodierer der ersten Instanz)
                        should_update_summary = (coder_index == 0 and instance_info['instance'] == 1)

                        # FIX: Enhanced Categories Logic - fÜge Fokuskategorie zu effective_categories hinzu
                        enhanced_categories = effective_categories.copy()
                        target_cat = instance_info['target_category']
                        if target_cat:
                            if target_cat and target_cat not in enhanced_categories:
                                if target_cat in categories:  
                                    enhanced_categories[target_cat] = categories[target_cat]  
                                    print(f"    🎯 Fokuskategorie '{target_cat}' zu verfÜgbaren Kategorien hinzugefÜgt")
                                else:
                                    print(f"    ❌ Fokuskategorie '{target_cat}' nicht in Kategorien vorhanden")
                        
                        if instance_info['target_category']:
                            # FIX: Mehrfachkodierung mit Fokus und Kontext (mit gefilterten Kategorien)
                            combined_result = await coder.code_chunk_with_focus_and_context(
                                text, enhanced_categories,  
                                focus_category=instance_info['target_category'],
                                focus_context=instance_info['category_context'],
                                current_summary=updated_summary if should_update_summary else current_summary,
                                segment_info=segment_info,
                                update_summary=should_update_summary
                            )
                        else:
                            # FIX: Standard Kontext-Kodierung (mit gefilterten Kategorien)
                            combined_result = await coder.code_chunk_with_progressive_context(
                                text, 
                                enhanced_categories,  
                                updated_summary if should_update_summary else current_summary,
                                segment_info
                            )
                        
                        if combined_result:
                            # Extrahiere Kodierungsergebnis und ggf. aktualisiertes Summary
                            coding_result = combined_result.get('coding_result', {})
                            
                            # Summary nur beim ersten Kodierer der ersten Instanz aktualisieren
                            if should_update_summary:
                                updated_summary = combined_result.get('updated_summary', current_summary)
                                self.document_summaries[doc_name] = updated_summary
                                print(f"ℹ️ Summary aktualisiert: {len(updated_summary.split())} WÖrter")
                            
                            # FIX: Erstelle erweiterten Kodierungseintrag mit Kategorie-PrÄferenzen
                            coding_entry = {
                                'segment_id': segment_id,
                                'coder_id': coder.coder_id,
                                'category': coding_result.get('category', ''),
                                'subcategories': coding_result.get('subcategories', []),
                                'justification': coding_result.get('justification', ''),
                                'confidence': coding_result.get('confidence', {}),
                                'text': text,
                                'paraphrase': coding_result.get('paraphrase', ''),
                                'keywords': coding_result.get('keywords', ''),
                                'context_summary': updated_summary,
                                'context_influence': coding_result.get('context_influence', ''),
                                'multiple_coding_instance': instance_info['instance'],
                                'total_coding_instances': instance_info['total_instances'],
                                'target_category': instance_info['target_category'],
                                'category_focus_used': bool(instance_info['target_category']),
                                # FIX: Neue Felder fuer Kategorie-PrÄferenzen
                                'category_preselection_used': bool(preferred_cats),
                                'preselected_categories': preferred_cats,
                                'category_filtering_applied': bool(preferred_cats),
                                'relevance_scores': preselection.get('relevance_scores', {}),
                                'preselection_reasoning': preselection.get('reasoning', '')
                            }
                            
                            # FIX: Validiere Subkategorien gegen die gewÄhlte Hauptkategorie
                            main_category = coding_entry['category']
                            original_subcats = coding_entry['subcategories']
                            if main_category and main_category != 'Nicht kodiert':
                                # FIX: Verwende enhanced_categories fuer Validierung (schon vorhanden)
                                categories_dict_for_validation = enhanced_categories
                                
                                if not categories_dict_for_validation:
                                    print(f"❌ KRITISCH: Kein categories_dict verfÜgbar fuer Validierung!")
                                    # Fallback: verwende original categories
                                    categories_dict_for_validation = categories
                                
                                try:
                                    validated_subcats = CategoryValidator.validate_subcategories_for_category(
                                        original_subcats, main_category, categories_dict_for_validation, warn_only=False
                                    )
                                    if len(validated_subcats) != len(original_subcats):
                                        print(f"    ❌ Subkategorien bereinigt: {len(original_subcats)} -> {len(validated_subcats)}")
                                    coding_entry['subcategories'] = validated_subcats
                                except Exception as e:
                                    print(f"    ⚠️ Subkategorien-Validierung fehlgeschlagen: {str(e)}")
                                    print(f"    🧾 Debug: main_category='{main_category}', enhanced_categories_keys={list(enhanced_categories.keys())[:5]}")
                                    # Behalte ursprÜngliche Subkategorien bei Validierungsfehlern
                            
                            
                            batch_results.append(coding_entry)
                            
                            if instance_info['total_instances'] > 1:
                                category_display = coding_entry['category']
                                if preferred_cats and category_display in preferred_cats:
                                    category_display += " 🎯"
                                print(f"        ✅ {coder.coder_id}: {category_display}")
                            else:
                                category_display = coding_entry['category']
                                if preferred_cats and category_display in preferred_cats:
                                    category_display += " 🎯"
                                print(f"  ✅ Kodierer {coder.coder_id}: {category_display}")
                        else:
                            print(f"  âš  Keine Kodierung von {coder.coder_id} erhalten")
                            
                    except Exception as e:
                        print(f"  âš  Fehler bei {coder.coder_id}: {str(e)}")
                        continue
        
        return batch_results
    
    def _extract_doc_and_chunk_id(self, segment_id: str) -> Tuple[str, str]:
        """Extrahiert Dokumentname und Chunk-ID aus segment_id."""
        parts = segment_id.split('_chunk_')
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            return segment_id, "unknown"

    def _extract_segment_sort_key(self, segment_id: str) -> Tuple[str, int]:
        """Erstellt einen SortierschlÜssel fuer die richtige Chunk-Reihenfolge."""
        try:
            doc_name, chunk_id = self._extract_doc_and_chunk_id(segment_id)
            return (doc_name, int(chunk_id) if chunk_id.isdigit() else 0)
        except Exception:
            return (segment_id, 0)
    
    async def _finalize_by_mode(self, analysis_mode: str, current_categories: Dict, 
                            deductive_categories: Dict, initial_categories: Dict) -> Dict:
        """
        KORRIGIERTE Finalisierung - gibt immer ein Dictionary zurÜck
        """
        try:
            if analysis_mode == 'inductive':
                print(f"\nℹ️ INDUCTIVE MODE Finalisierung:")
                print(f"   - Deduktive Kategorien: IGNORIERT")
                print(f"   - Induktive Kategorien: {len(current_categories)}")
                print(f"   -> Finales System: NUR {len(current_categories)} induktive Kategorien")
                return current_categories
                
            elif analysis_mode == 'grounded':
                # Im separaten Grounded Mode wurde bereits alles erledigt
                print(f"\n✅ GROUNDED MODE bereits vollstÄndig abgeschlossen")
                return current_categories
                
            elif analysis_mode == 'abductive':
                print(f"\nℹ️ ABDUCTIVE MODE Finalisierung:")
                print(f"   - Erweiterte deduktive Kategorien: {len(current_categories)}")
                return current_categories
                
            else:  # deductive oder andere
                print(f"\nℹ️ {analysis_mode.upper()} MODE Finalisierung:")
                print(f"   - Kategorien: {len(current_categories)}")
                return current_categories
                
        except Exception as e:
            print(f"Fehler in _finalize_by_mode: {str(e)}")
            # Fallback: Gebe wenigstens die aktuellen Kategorien zurÜck
            return current_categories or initial_categories or {}

    def _show_final_development_stats(self, final_categories: Dict, initial_categories: Dict, batch_count: int):
        """
        Zeigt finale Entwicklungsstatistiken
        """
        print(f"\n{'='*80}")
        print(f"🧾 KATEGORIENENTWICKLUNG ABGESCHLOSSEN")
        print(f"{'='*80}")
        
        analysis_mode = CONFIG.get('ANALYSIS_MODE', 'inductive')
        
        if analysis_mode == 'inductive':
            print(f"🧑‍💼 INDUCTIVE MODE - EigenstÄndiges induktives System:")
            print(f"   - Deduktive Kategorien: IGNORIERT")
            print(f"   - Entwickelte induktive Kategorien: {len(final_categories)}")
            print(f"   - Verarbeitete Batches: {batch_count}")
            
            # Subkategorien-Statistik
            total_subcats = sum(len(cat.subcategories) for cat in final_categories.values())
            print(f"   - Subkategorien: {total_subcats}")
            
        else:
            # Bestehende Logik fuer andere Modi - KORRIGIERT
            initial_count = len(initial_categories) if initial_categories else 0  # ✅ BUGFIX: len() hinzugefÜgt
            final_count = len(final_categories)
            new_count = final_count - initial_count  # ✅ Jetzt korrekt: int - int
            
            print(f"ℹ️ Entwicklungsbilanz:")
            print(f"   - Verarbeitete Batches: {batch_count}")
            print(f"   - Initial: {initial_count} Kategorien")
            print(f"   - Neu entwickelt: {new_count} Kategorien")
            print(f"   - Final: {final_count} Kategorien")
            
            # Subkategorien-Statistik
            total_subcats = sum(len(cat.subcategories) for cat in final_categories.values())
            print(f"   - Subkategorien: {total_subcats}")

        if (hasattr(self, 'inductive_coder') and 
            self.inductive_coder and 
            hasattr(self.inductive_coder, 'theoretical_saturation_history') and
            self.inductive_coder.theoretical_saturation_history):
            
            final_saturation = self.inductive_coder.theoretical_saturation_history[-1]
            print(f"\n🎯 Finale SÄttigung:")
            print(f"   - Theoretische SÄttigung: {final_saturation['theoretical_saturation']:.1%}")
            print(f"   - KategorienqualitÄt: {final_saturation['category_quality']:.1%}")
            print(f"   - DiversitÄt: {final_saturation['category_diversity']:.1%}")
        
        if (hasattr(self, 'inductive_coder') and 
            self.inductive_coder and 
            hasattr(self.inductive_coder, 'category_development_phases') and
            self.inductive_coder.category_development_phases):
            
            print(f"\n🧾 Entwicklungsphasen:")
            for phase in self.inductive_coder.category_development_phases:
                print(f"   Batch {phase['batch']}: +{phase['new_categories']} -> {phase['total_categories']} total")
    
    async def _save_grounded_checkpoint(self):
        """Speichere Grounded Mode Checkpoint zwischen Batches"""
        try:
            checkpoint_path = os.path.join(self.config['OUTPUT_DIR'], 'grounded_checkpoint.json')
            checkpoint_data = {
                'subcodes': self.grounded_subcodes_collection,
                'keywords': list(set(self.grounded_keywords_collection)),
                'batch_history': self.grounded_batch_history,
                'saturation_counter': self.grounded_saturation_counter,
                'segment_analyses_count': len(self.grounded_segment_analyses),
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                
            print(f"ðŸ’¾ Grounded Checkpoint gespeichert: {len(self.grounded_subcodes_collection)} Subcodes")
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern des Grounded Checkpoints: {str(e)}")

    async def _load_grounded_checkpoint(self):
        """Lade Grounded Mode Checkpoint falls vorhanden"""
        try:
            checkpoint_path = os.path.join(self.config['OUTPUT_DIR'], 'grounded_checkpoint.json')
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                self.grounded_subcodes_collection = checkpoint_data.get('subcodes', [])
                self.grounded_keywords_collection = checkpoint_data.get('keywords', [])
                self.grounded_batch_history = checkpoint_data.get('batch_history', [])
                self.grounded_saturation_counter = checkpoint_data.get('saturation_counter', 0)
                
                print(f"ðŸ’¾ Grounded Checkpoint geladen: {len(self.grounded_subcodes_collection)} Subcodes")
                print(f"   - Keywords: {len(self.grounded_keywords_collection)}")
                print(f"   - Batch-Historie: {len(self.grounded_batch_history)} EintrÄge")
                return True
        except Exception as e:
            print(f"❌ Fehler beim Laden des Grounded Checkpoints: {str(e)}")
        
        return False
    
    async def analyze_material(self, 
                            chunks: Dict[str, List[str]], 
                            initial_categories: Dict,
                            skip_inductive: bool = False,
                            batch_size: Optional[int] = None) -> Tuple[Dict, List]:
        """
        KORRIGIERTE Hauptanalyse mit besserer Fehlerbehandlung
        """
        try:
            token_counter.reset_session() # Token-Session zurÜcksetzen fuer neue Analyse

            self.escape_handler.start_monitoring()
            self.start_time = datetime.now()
            print(f"\nAnalyse gestartet um {self.start_time.strftime('%H:%M:%S')}")

            # Berechne _total_segments ZUERST
            all_segments = self._prepare_segments(chunks)
            self._total_segments = len(all_segments)
            
            # Speichere chunks als Instanzvariable
            self.chunks = chunks
            
            analysis_mode = CONFIG.get('ANALYSIS_MODE', 'inductive')
            
            # Reset Tracking-Variablen
            self.coding_results = []
            self.processed_segments = set()
            
            if batch_size is None:
                batch_size = CONFIG.get('BATCH_SIZE', 5)
            
            total_segments = len(all_segments)
            print(f"Verarbeite {total_segments} Segmente mit Batch-GrÖẞe {batch_size}...")
            self.history.log_analysis_start(total_segments, len(initial_categories))

            # GROUNDED MODE: Spezielle Behandlung
            if analysis_mode == 'grounded':
                result = await self._analyze_grounded_mode(
                    chunks, initial_categories, all_segments, batch_size
                )
            else:
                # Normale Modi (inductive, abductive, deductive)
                result = await self._analyze_normal_modes(
                    chunks, initial_categories, all_segments, skip_inductive, batch_size, analysis_mode
                )
            
            # KORRIGIERT: PrÜfe ob result ein Tupel ist
            if result is None:
                print("❌ Warnung: Analyse-Methode gab None zurÜck")
                return initial_categories, []
            
            if not isinstance(result, tuple) or len(result) != 2:
                print("❌ Warnung: Analyse-Methode gab kein gÜltiges Tupel zurÜck")
                return initial_categories, []
            
            final_categories, coding_results = result
            
            # Stoppe Escape-Handler
            self.escape_handler.stop_monitoring()
            self.end_time = datetime.now()

            return final_categories, coding_results
                    
        except Exception as e:
            self.end_time = datetime.now()
            print(f"Fehler in der Analyse: {str(e)}")
            traceback.print_exc()
            if hasattr(self, 'escape_handler'):
                self.escape_handler.stop_monitoring()
            raise

    async def _analyze_normal_modes(self, 
                                chunks: Dict[str, List[str]], 
                                initial_categories: Dict,
                                all_segments: List,
                                skip_inductive: bool,
                                batch_size: int,
                                analysis_mode: str) -> Tuple[Dict, List]:
        """
        Analysiert normale Modi (inductive, abductive, deductive)
        KORRIGIERT: Gibt immer ein Tupel zurÜck
        """
        
        # Kategoriensystem-Behandlung
        if analysis_mode == 'inductive':
            print(f"\nℹ️ INDUCTIVE MODE: Entwickle komplett neues induktives Kategoriensystem")
            current_categories = {}  # Leeres induktives System
            deductive_categories = {}  # LEER im inductive mode!
        elif analysis_mode == 'abductive':
            print(f"\nℹ️ ABDUCTIVE MODE: Erweitere deduktive Kategorien um Subkategorien")
            current_categories = initial_categories.copy()
            deductive_categories = initial_categories.copy()
        else:  # deductive
            current_categories = initial_categories.copy()
            deductive_categories = initial_categories.copy()

        if batch_size is None:
            batch_size = CONFIG.get('BATCH_SIZE', 5)
        
        total_segments = len(all_segments)
        print(f"Verarbeite {total_segments} Segmente mit Batch-GrÖẞe {batch_size}...")

        # Initialisiere ImprovedSaturationController
        saturation_controller = ImprovedSaturationController(analysis_mode)
        
        # HAUPTSCHLEIFE
        batch_count = 0
        use_context = CONFIG.get('CODE_WITH_CONTEXT', False)
        
        while True:
            # Escape-PrÜfung
            if self.check_escape_abort():
                print("\n🏁 Abbruch durch Benutzer erkannt...")
                await self._export_intermediate_results(chunks, current_categories, deductive_categories, initial_categories)
                return current_categories, self.coding_results
            
            batch = await self._get_next_batch(all_segments, batch_size)
            if not batch:
                break
                
            batch_count += 1
            material_percentage = (len(self.processed_segments) / total_segments) * 100
            
            print(f"\n{'='*60}")
            print(f"🧾 BATCH {batch_count}: {len(batch)} Segmente")
            print(f"ℹ️ Material verarbeitet: {material_percentage:.1f}%")
            print(f"{'='*60}")
            
            batch_start = time.time()
            
            try:
                # 1. ALLGEMEINE RELEVANZPRÜFUNG
                print(f"\n🕵️ Schritt 1: Erweiterte RelevanzprÜfung fuer Forschungsfrage...")

                # FIX: Escape-PrÜfung vor RelevanzprÜfung
                if self.check_escape_abort():
                    print("\n🏁 Abbruch vor RelevanzprÜfung erkannt...")
                    await self._export_intermediate_results(chunks, current_categories, deductive_categories, initial_categories)
                    return current_categories, self.coding_results
                
                if analysis_mode == 'deductive':
                    # FIX: Erweiterte RelevanzprÜfung fuer deduktiven Modus
                    extended_relevance_results = await self.relevance_checker.check_relevance_with_category_preselection(
                        batch, current_categories, analysis_mode
                    )
                    
                    # Filtere relevante Segmente und sammle Kategorie-PrÄferenzen
                    generally_relevant_batch = []
                    category_preselections = {}  # FIX: Neue Variable fuer Kategorie-PrÄferenzen
                    
                    for segment_id, text in batch:
                        result = extended_relevance_results.get(segment_id, {})
                        if result.get('is_relevant', False):
                            generally_relevant_batch.append((segment_id, text))
                            # FIX: Speichere Kategorie-PrÄferenzen fuer spaeteren Gebrauch
                            category_preselections[segment_id] = {
                                'preferred_categories': result.get('preferred_categories', []),
                                'relevance_scores': result.get('relevance_scores', {}),
                                'reasoning': result.get('reasoning', '')
                            }
                    
                    print(f"🧾 Erweiterte Relevanz: {len(generally_relevant_batch)} von {len(batch)} Segmenten relevant")
                    if category_preselections:
                        preselection_stats = {}
                        for prefs in category_preselections.values():
                            for cat in prefs['preferred_categories']:
                                preselection_stats[cat] = preselection_stats.get(cat, 0) + 1
                        print(f"🎯 Kategorie-PrÄferenzen: {preselection_stats}")
                        
                else:
                    # FIX: Standard-RelevanzprÜfung fuer andere Modi (unverÄndert)
                    general_relevance_results = await self.relevance_checker.check_relevance_batch(batch)
                    generally_relevant_batch = [
                        (segment_id, text) for segment_id, text in batch 
                        if general_relevance_results.get(segment_id, False)
                    ]
                    category_preselections = {}  # FIX: Leer fuer andere Modi
                    print(f"🧾 Allgemeine Relevanz: {len(generally_relevant_batch)} von {len(batch)} Segmente relevant")
                
                # Markiere alle Segmente als verarbeitet
                self.processed_segments.update(sid for sid, _ in batch)

                # FIX: Escape-PrÜfung nach RelevanzprÜfung
                if self.check_escape_abort():
                    print("\n🏁 Abbruch nach RelevanzprÜfung erkannt...")
                    await self._export_intermediate_results(chunks, current_categories, deductive_categories, initial_categories)
                    return current_categories, self.coding_results
                
                # 2. INDUKTIVE KATEGORIENENTWICKLUNG
                if not skip_inductive and generally_relevant_batch:
                    print(f"\n🕵️ NÄchster Schritt: Induktive Kategorienentwicklung...")
                    
                    # FIX: Escape-PrÜfung vor Kodierung
                    if self.check_escape_abort():
                        print("\n🏁 Abbruch vor Kodierung erkannt...")
                        await self._export_intermediate_results(chunks, current_categories, deductive_categories, initial_categories)
                        return current_categories, self.coding_results
                    
                    if analysis_mode in ['inductive', 'abductive']:
                        # Standard induktive Kategorienentwicklung
                        new_categories = await self._process_batch_inductively(
                            generally_relevant_batch, 
                            current_categories
                        )
                        
                        if new_categories:
                            before_count = len(current_categories)
                            # FIX: ZÄhle auch die Subkategorien fuer bessere Berichterstattung
                            before_subcategories = sum(len(cat.subcategories) for cat in current_categories.values())
                            
                            # Kategorien integrieren
                            current_categories = self._merge_category_systems(
                                current_categories,
                                new_categories
                            )
                            
                            # FIX: Aktualisiere Instanz-Attribut fuer Validierung
                            self.current_categories = current_categories

                            added_count = len(current_categories) - before_count
                            # FIX: ZÄhle auch die neuen Subkategorien
                            after_subcategories = sum(len(cat.subcategories) for cat in current_categories.values())
                            added_subcategories = after_subcategories - before_subcategories
                            
                            # FIX: Bessere Ausgabe je nach Analysemodus
                            if analysis_mode == 'abductive':
                                if added_count > 0:
                                    print(f"✅ {added_count} neue Hauptkategorien integriert")
                                if added_subcategories > 0:
                                    print(f"✅ {added_subcategories} neue Subkategorien integriert")
                                if added_count == 0 and added_subcategories == 0:
                                    print("✅ 0 neue Kategorien integriert (wie erwartet im abduktiven Modus)")
                            else:
                                print(f"✅ {added_count} neue Kategorien integriert")
                                if added_subcategories > 0:
                                    print(f"   🔀 ZusÄtzlich {added_subcategories} neue Subkategorien")
                            
                            # Aktualisiere ALLE Kodierer
                            for coder in self.deductive_coders:
                                await coder.update_category_system(current_categories)
                            
                            # FIX: Reset nur bei tatsÄchlichen Änderungen
                            if added_count > 0 or added_subcategories > 0:
                                saturation_controller.reset_stability_counter()
                        else:
                            saturation_controller.increment_stability_counter()

                    # FIX: Escape-PrÜfung nach Kodierung
                    if self.check_escape_abort():
                        print("\n🏁 Abbruch nach Kodierung erkannt...")
                        await self._export_intermediate_results(chunks, current_categories, deductive_categories, initial_categories)
                        return current_categories, self.coding_results
                    
                # 3. DEDUKTIVE KODIERUNG
                print(f"\n📝 NÄchster Schritt: Deduktive Kodierung aller {len(batch)} Segmente...")

                # FIX: Escape-PrÜfung vor Kodierung
                if self.check_escape_abort():
                    print("\n🏁 Abbruch vor Kodierung erkannt...")
                    await self._export_intermediate_results(chunks, current_categories, deductive_categories, initial_categories)
                    return current_categories, self.coding_results
                
                # Bestimme Kodiersystem je nach Modus
                if analysis_mode == 'inductive':
                    if len(current_categories) == 0:
                        coding_categories = {}
                        print(f"   🔀 Inductive Mode: Keine induktiven Kategorien -> 'Nicht kodiert'")
                    else:
                        coding_categories = current_categories
                        print(f"   🔀 Inductive Mode: Verwende {len(current_categories)} induktive Kategorien")
                elif analysis_mode == 'grounded':
                    # FIX: Im grounded mode nur rein induktive Kategorien verwenden
                    grounded_categories = {}
                    for name, cat in current_categories.items():
                        if name not in DEDUKTIVE_KATEGORIEN:
                            grounded_categories[name] = cat
                    
                    coding_categories = grounded_categories
                    print(f"   🔀 Grounded Mode: Verwende {len(grounded_categories)} rein induktive Kategorien")
                    print(f"   ❌ Ausgeschlossen: {len(current_categories) - len(grounded_categories)} deduktive Kategorien")
                else:
                    coding_categories = current_categories
                
                # FÜhre Kodierung durch
                if use_context:
                    batch_results = await self._code_batch_with_context(
                        batch, 
                        coding_categories,
                        category_preselections=category_preselections  # FIX: Neue Parameter
                    )
                else:
                    batch_results = await self._code_batch_deductively(
                        batch, 
                        coding_categories,
                        category_preselections=category_preselections  # FIX: Neue Parameter
                    )
            
                self.coding_results.extend(batch_results)
                
                # 4. SÄttigungsprÜfung
                batch_time = time.time() - batch_start
                material_percentage = (len(self.processed_segments) / total_segments) * 100
                total_batches = len(all_segments) / batch_size

                # Normale SÄttigungsprÜfung
                saturation_status = saturation_controller.assess_saturation(
                    current_categories=current_categories,
                    material_percentage=material_percentage,
                    batch_count=batch_count,
                    total_segments=self._total_segments
                )
            
                print(f"\n🧾 SÄttigungsstatus:")
                print(f"   🎯 Theoretische SÄttigung: {saturation_status['theoretical_saturation']:.1%}")
                print(f"   ℹ️ Materialabdeckung: {saturation_status['material_coverage']:.1%}")
                
                if saturation_status['is_saturated']:
                    print(f"\n🎯 SÄTTIGUNG ERREICHT nach {batch_count} Batches!")
                    break
                
                # Fortschrittsinfo
                print(f"\nℹ️ Fortschritt:")
                print(f"   - Verarbeitete Segmente: {len(self.processed_segments)}/{total_segments}")
                print(f"   - Aktuelle Kategorien: {len(current_categories)}")
                print(f"   - Kodierungen: {len(self.coding_results)}")
                print(f"   - Batch-Zeit: {batch_time:.2f}s")
                
            except Exception as e:
                print(f"Fehler bei Batch {batch_count}: {str(e)}")
                traceback.print_exc()
                continue

        # Finalisierung
        print(f"\n🏁 FINALISIERUNG ({analysis_mode.upper()} MODE):")

        final_categories = await self._finalize_by_mode(
            analysis_mode, current_categories, deductive_categories, initial_categories
        )

        # FIX: Escape-PrÜfung
        if self.check_escape_abort():
            print("\n🏁 Abbruch nach Kodierungsverarbeitung erkannt...")
            await self._export_intermediate_results(chunks, current_categories, deductive_categories, initial_categories)
            return current_categories, self.coding_results
        
        
        # Zeige finale Statistiken
        self._show_final_development_stats(final_categories, initial_categories, batch_count)
        
        # KORRIGIERT: Stelle sicher, dass immer ein Tupel zurÜckgegeben wird
        return final_categories, self.coding_results
    
    async def _analyze_grounded_mode(self, chunks: Dict[str, List[str]], initial_categories: Dict, 
                                all_segments: List, batch_size: int) -> Tuple[Dict, List]:
        """
        NEUE METHODE: Separate Grounded Mode Analyse
        """
        print("\nℹ️ GROUNDED MODE: Starte spezielle Subcode-Sammlung")
        
        if batch_size is None:
            batch_size = CONFIG.get('BATCH_SIZE', 5)
        
        # Initialisiere Grounded-spezifische Variablen
        self.grounded_subcodes_collection = []
        self.grounded_segment_analyses = []
        self.grounded_keywords_collection = []
        self.grounded_batch_history = []
        self.grounded_saturation_counter = 0
        
        batch_count = 0
        use_context = CONFIG.get('CODE_WITH_CONTEXT', False)
        
        # PHASE 1: NUR SUBCODE-SAMMLUNG (KEINE KODIERUNG)
        while True:
            if self.check_escape_abort():
                print("\n🏁 Abbruch durch Benutzer erkannt...")
                break
            
            batch = await self._get_next_batch(all_segments, batch_size)
            if not batch:
                break
                
            batch_count += 1
            material_percentage = (len(self.processed_segments) / len(all_segments)) * 100
            
            print(f"\n{'='*60}")
            print(f"🧾 GROUNDED BATCH {batch_count}: {len(batch)} Segmente (NUR SUBCODE-SAMMLUNG)")
            print(f"ℹ️ Material verarbeitet: {material_percentage:.1f}%")
            print(f"{'='*60}")
            
            # 1. RelevanzprÜfung
            general_relevance_results = await self.relevance_checker.check_relevance_batch(batch)
            generally_relevant_batch = [
                (segment_id, text) for segment_id, text in batch 
                if general_relevance_results.get(segment_id, False)
            ]
            
            # Markiere Segmente als verarbeitet
            self.processed_segments.update(sid for sid, _ in batch)
            
            # 2. NUR SUBCODE-SAMMLUNG (KEINE KODIERUNG!)
            if generally_relevant_batch:
                relevant_texts = [text for _, text in generally_relevant_batch]
                
                # Grounded-Analyse fuer Subcodes
                grounded_analysis = await self.inductive_coder.analyze_grounded_batch(
                    segments=relevant_texts,
                    material_percentage=material_percentage
                )
                
                # Sammle Subcodes zentral
                self._collect_grounded_subcodes(grounded_analysis, batch_count)
            
            # 3. SÄttigungsprÜfung (nur fuer Subcode-Sammlung)
            if await self._assess_grounded_saturation(batch_count, len(all_segments) / batch_size):
                print(f"\n🏁 GROUNDED SUBCODE-SAMMLUNG abgeschlossen nach {batch_count} Batches!")
                break
        
        print(f"\n🎯 GROUNDED PHASE 1 ABGESCHLOSSEN:")
        print(f"   - Gesammelte Subcodes: {len(self.grounded_subcodes_collection)}")
        print(f"   - Segment-Analysen: {len(self.grounded_segment_analyses)}")
        print(f"   - Keywords: {len(self.grounded_keywords_collection)}")
        
        # PHASE 2: HAUPTKATEGORIEN GENERIEREN
        if len(self.grounded_subcodes_collection) >= 5:
            print(f"\n🕵️ PHASE 2: Generiere Hauptkategorien aus Subcodes...")
            
            # Übergebe Subcodes an InductiveCoder
            self.inductive_coder.collected_subcodes = self.grounded_subcodes_collection
            self.inductive_coder.grounded_segment_analyses = self.grounded_segment_analyses
            
            # Generiere Hauptkategorien
            grounded_categories = await self.inductive_coder._generate_main_categories_from_subcodes(initial_categories)
            
            if grounded_categories:
                print(f"✅ {len(grounded_categories)} Hauptkategorien generiert")
                
                # Aktualisiere alle Kodierer mit den neuen Kategorien
                for coder in self.deductive_coders:
                    await coder.update_category_system(grounded_categories)
                
                # PHASE 3: KODIERUNG MIT GROUNDED KATEGORIEN
                print(f"\n📝 PHASE 3: Kodiere alle Segmente mit Grounded-Kategorien...")
                coding_results = await self._code_all_segments_with_grounded_categories(
                    all_segments, grounded_categories, use_context
                )
                
                self.coding_results = coding_results
                return grounded_categories, coding_results
            else:
                print("⚠️ Keine Hauptkategorien generiert - verwende initiale Kategorien")
                return initial_categories, []
        else:
            print(f"❌ Zu wenige Subcodes: {len(self.grounded_subcodes_collection)} < 5")
            return initial_categories, []

    def _collect_grounded_subcodes(self, grounded_analysis: Dict, batch_number: int):
        """
        NEUE METHODE: Sammle Subcodes aus Grounded-Analyse
        """
        new_subcodes_count = 0
        
        if grounded_analysis and 'segment_analyses' in grounded_analysis:
            print(f"🔀 Verarbeite {len(grounded_analysis['segment_analyses'])} Segment-Analysen")
            
            # Speichere alle Segment-Analysen
            self.grounded_segment_analyses.extend(grounded_analysis['segment_analyses'])
            
            for segment_analysis in grounded_analysis['segment_analyses']:
                subcodes = segment_analysis.get('subcodes', [])
                
                for subcode in subcodes:
                    subcode_name = subcode.get('name', '').strip()
                    if subcode_name:
                        # PrÜfe auf Duplikate
                        existing_names = [sc['name'] for sc in self.grounded_subcodes_collection]
                        
                        if subcode_name not in existing_names:
                            # Neuer Subcode
                            subcode_data = {
                                'name': subcode_name,
                                'definition': subcode.get('definition', ''),
                                'keywords': subcode.get('keywords', []),
                                'evidence': subcode.get('evidence', []),
                                'confidence': subcode.get('confidence', 0.7),
                                'batch_number': batch_number,
                                'source_segments': [segment_analysis.get('segment_text', '')[:100]]
                            }
                            
                            self.grounded_subcodes_collection.append(subcode_data)
                            new_subcodes_count += 1
                            
                            # Sammle Keywords
                            self.grounded_keywords_collection.extend(subcode.get('keywords', []))
                            
                            print(f"    ✅ Neuer Subcode: '{subcode_name}'")
                        else:
                            # Erweitere bestehenden Subcode
                            for existing_subcode in self.grounded_subcodes_collection:
                                if existing_subcode['name'] == subcode_name:
                                    # Erweitere ohne Duplikate
                                    existing_keywords = set(existing_subcode['keywords'])
                                    new_keywords = set(subcode.get('keywords', []))
                                    existing_subcode['keywords'] = list(existing_keywords | new_keywords)
                                    
                                    existing_subcode['evidence'].extend(subcode.get('evidence', []))
                                    existing_subcode['source_segments'].append(
                                        segment_analysis.get('segment_text', '')[:100]
                                    )
                                    
                                    # Aktualisiere Konfidenz
                                    old_conf = existing_subcode.get('confidence', 0.7)
                                    new_conf = subcode.get('confidence', 0.7)
                                    existing_subcode['confidence'] = (old_conf + new_conf) / 2
                                    
                                    print(f"    ℹ️ Subcode erweitert: '{subcode_name}'")
                                    break
        
        # Aktualisiere SÄttigungszÄhler
        if new_subcodes_count == 0:
            self.grounded_saturation_counter += 1
        else:
            self.grounded_saturation_counter = 0
        
        # Speichere Batch-Historie
        batch_info = {
            'batch_number': batch_number,
            'new_subcodes': new_subcodes_count,
            'total_subcodes': len(self.grounded_subcodes_collection),
            'material_percentage': (len(self.processed_segments) / self._total_segments) * 100
        }
        self.grounded_batch_history.append(batch_info)
        
        print(f"✅ SUBCODE-SAMMLUNG BATCH {batch_number}:")
        print(f"   - Neue Subcodes: {new_subcodes_count}")
        print(f"   - Gesamt Subcodes: {len(self.grounded_subcodes_collection)}")
        print(f"   - SÄttigungs-Counter: {self.grounded_saturation_counter}")

    async def _code_all_segments_with_grounded_categories(self, all_segments: List, 
                                                        grounded_categories: Dict, 
                                                        use_context: bool) -> List[Dict]:
        """
        NEUE METHODE: Kodiere alle Segmente mit den generierten Grounded-Kategorien
        """
        print(f"📝 Kodiere {len(all_segments)} Segmente mit {len(grounded_categories)} Grounded-Kategorien")
        
        coding_results = []
        batch_size = CONFIG.get('BATCH_SIZE', 5)
        
        # Erstelle Batches fuer die Kodierung
        for i in range(0, len(all_segments), batch_size):
            batch = all_segments[i:i + batch_size]
            print(f"   Kodiere Batch {i//batch_size + 1}: {len(batch)} Segmente")
            
            if use_context:
                batch_results = await self._code_batch_with_context(batch, grounded_categories)
            else:
                batch_results = await self._code_batch_deductively(batch, grounded_categories)
            
            coding_results.extend(batch_results)
            
            # Markiere als verarbeitet (falls noch nicht geschehen)
            for segment_id, _ in batch:
                self.processed_segments.add(segment_id)
        
        print(f"✅ Kodierung abgeschlossen: {len(coding_results)} Kodierungen erstellt")
        return coding_results
    
    async def _recode_segments_with_final_categories(self, final_categories: Dict[str, CategoryDefinition], chunks: Dict[str, List[str]]) -> None:
        """
        GROUNDED MODE: Kodiere alle Segmente nachtrÄglich mit generierten Hauptkategorien
        """
        print(f"\nℹ️ GROUNDED MODE: NachtrÄgliche Kodierung mit {len(final_categories)} Kategorien")
        
        # Aktualisiere ALLE Kodierer mit finalen Kategorien
        for coder in self.deductive_coders:
            success = await coder.update_category_system(final_categories)
            if success:
                print(f"   ✅ Kodierer {coder.coder_id} erfolgreich aktualisiert")
            else:
                print(f"   ⚠️ Fehler bei Kodierer {coder.coder_id}")
        
        # Rekonstruiere alle Segmente
        all_segments_to_recode = []
        for doc_name, doc_chunks in chunks.items():
            for chunk_id, chunk_text in enumerate(doc_chunks):
                segment_id = f"{doc_name}_chunk_{chunk_id}"
                all_segments_to_recode.append((segment_id, chunk_text))
        
        print(f"🧾 Kodiere {len(all_segments_to_recode)} Segmente mit Grounded-Kategorien")
        
        # Kodiere in Batches
        new_codings = []
        batch_size = 5
        
        for i in range(0, len(all_segments_to_recode), batch_size):
            batch = all_segments_to_recode[i:i + batch_size]
            print(f"   Batch {i//batch_size + 1}: {len(batch)} Segmente")
            
            for segment_id, segment_text in batch:
                try:
                    # Kodiere mit dem ersten Kodierer (stellvertretend)
                    coding_result = await self.deductive_coders[0].code_chunk(segment_text, final_categories)
                    
                    if coding_result and coding_result.category != 'Nicht kodiert':
                        new_coding = {
                            'segment_id': segment_id,
                            'coder_id': self.deductive_coders[0].coder_id,
                            'category': coding_result.category,
                            'subcategories': list(coding_result.subcategories),
                            'confidence': coding_result.confidence,
                            'justification': f"[Grounded Theory Nachkodierung] {coding_result.justification}",
                            'text': segment_text,
                            'paraphrase': getattr(coding_result, 'paraphrase', ''),
                            'keywords': getattr(coding_result, 'keywords', ''),
                            'grounded_recoded': True,
                            'multiple_coding_instance': 1,
                            'total_coding_instances': 1,
                            'target_category': '',
                            'category_focus_used': False
                        }
                        new_codings.append(new_coding)
                    else:
                        # "Nicht kodiert" Fallback
                        new_codings.append({
                            'segment_id': segment_id,
                            'coder_id': self.deductive_coders[0].coder_id,
                            'category': 'Nicht kodiert',
                            'subcategories': [],
                            'confidence': {'total': 1.0},
                            'justification': "Nicht relevant fuer Grounded-Kategorien",
                            'text': segment_text,
                            'grounded_recoded': False
                        })
                        
                except Exception as e:
                    print(f"      ⚠️ Fehler bei {segment_id}: {str(e)}")
                    continue
        
        # KRITISCH: Ersetze coding_results komplett
        if new_codings:
            print(f"ℹ️ Ersetze {len(self.coding_results)} alte durch {len(new_codings)} neue Kodierungen")
            self.coding_results = new_codings
            
            # Statistiken
            from collections import Counter
            category_dist = Counter(coding.get('category', 'Unbekannt') for coding in new_codings)
            print(f"\nℹ️ Kategorienverteilung nach Grounded-Nachkodierung:")
            for cat, count in category_dist.most_common():
                percentage = (count / len(new_codings)) * 100
                print(f"   - {cat}: {count} ({percentage:.1f}%)")
        else:
            print(f"⚠️ Keine Nachkodierungen erstellt")
    
    def _show_grounded_mode_statistics(self):
        """
        Zeigt detaillierte Statistiken fuer den Grounded Mode
        """
        if not hasattr(self, 'collected_subcodes'):
            return
            
        print(f"\n🧾 GROUNDED MODE STATISTIKEN:")
        print(f"{'='*50}")
        
        # Subcode-Statistiken
        print(f"🔀 Subcode-Sammlung:")
        print(f"   - Gesammelte Subcodes: {len(self.collected_subcodes)}")
        print(f"   - Segment-Analysen: {len(self.grounded_segment_analyses)}")
        
        if self.collected_subcodes:
            # Konfidenz-Analyse
            confidences = [sc.get('confidence', 0) for sc in self.collected_subcodes]
            avg_confidence = sum(confidences) / len(confidences)
            print(f"   - Durchschnittliche Konfidenz: {avg_confidence:.2f}")
            
            # Keywords-Analyse
            all_keywords = []
            for sc in self.collected_subcodes:
                all_keywords.extend(sc.get('keywords', []))
            
            keyword_counter = Counter(all_keywords)
            print(f"   - Einzigartige Keywords: {len(set(all_keywords))}")
            print(f"   - Keywords gesamt: {len(all_keywords)}")
            
            # Top Keywords
            top_keywords = keyword_counter.most_common(10)
            print(f"   - Top Keywords: {', '.join([f'{kw}({count})' for kw, count in top_keywords[:5]])}")
            
            # Subcode-Batch-Verteilung
            batch_dist = Counter(sc.get('batch_number', 0) for sc in self.collected_subcodes)
            print(f"   - Verteilung Über Batches: {dict(batch_dist)}")

    def _export_grounded_mode_details(self, output_dir: str):
        """
        Exportiert detaillierte Grounded Mode Daten fuer weitere Analyse
        """
        if not hasattr(self, 'collected_subcodes') or not self.collected_subcodes:
            return
            
        try:
            # Exportiere Subcodes als JSON
            subcodes_path = os.path.join(output_dir, f"grounded_subcodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            export_data = {
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'total_subcodes': len(self.collected_subcodes),
                    'total_segment_analyses': len(self.grounded_segment_analyses),
                    'analysis_mode': 'grounded'
                },
                'subcodes': self.collected_subcodes,
                'segment_analyses': self.grounded_segment_analyses[:100]  # Nur erste 100 fuer GrÖẞe
            }
            
            with open(subcodes_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            print(f"\n🔀 Grounded Mode Details exportiert: {subcodes_path}")
            
        except Exception as e:
            print(f"❌ Fehler beim Export der Grounded Mode Details: {str(e)}")

    def check_escape_abort(self) -> bool:
        """PrÜft ob durch Escape abgebrochen werden soll"""
        return (getattr(self, '_should_abort', False) or 
                getattr(self, '_escape_abort_requested', False) or
                (hasattr(self, 'escape_handler') and self.escape_handler.should_abort()))
    
    async def _export_intermediate_results(self, chunks, current_categories, 
                                         deductive_categories, initial_categories):
        """Exportiert Zwischenergebnisse bei Abbruch"""
        try:
            if not hasattr(self, 'end_time') or self.end_time is None:
                self.end_time = datetime.now()

            print("\n🧾 Exportiere Zwischenergebnisse...")
            
            # Erstelle einen speziellen Exporter fuer Zwischenergebnisse
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            exporter = ResultsExporter(
                output_dir=CONFIG['OUTPUT_DIR'],
                attribute_labels=CONFIG['ATTRIBUTE_LABELS'],
                analysis_manager=self
            )
            
            # Speichere Zwischenkategorien
            if current_categories:
                category_manager = CategoryManager(CONFIG['OUTPUT_DIR'])
                category_manager.save_codebook(
                    categories=current_categories,
                    filename=f"codebook_intermediate_{timestamp}.json"
                )
                print(f"🔀 Zwischenkategorien gespeichert: {len(current_categories)} Kategorien")
            
            # Exportiere Zwischenkodierungen falls vorhanden
            if self.coding_results:
                print(f"🧾 Exportiere {len(self.coding_results)} Zwischenkodierungen...")
                
                # Revision Manager fuer Export
                revision_manager = CategoryRevisionManager(
                    output_dir=CONFIG['OUTPUT_DIR'],
                    config=CONFIG
                )
                
                # Berechne eine grobe ReliabilitÄt fuer Zwischenergebnisse
                reliability = 0.8  # Placeholder
                
                await exporter.export_results(
                    codings=self.coding_results,
                    reliability=reliability,
                    categories=current_categories,
                    chunks=chunks,
                    revision_manager=revision_manager,
                    export_mode="consensus",
                    original_categories=initial_categories,
                    document_summaries=getattr(self, 'document_summaries', None),
                    is_intermediate_export=True  # FIX: Kennzeichnung als Zwischenexport
                )
                
                print("✅ Zwischenergebnisse erfolgreich exportiert!")
                print(f"ℹ️ Dateien im Ordner: {CONFIG['OUTPUT_DIR']}")
                print(f"ℹ️ Export-Datei: QCA-AID_Analysis_INTERMEDIATE_{timestamp}.xlsx")
            else:
                print("❌  Keine Kodierungen zum Exportieren vorhanden")
                
        except Exception as e:
            print(f"⚠️ Fehler beim Export der Zwischenergebnisse: {str(e)}")
            import traceback
            traceback.print_exc()

    def _merge_category_systems(self, 
                            current: Dict[str, CategoryDefinition], 
                            new: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        FÜhrt bestehendes und neues Kategoriensystem zusammen.
        
        Args:
            current: Bestehendes Kategoriensystem
            new: Neue Kategorien
            
        Returns:
            Dict[str, CategoryDefinition]: ZusammengefÜhrtes System
        """
        merged = current.copy()
        
        for name, category in new.items():
            if name not in merged:
                # Komplett neue Kategorie
                merged[name] = category
                print(f"\nðŸ†• Neue Hauptkategorie hinzugefÜgt: {name}")
                print(f"   Definition: {category.definition[:100]}...")
                if category.subcategories:
                    print("   Subkategorien:")
                    for sub_name in category.subcategories.keys():
                        print(f"   - {sub_name}")
            else:
                # Bestehende Kategorie aktualisieren
                existing = merged[name]
                
                # Sammle Änderungen fuer Debug-Ausgabe
                changes = []
                
                # PrÜfe auf neue/geÄnderte Definition
                new_definition = category.definition
                if len(new_definition) > len(existing.definition):
                    changes.append("Definition aktualisiert")
                else:
                    new_definition = existing.definition
                
                # Kombiniere Beispiele
                new_examples = list(set(existing.examples) | set(category.examples))
                if len(new_examples) > len(existing.examples):
                    changes.append(f"{len(new_examples) - len(existing.examples)} neue Beispiele")
                
                # Kombiniere Regeln
                new_rules = list(set(existing.rules) | set(category.rules))
                if len(new_rules) > len(existing.rules):
                    changes.append(f"{len(new_rules) - len(existing.rules)} neue Regeln")
                
                # Neue Subkategorien
                new_subcats = {**existing.subcategories, **category.subcategories}
                if len(new_subcats) > len(existing.subcategories):
                    changes.append(f"{len(new_subcats) - len(existing.subcategories)} neue Subkategorien")
                
                # Erstelle aktualisierte Kategorie
                merged[name] = CategoryDefinition(
                    name=name,
                    definition=new_definition,
                    examples=new_examples,
                    rules=new_rules,
                    subcategories=new_subcats,
                    added_date=existing.added_date,
                    modified_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                if changes:
                    print(f"\n🔀 Kategorie '{name}' aktualisiert:")
                    for change in changes:
                        print(f"   - {change}")
        
        return merged

    

    def _prepare_segments(self, chunks: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """
        Bereitet die Segmente fuer die Analyse vor.
        
        Args:
            chunks: Dictionary mit Dokumenten-Chunks
            
        Returns:
            List[Tuple[str, str]]: Liste von (segment_id, text) Tupeln
        """
        segments = []
        for doc_name, doc_chunks in chunks.items():
            for chunk_idx, chunk in enumerate(doc_chunks):
                segment_id = f"{doc_name}_chunk_{chunk_idx}"
                segments.append((segment_id, chunk))
        return segments


    def _find_similar_category(self, 
                                category: CategoryDefinition,
                                existing_categories: Dict[str, CategoryDefinition]) -> Optional[str]:
        """
        Findet Ähnliche existierende Kategorien basierend auf Namen und Definition.
        
        Args:
            category: Zu prÜfende Kategorie
            existing_categories: Bestehendes Kategoriensystem
            
        Returns:
            Optional[str]: Name der Ähnlichsten Kategorie oder None
        """
        try:
            best_match = None
            highest_similarity = 0.0
            
            for existing_name, existing_cat in existing_categories.items():
                # Berechne Ähnlichkeit basierend auf verschiedenen Faktoren
                
                # 1. Name-Ähnlichkeit (gewichtet: 0.3)
                name_similarity = self.inductive_coder._calculate_text_similarity(
                    category.name.lower(),
                    existing_name.lower()
                ) * 0.3
                
                # 2. Definitions-Ähnlichkeit (gewichtet: 0.5)
                definition_similarity = self.inductive_coder._calculate_text_similarity(
                    category.definition,
                    existing_cat.definition
                ) * 0.5
                
                # 3. Subkategorien-Überlappung (gewichtet: 0.2)
                subcats1 = set(category.subcategories.keys())
                subcats2 = set(existing_cat.subcategories.keys())
                if subcats1 and subcats2:
                    subcat_overlap = len(subcats1 & subcats2) / len(subcats1 | subcats2)
                else:
                    subcat_overlap = 0
                subcat_similarity = subcat_overlap * 0.2
                
                # GesamtÄhnlichkeit
                total_similarity = name_similarity + definition_similarity + subcat_similarity
                
                # Debug-Ausgabe fuer hohe Ähnlichkeiten
                if total_similarity > 0.5:
                    print(f"\nÄhnlichkeitsprÜfung fuer '{category.name}' und '{existing_name}':")
                    print(f"- Name-Ähnlichkeit: {name_similarity:.2f}")
                    print(f"- Definitions-Ähnlichkeit: {definition_similarity:.2f}")
                    print(f"- Subkategorien-Überlappung: {subcat_similarity:.2f}")
                    print(f"- Gesamt: {total_similarity:.2f}")
                
                # Update beste Übereinstimmung
                if total_similarity > highest_similarity:
                    highest_similarity = total_similarity
                    best_match = existing_name
            
            # Nur zurÜckgeben wenn Ähnlichkeit hoch genug
            if highest_similarity > 0.7:  # Schwellenwert fuer Ähnlichkeit
                print(f"\nâš  Hohe Ähnlichkeit ({highest_similarity:.2f}) gefunden:")
                print(f"- Neue Kategorie: {category.name}")
                print(f"- Existierende Kategorie: {best_match}")
                return best_match
                
            return None
            
        except Exception as e:
            print(f"Fehler bei ÄhnlichkeitsprÜfung: {str(e)}")
            return None

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Berechnet die Ähnlichkeit zwischen zwei Texten mit Caching."""
        cache_key = f"{hash(text1)}_{hash(text2)}"
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
            
        # Konvertiere Texte zu Sets von WÖrtern
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Berechne Jaccard-Ähnlichkeit
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Cache das Ergebnis
        self.similarity_cache[cache_key] = similarity
        self.validation_stats['similarity_calculations'] += 1
        
        return similarity
   
   
    def _prepare_segments(self, chunks: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """
        Bereitet die Segmente fuer die Analyse vor.
        
        Args:
            chunks: Dictionary mit Dokumenten-Chunks
            
        Returns:
            List[Tuple[str, str]]: Liste von (segment_id, text) Tupeln
        """
        segments = []
        for doc_name, doc_chunks in chunks.items():
            for chunk_idx, chunk in enumerate(doc_chunks):
                segment_id = f"{doc_name}_chunk_{chunk_idx}"
                segments.append((segment_id, chunk))
        return segments

    def _update_tracking(self,
                        batch_results: List[Dict],
                        processing_time: float,
                        batch_size: int) -> None:
        """
        Aktualisiert die Performance-Metriken.
        
        Args:
            batch_results: Ergebnisse des Batches
            processing_time: Verarbeitungszeit
            batch_size: GrÖẞe des Batches
        """
        self.coding_results.extend(batch_results)
        self.performance_metrics['batch_processing_times'].append(processing_time)
        
        # Berechne durchschnittliche Zeit pro Segment
        avg_time_per_segment = processing_time / batch_size
        self.performance_metrics['coding_times'].append(avg_time_per_segment)

    def _log_iteration_status(self,
                            material_percentage: float,
                            saturation_metrics: Dict,
                            num_results: int) -> None:
        """
        Protokolliert den Status der aktuellen Iteration.
        
        Args:
            material_percentage: Prozentsatz des verarbeiteten Materials
            saturation_metrics: Metriken der SÄttigungsprÜfung
            num_results: Anzahl der Kodierungen
        """
        try:
            # Erstelle Status-Dictionary
            status = {
                'timestamp': datetime.now().isoformat(),
                'material_processed': material_percentage,
                'saturation_metrics': saturation_metrics,
                'results_count': num_results,
                'processing_time': self.performance_metrics['batch_processing_times'][-1] if self.performance_metrics['batch_processing_times'] else 0
            }
            
            # FÜge Status zum Log hinzu
            self.analysis_log.append(status)
            
            # Debug-Ausgabe fuer wichtige Metriken
            print("\nIterations-Status:")
            print(f"- Material verarbeitet: {material_percentage:.1f}%")
            print(f"- Neue Kodierungen: {num_results}")
            print(f"- Verarbeitungszeit: {status['processing_time']:.2f}s")
            if saturation_metrics:
                print("- SÄttigungsmetriken:")
                for key, value in saturation_metrics.items():
                    print(f"  - {key}: {value}")
        except Exception as e:
            print(f"Warnung: Fehler beim Logging des Iterationsstatus: {str(e)}")
            # Fehler beim Logging sollte die Hauptanalyse nicht unterbrechen

    def _finalize_analysis(self,
                          final_categories: Dict,
                          initial_categories: Dict) -> Tuple[Dict, List]:
        """
        Schlieẞt die Analyse ab und bereitet die Ergebnisse vor.
        
        Args:
            final_categories: Finales Kategoriensystem
            initial_categories: Initiales Kategoriensystem
            
        Returns:
            Tuple[Dict, List]: (Finales Kategoriensystem, Kodierungsergebnisse)
        """
        # Berechne finale Statistiken
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        avg_time_per_segment = total_time / len(self.processed_segments)
        
        print("\nAnalyse abgeschlossen:")
        print(f"- Gesamtzeit: {total_time:.2f}s")
        print(f"- Durchschnittliche Zeit pro Segment: {avg_time_per_segment:.2f}s")
        print(f"- Verarbeitete Segmente: {len(self.processed_segments)}")
        print(f"- Finale Kategorien: {len(final_categories)}")
        print(f"- Gesamtanzahl Kodierungen: {len(self.coding_results)}")
        
        return final_categories, self.coding_results

    def get_analysis_report(self) -> Dict:
        """Erstellt einen detaillierten Analysebericht."""
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_segments': len(self.processed_segments),
            'total_codings': len(self.coding_results),
            'performance_metrics': {
                'avg_batch_time': (
                    sum(self.performance_metrics['batch_processing_times']) / 
                    len(self.performance_metrics['batch_processing_times'])
                ) if self.performance_metrics['batch_processing_times'] else 0,
                'total_batches': len(self.performance_metrics['batch_processing_times']),
                'coding_distribution': self._get_coding_distribution()
            },
            'coding_summary': self._get_coding_summary()
        }
    
    def _get_coding_distribution(self) -> Dict:
        """Analysiert die Verteilung der Kodierungen."""
        if not self.coding_results:
            return {}
            
        coders = Counter(coding['coder_id'] for coding in self.coding_results)
        categories = Counter(coding['category'] for coding in self.coding_results)
        
        return {
            'coders': dict(coders),
            'categories': dict(categories)
        }
    
    def _get_coding_summary(self) -> Dict:
        """Erstellt eine Zusammenfassung der Kodierungen."""
        if not self.coding_results:
            return {}
            
        return {
            'total_segments_coded': len(set(coding['segment_id'] for coding in self.coding_results)),
            'avg_codings_per_segment': len(self.coding_results) / len(self.processed_segments) if self.processed_segments else 0,
            'unique_categories': len(set(coding['category'] for coding in self.coding_results))
        }

    def get_progress_report(self) -> Dict:
        """
        Erstellt einen detaillierten Fortschrittsbericht fuer die laufende Analyse.
        
        Returns:
            Dict: Fortschrittsbericht mit aktuellen Metriken und Status
        """
        current_time = datetime.now()
        elapsed_time = (current_time - self.start_time).total_seconds()
        
        # Berechne durchschnittliche Verarbeitungszeiten
        avg_batch_time = statistics.mean(self.performance_metrics['batch_processing_times']) if self.performance_metrics['batch_processing_times'] else 0
        avg_coding_time = statistics.mean(self.performance_metrics['coding_times']) if self.performance_metrics['coding_times'] else 0
        
                
        # Berechne Verarbeitungsstatistiken
        total_segments = len(self.processed_segments)
        segments_per_hour = (total_segments / elapsed_time) * 3600 if elapsed_time > 0 else 0
        
        return {
            'progress': {
                'processed_segments': total_segments,
                'total_codings': len(self.coding_results),
                'segments_per_hour': round(segments_per_hour, 2),
                'elapsed_time': round(elapsed_time, 2)
            },
            'performance': {
                'avg_batch_processing_time': round(avg_batch_time, 2),
                'avg_coding_time': round(avg_coding_time, 2),
                'last_batch_time': round(self.performance_metrics['batch_processing_times'][-1], 2) if self.performance_metrics['batch_processing_times'] else 0
            },
            'status': {
                'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'last_update': self.analysis_log[-1]['timestamp'] if self.analysis_log else None
            }
        }

