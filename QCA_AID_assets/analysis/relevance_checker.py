"""
Relevanz-Prüfung für QCA-AID
=============================
Zentrale Klasse für Relevanzprüfungen mit Caching und Batch-Verarbeitung.
"""

import json
from ..utils.tracking.token_tracker import TokenTracker, get_global_token_counter
from ..utils.llm.response import LLMResponse
from ..utils.llm.factory import LLMProviderFactory
import time
import asyncio
from typing import List, Tuple, Dict, Optional, Any

from ..core.config import CONFIG, FORSCHUNGSFRAGE, KODIERREGELN
from ..core.data_models import CategoryDefinition
from ..QCA_Prompts import QCAPrompts

# Verwende globale Token-Counter Instanz
token_counter = get_global_token_counter()


class RelevanceChecker:
    """
    Zentrale Klasse für Relevanzprüfungen mit Caching und Batch-Verarbeitung.
    Reduziert API-Calls durch Zusammenfassung mehrerer Segmente.
    """
    
    def __init__(self, model_name: str, batch_size: int = 5, temperature: float = 0.3):
        self.model_name = model_name
        self.batch_size = batch_size
        self.temperature = float(temperature)

        # Hole Provider aus CONFIG
        provider_name = CONFIG.get('MODEL_PROVIDER', 'openai').lower()  # Fallback zu OpenAI
        try:
            # Wir lassen den LLMProvider sich selbst um die Client-Initialisierung kümmern
            # Übergebe model_name für Capability-Testing
            self.llm_provider = LLMProviderFactory.create_provider(provider_name, model_name=model_name)
        except Exception as e:
            print(f"Fehler bei Provider-Initialisierung: {str(e)}")
            raise
        
        # Cache für Relevanzprüfungen
        self.relevance_cache = {}
        self.relevance_details = {}
        
        # Tracking-Metriken
        self.total_segments = 0
        self.relevant_segments = 0
        self.api_calls = 0

        # Hole Ausschlussregeln aus KODIERREGELN
        self.exclusion_rules = KODIERREGELN.get('exclusion', [])
        print("\nRelevanceChecker initialisiert:")
        print(f"- {len(self.exclusion_rules)} Ausschlussregeln geladen")

        # Hole Mehrfachkodierungsparameter aus CONFIG
        self.multiple_codings_enabled = CONFIG.get('MULTIPLE_CODINGS', True)
        self.multiple_threshold = float(CONFIG.get('MULTIPLE_CODING_THRESHOLD', 0.7))

        # Cache für Mehrfachkodierungen
        self.multiple_coding_cache = {}
        
        print("\nRelevanceChecker initialisiert:")
        print(f"- {len(self.exclusion_rules)} Ausschlussregeln geladen")
        print(f"- Mehrfachkodierung: {'Aktiviert' if self.multiple_codings_enabled else 'Deaktiviert'}")
        if self.multiple_codings_enabled:
            print(f"- Mehrfachkodierung-Schwellenwert: {self.multiple_threshold}")
        
        # Prompt-Handler
        self.prompt_handler = QCAPrompts(
            forschungsfrage=FORSCHUNGSFRAGE,
            kodierregeln=KODIERREGELN,
            deduktive_kategorien=CONFIG.get('DEDUKTIVE_KATEGORIEN', {})
        )

    def _format_segments_for_batch(self, segments: List[Tuple[str, str]]) -> str:
        """
        Formatiert Segmente für Batch-Verarbeitung in der Relevanzprüfung
        
        Args:
            segments: Liste von (segment_id, text) Tupeln
            
        Returns:
            str: Formatierte Segmente für den Prompt
        """
        formatted_segments = []
        for i, (segment_id, text) in enumerate(segments, 1):
            # Begrenze Textlänge für bessere Performance
            truncated_text = text[:800] + "..." if len(text) > 800 else text
            # Entferne problematische Zeichen
            clean_text = truncated_text.replace('\n', ' ').replace('\r', ' ').strip()
            formatted_segments.append(f"SEGMENT {i}:\n{clean_text}\n")
        
        return "\n".join(formatted_segments)
    
    async def check_multiple_category_relevance(self, segments: List[Tuple[str, str]], 
                                            categories: Dict[str, CategoryDefinition]) -> Dict[str, List[Dict]]:
        """
        PARALLELISIERTE VERSION: Prüft ob Segmente für mehrere Hauptkategorien relevant sind.
        """
        if not self.multiple_codings_enabled:
            return {}
            
        # Filtere bereits gecachte Segmente
        uncached_segments = [
            (sid, text) for sid, text in segments 
            if sid not in self.multiple_coding_cache
        ]
        
        if not uncached_segments:
            return {sid: self.multiple_coding_cache[sid] for sid, _ in segments}

        try:
            print(f"🚀 Parallele Mehrfachkodierungs-Pruefung: {len(uncached_segments)} Segmente")
            
            # Bereite Kategorien-Kontext vor
            category_descriptions = []
            for cat_name, cat_def in categories.items():
                if cat_name not in ["Nicht kodiert", "Kein Kodierkonsens"]:
                    examples_list = list(cat_def.examples) if isinstance(cat_def.examples, set) else cat_def.examples
                    category_descriptions.append({
                        'name': cat_name,
                        'definition': cat_def.definition[:200] + '...' if len(cat_def.definition) > 200 else cat_def.definition,
                        'examples': examples_list[:2] if examples_list else []
                    })

            # 🚀 PARALLEL: Hilfsfunktion für einzelnes Segment
            async def check_single_segment_multiple(segment_id: str, text: str) -> Tuple[str, List[Dict]]:
                """Prüft ein einzelnes Segment auf Mehrfachkodierung parallel."""
                try:
                    prompt = self.prompt_handler.get_multiple_category_relevance_prompt(
                        segments_text=f"SEGMENT:\n{text}",
                        category_descriptions=category_descriptions,
                        multiple_threshold=self.multiple_threshold
                    )
                    
                    token_counter.start_request()
                    
                    # Erstelle Parameter-Dict
                    response = await self.llm_provider.create_completion(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch. Antworte ausschliesslich mit einem JSON-Objekt."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        response_format={"type": "json_object"}
                     )
                    
                    llm_response = LLMResponse(response)
                    try:
                        result = json.loads(llm_response.extract_json())
                    except json.JSONDecodeError as e:
                        print(f"‼️ JSONDecodeError in check_multiple_category_relevance for segment {segment_id}: {e}")
                        print(f"‼️ Raw LLM response: {llm_response.content}")
                        print(f"⚠️ Fallback: Keine Mehrfachkodierung für dieses Segment")
                        # Fallback: Keine Mehrfachkodierung
                        return segment_id, []
                    
                    
                    token_counter.track_response(response, self.model_name)
                    
                    # Verarbeite Ergebnisse - erwarte single segment format
                    segment_result = result.get('segment_results', [{}])[0] if result.get('segment_results') else result
                    
                    # Filtere Kategorien nach Schwellenwert
                    relevant_categories = []
                    for cat in segment_result.get('relevant_categories', []):
                        if cat.get('relevance_score', 0) >= self.multiple_threshold:
                            relevant_categories.append(cat)
                    
                    return segment_id, relevant_categories
                    
                except Exception as e:
                    print(f"[WARN] Fehler bei Mehrfachkodierungs-Pruefung {segment_id}: {str(e)}")
                    return segment_id, []  # Leer = keine Mehrfachkodierung
            
            # 🚀 Erstelle Tasks für alle Segmente
            tasks = [
                check_single_segment_multiple(segment_id, text) 
                for segment_id, text in uncached_segments
            ]
            
            # 🚀 Führe alle parallel aus
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processing_time = time.time() - start_time
            
            # Verarbeite Ergebnisse und aktualisiere Cache
            multiple_coding_results = {}
            successful_checks = 0
            segments_with_multiple = 0
            
            for result in results:
                if isinstance(result, Exception):
                    print(f"⚠️ Ausnahme bei Mehrfachkodierungs-Prüfung: {result}")
                    continue
                
                segment_id, relevant_categories = result
                
                # Cache aktualisieren
                self.multiple_coding_cache[segment_id] = relevant_categories
                multiple_coding_results[segment_id] = relevant_categories
                
                successful_checks += 1
                
                # Debug-Ausgabe für Mehrfachkodierung
                if len(relevant_categories) > 1:
                    segments_with_multiple += 1
                    print(f"  🔁 Mehrfachkodierung: {segment_id}")
                    for cat in relevant_categories:
                        print(f"    - {cat['category']}: {cat['relevance_score']:.2f}")
            
            print(f"⚡ {successful_checks} Mehrfachkodierungs-Prüfungen in {processing_time:.2f}s")
            print(f"🔄 {segments_with_multiple} Segmente mit Mehrfachkodierung identifiziert")
            
            # Kombiniere mit Cache für alle ursprünglichen Segmente
            final_results = {}
            for segment_id, text in segments:
                if segment_id in multiple_coding_results:
                    final_results[segment_id] = multiple_coding_results[segment_id]
                else:
                    final_results[segment_id] = self.multiple_coding_cache.get(segment_id, [])
            
            return final_results

        except Exception as e:
            print(f"Fehler bei paralleler Mehrfachkodierungs-Prüfung: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    
    async def check_relevance_batch(self, segments: List[Tuple[str, str]]) -> Dict[str, bool]:
        """
        Prüft die Relevanz mehrerer Segmente parallel mit Batch-Verarbeitung.
        Behält die bewährte Batch-Logik bei und fügt Parallelisierung nur für große Batches hinzu.
        
        Args:
            segments: Liste von (segment_id, text) Tupeln
            
        Returns:
            Dict[str, bool]: Mapping von segment_id zu Relevanz
        """
        # Filtere bereits gecachte Segmente
        uncached_segments = [
            (sid, text) for sid, text in segments 
            if sid not in self.relevance_cache
        ]
        
        if not uncached_segments:
            return {sid: self.relevance_cache[sid] for sid, _ in segments}

        try:
            print(f"🔍 Relevanzprüfung: {len(uncached_segments)} neue Segmente")
            
            # STRATEGIE: Kleine Batches (≤5) → bewährte Batch-Methode
            #           Große Batches (>5) → Parallelisierung in Sub-Batches
            
            if len(uncached_segments) <= 5:
                # BEWÄHRTE BATCH-METHODE für kleine Gruppen
                print(f"   📦 Verwende normale Batch-Methode für {len(uncached_segments)} Segmente")
                
                # Erstelle formatierten Text für Batch-Verarbeitung
                segments_text = "\n\n=== SEGMENT BREAK ===\n\n".join(
                    f"SEGMENT {i + 1}:\n{text}" 
                    for i, (_, text) in enumerate(uncached_segments)
                )

                prompt = self.prompt_handler.get_relevance_check_prompt(
                    segments_text=segments_text,
                    exclusion_rules=self.exclusion_rules
                )
                
                token_counter.start_request()
                
                # Ein API-Call für alle Segmente
                response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                
                llm_response = LLMResponse(response)
                try:
                    results = json.loads(llm_response.extract_json())
                except json.JSONDecodeError as e:
                    print(f"‼️ JSONDecodeError in check_relevance_batch: {e}")
                    print(f"‼️ Raw LLM response: {llm_response.content}")
                    print(f"⚠️ Fallback: Markiere alle Segmente als relevant")
                    # Fallback: Alle als relevant, damit Analyse weitergehen kann
                    for segment_id, _ in uncached_segments:
                        self.relevance_cache[segment_id] = True
                    return {sid: True for sid, _ in segments}
                
                
                token_counter.track_response(response, self.model_name)
                
                # Verarbeite Ergebnisse
                relevance_results = {}
                segment_results = results.get('segment_results', [])
                
                if len(segment_results) != len(uncached_segments):
                    print(f"⚠️ Warnung: Anzahl Ergebnisse ({len(segment_results)}) != Anzahl Segmente ({len(uncached_segments)})")
                    # Fallback: Alle als relevant markieren
                    for segment_id, _ in uncached_segments:
                        relevance_results[segment_id] = True
                        self.relevance_cache[segment_id] = True
                else:
                    for i, (segment_id, _) in enumerate(uncached_segments):
                        segment_result = segment_results[i]
                        is_relevant = segment_result.get('is_relevant', True)  # Default: relevant
                        
                        # Cache-Aktualisierung
                        self.relevance_cache[segment_id] = is_relevant
                        # Erweiterte Details-Speicherung mit Begründung
                        # FIX: Vereinheitlicht auf 'reasoning' (kein 'justification' Duplikat)
                        # 'reasoning' kommt direkt vom LLM und ist die Hauptbegründung
                        self.relevance_details[segment_id] = {
                            'confidence': segment_result.get('confidence', 0.8),
                            'key_aspects': segment_result.get('key_aspects', []),
                            'reasoning': segment_result.get('reasoning', segment_result.get('justification', 'Keine Begründung verfügbar')),
                            'is_relevant': is_relevant,
                            'main_themes': segment_result.get('main_themes', []),
                            'exclusion_match': segment_result.get('exclusion_match', False)
                        }
                        
                        relevance_results[segment_id] = is_relevant
                        
                        # Tracking-Aktualisierung
                        self.total_segments += 1
                        if is_relevant:
                            self.relevant_segments += 1
                
                self.api_calls += 1
                
            else:
                # PARALLELISIERUNG für große Batches
                print(f"   🚀 Verwende Parallelisierung in Sub-Batches fuer {len(uncached_segments)} Segmente")
                
                # Teile in Sub-Batches von je 3-4 Segmenten
                sub_batch_size = 3
                sub_batches = [
                    uncached_segments[i:i + sub_batch_size] 
                    for i in range(0, len(uncached_segments), sub_batch_size)
                ]
                
                async def process_sub_batch(sub_batch: List[Tuple[str, str]]) -> Dict[str, bool]:
                    """Verarbeitet einen Sub-Batch mit der bewährten Methode."""
                    try:
                        # Verwende dieselbe Batch-Logik wie oben
                        segments_text = "\n\n=== SEGMENT BREAK ===\n\n".join(
                            f"SEGMENT {i + 1}:\n{text}" 
                            for i, (_, text) in enumerate(sub_batch)
                        )

                        prompt = self.prompt_handler.get_relevance_check_prompt(
                            segments_text=segments_text,
                            exclusion_rules=self.exclusion_rules
                        )
                        
                        token_counter.start_request()
                        
                        response = await self.llm_provider.create_completion(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=self.temperature,
                            response_format={"type": "json_object"}
                        )
                        
                        llm_response = LLMResponse(response)
                        try:
                            results = json.loads(llm_response.extract_json())
                        except json.JSONDecodeError as e:
                            print(f"‼️ JSONDecodeError in process_sub_batch: {e}")
                            print(f"‼️ Raw LLM response: {llm_response.content}")
                            print(f"⚠️ Fallback für Sub-Batch: Markiere alle als relevant")
                            # Fallback: Alle als relevant
                            return {segment_id: True for segment_id, _ in sub_batch}
                        
                        
                        token_counter.track_response(response, self.model_name)
                        
                        # Verarbeite Sub-Batch Ergebnisse
                        sub_batch_results = {}
                        segment_results = results.get('segment_results', [])
                        
                        if len(segment_results) != len(sub_batch):
                            # Fallback bei Mismatch
                            for segment_id, _ in sub_batch:
                                sub_batch_results[segment_id] = True
                        else:
                            for i, (segment_id, _) in enumerate(sub_batch):
                                segment_result = segment_results[i]
                                is_relevant = segment_result.get('is_relevant', True)
                                sub_batch_results[segment_id] = is_relevant
                                
                                # Erweiterte Details-Speicherung mit Begründung
                                # FIX: Vereinheitlicht auf 'reasoning' (kein 'justification' Duplikat)
                                self.relevance_details[segment_id] = {
                                    'confidence': segment_result.get('confidence', 0.8),
                                    'key_aspects': segment_result.get('key_aspects', []),
                                    'reasoning': segment_result.get('reasoning', segment_result.get('justification', 'Keine Begründung verfügbar')),
                                    'is_relevant': is_relevant,
                                    'main_themes': segment_result.get('main_themes', []),
                                    'exclusion_match': segment_result.get('exclusion_match', False)
                                }
                        
                        return sub_batch_results
                        
                    except Exception as e:
                        print(f"⚠️ Fehler in Sub-Batch: {str(e)}")
                        # Fallback: Alle als relevant markieren
                        return {segment_id: True for segment_id, _ in sub_batch}
                
                # Führe alle Sub-Batches parallel aus
                start_time = time.time()
                tasks = [process_sub_batch(sub_batch) for sub_batch in sub_batches]
                sub_batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                processing_time = time.time() - start_time
                
                # Kombiniere Ergebnisse
                relevance_results = {}
                successful_batches = 0
                
                for result in sub_batch_results:
                    if isinstance(result, Exception):
                        print(f"⚠️ Sub-Batch Ausnahme: {result}")
                        continue
                    
                    if isinstance(result, dict):
                        relevance_results.update(result)
                        successful_batches += 1
                
                # Aktualisiere Cache und Tracking
                for segment_id, is_relevant in relevance_results.items():
                    self.relevance_cache[segment_id] = is_relevant
                    self.total_segments += 1
                    if is_relevant:
                        self.relevant_segments += 1
                
                # API-Calls = Anzahl der Sub-Batches
                self.api_calls += successful_batches
                
                print(f"   ⚡ {successful_batches} Sub-Batches in {processing_time:.2f}s verarbeitet")
            
            # Debug-Ausgabe der Ergebnisse
            relevant_count = sum(1 for is_relevant in relevance_results.values() if is_relevant)
            print(f"   ℹ️ Relevanz-Ergebnisse: {relevant_count}/{len(relevance_results)} als relevant eingestuft")
            
            # Kombiniere mit bereits gecachten Ergebnissen für finale Antwort
            final_results = {}
            for segment_id, text in segments:
                if segment_id in relevance_results:
                    final_results[segment_id] = relevance_results[segment_id]
                else:
                    final_results[segment_id] = self.relevance_cache.get(segment_id, True)
            
            return final_results

        except Exception as e:
            print(f"Fehler bei paralleler Relevanzprüfung: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback: Alle als relevant markieren bei Fehler
            return {sid: True for sid, _ in segments}
    

    async def check_relevance_with_category_preselection(self, segments: List[Tuple[str, str]], 
                                                        categories: Dict[str, CategoryDefinition] = None,
                                                        mode: str = 'deductive') -> Dict[str, Dict]:
        """
        Erweiterte Relevanzprüfung mit Hauptkategorie-Vorauswahl (nur für deduktiven Modus)
        
        Returns:
            Dict mit segment_id als Key und Dict mit 'is_relevant', 'preferred_categories', 'relevance_scores'
        """
        if mode != 'deductive' or not categories:
            # Fallback auf Standard-Relevanzprüfung für andere Modi
            standard_results = await self.check_relevance_batch(segments)
            return {
                sid: {'is_relevant': is_relevant, 'preferred_categories': [], 'relevance_scores': {}}
                for sid, is_relevant in standard_results.items()
            }
        
        print(f"🧫 Erweiterte Relevanzprüfung mit Kategorie-Vorauswahl fuer {len(segments)} Segmente...")
        
        # Cache-Key für erweiterte Relevanzprüfung
        cache_key_base = "extended_relevance"
        
        results = {}
        segments_to_process = []
        
        # Cache-Prüfung
        for segment_id, text in segments:
            cache_key = f"{cache_key_base}_{hash(text)}"
            if cache_key in self.relevance_cache:
                results[segment_id] = self.relevance_cache[cache_key]
            else:
                segments_to_process.append((segment_id, text))
        
        if not segments_to_process:
            return results
        
        # Batch-Verarbeitung für nicht-gecachte Segmente
        batch_text = self._format_segments_for_batch(segments_to_process)
        
        prompt = self.prompt_handler.get_relevance_with_category_preselection_prompt(
            batch_text, categories
        )
        
        try:
            self.api_calls += 1
            response = await self.llm_provider.create_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            llm_response = LLMResponse(response)
            batch_results = json.loads(llm_response.extract_json())
            
            # Verarbeite Batch-Ergebnisse
            segment_results = batch_results.get('segment_results', [])
            
            for i, (segment_id, text) in enumerate(segments_to_process):
                if i < len(segment_results):
                    segment_result = segment_results[i]
                    
                    result = {
                        'is_relevant': segment_result.get('is_relevant', False),
                        'preferred_categories': segment_result.get('preferred_categories', []),
                        'relevance_scores': segment_result.get('relevance_scores', {}),
                        'reasoning': segment_result.get('reasoning', '')
                    }
                    
                    # Cache speichern
                    cache_key = f"{cache_key_base}_{hash(text)}"
                    self.relevance_cache[cache_key] = result
                    
                    results[segment_id] = result
                else:
                    # Fallback bei unvollständigen Ergebnissen
                    results[segment_id] = {
                        'is_relevant': False,
                        'preferred_categories': [],
                        'relevance_scores': {},
                        'reasoning': 'Unvollständiges Batch-Ergebnis'
                    }
            
            print(f"✅ Erweiterte Relevanzprüfung abgeschlossen: {len([r for r in results.values() if r['is_relevant']])} relevante Segmente")
            
        except Exception as e:
            print(f"Fehler bei erweiterter Relevanzprüfung: {str(e)}")
            # Fallback auf Standard-Relevanzprüfung
            standard_results = await self.check_relevance_batch(segments_to_process)
            for segment_id, is_relevant in standard_results.items():
                results[segment_id] = {
                    'is_relevant': is_relevant,
                    'preferred_categories': [],
                    'relevance_scores': {},
                    'reasoning': 'Fallback nach Fehler'
                }
        
        return results
    
    def get_relevance_details(self, segment_id: str) -> Optional[Dict]:
        """Gibt detaillierte Relevanzinformationen für ein Segment zurück."""
        return self.relevance_details.get(segment_id)

    def get_statistics(self) -> Dict:
        """Gibt Statistiken zur Relevanzprüfung zurück."""
        return {
            'total_segments': self.total_segments,
            'relevant_segments': self.relevant_segments,
            'relevance_rate': self.relevant_segments / max(1, self.total_segments),
            'api_calls': self.api_calls,
            'cache_size': len(self.relevance_cache)
        }
