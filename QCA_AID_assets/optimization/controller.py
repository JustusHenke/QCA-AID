"""
Optimization Controller
Main controller for API call optimization across all analysis modes.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from collections import Counter
from datetime import datetime

from QCA_AID_assets.optimization.unified_relevance_analyzer import UnifiedRelevanceAnalyzer, create_unified_analyzer
from QCA_AID_assets.optimization.cache import ModeAwareCache
from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager
from QCA_AID_assets.optimization.tests.metrics_collector import get_global_metrics_collector
from QCA_AID_assets.core.data_models import CategoryDefinition


class AnalysisMode(str, Enum):
    """Analysis modes supported by the optimization system."""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    GROUNDED = "grounded"


class OptimizationController:
    """
    Central controller for optimizing API calls across all analysis modes.
    
    Provides mode-specific optimization strategies:
    - Deductive: Batch category validation
    - Inductive: Thematic batching with similarity grouping
    - Abductive: Hypothesis batching with conflict resolution
    - Grounded: Iterative batch coding with saturation tracking
    """
    
    def __init__(self, llm_provider, model_name: str = "gpt-4", output_dir: str = None, cache_dir: str = None, 
                 relevance_temperature: float = 0.3, inductive_temperature: float = 0.2,
                 multiple_coding_threshold: float = 0.7):
        """
        Initialize optimization controller.
        
        Args:
            llm_provider: LLM provider instance
            model_name: Model to use for optimization
            output_dir: Directory for output files
            cache_dir: Directory for cache files
            relevance_temperature: Temperature for relevance checking (default: 0.3)
            inductive_temperature: Temperature for inductive coding (default: 0.2)
            multiple_coding_threshold: Threshold for multiple coding decisions (default: 0.7)
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.output_dir = output_dir or "."
        self.cache_dir = cache_dir or ".cache"
        self.metrics_collector = get_global_metrics_collector()
        
        # Store temperatures for different modes
        self.relevance_temperature = relevance_temperature
        self.inductive_temperature = inductive_temperature
        
        # Initialize mode-specific analyzers with relevance temperature from config
        self.unified_analyzer = create_unified_analyzer(
            llm_provider=llm_provider,
            model_name=model_name,
            temperature=relevance_temperature,  # Verwende Temperature aus Config für Relevanzprüfung
            multiple_coding_threshold=multiple_coding_threshold  # Verwende Threshold aus Config
        )
        
        # Initialize base cache and dynamic cache manager
        base_cache = ModeAwareCache()
        self.dynamic_cache_manager = DynamicCacheManager(
            base_cache=base_cache,
            analysis_mode=None,  # Will be set per analysis
            reliability_db_path=f"{self.output_dir}/all_codings.json"
        )
        
        # Keep reference to base cache for backward compatibility
        self.cache = base_cache
        
        # Grounded Mode State (für Subcode-Sammlung über Batches hinweg)
        self.grounded_subcodes_collection = []
        self.grounded_segment_analyses = []
        self.grounded_keywords_collection = []
        
        # Load cache if exists, then clear it for fresh start
        try:
            self.cache.load_from_file("optimization_cache.json")
        except:
            pass
        
        # Clear cache at start of each analysis for fresh results
        self.cache.clear()
        
        # Ensure all stats fields are properly initialized after clear
        required_stats_fields = {
            "total_errors": 0,
            "error_contexts": [],
            "last_error_timestamp": None,
            "performance_timings": {},
            "cache_access_patterns": {},
            "eviction_reasons": {},
            "errors": {},
            "memory_usage_by_type": {}
        }
        
        for field, default_value in required_stats_fields.items():
            if field not in self.cache.stats:
                self.cache.stats[field] = default_value
        
        print("   🗑️ Cache geleert für frische Analyse")

            
        # Mode-specific configurations
        self.config = {
            AnalysisMode.DEDUCTIVE: {
                "batch_size": 5,
                "enable_batching": True,
                "quality_threshold": 0.8,
                "max_parallel_requests": 3
            },
            AnalysisMode.INDUCTIVE: {
                "batch_size": 3,
                "enable_batching": True,
                "similarity_threshold": 0.7,
                "quality_threshold": 0.75,
                "max_parallel_requests": 2
            },
            AnalysisMode.ABDUCTIVE: {
                "batch_size": 4,
                "enable_batching": True,
                "conflict_resolution": True,
                "quality_threshold": 0.78,
                "max_parallel_requests": 2
            },
            AnalysisMode.GROUNDED: {
                "batch_size": 3,
                "enable_batching": True,
                "saturation_threshold": 0.9,
                "quality_threshold": 0.72,
                "max_parallel_requests": 2
            }
        }
    
    def _serialize_category_definitions(self, category_definitions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Konvertiert CategoryDefinition-Objekte zu JSON-serialisierbarem Format.
        
        Args:
            category_definitions: Dictionary mit CategoryDefinition-Objekten oder Strings
            
        Returns:
            Dictionary mit serialisierbaren Daten
        """
        serializable_definitions = {}
        
        for name, definition in category_definitions.items():
            if hasattr(definition, 'definition'):
                # CategoryDefinition object
                serialized_subcategories = {}
                
                # FIX: Serialisiere auch Subkategorien rekursiv
                if hasattr(definition, 'subcategories') and definition.subcategories:
                    for subcat_name, subcat_obj in definition.subcategories.items():
                        if hasattr(subcat_obj, 'definition'):
                            # Subkategorie ist auch ein CategoryDefinition Objekt
                            serialized_subcategories[subcat_name] = {
                                'definition': subcat_obj.definition,
                                'examples': subcat_obj.examples if hasattr(subcat_obj, 'examples') else [],
                                'rules': subcat_obj.rules if hasattr(subcat_obj, 'rules') else []
                            }
                        else:
                            # Subkategorie ist bereits ein String oder Dict
                            serialized_subcategories[subcat_name] = subcat_obj
                
                serializable_definitions[name] = {
                    'definition': definition.definition,
                    'subcategories': serialized_subcategories,
                    'examples': definition.examples if hasattr(definition, 'examples') else [],
                    'rules': definition.rules if hasattr(definition, 'rules') else []
                }
            elif isinstance(definition, dict):
                # Already a dict, but check subcategories
                serialized_subcategories = {}
                subcats = definition.get('subcategories', {})
                
                if isinstance(subcats, dict):
                    for subcat_name, subcat_obj in subcats.items():
                        if hasattr(subcat_obj, 'definition'):
                            # Subkategorie ist ein CategoryDefinition Objekt
                            serialized_subcategories[subcat_name] = {
                                'definition': subcat_obj.definition,
                                'examples': subcat_obj.examples if hasattr(subcat_obj, 'examples') else [],
                                'rules': subcat_obj.rules if hasattr(subcat_obj, 'rules') else []
                            }
                        else:
                            # Subkategorie ist bereits serialisiert
                            serialized_subcategories[subcat_name] = subcat_obj
                
                serializable_definitions[name] = {
                    'definition': definition.get('definition', str(definition)),
                    'subcategories': serialized_subcategories,
                    'examples': definition.get('examples', []),
                    'rules': definition.get('rules', [])
                }
            else:
                # String or other type
                serializable_definitions[name] = {
                    'definition': str(definition),
                    'subcategories': {},
                    'examples': [],
                    'rules': []
                }
        
        return serializable_definitions
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics from dynamic cache manager.
        
        Returns:
            Dictionary with cache performance metrics and strategy information
        """
        try:
            cache_stats = self.dynamic_cache_manager.get_statistics()
            manager_info = self.dynamic_cache_manager.get_manager_info()
            reliability_summary = self.dynamic_cache_manager.get_reliability_summary()
            
            return {
                'cache_performance': {
                    'total_entries': cache_stats.total_entries,
                    'shared_entries': cache_stats.shared_entries,
                    'coder_specific_entries': cache_stats.coder_specific_entries,
                    'hit_rate_overall': cache_stats.hit_rate_overall,
                    'hit_rate_by_coder': cache_stats.hit_rate_by_coder,
                    'memory_usage_mb': cache_stats.memory_usage_mb,
                    'strategy_type': cache_stats.strategy_type,
                    'performance_metrics': cache_stats.performance_metrics
                },
                'manager_info': manager_info,
                'reliability_data': reliability_summary,
                'mode_specific_rules': self.dynamic_cache_manager.get_mode_specific_cache_rules()
            }
        except Exception as e:
            print(f"❌ Fehler beim Abrufen der Cache-Statistiken: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'cache_performance': {
                    'total_entries': 0,
                    'shared_entries': 0,
                    'coder_specific_entries': 0,
                    'hit_rate_overall': 0.0,
                    'hit_rate_by_coder': {},
                    'memory_usage_mb': 0.0,
                    'strategy_type': 'unknown',
                    'performance_metrics': {}
                },
                'manager_info': {},
                'reliability_data': {},
                'mode_specific_rules': {}
            }
    
    def export_cache_performance_report(self, filepath: str) -> None:
        """
        Export comprehensive cache performance report.
        
        Args:
            filepath: Path to export the performance report
        """
        try:
            self.dynamic_cache_manager.export_performance_report(filepath)
            print(f"✅ Cache-Performance-Report exportiert: {filepath}")
        except Exception as e:
            print(f"❌ Fehler beim Exportieren des Performance-Reports: {e}")
            raise
    
    async def analyze_segments(self,
                              segments: List[Dict[str, Any]],
                              analysis_mode: AnalysisMode,
                              category_definitions: Optional[Dict[str, str]] = None,
                              research_question: Optional[str] = None,
                              coding_rules: Optional[List[str]] = None,
                              temperature: Optional[float] = None,
                              current_categories: Optional[Dict[str, Any]] = None,
                              coder_settings: Optional[List[Dict[str, Any]]] = None,
                              batch_size: Optional[int] = None,
                              use_context: bool = False,
                              document_paraphrases: Optional[Dict[str, List[str]]] = None,
                              context_paraphrase_count: int = 3,
                              **kwargs) -> List[Dict[str, Any]]:
        """
        Analyze segments using optimized approach for the given mode.
        
        Args:
            segments: List of segments with 'segment_id' and 'text'
            analysis_mode: Analysis mode to use
            category_definitions: Category definitions (for deductive/abductive)
            research_question: Research question context
            coding_rules: Coding rules
            temperature: Temperature for coding (optional, uses mode-specific default)
            current_categories: Current category system (for inductive/abductive)
            coder_settings: List of coder configs with temperature and coder_id (for multi-coder)
            batch_size: User-specified batch size (optional, overrides mode defaults)
            use_context: Whether to use context paraphrases (CODE_WITH_CONTEXT)
            document_paraphrases: Dict mapping document names to lists of paraphrases
            context_paraphrase_count: Number of previous paraphrases to use as context
            **kwargs: Mode-specific parameters
            
        Returns:
            List of analysis results (includes coding results for all modes)
        """
        # Configure dynamic cache manager with analysis mode and coder settings
        self.dynamic_cache_manager.configure_analysis_mode(analysis_mode.value)
        if coder_settings:
            self.dynamic_cache_manager.configure_coders(coder_settings)
            print(f"   🔧 Cache-Manager konfiguriert: {len(coder_settings)} Kodierer, Modus: {analysis_mode.value}")
            
            # Log cache strategy for debugging
            strategy_info = self.dynamic_cache_manager.get_mode_specific_cache_rules()
            print(f"   📊 Cache-Strategie: {strategy_info['strategy_type']}")
            print(f"   🔄 Shared Operations: {strategy_info['shared_operations']}")
            print(f"   👤 Coder-Specific Operations: {strategy_info['coder_specific_operations']}")
        else:
            # Default single coder configuration
            default_coder_settings = [{'coder_id': 'auto_1', 'temperature': temperature or 0.3}]
            self.dynamic_cache_manager.configure_coders(default_coder_settings)
            print(f"   🔧 Cache-Manager konfiguriert: Standard Single-Coder, Modus: {analysis_mode.value}")
        
        # Override batch_size if provided by user
        if batch_size is not None:
            print(f"   ⚙️ User-Config Batch Size: {batch_size} (überschreibt Standard-Werte)")
            # Update all mode configs with user batch_size
            for mode in self.config:
                self.config[mode]['batch_size'] = batch_size
        
        # FIX: Kontext-Paraphrasen Setup
        if use_context:
            print(f"   📝 Kontext-Paraphrasen aktiviert: max. {context_paraphrase_count} pro Dokument")
            if document_paraphrases:
                total_paraphrases = sum(len(paraphrases) for paraphrases in document_paraphrases.values())
                print(f"   📝 Verfügbare Paraphrasen: {total_paraphrases} aus {len(document_paraphrases)} Dokumenten")
            else:
                print(f"   ⚠️ Kontext aktiviert, aber keine Paraphrasen verfügbar")
        else:
            print(f"   📝 Kontext-Paraphrasen deaktiviert")
        
        # Start metrics collection
        batch_id = f"{analysis_mode.value}_{int(asyncio.get_event_loop().time())}"
        self.metrics_collector.start_analysis(analysis_mode.value, batch_id)
        
        # Add all segments to batch
        for segment in segments:
            self.metrics_collector.add_segment(segment['segment_id'])
        
        try:
            # Select appropriate analysis strategy
            if analysis_mode == AnalysisMode.DEDUCTIVE:
                results = await self._analyze_deductive(
                    segments=segments,
                    category_definitions=category_definitions,
                    research_question=research_question,
                    coding_rules=coding_rules,
                    temperature=temperature,
                    coder_settings=coder_settings,
                    use_context=use_context,
                    document_paraphrases=document_paraphrases,
                    context_paraphrase_count=context_paraphrase_count,
                    **kwargs
                )
                # Deductive mode doesn't change categories
                return results, current_categories
            elif analysis_mode == AnalysisMode.INDUCTIVE:
                results = await self._analyze_inductive(
                    segments=segments,
                    research_question=research_question,
                    current_categories=current_categories,
                    coding_rules=coding_rules,
                    coder_settings=coder_settings,
                    temperature=temperature,
                    use_context=use_context,
                    document_paraphrases=document_paraphrases,
                    context_paraphrase_count=context_paraphrase_count,
                    **kwargs
                )
                # For inductive mode, also return updated categories
                updated_categories = getattr(self, '_last_developed_categories', current_categories)
                return results, updated_categories
            elif analysis_mode == AnalysisMode.ABDUCTIVE:
                results = await self._analyze_abductive(
                    segments=segments,
                    category_definitions=category_definitions,
                    research_question=research_question,
                    current_categories=current_categories,
                    coding_rules=coding_rules,
                    coder_settings=coder_settings,
                    temperature=temperature,
                    use_context=use_context,
                    document_paraphrases=document_paraphrases,
                    context_paraphrase_count=context_paraphrase_count,
                    **kwargs
                )
                # For abductive mode, also return updated categories
                updated_categories = getattr(self, '_last_extended_categories', current_categories)
                return results, updated_categories
            elif analysis_mode == AnalysisMode.GROUNDED:
                results = await self._analyze_grounded(
                    segments=segments,
                    research_question=research_question,
                    coder_settings=coder_settings,
                    temperature=temperature,
                    use_context=use_context,
                    document_paraphrases=document_paraphrases,
                    context_paraphrase_count=context_paraphrase_count,
                    **kwargs
                )
                # Grounded mode doesn't change categories in Phase 1
                return results, current_categories
            else:
                raise ValueError(f"Unsupported analysis mode: {analysis_mode}")
            
            # End batch and analysis
            self.metrics_collector.end_batch()
            self.metrics_collector.end_analysis()
            
            # Log final API call summary
            from QCA_AID_assets.utils.tracking.token_tracker import get_global_token_counter
            token_counter = get_global_token_counter()
            session_stats = token_counter.session_stats
            total_requests = session_stats.get('requests', 0)
            
            print(f"\n📊 API CALL SUMMARY:")
            print(f"   🔍 Total API Calls in dieser Analyse: {total_requests}")
            print(f"   📈 Segmente verarbeitet: {len(segments)}")
            if len(segments) > 0:
                calls_per_segment = total_requests / len(segments)
                print(f"   ⚡ Effizienz: {calls_per_segment:.2f} API Calls pro Segment")
            
            # Debug: Show detailed session stats
            print(f"   🔍 DEBUG Session Stats: {session_stats}")
            
            # Debug: Show expected vs actual calls for multi-coder scenarios
            if hasattr(self, '_debug_expected_calls'):
                print(f"   🔍 DEBUG Erwartete API Calls: {self._debug_expected_calls}")
                if total_requests != self._debug_expected_calls:
                    print(f"   ⚠️ DEBUG API Call Diskrepanz: Erwartet {self._debug_expected_calls}, Tatsächlich {total_requests}")
            
        except Exception as e:
            # Record failure in metrics
            self.metrics_collector.end_batch()
            self.metrics_collector.end_analysis()
            raise
    
    async def _analyze_deductive(self,
                                segments: List[Dict[str, Any]],
                                category_definitions: Dict[str, str],
                                research_question: str,
                                coding_rules: List[str],
                                temperature: Optional[float] = None,
                                coder_settings: Optional[List[Dict[str, Any]]] = None,
                                analysis_mode: str = 'deductive',
                                use_context: bool = False,
                                document_paraphrases: Optional[Dict[str, List[str]]] = None,
                                context_paraphrase_count: int = 3,
                                **kwargs) -> List[Dict[str, Any]]:
        """
        Optimized deductive analysis with batch processing and multi-coder support.
        
        Strategy: Batch category validation to reduce API calls.
        Multi-coder: If coder_settings provided, run analysis for each coder.
        """
        config = self.config[AnalysisMode.DEDUCTIVE]
        batch_size = config['batch_size']
        
        # DEBUG: Zeige coder_settings Info
        print(f"[DEBUG _analyze_deductive] coder_settings: {coder_settings}")
        if coder_settings:
            print(f"[DEBUG _analyze_deductive] Anzahl coder_settings: {len(coder_settings)}")
            for cs in coder_settings:
                print(f"  - {cs.get('coder_id', '?')}: temp={cs.get('temperature', '?')}")
        
        # DEBUG: Calculate expected API calls for verification
        num_coders = len(coder_settings) if coder_settings else 1
        
        # CORRECTED: Relevance and category preferences are also batched (but shared across coders)
        estimated_batches = (len(segments) + batch_size - 1) // batch_size
        relevance_calls = estimated_batches  # Relevance is batched but shared
        preference_calls = estimated_batches  # Category preferences are batched but shared
        expected_calls_base = relevance_calls + preference_calls  # Both are batched
        
        # If coder_settings provided, run analysis for each coder
        if coder_settings and len(coder_settings) > 1:
            all_results = []
            # Use intelligent cache management instead of disabling cache
            print(f"   🔄 Multi-Coder Mode: Intelligente Cache-Verwaltung für {len(coder_settings)} Kodierer")
            
            # DEBUG: Estimate expected API calls - CORRECTED CALCULATION
            # Shared calls (batched): relevance + category preferences
            # Per coder calls: coding (also batched)
            estimated_coding_calls = len(coder_settings) * estimated_batches
            self._debug_expected_calls = expected_calls_base + estimated_coding_calls
            
            print(f"   🔍 DEBUG Geschätzte API Calls: {self._debug_expected_calls}")
            print(f"      └─ Shared calls: {expected_calls_base} ({relevance_calls} relevance + {preference_calls} preferences batches)")
            print(f"      └─ Coding calls: {estimated_coding_calls} ({len(coder_settings)} coders × {estimated_batches} batches)")
            print(f"      └─ Segments: {len(segments)}, Batch size: {batch_size}")
            print(f"      └─ Formula: {estimated_batches} × (2 shared + {len(coder_settings)} coders) = {estimated_batches} × {2 + len(coder_settings)} = {self._debug_expected_calls}")
            
            # Track actual API calls for comparison
            from QCA_AID_assets.utils.tracking.token_tracker import get_global_token_counter
            token_counter = get_global_token_counter()
            calls_before = token_counter.session_stats.get('requests', 0)
            print(f"      └─ API calls before optimization: {calls_before}")
            
            # FIX: Führe Relevanzprüfung und Kategoriepräferenzen EINMAL für alle Kodierer durch
            print(f"   📊 Schritt 1: Erweiterte Relevanzprüfung mit Kategorie-Präferenzen (SHARED)...")
            print(f"   📋 Forschungsfrage: {research_question}")
            
            # Schritt 1a: Einfache Relevanzprüfung (SHARED)
            print(f"   🔍 API Call 1: Einfache Relevanzprüfung für {len(segments)} Segmente...")
            from ..core.config import CONFIG
            relevance_threshold = CONFIG.get('RELEVANCE_THRESHOLD', 0.0)
            relevance_results = await self.unified_analyzer.analyze_relevance_simple(
                segments=segments,
                research_question=research_question,
                batch_size=batch_size,
                relevance_threshold=relevance_threshold
            )
            
            # Alle als relevant markierten Segmente werden kodiert (kein zusätzlicher Threshold)
            relevant_segments = []
            for rel_result in relevance_results:
                seg_id = rel_result.get('segment_id', '')
                # Alle Segmente aus relevance_results sind bereits als relevant eingestuft
                seg = next((s for s in segments if s['segment_id'] == seg_id), None)
                if seg:
                    relevant_segments.append(seg)
            
            if not relevant_segments:
                print(f"   ✅ 0 von {len(segments)} Segmenten relevant - KEINE weiteren API Calls nötig")
                return []
            
            print(f"   ✅ {len(relevant_segments)} von {len(segments)} Segmenten relevant")
            
            # Schritt 1b: Kategoriepräferenzen für relevante Segmente (SHARED)
            print(f"   🔍 API Call 2: Kategoriepräferenzen für {len(relevant_segments)} relevante Segmente...")
            category_preference_results = await self.unified_analyzer.analyze_category_preferences(
                segments=relevant_segments,
                category_definitions=category_definitions,
                research_question=research_question,
                coding_rules=coding_rules,
                batch_size=batch_size
            )
            
            # Erstelle Kategoriepräferenzen-Dictionary mit verbessertem Logging
            category_preselections = {}
            for pref_result in category_preference_results:
                seg_id = pref_result.get('segment_id', '')
                # FIX: Fallback für unterschiedliche Feldnamen zwischen Modi
                preferred_cats = pref_result.get('preferred_categories', [])
                if not preferred_cats:
                    preferred_cats = pref_result.get('top_categories', [])
                
                category_preferences = pref_result.get('category_preferences', {})
                
                reasoning = pref_result.get('reasoning', '')
                if not reasoning:
                    reasoning = pref_result.get('preference_reasoning', '')
                
                if seg_id:
                    category_preselections[seg_id] = {
                        'preferred_categories': preferred_cats,
                        'category_preferences': category_preferences,
                        'reasoning': reasoning
                    }
                    
                    # Debug: Log category preferences for each segment
                    seg_id_short = seg_id[:30] + '...' if len(seg_id) > 30 else seg_id
                    print(f"   🎯 Segment: {seg_id_short}")
                    if preferred_cats:
                        print(f"      └─ Präferierte Kategorien: {', '.join(preferred_cats)}")
                        # Show scores for preferred categories
                        for cat in preferred_cats:
                            score = category_preferences.get(cat, 0.0)
                            print(f"         • {cat}: {score:.2f}")
                    else:
                        print(f"      └─ Keine starken Kategoriepräferenzen (alle Scores < 0.6)")
            
            print(f"   ✅ Kategoriepräferenzen für {len(category_preselections)}/{len(relevant_segments)} Segmente ermittelt")
            
            # Zeige Kategoriepräferenzen-Statistik
            if category_preselections:
                all_preferred = []
                for prefs in category_preselections.values():
                    all_preferred.extend(prefs.get('preferred_categories', []))
                
                from collections import Counter
                pref_counts = Counter(all_preferred)
                print(f"   📊 Kategorie-Präferenzen Zusammenfassung: {dict(pref_counts)}")
            else:
                print(f"   ⚠️ Keine Kategoriepräferenzen ermittelt - alle Segmente haben schwache Kategorie-Scores")
            
            # Jetzt für jeden Kodierer nur noch die Kodierung durchführen
            for coder_config in coder_settings:
                coder_id = coder_config.get('coder_id', 'auto_1')
                coder_temperature = coder_config.get('temperature', temperature)
                
                print(f"   🔄 Analysiere mit Kodierer '{coder_id}' (Temperature: {coder_temperature})")
                
                # Run analysis for this coder (NUR KODIERUNG, keine Relevanz/Präferenzen)
                coder_results = await self._analyze_deductive_coding_only(
                    relevant_segments=relevant_segments,
                    category_definitions=category_definitions,
                    research_question=research_question,
                    coding_rules=coding_rules,
                    temperature=coder_temperature,
                    coder_id=coder_id,
                    batch_size=batch_size,
                    config=config,
                    analysis_mode=analysis_mode,
                    use_context=use_context,
                    document_paraphrases=document_paraphrases,
                    context_paraphrase_count=context_paraphrase_count,
                    category_preselections=category_preselections
                )
                all_results.extend(coder_results)
            
            # DEBUG: Track actual API calls made by optimization controller
            calls_after = token_counter.session_stats.get('requests', 0)
            optimization_calls = calls_after - calls_before
            print(f"      └─ API calls after optimization: {calls_after}")
            print(f"      └─ Optimization controller calls: {optimization_calls}")
            print(f"      └─ Estimation accuracy: {optimization_calls}/{self._debug_expected_calls} = {(optimization_calls/self._debug_expected_calls*100):.1f}%" if self._debug_expected_calls > 0 else "")
            
            return all_results
        else:
            # Single coder mode
            coder_id = coder_settings[0].get('coder_id', 'auto_1') if coder_settings else 'auto_1'
            
            # DEBUG: Estimate expected API calls for single coder - CORRECTED
            estimated_coding_calls = 1 * estimated_batches  # Single coder
            self._debug_expected_calls = expected_calls_base + estimated_coding_calls
            
            print(f"   🔍 DEBUG Geschätzte API Calls (Single-Coder): {self._debug_expected_calls}")
            print(f"      └─ Shared calls: {expected_calls_base} ({relevance_calls} relevance + {preference_calls} preferences batches)")
            print(f"      └─ Coding calls: {estimated_coding_calls} (1 coder × {estimated_batches} batches)")
            print(f"      └─ Segments: {len(segments)}, Batch size: {batch_size}")
            print(f"      └─ Formula: {estimated_batches} × (2 shared + 1 coder) = {estimated_batches} × 3 = {self._debug_expected_calls}")
            
            # Track actual API calls for comparison
            from QCA_AID_assets.utils.tracking.token_tracker import get_global_token_counter
            token_counter = get_global_token_counter()
            calls_before = token_counter.session_stats.get('requests', 0)
            print(f"      └─ API calls before optimization: {calls_before}")
            
            result = await self._analyze_deductive_single(
                segments=segments,
                category_definitions=category_definitions,
                research_question=research_question,
                coding_rules=coding_rules,
                temperature=temperature,
                coder_id=coder_id,
                batch_size=batch_size,
                config=config,
                analysis_mode=analysis_mode,
                use_context=use_context,
                document_paraphrases=document_paraphrases,
                context_paraphrase_count=context_paraphrase_count
            )
            
            # DEBUG: Track actual API calls made by optimization controller
            calls_after = token_counter.session_stats.get('requests', 0)
            optimization_calls = calls_after - calls_before
            print(f"      └─ API calls after optimization: {calls_after}")
            print(f"      └─ Optimization controller calls: {optimization_calls}")
            print(f"      └─ Estimation accuracy: {optimization_calls}/{self._debug_expected_calls} = {(optimization_calls/self._debug_expected_calls*100):.1f}%" if self._debug_expected_calls > 0 else "")
            
            return result
    
    def _extract_doc_name_from_segment_id(self, segment_id: str) -> str:
        """
        Extrahiert Dokumentname aus Segment-ID.
        
        Beispiele:
        - "text_demo_wiss.pdf_chunk_0" -> "text_demo_wiss.pdf"
        - "trial_case_iv_2.pdf_chunk_1" -> "trial_case_iv_2.pdf"
        """
        if '_chunk_' in segment_id:
            return segment_id.split('_chunk_')[0]
        else:
            # Fallback: Verwende gesamte segment_id als doc_name
            return segment_id
    
    def _extract_chunk_number_from_segment_id(self, segment_id: str) -> int:
        """
        Extrahiert Chunk-Nummer aus segment_id.
        
        Beispiele:
        - "text_demo_wiss.pdf_chunk_0" -> 0
        - "text_demo_wiss.pdf_chunk_5" -> 5
        - "text_demo_wiss.pdf_chunk_3-1" -> 3 (Mehrfachkodierung)
        """
        parts = segment_id.split('_chunk_')
        if len(parts) >= 2:
            chunk_part = parts[1]
            # Handle multiple coding suffixes like -1, -2
            chunk_part = chunk_part.split('-')[0]
            try:
                return int(chunk_part)
            except ValueError:
                return 0
        return 0
    
    def _get_progressive_context_paraphrases(self, 
                                           segment_id: str, 
                                           document_paraphrases: Dict[str, List[str]], 
                                           context_paraphrase_count: int = 3) -> List[str]:
        """
        Ermittelt progressive Kontext-Paraphrasen für ein Segment.
        
        Nur Paraphrasen von vorherigen Chunks des gleichen Dokuments werden als Kontext verwendet.
        Beispiel:
        - chunk_0: keine Kontext-Paraphrasen
        - chunk_1: max. Paraphrase von chunk_0
        - chunk_5: max. Paraphrasen von chunk_2, chunk_3, chunk_4 (letzte 3)
        
        Args:
            segment_id: ID des aktuellen Segments (z.B. "text1_chunk_5")
            document_paraphrases: Alle verfügbaren Paraphrasen pro Dokument
            context_paraphrase_count: Maximale Anzahl Kontext-Paraphrasen
            
        Returns:
            Liste der Kontext-Paraphrasen für dieses Segment
        """
        doc_name = self._extract_doc_name_from_segment_id(segment_id)
        current_chunk_num = self._extract_chunk_number_from_segment_id(segment_id)
        
        if doc_name not in document_paraphrases:
            return []
        
        all_paraphrases = document_paraphrases[doc_name]
        
        # Für chunk_0 gibt es keine vorherigen Paraphrasen
        if current_chunk_num == 0:
            return []
        
        # Für chunk_N können maximal die Paraphrasen von chunk_0 bis chunk_(N-1) verwendet werden
        # Da die Paraphrasen in der Reihenfolge der Chunks gespeichert sind,
        # nehmen wir die ersten (current_chunk_num) Paraphrasen
        available_previous_paraphrases = all_paraphrases[:current_chunk_num]
        
        # Limitiere auf die letzten N Paraphrasen (wie in Standard-Analyse)
        context_paraphrases = available_previous_paraphrases[-context_paraphrase_count:]
        
        return context_paraphrases
    
    def _format_single_coding_result(self, r, coder_id: str, preferred_cats: List[str], seg_prefs: Dict) -> Dict:
        """
        Formatiert ein einzelnes Kodierungsergebnis (ohne Mehrfachkodierung).
        """
        return {
            'segment_id': r.segment_id,
            'result': {
                'primary_category': r.primary_category,
                'confidence': r.confidence,
                'all_categories': r.all_categories,
                'subcategories': r.subcategories if r.subcategories else [],
                'keywords': r.keywords if r.keywords else '',
                'paraphrase': r.paraphrase if r.paraphrase else '',
                'justification': r.justification if r.justification else '',
                'coder_id': coder_id,
                'category_preselection_used': bool(preferred_cats),
                'preferred_categories': preferred_cats,
                'preselection_reasoning': seg_prefs.get('reasoning', ''),
                # Standard-Kodierung Metadaten
                'multiple_coding_instance': 1,
                'total_coding_instances': 1,
                'target_category': '',
                'original_segment_id': r.segment_id,
                'is_multiple_coding': False
            },
            'analysis_mode': 'deductive',
            'timestamp': asyncio.get_event_loop().time()
        }
    
    def _format_single_coding_result_inductive(self, r, coder_id: str) -> Dict:
        """
        Formatiert ein einzelnes Kodierungsergebnis für inductive Mode (ohne Mehrfachkodierung).
        """
        # Enhanced logging
        seg_id_short = r.segment_id[:30] + '...' if len(r.segment_id) > 30 else r.segment_id
        print(f"   📝 Segment: {seg_id_short} → Kategorie: {r.primary_category}")
        if r.subcategories:
            print(f"      └─ Subkategorien: {r.subcategories}")
        print(f"      └─ Kodierer: {coder_id} | Konfidenz: {r.confidence:.2f}")
        if hasattr(r, 'justification') and r.justification:
            justification_short = r.justification[:80] + '...' if len(r.justification) > 80 else r.justification
            print(f"      └─ Begründung: {justification_short}")
        
        return {
            'segment_id': r.segment_id,
            'result': {
                'primary_category': r.primary_category,
                'confidence': r.confidence,
                'all_categories': r.all_categories,
                'subcategories': r.subcategories if r.subcategories else [],
                'keywords': r.keywords if r.keywords else '',
                'paraphrase': r.paraphrase if r.paraphrase else '',
                'justification': r.justification if r.justification else '',
                'coder_id': coder_id,
                'category_preselection_used': False,  # No preselection in inductive
                'preferred_categories': [],
                'preselection_reasoning': '',
                # Standard-Kodierung Metadaten
                'multiple_coding_instance': 1,
                'total_coding_instances': 1,
                'target_category': '',
                'original_segment_id': r.segment_id,
                'is_multiple_coding': False
            },
            'analysis_mode': 'inductive',
            'timestamp': asyncio.get_event_loop().time()
        }
    
    def _format_single_coding_result_abductive(self, r, coder_id: str, preferred_cats: List[str], seg_prefs: Dict) -> Dict:
        """
        Formatiert ein einzelnes Kodierungsergebnis für abductive Mode (ohne Mehrfachkodierung).
        """
        # Enhanced logging with coder information
        seg_id_short = r.segment_id[:30] + '...' if len(r.segment_id) > 30 else r.segment_id
        print(f"   📝 Segment: {seg_id_short} → Kategorie: {r.primary_category}")
        if r.subcategories:
            print(f"      └─ Subkategorien: {r.subcategories}")
        if preferred_cats:
            print(f"      └─ Präferierte Kategorien: {preferred_cats}")
        print(f"      └─ Kodierer: {coder_id} | Konfidenz: {r.confidence:.2f}")
        if hasattr(r, 'justification') and r.justification:
            justification_short = r.justification[:80] + '...' if len(r.justification) > 80 else r.justification
            print(f"      └─ Begründung: {justification_short}")
        
        return {
            'segment_id': r.segment_id,
            'result': {
                'primary_category': r.primary_category,
                'confidence': r.confidence,
                'all_categories': r.all_categories,
                'subcategories': r.subcategories if r.subcategories else [],
                'keywords': r.keywords if r.keywords else '',
                'paraphrase': r.paraphrase if r.paraphrase else '',
                'justification': r.justification if r.justification else '',
                'coder_id': coder_id,
                'category_preselection_used': bool(preferred_cats),
                'preferred_categories': preferred_cats,
                'preselection_reasoning': seg_prefs.get('reasoning', ''),
                # Standard-Kodierung Metadaten
                'multiple_coding_instance': 1,
                'total_coding_instances': 1,
                'target_category': '',
                'original_segment_id': r.segment_id,
                'is_multiple_coding': False
            },
            'analysis_mode': 'abductive',
            'timestamp': asyncio.get_event_loop().time()
        }

    async def _analyze_deductive_coding_only(self,
                                            relevant_segments: List[Dict[str, Any]],
                                            category_definitions: Dict[str, str],
                                            research_question: str,
                                            coding_rules: List[str],
                                            temperature: Optional[float],
                                            coder_id: str,
                                            batch_size: int,
                                            config: Dict[str, Any],
                                            analysis_mode: str = 'deductive',
                                            use_context: bool = False,
                                            document_paraphrases: Optional[Dict[str, List[str]]] = None,
                                            context_paraphrase_count: int = 3,
                                            category_preselections: Optional[Dict[str, Dict]] = None) -> List[Dict[str, Any]]:
        """
        Führt nur die Kodierung durch (ohne Relevanzprüfung und Kategoriepräferenzen).
        Für Multi-Coder-Szenarien wo Relevanz/Präferenzen bereits einmal berechnet wurden.
        """
        results = []
        
        if not relevant_segments:
            return results
        
        print(f"   📊 Schritt 3: Kodierung von {len(relevant_segments)} relevanten Segmenten...")
        
        # Process segments in batches
        total_batches = (len(relevant_segments) + batch_size - 1) // batch_size
        
        for i in range(0, len(relevant_segments), batch_size):
            batch = relevant_segments[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"   📦 Batch {batch_num}/{total_batches}: {len(batch)} Segmente")
            
            if config['enable_batching'] and len(batch) > 1:
                # Use batch processing with category preferences
                batch_results = await self._batch_analyze_deductive(
                    batch=batch,
                    category_definitions=category_definitions,
                    research_question=research_question,
                    coding_rules=coding_rules,
                    temperature=temperature,
                    coder_id=coder_id,
                    category_preselections=category_preselections,
                    analysis_mode=analysis_mode,
                    batch_size=batch_size,
                    use_context=use_context,
                    document_paraphrases=document_paraphrases,
                    context_paraphrase_count=context_paraphrase_count
                )
                results.extend(batch_results)
            else:
                # Process individually with category preferences
                use_temperature = temperature if temperature is not None else self.unified_analyzer.temperature
                for segment in batch:
                    # Get category preferences for this segment
                    seg_prefs = category_preselections.get(segment['segment_id'], {}) if category_preselections else {}
                    preferred_cats = seg_prefs.get('preferred_categories', [])
                    
                    # Filter categories if preferences exist
                    effective_categories = category_definitions
                    if preferred_cats:
                        effective_categories = {
                            name: definition for name, definition in category_definitions.items()
                            if name in preferred_cats
                        }
                        print(f"   🎯 Segment {segment['segment_id']}: Fokus auf {len(effective_categories)} präferierte Kategorien")
                    
                    # Single segment analysis using analyze_batch
                    batch_results = await self.unified_analyzer.analyze_batch(
                        segments=[segment],
                        category_definitions=effective_categories,
                        research_question=research_question,
                        coding_rules=coding_rules,
                        batch_size=1,
                        temperature=use_temperature
                    )
                    
                    if batch_results and len(batch_results) > 0:
                        result = batch_results[0]  # Erstes (und einziges) Ergebnis
                    else:
                        result = None
                    
                    if result:
                        formatted_result = {
                            'segment_id': segment['segment_id'],
                            'result': {
                                'primary_category': result.primary_category,
                                'confidence': result.confidence,
                                'all_categories': getattr(result, 'all_categories', [result.primary_category]),
                                'subcategories': getattr(result, 'subcategories', []),
                                'keywords': getattr(result, 'keywords', ''),
                                'paraphrase': getattr(result, 'paraphrase', ''),
                                'justification': getattr(result, 'justification', ''),
                                'coder_id': coder_id,
                                'category_preselection_used': bool(preferred_cats),
                                'preferred_categories': preferred_cats,
                                'preselection_reasoning': seg_prefs.get('reasoning', '')
                            },
                            'analysis_mode': analysis_mode,
                            'timestamp': asyncio.get_event_loop().time()
                        }
                        results.append(formatted_result)
        
        return results

    async def _analyze_deductive_single(self,
                                       segments: List[Dict[str, Any]],
                                       category_definitions: Dict[str, str],
                                       research_question: str,
                                       coding_rules: List[str],
                                       temperature: Optional[float],
                                       coder_id: str,
                                       batch_size: int,
                                       config: Dict[str, Any],
                                       analysis_mode: str = 'deductive',
                                       use_context: bool = False,
                                       document_paraphrases: Optional[Dict[str, List[str]]] = None,
                                       context_paraphrase_count: int = 3) -> List[Dict[str, Any]]:
        """
        Single coder deductive analysis with category preferences.
        """
        results = []
        
        # Step 1: Relevance check with category preferences (like standard analysis)
        print(f"   📊 Schritt 1: Erweiterte Relevanzprüfung mit Kategorie-Präferenzen...")
        print(f"   📋 Forschungsfrage: {research_question}")
        
        # Schritt 1a: Einfache Relevanzprüfung
        print(f"   🔍 API Call 1: Einfache Relevanzprüfung für {len(segments)} Segmente...")
        from ..core.config import CONFIG
        relevance_threshold = CONFIG.get('RELEVANCE_THRESHOLD', 0.0)
        relevance_results = await self.unified_analyzer.analyze_relevance_simple(
            segments=segments,
            research_question=research_question,
            batch_size=batch_size,
            relevance_threshold=relevance_threshold
        )
        
        # Alle als relevant markierten Segmente werden kodiert (kein zusätzlicher Threshold)
        relevant_segments = []
        for rel_result in relevance_results:
            seg_id = rel_result.get('segment_id', '')
            # Alle Segmente aus relevance_results sind bereits als relevant eingestuft
            seg = next((s for s in segments if s['segment_id'] == seg_id), None)
            if seg:
                relevant_segments.append(seg)
        
        if not relevant_segments:
            print(f"   ✅ 0 von {len(segments)} Segmenten relevant - KEINE weiteren API Calls nötig")
            return []
        
        # Schritt 1b: Kategoriepräferenzen für relevante Segmente
        print(f"   🔍 API Call 2: Kategoriepräferenzen für {len(relevant_segments)} relevante Segmente...")
        category_preference_results = await self.unified_analyzer.analyze_category_preferences(
            segments=relevant_segments,
            category_definitions=category_definitions,
            research_question=research_question,
            coding_rules=coding_rules,
            batch_size=batch_size
        )
        
        # Sammle Kategoriepräferenzen mit verbessertem Logging
        category_preselections = {}
        for pref_result in category_preference_results:
            seg_id = pref_result.get('segment_id', '')
            if seg_id:
                # FIX: Fallback für unterschiedliche Feldnamen zwischen Modi
                top_categories = pref_result.get('top_categories', [])
                if not top_categories:
                    top_categories = pref_result.get('preferred_categories', [])
                
                category_preferences = pref_result.get('category_preferences', {})
                
                reasoning = pref_result.get('preference_reasoning', '')
                if not reasoning:
                    reasoning = pref_result.get('reasoning', '')
                
                category_preselections[seg_id] = {
                    'preferred_categories': top_categories,
                    'category_preferences': category_preferences,
                    'relevance_scores': category_preferences,
                    'reasoning': reasoning
                }
                
                # Debug: Log category preferences for each segment
                seg_id_short = seg_id[:30] + '...' if len(seg_id) > 30 else seg_id
                print(f"   🎯 Segment: {seg_id_short}")
                if top_categories:
                    print(f"      └─ Präferierte Kategorien: {', '.join(top_categories)}")
                    # Show scores for preferred categories
                    for cat in top_categories:
                        score = category_preferences.get(cat, 0.0)
                        print(f"         • {cat}: {score:.2f}")
                else:
                    print(f"      └─ Keine starken Kategoriepräferenzen (alle Scores < 0.6)")
        
        print(f"   ✅ {len(relevant_segments)} von {len(segments)} Segmenten relevant")
        if category_preselections:
            preselection_stats = {}
            for prefs in category_preselections.values():
                for cat in prefs['preferred_categories']:
                    preselection_stats[cat] = preselection_stats.get(cat, 0) + 1
            print(f"   📊 Kategorie-Präferenzen Zusammenfassung: {dict(preselection_stats)}")
        else:
            print(f"   ⚠️ Keine Kategoriepräferenzen ermittelt - alle Segmente haben schwache Kategorie-Scores")
        
        if not relevant_segments:
            return []
        
        # Step 3: Process relevant segments with category preferences
        print(f"   🔍 Schritt 3: Kodierung von {len(relevant_segments)} relevanten Segmenten...")
        for i in range(0, len(relevant_segments), batch_size):
            batch = relevant_segments[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(relevant_segments) + batch_size - 1) // batch_size
            
            print(f"   📦 Batch {batch_num}/{total_batches}: {len(batch)} Segmente")
            
            if config['enable_batching'] and len(batch) > 1:
                # Use batch processing with category preferences
                batch_results = await self._batch_analyze_deductive(
                    batch=batch,
                    category_definitions=category_definitions,
                    research_question=research_question,
                    coding_rules=coding_rules,
                    temperature=temperature,
                    coder_id=coder_id,
                    category_preselections=category_preselections,  # Pass preferences
                    analysis_mode=analysis_mode,  # Pass the analysis mode
                    batch_size=batch_size,  # Pass batch_size parameter
                    use_context=use_context,
                    document_paraphrases=document_paraphrases,
                    context_paraphrase_count=context_paraphrase_count
                )
                results.extend(batch_results)
            else:
                # Process individually with category preferences
                use_temperature = temperature if temperature is not None else self.unified_analyzer.temperature
                for segment in batch:
                    # Get category preferences for this segment
                    seg_prefs = category_preselections.get(segment['segment_id'], {})
                    preferred_cats = seg_prefs.get('preferred_categories', [])
                    
                    # Filter category definitions to preferred categories if available
                    effective_categories = category_definitions
                    if preferred_cats:
                        effective_categories = {
                            name: definition for name, definition in category_definitions.items()
                            if name in preferred_cats
                        }
                        print(f"   🎯 Segment {segment['segment_id']}: Fokus auf {len(effective_categories)} präferierte Kategorien")
                    
                    # Temporarily set temperature for this call
                    original_temp = self.unified_analyzer.temperature
                    self.unified_analyzer.temperature = use_temperature
                    try:
                        result = await self.unified_analyzer.analyze_relevance_comprehensive(
                            segment_text=segment['text'],
                            segment_id=segment['segment_id'],
                            category_definitions=effective_categories,  # Use filtered categories
                            research_question=research_question,
                            coding_rules=coding_rules
                        )
                    finally:
                        self.unified_analyzer.temperature = original_temp
                    
                    # Convert to standard format (consistent with batch processing)
                    results.append({
                        'segment_id': result.segment_id,
                        'result': {
                            'primary_category': result.primary_category,
                            'confidence': result.confidence,
                            'all_categories': result.all_categories,
                            'subcategories': result.subcategories if result.subcategories else [],
                            'keywords': result.keywords if result.keywords else '',
                            'paraphrase': result.paraphrase if result.paraphrase else '',
                            'justification': result.justification if result.justification else '',
                            'coder_id': coder_id,
                            # Add category preference information
                            'category_preselection_used': bool(preferred_cats),
                            'preferred_categories': preferred_cats,
                            'preselection_reasoning': seg_prefs.get('reasoning', '')
                        },
                        'analysis_mode': 'deductive'
                    })
        
        return results
    
    async def _batch_analyze_deductive(self,
                                      batch: List[Dict[str, Any]],
                                      category_definitions: Dict[str, str],
                                      research_question: str,
                                      coding_rules: List[str],
                                      temperature: Optional[float] = None,
                                      coder_id: str = 'auto_1',
                                      category_preselections: Optional[Dict[str, Dict]] = None,
                                      analysis_mode: str = 'deductive',
                                      batch_size: int = 5,
                                      use_context: bool = False,
                                      document_paraphrases: Optional[Dict[str, List[str]]] = None,
                                      context_paraphrase_count: int = 3) -> List[Dict[str, Any]]:
        """
        Analyze batch of segments for deductive mode using intelligent caching.
        
        Uses DynamicCacheManager to determine shared vs coder-specific caching.
        Applies category preselections to filter categories for each segment.
        
        Args:
            batch: List of segments to analyze
            category_definitions: All available category definitions
            research_question: Research question context
            coding_rules: List of coding rules
            temperature: Temperature for LLM calls
            coder_id: Identifier for the coder
            category_preselections: Dict mapping segment_id to category preferences
        """
        # 1. Check Cache using intelligent strategy
        print(f"      🔍 Prüfe Cache für {len(batch)} Segmente...")
        cached_results = []
        uncached_segments = []
        
        for segment in batch:
            # FIX: Konvertiere CategoryDefinition-Objekte zu serialisierbarem Format für Cache-Keys
            serializable_category_definitions = self._serialize_category_definitions(category_definitions)
            
            # Use dynamic cache manager to determine cache key strategy
            if self.dynamic_cache_manager.should_cache_shared('coding'):
                # Use shared cache key for coding operation
                cache_key = self.dynamic_cache_manager.get_shared_cache_key(
                    'coding',
                    analysis_mode=analysis_mode,
                    segment_text=segment['text'],
                    category_definitions=serializable_category_definitions,  # FIX: Verwende serialisierbare Version
                    research_question=research_question,
                    coding_rules=coding_rules,
                    segment_id=segment['segment_id']
                )
            else:
                # Use coder-specific cache key
                cache_key = self.dynamic_cache_manager.get_coder_specific_key(
                    coder_id,
                    'coding',
                    analysis_mode=analysis_mode,
                    segment_text=segment['text'],
                    category_definitions=serializable_category_definitions,  # FIX: Verwende serialisierbare Version
                    research_question=research_question,
                    coding_rules=coding_rules,
                    segment_id=segment['segment_id']
                )
            
            # Try to get from cache using the determined key
            # FIX: Use the base cache with the generated cache key and serializable category definitions
            cached_val = self.cache.get(
                analysis_mode=analysis_mode,
                segment_text=segment['text'],
                category_definitions=serializable_category_definitions,  # FIX: Verwende serialisierbare Version
                research_question=research_question,
                coding_rules=coding_rules,
                segment_id=segment['segment_id'],
                operation_type='coding',
                coder_id=coder_id
            )
            
            if cached_val:
                # Check if cached result is compatible with current coder
                cached_coder_id = cached_val.get('result', {}).get('coder_id')
                if (self.dynamic_cache_manager.should_cache_shared('coding') or 
                    cached_coder_id == coder_id):
                    cached_results.append(cached_val)
                else:
                    uncached_segments.append(segment)
            else:
                uncached_segments.append(segment)
        
        # Log cache performance
        cache_hits = len(cached_results)
        cache_misses = len(uncached_segments)
        total_segments = len(batch)
        hit_rate = (cache_hits / total_segments * 100) if total_segments > 0 else 0
        
        print(f"      📊 Cache Performance: {cache_hits} Hits, {cache_misses} Misses ({hit_rate:.1f}% Hit Rate)")
        
        # 2. Process Uncached Segments
        new_results = []
        if uncached_segments:
            print(f"      🔍 API Call 3+: Kodierung von {len(uncached_segments)} Segmenten (Cache Miss)")
            
            # Apply category preselections if available
            # For batch processing, we need to handle segments with different preferred categories
            # Strategy: If all segments have similar preferences, use filtered categories
            # Otherwise, use all categories and let the analyzer decide
            
            effective_categories = category_definitions
            if category_preselections:
                # Collect all preferred categories across the batch
                all_preferred_cats = set()
                segments_with_prefs = 0
                for segment in uncached_segments:
                    seg_prefs = category_preselections.get(segment['segment_id'], {})
                    preferred_cats = seg_prefs.get('preferred_categories', [])
                    if preferred_cats:
                        all_preferred_cats.update(preferred_cats)
                        segments_with_prefs += 1
                
                # If most segments have preferences, use the union of preferred categories
                if segments_with_prefs > len(uncached_segments) / 2 and all_preferred_cats:
                    effective_categories = {
                        name: definition for name, definition in category_definitions.items()
                        if name in all_preferred_cats
                    }
                    print(f"   🎯 Batch: Fokus auf {len(effective_categories)} präferierte Kategorien (aus {len(category_definitions)} verfügbaren)")
            
            # FIX: Verwende progressive Kontext-Paraphrasen pro Segment
            # Für Batch-Processing nehmen wir die Kontext-Paraphrasen des ersten Segments als Repräsentativ
            # (In der Praxis sollten alle Segmente im Batch aus dem gleichen Dokument stammen)
            context_paraphrases = []
            if use_context and document_paraphrases and uncached_segments:
                first_segment_id = uncached_segments[0]['segment_id']
                context_paraphrases = self._get_progressive_context_paraphrases(
                    first_segment_id, document_paraphrases, context_paraphrase_count
                )
                
                if context_paraphrases:
                    doc_name = self._extract_doc_name_from_segment_id(first_segment_id)
                    chunk_num = self._extract_chunk_number_from_segment_id(first_segment_id)
                    print(f"      📝 Nutze {len(context_paraphrases)} progressive Kontext-Paraphrasen für Dokument '{doc_name}' (Chunk {chunk_num})")
                else:
                    first_segment_id = uncached_segments[0]['segment_id']
                    doc_name = self._extract_doc_name_from_segment_id(first_segment_id)
                    chunk_num = self._extract_chunk_number_from_segment_id(first_segment_id)
                    print(f"      📝 Keine Kontext-Paraphrasen für Dokument '{doc_name}' Chunk {chunk_num} (erste Chunks oder nicht verfügbar)")
            
            # Use true batch processing via unified analyzer with temperature and context
            batch_results = await self.unified_analyzer.analyze_batch(
                segments=uncached_segments,
                category_definitions=effective_categories,  # Use filtered categories
                research_question=research_question,
                coding_rules=coding_rules,
                batch_size=batch_size,  # Use configured batch_size instead of all segments
                temperature=temperature,
                context_paraphrases=context_paraphrases if context_paraphrases else None
            )
            
            print(f"   🔍 DEBUG: UnifiedAnalyzer returned {len(batch_results)} results")
            
            # FIX: Implementiere korrekte Mehrfachkodierung - KEINE separaten API Calls!
            expanded_results = []
            for r in batch_results:
                # Get category preselection info for this specific segment
                seg_prefs = category_preselections.get(r.segment_id, {}) if category_preselections else {}
                preferred_cats = seg_prefs.get('preferred_categories', [])
                
                if hasattr(r, 'requires_multiple_coding') and r.requires_multiple_coding:
                    # Mehrfachkodierung erforderlich - erstelle mehrere Ergebnisse aus EINER Antwort
                    relevant_categories = []
                    if hasattr(r, 'relevance_scores') and r.relevance_scores:
                        # Sammle alle Kategorien über dem Schwellenwert
                        for cat_name, score in r.relevance_scores.items():
                            if score >= self.unified_analyzer.multiple_coding_threshold:
                                relevant_categories.append({
                                    'category': cat_name,
                                    'relevance_score': score
                                })
                    
                    if len(relevant_categories) > 1:
                        print(f"   🔁 Mehrfachkodierung für Segment {r.segment_id}: {len(relevant_categories)} Kategorien")
                        
                        # Erstelle separate Ergebnisse für jede relevante Kategorie (OHNE zusätzliche API Calls)
                        for i, cat_info in enumerate(relevant_categories, 1):
                            target_category = cat_info['category']
                            
                            # Erstelle Mehrfachkodierungs-Instanz basierend auf der ursprünglichen Antwort
                            multiple_segment_id = f"{r.segment_id}-{i}"
                            
                            # Finde passende Subkategorien für diese Kategorie
                            target_subcategories = []
                            if hasattr(r, 'subcategories') and r.subcategories:
                                # Filtere Subkategorien die zu dieser Kategorie gehören
                                for subcat in r.subcategories:
                                    if isinstance(subcat, str) and target_category.lower() in subcat.lower():
                                        target_subcategories.append(subcat)
                                    elif isinstance(subcat, dict) and subcat.get('category') == target_category:
                                        target_subcategories.append(subcat.get('name', ''))
                                
                                # Fallback: Verwende alle Subkategorien wenn keine spezifischen gefunden
                                if not target_subcategories:
                                    target_subcategories = r.subcategories
                            
                            formatted_res = {
                                'segment_id': multiple_segment_id,
                                'result': {
                                    'primary_category': target_category,
                                    'confidence': cat_info['relevance_score'],  # Verwende Relevanz-Score als Konfidenz
                                    'all_categories': [target_category],
                                    'subcategories': target_subcategories,
                                    'keywords': r.keywords if hasattr(r, 'keywords') else '',
                                    'paraphrase': r.paraphrase if hasattr(r, 'paraphrase') else '',
                                    'justification': f"[Mehrfachkodierung {i}/{len(relevant_categories)}] {r.justification if hasattr(r, 'justification') else ''}",
                                    'coder_id': coder_id,
                                    'category_preselection_used': bool(preferred_cats),
                                    'preferred_categories': preferred_cats,
                                    'preselection_reasoning': seg_prefs.get('reasoning', ''),
                                    # Mehrfachkodierungs-Metadaten
                                    'multiple_coding_instance': i,
                                    'total_coding_instances': len(relevant_categories),
                                    'target_category': target_category,
                                    'original_segment_id': r.segment_id,
                                    'is_multiple_coding': True
                                },
                                'analysis_mode': analysis_mode,
                                'timestamp': asyncio.get_event_loop().time()
                            }
                            expanded_results.append(formatted_res)
                            
                            # Log multiple coding result
                            print(f"      📝 Mehrfachkodierung {i}: {target_category} (Konfidenz: {cat_info['relevance_score']:.2f})")
                    else:
                        # Nur eine Kategorie über Schwellenwert - normale Kodierung
                        formatted_res = self._format_single_coding_result(r, coder_id, preferred_cats, seg_prefs)
                        expanded_results.append(formatted_res)
                else:
                    # Keine Mehrfachkodierung erforderlich - normale Kodierung
                    formatted_res = self._format_single_coding_result(r, coder_id, preferred_cats, seg_prefs)
                    expanded_results.append(formatted_res)
            
            new_results = expanded_results
            
            # Cache und Reliability für alle Ergebnisse (normale + Mehrfachkodierung)
            for formatted_res in new_results:
                # Enhanced logging
                segment_id = formatted_res['segment_id']
                result_data = formatted_res['result']
                
                seg_id_short = segment_id[:30] + '...' if len(segment_id) > 30 else segment_id
                print(f"   📝 Segment: {seg_id_short} → Kategorie: {result_data['primary_category']}")
                if result_data.get('subcategories'):
                    print(f"      └─ Subkategorien: {result_data['subcategories']}")
                if result_data.get('preferred_categories'):
                    print(f"      └─ Präferierte Kategorien: {result_data['preferred_categories']}")
                print(f"      └─ Kodierer: {coder_id} | Konfidenz: {result_data['confidence']:.2f}")
                if result_data.get('justification'):
                    justification_short = result_data['justification'][:80] + '...' if len(result_data['justification']) > 80 else result_data['justification']
                    print(f"      └─ Begründung: {justification_short}")
                if result_data.get('is_multiple_coding'):
                    print(f"      └─ Mehrfachkodierung: {result_data['multiple_coding_instance']}/{result_data['total_coding_instances']} (Ziel: {result_data['target_category']})")
                
                # Add to cache
                serializable_category_definitions = self._serialize_category_definitions(category_definitions)
                
                # Für Mehrfachkodierung verwende die original segment_id für Cache-Lookup
                cache_segment_id = result_data.get('original_segment_id', segment_id)
                
                self.cache.set(
                    analysis_mode="deductive",
                    segment_text=next((s['text'] for s in uncached_segments if s['segment_id'] == cache_segment_id), ""),
                    value=formatted_res,
                    category_definitions=serializable_category_definitions,
                    research_question=research_question,
                    coding_rules=coding_rules,
                    segment_id=segment_id,  # Verwende die tatsächliche segment_id (mit -1, -2 suffix)
                    confidence=result_data['confidence']
                )
                
                # Store for reliability analysis
                from QCA_AID_assets.core.data_models import ExtendedCodingResult
                from datetime import datetime
                
                reliability_result = ExtendedCodingResult(
                    segment_id=segment_id,  # Verwende die tatsächliche segment_id (mit -1, -2 suffix)
                    coder_id=coder_id,
                    category=result_data['primary_category'],
                    subcategories=result_data.get('subcategories', []),
                    confidence=result_data['confidence'],
                    justification=result_data.get('justification', ''),
                    analysis_mode=analysis_mode,
                    timestamp=datetime.now(),
                    is_manual=False,
                    metadata={
                        'temperature': temperature,
                        'keywords': result_data.get('keywords', ''),
                        'paraphrase': result_data.get('paraphrase', ''),
                        'is_multiple_coding': result_data.get('is_multiple_coding', False),
                        'multiple_coding_instance': result_data.get('multiple_coding_instance', 1),
                        'total_coding_instances': result_data.get('total_coding_instances', 1),
                        'target_category': result_data.get('target_category', ''),
                        'original_segment_id': result_data.get('original_segment_id', segment_id)
                    }
                )
                
                try:
                    self.dynamic_cache_manager.store_for_reliability(reliability_result)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
        
        # 3. Merge Results
        final_results = cached_results + new_results
        
        # Save persistence periodically
        if uncached_segments:
            self.cache.save_to_file("optimization_cache.json")

        return final_results
    
    async def _store_results_for_reliability(self, results: List[Dict[str, Any]], analysis_mode: str, coder_id: str, temperature: Optional[float] = None) -> None:
        """
        Helper method to store coding results for reliability analysis.
        
        Args:
            results: List of coding results to store
            analysis_mode: Analysis mode (deductive, abductive, inductive, grounded)
            coder_id: Coder ID for the results
            temperature: Actual temperature used for this coder (optional)
        """
        if not results:
            return
        
        from QCA_AID_assets.core.data_models import ExtendedCodingResult
        from datetime import datetime
        
        # print(f"   💾 DEBUG: Storing {len(results)} results for reliability analysis (mode: {analysis_mode}, coder: {coder_id}, temp: {temperature})")
        
        for result in results:
            try:
                # Extract result data (handle different formats)
                if 'result' in result:
                    result_data = result['result']
                    segment_id = result.get('segment_id', result_data.get('segment_id', ''))
                    primary_category = result_data.get('primary_category', 'Nicht kodiert')
                    confidence_val = result_data.get('confidence', 0.7)
                    subcategories = result_data.get('subcategories', [])
                    keywords = result_data.get('keywords', '')
                    paraphrase = result_data.get('paraphrase', '')
                    justification = result_data.get('justification', '')
                    result_coder_id = result_data.get('coder_id', coder_id)
                else:
                    segment_id = result.get('segment_id', '')
                    primary_category = result.get('primary_category', 'Nicht kodiert')
                    confidence_val = result.get('confidence', 0.7)
                    subcategories = result.get('subcategories', [])
                    keywords = result.get('keywords', '')
                    paraphrase = result.get('paraphrase', '')
                    justification = result.get('justification', '')
                    result_coder_id = result.get('coder_id', coder_id)
                
                # Use actual temperature if provided, otherwise fallback to mode default
                actual_temperature = temperature if temperature is not None else getattr(self, f'{analysis_mode}_temperature', 0.3)
                
                # Create ExtendedCodingResult
                reliability_result = ExtendedCodingResult(
                    segment_id=segment_id,
                    coder_id=result_coder_id,
                    category=primary_category,
                    subcategories=subcategories if subcategories else [],
                    confidence=confidence_val,
                    justification=justification if justification else '',
                    analysis_mode=analysis_mode,
                    timestamp=datetime.now(),
                    is_manual=False,
                    metadata={
                        'temperature': actual_temperature,
                        'keywords': keywords if keywords else '',
                        'paraphrase': paraphrase if paraphrase else ''
                    }
                )
                
                # Store for reliability analysis using dynamic cache manager
                try:
                   #  print(f"   💾 DEBUG: Storing reliability result for {segment_id}, coder {result_coder_id}")
                    self.dynamic_cache_manager.store_for_reliability(reliability_result)
                    # print(f"   ✅ DEBUG: Successfully stored reliability result for {segment_id}")
                except Exception as e:
                    # print(f"   ❌ DEBUG: Failed to store reliability result for {segment_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as e:
                # print(f"   ❌ DEBUG: Failed to process result for reliability storage: {e}")
                import traceback
                traceback.print_exc()
    
    async def _analyze_inductive(self,
                                 segments: List[Dict[str, Any]],
                                 research_question: str,
                                 current_categories: Optional[Dict[str, Any]] = None,
                                 coding_rules: Optional[List[str]] = None,
                                 coder_settings: Optional[List[Dict[str, Any]]] = None,
                                 temperature: Optional[float] = None,
                                 use_context: bool = False,
                                 document_paraphrases: Optional[Dict[str, List[str]]] = None,
                                 context_paraphrase_count: int = 3,
                                 **kwargs) -> List[Dict[str, Any]]:
        """
        Optimized inductive analysis with thematic batching and saturation tracking.
        
        Workflow:
        1. Relevanzprüfung (filtert relevante Segmente) - SHARED across all coders
        2. Kategorienentwicklung (aus relevanten Segmenten) - SHARED across all coders mit Sättigungscontroller
        3. Deduktive Kodierung (mit entwickelten Kategorien, pro Kodierer)
        """
        config = self.config[AnalysisMode.INDUCTIVE]
        batch_size = config['batch_size']  # Get batch_size from config
        
        # Initialisiere Sättigungscontroller
        from QCA_AID_assets.analysis.saturation_controller import ImprovedSaturationController
        saturation_controller = ImprovedSaturationController('inductive')
        
        # 1. SHARED Relevanzprüfung (wie im Standard-Workflow)
        print(f"   📊 Schritt 1: Relevanzprüfung für {len(segments)} Segmente...")
        print(f"   📋 Forschungsfrage: {research_question}")
        print(f"   🔍 API Call 1: Einfache Relevanzprüfung für {len(segments)} Segmente...")
        from ..core.config import CONFIG
        relevance_threshold = CONFIG.get('RELEVANCE_THRESHOLD', 0.0)
        relevance_results = await self.unified_analyzer.analyze_relevance_simple(
            segments=segments,
            research_question=research_question,
            batch_size=batch_size,
            relevance_threshold=relevance_threshold
        )
        
        # Filtere relevante Segmente (Threshold: 0.3)
        relevant_segments = []
        relevance_map = {}
        for rel_result in relevance_results:
            seg_id = rel_result.get('segment_id', '')
            # Alle Segmente aus relevance_results sind bereits als relevant eingestuft
            seg = next((s for s in segments if s['segment_id'] == seg_id), None)
            if seg:
                relevant_segments.append(seg)
                relevance_map[seg_id] = rel_result
        
        print(f"   ✅ {len(relevant_segments)} von {len(segments)} Segmenten sind relevant")
        
        if not relevant_segments:
            print(f"   ✅ 0 von {len(segments)} Segmenten relevant - KEINE weiteren API Calls nötig")
            return []
        
        # 2. SHARED Kategorienentwicklung mit Batch-Tracking und Sättigungscontroller
        print(f"   🔍 Schritt 2: Kategorienentwicklung aus {len(relevant_segments)} relevanten Segmenten...")
        
        # Berechne Batch-Parameter für echtes Batch-Tracking
        category_batch_size = min(batch_size * 2, 10)  # Größere Batches für Kategorienentwicklung
        total_category_batches = (len(relevant_segments) + category_batch_size - 1) // category_batch_size
        
        print(f"   📦 Kategorienentwicklung in {total_category_batches} Batches (Batch-Größe: {category_batch_size})")
        
        developed_categories = current_categories.copy() if current_categories else {}
        all_category_results = []
        
        # Iteriere über Batches für Kategorienentwicklung
        for batch_idx in range(total_category_batches):
            start_idx = batch_idx * category_batch_size
            end_idx = min(start_idx + category_batch_size, len(relevant_segments))
            batch_segments = relevant_segments[start_idx:end_idx]
            
            batch_number = batch_idx + 1
            material_coverage = end_idx / len(segments) if segments else 0.0
            
            print(f"   🔍 API Call 2.{batch_number}: Induktive Kategorienentwicklung für Batch {batch_number}/{total_category_batches} ({len(batch_segments)} Segmente)")
            
            category_results = await self.unified_analyzer.analyze_batch_inductive(
                segments=batch_segments,
                research_question=research_question,
                batch_size=batch_size,  # Interne Batch-Größe für LLM-Calls
                temperature=self.inductive_temperature,
                existing_categories=developed_categories,
                batch_number=batch_number,
                total_batches=total_category_batches,
                material_coverage=material_coverage
            )
            
            all_category_results.extend(category_results)
            
            # Extrahiere neue Kategorien aus diesem Batch
            batch_categories = self._extract_categories_from_inductive_results(
                category_results, 
                developed_categories
            )
            
            # Prüfe ob neue Kategorien entwickelt wurden
            categories_before = len(developed_categories)
            developed_categories.update(batch_categories)
            categories_after = len(developed_categories)
            new_categories_count = categories_after - categories_before
            
            # Update Sättigungscontroller
            if new_categories_count > 0:
                print(f"   ✅ Batch {batch_number}: {new_categories_count} neue Kategorien entwickelt")
                saturation_controller.reset_stability_counter()
            else:
                print(f"   ⚠️ Batch {batch_number}: Keine neuen Kategorien entwickelt")
                saturation_controller.increment_stability_counter()
            
            # Prüfe Sättigung nach jedem Batch
            saturation_assessment = saturation_controller.assess_saturation(
                current_categories=developed_categories,
                material_percentage=material_coverage * 100,
                batch_count=batch_number,
                total_segments=len(segments)
            )
            
            print(f"   📊 Sättigungsbeurteilung Batch {batch_number}:")
            print(f"      - Theoretische Sättigung: {saturation_assessment['theoretical_saturation']:.2f}")
            print(f"      - Materialabdeckung: {saturation_assessment['material_coverage']:.1%}")
            print(f"      - Stabilität: {saturation_assessment['stability_batches']} Batches")
            print(f"      - Kategorienqualität: {saturation_assessment['category_quality']:.2f}")
            print(f"      - Sättigung erreicht: {'✅ JA' if saturation_assessment['is_saturated'] else '❌ NEIN'}")
            
            # Stoppe bei Sättigung (aber nur wenn mindestens 70% Material verarbeitet)
            if saturation_assessment['is_saturated'] and material_coverage >= 0.7:
                print(f"   🎯 SÄTTIGUNG ERREICHT nach Batch {batch_number}/{total_category_batches}")
                print(f"      Grund: {saturation_assessment['saturation_reason']}")
                break
            elif batch_number < total_category_batches:
                print(f"   ➡️ Setze Kategorienentwicklung fort (Batch {batch_number + 1}/{total_category_batches})")
        
        # Finale Sättigungsauswertung
        final_material_coverage = len(relevant_segments) / len(segments) if segments else 0.0
        final_saturation = saturation_controller.assess_saturation(
            current_categories=developed_categories,
            material_percentage=final_material_coverage * 100,
            batch_count=total_category_batches,
            total_segments=len(segments)
        )
        
        print(f"\n   🎯 FINALE SÄTTIGUNGSAUSWERTUNG:")
        print(f"      - Entwickelte Kategorien: {len(developed_categories)}")
        print(f"      - Theoretische Sättigung: {final_saturation['theoretical_saturation']:.2f}")
        print(f"      - Materialabdeckung: {final_saturation['material_coverage']:.1%}")
        print(f"      - Stabilität: {final_saturation['stability_batches']} stabile Batches")
        print(f"      - Kategorienqualität: {final_saturation['category_quality']:.2f}")
        print(f"      - Endstatus: {'🎯 GESÄTTIGT' if final_saturation['is_saturated'] else '⚠️ NICHT GESÄTTIGT'}")
        
        if not developed_categories:
            print("   ⚠️ Keine Kategorien entwickelt, überspringe Kodierung")
            return []
        
        print(f"   ✅ {len(developed_categories)} Kategorien für Kodierung verfügbar")
        
        # Store developed categories for return to analysis manager
        self._last_developed_categories = developed_categories
        
        # 3. Kodierung mit entwickelten Kategorien (pro Kodierer)
        # Konvertiere CategoryDefinitions zu Dict-Format für analyze_batch
        cat_defs = {
            k: {
                'definition': v.definition if hasattr(v, 'definition') else str(v),
                'subcategories': dict(v.subcategories) if hasattr(v, 'subcategories') and v.subcategories else {}
            }
            for k, v in developed_categories.items()
        }
        
        print(f"   🔍 Schritt 3: Kodierung von {len(relevant_segments)} relevanten Segmenten...")
        
        coding_results = []
        if coder_settings and len(coder_settings) > 1:
            # Multi-Coder: Kodiere mit jedem Kodierer separat (OHNE separate Relevanzprüfung)
            print(f"   🔄 Multi-Coder Mode: Kodierung mit {len(coder_settings)} Kodierern")
            for coder_config in coder_settings:
                coder_id = coder_config.get('coder_id', 'auto_1')
                coder_temp = coder_config.get('temperature', temperature or self.inductive_temperature)
                
                print(f"   🔄 Analysiere mit Kodierer '{coder_id}' (Temperature: {coder_temp})")
                
                # Direkte Kodierung OHNE separate Relevanzprüfung
                coder_results = await self._analyze_inductive_coding_only(
                    segments=relevant_segments,
                    category_definitions=cat_defs,
                    research_question=research_question,
                    coding_rules=coding_rules or [],
                    temperature=coder_temp,
                    coder_id=coder_id,
                    batch_size=batch_size,
                    use_context=use_context,
                    document_paraphrases=document_paraphrases,
                    context_paraphrase_count=context_paraphrase_count
                )
                
                # Store results for reliability analysis
                await self._store_results_for_reliability(coder_results, 'inductive', coder_id, coder_temp)
                
                coding_results.extend(coder_results)
                
        elif coder_settings:
            # Single-Coder
            coder_config = coder_settings[0]
            coder_id = coder_config.get('coder_id', 'auto_1')
            coder_temp = coder_config.get('temperature', temperature or self.inductive_temperature)
            
            print(f"   🔄 Single-Coder Mode: Kodierung mit '{coder_id}' (Temperature: {coder_temp})")
            
            # Direkte Kodierung OHNE separate Relevanzprüfung
            coder_results = await self._analyze_inductive_coding_only(
                segments=relevant_segments,
                category_definitions=cat_defs,
                research_question=research_question,
                coding_rules=coding_rules or [],
                temperature=coder_temp,
                coder_id=coder_id,
                batch_size=batch_size,
                use_context=use_context,
                document_paraphrases=document_paraphrases,
                context_paraphrase_count=context_paraphrase_count
            )
            
            # Store results for reliability analysis
            await self._store_results_for_reliability(coder_results, 'inductive', coder_id, coder_temp)
            
            coding_results.extend(coder_results)
        else:
            # Fallback: Standard-Temperature ohne spezifische Kodierer
            print(f"   🔄 Fallback Mode: Kodierung ohne spezifische Kodierer")
            fallback_results = await self._analyze_inductive_coding_only(
                segments=relevant_segments,
                category_definitions=cat_defs,
                research_question=research_question,
                coding_rules=coding_rules or [],
                temperature=temperature or self.inductive_temperature,
                coder_id='auto_fallback',
                batch_size=batch_size,
                use_context=use_context,
                document_paraphrases=document_paraphrases,
                context_paraphrase_count=context_paraphrase_count
            )
            
            # Store results for reliability analysis
            await self._store_results_for_reliability(fallback_results, 'inductive', 'auto_fallback', temperature or self.inductive_temperature)
            
            coding_results.extend(fallback_results)
        
        return coding_results
    
    async def _analyze_inductive_coding_only(self,
                                           segments: List[Dict[str, Any]],
                                           category_definitions: Dict[str, str],
                                           research_question: str,
                                           coding_rules: List[str],
                                           temperature: Optional[float],
                                           coder_id: str,
                                           batch_size: int,
                                           use_context: bool = False,
                                           document_paraphrases: Optional[Dict[str, List[str]]] = None,
                                           context_paraphrase_count: int = 3) -> List[Dict[str, Any]]:
        """
        Inductive coding only - no relevance check, just coding with developed categories.
        """
        results = []
        
        # Process segments in batches for coding
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(segments) + batch_size - 1) // batch_size
            
            print(f"   📦 Batch {batch_num}/{total_batches}: {len(batch)} Segmente")
            
            # FIX: Verwende progressive Kontext-Paraphrasen pro Segment
            context_paraphrases = []
            if use_context and document_paraphrases and batch:
                first_segment_id = batch[0]['segment_id']
                context_paraphrases = self._get_progressive_context_paraphrases(
                    first_segment_id, document_paraphrases, context_paraphrase_count
                )
                
                if context_paraphrases:
                    doc_name = self._extract_doc_name_from_segment_id(first_segment_id)
                    chunk_num = self._extract_chunk_number_from_segment_id(first_segment_id)
                    print(f"      📝 Nutze {len(context_paraphrases)} progressive Kontext-Paraphrasen für Dokument '{doc_name}' (Chunk {chunk_num}) (inductive)")
                else:
                    doc_name = self._extract_doc_name_from_segment_id(first_segment_id)
                    chunk_num = self._extract_chunk_number_from_segment_id(first_segment_id)
                    print(f"      📝 Keine Kontext-Paraphrasen für Dokument '{doc_name}' Chunk {chunk_num} (erste Chunks oder nicht verfügbar) (inductive)")
            
            # Use batch processing for efficient coding
            batch_results = await self.unified_analyzer.analyze_batch(
                segments=batch,
                category_definitions=category_definitions,
                research_question=research_question,
                coding_rules=coding_rules,
                batch_size=batch_size,
                temperature=temperature,
                context_paraphrases=context_paraphrases if context_paraphrases else None
            )
            
            # FIX: Implementiere korrekte Mehrfachkodierung für inductive Mode - KEINE separaten API Calls!
            expanded_results = []
            for r in batch_results:
                if hasattr(r, 'requires_multiple_coding') and r.requires_multiple_coding:
                    # Mehrfachkodierung erforderlich - erstelle mehrere Ergebnisse aus EINER Antwort
                    relevant_categories = []
                    if hasattr(r, 'relevance_scores') and r.relevance_scores:
                        for cat_name, score in r.relevance_scores.items():
                            if score >= self.unified_analyzer.multiple_coding_threshold:
                                relevant_categories.append({
                                    'category': cat_name,
                                    'relevance_score': score
                                })
                    
                    if len(relevant_categories) > 1:
                        print(f"   🔁 Mehrfachkodierung für Segment {r.segment_id}: {len(relevant_categories)} Kategorien (inductive)")
                        
                        # Erstelle separate Ergebnisse für jede relevante Kategorie (OHNE zusätzliche API Calls)
                        for i, cat_info in enumerate(relevant_categories, 1):
                            target_category = cat_info['category']
                            multiple_segment_id = f"{r.segment_id}-{i}"
                            
                            # Finde passende Subkategorien für diese Kategorie
                            target_subcategories = []
                            if hasattr(r, 'subcategories') and r.subcategories:
                                # Filtere Subkategorien die zu dieser Kategorie gehören
                                for subcat in r.subcategories:
                                    if isinstance(subcat, str) and target_category.lower() in subcat.lower():
                                        target_subcategories.append(subcat)
                                    elif isinstance(subcat, dict) and subcat.get('category') == target_category:
                                        target_subcategories.append(subcat.get('name', ''))
                                
                                # Fallback: Verwende alle Subkategorien wenn keine spezifischen gefunden
                                if not target_subcategories:
                                    target_subcategories = r.subcategories
                            
                            formatted_result = {
                                'segment_id': multiple_segment_id,
                                'result': {
                                    'primary_category': target_category,
                                    'confidence': cat_info['relevance_score'],  # Verwende Relevanz-Score als Konfidenz
                                    'all_categories': [target_category],
                                    'subcategories': target_subcategories,
                                    'keywords': r.keywords if hasattr(r, 'keywords') else '',
                                    'paraphrase': r.paraphrase if hasattr(r, 'paraphrase') else '',
                                    'justification': f"[Mehrfachkodierung {i}/{len(relevant_categories)}] {r.justification if hasattr(r, 'justification') else ''}",
                                    'coder_id': coder_id,
                                    'category_preselection_used': False,
                                    'preferred_categories': [],
                                    'preselection_reasoning': '',
                                    # Mehrfachkodierungs-Metadaten
                                    'multiple_coding_instance': i,
                                    'total_coding_instances': len(relevant_categories),
                                    'target_category': target_category,
                                    'original_segment_id': r.segment_id,
                                    'is_multiple_coding': True
                                },
                                'analysis_mode': 'inductive',
                                'timestamp': asyncio.get_event_loop().time()
                            }
                            expanded_results.append(formatted_result)
                            print(f"      📝 Mehrfachkodierung {i}: {target_category} (Konfidenz: {cat_info['relevance_score']:.2f})")
                    else:
                        # Nur eine Kategorie über Schwellenwert - normale Kodierung
                        expanded_results.append(self._format_single_coding_result_inductive(r, coder_id))
                else:
                    # Keine Mehrfachkodierung erforderlich - normale Kodierung
                    expanded_results.append(self._format_single_coding_result_inductive(r, coder_id))
            
            results.extend(expanded_results)
        
        return results
    
    def _extract_categories_from_inductive_results(self, 
                                                   category_results: List[Dict[str, Any]],
                                                   current_categories: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrahiert entwickelte Kategorien aus Inductive-Ergebnissen.
        
        Konvertiert Kategorienamen in CategoryDefinition-ähnliches Format.
        Verarbeitet sowohl das neue detaillierte Format als auch das alte Format.
        """
        from datetime import datetime
        from QCA_AID_assets.core.data_models import CategoryDefinition
        
        developed = {}
        
        # Sammle Sättigungsmetriken für Logging
        saturation_metrics = []
        
        for result in category_results:
            # Verarbeite development_assessment falls vorhanden
            if 'development_assessment' in result:
                assessment = result['development_assessment']
                saturation_metrics.append(assessment)
                
                # Log Sättigungsmetriken
                saturation = assessment.get('theoretical_saturation', 0.0)
                recommendation = assessment.get('recommendation', 'continue')
                print(f"   📊 Sättigungsmetriken: Sättigung={saturation:.2f}, Empfehlung={recommendation}")
            
            # Verarbeite Kategorien - unterstütze beide Formate
            categories = []
            category_definitions = {}
            subcategories = {}
            
            if 'categories' in result:
                # Neues Format
                categories = result.get('categories', [])
                category_definitions = result.get('category_definitions', {})
                subcategories = result.get('subcategories', {})
            elif 'new_categories' in result:
                # Standard-Analyse Format
                new_categories = result.get('new_categories', [])
                for cat_data in new_categories:
                    cat_name = cat_data.get('name', '')
                    if cat_name:
                        categories.append(cat_name)
                        category_definitions[cat_name] = cat_data.get('definition', f"Induktiv entwickelte Kategorie: {cat_name}")
                        
                        # Verarbeite Subkategorien
                        cat_subcategories = cat_data.get('subcategories', [])
                        if cat_subcategories:
                            subcategories[cat_name] = cat_subcategories
            
            # Erstelle CategoryDefinition-Objekte
            for cat_name in categories:
                if cat_name and cat_name not in developed and cat_name not in current_categories:
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    
                    # Erstelle Subkategorien-Dictionary
                    subcat_dict = {}
                    if cat_name in subcategories:
                        for subcat_data in subcategories[cat_name]:
                            if isinstance(subcat_data, dict):
                                subcat_name = subcat_data.get('name', '')
                                subcat_definition = subcat_data.get('definition', f"Subkategorie: {subcat_name}")
                            else:
                                subcat_name = str(subcat_data)
                                subcat_definition = f"Subkategorie: {subcat_name}"
                            
                            if subcat_name:
                                subcat_dict[subcat_name] = CategoryDefinition(
                                    name=subcat_name,
                                    definition=subcat_definition,
                                    examples=[],
                                    rules=[],
                                    subcategories={},
                                    added_date=current_date,
                                    modified_date=current_date
                                )
                    
                    # Erstelle Hauptkategorie
                    developed[cat_name] = CategoryDefinition(
                        name=cat_name,
                        definition=category_definitions.get(cat_name, f"Induktiv entwickelte Kategorie: {cat_name}"),
                        examples=[],
                        rules=[],
                        subcategories=subcat_dict,
                        added_date=current_date,
                        modified_date=current_date
                    )
                    
                    print(f"   ✅ Neue Kategorie entwickelt: '{cat_name}' mit {len(subcat_dict)} Subkategorien")
        
        # Log finale Sättigungsmetriken
        if saturation_metrics:
            avg_saturation = sum(m.get('theoretical_saturation', 0.0) for m in saturation_metrics) / len(saturation_metrics)
            recommendations = [m.get('recommendation', 'continue') for m in saturation_metrics]
            most_common_rec = max(set(recommendations), key=recommendations.count)
            
            print(f"   📊 Finale Sättigungsmetriken:")
            print(f"      - Durchschnittliche Sättigung: {avg_saturation:.2f}")
            print(f"      - Häufigste Empfehlung: {most_common_rec}")
        
        return developed
    
    async def _analyze_abductive(self,
                                 segments: List[Dict[str, Any]],
                                 category_definitions: Dict[str, str],
                                 research_question: str,
                                 current_categories: Optional[Dict[str, Any]] = None,
                                 coding_rules: Optional[List[str]] = None,
                                 coder_settings: Optional[List[Dict[str, Any]]] = None,
                                 temperature: Optional[float] = None,
                                 use_context: bool = False,
                                 document_paraphrases: Optional[Dict[str, List[str]]] = None,
                                 context_paraphrase_count: int = 3,
                                 **kwargs) -> List[Dict[str, Any]]:
        """
        Optimized abductive analysis with hypothesis batching.
        
        Workflow:
        1. Relevanzprüfung mit Kategoriepräferenzen (filtert relevante Segmente)
        2. Subkategorien-Entwicklung (erweitert bestehende Kategorien induktiv)
        3. Kodierung mit erweiterten Kategorien (pro Kodierer)
        """
        config = self.config[AnalysisMode.ABDUCTIVE]
        batch_size = config['batch_size']  # Get batch_size from config
        
        # 1. Relevanzprüfung (einfach, wie in Standard-Analyse)
        print(f"   📊 Schritt 1: Relevanzprüfung für {len(segments)} Segmente...")
        print(f"   📋 Forschungsfrage: {research_question}")
        print(f"   🔍 API Call 1: Einfache Relevanzprüfung für {len(segments)} Segmente...")
        from ..core.config import CONFIG
        relevance_threshold = CONFIG.get('RELEVANCE_THRESHOLD', 0.0)
        relevance_results = await self.unified_analyzer.analyze_relevance_simple(
            segments=segments,
            research_question=research_question,
            batch_size=batch_size,
            relevance_threshold=relevance_threshold
        )
        
        # Debug: Show relevance results with better logging
        print(f"   🔍 DEBUG: Einfache Relevance check returned {len(relevance_results)} results")
        relevant_count = 0
        for i, rel_result in enumerate(relevance_results):
            seg_id = rel_result.get('segment_id', 'MISSING')
            research_rel = rel_result.get('research_relevance', 0.0)
            reasoning = rel_result.get('relevance_reasoning', '')
            
            # Truncate long segment IDs and reasoning for readability
            seg_id_short = seg_id[:30] + '...' if len(seg_id) > 30 else seg_id
            reasoning_short = reasoning[:80] + '...' if len(reasoning) > 80 else reasoning
            
            status = "✅ RELEVANT" if research_rel >= 0.0 else "❌ NICHT RELEVANT"
            if research_rel >= 0.0:  # ALLE als relevant markierten Segmente kodieren
                relevant_count += 1
            
            print(f"   🔍 Segment {i+1}: {seg_id_short}")
            print(f"      └─ {status} (Score: {research_rel:.2f})")
            print(f"      └─ Begründung: {reasoning_short}")
        
        print(f"   📊 Relevanz-Zusammenfassung: {relevant_count}/{len(relevance_results)} Segmente relevant (Threshold: ≥0.3)")
        
        # Filtere relevante Segmente (Threshold: 0.3)
        relevant_segments = []
        relevance_map = {}
        
        for rel_result in relevance_results:
            seg_id = rel_result.get('segment_id', '')
            # Alle Segmente aus relevance_results sind bereits als relevant eingestuft
            seg = next((s for s in segments if s['segment_id'] == seg_id), None)
            if seg:
                relevant_segments.append(seg)
                relevance_map[seg_id] = rel_result
        
        print(f"   ✅ {len(relevant_segments)} von {len(segments)} Segmenten sind relevant")
        
        if not relevant_segments:
            print(f"   ✅ 0 von {len(segments)} Segmenten relevant - KEINE weiteren API Calls nötig")
            return []
        
        # 1.5. Kategoriepräferenzen für relevante Segmente bestimmen
        print(f"   📊 Schritt 1.5: Kategoriepräferenzen für {len(relevant_segments)} relevante Segmente...")
        print(f"   🔍 API Call 2: Kategoriepräferenzen für {len(relevant_segments)} relevante Segmente...")
        category_preference_results = await self.unified_analyzer.analyze_category_preferences(
            segments=relevant_segments,  # Nur relevante Segmente!
            category_definitions=category_definitions,
            research_question=research_question,
            coding_rules=coding_rules or [],
            batch_size=batch_size
        )
        
        # Sammle Kategoriepräferenzen mit verbessertem Logging
        category_preselections = {}
        for pref_result in category_preference_results:
            seg_id = pref_result.get('segment_id', '')
            if seg_id:  # Segment ist bereits als relevant bestätigt
                # FIX: Fallback für unterschiedliche Feldnamen zwischen Modi
                top_categories = pref_result.get('top_categories', [])
                if not top_categories:
                    top_categories = pref_result.get('preferred_categories', [])
                
                category_preferences = pref_result.get('category_preferences', {})
                
                reasoning = pref_result.get('preference_reasoning', '')
                if not reasoning:
                    reasoning = pref_result.get('reasoning', '')
                
                category_preselections[seg_id] = {
                    'preferred_categories': top_categories,
                    'category_preferences': category_preferences,
                    'relevance_scores': category_preferences,
                    'reasoning': reasoning
                }
                
                # Debug: Log category preferences for each segment
                seg_id_short = seg_id[:30] + '...' if len(seg_id) > 30 else seg_id
                print(f"   🎯 Segment: {seg_id_short}")
                if top_categories:
                    print(f"      └─ Präferierte Kategorien: {', '.join(top_categories)}")
                    # Show scores for preferred categories
                    for cat in top_categories:
                        score = category_preferences.get(cat, 0.0)
                        print(f"         • {cat}: {score:.2f}")
                else:
                    print(f"      └─ Keine starken Kategoriepräferenzen (alle Scores < 0.6)")
        
        if category_preselections:
            preselection_stats = {}
            for prefs in category_preselections.values():
                for cat in prefs['preferred_categories']:
                    preselection_stats[cat] = preselection_stats.get(cat, 0) + 1
            print(f"   📊 Kategorie-Präferenzen Zusammenfassung: {dict(preselection_stats)}")
        else:
            print(f"   ⚠️ Keine Kategoriepräferenzen ermittelt - alle Segmente haben schwache Kategorie-Scores")
        
        if not relevant_segments:
            return []
        
        # 2. Subkategorien-Entwicklung mit Sättigungscontroller
        print(f"   🔍 Schritt 2: Erweitere bestehende Kategorien um neue Subkategorien (aus {len(relevant_segments)} relevanten Segmenten)...")
        
        # Initialisiere Sättigungscontroller für abductive Mode
        from QCA_AID_assets.analysis.saturation_controller import ImprovedSaturationController
        saturation_controller = ImprovedSaturationController('abductive')
        
        # Berechne Batch-Parameter für echtes Batch-Tracking
        subcategory_batch_size = min(batch_size * 2, 8)  # Kleinere Batches für Subkategorien-Entwicklung
        total_subcategory_batches = (len(relevant_segments) + subcategory_batch_size - 1) // subcategory_batch_size
        
        print(f"   📦 Subkategorien-Entwicklung in {total_subcategory_batches} Batches (Batch-Größe: {subcategory_batch_size})")
        
        extended_categories = current_categories.copy() if current_categories else {}
        all_subcategory_results = []
        
        # Iteriere über Batches für Subkategorien-Entwicklung
        for batch_idx in range(total_subcategory_batches):
            start_idx = batch_idx * subcategory_batch_size
            end_idx = min(start_idx + subcategory_batch_size, len(relevant_segments))
            batch_segments = relevant_segments[start_idx:end_idx]
            
            batch_number = batch_idx + 1
            material_coverage = end_idx / len(segments) if segments else 0.0
            
            print(f"   🔍 API Call 3.{batch_number}: Abduktive Subkategorien-Entwicklung für Batch {batch_number}/{total_subcategory_batches} ({len(batch_segments)} Segmente)")
            
            abductive_temp = temperature or kwargs.get('temperature', self.inductive_temperature)
            subcategory_results = await self.unified_analyzer.analyze_batch_abductive(
                segments=batch_segments,
                category_definitions=category_definitions,
                research_question=research_question,
                batch_size=batch_size,  # Interne Batch-Größe für LLM-Calls
                temperature=abductive_temp
            )
            
            all_subcategory_results.extend(subcategory_results)
            
            # Erweitere Kategorien mit neuen Subkategorien aus diesem Batch
            categories_before = len(extended_categories)
            subcategories_before = sum(len(getattr(cat, 'subcategories', {})) for cat in extended_categories.values() if hasattr(cat, 'subcategories'))
            
            batch_extended_categories = self._extend_categories_with_subcategories(
                subcategory_results,
                extended_categories,
                category_definitions
            )
            
            extended_categories.update(batch_extended_categories)
            
            categories_after = len(extended_categories)
            subcategories_after = sum(len(getattr(cat, 'subcategories', {})) for cat in extended_categories.values() if hasattr(cat, 'subcategories'))
            
            new_subcategories_count = subcategories_after - subcategories_before
            
            # Update Sättigungscontroller
            if new_subcategories_count > 0:
                print(f"   ✅ Batch {batch_number}: {new_subcategories_count} neue Subkategorien entwickelt")
                saturation_controller.reset_stability_counter()
            else:
                print(f"   ⚠️ Batch {batch_number}: Keine neuen Subkategorien entwickelt")
                saturation_controller.increment_stability_counter()
            
            # Prüfe Sättigung nach jedem Batch
            saturation_assessment = saturation_controller.assess_saturation(
                current_categories=extended_categories,
                material_percentage=material_coverage * 100,
                batch_count=batch_number,
                total_segments=len(segments)
            )
            
            print(f"   📊 Sättigungsbeurteilung Batch {batch_number}:")
            print(f"      - Theoretische Sättigung: {saturation_assessment['theoretical_saturation']:.2f}")
            print(f"      - Materialabdeckung: {saturation_assessment['material_coverage']:.1%}")
            print(f"      - Stabilität: {saturation_assessment['stability_batches']} Batches")
            print(f"      - Kategorienqualität: {saturation_assessment['category_quality']:.2f}")
            print(f"      - Sättigung erreicht: {'✅ JA' if saturation_assessment['is_saturated'] else '❌ NEIN'}")
            
            # Stoppe bei Sättigung (aber nur wenn mindestens 60% Material verarbeitet - niedriger für abductive)
            if saturation_assessment['is_saturated'] and material_coverage >= 0.6:
                print(f"   🎯 SUBKATEGORIEN-SÄTTIGUNG ERREICHT nach Batch {batch_number}/{total_subcategory_batches}")
                print(f"      Grund: {saturation_assessment['saturation_reason']}")
                break
            elif batch_number < total_subcategory_batches:
                print(f"   ➡️ Setze Subkategorien-Entwicklung fort (Batch {batch_number + 1}/{total_subcategory_batches})")
        
        # Finale Sättigungsauswertung für Subkategorien
        final_material_coverage = len(relevant_segments) / len(segments) if segments else 0.0
        final_saturation = saturation_controller.assess_saturation(
            current_categories=extended_categories,
            material_percentage=final_material_coverage * 100,
            batch_count=total_subcategory_batches,
            total_segments=len(segments)
        )
        
        print(f"\n   🎯 FINALE SUBKATEGORIEN-SÄTTIGUNGSAUSWERTUNG:")
        print(f"      - Erweiterte Kategorien: {len(extended_categories)}")
        total_subcategories = sum(len(getattr(cat, 'subcategories', {})) for cat in extended_categories.values() if hasattr(cat, 'subcategories'))
        print(f"      - Gesamte Subkategorien: {total_subcategories}")
        print(f"      - Theoretische Sättigung: {final_saturation['theoretical_saturation']:.2f}")
        print(f"      - Materialabdeckung: {final_saturation['material_coverage']:.1%}")
        print(f"      - Stabilität: {final_saturation['stability_batches']} stabile Batches")
        print(f"      - Kategorienqualität: {final_saturation['category_quality']:.2f}")
        print(f"      - Endstatus: {'🎯 GESÄTTIGT' if final_saturation['is_saturated'] else '⚠️ NICHT GESÄTTIGT'}")
        
        # Store extended categories for return to analysis manager
        self._last_extended_categories = extended_categories
        
        # 3. Abduktive Kodierung mit erweiterten Kategorien (OHNE separate Relevanzprüfung)
        # FIX: Konvertiere CategoryDefinitions zu Dict-Format für analyze_batch
        # Stelle sicher, dass alle CategoryDefinition Objekte korrekt serialisiert werden
        cat_defs = {}
        for k, v in extended_categories.items():
            if hasattr(v, 'definition'):
                # CategoryDefinition object
                cat_defs[k] = {
                    'definition': v.definition,
                    'subcategories': {}
                }
                # Serialisiere auch Subkategorien
                if hasattr(v, 'subcategories') and v.subcategories:
                    for subcat_name, subcat_obj in v.subcategories.items():
                        if hasattr(subcat_obj, 'definition'):
                            cat_defs[k]['subcategories'][subcat_name] = subcat_obj.definition
                        else:
                            cat_defs[k]['subcategories'][subcat_name] = str(subcat_obj)
            elif isinstance(v, dict):
                # Already a dict
                cat_defs[k] = {
                    'definition': v.get('definition', str(v)),
                    'subcategories': v.get('subcategories', {})
                }
            else:
                # String or other type
                cat_defs[k] = {
                    'definition': str(v),
                    'subcategories': {}
                }
        
        print(f"   🔧 DEBUG: Serialisierte {len(cat_defs)} Kategorien für Cache-kompatible Kodierung")
        
        coding_results = []
        if coder_settings and len(coder_settings) > 1:
            print(f"   🔄 Multi-Coder Mode: Kodierung mit {len(coder_settings)} Kodierern")
            # Multi-Coder: Kodiere mit jedem Kodierer separat
            for coder_config in coder_settings:
                coder_id = coder_config.get('coder_id', 'auto_1')
                coder_temp = coder_config.get('temperature', temperature)
                
                print(f"   🔄 Analysiere mit Kodierer '{coder_id}' (Temperature: {coder_temp})")
                
                # Direkte Kodierung OHNE separate Relevanzprüfung
                coder_results = await self._batch_analyze_abductive_direct(
                    batch=relevant_segments,
                    category_definitions=cat_defs,
                    research_question=research_question,
                    coding_rules=coding_rules or [],
                    temperature=coder_temp,
                    coder_id=coder_id,
                    category_preselections=category_preselections,
                    batch_size=batch_size,
                    use_context=use_context,
                    document_paraphrases=document_paraphrases,
                    context_paraphrase_count=context_paraphrase_count
                )
                
                # Markiere als abductive mode
                for result in coder_results:
                    if 'result' in result:
                        result['result']['analysis_mode'] = 'abductive'
                    else:
                        result['analysis_mode'] = 'abductive'
                
                # Store results for reliability analysis
                await self._store_results_for_reliability(coder_results, 'abductive', coder_id, coder_temp)
                
                coding_results.extend(coder_results)
                
        elif coder_settings:
            # Single-Coder
            coder_config = coder_settings[0]
            coder_id = coder_config.get('coder_id', 'auto_1')
            coder_temp = coder_config.get('temperature', temperature)
            
            print(f"   🔄 Single-Coder Mode: Kodierung mit '{coder_id}' (Temperature: {coder_temp})")
            
            # Direkte Kodierung OHNE separate Relevanzprüfung
            coder_results = await self._batch_analyze_abductive_direct(
                batch=relevant_segments,
                category_definitions=cat_defs,
                research_question=research_question,
                coding_rules=coding_rules or [],
                temperature=coder_temp,
                coder_id=coder_id,
                category_preselections=category_preselections,
                batch_size=batch_size,
                use_context=use_context,
                document_paraphrases=document_paraphrases,
                context_paraphrase_count=context_paraphrase_count
            )
            
            # Markiere als abductive mode
            # Markiere als abductive mode
            for result in coder_results:
                if 'result' in result:
                    result['result']['analysis_mode'] = 'abductive'
                else:
                    result['analysis_mode'] = 'abductive'
            
            # Store results for reliability analysis
            await self._store_results_for_reliability(coder_results, 'abductive', coder_id, coder_temp)
            
            coding_results.extend(coder_results)
        else:
            # Fallback: Standard-Temperature ohne spezifische Kodierer
            print(f"   🔄 Fallback Mode: Kodierung ohne spezifische Kodierer")
            fallback_results = await self._batch_analyze_abductive_direct(
                batch=relevant_segments,
                category_definitions=cat_defs,
                research_question=research_question,
                coding_rules=coding_rules or [],
                temperature=temperature,
                coder_id='auto_fallback',
                category_preselections=category_preselections,
                batch_size=batch_size,
                use_context=use_context,
                document_paraphrases=document_paraphrases,
                context_paraphrase_count=context_paraphrase_count
            )
            
            # Markiere als abductive mode
            for result in fallback_results:
                if 'result' in result:
                    result['result']['analysis_mode'] = 'abductive'
                else:
                    result['analysis_mode'] = 'abductive'
            
            # Store results for reliability analysis
            await self._store_results_for_reliability(fallback_results, 'abductive', 'auto_fallback', temperature)
            
            coding_results.extend(fallback_results)
        
        # Zusammenfassung der Kodierungen pro Coder
        if coding_results:
            print(f"\n   📊 KODIERUNGS-ZUSAMMENFASSUNG:")
            coder_summary = {}
            category_summary = {}
            
            for result in coding_results:
                coder_id = result.get('result', {}).get('coder_id', 'unknown')
                category = result.get('result', {}).get('primary_category', 'unknown')
                
                # Count by coder
                if coder_id not in coder_summary:
                    coder_summary[coder_id] = 0
                coder_summary[coder_id] += 1
                
                # Count by category
                if category not in category_summary:
                    category_summary[category] = 0
                category_summary[category] += 1
            
            # Show coder summary
            for coder_id, count in coder_summary.items():
                print(f"      • {coder_id}: {count} Kodierungen")
            
            # Show category summary
            print(f"   📋 KATEGORIEN-VERTEILUNG:")
            for category, count in sorted(category_summary.items()):
                print(f"      • {category}: {count} Segmente")
        
        return coding_results
    
    def _extend_categories_with_subcategories(self,
                                             subcategory_results: List[Dict[str, Any]],
                                             current_categories: Dict[str, Any],
                                             category_definitions: Dict[str, str]) -> Dict[str, Any]:
        """
        Erweitert bestehende Kategorien um neue Subkategorien aus Abductive-Ergebnissen.
        Verarbeitet das neue abductive Schema mit extended_categories.
        """
        
        extended = current_categories.copy() if current_categories else {}
        
        # Handle new abductive schema format
        if subcategory_results and len(subcategory_results) > 0:
            # Extract extended_categories from the first result (they should all have the same)
            extended_categories_data = None
            for result in subcategory_results:
                if "_extended_categories" in result:
                    extended_categories_data = result["_extended_categories"]
                    break
            
            if extended_categories_data:
                # Process extended_categories to add new subcategories
                for hauptkategorie_name, kategorie_data in extended_categories_data.items():
                    new_subcategories = kategorie_data.get("new_subcategories", [])
                    
                    if new_subcategories and hauptkategorie_name in category_definitions:
                        # Ensure main category exists
                        if hauptkategorie_name not in extended:
                            current_date = datetime.now().strftime("%Y-%m-%d")
                            extended[hauptkategorie_name] = CategoryDefinition(
                                name=hauptkategorie_name,
                                definition=category_definitions[hauptkategorie_name],
                                examples=[],
                                rules=[],
                                subcategories={},
                                added_date=current_date,
                                modified_date=current_date
                            )
                        
                        # Add new subcategories
                        cat = extended[hauptkategorie_name]
                        if hasattr(cat, 'subcategories'):
                            for subcat_data in new_subcategories:
                                subcat_name = subcat_data.get("name")
                                subcat_definition = subcat_data.get("definition", f"Abduktiv entwickelte Subkategorie: {subcat_name}")
                                
                                if subcat_name and subcat_name not in cat.subcategories:
                                    current_date = datetime.now().strftime("%Y-%m-%d")
                                    cat.subcategories[subcat_name] = CategoryDefinition(
                                        name=subcat_name,
                                        definition=subcat_definition,
                                        examples=list(subcat_data.get("evidence", [])),
                                        rules=[],
                                        subcategories={},
                                        added_date=current_date,
                                        modified_date=current_date
                                    )
                                    print(f"      ✅ Neue Subkategorie '{subcat_name}' zu '{hauptkategorie_name}' hinzugefügt")
            else:
                # Fallback: Process segment assignments for subcategories
                for result in subcategory_results:
                    main_cat = result.get('main_category')
                    subcategories = result.get('subcategories', [])
                    
                    if main_cat and subcategories:
                        # Ensure main category exists
                        if main_cat not in extended:
                            if main_cat in category_definitions:
                                current_date = datetime.now().strftime("%Y-%m-%d")
                                extended[main_cat] = CategoryDefinition(
                                    name=main_cat,
                                    definition=category_definitions[main_cat],
                                    examples=[],
                                    rules=[],
                                    subcategories={},
                                    added_date=current_date,
                                    modified_date=current_date
                                )
                            else:
                                continue
                        
                        # Add subcategories to existing category
                        cat = extended[main_cat]
                        if hasattr(cat, 'subcategories'):
                            for subcat in subcategories:
                                if subcat not in cat.subcategories:
                                    current_date = datetime.now().strftime("%Y-%m-%d")
                                    cat.subcategories[subcat] = CategoryDefinition(
                                        name=subcat,
                                        definition=f"Abduktiv entwickelte Subkategorie: {subcat}",
                                        examples=[],
                                        rules=[],
                                        subcategories={},
                                        added_date=current_date,
                                        modified_date=current_date
                                    )
        
        return extended
    
    async def _batch_analyze_abductive_direct(self,
                                             batch: List[Dict[str, Any]],
                                             category_definitions: Dict[str, str],
                                             research_question: str,
                                             coding_rules: List[str],
                                             temperature: Optional[float] = None,
                                             coder_id: str = 'auto_1',
                                             category_preselections: Optional[Dict[str, Dict]] = None,
                                             batch_size: int = 5,
                                             use_context: bool = False,
                                             document_paraphrases: Optional[Dict[str, List[str]]] = None,
                                             context_paraphrase_count: int = 3) -> List[Dict[str, Any]]:
        """
        Direkte abduktive Kodierung ohne separate Relevanzprüfung.
        
        Diese Methode führt die Kodierung mit bereits gefilterten relevanten Segmenten durch
        und verwendet die Kategoriepräferenzen aus der vorherigen Relevanzprüfung.
        
        Args:
            batch: Liste der bereits als relevant identifizierten Segmente
            category_definitions: Erweiterte Kategoriedefinitionen (mit Subkategorien)
            research_question: Forschungsfrage
            coding_rules: Kodierregeln
            temperature: Temperature für LLM-Aufrufe
            coder_id: Kodierer-ID
            category_preselections: Kategoriepräferenzen aus der Relevanzprüfung
        """
        new_results = []
        
        if not batch:
            return new_results
        
        print(f"   🔍 DEBUG: Direkte abduktive Kodierung für {len(batch)} Segmente")
        
        # FIX: Verwende progressive Kontext-Paraphrasen pro Segment
        context_paraphrases = []
        if use_context and document_paraphrases and batch:
            first_segment_id = batch[0]['segment_id']
            context_paraphrases = self._get_progressive_context_paraphrases(
                first_segment_id, document_paraphrases, context_paraphrase_count
            )
            
            if context_paraphrases:
                doc_name = self._extract_doc_name_from_segment_id(first_segment_id)
                chunk_num = self._extract_chunk_number_from_segment_id(first_segment_id)
                print(f"      📝 Nutze {len(context_paraphrases)} progressive Kontext-Paraphrasen für Dokument '{doc_name}' (Chunk {chunk_num}) (abductive)")
            else:
                doc_name = self._extract_doc_name_from_segment_id(first_segment_id)
                chunk_num = self._extract_chunk_number_from_segment_id(first_segment_id)
                print(f"      📝 Keine Kontext-Paraphrasen für Dokument '{doc_name}' Chunk {chunk_num} (erste Chunks oder nicht verfügbar) (abductive)")
        
        # Verwende Batch-Processing für effiziente Kodierung
        batch_results = await self.unified_analyzer.analyze_batch(
            segments=batch,
            category_definitions=category_definitions,
            research_question=research_question,
            coding_rules=coding_rules,
            batch_size=batch_size,  # Use configured batch_size
            temperature=temperature,
            context_paraphrases=context_paraphrases if context_paraphrases else None
        )
        
        print(f"   🔍 DEBUG: Batch-Kodierung ergab {len(batch_results)} Ergebnisse")
        
        # FIX: Implementiere korrekte Mehrfachkodierung für abductive Mode - KEINE separaten API Calls!
        expanded_results = []
        for r in batch_results:
            # Get category preselection info for this segment
            seg_prefs = category_preselections.get(r.segment_id, {}) if category_preselections else {}
            preferred_cats = seg_prefs.get('preferred_categories', [])
            
            if hasattr(r, 'requires_multiple_coding') and r.requires_multiple_coding:
                # Mehrfachkodierung erforderlich - erstelle mehrere Ergebnisse aus EINER Antwort
                relevant_categories = []
                if hasattr(r, 'relevance_scores') and r.relevance_scores:
                    for cat_name, score in r.relevance_scores.items():
                        if score >= self.unified_analyzer.multiple_coding_threshold:
                            relevant_categories.append({
                                'category': cat_name,
                                'relevance_score': score
                            })
                
                if len(relevant_categories) > 1:
                    print(f"   🔁 Mehrfachkodierung für Segment {r.segment_id}: {len(relevant_categories)} Kategorien (abductive)")
                    
                    # Erstelle separate Ergebnisse für jede relevante Kategorie (OHNE zusätzliche API Calls)
                    for i, cat_info in enumerate(relevant_categories, 1):
                        target_category = cat_info['category']
                        multiple_segment_id = f"{r.segment_id}-{i}"
                        
                        # Finde passende Subkategorien für diese Kategorie
                        target_subcategories = []
                        if hasattr(r, 'subcategories') and r.subcategories:
                            # Filtere Subkategorien die zu dieser Kategorie gehören
                            for subcat in r.subcategories:
                                if isinstance(subcat, str) and target_category.lower() in subcat.lower():
                                    target_subcategories.append(subcat)
                                elif isinstance(subcat, dict) and subcat.get('category') == target_category:
                                    target_subcategories.append(subcat.get('name', ''))
                            
                            # Fallback: Verwende alle Subkategorien wenn keine spezifischen gefunden
                            if not target_subcategories:
                                target_subcategories = r.subcategories
                        
                        formatted_res = {
                            'segment_id': multiple_segment_id,
                            'result': {
                                'primary_category': target_category,
                                'confidence': cat_info['relevance_score'],  # Verwende Relevanz-Score als Konfidenz
                                'all_categories': [target_category],
                                'subcategories': target_subcategories,
                                'keywords': r.keywords if hasattr(r, 'keywords') else '',
                                'paraphrase': r.paraphrase if hasattr(r, 'paraphrase') else '',
                                'justification': f"[Mehrfachkodierung {i}/{len(relevant_categories)}] {r.justification if hasattr(r, 'justification') else ''}",
                                'coder_id': coder_id,
                                'category_preselection_used': bool(preferred_cats),
                                'preferred_categories': preferred_cats,
                                'preselection_reasoning': seg_prefs.get('reasoning', ''),
                                # Mehrfachkodierungs-Metadaten
                                'multiple_coding_instance': i,
                                'total_coding_instances': len(relevant_categories),
                                'target_category': target_category,
                                'original_segment_id': r.segment_id,
                                'is_multiple_coding': True
                            },
                            'analysis_mode': 'abductive',
                            'timestamp': asyncio.get_event_loop().time()
                        }
                        expanded_results.append(formatted_res)
                        print(f"      📝 Mehrfachkodierung {i}: {target_category} (Konfidenz: {cat_info['relevance_score']:.2f})")
                else:
                    # Nur eine Kategorie über Schwellenwert - normale Kodierung
                    expanded_results.append(self._format_single_coding_result_abductive(r, coder_id, preferred_cats, seg_prefs))
            else:
                # Keine Mehrfachkodierung erforderlich - normale Kodierung
                expanded_results.append(self._format_single_coding_result_abductive(r, coder_id, preferred_cats, seg_prefs))
        
        new_results = expanded_results
        
        # Cache alle Ergebnisse (normale + Mehrfachkodierung)
        for formatted_res in new_results:
            segment_id = formatted_res['segment_id']
            result_data = formatted_res['result']
            
            # Add to cache
            serializable_category_definitions = self._serialize_category_definitions(category_definitions)
            
            # Für Mehrfachkodierung verwende die original segment_id für Cache-Lookup
            cache_segment_id = result_data.get('original_segment_id', segment_id)
            
            self.cache.set(
                analysis_mode="abductive",
                segment_text=next((s['text'] for s in batch if s['segment_id'] == cache_segment_id), ""),
                value=formatted_res,
                category_definitions=serializable_category_definitions,
                research_question=research_question,
                coding_rules=coding_rules,
                segment_id=segment_id,  # Verwende die tatsächliche segment_id (mit -1, -2 suffix)
                coder_id=coder_id
            )
        
        return new_results
    
    async def _analyze_grounded(self,
                               segments: List[Dict[str, Any]],
                               research_question: str,
                               coder_settings: Optional[List[Dict[str, Any]]] = None,
                               temperature: Optional[float] = None,
                               use_context: bool = False,
                               document_paraphrases: Optional[Dict[str, List[str]]] = None,
                               context_paraphrase_count: int = 3,
                               **kwargs) -> List[Dict[str, Any]]:
        """
        Optimized grounded analysis - PHASE 1: Subcode-Sammlung.
        
        Workflow Phase 1 (SHARED across all coders):
        1. Relevanzprüfung (filtert relevante Segmente) - SHARED across all coders
        2. Subcode-Sammlung (extrahierte Codes für Theorieentwicklung) - SHARED across all coders
        3. Sammelt Subcodes zentral (stateful über Batches hinweg)
        
        Note: Grounded Theory typically uses single-coder approach for initial subcode collection.
        Multi-coder support can be added later for validation phases.
        
        Hinweis: Phase 2 (Hauptkategorien-Generierung) und Phase 3 (Kodierung)
        werden separat aufgerufen nach Abschluss aller Batches.
        """
        config = self.config[AnalysisMode.GROUNDED]
        batch_size = config['batch_size']  # Get batch_size from config
        
        # 1. SHARED Relevanzprüfung (wie im Standard-Workflow)
        print(f"   📊 Schritt 1: Relevanzprüfung für {len(segments)} Segmente...")
        print(f"   📋 Forschungsfrage: {research_question}")
        print(f"   🔍 API Call 1: Einfache Relevanzprüfung für {len(segments)} Segmente...")
        from ..core.config import CONFIG
        relevance_threshold = CONFIG.get('RELEVANCE_THRESHOLD', 0.0)
        relevance_results = await self.unified_analyzer.analyze_relevance_simple(
            segments=segments,
            research_question=research_question,
            batch_size=batch_size,
            relevance_threshold=relevance_threshold
        )
        
        # Filtere relevante Segmente (Threshold: 0.3)
        relevant_segments = []
        relevance_map = {}
        for rel_result in relevance_results:
            seg_id = rel_result.get('segment_id', '')
            # Alle Segmente aus relevance_results sind bereits als relevant eingestuft
            seg = next((s for s in segments if s['segment_id'] == seg_id), None)
            if seg:
                relevant_segments.append(seg)
                relevance_map[seg_id] = rel_result
        
        print(f"   ✅ {len(relevant_segments)} von {len(segments)} Segmenten sind relevant")
        
        if not relevant_segments:
            print(f"   ✅ 0 von {len(segments)} Segmenten relevant - KEINE weiteren API Calls nötig")
            return []
        
        # 2. SHARED Subcode-Sammlung
        print(f"   🔍 Schritt 2: Subcode-Sammlung aus {len(relevant_segments)} relevanten Segmenten...")
        print(f"   🔍 API Call 2: Grounded Subcode-Sammlung für {len(relevant_segments)} Segmente...")
        grounded_temp = temperature or kwargs.get('temperature', self.inductive_temperature)
        results = await self.unified_analyzer.analyze_batch_grounded(
            segments=relevant_segments,
            research_question=research_question,
            batch_size=batch_size,  # Use configured batch_size instead of all segments
            temperature=grounded_temp
        )
        
        # 3. Sammle Subcodes zentral (stateful)
        subcode_results = []
        for res in results:
            segment_id = res.get('segment_id', '')
            segment_text = next((s['text'] for s in relevant_segments if s['segment_id'] == segment_id), '')
            
            # Extrahiere Subcodes aus Ergebnis
            codes = res.get('codes', [])
            keywords = res.get('keywords', [])
            memo = res.get('memo', '')
            
            # Speichere Segment-Analyse
            segment_analysis = {
                'segment_id': segment_id,
                'segment_text': segment_text,
                'subcodes': codes,
                'keywords': keywords,
                'memo': memo
            }
            self.grounded_segment_analyses.append(segment_analysis)
            
            # Sammle Subcodes (verhindere Duplikate)
            new_subcodes_in_segment = []
            for code in codes:
                if isinstance(code, dict):
                    subcode_name = code.get('name', '').strip()
                    if subcode_name:
                        existing_names = [sc.get('name', '') for sc in self.grounded_subcodes_collection]
                        if subcode_name not in existing_names:
                            self.grounded_subcodes_collection.append({
                                'name': subcode_name,
                                'definition': code.get('definition', ''),
                                'keywords': code.get('keywords', keywords),
                                'evidence': code.get('evidence', []),
                                'confidence': code.get('confidence', 0.7),
                                'source_segments': [segment_text[:100]]
                            })
                            new_subcodes_in_segment.append(subcode_name)
                            print(f"    ✅ Neuer Subcode: '{subcode_name}'")
                elif isinstance(code, str) and code.strip():
                    subcode_name = code.strip()
                    existing_names = [sc.get('name', '') for sc in self.grounded_subcodes_collection]
                    if subcode_name not in existing_names:
                        self.grounded_subcodes_collection.append({
                            'name': subcode_name,
                            'definition': f"Grounded Theory Subcode: {subcode_name}",
                            'keywords': keywords,
                            'evidence': [],
                            'confidence': 0.7,
                            'source_segments': [segment_text[:100]]
                        })
                        new_subcodes_in_segment.append(subcode_name)
                        print(f"    ✅ Neuer Subcode: '{subcode_name}'")
            
            # Log Segment-Zusammenfassung
            seg_id_short = segment_id[:30] + '...' if len(segment_id) > 30 else segment_id
            if new_subcodes_in_segment:
                print(f"   📝 Segment {seg_id_short}: {len(new_subcodes_in_segment)} neue Subcodes")
                for subcode in new_subcodes_in_segment:
                    print(f"      • {subcode}")
            else:
                print(f"   📝 Segment {seg_id_short}: Keine neuen Subcodes (bereits bekannt)")
            
            if codes:
                print(f"      └─ Gesamt Codes in Segment: {len(codes)}")
            if keywords:
                print(f"      └─ Keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
            
            # Sammle Keywords
            if keywords:
                self.grounded_keywords_collection.extend(keywords)
            
            # FIX: Formatiere Ergebnis im gleichen Format wie deductive Mode für Exporter-Kompatibilität
            coder_id = coder_settings[0].get('coder_id', 'auto_1') if coder_settings else 'auto_1'
            
            # Bestimme Hauptkategorie basierend auf dominanten Subcodes
            primary_category = "Grounded_Analysis"
            if codes:
                # Verwende ersten/dominanten Code als Hauptkategorie
                if isinstance(codes[0], dict):
                    primary_category = codes[0].get('name', 'Grounded_Analysis')
                elif isinstance(codes[0], str):
                    primary_category = codes[0]
            
            # Extrahiere Subkategorien-Namen
            subcategory_names = []
            for code in codes:
                if isinstance(code, dict):
                    name = code.get('name', '')
                    if name and name != primary_category:
                        subcategory_names.append(name)
                elif isinstance(code, str) and code != primary_category:
                    subcategory_names.append(code)
            
            formatted_res = {
                'segment_id': segment_id,
                'result': {
                    'primary_category': primary_category,
                    'confidence': 0.8,  # Standard-Konfidenz für Grounded Analysis
                    'all_categories': [primary_category],
                    'subcategories': subcategory_names,
                    'keywords': ', '.join(keywords) if keywords else '',
                    'paraphrase': memo if memo else '',
                    'justification': f"Grounded Analysis: {len(codes)} Codes identifiziert",
                    'coder_id': coder_id,
                    'category_preselection_used': False,
                    'preferred_categories': [],
                    'preselection_reasoning': '',
                    # Grounded-spezifische Metadaten
                    'multiple_coding_instance': 1,
                    'total_coding_instances': 1,
                    'target_category': '',
                    'original_segment_id': segment_id,
                    'is_multiple_coding': False,
                    # Grounded-spezifische Daten (für interne Verwendung)
                    'grounded_subcodes': codes,
                    'grounded_memo': memo,
                    'grounded_phase': 'subcode_collection'
                },
                'analysis_mode': 'grounded',
                'timestamp': asyncio.get_event_loop().time()
            }
            subcode_results.append(formatted_res)
        
        print(f"   ✅ Gesamt Subcodes gesammelt: {len(self.grounded_subcodes_collection)}")
        
        # Zeige Subcode-Zusammenfassung
        if self.grounded_subcodes_collection:
            print(f"   📋 Subcode-Übersicht:")
            for i, subcode in enumerate(self.grounded_subcodes_collection[:10], 1):  # Zeige erste 10
                subcode_name = subcode.get('name', 'Unbekannt')
                print(f"      {i:2d}. {subcode_name}")
            
            if len(self.grounded_subcodes_collection) > 10:
                print(f"      ... und {len(self.grounded_subcodes_collection) - 10} weitere")
        
        return subcode_results
    
    async def generate_grounded_main_categories(self,
                                                research_question: str,
                                                initial_categories: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        PHASE 2: Generiere Hauptkategorien aus gesammelten Subcodes.
        
        Args:
            research_question: Forschungsfrage
            initial_categories: Initiale Kategorien (optional)
            
        Returns:
            Dict mit generierten Hauptkategorien als CategoryDefinitions
        """
        if len(self.grounded_subcodes_collection) < 5:
            print(f"❌ Zu wenige Subcodes für Hauptkategorien-Generierung: {len(self.grounded_subcodes_collection)} < 5")
            # WICHTIG: Im Grounded Mode KEINE initial_categories zurückgeben
            return {}
        
        print(f"\n🕵️ PHASE 2: Generiere Hauptkategorien aus {len(self.grounded_subcodes_collection)} Subcodes...")
        
        # Bereite Subcodes für LLM-Analyse vor
        subcodes_data = []
        all_keywords = []
        
        for subcode in self.grounded_subcodes_collection:
            subcodes_data.append({
                'name': subcode.get('name', ''),
                'definition': subcode.get('definition', ''),
                'keywords': subcode.get('keywords', []),
                'confidence': subcode.get('confidence', 0.7),
                'evidence': subcode.get('evidence', [])
            })
            all_keywords.extend(subcode.get('keywords', []))
        
        # Zähle Keywords
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(20)
        
        # Berechne durchschnittliche Konfidenz
        avg_confidence = sum(sc.get('confidence', 0.7) for sc in subcodes_data) / len(subcodes_data) if subcodes_data else 0.7
        
        # Erstelle Prompt für Hauptkategorien-Generierung
        prompt = self._build_main_categories_generation_prompt(
            subcodes_data=subcodes_data,
            top_keywords=top_keywords,
            research_question=research_question,
            avg_confidence=avg_confidence
        )
        
        try:
            from QCA_AID_assets.utils.tracking.token_tracker import get_global_token_counter
            from QCA_AID_assets.utils.llm.response import LLMResponse
            token_counter = get_global_token_counter()
            
            start_time = asyncio.get_event_loop().time()
            token_counter.start_request()
            
            response = await self.llm_provider.create_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für Grounded Theory. Du antwortest auf Deutsch. Antworte ausschließlich mit einem JSON-Objekt."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.inductive_temperature,
                response_format={"type": "json_object"}
            )
            
            token_counter.track_response(response, self.model_name)
            processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            llm_response = LLMResponse(response)
            result = json.loads(llm_response.extract_json())
            
            # Konvertiere zu CategoryDefinitions (OHNE initial_categories)
            generated_categories = self._parse_main_categories_from_grounded(result, None)
            
            print(f"✅ {len(generated_categories)} Hauptkategorien generiert")
            
            # Zeige generierte Hauptkategorien
            if generated_categories:
                print(f"   📋 Generierte Hauptkategorien:")
                for i, (cat_name, cat_def) in enumerate(generated_categories.items(), 1):
                    print(f"      {i:2d}. {cat_name}")
                    if hasattr(cat_def, 'subcategories') and cat_def.subcategories:
                        subcat_count = len(cat_def.subcategories)
                        print(f"          └─ {subcat_count} Subkategorien")
            
            return generated_categories
            
        except Exception as e:
            print(f"❌ Fehler bei Hauptkategorien-Generierung: {e}")
            import traceback
            traceback.print_exc()
            # WICHTIG: Im Grounded Mode KEINE initial_categories zurückgeben
            return {}
    
    def _build_main_categories_generation_prompt(self,
                                                 subcodes_data: List[Dict[str, Any]],
                                                 top_keywords: List[Tuple[str, int]],
                                                 research_question: str,
                                                 avg_confidence: float) -> str:
        """Erstellt Prompt für Hauptkategorien-Generierung aus Subcodes."""
        subcodes_text = ""
        for i, subcode in enumerate(subcodes_data, 1):
            keywords_str = ', '.join(subcode['keywords'][:10])
            subcodes_text += f"""
{i}. Subcode: {subcode['name']}
   Definition: {subcode['definition']}
   Keywords: {keywords_str}
   Konfidenz: {subcode['confidence']:.2f}
   Textbelege: {len(subcode.get('evidence', []))}
"""
        
        top_keywords_str = ', '.join([kw for kw, _ in top_keywords[:15]])
        
        return f"""Generiere Hauptkategorien aus gesammelten Subcodes nach Grounded Theory.

FORSCHUNGSFRAGE:
{research_question}

GESAMMELTE SUBCODES ({len(subcodes_data)}):
{subcodes_text}

TOP KEYWORDS:
{top_keywords_str}

DURCHSCHNITTLICHE KONFIDENZ: {avg_confidence:.2f}

AUFGABE:
1. Analysiere die Subcodes und identifiziere thematische Muster
2. Gruppiere verwandte Subcodes zu Hauptkategorien
3. Entwickle präzise Definitionen für jede Hauptkategorie
4. Stelle sicher, dass Hauptkategorien die Forschungsfrage abdecken

Antworte NUR mit folgendem JSON-Format:
{{
  "main_categories": [
    {{
      "name": "Hauptkategorie-Name",
      "definition": "Detaillierte Definition der Hauptkategorie",
      "related_subcodes": ["Subcode1", "Subcode2", ...],
      "confidence": 0.85,
      "coverage": "Beschreibung wie diese Kategorie die Forschungsfrage abdeckt"
    }},
    ...
  ],
  "analysis_summary": "Zusammenfassung der Kategorienbildung"
}}"""
    
    def _parse_main_categories_from_grounded(self,
                                            result: Dict[str, Any],
                                            initial_categories: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Konvertiert LLM-Ergebnis zu CategoryDefinitions."""
        main_categories = result.get('main_categories', [])
        # WICHTIG: Im Grounded Mode KEINE initial_categories verwenden - rein induktiv!
        generated = {}  # Leeres Dict - keine vordefinierten Kategorien!
        
        for cat_data in main_categories:
            cat_name = cat_data.get('name', '').strip()
            if not cat_name:
                continue
            
            # Erstelle CategoryDefinition
            related_subcodes = cat_data.get('related_subcodes', [])
            subcategories = {}
            
            # Erstelle Subkategorien aus related_subcodes
            for subcode_name in related_subcodes:
                # Finde Subcode-Details
                subcode_details = next(
                    (sc for sc in self.grounded_subcodes_collection if sc.get('name') == subcode_name),
                    None
                )
                
                if subcode_details:
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    subcategories[subcode_name] = CategoryDefinition(
                        name=subcode_name,
                        definition=subcode_details.get('definition', f"Subcode: {subcode_name}"),
                        examples=[],
                        rules=[],
                        subcategories={},
                        added_date=current_date,
                        modified_date=current_date
                    )
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            generated[cat_name] = CategoryDefinition(
                name=cat_name,
                definition=cat_data.get('definition', f"Hauptkategorie: {cat_name}"),
                examples=[],
                rules=[],
                subcategories=subcategories,
                added_date=current_date,
                modified_date=current_date
            )
        
        return generated
    
    async def code_with_grounded_categories(self,
                                           all_segments: List[Dict[str, Any]],
                                           grounded_categories: Dict[str, Any],
                                           research_question: str,
                                           coding_rules: Optional[List[str]] = None,
                                           coder_settings: Optional[List[Dict[str, Any]]] = None,
                                           batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        PHASE 3: Kodiere alle Segmente mit generierten Grounded-Kategorien.
        
        Args:
            all_segments: Alle Segmente (mit segment_id und text)
            grounded_categories: Generierte Hauptkategorien
            research_question: Forschungsfrage
            coding_rules: Kodierregeln
            coder_settings: Liste der Kodierer-Konfigurationen
            
        Returns:
            List von Kodierungsergebnissen
        """
        print(f"\n📝 PHASE 3: Kodiere {len(all_segments)} Segmente mit Grounded-Kategorien...")
        
        # Konvertiere CategoryDefinitions zu Dict-Format
        cat_defs = {
            k: {
                'definition': v.definition if hasattr(v, 'definition') else str(v),
                'subcategories': dict(v.subcategories) if hasattr(v, 'subcategories') and v.subcategories else {}
            }
            for k, v in grounded_categories.items()
        }
        
        coding_results = []
        
        if coder_settings:
            # Für jeden Kodierer separate API-Calls
            for coder_config in coder_settings:
                coder_temp = coder_config.get('temperature', self.inductive_temperature)
                coder_id = coder_config.get('coder_id', 'auto_1')
                print(f"   🔍 Kodierung mit {coder_id} (Temperature: {coder_temp})")
                
                try:
                    # Verwende analyze_batch direkt mit allen Segmenten (interne Batch-Logik)
                    results = await self.unified_analyzer.analyze_batch(
                        segments=all_segments,
                        category_definitions=cat_defs,
                        research_question=research_question,
                        coding_rules=coding_rules or [],
                        batch_size=batch_size,
                        temperature=coder_temp
                    )
                    
                    if results:
                        for result in results:
                            formatted_result = {
                                'segment_id': result.segment_id,
                                'result': {
                                    'primary_category': result.primary_category,
                                    'confidence': result.confidence,
                                    'all_categories': getattr(result, 'all_categories', [result.primary_category]),
                                    'subcategories': getattr(result, 'subcategories', []),
                                    'keywords': getattr(result, 'keywords', ''),
                                    'paraphrase': getattr(result, 'paraphrase', ''),
                                    'justification': getattr(result, 'justification', ''),
                                    'coder_id': coder_id,
                                    'analysis_mode': 'grounded',
                                    'grounded_recoded': True
                                },
                                'analysis_mode': 'grounded',
                                'timestamp': asyncio.get_event_loop().time()
                            }
                            coding_results.append(formatted_result)
                            
                            print(f"   📝 Segment: {result.segment_id} → Kategorie: {result.primary_category}")
                            print(f"      └─ Kodierer: {coder_id} | Konfidenz: {result.confidence:.2f}")
                    
                    # FIX: Store results for reliability analysis (use original results, not formatted)
                    if results:
                        # Konvertiere UnifiedAnalysisResult zu Dictionary-Format für _store_results_for_reliability
                        results_for_reliability = []
                        for result in results:
                            reliability_result = {
                                'segment_id': result.segment_id,
                                'primary_category': result.primary_category,
                                'confidence': result.confidence,
                                'subcategories': getattr(result, 'subcategories', []),
                                'keywords': getattr(result, 'keywords', ''),
                                'paraphrase': getattr(result, 'paraphrase', ''),
                                'justification': getattr(result, 'justification', ''),
                                'coder_id': coder_id
                            }
                            results_for_reliability.append(reliability_result)
                        
                        await self._store_results_for_reliability(results_for_reliability, 'grounded', coder_id, coder_temp)
                            
                except Exception as e:
                    print(f"   ❌ Fehler bei Kodierung mit {coder_id}: {e}")
                    continue
        else:
            # Fallback: Ein Kodierer
            print(f"   🔍 Fallback-Kodierung mit auto_1")
            
            try:
                # Verwende analyze_batch direkt mit allen Segmenten (interne Batch-Logik)
                results = await self.unified_analyzer.analyze_batch(
                    segments=all_segments,
                    category_definitions=cat_defs,
                    research_question=research_question,
                    coding_rules=coding_rules or [],
                    batch_size=batch_size,
                    temperature=self.inductive_temperature
                )
                
                if results:
                    for result in results:
                        formatted_result = {
                            'segment_id': result.segment_id,
                            'result': {
                                'primary_category': result.primary_category,
                                'confidence': result.confidence,
                                'all_categories': getattr(result, 'all_categories', [result.primary_category]),
                                'subcategories': getattr(result, 'subcategories', []),
                                'keywords': getattr(result, 'keywords', ''),
                                'paraphrase': getattr(result, 'paraphrase', ''),
                                'justification': getattr(result, 'justification', ''),
                                'coder_id': 'auto_1',
                                'analysis_mode': 'grounded',
                                'grounded_recoded': True
                            },
                            'analysis_mode': 'grounded',
                            'timestamp': asyncio.get_event_loop().time()
                        }
                        coding_results.append(formatted_result)
                        
                        print(f"   📝 Segment: {result.segment_id} → Kategorie: {result.primary_category}")
                        print(f"      └─ Kodierer: auto_1 | Konfidenz: {result.confidence:.2f}")
                
                # FIX: Store results for reliability analysis (use original results, not formatted)
                if results:
                    # Konvertiere UnifiedAnalysisResult zu Dictionary-Format für _store_results_for_reliability
                    results_for_reliability = []
                    for result in results:
                        reliability_result = {
                            'segment_id': result.segment_id,
                            'primary_category': result.primary_category,
                            'confidence': result.confidence,
                            'subcategories': getattr(result, 'subcategories', []),
                            'keywords': getattr(result, 'keywords', ''),
                            'paraphrase': getattr(result, 'paraphrase', ''),
                            'justification': getattr(result, 'justification', ''),
                            'coder_id': 'auto_1'
                        }
                        results_for_reliability.append(reliability_result)
                    
                    await self._store_results_for_reliability(results_for_reliability, 'grounded', 'auto_1', self.inductive_temperature)
                        
            except Exception as e:
                print(f"   ❌ Fehler bei Fallback-Kodierung: {e}")
        
        print(f"✅ {len(coding_results)} Kodierungen erstellt")
        return coding_results
    
    async def _code_single_segment_grounded(self, segment: Dict[str, Any], cat_defs: Dict[str, Any], 
                                          research_question: str, coding_rules: List[str], 
                                          temperature: float, coder_id: str) -> Optional[Dict[str, Any]]:
        """
        Hilfsmethode für normale (nicht-multiple) Kodierung eines einzelnen Segments im Grounded Mode.
        """
        try:
            result = await self.unified_analyzer.analyze_comprehensive(
                segment_text=segment['text'],
                category_definitions=cat_defs,
                research_question=research_question,
                coding_rules=coding_rules,
                temperature=temperature
            )
            
            if result:
                formatted_result = {
                    'segment_id': segment['segment_id'],
                    'result': {
                        'primary_category': result.primary_category,
                        'confidence': result.confidence,
                        'all_categories': result.all_categories,
                        'subcategories': result.subcategories if result.subcategories else [],
                        'keywords': result.keywords if result.keywords else '',
                        'paraphrase': result.paraphrase if result.paraphrase else '',
                        'justification': result.justification if result.justification else '',
                        'coder_id': coder_id,
                        'analysis_mode': 'grounded',
                        'grounded_recoded': True
                    },
                    'analysis_mode': 'grounded',
                    'timestamp': asyncio.get_event_loop().time()
                }
                return formatted_result
            
        except Exception as e:
            print(f"   ❌ Fehler bei Kodierung von Segment {segment['segment_id']}: {e}")
            
        return None
    
    def get_grounded_subcodes(self) -> List[Dict[str, Any]]:
        """Gibt gesammelte Subcodes zurück."""
        return self.grounded_subcodes_collection.copy()
    
    def reset_grounded_state(self):
        """Setzt Grounded State zurück (für neue Analyse)."""
        self.grounded_subcodes_collection = []
        self.grounded_segment_analyses = []
        self.grounded_keywords_collection = []
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """
        Get optimization performance metrics.
        
        Returns:
            Dictionary with optimization metrics
        """
        summary = self.metrics_collector.get_summary()
        
        if not summary:
            return {}
        
        # Calculate optimization metrics
        optimization_metrics = {
            'total_api_calls': summary.get('total_api_calls', 0),
            'total_segments': summary.get('total_segments', 0),
            'avg_api_calls_per_segment': summary.get('avg_api_calls_per_segment', 0),
            'avg_tokens_per_segment': summary.get('avg_tokens_per_segment', 0),
            'avg_processing_time_per_segment_ms': summary.get('avg_processing_time_per_segment_ms', 0),
            'analysis_modes': {}
        }
        
        # Add mode-specific metrics
        for mode, mode_data in summary.get('analysis_modes', {}).items():
            optimization_metrics['analysis_modes'][mode] = {
                'api_calls_per_segment': mode_data.get('avg_api_calls_per_segment', 0),
                'processing_time_ms': mode_data.get('avg_processing_time_per_segment_ms', 0),
                'tokens_per_segment': mode_data.get('avg_tokens_per_segment', 0),
                'success_rate': mode_data.get('success_rate', 0),
                'quality_scores': mode_data.get('quality_scores', []),
                'confidence_scores': mode_data.get('confidence_scores', [])
            }
        
        return optimization_metrics
    
    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """
        Compare current metrics with baseline.
        
        Args:
            baseline_file: Path to baseline metrics JSON file
            
        Returns:
            Comparison results
        """
        return self.metrics_collector.compare_with_baseline(baseline_file)


def create_optimization_controller(llm_provider, model_name: str = "gpt-4") -> OptimizationController:
    """
    Create an OptimizationController instance.
    
    Args:
        llm_provider: LLM provider instance
        model_name: Model to use
        
    Returns:
        OptimizationController instance
    """
    return OptimizationController(llm_provider=llm_provider, model_name=model_name)
