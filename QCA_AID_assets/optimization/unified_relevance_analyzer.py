"""
Unified Relevance Analyzer Prototype
Consolidates multiple API calls into a single comprehensive analysis for deductive mode.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics

from QCA_AID_assets.utils.llm.base import LLMProvider
from QCA_AID_assets.utils.llm.response import LLMResponse
from QCA_AID_assets.optimization.tests.metrics_collector import record_api_call
from QCA_AID_assets.utils.tracking.token_tracker import get_global_token_counter
from QCA_AID_assets.QCA_Prompts import QCAPrompts  # Import für Standard-Prompts

# Get global token counter instance
token_counter = get_global_token_counter()


@dataclass
class UnifiedAnalysisResult:
    """Results from unified analysis."""
    segment_id: str
    primary_category: str
    all_categories: List[Dict[str, Any]]  # All categories with relevance scores
    relevance_scores: Dict[str, float]
    confidence: float
    requires_multiple_coding: bool
    multiple_coding_threshold: float = 0.7
    # Extended fields for complete coding information
    subcategories: List[str] = None
    keywords: str = ""
    paraphrase: str = ""
    justification: str = ""


class UnifiedRelevanceAnalyzer:
    """
    Unified analyzer that combines multiple API calls into one comprehensive analysis.
    
    For deductive mode, replaces:
    1. Multiple category relevance checks
    2. Individual coding calls
    3. Confidence scoring calls
    
    With:
    1. Single comprehensive analysis that returns all needed information
    """
    
    def __init__(self, llm_provider: LLMProvider, 
                 model_name: str = "gpt-4",
                 temperature: float = 0.3,
                 multiple_coding_threshold: float = 0.7):
        """
        Initialize the unified analyzer.
        
        Args:
            llm_provider: LLM provider instance
            model_name: Model to use for analysis
            temperature: Temperature for LLM responses
            multiple_coding_threshold: Threshold for multiple coding decisions
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize prompt handler with standard prompts
        # Note: We'll use dummy values for initialization, real values come from config
        self.prompt_handler = QCAPrompts(
            forschungsfrage="Placeholder - wird zur Laufzeit gesetzt",
            kodierregeln=["Placeholder - wird zur Laufzeit gesetzt"],
            deduktive_kategorien={}
        )
        
        # Configuration
        self.max_categories_per_segment = 3
        self.min_relevance_score = 0.3
        self.multiple_coding_threshold = multiple_coding_threshold
        
    def update_prompt_context(self, research_question: str, coding_rules: List[str], categories: Dict = None):
        """
        Update the prompt handler with actual research context.
        
        Args:
            research_question: The actual research question
            coding_rules: List of coding rules
            categories: Dictionary of categories (optional)
        """
        self.prompt_handler = QCAPrompts(
            forschungsfrage=research_question,
            kodierregeln=coding_rules,
            deduktive_kategorien=categories or {}
        )
        
    async def analyze_relevance_simple(self,
                                      segments: List[Dict[str, str]],
                                      research_question: str,
                                      batch_size: int = 5,
                                      relevance_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Simple relevance check without category preferences (for inductive/abductive/grounded modes).
        Uses batching to respect batch_size limits.
        
        Args:
            segments: List of dicts with 'segment_id' and 'text'
            research_question: Research question context
            batch_size: Maximum segments per API call
            relevance_threshold: Minimum confidence for relevant segments (default: 0.0)
            
        Returns:
            List of dicts with relevance information (only segments marked as relevant by LLM)
        """
        all_results = []  # Alle LLM-Ergebnisse (relevant + nicht-relevant)
        results = []      # Nur die finalen relevanten Ergebnisse
        
        # Process in batches
        total_batches = (len(segments) + batch_size - 1) // batch_size
        print(f"   🔄 Verarbeite {len(segments)} Segmente in {total_batches} Batches (Batch-Größe: {batch_size})")
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            if not batch:
                continue
            
            batch_num = (i // batch_size) + 1
            print(f"   📦 Relevanz-Batch {batch_num}/{total_batches}: {len(batch)} Segmente")
                
            try:
                start_time = asyncio.get_event_loop().time()
                token_counter.start_request()
                
                prompt = self._build_simple_relevance_prompt(batch, research_question)
                
                response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf Deutsch. Antworte ausschließlich mit einem JSON-Objekt."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                token_counter.track_response(response, self.model_name)
                processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                
                record_api_call(
                    call_type="relevance_simple",
                    tokens_used=self._estimate_tokens(prompt, response),
                    processing_time_ms=processing_time_ms,
                    success=True
                )
                
                llm_response = LLMResponse(response)
                result = json.loads(llm_response.extract_json())
                
                # Adapt standard prompt response format to expected format
                standard_results = result.get("segment_results", [])
                
                for std_result in standard_results:
                    # Map segment_number to actual segment_id
                    segment_number = std_result.get("segment_number", 1)
                    if segment_number <= len(batch):
                        segment_id = batch[segment_number - 1]["segment_id"]
                    else:
                        # Fallback if segment_number is out of range
                        segment_id = f"unknown_segment_{segment_number}"
                    
                    # Extrahiere LLM-Entscheidung und Konfidenz
                    is_relevant = std_result.get("is_relevant", False)
                    confidence = std_result.get("confidence", 0.0)
                    research_relevance = float(confidence) if confidence is not None else 0.0
                    
                    # Erstelle Ergebnis für alle LLM-Bewertungen (für Statistik)
                    all_result = {
                        "segment_id": segment_id,
                        "is_relevant": is_relevant,
                        "research_relevance": research_relevance,
                        "relevance_reasoning": std_result.get("justification", "Keine Begründung verfügbar")
                    }
                    all_results.append(all_result)
                    
                    # Nur als relevant markierte Segmente mit hinreichender Konfidenz werden weitergegeben
                    if is_relevant and research_relevance >= relevance_threshold:
                        adapted_result = {
                            "segment_id": segment_id,
                            "research_relevance": research_relevance,
                            "relevance_reasoning": std_result.get("justification", "Keine Begründung verfügbar")
                        }
                        results.append(adapted_result)
                    
            except Exception as e:
                # Handle batch errors gracefully
                print(f"   ⚠️ Fehler in Relevanzprüfung Batch {i//batch_size + 1}: {e}")
                # Add default results for failed batch
                for segment in batch:
                    all_results.append({
                        "segment_id": segment["segment_id"],
                        "is_relevant": False,
                        "research_relevance": 0.0,
                        "relevance_reasoning": f"Fehler bei der Analyse: {str(e)}"
                    })
        
        # Statistiken berechnen
        llm_relevant_count = sum(1 for r in all_results if r.get('is_relevant', False))
        final_relevant_count = len(results)
        total_count = len(all_results)
        
        # Verbesserte Log-Ausgabe
        print(f"   🔍 DEBUG: Relevance check returned {total_count} total results")
        print(f"   📊 {llm_relevant_count} Segmente vom LLM als relevant für die Forschungsfrage identifiziert, darunter {final_relevant_count} Segmente mit hinreichender Konfidenz (≥{relevance_threshold})")
        
        # Show detailed results for relevant segments only
        for i, result in enumerate(results):
            seg_id = result.get('segment_id', 'MISSING')
            # Truncate long segment IDs for readability
            if len(seg_id) > 35:
                seg_id_display = seg_id[:32] + '...'
            else:
                seg_id_display = seg_id
            
            relevance = result.get('research_relevance', 0.0)
            status = "✅ RELEVANT"  # Alle Segmente in results sind relevant
            
            print(f"   🔍 Segment {i+1:2d}: {seg_id_display:<35} → {relevance:.1f} ({status})")
        
        return results
    
    def _build_simple_relevance_prompt(self, segments: List[Dict[str, str]], research_question: str) -> str:
        """Build relevance prompt using standard QCA_Prompts methodology."""
        # Update prompt handler with current research question
        self.update_prompt_context(research_question, ["Standard-Kodierregeln"], {})
        
        # Format segments for standard prompt
        segments_text = "\n\n".join([
            f"SEGMENT {i+1}:\nID: {s['segment_id']}\nTEXT: {s['text']}"
            for i, s in enumerate(segments)
        ])
        
        # Use standard relevance check prompt with empty exclusion rules
        exclusion_rules = []  # Can be extended based on configuration
        
        return self.prompt_handler.get_relevance_check_prompt(
            segments_text=segments_text,
            exclusion_rules=exclusion_rules
        )

    async def analyze_relevance_with_preferences(self,
                                                 segments: List[Dict[str, str]],
                                                 category_definitions: Dict[str, str],
                                                 research_question: str,
                                                 coding_rules: List[str],
                                                 relevance_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Analyze relevance of segments and determine category preferences.
        
        This API call provides:
        - Research question relevance for each segment (only segments marked as relevant by LLM)
        - Category preferences (which categories are most relevant)
        - Initial relevance scores
        
        Args:
            segments: List of dicts with 'segment_id' and 'text'
            category_definitions: Dictionary mapping category names to descriptions
            research_question: Research question context
            coding_rules: List of coding rules
            relevance_threshold: Minimum confidence for relevant segments (default: 0.0)
            
        Returns:
            List of dicts with relevance information and category preferences (only relevant segments)
        """
        all_results = []  # Alle LLM-Ergebnisse (relevant + nicht-relevant)
        results = []      # Nur die finalen relevanten Ergebnisse
        
        try:
            start_time = asyncio.get_event_loop().time()
            token_counter.start_request()
            
            prompt = self._build_relevance_preference_prompt(
                segments=segments,
                category_definitions=category_definitions,
                research_question=research_question,
                coding_rules=coding_rules
            )
            
            response = await self.llm_provider.create_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf Deutsch. Antworte ausschließlich mit einem JSON-Objekt."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            token_counter.track_response(response, self.model_name)
            processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            record_api_call(
                call_type="relevance_with_preferences",
                tokens_used=self._estimate_tokens(prompt, response),
                processing_time_ms=processing_time_ms,
                success=True
            )
            
            llm_response = LLMResponse(response)
            result = json.loads(llm_response.extract_json())
            
            # Debug: Log the raw response structure
            print(f"   🔍 DEBUG: Raw LLM response keys: {list(result.keys())}")
            if "segment_results" in result:
                print(f"   🔍 DEBUG: Found {len(result['segment_results'])} segment_results")
                if result['segment_results']:
                    first_result = result['segment_results'][0]
                    print(f"   🔍 DEBUG: First result keys: {list(first_result.keys())}")
                    print(f"   🔍 DEBUG: First result is_relevant: {first_result.get('is_relevant', 'MISSING')}")
                    print(f"   🔍 DEBUG: First result preferred_categories: {first_result.get('preferred_categories', 'MISSING')}")
            
            # Adapt standard prompt response format to expected format
            # Note: The standard category preselection prompt has a different format
            # We need to adapt it to the expected format for compatibility
            if "segment_results" in result:
                # Standard format adaptation
                standard_results = result.get("segment_results", [])
                
                for std_result in standard_results:
                    # Map segment_number to actual segment_id
                    segment_number = std_result.get("segment_number", 1)
                    if segment_number <= len(segments):
                        segment_id = segments[segment_number - 1]["segment_id"]
                    else:
                        # Fallback if segment_number is out of range
                        segment_id = f"unknown_segment_{segment_number}"
                    
                    # Extrahiere LLM-Entscheidung und Konfidenz
                    is_relevant = std_result.get("is_relevant", False)
                    confidence = std_result.get("confidence", 0.0)
                    research_relevance = float(confidence) if confidence is not None else (1.0 if is_relevant else 0.0)
                    
                    # Get preferred categories (categories with score >= 0.6)
                    preferred_categories = std_result.get("preferred_categories", [])
                    category_preferences = std_result.get("relevance_scores", {})
                    
                    # Erstelle Ergebnis für alle LLM-Bewertungen (für Statistik)
                    all_result = {
                        "segment_id": segment_id,
                        "is_relevant": is_relevant,
                        "research_relevance": research_relevance,
                        "category_preferences": category_preferences,
                        "top_categories": preferred_categories,
                        "relevance_reasoning": std_result.get("reasoning", "Keine Begründung verfügbar")
                    }
                    all_results.append(all_result)
                    
                    # Nur als relevant markierte Segmente mit hinreichender Konfidenz werden weitergegeben
                    if is_relevant and research_relevance >= relevance_threshold:
                        adapted_result = {
                            "segment_id": segment_id,
                            "research_relevance": research_relevance,
                            "category_preferences": category_preferences,
                            "top_categories": preferred_categories,
                            "relevance_reasoning": std_result.get("reasoning", "Keine Begründung verfügbar")
                        }
                        results.append(adapted_result)
                
                # Statistiken berechnen
                llm_relevant_count = sum(1 for r in all_results if r.get('is_relevant', False))
                final_relevant_count = len(results)
                total_count = len(all_results)
                
                # Verbesserte Log-Ausgabe
                print(f"   🔍 DEBUG: Relevance with preferences returned {total_count} total results")
                print(f"   📊 {llm_relevant_count} Segmente vom LLM als relevant für die Forschungsfrage identifiziert, darunter {final_relevant_count} Segmente mit hinreichender Konfidenz (≥{relevance_threshold})")
                
                return results
            else:
                # Fallback to original format if available
                return result.get("results", [])
            
        except Exception as e:
            processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            record_api_call(
                call_type="relevance_with_preferences",
                tokens_used=0,
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )
            raise
    
    def _build_relevance_preference_prompt(self,
                                          segments: List[Dict[str, str]],
                                          category_definitions: Dict[str, str],
                                          research_question: str,
                                          coding_rules: List[str]) -> str:
        """Build prompt for relevance checking with category preferences using standard QCA_Prompts."""
        # Update prompt handler with current context
        self.update_prompt_context(research_question, coding_rules, category_definitions)
        
        # Format segments for standard prompt
        segments_text = "\n\n".join([
            f"SEGMENT {i+1}:\nID: {s['segment_id']}\nTEXT: {s['text']}"
            for i, s in enumerate(segments)
        ])
        
        # Use standard relevance with category preselection prompt
        return self.prompt_handler.get_relevance_with_category_preselection_prompt(
            segments_text=segments_text,
            categories=category_definitions
        )

    async def analyze_category_preferences(self,
                                          segments: List[Dict[str, str]],
                                          category_definitions: Dict[str, str],
                                          research_question: str,
                                          coding_rules: List[str],
                                          batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Bestimme Kategoriepräferenzen für bereits als relevant bestätigte Segmente.
        Uses batching to respect batch_size limits.
        
        Diese Methode führt KEINE Relevanzprüfung durch, sondern bestimmt nur
        welche Kategorien am besten zu den Segmenten passen.
        
        Args:
            segments: Liste bereits relevanter Segmente mit 'segment_id' und 'text'
            category_definitions: Dictionary mapping category names to descriptions
            research_question: Research question context
            coding_rules: List of coding rules
            batch_size: Maximum segments per API call
            
        Returns:
            List of dicts with category preference information
        """
        results = []
        
        # Process in batches
        total_batches = (len(segments) + batch_size - 1) // batch_size
        print(f"   🔄 Verarbeite {len(segments)} Segmente in {total_batches} Batches (Batch-Größe: {batch_size})")
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            if not batch:
                continue
            
            batch_num = (i // batch_size) + 1
            print(f"   📦 Kategoriepräferenzen-Batch {batch_num}/{total_batches}: {len(batch)} Segmente")
                
            try:
                start_time = asyncio.get_event_loop().time()
                token_counter.start_request()
                
                prompt = self._build_category_preferences_prompt(
                    segments=batch,
                    category_definitions=category_definitions,
                    research_question=research_question,
                    coding_rules=coding_rules
                )
                
                response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf Deutsch. Antworte ausschließlich mit einem JSON-Objekt."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                token_counter.track_response(response, self.model_name)
                processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                
                record_api_call(
                    call_type="category_preferences",
                    tokens_used=self._estimate_tokens(prompt, response),
                    processing_time_ms=processing_time_ms,
                    success=True
                )
                
                llm_response = LLMResponse(response)
                json_text = llm_response.extract_json()
                
                # Additional safety check for JSON parsing
                if not json_text or json_text.strip() == "":
                    raise ValueError("Empty JSON response from LLM")
                
                result = json.loads(json_text)
                
                # Ensure result is a dictionary
                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict from LLM, got {type(result)}")
                
                # Debug: Log the structure of the result
                # print(f"   🔍 DEBUG: LLM response structure: {list(result.keys())}")
                if "segment_results" in result:
                    segment_results = result["segment_results"]
                    # print(f"   🔍 DEBUG: segment_results type: {type(segment_results)}, length: {len(segment_results) if isinstance(segment_results, list) else 'N/A'}")
                    if isinstance(segment_results, list) and len(segment_results) > 0:
                        # print(f"   🔍 DEBUG: First segment_result type: {type(segment_results[0])}")
                        if segment_results[0] is not None:
                            # print(f"   🔍 DEBUG: First segment_result keys: {list(segment_results[0].keys()) if isinstance(segment_results[0], dict) else 'Not a dict'}")
                            pass
                
                # Parse category preference results
                if "segment_results" in result:
                    standard_results = result.get("segment_results", [])
                    
                    # Ensure standard_results is a list
                    if not isinstance(standard_results, list):
                        print(f"   ⚠️ Expected list for segment_results, got {type(standard_results)}")
                        standard_results = []
                    
                    for std_result in standard_results:
                        # Skip None results
                        if std_result is None:
                            print(f"   ⚠️ Skipping None result in segment_results")
                            continue
                            
                        # Ensure std_result is a dictionary
                        if not isinstance(std_result, dict):
                            print(f"   ⚠️ Expected dict for segment result, got {type(std_result)}")
                            continue
                            
                        # Map segment_number to actual segment_id
                        segment_number = std_result.get("segment_number", 1)
                        if segment_number <= len(batch):
                            segment_id = batch[segment_number - 1]["segment_id"]
                        else:
                            # Fallback if segment_number is out of range
                            segment_id = f"unknown_segment_{segment_number}"
                            print(f"   ⚠️ Segment number {segment_number} out of range for batch size {len(batch)}")
                        
                        # Safely extract data with None checks
                        category_scores = std_result.get("category_scores", {})
                        preferred_categories = std_result.get("preferred_categories", [])
                        reasoning = std_result.get("reasoning", "Keine Begründung verfügbar")
                        
                        # Ensure extracted data has correct types
                        if not isinstance(category_scores, dict):
                            print(f"   ⚠️ Expected dict for category_scores, got {type(category_scores)}")
                            category_scores = {}
                        
                        if not isinstance(preferred_categories, list):
                            print(f"   ⚠️ Expected list for preferred_categories, got {type(preferred_categories)}")
                            preferred_categories = []
                        
                        if not isinstance(reasoning, str):
                            reasoning = str(reasoning) if reasoning is not None else "Keine Begründung verfügbar"
                        
                        adapted_result = {
                            "segment_id": segment_id,
                            "category_preferences": category_scores,
                            "top_categories": preferred_categories,
                            "preference_reasoning": reasoning
                        }
                        results.append(adapted_result)
                    
                    # Ensure we have results for all segments in the batch
                    processed_segment_ids = {r["segment_id"] for r in results if r["segment_id"] in [s["segment_id"] for s in batch]}
                    missing_segments = [s for s in batch if s["segment_id"] not in processed_segment_ids]
                    
                    if missing_segments:
                        print(f"   ⚠️ Creating default results for {len(missing_segments)} missing segments")
                        for segment in missing_segments:
                            results.append({
                                "segment_id": segment["segment_id"],
                                "category_preferences": {},
                                "top_categories": [],
                                "preference_reasoning": "Keine Kategoriepräferenzen ermittelt (fehlende LLM-Response)"
                            })
                else:
                    # Fallback to original format if available
                    print(f"   🔍 DEBUG: Using fallback format, available keys: {list(result.keys())}")
                    batch_results = result.get("results", [])
                    
                    # If no results found, create default entries
                    if not batch_results:
                        print(f"   ⚠️ No results found in LLM response, creating default entries")
                        for segment in batch:
                            results.append({
                                "segment_id": segment["segment_id"],
                                "category_preferences": {},
                                "top_categories": [],
                                "preference_reasoning": "Keine Kategoriepräferenzen ermittelt (LLM-Response unvollständig)"
                            })
                    else:
                        results.extend(batch_results)
                    
            except Exception as e:
                # Handle batch errors gracefully
                print(f"   ⚠️ Fehler in Kategoriepräferenzen Batch {i//batch_size + 1}: {e}")
                # Add default results for failed batch
                for segment in batch:
                    results.append({
                        "segment_id": segment["segment_id"],
                        "category_preferences": {},
                        "top_categories": [],
                        "preference_reasoning": f"Fehler bei der Analyse: {str(e)}"
                    })
        
        return results
    
    def _build_category_preferences_prompt(self,
                                          segments: List[Dict[str, str]],
                                          category_definitions: Dict[str, str],
                                          research_question: str,
                                          coding_rules: List[str]) -> str:
        """Build prompt for category preferences using standard QCA_Prompts."""
        # Update prompt handler with current context
        self.update_prompt_context(research_question, coding_rules, category_definitions)
        
        # Format segments for standard prompt
        segments_text = "\n\n".join([
            f"SEGMENT {i+1}:\nID: {s['segment_id']}\nTEXT: {s['text']}"
            for i, s in enumerate(segments)
        ])
        
        # Use new category preferences prompt
        return self.prompt_handler.get_category_preferences_prompt(
            segments_text=segments_text,
            categories=category_definitions
        )

    async def analyze_relevance_comprehensive(self,
                                            segment_text: str,
                                            segment_id: str,
                                            category_definitions: Dict[str, str],
                                            research_question: str,
                                            coding_rules: List[str]) -> UnifiedAnalysisResult:
        """
        Perform comprehensive relevance analysis for deductive mode.
        
        This single API call replaces multiple individual calls for:
        - Category relevance checking
        - Primary category selection
        - Multiple coding determination
        - Confidence scoring
        
        Args:
            segment_text: Text segment to analyze
            segment_id: Unique identifier for the segment
            category_definitions: Dictionary mapping category names to descriptions
            research_question: Research question context
            coding_rules: List of coding rules
            
        Returns:
            UnifiedAnalysisResult containing all analysis results
        """
        # Build comprehensive prompt
        prompt = self._build_comprehensive_prompt(
            segment_text=segment_text,
            category_definitions=category_definitions,
            research_question=research_question,
            coding_rules=coding_rules
        )
        
        try:
            # Track API call
            start_time = asyncio.get_event_loop().time()
            # Track token usage
            token_counter.start_request()
            
            response = await self.llm_provider.create_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse nach Mayring. Du antwortest auf Deutsch."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Track response
            token_counter.track_response(response, self.model_name)
            
            processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Record the API call
            record_api_call(
                call_type="unified_relevance_analysis",
                tokens_used=self._estimate_tokens(prompt, response),
                processing_time_ms=processing_time_ms,
                success=True
            )
            
            llm_response = LLMResponse(response)
            result = json.loads(llm_response.extract_json())
            
            # Parse comprehensive result
            return self._parse_comprehensive_result(result, segment_id)
            
        except Exception as e:
            processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Record failed API call
            record_api_call(
                call_type="unified_relevance_analysis",
                tokens_used=0,
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )
            raise
    
    def _build_comprehensive_prompt(self,
                                  segment_text: str,
                                  category_definitions: Dict[str, str],
                                  research_question: str,
                                  coding_rules: List[str],
                                  context_paraphrases: Optional[List[str]] = None) -> str:
        """
        Build a comprehensive prompt using centralized QCA_Prompts methodology.
        """
        # Update prompt handler with current context
        self.update_prompt_context(research_question, coding_rules, category_definitions)
        
        # Format categories for standard prompt
        categories_overview = []
        for name, definition in category_definitions.items():
            categories_overview.append({
                'name': name,
                'definition': definition,
                'subcategories': {},  # Can be extended
                'examples': [],
                'rules': []
            })
        
        # Use centralized comprehensive deductive prompt
        return self.prompt_handler.get_comprehensive_deductive_prompt(
            segment_text=segment_text,
            categories_overview=categories_overview,
            multiple_coding_threshold=self.multiple_coding_threshold,
            context_paraphrases=context_paraphrases
        )
    
    def _parse_comprehensive_result(self, result: Dict[str, Any], segment_id: str) -> UnifiedAnalysisResult:
        """
        Parse comprehensive analysis result into structured format.
        """
        # Extract relevance scores
        relevance_scores = result.get("relevance_scores", {})
        
        # Get primary category
        primary_category = result.get("primary_category", "")
        
        # Get all categories with details
        all_categories = result.get("all_categories", [])
        
        # Determine if multiple coding is required
        requires_multiple_coding = result.get("requires_multiple_coding", False)
        
        # Get confidence score
        confidence = result.get("confidence", 0.5)
        
        # Extract additional coding details
        subcategories = result.get("subcategories", [])
        keywords = result.get("keywords", "")
        paraphrase = result.get("paraphrase", "")
        justification = result.get("justification", "")
        
        return UnifiedAnalysisResult(
            segment_id=segment_id,
            primary_category=primary_category,
            all_categories=all_categories,
            relevance_scores=relevance_scores,
            confidence=confidence,
            requires_multiple_coding=requires_multiple_coding,
            multiple_coding_threshold=self.multiple_coding_threshold,
            subcategories=subcategories,
            keywords=keywords,
            paraphrase=paraphrase,
            justification=justification
        )
    
    def _estimate_tokens(self, prompt: str, response: Dict[str, Any]) -> int:
        """
        Estimate token usage for API call tracking.
        """
        # Simple estimation: ~4 tokens per 100 characters
        prompt_chars = len(prompt)
        
        if isinstance(response, dict) and 'choices' in response:
            response_text = response['choices'][0].get('message', {}).get('content', '')
            response_chars = len(response_text)
        else:
            response_chars = 0
            
        total_chars = prompt_chars + response_chars
        return max(1, int(total_chars * 0.04))
    
    async def analyze_batch(self,
                           segments: List[Dict[str, str]],
                           category_definitions: Dict[str, str],
                           research_question: str,
                           coding_rules: List[str],
                           batch_size: int = 5,
                           temperature: Optional[float] = None,
                           context_paraphrases: Optional[List[str]] = None) -> List[UnifiedAnalysisResult]:
        """
        Analyze a batch of segments with unified processing.
        
        Args:
            segments: List of dicts with 'segment_id' and 'text'
            category_definitions: Dictionary mapping category names to descriptions
            research_question: Research question context
            coding_rules: List of coding rules
            batch_size: Maximum segments per API call
            
        Returns:
            List of UnifiedAnalysisResult objects
        """
        results = []
        
        # Process in batches
        total_batches = (len(segments) + batch_size - 1) // batch_size
        print(f"   🔄 Verarbeite {len(segments)} Segmente in {total_batches} Batches (Batch-Größe: {batch_size})")
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            
            if len(batch) == 0:
                continue
            
            batch_num = (i // batch_size) + 1
            print(f"   Kodierungs-Teil-Batch {batch_num}/{total_batches}: {len(batch)} Segmente")
                
            try:
                # Track API call
                start_time = asyncio.get_event_loop().time()
                token_counter.start_request()  # Track request start
                
                # Build batch prompt
                prompt = self._build_batch_prompt(
                    segments=batch,
                    category_definitions=category_definitions,
                    research_question=research_question,
                    coding_rules=coding_rules,
                    context_paraphrases=context_paraphrases
                )
                
                # Use provided temperature or fallback to instance temperature
                use_temperature = temperature if temperature is not None else self.temperature
                
                response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse. Du antwortest auf deutsch. Antworte ausschliesslich mit einem JSON-Objekt."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=use_temperature,
                    response_format={"type": "json_object"}
                )
                
                token_counter.track_response(response, self.model_name)  # Track response
                processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                
                # Record the API call (for additional metrics)
                record_api_call(
                    call_type="unified_relevance_analysis_batch",
                    tokens_used=self._estimate_tokens(prompt, response),
                    processing_time_ms=processing_time_ms,
                    success=True
                )
                
                llm_response = LLMResponse(response)
                batch_result_json = json.loads(llm_response.extract_json())
                
                # Parse batch results
                parsed_results = self._parse_batch_result(batch_result_json, batch)
                results.extend(parsed_results)
                
            except Exception as e:
                # Log error and potentially implement fallback here
                processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                record_api_call(
                    call_type="unified_relevance_analysis_batch",
                    tokens_used=0,
                    processing_time_ms=processing_time_ms,
                    success=False,
                    error_message=str(e)
                )
                # Fallback to individual processing on failure
                # OR raise/return partial errors. For now, raising.
                raise e

        return results

    def _build_batch_prompt(self,
                          segments: List[Dict[str, str]],
                          category_definitions: Dict[str, str],
                          research_question: str,
                          coding_rules: List[str],
                          context_paraphrases: Optional[List[str]] = None) -> str:
        """
        Build batch prompt using centralized QCA_Prompts methodology.
        """
        # Update prompt handler with current context
        self.update_prompt_context(research_question, coding_rules, category_definitions)
        
        # Format categories for standard prompt
        categories_overview = []
        for name, definition in category_definitions.items():
            if hasattr(definition, 'definition'):
                # CategoryDefinition object
                categories_overview.append({
                    'name': name,
                    'definition': definition.definition,
                    'subcategories': definition.subcategories or {},
                    'examples': definition.examples or [],
                    'rules': definition.rules or []
                })
            else:
                # String definition
                categories_overview.append({
                    'name': name,
                    'definition': str(definition),
                    'subcategories': {},
                    'examples': [],
                    'rules': []
                })
        
        # Use centralized batch deductive prompt
        return self.prompt_handler.get_batch_deductive_prompt(
            segments=segments,
            categories_overview=categories_overview,
            context_paraphrases=context_paraphrases
        )

    def _parse_batch_result(self, batch_json: Dict[str, Any], original_segments: List[Dict[str, str]]) -> List[UnifiedAnalysisResult]:
        """
        Parse the batch JSON response into UnifiedAnalysisResult objects.
        """
        results_list = batch_json.get("results", [])
        parsed_objects = []
        
        # Create a map for quick lookup of original segment data if needed
        # (though currently we just need the IDs to match up or verify)
        
        for item in results_list:
            segment_id = item.get("segment_id")
            
            # Basic validation that we got a result for a known segment
            if not segment_id:
                continue

            parsed_objects.append(self._parse_comprehensive_result(item, segment_id))
            
        return parsed_objects
    
    async def analyze_batch_inductive(self,
                                    segments: List[Dict[str, str]],
                                    research_question: str,
                                    batch_size: int = 3,
                                    temperature: Optional[float] = None,
                                    context_paraphrases: Optional[List[str]] = None,
                                    existing_categories: Optional[Dict[str, Any]] = None,
                                    batch_number: int = 1,
                                    total_batches: int = 1,
                                    material_coverage: float = 0.0) -> List[Dict[str, Any]]:
        """
        Analyze a batch of segments for inductive category development with saturation tracking.
        
        Args:
            segments: Segments to analyze
            research_question: Research question
            batch_size: Size of processing batches
            temperature: LLM temperature
            context_paraphrases: Context paraphrases (not used in inductive mode)
            existing_categories: Categories from previous batches
            batch_number: Current batch number
            total_batches: Total number of batches
            material_coverage: Percentage of material processed so far
        """
        results = []
        
        # Process in batches
        total_internal_batches = (len(segments) + batch_size - 1) // batch_size
        print(f"   🔄 Verarbeite {len(segments)} Segmente in {total_internal_batches} internen Batches (Batch-Größe: {batch_size})")
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            if not batch: continue
            
            internal_batch_num = (i // batch_size) + 1
            print(f"   📦 Induktive-Batch {internal_batch_num}/{total_internal_batches}: {len(batch)} Segmente")
            
            try:
                from QCA_AID_assets.utils.tracking.token_tracker import get_global_token_counter
                token_counter = get_global_token_counter()
                token_counter.start_request()  # Track request start
                
                prompt = self._build_inductive_batch_prompt(
                    batch, 
                    research_question,
                    existing_categories=existing_categories,
                    batch_number=batch_number,
                    total_batches=total_batches,
                    material_coverage=material_coverage
                )
                
                # Use provided temperature or fallback to instance temperature
                use_temperature = temperature if temperature is not None else self.temperature
                
                response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse und induktive Kategorienentwicklung. Du antwortest auf Deutsch. Antworte ausschließlich mit einem JSON-Objekt."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=use_temperature,
                    response_format={"type": "json_object"}
                )
                
                token_counter.track_response(response, self.model_name)
                
                # Parse response
                from QCA_AID_assets.utils.llm.response import LLMResponse
                llm_response = LLMResponse(response)
                result = json.loads(llm_response.extract_json())
                
                # Extract results and development assessment
                batch_results = result.get('results', [])
                development_assessment = result.get('development_assessment', {})
                
                # Add development assessment to each result for tracking
                for batch_result in batch_results:
                    batch_result['development_assessment'] = development_assessment
                
                results.extend(batch_results)
                
                print(f"   ✅ Batch {internal_batch_num}: {len(batch_results)} Ergebnisse")
                if development_assessment:
                    saturation = development_assessment.get('theoretical_saturation', 0.0)
                    recommendation = development_assessment.get('recommendation', 'continue')
                    print(f"      📊 Sättigung: {saturation:.2f}, Empfehlung: {recommendation}")
                
            except Exception as e:
                print(f"   ❌ Fehler in Induktive-Batch {internal_batch_num}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        return results

    def _build_inductive_batch_prompt(self, segments: List[Dict[str, str]], research_question: str, 
                                     existing_categories: Optional[Dict[str, Any]] = None,
                                     batch_number: int = 1, total_batches: int = 1,
                                     material_coverage: float = 0.0) -> str:
        """
        Build inductive batch prompt using the same methodology as standard analysis.
        
        Args:
            segments: List of segments to analyze
            research_question: Research question
            existing_categories: Existing categories from previous batches
            batch_number: Current batch number
            total_batches: Total number of batches
            material_coverage: Percentage of material processed so far
        """
        # Update prompt handler with current context
        self.update_prompt_context(research_question, ["Standard-Kodierregeln"], {})
        
        # Format segments for prompt
        segments_text = "\n\n".join([f"SEGMENT {i+1}:\n{s['text']}" for i, s in enumerate(segments)])
        
        # Formatiere bestehende induktive Kategorien als Kontext (aber nicht als Einschränkung)
        existing_context = ""
        if existing_categories:
            existing_names = list(existing_categories.keys())
            existing_context = f"""
            BESTEHENDE INDUKTIVE KATEGORIEN (als Kontext, NICHT als Einschränkung):
            {', '.join(existing_names)}
            
            WICHTIG: Entwickle NEUE, EIGENSTÄNDIGE Kategorien, die sich thematisch von den bestehenden unterscheiden.
            Beachte aber die bereits entwickelten Kategorien um Redundanzen zu vermeiden.
            """
        
        # Material- und Batch-Kontext für Sättigungsbeurteilung
        progress_context = f"""
        BATCH-KONTEXT:
        - Aktueller Batch: {batch_number}/{total_batches}
        - Materialabdeckung: {material_coverage:.1%}
        - Bereits entwickelte Kategorien: {len(existing_categories) if existing_categories else 0}
        """
        
        return f"""
        FORSCHUNGSFRAGE (ZENTRALE ORIENTIERUNG):
        {research_question}

        INDUCTIVE MODE: Vollständige induktive Kategorienentwicklung

        {existing_context}

        {progress_context}

        WICHTIG - FORSCHUNGSFRAGE BERÜCKSICHTIGEN:
        - Prüfe ZUERST, ob ein Textsegment zur Forschungsfrage relevant ist
        - Entwickle NUR Kategorien für Aspekte, die zur Beantwortung der Forschungsfrage beitragen
        - Kategoriennamen sollten Terminologie der Forschungsfrage aufgreifen, wo sinnvoll
        - Irrelevante textliche Details NICHT kategorisieren

        AUFGABE: Entwickle völlig NEUE Hauptkategorien aus den folgenden Textsegmenten.
        Dies ist ein eigenständiges induktives Kategoriensystem, unabhängig von deduktiven Kategorien.

        REGELN FÜR INDUCTIVE MODE:
        - Entwickle 1-4 NEUE Hauptkategorien
        - Jede Kategorie muss mindestens 2 Textbelege haben
        - Konfidenz mindestens 0.7
        - Kategorien müssen thematisch eigenständig und relevant für die Forschungsfrage sein
        - Erstelle auch 2-4 Subkategorien pro Hauptkategorie
        - Kategorien sollen neue Aspekte der Forschungsfrage beleuchten
        - Vermeide Redundanzen zu bereits entwickelten Kategorien

        SÄTTIGUNGSBEURTEILUNG:
        - Bewerte die theoretische Sättigung basierend auf Kategorienqualität und Themenabdeckung
        - Berücksichtige den Materialfortschritt und bereits entwickelte Kategorien
        - Empfehle "continue" wenn neue Themen gefunden werden
        - Empfehle "pause" wenn wenige neue Aspekte, aber Material noch nicht vollständig
        - Empfehle "stop" wenn hohe Sättigung erreicht oder Material vollständig verarbeitet

        TEXTSEGMENTE:
        {segments_text}

        Antworte NUR mit JSON:
        {{
            "results": [
                {{
                    "segment_id": "segment_id_here",
                    "categories": ["Kategorie1", "Kategorie2"],
                    "category_definitions": {{
                        "Kategorie1": "Ausführliche Definition (mindestens 20 Wörter)",
                        "Kategorie2": "Ausführliche Definition (mindestens 20 Wörter)"
                    }},
                    "subcategories": {{
                        "Kategorie1": [
                            {{
                                "name": "Subkategorie Name", 
                                "definition": "Subkategorie Definition"
                            }}
                        ]
                    }},
                    "confidence": 0.8,
                    "reasoning": "Begründung der Kategorienentwicklung",
                    "evidence": ["Textbelege aus den Segmenten"],
                    "thematic_justification": "Warum diese Kategorien eigenständige Themenbereiche abbilden"
                }}
            ],
            "development_assessment": {{
                "categories_developed": 0,
                "theoretical_saturation": 0.0,
                "material_coverage": {material_coverage},
                "new_themes_found": true,
                "category_quality": 0.0,
                "category_diversity": 0.0,
                "recommendation": "continue",
                "saturation_reasoning": "Begründung für die Sättigungseinschätzung"
            }}
        }}
        """

    async def analyze_batch_abductive(self,
                                    segments: List[Dict[str, str]],
                                    category_definitions: Dict[str, str],
                                    research_question: str,
                                    batch_size: int = 4,
                                    temperature: Optional[float] = None,
                                    context_paraphrases: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Analyze batch for abductive mode (extending knowledge).
        """
        results = []
        
        # Process in batches
        total_batches = (len(segments) + batch_size - 1) // batch_size
        print(f"   🔄 Verarbeite {len(segments)} Segmente in {total_batches} Batches (Batch-Größe: {batch_size})")
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            if not batch: continue
            
            batch_num = (i // batch_size) + 1
            print(f"   📦 Abduktive-Batch {batch_num}/{total_batches}: {len(batch)} Segmente")
            
            try:
                token_counter.start_request()  # Track request start
                
                prompt = self._build_abductive_batch_prompt(batch, category_definitions, research_question)
                # Use provided temperature or fallback to instance temperature
                use_temperature = temperature if temperature is not None else self.temperature
                
                response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für qualitative Inhaltsanalyse (Abduktiv). Antworte als JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=use_temperature,
                    response_format={"type": "json_object"}
                )
                
                token_counter.track_response(response, self.model_name)  # Track response
                
                llm_response = LLMResponse(response)
                batch_json = json.loads(llm_response.extract_json())
                
                # Handle new abductive schema with extended_categories and segment_assignments
                if "segment_assignments" in batch_json:
                    # Store extended_categories information for later processing
                    batch_results = batch_json.get("segment_assignments", [])
                    
                    # Add extended_categories info to each result for processing
                    extended_categories = batch_json.get("extended_categories", {})
                    for result in batch_results:
                        result["_extended_categories"] = extended_categories
                    
                    results.extend(batch_results)
                else:
                    # Fallback for old schema
                    results.extend(batch_json.get("results", []))
            except Exception:
                pass
        return results

    def _build_abductive_batch_prompt(self, segments: List[Dict[str, str]], categories: Dict[str, str], rq: str) -> str:
        """Build abductive batch prompt using standard QCA_Prompts methodology."""
        # Update prompt handler with current context
        self.update_prompt_context(rq, ["Standard-Kodierregeln"], categories)
        
        # Format segments for standard prompt
        segments_text = [f"SEGMENT {i+1}: {s['text']}" for i, s in enumerate(segments)]
        
        # Format existing categories
        categories_text = "\n".join([f"- {name}: {definition}" for name, definition in categories.items()])
        
        # Use centralized mode instructions and schema
        mode_instructions = self.prompt_handler.get_abductive_mode_instructions(categories_text)
        json_schema = self.prompt_handler.get_abductive_json_schema()
        
        return self.prompt_handler.get_category_batch_analysis_prompt(
            current_categories_text=categories_text,
            segments=segments_text,
            mode_instructions=mode_instructions,
            json_schema=json_schema
        )

    async def analyze_batch_grounded(self,
                                   segments: List[Dict[str, str]],
                                   research_question: str,
                                   batch_size: int = 3,
                                   temperature: Optional[float] = None,
                                   context_paraphrases: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Analyze batch for grounded theory (open coding).
        """
        results = []
        
        # Process in batches
        total_batches = (len(segments) + batch_size - 1) // batch_size
        print(f"   🔄 Verarbeite {len(segments)} Segmente in {total_batches} Batches (Batch-Größe: {batch_size})")
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            if not batch: continue
            
            batch_num = (i // batch_size) + 1
            print(f"   📦 Grounded-Batch {batch_num}/{total_batches}: {len(batch)} Segmente")
            
            try:
                token_counter.start_request()  # Track request start
                
                prompt = self._build_grounded_batch_prompt(batch, research_question)
                # Use provided temperature or fallback to instance temperature
                use_temperature = temperature if temperature is not None else self.temperature
                
                response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte für Grounded Theory (Open Coding). Antworte als JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=use_temperature,
                    response_format={"type": "json_object"}
                )
                
                token_counter.track_response(response, self.model_name)  # Track response
                
                llm_response = LLMResponse(response)
                batch_json = json.loads(llm_response.extract_json())
                results.extend(batch_json.get("results", []))
            except Exception:
                pass
        return results

    def _build_grounded_batch_prompt(self, segments: List[Dict[str, str]], rq: str) -> str:
        """Build grounded batch prompt using standard QCA_Prompts methodology."""
        # Update prompt handler with current context
        self.update_prompt_context(rq, ["Standard-Kodierregeln"], {})
        
        # Format segments for standard prompt
        segments_text = [f"SEGMENT {i+1}: {s['text']}" for i, s in enumerate(segments)]
        
        # Use centralized JSON schema
        json_schema = self.prompt_handler.get_grounded_json_schema()
        
        return self.prompt_handler.get_grounded_analysis_prompt(
            segments=segments_text,
            existing_subcodes=[],  # Wird später mit tatsächlichen Subcodes gefüllt
            json_schema=json_schema
        )
    
    def calculate_savings(self, 
                         individual_calls_per_segment: float,
                         batch_size: int = 5) -> Dict[str, float]:
        """
        Calculate potential API call savings.
        
        Args:
            individual_calls_per_segment: Average API calls per segment in current implementation
            batch_size: Target batch size for optimization
            
        Returns:
            Dictionary with savings metrics
        """
        # Current implementation: multiple calls per segment
        # Our implementation: 1 call per segment (or less with batching)
        
        if batch_size > 1:
            # With perfect batching: 1 call per batch
            calls_per_segment_optimized = 1.0 / batch_size
        else:
            calls_per_segment_optimized = 1.0
        
        reduction = individual_calls_per_segment - calls_per_segment_optimized
        reduction_percent = (reduction / individual_calls_per_segment) * 100 if individual_calls_per_segment > 0 else 0
        
        return {
            "current_calls_per_segment": individual_calls_per_segment,
            "optimized_calls_per_segment": calls_per_segment_optimized,
            "reduction_per_segment": reduction,
            "reduction_percent": reduction_percent,
            "estimated_savings_100_segments": reduction * 100,
            "batch_size": batch_size
        }


# Factory function for easy integration
def create_unified_analyzer(llm_provider: LLMProvider, 
                           model_name: str = "gpt-4",
                           temperature: float = 0.3,
                           multiple_coding_threshold: float = 0.7) -> UnifiedRelevanceAnalyzer:
    """
    Create a UnifiedRelevanceAnalyzer instance.
    
    Args:
        llm_provider: LLM provider instance
        model_name: Model to use
        temperature: Temperature setting
        multiple_coding_threshold: Threshold for multiple coding decisions
        
    Returns:
        UnifiedRelevanceAnalyzer instance
    """
    return UnifiedRelevanceAnalyzer(
        llm_provider=llm_provider,
        model_name=model_name,
        temperature=temperature,
        multiple_coding_threshold=multiple_coding_threshold
    )


# Test function
async def test_unified_analyzer():
    """Test the unified analyzer with sample data."""
    import sys
    import os
    
    # Add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    # Mock LLM provider for testing
    class MockLLMProvider:
        async def create_completion(self, model, messages, temperature, response_format):
            # Return mock response
            prompt = messages[1]['content'] if len(messages) > 1 else messages[0]['content']
            
            # Simulate comprehensive analysis response
            mock_response = {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "segment_id": "test_segment_001",
                            "relevance_scores": {
                                "Environmental Impact": 0.85,
                                "Economic Factors": 0.72,
                                "Policy Implementation": 0.63,
                                "Technology Adoption": 0.41,
                                "Stakeholder Engagement": 0.58
                            },
                            "primary_category": "Environmental Impact",
                            "all_categories": [
                                {
                                    "category": "Environmental Impact",
                                    "relevance_score": 0.85,
                                    "reasoning": "Segment diskutiert Klimawandelanpassung"
                                },
                                {
                                    "category": "Economic Factors", 
                                    "relevance_score": 0.72,
                                    "reasoning": "Erwähnt regionale Vulnerabilitäten"
                                }
                            ],
                            "requires_multiple_coding": True,
                            "confidence": 0.88,
                            "analysis_summary": "Segment ist relevant für Umweltauswirkungen und Wirtschaftsfaktoren"
                        })
                    }
                }]
            }
            
            return mock_response
    
    # Create analyzer with mock provider
    analyzer = UnifiedRelevanceAnalyzer(
        llm_provider=MockLLMProvider(),
        model_name="gpt-4",
        temperature=0.3
    )
    
    # Test data
    category_definitions = {
        "Environmental Impact": "Auswirkungen auf die Umwelt und Ökosysteme",
        "Economic Factors": "Wirtschaftliche Aspekte und Kosten-Nutzen-Analysen",
        "Policy Implementation": "Umsetzung politischer Maßnahmen",
        "Technology Adoption": "Einführung und Nutzung neuer Technologien",
        "Stakeholder Engagement": "Einbindung von Interessengruppen"
    }
    
    research_question = "Wie beeinflussen Klimawandelanpassungsstrategien regionale Vulnerabilitäten?"
    coding_rules = [
        "Konzentriere dich auf explizite Aussagen über Auswirkungen",
        "Berücksichtige implizite Zusammenhänge",
        "Vermeide Doppelkodierung ähnlicher Konzepte"
    ]
    
    # Test single segment analysis
    result = await analyzer.analyze_relevance_comprehensive(
        segment_text="Climate change adaptation strategies must account for regional variations in vulnerability.",
        segment_id="test_segment_001",
        category_definitions=category_definitions,
        research_question=research_question,
        coding_rules=coding_rules
    )
    
    print("Unified Analysis Result:")
    print(f"  Segment ID: {result.segment_id}")
    print(f"  Primary Category: {result.primary_category}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Requires Multiple Coding: {result.requires_multiple_coding}")
    print(f"  Relevance Scores: {result.relevance_scores}")
    
    # Calculate savings
    savings = analyzer.calculate_savings(
        individual_calls_per_segment=2.2,
        batch_size=5
    )
    
    print("\nEstimated Savings:")
    print(f"  Current API calls/segment: {savings['current_calls_per_segment']:.2f}")
    print(f"  Optimized API calls/segment: {savings['optimized_calls_per_segment']:.2f}")
    print(f"  Reduction: {savings['reduction_percent']:.1f}%")
    print(f"  Savings per 100 segments: {savings['estimated_savings_100_segments']:.0f} calls")
    
    return result


if __name__ == "__main__":
    # Run test
    asyncio.run(test_unified_analyzer())