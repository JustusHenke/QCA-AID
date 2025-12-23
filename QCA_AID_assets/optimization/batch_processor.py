"""
Batch Processor
Handles batch processing of segments for API call optimization.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
import statistics
from dataclasses import dataclass
from QCA_AID_assets.optimization.unified_relevance_analyzer import create_unified_analyzer


@dataclass
class BatchResult:
    """Results from batch processing."""
    batch_id: str
    segment_ids: List[str]
    analysis_mode: str
    api_calls_made: int
    api_calls_saved: int
    processing_time_ms: float
    success: bool
    results: List[Dict[str, Any]]
    error_message: Optional[str] = None


class BatchProcessor:
    """
    Processes segments in batches to optimize API calls.
    
    Implements different batching strategies for each analysis mode:
    - Deductive: Category-based batching
    - Inductive: Similarity-based batching
    - Abductive: Hypothesis-based batching
    - Grounded: Iterative batching with saturation
    """
    
    def __init__(self, llm_provider, model_name: str = "gpt-4"):
        """
        Initialize batch processor.
        
        Args:
            llm_provider: LLM provider instance
            model_name: Model to use for batch processing
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.unified_analyzer = create_unified_analyzer(llm_provider, model_name)
        
        # Batching configurations
        self.config = {
            "deductive": {
                "max_batch_size": 5,
                "min_batch_size": 2,
                "similarity_threshold": 0.6,
                "quality_threshold": 0.75,
                "fallback_to_single": True
            },
            "inductive": {
                "max_batch_size": 3,
                "min_batch_size": 2,
                "similarity_threshold": 0.7,
                "quality_threshold": 0.7,
                "fallback_to_single": True
            },
            "abductive": {
                "max_batch_size": 4,
                "min_batch_size": 2,
                "similarity_threshold": 0.65,
                "quality_threshold": 0.72,
                "fallback_to_single": True
            },
            "grounded": {
                "max_batch_size": 3,
                "min_batch_size": 2,
                "similarity_threshold": 0.75,
                "quality_threshold": 0.68,
                "fallback_to_single": True,
                "saturation_check_frequency": 5  # batches
            }
        }
    
    async def process_batch(self,
                           segments: List[Dict[str, Any]],
                           analysis_mode: str,
                           **kwargs) -> BatchResult:
        """
        Process a batch of segments.
        
        Args:
            segments: List of segments with 'segment_id' and 'text'
            analysis_mode: Analysis mode
            **kwargs: Mode-specific parameters
            
        Returns:
            BatchResult with processing results
        """
        batch_id = f"{analysis_mode}_batch_{int(asyncio.get_event_loop().time())}"
        
        try:
            # Validate batch
            if not segments:
                return BatchResult(
                    batch_id=batch_id,
                    segment_ids=[],
                    analysis_mode=analysis_mode,
                    api_calls_made=0,
                    api_calls_saved=0,
                    processing_time_ms=0,
                    success=True,
                    results=[]
                )
            
            # Apply mode-specific batching strategy
            if analysis_mode == "deductive":
                return await self._process_deductive_batch(segments, batch_id, **kwargs)
            elif analysis_mode == "inductive":
                return await self._process_inductive_batch(segments, batch_id, **kwargs)
            elif analysis_mode == "abductive":
                return await self._process_abductive_batch(segments, batch_id, **kwargs)
            elif analysis_mode == "grounded":
                return await self._process_grounded_batch(segments, batch_id, **kwargs)
            else:
                raise ValueError(f"Unsupported analysis mode: {analysis_mode}")
                
        except Exception as e:
            # Fallback to individual processing
            return await self._fallback_processing(segments, analysis_mode, batch_id, str(e), **kwargs)
    
    async def _process_deductive_batch(self,
                                     segments: List[Dict[str, Any]],
                                     batch_id: str,
                                     category_definitions: Optional[Dict[str, str]] = None,
                                     research_question: Optional[str] = None,
                                     coding_rules: Optional[List[str]] = None,
                                     **kwargs) -> BatchResult:
        """
        Process deductive batch with category-based optimization.
        
        Strategy: Single API call to analyze all segments against all categories.
        """
        config = self.config["deductive"]
        max_batch_size = config["max_batch_size"]
        
        # Limit batch size
        if len(segments) > max_batch_size:
            segments = segments[:max_batch_size]
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Use unified analyzer for batch processing
            results = await self.unified_analyzer.analyze_batch(
                segments=segments,
                category_definitions=category_definitions,
                research_question=research_question,
                coding_rules=coding_rules,
                batch_size=len(segments)
            )
            
            # Convert results to dict format
            dict_results = []
            for r in results:
                dict_results.append({
                    'segment_id': r.segment_id,
                    'primary_category': r.primary_category,
                    'all_categories': r.all_categories,
                    'relevance_scores': r.relevance_scores,
                    'confidence': r.confidence,
                    'requires_multiple_coding': r.requires_multiple_coding,
                    'analysis_mode': 'deductive'
                })
                
            processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return BatchResult(
                batch_id=batch_id,
                segment_ids=[s['segment_id'] for s in segments],
                analysis_mode="deductive",
                api_calls_made=1,
                api_calls_saved=len(segments) * 2 - 1 if len(segments) > 0 else 0,
                processing_time_ms=processing_time_ms,
                success=True,
                results=dict_results
            )
            
        except Exception as e:
            return await self._fallback_processing(segments, "deductive", batch_id, str(e), **kwargs)
    
    def _build_deductive_batch_prompt(self,
                                     segments: List[Dict[str, Any]],
                                     category_definitions: Dict[str, str],
                                     research_question: str,
                                     coding_rules: List[str]) -> str:
        """
        Build prompt for deductive batch analysis using centralized QCA_Prompts.
        """
        # Import QCA_Prompts if not already available
        try:
            from QCA_AID_assets.QCA_Prompts import QCAPrompts
        except ImportError:
            # Fallback to simple prompt if import fails
            return "placeholder prompt"
        
        # Initialize prompt handler
        prompt_handler = QCAPrompts(
            forschungsfrage=research_question,
            kodierregeln=coding_rules,
            deduktive_kategorien=category_definitions
        )
        
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
        
        # Convert segments to expected format
        formatted_segments = []
        for segment in segments:
            formatted_segments.append({
                'segment_id': segment.get('segment_id', ''),
                'text': segment.get('text', '')
            })
        
        # Use centralized batch deductive prompt
        return prompt_handler.get_batch_deductive_prompt(
            segments=formatted_segments,
            categories_overview=categories_overview,
            context_paraphrases=None
        )
    
    def _parse_deductive_batch_response(self, response: Dict[str, Any], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Implementation placeholder
        return []
    
    async def _fallback_processing(self,
                                  segments: List[Dict[str, Any]],
                                  analysis_mode: str,
                                  batch_id: str,
                                  error_message: str,
                                  **kwargs) -> BatchResult:
        # Implementation placeholder
        return BatchResult(
            batch_id=batch_id,
            segment_ids=[s['segment_id'] for s in segments],
            analysis_mode=analysis_mode,
            api_calls_made=0,
            api_calls_saved=0,
            processing_time_ms=0,
            success=False,
            results=[],
            error_message=error_message
        )
    
    def calculate_batch_efficiency(self,
                                  batch_result: BatchResult,
                                  baseline_calls_per_segment: float) -> Dict[str, float]:
        """
        Calculate efficiency metrics for batch processing.
        
        Args:
            batch_result: Batch processing result
            baseline_calls_per_segment: Baseline API calls per segment
            
        Returns:
            Efficiency metrics
        """
        if not batch_result.success or not batch_result.segment_ids:
            return {
                'efficiency': 0.0,
                'savings_percentage': 0.0,
                'calls_per_segment': baseline_calls_per_segment,
                'time_per_segment_ms': 0.0
            }
        
        segment_count = len(batch_result.segment_ids)
        
        # Calculate metrics
        calls_per_segment = batch_result.api_calls_made / segment_count if segment_count > 0 else 0
        time_per_segment_ms = batch_result.processing_time_ms / segment_count if segment_count > 0 else 0
        
        # Calculate efficiency (0-1, higher is better)
        if baseline_calls_per_segment > 0:
            efficiency = (baseline_calls_per_segment - calls_per_segment) / baseline_calls_per_segment
            efficiency = max(0.0, min(1.0, efficiency))
        else:
            efficiency = 0.0
        
        savings_percentage = efficiency * 100
        
        return {
            'efficiency': efficiency,
            'savings_percentage': savings_percentage,
            'calls_per_segment': calls_per_segment,
            'time_per_segment_ms': time_per_segment_ms,
            'total_calls_saved': batch_result.api_calls_saved,
            'segment_count': segment_count
        }