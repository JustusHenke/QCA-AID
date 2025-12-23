"""
Optimization module for QCA-AID API call efficiency improvements.

This module provides optimized processing for all four analysis modes:
- Deductive: Consolidated relevance checks and batch processing
- Inductive: Batch category development
- Abductive: Batch subcategory extension  
- Grounded: Batch subcode extraction with saturation optimization

Extended with Dynamic Cache System for Multi-Coder support and Strategy Pattern.
"""

from .controller import OptimizationController
from .unified_relevance_analyzer import UnifiedRelevanceAnalyzer
from .batch_processor import BatchProcessor
from .cache import (
    ModeAwareCache, 
    CacheEntry, 
    CacheStatistics, 
    CacheEventListener, 
    DefaultCacheEventListener,
    CacheInterface
)
from .cache_strategies import (
    CacheStrategy,
    SingleCoderCacheStrategy,
    MultiCoderCacheStrategy,
    ModeSpecificCacheStrategy,
    CacheStrategyFactory
)
from .confidence_scorer import ConfidenceScorer

__all__ = [
    'OptimizationController',
    'UnifiedRelevanceAnalyzer',
    'BatchProcessor',
    'ModeAwareCache',
    'CacheEntry',
    'CacheStatistics', 
    'CacheEventListener',
    'DefaultCacheEventListener',
    'CacheInterface',
    'CacheStrategy',
    'SingleCoderCacheStrategy',
    'MultiCoderCacheStrategy',
    'ModeSpecificCacheStrategy',
    'CacheStrategyFactory',
    'ConfidenceScorer'
]

__version__ = '0.3.0'