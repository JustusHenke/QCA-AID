"""
Mode-Aware Cache
Intelligent caching system for intermediate results to reduce API calls.
Extended for Dynamic Cache System with Multi-Coder support and Strategy Pattern.
"""

import json
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import OrderedDict
from abc import ABC, abstractmethod

# Import strategy classes
from .cache_strategies import CacheStrategy, CacheStrategyFactory, SingleCoderCacheStrategy

# Set up logger for cache operations
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Extended cache entry with metadata for multi-coder support."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    analysis_mode: str = ""
    segment_id: str = ""
    category: str = ""
    confidence: float = 0.0
    # New fields for dynamic cache system
    coder_id: Optional[str] = None
    operation_type: str = ""
    is_shared: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata dict if None."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CacheStatistics:
    """Statistics for cache performance monitoring."""
    total_entries: int
    shared_entries: int
    coder_specific_entries: int
    hit_rate_overall: float
    hit_rate_by_coder: Dict[str, float]
    memory_usage_mb: float
    strategy_type: str
    hits_by_operation: Dict[str, int] = None
    misses_by_operation: Dict[str, int] = None
    # Enhanced monitoring fields
    memory_usage_by_type: Dict[str, float] = None
    cache_efficiency_by_coder: Dict[str, Dict[str, float]] = None
    error_counts: Dict[str, int] = None
    performance_metrics: Dict[str, Any] = None
    # Manual coder integration fields
    manual_coder_stats: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize operation stats if None."""
        if self.hits_by_operation is None:
            self.hits_by_operation = {}
        if self.misses_by_operation is None:
            self.misses_by_operation = {}
        if self.memory_usage_by_type is None:
            self.memory_usage_by_type = {}
        if self.cache_efficiency_by_coder is None:
            self.cache_efficiency_by_coder = {}
        if self.error_counts is None:
            self.error_counts = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.manual_coder_stats is None:
            self.manual_coder_stats = {}


class CacheEventListener(ABC):
    """Abstract base class for cache event listeners."""
    
    @abstractmethod
    def on_cache_hit(self, key: str, entry: CacheEntry) -> None:
        """Called when cache hit occurs."""
        pass
    
    @abstractmethod
    def on_cache_miss(self, key: str, operation_type: str) -> None:
        """Called when cache miss occurs."""
        pass
    
    @abstractmethod
    def on_cache_set(self, key: str, entry: CacheEntry) -> None:
        """Called when entry is stored in cache."""
        pass
    
    @abstractmethod
    def on_cache_evict(self, key: str, entry: CacheEntry) -> None:
        """Called when entry is evicted from cache."""
        pass
    
    @abstractmethod
    def on_cache_invalidate(self, keys: List[str], reason: str) -> None:
        """Called when entries are invalidated."""
        pass


class DefaultCacheEventListener(CacheEventListener):
    """Default implementation of cache event listener with logging."""
    
    def on_cache_hit(self, key: str, entry: CacheEntry) -> None:
        """Log cache hit."""
        logger.debug(f"Cache HIT: {key[:16]}... (coder: {entry.coder_id}, op: {entry.operation_type})")
    
    def on_cache_miss(self, key: str, operation_type: str) -> None:
        """Log cache miss."""
        logger.debug(f"Cache MISS: {key[:16]}... (op: {operation_type})")
    
    def on_cache_set(self, key: str, entry: CacheEntry) -> None:
        """Log cache set."""
        logger.debug(f"Cache SET: {key[:16]}... (coder: {entry.coder_id}, op: {entry.operation_type}, shared: {entry.is_shared})")
    
    def on_cache_evict(self, key: str, entry: CacheEntry) -> None:
        """Log cache eviction."""
        logger.info(f"Cache EVICT: {key[:16]}... (coder: {entry.coder_id}, op: {entry.operation_type})")
    
    def on_cache_invalidate(self, keys: List[str], reason: str) -> None:
        """Log cache invalidation."""
        logger.info(f"Cache INVALIDATE: {len(keys)} entries, reason: {reason}")


class CacheInterface(ABC):
    """Abstract interface for cache operations."""
    
    @abstractmethod
    def get_cache_key(self, operation_type: str, coder_id: Optional[str] = None, **params) -> str:
        """Generate cache key for operation."""
        pass
    
    @abstractmethod
    def should_cache_shared(self, operation_type: str) -> bool:
        """Determine if operation should use shared cache."""
        pass
    
    @abstractmethod
    def should_cache_per_coder(self, operation_type: str) -> bool:
        """Determine if operation should use per-coder cache."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics."""
        pass
    
    @abstractmethod
    def add_event_listener(self, listener: CacheEventListener) -> None:
        """Add event listener for monitoring."""
        pass
    
    @abstractmethod
    def remove_event_listener(self, listener: CacheEventListener) -> None:
        """Remove event listener."""
        pass


class ModeAwareCache(CacheInterface):
    """
    Intelligent cache that understands analysis modes and caching strategies.
    Extended with multi-coder support, monitoring capabilities, and strategy pattern.
    
    Features:
    - Mode-specific TTL (Time To Live)
    - Confidence-based cache validation
    - Similarity-based cache lookup
    - Automatic cache invalidation
    - Memory-aware eviction policies
    - Multi-coder support with shared/specific caching
    - Event-based monitoring and logging
    - Pluggable cache strategies
    """
    
    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 3600, strategy: Optional[CacheStrategy] = None):
        """
        Initialize mode-aware cache.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl_seconds: Default time-to-live in seconds
            strategy: Optional cache strategy (defaults to SingleCoderCacheStrategy)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        
        # Set default strategy if none provided
        self.strategy = strategy or SingleCoderCacheStrategy()
        
        # Event listeners for monitoring
        self.event_listeners: List[CacheEventListener] = []
        self.add_event_listener(DefaultCacheEventListener())
        
        # Mode-specific configurations
        self.mode_configs = {
            "deductive": {
                "ttl": 7200,  # 2 hours
                "confidence_threshold": 0.7,
                "similarity_threshold": 0.8,
                "max_entries_per_category": 50
            },
            "inductive": {
                "ttl": 3600,  # 1 hour
                "confidence_threshold": 0.65,
                "similarity_threshold": 0.75,
                "max_entries_per_category": 100
            },
            "abductive": {
                "ttl": 5400,  # 1.5 hours
                "confidence_threshold": 0.68,
                "similarity_threshold": 0.7,
                "max_entries_per_category": 75
            },
            "grounded": {
                "ttl": 1800,  # 30 minutes (shorter due to iterative nature)
                "confidence_threshold": 0.6,
                "similarity_threshold": 0.65,
                "max_entries_per_category": 150
            }
        }
        
        # Cache storage: OrderedDict for LRU eviction
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Indexes for fast lookup
        self.mode_index: Dict[str, List[str]] = {}
        self.segment_index: Dict[str, List[str]] = {}
        self.category_index: Dict[str, List[str]] = {}
        # New indexes for multi-coder support
        self.coder_index: Dict[str, List[str]] = {}
        self.operation_index: Dict[str, List[str]] = {}
        self.shared_index: List[str] = []
        
        # Enhanced statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
            "total_entries": 0,
            "hits_by_coder": {},
            "misses_by_coder": {},
            "hits_by_operation": {},
            "misses_by_operation": {},
            # Enhanced monitoring fields
            "errors": {},
            "error_contexts": [],
            "memory_usage_by_type": {},
            "performance_timings": {},
            "cache_access_patterns": {},
            "eviction_reasons": {},
            "last_error_timestamp": None,
            "total_errors": 0
        }
    
    def set_strategy(self, strategy: CacheStrategy) -> None:
        """
        Set a new cache strategy.
        
        Args:
            strategy: New cache strategy to use
        """
        old_strategy = self.strategy
        self.strategy = strategy
        
        logger.info(f"Cache strategy changed from {old_strategy.name} to {strategy.name}")
        
        # Invalidate operations that might be affected by strategy change
        operations_to_invalidate = strategy.invalidate_on_config_change()
        if operations_to_invalidate:
            self._invalidate_operations(operations_to_invalidate, f"strategy_change:{old_strategy.name}->{strategy.name}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about current cache strategy.
        
        Returns:
            Strategy information dictionary
        """
        return self.strategy.get_strategy_info()
    
    def _invalidate_operations(self, operations: List[str], reason: str) -> None:
        """
        Invalidate cache entries for specific operations.
        
        Args:
            operations: List of operation types to invalidate
            reason: Reason for invalidation
        """
        keys_to_remove = []
        
        for operation in operations:
            if operation in self.operation_index:
                keys_to_remove.extend(self.operation_index[operation])
        
        # Remove duplicates
        keys_to_remove = list(set(keys_to_remove))
        
        for key in keys_to_remove:
            if key in self.cache:
                entry = self.cache[key]
                self._remove_entry_from_indexes(key, entry)
                del self.cache[key]
                self.stats["invalidations"] += 1
        
        self.stats["total_entries"] = len(self.cache)
        
        if keys_to_remove:
            self._notify_cache_invalidate(keys_to_remove, reason)
    
    def _remove_entry_from_indexes(self, key: str, entry: CacheEntry) -> None:
        """
        Remove entry from all indexes.
        
        Args:
            key: Cache key
            entry: Cache entry
        """
        # Remove from mode index
        if entry.analysis_mode in self.mode_index:
            self.mode_index[entry.analysis_mode] = [
                k for k in self.mode_index[entry.analysis_mode] if k != key
            ]
        
        # Remove from segment index
        if entry.segment_id and entry.segment_id in self.segment_index:
            self.segment_index[entry.segment_id] = [
                k for k in self.segment_index[entry.segment_id] if k != key
            ]
        
        # Remove from category index
        if entry.category and entry.category in self.category_index:
            self.category_index[entry.category] = [
                k for k in self.category_index[entry.category] if k != key
            ]
        
        # Remove from coder index
        if entry.coder_id and entry.coder_id in self.coder_index:
            self.coder_index[entry.coder_id] = [
                k for k in self.coder_index[entry.coder_id] if k != key
            ]
        
        # Remove from operation index
        if entry.operation_type in self.operation_index:
            self.operation_index[entry.operation_type] = [
                k for k in self.operation_index[entry.operation_type] if k != key
            ]
        
        # Remove from shared index
        if entry.is_shared and key in self.shared_index:
            self.shared_index.remove(key)
    
    def add_event_listener(self, listener: CacheEventListener) -> None:
        """Add event listener for monitoring."""
        if listener not in self.event_listeners:
            self.event_listeners.append(listener)
    
    def remove_event_listener(self, listener: CacheEventListener) -> None:
        """Remove event listener."""
        if listener in self.event_listeners:
            self.event_listeners.remove(listener)
    
    def _notify_cache_hit(self, key: str, entry: CacheEntry) -> None:
        """Notify all listeners of cache hit."""
        for listener in self.event_listeners:
            try:
                listener.on_cache_hit(key, entry)
            except Exception as e:
                logger.warning(f"Error in cache hit listener: {e}")
    
    def _notify_cache_miss(self, key: str, operation_type: str) -> None:
        """Notify all listeners of cache miss."""
        for listener in self.event_listeners:
            try:
                listener.on_cache_miss(key, operation_type)
            except Exception as e:
                logger.warning(f"Error in cache miss listener: {e}")
    
    def _notify_cache_set(self, key: str, entry: CacheEntry) -> None:
        """Notify all listeners of cache set."""
        for listener in self.event_listeners:
            try:
                listener.on_cache_set(key, entry)
            except Exception as e:
                logger.warning(f"Error in cache set listener: {e}")
    
    def _notify_cache_evict(self, key: str, entry: CacheEntry) -> None:
        """Notify all listeners of cache eviction."""
        for listener in self.event_listeners:
            try:
                listener.on_cache_evict(key, entry)
            except Exception as e:
                logger.warning(f"Error in cache evict listener: {e}")
    
    def _notify_cache_invalidate(self, keys: List[str], reason: str) -> None:
        """Notify all listeners of cache invalidation."""
        for listener in self.event_listeners:
            try:
                listener.on_cache_invalidate(keys, reason)
            except Exception as e:
                logger.warning(f"Error in cache invalidate listener: {e}")
    
    def _log_error_with_context(self, error: Exception, operation: str, context: Dict[str, Any]) -> None:
        """
        Log error with detailed context for debugging and monitoring.
        
        Args:
            error: Exception that occurred
            operation: Operation that failed
            context: Additional context information
        """
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'operation': operation,
            'context': context,
            'cache_state': {
                'total_entries': len(self.cache),
                'strategy': self.strategy.name,
                'memory_usage_mb': self._calculate_memory_usage()
            }
        }
        
        # Store error in statistics
        error_key = f"{operation}:{type(error).__name__}"
        self.stats["errors"][error_key] = self.stats["errors"].get(error_key, 0) + 1
        self.stats["error_contexts"].append(error_info)
        self.stats["last_error_timestamp"] = error_info['timestamp']
        self.stats["total_errors"] += 1
        
        # Keep only last 100 error contexts to prevent memory bloat
        if len(self.stats["error_contexts"]) > 100:
            self.stats["error_contexts"] = self.stats["error_contexts"][-100:]
        
        # Log with appropriate level based on error frequency
        error_count = self.stats["errors"][error_key]
        if error_count == 1:
            logger.error(f"Cache error in {operation}: {error} | Context: {json.dumps(context, indent=2)}")
        elif error_count <= 5:
            logger.warning(f"Cache error in {operation} (#{error_count}): {error}")
        else:
            logger.debug(f"Cache error in {operation} (#{error_count}): {error}")
    
    def _calculate_memory_usage(self) -> float:
        """
        Calculate detailed memory usage for different cache types.
        
        Returns:
            Total memory usage in MB
        """
        try:
            import sys
            
            total_size = 0
            shared_size = 0
            coder_specific_size = 0
            
            for key, entry in self.cache.items():
                entry_size = sys.getsizeof(entry.value) + sys.getsizeof(entry.key) + sys.getsizeof(entry)
                total_size += entry_size
                
                if entry.is_shared:
                    shared_size += entry_size
                else:
                    coder_specific_size += entry_size
            
            # Update memory usage by type
            self.stats["memory_usage_by_type"] = {
                "total_mb": total_size / (1024 * 1024),
                "shared_mb": shared_size / (1024 * 1024),
                "coder_specific_mb": coder_specific_size / (1024 * 1024),
                "index_overhead_mb": self._calculate_index_memory() / (1024 * 1024)
            }
            
            return total_size / (1024 * 1024)
            
        except Exception as e:
            self._log_error_with_context(e, "memory_calculation", {})
            # Fallback to simple estimation
            return len(self.cache) * 0.001
    
    def _calculate_index_memory(self) -> int:
        """
        Calculate memory usage of cache indexes.
        
        Returns:
            Index memory usage in bytes
        """
        import sys
        
        index_size = 0
        index_size += sys.getsizeof(self.mode_index)
        index_size += sys.getsizeof(self.segment_index)
        index_size += sys.getsizeof(self.category_index)
        index_size += sys.getsizeof(self.coder_index)
        index_size += sys.getsizeof(self.operation_index)
        index_size += sys.getsizeof(self.shared_index)
        
        return index_size
    
    def _track_performance_timing(self, operation: str, duration_ms: float, success: bool) -> None:
        """
        Track performance timing for cache operations.
        
        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether operation was successful
        """
        if operation not in self.stats["performance_timings"]:
            self.stats["performance_timings"][operation] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_duration_ms": 0.0,
                "avg_duration_ms": 0.0,
                "min_duration_ms": float('inf'),
                "max_duration_ms": 0.0
            }
        
        timing_stats = self.stats["performance_timings"][operation]
        timing_stats["total_calls"] += 1
        
        if success:
            timing_stats["successful_calls"] += 1
        else:
            timing_stats["failed_calls"] += 1
        
        timing_stats["total_duration_ms"] += duration_ms
        timing_stats["avg_duration_ms"] = timing_stats["total_duration_ms"] / timing_stats["total_calls"]
        timing_stats["min_duration_ms"] = min(timing_stats["min_duration_ms"], duration_ms)
        timing_stats["max_duration_ms"] = max(timing_stats["max_duration_ms"], duration_ms)
    
    def _track_cache_access_pattern(self, operation_type: str, coder_id: Optional[str], is_hit: bool) -> None:
        """
        Track cache access patterns for analysis.
        
        Args:
            operation_type: Type of operation
            coder_id: Coder ID if applicable
            is_hit: Whether this was a cache hit
        """
        pattern_key = f"{operation_type}:{coder_id or 'shared'}"
        
        if pattern_key not in self.stats["cache_access_patterns"]:
            self.stats["cache_access_patterns"][pattern_key] = {
                "total_accesses": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "last_access": None
            }
        
        pattern = self.stats["cache_access_patterns"][pattern_key]
        pattern["total_accesses"] += 1
        pattern["last_access"] = datetime.now().isoformat()
        
        if is_hit:
            pattern["hits"] += 1
        else:
            pattern["misses"] += 1
        
        pattern["hit_rate"] = pattern["hits"] / pattern["total_accesses"]
    
    def get_cache_key(self, operation_type: str, coder_id: Optional[str] = None, **params) -> str:
        """
        Generate cache key for operation using current strategy.
        
        Args:
            operation_type: Type of operation (e.g., 'relevance_check', 'coding')
            coder_id: Optional coder ID for coder-specific operations
            **params: Additional parameters for key generation
            
        Returns:
            Cache key string
        """
        return self.strategy.get_cache_key(operation_type, coder_id, **params)
    
    def should_cache_shared(self, operation_type: str) -> bool:
        """
        Determine if operation should use shared cache using current strategy.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            True if operation should be shared across coders
        """
        return self.strategy.should_cache_shared(operation_type)
    
    def should_cache_per_coder(self, operation_type: str) -> bool:
        """
        Determine if operation should use per-coder cache using current strategy.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            True if operation should be cached per coder
        """
        return self.strategy.should_cache_per_coder(operation_type)
    
    def get_statistics(self) -> CacheStatistics:
        """
        Get comprehensive cache statistics.
        
        Returns:
            CacheStatistics object with current metrics
        """
        total_hits_misses = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_hits_misses if total_hits_misses > 0 else 0
        
        # Calculate hit rates by coder
        hit_rate_by_coder = {}
        for coder_id in self.stats["hits_by_coder"]:
            coder_hits = self.stats["hits_by_coder"].get(coder_id, 0)
            coder_misses = self.stats["misses_by_coder"].get(coder_id, 0)
            coder_total = coder_hits + coder_misses
            hit_rate_by_coder[coder_id] = coder_hits / coder_total if coder_total > 0 else 0
        
        # Calculate cache efficiency by coder (includes performance metrics)
        cache_efficiency_by_coder = {}
        for coder_id in set(list(self.stats["hits_by_coder"].keys()) + list(self.stats["misses_by_coder"].keys())):
            hits = self.stats["hits_by_coder"].get(coder_id, 0)
            misses = self.stats["misses_by_coder"].get(coder_id, 0)
            total = hits + misses
            
            cache_efficiency_by_coder[coder_id] = {
                "hit_rate": hits / total if total > 0 else 0,
                "total_requests": total,
                "hits": hits,
                "misses": misses,
                "efficiency_score": (hits * 2 - misses) / max(total, 1)  # Weighted efficiency
            }
        
        # Count shared vs coder-specific entries
        shared_entries = len(self.shared_index)
        coder_specific_entries = len(self.cache) - shared_entries
        
        # Calculate memory usage
        memory_usage_mb = self._calculate_memory_usage()
        
        # Prepare performance metrics
        performance_metrics = {
            "total_errors": self.stats["total_errors"],
            "error_rate": self.stats["total_errors"] / max(total_hits_misses, 1),
            "last_error": self.stats["last_error_timestamp"],
            "performance_timings": dict(self.stats["performance_timings"]),
            "access_patterns": dict(self.stats["cache_access_patterns"]),
            "eviction_stats": {
                "total_evictions": self.stats["evictions"],
                "eviction_reasons": dict(self.stats["eviction_reasons"])
            }
        }
        
        return CacheStatistics(
            total_entries=len(self.cache),
            shared_entries=shared_entries,
            coder_specific_entries=coder_specific_entries,
            hit_rate_overall=hit_rate,
            hit_rate_by_coder=hit_rate_by_coder,
            memory_usage_mb=memory_usage_mb,
            strategy_type=self.strategy.name,
            hits_by_operation=self.stats["hits_by_operation"].copy(),
            misses_by_operation=self.stats["misses_by_operation"].copy(),
            memory_usage_by_type=self.stats["memory_usage_by_type"].copy(),
            cache_efficiency_by_coder=cache_efficiency_by_coder,
            error_counts=self.stats["errors"].copy(),
            performance_metrics=performance_metrics
        )
    
    def _generate_key(self, 
                     analysis_mode: str,
                     segment_text: str,
                     category_definitions: Optional[Dict[str, str]] = None,
                     research_question: Optional[str] = None,
                     coding_rules: Optional[List[str]] = None) -> str:
        """
        Generate cache key from analysis parameters.
        
        Args:
            analysis_mode: Analysis mode
            segment_text: Segment text
            category_definitions: Category definitions (for deductive/abductive)
            research_question: Research question
            coding_rules: Coding rules
            
        Returns:
            Cache key string
        """
        # Create hashable representation
        components = [
            analysis_mode,
            segment_text.strip().lower(),
            research_question.strip().lower() if research_question else "",
        ]
        
        if category_definitions:
            # Sort categories for consistent hashing
            sorted_cats = sorted(category_definitions.items())
            components.append(json.dumps(sorted_cats, sort_keys=True))
        
        if coding_rules:
            # Sort rules for consistent hashing
            sorted_rules = sorted([r.strip().lower() for r in coding_rules])
            components.append(json.dumps(sorted_rules, sort_keys=True))
        
        # Generate SHA256 hash
        key_string = "|".join(components)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity (0-1).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Simple similarity based on shared words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get(self, 
            analysis_mode: str,
            segment_text: str,
            category_definitions: Optional[Dict[str, str]] = None,
            research_question: Optional[str] = None,
            coding_rules: Optional[List[str]] = None,
            segment_id: Optional[str] = None,
            category: Optional[str] = None,
            min_confidence: float = 0.0,
            operation_type: str = "unknown",
            coder_id: Optional[str] = None) -> Optional[Any]:
        """
        Get cached result if available and valid.
        
        Args:
            analysis_mode: Analysis mode
            segment_text: Segment text
            category_definitions: Category definitions
            research_question: Research question
            coding_rules: Coding rules
            segment_id: Segment ID (for indexing)
            category: Category (for indexing)
            min_confidence: Minimum confidence score required
            operation_type: Type of operation for monitoring
            coder_id: Coder ID for coder-specific operations
            
        Returns:
            Cached value or None if not found/invalid
        """
        start_time = time.time()
        success = False
        result = None
        
        try:
            # Generate primary key
            primary_key = self.get_cache_key(
                operation_type=operation_type,
                coder_id=coder_id,
                analysis_mode=analysis_mode,
                segment_text=segment_text,
                category_definitions=category_definitions,
                research_question=research_question,
                coding_rules=coding_rules
            )
            
            # Try exact match first
            if primary_key in self.cache:
                entry = self.cache[primary_key]
                
                # Check if entry is still valid
                if self._is_valid_entry(entry, analysis_mode, min_confidence):
                    # Update LRU order and access count
                    self.cache.move_to_end(primary_key)
                    entry.access_count += 1
                    entry.timestamp = time.time()
                    
                    # Update statistics
                    self.stats["hits"] += 1
                    if coder_id:
                        self.stats["hits_by_coder"][coder_id] = self.stats["hits_by_coder"].get(coder_id, 0) + 1
                    self.stats["hits_by_operation"][operation_type] = self.stats["hits_by_operation"].get(operation_type, 0) + 1
                    
                    # Track access pattern
                    self._track_cache_access_pattern(operation_type, coder_id, True)
                    
                    # Notify listeners
                    self._notify_cache_hit(primary_key, entry)
                    
                    result = entry.value
                    success = True
                    return result
            
            # If no exact match, try similarity-based lookup for shared operations
            if self.should_cache_shared(operation_type):
                config = self.mode_configs.get(analysis_mode, {})
                similarity_threshold = config.get("similarity_threshold", 0.7)
                
                # Get entries for this mode
                mode_entries = self.mode_index.get(analysis_mode, [])
                
                for cache_key in mode_entries:
                    if cache_key not in self.cache:
                        continue
                        
                    entry = self.cache[cache_key]
                    
                    # Check if entry is valid and shared
                    if not self._is_valid_entry(entry, analysis_mode, min_confidence) or not entry.is_shared:
                        continue
                    
                    # Calculate similarity with cached segment
                    cached_text = entry.value.get("segment_text", "") if isinstance(entry.value, dict) else ""
                    similarity = self._text_similarity(segment_text, cached_text)
                    
                    if similarity >= similarity_threshold:
                        # Found similar entry
                        entry.access_count += 1
                        entry.timestamp = time.time()
                        self.cache.move_to_end(cache_key)
                        
                        # Update statistics
                        self.stats["hits"] += 1
                        if coder_id:
                            self.stats["hits_by_coder"][coder_id] = self.stats["hits_by_coder"].get(coder_id, 0) + 1
                        self.stats["hits_by_operation"][operation_type] = self.stats["hits_by_operation"].get(operation_type, 0) + 1
                        
                        # Track access pattern
                        self._track_cache_access_pattern(operation_type, coder_id, True)
                        
                        # Notify listeners
                        self._notify_cache_hit(cache_key, entry)
                        
                        result = entry.value
                        success = True
                        return result
            
            # Cache miss
            self.stats["misses"] += 1
            if coder_id:
                self.stats["misses_by_coder"][coder_id] = self.stats["misses_by_coder"].get(coder_id, 0) + 1
            self.stats["misses_by_operation"][operation_type] = self.stats["misses_by_operation"].get(operation_type, 0) + 1
            
            # Track access pattern
            self._track_cache_access_pattern(operation_type, coder_id, False)
            
            # Notify listeners
            self._notify_cache_miss(primary_key, operation_type)
            
            success = True  # No error occurred, just a miss
            return None
            
        except Exception as e:
            # Log error with context
            context = {
                "operation_type": operation_type,
                "coder_id": coder_id,
                "analysis_mode": analysis_mode,
                "segment_text_length": len(segment_text) if segment_text else 0,
                "has_category_definitions": category_definitions is not None,
                "min_confidence": min_confidence
            }
            self._log_error_with_context(e, "cache_get", context)
            return None
            
        finally:
            # Track performance timing
            duration_ms = (time.time() - start_time) * 1000
            self._track_performance_timing("cache_get", duration_ms, success)
    
    def set(self, 
            analysis_mode: str,
            segment_text: str,
            value: Any,
            category_definitions: Optional[Dict[str, str]] = None,
            research_question: Optional[str] = None,
            coding_rules: Optional[List[str]] = None,
            segment_id: Optional[str] = None,
            category: Optional[str] = None,
            confidence: float = 0.5,
            operation_type: str = "unknown",
            coder_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None):
        """
        Store result in cache.
        
        Args:
            analysis_mode: Analysis mode
            segment_text: Segment text
            value: Value to cache
            category_definitions: Category definitions
            research_question: Research question
            coding_rules: Coding rules
            segment_id: Segment ID (for indexing)
            category: Category (for indexing)
            confidence: Confidence score for this result
            operation_type: Type of operation for monitoring
            coder_id: Coder ID for coder-specific operations
            metadata: Additional metadata for the entry
        """
        start_time = time.time()
        success = False
        
        try:
            # Generate key
            key = self.get_cache_key(
                operation_type=operation_type,
                coder_id=coder_id,
                analysis_mode=analysis_mode,
                segment_text=segment_text,
                category_definitions=category_definitions,
                research_question=research_question,
                coding_rules=coding_rules
            )
            
            # Check cache size and evict if needed
            if len(self.cache) >= self.max_size:
                self._evict_entries()
            
            # Determine if this should be a shared entry
            is_shared = self.should_cache_shared(operation_type)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=1,
                analysis_mode=analysis_mode,
                segment_id=segment_id or "",
                category=category or "",
                confidence=confidence,
                coder_id=coder_id,
                operation_type=operation_type,
                is_shared=is_shared,
                metadata=metadata or {}
            )
            
            # Store in cache
            self.cache[key] = entry
            self.cache.move_to_end(key)
            
            # Update indexes
            if analysis_mode not in self.mode_index:
                self.mode_index[analysis_mode] = []
            self.mode_index[analysis_mode].append(key)
            
            if segment_id:
                if segment_id not in self.segment_index:
                    self.segment_index[segment_id] = []
                self.segment_index[segment_id].append(key)
            
            if category:
                if category not in self.category_index:
                    self.category_index[category] = []
                self.category_index[category].append(key)
            
            # New indexes for multi-coder support
            if coder_id:
                if coder_id not in self.coder_index:
                    self.coder_index[coder_id] = []
                self.coder_index[coder_id].append(key)
            
            if operation_type not in self.operation_index:
                self.operation_index[operation_type] = []
            self.operation_index[operation_type].append(key)
            
            if is_shared:
                self.shared_index.append(key)
            
            # Update statistics
            self.stats["total_entries"] = len(self.cache)
            
            # Notify listeners
            self._notify_cache_set(key, entry)
            
            success = True
            
        except Exception as e:
            # Log error with context
            context = {
                "operation_type": operation_type,
                "coder_id": coder_id,
                "analysis_mode": analysis_mode,
                "segment_text_length": len(segment_text) if segment_text else 0,
                "value_type": type(value).__name__,
                "confidence": confidence,
                "cache_size": len(self.cache),
                "max_size": self.max_size
            }
            self._log_error_with_context(e, "cache_set", context)
            
        finally:
            # Track performance timing
            duration_ms = (time.time() - start_time) * 1000
            self._track_performance_timing("cache_set", duration_ms, success)
    
    def _is_valid_entry(self, entry: CacheEntry, analysis_mode: str, min_confidence: float) -> bool:
        """
        Check if cache entry is still valid.
        
        Args:
            entry: Cache entry to check
            analysis_mode: Current analysis mode
            min_confidence: Minimum confidence required
            
        Returns:
            True if entry is valid
        """
        # Check TTL
        config = self.mode_configs.get(analysis_mode, {})
        ttl = config.get("ttl", self.default_ttl)
        
        if time.time() - entry.timestamp > ttl:
            return False
        
        # Check confidence
        if entry.confidence < min_confidence:
            return False
        
        # Check mode-specific confidence threshold
        confidence_threshold = config.get("confidence_threshold", 0.0)
        if entry.confidence < confidence_threshold:
            return False
        
        # Check if entry belongs to requested mode
        if entry.analysis_mode != analysis_mode:
            return False
        
        return True
    
    def export_statistics(self, filepath: str, include_detailed: bool = True) -> None:
        """
        Export cache statistics to JSON file for performance analysis.
        
        Args:
            filepath: Path to export file
            include_detailed: Whether to include detailed performance data
        """
        try:
            stats = self.get_statistics()
            
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'cache_statistics': {
                    'total_entries': stats.total_entries,
                    'shared_entries': stats.shared_entries,
                    'coder_specific_entries': stats.coder_specific_entries,
                    'hit_rate_overall': stats.hit_rate_overall,
                    'hit_rate_by_coder': stats.hit_rate_by_coder,
                    'memory_usage_mb': stats.memory_usage_mb,
                    'strategy_type': stats.strategy_type,
                    'hits_by_operation': stats.hits_by_operation,
                    'misses_by_operation': stats.misses_by_operation,
                    'memory_usage_by_type': stats.memory_usage_by_type,
                    'cache_efficiency_by_coder': stats.cache_efficiency_by_coder,
                    'error_counts': stats.error_counts,
                    'performance_metrics': stats.performance_metrics
                }
            }
            
            if include_detailed:
                export_data['detailed_statistics'] = {
                    'raw_stats': dict(self.stats),
                    'cache_configuration': {
                        'max_size': self.max_size,
                        'default_ttl': self.default_ttl,
                        'mode_configs': self.mode_configs,
                        'strategy_info': self.get_strategy_info()
                    },
                    'index_sizes': {
                        'mode_index': {mode: len(keys) for mode, keys in self.mode_index.items()},
                        'coder_index': {coder: len(keys) for coder, keys in self.coder_index.items()},
                        'operation_index': {op: len(keys) for op, keys in self.operation_index.items()},
                        'shared_index_size': len(self.shared_index)
                    },
                    'recent_errors': self.stats["error_contexts"][-10:] if self.stats["error_contexts"] else []
                }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Cache statistics exported to {filepath}")
            
        except Exception as e:
            context = {
                "filepath": filepath,
                "include_detailed": include_detailed,
                "cache_size": len(self.cache)
            }
            self._log_error_with_context(e, "statistics_export", context)
            raise
    
    def get_error_report(self) -> Dict[str, Any]:
        """
        Get detailed error report for debugging.
        
        Returns:
            Dictionary with error analysis
        """
        try:
            total_operations = self.stats["hits"] + self.stats["misses"]
            
            error_report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_errors': self.stats["total_errors"],
                    'error_rate': self.stats["total_errors"] / max(total_operations, 1),
                    'last_error': self.stats["last_error_timestamp"],
                    'most_common_errors': sorted(
                        self.stats["errors"].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                },
                'error_breakdown': dict(self.stats["errors"]),
                'recent_error_contexts': self.stats["error_contexts"][-20:] if self.stats["error_contexts"] else [],
                'error_patterns': self._analyze_error_patterns()
            }
            
            return error_report
            
        except Exception as e:
            logger.error(f"Failed to generate error report: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'summary': {'total_errors': self.stats.get("total_errors", 0)}
            }
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """
        Analyze error patterns for insights.
        
        Returns:
            Dictionary with error pattern analysis
        """
        patterns = {
            'errors_by_operation': {},
            'errors_by_hour': {},
            'error_frequency_trend': [],
            'common_error_contexts': {}
        }
        
        try:
            # Analyze errors by operation
            for error_context in self.stats["error_contexts"]:
                operation = error_context.get('operation', 'unknown')
                if operation not in patterns['errors_by_operation']:
                    patterns['errors_by_operation'][operation] = 0
                patterns['errors_by_operation'][operation] += 1
            
            # Analyze errors by hour (last 24 hours)
            from datetime import datetime, timedelta
            now = datetime.now()
            for error_context in self.stats["error_contexts"]:
                try:
                    error_time = datetime.fromisoformat(error_context['timestamp'])
                    if now - error_time <= timedelta(hours=24):
                        hour = error_time.hour
                        if hour not in patterns['errors_by_hour']:
                            patterns['errors_by_hour'][hour] = 0
                        patterns['errors_by_hour'][hour] += 1
                except:
                    continue
            
            # Analyze common error contexts
            context_keys = {}
            for error_context in self.stats["error_contexts"]:
                context = error_context.get('context', {})
                for key, value in context.items():
                    if key not in context_keys:
                        context_keys[key] = {}
                    value_str = str(value)
                    if value_str not in context_keys[key]:
                        context_keys[key][value_str] = 0
                    context_keys[key][value_str] += 1
            
            patterns['common_error_contexts'] = context_keys
            
        except Exception as e:
            logger.warning(f"Error pattern analysis failed: {e}")
        
        return patterns
    
    def reset_statistics(self, keep_errors: bool = False) -> None:
        """
        Reset cache statistics.
        
        Args:
            keep_errors: Whether to keep error history
        """
        try:
            # Reset basic stats
            self.stats.update({
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "invalidations": 0,
                "hits_by_coder": {},
                "misses_by_coder": {},
                "hits_by_operation": {},
                "misses_by_operation": {},
                "performance_timings": {},
                "cache_access_patterns": {},
                "eviction_reasons": {}
            })
            
            if not keep_errors:
                self.stats.update({
                    "errors": {},
                    "error_contexts": [],
                    "last_error_timestamp": None,
                    "total_errors": 0
                })
            
            # Update total entries count
            self.stats["total_entries"] = len(self.cache)
            
            logger.info(f"Cache statistics reset (keep_errors={keep_errors})")
            
        except Exception as e:
            context = {"keep_errors": keep_errors}
            self._log_error_with_context(e, "statistics_reset", context)
    
    def _evict_entries(self):
        """Evict least recently used entries to maintain cache size."""
        num_to_evict = max(1, len(self.cache) // 10)  # Evict 10% of cache
        
        evicted = 0
        evicted_keys = []
        eviction_reason = "lru_size_limit"
        
        try:
            while evicted < num_to_evict and self.cache:
                # Get LRU entry
                key, entry = self.cache.popitem(last=False)
                evicted_keys.append(key)
                
                # Remove from indexes using helper method
                self._remove_entry_from_indexes(key, entry)
                
                # Track eviction reason
                if eviction_reason not in self.stats["eviction_reasons"]:
                    self.stats["eviction_reasons"][eviction_reason] = 0
                self.stats["eviction_reasons"][eviction_reason] += 1
                
                # Notify listeners
                self._notify_cache_evict(key, entry)
                
                evicted += 1
                self.stats["evictions"] += 1
            
            # Notify about eviction batch
            if evicted_keys:
                self._notify_cache_invalidate(evicted_keys, eviction_reason)
                
        except Exception as e:
            # Log eviction error with context
            context = {
                "num_to_evict": num_to_evict,
                "evicted_so_far": evicted,
                "cache_size": len(self.cache),
                "max_size": self.max_size
            }
            self._log_error_with_context(e, "cache_eviction", context)
    
    def invalidate_by_mode(self, analysis_mode: str):
        """
        Invalidate all cache entries for a specific analysis mode.
        
        Args:
            analysis_mode: Mode to invalidate
        """
        if analysis_mode not in self.mode_index:
            return
        
        keys_to_remove = self.mode_index[analysis_mode].copy()
        for key in keys_to_remove:
            if key in self.cache:
                entry = self.cache[key]
                self._remove_entry_from_indexes(key, entry)
                del self.cache[key]
                self.stats["invalidations"] += 1
        
        # Clear mode index
        self.mode_index[analysis_mode] = []
        self.stats["total_entries"] = len(self.cache)
        
        # Notify listeners
        if keys_to_remove:
            self._notify_cache_invalidate(keys_to_remove, f"mode_invalidation:{analysis_mode}")
    
    def invalidate_by_coder(self, coder_id: str):
        """
        Invalidate all cache entries for a specific coder.
        
        Args:
            coder_id: Coder ID to invalidate
        """
        if coder_id not in self.coder_index:
            return
        
        keys_to_remove = self.coder_index[coder_id].copy()
        for key in keys_to_remove:
            if key in self.cache:
                entry = self.cache[key]
                self._remove_entry_from_indexes(key, entry)
                del self.cache[key]
                self.stats["invalidations"] += 1
        
        # Clear coder index
        self.coder_index[coder_id] = []
        self.stats["total_entries"] = len(self.cache)
        
        # Notify listeners
        if keys_to_remove:
            self._notify_cache_invalidate(keys_to_remove, f"coder_invalidation:{coder_id}")
    
    def invalidate_by_operation(self, operation_type: str):
        """
        Invalidate all cache entries for a specific operation type.
        
        Args:
            operation_type: Operation type to invalidate
        """
        if operation_type not in self.operation_index:
            return
        
        keys_to_remove = self.operation_index[operation_type].copy()
        for key in keys_to_remove:
            if key in self.cache:
                entry = self.cache[key]
                self._remove_entry_from_indexes(key, entry)
                del self.cache[key]
                self.stats["invalidations"] += 1
        
        # Clear operation index
        self.operation_index[operation_type] = []
        self.stats["total_entries"] = len(self.cache)
        
        # Notify listeners
        if keys_to_remove:
            self._notify_cache_invalidate(keys_to_remove, f"operation_invalidation:{operation_type}")
    
    def invalidate_by_segment(self, segment_id: str):
        """
        Invalidate all cache entries for a specific segment.
        
        Args:
            segment_id: Segment ID to invalidate
        """
        if segment_id not in self.segment_index:
            return
        
        keys_to_remove = self.segment_index[segment_id].copy()
        for key in keys_to_remove:
            if key in self.cache:
                entry = self.cache[key]
                self._remove_entry_from_indexes(key, entry)
                del self.cache[key]
                self.stats["invalidations"] += 1
        
        # Clear segment index
        self.segment_index[segment_id] = []
        self.stats["total_entries"] = len(self.cache)
        
        # Notify listeners
        if keys_to_remove:
            self._notify_cache_invalidate(keys_to_remove, f"segment_invalidation:{segment_id}")
    
    def invalidate_by_category(self, category: str):
        """
        Invalidate all cache entries for a specific category.
        
        Args:
            category: Category to invalidate
        """
        if category not in self.category_index:
            return
        
        keys_to_remove = self.category_index[category].copy()
        for key in keys_to_remove:
            if key in self.cache:
                entry = self.cache[key]
                self._remove_entry_from_indexes(key, entry)
                del self.cache[key]
                self.stats["invalidations"] += 1
        
        # Clear category index
        self.category_index[category] = []
        self.stats["total_entries"] = len(self.cache)
        
        # Notify listeners
        if keys_to_remove:
            self._notify_cache_invalidate(keys_to_remove, f"category_invalidation:{category}")
    
    def clear(self):
        """Clear entire cache."""
        keys_to_remove = list(self.cache.keys())
        
        self.cache.clear()
        self.mode_index.clear()
        self.segment_index.clear()
        self.category_index.clear()
        # Clear new indexes
        self.coder_index.clear()
        self.operation_index.clear()
        self.shared_index.clear()
        
        self.stats["total_entries"] = 0
        self.stats["evictions"] = 0
        self.stats["invalidations"] = 0
        
        # Notify listeners
        if keys_to_remove:
            self._notify_cache_invalidate(keys_to_remove, "clear_all")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics (legacy method for backward compatibility).
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.get_statistics()
        
        # Convert to legacy format for backward compatibility
        return {
            "total_entries": stats.total_entries,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": stats.hit_rate_overall,
            "evictions": self.stats["evictions"],
            "invalidations": self.stats["invalidations"],
            "avg_confidence": self._calculate_avg_confidence(),
            "mode_distribution": self._get_mode_distribution(),
            "max_size": self.max_size,
            "memory_usage_percentage": (len(self.cache) / self.max_size) * 100 if self.max_size > 0 else 0,
            # New fields
            "shared_entries": stats.shared_entries,
            "coder_specific_entries": stats.coder_specific_entries,
            "hit_rate_by_coder": stats.hit_rate_by_coder,
            "hits_by_operation": stats.hits_by_operation,
            "misses_by_operation": stats.misses_by_operation
        }
    
    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence across all entries."""
        if not self.cache:
            return 0.0
        
        total_confidence = sum(entry.confidence for entry in self.cache.values())
        return total_confidence / len(self.cache)
    
    def _get_mode_distribution(self) -> Dict[str, int]:
        """Get distribution of entries by analysis mode."""
        mode_distribution = {}
        for mode, keys in self.mode_index.items():
            mode_distribution[mode] = len(keys)
        return mode_distribution
    
    def save_to_file(self, filepath: str):
        """
        Save cache to file.
        
        Args:
            filepath: Path to save cache
        """
        cache_data = {
            "entries": [
                {
                    "key": entry.key,
                    "value": entry.value,
                    "timestamp": entry.timestamp,
                    "access_count": entry.access_count,
                    "analysis_mode": entry.analysis_mode,
                    "segment_id": entry.segment_id,
                    "category": entry.category,
                    "confidence": entry.confidence,
                    # New fields
                    "coder_id": entry.coder_id,
                    "operation_type": entry.operation_type,
                    "is_shared": entry.is_shared,
                    "metadata": entry.metadata
                }
                for entry in self.cache.values()
            ],
            "stats": self.stats,
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
            "mode_configs": self.mode_configs,
            "timestamp": time.time(),
            "version": "2.0"  # Version for compatibility
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, default=str)
    
    def load_from_file(self, filepath: str):
        """
        Load cache from file.
        
        Args:
            filepath: Path to load cache from
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Clear existing cache
            self.clear()
            
            # Restore configuration
            self.max_size = cache_data.get("max_size", self.max_size)
            self.default_ttl = cache_data.get("default_ttl", self.default_ttl)
            self.mode_configs = cache_data.get("mode_configs", self.mode_configs)
            
            # Check version for compatibility
            version = cache_data.get("version", "1.0")
            
            # Restore entries
            for entry_data in cache_data.get("entries", []):
                # Handle both old and new format
                if version == "1.0":
                    # Old format - add default values for new fields
                    entry = CacheEntry(
                        key=entry_data["key"],
                        value=entry_data["value"],
                        timestamp=entry_data["timestamp"],
                        access_count=entry_data.get("access_count", 0),
                        analysis_mode=entry_data.get("analysis_mode", ""),
                        segment_id=entry_data.get("segment_id", ""),
                        category=entry_data.get("category", ""),
                        confidence=entry_data.get("confidence", 0.5),
                        coder_id=None,
                        operation_type="unknown",
                        is_shared=True,  # Default to shared for old entries
                        metadata={}
                    )
                else:
                    # New format
                    entry = CacheEntry(
                        key=entry_data["key"],
                        value=entry_data["value"],
                        timestamp=entry_data["timestamp"],
                        access_count=entry_data.get("access_count", 0),
                        analysis_mode=entry_data.get("analysis_mode", ""),
                        segment_id=entry_data.get("segment_id", ""),
                        category=entry_data.get("category", ""),
                        confidence=entry_data.get("confidence", 0.5),
                        coder_id=entry_data.get("coder_id"),
                        operation_type=entry_data.get("operation_type", "unknown"),
                        is_shared=entry_data.get("is_shared", True),
                        metadata=entry_data.get("metadata", {})
                    )
                
                self.cache[entry.key] = entry
                
                # Rebuild indexes
                if entry.analysis_mode:
                    if entry.analysis_mode not in self.mode_index:
                        self.mode_index[entry.analysis_mode] = []
                    self.mode_index[entry.analysis_mode].append(entry.key)
                
                if entry.segment_id:
                    if entry.segment_id not in self.segment_index:
                        self.segment_index[entry.segment_id] = []
                    self.segment_index[entry.segment_id].append(entry.key)
                
                if entry.category:
                    if entry.category not in self.category_index:
                        self.category_index[entry.category] = []
                    self.category_index[entry.category].append(entry.key)
                
                # Rebuild new indexes
                if entry.coder_id:
                    if entry.coder_id not in self.coder_index:
                        self.coder_index[entry.coder_id] = []
                    self.coder_index[entry.coder_id].append(entry.key)
                
                if entry.operation_type not in self.operation_index:
                    self.operation_index[entry.operation_type] = []
                self.operation_index[entry.operation_type].append(entry.key)
                
                if entry.is_shared:
                    self.shared_index.append(entry.key)
            
            # Restore stats
            self.stats = cache_data.get("stats", self.stats)
            self.stats["total_entries"] = len(self.cache)
            
            # Initialize new stats fields if missing
            if "hits_by_coder" not in self.stats:
                self.stats["hits_by_coder"] = {}
            if "misses_by_coder" not in self.stats:
                self.stats["misses_by_coder"] = {}
            if "hits_by_operation" not in self.stats:
                self.stats["hits_by_operation"] = {}
            if "misses_by_operation" not in self.stats:
                self.stats["misses_by_operation"] = {}
            
        except Exception as e:
            logger.warning(f"Could not load cache from {filepath}: {e}")
            # Keep existing cache on load failure