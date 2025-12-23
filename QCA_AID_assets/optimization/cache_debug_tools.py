"""
Cache Debug Tools and Test Mode Support
======================================
Debugging and testing utilities for the Dynamic Cache System.
Provides test mode, benchmarking, cache dump/restore, and enhanced logging.
"""

import json
import time
import logging
import hashlib
import pickle
import gzip
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from enum import Enum

from .cache import CacheEntry, CacheStatistics, ModeAwareCache
from .dynamic_cache_manager import DynamicCacheManager

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Logging verbosity levels for cache debugging."""
    SILENT = 0      # No cache logging
    ERROR = 1       # Only errors
    WARNING = 2     # Errors and warnings
    INFO = 3        # Basic operations
    DEBUG = 4       # Detailed operations
    TRACE = 5       # All cache operations with full context


class TestMode(Enum):
    """Test modes for deterministic cache behavior."""
    DISABLED = "disabled"           # Normal cache behavior
    DETERMINISTIC = "deterministic" # Fixed seeds, predictable results
    REPLAY = "replay"              # Replay from recorded session
    RECORD = "record"              # Record session for later replay


@dataclass
class BenchmarkResult:
    """Result of cache performance benchmark."""
    test_name: str
    duration_seconds: float
    operations_count: int
    operations_per_second: float
    hit_rate: float
    memory_usage_mb: float
    cache_size: int
    error_count: int
    metadata: Dict[str, Any]


@dataclass
class CacheDumpEntry:
    """Serializable cache entry for dump/restore operations."""
    key: str
    value_type: str
    value_size_bytes: int
    timestamp: float
    access_count: int
    analysis_mode: str
    segment_id: str
    category: str
    confidence: float
    coder_id: Optional[str]
    operation_type: str
    is_shared: bool
    metadata: Dict[str, Any]
    
    @classmethod
    def from_cache_entry(cls, entry: CacheEntry) -> 'CacheDumpEntry':
        """Create dump entry from cache entry."""
        # Calculate value size
        try:
            value_size = len(pickle.dumps(entry.value))
        except Exception:
            value_size = 0
        
        return cls(
            key=entry.key,
            value_type=type(entry.value).__name__,
            value_size_bytes=value_size,
            timestamp=entry.timestamp,
            access_count=entry.access_count,
            analysis_mode=entry.analysis_mode,
            segment_id=entry.segment_id,
            category=entry.category,
            confidence=entry.confidence,
            coder_id=entry.coder_id,
            operation_type=entry.operation_type,
            is_shared=entry.is_shared,
            metadata=entry.metadata.copy() if entry.metadata else {}
        )


@dataclass
class CacheDump:
    """Complete cache dump for debugging and analysis."""
    timestamp: str
    manager_type: str
    analysis_mode: str
    strategy_name: str
    total_entries: int
    entries: List[CacheDumpEntry]
    statistics: Dict[str, Any]
    indexes: Dict[str, Any]
    configuration: Dict[str, Any]
    metadata: Dict[str, Any]


class CacheDebugLogger:
    """Enhanced logging system for cache debugging with configurable verbosity."""
    
    def __init__(self, level: LogLevel = LogLevel.INFO, log_file: Optional[str] = None):
        """
        Initialize debug logger.
        
        Args:
            level: Logging verbosity level
            log_file: Optional file to write logs to
        """
        self.level = level
        self.log_file = log_file
        self.operation_counts = {}
        self.error_contexts = []
        
        # Set up file handler if specified
        if log_file:
            self._setup_file_logging(log_file)
    
    def _setup_file_logging(self, log_file: str) -> None:
        """Set up file logging handler."""
        try:
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            cache_logger = logging.getLogger('QCA_AID_assets.optimization')
            cache_logger.addHandler(file_handler)
            cache_logger.setLevel(logging.DEBUG)
            
        except Exception as e:
            logger.warning(f"Failed to set up file logging: {e}")
    
    def log_operation(self, operation: str, details: Dict[str, Any], level: LogLevel = LogLevel.DEBUG) -> None:
        """
        Log cache operation with context.
        
        Args:
            operation: Operation type
            details: Operation details
            level: Log level for this operation
        """
        if level.value > self.level.value:
            return
        
        # Count operations
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
        
        # Format message based on level
        if self.level == LogLevel.TRACE:
            message = f"CACHE {operation}: {json.dumps(details, indent=2)}"
        elif self.level == LogLevel.DEBUG:
            key_details = {k: v for k, v in details.items() if k in ['key', 'coder_id', 'operation_type', 'hit_rate']}
            message = f"CACHE {operation}: {key_details}"
        else:
            message = f"CACHE {operation}: {details.get('key', 'unknown')[:16]}..."
        
        # Log at appropriate level
        if level == LogLevel.ERROR:
            logger.error(message)
        elif level == LogLevel.WARNING:
            logger.warning(message)
        elif level == LogLevel.INFO:
            logger.info(message)
        else:
            logger.debug(message)
    
    def log_error(self, error: str, context: Dict[str, Any]) -> None:
        """
        Log error with full context.
        
        Args:
            error: Error message
            context: Error context
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'context': context
        }
        
        self.error_contexts.append(error_entry)
        
        if self.level.value >= LogLevel.ERROR.value:
            if self.level == LogLevel.TRACE:
                logger.error(f"CACHE ERROR: {error}\nContext: {json.dumps(context, indent=2)}")
            else:
                logger.error(f"CACHE ERROR: {error}")
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get summary of logged operations."""
        return {
            'operation_counts': self.operation_counts.copy(),
            'total_operations': sum(self.operation_counts.values()),
            'error_count': len(self.error_contexts),
            'recent_errors': self.error_contexts[-5:] if self.error_contexts else []
        }


class TestModeManager:
    """Manager for test modes and deterministic cache behavior."""
    
    def __init__(self):
        """Initialize test mode manager."""
        self.mode = TestMode.DISABLED
        self.seed = None
        self.recorded_operations = []
        self.replay_index = 0
        self.recording_file = None
        
    def enable_deterministic_mode(self, seed: int = 42) -> None:
        """
        Enable deterministic test mode.
        
        Args:
            seed: Random seed for deterministic behavior
        """
        self.mode = TestMode.DETERMINISTIC
        self.seed = seed
        
        # Set random seeds for deterministic behavior
        import random
        random.seed(seed)
        
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        
        logger.info(f"Deterministic test mode enabled with seed {seed}")
    
    def start_recording(self, recording_file: str) -> None:
        """
        Start recording cache operations for later replay.
        
        Args:
            recording_file: File to save recorded operations
        """
        self.mode = TestMode.RECORD
        self.recording_file = recording_file
        self.recorded_operations = []
        
        logger.info(f"Started recording cache operations to {recording_file}")
    
    def stop_recording(self) -> None:
        """Stop recording and save operations to file."""
        if self.mode != TestMode.RECORD or not self.recording_file:
            return
        
        try:
            recording_data = {
                'timestamp': datetime.now().isoformat(),
                'seed': self.seed,
                'operations': self.recorded_operations
            }
            
            with open(self.recording_file, 'w', encoding='utf-8') as f:
                json.dump(recording_data, f, indent=2)
            
            logger.info(f"Saved {len(self.recorded_operations)} operations to {self.recording_file}")
            
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
        
        self.mode = TestMode.DISABLED
        self.recording_file = None
    
    def start_replay(self, recording_file: str) -> None:
        """
        Start replaying recorded operations.
        
        Args:
            recording_file: File with recorded operations
        """
        try:
            with open(recording_file, 'r', encoding='utf-8') as f:
                recording_data = json.load(f)
            
            self.mode = TestMode.REPLAY
            self.recorded_operations = recording_data['operations']
            self.replay_index = 0
            
            # Set seed if available
            if 'seed' in recording_data:
                self.enable_deterministic_mode(recording_data['seed'])
            
            logger.info(f"Started replaying {len(self.recorded_operations)} operations from {recording_file}")
            
        except Exception as e:
            logger.error(f"Failed to load recording: {e}")
            self.mode = TestMode.DISABLED
    
    def record_operation(self, operation_type: str, params: Dict[str, Any], result: Any) -> None:
        """
        Record a cache operation.
        
        Args:
            operation_type: Type of operation
            params: Operation parameters
            result: Operation result
        """
        if self.mode != TestMode.RECORD:
            return
        
        operation_record = {
            'timestamp': time.time(),
            'operation_type': operation_type,
            'params': params,
            'result_type': type(result).__name__,
            'result_hash': self._hash_result(result)
        }
        
        self.recorded_operations.append(operation_record)
    
    def get_replay_operation(self) -> Optional[Dict[str, Any]]:
        """
        Get next operation from replay.
        
        Returns:
            Next operation or None if replay finished
        """
        if self.mode != TestMode.REPLAY or self.replay_index >= len(self.recorded_operations):
            return None
        
        operation = self.recorded_operations[self.replay_index]
        self.replay_index += 1
        return operation
    
    def _hash_result(self, result: Any) -> str:
        """Create hash of result for verification."""
        try:
            result_str = json.dumps(result, sort_keys=True, default=str)
            return hashlib.sha256(result_str.encode()).hexdigest()[:16]
        except Exception:
            return "unhashable"
    
    def is_test_mode_active(self) -> bool:
        """Check if any test mode is active."""
        return self.mode != TestMode.DISABLED
    
    def get_test_mode_info(self) -> Dict[str, Any]:
        """Get information about current test mode."""
        return {
            'mode': self.mode.value,
            'seed': self.seed,
            'recorded_operations': len(self.recorded_operations),
            'replay_index': self.replay_index,
            'recording_file': self.recording_file
        }


class CacheBenchmark:
    """Benchmarking tools for cache performance measurement."""
    
    def __init__(self, cache_manager: DynamicCacheManager):
        """
        Initialize benchmark suite.
        
        Args:
            cache_manager: Cache manager to benchmark
        """
        self.cache_manager = cache_manager
        self.results = []
    
    def run_basic_performance_test(self, operations_count: int = 1000) -> BenchmarkResult:
        """
        Run basic cache performance test.
        
        Args:
            operations_count: Number of operations to perform
            
        Returns:
            Benchmark result
        """
        logger.info(f"Starting basic performance test with {operations_count} operations")
        
        start_time = time.time()
        start_stats = self.cache_manager.get_statistics()
        
        # Generate test operations
        test_operations = self._generate_test_operations(operations_count)
        
        # Execute operations
        hit_count = 0
        error_count = 0
        
        for operation in test_operations:
            try:
                # Simulate cache operation
                cache_key = self.cache_manager.get_shared_cache_key(
                    operation['type'], 
                    **operation['params']
                )
                
                # Check if in cache (simulated hit/miss)
                if cache_key in self.cache_manager.base_cache.cache:
                    hit_count += 1
                else:
                    # Simulate storing result
                    self._simulate_cache_store(cache_key, operation)
                    
            except Exception as e:
                error_count += 1
                logger.debug(f"Benchmark operation error: {e}")
        
        end_time = time.time()
        end_stats = self.cache_manager.get_statistics()
        
        # Calculate metrics
        duration = end_time - start_time
        ops_per_second = operations_count / duration if duration > 0 else 0
        hit_rate = hit_count / operations_count if operations_count > 0 else 0
        
        result = BenchmarkResult(
            test_name="basic_performance",
            duration_seconds=duration,
            operations_count=operations_count,
            operations_per_second=ops_per_second,
            hit_rate=hit_rate,
            memory_usage_mb=end_stats.memory_usage_mb,
            cache_size=end_stats.total_entries,
            error_count=error_count,
            metadata={
                'start_cache_size': start_stats.total_entries,
                'end_cache_size': end_stats.total_entries,
                'cache_growth': end_stats.total_entries - start_stats.total_entries
            }
        )
        
        self.results.append(result)
        logger.info(f"Basic performance test completed: {ops_per_second:.1f} ops/sec, {hit_rate:.2%} hit rate")
        
        return result
    
    def run_multi_coder_benchmark(self, coder_count: int = 3, operations_per_coder: int = 500) -> BenchmarkResult:
        """
        Run multi-coder performance benchmark.
        
        Args:
            coder_count: Number of coders to simulate
            operations_per_coder: Operations per coder
            
        Returns:
            Benchmark result
        """
        logger.info(f"Starting multi-coder benchmark: {coder_count} coders, {operations_per_coder} ops each")
        
        # Configure coders
        coder_settings = [
            {'coder_id': f'coder_{i}', 'temperature': 0.1 + (i * 0.1)}
            for i in range(coder_count)
        ]
        self.cache_manager.configure_coders(coder_settings)
        
        start_time = time.time()
        start_stats = self.cache_manager.get_statistics()
        
        total_operations = coder_count * operations_per_coder
        hit_count = 0
        error_count = 0
        
        # Run operations for each coder
        for coder_setting in coder_settings:
            coder_id = coder_setting['coder_id']
            
            for _ in range(operations_per_coder):
                try:
                    # Generate operation
                    operation = self._generate_single_operation()
                    
                    # Get coder-specific key
                    cache_key = self.cache_manager.get_coder_specific_key(
                        coder_id,
                        operation['type'],
                        **operation['params']
                    )
                    
                    # Check cache
                    if cache_key in self.cache_manager.base_cache.cache:
                        hit_count += 1
                    else:
                        self._simulate_cache_store(cache_key, operation, coder_id)
                        
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Multi-coder benchmark error: {e}")
        
        end_time = time.time()
        end_stats = self.cache_manager.get_statistics()
        
        # Calculate metrics
        duration = end_time - start_time
        ops_per_second = total_operations / duration if duration > 0 else 0
        hit_rate = hit_count / total_operations if total_operations > 0 else 0
        
        result = BenchmarkResult(
            test_name="multi_coder_performance",
            duration_seconds=duration,
            operations_count=total_operations,
            operations_per_second=ops_per_second,
            hit_rate=hit_rate,
            memory_usage_mb=end_stats.memory_usage_mb,
            cache_size=end_stats.total_entries,
            error_count=error_count,
            metadata={
                'coder_count': coder_count,
                'operations_per_coder': operations_per_coder,
                'shared_entries': end_stats.shared_entries,
                'coder_specific_entries': end_stats.coder_specific_entries,
                'hit_rate_by_coder': end_stats.hit_rate_by_coder
            }
        )
        
        self.results.append(result)
        logger.info(f"Multi-coder benchmark completed: {ops_per_second:.1f} ops/sec, {hit_rate:.2%} hit rate")
        
        return result
    
    def run_memory_stress_test(self, target_memory_mb: float = 100) -> BenchmarkResult:
        """
        Run memory stress test to measure cache behavior under memory pressure.
        
        Args:
            target_memory_mb: Target memory usage in MB
            
        Returns:
            Benchmark result
        """
        logger.info(f"Starting memory stress test, target: {target_memory_mb} MB")
        
        start_time = time.time()
        operations_count = 0
        error_count = 0
        
        # Keep adding entries until target memory is reached
        while True:
            try:
                current_stats = self.cache_manager.get_statistics()
                
                if current_stats.memory_usage_mb >= target_memory_mb:
                    break
                
                # Generate large operation
                operation = self._generate_large_operation()
                cache_key = self.cache_manager.get_shared_cache_key(
                    operation['type'],
                    **operation['params']
                )
                
                self._simulate_cache_store(cache_key, operation)
                operations_count += 1
                
                # Safety check to prevent infinite loop
                if operations_count > 10000:
                    logger.warning("Memory stress test reached operation limit")
                    break
                    
            except Exception as e:
                error_count += 1
                logger.debug(f"Memory stress test error: {e}")
                break
        
        end_time = time.time()
        final_stats = self.cache_manager.get_statistics()
        
        result = BenchmarkResult(
            test_name="memory_stress",
            duration_seconds=end_time - start_time,
            operations_count=operations_count,
            operations_per_second=operations_count / (end_time - start_time) if end_time > start_time else 0,
            hit_rate=0.0,  # Not applicable for stress test
            memory_usage_mb=final_stats.memory_usage_mb,
            cache_size=final_stats.total_entries,
            error_count=error_count,
            metadata={
                'target_memory_mb': target_memory_mb,
                'achieved_memory_mb': final_stats.memory_usage_mb,
                'memory_efficiency': operations_count / final_stats.memory_usage_mb if final_stats.memory_usage_mb > 0 else 0
            }
        )
        
        self.results.append(result)
        logger.info(f"Memory stress test completed: {final_stats.memory_usage_mb:.1f} MB, {operations_count} operations")
        
        return result
    
    def _generate_test_operations(self, count: int) -> List[Dict[str, Any]]:
        """Generate test operations for benchmarking."""
        operations = []
        operation_types = ['relevance_check', 'category_development', 'coding', 'confidence_scoring']
        
        for i in range(count):
            op_type = operation_types[i % len(operation_types)]
            
            operation = {
                'type': op_type,
                'params': {
                    'segment_text': f'Test segment {i}',
                    'analysis_mode': 'deductive',
                    'research_question': 'Test research question'
                }
            }
            
            operations.append(operation)
        
        return operations
    
    def _generate_single_operation(self) -> Dict[str, Any]:
        """Generate a single test operation."""
        import random
        
        operation_types = ['relevance_check', 'category_development', 'coding', 'confidence_scoring']
        op_type = random.choice(operation_types)
        
        return {
            'type': op_type,
            'params': {
                'segment_text': f'Random segment {random.randint(1, 1000)}',
                'analysis_mode': random.choice(['deductive', 'inductive', 'abductive', 'grounded']),
                'research_question': f'Research question {random.randint(1, 10)}'
            }
        }
    
    def _generate_large_operation(self) -> Dict[str, Any]:
        """Generate operation with large data for memory testing."""
        large_text = "Large test segment " * 1000  # ~17KB of text
        
        return {
            'type': 'coding',
            'params': {
                'segment_text': large_text,
                'analysis_mode': 'deductive',
                'research_question': 'Memory stress test',
                'large_data': list(range(1000))  # Additional large data
            }
        }
    
    def _simulate_cache_store(self, cache_key: str, operation: Dict[str, Any], coder_id: Optional[str] = None) -> None:
        """Simulate storing operation result in cache."""
        # Create mock result
        mock_result = {
            'category': 'test_category',
            'confidence': 0.85,
            'justification': 'Test justification',
            'timestamp': time.time()
        }
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            value=mock_result,
            timestamp=time.time(),
            analysis_mode=operation['params'].get('analysis_mode', ''),
            segment_id=f"segment_{hash(operation['params'].get('segment_text', '')) % 1000}",
            category='test_category',
            confidence=0.85,
            coder_id=coder_id,
            operation_type=operation['type'],
            is_shared=coder_id is None,
            metadata=operation['params']
        )
        
        # Store in cache
        self.cache_manager.base_cache.cache[cache_key] = entry
    
    def export_benchmark_results(self, filepath: str) -> None:
        """
        Export benchmark results to file.
        
        Args:
            filepath: Path to export results
        """
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'benchmark_count': len(self.results),
                'results': [asdict(result) for result in self.results],
                'summary': self._generate_benchmark_summary()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Benchmark results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export benchmark results: {e}")
            raise
    
    def _generate_benchmark_summary(self) -> Dict[str, Any]:
        """Generate summary of all benchmark results."""
        if not self.results:
            return {}
        
        return {
            'total_tests': len(self.results),
            'average_ops_per_second': sum(r.operations_per_second for r in self.results) / len(self.results),
            'average_hit_rate': sum(r.hit_rate for r in self.results) / len(self.results),
            'total_operations': sum(r.operations_count for r in self.results),
            'total_errors': sum(r.error_count for r in self.results),
            'max_memory_usage': max(r.memory_usage_mb for r in self.results),
            'test_types': list(set(r.test_name for r in self.results))
        }


class CacheDumpManager:
    """Manager for cache dump and restore operations."""
    
    def __init__(self, cache_manager: DynamicCacheManager):
        """
        Initialize dump manager.
        
        Args:
            cache_manager: Cache manager to dump/restore
        """
        self.cache_manager = cache_manager
    
    def create_dump(self, include_values: bool = False, compress: bool = True) -> CacheDump:
        """
        Create complete cache dump.
        
        Args:
            include_values: Whether to include actual cached values
            compress: Whether to compress large values
            
        Returns:
            Cache dump object
        """
        logger.info(f"Creating cache dump (include_values={include_values}, compress={compress})")
        
        # Get current statistics and info
        stats = self.cache_manager.get_statistics()
        manager_info = self.cache_manager.get_manager_info()
        
        # Create dump entries
        dump_entries = []
        for key, entry in self.cache_manager.base_cache.cache.items():
            dump_entry = CacheDumpEntry.from_cache_entry(entry)
            
            # Optionally include values
            if include_values:
                try:
                    if compress and dump_entry.value_size_bytes > 1024:  # Compress values > 1KB
                        compressed_value = gzip.compress(pickle.dumps(entry.value))
                        dump_entry.metadata['compressed_value'] = compressed_value.hex()
                        dump_entry.metadata['is_compressed'] = True
                    else:
                        dump_entry.metadata['value'] = entry.value
                        dump_entry.metadata['is_compressed'] = False
                except Exception as e:
                    logger.warning(f"Failed to serialize value for key {key[:16]}...: {e}")
                    dump_entry.metadata['serialization_error'] = str(e)
            
            dump_entries.append(dump_entry)
        
        # Create complete dump
        cache_dump = CacheDump(
            timestamp=datetime.now().isoformat(),
            manager_type=manager_info['manager_type'],
            analysis_mode=manager_info['analysis_mode'],
            strategy_name=stats.strategy_type,
            total_entries=len(dump_entries),
            entries=dump_entries,
            statistics=asdict(stats),
            indexes={
                'mode_index_sizes': {mode: len(keys) for mode, keys in self.cache_manager.base_cache.mode_index.items()},
                'coder_index_sizes': {coder: len(keys) for coder, keys in self.cache_manager.base_cache.coder_index.items()},
                'operation_index_sizes': {op: len(keys) for op, keys in self.cache_manager.base_cache.operation_index.items()},
                'shared_index_size': len(self.cache_manager.base_cache.shared_index)
            },
            configuration=manager_info,
            metadata={
                'include_values': include_values,
                'compress': compress,
                'dump_size_entries': len(dump_entries),
                'total_value_size_bytes': sum(entry.value_size_bytes for entry in dump_entries)
            }
        )
        
        logger.info(f"Cache dump created: {len(dump_entries)} entries, {cache_dump.metadata['total_value_size_bytes']} bytes")
        
        return cache_dump
    
    def save_dump(self, dump: CacheDump, filepath: str, format: str = 'json') -> None:
        """
        Save cache dump to file.
        
        Args:
            dump: Cache dump to save
            filepath: Path to save file
            format: Format ('json' or 'pickle')
        """
        try:
            if format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(asdict(dump), f, indent=2, default=str)
            elif format == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(dump, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Cache dump saved to {filepath} ({format} format)")
            
        except Exception as e:
            logger.error(f"Failed to save cache dump: {e}")
            raise
    
    def load_dump(self, filepath: str, format: str = 'json') -> CacheDump:
        """
        Load cache dump from file.
        
        Args:
            filepath: Path to dump file
            format: Format ('json' or 'pickle')
            
        Returns:
            Loaded cache dump
        """
        try:
            if format == 'json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    dump_data = json.load(f)
                
                # Convert back to CacheDump object
                dump = CacheDump(**dump_data)
                
                # Convert entries back to CacheDumpEntry objects
                dump.entries = [CacheDumpEntry(**entry_data) for entry_data in dump_data['entries']]
                
            elif format == 'pickle':
                with open(filepath, 'rb') as f:
                    dump = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Cache dump loaded from {filepath}: {len(dump.entries)} entries")
            return dump
            
        except Exception as e:
            logger.error(f"Failed to load cache dump: {e}")
            raise
    
    def restore_from_dump(self, dump: CacheDump, clear_existing: bool = True, 
                         restore_values: bool = False) -> Dict[str, Any]:
        """
        Restore cache from dump.
        
        Args:
            dump: Cache dump to restore from
            clear_existing: Whether to clear existing cache first
            restore_values: Whether to restore cached values (if available)
            
        Returns:
            Restoration result summary
        """
        logger.info(f"Restoring cache from dump: {len(dump.entries)} entries")
        
        if clear_existing:
            self.cache_manager.base_cache.clear()
        
        restored_count = 0
        skipped_count = 0
        error_count = 0
        
        for dump_entry in dump.entries:
            try:
                # Create cache entry
                cache_entry = CacheEntry(
                    key=dump_entry.key,
                    value=None,  # Will be set below if available
                    timestamp=dump_entry.timestamp,
                    access_count=dump_entry.access_count,
                    analysis_mode=dump_entry.analysis_mode,
                    segment_id=dump_entry.segment_id,
                    category=dump_entry.category,
                    confidence=dump_entry.confidence,
                    coder_id=dump_entry.coder_id,
                    operation_type=dump_entry.operation_type,
                    is_shared=dump_entry.is_shared,
                    metadata=dump_entry.metadata.copy()
                )
                
                # Restore value if available and requested
                if restore_values and 'value' in dump_entry.metadata:
                    if dump_entry.metadata.get('is_compressed', False):
                        # Decompress value
                        compressed_data = bytes.fromhex(dump_entry.metadata['compressed_value'])
                        cache_entry.value = pickle.loads(gzip.decompress(compressed_data))
                    else:
                        cache_entry.value = dump_entry.metadata['value']
                else:
                    # Use placeholder value
                    cache_entry.value = {'restored_from_dump': True, 'original_type': dump_entry.value_type}
                
                # Store in cache
                self.cache_manager.base_cache.cache[dump_entry.key] = cache_entry
                
                # Update indexes (simplified - full index rebuild would be better)
                if dump_entry.operation_type:
                    if dump_entry.operation_type not in self.cache_manager.base_cache.operation_index:
                        self.cache_manager.base_cache.operation_index[dump_entry.operation_type] = []
                    self.cache_manager.base_cache.operation_index[dump_entry.operation_type].append(dump_entry.key)
                
                if dump_entry.is_shared:
                    self.cache_manager.base_cache.shared_index.append(dump_entry.key)
                
                restored_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to restore entry {dump_entry.key[:16]}...: {e}")
                error_count += 1
        
        # Update cache statistics
        self.cache_manager.base_cache.stats["total_entries"] = len(self.cache_manager.base_cache.cache)
        
        result = {
            'restored_entries': restored_count,
            'skipped_entries': skipped_count,
            'error_count': error_count,
            'total_entries_after_restore': len(self.cache_manager.base_cache.cache),
            'restore_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Cache restoration completed: {restored_count} restored, {error_count} errors")
        
        return result
    
    def compare_dumps(self, dump1: CacheDump, dump2: CacheDump) -> Dict[str, Any]:
        """
        Compare two cache dumps.
        
        Args:
            dump1: First dump
            dump2: Second dump
            
        Returns:
            Comparison result
        """
        # Create key sets
        keys1 = set(entry.key for entry in dump1.entries)
        keys2 = set(entry.key for entry in dump2.entries)
        
        # Find differences
        only_in_dump1 = keys1 - keys2
        only_in_dump2 = keys2 - keys1
        common_keys = keys1 & keys2
        
        # Compare common entries
        entry_differences = []
        for key in common_keys:
            entry1 = next(e for e in dump1.entries if e.key == key)
            entry2 = next(e for e in dump2.entries if e.key == key)
            
            differences = {}
            for field in ['timestamp', 'access_count', 'confidence', 'coder_id', 'operation_type', 'is_shared']:
                val1 = getattr(entry1, field)
                val2 = getattr(entry2, field)
                if val1 != val2:
                    differences[field] = {'dump1': val1, 'dump2': val2}
            
            if differences:
                entry_differences.append({'key': key[:16] + '...', 'differences': differences})
        
        comparison = {
            'dump1_info': {
                'timestamp': dump1.timestamp,
                'total_entries': dump1.total_entries,
                'strategy': dump1.strategy_name
            },
            'dump2_info': {
                'timestamp': dump2.timestamp,
                'total_entries': dump2.total_entries,
                'strategy': dump2.strategy_name
            },
            'key_differences': {
                'only_in_dump1': len(only_in_dump1),
                'only_in_dump2': len(only_in_dump2),
                'common_keys': len(common_keys)
            },
            'entry_differences': entry_differences[:10],  # Limit to first 10
            'total_entry_differences': len(entry_differences)
        }
        
        logger.info(f"Dump comparison completed: {len(entry_differences)} entry differences found")
        
        return comparison


@contextmanager
def cache_debug_session(cache_manager: DynamicCacheManager, 
                       log_level: LogLevel = LogLevel.DEBUG,
                       log_file: Optional[str] = None,
                       test_mode: Optional[TestMode] = None,
                       test_seed: int = 42):
    """
    Context manager for cache debugging session.
    
    Args:
        cache_manager: Cache manager to debug
        log_level: Logging verbosity level
        log_file: Optional log file
        test_mode: Optional test mode to enable
        test_seed: Seed for deterministic test mode
    """
    # Set up debug logger
    debug_logger = CacheDebugLogger(log_level, log_file)
    
    # Set up test mode manager
    test_manager = TestModeManager()
    if test_mode == TestMode.DETERMINISTIC:
        test_manager.enable_deterministic_mode(test_seed)
    
    # Store original logging level
    original_level = logging.getLogger('QCA_AID_assets.optimization').level
    
    try:
        # Set debug logging level
        if log_level != LogLevel.SILENT:
            logging.getLogger('QCA_AID_assets.optimization').setLevel(logging.DEBUG)
        
        logger.info(f"Cache debug session started (level: {log_level.name}, test_mode: {test_mode})")
        
        yield {
            'debug_logger': debug_logger,
            'test_manager': test_manager,
            'benchmark': CacheBenchmark(cache_manager),
            'dump_manager': CacheDumpManager(cache_manager)
        }
        
    finally:
        # Restore original logging level
        logging.getLogger('QCA_AID_assets.optimization').setLevel(original_level)
        
        # Stop any active recording
        if test_manager.mode == TestMode.RECORD:
            test_manager.stop_recording()
        
        # Log session summary
        summary = debug_logger.get_operation_summary()
        logger.info(f"Cache debug session ended: {summary['total_operations']} operations, {summary['error_count']} errors")