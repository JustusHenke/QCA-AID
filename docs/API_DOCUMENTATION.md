# Dynamic Cache Manager API Documentation

## Overview

The `DynamicCacheManager` is the central component of the Dynamic Cache System that provides intelligent cache strategy selection and management for QCA-AID's multi-coder analysis workflows. It automatically switches between single-coder and multi-coder strategies, supports intercoder reliability analysis, and provides comprehensive monitoring and debugging capabilities.

## Table of Contents

1. [Core API](#core-api)
2. [Configuration Management](#configuration-management)
3. [Cache Operations](#cache-operations)
4. [Reliability Data Management](#reliability-data-management)
5. [Monitoring and Statistics](#monitoring-and-statistics)
6. [Plugin System](#plugin-system)
7. [Test Mode and Debugging](#test-mode-and-debugging)
8. [Migration and Consistency](#migration-and-consistency)
9. [Performance Benchmarking](#performance-benchmarking)
10. [Error Handling](#error-handling)

## Core API

### Class: DynamicCacheManager

```python
class DynamicCacheManager:
    """
    Central cache manager that dynamically selects and manages cache strategies.
    
    Features:
    - Automatic strategy selection based on coder count
    - Hot-reload configuration changes
    - Shared and coder-specific cache key generation
    - Intercoder reliability data management
    - Configuration change detection and cache migration
    - Test modes for deterministic behavior
    - Enhanced debugging and logging capabilities
    - Performance benchmarking tools
    - Cache dump and restore functionality
    """
```

#### Constructor

```python
def __init__(self, 
             base_cache: ModeAwareCache, 
             config_file: Optional[str] = None,
             reliability_db_path: Optional[str] = None, 
             analysis_mode: Optional[str] = None,
             enable_plugins: bool = True, 
             plugins_directory: Optional[str] = None,
             test_mode: bool = False, 
             debug_level: str = "INFO") -> None:
    """
    Initialize dynamic cache manager.
    
    Args:
        base_cache: Base ModeAwareCache instance to manage
        config_file: Optional path to configuration file for hot-reload
        reliability_db_path: Optional path to reliability database file 
                           (default: output/reliability_data.db, relative to configured project root)
        analysis_mode: Optional analysis mode for mode-specific caching strategies
        enable_plugins: Whether to enable plugin system
        plugins_directory: Optional directory to load plugins from
        test_mode: Whether to enable test mode for deterministic behavior
        debug_level: Debug logging level (SILENT, ERROR, WARNING, INFO, DEBUG, TRACE)
    """
```

**Example Usage:**

```python
from QCA_AID_assets.optimization.cache import ModeAwareCache
from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager

# Basic initialization
cache = ModeAwareCache()
manager = DynamicCacheManager(cache)

# Advanced initialization with all options
manager = DynamicCacheManager(
    base_cache=cache,
    config_file="cache_config.json",
    reliability_db_path="output/reliability_data.db",
    analysis_mode="inductive",
    enable_plugins=True,
    plugins_directory="custom_plugins/",
    test_mode=False,
    debug_level="INFO"
)
```

## Configuration Management

### configure_coders()

```python
def configure_coders(self, coder_settings: List[Dict[str, Any]]) -> None:
    """
    Configure coders and update cache strategy if needed.
    
    Args:
        coder_settings: List of coder configurations with 'coder_id' and 'temperature'
    """
```

**Example:**

```python
# Configure multiple coders
manager.configure_coders([
    {'coder_id': 'coder_1', 'temperature': 0.1},
    {'coder_id': 'coder_2', 'temperature': 0.3},
    {'coder_id': 'coder_3', 'temperature': 0.5}
])

# Single coder configuration
manager.configure_coders([
    {'coder_id': 'main_coder', 'temperature': 0.2}
])

# Clear all coders
manager.configure_coders([])
```

### configure_analysis_mode()

```python
def configure_analysis_mode(self, analysis_mode: str) -> None:
    """
    Configure analysis mode and update cache strategy if needed.
    
    Args:
        analysis_mode: Analysis mode (inductive, abductive, grounded, deductive)
    """
```

**Example:**

```python
# Set analysis mode
manager.configure_analysis_mode("inductive")

# Mode-specific cache rules will be applied automatically
mode_rules = manager.get_mode_specific_cache_rules()
print(f"Shared operations: {mode_rules['shared_operations']}")
print(f"Coder-specific operations: {mode_rules['coder_specific_operations']}")
```

### reload_configuration()

```python
def reload_configuration(self) -> bool:
    """
    Reload configuration from file if available.
    
    Returns:
        True if configuration was reloaded, False if no config file or no changes
    """
```

**Example:**

```python
# Hot-reload configuration from file
if manager.reload_configuration():
    print("Configuration reloaded successfully")
    print(f"New strategy: {manager.get_cache_strategy().name}")
else:
    print("No configuration changes detected")
```

## Cache Operations

### Cache Key Generation

```python
def get_shared_cache_key(self, operation: str, **params) -> str:
    """
    Generate cache key for shared operations (used across all coders).
    
    Args:
        operation: Operation type (e.g., 'relevance_check', 'category_development')
        **params: Additional parameters for key generation
        
    Returns:
        Cache key string for shared operation
    """

def get_coder_specific_key(self, coder_id: str, operation: str, **params) -> str:
    """
    Generate cache key for coder-specific operations.
    
    Args:
        coder_id: Coder identifier
        operation: Operation type (e.g., 'coding', 'confidence_scoring')
        **params: Additional parameters for key generation
        
    Returns:
        Cache key string for coder-specific operation
    """
```

**Example:**

```python
# Generate shared cache key
shared_key = manager.get_shared_cache_key(
    'relevance_check',
    segment_text="This is a test segment",
    research_question="What are the main themes?"
)

# Generate coder-specific cache key
coder_key = manager.get_coder_specific_key(
    'coder_1',
    'coding',
    segment_text="This is a test segment",
    category_definitions={"theme1": "Definition 1"}
)
```

### Cache Strategy Queries

```python
def should_cache_shared(self, operation: str) -> bool:
    """
    Check if operation should use shared caching.
    
    Args:
        operation: Operation type
        
    Returns:
        True if operation should be cached as shared
    """

def should_cache_per_coder(self, operation: str) -> bool:
    """
    Check if operation should use per-coder caching.
    
    Args:
        operation: Operation type
        
    Returns:
        True if operation should be cached per coder
    """

def get_cache_strategy(self) -> CacheStrategy:
    """
    Get current cache strategy.
    
    Returns:
        Current cache strategy instance
    """
```

**Example:**

```python
# Check caching behavior for operations
if manager.should_cache_shared('relevance_check'):
    print("Relevance check will be shared across coders")

if manager.should_cache_per_coder('coding'):
    print("Coding will be cached separately per coder")

# Get current strategy info
strategy = manager.get_cache_strategy()
print(f"Current strategy: {strategy.name}")
strategy_info = strategy.get_strategy_info()
print(f"Strategy type: {strategy_info['type']}")
```

## Reliability Data Management

### store_for_reliability()

```python
def store_for_reliability(self, coding_result: ExtendedCodingResult) -> None:
    """
    Store coding result for intercoder reliability analysis.
    
    Args:
        coding_result: Extended coding result with metadata
    """
```

### get_reliability_data()

```python
def get_reliability_data(self, segment_ids: Optional[List[str]] = None) -> List[ExtendedCodingResult]:
    """
    Get reliability data for intercoder analysis.
    
    Args:
        segment_ids: Optional list of segment IDs to filter by
        
    Returns:
        List of coding results for reliability analysis
    """
```

### Manual Coder Integration

```python
def store_manual_coding(self, 
                       segment_id: str, 
                       category: str, 
                       subcategories: List[str], 
                       justification: str, 
                       confidence: float, 
                       analysis_mode: str,
                       manual_coder_id: Optional[str] = None) -> None:
    """
    Store manual coding result with automatic "manual" ID assignment.
    
    Args:
        segment_id: Segment identifier
        category: Main category
        subcategories: List of subcategories
        justification: Coding justification
        confidence: Confidence score
        analysis_mode: Analysis mode used
        manual_coder_id: Optional specific manual coder ID (defaults to "manual")
    """

def get_manual_codings(self, segment_ids: Optional[List[str]] = None) -> List[ExtendedCodingResult]:
    """
    Get only manual coding results.
    
    Args:
        segment_ids: Optional list of segment IDs to filter by
        
    Returns:
        List of manual coding results
    """

def get_automatic_codings(self, segment_ids: Optional[List[str]] = None) -> List[ExtendedCodingResult]:
    """
    Get only automatic coding results.
    
    Args:
        segment_ids: Optional list of segment IDs to filter by
        
    Returns:
        List of automatic coding results
    """
```

**Example:**

```python
from QCA_AID_assets.core.data_models import ExtendedCodingResult
from datetime import datetime

# Store automatic coding result
auto_result = ExtendedCodingResult(
    segment_id="seg_001",
    coder_id="coder_1",
    category="theme_a",
    subcategories=["subtheme_1"],
    confidence=0.85,
    justification="Strong indicators present",
    analysis_mode="inductive",
    timestamp=datetime.now(),
    is_manual=False
)
manager.store_for_reliability(auto_result)

# Store manual coding result
manager.store_manual_coding(
    segment_id="seg_001",
    category="theme_a",
    subcategories=["subtheme_1"],
    justification="Manual review confirms theme",
    confidence=0.90,
    analysis_mode="inductive"
)

# Get reliability data for analysis
reliability_data = manager.get_reliability_data(["seg_001"])
manual_codings = manager.get_manual_codings(["seg_001"])
auto_codings = manager.get_automatic_codings(["seg_001"])

print(f"Total codings: {len(reliability_data)}")
print(f"Manual codings: {len(manual_codings)}")
print(f"Automatic codings: {len(auto_codings)}")
```

### Reliability Analysis

```python
def get_reliability_summary(self) -> Dict[str, Any]:
    """
    Get summary of reliability data for monitoring.
    
    Returns:
        Dictionary with reliability data summary including manual coder statistics
    """

def test_manual_auto_coder_combination(self, test_segments: List[str]) -> Dict[str, Any]:
    """
    Test manual + auto-coder combinations for reliability analysis.
    
    Args:
        test_segments: List of segment IDs to test with
        
    Returns:
        Dictionary with test results and reliability metrics
    """

def get_segments_for_reliability_analysis(self) -> List[str]:
    """
    Get segments that have multiple coders for reliability analysis.
    
    Returns:
        List of segment IDs suitable for reliability analysis
    """
```

**Example:**

```python
# Get reliability summary
summary = manager.get_reliability_summary()
print(f"Total segments: {summary['total_segments']}")
print(f"Total codings: {summary['total_codings']}")
print(f"Manual codings: {summary['manual_codings']}")
print(f"Automatic codings: {summary['automatic_codings']}")
print(f"Codings per coder: {summary['codings_per_coder']}")

# Test manual + auto coder combination
test_segments = manager.get_segments_for_reliability_analysis()[:10]
test_results = manager.test_manual_auto_coder_combination(test_segments)

print(f"Segments with both types: {test_results['segments_with_both']}")
print(f"Agreement rate: {test_results['reliability_metrics']['agreement_rate']:.2%}")
print(f"Cache isolation verified: {test_results['cache_isolation_verified']}")
```

## Monitoring and Statistics

### get_statistics()

```python
def get_statistics(self) -> CacheStatistics:
    """
    Get comprehensive cache statistics including strategy information and manual coder separation.
    
    Returns:
        Enhanced cache statistics with strategy info and manual coder separation
    """
```

### get_monitoring_dashboard_data()

```python
def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
    """
    Get data formatted for monitoring dashboard display including manual coder statistics.
    
    Returns:
        Dictionary with dashboard-ready monitoring data including manual coder info
    """
```

### export_performance_report()

```python
def export_performance_report(self, filepath: str) -> None:
    """
    Export comprehensive performance report including cache and reliability data.
    
    Args:
        filepath: Path to export file
    """
```

**Example:**

```python
# Get detailed statistics
stats = manager.get_statistics()
print(f"Total entries: {stats.total_entries}")
print(f"Hit rate: {stats.hit_rate_overall:.2%}")
print(f"Memory usage: {stats.memory_usage_mb:.1f} MB")
print(f"Strategy: {stats.strategy_type}")

# Get dashboard data
dashboard = manager.get_monitoring_dashboard_data()
print(f"Status: {dashboard['status']}")
print(f"Key metrics: {dashboard['key_metrics']}")
print(f"Manual coder info: {dashboard['manual_coder_info']}")

# Export comprehensive report
manager.export_performance_report("cache_performance_report.json")
```

## Plugin System

### Plugin Management

```python
def load_plugin(self, 
               plugin_path: str, 
               plugin_name: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Load a plugin from file.
    
    Args:
        plugin_path: Path to plugin file
        plugin_name: Optional name for plugin
        config: Optional configuration for plugin
        
    Returns:
        True if loading successful
    """

def unload_plugin(self, plugin_name: str) -> bool:
    """
    Unload a plugin.
    
    Args:
        plugin_name: Name of plugin to unload
        
    Returns:
        True if unloading successful
    """

def get_available_strategies(self) -> Dict[str, str]:
    """
    Get all available cache strategies (built-in and plugin-provided).
    
    Returns:
        Dictionary mapping strategy types to descriptions
    """
```

### Custom Strategy Management

```python
def create_custom_strategy(self, strategy_type: str, **kwargs) -> Optional[CacheStrategy]:
    """
    Create a custom strategy using plugins.
    
    Args:
        strategy_type: Type of strategy to create
        **kwargs: Additional parameters for strategy creation
        
    Returns:
        Strategy instance or None if creation failed
    """

def set_custom_strategy(self, strategy: CacheStrategy) -> None:
    """
    Set a custom strategy as the current strategy.
    
    Args:
        strategy: Custom strategy instance to use
    """
```

**Example:**

```python
# Load custom plugin
success = manager.load_plugin(
    "plugins/my_custom_strategy.py",
    plugin_name="MyCustomStrategy",
    config={"param1": "value1"}
)

if success:
    print("Plugin loaded successfully")
    
    # Get available strategies
    strategies = manager.get_available_strategies()
    print(f"Available strategies: {list(strategies.keys())}")
    
    # Create custom strategy
    custom_strategy = manager.create_custom_strategy(
        "MyCustomStrategy",
        special_param="custom_value"
    )
    
    if custom_strategy:
        # Set as current strategy
        manager.set_custom_strategy(custom_strategy)
        print(f"Using custom strategy: {custom_strategy.name}")

# Validate plugin system
validation_results = manager.validate_plugin_system()
for plugin_name, errors in validation_results.items():
    if errors:
        print(f"Plugin {plugin_name} has errors: {errors}")
```

## Test Mode and Debugging

### Test Mode Management

```python
def enable_test_mode(self, 
                    mode: str = "deterministic", 
                    seed: int = 42, 
                    recording_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Enable test mode for deterministic cache behavior.
    
    Args:
        mode: Test mode ('deterministic', 'record', 'replay')
        seed: Random seed for deterministic mode
        recording_file: File for record/replay operations
        
    Returns:
        Test mode status
    """

def disable_test_mode(self) -> Dict[str, Any]:
    """
    Disable test mode and return to normal operation.
    
    Returns:
        Disable operation status
    """

def get_test_mode_status(self) -> Dict[str, Any]:
    """
    Get current test mode status.
    
    Returns:
        Test mode status information
    """
```

### Debug Logging

```python
def set_debug_level(self, level: str) -> Dict[str, Any]:
    """
    Set debug logging level.
    
    Args:
        level: Debug level (SILENT, ERROR, WARNING, INFO, DEBUG, TRACE)
        
    Returns:
        Operation status
    """

def get_debug_summary(self) -> Dict[str, Any]:
    """
    Get debug operation summary.
    
    Returns:
        Debug summary information
    """

def start_debug_session(self, 
                       log_file: Optional[str] = None, 
                       test_mode: Optional[str] = None, 
                       test_seed: int = 42) -> Dict[str, Any]:
    """
    Start comprehensive debug session.
    
    Args:
        log_file: Optional file to log debug information
        test_mode: Optional test mode to enable
        test_seed: Seed for deterministic test mode
        
    Returns:
        Debug session status
    """
```

**Example:**

```python
# Enable deterministic test mode
test_result = manager.enable_test_mode("deterministic", seed=42)
print(f"Test mode enabled: {test_result['status']}")

# Set debug level
debug_result = manager.set_debug_level("DEBUG")
print(f"Debug level set to: {debug_result['new_level']}")

# Start comprehensive debug session
session = manager.start_debug_session(
    log_file="cache_debug.log",
    test_mode="deterministic",
    test_seed=42
)
print(f"Debug session started: {session['session_started']}")

# Get debug summary
summary = manager.get_debug_summary()
print(f"Operations logged: {summary['operation_summary']}")

# Disable test mode when done
manager.disable_test_mode()
```

## Migration and Consistency

### Cache Migration

```python
def get_migration_status(self) -> Dict[str, Any]:
    """
    Get status of cache migration operations.
    
    Returns:
        Dictionary with migration status information
    """

def force_consistency_check(self) -> Dict[str, Any]:
    """
    Force a consistency check and return detailed results.
    
    Returns:
        Dictionary with consistency check results
    """

def repair_cache_consistency(self, auto_fix: bool = True) -> Dict[str, Any]:
    """
    Repair cache consistency issues.
    
    Args:
        auto_fix: Whether to automatically fix issues or just report them
        
    Returns:
        Dictionary with repair results
    """
```

### Selective Cache Management

```python
def perform_selective_cache_clear(self, criteria: Dict[str, Any]) -> int:
    """
    Perform selective cache clearing based on criteria.
    
    Args:
        criteria: Dictionary with clearing criteria:
            - 'operations': List of operation types to clear
            - 'coders': List of coder IDs to clear
            - 'modes': List of analysis modes to clear
            - 'older_than_hours': Clear entries older than X hours
            - 'confidence_below': Clear entries with confidence below threshold
            
    Returns:
        Number of entries cleared
    """
```

**Example:**

```python
# Check migration status
migration_status = manager.get_migration_status()
print(f"Has backup: {migration_status['has_backup']}")
print(f"Consistency status: {migration_status['consistency_status']}")

# Force consistency check
consistency_result = manager.force_consistency_check()
print(f"Issues found: {consistency_result['total_issues']}")

if consistency_result['total_issues'] > 0:
    # Repair consistency issues
    repair_result = manager.repair_cache_consistency(auto_fix=True)
    print(f"Fixed {repair_result['fixed_issues']} issues")

# Selective cache clearing
cleared_count = manager.perform_selective_cache_clear({
    'operations': ['coding'],
    'older_than_hours': 24,
    'confidence_below': 0.5
})
print(f"Cleared {cleared_count} entries")
```

## Performance Benchmarking

### Benchmark Execution

```python
def create_benchmark_suite(self) -> Optional['CacheBenchmark']:
    """
    Create benchmark suite for performance testing.
    
    Returns:
        Benchmark suite instance or None if not available
    """

def run_performance_benchmark(self, test_type: str = "basic", **kwargs) -> Dict[str, Any]:
    """
    Run performance benchmark.
    
    Args:
        test_type: Type of benchmark ('basic', 'multi_coder', 'memory_stress')
        **kwargs: Additional parameters for benchmark
        
    Returns:
        Benchmark results
    """
```

### Cache Dump and Restore

```python
def create_cache_dump(self, 
                     filepath: str, 
                     include_values: bool = False, 
                     compress: bool = True, 
                     format: str = 'json') -> Dict[str, Any]:
    """
    Create cache dump for debugging.
    
    Args:
        filepath: Path to save dump
        include_values: Whether to include cached values
        compress: Whether to compress large values
        format: Dump format ('json' or 'pickle')
        
    Returns:
        Dump operation status
    """

def restore_cache_dump(self, 
                      filepath: str, 
                      clear_existing: bool = True, 
                      restore_values: bool = False, 
                      format: str = 'json') -> Dict[str, Any]:
    """
    Restore cache from dump.
    
    Args:
        filepath: Path to dump file
        clear_existing: Whether to clear existing cache
        restore_values: Whether to restore cached values
        format: Dump format ('json' or 'pickle')
        
    Returns:
        Restore operation status
    """
```

**Example:**

```python
# Run basic performance benchmark
basic_result = manager.run_performance_benchmark(
    test_type='basic',
    operations_count=1000
)
print(f"Operations/sec: {basic_result['benchmark_result']['operations_per_second']:.1f}")
print(f"Hit rate: {basic_result['benchmark_result']['hit_rate']:.2%}")

# Run multi-coder benchmark
multi_result = manager.run_performance_benchmark(
    test_type='multi_coder',
    coder_count=3,
    operations_per_coder=500
)
print(f"Total operations: {multi_result['benchmark_result']['operations_count']}")

# Create cache dump
dump_result = manager.create_cache_dump(
    'cache_dump.json',
    include_values=True,
    compress=True
)
print(f"Dumped {dump_result['total_entries']} entries to {dump_result['filepath']}")

# Restore from dump
restore_result = manager.restore_cache_dump(
    'cache_dump.json',
    clear_existing=True,
    restore_values=True
)
print(f"Restored {restore_result['restore_result']['restored_entries']} entries")
```

## Error Handling

### Common Error Scenarios

The DynamicCacheManager handles various error scenarios gracefully:

1. **Configuration Errors**: Invalid coder settings fall back to default single-coder strategy
2. **Strategy Migration Failures**: Automatic rollback to previous state with consistency checks
3. **Database Connection Issues**: Fallback to in-memory reliability data storage
4. **Plugin Loading Errors**: Continue with built-in strategies, log warnings
5. **Memory Pressure**: Automatic cache eviction with LRU policy

### Error Information

```python
# Get error information from cache statistics
stats = manager.get_statistics()
error_rate = stats.performance_metrics.get('error_rate', 0)
error_counts = stats.error_counts

print(f"Error rate: {error_rate:.2%}")
print(f"Error counts by type: {error_counts}")

# Get detailed error report
dashboard_data = manager.get_monitoring_dashboard_data()
alerts = dashboard_data['alerts']

for alert in alerts:
    print(f"Alert: {alert}")
```

### Exception Handling Best Practices

```python
try:
    # Configure coders
    manager.configure_coders(coder_settings)
    
    # Perform cache operations
    shared_key = manager.get_shared_cache_key('relevance_check', **params)
    
except Exception as e:
    # Log error and get system state
    logger.error(f"Cache operation failed: {e}")
    
    # Check system health
    consistency_result = manager.force_consistency_check()
    if consistency_result['total_issues'] > 0:
        # Attempt repair
        repair_result = manager.repair_cache_consistency(auto_fix=True)
        logger.info(f"Repaired {repair_result['fixed_issues']} consistency issues")
    
    # Export debug information
    manager.export_performance_report("error_debug_report.json")
```

## Integration Examples

### Basic Integration

```python
from QCA_AID_assets.optimization.cache import ModeAwareCache
from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager

# Initialize cache system
cache = ModeAwareCache()
manager = DynamicCacheManager(cache, analysis_mode="inductive")

# Configure for multi-coder analysis
manager.configure_coders([
    {'coder_id': 'coder_1', 'temperature': 0.1},
    {'coder_id': 'coder_2', 'temperature': 0.3}
])

# Use in analysis workflow
def perform_analysis(segment_text, research_question):
    # Check if relevance already cached (shared operation)
    relevance_key = manager.get_shared_cache_key(
        'relevance_check',
        segment_text=segment_text,
        research_question=research_question
    )
    
    # Perform coding for each coder (coder-specific operation)
    for coder_config in manager.coder_settings:
        coder_id = coder_config['coder_id']
        coding_key = manager.get_coder_specific_key(
            coder_id,
            'coding',
            segment_text=segment_text,
            category_definitions=category_definitions
        )
        
        # Store coding result for reliability analysis
        coding_result = ExtendedCodingResult(...)
        manager.store_for_reliability(coding_result)
```

### Advanced Integration with Manual Coders

```python
# Configure system for manual + automatic coder combination
manager.configure_coders([
    {'coder_id': 'auto_coder_1', 'temperature': 0.2},
    {'coder_id': 'auto_coder_2', 'temperature': 0.4}
])

# Process automatic codings
for segment in segments:
    for coder_config in manager.coder_settings:
        # Perform automatic coding
        result = perform_automatic_coding(segment, coder_config)
        manager.store_for_reliability(result)

# Process manual codings from GUI
def handle_manual_coding(segment_id, category, subcategories, justification, confidence):
    manager.store_manual_coding(
        segment_id=segment_id,
        category=category,
        subcategories=subcategories,
        justification=justification,
        confidence=confidence,
        analysis_mode=manager.analysis_mode
    )

# Analyze intercoder reliability
reliability_segments = manager.get_segments_for_reliability_analysis()
test_results = manager.test_manual_auto_coder_combination(reliability_segments)

print(f"Agreement rate: {test_results['reliability_metrics']['agreement_rate']:.2%}")
print(f"Cache isolation verified: {test_results['cache_isolation_verified']}")
```

## Version Compatibility

### API Versioning

The DynamicCacheManager maintains backward compatibility with the existing ModeAwareCache API:

- All existing cache operations continue to work unchanged
- New functionality is additive and optional
- Configuration changes are detected automatically
- Migration between strategies is handled transparently

### Deprecation Notices

- Direct cache manipulation methods are deprecated in favor of strategy-based operations
- Manual cache key generation is deprecated in favor of `get_shared_cache_key()` and `get_coder_specific_key()`
- Direct reliability data storage is deprecated in favor of `store_for_reliability()`

## Performance Considerations

### Memory Usage

- Shared cache entries reduce memory usage in multi-coder scenarios
- LRU eviction prevents unbounded memory growth
- Compressed cache dumps minimize storage requirements
- Manual coder isolation prevents cache pollution

### CPU Performance

- Strategy selection is cached and only recalculated on configuration changes
- Cache key generation is optimized for common operation patterns
- Database operations are batched for efficiency
- Plugin loading is performed once at initialization

### Network/IO Performance

- Reliability database uses SQLite for efficient local storage
- Cache dumps support compression for reduced file sizes
- Configuration hot-reload minimizes system restarts
- Batch operations reduce database transaction overhead

## Related Documentation

- [Migration Guide](MIGRATION_GUIDE.md) - Migrating from old cache system
- [Configuration Guide](CONFIGURATION_GUIDE.md) - Configuration options and best practices
- [Performance Benchmarks](PERFORMANCE_BENCHMARKS.md) - Performance analysis and comparisons
- [Test Mode Guide](TEST_MODE_AND_DEBUGGING_GUIDE.md) - Testing and debugging features
- [Plugin Development Guide](PLUGIN_DEVELOPMENT_GUIDE.md) - Creating custom cache strategies

## Support and Troubleshooting

For issues and questions:

1. Check the [Migration Guide](MIGRATION_GUIDE.md) for common migration issues
2. Review the [Test Mode Guide](TEST_MODE_AND_DEBUGGING_GUIDE.md) for debugging techniques
3. Use `manager.export_performance_report()` to generate diagnostic information
4. Enable debug logging with `manager.set_debug_level('DEBUG')` for detailed operation logs