# Test Mode and Debugging Features Guide

## Overview

The Dynamic Cache System includes comprehensive test mode and debugging features to support development, testing, and troubleshooting. This guide covers all available features and how to use them.

## Features

### 1. Test Modes for Deterministic Behavior

The system supports multiple test modes to ensure reproducible and predictable cache behavior during testing:

#### Deterministic Mode
Enables fixed random seeds for reproducible results across test runs.

```python
from QCA_AID_assets.optimization.cache import ModeAwareCache
from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager

# Initialize with test mode enabled
cache = ModeAwareCache()
manager = DynamicCacheManager(cache, test_mode=True, debug_level='DEBUG')

# Enable deterministic mode with specific seed
result = manager.enable_test_mode('deterministic', seed=42)
print(f"Test mode enabled: {result}")
```

#### Record Mode
Records all cache operations for later replay and analysis.

```python
# Start recording cache operations
result = manager.enable_test_mode('record', recording_file='cache_session.json')

# ... perform cache operations ...

# Stop recording (automatically saves to file)
manager.disable_test_mode()
```

#### Replay Mode
Replays previously recorded cache operations for regression testing.

```python
# Replay recorded session
result = manager.enable_test_mode('replay', recording_file='cache_session.json')

# Operations will be replayed from the recording
```

### 2. Logging Verbosity Levels

Six logging levels provide granular control over debug output:

- **SILENT**: No cache logging
- **ERROR**: Only errors
- **WARNING**: Errors and warnings
- **INFO**: Basic operations
- **DEBUG**: Detailed operations
- **TRACE**: All cache operations with full context

```python
# Set logging level
manager.set_debug_level('TRACE')

# Get current debug status
debug_summary = manager.get_debug_summary()
print(f"Current level: {debug_summary['debug_level']}")
print(f"Operations logged: {debug_summary['operation_summary']}")
```

### 3. Performance Benchmarking Tools

Built-in benchmarking suite for measuring cache performance:

#### Basic Performance Test
```python
# Run basic performance benchmark
result = manager.run_performance_benchmark(
    test_type='basic',
    operations_count=1000
)

print(f"Operations/sec: {result['benchmark_result']['operations_per_second']:.1f}")
print(f"Hit rate: {result['benchmark_result']['hit_rate']:.2%}")
print(f"Memory usage: {result['benchmark_result']['memory_usage_mb']:.1f} MB")
```

#### Multi-Coder Benchmark
```python
# Test multi-coder performance
result = manager.run_performance_benchmark(
    test_type='multi_coder',
    coder_count=3,
    operations_per_coder=500
)

print(f"Total operations: {result['benchmark_result']['operations_count']}")
print(f"Shared entries: {result['benchmark_result']['metadata']['shared_entries']}")
print(f"Coder-specific entries: {result['benchmark_result']['metadata']['coder_specific_entries']}")
```

#### Memory Stress Test
```python
# Test cache behavior under memory pressure
result = manager.run_performance_benchmark(
    test_type='memory_stress',
    target_memory_mb=100
)

print(f"Target memory: {result['benchmark_result']['metadata']['target_memory_mb']} MB")
print(f"Achieved memory: {result['benchmark_result']['metadata']['achieved_memory_mb']} MB")
```

#### Export Benchmark Results
```python
# Create benchmark suite for custom tests
benchmark = manager.create_benchmark_suite()

# Run multiple benchmarks
benchmark.run_basic_performance_test(operations_count=1000)
benchmark.run_multi_coder_benchmark(coder_count=3, operations_per_coder=500)
benchmark.run_memory_stress_test(target_memory_mb=50)

# Export all results
benchmark.export_benchmark_results('benchmark_results.json')
```

### 4. Cache Dump and Restore

Capture and restore complete cache state for debugging and analysis:

#### Create Cache Dump
```python
# Create dump without values (metadata only)
result = manager.create_cache_dump(
    filepath='cache_dump.json',
    include_values=False,
    compress=True,
    format='json'
)

print(f"Dumped {result['total_entries']} entries")
print(f"File: {result['filepath']}")

# Create dump with values (for full restoration)
result = manager.create_cache_dump(
    filepath='cache_dump_full.json',
    include_values=True,
    compress=True,
    format='json'
)
```

#### Restore from Dump
```python
# Restore cache from dump
result = manager.restore_cache_dump(
    filepath='cache_dump.json',
    clear_existing=True,
    restore_values=False,
    format='json'
)

print(f"Restored {result['restore_result']['restored_entries']} entries")
print(f"Errors: {result['restore_result']['error_count']}")
```

#### Compare Dumps
```python
from QCA_AID_assets.optimization.cache_debug_tools import CacheDumpManager

dump_manager = CacheDumpManager(manager)

# Load two dumps
dump1 = dump_manager.load_dump('cache_dump_before.json')
dump2 = dump_manager.load_dump('cache_dump_after.json')

# Compare them
comparison = dump_manager.compare_dumps(dump1, dump2)

print(f"Only in dump1: {comparison['key_differences']['only_in_dump1']}")
print(f"Only in dump2: {comparison['key_differences']['only_in_dump2']}")
print(f"Common keys: {comparison['key_differences']['common_keys']}")
print(f"Entry differences: {comparison['total_entry_differences']}")
```

## Complete Debug Session Example

```python
from QCA_AID_assets.optimization.cache import ModeAwareCache
from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager

# Initialize cache manager with debugging enabled
cache = ModeAwareCache()
manager = DynamicCacheManager(
    cache,
    test_mode=True,
    debug_level='DEBUG'
)

# Start comprehensive debug session
session = manager.start_debug_session(
    log_file='cache_debug.log',
    test_mode='deterministic',
    test_seed=42
)

print(f"Debug session started: {session['session_started']}")

# Configure coders for testing
manager.configure_coders([
    {'coder_id': 'coder_1', 'temperature': 0.1},
    {'coder_id': 'coder_2', 'temperature': 0.3},
    {'coder_id': 'coder_3', 'temperature': 0.5}
])

# Run performance benchmarks
basic_result = manager.run_performance_benchmark('basic', operations_count=500)
multi_result = manager.run_performance_benchmark('multi_coder', coder_count=3)

# Create cache dump for analysis
dump_result = manager.create_cache_dump(
    'debug_cache_dump.json',
    include_values=True,
    compress=True
)

# Export comprehensive debug report
report_result = manager.export_debug_report('debug_report.json')

print(f"Debug report exported: {report_result['filepath']}")
print(f"Report sections: {report_result['report_sections']}")

# Get test mode status
test_status = manager.get_test_mode_status()
print(f"Test mode active: {test_status['active']}")
print(f"Test mode info: {test_status['info']}")

# Disable test mode when done
manager.disable_test_mode()
```

## Debug Report Contents

The comprehensive debug report includes:

1. **Manager Info**: Current configuration and state
2. **Cache Statistics**: Hit rates, memory usage, entry counts
3. **Debug Summary**: Operation counts, error logs
4. **Test Mode Status**: Current test mode and settings
5. **Consistency Check**: Cache integrity validation
6. **Reliability Summary**: Intercoder reliability data
7. **Performance Metrics**: Detailed performance statistics

## Context Manager for Debug Sessions

Use the context manager for automatic setup and cleanup:

```python
from QCA_AID_assets.optimization.cache_debug_tools import cache_debug_session, LogLevel, TestMode

with cache_debug_session(
    manager,
    log_level=LogLevel.DEBUG,
    log_file='cache_debug.log',
    test_mode=TestMode.DETERMINISTIC,
    test_seed=42
) as debug_tools:
    
    # Access debug tools
    debug_logger = debug_tools['debug_logger']
    test_manager = debug_tools['test_manager']
    benchmark = debug_tools['benchmark']
    dump_manager = debug_tools['dump_manager']
    
    # Run tests
    benchmark.run_basic_performance_test(1000)
    
    # Create dump
    cache_dump = dump_manager.create_dump(include_values=True)
    dump_manager.save_dump(cache_dump, 'session_dump.json')
    
    # Get operation summary
    summary = debug_logger.get_operation_summary()
    print(f"Total operations: {summary['total_operations']}")
    print(f"Errors: {summary['error_count']}")

# Cleanup happens automatically
```

## Best Practices

### For Development
1. Use **DEBUG** or **TRACE** level for detailed operation tracking
2. Enable **deterministic mode** for reproducible test results
3. Create cache dumps before and after changes to compare behavior

### For Testing
1. Use **record/replay mode** for regression testing
2. Run benchmarks to establish performance baselines
3. Use test mode to ensure consistent test results

### For Production Debugging
1. Use **INFO** or **WARNING** level to minimize overhead
2. Create cache dumps when issues occur
3. Export debug reports for analysis
4. Use selective cache clearing to test specific scenarios

### For Performance Analysis
1. Run all three benchmark types (basic, multi-coder, memory stress)
2. Export benchmark results for comparison
3. Monitor cache statistics over time
4. Use memory stress tests to find optimal cache sizes

## Troubleshooting

### High Memory Usage
```python
# Run memory stress test to identify issues
result = manager.run_performance_benchmark('memory_stress', target_memory_mb=100)

# Check memory efficiency
efficiency = result['benchmark_result']['metadata']['memory_efficiency']
print(f"Memory efficiency: {efficiency:.2f} operations/MB")

# Perform selective cache clearing if needed
cleared = manager.perform_selective_cache_clear({
    'older_than_hours': 24,
    'confidence_below': 0.5
})
print(f"Cleared {cleared} entries")
```

### Low Hit Rates
```python
# Get detailed statistics
stats = manager.get_statistics()

# Check hit rates by coder
for coder_id, efficiency in stats.cache_efficiency_by_coder.items():
    print(f"Coder {coder_id}: {efficiency['hit_rate']:.2%} hit rate")

# Check hit rates by operation
for operation, hits in stats.hits_by_operation.items():
    total = hits + stats.misses_by_operation.get(operation, 0)
    hit_rate = hits / total if total > 0 else 0
    print(f"Operation {operation}: {hit_rate:.2%} hit rate")
```

### Cache Consistency Issues
```python
# Force consistency check
consistency_result = manager.force_consistency_check()

print(f"Total issues: {consistency_result['total_issues']}")
print(f"Issues found: {consistency_result['issues']}")

# Repair if needed
if consistency_result['total_issues'] > 0:
    repair_result = manager.repair_cache_consistency(auto_fix=True)
    print(f"Fixed {repair_result['fixed_issues']} issues")
    print(f"Remaining issues: {repair_result['issues_after']}")
```

## API Reference

### DynamicCacheManager Debug Methods

- `enable_test_mode(mode, seed, recording_file)` - Enable test mode
- `disable_test_mode()` - Disable test mode
- `get_test_mode_status()` - Get test mode status
- `set_debug_level(level)` - Set logging verbosity
- `get_debug_summary()` - Get debug operation summary
- `create_benchmark_suite()` - Create benchmark suite
- `run_performance_benchmark(test_type, **kwargs)` - Run benchmark
- `create_cache_dump(filepath, include_values, compress, format)` - Create dump
- `restore_cache_dump(filepath, clear_existing, restore_values, format)` - Restore dump
- `start_debug_session(log_file, test_mode, test_seed)` - Start debug session
- `export_debug_report(filepath)` - Export debug report
- `force_consistency_check()` - Check cache consistency
- `repair_cache_consistency(auto_fix)` - Repair consistency issues

### Cache Debug Tools

- `CacheDebugLogger` - Enhanced logging with operation tracking
- `TestModeManager` - Test mode management
- `CacheBenchmark` - Performance benchmarking suite
- `CacheDumpManager` - Cache dump and restore operations
- `cache_debug_session()` - Context manager for debug sessions

## Requirements Validation

This implementation fulfills all requirements from Task 11:

✅ **Test-Mode für deterministische Cache-Ergebnisse**
- Deterministic mode with configurable seeds
- Record/replay functionality for regression testing
- Test mode status tracking and management

✅ **Verschiedene Logging-Verbosity-Level**
- Six logging levels (SILENT, ERROR, WARNING, INFO, DEBUG, TRACE)
- Dynamic level switching without restart
- Operation counting and error tracking

✅ **Benchmarking-Tools für Performance-Messung**
- Basic performance benchmarks
- Multi-coder performance testing
- Memory stress testing
- Benchmark result export and analysis

✅ **Cache-Dump und -Restore Funktionen für Debugging**
- Complete cache state dumps (with/without values)
- Compressed dump support
- Cache restoration from dumps
- Dump comparison for analysis
- Multiple format support (JSON, pickle)

## Related Documentation

- [Dynamic Cache System Design](design.md)
- [Cache Strategies Guide](CACHE_STRATEGIES.md)
- [Plugin System Guide](PLUGIN_SYSTEM.md)
- [API Documentation](API_DOCUMENTATION.md)
