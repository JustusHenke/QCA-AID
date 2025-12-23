# Migration Guide: From Old Cache System to Dynamic Cache System

## Overview

This guide helps you migrate from the old cache system (where cache was completely disabled for multi-coder scenarios) to the new Dynamic Cache System that intelligently manages cache for both single and multi-coder workflows.

## Table of Contents

1. [Why Migrate?](#why-migrate)
2. [Breaking Changes](#breaking-changes)
3. [Migration Steps](#migration-steps)
4. [Code Migration Examples](#code-migration-examples)
5. [Configuration Migration](#configuration-migration)
6. [Testing Your Migration](#testing-your-migration)
7. [Rollback Plan](#rollback-plan)
8. [Common Issues and Solutions](#common-issues-and-solutions)

## Why Migrate?

### Benefits of the New System

**Performance Improvements:**
- **50-70% reduction in API calls** for multi-coder scenarios
- Shared operations (relevance checking, category development) cached once instead of per-coder
- Intelligent cache key generation prevents unnecessary cache misses

**Methodological Integrity:**
- Respects QCA methodological principles for each analysis mode
- Proper separation of shared vs. coder-specific operations
- Manual coder isolation ensures no cross-contamination

**Reliability Analysis:**
- Built-in intercoder reliability data collection
- Persistent storage of all coding results
- Support for manual + automatic coder combinations

**Operational Benefits:**
- Hot-reload configuration changes without restart
- Comprehensive monitoring and statistics
- Plugin system for custom cache strategies
- Test modes for reproducible testing

### Old System Limitations

The old system had these limitations:

```python
# OLD SYSTEM - OptimizationController
if len(coder_settings) > 1:
    # Cache completely disabled for multi-coder!
    self.cache.disable()
    logger.info("Cache disabled for multi-coder scenario")
```

**Problems:**
- ❌ All operations repeated for each coder (even shared ones)
- ❌ No intercoder reliability data collection
- ❌ No distinction between shared and coder-specific operations
- ❌ Manual coders not supported
- ❌ No performance monitoring

## Breaking Changes

### API Changes

#### 1. Cache Initialization

**OLD:**
```python
from QCA_AID_assets.optimization.cache import ModeAwareCache

cache = ModeAwareCache()
# Cache was managed directly by OptimizationController
```

**NEW:**
```python
from QCA_AID_assets.optimization.cache import ModeAwareCache
from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager

cache = ModeAwareCache()
manager = DynamicCacheManager(cache, analysis_mode="inductive")
```

#### 2. Coder Configuration

**OLD:**
```python
# Coder settings passed directly to controller
controller = OptimizationController(
    cache=cache,
    coder_settings=coder_settings
)
```

**NEW:**
```python
# Coder settings configured through manager
manager.configure_coders(coder_settings)
```

#### 3. Cache Key Generation

**OLD:**
```python
# Manual cache key generation
cache_key = f"{operation}_{segment_id}_{hash(params)}"
```

**NEW:**
```python
# Strategy-based cache key generation
if manager.should_cache_shared(operation):
    cache_key = manager.get_shared_cache_key(operation, **params)
else:
    cache_key = manager.get_coder_specific_key(coder_id, operation, **params)
```

### Behavioral Changes

#### 1. Multi-Coder Cache Behavior

**OLD:** Cache completely disabled for multiple coders
**NEW:** Intelligent caching with shared and coder-specific separation

#### 2. Manual Coder Handling

**OLD:** Manual coders not supported in cache system
**NEW:** Manual coders fully integrated with automatic isolation

#### 3. Reliability Data

**OLD:** No built-in reliability data collection
**NEW:** Automatic collection and persistent storage

## Migration Steps

### Step 1: Update Dependencies

Ensure you have the latest version of the optimization module:

```python
# Check if new components are available
try:
    from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager
    from QCA_AID_assets.optimization.cache_strategies import CacheStrategyFactory
    from QCA_AID_assets.optimization.reliability_database import ReliabilityDatabase
    print("✓ New cache system components available")
except ImportError as e:
    print(f"✗ Missing components: {e}")
    print("Please update to the latest version")
```

### Step 2: Update OptimizationController Integration

**OLD CODE (optimization/controller.py):**

```python
class OptimizationController:
    def __init__(self, cache, coder_settings, ...):
        self.cache = cache
        self.coder_settings = coder_settings
        
        # Disable cache for multi-coder
        if len(coder_settings) > 1:
            self.cache.disable()
            logger.info("Cache disabled for multi-coder scenario")
```

**NEW CODE (optimization/controller.py):**

```python
class OptimizationController:
    def __init__(self, cache, coder_settings, analysis_mode, ...):
        # Initialize dynamic cache manager
        self.cache_manager = DynamicCacheManager(
            base_cache=cache,
            analysis_mode=analysis_mode,
            reliability_db_path="output/reliability_data.db"
        )
        
        # Configure coders
        self.cache_manager.configure_coders(coder_settings)
        
        # Cache is now intelligently managed
        logger.info(f"Cache strategy: {self.cache_manager.get_cache_strategy().name}")
```

### Step 3: Update Cache Operations

**OLD CODE:**

```python
def check_relevance(self, segment_text, research_question):
    # Manual cache key generation
    cache_key = f"relevance_{hash(segment_text)}_{hash(research_question)}"
    
    # Check cache
    if cache_key in self.cache:
        return self.cache[cache_key]
    
    # Perform operation
    result = self.llm_provider.check_relevance(segment_text, research_question)
    
    # Store in cache
    self.cache[cache_key] = result
    return result
```

**NEW CODE:**

```python
def check_relevance(self, segment_text, research_question):
    # Generate shared cache key (relevance is shared across coders)
    cache_key = self.cache_manager.get_shared_cache_key(
        'relevance_check',
        segment_text=segment_text,
        research_question=research_question
    )
    
    # Check cache through base cache
    if cache_key in self.cache_manager.base_cache:
        return self.cache_manager.base_cache[cache_key]
    
    # Perform operation
    result = self.llm_provider.check_relevance(segment_text, research_question)
    
    # Store in cache
    self.cache_manager.base_cache[cache_key] = result
    return result
```

### Step 4: Update Coding Operations

**OLD CODE:**

```python
def perform_coding(self, segment_text, coder_id, category_definitions):
    # No caching for multi-coder scenarios
    result = self.llm_provider.code_segment(
        segment_text, 
        category_definitions,
        temperature=self.get_coder_temperature(coder_id)
    )
    return result
```

**NEW CODE:**

```python
def perform_coding(self, segment_text, coder_id, category_definitions):
    # Generate coder-specific cache key
    cache_key = self.cache_manager.get_coder_specific_key(
        coder_id,
        'coding',
        segment_text=segment_text,
        category_definitions=category_definitions
    )
    
    # Check cache
    if cache_key in self.cache_manager.base_cache:
        cached_result = self.cache_manager.base_cache[cache_key]
        
        # Store for reliability analysis
        self.cache_manager.store_for_reliability(cached_result)
        return cached_result
    
    # Perform operation
    result = self.llm_provider.code_segment(
        segment_text, 
        category_definitions,
        temperature=self.get_coder_temperature(coder_id)
    )
    
    # Create extended coding result
    from QCA_AID_assets.core.data_models import ExtendedCodingResult
    from datetime import datetime
    
    extended_result = ExtendedCodingResult(
        segment_id=segment_id,
        coder_id=coder_id,
        category=result['category'],
        subcategories=result.get('subcategories', []),
        confidence=result.get('confidence', 0.0),
        justification=result.get('justification', ''),
        analysis_mode=self.cache_manager.analysis_mode,
        timestamp=datetime.now(),
        is_manual=False
    )
    
    # Store in cache
    self.cache_manager.base_cache[cache_key] = extended_result
    
    # Store for reliability analysis
    self.cache_manager.store_for_reliability(extended_result)
    
    return result
```

### Step 5: Add Manual Coder Support

**NEW CODE:**

```python
def handle_manual_coding(self, segment_id, category, subcategories, 
                        justification, confidence):
    """Handle manual coding from GUI."""
    self.cache_manager.store_manual_coding(
        segment_id=segment_id,
        category=category,
        subcategories=subcategories,
        justification=justification,
        confidence=confidence,
        analysis_mode=self.cache_manager.analysis_mode
    )
    
    logger.info(f"Manual coding stored for segment {segment_id}")
```

### Step 6: Add Monitoring and Statistics

**NEW CODE:**

```python
def get_analysis_statistics(self):
    """Get comprehensive analysis statistics."""
    # Get cache statistics
    cache_stats = self.cache_manager.get_statistics()
    
    # Get reliability summary
    reliability_summary = self.cache_manager.get_reliability_summary()
    
    return {
        'cache': {
            'total_entries': cache_stats.total_entries,
            'hit_rate': cache_stats.hit_rate_overall,
            'memory_usage_mb': cache_stats.memory_usage_mb,
            'strategy': cache_stats.strategy_type
        },
        'reliability': {
            'total_segments': reliability_summary['total_segments'],
            'total_codings': reliability_summary['total_codings'],
            'manual_codings': reliability_summary['manual_codings'],
            'automatic_codings': reliability_summary['automatic_codings'],
            'segments_with_multiple_coders': reliability_summary['segments_with_multiple_coders']
        }
    }
```

## Code Migration Examples

### Example 1: Simple Single-Coder Migration

**BEFORE:**

```python
from QCA_AID_assets.optimization.cache import ModeAwareCache
from QCA_AID_assets.optimization.controller import OptimizationController

cache = ModeAwareCache()
controller = OptimizationController(
    cache=cache,
    coder_settings=[{'coder_id': 'main', 'temperature': 0.2}],
    analysis_mode='inductive'
)

# Run analysis
results = controller.run_analysis(segments)
```

**AFTER:**

```python
from QCA_AID_assets.optimization.cache import ModeAwareCache
from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager
from QCA_AID_assets.optimization.controller import OptimizationController

cache = ModeAwareCache()
manager = DynamicCacheManager(cache, analysis_mode='inductive')
manager.configure_coders([{'coder_id': 'main', 'temperature': 0.2}])

controller = OptimizationController(
    cache_manager=manager,
    analysis_mode='inductive'
)

# Run analysis (cache now works efficiently)
results = controller.run_analysis(segments)

# Get statistics
stats = manager.get_statistics()
print(f"Cache hit rate: {stats.hit_rate_overall:.2%}")
```

### Example 2: Multi-Coder Migration

**BEFORE:**

```python
# Multi-coder setup - cache was disabled!
controller = OptimizationController(
    cache=cache,
    coder_settings=[
        {'coder_id': 'coder_1', 'temperature': 0.1},
        {'coder_id': 'coder_2', 'temperature': 0.3},
        {'coder_id': 'coder_3', 'temperature': 0.5}
    ],
    analysis_mode='inductive'
)

# All operations repeated for each coder
results = controller.run_analysis(segments)
```

**AFTER:**

```python
# Multi-coder setup - intelligent caching!
manager = DynamicCacheManager(cache, analysis_mode='inductive')
manager.configure_coders([
    {'coder_id': 'coder_1', 'temperature': 0.1},
    {'coder_id': 'coder_2', 'temperature': 0.3},
    {'coder_id': 'coder_3', 'temperature': 0.5}
])

controller = OptimizationController(
    cache_manager=manager,
    analysis_mode='inductive'
)

# Shared operations cached once, coding cached per coder
results = controller.run_analysis(segments)

# Get reliability data
reliability_data = manager.get_reliability_data()
print(f"Collected {len(reliability_data)} coding results for reliability analysis")
```

### Example 3: Manual + Automatic Coder Migration

**BEFORE:**

```python
# Manual coders not supported in old system
# Had to use separate tracking mechanism
```

**AFTER:**

```python
# Configure automatic coders
manager.configure_coders([
    {'coder_id': 'auto_1', 'temperature': 0.2},
    {'coder_id': 'auto_2', 'temperature': 0.4}
])

# Run automatic analysis
auto_results = controller.run_analysis(segments)

# Add manual codings from GUI
for manual_coding in gui_manual_codings:
    manager.store_manual_coding(
        segment_id=manual_coding['segment_id'],
        category=manual_coding['category'],
        subcategories=manual_coding['subcategories'],
        justification=manual_coding['justification'],
        confidence=manual_coding['confidence'],
        analysis_mode='inductive'
    )

# Test reliability with manual + automatic combination
test_segments = manager.get_segments_for_reliability_analysis()
test_results = manager.test_manual_auto_coder_combination(test_segments)

print(f"Agreement rate: {test_results['reliability_metrics']['agreement_rate']:.2%}")
print(f"Cache isolation verified: {test_results['cache_isolation_verified']}")
```

## Configuration Migration

### Old Configuration Format

**OLD (config.json):**

```json
{
  "cache_enabled": true,
  "cache_size": 1000,
  "coder_settings": [
    {"coder_id": "coder_1", "temperature": 0.2}
  ]
}
```

### New Configuration Format

**NEW (config.json):**

```json
{
  "cache_config": {
    "enabled": true,
    "max_size": 1000,
    "enable_plugins": true,
    "plugins_directory": "custom_plugins/",
    "test_mode": false,
    "debug_level": "INFO"
  },
  "coder_settings": [
    {"coder_id": "coder_1", "temperature": 0.2},
    {"coder_id": "coder_2", "temperature": 0.4}
  ],
  "analysis_mode": "inductive",
  "reliability_db_path": "output/reliability_data.db"
}
```

### Configuration Loading

**NEW CODE:**

```python
import json

def load_configuration(config_file):
    """Load configuration and initialize cache manager."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Initialize cache
    cache = ModeAwareCache()
    
    # Initialize manager with configuration
    cache_config = config.get('cache_config', {})
    manager = DynamicCacheManager(
        base_cache=cache,
        config_file=config_file,
        reliability_db_path=config.get('reliability_db_path'),
        analysis_mode=config.get('analysis_mode'),
        enable_plugins=cache_config.get('enable_plugins', True),
        plugins_directory=cache_config.get('plugins_directory'),
        test_mode=cache_config.get('test_mode', False),
        debug_level=cache_config.get('debug_level', 'INFO')
    )
    
    # Configure coders
    manager.configure_coders(config.get('coder_settings', []))
    
    return manager
```

## Testing Your Migration

### Step 1: Unit Tests

Create tests to verify migration:

```python
import pytest
from QCA_AID_assets.optimization.cache import ModeAwareCache
from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager

def test_single_coder_migration():
    """Test single coder scenario works as before."""
    cache = ModeAwareCache()
    manager = DynamicCacheManager(cache, analysis_mode='inductive')
    
    # Configure single coder
    manager.configure_coders([{'coder_id': 'main', 'temperature': 0.2}])
    
    # Verify strategy
    assert manager.get_cache_strategy().name == "SingleCoder"
    
    # Test cache operations
    key = manager.get_shared_cache_key('relevance_check', segment_text="test")
    assert key is not None

def test_multi_coder_migration():
    """Test multi-coder scenario uses new strategy."""
    cache = ModeAwareCache()
    manager = DynamicCacheManager(cache, analysis_mode='inductive')
    
    # Configure multiple coders
    manager.configure_coders([
        {'coder_id': 'coder_1', 'temperature': 0.1},
        {'coder_id': 'coder_2', 'temperature': 0.3}
    ])
    
    # Verify strategy changed
    assert manager.get_cache_strategy().name == "MultiCoder"
    
    # Test shared operations
    assert manager.should_cache_shared('relevance_check')
    
    # Test coder-specific operations
    assert manager.should_cache_per_coder('coding')

def test_manual_coder_integration():
    """Test manual coder integration."""
    cache = ModeAwareCache()
    manager = DynamicCacheManager(cache, analysis_mode='inductive')
    
    # Store manual coding
    manager.store_manual_coding(
        segment_id="seg_001",
        category="theme_a",
        subcategories=["sub_1"],
        justification="Manual review",
        confidence=0.9,
        analysis_mode="inductive"
    )
    
    # Verify storage
    manual_codings = manager.get_manual_codings(["seg_001"])
    assert len(manual_codings) == 1
    assert manual_codings[0].coder_id == "manual"
    assert manual_codings[0].is_manual == True
```

### Step 2: Integration Tests

Test with real analysis workflow:

```python
def test_full_analysis_workflow():
    """Test complete analysis workflow with new cache system."""
    # Initialize
    cache = ModeAwareCache()
    manager = DynamicCacheManager(cache, analysis_mode='inductive')
    manager.configure_coders([
        {'coder_id': 'coder_1', 'temperature': 0.2},
        {'coder_id': 'coder_2', 'temperature': 0.4}
    ])
    
    # Simulate analysis
    test_segments = ["segment_1", "segment_2", "segment_3"]
    
    for segment_id in test_segments:
        # Shared operation (relevance check)
        relevance_key = manager.get_shared_cache_key(
            'relevance_check',
            segment_text=segment_id
        )
        
        # Coder-specific operations
        for coder_config in manager.coder_settings:
            coder_id = coder_config['coder_id']
            coding_key = manager.get_coder_specific_key(
                coder_id,
                'coding',
                segment_text=segment_id
            )
    
    # Verify statistics
    stats = manager.get_statistics()
    assert stats.total_entries > 0
    assert stats.shared_entries > 0
    assert stats.coder_specific_entries > 0
    
    # Verify reliability data
    reliability_summary = manager.get_reliability_summary()
    assert reliability_summary['total_segments'] == len(test_segments)
    assert reliability_summary['total_codings'] == len(test_segments) * 2  # 2 coders
```

### Step 3: Performance Comparison

Compare old vs. new system performance:

```python
def test_performance_improvement():
    """Verify performance improvement over old system."""
    import time
    
    # Setup
    cache = ModeAwareCache()
    manager = DynamicCacheManager(cache, analysis_mode='inductive')
    manager.configure_coders([
        {'coder_id': 'coder_1', 'temperature': 0.2},
        {'coder_id': 'coder_2', 'temperature': 0.4},
        {'coder_id': 'coder_3', 'temperature': 0.6}
    ])
    
    # Run benchmark
    result = manager.run_performance_benchmark(
        test_type='multi_coder',
        coder_count=3,
        operations_per_coder=100
    )
    
    # Verify improvements
    assert result['benchmark_result']['hit_rate'] > 0.3  # At least 30% hit rate
    assert result['benchmark_result']['operations_per_second'] > 50  # Reasonable throughput
    
    print(f"Performance Results:")
    print(f"  Hit rate: {result['benchmark_result']['hit_rate']:.2%}")
    print(f"  Operations/sec: {result['benchmark_result']['operations_per_second']:.1f}")
    print(f"  Memory usage: {result['benchmark_result']['memory_usage_mb']:.1f} MB")
```

## Rollback Plan

If you need to rollback to the old system:

### Step 1: Preserve Old Code

Before migration, create a backup:

```bash
# Backup old controller
cp QCA_AID_assets/optimization/controller.py QCA_AID_assets/optimization/controller.py.backup

# Backup old cache
cp QCA_AID_assets/optimization/cache.py QCA_AID_assets/optimization/cache.py.backup
```

### Step 2: Rollback Procedure

```python
# Rollback to old system
def rollback_to_old_cache_system():
    """Rollback to old cache system if needed."""
    import shutil
    
    # Restore old files
    shutil.copy(
        'QCA_AID_assets/optimization/controller.py.backup',
        'QCA_AID_assets/optimization/controller.py'
    )
    
    shutil.copy(
        'QCA_AID_assets/optimization/cache.py.backup',
        'QCA_AID_assets/optimization/cache.py'
    )
    
    print("Rolled back to old cache system")
    print("Note: You will lose multi-coder cache benefits")
```

### Step 3: Data Preservation

Export reliability data before rollback:

```python
# Export reliability data before rollback
manager.export_reliability_data("reliability_data_backup.json")

# Export cache dump
manager.create_cache_dump("cache_backup.json", include_values=True)

# Export performance report
manager.export_performance_report("performance_report_backup.json")
```

## Common Issues and Solutions

### Issue 1: Import Errors

**Problem:**
```python
ImportError: cannot import name 'DynamicCacheManager'
```

**Solution:**
```python
# Ensure you have the latest version
import sys
sys.path.insert(0, '/path/to/QCA_AID_assets')

# Verify installation
try:
    from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager
    print("✓ DynamicCacheManager available")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Please update to the latest version")
```

### Issue 2: Configuration Not Loading

**Problem:**
```python
# Configuration changes not taking effect
```

**Solution:**
```python
# Use hot-reload
if manager.reload_configuration():
    print("Configuration reloaded")
else:
    print("No changes detected")

# Or restart with new configuration
manager = DynamicCacheManager(
    cache,
    config_file="config.json",
    analysis_mode="inductive"
)
manager.configure_coders(new_coder_settings)
```

### Issue 3: Cache Not Working for Multi-Coder

**Problem:**
```python
# Cache hit rate is 0% for multi-coder
```

**Solution:**
```python
# Check strategy
strategy = manager.get_cache_strategy()
print(f"Current strategy: {strategy.name}")

# Verify coder configuration
print(f"Configured coders: {len(manager.coder_settings)}")

# Check operation classification
print(f"Shared operations: {strategy.shared_operations}")
print(f"Coder-specific operations: {strategy.coder_specific_operations}")

# Force consistency check
consistency_result = manager.force_consistency_check()
if consistency_result['total_issues'] > 0:
    manager.repair_cache_consistency(auto_fix=True)
```

### Issue 4: Reliability Data Not Persisting

**Problem:**
```python
# Reliability data lost after restart
```

**Solution:**
```python
# Verify database path
db_info = manager.get_reliability_database_info()
print(f"Database path: {db_info['database_path']}")
print(f"Database size: {db_info['database_size_mb']:.1f} MB")

# Check if data is being stored
summary = manager.get_reliability_summary()
print(f"Total codings: {summary['total_codings']}")

# Manually backup database
manager.backup_reliability_database("reliability_backup.db")
```

### Issue 5: Memory Usage Too High

**Problem:**
```python
# Cache using too much memory
```

**Solution:**
```python
# Check memory usage
stats = manager.get_statistics()
print(f"Memory usage: {stats.memory_usage_mb:.1f} MB")
print(f"Total entries: {stats.total_entries}")

# Perform selective clearing
cleared = manager.perform_selective_cache_clear({
    'older_than_hours': 24,
    'confidence_below': 0.5
})
print(f"Cleared {cleared} entries")

# Run memory stress test to find optimal size
result = manager.run_performance_benchmark(
    test_type='memory_stress',
    target_memory_mb=100
)
print(f"Memory efficiency: {result['benchmark_result']['metadata']['memory_efficiency']:.2f} ops/MB")
```

## Migration Checklist

Use this checklist to track your migration progress:

- [ ] **Preparation**
  - [ ] Backup old code
  - [ ] Review breaking changes
  - [ ] Update dependencies
  - [ ] Create test environment

- [ ] **Code Migration**
  - [ ] Update cache initialization
  - [ ] Update coder configuration
  - [ ] Update cache operations
  - [ ] Add manual coder support
  - [ ] Add monitoring and statistics

- [ ] **Configuration Migration**
  - [ ] Update configuration format
  - [ ] Set up reliability database path
  - [ ] Configure analysis mode
  - [ ] Set up plugin directory (if using plugins)

- [ ] **Testing**
  - [ ] Run unit tests
  - [ ] Run integration tests
  - [ ] Run performance benchmarks
  - [ ] Test manual coder integration
  - [ ] Verify reliability data collection

- [ ] **Deployment**
  - [ ] Deploy to test environment
  - [ ] Monitor performance
  - [ ] Verify cache statistics
  - [ ] Check reliability data
  - [ ] Deploy to production

- [ ] **Post-Migration**
  - [ ] Monitor system health
  - [ ] Review performance reports
  - [ ] Collect user feedback
  - [ ] Document any issues
  - [ ] Plan optimization improvements

## Getting Help

If you encounter issues during migration:

1. **Check Documentation:**
   - [API Documentation](API_DOCUMENTATION.md)
   - [Configuration Guide](CONFIGURATION_GUIDE.md)
   - [Test Mode Guide](TEST_MODE_AND_DEBUGGING_GUIDE.md)

2. **Enable Debug Logging:**
   ```python
   manager.set_debug_level('DEBUG')
   manager.start_debug_session(log_file='migration_debug.log')
   ```

3. **Export Diagnostic Information:**
   ```python
   manager.export_performance_report('migration_diagnostics.json')
   ```

4. **Run Consistency Checks:**
   ```python
   consistency_result = manager.force_consistency_check()
   if consistency_result['total_issues'] > 0:
       manager.repair_cache_consistency(auto_fix=True)
   ```

## Next Steps

After successful migration:

1. **Optimize Performance:**
   - Review [Performance Benchmarks](PERFORMANCE_BENCHMARKS.md)
   - Tune cache size and eviction policies
   - Consider custom cache strategies via plugins

2. **Enhance Monitoring:**
   - Set up regular performance reports
   - Monitor cache hit rates
   - Track reliability data collection

3. **Explore Advanced Features:**
   - Custom cache strategies via plugins
   - Test modes for reproducible testing
   - Advanced debugging capabilities

4. **Plan Future Improvements:**
   - Identify optimization opportunities
   - Consider additional cache strategies
   - Enhance reliability analysis capabilities