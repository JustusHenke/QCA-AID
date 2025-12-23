"""
Dynamic Cache Manager
====================
Central manager for intelligent cache strategy selection and management.
Automatically switches between single-coder and multi-coder strategies based on configuration.
Supports plugin-based strategy extensions for custom caching behaviors.
Enhanced with test modes and debugging features.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from .cache import ModeAwareCache, CacheStatistics
from .cache_strategies import CacheStrategy, CacheStrategyFactory, SingleCoderCacheStrategy, MultiCoderCacheStrategy, ModeSpecificCacheStrategy
from .reliability_database import ReliabilityDatabase
from ..core.data_models import ExtendedCodingResult

logger = logging.getLogger(__name__)


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
    
    def __init__(self, base_cache: ModeAwareCache, config_file: Optional[str] = None, 
                 reliability_db_path: Optional[str] = None, analysis_mode: Optional[str] = None,
                 enable_plugins: bool = True, plugins_directory: Optional[str] = None,
                 test_mode: bool = False, debug_level: str = "INFO"):
        """
        Initialize dynamic cache manager.
        
        Args:
            base_cache: Base ModeAwareCache instance to manage
            config_file: Optional path to configuration file for hot-reload
            reliability_db_path: Optional path to reliability database file 
                               (default: output/all_codings.json, relative to configured project root)
            analysis_mode: Optional analysis mode for mode-specific caching strategies
            enable_plugins: Whether to enable plugin system
            plugins_directory: Optional directory to load plugins from
            test_mode: Whether to enable test mode for deterministic behavior
            debug_level: Debug logging level (SILENT, ERROR, WARNING, INFO, DEBUG, TRACE)
        """
        self.base_cache = base_cache
        self.config_file = config_file
        self.analysis_mode = analysis_mode
        self.enable_plugins = enable_plugins
        
        # Test mode and debugging setup
        self.test_mode = test_mode
        self.debug_level = debug_level
        self._setup_debugging()
        
        # Current configuration
        self.coder_settings: List[Dict[str, Any]] = []
        self.current_strategy: Optional[CacheStrategy] = None
        self.last_config_hash: Optional[str] = None
        
        # Plugin system setup
        self.plugin_registry: Optional[CacheStrategyRegistry] = None
        if enable_plugins:
            self._initialize_plugin_system(plugins_directory)
        
        # Reliability database for intercoder analysis
        self.reliability_data: Dict[str, List[ExtendedCodingResult]] = {}
        self.reliability_db = ReliabilityDatabase(reliability_db_path or "output/all_codings.json")
        
        # Initialize with default strategy
        self._update_strategy()
        
        logger.info(f"DynamicCacheManager initialized for {analysis_mode or 'default'} mode" +
                   (f" with plugins {'enabled' if enable_plugins else 'disabled'}" +
                    f", test_mode={'enabled' if test_mode else 'disabled'}"))
    
    def _setup_debugging(self) -> None:
        """Set up debugging and test mode capabilities."""
        try:
            from .cache_debug_tools import LogLevel, TestModeManager, CacheDebugLogger
            
            # Set up debug logger
            log_level_map = {
                'SILENT': LogLevel.SILENT,
                'ERROR': LogLevel.ERROR,
                'WARNING': LogLevel.WARNING,
                'INFO': LogLevel.INFO,
                'DEBUG': LogLevel.DEBUG,
                'TRACE': LogLevel.TRACE
            }
            
            self.debug_logger = CacheDebugLogger(
                level=log_level_map.get(self.debug_level.upper(), LogLevel.INFO)
            )
            
            # Set up test mode manager
            self.test_manager = TestModeManager()
            if self.test_mode:
                self.test_manager.enable_deterministic_mode()
            
            logger.debug(f"Debugging setup completed: level={self.debug_level}, test_mode={self.test_mode}")
            
        except ImportError as e:
            logger.warning(f"Debug tools not available: {e}")
            self.debug_logger = None
            self.test_manager = None
    def _initialize_plugin_system(self, plugins_directory: Optional[str] = None) -> None:
        """
        Initialize the plugin system and load available plugins.
        
        Args:
            plugins_directory: Optional directory to load plugins from
        """
        try:
            # Lazy import to avoid circular imports
            from .cache_plugin_interface import CacheStrategyRegistry, get_global_registry
            
            # Get or create plugin registry
            self.plugin_registry = get_global_registry()
            
            # Set registry in factory for strategy creation
            CacheStrategyFactory.set_plugin_registry(self.plugin_registry)
            
            # Load plugins from directory if specified
            if plugins_directory:
                results = CacheStrategyFactory.load_plugins_from_directory(plugins_directory)
                loaded_count = sum(1 for success in results.values() if success)
                logger.info(f"Loaded {loaded_count}/{len(results)} plugins from {plugins_directory}")
            else:
                # Try to load from default plugins directory
                import os
                default_plugins_dir = os.path.join(os.path.dirname(__file__), 'plugins')
                if os.path.exists(default_plugins_dir):
                    results = CacheStrategyFactory.load_plugins_from_directory(default_plugins_dir)
                    loaded_count = sum(1 for success in results.values() if success)
                    if loaded_count > 0:
                        logger.info(f"Loaded {loaded_count} plugins from default directory")
            
            # Log available strategies
            available_strategies = CacheStrategyFactory.get_available_strategies()
            logger.info(f"Available cache strategies: {list(available_strategies.keys())}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize plugin system: {e}")
            self.plugin_registry = None
            self.enable_plugins = False
    
    def configure_analysis_mode(self, analysis_mode: str) -> None:
        """
        Configure analysis mode and update cache strategy if needed.
        
        Args:
            analysis_mode: Analysis mode (inductive, abductive, grounded, deductive)
        """
        old_mode = self.analysis_mode
        self.analysis_mode = analysis_mode.lower() if analysis_mode else None
        
        if old_mode != self.analysis_mode:
            logger.info(f"Analysis mode changed: {old_mode} -> {self.analysis_mode}")
            
            # Update strategy to reflect new analysis mode
            self._update_strategy()
            
            # Perform cache migration/invalidation for mode change
            if old_mode and self.analysis_mode:
                self._handle_mode_change(old_mode, self.analysis_mode)
    
    def _handle_mode_change(self, old_mode: str, new_mode: str) -> None:
        """
        Handle cache migration/invalidation when analysis mode changes.
        
        Args:
            old_mode: Previous analysis mode
            new_mode: New analysis mode
        """
        logger.info(f"Handling analysis mode change: {old_mode} -> {new_mode}")
        
        # Different modes have different methodological requirements
        # Some operations may need to be invalidated when switching modes
        
        mode_specific_invalidations = {
            ('deductive', 'inductive'): ['category_development'],
            ('deductive', 'abductive'): ['subcategory_development'],
            ('deductive', 'grounded'): ['category_development', 'coding'],
            ('inductive', 'deductive'): ['relevance_check'],
            ('inductive', 'abductive'): ['subcategory_development'],
            ('inductive', 'grounded'): ['coding'],
            ('abductive', 'deductive'): ['relevance_check'],
            ('abductive', 'inductive'): ['category_development'],
            ('abductive', 'grounded'): ['coding'],
            ('grounded', 'deductive'): ['relevance_check', 'coding'],
            ('grounded', 'inductive'): ['category_development', 'coding'],
            ('grounded', 'abductive'): ['subcategory_development', 'coding']
        }
        
        operations_to_invalidate = mode_specific_invalidations.get((old_mode, new_mode), [])
        
        if operations_to_invalidate:
            self._invalidate_operations(operations_to_invalidate, f"mode_change:{old_mode}->{new_mode}")
            logger.info(f"Invalidated {len(operations_to_invalidate)} operation types due to mode change")
    
    def _invalidate_operations(self, operations: List[str], reason: str) -> None:
        """
        Invalidate cache entries for specific operations.
        
        Args:
            operations: List of operation types to invalidate
            reason: Reason for invalidation
        """
        keys_to_remove = []
        
        for operation in operations:
            if operation in self.base_cache.operation_index:
                keys_to_remove.extend(self.base_cache.operation_index[operation])
        
        # Remove duplicates
        keys_to_remove = list(set(keys_to_remove))
        
        for key in keys_to_remove:
            if key in self.base_cache.cache:
                entry = self.base_cache.cache[key]
                self.base_cache._remove_entry_from_indexes(key, entry)
                del self.base_cache.cache[key]
                self.base_cache.stats["invalidations"] += 1
        
        self.base_cache.stats["total_entries"] = len(self.base_cache.cache)
        
        if keys_to_remove:
            self.base_cache._notify_cache_invalidate(keys_to_remove, reason)
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries for operations: {operations}")
    
    def get_mode_specific_cache_rules(self) -> Dict[str, Any]:
        """
        Get current mode-specific cache rules and operation classifications.
        
        Returns:
            Dictionary with mode-specific cache information
        """
        if not self.current_strategy or not hasattr(self.current_strategy, 'analysis_mode'):
            return {
                'analysis_mode': self.analysis_mode,
                'strategy_type': 'non_mode_specific',
                'shared_operations': [],
                'coder_specific_operations': [],
                'methodological_principles': {}
            }
        
        strategy_info = self.current_strategy.get_strategy_info()
        
        return {
            'analysis_mode': self.analysis_mode,
            'strategy_type': strategy_info.get('type', 'unknown'),
            'strategy_name': strategy_info.get('name', 'unknown'),
            'shared_operations': strategy_info.get('shared_operations', []),
            'coder_specific_operations': strategy_info.get('coder_specific_operations', []),
            'methodological_principles': strategy_info.get('methodological_principles', {}),
            'supports_multi_coder': strategy_info.get('supports_multi_coder', False)
        }
    
    def configure_coders(self, coder_settings: List[Dict[str, Any]]) -> None:
        """
        Configure coders and update cache strategy if needed.
        
        Args:
            coder_settings: List of coder configurations with 'coder_id' and 'temperature'
        """
        old_coder_count = len(self.coder_settings)
        old_strategy_name = self.current_strategy.name if self.current_strategy else "None"
        
        # Update coder settings
        self.coder_settings = coder_settings.copy() if coder_settings else []
        
        # Calculate configuration hash for change detection
        config_hash = self._calculate_config_hash()
        config_changed = config_hash != self.last_config_hash
        self.last_config_hash = config_hash
        
        new_coder_count = len(self.coder_settings)
        
        logger.info(f"Coder configuration updated: {old_coder_count} -> {new_coder_count} coders")
        
        # Update strategy if coder count changed or configuration changed
        if old_coder_count != new_coder_count or config_changed:
            self._update_strategy()
            
            new_strategy_name = self.current_strategy.name if self.current_strategy else "None"
            
            if old_strategy_name != new_strategy_name:
                logger.info(f"Cache strategy changed: {old_strategy_name} -> {new_strategy_name}")
                
                # Perform cache migration/invalidation if needed
                self._handle_strategy_change(old_strategy_name, new_strategy_name)
        """
        Configure coders and update cache strategy if needed.
        
        Args:
            coder_settings: List of coder configurations with 'coder_id' and 'temperature'
        """
        old_coder_count = len(self.coder_settings)
        old_strategy_name = self.current_strategy.name if self.current_strategy else "None"
        
        # Update coder settings
        self.coder_settings = coder_settings.copy() if coder_settings else []
        
        # Calculate configuration hash for change detection
        config_hash = self._calculate_config_hash()
        config_changed = config_hash != self.last_config_hash
        self.last_config_hash = config_hash
        
        new_coder_count = len(self.coder_settings)
        
        logger.info(f"Coder configuration updated: {old_coder_count} -> {new_coder_count} coders")
        
        # Update strategy if coder count changed or configuration changed
        if old_coder_count != new_coder_count or config_changed:
            self._update_strategy()
            
            new_strategy_name = self.current_strategy.name if self.current_strategy else "None"
            
            if old_strategy_name != new_strategy_name:
                logger.info(f"Cache strategy changed: {old_strategy_name} -> {new_strategy_name}")
                
                # Perform cache migration/invalidation if needed
                self._handle_strategy_change(old_strategy_name, new_strategy_name)
    
    def _calculate_config_hash(self) -> str:
        """
        Calculate hash of current configuration for change detection.
        
        Returns:
            Configuration hash string
        """
        import hashlib
        
        # Create hashable representation of configuration
        config_data = {
            'coder_count': len(self.coder_settings),
            'coder_ids': sorted([cs.get('coder_id', '') for cs in self.coder_settings]),
            'temperatures': sorted([cs.get('temperature', 0.0) for cs in self.coder_settings])
        }
        
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]
    
    def _update_strategy(self) -> None:
        """Update cache strategy based on current coder configuration and analysis mode."""
        coder_count = len(self.coder_settings)
        
        # Create appropriate strategy with analysis mode support
        new_strategy = CacheStrategyFactory.create_strategy(
            coder_count=coder_count,
            analysis_mode=self.analysis_mode
        )
        
        # Set strategy on base cache
        self.base_cache.set_strategy(new_strategy)
        self.current_strategy = new_strategy
        
        logger.info(f"Cache strategy updated to: {new_strategy.name} for {coder_count} coder(s)" + 
                   (f" in {self.analysis_mode} mode" if self.analysis_mode else ""))
    
    def _handle_strategy_change(self, old_strategy: str, new_strategy: str) -> None:
        """
        Handle cache migration/invalidation when strategy changes.
        
        Args:
            old_strategy: Previous strategy name
            new_strategy: New strategy name
        """
        logger.info(f"Handling strategy change: {old_strategy} -> {new_strategy}")
        
        # Perform cache migration with rollback support
        migration_success = self._migrate_cache_entries(old_strategy, new_strategy)
        
        if not migration_success:
            logger.warning("Cache migration failed, performing rollback")
            self._rollback_migration(old_strategy)
            return
        
        # Perform consistency checks after migration
        consistency_issues = self._check_cache_consistency()
        if consistency_issues:
            logger.warning(f"Cache consistency issues detected: {consistency_issues}")
            self._fix_consistency_issues(consistency_issues)
        
        # Log strategy change for monitoring
        operations_invalidated = self.current_strategy.invalidate_on_config_change()
        self._log_strategy_change(old_strategy, new_strategy, operations_invalidated)
    
    def _log_strategy_change(self, old_strategy: str, new_strategy: str, invalidated_ops: List[str]) -> None:
        """
        Log strategy change for debugging and monitoring.
        
        Args:
            old_strategy: Previous strategy name
            new_strategy: New strategy name
            invalidated_ops: List of invalidated operations
        """
        change_info = {
            'timestamp': datetime.now().isoformat(),
            'old_strategy': old_strategy,
            'new_strategy': new_strategy,
            'coder_count': len(self.coder_settings),
            'invalidated_operations': invalidated_ops,
            'cache_stats_before': self.base_cache.get_statistics().total_entries
        }
        
        logger.info(f"Strategy change logged: {json.dumps(change_info, indent=2)}")
    
    def get_cache_strategy(self) -> CacheStrategy:
        """
        Get current cache strategy.
        
        Returns:
            Current cache strategy instance
        """
        return self.current_strategy
    
    def get_shared_cache_key(self, operation: str, **params) -> str:
        """
        Generate cache key for shared operations (used across all coders).
        
        Args:
            operation: Operation type (e.g., 'relevance_check', 'category_development')
            **params: Additional parameters for key generation
            
        Returns:
            Cache key string for shared operation
        """
        if not self.current_strategy.should_cache_shared(operation):
            logger.warning(f"Operation '{operation}' is not configured as shared but shared key requested")
        
        return self.current_strategy.get_cache_key(operation, coder_id=None, **params)
    
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
        if not self.current_strategy.should_cache_per_coder(operation):
            logger.warning(f"Operation '{operation}' is not configured as coder-specific but coder-specific key requested")
        
        return self.current_strategy.get_cache_key(operation, coder_id=coder_id, **params)
    
    def should_cache_shared(self, operation: str) -> bool:
        """
        Check if operation should use shared caching.
        
        Args:
            operation: Operation type
            
        Returns:
            True if operation should be cached as shared
        """
        return self.current_strategy.should_cache_shared(operation)
    
    def should_cache_per_coder(self, operation: str) -> bool:
        """
        Check if operation should use per-coder caching.
        
        Args:
            operation: Operation type
            
        Returns:
            True if operation should be cached per coder
        """
        return self.current_strategy.should_cache_per_coder(operation)
    
    def store_for_reliability(self, coding_result: ExtendedCodingResult) -> None:
        """
        Store coding result for intercoder reliability analysis.
        
        Args:
            coding_result: Extended coding result with metadata
        """
        # print(f"   🔍 DEBUG: store_for_reliability called for segment {coding_result.segment_id}, coder {coding_result.coder_id}")
        # print(f"   🔍 DEBUG: Category: {coding_result.category}, is_relevant: {coding_result.metadata.get('is_relevant', True) if coding_result.metadata else True}")
        
        # Store in memory for quick access
        segment_id = coding_result.segment_id
        
        if segment_id not in self.reliability_data:
            self.reliability_data[segment_id] = []
        
        # Check if we already have a result from this coder for this segment
        existing_idx = None
        for i, existing_result in enumerate(self.reliability_data[segment_id]):
            if existing_result.coder_id == coding_result.coder_id:
                existing_idx = i
                break
        
        if existing_idx is not None:
            # Update existing result
            self.reliability_data[segment_id][existing_idx] = coding_result
            logger.debug(f"Updated reliability data for segment {segment_id}, coder {coding_result.coder_id}")
            # print(f"   ✅ DEBUG: Updated in-memory reliability data for {segment_id}")
        else:
            # Add new result
            self.reliability_data[segment_id].append(coding_result)
            logger.debug(f"Added reliability data for segment {segment_id}, coder {coding_result.coder_id}")
            # print(f"   ✅ DEBUG: Added to in-memory reliability data for {segment_id}")
        
        # Store in persistent database
        try:
            # print(f"   💾 DEBUG: Storing to persistent database...")
            self.reliability_db.store_coding_result(coding_result)
            # print(f"   ✅ DEBUG: Successfully stored to persistent database")
        except Exception as e:
            # print(f"   ❌ DEBUG: Failed to store to persistent database: {e}")
            import traceback
            traceback.print_exc()
        
        # Mark manual coders with "manual" ID if needed
        if coding_result.is_manual and coding_result.coder_id != "manual":
            # Create a copy with manual ID for consistency
            manual_result = ExtendedCodingResult(
                segment_id=coding_result.segment_id,
                coder_id="manual",
                category=coding_result.category,
                subcategories=coding_result.subcategories,
                confidence=coding_result.confidence,
                justification=coding_result.justification,
                analysis_mode=coding_result.analysis_mode,
                timestamp=coding_result.timestamp,
                is_manual=True,
                metadata={**coding_result.metadata, 'original_coder_id': coding_result.coder_id}
            )
            self.reliability_db.store_coding_result(manual_result)
    
    def get_reliability_data(self, segment_ids: Optional[List[str]] = None, exclude_non_relevant: bool = True) -> List[ExtendedCodingResult]:
        """
        Get reliability data for intercoder analysis.
        
        Args:
            segment_ids: Optional list of segment IDs to filter by
            exclude_non_relevant: Whether to exclude non-relevant segments from reliability calculation (default: True)
            
        Returns:
            List of coding results for reliability analysis (excluding non-relevant segments by default)
        """
        # Try to get from persistent database first
        try:
            db_results = self.reliability_db.get_coding_results(segment_ids=segment_ids)
            if db_results:
                # FIX: Filter out non-relevant segments if requested
                if exclude_non_relevant:
                    filtered_results = []
                    for result in db_results:
                        # Check metadata for exclusion flag
                        metadata = result.metadata or {}
                        exclude_from_reliability = metadata.get('exclude_from_reliability', False)
                        is_relevant = metadata.get('is_relevant', True)
                        
                        # Exclude if marked as non-relevant or explicitly excluded
                        if not exclude_from_reliability and is_relevant:
                            filtered_results.append(result)
                        else:
                            logger.debug(f"Excluding segment {result.segment_id} from reliability (is_relevant={is_relevant}, exclude_from_reliability={exclude_from_reliability})")
                    
                    print(f"   🔍 Reliability Filter: {len(db_results)} total → {len(filtered_results)} relevant für Reliabilität")
                    return filtered_results
                else:
                    return db_results
        except Exception as e:
            logger.warning(f"Failed to get reliability data from database: {e}")
        
        # Fallback to in-memory data
        if segment_ids is None:
            # Return all reliability data
            all_results = []
            for segment_results in self.reliability_data.values():
                all_results.extend(segment_results)
        else:
            # Return data for specific segments
            all_results = []
            for segment_id in segment_ids:
                if segment_id in self.reliability_data:
                    all_results.extend(self.reliability_data[segment_id])
        
        # FIX: Filter out non-relevant segments from in-memory data if requested
        if exclude_non_relevant:
            filtered_results = []
            for result in all_results:
                metadata = result.metadata or {}
                exclude_from_reliability = metadata.get('exclude_from_reliability', False)
                is_relevant = metadata.get('is_relevant', True)
                
                if not exclude_from_reliability and is_relevant:
                    filtered_results.append(result)
            
            return filtered_results
        else:
            return all_results
    
    def get_all_coding_data(self, segment_ids: Optional[List[str]] = None) -> List[ExtendedCodingResult]:
        """
        Get ALL coding data including non-relevant segments.
        
        Args:
            segment_ids: Optional list of segment IDs to filter by
            
        Returns:
            List of all coding results (including non-relevant segments)
        """
        return self.get_reliability_data(segment_ids=segment_ids, exclude_non_relevant=False)
    
    def get_reliability_summary(self) -> Dict[str, Any]:
        """
        Get summary of reliability data for monitoring.
        
        Returns:
            Dictionary with reliability data summary including manual coder statistics
        """
        # Try to get from persistent database first
        try:
            db_summary = self.reliability_db.get_reliability_summary()
            if db_summary['total_codings'] > 0:
                # Enhance with manual coder breakdown
                manual_codings = db_summary.get('manual_codings', 0)
                automatic_codings = db_summary.get('automatic_codings', 0)
                
                # Add manual coder specific statistics
                db_summary['manual_coder_stats'] = {
                    'total_manual_codings': manual_codings,
                    'total_automatic_codings': automatic_codings,
                    'manual_percentage': (manual_codings / db_summary['total_codings'] * 100) if db_summary['total_codings'] > 0 else 0,
                    'manual_coders': [coder_id for coder_id in db_summary.get('codings_per_coder', {}).keys() 
                                     if coder_id == 'manual' or 'human' in coder_id.lower() or 'manual' in coder_id.lower()]
                }
                
                return db_summary
        except Exception as e:
            logger.warning(f"Failed to get reliability summary from database: {e}")
        
        # Fallback to in-memory data
        total_segments = len(self.reliability_data)
        total_codings = sum(len(results) for results in self.reliability_data.values())
        
        # Count codings per coder and separate manual/automatic
        coder_counts = {}
        analysis_modes = set()
        manual_codings = 0
        automatic_codings = 0
        manual_coders = []
        
        for segment_results in self.reliability_data.values():
            for result in segment_results:
                coder_id = result.coder_id
                coder_counts[coder_id] = coder_counts.get(coder_id, 0) + 1
                analysis_modes.add(result.analysis_mode)
                
                # Track manual vs automatic
                if result.is_manual:
                    manual_codings += 1
                    if coder_id not in manual_coders:
                        manual_coders.append(coder_id)
                else:
                    automatic_codings += 1
        
        return {
            'total_segments': total_segments,
            'total_codings': total_codings,
            'manual_codings': manual_codings,
            'automatic_codings': automatic_codings,
            'codings_per_coder': coder_counts,
            'analysis_modes': list(analysis_modes),
            'segments_with_multiple_coders': sum(1 for results in self.reliability_data.values() if len(results) > 1),
            'manual_coder_stats': {
                'total_manual_codings': manual_codings,
                'total_automatic_codings': automatic_codings,
                'manual_percentage': (manual_codings / total_codings * 100) if total_codings > 0 else 0,
                'manual_coders': manual_coders
            }
        }
    
    def reload_configuration(self) -> bool:
        """
        Reload configuration from file if available.
        
        Returns:
            True if configuration was reloaded, False if no config file or no changes
        """
        if not self.config_file:
            return False
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Extract coder settings from configuration
            new_coder_settings = config_data.get('coder_settings', [])
            
            # Calculate new config hash
            old_hash = self.last_config_hash
            self.coder_settings = new_coder_settings
            new_hash = self._calculate_config_hash()
            
            if old_hash != new_hash:
                logger.info(f"Configuration reloaded from {self.config_file}")
                self.configure_coders(new_coder_settings)
                return True
            else:
                logger.debug("Configuration file checked, no changes detected")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to reload configuration from {self.config_file}: {e}")
            return False
    
    def get_statistics(self) -> CacheStatistics:
        """
        Get comprehensive cache statistics including strategy information and manual coder isolation.
        
        Returns:
            Enhanced cache statistics with strategy info and manual coder separation
        """
        base_stats = self.base_cache.get_statistics()
        
        # Add strategy information
        base_stats.strategy_type = self.current_strategy.name if self.current_strategy else "Unknown"
        
        # Add manual coder isolation statistics
        manual_coder_stats = self._get_manual_coder_cache_stats()
        base_stats.manual_coder_stats = manual_coder_stats
        
        return base_stats
    
    def _get_manual_coder_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics specifically for manual coder operations.
        
        Returns:
            Dictionary with manual coder cache statistics
        """
        manual_stats = {
            'manual_cache_entries': 0,
            'manual_hit_rate': 0.0,
            'manual_operations': {},
            'manual_memory_usage_mb': 0.0,
            'isolated_from_automatic': True
        }
        
        try:
            # Count manual coder cache entries
            manual_entries = 0
            manual_memory = 0
            manual_operations = {}
            
            for key, entry in self.base_cache.cache.items():
                # Check if entry is from manual coder
                if (entry.coder_id == "manual" or 
                    (entry.coder_id and ('manual' in entry.coder_id.lower() or 'human' in entry.coder_id.lower())) or
                    entry.metadata.get('source') == 'manual_gui'):
                    
                    manual_entries += 1
                    
                    # Estimate memory usage (rough calculation)
                    entry_size = len(str(entry.value)) + len(str(entry.key)) + 200  # overhead
                    manual_memory += entry_size
                    
                    # Track operations
                    op_type = entry.operation_type
                    if op_type not in manual_operations:
                        manual_operations[op_type] = {'count': 0, 'hit_rate': 0.0}
                    manual_operations[op_type]['count'] += 1
            
            # Calculate manual hit rate from base cache stats
            manual_hit_rate = 0.0
            if 'manual' in self.base_cache.stats.get('hit_rate_by_coder', {}):
                manual_hit_rate = self.base_cache.stats['hit_rate_by_coder']['manual']
            
            manual_stats.update({
                'manual_cache_entries': manual_entries,
                'manual_hit_rate': manual_hit_rate,
                'manual_operations': manual_operations,
                'manual_memory_usage_mb': manual_memory / (1024 * 1024),
                'isolated_from_automatic': True  # Manual coders are always isolated
            })
            
        except Exception as e:
            logger.warning(f"Failed to calculate manual coder cache stats: {e}")
            manual_stats['error'] = str(e)
        
        return manual_stats
    
    def export_performance_report(self, filepath: str) -> None:
        """
        Export comprehensive performance report including cache and reliability data.
        
        Args:
            filepath: Path to export file
        """
        try:
            # Get cache statistics
            cache_stats = self.get_statistics()
            
            # Get reliability summary
            reliability_summary = self.get_reliability_summary()
            
            # Get manager info
            manager_info = self.get_manager_info()
            
            # Get error report from base cache
            error_report = self.base_cache.get_error_report()
            
            performance_report = {
                'timestamp': datetime.now().isoformat(),
                'report_type': 'dynamic_cache_performance',
                'manager_info': manager_info,
                'cache_statistics': {
                    'total_entries': cache_stats.total_entries,
                    'shared_entries': cache_stats.shared_entries,
                    'coder_specific_entries': cache_stats.coder_specific_entries,
                    'hit_rate_overall': cache_stats.hit_rate_overall,
                    'hit_rate_by_coder': cache_stats.hit_rate_by_coder,
                    'memory_usage_mb': cache_stats.memory_usage_mb,
                    'memory_usage_by_type': cache_stats.memory_usage_by_type,
                    'cache_efficiency_by_coder': cache_stats.cache_efficiency_by_coder,
                    'hits_by_operation': cache_stats.hits_by_operation,
                    'misses_by_operation': cache_stats.misses_by_operation,
                    'strategy_type': cache_stats.strategy_type
                },
                'reliability_data': reliability_summary,
                'error_analysis': error_report,
                'performance_insights': self._generate_performance_insights(cache_stats, reliability_summary)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(performance_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Performance report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export performance report: {e}")
            raise
    
    def _generate_performance_insights(self, cache_stats: CacheStatistics, reliability_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate performance insights from statistics.
        
        Args:
            cache_stats: Cache statistics
            reliability_summary: Reliability data summary
            
        Returns:
            Dictionary with performance insights
        """
        insights = {
            'cache_efficiency': 'unknown',
            'memory_efficiency': 'unknown',
            'error_level': 'unknown',
            'recommendations': [],
            'alerts': []
        }
        
        try:
            # Analyze cache efficiency
            if cache_stats.hit_rate_overall >= 0.8:
                insights['cache_efficiency'] = 'excellent'
            elif cache_stats.hit_rate_overall >= 0.6:
                insights['cache_efficiency'] = 'good'
            elif cache_stats.hit_rate_overall >= 0.4:
                insights['cache_efficiency'] = 'fair'
            else:
                insights['cache_efficiency'] = 'poor'
                insights['recommendations'].append("Consider reviewing cache strategy or increasing cache size")
            
            # Analyze memory efficiency
            if cache_stats.memory_usage_mb < 50:
                insights['memory_efficiency'] = 'excellent'
            elif cache_stats.memory_usage_mb < 100:
                insights['memory_efficiency'] = 'good'
            elif cache_stats.memory_usage_mb < 200:
                insights['memory_efficiency'] = 'fair'
            else:
                insights['memory_efficiency'] = 'poor'
                insights['recommendations'].append("Consider reducing cache size or implementing more aggressive eviction")
            
            # Analyze error level
            error_rate = cache_stats.performance_metrics.get('error_rate', 0)
            if error_rate < 0.01:
                insights['error_level'] = 'low'
            elif error_rate < 0.05:
                insights['error_level'] = 'moderate'
            else:
                insights['error_level'] = 'high'
                insights['alerts'].append(f"High error rate detected: {error_rate:.2%}")
            
            # Check for coder-specific issues
            for coder_id, efficiency in cache_stats.cache_efficiency_by_coder.items():
                if efficiency['hit_rate'] < 0.3:
                    insights['alerts'].append(f"Low hit rate for coder {coder_id}: {efficiency['hit_rate']:.2%}")
            
            # Check reliability data coverage
            if reliability_summary['total_codings'] > 0:
                coders_with_data = len(reliability_summary['codings_per_coder'])
                if coders_with_data < 2:
                    insights['recommendations'].append("Consider adding more coders for reliability analysis")
                elif reliability_summary['segments_with_multiple_coders'] < reliability_summary['total_segments'] * 0.5:
                    insights['recommendations'].append("Increase multi-coder coverage for better reliability analysis")
            
            # Memory usage recommendations
            memory_by_type = cache_stats.memory_usage_by_type
            if memory_by_type.get('shared_mb', 0) > memory_by_type.get('coder_specific_mb', 0) * 2:
                insights['recommendations'].append("Consider optimizing shared cache entries for memory efficiency")
            
        except Exception as e:
            logger.warning(f"Failed to generate performance insights: {e}")
            insights['error'] = str(e)
        
        return insights
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data formatted for monitoring dashboard display including manual coder statistics.
        
        Returns:
            Dictionary with dashboard-ready monitoring data including manual coder info
        """
        try:
            cache_stats = self.get_statistics()
            reliability_summary = self.get_reliability_summary()
            manager_info = self.get_manager_info()
            
            # Get manual coder specific statistics
            manual_coder_stats = getattr(cache_stats, 'manual_coder_stats', {})
            
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'status': 'healthy',  # Will be updated based on analysis
                'key_metrics': {
                    'cache_hit_rate': f"{cache_stats.hit_rate_overall:.1%}",
                    'total_entries': cache_stats.total_entries,
                    'memory_usage': f"{cache_stats.memory_usage_mb:.1f} MB",
                    'active_coders': len(cache_stats.hit_rate_by_coder),
                    'strategy': cache_stats.strategy_type,
                    'reliability_segments': reliability_summary['total_segments'],
                    'manual_coders': len(reliability_summary.get('manual_coder_stats', {}).get('manual_coders', [])),
                    'manual_codings': reliability_summary.get('manual_coder_stats', {}).get('total_manual_codings', 0)
                },
                'performance_indicators': {
                    'cache_efficiency': self._get_efficiency_indicator(cache_stats.hit_rate_overall),
                    'memory_efficiency': self._get_memory_indicator(cache_stats.memory_usage_mb),
                    'error_rate': cache_stats.performance_metrics.get('error_rate', 0),
                    'coder_balance': self._get_coder_balance_indicator(cache_stats.cache_efficiency_by_coder),
                    'manual_coder_integration': self._get_manual_coder_integration_indicator(manual_coder_stats, reliability_summary)
                },
                'recent_activity': {
                    'hits_by_operation': dict(list(cache_stats.hits_by_operation.items())[:5]),
                    'top_coders_by_activity': self._get_top_coders_by_activity(cache_stats.cache_efficiency_by_coder),
                    'recent_errors': cache_stats.error_counts
                },
                'manual_coder_info': {
                    'cache_isolation_verified': manual_coder_stats.get('isolated_from_automatic', False),
                    'manual_cache_entries': manual_coder_stats.get('manual_cache_entries', 0),
                    'manual_hit_rate': f"{manual_coder_stats.get('manual_hit_rate', 0):.1%}",
                    'manual_memory_usage': f"{manual_coder_stats.get('manual_memory_usage_mb', 0):.1f} MB",
                    'manual_percentage': f"{reliability_summary.get('manual_coder_stats', {}).get('manual_percentage', 0):.1f}%"
                },
                'alerts': self._generate_dashboard_alerts(cache_stats, reliability_summary)
            }
            
            # Determine overall status including manual coder considerations
            error_rate = cache_stats.performance_metrics.get('error_rate', 0)
            manual_integration_ok = manual_coder_stats.get('isolated_from_automatic', True)
            
            if error_rate > 0.1 or cache_stats.hit_rate_overall < 0.2 or not manual_integration_ok:
                dashboard_data['status'] = 'critical'
            elif error_rate > 0.05 or cache_stats.hit_rate_overall < 0.4:
                dashboard_data['status'] = 'warning'
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard data: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e),
                'key_metrics': {},
                'performance_indicators': {},
                'recent_activity': {},
                'manual_coder_info': {},
                'alerts': [f"Dashboard data generation failed: {str(e)}"]
            }
    
    def _get_manual_coder_integration_indicator(self, manual_stats: Dict[str, Any], 
                                              reliability_summary: Dict[str, Any]) -> str:
        """
        Get manual coder integration health indicator.
        
        Args:
            manual_stats: Manual coder cache statistics
            reliability_summary: Reliability data summary
            
        Returns:
            Integration health indicator string
        """
        try:
            # Check cache isolation
            if not manual_stats.get('isolated_from_automatic', True):
                return "isolation_violation"
            
            # Check if manual coders are active
            manual_codings = reliability_summary.get('manual_coder_stats', {}).get('total_manual_codings', 0)
            total_codings = reliability_summary.get('total_codings', 0)
            
            if manual_codings == 0:
                return "no_manual_activity"
            elif total_codings > 0:
                manual_percentage = manual_codings / total_codings
                if manual_percentage > 0.3:
                    return "high_manual_activity"
                elif manual_percentage > 0.1:
                    return "moderate_manual_activity"
                else:
                    return "low_manual_activity"
            
            return "unknown"
            
        except Exception as e:
            logger.warning(f"Failed to calculate manual coder integration indicator: {e}")
            return "error"
    
    def _get_efficiency_indicator(self, hit_rate: float) -> str:
        """Get efficiency indicator based on hit rate."""
        if hit_rate >= 0.8:
            return "excellent"
        elif hit_rate >= 0.6:
            return "good"
        elif hit_rate >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _get_memory_indicator(self, memory_mb: float) -> str:
        """Get memory efficiency indicator."""
        if memory_mb < 50:
            return "excellent"
        elif memory_mb < 100:
            return "good"
        elif memory_mb < 200:
            return "fair"
        else:
            return "poor"
    
    def _get_coder_balance_indicator(self, coder_efficiency: Dict[str, Dict[str, float]]) -> str:
        """Get coder balance indicator."""
        if not coder_efficiency:
            return "no_data"
        
        hit_rates = [eff['hit_rate'] for eff in coder_efficiency.values()]
        if not hit_rates:
            return "no_data"
        
        min_rate = min(hit_rates)
        max_rate = max(hit_rates)
        
        if max_rate - min_rate < 0.2:
            return "balanced"
        elif max_rate - min_rate < 0.4:
            return "moderate_imbalance"
        else:
            return "high_imbalance"
    
    def _get_top_coders_by_activity(self, coder_efficiency: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Get top coders by activity level."""
        coders = []
        for coder_id, efficiency in coder_efficiency.items():
            coders.append({
                'coder_id': coder_id,
                'total_requests': efficiency['total_requests'],
                'hit_rate': efficiency['hit_rate'],
                'efficiency_score': efficiency['efficiency_score']
            })
        
        # Sort by total requests (activity level)
        coders.sort(key=lambda x: x['total_requests'], reverse=True)
        return coders[:5]
    
    def _generate_dashboard_alerts(self, cache_stats: CacheStatistics, reliability_summary: Dict[str, Any]) -> List[str]:
        """Generate alerts for dashboard."""
        alerts = []
        
        # Cache performance alerts
        if cache_stats.hit_rate_overall < 0.3:
            alerts.append(f"Low cache hit rate: {cache_stats.hit_rate_overall:.1%}")
        
        # Memory alerts
        if cache_stats.memory_usage_mb > 200:
            alerts.append(f"High memory usage: {cache_stats.memory_usage_mb:.1f} MB")
        
        # Error alerts
        error_rate = cache_stats.performance_metrics.get('error_rate', 0)
        if error_rate > 0.05:
            alerts.append(f"High error rate: {error_rate:.2%}")
        
        # Coder-specific alerts
        for coder_id, efficiency in cache_stats.cache_efficiency_by_coder.items():
            if efficiency['hit_rate'] < 0.2:
                alerts.append(f"Very low hit rate for coder {coder_id}: {efficiency['hit_rate']:.1%}")
        
        # Reliability alerts
        if reliability_summary['total_segments'] > 0 and reliability_summary['segments_with_multiple_coders'] == 0:
            alerts.append("No segments with multiple coders for reliability analysis")
        
        return alerts
    
    def get_manager_info(self) -> Dict[str, Any]:
        """
        Get information about the dynamic cache manager state.
        
        Returns:
            Dictionary with manager information
        """
        strategy_info = self.current_strategy.get_strategy_info() if self.current_strategy else {}
        
        manager_info = {
            'manager_type': 'DynamicCacheManager',
            'analysis_mode': self.analysis_mode,
            'coder_count': len(self.coder_settings),
            'coder_ids': [cs.get('coder_id', 'unknown') for cs in self.coder_settings],
            'current_strategy': strategy_info,
            'mode_specific_rules': self.get_mode_specific_cache_rules(),
            'config_file': self.config_file,
            'last_config_hash': self.last_config_hash,
            'reliability_summary': self.get_reliability_summary(),
            'cache_statistics': self.get_statistics().total_entries,
            'plugins_enabled': self.enable_plugins
        }
        
        # Add plugin information if enabled
        if self.enable_plugins and self.plugin_registry:
            manager_info['plugin_system'] = self.get_plugin_system_info()
        
        return manager_info
    
    def get_plugin_system_info(self) -> Dict[str, Any]:
        """
        Get information about the plugin system status.
        
        Returns:
            Dictionary with plugin system information
        """
        if not self.enable_plugins or not self.plugin_registry:
            return {
                'enabled': False,
                'error': 'Plugin system not enabled or not initialized'
            }
        
        try:
            registry_status = self.plugin_registry.get_registry_status()
            available_strategies = self.plugin_registry.get_available_strategies()
            
            return {
                'enabled': True,
                'registry_status': registry_status,
                'available_plugin_strategies': available_strategies,
                'total_plugins': registry_status['total_plugins'],
                'total_plugin_strategies': registry_status['total_strategies'],
                'plugins_with_errors': registry_status['plugins_with_errors']
            }
            
        except Exception as e:
            return {
                'enabled': True,
                'error': f'Failed to get plugin system info: {str(e)}'
            }
    
    def load_plugin(self, plugin_path: str, plugin_name: Optional[str] = None,
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
        if not self.enable_plugins or not self.plugin_registry:
            logger.error("Plugin system not enabled, cannot load plugin")
            return False
        
        success = self.plugin_registry.load_plugin_from_file(plugin_path, plugin_name, config)
        
        if success:
            logger.info(f"Successfully loaded plugin from {plugin_path}")
            # Update available strategies in factory
            CacheStrategyFactory.set_plugin_registry(self.plugin_registry)
        else:
            logger.error(f"Failed to load plugin from {plugin_path}")
        
        return success
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Name of plugin to unload
            
        Returns:
            True if unloading successful
        """
        if not self.enable_plugins or not self.plugin_registry:
            logger.error("Plugin system not enabled, cannot unload plugin")
            return False
        
        success = self.plugin_registry.unregister_plugin(plugin_name)
        
        if success:
            logger.info(f"Successfully unloaded plugin {plugin_name}")
        else:
            logger.error(f"Failed to unload plugin {plugin_name}")
        
        return success
    
    def get_available_strategies(self) -> Dict[str, str]:
        """
        Get all available cache strategies (built-in and plugin-provided).
        
        Returns:
            Dictionary mapping strategy types to descriptions
        """
        return CacheStrategyFactory.get_available_strategies()
    
    def create_custom_strategy(self, strategy_type: str, **kwargs) -> Optional[CacheStrategy]:
        """
        Create a custom strategy using plugins.
        
        Args:
            strategy_type: Type of strategy to create
            **kwargs: Additional parameters for strategy creation
            
        Returns:
            Strategy instance or None if creation failed
        """
        if not self.enable_plugins or not self.plugin_registry:
            logger.error("Plugin system not enabled, cannot create custom strategy")
            return None
        
        # Add current manager context to kwargs
        kwargs.update({
            'coder_count': len(self.coder_settings),
            'analysis_mode': self.analysis_mode
        })
        
        strategy = self.plugin_registry.create_strategy(strategy_type, **kwargs)
        
        if strategy:
            logger.info(f"Created custom strategy: {strategy.name}")
        else:
            logger.warning(f"Failed to create custom strategy: {strategy_type}")
        
        return strategy
    
    def set_custom_strategy(self, strategy: CacheStrategy) -> None:
        """
        Set a custom strategy as the current strategy.
        
        Args:
            strategy: Custom strategy instance to use
        """
        old_strategy_name = self.current_strategy.name if self.current_strategy else "None"
        
        # Set strategy on base cache
        self.base_cache.set_strategy(strategy)
        self.current_strategy = strategy
        
        logger.info(f"Set custom cache strategy: {old_strategy_name} -> {strategy.name}")
        
        # Perform cache migration if needed
        if old_strategy_name != strategy.name:
            self._handle_strategy_change(old_strategy_name, strategy.name)
    
    def validate_plugin_system(self) -> Dict[str, List[str]]:
        """
        Validate all plugins in the system.
        
        Returns:
            Dictionary mapping plugin names to validation errors
        """
        if not self.enable_plugins or not self.plugin_registry:
            return {'system': ['Plugin system not enabled or not initialized']}
        
        return self.plugin_registry.validate_all_plugins()
    
    def export_plugin_configuration(self, filepath: str) -> None:
        """
        Export current plugin configuration to file.
        
        Args:
            filepath: Path to export configuration to
        """
        if not self.enable_plugins or not self.plugin_registry:
            logger.error("Plugin system not enabled, cannot export configuration")
            return
        
        try:
            config_data = {
                'timestamp': datetime.now().isoformat(),
                'manager_info': self.get_manager_info(),
                'plugin_system': self.get_plugin_system_info(),
                'available_strategies': self.get_available_strategies()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Plugin configuration exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export plugin configuration: {e}")
            raise
    
    def clear_reliability_data(self, segment_ids: Optional[List[str]] = None) -> None:
        """
        Clear reliability data for specified segments or all segments.
        
        Args:
            segment_ids: Optional list of segment IDs to clear, None for all
        """
        # Clear from persistent database
        try:
            cleared_count = self.reliability_db.delete_coding_results(segment_ids=segment_ids)
            logger.info(f"Cleared {cleared_count} entries from reliability database")
        except Exception as e:
            logger.warning(f"Failed to clear reliability data from database: {e}")
        
        # Clear from in-memory data
        if segment_ids is None:
            # Clear all reliability data
            cleared_count = sum(len(results) for results in self.reliability_data.values())
            self.reliability_data.clear()
            logger.info(f"Cleared all in-memory reliability data ({cleared_count} entries)")
        else:
            # Clear specific segments
            cleared_count = 0
            for segment_id in segment_ids:
                if segment_id in self.reliability_data:
                    cleared_count += len(self.reliability_data[segment_id])
                    del self.reliability_data[segment_id]
            logger.info(f"Cleared in-memory reliability data for {len(segment_ids)} segments ({cleared_count} entries)")
    
    def export_reliability_data(self, filepath: str) -> None:
        """
        Export reliability data to JSON file.
        
        Args:
            filepath: Path to export file
        """
        try:
            # Export from persistent database
            self.reliability_db.export_to_json(filepath)
            logger.info(f"Reliability data exported to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to export from database, using in-memory data: {e}")
            
            # Fallback to in-memory export
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'manager_info': self.get_manager_info(),
                'reliability_data': {
                    segment_id: [result.to_dict() for result in results]
                    for segment_id, results in self.reliability_data.items()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Reliability data exported to {filepath}")
    
    def import_reliability_data(self, filepath: str) -> None:
        """
        Import reliability data from JSON file.
        
        Args:
            filepath: Path to import file
        """
        try:
            # Import to persistent database
            imported_count = self.reliability_db.import_from_json(filepath, clear_existing=True)
            
            # Also load into memory for quick access
            self.reliability_data.clear()
            db_results = self.reliability_db.get_coding_results()
            
            for result in db_results:
                segment_id = result.segment_id
                if segment_id not in self.reliability_data:
                    self.reliability_data[segment_id] = []
                self.reliability_data[segment_id].append(result)
            
            logger.info(f"Imported {imported_count} reliability entries from {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to import to database, using in-memory import: {e}")
            
            # Fallback to in-memory import
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
                
                # Clear existing data
                self.reliability_data.clear()
                
                # Import reliability data
                reliability_data = import_data.get('reliability_data', {})
                for segment_id, results_data in reliability_data.items():
                    self.reliability_data[segment_id] = [
                        ExtendedCodingResult.from_dict(result_data)
                        for result_data in results_data
                    ]
                
                imported_count = sum(len(results) for results in self.reliability_data.values())
                logger.info(f"Imported reliability data from {filepath} ({imported_count} entries)")
                
            except Exception as e:
                logger.error(f"Failed to import reliability data from {filepath}: {e}")
                raise
    def store_manual_coding(self, segment_id: str, category: str, subcategories: List[str], 
                           justification: str, confidence: float, analysis_mode: str,
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
        coder_id = manual_coder_id or "manual"
        
        manual_result = ExtendedCodingResult(
            segment_id=segment_id,
            coder_id=coder_id,
            category=category,
            subcategories=subcategories,
            confidence=confidence,
            justification=justification,
            analysis_mode=analysis_mode,
            timestamp=datetime.now(),
            is_manual=True,
            metadata={'source': 'manual_gui', 'original_coder_id': manual_coder_id}
        )
        
        self.store_for_reliability(manual_result)
        logger.info(f"Stored manual coding for segment {segment_id} with coder ID '{coder_id}'")
    
    def get_manual_codings(self, segment_ids: Optional[List[str]] = None) -> List[ExtendedCodingResult]:
        """
        Get only manual coding results.
        
        Args:
            segment_ids: Optional list of segment IDs to filter by
            
        Returns:
            List of manual coding results
        """
        try:
            return self.reliability_db.get_coding_results(
                segment_ids=segment_ids,
                include_manual=True,
                include_automatic=False
            )
        except Exception as e:
            logger.warning(f"Failed to get manual codings from database: {e}")
            
            # Fallback to in-memory data
            all_results = self.get_reliability_data(segment_ids)
            return [result for result in all_results if result.is_manual]
    
    def get_automatic_codings(self, segment_ids: Optional[List[str]] = None) -> List[ExtendedCodingResult]:
        """
        Get only automatic coding results.
        
        Args:
            segment_ids: Optional list of segment IDs to filter by
            
        Returns:
            List of automatic coding results
        """
        try:
            return self.reliability_db.get_coding_results(
                segment_ids=segment_ids,
                include_manual=False,
                include_automatic=True
            )
        except Exception as e:
            logger.warning(f"Failed to get automatic codings from database: {e}")
            
            # Fallback to in-memory data
            all_results = self.get_reliability_data(segment_ids)
            return [result for result in all_results if not result.is_manual]
    
    def test_manual_auto_coder_combination(self, test_segments: List[str]) -> Dict[str, Any]:
        """
        Test manual + auto-coder combinations for reliability analysis.
        
        Args:
            test_segments: List of segment IDs to test with
            
        Returns:
            Dictionary with test results and reliability metrics
        """
        logger.info(f"Testing manual + auto-coder combination with {len(test_segments)} segments")
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'test_segments': test_segments,
            'manual_codings': 0,
            'automatic_codings': 0,
            'segments_with_both': 0,
            'reliability_metrics': {},
            'cache_isolation_verified': False,
            'errors': []
        }
        
        try:
            # Get all codings for test segments
            all_codings = self.get_reliability_data(segment_ids=test_segments)
            
            # Separate manual and automatic codings
            manual_codings = [c for c in all_codings if c.is_manual]
            automatic_codings = [c for c in all_codings if not c.is_manual]
            
            test_results['manual_codings'] = len(manual_codings)
            test_results['automatic_codings'] = len(automatic_codings)
            
            # Check segments with both manual and automatic codings
            segments_with_both = set()
            manual_segments = set(c.segment_id for c in manual_codings)
            auto_segments = set(c.segment_id for c in automatic_codings)
            segments_with_both = manual_segments.intersection(auto_segments)
            
            test_results['segments_with_both'] = len(segments_with_both)
            
            # Verify cache isolation
            cache_isolation_verified = self._verify_manual_cache_isolation()
            test_results['cache_isolation_verified'] = cache_isolation_verified
            
            # Calculate basic reliability metrics if we have both types
            if segments_with_both:
                reliability_metrics = self._calculate_basic_reliability_metrics(
                    manual_codings, automatic_codings, list(segments_with_both)
                )
                test_results['reliability_metrics'] = reliability_metrics
            
            # Test cache behavior with mixed coders
            cache_test_results = self._test_mixed_coder_cache_behavior()
            test_results['cache_behavior'] = cache_test_results
            
            logger.info(f"Manual + auto-coder test completed: {test_results['segments_with_both']} segments with both types")
            
        except Exception as e:
            error_msg = f"Manual + auto-coder test failed: {e}"
            logger.error(error_msg)
            test_results['errors'].append(error_msg)
        
        return test_results
    
    def _verify_manual_cache_isolation(self) -> bool:
        """
        Verify that manual coder cache entries are properly isolated from automatic coders.
        
        Returns:
            True if isolation is verified, False otherwise
        """
        try:
            # Check that manual entries have correct coder_id
            manual_entries_found = False
            isolation_violations = []
            
            for key, entry in self.base_cache.cache.items():
                # Check for manual entries
                if (entry.coder_id == "manual" or 
                    (entry.coder_id and ('manual' in entry.coder_id.lower() or 'human' in entry.coder_id.lower())) or
                    entry.metadata.get('source') == 'manual_gui'):
                    
                    manual_entries_found = True
                    
                    # Verify isolation: manual entries should not be shared with automatic coders
                    if entry.is_shared and entry.operation_type in ['coding', 'confidence_scoring']:
                        isolation_violations.append(f"Manual entry {key[:16]}... marked as shared for coder-specific operation")
                    
                    # Verify manual entries have proper metadata
                    if not entry.metadata.get('source') and entry.coder_id == "manual":
                        logger.debug(f"Manual entry {key[:16]}... missing source metadata")
            
            if isolation_violations:
                logger.warning(f"Cache isolation violations found: {isolation_violations}")
                return False
            
            if manual_entries_found:
                logger.debug("Manual coder cache isolation verified successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify manual cache isolation: {e}")
            return False
    
    def _calculate_basic_reliability_metrics(self, manual_codings: List[ExtendedCodingResult], 
                                           automatic_codings: List[ExtendedCodingResult],
                                           common_segments: List[str]) -> Dict[str, Any]:
        """
        Calculate basic reliability metrics between manual and automatic codings.
        
        Args:
            manual_codings: List of manual coding results
            automatic_codings: List of automatic coding results
            common_segments: List of segments coded by both types
            
        Returns:
            Dictionary with reliability metrics
        """
        metrics = {
            'agreement_rate': 0.0,
            'category_matches': 0,
            'total_comparisons': 0,
            'confidence_correlation': 0.0,
            'detailed_agreements': []
        }
        
        try:
            # Group codings by segment
            manual_by_segment = {c.segment_id: c for c in manual_codings if c.segment_id in common_segments}
            auto_by_segment = {c.segment_id: c for c in automatic_codings if c.segment_id in common_segments}
            
            category_matches = 0
            total_comparisons = 0
            confidence_pairs = []
            
            for segment_id in common_segments:
                if segment_id in manual_by_segment and segment_id in auto_by_segment:
                    manual_result = manual_by_segment[segment_id]
                    auto_result = auto_by_segment[segment_id]
                    
                    # Compare categories
                    category_match = manual_result.category == auto_result.category
                    if category_match:
                        category_matches += 1
                    
                    total_comparisons += 1
                    
                    # Collect confidence values for correlation
                    confidence_pairs.append((manual_result.confidence, auto_result.confidence))
                    
                    # Store detailed agreement info
                    metrics['detailed_agreements'].append({
                        'segment_id': segment_id,
                        'manual_category': manual_result.category,
                        'auto_category': auto_result.category,
                        'category_match': category_match,
                        'manual_confidence': manual_result.confidence,
                        'auto_confidence': auto_result.confidence
                    })
            
            # Calculate agreement rate
            if total_comparisons > 0:
                metrics['agreement_rate'] = category_matches / total_comparisons
                metrics['category_matches'] = category_matches
                metrics['total_comparisons'] = total_comparisons
            
            # Calculate confidence correlation (simple correlation)
            if len(confidence_pairs) > 1:
                manual_confidences = [pair[0] for pair in confidence_pairs]
                auto_confidences = [pair[1] for pair in confidence_pairs]
                
                # Simple Pearson correlation
                import statistics
                if len(set(manual_confidences)) > 1 and len(set(auto_confidences)) > 1:
                    mean_manual = statistics.mean(manual_confidences)
                    mean_auto = statistics.mean(auto_confidences)
                    
                    numerator = sum((m - mean_manual) * (a - mean_auto) 
                                  for m, a in zip(manual_confidences, auto_confidences))
                    
                    manual_var = sum((m - mean_manual) ** 2 for m in manual_confidences)
                    auto_var = sum((a - mean_auto) ** 2 for a in auto_confidences)
                    
                    if manual_var > 0 and auto_var > 0:
                        correlation = numerator / (manual_var * auto_var) ** 0.5
                        metrics['confidence_correlation'] = correlation
            
        except Exception as e:
            logger.warning(f"Failed to calculate reliability metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _test_mixed_coder_cache_behavior(self) -> Dict[str, Any]:
        """
        Test cache behavior with mixed manual and automatic coders.
        
        Returns:
            Dictionary with cache behavior test results
        """
        test_results = {
            'manual_entries_isolated': True,
            'shared_operations_work': True,
            'coder_specific_operations_separated': True,
            'cache_key_generation_correct': True,
            'errors': []
        }
        
        try:
            # Test 1: Verify manual entries are isolated
            manual_entries = []
            auto_entries = []
            
            for key, entry in self.base_cache.cache.items():
                if (entry.coder_id == "manual" or 
                    (entry.coder_id and ('manual' in entry.coder_id.lower() or 'human' in entry.coder_id.lower()))):
                    manual_entries.append(entry)
                elif entry.coder_id and entry.coder_id != "manual":
                    auto_entries.append(entry)
            
            # Check for improper sharing
            for manual_entry in manual_entries:
                if manual_entry.operation_type in ['coding', 'confidence_scoring'] and manual_entry.is_shared:
                    test_results['manual_entries_isolated'] = False
                    test_results['errors'].append(f"Manual entry for {manual_entry.operation_type} incorrectly marked as shared")
            
            # Test 2: Verify shared operations work for both types
            shared_ops = ['relevance_check', 'category_development']
            for op in shared_ops:
                # Generate keys for both manual and automatic
                try:
                    manual_key = self.get_shared_cache_key(op, test_param="test")
                    auto_key = self.get_shared_cache_key(op, test_param="test")
                    
                    if manual_key != auto_key:
                        test_results['shared_operations_work'] = False
                        test_results['errors'].append(f"Shared operation {op} generates different keys for manual/auto")
                        
                except Exception as e:
                    test_results['shared_operations_work'] = False
                    test_results['errors'].append(f"Failed to generate shared key for {op}: {e}")
            
            # Test 3: Verify coder-specific operations are separated
            coder_specific_ops = ['coding', 'confidence_scoring']
            for op in coder_specific_ops:
                try:
                    manual_key = self.get_coder_specific_key("manual", op, test_param="test")
                    auto_key = self.get_coder_specific_key("auto_1", op, test_param="test")
                    
                    if manual_key == auto_key:
                        test_results['coder_specific_operations_separated'] = False
                        test_results['errors'].append(f"Coder-specific operation {op} generates same key for different coders")
                        
                except Exception as e:
                    test_results['coder_specific_operations_separated'] = False
                    test_results['errors'].append(f"Failed to generate coder-specific key for {op}: {e}")
            
            logger.debug(f"Mixed coder cache behavior test completed with {len(test_results['errors'])} errors")
            
        except Exception as e:
            error_msg = f"Mixed coder cache behavior test failed: {e}"
            logger.error(error_msg)
            test_results['errors'].append(error_msg)
        
        return test_results
    
    def get_segments_for_reliability_analysis(self) -> List[str]:
        """
        Get segments that have multiple coders for reliability analysis.
        
        Returns:
            List of segment IDs suitable for reliability analysis
        """
        try:
            return self.reliability_db.get_segments_for_reliability_analysis()
        except Exception as e:
            logger.warning(f"Failed to get segments from database: {e}")
            
            # Fallback to in-memory data
            multi_coder_segments = []
            for segment_id, results in self.reliability_data.items():
                unique_coders = set(result.coder_id for result in results)
                if len(unique_coders) > 1:
                    multi_coder_segments.append(segment_id)
            return multi_coder_segments
    
    def backup_reliability_database(self, backup_path: str) -> None:
        """
        Create a backup of the reliability database.
        
        Args:
            backup_path: Path for backup file
        """
        try:
            self.reliability_db.backup_database(backup_path)
            logger.info(f"Reliability database backed up to {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup reliability database: {e}")
            raise
    
    def get_reliability_database_info(self) -> Dict[str, Any]:
        """
        Get information about the reliability database.
        
        Returns:
            Dictionary with database information
        """
        try:
            return self.reliability_db.get_database_info()
        except Exception as e:
            logger.warning(f"Failed to get database info: {e}")
            return {
                'database_path': 'unknown',
                'database_size_bytes': 0,
                'database_size_mb': 0.0,
                'version': 'unknown',
                'tables': [],
                'summary': self.get_reliability_summary()
            }
    
    def _migrate_cache_entries(self, old_strategy: str, new_strategy: str) -> bool:
        """
        Migrate cache entries when strategy changes.
        
        Args:
            old_strategy: Previous strategy name
            new_strategy: New strategy name
            
        Returns:
            True if migration successful, False otherwise
        """
        try:
            logger.info(f"Starting cache migration: {old_strategy} -> {new_strategy}")
            
            # Create backup of current cache state
            migration_backup = self._create_migration_backup()
            
            # Get all current cache entries
            all_entries = list(self.base_cache.cache.items())
            migrated_count = 0
            invalidated_count = 0
            
            for cache_key, entry in all_entries:
                try:
                    # Determine if entry needs migration or invalidation
                    migration_action = self._determine_migration_action(entry, old_strategy, new_strategy)
                    
                    if migration_action == "migrate":
                        # Regenerate cache key with new strategy
                        new_key = self._regenerate_cache_key(entry)
                        
                        if new_key != cache_key:
                            # Key changed, need to migrate
                            self._migrate_single_entry(cache_key, new_key, entry)
                            migrated_count += 1
                        # If key is same, no migration needed
                        
                    elif migration_action == "invalidate":
                        # Remove entry that's no longer valid
                        self.base_cache._remove_entry_from_indexes(cache_key, entry)
                        del self.base_cache.cache[cache_key]
                        invalidated_count += 1
                        
                    # Update entry metadata to reflect new strategy
                    if cache_key in self.base_cache.cache:
                        self.base_cache.cache[cache_key].metadata['migrated_from'] = old_strategy
                        self.base_cache.cache[cache_key].metadata['migration_timestamp'] = datetime.now().isoformat()
                        
                except Exception as e:
                    logger.warning(f"Failed to migrate entry {cache_key}: {e}")
                    # Continue with other entries
                    continue
            
            logger.info(f"Cache migration completed: {migrated_count} migrated, {invalidated_count} invalidated")
            
            # Store migration backup info for potential rollback
            self._migration_backup = migration_backup
            self._migration_backup['migration_stats'] = {
                'migrated_count': migrated_count,
                'invalidated_count': invalidated_count,
                'total_processed': len(all_entries)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Cache migration failed: {e}")
            return False
    
    def _create_migration_backup(self) -> Dict[str, Any]:
        """
        Create backup of cache state before migration.
        
        Returns:
            Backup data dictionary
        """
        backup = {
            'timestamp': datetime.now().isoformat(),
            'cache_entries': {},
            'indexes': {
                'mode_index': dict(self.base_cache.mode_index),
                'segment_index': dict(self.base_cache.segment_index),
                'category_index': dict(self.base_cache.category_index),
                'coder_index': dict(self.base_cache.coder_index),
                'operation_index': dict(self.base_cache.operation_index),
                'shared_index': list(self.base_cache.shared_index)
            },
            'stats': dict(self.base_cache.stats)
        }
        
        # Create deep copy of cache entries
        for key, entry in self.base_cache.cache.items():
            backup['cache_entries'][key] = {
                'key': entry.key,
                'value': entry.value,
                'timestamp': entry.timestamp,
                'access_count': entry.access_count,
                'analysis_mode': entry.analysis_mode,
                'segment_id': entry.segment_id,
                'category': entry.category,
                'confidence': entry.confidence,
                'coder_id': entry.coder_id,
                'operation_type': entry.operation_type,
                'is_shared': entry.is_shared,
                'metadata': dict(entry.metadata) if entry.metadata else {}
            }
        
        logger.debug(f"Created migration backup with {len(backup['cache_entries'])} entries")
        return backup
    
    def _determine_migration_action(self, entry, old_strategy: str, new_strategy: str) -> str:
        """
        Determine what action to take for a cache entry during migration.
        
        Args:
            entry: Cache entry to evaluate
            old_strategy: Previous strategy name
            new_strategy: New strategy name
            
        Returns:
            Action to take: "migrate", "invalidate", or "keep"
        """
        operation_type = entry.operation_type
        
        # Check if operation is supported by new strategy
        if new_strategy == "SingleCoder":
            # All operations are supported in single coder mode
            return "migrate"
        elif new_strategy == "MultiCoder":
            # Check if operation classification changed
            old_was_shared = (old_strategy == "SingleCoder" or 
                            operation_type in self.current_strategy.shared_operations)
            new_is_shared = operation_type in self.current_strategy.shared_operations
            
            if old_was_shared != new_is_shared:
                # Operation classification changed, need to migrate
                return "migrate"
            else:
                # Classification same, keep entry
                return "keep"
        
        # Default to migration for unknown strategies
        return "migrate"
    
    def _regenerate_cache_key(self, entry) -> str:
        """
        Regenerate cache key for entry using current strategy.
        
        Args:
            entry: Cache entry to regenerate key for
            
        Returns:
            New cache key
        """
        # Extract parameters from entry metadata or reconstruct from entry
        params = {
            'analysis_mode': entry.analysis_mode,
            'segment_text': entry.metadata.get('segment_text', ''),
            'research_question': entry.metadata.get('research_question', ''),
            'category_definitions': entry.metadata.get('category_definitions'),
            'coding_rules': entry.metadata.get('coding_rules')
        }
        
        return self.current_strategy.get_cache_key(
            entry.operation_type,
            entry.coder_id,
            **params
        )
    
    def _migrate_single_entry(self, old_key: str, new_key: str, entry) -> None:
        """
        Migrate a single cache entry to new key.
        
        Args:
            old_key: Original cache key
            new_key: New cache key
            entry: Cache entry to migrate
        """
        # Remove from old location
        self.base_cache._remove_entry_from_indexes(old_key, entry)
        del self.base_cache.cache[old_key]
        
        # Update entry key
        entry.key = new_key
        
        # Add to new location
        self.base_cache.cache[new_key] = entry
        self.base_cache.cache.move_to_end(new_key)
        
        # Update indexes with new key
        if entry.analysis_mode:
            if entry.analysis_mode not in self.base_cache.mode_index:
                self.base_cache.mode_index[entry.analysis_mode] = []
            self.base_cache.mode_index[entry.analysis_mode].append(new_key)
        
        if entry.segment_id:
            if entry.segment_id not in self.base_cache.segment_index:
                self.base_cache.segment_index[entry.segment_id] = []
            self.base_cache.segment_index[entry.segment_id].append(new_key)
        
        if entry.category:
            if entry.category not in self.base_cache.category_index:
                self.base_cache.category_index[entry.category] = []
            self.base_cache.category_index[entry.category].append(new_key)
        
        if entry.coder_id:
            if entry.coder_id not in self.base_cache.coder_index:
                self.base_cache.coder_index[entry.coder_id] = []
            self.base_cache.coder_index[entry.coder_id].append(new_key)
        
        if entry.operation_type not in self.base_cache.operation_index:
            self.base_cache.operation_index[entry.operation_type] = []
        self.base_cache.operation_index[entry.operation_type].append(new_key)
        
        if entry.is_shared:
            self.base_cache.shared_index.append(new_key)
        
        logger.debug(f"Migrated cache entry: {old_key[:16]}... -> {new_key[:16]}...")
    
    def _rollback_migration(self, target_strategy: str) -> None:
        """
        Rollback failed migration to previous state.
        
        Args:
            target_strategy: Strategy to rollback to
        """
        if not hasattr(self, '_migration_backup') or not self._migration_backup:
            logger.error("No migration backup available for rollback")
            return
        
        try:
            logger.info("Starting cache migration rollback")
            backup = self._migration_backup
            
            # Clear current cache
            self.base_cache.clear()
            
            # Restore cache entries from backup
            from .cache import CacheEntry
            
            for key, entry_data in backup['cache_entries'].items():
                entry = CacheEntry(
                    key=entry_data['key'],
                    value=entry_data['value'],
                    timestamp=entry_data['timestamp'],
                    access_count=entry_data['access_count'],
                    analysis_mode=entry_data['analysis_mode'],
                    segment_id=entry_data['segment_id'],
                    category=entry_data['category'],
                    confidence=entry_data['confidence'],
                    coder_id=entry_data['coder_id'],
                    operation_type=entry_data['operation_type'],
                    is_shared=entry_data['is_shared'],
                    metadata=entry_data['metadata']
                )
                
                self.base_cache.cache[key] = entry
            
            # Restore indexes
            self.base_cache.mode_index = backup['indexes']['mode_index']
            self.base_cache.segment_index = backup['indexes']['segment_index']
            self.base_cache.category_index = backup['indexes']['category_index']
            self.base_cache.coder_index = backup['indexes']['coder_index']
            self.base_cache.operation_index = backup['indexes']['operation_index']
            self.base_cache.shared_index = backup['indexes']['shared_index']
            
            # Restore stats
            self.base_cache.stats = backup['stats']
            
            logger.info(f"Cache migration rollback completed, restored {len(backup['cache_entries'])} entries")
            
        except Exception as e:
            logger.error(f"Cache migration rollback failed: {e}")
            # Clear cache as last resort
            self.base_cache.clear()
    
    def _check_cache_consistency(self) -> List[str]:
        """
        Check cache consistency and return list of issues found.
        
        Returns:
            List of consistency issue descriptions
        """
        issues = []
        
        try:
            # Check 1: Verify all cache entries exist in indexes
            cache_keys = set(self.base_cache.cache.keys())
            
            # Check mode index consistency
            mode_index_keys = set()
            for mode_keys in self.base_cache.mode_index.values():
                mode_index_keys.update(mode_keys)
            
            orphaned_in_mode_index = mode_index_keys - cache_keys
            if orphaned_in_mode_index:
                issues.append(f"Mode index contains {len(orphaned_in_mode_index)} orphaned keys")
            
            # Check coder index consistency
            coder_index_keys = set()
            for coder_keys in self.base_cache.coder_index.values():
                coder_index_keys.update(coder_keys)
            
            orphaned_in_coder_index = coder_index_keys - cache_keys
            if orphaned_in_coder_index:
                issues.append(f"Coder index contains {len(orphaned_in_coder_index)} orphaned keys")
            
            # Check operation index consistency
            operation_index_keys = set()
            for op_keys in self.base_cache.operation_index.values():
                operation_index_keys.update(op_keys)
            
            orphaned_in_operation_index = operation_index_keys - cache_keys
            if orphaned_in_operation_index:
                issues.append(f"Operation index contains {len(orphaned_in_operation_index)} orphaned keys")
            
            # Check shared index consistency
            shared_index_keys = set(self.base_cache.shared_index)
            orphaned_in_shared_index = shared_index_keys - cache_keys
            if orphaned_in_shared_index:
                issues.append(f"Shared index contains {len(orphaned_in_shared_index)} orphaned keys")
            
            # Check 2: Verify entry classification matches strategy
            for key, entry in self.base_cache.cache.items():
                expected_shared = self.current_strategy.should_cache_shared(entry.operation_type)
                if entry.is_shared != expected_shared:
                    issues.append(f"Entry {key[:16]}... has incorrect shared classification")
            
            # Check 3: Verify coder-specific entries have coder_id
            for key, entry in self.base_cache.cache.items():
                if self.current_strategy.should_cache_per_coder(entry.operation_type):
                    if not entry.coder_id:
                        issues.append(f"Coder-specific entry {key[:16]}... missing coder_id")
            
            # Check 4: Verify shared entries don't have coder_id (unless manual)
            for key, entry in self.base_cache.cache.items():
                if self.current_strategy.should_cache_shared(entry.operation_type):
                    if entry.coder_id and entry.coder_id != "manual":
                        issues.append(f"Shared entry {key[:16]}... has unexpected coder_id: {entry.coder_id}")
            
            logger.debug(f"Cache consistency check completed, found {len(issues)} issues")
            
        except Exception as e:
            logger.error(f"Cache consistency check failed: {e}")
            issues.append(f"Consistency check error: {str(e)}")
        
        return issues
    
    def _fix_consistency_issues(self, issues: List[str]) -> None:
        """
        Attempt to fix cache consistency issues.
        
        Args:
            issues: List of consistency issues to fix
        """
        logger.info(f"Attempting to fix {len(issues)} consistency issues")
        
        try:
            # Fix 1: Remove orphaned keys from indexes
            cache_keys = set(self.base_cache.cache.keys())
            
            # Clean mode index
            for mode, keys in list(self.base_cache.mode_index.items()):
                valid_keys = [k for k in keys if k in cache_keys]
                self.base_cache.mode_index[mode] = valid_keys
                if not valid_keys:
                    del self.base_cache.mode_index[mode]
            
            # Clean coder index
            for coder, keys in list(self.base_cache.coder_index.items()):
                valid_keys = [k for k in keys if k in cache_keys]
                self.base_cache.coder_index[coder] = valid_keys
                if not valid_keys:
                    del self.base_cache.coder_index[coder]
            
            # Clean operation index
            for operation, keys in list(self.base_cache.operation_index.items()):
                valid_keys = [k for k in keys if k in cache_keys]
                self.base_cache.operation_index[operation] = valid_keys
                if not valid_keys:
                    del self.base_cache.operation_index[operation]
            
            # Clean shared index
            self.base_cache.shared_index = [k for k in self.base_cache.shared_index if k in cache_keys]
            
            # Fix 2: Correct entry classifications
            for key, entry in self.base_cache.cache.items():
                expected_shared = self.current_strategy.should_cache_shared(entry.operation_type)
                if entry.is_shared != expected_shared:
                    entry.is_shared = expected_shared
                    
                    # Update shared index
                    if expected_shared and key not in self.base_cache.shared_index:
                        self.base_cache.shared_index.append(key)
                    elif not expected_shared and key in self.base_cache.shared_index:
                        self.base_cache.shared_index.remove(key)
            
            # Fix 3: Remove entries with invalid coder_id configuration
            keys_to_remove = []
            for key, entry in self.base_cache.cache.items():
                # Remove coder-specific entries without coder_id
                if (self.current_strategy.should_cache_per_coder(entry.operation_type) and 
                    not entry.coder_id):
                    keys_to_remove.append(key)
                
                # Remove shared entries with unexpected coder_id
                elif (self.current_strategy.should_cache_shared(entry.operation_type) and 
                      entry.coder_id and entry.coder_id != "manual"):
                    keys_to_remove.append(key)
            
            # Remove invalid entries
            for key in keys_to_remove:
                if key in self.base_cache.cache:
                    entry = self.base_cache.cache[key]
                    self.base_cache._remove_entry_from_indexes(key, entry)
                    del self.base_cache.cache[key]
            
            logger.info(f"Fixed consistency issues, removed {len(keys_to_remove)} invalid entries")
            
        except Exception as e:
            logger.error(f"Failed to fix consistency issues: {e}")
    
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
        logger.info(f"Performing selective cache clear with criteria: {criteria}")
        
        keys_to_remove = set()
        
        try:
            # Clear by operations
            if 'operations' in criteria:
                for operation in criteria['operations']:
                    if operation in self.base_cache.operation_index:
                        keys_to_remove.update(self.base_cache.operation_index[operation])
            
            # Clear by coders
            if 'coders' in criteria:
                for coder in criteria['coders']:
                    if coder in self.base_cache.coder_index:
                        keys_to_remove.update(self.base_cache.coder_index[coder])
            
            # Clear by modes
            if 'modes' in criteria:
                for mode in criteria['modes']:
                    if mode in self.base_cache.mode_index:
                        keys_to_remove.update(self.base_cache.mode_index[mode])
            
            # Clear by age
            if 'older_than_hours' in criteria:
                cutoff_time = time.time() - (criteria['older_than_hours'] * 3600)
                for key, entry in self.base_cache.cache.items():
                    if entry.timestamp < cutoff_time:
                        keys_to_remove.add(key)
            
            # Clear by confidence
            if 'confidence_below' in criteria:
                threshold = criteria['confidence_below']
                for key, entry in self.base_cache.cache.items():
                    if entry.confidence < threshold:
                        keys_to_remove.add(key)
            
            # Remove selected entries
            cleared_count = 0
            for key in keys_to_remove:
                if key in self.base_cache.cache:
                    entry = self.base_cache.cache[key]
                    self.base_cache._remove_entry_from_indexes(key, entry)
                    del self.base_cache.cache[key]
                    cleared_count += 1
            
            # Update statistics
            self.base_cache.stats["total_entries"] = len(self.base_cache.cache)
            self.base_cache.stats["invalidations"] += cleared_count
            
            # Notify listeners
            if keys_to_remove:
                self.base_cache._notify_cache_invalidate(
                    list(keys_to_remove), 
                    f"selective_clear:{criteria}"
                )
            
            logger.info(f"Selective cache clear completed: {cleared_count} entries removed")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Selective cache clear failed: {e}")
            return 0
    
    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get status of cache migration operations.
        
        Returns:
            Dictionary with migration status information
        """
        migration_info = {
            'has_backup': hasattr(self, '_migration_backup') and self._migration_backup is not None,
            'backup_timestamp': None,
            'migration_stats': None,
            'consistency_status': 'unknown'
        }
        
        if hasattr(self, '_migration_backup') and self._migration_backup:
            migration_info['backup_timestamp'] = self._migration_backup.get('timestamp')
            migration_info['migration_stats'] = self._migration_backup.get('migration_stats')
        
        # Check current consistency
        try:
            consistency_issues = self._check_cache_consistency()
            migration_info['consistency_status'] = 'clean' if not consistency_issues else 'issues_found'
            migration_info['consistency_issues'] = consistency_issues
        except Exception as e:
            migration_info['consistency_status'] = 'check_failed'
            migration_info['consistency_error'] = str(e)
        
        return migration_info
    
    def force_consistency_check(self) -> Dict[str, Any]:
        """
        Force a consistency check and return detailed results.
        
        Returns:
            Dictionary with consistency check results
        """
        logger.info("Performing forced consistency check")
        
        try:
            issues = self._check_cache_consistency()
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'total_issues': len(issues),
                'issues': issues,
                'cache_stats': {
                    'total_entries': len(self.base_cache.cache),
                    'shared_entries': len(self.base_cache.shared_index),
                    'coder_specific_entries': len(self.base_cache.cache) - len(self.base_cache.shared_index),
                    'index_sizes': {
                        'mode_index': sum(len(keys) for keys in self.base_cache.mode_index.values()),
                        'coder_index': sum(len(keys) for keys in self.base_cache.coder_index.values()),
                        'operation_index': sum(len(keys) for keys in self.base_cache.operation_index.values()),
                        'shared_index': len(self.base_cache.shared_index)
                    }
                }
            }
            
            if issues:
                logger.warning(f"Consistency check found {len(issues)} issues")
            else:
                logger.info("Consistency check passed - no issues found")
            
            return result
            
        except Exception as e:
            logger.error(f"Forced consistency check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'total_issues': -1,
                'issues': [],
                'cache_stats': {}
            }
    
    def repair_cache_consistency(self, auto_fix: bool = True) -> Dict[str, Any]:
        """
        Repair cache consistency issues.
        
        Args:
            auto_fix: Whether to automatically fix issues or just report them
            
        Returns:
            Dictionary with repair results
        """
        logger.info(f"Starting cache consistency repair (auto_fix={auto_fix})")
        
        try:
            # First, check for issues
            issues_before = self._check_cache_consistency()
            
            if not issues_before:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'issues_before': 0,
                    'issues_after': 0,
                    'fixed_issues': 0,
                    'status': 'no_issues_found'
                }
            
            if auto_fix:
                # Attempt to fix issues
                self._fix_consistency_issues(issues_before)
                
                # Check again after fixes
                issues_after = self._check_cache_consistency()
                fixed_count = len(issues_before) - len(issues_after)
                
                return {
                    'timestamp': datetime.now().isoformat(),
                    'issues_before': len(issues_before),
                    'issues_after': len(issues_after),
                    'fixed_issues': fixed_count,
                    'remaining_issues': issues_after,
                    'status': 'repair_completed' if not issues_after else 'partial_repair'
                }
            else:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'issues_before': len(issues_before),
                    'issues_found': issues_before,
                    'status': 'issues_identified_no_fix'
                }
                
        except Exception as e:
            logger.error(f"Cache consistency repair failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'repair_failed'
            }
    
    # ========================================
    # Test Mode and Debugging Methods
    # ========================================
    
    def enable_test_mode(self, mode: str = "deterministic", seed: int = 42, 
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
        if not self.test_manager:
            return {'error': 'Test mode not available - debug tools not initialized'}
        
        try:
            from .cache_debug_tools import TestMode
            
            if mode == "deterministic":
                self.test_manager.enable_deterministic_mode(seed)
                logger.info(f"Deterministic test mode enabled with seed {seed}")
                
            elif mode == "record":
                if not recording_file:
                    recording_file = f"cache_recording_{int(time.time())}.json"
                self.test_manager.start_recording(recording_file)
                logger.info(f"Recording mode started, saving to {recording_file}")
                
            elif mode == "replay":
                if not recording_file:
                    return {'error': 'Recording file required for replay mode'}
                self.test_manager.start_replay(recording_file)
                logger.info(f"Replay mode started from {recording_file}")
                
            else:
                return {'error': f'Unknown test mode: {mode}'}
            
            return {
                'status': 'success',
                'mode': mode,
                'seed': seed if mode == "deterministic" else None,
                'recording_file': recording_file,
                'test_mode_info': self.test_manager.get_test_mode_info()
            }
            
        except Exception as e:
            logger.error(f"Failed to enable test mode: {e}")
            return {'error': str(e)}
    
    def disable_test_mode(self) -> Dict[str, Any]:
        """
        Disable test mode and return to normal operation.
        
        Returns:
            Disable operation status
        """
        if not self.test_manager:
            return {'error': 'Test mode not available'}
        
        try:
            # Stop any active recording
            if self.test_manager.mode.value == "record":
                self.test_manager.stop_recording()
            
            # Reset to disabled mode
            from .cache_debug_tools import TestMode
            self.test_manager.mode = TestMode.DISABLED
            
            logger.info("Test mode disabled")
            
            return {
                'status': 'success',
                'message': 'Test mode disabled'
            }
            
        except Exception as e:
            logger.error(f"Failed to disable test mode: {e}")
            return {'error': str(e)}
    
    def get_test_mode_status(self) -> Dict[str, Any]:
        """
        Get current test mode status.
        
        Returns:
            Test mode status information
        """
        if not self.test_manager:
            return {'available': False, 'error': 'Test mode not available'}
        
        return {
            'available': True,
            'active': self.test_manager.is_test_mode_active(),
            'info': self.test_manager.get_test_mode_info()
        }
    
    def set_debug_level(self, level: str) -> Dict[str, Any]:
        """
        Set debug logging level.
        
        Args:
            level: Debug level (SILENT, ERROR, WARNING, INFO, DEBUG, TRACE)
            
        Returns:
            Operation status
        """
        if not self.debug_logger:
            return {'error': 'Debug logger not available'}
        
        try:
            from .cache_debug_tools import LogLevel
            
            level_map = {
                'SILENT': LogLevel.SILENT,
                'ERROR': LogLevel.ERROR,
                'WARNING': LogLevel.WARNING,
                'INFO': LogLevel.INFO,
                'DEBUG': LogLevel.DEBUG,
                'TRACE': LogLevel.TRACE
            }
            
            if level.upper() not in level_map:
                return {'error': f'Invalid debug level: {level}'}
            
            old_level = self.debug_logger.level.name
            self.debug_logger.level = level_map[level.upper()]
            self.debug_level = level.upper()
            
            logger.info(f"Debug level changed: {old_level} -> {level.upper()}")
            
            return {
                'status': 'success',
                'old_level': old_level,
                'new_level': level.upper()
            }
            
        except Exception as e:
            logger.error(f"Failed to set debug level: {e}")
            return {'error': str(e)}
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """
        Get debug operation summary.
        
        Returns:
            Debug summary information
        """
        if not self.debug_logger:
            return {'error': 'Debug logger not available'}
        
        try:
            summary = self.debug_logger.get_operation_summary()
            
            return {
                'debug_level': self.debug_level,
                'operation_summary': summary,
                'test_mode_status': self.get_test_mode_status(),
                'cache_statistics': self.get_statistics().total_entries
            }
            
        except Exception as e:
            logger.error(f"Failed to get debug summary: {e}")
            return {'error': str(e)}
    
    def create_benchmark_suite(self) -> Optional['CacheBenchmark']:
        """
        Create benchmark suite for performance testing.
        
        Returns:
            Benchmark suite instance or None if not available
        """
        try:
            from .cache_debug_tools import CacheBenchmark
            return CacheBenchmark(self)
        except ImportError as e:
            logger.warning(f"Benchmark tools not available: {e}")
            return None
    
    def run_performance_benchmark(self, test_type: str = "basic", **kwargs) -> Dict[str, Any]:
        """
        Run performance benchmark.
        
        Args:
            test_type: Type of benchmark ('basic', 'multi_coder', 'memory_stress')
            **kwargs: Additional parameters for benchmark
            
        Returns:
            Benchmark results
        """
        benchmark = self.create_benchmark_suite()
        if not benchmark:
            return {'error': 'Benchmark tools not available'}
        
        try:
            if test_type == "basic":
                result = benchmark.run_basic_performance_test(
                    operations_count=kwargs.get('operations_count', 1000)
                )
            elif test_type == "multi_coder":
                result = benchmark.run_multi_coder_benchmark(
                    coder_count=kwargs.get('coder_count', 3),
                    operations_per_coder=kwargs.get('operations_per_coder', 500)
                )
            elif test_type == "memory_stress":
                result = benchmark.run_memory_stress_test(
                    target_memory_mb=kwargs.get('target_memory_mb', 100)
                )
            else:
                return {'error': f'Unknown benchmark type: {test_type}'}
            
            return {
                'status': 'success',
                'benchmark_result': {
                    'test_name': result.test_name,
                    'duration_seconds': result.duration_seconds,
                    'operations_count': result.operations_count,
                    'operations_per_second': result.operations_per_second,
                    'hit_rate': result.hit_rate,
                    'memory_usage_mb': result.memory_usage_mb,
                    'cache_size': result.cache_size,
                    'error_count': result.error_count,
                    'metadata': result.metadata
                }
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {'error': str(e)}
    
    def create_cache_dump(self, filepath: str, include_values: bool = False, 
                         compress: bool = True, format: str = 'json') -> Dict[str, Any]:
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
        try:
            from .cache_debug_tools import CacheDumpManager
            
            dump_manager = CacheDumpManager(self)
            cache_dump = dump_manager.create_dump(include_values, compress)
            dump_manager.save_dump(cache_dump, filepath, format)
            
            return {
                'status': 'success',
                'filepath': filepath,
                'format': format,
                'total_entries': cache_dump.total_entries,
                'include_values': include_values,
                'compress': compress,
                'dump_size_bytes': cache_dump.metadata.get('total_value_size_bytes', 0)
            }
            
        except Exception as e:
            logger.error(f"Cache dump failed: {e}")
            return {'error': str(e)}
    
    def restore_cache_dump(self, filepath: str, clear_existing: bool = True, 
                          restore_values: bool = False, format: str = 'json') -> Dict[str, Any]:
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
        try:
            from .cache_debug_tools import CacheDumpManager
            
            dump_manager = CacheDumpManager(self)
            cache_dump = dump_manager.load_dump(filepath, format)
            result = dump_manager.restore_from_dump(cache_dump, clear_existing, restore_values)
            
            return {
                'status': 'success',
                'filepath': filepath,
                'format': format,
                'restore_result': result
            }
            
        except Exception as e:
            logger.error(f"Cache restore failed: {e}")
            return {'error': str(e)}
    
    def start_debug_session(self, log_file: Optional[str] = None, 
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
        try:
            # Set up file logging if requested
            if log_file and self.debug_logger:
                self.debug_logger._setup_file_logging(log_file)
            
            # Enable test mode if requested
            if test_mode:
                test_result = self.enable_test_mode(test_mode, test_seed)
                if 'error' in test_result:
                    return test_result
            
            session_info = {
                'status': 'success',
                'session_started': datetime.now().isoformat(),
                'debug_level': self.debug_level,
                'log_file': log_file,
                'test_mode': test_mode,
                'test_seed': test_seed if test_mode == "deterministic" else None,
                'cache_stats': {
                    'total_entries': len(self.base_cache.cache),
                    'strategy': self.current_strategy.name if self.current_strategy else 'None'
                }
            }
            
            logger.info(f"Debug session started: {session_info}")
            return session_info
            
        except Exception as e:
            logger.error(f"Failed to start debug session: {e}")
            return {'error': str(e)}
    
    def export_debug_report(self, filepath: str) -> Dict[str, Any]:
        """
        Export comprehensive debug report.
        
        Args:
            filepath: Path to save debug report
            
        Returns:
            Export operation status
        """
        try:
            debug_report = {
                'timestamp': datetime.now().isoformat(),
                'manager_info': self.get_manager_info(),
                'cache_statistics': self.get_statistics().__dict__,
                'debug_summary': self.get_debug_summary(),
                'test_mode_status': self.get_test_mode_status(),
                'consistency_check': self.force_consistency_check(),
                'reliability_summary': self.get_reliability_summary(),
                'performance_metrics': self.base_cache.stats.copy()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(debug_report, f, indent=2, default=str)
            
            logger.info(f"Debug report exported to {filepath}")
            
            return {
                'status': 'success',
                'filepath': filepath,
                'report_sections': list(debug_report.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to export debug report: {e}")
            return {'error': str(e)}