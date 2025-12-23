"""
Example Custom Cache Strategy Plugin
===================================
Demonstrates how to create a custom cache strategy plugin for the dynamic cache system.
This example implements a performance-optimized strategy for high-throughput scenarios.
"""

import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime

from QCA_AID_assets.optimization.cache_plugin_interface import CachePluginInterface
from QCA_AID_assets.optimization.cache_strategies import CacheStrategy

logger = logging.getLogger(__name__)


class PerformanceOptimizedCacheStrategy(CacheStrategy):
    """
    Performance-optimized cache strategy that prioritizes speed over methodological separation.
    
    This strategy is designed for high-throughput scenarios where cache performance
    is more important than strict methodological compliance.
    """
    
    def __init__(self, coder_count: int = 1, aggressive_sharing: bool = True, 
                 cache_everything: bool = False):
        """
        Initialize performance-optimized cache strategy.
        
        Args:
            coder_count: Number of coders configured
            aggressive_sharing: Whether to aggressively share operations across coders
            cache_everything: Whether to cache all operations regardless of type
        """
        super().__init__(f"PerformanceOptimized_{coder_count}coder{'s' if coder_count != 1 else ''}")
        self.coder_count = coder_count
        self.aggressive_sharing = aggressive_sharing
        self.cache_everything = cache_everything
        
        # Define operation classifications based on performance optimization
        if aggressive_sharing or coder_count <= 1:
            # Maximize sharing for performance
            self.shared_operations = {
                'relevance_check',
                'category_development',
                'subcode_collection',
                'subcategory_development',
                'coding',  # Shared even coding for performance
                'confidence_scoring'
            }
            self.coder_specific_operations = {
                'manual_coding'  # Only manual coding remains coder-specific
            }
        else:
            # Standard separation but optimized
            self.shared_operations = {
                'relevance_check',
                'category_development',
                'subcode_collection',
                'subcategory_development'
            }
            self.coder_specific_operations = {
                'coding',
                'confidence_scoring',
                'manual_coding'
            }
        
        # Override if cache_everything is enabled
        if cache_everything:
            all_ops = self.shared_operations | self.coder_specific_operations
            self.shared_operations = all_ops
            self.coder_specific_operations = set()
        
        logger.info(f"Initialized {self.name} with aggressive_sharing={aggressive_sharing}, "
                   f"cache_everything={cache_everything}")
    
    def should_cache_shared(self, operation: str) -> bool:
        """
        Determine if operation should be shared (optimized for performance).
        
        Args:
            operation: Operation type
            
        Returns:
            True if operation should be shared
        """
        return operation in self.shared_operations
    
    def should_cache_per_coder(self, operation: str) -> bool:
        """
        Determine if operation should be cached per coder.
        
        Args:
            operation: Operation type
            
        Returns:
            True if operation should be cached per coder
        """
        return operation in self.coder_specific_operations
    
    def get_cache_key(self, operation: str, coder_id: Optional[str] = None, **params) -> str:
        """
        Generate optimized cache key with minimal overhead.
        
        Args:
            operation: Operation type
            coder_id: Coder ID for coder-specific operations
            **params: Additional parameters
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Simplified key generation for performance
        components = [
            operation,
            params.get('analysis_mode', ''),
        ]
        
        # Add coder ID only if truly needed
        if self.should_cache_per_coder(operation) and coder_id:
            components.append(f"coder:{coder_id}")
        
        # Use only essential parameters to minimize key generation overhead
        segment_text = params.get('segment_text', '')
        if segment_text:
            # Use hash of segment text instead of full text for performance
            segment_hash = hashlib.md5(segment_text.encode('utf-8')).hexdigest()[:8]
            components.append(f"seg:{segment_hash}")
        
        research_question = params.get('research_question', '')
        if research_question:
            # Use hash of research question for performance
            rq_hash = hashlib.md5(research_question.encode('utf-8')).hexdigest()[:8]
            components.append(f"rq:{rq_hash}")
        
        # Generate final key
        key_string = "|".join(components)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def invalidate_on_config_change(self) -> List[str]:
        """
        Get operations to invalidate on config change (minimal for performance).
        
        Returns:
            Minimal list of operations to invalidate
        """
        # Only invalidate truly configuration-dependent operations
        return ['relevance_check']
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about performance-optimized strategy.
        
        Returns:
            Strategy information
        """
        return {
            'name': self.name,
            'type': 'performance_optimized',
            'description': 'Performance-optimized caching strategy for high-throughput scenarios',
            'shared_operations': list(self.shared_operations),
            'coder_specific_operations': list(self.coder_specific_operations),
            'supports_multi_coder': self.coder_count > 1,
            'performance_features': {
                'aggressive_sharing': self.aggressive_sharing,
                'cache_everything': len(self.coder_specific_operations) == 0,
                'optimized_key_generation': True,
                'minimal_invalidation': True
            }
        }


class MemoryEfficientCacheStrategy(CacheStrategy):
    """
    Memory-efficient cache strategy that minimizes memory usage.
    
    This strategy is designed for resource-constrained environments where
    memory usage is a primary concern.
    """
    
    def __init__(self, coder_count: int = 1, max_shared_operations: int = 2):
        """
        Initialize memory-efficient cache strategy.
        
        Args:
            coder_count: Number of coders configured
            max_shared_operations: Maximum number of operations to cache as shared
        """
        super().__init__(f"MemoryEfficient_{coder_count}coder{'s' if coder_count != 1 else ''}")
        self.coder_count = coder_count
        self.max_shared_operations = max_shared_operations
        
        # Prioritize most important operations for sharing
        priority_operations = [
            'relevance_check',      # Highest priority - most reusable
            'category_development', # High priority - expensive to compute
            'subcategory_development', # Medium priority
            'subcode_collection'    # Lower priority
        ]
        
        # Select top operations for sharing based on memory constraints
        self.shared_operations = set(priority_operations[:max_shared_operations])
        
        # Everything else is coder-specific or not cached
        all_operations = {
            'relevance_check', 'category_development', 'subcode_collection',
            'subcategory_development', 'coding', 'confidence_scoring', 'manual_coding'
        }
        self.coder_specific_operations = all_operations - self.shared_operations
        
        logger.info(f"Initialized {self.name} with max_shared_operations={max_shared_operations}")
    
    def should_cache_shared(self, operation: str) -> bool:
        """
        Determine if operation should be shared (memory-optimized).
        
        Args:
            operation: Operation type
            
        Returns:
            True if operation should be shared
        """
        return operation in self.shared_operations
    
    def should_cache_per_coder(self, operation: str) -> bool:
        """
        Determine if operation should be cached per coder.
        
        Args:
            operation: Operation type
            
        Returns:
            True if operation should be cached per coder
        """
        # In memory-efficient mode, only cache per coder if single coder
        # or if it's a high-value operation
        if self.coder_count <= 1:
            return operation in self.coder_specific_operations
        
        # For multi-coder, only cache high-value coder-specific operations
        high_value_operations = {'coding', 'manual_coding'}
        return operation in (self.coder_specific_operations & high_value_operations)
    
    def get_cache_key(self, operation: str, coder_id: Optional[str] = None, **params) -> str:
        """
        Generate memory-efficient cache key.
        
        Args:
            operation: Operation type
            coder_id: Coder ID for coder-specific operations
            **params: Additional parameters
            
        Returns:
            Compact cache key string
        """
        import hashlib
        
        # Use compact key generation to minimize memory overhead
        components = [
            operation[:4],  # Abbreviated operation name
            params.get('analysis_mode', '')[:3],  # Abbreviated mode
        ]
        
        # Add coder ID only if needed
        if self.should_cache_per_coder(operation) and coder_id:
            components.append(coder_id[:8])  # Truncated coder ID
        
        # Use hashes for large parameters to save memory
        segment_text = params.get('segment_text', '')
        if segment_text:
            # Use shorter hash for memory efficiency
            segment_hash = hashlib.md5(segment_text.encode('utf-8')).hexdigest()[:6]
            components.append(segment_hash)
        
        # Generate compact key
        key_string = "|".join(components)
        # Use shorter hash for final key
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()[:16]
    
    def invalidate_on_config_change(self) -> List[str]:
        """
        Get operations to invalidate on config change (memory-conscious).
        
        Returns:
            List of operations to invalidate
        """
        # Invalidate all shared operations to free memory
        return list(self.shared_operations)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about memory-efficient strategy.
        
        Returns:
            Strategy information
        """
        return {
            'name': self.name,
            'type': 'memory_efficient',
            'description': 'Memory-efficient caching strategy for resource-constrained environments',
            'shared_operations': list(self.shared_operations),
            'coder_specific_operations': list(self.coder_specific_operations),
            'supports_multi_coder': self.coder_count > 1,
            'memory_features': {
                'max_shared_operations': self.max_shared_operations,
                'compact_keys': True,
                'selective_coder_caching': self.coder_count > 1,
                'aggressive_invalidation': True
            }
        }


class ExampleCustomCachePlugin(CachePluginInterface):
    """
    Example plugin demonstrating custom cache strategies.
    
    This plugin provides two custom strategies:
    1. PerformanceOptimizedCacheStrategy - For high-throughput scenarios
    2. MemoryEfficientCacheStrategy - For resource-constrained environments
    """
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """
        Get information about this example plugin.
        
        Returns:
            Plugin metadata
        """
        return {
            'name': 'ExampleCustomCachePlugin',
            'version': '1.0.0',
            'description': 'Example plugin demonstrating custom cache strategies for different use cases',
            'author': 'QCA-AID Development Team',
            'supported_modes': ['inductive', 'abductive', 'grounded', 'deductive'],
            'strategy_types': ['performance_optimized', 'memory_efficient'],
            'features': [
                'High-performance caching for throughput optimization',
                'Memory-efficient caching for resource constraints',
                'Configurable operation sharing policies',
                'Optimized key generation algorithms'
            ]
        }
    
    def create_strategy(self, strategy_type: str, **kwargs) -> CacheStrategy:
        """
        Create a cache strategy instance.
        
        Args:
            strategy_type: Type of strategy to create
            **kwargs: Additional parameters for strategy creation
            
        Returns:
            Cache strategy instance
            
        Raises:
            ValueError: If strategy_type is not supported
        """
        coder_count = kwargs.get('coder_count', 1)
        
        if strategy_type == 'performance_optimized':
            return PerformanceOptimizedCacheStrategy(
                coder_count=coder_count,
                aggressive_sharing=kwargs.get('aggressive_sharing', True),
                cache_everything=kwargs.get('cache_everything', False)
            )
        elif strategy_type == 'memory_efficient':
            return MemoryEfficientCacheStrategy(
                coder_count=coder_count,
                max_shared_operations=kwargs.get('max_shared_operations', 2)
            )
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate coder_count
        coder_count = config.get('coder_count', 1)
        if not isinstance(coder_count, int) or coder_count < 1:
            errors.append("coder_count must be a positive integer")
        
        # Validate performance_optimized specific config
        if 'aggressive_sharing' in config:
            if not isinstance(config['aggressive_sharing'], bool):
                errors.append("aggressive_sharing must be a boolean")
        
        if 'cache_everything' in config:
            if not isinstance(config['cache_everything'], bool):
                errors.append("cache_everything must be a boolean")
        
        # Validate memory_efficient specific config
        if 'max_shared_operations' in config:
            max_shared = config['max_shared_operations']
            if not isinstance(max_shared, int) or max_shared < 0 or max_shared > 10:
                errors.append("max_shared_operations must be an integer between 0 and 10")
        
        return errors
    
    def get_supported_strategy_types(self) -> List[str]:
        """
        Get list of strategy types supported by this plugin.
        
        Returns:
            List of strategy type names
        """
        return ['performance_optimized', 'memory_efficient']


# Plugin instance for easy importing
plugin_instance = ExampleCustomCachePlugin()