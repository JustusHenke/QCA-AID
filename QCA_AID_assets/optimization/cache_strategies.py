"""
Cache Strategy Pattern Implementation
====================================
Implements different caching strategies for single-coder and multi-coder scenarios.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""
    
    def __init__(self, name: str):
        """
        Initialize cache strategy.
        
        Args:
            name: Name of the strategy
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def should_cache_shared(self, operation: str) -> bool:
        """
        Determine if operation should use shared cache.
        
        Args:
            operation: Operation type
            
        Returns:
            True if operation should be shared across coders
        """
        pass
    
    @abstractmethod
    def should_cache_per_coder(self, operation: str) -> bool:
        """
        Determine if operation should use per-coder cache.
        
        Args:
            operation: Operation type
            
        Returns:
            True if operation should be cached per coder
        """
        pass
    
    @abstractmethod
    def get_cache_key(self, operation: str, coder_id: Optional[str] = None, **params) -> str:
        """
        Generate cache key for operation.
        
        Args:
            operation: Operation type
            coder_id: Optional coder ID
            **params: Additional parameters
            
        Returns:
            Cache key string
        """
        pass
    
    @abstractmethod
    def invalidate_on_config_change(self) -> List[str]:
        """
        Get list of operation types that should be invalidated on config change.
        
        Returns:
            List of operation types to invalidate
        """
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about this strategy.
        
        Returns:
            Dictionary with strategy information
        """
        pass


class SingleCoderCacheStrategy(CacheStrategy):
    """Cache strategy for single coder scenarios."""
    
    def __init__(self):
        """Initialize single coder cache strategy."""
        super().__init__("SingleCoder")
        
        # All operations are treated as shared in single-coder mode
        self.all_operations = {
            'relevance_check',
            'category_development',
            'subcode_collection',
            'subcategory_development',
            'coding',
            'confidence_scoring',
            'manual_coding'
        }
    
    def should_cache_shared(self, operation: str) -> bool:
        """
        In single-coder mode, all operations are effectively shared.
        
        Args:
            operation: Operation type
            
        Returns:
            True for all operations
        """
        return operation in self.all_operations
    
    def should_cache_per_coder(self, operation: str) -> bool:
        """
        In single-coder mode, no operations need per-coder caching.
        
        Args:
            operation: Operation type
            
        Returns:
            False for all operations
        """
        return False
    
    def get_cache_key(self, operation: str, coder_id: Optional[str] = None, **params) -> str:
        """
        Generate cache key without coder ID for single-coder mode.
        
        Args:
            operation: Operation type
            coder_id: Ignored in single-coder mode
            **params: Additional parameters
            
        Returns:
            Cache key string
        """
        import hashlib
        import json
        
        # Build key components without coder ID
        components = [
            operation,
            params.get('analysis_mode', ''),
            (params.get('segment_text') or '').strip().lower(),
            (params.get('research_question') or '').strip().lower(),
        ]
        
        # Add category definitions if present
        category_definitions = params.get('category_definitions')
        if category_definitions:
            sorted_cats = sorted(category_definitions.items())
            components.append(json.dumps(sorted_cats, sort_keys=True))
        
        # Add coding rules if present
        coding_rules = params.get('coding_rules')
        if coding_rules:
            sorted_rules = sorted([r.strip().lower() for r in coding_rules])
            components.append(json.dumps(sorted_rules, sort_keys=True))
        
        # Generate hash
        key_string = "|".join(components)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def invalidate_on_config_change(self) -> List[str]:
        """
        Get operations to invalidate on config change.
        
        Returns:
            All operations since they might be affected by config changes
        """
        return list(self.all_operations)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about single coder strategy.
        
        Returns:
            Strategy information
        """
        return {
            'name': self.name,
            'type': 'single_coder',
            'description': 'Standard caching for single coder scenarios',
            'shared_operations': list(self.all_operations),
            'coder_specific_operations': [],
            'supports_multi_coder': False
        }


class MultiCoderCacheStrategy(CacheStrategy):
    """Cache strategy for multi-coder scenarios with intelligent shared/specific caching."""
    
    def __init__(self):
        """Initialize multi-coder cache strategy."""
        super().__init__("MultiCoder")
        
        # Operations that should be shared across all coders
        self.shared_operations = {
            'relevance_check',
            'category_development',
            'subcode_collection',
            'subcategory_development'
        }
        
        # Operations that should be cached per coder
        self.coder_specific_operations = {
            'coding',
            'confidence_scoring',
            'manual_coding'
        }
    
    def should_cache_shared(self, operation: str) -> bool:
        """
        Determine if operation should be shared across coders.
        
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
        Generate cache key with or without coder ID based on operation type.
        
        Args:
            operation: Operation type
            coder_id: Coder ID for coder-specific operations
            **params: Additional parameters
            
        Returns:
            Cache key string
        """
        import hashlib
        import json
        
        # Build key components
        components = [
            operation,
            params.get('analysis_mode', ''),
            (params.get('segment_text') or '').strip().lower(),
            (params.get('research_question') or '').strip().lower(),
        ]
        
        # Add coder ID for coder-specific operations
        if self.should_cache_per_coder(operation) and coder_id:
            components.append(f"coder:{coder_id}")
        
        # Add category definitions if present
        category_definitions = params.get('category_definitions')
        if category_definitions:
            sorted_cats = sorted(category_definitions.items())
            components.append(json.dumps(sorted_cats, sort_keys=True))
        
        # Add coding rules if present
        coding_rules = params.get('coding_rules')
        if coding_rules:
            sorted_rules = sorted([r.strip().lower() for r in coding_rules])
            components.append(json.dumps(sorted_rules, sort_keys=True))
        
        # Generate hash
        key_string = "|".join(components)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def invalidate_on_config_change(self) -> List[str]:
        """
        Get operations to invalidate on config change.
        
        Returns:
            Operations that might be affected by configuration changes
        """
        # Shared operations are more likely to be affected by config changes
        return list(self.shared_operations)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about multi-coder strategy.
        
        Returns:
            Strategy information
        """
        return {
            'name': self.name,
            'type': 'multi_coder',
            'description': 'Intelligent caching for multi-coder scenarios with shared/specific operations',
            'shared_operations': list(self.shared_operations),
            'coder_specific_operations': list(self.coder_specific_operations),
            'supports_multi_coder': True
        }


class ModeSpecificCacheStrategy(CacheStrategy):
    """Cache strategy that adapts behavior based on analysis mode and coder configuration."""
    
    def __init__(self, analysis_mode: str, coder_count: int):
        """
        Initialize mode-specific cache strategy.
        
        Args:
            analysis_mode: Analysis mode (inductive, abductive, grounded, deductive)
            coder_count: Number of coders configured
        """
        super().__init__(f"ModeSpecific_{analysis_mode}_{coder_count}coder{'s' if coder_count != 1 else ''}")
        self.analysis_mode = analysis_mode.lower()
        self.coder_count = coder_count
        
        # Define mode-specific operation classifications
        self._define_mode_operations()
        
        self.logger.info(f"Initialized mode-specific cache strategy for {analysis_mode} mode with {coder_count} coder(s)")
    
    def _define_mode_operations(self):
        """Define which operations should be shared vs coder-specific for each mode."""
        
        if self.analysis_mode == 'inductive':
            # Inductive Mode: Category development shared, coding per coder
            self.shared_operations = {
                'relevance_check',
                'category_development',  # Shared category development
                'subcode_collection'
            }
            self.coder_specific_operations = {
                'coding',  # Each coder codes independently
                'confidence_scoring',
                'manual_coding'
            }
            
        elif self.analysis_mode == 'abductive':
            # Abductive Mode: Subcategory development shared, coding per coder
            self.shared_operations = {
                'relevance_check',
                'category_development',  # Base categories shared
                'subcategory_development',  # Subcategory extension shared
                'subcode_collection'
            }
            self.coder_specific_operations = {
                'coding',  # Each coder codes with extended categories
                'confidence_scoring',
                'manual_coding'
            }
            
        elif self.analysis_mode == 'grounded':
            # Grounded Mode: Subcode collection shared, coding per coder in phase 3
            self.shared_operations = {
                'relevance_check',
                'subcode_collection'  # Collect subcodes across all segments
            }
            self.coder_specific_operations = {
                'coding',  # Phase 3: Each coder applies generated categories
                'manual_coding'  # Manual coding allowed
            }
            
        elif self.analysis_mode == 'deductive':
            # Deductive Mode: Relevance check shared, coding per coder
            self.shared_operations = {
                'relevance_check'  # Shared relevance assessment
            }
            self.coder_specific_operations = {
                'coding',  # Each coder applies predefined categories
                'confidence_scoring',
                'manual_coding'
            }
            
        else:
            # Default/unknown mode: Conservative approach
            self.shared_operations = {
                'relevance_check'
            }
            self.coder_specific_operations = {
                'coding',
                'confidence_scoring',
                'manual_coding',
                'category_development',
                'subcategory_development',
                'subcode_collection'
            }
            self.logger.warning(f"Unknown analysis mode '{self.analysis_mode}', using conservative caching")
    
    def should_cache_shared(self, operation: str) -> bool:
        """
        Determine if operation should be shared based on analysis mode.
        
        Args:
            operation: Operation type
            
        Returns:
            True if operation should be shared across coders
        """
        # In single-coder scenarios, treat everything as shared for efficiency
        if self.coder_count <= 1:
            return operation in (self.shared_operations | self.coder_specific_operations)
        
        # Multi-coder scenarios: respect methodological principles
        return operation in self.shared_operations
    
    def should_cache_per_coder(self, operation: str) -> bool:
        """
        Determine if operation should be cached per coder based on analysis mode.
        
        Args:
            operation: Operation type
            
        Returns:
            True if operation should be cached per coder
        """
        # Single-coder scenarios don't need per-coder caching
        if self.coder_count <= 1:
            return False
        
        # Multi-coder scenarios: cache per coder for coder-specific operations
        return operation in self.coder_specific_operations
    
    def get_cache_key(self, operation: str, coder_id: Optional[str] = None, **params) -> str:
        """
        Generate cache key with mode-specific considerations.
        
        Args:
            operation: Operation type
            coder_id: Coder ID for coder-specific operations
            **params: Additional parameters
            
        Returns:
            Cache key string
        """
        import hashlib
        import json
        
        # Build key components
        components = [
            operation,
            self.analysis_mode,  # Include analysis mode in key
            params.get('analysis_mode', ''),
            (params.get('segment_text') or '').strip().lower(),
            (params.get('research_question') or '').strip().lower(),
        ]
        
        # Add coder ID for coder-specific operations in multi-coder scenarios
        if self.coder_count > 1 and self.should_cache_per_coder(operation) and coder_id:
            components.append(f"coder:{coder_id}")
        
        # Add mode-specific parameters
        if self.analysis_mode == 'grounded':
            # In grounded mode, include subcode collection state
            subcodes = params.get('existing_subcodes', [])
            if subcodes:
                components.append(f"subcodes:{len(subcodes)}")
        
        # Add category definitions if present
        category_definitions = params.get('category_definitions')
        if category_definitions:
            sorted_cats = sorted(category_definitions.items())
            components.append(json.dumps(sorted_cats, sort_keys=True))
        
        # Add coding rules if present
        coding_rules = params.get('coding_rules')
        if coding_rules:
            sorted_rules = sorted([r.strip().lower() for r in coding_rules])
            components.append(json.dumps(sorted_rules, sort_keys=True))
        
        # Generate hash
        key_string = "|".join(components)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def invalidate_on_config_change(self) -> List[str]:
        """
        Get operations to invalidate on config change based on analysis mode.
        
        Returns:
            Operations that should be invalidated when configuration changes
        """
        # Mode-specific invalidation rules
        if self.analysis_mode == 'inductive':
            # Category development might be affected by config changes
            return ['category_development', 'relevance_check']
        elif self.analysis_mode == 'abductive':
            # Both category and subcategory development affected
            return ['category_development', 'subcategory_development', 'relevance_check']
        elif self.analysis_mode == 'grounded':
            # Subcode collection affected by config changes
            return ['subcode_collection', 'relevance_check']
        elif self.analysis_mode == 'deductive':
            # Relevance check most affected in deductive mode
            return ['relevance_check']
        else:
            # Conservative: invalidate shared operations
            return list(self.shared_operations)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about mode-specific strategy.
        
        Returns:
            Strategy information
        """
        return {
            'name': self.name,
            'type': 'mode_specific',
            'analysis_mode': self.analysis_mode,
            'coder_count': self.coder_count,
            'description': f'Mode-specific caching for {self.analysis_mode} analysis with {self.coder_count} coder(s)',
            'shared_operations': list(self.shared_operations),
            'coder_specific_operations': list(self.coder_specific_operations),
            'supports_multi_coder': self.coder_count > 1,
            'methodological_principles': self._get_methodological_principles()
        }
    
    def _get_methodological_principles(self) -> Dict[str, str]:
        """
        Get methodological principles for this analysis mode.
        
        Returns:
            Dictionary describing methodological principles
        """
        principles = {
            'inductive': {
                'category_development': 'Categories developed inductively from data, shared across coders',
                'coding': 'Each coder applies developed categories independently for reliability',
                'cache_strategy': 'Share category development, separate coding per coder'
            },
            'abductive': {
                'category_development': 'Base categories extended with subcategories, shared development',
                'coding': 'Each coder applies extended categories independently',
                'cache_strategy': 'Share category and subcategory development, separate coding per coder'
            },
            'grounded': {
                'subcode_collection': 'Collect subcodes across all segments without predefined categories',
                'coding': 'No automatic coding in phase 1, only manual coding allowed',
                'cache_strategy': 'Share subcode collection, no automatic coding cache'
            },
            'deductive': {
                'relevance_check': 'Assess relevance with predefined categories, shared assessment',
                'coding': 'Each coder applies predefined categories independently',
                'cache_strategy': 'Share relevance assessment, separate coding per coder'
            }
        }
        
        return principles.get(self.analysis_mode, {
            'unknown_mode': 'Conservative caching approach for unknown analysis mode'
        })


class CacheStrategyFactory:
    """Factory for creating cache strategies based on configuration with plugin support."""
    
    _strategies = {
        'single_coder': SingleCoderCacheStrategy,
        'multi_coder': MultiCoderCacheStrategy,
        'mode_specific': ModeSpecificCacheStrategy
    }
    
    _plugin_registry = None
    
    @classmethod
    def set_plugin_registry(cls, registry):
        """
        Set the plugin registry for accessing plugin-provided strategies.
        
        Args:
            registry: CacheStrategyRegistry instance
        """
        cls._plugin_registry = registry
        logger.info("Plugin registry set for CacheStrategyFactory")
    
    @classmethod
    def create_strategy(cls, coder_count: int, strategy_type: Optional[str] = None, 
                       analysis_mode: Optional[str] = None, **kwargs) -> CacheStrategy:
        """
        Create appropriate cache strategy based on coder count and analysis mode.
        
        Args:
            coder_count: Number of coders configured
            strategy_type: Optional explicit strategy type
            analysis_mode: Optional analysis mode for mode-specific strategies
            **kwargs: Additional parameters for strategy creation
            
        Returns:
            Cache strategy instance
        """
        # First try to create from plugins if registry is available
        if strategy_type and cls._plugin_registry:
            plugin_strategy = cls._plugin_registry.create_strategy(
                strategy_type, 
                coder_count=coder_count, 
                analysis_mode=analysis_mode,
                **kwargs
            )
            if plugin_strategy:
                logger.info(f"Created plugin strategy: {plugin_strategy.name}")
                return plugin_strategy
        
        # Fall back to built-in strategies
        if strategy_type:
            if strategy_type not in cls._strategies:
                logger.warning(f"Unknown strategy type: {strategy_type}, falling back to auto-selection")
                strategy_type = None
        
        if not strategy_type:
            # Auto-select strategy based on coder count and analysis mode
            if analysis_mode and analysis_mode.lower() in ['inductive', 'abductive', 'grounded', 'deductive']:
                strategy_type = 'mode_specific'
            elif coder_count <= 1:
                strategy_type = 'single_coder'
            else:
                strategy_type = 'multi_coder'
        
        strategy_class = cls._strategies[strategy_type]
        
        # Create strategy with appropriate parameters
        if strategy_type == 'mode_specific':
            if not analysis_mode:
                logger.warning("Mode-specific strategy requested but no analysis mode provided, using 'deductive'")
                analysis_mode = 'deductive'
            strategy = strategy_class(analysis_mode, coder_count)
        else:
            strategy = strategy_class()
        
        logger.info(f"Created cache strategy: {strategy.name} for {coder_count} coder(s)" + 
                   (f" in {analysis_mode} mode" if analysis_mode else ""))
        return strategy
    
    @classmethod
    def get_available_strategies(cls) -> Dict[str, str]:
        """
        Get available strategy types and their descriptions.
        
        Returns:
            Dictionary mapping strategy type to description
        """
        strategies = {}
        
        # Add built-in strategies
        for strategy_type, strategy_class in cls._strategies.items():
            if strategy_type == 'mode_specific':
                # Create a temporary instance for mode_specific
                temp_strategy = strategy_class('deductive', 1)
            else:
                temp_strategy = strategy_class()
            info = temp_strategy.get_strategy_info()
            strategies[strategy_type] = info['description']
        
        # Add plugin strategies if registry is available
        if cls._plugin_registry:
            plugin_strategies = cls._plugin_registry.get_available_strategies()
            for strategy_type, strategy_info in plugin_strategies.items():
                strategies[strategy_type] = strategy_info['description']
        
        return strategies
    
    @classmethod
    def get_strategy_info(cls, strategy_type: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific strategy type.
        
        Args:
            strategy_type: Strategy type to get information for
            
        Returns:
            Strategy information dictionary or None if not found
        """
        # Check plugins first
        if cls._plugin_registry:
            plugin_strategies = cls._plugin_registry.get_available_strategies()
            if strategy_type in plugin_strategies:
                return plugin_strategies[strategy_type]
        
        # Check built-in strategies
        if strategy_type in cls._strategies:
            strategy_class = cls._strategies[strategy_type]
            if strategy_type == 'mode_specific':
                temp_strategy = strategy_class('deductive', 1)
            else:
                temp_strategy = strategy_class()
            return temp_strategy.get_strategy_info()
        
        return None
    
    @classmethod
    def register_strategy(cls, strategy_type: str, strategy_class: type):
        """
        Register a new cache strategy type.
        
        Args:
            strategy_type: Type identifier for the strategy
            strategy_class: Strategy class that inherits from CacheStrategy
        """
        if not issubclass(strategy_class, CacheStrategy):
            raise ValueError(f"Strategy class must inherit from CacheStrategy")
        
        cls._strategies[strategy_type] = strategy_class
        logger.info(f"Registered new cache strategy: {strategy_type}")
    
    @classmethod
    def load_plugins_from_directory(cls, plugins_dir: str) -> Dict[str, bool]:
        """
        Load all plugins from a directory.
        
        Args:
            plugins_dir: Directory containing plugin files
            
        Returns:
            Dictionary mapping plugin names to load success status
        """
        if not cls._plugin_registry:
            logger.error("No plugin registry set, cannot load plugins")
            return {}
        
        import os
        from pathlib import Path
        
        results = {}
        plugins_path = Path(plugins_dir)
        
        if not plugins_path.exists():
            logger.warning(f"Plugins directory does not exist: {plugins_dir}")
            return results
        
        # Load all Python files in the plugins directory
        for plugin_file in plugins_path.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue  # Skip __init__.py and similar files
            
            plugin_name = plugin_file.stem
            success = cls._plugin_registry.load_plugin_from_file(
                str(plugin_file), 
                plugin_name
            )
            results[plugin_name] = success
            
            if success:
                logger.info(f"Successfully loaded plugin: {plugin_name}")
            else:
                logger.info(f"ℹ️ Kein Cache-Plugin gefunden: {plugin_name}")
        
        return results
    
    @classmethod
    def get_plugin_registry_status(cls) -> Dict[str, Any]:
        """
        Get status of the plugin registry.
        
        Returns:
            Registry status information
        """
        if not cls._plugin_registry:
            return {
                'registry_available': False,
                'error': 'No plugin registry set'
            }
        
        status = cls._plugin_registry.get_registry_status()
        status['registry_available'] = True
        return status