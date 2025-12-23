"""
Cache Plugin Interface
=====================
Plugin interface for extending the dynamic cache system with custom strategies.
Provides a registry system for loading and validating cache strategy plugins.
"""

import logging
import importlib
import importlib.util
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class CachePluginInterface(ABC):
    """
    Abstract base class for cache strategy plugins.
    
    All cache strategy plugins must inherit from this interface and implement
    the required methods for strategy creation and validation.
    """
    
    @abstractmethod
    def get_plugin_info(self) -> Dict[str, Any]:
        """
        Get information about this plugin.
        
        Returns:
            Dictionary with plugin metadata:
            - name: Plugin name
            - version: Plugin version
            - description: Plugin description
            - author: Plugin author
            - supported_modes: List of analysis modes supported
            - strategy_types: List of strategy types provided
        """
        pass
    
    @abstractmethod
    def create_strategy(self, strategy_type: str, **kwargs):
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
        pass
    
    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        pass
    
    @abstractmethod
    def get_supported_strategy_types(self) -> List[str]:
        """
        Get list of strategy types supported by this plugin.
        
        Returns:
            List of strategy type names
        """
        pass


class CacheStrategyRegistry:
    """
    Registry for managing cache strategy plugins and their lifecycle.
    
    Provides functionality for:
    - Loading plugins from files or modules
    - Validating plugin implementations
    - Creating strategy instances from plugins
    - Managing plugin lifecycle
    """
    
    def __init__(self):
        """Initialize the cache strategy registry."""
        self.plugins: Dict[str, CachePluginInterface] = {}
        self.strategy_mappings: Dict[str, str] = {}  # strategy_type -> plugin_name
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.validation_errors: Dict[str, List[str]] = {}
        
        logger.info("CacheStrategyRegistry initialized")
    
    def register_plugin(self, plugin_name: str, plugin: CachePluginInterface, 
                       config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a cache strategy plugin.
        
        Args:
            plugin_name: Unique name for the plugin
            plugin: Plugin instance implementing CachePluginInterface
            config: Optional configuration for the plugin
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Basic validation
            if not isinstance(plugin, CachePluginInterface):
                self.validation_errors[plugin_name] = ["Plugin must inherit from CachePluginInterface"]
                return False
            
            # Validate plugin configuration if provided
            if config:
                config_errors = plugin.validate_configuration(config)
                if config_errors:
                    self.validation_errors[plugin_name] = config_errors
                    logger.error(f"Plugin configuration validation failed for '{plugin_name}': {config_errors}")
                    return False
                self.plugin_configs[plugin_name] = config.copy()
            
            # Check for strategy type conflicts
            supported_types = plugin.get_supported_strategy_types()
            conflicts = []
            for strategy_type in supported_types:
                if strategy_type in self.strategy_mappings:
                    existing_plugin = self.strategy_mappings[strategy_type]
                    conflicts.append(f"Strategy type '{strategy_type}' already provided by plugin '{existing_plugin}'")
            
            if conflicts:
                self.validation_errors[plugin_name] = conflicts
                logger.error(f"Plugin registration failed for '{plugin_name}': {conflicts}")
                return False
            
            # Register plugin and update mappings
            self.plugins[plugin_name] = plugin
            for strategy_type in supported_types:
                self.strategy_mappings[strategy_type] = plugin_name
            
            # Clear any previous validation errors
            if plugin_name in self.validation_errors:
                del self.validation_errors[plugin_name]
            
            plugin_info = plugin.get_plugin_info()
            logger.info(f"Successfully registered plugin '{plugin_name}' v{plugin_info.get('version', 'unknown')} "
                       f"providing strategies: {supported_types}")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to register plugin '{plugin_name}': {str(e)}"
            self.validation_errors[plugin_name] = [error_msg]
            logger.error(error_msg)
            return False
    
    def create_strategy(self, strategy_type: str, **kwargs):
        """
        Create a cache strategy instance using registered plugins.
        
        Args:
            strategy_type: Type of strategy to create
            **kwargs: Additional parameters for strategy creation
            
        Returns:
            Cache strategy instance or None if not found
        """
        try:
            if strategy_type not in self.strategy_mappings:
                logger.warning(f"Strategy type '{strategy_type}' not found in registry")
                return None
            
            plugin_name = self.strategy_mappings[strategy_type]
            plugin = self.plugins[plugin_name]
            
            # Get plugin configuration
            plugin_config = self.plugin_configs.get(plugin_name, {})
            
            # Merge plugin config with kwargs
            merged_kwargs = {**plugin_config, **kwargs}
            
            # Create strategy
            strategy = plugin.create_strategy(strategy_type, **merged_kwargs)
            
            logger.info(f"Created strategy '{strategy_type}' using plugin '{plugin_name}'")
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to create strategy '{strategy_type}': {e}")
            return None
    
    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available strategies.
        
        Returns:
            Dictionary mapping strategy types to their information
        """
        strategies = {}
        
        for strategy_type, plugin_name in self.strategy_mappings.items():
            plugin = self.plugins[plugin_name]
            plugin_info = plugin.get_plugin_info()
            
            strategies[strategy_type] = {
                'plugin_name': plugin_name,
                'plugin_version': plugin_info.get('version', 'unknown'),
                'description': plugin_info.get('description', 'No description'),
                'author': plugin_info.get('author', 'Unknown'),
                'supported_modes': plugin_info.get('supported_modes', [])
            }
        
        return strategies
    
    def get_registry_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the plugin registry.
        
        Returns:
            Dictionary with registry status information
        """
        return {
            'total_plugins': len(self.plugins),
            'total_strategies': len(self.strategy_mappings),
            'plugins': list(self.plugins.keys()),
            'available_strategies': list(self.strategy_mappings.keys()),
            'plugins_with_errors': list(self.validation_errors.keys()),
            'strategy_mappings': dict(self.strategy_mappings),
            'validation_summary': {
                plugin_name: len(errors) for plugin_name, errors in self.validation_errors.items()
            }
        }
    
    def load_plugin_from_file(self, plugin_path: str, plugin_name: str, 
                             config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load a plugin from a Python file.
        
        Args:
            plugin_path: Path to the plugin Python file
            plugin_name: Name to register the plugin under
            config: Optional configuration for the plugin
            
        Returns:
            True if loading and registration successful, False otherwise
        """
        try:
            import importlib.util
            import sys
            from pathlib import Path
            
            plugin_file = Path(plugin_path)
            if not plugin_file.exists():
                error_msg = f"Plugin file not found: {plugin_path}"
                self.validation_errors[plugin_name] = [error_msg]
                logger.error(error_msg)
                return False
            
            # Load module from file
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec is None or spec.loader is None:
                error_msg = f"Could not create module spec for: {plugin_path}"
                self.validation_errors[plugin_name] = [error_msg]
                logger.error(error_msg)
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class in module
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, CachePluginInterface) and 
                    attr != CachePluginInterface):
                    plugin_class = attr
                    break
            
            if plugin_class is None:
                error_msg = f"No CachePluginInterface implementation found in: {plugin_path}"
                self.validation_errors[plugin_name] = [error_msg]
                logger.error(error_msg)
                return False
            
            # Create plugin instance
            plugin_instance = plugin_class()
            
            # Register the plugin
            success = self.register_plugin(plugin_name, plugin_instance, config)
            
            if success:
                logger.info(f"Successfully loaded plugin '{plugin_name}' from {plugin_path}")
            else:
                logger.error(f"Failed to register loaded plugin '{plugin_name}'")
            
            return success
            
        except Exception as e:
            error_msg = f"ℹ️ Kein Cache-Plugin gefunden: {plugin_name}"
            self.validation_errors[plugin_name] = [str(e)]
            logger.info(error_msg)  # Use info instead of error for missing plugins
            return False


# Global registry instance
_global_registry = CacheStrategyRegistry()


def get_global_registry() -> CacheStrategyRegistry:
    """
    Get the global cache strategy registry instance.
    
    Returns:
        Global registry instance
    """
    return _global_registry


def register_plugin(plugin_name: str, plugin: CachePluginInterface, 
                   config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Register a plugin with the global registry.
    
    Args:
        plugin_name: Unique name for the plugin
        plugin: Plugin instance
        config: Optional configuration
        
    Returns:
        True if registration successful
    """
    return _global_registry.register_plugin(plugin_name, plugin, config)