"""
Test Script for Cache Plugin System
===================================
Simple test to validate the plugin interface and example plugin functionality.
"""

import logging
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from QCA_AID_assets.optimization.cache_plugin_interface import CacheStrategyRegistry, get_global_registry
from QCA_AID_assets.optimization.plugins.example_custom_strategy import ExampleCustomCachePlugin
from QCA_AID_assets.optimization.cache_strategies import CacheStrategyFactory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_plugin_interface():
    """Test the basic plugin interface functionality."""
    logger.info("Testing plugin interface...")
    
    # Create plugin instance
    plugin = ExampleCustomCachePlugin()
    
    # Test plugin info
    info = plugin.get_plugin_info()
    logger.info(f"Plugin info: {info}")
    
    # Test supported strategy types
    strategy_types = plugin.get_supported_strategy_types()
    logger.info(f"Supported strategies: {strategy_types}")
    
    # Test configuration validation
    valid_config = {
        'coder_count': 2,
        'aggressive_sharing': True,
        'max_shared_operations': 3
    }
    errors = plugin.validate_configuration(valid_config)
    logger.info(f"Valid config errors: {errors}")
    
    invalid_config = {
        'coder_count': -1,
        'aggressive_sharing': 'not_a_bool',
        'max_shared_operations': 20
    }
    errors = plugin.validate_configuration(invalid_config)
    logger.info(f"Invalid config errors: {errors}")
    
    # Test strategy creation
    try:
        strategy = plugin.create_strategy('performance_optimized', coder_count=2, aggressive_sharing=True)
        logger.info(f"Created strategy: {strategy.name}")
        
        strategy_info = strategy.get_strategy_info()
        logger.info(f"Strategy info: {strategy_info}")
        
    except Exception as e:
        logger.error(f"Failed to create strategy: {e}")
    
    logger.info("Plugin interface test completed")


def test_registry():
    """Test the plugin registry functionality."""
    logger.info("Testing plugin registry...")
    
    # Get global registry
    registry = get_global_registry()
    
    # Register plugin
    plugin = ExampleCustomCachePlugin()
    success = registry.register_plugin('example_plugin', plugin)
    logger.info(f"Plugin registration success: {success}")
    
    # Get registry status
    status = registry.get_registry_status()
    logger.info(f"Registry status: {status}")
    
    # Get available strategies
    strategies = registry.get_available_strategies()
    logger.info(f"Available strategies: {strategies}")
    
    # Test strategy creation through registry
    try:
        strategy = registry.create_strategy('performance_optimized', coder_count=3)
        logger.info(f"Created strategy through registry: {strategy.name}")
    except Exception as e:
        logger.error(f"Failed to create strategy through registry: {e}")
    
    # Test plugin validation
    validation_results = registry.validate_all_plugins()
    logger.info(f"Plugin validation results: {validation_results}")
    
    logger.info("Registry test completed")


def test_factory_integration():
    """Test integration with CacheStrategyFactory."""
    logger.info("Testing factory integration...")
    
    # Set up registry with factory
    registry = get_global_registry()
    plugin = ExampleCustomCachePlugin()
    registry.register_plugin('example_plugin', plugin)
    
    CacheStrategyFactory.set_plugin_registry(registry)
    
    # Test factory strategy creation
    try:
        # Test plugin strategy creation
        strategy = CacheStrategyFactory.create_strategy(
            coder_count=2,
            strategy_type='performance_optimized',
            aggressive_sharing=True
        )
        logger.info(f"Factory created plugin strategy: {strategy.name}")
        
        # Test fallback to built-in strategy
        strategy = CacheStrategyFactory.create_strategy(
            coder_count=1,
            strategy_type='single_coder'
        )
        logger.info(f"Factory created built-in strategy: {strategy.name}")
        
    except Exception as e:
        logger.error(f"Factory integration test failed: {e}")
    
    # Test available strategies
    available = CacheStrategyFactory.get_available_strategies()
    logger.info(f"Factory available strategies: {list(available.keys())}")
    
    logger.info("Factory integration test completed")


def test_plugin_loading():
    """Test loading plugins from files."""
    logger.info("Testing plugin loading from files...")
    
    registry = get_global_registry()
    
    # Test loading from module
    try:
        plugins_dir = Path(__file__).parent / 'plugins'
        plugin_file = plugins_dir / 'example_custom_strategy.py'
        
        if plugin_file.exists():
            success = registry.load_plugin_from_file(str(plugin_file), 'loaded_example')
            logger.info(f"Plugin loading from file success: {success}")
            
            if success:
                status = registry.get_registry_status()
                logger.info(f"Registry status after loading: {status}")
        else:
            logger.warning(f"Plugin file not found: {plugin_file}")
    
    except Exception as e:
        logger.error(f"Plugin loading test failed: {e}")
    
    logger.info("Plugin loading test completed")


def main():
    """Run all plugin system tests."""
    logger.info("Starting cache plugin system tests...")
    
    try:
        test_plugin_interface()
        print("-" * 50)
        
        test_registry()
        print("-" * 50)
        
        test_factory_integration()
        print("-" * 50)
        
        test_plugin_loading()
        print("-" * 50)
        
        logger.info("All plugin system tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Plugin system tests failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())