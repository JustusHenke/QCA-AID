"""
Cache Strategy Plugins
=====================
Directory for cache strategy plugins that extend the dynamic cache system.
"""

from .example_custom_strategy import ExampleCustomCachePlugin, plugin_instance

__all__ = ['ExampleCustomCachePlugin', 'plugin_instance']
