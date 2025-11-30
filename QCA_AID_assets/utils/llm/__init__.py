"""
LLM Provider Package

Handles communication with different LLM APIs:
- OpenAI (GPT-4, GPT-5 family)
- Anthropic (Claude family)
- Mistral
- OpenRouter (unified access to multiple providers)
- Local (LM Studio, Ollama)

Features:
- Async/await support
- Temperature parameter capability detection
- JSON response format handling
- Response wrapper for unified interface
- Provider manager for multi-provider support

Exports:
  - LLMProvider: Abstract base class for all providers
  - OpenAIProvider: OpenAI implementation
  - AnthropicProvider: Anthropic implementation
  - MistralProvider: Mistral implementation
  - OpenRouterProvider: OpenRouter implementation
  - LocalProvider: Local LLM server implementation
  - LLMProviderFactory: Factory for creating providers
  - LLMResponse: Response wrapper for unified interface
  - NormalizedModel: Unified model format for all providers
  - CacheMetadata: Cache metadata for TTL management
  - ProviderLoadError: Exception for provider loading errors
  - ValidationError: Exception for validation errors
  - CacheError: Exception for cache-related errors
  - CacheManager: Manages local caching with TTL
  - ProviderLoader: Loads provider configs from Catwalk or local files
  - ModelRegistry: Manages and filters normalized models
  - LocalDetector: Detects local LLM models from LM Studio and Ollama
  - LLMProviderManager: Main class for managing multiple LLM providers
"""

from .base import LLMProvider
from .response import LLMResponse
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .mistral_provider import MistralProvider
from .openrouter_provider import OpenRouterProvider
from .local_provider import LocalProvider
from .factory import LLMProviderFactory
from .models import (
    NormalizedModel,
    CacheMetadata,
    ProviderLoadError,
    ValidationError,
    CacheError
)
from .cache_manager import CacheManager
from .provider_loader import ProviderLoader
from .model_registry import ModelRegistry
from .local_detector import LocalDetector
from .provider_manager import LLMProviderManager

__all__ = [
    'LLMProvider',
    'LLMResponse',
    'OpenAIProvider',
    'AnthropicProvider',
    'MistralProvider',
    'OpenRouterProvider',
    'LocalProvider',
    'LLMProviderFactory',
    'NormalizedModel',
    'CacheMetadata',
    'ProviderLoadError',
    'ValidationError',
    'CacheError',
    'CacheManager',
    'ProviderLoader',
    'ModelRegistry',
    'LocalDetector',
    'LLMProviderManager',
]
