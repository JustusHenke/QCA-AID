"""
LLM Provider Package

Handles communication with different LLM APIs:
- OpenAI (GPT-4, GPT-5 family)
- Mistral

Features:
- Async/await support
- Temperature parameter capability detection
- JSON response format handling
- Response wrapper for unified interface

Exports:
  - LLMProvider: Abstract base class for all providers
  - OpenAIProvider: OpenAI implementation
  - MistralProvider: Mistral implementation
  - LLMProviderFactory: Factory for creating providers
  - LLMResponse: Response wrapper for unified interface
"""

from .base import LLMProvider
from .response import LLMResponse
from .openai_provider import OpenAIProvider
from .mistral_provider import MistralProvider
from .factory import LLMProviderFactory

__all__ = [
    'LLMProvider',
    'LLMResponse',
    'OpenAIProvider',
    'MistralProvider',
    'LLMProviderFactory',
]
