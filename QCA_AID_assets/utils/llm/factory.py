"""
LLM Provider Factory

Factory pattern for creating LLM provider instances based on provider name.
"""

from typing import Optional

from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .mistral_provider import MistralProvider
from .anthropic_provider import AnthropicProvider
from .openrouter_provider import OpenRouterProvider
from .local_provider import LocalProvider


class LLMProviderFactory:
    """
    Factory Klasse zur Erstellung von LLM Providern.
    
    Uses factory pattern to dynamically instantiate the correct provider
    (OpenAI, Mistral, etc.) based on string identifier.
    """
    
    @staticmethod
    def create_provider(provider_name: str, 
                       model_name: Optional[str] = None,
                       api_key: Optional[str] = None,
                       base_url: Optional[str] = None) -> LLMProvider:
        """
        Erstellt einen LLM Provider basierend auf dem Namen.
        
        Supports multiple names for each provider and allows optional API key
        and base URL overrides.
        
        Args:
            provider_name: Name des Providers ('openai', 'anthropic', 'mistral', 'openrouter', 'local')
            model_name: Optional - Model Name zum Testen der Capabilities
            api_key: Optional - API Key (überschreibt Umgebungsvariable)
            base_url: Optional - Base URL (nur für 'local' Provider)
            
        Returns:
            LLMProvider: Initialisierter Provider
            
        Raises:
            ValueError: Wenn ein unbekannter Provider angefordert wird
            Exception: If provider initialization fails (missing API keys, etc.)
            
        Example:
            >>> factory = LLMProviderFactory()
            >>> provider = factory.create_provider('openai', model_name='gpt-4o-mini')
            >>> # provider is now an OpenAIProvider instance with test_model_capabilities called
            >>> 
            >>> # Create local provider with custom URL
            >>> local_provider = factory.create_provider('local', base_url='http://localhost:11434/v1')
        """
        provider_name = provider_name.lower().strip()
        
        print(f"🔧 Initialisiere LLM Provider: {provider_name}")
        
        try:
            # Handle API key override if provided
            if api_key:
                import os
                # Temporarily set environment variable for this provider
                if provider_name in ['openai', 'gpt']:
                    os.environ['OPENAI_API_KEY'] = api_key
                elif provider_name in ['anthropic', 'claude']:
                    os.environ['ANTHROPIC_API_KEY'] = api_key
                elif provider_name in ['mistral', 'mistralai']:
                    os.environ['MISTRAL_API_KEY'] = api_key
                elif provider_name == 'openrouter':
                    os.environ['OPENROUTER_API_KEY'] = api_key
            
            # Create provider based on name
            if provider_name in ['openai', 'gpt']:
                provider = OpenAIProvider()
                # Teste Model-Capabilities falls model_name vorhanden
                if model_name:
                    provider.test_model_capabilities(model_name)
                return provider
                
            elif provider_name in ['anthropic', 'claude']:
                provider = AnthropicProvider()
                # Teste Model-Capabilities falls model_name vorhanden
                if model_name:
                    provider.test_model_capabilities(model_name)
                return provider
                
            elif provider_name in ['mistral', 'mistralai']:
                provider = MistralProvider()
                # Teste Model-Capabilities falls model_name vorhanden
                if model_name:
                    provider.test_model_capabilities(model_name)
                return provider
                
            elif provider_name == 'openrouter':
                provider = OpenRouterProvider()
                # Teste Model-Capabilities falls model_name vorhanden
                if model_name:
                    provider.test_model_capabilities(model_name)
                return provider
                
            elif provider_name == 'local':
                provider = LocalProvider(base_url=base_url)
                # Teste Model-Capabilities falls model_name vorhanden
                if model_name:
                    provider.test_model_capabilities(model_name)
                return provider
                
            else:
                raise ValueError(
                    f"❌ Unbekannter LLM Provider: {provider_name}. "
                    f"Unterstützte Provider: 'openai' (oder 'gpt'), 'anthropic' (oder 'claude'), "
                    f"'mistral' (oder 'mistralai'), 'openrouter', 'local'"
                )
                
        except Exception as e:
            print(f"❌ ‼️ Fehler bei Provider-Erstellung: {str(e)}")
            raise
