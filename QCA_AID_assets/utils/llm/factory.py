"""
LLM Provider Factory

Factory pattern for creating LLM provider instances based on provider name.
"""

from typing import Optional

from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .mistral_provider import MistralProvider


class LLMProviderFactory:
    """
    Factory Klasse zur Erstellung von LLM Providern.
    
    Uses factory pattern to dynamically instantiate the correct provider
    (OpenAI, Mistral, etc.) based on string identifier.
    """
    
    @staticmethod
    def create_provider(provider_name: str, model_name: Optional[str] = None) -> LLMProvider:
        """
        Erstellt einen LLM Provider basierend auf dem Namen.
        
        Supports multiple names for each provider (e.g., 'openai' and 'gpt' both
        create OpenAIProvider). Optionally tests model capabilities on first use.
        
        Args:
            provider_name: Name des Providers ('openai'/'gpt' oder 'mistral'/'mistralai')
            model_name: Optional - Model Name zum Testen der Capabilities
            
        Returns:
            LLMProvider: Initialisierter Provider (OpenAIProvider oder MistralProvider)
            
        Raises:
            ValueError: Wenn ein unbekannter Provider angefordert wird
            Exception: If provider initialization fails (missing API keys, etc.)
            
        Example:
            >>> factory = LLMProviderFactory()
            >>> provider = factory.create_provider('openai', model_name='gpt-4o-mini')
            >>> # provider is now an OpenAIProvider instance with test_model_capabilities called
        """
        provider_name = provider_name.lower().strip()
        
        print(f"🔧 Initialisiere LLM Provider: {provider_name}")
        
        try:
            if provider_name in ['openai', 'gpt']:
                provider = OpenAIProvider()
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
                
            else:
                raise ValueError(
                    f"❌ Unbekannter LLM Provider: {provider_name}. "
                    f"Unterstützte Provider: 'openai' (oder 'gpt'), 'mistral' (oder 'mistralai')"
                )
                
        except Exception as e:
            print(f"❌ [ERROR] Fehler bei Provider-Erstellung: {str(e)}")
            raise
