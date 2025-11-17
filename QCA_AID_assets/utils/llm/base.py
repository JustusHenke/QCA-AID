"""
Abstract LLM Provider Base Class

Defines the interface that all LLM provider implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class LLMProvider(ABC):
    """
    Abstrakte Basisklasse für LLM Provider.
    
    Defines the interface that all LLM provider implementations (OpenAI, Mistral, etc.)
    must implement to provide async chat completions with consistent parameter handling.
    
    Attributes:
        client: The initialized LLM provider client
        model_name: Name of the model being used (for capability tracking)
    """
    
    def __init__(self) -> None:
        """Initialize the provider"""
        self.client = None
        self.model_name = None
        self.initialize_client()
    
    @abstractmethod
    def initialize_client(self) -> None:
        """
        Initialisiert den Client für den jeweiligen Provider.
        
        Must be implemented by subclasses to set up the actual LLM client
        (OpenAI AsyncOpenAI, Mistral client, etc.)
        
        Raises:
            ImportError: If required libraries are not installed
            ValueError: If required credentials are missing
            Exception: For other initialization errors
        """
        pass
    
    @abstractmethod
    async def create_completion(self,
                              model: str,
                              messages: List[Dict],
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              response_format: Optional[Dict] = None) -> Any:
        """
        Erstellt eine Chat Completion.
        
        Creates an async chat completion request to the LLM provider.
        Implementations should handle parameter compatibility and fallback strategies.
        
        Args:
            model: Name des zu verwendenden Modells
            messages: Liste der Chat-Nachrichten im Format [{"role": "...", "content": "..."}]
            temperature: Temperatur für die Antwortgenerierung (0.0-2.0, provider-dependent)
            max_tokens: Maximale Anzahl von Tokens in der Antwort (optional)
            response_format: Format der Antwort, z.B. {"type": "json_object"} (optional)
            
        Returns:
            Provider-specific response object (will be wrapped in LLMResponse)
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If API call fails
        """
        pass
    
    def test_model_capabilities(self, model: str) -> None:
        """
        Testet die Capabilities eines Models (z.B. temperature-Parameter).
        
        Default-Implementierung: Kann von Subclasses überschrieben werden.
        This method allows providers to test model capabilities on first use
        and cache the results for efficient fallback strategies.
        
        Args:
            model: Name des Modells zum Testen
            
        Note:
            This is called during initialization but the actual testing happens
            during the first API call in create_completion().
        """
        # Default: keine Tests durchführen
        pass
