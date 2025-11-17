"""
LLM Response Wrapper

Provides a unified interface for responses from different LLM providers
(OpenAI, Mistral, etc.) to ensure consistent access to response content.
"""

from typing import Dict, Optional, Any


class LLMResponse:
    """
    Wrapper für LLM-Antworten um einheitliche Schnittstelle zu gewährleisten.
    
    Handles responses from multiple LLM providers (OpenAI, Mistral, etc.)
    and provides a consistent interface regardless of provider format.
    
    Attributes:
        content: The text content of the response
        model: Name of the model that generated the response
        usage: Token usage statistics (if available)
    """
    
    content: str
    model: str = ""
    usage: Optional[Dict] = None
    
    def __init__(self, response: Any) -> None:
        """
        Initialisiert LLMResponse basierend auf dem Provider-Response-Format.
        
        Detects the response format and extracts content, model name, and usage
        information in a provider-agnostic way.
        
        Args:
            response: Rohe Antwort vom LLM Provider (OpenAI oder Mistral Format)
            
        Raises:
            None - Falls das Format nicht erkannt wird, wird ein Fallback verwendet
        """
        if hasattr(response, 'choices') and response.choices:
            # OpenAI Format (standard ChatCompletion response)
            self.content = response.choices[0].message.content
            self.model = getattr(response, 'model', '')
            self.usage = getattr(response, 'usage', None)
        elif hasattr(response, 'content'):
            # Mistral Format (direct content attribute)
            self.content = response.content
            self.model = getattr(response, 'model', '')
            self.usage = getattr(response, 'usage', None)
        else:
            # Fallback für unbekannte Formate
            self.content = str(response)
            self.model = "unknown"
            self.usage = None
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"LLMResponse(model={self.model}, content_length={len(self.content)}, usage={self.usage})"
