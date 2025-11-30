"""
OpenRouter Provider Implementation

Async wrapper for OpenRouter API which provides access to multiple LLM providers
through a unified OpenAI-compatible interface.
"""

import os
from typing import List, Dict, Optional, Any

from .base import LLMProvider


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter Provider Implementation.
    
    Handles async chat completions with OpenRouter's API, which provides
    access to multiple providers (OpenAI, Anthropic, Mistral, etc.) through
    a single API key and OpenAI-compatible interface.
    
    OpenRouter uses model IDs like:
    - openai/gpt-4o
    - anthropic/claude-3-opus
    - mistral/mistral-large
    """
    
    def initialize_client(self) -> None:
        """
        Initialisiert den OpenRouter Client.
        
        Sets up the AsyncOpenAI client configured for OpenRouter's endpoint.
        
        Raises:
            ImportError: If openai library not installed
            ValueError: If OPENROUTER_API_KEY environment variable not found
            Exception: For other initialization errors
        """
        try:
            from openai import AsyncOpenAI
            
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY nicht in Umgebungsvariablen gefunden")
            
            # OpenRouter uses OpenAI-compatible API with custom base URL
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            print("✅ OpenRouter Client erfolgreich initialisiert")
            
        except ImportError:
            raise ImportError("OpenAI Bibliothek nicht installiert. Bitte installieren Sie: pip install openai")
        except Exception as e:
            raise Exception(f"Fehler bei OpenRouter Client-Initialisierung: {str(e)}")
    
    async def create_completion(self,
                              model: str,
                              messages: List[Dict],
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              response_format: Optional[Dict] = None) -> Any:
        """
        Erstellt eine OpenRouter Chat Completion.
        
        OpenRouter uses OpenAI-compatible API, so parameter handling is similar.
        Model IDs should include provider prefix (e.g., 'mistral/mistral-large').
        
        Args:
            model: Name des zu verwendenden Modells (z.B. 'mistral/mistral-large', 'anthropic/claude-3-opus')
            messages: Liste der Chat-Nachrichten im Format [{"role": "...", "content": "..."}]
            temperature: Temperatur für die Antwortgenerierung (0.0-2.0)
            max_tokens: Maximale Anzahl von Tokens (optional)
            response_format: Format der Antwort, z.B. {"type": "json_object"} (optional)
            
        Returns:
            OpenAI-compatible ChatCompletion Response object
            
        Raises:
            Exception: If API call fails
        """
        from ..tracking.token_tracker import get_global_token_counter
        token_counter = get_global_token_counter()
        
        try:
            # Prüfe Capability-Cache für dieses Model
            supports_temperature = token_counter.model_capabilities.get(model, None)
            
            # Erstelle Parameter-Dict
            params = {
                'model': model,
                'messages': messages
            }
            
            # Temperature handling mit Capability-Check
            if supports_temperature is None:
                # Keine Information vorhanden - versuche mit temperature
                if temperature is not None:
                    params['temperature'] = temperature
            elif supports_temperature:
                # Model unterstützt temperature - nutze es
                if temperature is not None:
                    params['temperature'] = temperature
            # else: Model unterstützt temperature nicht - füge es nicht hinzu
            
            # Füge optionale Parameter hinzu
            if max_tokens:
                params['max_tokens'] = max_tokens
            
            # Füge response_format nur hinzu wenn explizit übergeben
            if response_format is not None:
                params['response_format'] = response_format
            
            # API Call mit Fehlerbehandlung
            try:
                response = await self.client.chat.completions.create(**params)
                # Wenn erfolgreich und temperature war im Request und noch nicht getestet
                if 'temperature' in params and supports_temperature is None:
                    token_counter.model_capabilities[model] = True
                    print(f"✅ Model {model} unterstützt temperature-Parameter")
                return response
                
            except Exception as api_error:
                # Prüfe ob der Fehler temperature-bezogen ist
                error_msg = str(api_error)
                
                if 'temperature' in error_msg.lower() and 'temperature' in params:
                    # Markiere Model als nicht-unterstützt
                    token_counter.model_capabilities[model] = False
                    print(f"⚠️  Model {model} unterstützt temperature-Parameter NICHT. Retry ohne...")
                    
                    # Versuche erneut ohne temperature
                    params_without_temp = {k: v for k, v in params.items() if k != 'temperature'}
                    try:
                        response = await self.client.chat.completions.create(**params_without_temp)
                        return response
                    except Exception as retry_error:
                        print(f"❌ ‼️ Auch Retry ohne temperature fehlgeschlagen: {str(retry_error)}")
                        raise retry_error
                else:
                    # Anderer Fehler - weiterleiten
                    raise api_error
            return response
            
        except Exception as e:
            print(f"❌ ‼️ Fehler bei OpenRouter API Call: {str(e)}")
            raise
