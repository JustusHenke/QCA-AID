"""
Local Provider Implementation

Async wrapper for local LLM servers (LM Studio, Ollama) which provide
OpenAI-compatible APIs.
"""

import os
from typing import List, Dict, Optional, Any

from .base import LLMProvider


class LocalProvider(LLMProvider):
    """
    Local Provider Implementation.
    
    Handles async chat completions with local LLM servers:
    - LM Studio: http://localhost:1234/v1 (OpenAI-compatible)
    - Ollama: http://localhost:11434/v1 (OpenAI-compatible)
    
    No API key required for local servers.
    """
    
    def __init__(self, base_url: Optional[str] = None) -> None:
        """
        Initialize LocalProvider with optional custom base URL.
        
        Args:
            base_url: Custom base URL for local server (defaults to LM Studio URL)
        """
        self.base_url = base_url or "http://localhost:1234/v1"
        super().__init__()
    
    def initialize_client(self) -> None:
        """
        Initialisiert den Local Client.
        
        Sets up the AsyncOpenAI client configured for local server endpoint.
        No API key required.
        
        Raises:
            ImportError: If openai library not installed
            Exception: For other initialization errors
        """
        try:
            from openai import AsyncOpenAI
            
            # Local servers don't require API keys, but OpenAI client requires one
            # Use a dummy key for local servers
            self.client = AsyncOpenAI(
                api_key="not-needed",
                base_url=self.base_url
            )
            print(f"✅ Local Client erfolgreich initialisiert (URL: {self.base_url})")
            
        except ImportError:
            raise ImportError("OpenAI Bibliothek nicht installiert. Bitte installieren Sie: pip install openai")
        except Exception as e:
            raise Exception(f"Fehler bei Local Client-Initialisierung: {str(e)}")
    
    async def create_completion(self,
                              model: str,
                              messages: List[Dict],
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              response_format: Optional[Dict] = None) -> Any:
        """
        Erstellt eine Local Chat Completion.
        
        Local servers (LM Studio, Ollama) use OpenAI-compatible API.
        Parameter support may vary by server and model.
        
        Args:
            model: Name des zu verwendenden Modells (z.B. 'llama-3.1-8b', 'mistral-7b')
            messages: Liste der Chat-Nachrichten im Format [{"role": "...", "content": "..."}]
            temperature: Temperatur für die Antwortgenerierung (0.0-2.0)
            max_tokens: Maximale Anzahl von Tokens (optional)
            response_format: Format der Antwort (may not be supported by all local models)
            
        Returns:
            OpenAI-compatible ChatCompletion Response object
            
        Raises:
            Exception: If API call fails (e.g., server not running)
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
            # else: Model unterstützt temperature nicht - Füge es nicht hinzu
            
            # Füge optionale Parameter hinzu
            if max_tokens:
                params['max_tokens'] = max_tokens
            
            # Füge response_format nur hinzu wenn explizit übergeben.
            # Viele lokalen Server implementieren bereits den neueren 'json_schema'-Standard
            # und lehnen das ältere 'json_object' Format ab. Wir konvertieren daher
            # automatisch zu einem kompatiblen Schema, das beliebige JSON-Objekte erlaubt.
            if response_format is not None:
                rf = dict(response_format)
                if rf.get('type') == 'json_object':
                    rf = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "generic_json_response",
                            "schema": {
                                "type": "object",
                                "additionalProperties": True
                            }
                        }
                    }
                params['response_format'] = rf
            
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
            
        except Exception as e:
            error_msg = str(e)
            if 'connection' in error_msg.lower() or 'refused' in error_msg.lower():
                print(f"❌ ‼️ Lokaler Server nicht erreichbar unter {self.base_url}")
            else:
                print(f"❌ ‼️ Fehler bei Local API Call: {error_msg}")
            raise
