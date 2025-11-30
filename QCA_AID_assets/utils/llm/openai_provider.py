"""
OpenAI Provider Implementation

Async wrapper for OpenAI API with intelligent parameter handling and
model capability detection for temperature and response_format support.
"""

import os
import asyncio
from typing import List, Dict, Optional, Any

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    OpenAI Provider Implementation.
    
    Handles async chat completions with OpenAI's API, including:
    - Intelligent temperature parameter handling (detects model support)
    - Response format fallback (JSON mode not supported on all models)
    - Async/await support for non-blocking I/O
    - Capability caching to avoid repeated failed attempts
    """
    
    def initialize_client(self) -> None:
        """
        Initialisiert den OpenAI Client.
        
        Sets up the AsyncOpenAI client with API key from environment.
        
        Raises:
            ImportError: If openai library not installed
            ValueError: If OPENAI_API_KEY environment variable not found
            Exception: For other initialization errors
        """
        try:
            from openai import AsyncOpenAI
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY nicht in Umgebungsvariablen gefunden")
            
            self.client = AsyncOpenAI(api_key=api_key)
            print("✅ OpenAI Client erfolgreich initialisiert")
            
        except ImportError:
            raise ImportError("OpenAI Bibliothek nicht installiert. Bitte installieren Sie: pip install openai")
        except Exception as e:
            raise Exception(f"Fehler bei OpenAI Client-Initialisierung: {str(e)}")
    
    def test_model_capabilities(self, model: str) -> None:
        """
        Markiert ein Model für Capability-Testing beim ersten API Call.
        
        Der Test wird asynchron beim ersten create_completion durchgeführt
        und die Ergebnisse werden gecacht für zukünftige Aufrufe.
        
        Args:
            model: Name des Modells zum Testen
        """
        # Diese Methode wird vom LLMProviderFactory aufgerufen
        # Die Implementierung ist in token_counter.model_capabilities
        pass
    
    async def create_completion(self,
                              model: str,
                              messages: List[Dict],
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              response_format: Optional[Dict] = None) -> Any:
        """
        Erstellt eine OpenAI Chat Completion mit intelligenter Parameterbehandlung.
        
        Handles model capability detection:
        1. First attempt: Try with temperature + response_format (if provided)
        2. If temperature error: Retry without temperature
        3. If response_format error: Retry without response_format
        4. Final fallback: Only model, messages, max_tokens
        
        Args:
            model: Name des zu verwendenden Modells (z.B. 'gpt-4o-mini', 'gpt-5-nano')
            messages: Liste der Chat-Nachrichten im Format [{"role": "...", "content": "..."}]
            temperature: Temperatur für die Antwortgenerierung (0.0-2.0)
            max_tokens: Maximale Anzahl von Tokens (optional)
            response_format: Format der Antwort, z.B. {"type": "json_object"} (optional)
            
        Returns:
            OpenAI ChatCompletion Response object
            
        Raises:
            Exception: If all retry attempts fail
        """
        try:
            # Import globale token_counter Instanz
            from ..tracking.token_tracker import get_global_token_counter
            token_counter = get_global_token_counter()
            
            # Erstelle Parameter-Dict
            params = {
                'model': model,
                'messages': messages,
            }

            # Prüfe Capability-Cache für dieses Model
            supports_temperature = token_counter.model_capabilities.get(model, None)

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
                        # Zweiter Versuch auch fehlgeschlagen - versuche auch ohne response_format
                        retry_error_msg = str(retry_error)
                        if 'response_format' in retry_error_msg.lower():
                            print(f"⚠️  Model {model} unterstützt auch response_format nicht. Retry ohne beide...")
                            params_minimal = {k: v for k, v in params.items() if k not in ['temperature', 'response_format']}
                            try:
                                response = await self.client.chat.completions.create(**params_minimal)
                                return response
                            except Exception as minimal_error:
                                print(f"❌ ‼️ Auch minimal-Versuch fehlgeschlagen: {str(minimal_error)[:200]}")
                                token_counter.track_error(self.model_name)
                                raise
                        else:
                            # Anderer Fehler
                            print(f"❌ ‼️ Retry ohne temperature fehlgeschlagen: {retry_error_msg[:200]}")
                            token_counter.track_error(self.model_name)
                            raise
                else:
                    # Anderer Fehler - logge und rethrow
                    print(f"❌ ‼️ API-Fehler (nicht temperature-bezogen): {error_msg[:200]}")
                    token_counter.track_error(self.model_name)
                    raise
                    
        except Exception as e:
            print(f"❌ ‼️ Unerwarteter Fehler in create_completion: {str(e)[:200]}")
            try:
                from ...analysis.deductive_coding import token_counter
                token_counter.track_error(self.model_name)
            except:
                pass
            raise
