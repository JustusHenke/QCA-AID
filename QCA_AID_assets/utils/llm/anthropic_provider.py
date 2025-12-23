"""
Anthropic Provider Implementation

Async wrapper for Anthropic API with intelligent parameter handling.
"""

import os
from typing import List, Dict, Optional, Any

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    """
    Anthropic Provider Implementation.
    
    Handles async chat completions with Anthropic's API (Claude models).
    Anthropic uses a different message format and parameter structure
    compared to OpenAI.
    """
    
    def initialize_client(self) -> None:
        """
        Initialisiert den Anthropic Client.
        
        Sets up the AsyncAnthropic client with API key from environment.
        
        Raises:
            ImportError: If anthropic library not installed
            ValueError: If ANTHROPIC_API_KEY environment variable not found
            Exception: For other initialization errors
        """
        try:
            from anthropic import AsyncAnthropic
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY nicht in Umgebungsvariablen gefunden")
            
            self.client = AsyncAnthropic(api_key=api_key)
            print("✅ Anthropic Client erfolgreich initialisiert")
            
        except ImportError:
            raise ImportError("Anthropic Bibliothek nicht installiert. Bitte installieren Sie: pip install anthropic")
        except Exception as e:
            raise Exception(f"Fehler bei Anthropic Client-Initialisierung: {str(e)}")
    
    async def create_completion(self,
                              model: str,
                              messages: List[Dict],
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              response_format: Optional[Dict] = None) -> Any:
        """
        Erstellt eine Anthropic Chat Completion.
        
        Note: Anthropic requires max_tokens to be specified and uses a different
        message format. System messages are handled separately.
        
        Args:
            model: Name des zu verwendenden Modells (z.B. 'claude-3-opus', 'claude-3-sonnet')
            messages: Liste der Chat-Nachrichten im Format [{"role": "...", "content": "..."}]
            temperature: Temperatur für die Antwortgenerierung (0.0-1.0)
            max_tokens: Maximale Anzahl von Tokens (required by Anthropic, defaults to 4096)
            response_format: Format der Antwort (wird bei Anthropic ignoriert)
            
        Returns:
            Anthropic Message Response object
            
        Raises:
            Exception: If API call fails
        """
        from ..tracking.token_tracker import get_global_token_counter
        token_counter = get_global_token_counter()
        
        try:
            # Anthropic requires max_tokens
            if max_tokens is None:
                max_tokens = 4096
            
            # Extract system message if present
            system_message = None
            filtered_messages = []
            
            for msg in messages:
                if msg.get('role') == 'system':
                    system_message = msg.get('content', '')
                else:
                    filtered_messages.append(msg)
            
            # Prüfe Capability-Cache für dieses Model
            supports_temperature = token_counter.model_capabilities.get(model, None)
            
            # Erstelle Parameter-Dict
            params = {
                'model': model,
                'messages': filtered_messages,
                'max_tokens': max_tokens
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
            
            # Add system message if present
            if system_message:
                params['system'] = system_message
            
            # Hinweis: Anthropic unterstützt response_format nicht direkt
            if response_format:
                print("⚠️  Warnung: response_format wird von Anthropic nicht direkt unterstützt")
            
            # API Call mit Fehlerbehandlung
            try:
                response = await self.client.messages.create(**params)
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
                        response = await self.client.messages.create(**params_without_temp)
                        return response
                    except Exception as retry_error:
                        print(f"❌ ‼️ Auch Retry ohne temperature fehlgeschlagen: {str(retry_error)}")
                        raise retry_error
                else:
                    # Anderer Fehler - weiterleiten
                    raise api_error
            
        except Exception as e:
            print(f"❌ ‼️ Fehler bei Anthropic API Call: {str(e)}")
            raise
