"""
Mistral Provider Implementation

Async wrapper for Mistral API with synchronous-to-async conversion.
"""

import os
import asyncio
from typing import List, Dict, Optional, Any

from .base import LLMProvider


class MistralProvider(LLMProvider):
    """
    Mistral Provider Implementation.
    
    Handles async chat completions with Mistral's API. Since Mistral's
    client is synchronous, this wraps calls in a thread pool executor.
    
    Note: Mistral has limited parameter support compared to OpenAI.
    Response format parameter is ignored.
    """
    
    def initialize_client(self) -> None:
        """
        Initialisiert den Mistral Client.
        
        Sets up the Mistral client with API key from environment.
        
        Raises:
            ImportError: If mistralai library not installed
            ValueError: If MISTRAL_API_KEY environment variable not found
            Exception: For other initialization errors
        """
        try:
            from mistralai import Mistral
            
            api_key = os.getenv('MISTRAL_API_KEY')
            if not api_key:
                raise ValueError("MISTRAL_API_KEY nicht in Umgebungsvariablen gefunden")
            
            self.client = Mistral(api_key=api_key)
            print("✅ Mistral Client erfolgreich initialisiert")
            
        except ImportError:
            raise ImportError("Mistral Bibliothek nicht installiert. Bitte installieren Sie: pip install mistralai")
        except Exception as e:
            raise Exception(f"Fehler bei Mistral Client-Initialisierung: {str(e)}")
    
    async def create_completion(self,
                              model: str,
                              messages: List[Dict],
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              response_format: Optional[Dict] = None) -> Any:
        """
        Erstellt eine Mistral Chat Completion.
        
        Note: response_format parameter is not supported by Mistral and is ignored.
        
        Args:
            model: Name des zu verwendenden Modells
            messages: Liste der Chat-Nachrichten im Format [{"role": "...", "content": "..."}]
            temperature: Temperatur für die Antwortgenerierung (0.0-1.0)
            max_tokens: Maximale Anzahl von Tokens (optional)
            response_format: Format der Antwort (wird bei Mistral ignoriert)
            
        Returns:
            Mistral ChatCompletion Response object
            
        Raises:
            Exception: If API call fails
        """
        try:
            # Erstelle Parameter-Dict
            params = {
                'model': model,
                'messages': messages,
                'temperature': temperature
            }
            
            # Füge optionale Parameter hinzu
            if max_tokens:
                params['max_tokens'] = max_tokens
            
            # Hinweis: Mistral unterstützt response_format möglicherweise nicht
            if response_format:
                print("⚠️  Warnung: response_format wird von Mistral möglicherweise nicht unterstützt")
            
            # API Call wrapped in async executor (Mistral client ist synchron)
            response = await self._make_async_call(params)
            return response
            
        except Exception as e:
            print(f"❌ [ERROR] Fehler bei Mistral API Call: {str(e)}")
            raise
    
    async def _make_async_call(self, params: Dict) -> Any:
        """
        Wrapper um synchrone Mistral Calls asynchron zu machen.
        
        Mistral's client library is synchronous, so we run it in a thread pool
        to avoid blocking the event loop.
        
        Args:
            params: Parameters to pass to Mistral API
            
        Returns:
            Response from Mistral API
            
        Raises:
            Exception: If the call fails
        """
        try:
            # Führe synchronen Call in Thread Pool aus
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.complete(**params)
            )
            return response
        except Exception as e:
            print(f"❌ [ERROR] Fehler bei async Mistral Call: {str(e)}")
            raise
