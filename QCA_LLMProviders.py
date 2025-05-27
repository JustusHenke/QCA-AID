"""
LLM Provider Module für QCA-AID
Unterstützt OpenAI und Mistral APIs mit einheitlicher Schnittstelle
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Lade Umgebungsvariablen
env_path = os.path.join(os.path.expanduser("~"), '.environ.env')
load_dotenv(env_path)

@dataclass
class LLMResponse:
    """Wrapper für LLM-Antworten um einheitliche Schnittstelle zu gewährleisten"""
    content: str
    model: str = ""
    usage: Dict = None
    
    def __init__(self, response):
        """
        Initialisiert LLMResponse basierend auf dem Provider-Response-Format
        
        Args:
            response: Rohe Antwort vom LLM Provider (OpenAI oder Mistral Format)
        """
        if hasattr(response, 'choices') and response.choices:
            # OpenAI Format
            self.content = response.choices[0].message.content
            self.model = getattr(response, 'model', '')
            self.usage = getattr(response, 'usage', None)
        elif hasattr(response, 'content'):
            # Mistral Format
            self.content = response.content
            self.model = getattr(response, 'model', '')
            self.usage = getattr(response, 'usage', None)
        else:
            # Fallback für unbekannte Formate
            self.content = str(response)
            self.model = "unknown"
            self.usage = None

class LLMProvider(ABC):
    """Abstrakte Basisklasse für LLM Provider"""
    
    def __init__(self):
        self.client = None
        self.model_name = None
        self.initialize_client()
    
    @abstractmethod
    def initialize_client(self):
        """Initialisiert den Client für den jeweiligen Provider"""
        pass
    
    @abstractmethod
    async def create_completion(self, 
                              model: str,
                              messages: List[Dict],
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              response_format: Optional[Dict] = None) -> Any:
        """Erstellt eine Chat Completion"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI Provider Implementation"""
    
    def initialize_client(self):
        """Initialisiert den OpenAI Client"""
        try:
            from openai import AsyncOpenAI
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY nicht in Umgebungsvariablen gefunden")
            
            self.client = AsyncOpenAI(api_key=api_key)
            print("OpenAI Client erfolgreich initialisiert")
            
        except ImportError:
            raise ImportError("OpenAI Bibliothek nicht installiert. Bitte installieren Sie: pip install openai")
        except Exception as e:
            raise Exception(f"Fehler bei OpenAI Client-Initialisierung: {str(e)}")
    
    async def create_completion(self, 
                              model: str,
                              messages: List[Dict],
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              response_format: Optional[Dict] = None) -> Any:
        """
        Erstellt eine OpenAI Chat Completion
        
        Args:
            model: Name des zu verwendenden Modells
            messages: Liste der Chat-Nachrichten
            temperature: Temperatur für die Antwortgenerierung
            max_tokens: Maximale Anzahl von Tokens (optional)
            response_format: Format der Antwort (optional, z.B. {"type": "json_object"})
            
        Returns:
            OpenAI ChatCompletion Response
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
            if response_format:
                params['response_format'] = response_format
            
            # API Call
            response = await self.client.chat.completions.create(**params)
            return response
            
        except Exception as e:
            print(f"Fehler bei OpenAI API Call: {str(e)}")
            raise

class MistralProvider(LLMProvider):
    """Mistral Provider Implementation"""
    
    def initialize_client(self):
        """Initialisiert den Mistral Client"""
        try:
            from mistralai import Mistral
            
            api_key = os.getenv('MISTRAL_API_KEY')
            if not api_key:
                raise ValueError("MISTRAL_API_KEY nicht in Umgebungsvariablen gefunden")
            
            self.client = Mistral(api_key=api_key)
            print("Mistral Client erfolgreich initialisiert")
            
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
        Erstellt eine Mistral Chat Completion
        
        Args:
            model: Name des zu verwendenden Modells
            messages: Liste der Chat-Nachrichten
            temperature: Temperatur für die Antwortgenerierung
            max_tokens: Maximale Anzahl von Tokens (optional)
            response_format: Format der Antwort (optional, wird bei Mistral ignoriert)
            
        Returns:
            Mistral ChatCompletion Response
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
                print("Warnung: response_format wird von Mistral möglicherweise nicht unterstützt")
            
            # API Call (synchron, da Mistral möglicherweise kein async unterstützt)
            response = await self._make_async_call(params)
            return response
            
        except Exception as e:
            print(f"Fehler bei Mistral API Call: {str(e)}")
            raise
    
    async def _make_async_call(self, params):
        """Wrapper um synchrone Mistral Calls asynchron zu machen"""
        try:
            # Führe synchronen Call in Thread Pool aus
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.chat.complete(**params)
            )
            return response
        except Exception as e:
            print(f"Fehler bei async Mistral Call: {str(e)}")
            raise

class LLMProviderFactory:
    """Factory Klasse zur Erstellung von LLM Providern"""
    
    @staticmethod
    def create_provider(provider_name: str) -> LLMProvider:
        """
        Erstellt einen LLM Provider basierend auf dem Namen
        
        Args:
            provider_name: Name des Providers ('openai' oder 'mistral')
            
        Returns:
            LLMProvider: Initialisierter Provider
            
        Raises:
            ValueError: Wenn ein unbekannter Provider angefordert wird
        """
        provider_name = provider_name.lower().strip()
        
        print(f"Initialisiere LLM Provider: {provider_name}")
        
        try:
            if provider_name in ['openai', 'gpt']:
                return OpenAIProvider()
            elif provider_name in ['mistral', 'mistralai']:
                return MistralProvider()
            else:
                raise ValueError(f"Unbekannter LLM Provider: {provider_name}. "
                               f"Unterstützte Provider: 'openai', 'mistral'")
                
        except Exception as e:
            print(f"Fehler bei Provider-Erstellung: {str(e)}")
            raise

# Beispiel für die Verwendung:
async def test_provider():
    """Testfunktion für die Provider"""
    try:
        # OpenAI Provider testen
        openai_provider = LLMProviderFactory.create_provider('openai')
        
        messages = [
            {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
            {"role": "user", "content": "Hallo, wie geht es dir?"}
        ]
        
        response = await openai_provider.create_completion(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.7
        )
        
        # Verwende LLMResponse Wrapper
        llm_response = LLMResponse(response)
        print(f"Antwort: {llm_response.content}")
        
    except Exception as e:
        print(f"Test fehlgeschlagen: {str(e)}")

if __name__ == "__main__":
    # Test ausführen
    asyncio.run(test_provider())