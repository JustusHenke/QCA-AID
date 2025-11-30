"""
LLM Response Wrapper

Provides a unified interface for responses from different LLM providers
(OpenAI, Anthropic, Mistral, etc.) to ensure consistent access to response content.
"""

from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class LLMResponse:
    """
    Wrapper für LLM-Antworten um einheitliche Schnittstelle zu gewährleisten.
    
    Handles responses from multiple LLM providers (OpenAI, Anthropic, Mistral, etc.)
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
            response: Rohe Antwort vom LLM Provider (OpenAI, Anthropic oder Mistral Format)
            
        Raises:
            None - Falls das Format nicht erkannt wird, wird ein Fallback verwendet
        """
        try:
            if hasattr(response, 'choices') and response.choices:
                # OpenAI Format (standard ChatCompletion response)
                self.content = response.choices[0].message.content
                self.model = getattr(response, 'model', '')
                self.usage = getattr(response, 'usage', None)
                logger.debug(f"Parsed OpenAI response: model={self.model}")
                
            elif hasattr(response, 'content') and isinstance(response.content, list):
                # Anthropic Format (content is a list of ContentBlock objects)
                # Extract text from first text block
                if response.content and len(response.content) > 0:
                    first_block = response.content[0]
                    if hasattr(first_block, 'text'):
                        self.content = first_block.text
                    elif hasattr(first_block, 'content'):
                        self.content = first_block.content
                    else:
                        self.content = str(first_block)
                else:
                    self.content = ""
                    logger.warning("Anthropic response has empty content list")
                    
                self.model = getattr(response, 'model', '')
                
                # Anthropic usage format: response.usage with input_tokens and output_tokens
                if hasattr(response, 'usage'):
                    usage = response.usage
                    self.usage = {
                        'prompt_tokens': getattr(usage, 'input_tokens', 0),
                        'completion_tokens': getattr(usage, 'output_tokens', 0),
                        'total_tokens': getattr(usage, 'input_tokens', 0) + getattr(usage, 'output_tokens', 0)
                    }
                else:
                    self.usage = None
                    
                logger.debug(f"Parsed Anthropic response: model={self.model}, content_length={len(self.content)}")
                
            elif hasattr(response, 'content') and isinstance(response.content, str):
                # Mistral Format (direct content attribute as string)
                self.content = response.content
                self.model = getattr(response, 'model', '')
                self.usage = getattr(response, 'usage', None)
                logger.debug(f"Parsed Mistral response: model={self.model}")
                
            else:
                # Fallback für unbekannte Formate
                logger.warning(f"Unknown response format: {type(response)}, attributes: {dir(response)}")
                self.content = str(response)
                self.model = "unknown"
                self.usage = None
                
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response type: {type(response)}")
            logger.error(f"Response attributes: {dir(response) if hasattr(response, '__dir__') else 'N/A'}")
            # Fallback
            self.content = str(response)
            self.model = "error"
            self.usage = None
    
    def extract_json(self) -> str:
        """
        Extrahiert JSON aus dem Response-Content.
        
        Anthropic (Claude) gibt oft JSON in Markdown-Code-Blöcken zurück:
        ```json
        {"key": "value"}
        ```
        
        Diese Methode extrahiert das reine JSON.
        
        Returns:
            str: Bereinigter JSON-String
        """
        import re
        
        content = self.content.strip()
        
        # Prüfe ob Content mit ```json oder ``` beginnt
        if content.startswith('```'):
            # Extrahiere JSON aus Code-Block
            # Pattern: ```json\n{...}\n``` oder ```\n{...}\n```
            pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                logger.debug(f"Extracted JSON from markdown code block (length: {len(extracted)})")
                return extracted
        
        # Wenn kein Code-Block gefunden, versuche JSON direkt zu finden
        # Suche nach { ... } oder [ ... ]
        json_pattern = r'(\{.*\}|\[.*\])'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            logger.debug(f"Extracted JSON from content (length: {len(extracted)})")
            return extracted
        
        # Fallback: Gib Original-Content zurück
        logger.debug("No JSON extraction needed, returning original content")
        return content
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"LLMResponse(model={self.model}, content_length={len(self.content)}, usage={self.usage})"
