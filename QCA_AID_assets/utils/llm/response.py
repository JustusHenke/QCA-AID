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
                self.content = response.choices[0].message.content or ""
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
        Extrahiert JSON aus dem Response-Content mit robustem Parsing.
        
        Anthropic (Claude) gibt oft JSON in Markdown-Code-Blöcken zurück:
        ```json
        {"key": "value"}
        ```
        
        Diese Methode extrahiert das reine JSON und versucht unvollständige
        JSON-Responses zu reparieren.
        
        Returns:
            str: Bereinigter JSON-String
        """
        import re
        import json as json_module
        
        content = self.content.strip()
        
        # 1. Prüfe ob Content mit ```json oder ``` beginnt
        if content.startswith('```'):
            # Extrahiere JSON aus Code-Block
            # Pattern: ```json\n{...}\n``` oder ```\n{...}\n```
            pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                logger.debug(f"Extracted JSON from markdown code block (length: {len(extracted)})")
                return self._repair_json(extracted)
        
        # 2. Wenn kein Code-Block gefunden, versuche JSON direkt zu finden
        # Suche nach { oder [ am Anfang
        first_brace = content.find('{')
        first_bracket = content.find('[')
        
        start_pos = -1
        json_char = None
        
        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start_pos = first_brace
            json_char = '{'
        elif first_bracket != -1:
            start_pos = first_bracket
            json_char = '['
        
        if start_pos != -1:
            extracted = content[start_pos:].strip()
            logger.debug(f"Extracted JSON from content starting at position {start_pos} (length: {len(extracted)})")
            return self._repair_json(extracted)
        
        # Fallback: Gib Original-Content zurück
        logger.debug("No JSON extraction needed, returning original content")
        return content
    
    def _repair_json(self, json_str: str) -> str:
        """
        Versucht unvollständige oder malformed JSON zu reparieren.
        
        Handles common LLM truncation issues:
        - Unterminated strings (missing closing quote)
        - Missing closing brackets/braces
        - Trailing commas before closing brackets
        
        Args:
            json_str: Möglicherweise unvollständiger JSON-String
            
        Returns:
            str: Reparierter JSON-String
        """
        import json as json_module
        import re
        
        # Versuche zuerst direktes Parsen
        try:
            json_module.loads(json_str)
            return json_str
        except json_module.JSONDecodeError:
            pass
        
        # --- Reparatur-Strategie ---
        repaired = json_str
        
        # 1. Entferne unvollständiges letztes Element bei abgeschnittenem JSON
        #    Suche das letzte vollständige JSON-Objekt/Array-Element
        #    Typisches Muster: ..., "key": "unvollständiger Wert
        
        # 1a. Schließe unterminated Strings
        #     Zähle Anführungszeichen (ohne escaped quotes)
        in_string = False
        last_quote_pos = -1
        i = 0
        while i < len(repaired):
            ch = repaired[i]
            if ch == '\\' and in_string:
                i += 2  # Skip escaped character
                continue
            if ch == '"':
                in_string = not in_string
                if in_string:
                    last_quote_pos = i
            i += 1
        
        if in_string and last_quote_pos >= 0:
            # String ist nicht geschlossen - zwei Strategien:
            # A) Versuche den unvollständigen Wert abzuschneiden und das letzte
            #    vollständige Element zu behalten
            # B) Schließe den String einfach
            
            # Strategie A: Schneide das letzte unvollständige key-value Paar ab
            # Finde das letzte Komma vor dem offenen String
            search_region = repaired[:last_quote_pos]
            last_comma = search_region.rfind(',')
            
            if last_comma > 0:
                # Schneide ab dem letzten Komma ab
                truncated = repaired[:last_comma]
                # Schließe offene Klammern
                truncated = self._close_brackets(truncated)
                try:
                    json_module.loads(truncated)
                    logger.debug("Repaired JSON by truncating incomplete last element at comma")
                    return truncated
                except json_module.JSONDecodeError:
                    pass
            
            # Strategie B: Schließe den offenen String
            repaired = repaired + '"'
            logger.debug("Repaired JSON by closing unterminated string")
        
        # 2. Entferne trailing commas vor schließenden Klammern
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        
        # 3. Schließe fehlende Klammern
        repaired = self._close_brackets(repaired)
        
        # 4. Versuche nochmal zu parsen
        try:
            json_module.loads(repaired)
            logger.debug("Successfully repaired JSON")
            return repaired
        except json_module.JSONDecodeError:
            pass
        
        # 5. Letzter Versuch: Schneide zeichenweise vom Ende ab bis valides JSON entsteht
        #    (nur für Objekte/Arrays mit mindestens einem vollständigen Element)
        for cut_pos in range(len(json_str) - 1, max(0, len(json_str) - 500), -1):
            candidate = json_str[:cut_pos]
            # Entferne trailing commas
            candidate = re.sub(r',\s*$', '', candidate)
            candidate = self._close_brackets(candidate)
            try:
                json_module.loads(candidate)
                logger.debug(f"Repaired JSON by truncating last {len(json_str) - cut_pos} chars")
                return candidate
            except json_module.JSONDecodeError:
                continue
        
        logger.debug("Could not repair JSON, returning best effort")
        return repaired
    
    def _close_brackets(self, json_str: str) -> str:
        """Füge fehlende schließende Klammern/Brackets hinzu."""
        # Zähle unter Berücksichtigung von Strings
        stack = []
        in_string = False
        i = 0
        while i < len(json_str):
            ch = json_str[i]
            if ch == '\\' and in_string:
                i += 2
                continue
            if ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch in ('{', '['):
                    stack.append(ch)
                elif ch == '}' and stack and stack[-1] == '{':
                    stack.pop()
                elif ch == ']' and stack and stack[-1] == '[':
                    stack.pop()
            i += 1
        
        # Schließe in umgekehrter Reihenfolge
        for bracket in reversed(stack):
            if bracket == '{':
                json_str += '}'
            elif bracket == '[':
                json_str += ']'
        
        return json_str
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"LLMResponse(model={self.model}, content_length={len(self.content)}, usage={self.usage})"
