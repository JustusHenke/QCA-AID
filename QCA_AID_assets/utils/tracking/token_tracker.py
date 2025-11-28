"""
Token Tracker for LLM Cost & Usage Accounting

Tracks token usage, costs, and statistics across LLM API calls.
Supports multiple models with intelligent pricing and fallbacks.
"""

import json
import os
from datetime import date
from typing import Dict, Optional, Any


# Globale Token-Counter Instanz (Singleton-Pattern)
_global_token_counter = None


def get_global_token_counter():
    """
    Gibt die globale Token-Counter Instanz zurück (Singleton).
    
    Returns:
        TokenTracker: Die globale Token-Counter Instanz
    """
    global _global_token_counter
    if _global_token_counter is None:
        _global_token_counter = TokenTracker()
    return _global_token_counter


class TokenTracker:
    """
    Verfolgt Token-Verbrauch und LLM-Kosten mit persistenten Statistiken.
    
    Features:
    - Per-model token counting
    - Cost calculation with model-specific pricing
    - Session and daily statistics
    - Model capability caching (for parameter support detection)
    - Rate limit warnings
    - Detailed reporting
    
    The tracker maintains both session stats (current run) and daily stats
    (persisted to disk for cross-session accumulation).
    """
    
    def __init__(self) -> None:
        """Initialize token tracker with session and daily statistics."""
        self.session_stats = {'input': 0, 'output': 0, 'requests': 0, 'cost': 0.0}
        self.daily_stats = self.load_daily_stats()
        
        # Cache für Model-Capabilities (z.B. welche Parameter unterstützt werden)
        # Format: {'model_name': True/False/None}
        # None = not yet tested, True = supports temperature, False = doesn't support
        self.model_capabilities = {}
        
        # Umfassende Model-Preisliste für alle unterstützten Provider
        self.model_prices = {
            # === CLAUDE MODELLE ===
            'claude-sonnet-4-20250514': {'input': 0.000015, 'output': 0.000075},
            'claude-opus-4-20241022': {'input': 0.000075, 'output': 0.000375},
            'claude-3-5-sonnet-20241022': {'input': 0.000003, 'output': 0.000015},
            'claude-3-5-haiku-20241022': {'input': 0.00000025, 'output': 0.00000125},
            
            # === OPENAI GPT-4.1 SERIE ===
            'gpt-4.1': {'input': 0.000003, 'output': 0.000015},
            'gpt-4.1-mini': {'input': 0.000001, 'output': 0.000004},
            'gpt-4.1-nano': {'input': 0.0000001, 'output': 0.0000004},
            
            # === OPENAI GPT-4O SERIE ===
            'gpt-4o': {'input': 0.000003, 'output': 0.00001},
            'gpt-4o-2024-11-20': {'input': 0.000003, 'output': 0.00001},
            'gpt-4o-mini': {'input': 0.00000015, 'output': 0.0000006},
            'gpt-4o-mini-2024-07-18': {'input': 0.00000015, 'output': 0.0000006},
            
            # === GPT-4O AUDIO/REALTIME ===
            'gpt-4o-realtime-preview': {'input': 0.000005, 'output': 0.00002},
            'gpt-4o-audio-preview': {'input': 0.000005, 'output': 0.00002},
            
            # === LEGACY GPT-4 MODELLE ===
            'gpt-4': {'input': 0.00003, 'output': 0.00006},
            'gpt-4-turbo': {'input': 0.00001, 'output': 0.00003},
            'gpt-4-turbo-2024-04-09': {'input': 0.00001, 'output': 0.00003},
            'gpt-4-1106-preview': {'input': 0.00001, 'output': 0.00003},
            'gpt-4-vision-preview': {'input': 0.00001, 'output': 0.00003},
            
            # === GPT-3.5 TURBO ===
            'gpt-3.5-turbo': {'input': 0.000001, 'output': 0.000002},
            'gpt-3.5-turbo-0125': {'input': 0.000001, 'output': 0.000002},
            'gpt-3.5-turbo-instruct': {'input': 0.0000015, 'output': 0.000002},
            
            # === BATCH API PREISE (50% Rabatt) ===
            'gpt-4o-batch': {'input': 0.0000015, 'output': 0.000005},
            'gpt-4o-mini-batch': {'input': 0.000000075, 'output': 0.0000003},
            'gpt-4-turbo-batch': {'input': 0.000005, 'output': 0.000015},
        }
        
        self.request_start_time = None
        self.debug_calls = []
    
    def get_model_price(self, model_name: str) -> Dict[str, float]:
        """
        Ermittelt den Preis für ein Modell mit intelligenter Fallback-Logik.
        
        Supports exact matches and intelligent family-based fallbacks for
        unknown models. Uses model name patterns to infer pricing.
        
        Args:
            model_name: Name of the model (e.g., 'gpt-4o-mini', 'claude-opus')
            
        Returns:
            Dictionary with 'input' and 'output' keys (prices per 1M tokens)
        """
        # Exakte Übereinstimmung
        if model_name in self.model_prices:
            return self.model_prices[model_name]
        
        # Fallback-Logik für ähnliche Modelle
        model_lower = model_name.lower()
        
        # GPT-4.1 Familie
        if 'gpt-4.1' in model_lower:
            if 'nano' in model_lower:
                return self.model_prices['gpt-4.1-nano']
            elif 'mini' in model_lower:
                return self.model_prices['gpt-4.1-mini']
            else:
                return self.model_prices['gpt-4.1']
        
        # GPT-4o Familie
        elif 'gpt-4o' in model_lower:
            if 'mini' in model_lower:
                return self.model_prices['gpt-4o-mini']
            elif 'batch' in model_lower:
                return self.model_prices['gpt-4o-batch']
            else:
                return self.model_prices['gpt-4o']
        
        # GPT-4 Familie
        elif 'gpt-4' in model_lower:
            if 'turbo' in model_lower:
                return self.model_prices['gpt-4-turbo']
            else:
                return self.model_prices['gpt-4']
        
        # GPT-3.5 Familie
        elif 'gpt-3.5' in model_lower:
            return self.model_prices['gpt-3.5-turbo']
        
        # Claude Familie
        elif 'claude' in model_lower:
            if 'sonnet-4' in model_lower:
                return self.model_prices['claude-sonnet-4-20250514']
            elif 'opus-4' in model_lower:
                return self.model_prices['claude-opus-4-20241022']
            else:
                return self.model_prices['claude-3-5-sonnet-20241022']
        
        # GPT-5 Familie (neue Modelle)
        elif 'gpt-5' in model_lower:
            if 'nano' in model_lower:
                return {'input': 0.000001, 'output': 0.000004}
            elif 'mini' in model_lower:
                return {'input': 0.000002, 'output': 0.000008}
            else:
                return {'input': 0.000005, 'output': 0.000020}
        
        # Mistral Familie
        elif 'mistral' in model_lower:
            if 'large' in model_lower:
                return {'input': 0.000002, 'output': 0.000006}
            else:
                return {'input': 0.000001, 'output': 0.000003}
        
        # Intelligente Schätzung basierend auf Größenkategorien
        if any(x in model_lower for x in ['nano', 'mini', 'small', 'lite']):
            return {'input': 0.000001, 'output': 0.000004}
        
        if any(x in model_lower for x in ['standard', 'base', 'default']):
            return {'input': 0.000003, 'output': 0.000010}
        
        # Fallback für große Modelle
        return {'input': 0.000005, 'output': 0.000020}
    
    def load_daily_stats(self) -> Dict:
        """
        Lade Tagesstatistiken aus Datei.
        
        Returns today's stats if file exists and is from today,
        otherwise returns zeroed stats for new day.
        
        Returns:
            Dictionary with 'input', 'output', 'requests', 'cost' keys
        """
        today = str(date.today())
        try:
            if os.path.exists('token_stats.json'):
                with open('token_stats.json', 'r') as f:
                    data = json.load(f)
                    if data.get('date') == today:
                        return data.get('stats', {'input': 0, 'output': 0, 'requests': 0, 'cost': 0.0})
        except Exception:
            pass
        return {'input': 0, 'output': 0, 'requests': 0, 'cost': 0.0}
    
    def save_daily_stats(self) -> None:
        """Speichere Tagesstatistiken in Datei für Persistenz über Sessions."""
        data = {
            'date': str(date.today()),
            'stats': self.daily_stats
        }
        try:
            with open('token_stats.json', 'w') as f:
                json.dump(data, f)
        except Exception:
            pass
    
    def start_request(self) -> None:
        """Markiere Start einer API-Anfrage für Timing."""
        import time
        self.request_start_time = time.time()
    
    def add_tokens(self, input_tokens: int, output_tokens: int = 0) -> None:
        """
        Kompatibilitätsmethode für Legacy-Code.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens (default 0)
        """
        import time
        
        # Update session stats
        self.session_stats['input'] += input_tokens
        self.session_stats['output'] += output_tokens
        self.session_stats['requests'] += 1
        
        # Update daily stats
        self.daily_stats['input'] += input_tokens
        self.daily_stats['output'] += output_tokens
        self.daily_stats['requests'] += 1
        self.save_daily_stats()
        
        # Debug tracking
        self.debug_calls.append({
            'method': 'add_tokens',
            'input': input_tokens,
            'output': output_tokens,
            'time': time.time(),
            'session_total_after': self.session_stats['input'] + self.session_stats['output']
        })
    
    def track_response(self, response_data: Any, model: str) -> None:
        """
        Verfolge Token-Verbrauch aus LLM-Response.
        
        Extracts usage information from provider-specific response formats
        and updates cost statistics using model-specific pricing.
        
        Args:
            response_data: Response object from LLM provider (OpenAI, Mistral, etc.)
            model: Name of the model that generated the response
        """
        import time
        
        try:
            # Finde Usage-Daten (verschiedene Provider-Formate)
            usage_data = None
            if hasattr(response_data, 'usage'):
                usage_data = response_data.usage
            elif isinstance(response_data, dict) and 'usage' in response_data:
                usage_data = response_data['usage']
            else:
                return
            
            if not usage_data:
                return
            
            input_tokens = 0
            output_tokens = 0
            
            # Multi-Provider Token-Extraktion
            if hasattr(usage_data, 'prompt_tokens'):
                input_tokens = usage_data.prompt_tokens
            elif hasattr(usage_data, 'input_tokens'):
                input_tokens = usage_data.input_tokens
            elif isinstance(usage_data, dict):
                input_tokens = usage_data.get('prompt_tokens', 0) or usage_data.get('input_tokens', 0)
            
            if hasattr(usage_data, 'completion_tokens'):
                output_tokens = usage_data.completion_tokens
            elif hasattr(usage_data, 'output_tokens'):
                output_tokens = usage_data.output_tokens
            elif isinstance(usage_data, dict):
                output_tokens = usage_data.get('completion_tokens', 0) or usage_data.get('output_tokens', 0)
            
            if input_tokens > 0 or output_tokens > 0:
                # Berechne Kosten basierend auf Model-Preisen
                price = self.get_model_price(model)
                cost = (input_tokens * price['input']) + (output_tokens * price['output'])
                
                # Update statistics
                self.session_stats['input'] += input_tokens
                self.session_stats['output'] += output_tokens
                self.session_stats['requests'] += 1
                self.session_stats['cost'] += cost
                
                self.daily_stats['input'] += input_tokens
                self.daily_stats['output'] += output_tokens
                self.daily_stats['requests'] += 1
                self.daily_stats['cost'] += cost
                self.save_daily_stats()
                
                # Check for rate limit warnings
                self.check_rate_limits(input_tokens, output_tokens)
                    
        except Exception as e:
            print(f"[ERROR] Token-Tracking: {e}")
    
    def track_error(self, model: Optional[str] = None) -> None:
        """
        Track LLM API errors without counting tokens.
        
        Args:
            model: Optional model name for logging
        """
        self.session_stats['requests'] += 1
        self.daily_stats['requests'] += 1
        self.save_daily_stats()
    
    def check_rate_limits(self, input_tokens: int, output_tokens: int) -> None:
        """
        Check and warn for potential rate limits.
        
        Args:
            input_tokens: Input tokens for this request
            output_tokens: Output tokens for this request
        """
        total_tokens = input_tokens + output_tokens
        
        if total_tokens > 50000:
            print(f"[WARN] High token usage: {total_tokens:,} tokens")
        
        if self.daily_stats['cost'] > 10.0:
            print(f"[WARN] High daily cost: ${self.daily_stats['cost']:.2f}")
        
        if self.daily_stats['requests'] > 1000:
            print(f"[WARN] Many requests today: {self.daily_stats['requests']}")
    
    def get_report(self) -> str:
        """
        Detaillierter Report über Token-Verbrauch und Kosten.
        
        Returns:
            Formatted string with session and daily statistics
        """
        return (f"📊 TOKEN-STATISTIKEN\n"
                f"{'='*60}\n"
                f"🎯 Session-Statistiken:\n"
                f"   Input Tokens: {self.session_stats['input']:,}\n"
                f"   Output Tokens: {self.session_stats['output']:,}\n"
                f"   Total Tokens: {self.session_stats['input'] + self.session_stats['output']:,}\n"
                f"   Requests: {self.session_stats['requests']}\n"
                f"   Session Cost: ${self.session_stats['cost']:.4f}\n"
                f"\n📅 Daily Statistics:\n"
                f"   Input Tokens: {self.daily_stats['input']:,}\n"
                f"   Output Tokens: {self.daily_stats['output']:,}\n"
                f"   Total Tokens: {self.daily_stats['input'] + self.daily_stats['output']:,}\n"
                f"   Requests: {self.daily_stats['requests']}\n"
                f"   Daily Cost: ${self.daily_stats['cost']:.4f}\n"
                f"{'='*60}")
    
    def get_session_cost(self) -> float:
        """Get total cost for current session."""
        return self.session_stats['cost']
    
    def get_daily_cost(self) -> float:
        """Get total cost for current day."""
        return self.daily_stats['cost']
    
    def reset_session(self) -> None:
        """Reset session statistics (keeps daily stats)."""
        self.session_stats = {'input': 0, 'output': 0, 'requests': 0, 'cost': 0.0}
        print("✅ Session statistics reset")
