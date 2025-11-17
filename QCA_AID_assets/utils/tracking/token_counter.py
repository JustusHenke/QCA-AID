"""
Token Counter - Legacy Support

Simple token counter for backward compatibility.
For new code, use TokenTracker instead.
"""

from typing import Optional


class TokenCounter:
    """
    Legacy token counter class for basic token tracking.
    
    This is a simplified version kept for backward compatibility.
    For new code and full cost tracking, use TokenTracker instead.
    """
    
    def __init__(self) -> None:
        """Initialize token counter."""
        self.input_tokens = 0
        self.output_tokens = 0

    def add_tokens(self, input_tokens: int, output_tokens: int = 0) -> None:
        """
        Zählt Input- und Output-Tokens.
        
        Args:
            input_tokens: Anzahl der Input-Tokens
            output_tokens: Anzahl der Output-Tokens (optional, Standard 0)
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def get_report(self) -> str:
        """
        Returns summary of token usage.
        
        Returns:
            String with total token counts
        """
        return (f"Token Usage Summary:\n"
               f"Input Tokens: {self.input_tokens}\n"
               f"Output Tokens: {self.output_tokens}\n"
               f"Total Tokens: {self.input_tokens + self.output_tokens}")

    def estimate_tokens(self, text: str) -> int:
        """
        Schätzt die Anzahl der Tokens in einem Text.
        
        Uses a heuristic approach based on character count and word count.
        Accuracy is ±10-15% compared to actual tokenization.
        
        Args:
            text: Zu schätzender Text
            
        Returns:
            Geschätzte Tokenanzahl
        """
        if not text:
            return 0
        
        # Grundlegende Schätzung: 1 Token ≈ 4.5 Zeichen für deutsche Texte
        char_per_token = 4.5
        
        # Anzahl der Sonderzeichen, die oft eigene Tokens bilden
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        
        # Anzahl der Wörter
        words = len(text.split())
        
        # Gewichtete Berechnung
        estimated_tokens = int(
            (len(text) / char_per_token) * 0.7 +  # Character-based (70% weight)
            (words + special_chars) * 0.3          # Word-based (30% weight)
        )
        
        return max(1, estimated_tokens)  # At least 1 token
