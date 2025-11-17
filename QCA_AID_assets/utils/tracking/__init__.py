"""
Tracking Package

Manages LLM token usage, API costs, and session statistics.

Features:
- Per-model token accounting
- Cost calculation across providers (OpenAI, Mistral, Claude)
- Session and daily statistics
- Model pricing management
- Error tracking and reporting

Exports:
  - TokenTracker: Main tracking class with cost calculation
  - TokenCounter: Legacy support for simple token counting
"""

from .token_tracker import TokenTracker
from .token_counter import TokenCounter

__all__ = [
    'TokenTracker',
    'TokenCounter',
]
