"""
Dialog Helpers - GUI setup and formatting for QCA-AID

Provides utilities for dialog configuration and multiple coding information.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime


def setup_manual_coding_window_enhanced(parent, categories: List[str]) -> Dict[str, Any]:
    """
    Sets up enhanced manual coding window configuration.
    
    Args:
        parent: Parent widget/window
        categories: List of category names
        
    Returns:
        Dictionary with window configuration
    """
    return {
        'parent': parent,
        'categories': categories,
        'allow_multiple': True,
        'show_confidence': True,
        'show_justification': True,
        'window_title': 'Manuelle Kodierung',
        'width': 1000,
        'height': 800
    }


def create_multiple_coding_results(
    segment_id: str,
    coding_decisions: List[Dict],
    timestamp: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Creates a formatted result object for multiple coding decisions.
    
    Args:
        segment_id: ID of the segment being coded
        coding_decisions: List of coding decision dictionaries
        timestamp: Optional timestamp for the decision
        
    Returns:
        Formatted result dictionary
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return {
        'segment_id': segment_id,
        'coding_decisions': coding_decisions,
        'decision_count': len(coding_decisions),
        'timestamp': timestamp.isoformat(),
        'has_consensus': len(set(c.get('category', '') for c in coding_decisions)) == 1
    }


def show_multiple_coding_info(coding_decisions: List[Dict]) -> str:
    """
    Generates an info string about multiple coding decisions.
    
    Args:
        coding_decisions: List of coding decision dictionaries
        
    Returns:
        Formatted information string
    """
    if not coding_decisions:
        return "Keine Kodierungsentscheidungen vorhanden"
    
    categories = set(c.get('category', 'unknown') for c in coding_decisions)
    info = f"Mehrfachkodierung: {len(coding_decisions)} Entscheidungen über {len(categories)} Kategorien\n"
    info += "Kategorien: " + ", ".join(sorted(categories))
    
    return info


__all__ = [
    'setup_manual_coding_window_enhanced',
    'create_multiple_coding_results',
    'show_multiple_coding_info',
]
