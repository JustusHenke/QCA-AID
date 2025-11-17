"""
Dialog Helpers - GUI setup and formatting for QCA-AID

Provides utilities for dialog configuration and multiple coding information.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import tkinter.messagebox as messagebox


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
    selected_categories: List[Dict],
    text: str,
    coder_id: str,
    timestamp: Optional[datetime] = None
) -> Any:
    """
    Creates a formatted result object for multiple coding decisions.
    
    Args:
        selected_categories: List of selected category dictionaries
        text: Text that was coded
        coder_id: ID of the coder
        timestamp: Optional timestamp for the decision
        
    Returns:
        List of CodingResult objects or formatted dictionaries
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    from ..core.data_models import CodingResult
    
    results = []
    for category in selected_categories:
        result = CodingResult(
            category=category.get('main_category', category.get('name')),
            subcategories=tuple([category.get('name')]) if category.get('type') == 'sub' else (),
            justification=f"Manuelle Mehrfachkodierung von Coder {coder_id}",
            confidence={'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
            text_references=(text[:100] if text else "",)
        )
        results.append(result)
    
    return results


def show_multiple_coding_info(
    parent,
    decision_count: int,
    main_categories: List[str]
) -> bool:
    """
    Shows information about multiple coding and asks for confirmation.
    
    Args:
        parent: Parent widget
        decision_count: Number of coding decisions
        main_categories: List of main category names
        
    Returns:
        True if user confirmed, False otherwise
    """
    message = (
        f"Mehrfachkodierung: {decision_count} Entscheidungen\n"
        f"Hauptkategorien: {', '.join(sorted(main_categories))}\n\n"
        f"Möchten Sie diese Mehrfachkodierung speichern?"
    )
    
    return messagebox.askyesno("Mehrfachkodierung bestätigen", message, parent=parent)


__all__ = [
    'setup_manual_coding_window_enhanced',
    'create_multiple_coding_results',
    'show_multiple_coding_info',
]
