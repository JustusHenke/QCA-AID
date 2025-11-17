"""
Validators - Data validation utilities for QCA-AID

Provides validation functions for segments, categories, and coding decisions.
"""

from typing import Dict, List, Optional, Any, Tuple


def validate_category_specific_segments(category_segments: List[Dict]) -> Dict[str, Any]:
    """
    Validates segments grouped by specific categories for review.
    
    Args:
        category_segments: List of segment dictionaries to validate
        
    Returns:
        Dictionary with valid_segments, warnings, and errors
    """
    try:
        validation_results = {
            'valid_segments': [],
            'warnings': [],
            'errors': []
        }
        
        for segment in category_segments:
            if not segment.get('text'):
                validation_results['errors'].append(
                    f"Segment missing text: {segment.get('id', 'unknown')}"
                )
                continue
            
            if not segment.get('category'):
                validation_results['warnings'].append(
                    f"Segment missing category: {segment.get('id', 'unknown')}"
                )
                continue
            
            validation_results['valid_segments'].append(segment)
        
        return validation_results
    
    except Exception as e:
        print(f"Fehler bei Validierung der kategorie-spezifischen Segmente: {str(e)}")
        return {'valid_segments': category_segments, 'warnings': [], 'errors': [str(e)]}


def validate_multiple_selection(
    selected_indices: List[int],
    category_map: Dict[int, Dict],
    min_count: int = 1,
    max_count: Optional[int] = None
) -> Tuple[bool, str, List[Dict]]:
    """
    Validates a multiple selection against the category map.
    
    Args:
        selected_indices: List of selected indices from listbox
        category_map: Mapping of indices to category information
        min_count: Minimum number of selections required
        max_count: Maximum number of selections allowed
        
    Returns:
        Tuple of (is_valid, error_message, selected_categories)
    """
    errors = []
    
    # Check count constraints
    if len(selected_indices) < min_count:
        errors.append(
            f"Mindestens {min_count} Element(e) erforderlich, {len(selected_indices)} ausgewählt"
        )
    
    if max_count is not None and len(selected_indices) > max_count:
        errors.append(
            f"Maximal {max_count} Element(e) erlaubt, {len(selected_indices)} ausgewählt"
        )
    
    # Get selected categories from map
    selected_categories = []
    for idx in selected_indices:
        if idx in category_map:
            selected_categories.append(category_map[idx])
        else:
            errors.append(f"Ungültiger Index: {idx}")
    
    is_valid = len(errors) == 0
    error_message = "; ".join(errors) if errors else ""
    
    return is_valid, error_message, selected_categories


def validate_multiple_selection_legacy(
    selected_items: List[str],
    allowed_items: List[str],
    min_count: int = 1,
    max_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Legacy function: Validates a multiple selection against allowed items.
    
    Args:
        selected_items: Items that were selected
        allowed_items: List of items that are allowed
        min_count: Minimum number of selections required
        max_count: Maximum number of selections allowed
        
    Returns:
        Dictionary with validation result
    """
    result = {
        'valid': False,
        'count': len(selected_items),
        'errors': []
    }
    
    # Check count constraints
    if len(selected_items) < min_count:
        result['errors'].append(
            f"Mindestens {min_count} Element(e) erforderlich, {len(selected_items)} ausgewählt"
        )
    
    if max_count is not None and len(selected_items) > max_count:
        result['errors'].append(
            f"Maximal {max_count} Element(e) erlaubt, {len(selected_items)} ausgewählt"
        )
    
    # Check if all items are allowed
    invalid_items = [item for item in selected_items if item not in allowed_items]
    if invalid_items:
        result['errors'].append(f"Ungültige Element(e): {', '.join(invalid_items)}")
    
    result['valid'] = len(result['errors']) == 0
    return result


__all__ = [
    'validate_category_specific_segments',
    'validate_multiple_selection',
    'validate_multiple_selection_legacy',
]
