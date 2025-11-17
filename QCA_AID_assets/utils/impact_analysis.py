"""
Impact Analysis - Analyze changes in coding decisions

Provides utilities for analyzing and reporting impacts of coding modifications.
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime
import os
import json


def analyze_multiple_coding_impact(
    original_codings: List[Dict],
    modified_codings: List[Dict]
) -> Dict[str, Any]:
    """
    Analyzes the impact of multiple coding changes on coding consistency.
    Compares original and modified coding decisions.
    
    Args:
        original_codings: List of original coding decisions
        modified_codings: List of modified coding decisions
        
    Returns:
        Analysis dictionary with consistency metrics and changes
    """
    analysis = {
        'total_original': len(original_codings),
        'total_modified': len(modified_codings),
        'changed_count': 0,
        'consistency_rate': 0.0,
        'category_changes': defaultdict(int),
        'confidence_changes': []
    }
    
    # Build mapping of original codings by segment_id
    original_map = {c.get('segment_id', ''): c for c in original_codings}
    
    for modified in modified_codings:
        segment_id = modified.get('segment_id', '')
        original = original_map.get(segment_id)
        
        if original:
            # Check if category changed
            if original.get('category') != modified.get('category'):
                analysis['changed_count'] += 1
                analysis['category_changes'][original.get('category', 'unknown')] += 1
            
            # Track confidence changes
            orig_conf = original.get('confidence', 0)
            mod_conf = modified.get('confidence', 0)
            if orig_conf != mod_conf:
                analysis['confidence_changes'].append({
                    'segment_id': segment_id,
                    'original_confidence': orig_conf,
                    'modified_confidence': mod_conf,
                    'change': mod_conf - orig_conf
                })
    
    # Calculate consistency rate
    if analysis['total_modified'] > 0:
        analysis['consistency_rate'] = (
            (analysis['total_modified'] - analysis['changed_count']) /
            analysis['total_modified']
        )
    
    return analysis


def export_multiple_coding_report(analysis: Dict, output_dir: str) -> Optional[str]:
    """
    Exports a report about multiple coding analysis.
    
    Args:
        analysis: Analysis dictionary to export
        output_dir: Directory to save the report
        
    Returns:
        Path to exported report file, or None if export failed
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"multiple_coding_report_{timestamp}.json")
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Multiple coding report exported to: {report_path}")
        return report_path
    
    except Exception as e:
        print(f"Fehler beim Export des Mehrfachkodierungs-Reports: {str(e)}")
        return None


__all__ = [
    'analyze_multiple_coding_impact',
    'export_multiple_coding_report',
]
