"""
Export Helper Functions

Text sanitization, color generation, and formatting utilities for Excel export.
"""

import re
import colorsys
import unicodedata
from typing import Dict, List


def sanitize_text_for_excel(text: str) -> str:
    """
    Erweiterte Textbereinigung für Excel-Export.
    Entfernt Artefakte aus der Dokumentverarbeitung.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Cleaned text safe for Excel export
    """
    if not text:
        return ""
    
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Remove file path artifacts
    text = re.sub(r'file:///[^\s\]]+', '', text)
    text = re.sub(r'[A-Za-z]:/[^\s\]]*\.txt', '', text)
    text = re.sub(r'/[^\s\]]*\.txt', '', text)
    
    # Remove leading/trailing brackets with metadata
    text = re.sub(r'^\s*\]', '', text)
    text = re.sub(r'\[\s*$', '', text)
    
    # Remove empty brackets
    text = re.sub(r'\[\s*\]', '', text)
    
    # Remove chunk separators and metadata markers
    text = re.sub(r'^\s*---+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*===+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove redundant brackets at sentence start
    text = re.sub(r'(?<=\s)\]([A-Z])', r'\1', text)
    text = re.sub(r'^\]([A-Z])', r'\1', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\uFFFE\uFFFF]', '', text)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive line breaks (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalize whitespace
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove space before punctuation
    text = re.sub(r' +([.!?,:;])', r'\1', text)
    
    # Replace problematic special characters
    problematic_chars = {
        '☺': ':)', '☻': ':)', '♥': '<3', '♦': 'diamond', '♣': 'club', '♠': 'spade',
        '†': '+', '‡': '++', '•': '*', '‰': 'promille', '™': '(TM)', '©': '(C)',
        '®': '(R)', '§': 'section', '¶': 'paragraph', '±': '+/-'
    }
    
    for char, replacement in problematic_chars.items():
        text = text.replace(char, replacement)
    
    # Normalize Unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove private use area characters
    text = re.sub(r'[\uE000-\uF8FF]', '', text)
    
    # Limit text length for Excel cells (Excel limit: 32,767 characters)
    if len(text) > 32760:  # Safety buffer
        text = text[:32760] + "..."
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Ensure text is not empty after cleanup
    if not text.strip():
        return ""
    
    return text


def generate_pastel_colors(num_colors: int) -> List[str]:
    """
    Generiert eine Palette mit Pastellfarben.
    
    Args:
        num_colors: Anzahl der benötigten Farben
    
    Returns:
        Liste von Hex-Farbcodes in Pastelltönen
    """
    pastel_colors = []
    for i in range(num_colors):
        # Wähle Hue gleichmäßig über Farbkreis
        hue = i / num_colors
        # Konvertiere HSV zu RGB mit hoher Helligkeit und Sättigung
        rgb = colorsys.hsv_to_rgb(hue, 0.4, 0.95)
        # Konvertiere RGB zu Hex
        hex_color = 'FF{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        pastel_colors.append(hex_color)
    
    return pastel_colors


def format_confidence(confidence: Dict) -> str:
    """
    Formatiert die Konfidenz-Werte für den Export.
    
    Args:
        confidence: Confidence values (dict, float, or string)
        
    Returns:
        Formatted confidence string
    """
    try:
        if isinstance(confidence, dict):
            formatted_values = []
            # Process each confidence value
            for key, value in confidence.items():
                if isinstance(value, (int, float)):
                    formatted_values.append(f"{key}: {value:.2f}")
                elif isinstance(value, dict):
                    # Process nested confidence values
                    nested_values = [f"{k}: {v:.2f}" for k, v in value.items()
                                   if isinstance(v, (int, float))]
                    if nested_values:
                        formatted_values.append(f"{key}: {', '.join(nested_values)}")
                elif isinstance(value, str):
                    formatted_values.append(f"{key}: {value}")
            
            return "\n".join(formatted_values)
        elif isinstance(confidence, (int, float)):
            return f"{float(confidence):.2f}"
        elif isinstance(confidence, str):
            return confidence
        else:
            return "0.00"
    
    except Exception as e:
        print(f"Fehler bei Konfidenz-Formatierung: {str(e)}")
        return "0.00"


__all__ = [
    'sanitize_text_for_excel',
    'generate_pastel_colors',
    'format_confidence',
]
