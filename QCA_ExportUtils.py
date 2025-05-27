import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from typing import Dict, List

def _sanitize_text_for_excel(text):
    """
    Bereinigt Text für Excel-Export, entfernt ungültige Zeichen.
    
    Args:
        text: Zu bereinigender Text
        
    Returns:
        str: Bereinigter Text ohne problematische Zeichen
    """
    if text is None:
        return ""
        
    if not isinstance(text, str):
        # Konvertiere zu String falls nötig
        text = str(text)
    
    # Liste von problematischen Zeichen, die in Excel Probleme verursachen können
    # Hier definieren wir Steuerzeichen und einige bekannte Problemzeichen
    problematic_chars = [
        # ASCII-Steuerzeichen 0-31 außer Tab (9), LF (10) und CR (13)
        *[chr(i) for i in range(0, 9)],
        *[chr(i) for i in range(11, 13)],
        *[chr(i) for i in range(14, 32)],
        # Einige bekannte problematische Sonderzeichen
        '\u0000', '\u0001', '\u0002', '\u0003', '\ufffe', '\uffff',
        # Emojis und andere Sonderzeichen, die Probleme verursachen könnten
        '☺', '☻', '♥', '♦', '♣', '♠'
    ]
    
    # Ersetze alle problematischen Zeichen
    for char in problematic_chars:
        text = text.replace(char, '')
    
    # Alternative Methode mit Regex für Steuerzeichen
    import re
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFE\uFFFF]', '', text)
    
    return text

def _generate_pastel_colors(num_colors):
    """
    Generiert eine Palette mit Pastellfarben.
    
    Args:
        num_colors (int): Anzahl der benötigten Farben
    
    Returns:
        List[str]: Liste von Hex-Farbcodes in Pastelltönen
    """
    import colorsys
    
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

    
def _format_confidence(confidence: dict) -> str:
    """Formatiert die Konfidenz-Werte für den Export"""
    try:
        if isinstance(confidence, dict):
            formatted_values = []
            # Verarbeite jeden Konfidenzwert einzeln
            for key, value in confidence.items():
                if isinstance(value, (int, float)):
                    formatted_values.append(f"{key}: {value:.2f}")
                elif isinstance(value, dict):
                    # Verarbeite verschachtelte Konfidenzwerte
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