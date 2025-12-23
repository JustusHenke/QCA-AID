"""
Prompt Templates for QCA-AID Explorer

This module contains standard prompt templates for various analysis types
used in the QCA-AID Explorer application.
"""

from typing import Dict


def get_default_prompts() -> Dict[str, str]:
    """
    Gibt die Standard-Prompts für verschiedene Analysetypen zurück.
    
    Returns:
        Dictionary mit Prompt-Templates für verschiedene Analysetypen.
        verfügbare Keys:
        - "paraphrase": Template für die Analyse paraphrasierter Textabschnitte
        - "reasoning": Template für die Analyse von Begründungen
    """
    prompts = {
        "paraphrase": """
Bitte analysieren Sie die folgenden paraphrasierten Textabschnitte und erstellen Sie einen thematischen Überblick:

{text}

Bitte geben Sie an:
1. Zusammenfassung identifizierter Hauptthemen. Ergänze dies anschließend mit konkreten Beispielen
2. Zusammenfassung wichtiger Muster und Beziehungen. Ergänze dies anschließend mit konkreten Beispielen
3. Zusammenfassung der Ergebnisse
        """,
        
        "reasoning": """
Bitte analysieren Sie die folgenden Begründungen für die Inklusion von Textsegementen in diesen Kategorien: {filters} 

Erstellen Sie eine umfassende Zusammenfassung:

{text}

Bitte geben Sie an:
1. Identifizierte Hauptargumente, die zur Inklusion in die Kategorien führten
2. Gemeinsame Muster in der Argumentation
3. Zusammenfassung der wichtigsten Begründungen
        """
    }
    return prompts
