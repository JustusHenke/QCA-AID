"""
QCA-AID Prompt Templates
=======================

Zentrale Sammlung aller Prompts für die qualitative Inhaltsanalyse.
Getrennt vom Hauptskript für bessere Wartbarkeit und Übersichtlichkeit.

Author: Justus Henke
"""

from typing import Dict, List, Optional, Any
import json


from dataclasses import dataclass

@dataclass
class CategoryDefinition:
    """Datenklasse für eine Kategorie im Kodiersystem"""
    name: str
    definition: str
    examples: List[str]
    rules: List[str]
    subcategories: Dict[str, str]
    added_date: str
    modified_date: str

class ConfidenceScales:
    """
    Zentrale Definitionen für alle Konfidenz-Skalen in QCA-AID Prompts
    Sorgt für einheitliche und nachvollziehbare Score-Vergabe
    """
    
    # Standard 0.0-1.0 Skala für allgemeine Verwendung
    STANDARD_SCALE = {
        "sehr_hoch": "0.9-1.0 = Sehr hohe Konfidenz (eindeutig, ohne Zweifel)",
        "hoch": "0.7-0.9 = Hohe Konfidenz (klar erkennbar, minimale Unsicherheit)", 
        "mittel": "0.5-0.7 = Mittlere Konfidenz (wahrscheinlich, aber mit Vorbehalten)",
        "niedrig": "0.3-0.5 = Niedrige Konfidenz (unsicher, schwer einschätzbar)",
        "sehr_niedrig": "0.0-0.3 = Sehr niedrige Konfidenz (sehr unsicher, eher nicht)"
    }
    
    # Spezielle Skala für Kategorie-Relevanz (wie im Relevance-Prompt)
    CATEGORY_RELEVANCE_SCALE = {
        "definitiv": "0.6+ = Definitiv relevant für diese Kategorie",
        "moeglich": "0.4-0.6 = Möglicherweise relevant", 
        "unwahrscheinlich": "<0.4 = Wahrscheinlich nicht relevant"
    }
    
    # Skala für Kodierungs-Konfidenz (präziser als Standard)
    CODING_CONFIDENCE_SCALE = {
        "eindeutig": "0.8-1.0 = Eindeutige Zuordnung (klare Textbelege, keine Zweifel)",
        "klar": "0.6-0.8 = Klare Zuordnung (gute Textbelege, minimale Unsicherheit)",
        "wahrscheinlich": "0.4-0.6 = Wahrscheinliche Zuordnung (schwächere Belege, moderate Unsicherheit)",
        "unsicher": "0.2-0.4 = Unsichere Zuordnung (wenige Belege, hohe Unsicherheit)",
        "sehr_unsicher": "0.0-0.2 = Sehr unsichere Zuordnung (kaum Belege, maximale Unsicherheit)"
    }
    
    # Skala für Subkategorie-Konfidenz
    SUBCATEGORY_CONFIDENCE_SCALE = {
        "perfekt": "0.9-1.0 = Perfekte Passung (Subkategorie trifft exakt zu)",
        "gut": "0.7-0.9 = Gute Passung (Subkategorie passt gut zum Text)",
        "akzeptabel": "0.5-0.7 = Akzeptable Passung (Subkategorie ist angemessen)",
        "schwach": "0.3-0.5 = Schwache Passung (Subkategorie nur teilweise zutreffend)",
        "unpassend": "0.0-0.3 = Unpassende Subkategorie (Subkategorie trifft nicht zu)"
    }
    
    # Skala für induktive Kategorien-Entwicklung
    INDUCTIVE_CONFIDENCE_SCALE = {
        "stark": "0.8-1.0 = Starke Kategorie (deutliches Muster, mehrere Belege)",
        "solide": "0.6-0.8 = Solide Kategorie (erkennbares Muster, ausreichende Belege)",
        "schwach": "0.4-0.6 = Schwache Kategorie (unsicheres Muster, wenige Belege)", 
        "ungeeignet": "0.0-0.4 = Ungeeignete Kategorie (kein klares Muster)"
    }
    
    @staticmethod
    def get_scale_instructions(scale_type: str = "standard") -> str:
        """
        Gibt formatierte Skalen-Anweisungen für Prompts zurück
        
        Args:
            scale_type: Typ der Skala ('standard', 'category_relevance', 'coding', 'subcategory', 'inductive')
            
        Returns:
            str: Formatierte Skalen-Anweisungen
        """
        scales = {
            "standard": ConfidenceScales.STANDARD_SCALE,
            "category_relevance": ConfidenceScales.CATEGORY_RELEVANCE_SCALE, 
            "coding": ConfidenceScales.CODING_CONFIDENCE_SCALE,
            "subcategory": ConfidenceScales.SUBCATEGORY_CONFIDENCE_SCALE,
            "inductive": ConfidenceScales.INDUCTIVE_CONFIDENCE_SCALE
        }
        
        if scale_type not in scales:
            scale_type = "standard"
        
        scale = scales[scale_type]
        
        lines = ["KONFIDENZ-SKALA:"]
        for key, description in scale.items():
            lines.append(f"- {description}")
        
        return "\n".join(lines)
    
    @staticmethod  
    def get_confidence_guidelines(scale_type: str = "standard", 
                                context: str = "allgemein") -> str:
        """
        Gibt kontextspezifische Konfidenz-Richtlinien zurück
        
        Args:
            scale_type: Typ der Skala
            context: Kontext der Anwendung (z.B. "kodierung", "relevanz", "kategorien")
            
        Returns:
            str: Kontextspezifische Richtlinien
        """
        scale_instructions = ConfidenceScales.get_scale_instructions(scale_type)
        
        context_guidelines = {
            "kodierung": [
                "Begründe Konfidenz-Scores mit konkreten Textstellen",
                "Hohe Scores nur bei eindeutigen sprachlichen Indikatoren",
                "Bei Ambiguität: niedrigere Scores vergeben"
            ],
            "relevanz": [
                "Sei konservativ aber nicht zu restriktiv", 
                "Lieber mehrere Kategorien mit mittleren Scores als eine mit hohem Score",
                "Bei Unsicherheit zwischen Kategorien: beide einschließen"
            ],
            "kategorien": [
                "Neue Kategorien brauchen starke Belege für hohe Scores",
                "Bestehende Kategorien: Scores basierend auf Passung",
                "Subkategorien: Scores relativ zur Hauptkategorie"
            ],
            "allgemein": [
                "Scores immer mit Begründung versehen",
                "Konsistenz über alle Bewertungen hinweg beachten",
                "Ehrlich bei Unsicherheit - niedrige Scores sind okay"
            ]
        }
        
        guidelines = context_guidelines.get(context, context_guidelines["allgemein"])
        
        result = [scale_instructions, "", "RICHTLINIEN FÜR SCORE-VERGABE:"]
        for guideline in guidelines:
            result.append(f"- {guideline}")
        
        return "\n".join(result)
    
    @staticmethod
    def validate_confidence_score(score: float, scale_type: str = "standard") -> tuple:
        """
        Validiert einen Konfidenz-Score und gibt Feedback
        
        Args:
            score: Zu validierender Score (0.0-1.0)
            scale_type: Typ der verwendeten Skala
            
        Returns:
            tuple: (is_valid: bool, feedback: str, category: str)
        """
        if not (0.0 <= score <= 1.0):
            return False, f"Score {score} liegt außerhalb des gültigen Bereichs 0.0-1.0", "invalid"
        
        # Bestimme Kategorie basierend auf Score und Skala
        if scale_type == "category_relevance":
            if score >= 0.6:
                return True, "Definitiv relevant", "definitiv"
            elif score >= 0.4:
                return True, "Möglicherweise relevant", "moeglich"
            else:
                return True, "Wahrscheinlich nicht relevant", "unwahrscheinlich"
                
        elif scale_type == "coding":
            if score >= 0.8:
                return True, "Eindeutige Zuordnung", "eindeutig"
            elif score >= 0.6:
                return True, "Klare Zuordnung", "klar"
            elif score >= 0.4:
                return True, "Wahrscheinliche Zuordnung", "wahrscheinlich"
            elif score >= 0.2:
                return True, "Unsichere Zuordnung", "unsicher"
            else:
                return True, "Sehr unsichere Zuordnung", "sehr_unsicher"
                
        else:  # standard scale
            if score >= 0.9:
                return True, "Sehr hohe Konfidenz", "sehr_hoch"
            elif score >= 0.7:
                return True, "Hohe Konfidenz", "hoch"
            elif score >= 0.5:
                return True, "Mittlere Konfidenz", "mittel"
            elif score >= 0.3:
                return True, "Niedrige Konfidenz", "niedrig"
            else:
                return True, "Sehr niedrige Konfidenz", "sehr_niedrig"

class QCAPrompts:
    """Zentrale Klasse für alle QCA-AID Prompts"""
    
    def __init__(self, forschungsfrage: str, kodierregeln: Dict, deduktive_kategorien: Dict):
        self.FORSCHUNGSFRAGE = forschungsfrage
        self.KODIERREGELN = kodierregeln
        self.DEDUKTIVE_KATEGORIEN = deduktive_kategorien
        # FIX: Alias für bessere Kompatibilität hinzuFügen
        self.deduktive_kategorien = deduktive_kategorien

        # print(f"QCAPrompts initialisiert:")
        # print(f"- Forschungsfrage: {len(self.FORSCHUNGSFRAGE)} Zeichen")
        # print(f"- Kodierregeln: {len(self.KODIERREGELN)} Kategorien")
        # print(f"- Deduktive Kategorien: {len(self.DEDUKTIVE_KATEGORIEN)} Kategorien")
    
    def _format_categories_for_prompt(self, categories_overview: List[Dict]) -> str:
        """
        Formatiert Kategorien für Prompt-Verwendung.
        
        Args:
            categories_overview: Liste von Kategorie-Dictionaries mit 'name', 'definition', 'subcategories'
            
        Returns:
            Formatierter Kategorien-Text für Prompts
        """
        formatted_categories = []
        
        for cat in categories_overview:
            cat_name = cat.get('name', 'Unbekannte Kategorie')
            cat_definition = cat.get('definition', 'Keine Definition verfügbar')
            subcategories = cat.get('subcategories', {})
            
            # Hauptkategorie formatieren
            cat_text = f"**{cat_name}**: {cat_definition}"
            
            # Subkategorien hinzufügen falls vorhanden
            if subcategories:
                subcat_list = []
                for subcat_name, subcat_def in subcategories.items():
                    if isinstance(subcat_def, str):
                        subcat_list.append(f"  - {subcat_name}: {subcat_def}")
                    elif hasattr(subcat_def, 'definition'):
                        subcat_list.append(f"  - {subcat_name}: {subcat_def.definition}")
                    else:
                        subcat_list.append(f"  - {subcat_name}")
                
                if subcat_list:
                    cat_text += "\n" + "\n".join(subcat_list)
            
            formatted_categories.append(cat_text)
        
        return "\n\n".join(formatted_categories)
    
    def get_deductive_coding_prompt(self, chunk: str, categories_overview: List[Dict], 
                                    context_paraphrases: Optional[List[str]] = None) -> str:
        """
        Prompt für deduktive Kodierung von Text-Chunks mit Subkategorie-Validierung.
        
        Args:
            chunk: Zu kodierender Text
            categories_overview: Liste der verfügbaren Kategorien
            context_paraphrases: Optionale Liste von Paraphrasen vorheriger Chunks als Kontext
        """
        # FIX: Verwende spezialisierte Coding-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="coding", 
            context="kodierung"
        )
        
        # NEU: Erstelle Kontext-Sektion wenn Paraphrasen vorhanden
        context_section = ""
        if context_paraphrases and len(context_paraphrases) > 0:
            context_section = "\n## KONTEXT (vorherige Chunks):\n\n"
            context_section += "Die folgenden Paraphrasen geben Kontext über bereits analysierte Textabschnitte.\n"
            context_section += "Nutze sie NUR um das aktuelle Textsegment besser zu verstehen und die passendste Kategorie zu wählen.\n"
            context_section += "Der Kontext hilft bei impliziten Bezügen und thematischen Zusammenhängen.\n\n"
            
            for i, paraphrase in enumerate(context_paraphrases, 1):
                context_section += f"[Vorheriger Chunk {i}]: {paraphrase}\n"
            
            context_section += "\nWICHTIG: Kodiere nur das AKTUELLE TEXTSEGMENT unten, nicht die Kontext-Paraphrasen!\n"

        return f"""
            Analysiere folgenden Text im Kontext der Forschungsfrage:
            "{self.FORSCHUNGSFRAGE}"
            {context_section}
            ## TEXT ZU KODIEREN (aktuelles Segment):
            {chunk}

            ## KATEGORIENSYSTEM:
            {json.dumps(categories_overview, indent=2, ensure_ascii=False)}

            ## KODIERREGELN:
            {json.dumps(self.KODIERREGELN, indent=2, ensure_ascii=False)}

            {confidence_guidelines}

            ## WICHTIG - KATEGORIEN- UND SUBKATEGORIEN-ZUORDNUNG:
            
            1. HAUPTKATEGORIEN-WAHL:
            - Vergleiche den Text systematisch mit JEDER Kategoriendefinition
            - Nutze den Kontext um implizite Bezüge besser zu verstehen
            - Wähle GENAU EINE Hauptkategorie mit der besten Passung
            - Bei keiner eindeutigen Passung: "Keine passende Kategorie"
            
            2. SUBKATEGORIEN-VALIDIERUNG - KRITISCH WICHTIG:
            - SUBKATEGORIEN DÜRFEN NUR AUS DER GEWÄHLTEN HAUPTKATEGORIE STAMMEN
            - Prüfe NUR die Subkategorien der gewählten Hauptkategorie
            - NIEMALS Subkategorien aus anderen Hauptkategorien wählen
            - Wenn du "Akteure" als Hauptkategorie wählst, darfst du KEINE Subkategorien aus "Prozesse" oder anderen Kategorien nehmen
            
            3. SUBKATEGORIEN-AUSWAHL:
            - Analysiere ALLE Subkategorien der gewählten Hauptkategorie
            - Wähle die 1-2 am besten passenden Subkategorien
            - Bei mehreren passenden: Wähle die spezifischste
            - Begründe die Subkategorie-Wahl explizit
            
            4. QUALITÄTSKONTROLLE:
            - Validiere am Ende: Gehören alle gewählten Subkategorien zur gewählten Hauptkategorie?
            - Bei Zweifel: Lieber weniger Subkategorien wählen
            - Dokumentiere die Zuordnungslogik transparent

            ## LIEFERE:

            Antworte ausschließlich mit einem JSON-Objekt:
            {{
                "paraphrase": "Prägnante Paraphrase max. 40 Wörter",
                "keywords": "2-3 zentrale Begriffe",
                "category": "GENAU EINE Hauptkategorie oder 'Keine passende Kategorie'",
                "subcategories": ["NUR Subkategorien der gewählten Hauptkategorie"],
                "justification": "MUSS enthalten: 1. Warum diese Hauptkategorie? 2. Warum diese Subkategorien aus dieser Hauptkategorie? 3. Konkrete Textstellen als Belege",
                "confidence": {{
                    "total": 0.00-1.00,
                    "category": 0.00-1.00,
                    "subcategories": 0.00-1.00
                }},
                "text_references": ["Konkrete Textstellen"],
                "definition_matches": ["Passende Definitionsaspekte"],
                "subcategory_validation": {{
                    "chosen_main_category": "Name der gewählten Hauptkategorie",
                    "available_subcategories_for_main": ["Alle verfügbaren Subkategorien dieser Hauptkategorie"],
                    "chosen_subcategories": ["Gewählte Subkategorien"],
                    "validation_check": "Bestätigung: Alle gewählten Subkategorien gehören zur gewählten Hauptkategorie"
                }}
            }}
            """
    
    def get_relevance_check_prompt(self, segments_text: str, exclusion_rules: List[str]) -> str:
        """Prompt für Batch-Relevanzprüfung"""
        # Platzhalter - vollständiger Prompt für Relevanzprüfung mit Ausschlussregeln
        # FIX: Verwende spezialisierte Relevanz-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="relevance", 
            context="relevanz"
        )
        
        return f"""
            Analysiere die Relevanz der folgenden Textsegmente für die Forschungsfrage:
            "{self.FORSCHUNGSFRAGE}"
            
            {confidence_guidelines}

            PRÜFUNGSREIHENFOLGE - Analysiere jedes Segment in dieser Reihenfolge:

            1. THEMATISCHE VORPRÜFUNG:
            
            Führe ZUERST eine grundlegende thematische Analyse durch:
            - Identifiziere den Gegenstand, die Kernthemen und zentralen Konzepte der Forschungsfrage
            - Prüfe, ob der Text den Gegenstand und diese Kernthemen überhaupt behandelt
            - Stelle fest, ob ein hinreichender inhaltlicher Zusammenhang zur Forschungsfrage besteht
            - Falls NEIN: Sofort als nicht relevant markieren
            - Falls JA: Weiter mit detaillierter Prüfung

            2. AUSSCHLUSSKRITERIEN:
            {chr(10).join(f"- {rule}" for rule in exclusion_rules)}

            3. TEXTSORTENSPEZIFISCHE PRÜFUNG:

            INTERVIEWS/GESPRÄCHE:
            - Direkte Erfahrungsberichte zum Forschungsthema
            - Persönliche Einschätzungen relevanter Akteure
            - Konkrete Beispiele aus der Praxis
            - Implizites Erfahrungswissen zum Thema
            - NICHT relevant: Interviewerfrage, sofern Sie nicht den Interviewten paraphrasiert

            DOKUMENTE/BERICHTE:
            - Faktische Informationen zum Forschungsgegenstand
            - Formale Regelungen und Vorgaben
            - Dokumentierte Prozesse und Strukturen
            - Institutionelle Rahmenbedingungen

            PROTOKOLLE/NOTIZEN:
            - Beobachtete Handlungen und Interaktionen
            - Situationsbeschreibungen zum Thema
            - Dokumentierte Entscheidungen
            - Relevante Kontextinformationen

            4. QUALITÄTSKRITERIEN:

            AUSSAGEKRAFT:
            - Spezifische Information zum Forschungsthema
            - Substanzielle, nicht-triviale Aussagen
            - Präzise und gehaltvolle Information

            ANWENDBARKEIT:
            - Direkter Bezug zur Forschungsfrage
            - Beitrag zur Beantwortung der Forschungsfrage
            - Erkenntnispotenzial für die Untersuchung

            KONTEXTRELEVANZ:
            - Bedeutung für das Verständnis des Forschungsgegenstands
            - Hilfe bei der Interpretation anderer Informationen
            - Notwendigkeit für die thematische Einordnung

            TEXTSEGMENTE:
            {segments_text}

            Antworte NUR mit einem JSON-Objekt:
            {{
                "segment_results": [
                    {{
                        "segment_number": 1,
                        "is_relevant": true/false,
                        "confidence": 0.0-1.0,
                        "text_type": "interview|dokument|protokoll|andere",
                        "key_aspects": ["konkrete", "für", "die", "Forschungsfrage", "relevante", "Aspekte"],
                        "justification": "WICHTIG: Erkläre deine Entscheidung klar und eindeutig. Bei NICHT-relevanten Segmenten: Erkläre konkret WARUM das Segment nicht relevant ist (z.B. 'zu allgemein', 'kein direkter Bezug', 'nur administrative Info'). Bei relevanten Segmenten: Erkläre konkret WAS das Segment relevant macht."
                    }},
                    ...
                ]
            }}

            WICHTIGE HINWEISE:
            - Führe IMMER ZUERST die thematische Vorprüfung durch
            - Identifiziere die Kernthemen der Forschungsfrage und prüfe deren Präsenz im Text
            - Markiere Segmente als nicht relevant, wenn sie die Kernthemen nicht behandeln
            - Bei Unsicherheit (confidence < 0.75) als nicht relevant markieren
            - Gib bei JEDEM Segment eine klare Begründung (relevant UND nicht relevant)
            - Bei nicht-relevanten Segmenten: Erkläre konkret warum nicht relevant (z.B. "Behandelt nur Metadaten", "Thematisch nicht passend zur Forschungsfrage", "Zu unspezifisch")
            - Sei streng bei der thematischen Vorprüfung
            """

    def get_category_preferences_prompt(self, segments_text: str, 
                                       categories: Dict[str, Any]) -> str:
        """
        Prompt für reine Kategoriepräferenz-Bestimmung (OHNE Relevanzprüfung).
        
        Für bereits als relevant bestätigte Segmente.
        
        Args:
            segments_text: Formatierte Textsegmente
            categories: Dictionary mit CategoryDefinition-Objekten
            
        Returns:
            str: Formatierter Prompt für Kategoriepräferenzen
        """
        # FIX: Sichere Extraktion der Kategorie-Beschreibungen
        category_descriptions = {}
        for name, cat_def in categories.items():
            # FIX: Prüfe ob cat_def ein CategoryDefinition-Objekt ist
            if hasattr(cat_def, 'definition') and hasattr(cat_def, 'subcategories'):
                # CategoryDefinition-Objekt (abduktive Analyse)
                category_descriptions[name] = {
                    'definition': cat_def.definition,
                    'subcategories': list(cat_def.subcategories.keys()) if cat_def.subcategories else []
                }
            else:
                # FIX: Fallback für andere Datenstrukturen
                if isinstance(cat_def, dict):
                    # Dictionary-Format (bereits serialisiert)
                    category_descriptions[name] = {
                        'definition': cat_def.get('definition', 'Keine Definition verfügbar'),
                        'subcategories': list(cat_def.get('subcategories', {}).keys()) if cat_def.get('subcategories') else []
                    }
                else:
                    # String-Format (deduktive Analyse)
                    category_descriptions[name] = {
                        'definition': str(cat_def) if cat_def else 'Keine Definition verfügbar',
                        'subcategories': []
                    }

        # FIX: Verwende einheitliche Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="category_relevance", 
            context="kategorien"
        )

        return f"""
        AUFGABE: Bestimme für jedes bereits relevante Textsegment die Kategoriepräferenzen für die Forschungsfrage:
        "{self.FORSCHUNGSFRAGE}"
        
        WICHTIG: Alle Segmente sind bereits als RELEVANT zur Forschungsfrage bestätigt.
        Deine Aufgabe ist NUR die Bestimmung der Kategoriepräferenzen.
        
        HAUPTKATEGORIEN:
        {json.dumps(category_descriptions, indent=2, ensure_ascii=False)}
        
        TEXTSEGMENTE (alle bereits relevant):
        {segments_text}

        {confidence_guidelines}
        
        ANALYSIERE JEDES SEGMENT:
        
        KATEGORIE-PRÄFERENZ-BEWERTUNG:
        - Bewerte JEDE Hauptkategorie: Wie gut passt das Segment zu dieser Kategorie? (0.0-1.0)
        - Identifiziere 1-3 wahrscheinlichste Kategorien (Score ≥ 0.6)
        - Begründe kurz die Kategorie-Einschätzungen
        
        BEWERTUNGSKRITERIEN:
        - Score 0.8-1.0: "Perfekte Passung - Segment gehört definitiv zu dieser Kategorie"
        - Score 0.6-0.8: "Gute Passung - Segment passt gut zu dieser Kategorie"
        - Score 0.4-0.6: "Mögliche Passung - Segment könnte zu dieser Kategorie gehören"
        - Score 0.0-0.4: "Schwache Passung - Segment passt nicht gut zu dieser Kategorie"
        
        QUALITÄTSKONTROLLE:
        - Jedes Segment SOLLTE mindestens eine Kategorie ≥ 0.6 haben
        - Preferred_categories enthält nur Kategorien mit Score ≥ 0.6
        - Reasoning muss konkrete Textbezüge enthalten
        - Bei Unsicherheit zwischen Kategorien: beide einschließen wenn Score ≥ 0.6
        
        Antworte ausschließlich mit JSON:
        {{
            "segment_results": [
                {{
                    "segment_number": 1,
                    "category_scores": {{"Kategorie1": 0.8, "Kategorie2": 0.3, ...}},
                    "preferred_categories": ["Kategorie1"],
                    "reasoning": "Konkrete Begründung mit Textbezug für Kategorie-Einschätzung"
                }}
            ]
        }}
        """

    # FIX: Neue Prompt-Methode für Kategorie-Vorauswahl hinzuFügen
    def get_relevance_with_category_preselection_prompt(self, segments_text: str, 
                                                       categories: Dict[str, Any]) -> str:
        """
        Prompt für Relevanzprüfung mit Hauptkategorie-Vorauswahl
        
        Args:
            segments_text: Formatierte Textsegmente für Batch-Verarbeitung
            categories: Dictionary mit CategoryDefinition-Objekten
            
        Returns:
            str: Formatierter Prompt für die AI
        """
        # FIX: Sichere Extraktion der Kategorie-Beschreibungen
        category_descriptions = {}
        for name, cat_def in categories.items():
            # FIX: Prüfe ob cat_def ein CategoryDefinition-Objekt ist
            if hasattr(cat_def, 'definition') and hasattr(cat_def, 'subcategories'):
                # CategoryDefinition-Objekt (abduktive Analyse)
                category_descriptions[name] = {
                    'definition': cat_def.definition,
                    'subcategories': list(cat_def.subcategories.keys()) if cat_def.subcategories else []
                }
            else:
                # FIX: Fallback für andere Datenstrukturen
                if isinstance(cat_def, dict):
                    # Dictionary-Format (bereits serialisiert)
                    category_descriptions[name] = {
                        'definition': cat_def.get('definition', 'Keine Definition verfügbar'),
                        'subcategories': list(cat_def.get('subcategories', {}).keys()) if cat_def.get('subcategories') else []
                    }
                else:
                    # String-Format (deduktive Analyse)
                    category_descriptions[name] = {
                        'definition': str(cat_def) if cat_def else 'Keine Definition verfügbar',
                        'subcategories': []
                    }

        # FIX: Verwende einheitliche Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="category_relevance", 
            context="relevanz"
        )

        return f"""
        AUFGABE: Analysiere jedes Textsegment in zwei Stufen für die Forschungsfrage:
        "{self.FORSCHUNGSFRAGE}"
        
        HAUPTKATEGORIEN:
        {json.dumps(category_descriptions, indent=2, ensure_ascii=False)}
        
        TEXTSEGMENTE:
        {segments_text}

        {confidence_guidelines}
        
        ANALYSIERE JEDES SEGMENT:
        
        1. ALLGEMEINE RELEVANZ:
        - Ist das Segment DIREKT für die Forschungsfrage relevant? (ja/nein)
        - NUR JA wenn das Segment konkrete Inhalte zur Forschungsfrage enthält
        - NEIN wenn das Segment nur allgemeine/unspezifische Informationen enthält
        
        2. KATEGORIE-PRÄFERENZ (nur falls relevant):
        - Bewerte JEDE Hauptkategorie: Wie wahrscheinlich passt das Segment zu dieser Kategorie? (0.0-1.0)
        - Identifiziere 1-3 wahrscheinlichste Kategorien (Score ≥ 0.6)
        - Begründe kurz die Kategorie-Einschätzungen
        
        WICHTIG FÜR RELEVANZ-BEWERTUNG:
        - Sei STRENG bei der Relevanzprüfung
        - Nur Segmente mit DIREKTEM Bezug zur Forschungsfrage sind relevant
        - Allgemeine Informationen ohne spezifischen Bezug sind NICHT relevant
        - Test-Segmente oder Beispiel-Texte sind NICHT relevant
        - Bei Zweifel: eher als NICHT relevant einstufen
        
        KONSISTENZ-REGEL:
        - Wenn du is_relevant: false setzt, dann erkläre in der Begründung WARUM es nicht relevant ist
        - Wenn du is_relevant: true setzt, dann erkläre in der Begründung WAS es relevant macht
        - Vermeide widersprüchliche Aussagen zwischen is_relevant und reasoning
        
        WICHTIG FÜR KATEGORIE-VORAUSWAHL:
        - Sei konservativ aber nicht zu restriktiv
        - Lieber 2-3 Kategorien vorschlagen als nur eine, wenn mehrere plausibel sind
        - Score 0.6+ bedeutet "definitiv relevant für diese Kategorie"
        - Score 0.4-0.6 bedeutet "möglicherweise relevant"  
        - Score <0.4 bedeutet "wahrscheinlich nicht relevant"
        - Bei Unsicherheit zwischen zwei Kategorien: beide einschließen
        
        QUALITÄTSKONTROLLE:
        - Jedes relevante Segment MUSS mindestens eine Kategorie ≥ 0.6 haben
        - Preferred_categories enthält nur Kategorien mit Score ≥ 0.6
        - Reasoning muss konkrete Textbezüge enthalten
        
        Antworte ausschließlich mit JSON:
        {{
            "segment_results": [
                {{
                    "segment_number": 1,
                    "is_relevant": true/false,
                    "relevance_scores": {{"Kategorie1": 0.8, "Kategorie2": 0.3, ...}},
                    "preferred_categories": ["Kategorie1", "Kategorie2"],
                    "reasoning": "WICHTIG: Erkläre deine Entscheidung klar und eindeutig. Bei NICHT-relevanten Segmenten: Erkläre konkret WARUM das Segment nicht relevant ist (z.B. 'zu allgemein', 'kein direkter Bezug', 'nur administrative Info'). Bei relevanten Segmenten: Erkläre konkret WAS das Segment relevant macht und welche Kategorien passen."
                }}
            ]
        }}
        """

    

    def get_multiple_category_relevance_prompt(self, segments_text: str, category_descriptions: List[Dict], multiple_threshold: float) -> str:
        """Prompt für Mehrfachkodierungs-Relevanzprüfung"""
        # FIX: Verwende spezialisierte Relevanz-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="relevance", 
            context="relevanz"
        )
        
        return f"""
            Analysiere die folgenden Textsegmente auf Mehrfachkodierung für verschiedene Hauptkategorien.
            
            FORSCHUNGSFRAGE: "{self.FORSCHUNGSFRAGE}"
            
            HAUPTKATEGORIEN:
            {json.dumps(category_descriptions, indent=2, ensure_ascii=False)}

            {confidence_guidelines}
            
            AUFGABE - MEHRFACHKODIERUNGS-ANALYSE:
            Prüfe für jedes Segment, ob es für MEHRERE Hauptkategorien gleichzeitig relevant ist.
            Ein Segment kann mehrfach kodiert werden, wenn es mindestens {multiple_threshold} ({int(float(multiple_threshold)*100)}%) Relevanz 
            für verschiedene Hauptkategorien hat.
            
            BEISPIEL für Mehrfachkodierung:
            "Wir haben seit Jahren ein Referat für Forschungsförderung, aber am Ende sind das immer 
            Aushandlungsprozesse zwischen den Beteiligten"
            → Könnte sowohl "Strukturen" (Referat) als auch "Prozesse" (Aushandlung) zugeordnet werden
            
            KRITERIEN FÜR MEHRFACHKODIERUNG:
            - Mindestens {int(float(multiple_threshold)*100)}% Relevanz für jede zugeordnete Kategorie
            - Verschiedene Aspekte des Segments sprechen verschiedene Kategorien an
            - Keine künstliche Aufblähung - nur wenn wirklich mehrere Themen behandelt werden
            - Textinhalt muss substanziell genug für mehrere Zuordnungen sein
            
            TEXTSEGMENTE:
            {segments_text}

            Antworte NUR mit einem JSON-Objekt:
            {{
                "segment_results": [
                    {{
                        "segment_number": 1,
                        "relevant_categories": [
                            {{
                                "category": "Kategoriename",
                                "relevance_score": 0.0-1.0,
                                "justification": "Begründung warum relevant",
                                "text_aspects": ["Welche Textteile sprechen diese Kategorie an"]
                            }}
                        ],
                        "multiple_coding": true/false,
                        "multiple_coding_justification": "Warum Mehrfachkodierung gerechtfertigt ist oder nicht"
                    }}
                ]
            }}
            """

    def get_focus_coding_prompt(self, chunk: str, categories_overview: List[Dict], 
                                focus_category: str, focus_context: Dict,
                                context_paraphrases: Optional[List[str]] = None) -> str:
        """Prompt für fokussierte Kodierung mit strenger Subkategorie-Validierung"""
        # FIX: Verwende spezialisierte Subkategorie-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="subcategory", 
            context="subkategorie"
        )
        
        # NEU: Erstelle Kontext-Sektion wenn Paraphrasen vorhanden
        context_section = ""
        if context_paraphrases and len(context_paraphrases) > 0:
            context_section = "\n## KONTEXT (vorherige Chunks):\n\n"
            context_section += "Die folgenden Paraphrasen geben Kontext über bereits analysierte Textabschnitte.\n"
            context_section += "Nutze sie NUR um das aktuelle Textsegment besser zu verstehen.\n\n"
            
            for i, paraphrase in enumerate(context_paraphrases, 1):
                context_section += f"[Vorheriger Chunk {i}]: {paraphrase}\n"
            
            context_section += "\nWICHTIG: Kodiere nur das AKTUELLE TEXTSEGMENT unten!\n"
        
        return f"""
            MEHRFACHKODIERUNGS-FOKUS auf: "{focus_category}"
            
            Forschungsfrage: "{self.FORSCHUNGSFRAGE}"

            {confidence_guidelines}
            {context_section}
            ## TEXT ZU KODIEREN (aktuelles Segment):
            {chunk}

            ## FOCUS-KONTEXT:
            - Relevanz-Score für {focus_category}: {focus_context.get('relevance_score', 0):.2f}
            - Begründung: {focus_context.get('justification', '')}
            
            ## KATEGORIENSYSTEM:
            {json.dumps(categories_overview, indent=2, ensure_ascii=False)}

            ## KRITISCH WICHTIG - SUBKATEGORIE-ZUORDNUNG:
            
            1. HAUPTKATEGORIEN-ENTSCHEIDUNG:
            - Nutze den Kontext um implizite Bezüge besser zu verstehen
            - Bevorzuge "{focus_category}" wenn >= 60% Relevanz
            - Andere Kategorie nur bei deutlich höherer Passung (>80%)
            
            2. SUBKATEGORIEN-VALIDIERUNG - ABSOLUT KRITISCH:
            - SUBKATEGORIEN DÜRFEN NUR AUS DER GEWÄHLTEN HAUPTKATEGORIE STAMMEN
            - Wenn du "{focus_category}" wählst, prüfe NUR dessen Subkategorien
            - NIEMALS Subkategorien aus anderen Hauptkategorien mischen
            - Validiere am Ende: Gehören alle Subkategorien zur gewählten Hauptkategorie?
            
            3. FOCUS-SPEZIFISCHE SUBKATEGORIEN:
            - Bei Wahl der Fokus-Kategorie: Analysiere deren Subkategorien gründlich
            - Wähle die 1-2 am besten zum Text passenden Subkategorien
            - Begründe die Subkategorie-Wahl im Kontext des Fokus

            ## LIEFERE:

            Antworte ausschließlich mit einem JSON-Objekt:
            {{
                "paraphrase": "Fokussierte Paraphrase max. 40 Wörter",
                "keywords": "2-3 Begriffe mit Fokus-Bezug",
                "category": "Gewählte Hauptkategorie",
                "subcategories": ["NUR Subkategorien der gewählten Hauptkategorie"],
                "justification": "MUSS enthalten: 1. Fokus-Einfluss 2. Hauptkategorie-Begründung 3. Subkategorie-Zuordnung zur gewählten Hauptkategorie",
                "confidence": {{
                    "total": 0.00-1.00,
                    "category": 0.00-1.00,
                    "subcategories": 0.00-1.00
                }},
                "focus_adherence": {{
                    "followed_focus": true/false,
                    "focus_category_score": 0.00-1.00,
                    "chosen_category_score": 0.00-1.00,
                    "deviation_reason": "Grund für Abweichung (falls zutreffend)"
                }},
                "subcategory_validation": {{
                    "chosen_main_category": "Name der gewählten Hauptkategorie",
                    "available_subcategories_for_main": ["Alle Subkategorien dieser Hauptkategorie"],
                    "chosen_subcategories": ["Gewählte Subkategorien"],
                    "validation_confirmed": true/false,
                    "validation_note": "Bestätigung der korrekten Zuordnung"
                }}
            }}
            """
    
    def get_focus_context_coding_prompt(self, chunk: str, categories_overview: List[Dict], focus_category: str, focus_context: Dict, current_summary: str, position_info: Dict, summary_update_prompt: str, update_summary: bool = False) -> str:
        """Prompt für fokussierte Kodierung mit Kontext"""

        # FIX: Verwende spezialisierte Coding-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="coding", 
            context="kodierung"
        )
        
        return f"""
            ## AUFGABE 1: FOKUSSIERTE KODIERUNG MIT KONTEXT
            
            Analysiere folgenden Text mit FOKUS auf die Kategorie "{focus_category}".
            
            MEHRFACHKODIERUNGS-KONTEXT:
            - Dieser Text wurde bereits als relevant für mehrere Kategorien identifiziert
            - Fokus-Kategorie: {focus_category}
            - Relevanz-Score: {focus_context.get('relevance_score', 0):.2f}
            - Begründung: {focus_context.get('justification', '')}
            - Relevante Textaspekte: {', '.join(focus_context.get('text_aspects', []))}
            
            ### PROGRESSIVER DOKUMENTKONTEXT (bisherige relevante Inhalte):
            {current_summary if current_summary else "Noch keine relevanten Inhalte für dieses Dokument erfasst."}
            
            Forschungsfrage: "{self.FORSCHUNGSFRAGE}"
            
            ### TEXTSEGMENT ZU KODIEREN:
            {chunk}
            
            {position_info}

            ### KATEGORIENSYSTEM:
            {json.dumps(categories_overview, indent=2, ensure_ascii=False)}

            ### KODIERREGELN:
            {json.dumps(self.KODIERREGELN, indent=2, ensure_ascii=False)}

            {confidence_guidelines}

            ### WICHTIG - FOKUSSIERTE MEHRFACHKODIERUNG MIT KONTEXT: 
            
            1. FOKUS-ORIENTIERTE ANALYSE:
            - Du kodierst diesen Text im Rahmen einer MEHRFACHKODIERUNG mit KONTEXT
            - Konzentriere dich auf die Aspekte des Texts, die zu "{focus_category}" passen
            - Berücksichtige den DOKUMENTKONTEXT für tieferes Verständnis des TEXTSEGMENTS
            - Die bereits identifizierten relevanten Textaspekte sollen berücksichtigt werden
            - Andere Kategorien nur wählen, wenn sie DEUTLICH besser passen (>80% vs <60% Fokus-Kategorie)
            
            2. KONTEXT-INTEGRATION:
            - Prüfe, ob das aktuelle TEXTSEGMENT bisherige Einschätzungen bestätigt, ergänzt oder widerspricht
            - Formuliere eine klare Begründung, die den Kontext explizit einbezieht
            
            3. FOKUS-KATEGORIE BEVORZUGUNG:
            - "{focus_category}" sollte bevorzugt werden, wenn sie mindestens 60% Relevanz hat
            - Berücksichtige den vorgegebenen Fokus-Kontext bei der Entscheidung
            - Dokumentiere, ob du dem Fokus gefolgt bist oder eine andere Kategorie gewählt hast
            
            4. SUBKATEGORIEN-FOKUS:
            - Bei Wahl der Fokus-Kategorie: Analysiere ALLE Subkategorien gründlich
            - Wähle die Subkategorien, die zu den relevanten Textaspekten passen
            
            5. TRANSPARENZ:
            - Dokumentiere in der Begründung den Einfluss sowohl des Kategorie-Fokus als auch des Dokument-Kontexts
            - Erkläre, warum du dem Fokus gefolgt bist oder davon abgewichen bist

            ### LIEFERE:

            1. PARAPHRASE:
            - Erfasse den zentralen Inhalt mit Fokus auf "{focus_category}"-relevante Aspekte
            - IGNORIERE dafür den Dokumenttext
            - Betone die Aspekte, die zur Fokus-Kategorie passen
            - Max. 40 Wörter, sachlich und deskriptiv

            2. SCHLÜSSELWÖRTER:
            - 2-3 zentrale Begriffe mit Bezug zur Fokus-Kategorie
            - Berücksichtige die identifizierten relevanten Textaspekte

            3. KATEGORIENZUORDNUNG:
            - Bevorzuge "{focus_category}" wenn >= 60% Relevanz gegeben
            - Andere Kategorie nur bei deutlich höherer Passung (>80%)
            - Begründung muss sowohl Fokus-Einfluss als auch Kontext-Einfluss dokumentieren

            {summary_update_prompt}

            Antworte mit EINEM JSON-Objekt, das {'BEIDE Aufgaben umfasst' if update_summary else 'die Kodierung umfasst'}:
            {{
                "coding_result": {{
                    "paraphrase": "Fokussierte Paraphrase des TEXTSEGMENTS hier",
                    "keywords": "Fokussierte Schlüsselwörter hier",
                    "category": "Name der Hauptkategorie",
                    "subcategories": ["Subkategorie1", "Subkategorie2"],
                    "justification": "Begründung mit Fokus- und Kontext-Dokumentation: 1. Einfluss des Kategorie-Fokus, 2. Einfluss des Dokumentkontexts, 3. Konkrete Textstellen",
                    "confidence": {{
                        "total": 0.00-1.00,
                        "category": 0.00-1.00,
                        "subcategories": 0.00-1.00
                    }},
                    "text_references": ["Relevante Textstellen"],
                    "definition_matches": ["Passende Definitionsaspekte"],
                    "context_influence": "Wie der Dokumentkontext die Kodierung beeinflusst hat",
                    "focus_adherence": {{
                        "followed_focus": true/false,
                        "focus_category_score": 0.00-1.00,
                        "chosen_category_score": 0.00-1.00,
                        "deviation_reason": "Grund für Abweichung vom Fokus (falls zutreffend)"
                    }}
                }}{"," if update_summary else ""}
                {"updated_summary: Das aktualisierte Document-Summary mit max. 70 Wörtern" if update_summary else ""}
            }}
            """

    def get_category_batch_analysis_prompt(self, current_categories_text: str, segments: List[str], mode_instructions: str, json_schema: str) -> str:
        """Prompt für Batch-Kategorienanalyse (induktiv/abduktiv)"""
        
        # FIX: Verwende spezialisierte Coding-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="coding", 
            context="kategorienentwicklung"
        )
        
        segments_text = "\n\n".join([f"SEGMENT {i+1}:\n{segment}" for i, segment in enumerate(segments)])
        
        return f"""
        Führe eine systematische Kategorienanalyse für die folgenden Textsegmente durch.
        
        FORSCHUNGSFRAGE:
        {self.FORSCHUNGSFRAGE}
        
        KODIERREGELN:
        {chr(10).join(f"- {regel}" for regel in self.KODIERREGELN)}
        
        AKTUELLE KATEGORIEN:
        {current_categories_text}
        
        MODUS-SPEZIFISCHE ANWEISUNGEN:
        {mode_instructions}
        
        {confidence_guidelines}
        
        TEXTSEGMENTE ZU ANALYSIEREN:
        {segments_text}
        
        ALLGEMEINE QUALITÄTSKRITERIEN:
        
        1. FORSCHUNGSFRAGEN-BEZUG:
        - Alle Kategorien müssen zur Beantwortung der Forschungsfrage beitragen
        - Kategorien sollen zentrale Aspekte der Forschungsfrage abdecken
        - Irrelevante oder tangentiale Aspekte ausschließen
        
        2. KATEGORIENQUALITÄT:
        - Präzise und eindeutige Benennung
        - Klare Abgrenzung zwischen Kategorien
        - Empirisch gut belegte Kategorien (durch Textstellen)
        - Angemessene Abstraktionsebene
        
        3. SYSTEMATIK:
        - Konsistente Kategorienentwicklung über alle Segmente
        - Berücksichtigung bestehender Kategorien
        - Logische Kategorienstruktur
        
        4. BEGRÜNDUNG:
        - Jede Kategorienentscheidung muss begründet werden
        - Bezug zu konkreten Textstellen herstellen
        - Verbindung zur Forschungsfrage erläutern
        
        Antworte ausschließlich im angegebenen JSON-Format:
        {json_schema}
        """

    def get_grounded_analysis_prompt(self, segments: List[str], existing_subcodes: List[str], json_schema: str) -> str:
        """Prompt für Grounded Theory Analyse"""

        # FIX: Verwende spezialisierte Coding-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="coding", 
            context="kodierung"
        )

        return f"""
            Analysiere folgende Textsegmente im Sinne der Grounded Theory.
            Identifiziere Subcodes und Keywords, ohne sie bereits Hauptkategorien zuzuordnen.
            
            FORSCHUNGSFRAGE (ZENTRALE ORIENTIERUNG):
            {self.FORSCHUNGSFRAGE}
            
            WICHTIG - FORSCHUNGSFRAGE BERÜCKSICHTIGEN:
            - Prüfe ZUERST, ob ein Textsegment zur Forschungsfrage relevant ist
            - Entwickle NUR Subcodes für Aspekte, die zur Beantwortung der Forschungsfrage beitragen
            - Filtere textnahe Details heraus, die keinen Bezug zur Forschungsfrage haben
            - Subcode-Namen sollten Terminologie der Forschungsfrage aufgreifen, wo sinnvoll
            
            {"BEREITS IDENTIFIZIERTE SUBCODES:" if existing_subcodes else ""}
            {json.dumps(existing_subcodes, indent=2, ensure_ascii=False) if existing_subcodes else ""}
            
            TEXTSEGMENTE:
            {json.dumps(segments, indent=2, ensure_ascii=False)}

            {confidence_guidelines}
            
            ANWEISUNGEN FÜR DEN GROUNDED THEORY MODUS:
            
            1. OFFENES KODIEREN MIT FORSCHUNGSFOKUS UND KLARER ABSTRAKTIONSHIERARCHIE:
            - KEYWORDS: Extrahiere TEXTNAHE, KONKRETE Begriffe und Phrasen direkt aus dem Text
            * NUR wenn sie zur Forschungsfrage relevant sind
            - SUBCODES: Entwickle eine ERSTE ABSTRAKTIONSEBENE basierend auf den Keywords
            * Fasse ähnliche Keywords zu einem gemeinsamen, leicht abstrahierten Subcode zusammen
            * Verwende eine präzisere Sprache als für spätere Hauptkategorien
            * Formuliere Subcodes als analytische Konzepte, NICHT als bloße Wiederholung der Keywords
            * Berücksichtige die Terminologie der Forschungsfrage
            
            2. WICHTIG - PROGRESSIVE ANREICHERUNG DES SUBCODE-SETS:
            - Berücksichtige die bereits identifizierten Subcodes
            - Fokussiere auf NEUE KOMPLEMENTÄRE Subcodes, die noch nicht erfasst wurden
            - Bei thematischer Überschneidung mit bestehenden Subcodes: verfeinere oder differenziere
            - Keine Duplikate zu bestehenden Subcodes erstellen
            
            3. QUALITÄTSKRITERIEN FÜR SUBCODES:
            - Empirisch gut belegt durch Textstellen
            - Präzise und eindeutig benannt
            - Analytisch wertvoll für die Forschungsfrage (ZENTRAL!)
            - Trennscharf zu anderen Subcodes
            - ABSTRAKTION: Erkennbar abstrakter als die Keywords, aber konkreter als spätere Hauptkategorien
            - Nicht zu allgemein oder zu spezifisch
            
            4. KEYWORDS IMMER MIT SUBCODES VERKNÜPFEN:
            - Jeder Subcode muss mindestens 3-5 Keywords erhalten
            - Keywords sollen DIREKTER TEXTAUSZUG sein, während Subcodes erste Interpretation darstellen
            - Keywords werden später für die Kategorienbildung verwendet
            - Keine allgemeinen Keywords ohne Subcode-Zugehörigkeit
            
            5. UNTERSCHIED KEYWORDS VS. SUBCODES - BEISPIEL:
            Keywords: "Antragsverfahren", "Bewertungskriterien", "Fördermittelbeantragung"
            Subcode: "Formale Verfahrensstruktur"
            
            Keywords: "mangelnde Abstimmung", "fehlende Kommunikation", "keine Rückmeldung" 
            Subcode: "Kommunikationsdefizite"
            
            Antworte NUR mit einem JSON-Objekt:
            {json_schema}
            """
    def get_main_categories_generation_prompt(self, subcodes_data: List[Dict], top_keywords: List[str], avg_confidence: float) -> str:
        """
        Erstellt einen Prompt für die Generierung von Hauptkategorien aus gesammelten Subcodes.
        
        Args:
            subcodes_data: Liste der gesammelten Subcodes mit Definitionen und Keywords
            top_keywords: Liste der häufigsten Keywords als (keyword, count) Tupel
            avg_confidence: Durchschnittliche Konfidenz der Subcodes
            
        Returns:
            str: Formatierter Prompt für die Hauptkategorien-Generierung
        """
        # FIX: Verwende spezialisierte Coding-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="coding", 
            context="kodierung"
        )
        
        # Erstelle formatierten Subcode-Text
        subcodes_text = ""
        for i, subcode in enumerate(subcodes_data, 1):
            keywords_str = ', '.join(subcode['keywords'][:10])  # Top 10 Keywords
            subcodes_text += f"""
            
            {i}. Subcode: {subcode['name']}
            Definition: {subcode['definition']}
            Keywords: {keywords_str}
            Konfidenz: {subcode['confidence']:.2f}
            Textbelege: {len(subcode['evidence'])}
            """
        
        top_keywords_str = ', '.join([kw for kw, _ in top_keywords[:10]])
        
        return f"""
        FORSCHUNGSFRAGE (ZENTRALE ORIENTIERUNG):
        {self.FORSCHUNGSFRAGE}
        
        GROUNDED THEORY: Generiere Hauptkategorien aus gesammelten Subcodes
        
        Du erhältst {len(subcodes_data)} Subcodes mit ihren Keywords, die während einer Grounded Theory Analyse gesammelt wurden. 
        Deine Aufgabe ist es, diese zu thematisch kohärenten Hauptkategorien zu gruppieren.

        {confidence_guidelines}
        
        WICHTIG - FORSCHUNGSFRAGE BERÜCKSICHTIGEN:
        - Jede Hauptkategorie MUSS einen erkennbaren Beitrag zur Beantwortung der Forschungsfrage leisten
        - Gruppiere Subcodes so, dass die entstehenden Hauptkategorien relevante Aspekte der Forschungsfrage abbilden
        - Kategoriennamen sollten Terminologie der Forschungsfrage aufgreifen, wo sinnvoll
        - Filtere Subcodes heraus, die keinen relevanten Bezug zur Forschungsfrage haben
        
        GESAMMELTE SUBCODES UND KEYWORDS:
        {subcodes_text}
        
        GROUNDED THEORY ANALYSE-ANWEISUNGEN:
        1. Analysiere die thematischen Verbindungen zwischen den Subcodes IM KONTEXT DER FORSCHUNGSFRAGE
        2. Gruppiere verwandte Subcodes zu 3-6 kohärenten Hauptkategorien
        3. Jede Hauptkategorie sollte mindestens 2-3 Subcodes enthalten
        4. Erstelle aussagekräftige Namen und Definitionen für die Hauptkategorien
        5. Ordne die Subcodes als Subkategorien den Hauptkategorien zu
        6. Berücksichtige die Keyword-Häufigkeiten zur Themenfindung
        
        TOP KEYWORDS ZUR ORIENTIERUNG: {top_keywords_str}

        AUFGABE - DREISTUFIGE ABSTRAKTIONSHIERARCHIE:
            1. KEYWORDS (bereits vorhanden): Textnahe, spezifische Begriffe und Phrasen direkt aus dem Material
            2. SUBCODES (bereits vorhanden): Erste Abstraktionsebene, fassen ähnliche Keywords zusammen
            3. HAUPTKATEGORIEN (zu generieren): DEUTLICH HÖHERE ABSTRAKTIONSEBENE als Subcodes

            RICHTLINIEN FÜR HAUPTKATEGORIEN:
            1. MAXIMALE ABSTRAKTION: Eine Hauptkategorie sollte deutlich abstrakter sein als die zugeordneten Subcodes
            2. THEORETISCHE KONZEPTE: Nutze sozialwissenschaftliche Konzepte und Terminologie auf hohem Abstraktionsniveau
            3. ÜBERGREIFENDE LOGIK: Identifiziere die gemeinsame strukturelle oder prozessuale Logik hinter verschiedenen Subcodes
            4. ANALYTISCHE DIMENSIONEN: Entwickle Hauptkategorien entlang analytischer Dimensionen wie:
            - Strukturelle vs. handlungsbezogene Aspekte
            - Formale vs. informelle Prozesse
            - Institutionelle vs. individuelle Faktoren
            - Manifeste vs. latente Funktionen

            BEISPIEL FÜR ABSTRAKTIONSHIERARCHIE:
            - Keywords: "Stundenplangestaltung", "Raumzuweisung", "Prüfungstermine"
            - Subcode: "Administrative Koordinationsaufgaben"
            - Hauptkategorie: "Organisationale Steuerungsmechanismen"

            - Keywords: "Zielvereinbarungen", "Budgetverhandlungen", "Leistungsmessungen" 
            - Subcode: "Leistungsbezogene Mittelzuweisung"
            - Hauptkategorie: "Governance durch Anreizstrukturen"

            STRATEGIE FÜR STÄRKERE VERDICHTUNG:
            - Identifiziere übergeordnete theoretische Konzepte, die mehrere ähnliche Phänomene zusammenfassen
            - Bevorzuge breitere Kategorien mit mehr Subcodes gegenüber eng definierten Kategorien mit wenigen Subcodes
            - Verwende für die Hauptkategorien Begriffe mit DEUTLICH HÖHEREM Abstraktionsgrad als die Subcodes
            - Erstelle eine konzeptionelle Hierarchie, die Zusammenhänge zwischen den Kategorien verdeutlicht
            - Ordne grenzwertige Subcodes der jeweils passenderen Hauptkategorie zu

            WICHTIG - REDUZIERUNG DER GESAMTANZAHL:
            - Identifiziere Redundanzen oder konzeptionelle Überlappungen zwischen möglichen Hauptkategorien
            - Erstelle WENIGER, dafür BREITERE Hauptkategorien (maximal 6)
            - Vermeide rein deskriptive oder zu eng gefasste Hauptkategorien
            - Arbeite auf höherer theoretischer Abstraktionsebene

            WICHTIG - VOLLSTÄNDIGES MAPPING:
            - Für das spätere Matching ist es ZWINGEND ERFORDERLICH, dass JEDER Subcode einer Hauptkategorie zugeordnet wird
            - Es muss eine klare 1:n Beziehung zwischen Hauptkategorien und Subcodes geben
            - Jeder Subcode muss genau einer Hauptkategorie zugeordnet sein
        
        Antworte AUSSCHLIESSLICH mit diesem JSON-Format:
        {{
            "main_categories": [
                {{
                    "name": "Hauptkategorie Name",
                    "definition": "Umfassende Definition der Hauptkategorie (mindestens 30 Wörter)",
                    "characteristic_keywords": ["Schlüssel", "Keywords", "für", "diese", "Kategorie"],
                    "examples": ["Beispiel1", "Beispiel2"],
                    "rules": ["Kodierregel1", "Kodierregel2"],
                    "subcodes": [
                        {{
                            "name": "Subcode Name aus der Liste oben",
                            "definition": "Definition des Subcodes",
                            "rationale": "Warum dieser Subcode zu dieser Hauptkategorie gehört"
                        }}
                    ],
                    "thematic_justification": "Warum diese Subcodes thematisch zusammengehören"
                }}
            ],
            "subcode_mappings": {{
                "Subcode Name": "Hauptkategorie Name"
            }},
            "meta_analysis": {{
                "total_subcodes_processed": {len(subcodes_data)},
                "total_main_categories": 0,
                "theoretical_saturation": 0.0,
                "coverage": 0.0,
                "justification": "Begründung für die Kategorienbildung"
            }}
        }}
        """
    
    def get_analyze_for_subcategories_prompt(self, segments_text: str, categories_context: List[Dict]) -> str:
        
        # FIX: Verwende spezialisierte Coding-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="coding", 
            context="kodierung"
        )
        
        return f"""
        ABDUKTIVER MODUS: Entwickle NUR neue Subkategorien für bestehende Hauptkategorien.

        BESTEHENDE HAUPTKATEGORIEN:
        {json.dumps(categories_context, indent=2, ensure_ascii=False)}

        {confidence_guidelines}

        STRIKTE REGELN FÜR ABDUKTIVEN MODUS:
        - KEINE neuen Hauptkategorien entwickeln
        - NUR neue Subkategorien für bestehende Hauptkategorien
        - Subkategorien müssen neue, relevante Themenaspekte der Hauptkategorie abbilden
        - Mindestens 2 Textbelege pro neuer Subkategorie
        - Konfidenz mindestens 0.7
        - Nur Subkategorien vorschlagen, die das bestehende System sinnvoll verfeinern
        
        TEXTSEGMENTE:
        {segments_text}
        
        Antworte NUR mit JSON:
        {{
            "extended_categories": {{
                "hauptkategorie_name": {{
                    "new_subcategories": [
                        {{
                            "name": "Subkategorie Name",
                            "definition": "Definition der Subkategorie",
                            "evidence": ["Textbelege aus den Segmenten"],
                            "confidence": 0.0-1.0,
                            "thematic_novelty": "Warum diese Subkategorie einen neuen Aspekt der Hauptkategorie abbildet"
                        }}
                    ]
                }}
            }},
            "development_assessment": {{
                "subcategories_developed": 0,
                "saturation_indicators": ["Liste von Hinweisen auf Sättigung"],
                "recommendation": "continue/pause/stop"
            }}
        }}
        """
    
    def get_definition_enhancement_prompt(self, category_data: Dict) -> str:
        """
        Erstellt einen Prompt zur Verbesserung oder Zusammenführung von Kategoriendefinitionen.
        
        Args:
            category_data: Dictionary mit Kategorieinformationen
                        - Für einzelne Kategorie: 'name', 'definition', 'examples'
                        - Für Zusammenführung: 'definition1', 'definition2'
            
        Returns:
            str: Formatierter Prompt für die Definitionsverbesserung
        """
        # FIX: Verwende spezialisierte Coding-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="coding", 
            context="kodierung"
        )

        if 'definition1' in category_data and 'definition2' in category_data:
            # Zusammenführung von zwei Definitionen
            return f"""
            Führe diese beiden Kategoriendefinitionen zu einer kohärenten, präzisen Definition zusammen:
            
            Definition 1: {category_data['definition1']}
            Definition 2: {category_data['definition2']}
            
            Erstelle eine neue Definition die:
            - Die Kernaspekte beider Definitionen vereint
            - Redundanzen eliminiert
            - Klar und verständlich ist
            - Mindestens 20 Wörter hat
            - Zur Forschungsfrage "{self.FORSCHUNGSFRAGE}" passt

            BEISPIEL GUTER DEFINITION:
            "Qualitätssicherungsprozesse umfassen alle systematischen Verfahren und Maßnahmen zur 
            Überprüfung und Gewährleistung definierter Qualitätsstandards in der Hochschule. Sie 
            beinhalten sowohl interne Evaluationen und Audits als auch externe Begutachtungen und 
            Akkreditierungen. Im Gegensatz zum allgemeinen Qualitätsmanagement fokussieren sie 
            sich auf die konkrete Durchführung und Dokumentation von qualitätssichernden 
            Maßnahmen."

            {confidence_guidelines}

            Antworte nur mit der neuen Definition:
            """
        else:
            # Verbesserung einer einzelnen Definition
            return f"""
            Verbessere die folgende Kategoriendefinition für die qualitative Inhaltsanalyse:
            
            Kategorie: {category_data.get('name', 'Unbenannt')}
            Aktuelle Definition: {category_data.get('definition', '')}
            Beispiele: {', '.join(category_data.get('examples', []))}
            
            Erstelle eine verbesserte Definition die:
            - Präzise und klar formuliert ist
            - Mindestens 20 Wörter umfasst
            - Zur Forschungsfrage "{self.forschungsfrage}" passt
            - Die Abgrenzung zu anderen Kategorien ermöglicht
            - Konkrete Kodierhinweise enthält
            
            {confidence_guidelines}

            Antworte nur mit der verbesserten Definition:
            """
    def _get_subcategory_generation_prompt(self, category: Dict) -> str:
        """
        Prompt zur Generierung fehlender Subkategorien.
        """
        # FIX: Verwende spezialisierte Coding-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="coding", 
            context="kodierung"
        )

        return f"""Entwickle passende Subkategorien für die folgende Hauptkategorie.

        HAUPTKATEGORIE: {category['name']}
        DEFINITION: {category['definition']}
        BEISPIEL: {category.get('example', '')}

        ANFORDERUNGEN AN SUBKATEGORIEN:
        1. 2-4 logisch zusammengehörende Aspekte
        2. Eindeutig der Hauptkategorie zuordenbar
        3. Sich gegenseitig ausschließend
        4. Mit kurzer Erläuterung (1 Satz)
        5. Relevant für die Forschungsfrage: {self.FORSCHUNGSFRAGE}

        BEISPIEL GUTER SUBKATEGORIEN:
        "Qualitätssicherungsprozesse":
        - "Interne Evaluation": Systematische Selbstbewertung durch die Hochschule
        - "Externe Begutachtung": Qualitätsprüfung durch unabhängige Gutachter
        - "Akkreditierung": Formale Anerkennung durch Akkreditierungsagenturen

        {confidence_guidelines}

        Antworte nur mit einem JSON-Array von Subkategorien:
        ["Subkategorie 1: Erläuterung", "Subkategorie 2: Erläuterung", ...]"""
    

    def _get_category_extraction_prompt(self, segment: str) -> str:
        """
        Detaillierter Prompt für die Kategorienentwicklung mit Fokus auf 
        hierarchische Einordnung und Vermeidung zu spezifischer Hauptkategorien.
        """
        # FIX: Verwende spezialisierte Coding-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="coding", 
            context="kodierung"
        )

        return f"""Analysiere das folgende Textsegment für die Entwicklung induktiver Kategorien.
        Forschungsfrage: "{self.FORSCHUNGSFRAGE}"

        WICHTIG - HIERARCHISCHE EINORDNUNG:
        
        1. PRÜFE ZUERST EINORDNUNG IN BESTEHENDE HAUPTKATEGORIEN
        Aktuelle Hauptkategorien:
        {json.dumps(self.DEDUKTIVE_KATEGORIEN, indent=2, ensure_ascii=False)}

        {confidence_guidelines}

        2. ENTSCHEIDUNGSREGELN FÜR KATEGORIENEBENE:

        HAUPTKATEGORIEN (nur wenn wirklich nötig):
        - Beschreiben übergeordnete Themenkomplexe oder Handlungsfelder
        - Umfassen mehrere verwandte Aspekte unter einem konzeptionellen Dach
        - Sind abstrakt genug für verschiedene Unterthemen
        
        GUTE BEISPIELE für Hauptkategorien:
        - "Wissenschaftlicher Nachwuchs" (übergeordneter Themenkomplex)
        - "Wissenschaftliches Personal" (breites Handlungsfeld)
        - "Berufungsstrategien" (strategische Ebene)
        
        SCHLECHTE BEISPIELE für Hauptkategorien:
        - "Promotion in Unternehmen" (→ besser als Subkategorie unter "Wissenschaftlicher Nachwuchs")
        - "Rolle wissenschaftlicher Mitarbeiter" (→ besser als Subkategorie unter "Wissenschaftliches Personal")
        - "Forschungsinteresse bei Berufungen" (→ besser als Subkategorie unter "Berufungsstrategien")

        SUBKATEGORIEN (bevorzugt):
        - Beschreiben spezifische Aspekte einer Hauptkategorie
        - Konkretisieren einzelne Phänomene oder Handlungen
        - Sind empirisch direkt beobachtbar

        3. ENTWICKLUNGSSCHRITTE:
        a) Prüfe ZUERST, ob der Inhalt als Subkategorie in bestehende Hauptkategorien passt
        b) Entwickle neue Subkategorien für bestehende Hauptkategorien
        c) NUR WENN NÖTIG: Schlage neue Hauptkategorie vor, wenn wirklich neuer Themenkomplex

        TEXT:
        {segment}

        Antworte ausschließlich mit einem JSON-Objekt:
        {{
            "categorization_type": "subcategory_extension" | "new_main_category",
            "analysis": {{
                "existing_main_category": "Name der passenden Hauptkategorie oder null",
                "justification": "Begründung der hierarchischen Einordnung"
            }},
            "suggested_changes": {{
                "new_subcategories": [
                    {{
                        "name": "Name der Subkategorie",
                        "definition": "Definition",
                        "example": "Textstelle als Beleg"
                    }}
                ],
                "new_main_category": null | {{  // nur wenn wirklich nötig
                    "name": "Name der neuen Hauptkategorie",
                    "definition": "Definition",
                    "justification": "Ausführliche Begründung, warum neue Hauptkategorie nötig",
                    "initial_subcategories": [
                        {{
                            "name": "Name der Subkategorie",
                            "definition": "Definition"
                        }}
                    ]
                }}
            }},
            "confidence": {{
                "hierarchy_level": 0.0-1.0,
                "categorization": 0.0-1.0
            }}
        }}

        WICHTIG:
        - Bevorzuge IMMER die Entwicklung von Subkategorien
        - Neue Hauptkategorien nur bei wirklich übergeordneten Themenkomplexen
        - Prüfe genau die konzeptionelle Ebene der Kategorisierung
        - Achte auf angemessenes Abstraktionsniveau
        """

    def get_segment_relevance_assessment_prompt(self, chunk: str) -> str:
        """
        Erstellt einen Prompt zur Bewertung der Relevanz eines Segments für die Kategorienentwicklung.
        
        Args:
            segment: Zu bewertender Textabschnitt
            
        Returns:
            str: Formatierter Prompt für die Relevanzbeurteilung
        """
        # FIX: Verwende spezialisierte Relevanz-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="relevance", 
            context="relevanz"
        )
        
        return f"""
        Analysiere sorgfältig die Relevanz des folgenden Texts für die Forschungsfrage:
        "{self.FORSCHUNGSFRAGE}"
        
        TEXT:
        {chunk}
        
        Prüfe systematisch:
        1. Inhaltlicher Bezug: Behandelt der Text explizit Aspekte der Forschungsfrage?
        2. Aussagekraft: Enthält der Text konkrete, analysierbare Aussagen?
        3. Substanz: Geht der Text über oberflächliche/beiläufige Erwähnungen hinaus?
        4. Kontext: Ist der Bezug zur Forschungsfrage eindeutig und nicht nur implizit?
        
        {confidence_guidelines}

        Antworte NUR mit einem JSON-Objekt:
        {{
            "relevance_score": 0.0-1.0,
            "justification": "Kurze Begründung der Bewertung",
            "key_aspects": ["Liste", "relevanter", "Aspekte"]
        }}
        """

    def get_comprehensive_deductive_prompt(self, segment_text: str, categories_overview: List[Dict], 
                                         multiple_coding_threshold: float = 0.7,
                                         context_paraphrases: Optional[List[str]] = None) -> str:
        """
        Prompt für umfassende deduktive Analyse (für Optimierung).
        Erweitert den Standard-Deduktiv-Prompt um zusätzliche Analyseschritte.
        """
        
        # Basis-Prompt verwenden
        base_prompt = self.get_deductive_coding_prompt(
            chunk=segment_text,
            categories_overview=categories_overview,
            context_paraphrases=context_paraphrases
        )
        
        # Erweiterte Analyse-Anweisungen hinzufügen
        comprehensive_extension = f"""

ERWEITERTE ANALYSE FÜR OPTIMIERUNG:
Zusätzlich zur Standard-Kodierung, führe folgende Analysen durch:

1. RELEVANZSCORES: Bewerte die Relevanz für JEDE Kategorie (0.0-1.0)
   - Auch für Kategorien, die nicht als Hauptkategorie gewählt werden
   - Berücksichtige Forschungsfrage und Kodierregeln

2. MEHRFACHKODIERUNG: Prüfe ob Mehrfachkodierung nötig ist
   - Schwellenwert: {multiple_coding_threshold}
   - Wenn 2+ Kategorien diesen Schwellenwert erreichen → Mehrfachkodierung erforderlich

3. ALLE KATEGORIEN: Liste alle Kategorien mit Relevanzscores und Begründungen
   - Auch niedrig bewertete Kategorien mit Begründung warum nicht relevant

4. KEYWORDS: Extrahiere 3-5 zentrale Schlüsselwörter aus dem Segment

5. PARAPHRASE: Erstelle präzise Zusammenfassung des Segments

Erweitere das JSON-Format um folgende Felder:
{{
  "relevance_scores": {{
    "Kategorie1": 0.85,
    "Kategorie2": 0.42,
    "Kategorie3": 0.15
  }},
  "all_categories": [
    {{
      "category": "Kategorie1",
      "relevance_score": 0.85,
      "reasoning": "Begründung warum relevant/nicht relevant"
    }},
    {{
      "category": "Kategorie2", 
      "relevance_score": 0.42,
      "reasoning": "Begründung warum relevant/nicht relevant"
    }}
  ],
  "requires_multiple_coding": true/false,
  "keywords": "Schlüsselwort1, Schlüsselwort2, Schlüsselwort3",
  "paraphrase": "Zusammenfassung des Segments"
}}
"""
        
        return base_prompt + comprehensive_extension

    def get_batch_deductive_prompt(self, segments: List[Dict[str, str]], categories_overview: List[Dict],
                                 context_paraphrases: Optional[List[str]] = None) -> str:
        """
        Prompt für Batch-Verarbeitung deduktiver Kodierung.
        """
        
        # FIX: Verwende spezialisierte Coding-Konfidenz-Skala
        confidence_guidelines = ConfidenceScales.get_confidence_guidelines(
            scale_type="coding", 
            context="kodierung"
        )
        
        # Kategorien formatieren
        categories_text = self._format_categories_for_prompt(categories_overview)
        
        # Segmente formatieren
        segments_text = "\n\n".join([
            f"SEGMENT {i+1} (ID: {seg['segment_id']}):\n{seg['text']}"
            for i, seg in enumerate(segments)
        ])
        
        return f"""
        Führe eine deduktive Kodierung für die folgenden {len(segments)} Segmente durch.
        
        FORSCHUNGSFRAGE:
        {self.FORSCHUNGSFRAGE}
        
        KODIERREGELN:
        {chr(10).join(f"- {regel}" for regel in self.KODIERREGELN)}
        
        {confidence_guidelines}
        
        KATEGORIEN:
        {categories_text}
        
        {"KONTEXT-PARAPHRASEN:" if context_paraphrases else ""}
        {chr(10).join(f"- {para}" for para in (context_paraphrases or []))}
        
        ZU ANALYSIERENDE SEGMENTE:
        {segments_text}
        
        ANWEISUNGEN FÜR JEDES SEGMENT:
        
        1. KATEGORIEZUORDNUNG:
        - Wähle die am besten passende Hauptkategorie
        - Berücksichtige alle Kodierregeln und die Forschungsfrage
        - Bei Unsicherheit: Wähle die Kategorie mit dem stärksten Textbezug
        
        2. SUBKATEGORIEN:
        - Wähle passende Subkategorien der Hauptkategorie (falls verfügbar)
        - Maximal 3 Subkategorien pro Segment
        
        3. KONFIDENZ:
        - Bewerte die Sicherheit der Kodierung (0.0-1.0)
        - Berücksichtige Eindeutigkeit des Textbezugs
        
        4. BEGRÜNDUNG:
        - Erkläre die Kategoriewahl mit direktem Bezug zum Text
        - Nenne konkrete Textstellen als Belege
        
        Antworte NUR mit folgendem JSON-Format:
        {{
          "results": [
            {{
              "segment_id": "segment_id_from_text",
              "primary_category": "Kategoriename",
              "subcategories": ["Subkategorie1", "Subkategorie2"],
              "confidence": 0.92,
              "justification": "Begründung mit Textbezug",
              "paraphrase": "Zusammenfassung des Segments",
              "keywords": "keyword1, keyword2, keyword3"
            }}
          ]
        }}
        """

    def get_inductive_mode_instructions(self) -> str:
        """Mode-spezifische Anweisungen für induktive Kategorienentwicklung."""
        return """
        INDUKTIVE KATEGORIENENTWICKLUNG:
        - Entwickle Kategorien direkt aus dem Material
        - Keine vorgegebenen Kategorien verwenden
        - Kategorien sollen die Inhalte der Segmente widerspiegeln
        - Jede Kategorie braucht eine klare Definition
        - Kategorien müssen zur Forschungsfrage beitragen
        """

    def get_inductive_json_schema(self) -> str:
        """JSON-Schema für induktive Kategorienentwicklung."""
        return '''
        {
          "results": [
            {
              "segment_id": "segment_id_here",
              "categories": ["Kategorie1", "Kategorie2"],
              "category_definitions": {
                "Kategorie1": "Definition der Kategorie",
                "Kategorie2": "Definition der Kategorie"
              },
              "confidence": 0.8,
              "reasoning": "Begründung der Kategorienentwicklung"
            }
          ]
        }
        '''

    def get_abductive_mode_instructions(self, categories_text: str) -> str:
        """Mode-spezifische Anweisungen für abduktive Analyse."""
        return f"""
        ABDUKTIVER MODUS: Entwickle NUR neue Subkategorien für bestehende Hauptkategorien.

        BESTEHENDE HAUPTKATEGORIEN:
        {categories_text}

        STRIKTE REGELN FÜR ABDUKTIVEN MODUS:
        - KEINE neuen Hauptkategorien entwickeln
        - NUR neue Subkategorien für bestehende Hauptkategorien
        - Subkategorien müssen neue, relevante Themenaspekte der Hauptkategorie abbilden
        - Mindestens 2 Textbelege pro neuer Subkategorie
        - Konfidenz mindestens 0.7
        - Nur Subkategorien vorschlagen, die das bestehende System sinnvoll verfeinern
        """

    def get_abductive_json_schema(self) -> str:
        """JSON-Schema für abduktive Analyse."""
        return '''
        {
          "extended_categories": {
            "hauptkategorie_name": {
              "new_subcategories": [
                {
                  "name": "Subkategorie Name",
                  "definition": "Definition der Subkategorie",
                  "evidence": ["Textbelege aus den Segmenten"],
                  "confidence": 0.0,
                  "thematic_novelty": "Warum diese Subkategorie einen neuen Aspekt der Hauptkategorie abbildet"
                }
              ]
            }
          },
          "development_assessment": {
            "subcategories_developed": 0,
            "saturation_indicators": ["Liste von Hinweisen auf Sättigung"],
            "recommendation": "continue/pause/stop"
          },
          "segment_assignments": [
            {
              "segment_id": "segment_id_here",
              "main_category": "Hauptkategorie",
              "subcategories": ["Bestehende_Subkategorie1", "Neue_Subkategorie2"],
              "confidence": 0.9,
              "reasoning": "Begründung der Kategoriezuordnung"
            }
          ]
        }
        '''

    def get_grounded_json_schema(self) -> str:
        """JSON-Schema für Grounded Theory Analyse."""
        return '''
        {
          "results": [
            {
              "segment_id": "segment_id_here",
              "codes": ["Code1", "Code2", "Code3"],
              "keywords": ["keyword1", "keyword2"],
              "memo": "Analytisches Memo zu diesem Segment",
              "concepts": ["Konzept1", "Konzept2"],
              "relationships": ["Beziehung zwischen Konzepten"],
              "confidence": 0.8
            }
          ]
        }
        '''