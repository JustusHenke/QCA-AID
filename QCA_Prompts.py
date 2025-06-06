"""
QCA-AID Prompt Templates
=======================

Zentrale Sammlung aller Prompts für die qualitative Inhaltsanalyse.
Getrennt vom Hauptskript für bessere Wartbarkeit und Übersichtlichkeit.

Author: Justus Henke
"""

from typing import Dict, List
import json

class QCAPrompts:
    """Zentrale Klasse für alle QCA-AID Prompts"""
    
    def __init__(self, forschungsfrage: str, kodierregeln: Dict, deduktive_kategorien: Dict):
        self.FORSCHUNGSFRAGE = forschungsfrage
        self.KODIERREGELN = kodierregeln
        self.DEDUKTIVE_KATEGORIEN = deduktive_kategorien

        # print(f"QCAPrompts initialisiert:")
        # print(f"- Forschungsfrage: {len(self.FORSCHUNGSFRAGE)} Zeichen")
        # print(f"- Kodierregeln: {len(self.KODIERREGELN)} Kategorien")
        # print(f"- Deduktive Kategorien: {len(self.DEDUKTIVE_KATEGORIEN)} Kategorien")
    
    def get_deductive_coding_prompt(self, chunk: str, categories_overview: List[Dict]) -> str:
        """
        Prompt für deduktive Kodierung von Text-Chunks mit Subkategorie-Validierung.
        """
        return f"""
            Analysiere folgenden Text im Kontext der Forschungsfrage:
            "{self.FORSCHUNGSFRAGE}"
            
            ## TEXT:
            {chunk}

            ## KATEGORIENSYSTEM:
            {json.dumps(categories_overview, indent=2, ensure_ascii=False)}

            ## KODIERREGELN:
            {json.dumps(self.KODIERREGELN, indent=2, ensure_ascii=False)}

            ## WICHTIG - KATEGORIEN- UND SUBKATEGORIEN-ZUORDNUNG:
            
            1. HAUPTKATEGORIEN-WAHL:
            - Vergleiche den Text systematisch mit JEDER Kategoriendefinition
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
        return f"""
            Analysiere die Relevanz der folgenden Textsegmente für die Forschungsfrage:
            "{self.FORSCHUNGSFRAGE}"
            
            PRÜFUNGSREIHENFOLGE - Analysiere jedes Segment in dieser Reihenfolge:

            1. THEMATISCHE VORPRÜFUNG:
            
            Führe ZUERST eine grundlegende thematische Analyse durch:
            - Identifiziere den Gegenstand, die Kernthemen und zentralen Konzepte der Forschungsfrage
            - Prüfe, ob der Text den Gegenstand und diese Kernthemen überhaupt behandelt
            - Stelle fest, ob ein hinreichender inhaltlicher Zusammenhang zur Forschungsfrage besteht
            - Falls NEIN: Sofort als nicht relevant markieren
            - Falls JA: Weiter mit detaillierter Prüfung

            2. AUSSCHLUSSKRITERIEN:
            {exclusion_rules}

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
                        "justification": "Begründung der Relevanz unter Berücksichtigung der Textsorte"
                    }},
                    ...
                ]
            }}

            WICHTIGE HINWEISE:
            - Führe IMMER ZUERST die thematische Vorprüfung durch
            - Identifiziere die Kernthemen der Forschungsfrage und prüfe deren Präsenz im Text
            - Markiere Segmente als nicht relevant, wenn sie die Kernthemen nicht behandeln
            - Bei Unsicherheit (confidence < 0.75) als nicht relevant markieren
            - Gib bei thematischer Nicht-Passung eine klare Begründung
            - Sei streng bei der thematischen Vorprüfung
            """

    def get_multiple_category_relevance_prompt(self, segments_text: str, category_descriptions: List[Dict], multiple_threshold: float) -> str:
        """Prompt für Mehrfachkodierungs-Relevanzprüfung"""
        return f"""
            Analysiere die folgenden Textsegmente auf Mehrfachkodierung für verschiedene Hauptkategorien.
            
            FORSCHUNGSFRAGE: "{self.FORSCHUNGSFRAGE}"
            
            HAUPTKATEGORIEN:
            {json.dumps(category_descriptions, indent=2, ensure_ascii=False)}
            
            AUFGABE - MEHRFACHKODIERUNGS-ANALYSE:
            Prüfe für jedes Segment, ob es für MEHRERE Hauptkategorien gleichzeitig relevant ist.
            Ein Segment kann mehrfach kodiert werden, wenn es mindestens {multiple_threshold} ({int(multiple_threshold*100)}%) Relevanz 
            für verschiedene Hauptkategorien hat.
            
            BEISPIEL für Mehrfachkodierung:
            "Wir haben seit Jahren ein Referat für Forschungsförderung, aber am Ende sind das immer 
            Aushandlungsprozesse zwischen den Beteiligten"
            → Könnte sowohl "Strukturen" (Referat) als auch "Prozesse" (Aushandlung) zugeordnet werden
            
            KRITERIEN FÜR MEHRFACHKODIERUNG:
            - Mindestens {int(multiple_threshold*100)}% Relevanz für jede zugeordnete Kategorie
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

    def get_focus_coding_prompt(self, chunk: str, categories_overview: List[Dict], focus_category: str, focus_context: Dict) -> str:
        """Prompt für fokussierte Kodierung mit strenger Subkategorie-Validierung"""
        return f"""
            MEHRFACHKODIERUNGS-FOKUS auf: "{focus_category}"
            
            Forschungsfrage: "{self.FORSCHUNGSFRAGE}"
            
            ## TEXT:
            {chunk}

            ## FOCUS-KONTEXT:
            - Relevanz-Score für {focus_category}: {focus_context.get('relevance_score', 0):.2f}
            - Begründung: {focus_context.get('justification', '')}
            
            ## KATEGORIENSYSTEM:
            {json.dumps(categories_overview, indent=2, ensure_ascii=False)}

            ## KRITISCH WICHTIG - SUBKATEGORIE-ZUORDNUNG:
            
            1. HAUPTKATEGORIEN-ENTSCHEIDUNG:
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
    
    def get_progressive_context_prompt(self, chunk: str, categories_overview: List[Dict], current_summary: str, position_info: Dict, summary_update_prompt: str) -> str:
        """Prompt für progressive Kontext-Kodierung"""
        return f"""
            ## AUFGABE 1: KODIERUNG
            Analysiere folgenden Text im Kontext der Forschungsfrage:
            "{self.FORSCHUNGSFRAGE}"
            
            ### PROGRESSIVER DOKUMENTKONTEXT (bisherige relevante Inhalte):
            {current_summary if current_summary else "Noch keine relevanten Inhalte für dieses Dokument erfasst."}
            
            ### TEXTSEGMENT ZU KODIEREN:
            {chunk}
            
            {position_info}

            ### KATEGORIENSYSTEM:
            {json.dumps(categories_overview, indent=2, ensure_ascii=False)}

            ### KODIERREGELN:
            {json.dumps(self.KODIERREGELN, indent=2, ensure_ascii=False)}

            ### WICHTIG - PROGRESSIVE KONTEXTANALYSE UND GENAUE KATEGORIENZUORDNUNG: 
            
            1. KATEGORIENVERGLEICH:
            - Vergleiche das TEXTSEGMENT systematisch mit JEDER Kategoriendefinition
            - Prüfe explizit die Übereinstimmung mit den Beispielen jeder Kategorie
            - Identifiziere wörtliche und sinngemäße Übereinstimmungen
            - Dokumentiere auch teilweise Übereinstimmungen für die Nachvollziehbarkeit
            - Berücksichtige den DOKUMENTKONTEXT für tieferes Verständnis des TEXTSEGMENTS
            
            2. SUBKATEGORIENVERGLEICH:
            - Bei passender Hauptkategorie: Prüfe ALLE zugehörigen Subkategorien
            - WICHTIG: Wähle PRIMÄR NUR DIE EINE Subkategorie mit der höchsten Passung zum Text
            - Vergebe weitere Subkategorien NUR, wenn der Text EINDEUTIG mehrere Subkategorien gleichgewichtig adressiert
            - Bei mehreren passenden Subkategorien mit ähnlicher Relevanz: Begründe die Wahl
            
            3. KONTEXT-INTEGRATION:
            - Prüfe, ob das aktuelle TEXTSEGMENT bisherige Einschätzungen bestätigt, ergänzt oder widerspricht
            - Bei Kategorien wie "dominante Akteure": Besonders wichtig, den bisherigen Kontext zu berücksichtigen
            - Formuliere eine klare Begründung, die den Kontext explizit einbezieht
            
            4. ENTSCHEIDUNGSREGELN:
            - Kodiere nur bei eindeutiger Übereinstimmung mit Definition UND Beispielen
            - Bei konkurrierenden Kategorien: Dokumentiere die Abwägung
            - Bei niedriger Konfidenz (<0.80): Wähle "Keine passende Kategorie"
            - Die Interpretation muss sich auf konkrete Textstellen stützen
            - Keine Annahmen oder Spekulationen über den Text hinaus
            - Prüfe die Relevanz für die Forschungsfrage explizit

            5. QUALITÄTSSICHERUNG:
            - Stelle intersubjektive Nachvollziehbarkeit sicher
            - Dokumentiere Grenzfälle und Abwägungen transparent
            - Prüfe die Konsistenz mit bisherigen Kodierungen
            - Bei Unsicherheiten: Lieber konservativ kodieren
            
            ### LIEFERE IN DER KODIERUNG:

            1. PARAPHRASE:
            - Erfasse den zentralen Inhalt des TEXTSEGMENTS in max. 40 Wörtern
            - IGNORIERE dafür den Dokumenttext
            - Verwende sachliche, deskriptive Sprache
            - Bleibe nah am Originaltext ohne Interpretation

            2. SCHLÜSSELWÖRTER:
            - 2-3 zentrale Begriffe aus dem Text
            - Wähle bedeutungstragende Terme
            - Vermeide zu allgemeine Begriffe

            3. KATEGORIENZUORDNUNG:
            - Entweder präzise Kategorie oder "Keine passende Kategorie"
            - Ausführliche Begründung mit Textbelegen
            - Transparente Konfidenzeinschätzung
            - Erläutere explizit den Einfluss des Dokumentkontexts

            {summary_update_prompt}

            Antworte mit EINEM JSON-Objekt, das BEIDE Aufgaben umfasst:
            {{
                "coding_result": {{
                    "paraphrase": "Deine prägnante Paraphrase des TEXTSEGMENTS hier",
                    "keywords": "Deine Schlüsselwörter hier",
                    "category": "Name der Hauptkategorie oder 'Keine passende Kategorie'",
                    "subcategories": ["Subkategorie", "Subkategorie"],
                    "justification": "Begründung muss enthalten: 1. Konkrete Textstellen, 2. Bezug zur Kategoriendefinition, 3. Verbindung zur Forschungsfrage",
                    "confidence": {{
                        "total": 0.00-1.00,
                        "category": 0.00-1.00,
                        "subcategories": 0.00-1.00
                    }},
                    "text_references": ["Relevante Textstellen"],
                    "definition_matches": ["Welche Aspekte der Definition passen"],
                    "context_influence": "Wie der Dokumentkontext die Kodierung beeinflusst hat"
                }},
                "updated_summary": "Das aktualisierte Document-Summary mit max. 70 Wörtern"
            }}
            """

    def get_focus_context_coding_prompt(self, chunk: str, categories_overview: List[Dict], focus_category: str, focus_context: Dict, current_summary: str, position_info: Dict, summary_update_prompt: str, update_summary: bool = False) -> str:
        """Prompt für fokussierte Kodierung mit Kontext"""
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
        """Prompt für induktive Kategorienentwicklung"""
        # Platzhalter - Prompt für Batch-Analyse und Kategorienentwicklung
        pass

    def get_grounded_analysis_prompt(self, segments: List[str], existing_subcodes: List[str], json_schema: str) -> str:
        """Prompt für Grounded Theory Analyse"""
        return f"""
            Analysiere folgende Textsegmente im Sinne der Grounded Theory.
            Identifiziere Subcodes und Keywords, ohne sie bereits Hauptkategorien zuzuordnen.
            
            FORSCHUNGSFRAGE:
            {self.FORSCHUNGSFRAGE}
            
            {"BEREITS IDENTIFIZIERTE SUBCODES:" if existing_subcodes else ""}
            {json.dumps(existing_subcodes, indent=2, ensure_ascii=False) if existing_subcodes else ""}
            
            TEXTSEGMENTE:
            {json.dumps(segments, indent=2, ensure_ascii=False)}
            
            ANWEISUNGEN FÜR DEN GROUNDED THEORY MODUS:
            
            1. OFFENES KODIEREN MIT KLARER ABSTRAKTIONSHIERARCHIE:
            - KEYWORDS: Extrahiere TEXTNAHE, KONKRETE Begriffe und Phrasen direkt aus dem Text
            - SUBCODES: Entwickle eine ERSTE ABSTRAKTIONSEBENE basierend auf den Keywords
            * Fasse ähnliche Keywords zu einem gemeinsamen, leicht abstrahierten Subcode zusammen
            * Verwende eine präzisere Sprache als für spätere Hauptkategorien
            * Formuliere Subcodes als analytische Konzepte, NICHT als bloße Wiederholung der Keywords
            
            2. WICHTIG - PROGRESSIVE ANREICHERUNG DES SUBCODE-SETS:
            - Berücksichtige die bereits identifizierten Subcodes
            - Fokussiere auf NEUE KOMPLEMENTÄRE Subcodes, die noch nicht erfasst wurden
            - Bei thematischer Überschneidung mit bestehenden Subcodes: verfeinere oder differenziere
            - Keine Duplikate zu bestehenden Subcodes erstellen
            
            3. QUALITÄTSKRITERIEN FÜR SUBCODES:
            - Empirisch gut belegt durch Textstellen
            - Präzise und eindeutig benannt
            - Analytisch wertvoll für die Forschungsfrage
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
        GROUNDED THEORY: Generiere Hauptkategorien aus gesammelten Subcodes
        
        Du erhältst {len(subcodes_data)} Subcodes mit ihren Keywords, die während einer Grounded Theory Analyse gesammelt wurden. 
        Deine Aufgabe ist es, diese zu thematisch kohärenten Hauptkategorien zu gruppieren.
        
        FORSCHUNGSFRAGE: {self.FORSCHUNGSFRAGE}
        
        GESAMMELTE SUBCODES UND KEYWORDS:
        {subcodes_text}
        
        GROUNDED THEORY ANALYSE-ANWEISUNGEN:
        1. Analysiere die thematischen Verbindungen zwischen den Subcodes
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
        return f"""
        ABDUKTIVER MODUS: Entwickle NUR neue Subkategorien für bestehende Hauptkategorien.

        BESTEHENDE HAUPTKATEGORIEN:
        {json.dumps(categories_context, indent=2, ensure_ascii=False)}

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
            
            Antworte nur mit der verbesserten Definition:
            """
    def _get_subcategory_generation_prompt(self, category: Dict) -> str:
        """
        Prompt zur Generierung fehlender Subkategorien.
        """
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

        Antworte nur mit einem JSON-Array von Subkategorien:
        ["Subkategorie 1: Erläuterung", "Subkategorie 2: Erläuterung", ...]"""
    

    def _get_category_extraction_prompt(self, segment: str) -> str:
        """
        Detaillierter Prompt für die Kategorienentwicklung mit Fokus auf 
        hierarchische Einordnung und Vermeidung zu spezifischer Hauptkategorien.
        """
        return f"""Analysiere das folgende Textsegment für die Entwicklung induktiver Kategorien.
        Forschungsfrage: "{self.FORSCHUNGSFRAGE}"

        WICHTIG - HIERARCHISCHE EINORDNUNG:
        
        1. PRÜFE ZUERST EINORDNUNG IN BESTEHENDE HAUPTKATEGORIEN
        Aktuelle Hauptkategorien:
        {json.dumps(self.DEDUKTIVE_KATEGORIEN, indent=2, ensure_ascii=False)}

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
        
        Antworte NUR mit einem JSON-Objekt:
        {{
            "relevance_score": 0.0-1.0,
            "justification": "Kurze Begründung der Bewertung",
            "key_aspects": ["Liste", "relevanter", "Aspekte"]
        }}
        """