"""
Deduktive Kodierung für QCA-AID
================================
Automatisches deduktives Codieren von Text-Chunks anhand des Kodierleitfadens.
"""

import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any

from ..core.config import CONFIG, FORSCHUNGSFRAGE, KODIERREGELN, DEDUKTIVE_KATEGORIEN
from ..core.data_models import CategoryDefinition, CodingResult
from ..core.validators import CategoryValidator
from ..QCA_Utils import LLMProviderFactory, LLMResponse, TokenTracker
from ..QCA_Prompts import QCAPrompts

# Globaler Token-Counter
token_counter = TokenTracker()


class DeductiveCategoryBuilder:
    """
    Baut ein initiales, theoriebasiertes Kategoriensystem auf.
    """
    def load_theoretical_categories(self) -> Dict[str, CategoryDefinition]:
        """
        LÄdt die vordefinierten deduktiven Kategorien.
        
        Returns:
            Dict[str, CategoryDefinition]: Dictionary mit Kategorienamen und deren Definitionen
        """
        categories = {}
        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Konvertiere DEDUKTIVE_KATEGORIEN in CategoryDefinition-Objekte
            for name, data in DEDUKTIVE_KATEGORIEN.items():
                # Stelle sicher, dass rules als Liste vorhanden ist
                rules = data.get("rules", [])
                if not isinstance(rules, list):
                    rules = [str(rules)] if rules else []
                
                categories[name] = CategoryDefinition(
                    name=name,
                    definition=data.get("definition", ""),
                    examples=data.get("examples", []),
                    rules=rules,  # Übergebe die validierte rules Liste
                    subcategories=data.get("subcategories", {}),
                    added_date=today,
                    modified_date=today
                )
                
            return categories
            
        except Exception as e:
            print(f"Fehler beim Laden der Kategorien: {str(e)}")
            print("Aktuelle Kategorie:", name)
            print("Kategorie-Daten:", json.dumps(data, indent=2, ensure_ascii=False))
            raise



# --- Klasse: DeductiveCoder ---
# Aufgabe: Automatisches deduktives Codieren von Text-Chunks anhand des Leitfadens
class DeductiveCoder:
    """
    Ordnet Text-Chunks automatisch deduktive Kategorien zu basierend auf dem Kodierleitfaden.
    Nutzt GPT-4-Mini fuer die qualitative Inhaltsanalyse nach Mayring.
    """
    
    def __init__(self, model_name: str, temperature: str, coder_id: str, skip_inductive: bool = False):
        """
        Initializes the DeductiveCoder with configuration for the GPT model.
        
        Args:
            model_name (str): Name of the GPT model to use
            temperature (float): Controls randomness in the model's output 
                - Lower values (e.g., 0.3) make output more focused and deterministic
                - Higher values (e.g., 0.7) make output more creative and diverse
            coder_id (str): Unique identifier for this coder instance
        """
        self.model_name = model_name
        self.temperature = float(temperature)
        self.coder_id = coder_id
        self.skip_inductive = skip_inductive

        # self.load_theoretical_categories = DeductiveCategoryBuilder.load_theoretical_categories()
        category_builder = DeductiveCategoryBuilder()
        initial_categories = category_builder.load_theoretical_categories()

        # FIX: Initialisiere current_categories modus-abhÄngig
        analysis_mode = CONFIG.get('ANALYSIS_MODE', 'deductive')
        
        if analysis_mode == 'grounded':
            # Im grounded mode starten wir mit leeren Kategorien
            self.current_categories = {}
            print(f"🧾 Kodierer {coder_id}: Grounded Mode - startet ohne deduktive Kategorien")
        else:
            # Alle anderen Modi laden deduktive Kategorien
            try:
                self.current_categories = initial_categories
                print(f"🧾 Kodierer {coder_id}: {len(self.current_categories)} deduktive Kategorien geladen ({analysis_mode} mode)")
            except Exception as e:
                print(f"❌ Fehler beim Laden der Kategorien fuer Kodierer {coder_id}: {str(e)}")
                self.current_categories = {}
           
        # Hole Provider aus CONFIG
        provider_name = CONFIG.get('MODEL_PROVIDER', 'openai').lower()  # Fallback zu OpenAI
        try:
            # Wir lassen den LLMProvider sich selbst um die Client-Initialisierung kÜmmern
            # Übergebe model_name für Capability-Testing
            self.llm_provider = LLMProviderFactory.create_provider(provider_name, model_name=model_name)
            print(f"🤖 LLM Provider '{provider_name}' fuer Kodierer {coder_id} initialisiert")
        except Exception as e:
            print(f"Fehler bei Provider-Initialisierung fuer {coder_id}: {str(e)}")
            raise
        
        # FIX: Prompt-Handler nur im grounded mode ohne deduktive Kategorien initialisieren
        analysis_mode = CONFIG.get('ANALYSIS_MODE', 'deductive')
        if analysis_mode == 'grounded':
            # FIX: Im grounded mode KEINE deduktiven Kategorien verwenden
            self.prompt_handler = QCAPrompts(
                forschungsfrage=FORSCHUNGSFRAGE,
                kodierregeln=KODIERREGELN,
                deduktive_kategorien={}  # FIX: Leeres Dict statt DEDUKTIVE_KATEGORIEN
            )
            print(f"   ℹ️ Grounded Mode: Kodierer {coder_id} ohne vordefinierte Kategorien initialisiert")
        else:
            # Normale Modi verwenden deduktive Kategorien
            self.prompt_handler = QCAPrompts(
                forschungsfrage=FORSCHUNGSFRAGE,
                kodierregeln=KODIERREGELN,
                deduktive_kategorien=DEDUKTIVE_KATEGORIEN
            )

    async def update_category_system(self, categories: Dict[str, CategoryDefinition]) -> bool:
        """
        Aktualisiert das Kategoriensystem des Kodierers.
        
        Args:
            categories: Neues/aktualisiertes Kategoriensystem
            
        Returns:
            bool: True wenn Update erfolgreich
        """
        try:
            # FIX: PrÜfe Analysemodus vor Update
            analysis_mode = CONFIG.get('ANALYSIS_MODE', 'deductive')
            
            if analysis_mode == 'grounded':
                # FIX: Im grounded mode nur mit rein induktiven Kategorien arbeiten
                # Filtere alle deduktiven Kategorien heraus
                grounded_categories = {}
                for name, cat in categories.items():
                    if name not in DEDUKTIVE_KATEGORIEN:
                        grounded_categories[name] = cat
                
                print(f"   ℹ️ Grounded Mode: Kodierer {self.coder_id} aktualisiert mit {len(grounded_categories)} rein induktiven Kategorien")
                print(f"   🔀 Ausgeschlossen: {len(categories) - len(grounded_categories)} deduktive Kategorien")
                
                # Verwende nur die gefilterten Kategorien
                categories_to_use = grounded_categories
            else:
                # Andere Modi verwenden alle Kategorien
                categories_to_use = categories
                print(f"   ℹ️ {analysis_mode.upper()} Mode: Kodierer {self.coder_id} aktualisiert mit {len(categories_to_use)} Kategorien")

            # Konvertiere CategoryDefinition in serialisierbares Dict
            categories_dict = {
                name: {
                    'definition': cat.definition,
                    'examples': list(cat.examples) if isinstance(cat.examples, set) else cat.examples,
                    'rules': list(cat.rules) if isinstance(cat.rules, set) else cat.rules,
                    'subcategories': dict(cat.subcategories) if isinstance(cat.subcategories, set) else cat.subcategories
                } for name, cat in categories_to_use.items()
            }

            # FIX: Aktualisiere das Kontextwissen des Kodierers modusabhÄngig
            if analysis_mode == 'grounded':
                prompt = f"""
                GROUNDED THEORY MODUS: Das Kategoriensystem wurde mit rein induktiven Kategorien aktualisiert.
                
                Aktuelle induktive Kategorien:
                {json.dumps(categories_dict, indent=2, ensure_ascii=False)}
                
                WICHTIGE REGELN FÜR GROUNDED KODIERUNG:
                1. Verwende NUR die oben aufgefÜhrten induktiven Kategorien
                2. KEINE deduktiven Kategorien aus dem ursprÜnglichen Codebook verwenden
                3. Diese Kategorien wurden bottom-up aus den Texten entwickelt
                4. Kodiere nur dann, wenn das Segment eindeutig zu einer dieser Kategorien passt
                5. Bei Unsicherheit: "Nicht kodiert" verwenden
                
                Antworte einfach mit "Verstanden" wenn du die Anweisung erhalten hast.
                """
            else:
                prompt = f"""
                Das Kategoriensystem wurde aktualisiert. Neue Zusammensetzung:
                {json.dumps(categories_dict, indent=2, ensure_ascii=False)}
                
                BerÜcksichtige bei der Kodierung:
                1. Verwende alle verfÜgbaren Kategorien entsprechend ihrer Definitionen
                2. PrÜfe auch Subkategorien bei der Zuordnung
                3. Kodiere nur bei eindeutiger Zuordnung
                
                Antworte einfach mit "Verstanden" wenn du die Anweisung erhalten hast.
                """

            # Aktualisiere internes Kategoriensystem
            # FIX: Verwende gefilterte Kategorien statt alle
            self.current_categories = categories_to_use
            
            # FIX: Aktualisiere den Prompt-Handler korrekt
            if analysis_mode == 'grounded':
                # FIX: Im grounded mode neue QCAPrompts ohne deduktive Kategorien erstellen
                self.prompt_handler = QCAPrompts(
                    forschungsfrage=self.prompt_handler.FORSCHUNGSFRAGE,
                    kodierregeln=self.prompt_handler.KODIERREGELN,
                    deduktive_kategorien={}  # FIX: Leeres Dict fuer grounded mode
                )
            else:
                # FIX: FÜr andere Modi QCAPrompts mit aktualisierten Kategorien erstellen
                self.prompt_handler = QCAPrompts(
                    forschungsfrage=self.prompt_handler.FORSCHUNGSFRAGE,
                    kodierregeln=self.prompt_handler.KODIERREGELN,
                    deduktive_kategorien=categories_dict  # FIX: Verwende aktualisierte Kategorien
                )

            # Test-Anfrage um sicherzustellen, dass das System bereit ist
            response = await self.llm_provider.create_completion(
                model=self.model_name,  # FIX: Verwende self.model_name
                messages=[
                    {"role": "system", "content": "Du bist ein QCA-Kodierer. Antworte auf deutsch."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            # FIX: Korrekte Antwort-Verarbeitung
            llm_response = LLMResponse(response)
            if "verstanden" in llm_response.content.lower():
                return True
            else:
                print(f"❌ Unerwartete Antwort von Kodierer {self.coder_id}: {llm_response.content}")
                return True  # Fahre trotzdem fort
                
        except Exception as e:
            print(f"⚠️ Fehler beim Update von Kodierer {self.coder_id}: {str(e)}")
            return False
        
    async def code_chunk_with_context_switch(self, 
                                         chunk: str, 
                                         categories: Dict[str, CategoryDefinition],
                                         use_context: bool = True,
                                         context_data: Dict = None) -> Dict:
        """
        Wrapper-Methode, die basierend auf dem use_context Parameter
        zwischen code_chunk und code_chunk_with_progressive_context wechselt.
        
        Args:
            chunk: Zu kodierender Text
            categories: Kategoriensystem
            use_context: Ob Kontext verwendet werden soll
            context_data: Kontextdaten (current_summary und segment_info)
            
        Returns:
            Dict: Kodierungsergebnis (und ggf. aktualisiertes Summary)
        """
        if use_context and context_data:
            # Mit progressivem Kontext kodieren
            return await self.code_chunk_with_progressive_context(
                chunk, 
                categories, 
                context_data.get('current_summary', ''),
                context_data.get('segment_info', {})
            )
        else:
            # Klassische Kodierung ohne Kontext
            result = await self.code_chunk(chunk, categories)
            
            # Wandle CodingResult in Dictionary um, wenn nÖtig
            if result and isinstance(result, CodingResult):
                return {
                    'coding_result': result.to_dict(),
                    'updated_summary': context_data.get('current_summary', '') if context_data else ''
                }
            elif result and isinstance(result, dict):
                return {
                    'coding_result': result,
                    'updated_summary': context_data.get('current_summary', '') if context_data else ''
                }
            else:
                return None
            
    async def code_chunk(self, chunk: str, categories: Optional[Dict[str, CategoryDefinition]] = None, 
                        is_last_segment: bool = False, preferred_cats: Optional[List[str]] = None) -> Optional[CodingResult]:  # FIX: Neuer Parameter hinzugefÜgt
        """
        Kodiert einen Text-Chunk basierend auf dem aktuellen Kategoriensystem.
        
        Args:
            chunk: Zu kodierender Text
            categories: Kategoriensystem (optional, verwendet current_categories als Fallback)
            is_last_segment: Ob dies das letzte Segment ist
            preferred_cats: Liste bevorzugter Kategorien fuer gefilterte Kodierung  # FIX: Neue FunktionalitÄt
            
        Returns:
            Optional[CodingResult]: Kodierungsergebnis oder None bei Fehlern
        """
        try:
            # Verwende das interne Kategoriensystem wenn vorhanden, sonst das Übergebene
            current_categories = categories
            
            if not current_categories:
                print(f"Fehler: Kein Kategoriensystem fuer Kodierer {self.coder_id} verfÜgbar")
                return None

            # FIX: Kategorien-Filterung basierend auf preferred_cats
            if preferred_cats:
                # Filtere Kategorien basierend auf Vorauswahl
                filtered_categories = {
                    name: cat for name, cat in current_categories.items() 
                    if name in preferred_cats
                }
                
                if filtered_categories:
                    print(f"    [TARGET] Gefilterte Kodierung fuer {self.coder_id}: {len(filtered_categories)}/{len(current_categories)} Kategorien")
                    print(f"    🔀‹ Fokus auf: {', '.join(preferred_cats)}")
                    effective_categories = filtered_categories
                else:
                    print(f"    ❌ Keine der bevorzugten Kategorien {preferred_cats} gefunden - nutze alle Kategorien")
                    effective_categories = current_categories
            else:
                # Standard-Verhalten: alle Kategorien verwenden
                effective_categories = current_categories
            # FIX: Ende der neuen Filterlogik

            # Erstelle formatierte KategorienÜbersicht
            categories_overview = []
            for name, cat in effective_categories.items():  # FIX: Nutze gefilterte Kategorien
                category_info = {
                    'name': name,
                    'definition': cat.definition,
                    'examples': list(cat.examples) if isinstance(cat.examples, set) else cat.examples,
                    'rules': list(cat.rules) if isinstance(cat.rules, set) else cat.rules,
                    'subcategories': {}
                }
                
                # FÜge Subkategorien hinzu
                for sub_name, sub_def in cat.subcategories.items():
                    category_info['subcategories'][sub_name] = sub_def
                    
                categories_overview.append(category_info)

            # Erstelle Prompt
            prompt = self.prompt_handler.get_deductive_coding_prompt(
                chunk=chunk,
                categories_overview=categories_overview
            )
            
            try:
                token_counter.start_request()

                response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte fuer qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                # Verarbeite Response
                llm_response = LLMResponse(response)
                try:
                    result = json.loads(llm_response.content)
                except json.JSONDecodeError as e:
                    print(f"[ERROR][{self.coder_id}] JSONDecodeError in code_chunk: {e}")
                    print(f"[ERROR][{self.coder_id}] Raw LLM response: {llm_response.content}")
                    token_counter.track_error(self.model_name)
                    return None
                
                token_counter.track_response(response, self.model_name)
                
                if result and isinstance(result, dict):
                    # FIX: Strikte Subkategorie-Validierung anwenden
                    main_category = result.get('category', '')
                    original_subcats = result.get('subcategories', [])
                    
                    validated_subcats = CategoryValidator.validate_subcategories_for_category(
                        original_subcats, main_category, self.current_categories, warn_only=False
                    )
                    result['subcategories'] = validated_subcats
                    # FIX: Ende der Subkategorie-Validierung
                    
                    coding_result = CodingResult(
                        category=result.get('category', 'Nicht kodiert'),
                        subcategories=tuple(validated_subcats),  # [OK] Tuple statt Set
                        confidence=result.get('confidence', {}),
                        justification=result.get('justification', ''),
                        paraphrase=result.get('paraphrase', ''),
                        keywords=result.get('keywords', ''),
                        text_references=tuple(result.get('text_references', [])),  # [OK] Tuple
                        uncertainties=tuple(result.get('uncertainties', [])) if result.get('uncertainties') else None
                    )
                                        
                    return coding_result
                else:
                    print(f"UngÜltige API-Antwort von {self.coder_id}")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"JSON-Fehler bei {self.coder_id}: {str(e)}")
                print(f"Rohe Antwort: {llm_response.content[:200]}...")
                return None
            except Exception as e:
                print(f"API-Fehler bei {self.coder_id}: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Allgemeiner Fehler bei der Kodierung durch {self.coder_id}: {str(e)}")
            return None
    
    async def code_chunk_with_progressive_context(self, 
                                          chunk: str, 
                                          categories: Dict[str, CategoryDefinition],
                                          current_summary: str,
                                          segment_info: Dict) -> Dict:
        """
        Kodiert einen Text-Chunk und aktualisiert gleichzeitig das Dokument-Summary
        mit einem dreistufigen Reifungsmodell.
        
        Args:
            chunk: Zu kodierender Text
            categories: Kategoriensystem
            current_summary: Aktuelles Dokument-Summary
            segment_info: ZusÄtzliche Informationen Über das Segment
            
        Returns:
            Dict: EnthÄlt sowohl Kodierungsergebnis als auch aktualisiertes Summary
        """
        try:
            
            current_categories = categories
            
            if not current_categories:
                print(f"Fehler: Kein Kategoriensystem fuer Kodierer {self.coder_id} verfÜgbar")
                return None

            print(f"\nDeduktiver Kodierer 🤖 **{self.coder_id}** verarbeitet Chunk mit progressivem Kontext...")
            
            # Erstelle formatierte KategorienÜbersicht
            categories_overview = []
            for name, cat in current_categories.items():
                category_info = {
                    'name': name,
                    'definition': cat.definition,
                    'examples': list(cat.examples) if isinstance(cat.examples, set) else cat.examples,
                    'rules': list(cat.rules) if isinstance(cat.rules, set) else cat.rules,
                    'subcategories': {}
                }
                
                # FÜge Subkategorien hinzu
                for sub_name, sub_def in cat.subcategories.items():
                    category_info['subcategories'][sub_name] = sub_def
                    
                categories_overview.append(category_info)
            
            # Position im Dokument und Fortschritt berechnen
            position_info = f"Segment: {segment_info.get('position', '')}"
            doc_name = segment_info.get('doc_name', 'Unbekanntes Dokument')
            
            # Berechne die relative Position im Dokument (fuer das Reifungsmodell)
            chunk_id = 0
            total_chunks = 1
            if 'position' in segment_info:
                try:
                    # Extrahiere Chunk-Nummer aus "Chunk X"
                    chunk_id = int(segment_info['position'].split()[-1])
                    
                    # SchÄtze Gesamtanzahl der Chunks (basierend auf bisherigen Chunks)
                    # Alternative: TatsÄchliche Anzahl Übergeben, falls verfÜgbar
                    total_chunks = max(chunk_id * 1.5, 20)  # SchÄtzung
                    
                    document_progress = chunk_id / total_chunks
                    print(f"Dokumentfortschritt: ca. {document_progress:.1%}")
                except (ValueError, IndexError):
                    document_progress = 0.5  # Fallback
            else:
                document_progress = 0.5  # Fallback
                
            # Bestimme die aktuelle Reifephase basierend auf dem Fortschritt
            if document_progress < 0.3:
                reifephase = "PHASE 1 (Sammlung)"
                max_aenderung = "50%"
            elif document_progress < 0.7:
                reifephase = "PHASE 2 (Konsolidierung)"
                max_aenderung = "30%"
            else:
                reifephase = "PHASE 3 (PrÄzisierung)"
                max_aenderung = "10%"
                
            print(f"Summary-Reifephase: {reifephase}, max. Änderung: {max_aenderung}")
            
            # Angepasster Prompt basierend auf dem dreistufigen Reifungsmodell
            # Verbesserter summary_update_prompt fuer die _code_chunk_with_progressive_context Methode

            summary_update_prompt = f"""
            ## AUFGABE 2: SUMMARY-UPDATE ({reifephase}, {int(document_progress*100)}%)

            """

            # Robustere Phasen-spezifische Anweisungen
            if document_progress < 0.3:
                summary_update_prompt += """
            SAMMLUNG (0-30%) - STRUKTURIERTER AUFBAU:
            - SCHLÜSSELINFORMATIONEN: Beginne mit einer LISTE wichtigster Konzepte im Telegrammstil
            - FORMAT: "Thema1: Kernaussage; Thema2: Kernaussage" 
            - SPEICHERSTRUKTUR: Speichere alle Informationen in KATEGORIEN (z.B. Akteure, Prozesse, Faktoren)
            - KEINE EINLEITUNGEN oder narrative Elemente, NUR Fakten und Verbindungen
            - BEHALTE IMMER: Bereits dokumentierte SchlÜsselkonzepte mÜssen bestehen bleiben
            """
            elif document_progress < 0.7:
                summary_update_prompt += """
            KONSOLIDIERUNG (30-70%) - HIERARCHISCHE ORGANISATION:
            - SCHLÜSSELINFORMATIONEN BEWAHREN: Alle bisherigen Hauptkategorien beibehalten
            - NEUE STRUKTUR: Als hierarchische Liste mit Kategorien und Unterpunkten organisieren
            - KOMPRIMIEREN: Details aus gleichen Themenbereichen zusammenfÜhren
            - PRIORITÄTSFORMAT: "Kategorie: Hauptpunkt1; Hauptpunkt2 -> Detail"
            - STATT LAe–SCHEN: Verwandte Inhalte zusammenfassen, aber KEINE Kategorien eliminieren
            """
            else:
                summary_update_prompt += """
            PRÄZISIERUNG (70-100%) - VERDICHTUNG MIT THESAURUS:
            - THESAURUS-METHODE: Jede Kategorie braucht genau 1-2 SÄtze im Telegrammstil
            - HAUPTKONZEPTE STABIL HALTEN: Alle identifizierten Kategorien mÜssen enthalten bleiben
            - ABSTRAHIEREN: Einzelinformationen innerhalb einer Kategorie verdichten
            - STABILITÄTSPRINZIP: Einmal erkannte wichtige ZusammenhÄnge dÜrfen nicht verloren gehen
            - PRIORITÄTSORDNUNG: Wichtigste Informationen IMMER am Anfang jeder Kategorie
            """

            # Allgemeine Kriterien fuer StabilitÄt und Komprimierung
            summary_update_prompt += """

            INFORMATIONSERHALTUNGS-SYSTEM:
            - MAXIMUM 80 WAe–RTER - Komprimiere alte statt neue Informationen zu verwerfen
            - KATEGORIEBASIERT: Jedes Summary muss immer in 3-5 klare Themenkategorien strukturiert sein
            - SCHLÜSSELPRINZIP: Bilde das Summary als INFORMATIONALE HIERARCHIE:
            1. Stufe: Immer stabile Themenkategorien
            2. Stufe: Zentrale Aussagen zu jeder Kategorie
            3. Stufe: ErgÄnzende Details (diese kÖnnen komprimiert werden)
            - STABILITÄTSGARANTIE: Neue Iteration darf niemals vorherige Kategorie-Level-1-Information verlieren
            - KOMPRIMIERUNGSSTRATEGIE: Bei Platzmangel Details (Stufe 3) zusammenfassen statt zu entfernen
            - FORMAT: "Kategorie1: Hauptpunkt; Hauptpunkt. Kategorie2: Hauptpunkt; Detail." (mit Doppelpunkten)
            - GRUNDREGEL: Neue Informationen ergÄnzen bestehende Kategorien statt sie zu ersetzen
            """
            
            # Prompt mit erweiterter Aufgabe fuer Summary-Update
            prompt = self.prompt_handler.get_progressive_context_prompt(
                chunk=chunk,
                categories_overview=categories_overview,
                current_summary=current_summary,
                position_info=position_info,
                summary_update_prompt=summary_update_prompt
            )
            
            # API-Call
            token_counter.start_request()

            response = await self.llm_provider.create_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte fuer qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Verarbeite Response mit Wrapper
            llm_response = LLMResponse(response)
            try:
                result = json.loads(llm_response.content)
            except json.JSONDecodeError as e:
                print(f"[ERROR][{self.coder_id}] JSONDecodeError: {e}")
                print(f"[ERROR][{self.coder_id}] Raw LLM response: {llm_response.content}")
                token_counter.track_error(self.model_name)
                return None # Or handle more gracefully
            

            
            token_counter.track_response(response, self.model_name)
            
            # Extrahiere relevante Teile
            if result and isinstance(result, dict):
                coding_result = result.get('coding_result', {})
                updated_summary = result.get('updated_summary', current_summary)
                
                # PrÜfe Wortlimit beim Summary
                if len(updated_summary.split()) > 80:  # Etwas Spielraum Über 70
                    words = updated_summary.split()
                    updated_summary = ' '.join(words[:70])
                    print(f"❌ Summary wurde gekÜrzt: {len(words)} -> 70 WÖrter")
                
                # Analyse der VerÄnderungen
                if current_summary:
                    # Berechne Prozent der Änderung
                    old_words = set(current_summary.lower().split())
                    new_words = set(updated_summary.lower().split())
                    
                    if old_words:
                        # Jaccard-Distanz als Maẞ fuer VerÄnderung
                        unchanged = len(old_words.intersection(new_words))
                        total = len(old_words.union(new_words))
                        change_percent = (1 - (unchanged / total)) * 100
                        
                        print(f"Summary Änderung: {change_percent:.1f}% (Ziel: max. {max_aenderung})")
                
                if coding_result:
                    paraphrase = coding_result.get('paraphrase', '')
                    if paraphrase:
                        print(f"\nðŸ—’ï¸  Paraphrase: {paraphrase}")
                    print(f"  ✅ Kodierung von {self.coder_id}: 📝  {coding_result.get('category', '')}")
                    print(f"  ✅ Subkategorien von {self.coder_id}: 📝  {', '.join(coding_result.get('subcategories', []))}")
                    print(f"  ✅ Keywords von {self.coder_id}: 📝  {coding_result.get('keywords', '')}")
                    print(f"\n🔀 Summary fuer {doc_name} aktualisiert ({len(updated_summary.split())} WÖrter):")
                    print(f"{updated_summary[:1000]}..." if len(updated_summary) > 100 else f"🔀„ {updated_summary}")
                    
                    # Kombiniertes Ergebnis zurÜckgeben
                    return {
                        'coding_result': coding_result,
                        'updated_summary': updated_summary
                    }
                else:
                    print(f"  âœ— Keine gÜltige Kodierung erhalten")
                    return None
            else:
                print("  âœ— Keine gÜltige Antwort erhalten")
                return None
                
        except Exception as e:
            print(f"Fehler bei der Kodierung durch {self.coder_id}: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()
            return None

    async def code_chunk_with_focus(self, chunk: str, categories: Dict[str, CategoryDefinition], 
                                focus_category: str, focus_context: Dict) -> Optional[CodingResult]:
        """
        Kodiert einen Text-Chunk mit Fokus auf eine bestimmte Kategorie (fuer Mehrfachkodierung).
        
        Args:
            chunk: Zu kodierender Text
            categories: Kategoriensystem  
            focus_category: Kategorie auf die fokussiert werden soll
            focus_context: Kontext zur fokussierten Kategorie mit 'justification', 'text_aspects', 'relevance_score'
            
        Returns:
            Optional[CodingResult]: Kodierungsergebnis mit Fokus-Kennzeichnung
        """
        
        try:
            if not categories:
                print(f"Fehler: Kein Kategoriensystem fuer Kodierer {self.coder_id} verfÜgbar")
                return None

            # print(f"    [TARGET] Fokuskodierung fuer Kategorie: {focus_category} (Relevanz: {focus_context.get('relevance_score', 0):.2f})")

            # Erstelle formatierte KategorienÜbersicht mit Fokus-Hervorhebung
            categories_overview = []
            for name, cat in categories.items():  
                # Hebe Fokus-Kategorie hervor
                display_name = name
                if name == focus_category:
                    display_name = name
                
                category_info = {
                    'name': display_name,
                    'definition': cat.definition,
                    'examples': list(cat.examples) if isinstance(cat.examples, set) else cat.examples,
                    'rules': list(cat.rules) if isinstance(cat.rules, set) else cat.rules,
                    'subcategories': {}
                }
                
                # FÜge Subkategorien hinzu
                for sub_name, sub_def in cat.subcategories.items():
                    category_info['subcategories'][sub_name] = sub_def
                    
                categories_overview.append(category_info)

            # Erstelle fokussierten Prompt
            prompt = self.prompt_handler.get_focus_coding_prompt(
                chunk=chunk,
                categories_overview=categories_overview,
                focus_category=focus_category,
                focus_context=focus_context
            )
            
            try:
                token_counter.start_request()

                response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte fuer qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                # Verarbeite Response mit Wrapper
                llm_response = LLMResponse(response)
                try:
                    result = json.loads(llm_response.content)
                except json.JSONDecodeError as e:
                    print(f"[ERROR][{self.coder_id}] JSONDecodeError in code_chunk_with_progressive_context: {e}")
                    print(f"[ERROR][{self.coder_id}] Raw LLM response: {llm_response.content}")
                    token_counter.track_error(self.model_name)
                    return None

                
                token_counter.track_response(response, self.model_name)
                
                if result and isinstance(result, dict):
                    if result.get('category'):
                        # Verarbeite Paraphrase
                        paraphrase = result.get('paraphrase', '')
                        if paraphrase:
                            print(f"      ðŸ—’ï¸  Fokus-Paraphrase: {paraphrase}")

                        # Dokumentiere Fokus-Adherence
                        focus_adherence = result.get('focus_adherence', {})
                        followed_focus = focus_adherence.get('followed_focus', True)
                        focus_icon = "🎯" if followed_focus else "ℹ️"
                        
                        print(f"      {focus_icon} Fokuskodierung von {self.coder_id}: 📝  {result.get('category', '')}")
                        print(f"      ✅ Subkategorien: 📝  {', '.join(result.get('subcategories', []))}")
                        print(f"      ✅ Keywords: 📝  {result.get('keywords', '')}")
                        
                        if not followed_focus:
                            deviation_reason = focus_adherence.get('deviation_reason', 'Nicht angegeben')
                            print(f"      ❌ Fokus-Abweichung: {deviation_reason}")

                        # Debug-Ausgaben fuer Fokus-Details
                        if focus_adherence:
                            focus_score = focus_adherence.get('focus_category_score', 0)
                            chosen_score = focus_adherence.get('chosen_category_score', 0)
                            print(f"      🧾 Fokus-Score: {focus_score:.2f}, GewÄhlt-Score: {chosen_score:.2f}")

                        # Erweiterte BegrÜndung mit Fokus-Kennzeichnung
                        original_justification = result.get('justification', '')
                        focus_prefix = f"[FOKUS: {focus_category}, Adherence: {'Ja' if followed_focus else 'Nein'}] "
                        enhanced_justification = focus_prefix + original_justification
                    
                        return CodingResult(
                            category=result.get('category', ''),
                            subcategories=tuple(result.get('subcategories', [])),
                            justification=enhanced_justification,
                            confidence=result.get('confidence', {'total': 0.0, 'category': 0.0, 'subcategories': 0.0}),
                            text_references=tuple([chunk[:100]]),
                            uncertainties=None,
                            paraphrase=result.get('paraphrase', ''),
                            keywords=result.get('keywords', '')
                        )
                    else:
                        print("      âœ— Keine passende Kategorie gefunden")
                        return None
                    
            except Exception as e:
                print(f"Fehler bei API Call fuer fokussierte Kodierung: {str(e)}")
                return None          

        except Exception as e:
            print(f"Fehler bei der fokussierten Kodierung durch {self.coder_id}: {str(e)}")
            return None

    async def code_chunk_with_focus_and_context(self, 
                                            chunk: str, 
                                            categories: Dict[str, CategoryDefinition],
                                            focus_category: str,
                                            focus_context: Dict,
                                            current_summary: str,
                                            segment_info: Dict,
                                            update_summary: bool = False) -> Dict:
        """
        Kodiert einen Text-Chunk mit Fokus auf eine Kategorie UND progressivem Kontext.
        Kombiniert die FunktionalitÄt von code_chunk_with_focus und code_chunk_with_progressive_context.
        
        Args:
            chunk: Zu kodierender Text
            categories: Kategoriensystem
            focus_category: Kategorie auf die fokussiert werden soll
            focus_context: Kontext zur fokussierten Kategorie
            current_summary: Aktuelles Dokument-Summary
            segment_info: ZusÄtzliche Informationen Über das Segment
            update_summary: Ob das Summary aktualisiert werden soll
            
        Returns:
            Dict: EnthÄlt sowohl Kodierungsergebnis als auch aktualisiertes Summary
        """
        try:
            if not categories:
                print(f"Fehler: Kein Kategoriensystem fuer Kodierer {self.coder_id} verfÜgbar")
                return None

            print(f"    [TARGET] Fokuskodierung fuer Kategorie: {focus_category} (Relevanz: {focus_context.get('relevance_score', 0):.2f})")

            # Erstelle formatierte KategorienÜbersicht mit Fokus-Hervorhebung
            categories_overview = []
            for name, cat in categories.items():  
                # Hebe Fokus-Kategorie hervor
                display_name = name
                if name == focus_category:
                    display_name = name
                
                category_info = {
                    'name': display_name,
                    'definition': cat.definition,
                    'examples': list(cat.examples) if isinstance(cat.examples, set) else cat.examples,
                    'rules': list(cat.rules) if isinstance(cat.rules, set) else cat.rules,
                    'subcategories': {}
                }
                
                # FÜge Subkategorien hinzu
                for sub_name, sub_def in cat.subcategories.items():
                    category_info['subcategories'][sub_name] = sub_def
                    
                categories_overview.append(category_info)

            # FIX: DEBUG - PrÜfe ob gefilterte Kategorien wirklich im Prompt landen
            # print(f"    🕵️ DEBUG: Categories Overview fuer Prompt:")
            # for cat_info in categories_overview:
            #     cat_name = cat_info['name']
            #     subcats = list(cat_info['subcategories'].keys())
            #     print(f"      - {cat_name}: {subcats}")

            # Position im Dokument und Fortschritt berechnen
            position_info = f"Segment: {segment_info.get('position', '')}"
            doc_name = segment_info.get('doc_name', 'Unbekanntes Dokument')
            
            # FIX: Summary-Update-Anweisungen mit dreistufigem Reifungsmodell (gleiche Logik wie in code_chunk_with_progressive_context)
            summary_update_prompt = ""
            if update_summary:
                # Berechne die relative Position im Dokument (fuer das Reifungsmodell)
                chunk_id = 0
                total_chunks = 1
                if 'position' in segment_info:
                    try:
                        # Extrahiere Chunk-Nummer aus "Chunk X"
                        chunk_id = int(segment_info['position'].split()[-1])
                        
                        # SchÄtze Gesamtanzahl der Chunks (basierend auf bisherigen Chunks)
                        # Alternative: TatsÄchliche Anzahl Übergeben, falls verfÜgbar
                        total_chunks = max(chunk_id * 1.5, 20)  # SchÄtzung
                        
                        document_progress = chunk_id / total_chunks
                        print(f"        Dokumentfortschritt: ca. {document_progress:.1%}")
                    except (ValueError, IndexError):
                        document_progress = 0.5  # Fallback
                else:
                    document_progress = 0.5  # Fallback
                    
                # Bestimme die aktuelle Reifephase basierend auf dem Fortschritt
                if document_progress < 0.3:
                    reifephase = "PHASE 1 (Sammlung)"
                    max_aenderung = "50%"
                elif document_progress < 0.7:
                    reifephase = "PHASE 2 (Konsolidierung)"
                    max_aenderung = "30%"
                else:
                    reifephase = "PHASE 3 (PrÄzisierung)"
                    max_aenderung = "10%"
                    
                print(f"        Summary-Reifephase: {reifephase}, max. Änderung: {max_aenderung}")
                
                # Angepasster Prompt basierend auf dem dreistufigen Reifungsmodell
                summary_update_prompt = f"""
                ## AUFGABE 2: SUMMARY-UPDATE ({reifephase}, {int(document_progress*100)}%)

                """

                # Robustere Phasen-spezifische Anweisungen
                if document_progress < 0.3:
                    summary_update_prompt += """
                SAMMLUNG (0-30%) - STRUKTURIERTER AUFBAU:
                - SCHLÜSSELINFORMATIONEN: Beginne mit einer LISTE wichtigster Konzepte im Telegrammstil
                - FORMAT: "Thema1: Kernaussage; Thema2: Kernaussage" 
                - SPEICHERSTRUKTUR: Speichere alle Informationen in KATEGORIEN (z.B. Akteure, Prozesse, Faktoren)
                - KEINE EINLEITUNGEN oder narrative Elemente, NUR Fakten und Verbindungen
                - BEHALTE IMMER: Bereits dokumentierte SchlÜsselkonzepte mÜssen bestehen bleiben
                """
                elif document_progress < 0.7:
                    summary_update_prompt += """
                KONSOLIDIERUNG (30-70%) - HIERARCHISCHE ORGANISATION:
                - SCHLÜSSELINFORMATIONEN BEWAHREN: Alle bisherigen Hauptkategorien beibehalten
                - NEUE STRUKTUR: Als hierarchische Liste mit Kategorien und Unterpunkten organisieren
                - KOMPRIMIEREN: Details aus gleichen Themenbereichen zusammenfÜhren
                - PRIORITÄTSFORMAT: "Kategorie: Hauptpunkt1; Hauptpunkt2 -> Detail"
                - STATT LAe–SCHEN: Verwandte Inhalte zusammenfassen, aber KEINE Kategorien eliminieren
                """
                else:
                    summary_update_prompt += """
                PRÄZISIERUNG (70-100%) - VERDICHTUNG MIT THESAURUS:
                - THESAURUS-METHODE: Jede Kategorie braucht genau 1-2 SÄtze im Telegrammstil
                - HAUPTKONZEPTE STABIL HALTEN: Alle identifizierten Kategorien mÜssen enthalten bleiben
                - ABSTRAHIEREN: Einzelinformationen innerhalb einer Kategorie verdichten
                - STABILITÄTSPRINZIP: Einmal erkannte wichtige ZusammenhÄnge dÜrfen nicht verloren gehen
                - PRIORITÄTSORDNUNG: Wichtigste Informationen IMMER am Anfang jeder Kategorie
                """

                # Allgemeine Kriterien fuer StabilitÄt und Komprimierung
                summary_update_prompt += """

                INFORMATIONSERHALTUNGS-SYSTEM:
                - MAXIMUM 80 WAe–RTER - Komprimiere alte statt neue Informationen zu verwerfen
                - KATEGORIEBASIERT: Jedes Summary muss immer in 3-5 klare Themenkategorien strukturiert sein
                - SCHLÜSSELPRINZIP: Bilde das Summary als INFORMATIONALE HIERARCHIE:
                1. Stufe: Immer stabile Themenkategorien
                2. Stufe: Zentrale Aussagen zu jeder Kategorie
                3. Stufe: ErgÄnzende Details (diese kÖnnen komprimiert werden)
                - STABILITÄTSGARANTIE: Neue Iteration darf niemals vorherige Kategorie-Level-1-Information verlieren
                - KOMPRIMIERUNGSSTRATEGIE: Bei Platzmangel Details (Stufe 3) zusammenfassen statt zu entfernen
                - FORMAT: "Kategorie1: Hauptpunkt; Hauptpunkt. Kategorie2: Hauptpunkt; Detail." (mit Doppelpunkten)
                - GRUNDREGEL: Neue Informationen ergÄnzen bestehende Kategorien statt sie zu ersetzen
                """

            # Erstelle fokussierten Prompt mit Kontext
            prompt = self.prompt_handler.get_focus_context_coding_prompt(
                chunk=chunk,
                categories_overview=categories_overview,
                focus_category=focus_category,
                focus_context=focus_context,
                current_summary=current_summary,
                position_info=position_info,
                summary_update_prompt=summary_update_prompt,
                update_summary=update_summary
            )
            

            # API-Call
            token_counter.start_request()

            response = await self.llm_provider.create_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte fuer qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Verarbeite Response mit Wrapper
            llm_response = LLMResponse(response)
            try:
                result = json.loads(llm_response.content)
            except json.JSONDecodeError as e:
                print(f"[ERROR][{self.coder_id}] JSONDecodeError: {e}")
                print(f"[ERROR][{self.coder_id}] Raw LLM response: {llm_response.content}")
                token_counter.track_error(self.model_name)
                return None # Or handle more gracefully
            

            
            token_counter.track_response(response, self.model_name)
            
            # Extrahiere relevante Teile
            if result and isinstance(result, dict):
                coding_result = result.get('coding_result', {})
                
                # Summary nur aktualisieren wenn angefordert
                if update_summary:
                    updated_summary = result.get('updated_summary', current_summary)
                    
                    # PrÜfe Wortlimit beim Summary
                    if len(updated_summary.split()) > 80:
                        words = updated_summary.split()
                        updated_summary = ' '.join(words[:70])
                        print(f"        ❌ Summary wurde gekÜrzt: {len(words)} -> 70 WÖrter")
                    
                    # FIX: Analyse der VerÄnderungen (gleich wie in code_chunk_with_progressive_context)
                    if current_summary:
                        # Berechne Prozent der Änderung
                        old_words = set(current_summary.lower().split())
                        new_words = set(updated_summary.lower().split())
                        
                        if old_words:
                            # Jaccard-Distanz als Maẞ fuer VerÄnderung
                            unchanged = len(old_words.intersection(new_words))
                            total = len(old_words.union(new_words))
                            change_percent = (1 - (unchanged / total)) * 100
                            
                            print(f"        Summary Änderung: {change_percent:.1f}% (Ziel: max. {max_aenderung})")
                else:
                    updated_summary = current_summary
                
                if coding_result:
                    paraphrase = coding_result.get('paraphrase', '')
                    if paraphrase:
                        print(f"        ðŸ—’ï¸  Fokus-Kontext-Paraphrase: {paraphrase}")

                    # Dokumentiere Fokus-Adherence
                    focus_adherence = coding_result.get('focus_adherence', {})
                    followed_focus = focus_adherence.get('followed_focus', True)
                    focus_icon = "🎯" if followed_focus else "ℹ️"
                    
                    print(f"        {focus_icon} Fokus-Kontext-Kodierung von {self.coder_id}: 📝  {coding_result.get('category', '')}")
                    print(f"        ✅ Subkategorien: 📝  {', '.join(coding_result.get('subcategories', []))}")
                    print(f"        ✅ Keywords: 📝  {coding_result.get('keywords', '')}")
                    
                    if not followed_focus:
                        deviation_reason = focus_adherence.get('deviation_reason', 'Nicht angegeben')
                        print(f"        ❌ Fokus-Abweichung: {deviation_reason}")

                    if update_summary:
                        print(f"        🔀 Summary fuer {doc_name} aktualisiert ({len(updated_summary.split())} WÖrter):")
                        print(f"        {updated_summary[:100]}..." if len(updated_summary) > 100 else f"        🔀„ {updated_summary}")
                    
                    # Kombiniertes Ergebnis zurÜckgeben
                    return {
                        'coding_result': coding_result,
                        'updated_summary': updated_summary
                    }
                else:
                    print(f"        âœ— Keine gÜltige Kodierung erhalten")
                    return None
            else:
                print("        âœ— Keine gÜltige Antwort erhalten")
                return None
                
        except Exception as e:
            print(f"Fehler bei der fokussierten Kontext-Kodierung durch {self.coder_id}: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()
            return None
        
    async def _check_relevance(self, chunk: str) -> bool:
        """
        PrÜft die Relevanz eines Chunks fuer die Forschungsfrage.
        
        Args:
            chunk: Zu prÜfender Text
            
        Returns:
            bool: True wenn der Text relevant ist
        """
        try:
            prompt = self.prompt_handler.get_segment_relevance_assessment_prompt(chunk)

            token_counter.start_request()
            
            response = await self.llm_provider.create_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein Experte fuer qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
            # Verarbeite Response mit Wrapper
            llm_response = LLMResponse(response)
            try:
                result = json.loads(llm_response.content)
            except json.JSONDecodeError as e:
                print(f"[ERROR][{self.coder_id}] JSONDecodeError: {e}")
                print(f"[ERROR][{self.coder_id}] Raw LLM response: {llm_response.content}")
                token_counter.track_error(self.model_name)
                return None # Or handle more gracefully
            
            
            
            token_counter.track_response(response, self.model_name)

            # Detaillierte Ausgabe der RelevanzprÜfung
            if result.get('is_relevant'):
                print(f"✅ Relevanz bestÄtigt (Konfidenz: {result.get('confidence', 0):.2f})")
                if result.get('key_aspects'):
                    print("  Relevante Aspekte:")

                    for aspect in result['key_aspects']:
                        print(f"  - {aspect}")
            else:
                print(f"⚠️ Nicht relevant: {result.get('justification', 'Keine BegrÜndung')}")

            return result.get('is_relevant', False)

        except Exception as e:
            print(f"Fehler bei der RelevanzprÜfung: {str(e)}")
            return True  # Im Zweifelsfall als relevant markieren

    
    def _validate_coding(self, result: dict) -> Optional[CodingResult]:
        """Validiert und konvertiert das API-Ergebnis in ein CodingResult-Objekt"""
        try:
            return CodingResult(
                category=result.get('category', ''),
                subcategories=result.get('subcategories', []),
                justification=result.get('justification', ''),
                confidence=result.get('confidence', {'total': 0, 'category': 0, 'subcategories': 0}),
                text_references=result.get('text_references', []),
                uncertainties=result.get('uncertainties', [])
            )
        except Exception as e:
            print(f"Fehler bei der Validierung des Kodierungsergebnisses: {str(e)}")
            return None

