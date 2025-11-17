"""
Induktive Kodierung für QCA-AID
================================
Ergänzung deduktiver Kategorien durch induktive Kategorien mittels LLM.
"""

import json
from ..utils.tracking.token_tracker import TokenTracker
from ..utils.llm.response import LLMResponse
from ..utils.llm.factory import LLMProviderFactory
import asyncio
from typing import Dict, Optional, List, Set, Tuple, Any
from collections import defaultdict

from ..core.config import CONFIG, FORSCHUNGSFRAGE, KODIERREGELN, DEDUKTIVE_KATEGORIEN
from ..core.data_models import CategoryDefinition, CodingResult
from ..management import DevelopmentHistory
from ..QCA_Prompts import QCAPrompts, ConfidenceScales

# Globaler Token-Counter
token_counter = TokenTracker()


class InductiveCoder:
    """
    Vereinfachter induktiver Kodierer mit strikter 2-Phasen-Struktur:
    Phase 1: Kategoriensystem-Aufbau (mit strenger SÄttigung)
    Phase 2: Kodierung mit festem System
    """
    
    def __init__(self, model_name: str, temperature: float, history: DevelopmentHistory, output_dir: str, config: dict = None):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.output_dir = output_dir
        self.history = history
        self.config = config or CONFIG  # KORREKTUR: Speichere config
        
        # LLM Provider (unverÄndert)
        provider_name = CONFIG.get('MODEL_PROVIDER', 'openai').lower()
        # Übergebe model_name für Capability-Testing
        self.llm_provider = LLMProviderFactory.create_provider(provider_name, model_name=model_name)
        
        # Cache und Tracking (unverÄndert)
        self.category_cache = {}
        self.analysis_cache = {}
        self.batch_results = []
        self.similarity_cache = {}
        
        # VERBESSERTE SÄttigungsschwellen (aus dem verbesserten Code)
        self.MIN_CONFIDENCE = 0.7
        self.MIN_EXAMPLES = 2
        self.MIN_CATEGORY_USAGE = 2
        self.MAX_CATEGORIES_PER_BATCH = 5
        
        # VERSCHÄRFTE SÄttigungskriterien (aus dem verbesserten Code)
        self.MIN_BATCHES_BEFORE_SATURATION = 5
        self.MIN_MATERIAL_COVERAGE = 0.8
        self.STABILITY_THRESHOLD = 3
        
        # Theoretische SÄttigungsmetriken (aus dem verbesserten Code)
        self.theoretical_saturation_history = []
        self.category_development_phases = []
        
        # Phasen-Management (unverÄndert)
        self.current_phase = "development"
        self.categories_locked = False
        self.development_complete = False
        
        # SÄttigungs-Tracking (unverÄndert)
        self.batches_without_new_categories = 0
        self.category_usage_history = {}
        self.rejected_categories = []
        
        # FÜr Grounded Theory Modus (unverÄndert)
        self.collected_subcodes = []
        self.segment_analyses = []

        self.discovered_aspects = set()
        self.batch_metrics = []
        
        # Prompt-Handler (unverÄndert)
        self.prompt_handler = QCAPrompts(
            forschungsfrage=FORSCHUNGSFRAGE,
            kodierregeln=KODIERREGELN,
            deduktive_kategorien=DEDUKTIVE_KATEGORIEN
        )

        print(f"\nðŸ”¬ Induktive Kodierung initialisiert:")
        print(f"- Min. Batches vor SÄttigung: {self.MIN_BATCHES_BEFORE_SATURATION}")
        print(f"- Min. Materialabdeckung: {self.MIN_MATERIAL_COVERAGE:.0%}")
        print(f"- StabilitÄtsschwelle: {self.STABILITY_THRESHOLD} Batches")
    
    
    
        
    def _create_proper_batches(self, segments: List[str], batch_size: int) -> List[List[str]]:
        """
        VERBESSERT: Erstellt Batches ohne kÜnstliche GrÖẞenreduzierung
        """
        if not segments:
            return []
        
        print(f"🔀¦ Erstelle Batches: {len(segments)} Segmente -> Batch-GrÖẞe {batch_size}")
        
        batches = []
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batches.append(batch)
        
        print(f"🔀¦ Ergebnis: {len(batches)} gleichmÄẞige Batches erstellt")
        return batches

    async def _validate_and_integrate_strict(self, candidates: Dict[str, CategoryDefinition], 
                                           existing: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        Validierung und automatische Konsolidierung neuer Kategorien
        """
        validated = {}
        
        for name, category in candidates.items():
            # 1. ÄhnlichkeitsprÜfung
            similar_existing = self._find_similar_category(category, existing)
            if similar_existing:
                print(f"ℹ️ '{name}' zu Ähnlich zu '{similar_existing}' - wird konsolidiert")
                # Automatische Konsolidierung statt Ablehnung
                consolidated = await self._auto_merge_categories(
                    category, existing[similar_existing], name, similar_existing
                )
                if consolidated:
                    existing[similar_existing] = consolidated
                    # WICHTIG: Nutzung fuer konsolidierte Kategorie erhÖhen
                    self.category_usage_history[similar_existing] = self.category_usage_history.get(similar_existing, 0) + 1
                continue
            
            # 2. QualitÄtsprÜfung
            if await self._meets_quality_standards(category):
                validated[name] = category
                # WICHTIG: Nutzung fuer neue Kategorie setzen
                self.category_usage_history[name] = self.category_usage_history.get(name, 0) + 1
                print(f"[OK] '{name}' validiert (Nutzung: {self.category_usage_history[name]})")
            else:
                print(f"⚠️ '{name}' erfÜllt QualitÄtsstandards nicht")
        
        return validated
    
    async def _consolidate_categories(self, categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        Automatische Konsolidierung Ähnlicher Kategorien
        """
        print("\nℹ️ Starte automatische Konsolidierung...")
        
        consolidated = categories.copy()
        merge_candidates = []
        
        # Finde Konsolidierungskandidaten
        category_names = list(consolidated.keys())
        for i in range(len(category_names)):
            for j in range(i + 1, len(category_names)):
                name1, name2 = category_names[i], category_names[j]
                if name1 in consolidated and name2 in consolidated:
                    similarity = self._calculate_category_similarity(
                        consolidated[name1], consolidated[name2]
                    )
                    if similarity > 0.7:  # similarity_threshold
                        merge_candidates.append((name1, name2, similarity))
        
        # Sortiere nach Ähnlichkeit
        merge_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # FÜhre Konsolidierungen durch
        for name1, name2, similarity in merge_candidates[:3]:  # Max 3 Merges pro Runde
            if name1 in consolidated and name2 in consolidated:
                print(f"ðŸ”— Konsolidiere '{name1}' + '{name2}' (Ähnlichkeit: {similarity:.2f})")
                merged = await self._merge_categories_intelligent(
                    consolidated[name1], consolidated[name2], name1, name2
                )
                if merged:
                    # Verwende den besseren Namen
                    better_name = self._choose_better_name(name1, name2)
                    consolidated[better_name] = merged
                    
                    # Entferne die anderen
                    other_name = name2 if better_name == name1 else name1
                    del consolidated[other_name]
                    
                    print(f"[OK] Konsolidiert zu '{better_name}'")
        
        return consolidated
    
    async def _finalize_categories(self, categories: Dict[str, CategoryDefinition]) -> Dict[str, CategoryDefinition]:
        """
        Finale Bereinigung des Kategoriensystems
        """
        print("\nðŸ§¹ Finale Bereinigung...")
        
        cleaned = {}
        
        for name, category in categories.items():
            # KORRIGIERT: Verwende deutlich niedrigere Schwelle oder Überspringe Check
            usage_count = self.category_usage_history.get(name, 0)
            
            # TEMPORÄRER FIX: Akzeptiere alle Kategorien in der Entwicklungsphase
            if self.current_phase == "development":
                print(f"[OK] '{name}' Übernommen (Entwicklungsphase)")
                cleaned[name] = category
                continue
                
            # KORRIGIERT: Viel niedrigere Schwelle
            min_usage = max(1, self.MIN_CATEGORY_USAGE // 3)  # 1 statt 3
            
            if usage_count >= min_usage:
                # Verbessere Definition falls nÖtig
                if len(category.definition.split()) < 20:
                    enhanced = await self._enhance_category_definition(category)
                    if enhanced:
                        category = category.replace(definition=enhanced.definition)
                
                cleaned[name] = category
                print(f"[OK] '{name}' Übernommen (Nutzung: {usage_count})")
            else:
                print(f"⚠️ '{name}' entfernt (Zu wenig genutzt: {usage_count}, Mindest: {min_usage})")
        
        return cleaned
    
    def _update_usage_history(self, category_names: List[str]) -> None:
        """
        Aktualisiert die Nutzungshistorie fuer Kategorien
        """
        for name in category_names:
            if name in self.category_usage_history:
                self.category_usage_history[name] += 1
            else:
                self.category_usage_history[name] = 1
        
        print(f"🧾 Nutzungshistorie aktualisiert fuer: {category_names}")
        print(f"    Aktuelle Nutzung: {dict(list(self.category_usage_history.items())[-3:])}")

    def _create_category_definition(self, cat_data: dict) -> CategoryDefinition:
        """
        Erstellt CategoryDefinition aus API-Response Dictionary
        """
        try:
            return CategoryDefinition(
                name=cat_data.get('name', ''),
                definition=cat_data.get('definition', ''),
                examples=cat_data.get('evidence', []),
                rules=[],  # Wird spaeter entwickelt
                subcategories={
                    sub.get('name', ''): sub.get('definition', '')
                    for sub in cat_data.get('subcategories', [])
                },
                added_date=datetime.now().strftime("%Y-%m-%d"),
                modified_date=datetime.now().strftime("%Y-%m-%d")
            )
        except Exception as e:
            print(f"Fehler bei CategoryDefinition-Erstellung: {str(e)}")
            return None
    
    
    
    def _format_existing_categories(self, categories: Dict[str, CategoryDefinition]) -> str:
        """Formatiert bestehende Kategorien fuer Prompt"""
        if not categories:
            return "Keine bestehenden Kategorien."
        
        formatted = []
        for name, cat in categories.items():
            definition_preview = cat.definition[:100] + "..." if len(cat.definition) > 100 else cat.definition
            formatted.append(f"- {name}: {definition_preview}")
        
        return "\n".join(formatted)

    
    async def develop_category_system(self, segments: List[str], initial_categories: Dict[str, CategoryDefinition] = None) -> Dict[str, CategoryDefinition]:
        """
        VERBESSERTE Kategorienentwicklung mit korrekter SÄttigungslogik
        """
        print(f"\n🕵️ Starte verbesserte induktive Entwicklung mit {len(segments)} Segmenten")
        
        current_categories = initial_categories.copy() if initial_categories else {}
        analysis_mode = CONFIG.get('ANALYSIS_MODE', 'inductive')
        
        print(f"\n🧾 Analysemodus: {analysis_mode.upper()}")
        
        # Reset Tracking
        self.theoretical_saturation_history = []
        self.category_development_phases = []
        self.batches_without_new_categories = 0
        
        # VERBESSERTE Batch-Erstellung (keine kÜnstliche Reduzierung)
        print("\n🔀¦ Erstelle optimierte Batches...")

        # Erstelle Batches direkt
        effective_batch_size = min(CONFIG.get('BATCH_SIZE', 5), len(segments))
        batches = self._create_proper_batches(segments, effective_batch_size)
        
        
        print(f"🧾 Batch-Konfiguration:")
        print(f"- Relevante Segmente: {len(segments)}")
        print(f"- Batch-GrÖẞe: {effective_batch_size}")
        print(f"- Anzahl Batches: {len(batches)}")
        
        working_categories = current_categories.copy()
        
        # HAUPTSCHLEIFE mit verbesserter SÄttigungslogik
        for batch_idx, batch in enumerate(batches):
            print(f"\n{'='*60}")
            print(f"🧾 BATCH {batch_idx + 1}/{len(batches)} - Kategorienentwicklung")
            print(f"{'='*60}")
            
            # Analysiere Batch
            new_candidates = await self._analyze_batch_improved(batch, working_categories, analysis_mode)
            
            # Validiere und integriere neue Kategorien
            if new_candidates:
                validated_categories = await self._validate_and_integrate_strict(new_candidates, working_categories)
                
                if validated_categories:
                    before_count = len(working_categories)
                    working_categories.update(validated_categories)
                    added_count = len(working_categories) - before_count
                    
                    print(f"[OK] {added_count} neue Kategorien integriert")
                    self.batches_without_new_categories = 0
                    self._update_usage_history(list(validated_categories.keys()))
                    
                    # Dokumentiere Entwicklungsphase
                    self.category_development_phases.append({
                        'batch': batch_idx + 1,
                        'new_categories': added_count,
                        'total_categories': len(working_categories),
                        'material_coverage': (batch_idx + 1) / len(batches)
                    })
                else:
                    print("⚠️ Keine Kategorien haben strenge Validierung bestanden")
                    self.batches_without_new_categories += 1
            else:
                print("ℹ️ Keine neuen Kategorien in diesem Batch")
                self.batches_without_new_categories += 1
            
            # VERBESSERTE SÄttigungsprÜfung
            saturation_metrics = self._assess_comprehensive_saturation(
                working_categories, 
                batch_idx + 1, 
                len(batches)
            )
            
            print(f"\nℹ️ SÄTTIGUNGSANALYSE:")
            print(f"- Theoretische SÄttigung: {saturation_metrics['theoretical_saturation']:.2f}")
            print(f"- Materialabdeckung: {saturation_metrics['material_coverage']:.1%}")
            print(f"- Stabile Batches: {saturation_metrics['stable_batches']}")
            print(f"- KategorienqualitÄt: {saturation_metrics['category_quality']:.2f}")
            print(f"- DiversitÄt: {saturation_metrics['category_diversity']:.2f}")
            
            # Speichere SÄttigungshistorie
            self.theoretical_saturation_history.append(saturation_metrics)
            
            # PrÜfe ALLE SÄttigungskriterien
            if self._check_comprehensive_saturation(saturation_metrics, batch_idx + 1, len(batches)):
                print(f"\n🏁 VOLLSTÄNDIGE SÄTTIGUNG erreicht nach Batch {batch_idx + 1}")
                print(f"🧾 SÄttigungsgrund:")
                for criterion, value in saturation_metrics.items():
                    print(f"   - {criterion}: {value}")
                break
            else:
                print(f"\nℹ️ SÄttigung noch nicht erreicht - fortsetzen")
                self._log_saturation_progress(saturation_metrics)
            
            # Zwischenkonsolidierung alle 3 Batches
            if (batch_idx + 1) % 3 == 0:
                print(f"\nℹ️ Zwischenkonsolidierung nach Batch {batch_idx + 1}")
                working_categories = await self._consolidate_categories(working_categories)
        
        # Finale Bereinigung und QualitÄtssicherung
        final_categories = await self._finalize_categories(working_categories)
        
        # Zeige finale Entwicklungsstatistiken
        self._show_development_summary(final_categories, initial_categories)
        
        return final_categories

    def _create_proper_batches(self, segments: List[str], batch_size: int) -> List[List[str]]:
        """
        VERBESSERT: Erstellt Batches ohne kÜnstliche GrÖẞenreduzierung
        """
        if not segments:
            return []
        
        print(f"🔀¦ Erstelle Batches: {len(segments)} Segmente -> Batch-GrÖẞe {batch_size}")
        
        batches = []
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batches.append(batch)
        
        print(f"🔀¦ Ergebnis: {len(batches)} gleichmÄẞige Batches erstellt")
        return batches

    def _assess_comprehensive_saturation(self, categories: Dict[str, CategoryDefinition], 
                                       current_batch: int, total_batches: int) -> Dict[str, float]:
        """
        VERBESSERTE umfassende SÄttigungsbeurteilung
        """
        # 1. Theoretische SÄttigung (KategorienqualitÄt und -vollstÄndigkeit)
        theoretical_saturation = self._calculate_theoretical_saturation(categories)
        
        # 2. Materialabdeckung
        material_coverage = current_batch / total_batches
        
        # 3. StabilitÄt (Batches ohne neue Kategorien)
        stability_ratio = self.batches_without_new_categories / max(1, current_batch)
        
        # 4. KategorienqualitÄt (Definition, Beispiele, Subkategorien)
        category_quality = self._assess_category_quality(categories)
        
        # 5. Kategorien-DiversitÄt (thematische Abdeckung)
        category_diversity = self._calculate_category_diversity(categories)
        
        return {
            'theoretical_saturation': theoretical_saturation,
            'material_coverage': material_coverage,
            'stable_batches': self.batches_without_new_categories,
            'stability_ratio': stability_ratio,
            'category_quality': category_quality,
            'category_diversity': category_diversity,
            'total_categories': len(categories)
        }

    def _calculate_theoretical_saturation(self, categories: Dict[str, CategoryDefinition]) -> float:
        """
        Berechnet theoretische SÄttigung basierend auf Kategorienreife und Forschungsabdeckung
        """
        if not categories:
            return 0.0
        
        # 1. Kategorienreife (Definition, Beispiele, Subkategorien)
        maturity_scores = []
        for cat in categories.values():
            score = 0
            # Definition (0-0.4)
            def_score = min(len(cat.definition.split()) / 30, 0.4)
            # Beispiele (0-0.3)
            example_score = min(len(cat.examples) / 5, 0.3)
            # Subkategorien (0-0.3)
            subcat_score = min(len(cat.subcategories) / 4, 0.3)
            
            total_score = def_score + example_score + subcat_score
            maturity_scores.append(total_score)
        
        avg_maturity = sum(maturity_scores) / len(maturity_scores)
        
        # 2. Forschungsabdeckung (Anzahl und DiversitÄt der Kategorien)
        # SchÄtze optimale Kategorienanzahl basierend auf Forschungsfrage
        estimated_optimal = 8  # Typisch fuer qualitative Analysen
        coverage_ratio = min(len(categories) / estimated_optimal, 1.0)
        
        # 3. Kombinierte theoretische SÄttigung
        theoretical_saturation = (avg_maturity * 0.7) + (coverage_ratio * 0.3)
        
        return min(theoretical_saturation, 1.0)

    def _assess_category_quality(self, categories: Dict[str, CategoryDefinition]) -> float:
        """
        Bewertet die durchschnittliche QualitÄt aller Kategorien
        """
        if not categories:
            return 0.0
        
        quality_scores = []
        for cat in categories.values():
            score = 0
            
            # Definition ausreichend (0-0.4)
            if len(cat.definition.split()) >= 20:
                score += 0.4
            elif len(cat.definition.split()) >= 10:
                score += 0.2
            
            # Beispiele vorhanden (0-0.3)
            if len(cat.examples) >= 3:
                score += 0.3
            elif len(cat.examples) >= 1:
                score += 0.15
            
            # Subkategorien entwickelt (0-0.3)
            if len(cat.subcategories) >= 3:
                score += 0.3
            elif len(cat.subcategories) >= 1:
                score += 0.15
            
            quality_scores.append(score)
        
        return sum(quality_scores) / len(quality_scores)

    def _calculate_category_diversity(self, categories: Dict[str, CategoryDefinition]) -> float:
        """
        Berechnet thematische DiversitÄt der Kategorien
        """
        if not categories:
            return 0.0
        
        # Sammle SchlÜsselwÖrter aus allen Definitionen
        all_keywords = set()
        for cat in categories.values():
            words = cat.definition.lower().split()
            keywords = [w for w in words if len(w) > 4]  # Nur lÄngere WÖrter
            all_keywords.update(keywords[:5])  # Top 5 pro Kategorie
        
        # DiversitÄt = VerhÄltnis von einzigartigen Begriffen zu Kategorien
        diversity = len(all_keywords) / (len(categories) * 3)  # Normalisiert
        return min(diversity, 1.0)

    def _check_comprehensive_saturation(self, saturation_metrics: Dict[str, float], 
                                      current_batch: int, total_batches: int) -> bool:
        """
        VERSCHÄRFTE SÄttigungsprÜfung mit mehreren Kriterien
        """
        # Mindestkriterien
        min_batches = max(self.MIN_BATCHES_BEFORE_SATURATION, total_batches * 0.3)
        min_material = self.MIN_MATERIAL_COVERAGE
        min_stability = self.STABILITY_THRESHOLD
        
        # PrÜfe alle Kriterien
        criteria_met = {
            'min_batches': current_batch >= min_batches,
            'material_coverage': saturation_metrics['material_coverage'] >= min_material,
            'theoretical_saturation': saturation_metrics['theoretical_saturation'] >= 0.8,
            'category_quality': saturation_metrics['category_quality'] >= 0.7,
            'stability': saturation_metrics['stable_batches'] >= min_stability,
            'sufficient_categories': saturation_metrics['total_categories'] >= 3
        }
        
        print(f"\n🕵️ SÄttigungskriterien:")
        for criterion, met in criteria_met.items():
            status = "[OK]" if met else "⚠️"
            print(f"   {status} {criterion}: {met}")
        
        # SÄttigung nur wenn ALLE Kriterien erfÜllt
        is_saturated = all(criteria_met.values())
        
        if is_saturated:
            print(f"\n[TARGET] ALLE SÄttigungskriterien erfÜllt!")
        else:
            missing = [k for k, v in criteria_met.items() if not v]
            print(f"\nℹ️ Fehlende Kriterien: {', '.join(missing)}")
        
        return is_saturated

    def _create_inductive_mode_prompt(self, segments_text: str, existing_categories: Dict[str, CategoryDefinition]) -> str:
        """
        Erstellt spezifischen Prompt fuer INDUCTIVE MODE (vollstÄndige induktive Kategorienentwicklung)
        """
        # Formatiere bestehende induktive Kategorien als Kontext (aber nicht als EinschrÄnkung)
        existing_context = ""
        if existing_categories:
            existing_names = list(existing_categories.keys())
            existing_context = f"""
            BESTEHENDE INDUKTIVE KATEGORIEN (als Kontext, NICHT als EinschrÄnkung):
            {', '.join(existing_names)}
            
            WICHTIG: Entwickle NEUE, EIGENSTÄNDIGE Kategorien, die sich thematisch von den bestehenden unterscheiden.
            Beachte aber die bereits entwickelten Kategorien um Redundanzen zu vermeiden.
            """
        
        return f"""
        INDUCTIVE MODE: VollstÄndige induktive Kategorienentwicklung

        {existing_context}

        AUFGABE: Entwickle voellig NEUE Hauptkategorien aus den folgenden Textsegmenten.
        Dies ist ein eigenstÄndiges induktives Kategoriensystem, unabhÄngig von deduktiven Kategorien.

        REGELN FÜR INDUCTIVE MODE:
        - Entwickle 1-{self.MAX_CATEGORIES_PER_BATCH} NEUE Hauptkategorien
        - Jede Kategorie muss mindestens {self.MIN_EXAMPLES} Textbelege haben
        - Konfidenz mindestens {self.MIN_CONFIDENCE}
        - Kategorien mÜssen thematisch eigenstÄndig und relevant sein
        - Erstelle auch 2-4 Subkategorien pro Hauptkategorie
        - Kategorien sollen neue Aspekte der Forschungsfrage beleuchten
        - Vermeide Redundanzen zu bereits entwickelten Kategorien

        FORSCHUNGSFRAGE: {FORSCHUNGSFRAGE}

        TEXTSEGMENTE:
        {segments_text}

        Antworte NUR mit JSON:
        {{
            "new_categories": [
                {{
                    "name": "Kategorie Name",
                    "definition": "AusfÜhrliche Definition (mindestens 20 WÖrter)",
                    "evidence": ["Textbelege aus den Segmenten"],
                    "confidence": 0.0-1.0,
                    "subcategories": [
                        {{
                            "name": "Subkategorie Name", 
                            "definition": "Subkategorie Definition"
                        }}
                    ],
                    "thematic_justification": "Warum diese Kategorie einen eigenstÄndigen Themenbereich abbildet"
                }}
            ],
            "development_assessment": {{
                "categories_developed": 0,
                "theoretical_saturation": 0.0-1.0,
                "new_themes_found": true/false,
                "recommendation": "continue/pause/stop"
            }}
        }}
        """
    
    def _create_abductive_mode_prompt(self, segments_text: str, existing_categories: Dict[str, CategoryDefinition]) -> str:
        """
        Erstellt spezifischen Prompt fuer ABDUCTIVE MODE (nur Subkategorien)
        """
        categories_context = []
        for cat_name, cat_def in existing_categories.items():
            categories_context.append({
                'name': cat_name,
                'definition': cat_def.definition[:200],
                'existing_subcategories': list(cat_def.subcategories.keys())
            })

        return f"""
        ABDUKTIVER MODUS: Entwickle NUR neue Subkategorien fuer bestehende Hauptkategorien.

        BESTEHENDE HAUPTKATEGORIEN:
        {json.dumps(categories_context, indent=2, ensure_ascii=False)}

        STRIKTE REGELN FÜR ABDUKTIVEN MODUS:
        - KEINE neuen Hauptkategorien entwickeln
        - NUR neue Subkategorien fuer bestehende Hauptkategorien
        - Subkategorien mÜssen neue, relevante Themenaspekte abbilden
        - Mindestens {self.MIN_EXAMPLES} Textbelege pro Subkategorie
        - Konfidenz mindestens {self.MIN_CONFIDENCE}
        - PrÜfe JEDE bestehende Hauptkategorie auf mÖgliche neue Subkategorien
        
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
                            "evidence": ["Textbelege"],
                            "confidence": 0.0-1.0,
                            "thematic_novelty": "Warum diese Subkategorie einen neuen Aspekt abbildet"
                        }}
                    ]
                }}
            }},
            "saturation_assessment": {{
                "subcategory_saturation": 0.0-1.0,
                "new_aspects_found": true/false,
                "recommendation": "continue/pause/stop"
            }}
        }}
        """

    def _create_standard_prompt(self, segments_text: str, existing_categories: Dict[str, CategoryDefinition]) -> str:
        """
        Erstellt Standard-Prompt fuer allgemeine induktive Kategorienentwicklung
        """
        existing_context = ""
        if existing_categories:
            existing_names = list(existing_categories.keys())
            existing_context = f"Bestehende Kategorien: {', '.join(existing_names)}"

        return f"""
        STANDARD INDUKTIVE KATEGORIENENTWICKLUNG

        {existing_context}

        AUFGABE: Entwickle neue Kategorien aus den folgenden Textsegmenten.

        ALLGEMEINE REGELN:
        - Entwickle 1-{self.MAX_CATEGORIES_PER_BATCH} neue Kategorien
        - Jede Kategorie braucht mindestens {self.MIN_EXAMPLES} Textbelege
        - Konfidenz mindestens {self.MIN_CONFIDENCE}
        - Erstelle aussagekrÄftige Definitionen
        - FÜge relevante Subkategorien hinzu

        FORSCHUNGSFRAGE: {FORSCHUNGSFRAGE}

        TEXTSEGMENTE:
        {segments_text}

        Antworte NUR mit JSON:
        {{
            "new_categories": [
                {{
                    "name": "Kategorie Name",
                    "definition": "Kategorie Definition",
                    "evidence": ["Textbelege"],
                    "confidence": 0.0-1.0,
                    "subcategories": [
                        {{"name": "Subkategorie", "definition": "Definition"}}
                    ]
                }}
            ]
        }}
        """
    
    async def _analyze_batch_improved(self, batch: List[str], existing_categories: Dict[str, CategoryDefinition], analysis_mode: str) -> Dict[str, CategoryDefinition]:
        """
        VERBESSERTE Batch-Analyse mit modusabhÄngiger Logik
        """
        segments_text = "\n\n=== SEGMENT BREAK ===\n\n".join(
            f"SEGMENT {i + 1}:\n{text}" 
            for i, text in enumerate(batch)
        )

        # ModusabhÄngige Prompt-Erstellung
        if analysis_mode == 'inductive':
            prompt = self._create_inductive_mode_prompt(segments_text, existing_categories)
        elif analysis_mode == 'abductive':
            prompt = self._create_abductive_mode_prompt(segments_text, existing_categories)
        else:
            prompt = self._create_standard_prompt(segments_text, existing_categories)

        try:
            token_counter.start_request()
            
            response = await self.llm_provider.create_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte fuer qualitative Inhaltsanalyse. Antworte auf deutsch. Antworte ausschliesslich mit einem JSON-Objekt."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            llm_response = LLMResponse(response)
            result = json.loads(llm_response.content)
            
            
            token_counter.track_response(response, self.model_name)
            
            # Verarbeite Ergebnisse
            candidates = {}
            
            for cat_data in result.get('new_categories', []):
                if cat_data.get('confidence', 0) >= self.MIN_CONFIDENCE:
                    candidates[cat_data['name']] = self._create_category_definition(cat_data)
                    print(f"[OK] Neuer Kandidat: '{cat_data['name']}' (Konfidenz: {cat_data.get('confidence', 0):.2f})")
            
            return candidates
            
        except Exception as e:
            print(f"Fehler bei verbesserter Batch-Analyse: {str(e)}")
            return {}

    def _create_abductive_mode_prompt(self, segments_text: str, existing_categories: Dict[str, CategoryDefinition]) -> str:
        """
        Erstellt spezifischen Prompt fuer abduktiven Modus (nur Subkategorien)
        """
        categories_context = []
        for cat_name, cat_def in existing_categories.items():
            categories_context.append({
                'name': cat_name,
                'definition': cat_def.definition,
                'existing_subcategories': list(cat_def.subcategories.keys())
            })

        return f"""
        ABDUKTIVER MODUS: Entwickle NUR neue Subkategorien fuer bestehende Hauptkategorien.

        BESTEHENDE HAUPTKATEGORIEN:
        {json.dumps(categories_context, indent=2, ensure_ascii=False)}

        STRIKTE REGELN:
        - KEINE neuen Hauptkategorien entwickeln
        - NUR neue Subkategorien fuer bestehende Hauptkategorien
        - Subkategorien mÜssen neue, relevante Themenaspekte abbilden
        - Mindestens {self.MIN_EXAMPLES} Textbelege pro Subkategorie
        - Konfidenz mindestens {self.MIN_CONFIDENCE}
        
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
                            "evidence": ["Textbelege"],
                            "confidence": 0.0-1.0,
                            "thematic_novelty": "Warum diese Subkategorie einen neuen Aspekt abbildet"
                        }}
                    ]
                }}
            }},
            "saturation_assessment": {{
                "subcategory_saturation": 0.0-1.0,
                "new_aspects_found": true/false,
                "recommendation": "continue/pause/stop"
            }}
        }}
        """

    def _log_saturation_progress(self, saturation_metrics: Dict[str, float]) -> None:
        """
        Protokolliert SÄttigungsfortschritt fuer Benutzer-Feedback
        """
        print(f"\n🧾 SÄttigungsfortschritt:")
        print(f"   [TARGET] Theoretische SÄttigung: {saturation_metrics['theoretical_saturation']:.1%}")
        print(f"   ℹ️ Materialabdeckung: {saturation_metrics['material_coverage']:.1%}")
        print(f"   ℹ️ StabilitÄt: {saturation_metrics['stable_batches']} Batches ohne neue Kategorien")
        print(f"   â­ KategorienqualitÄt: {saturation_metrics['category_quality']:.1%}")
        print(f"   ðŸŒˆ DiversitÄt: {saturation_metrics['category_diversity']:.1%}")

    def _show_development_summary(self, final_categories: Dict[str, CategoryDefinition], 
                                initial_categories: Dict[str, CategoryDefinition]) -> None:
        """
        Zeigt finale Entwicklungsstatistiken
        """
        print(f"\n{'='*60}")
        print(f"🧾 KATEGORIENENTWICKLUNG ABGESCHLOSSEN")
        print(f"{'='*60}")
        
        # Grundstatistiken
        initial_count = len(initial_categories) if initial_categories else 0
        final_count = len(final_categories)
        new_categories = final_count - initial_count
        
        print(f"ℹ️ Kategorien-Bilanz:")
        print(f"   - Initial: {initial_count}")
        print(f"   - Neu entwickelt: {new_categories}")
        print(f"   - Final: {final_count}")
        
        # SÄttigungshistorie
        if self.theoretical_saturation_history:
            final_saturation = self.theoretical_saturation_history[-1]
            print(f"\n[TARGET] Finale SÄttigung:")
            print(f"   - Theoretische SÄttigung: {final_saturation['theoretical_saturation']:.1%}")
            print(f"   - KategorienqualitÄt: {final_saturation['category_quality']:.1%}")
            print(f"   - DiversitÄt: {final_saturation['category_diversity']:.1%}")
        
        # Entwicklungsphasen
        if self.category_development_phases:
            print(f"\n🧾 Entwicklungsphasen:")
            for phase in self.category_development_phases:
                print(f"   Batch {phase['batch']}: +{phase['new_categories']} -> {phase['total_categories']} total")

    
    
    def _format_existing_categories(self, categories: Dict[str, CategoryDefinition]) -> str:
        """Formatiert bestehende Kategorien fuer Prompt"""
        if not categories:
            return "Keine bestehenden Kategorien."
        
        formatted = []
        for name, cat in categories.items():
            formatted.append(f"- {name}: {cat.definition[:100]}...")
        
        return "\n".join(formatted)

    
    async def _enhance_category_definition(self, category: CategoryDefinition) -> Optional[CategoryDefinition]:
        """Verbessert Kategoriendefinition"""
        try:
            prompt = self.prompt_handler._get_definition_enhancement_prompt({
                'name': category.name,
                'definition': category.definition,
                'examples': category.examples
            })
            
            response = await self.llm_provider.create_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte fuer qualitative Inhaltsanalyse."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            enhanced_def = response.choices[0].message.content.strip()
            
            if len(enhanced_def.split()) >= 20:
                return category.replace(definition=enhanced_def)
            
        except Exception as e:
            print(f"Fehler bei Definition-Verbesserung: {str(e)}")
        
        return None
    
    async def analyze_grounded_batch(self, segments: List[str], material_percentage: float) -> Dict[str, Any]:
        """
        Analysiert einen Batch von Segmenten im 'grounded' Modus.
        Extrahiert Subcodes und Keywords ohne direkte Zuordnung zu Hauptkategorien.
        Sorgt fuer angemessenen Abstand zwischen Keywords und Subcodes.
        
        Args:
            segments: Liste der Textsegmente
            material_percentage: Prozentsatz des verarbeiteten Materials
            
        Returns:
            Dict[str, Any]: Analyseergebnisse mit Subcodes und Keywords
        """
        try:
            # Cache-Key erstellen
            cache_key = (
                tuple(segments),
                'grounded'
            )
            
            # PrÜfe Cache
            if cache_key in self.analysis_cache:
                print("Nutze gecachte Analyse")
                return self.analysis_cache[cache_key]

            # Bestehende Subcodes sammeln
            existing_subcodes = []
            if hasattr(self, 'collected_subcodes'):
                existing_subcodes = [sc.get('name', '') for sc in self.collected_subcodes if isinstance(sc, dict)]
            
            # Definiere JSON-Schema fuer den grounded Modus
            json_schema = '''{
                "segment_analyses": [
                    {
                        "segment_text": "Textsegment",
                        "subcodes": [
                            {
                                "name": "Subcode-Name",
                                "definition": "Definition des Subcodes",
                                "evidence": ["Textbelege"],
                                "keywords": ["SchlÜsselwÖrter des Subcodes"],
                                "confidence": 0.0-1.0
                            }
                        ],
                        "memo": "Analytische Notizen zum Segment"
                    }
                ],
                "abstraction_quality": {
                    "keyword_subcode_distinction": 0.0-1.0,
                    "comment": "Bewertung der Abstraktionshierarchie"
                },
                "saturation_metrics": {
                    "new_aspects_found": true/false,
                    "coverage": 0.0-1.0,
                    "justification": "BegrÜndung"
                }
            }'''

            # Verbesserter Prompt mit Fokus auf die Abstraktionshierarchie
            prompt = self.prompt_handler.get_grounded_analysis_prompt(
                segments=segments,
                existing_subcodes=existing_subcodes,
                json_schema=json_schema
            )
            
            token_counter.start_request()

            # API-Call
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
            result = json.loads(llm_response.content)
            
            
            token_counter.track_response(response, self.model_name)

            # Cache das Ergebnis
            self.analysis_cache[cache_key] = result
            
            # Bewertung der AbstraktionsqualitÄt
            abstraction_quality = result.get('abstraction_quality', {})
            if abstraction_quality and 'keyword_subcode_distinction' in abstraction_quality:
                quality_score = abstraction_quality['keyword_subcode_distinction']
                quality_comment = abstraction_quality.get('comment', '')
                print(f"\nAbstraktionsqualitÄt: {quality_score:.2f}/1.0")
                print(f"Kommentar: {quality_comment}")
            
            # Debug-Ausgabe und verbesserte Fortschrittsanzeige
            segment_count = len(result.get('segment_analyses', []))
            
            # ZÄhle Subcodes und ihre Keywords
            subcode_count = 0
            keyword_count = 0
            new_subcodes = []
            
            for analysis in result.get('segment_analyses', []):
                subcodes = analysis.get('subcodes', [])
                subcode_count += len(subcodes)
                
                for subcode in subcodes:
                    new_subcodes.append(subcode)
                    keyword_count += len(subcode.get('keywords', []))
                    
                    # Zeige Abstraktionsbeispiele fuer besseres Monitoring
                    keywords = subcode.get('keywords', [])
                    if keywords and len(keywords) > 0:
                        print(f"\nAbstraktionsbeispiel:")
                        print(f"Keywords: {', '.join(keywords[:3])}" + ("..." if len(keywords) > 3 else ""))
                        print(f"Subcode: {subcode.get('name', '')}")
            
            # Erweiterte Fortschrittsanzeige
            print(f"\nGrounded Analyse fuer {segment_count} Segmente abgeschlossen:")
            print(f"- {subcode_count} neue Subcodes identifiziert")
            print(f"- {keyword_count} Keywords mit Subcodes verknÜpft")
            print(f"- Material-Fortschritt: {material_percentage:.1f}%")
            
            # Progress Bar fuer Gesamtfortschritt der Subcode-Sammlung
            if hasattr(self, 'collected_subcodes'):
                total_collected = len(self.collected_subcodes) + subcode_count
                # Einfache ASCII Progress Bar
                bar_length = 30
                filled_length = int(bar_length * material_percentage / 100)
                bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)
                
                print(f"\nGesamtfortschritt Grounded-Analyse:")
                print(f"[{bar}] {material_percentage:.1f}%")
                print(f"Bisher gesammelt: {total_collected} Subcodes mit ihren Keywords")
            
            return result
            
        except Exception as e:
            print(f"Fehler bei Grounded-Analyse: {str(e)}")
            print("Details:")
            traceback.print_exc()
            return {}
    
    async def _prefilter_segments(self, segments: List[str]) -> List[str]:
        """
        Filtert Segmente nach Relevanz fuer Kategorienentwicklung.
        Optimiert durch Parallelverarbeitung und Caching.
        """
        async def check_segment(segment: str) -> Tuple[str, float]:
            cache_key = hash(segment)
            if cache_key in self.category_cache:
                return segment, self.category_cache[cache_key]
            
            relevance = await self._assess_segment_relevance(segment)
            self.category_cache[cache_key] = relevance
            return segment, relevance
        
        # Parallele RelevanzprÜfung
        tasks = [check_segment(seg) for seg in segments]
        results = await asyncio.gather(*tasks)
        
        # Filter relevante Segmente
        return [seg for seg, relevance in results if relevance > self.MIN_CONFIDENCE]

    async def _assess_segment_relevance(self, segment: str) -> float:
        """
        Bewertet die Relevanz eines Segments fuer die Kategorienentwicklung.
        """
        prompt = self.prompt_handler.get_segment_relevance_assessment_prompt(segment)
                
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
            result = json.loads(llm_response.content)

            
            token_counter.track_response(response, self.model_name)

            return float(result.get('relevance_score', 0))
            
        except Exception as e:
            print(f"Error in relevance assessment: {str(e)}")
            return 0.0
    
    def _create_batches(self, segments: List[str], batch_size: int = None) -> List[List[str]]:
        """
        Creates batches of segments for processing.
        
        Args:
            segments: List of text segments to process
            batch_size: Optional custom batch size (defaults to self.BATCH_SIZE)
            
        Returns:
            List[List[str]]: List of segment batches
        """
        batch_size = self.BATCH_SIZE
            
        return [
            segments[i:i + batch_size] 
            for i in range(0, len(segments), batch_size)
        ]
    
    async def _generate_main_categories_from_subcodes(self, initial_categories: Dict[str, CategoryDefinition] = None) -> Dict[str, CategoryDefinition]:
        """
        Generiert Hauptkategorien aus den gesammelten Subcodes - VOLLSTÄNDIGE GROUNDED THEORY IMPLEMENTIERUNG
        """
        try:
            # Hole gesammelte Subcodes (mehrere Quellen probieren)
            collected_subcodes = []
            
            if hasattr(self, 'collected_subcodes') and self.collected_subcodes:
                collected_subcodes = self.collected_subcodes
                print(f"🧾 Verwende Subcodes aus InductiveCoder: {len(collected_subcodes)}")
            elif hasattr(self, 'analysis_manager') and hasattr(self.analysis_manager, 'collected_subcodes'):
                collected_subcodes = self.analysis_manager.collected_subcodes
                print(f"🧾 Verwende Subcodes aus AnalysisManager: {len(collected_subcodes)}")
            else:
                print("❌ Keine gesammelten Subcodes gefunden - prÜfe verfÜgbare Attribute:")
                for attr in dir(self):
                    if 'subcode' in attr.lower():
                        print(f"   - {attr}: {getattr(self, attr, 'N/A')}")
                return initial_categories or {}
            
            if len(collected_subcodes) < 5:
                print(f"❌ Zu wenige Subcodes fuer Hauptkategorien-Generierung: {len(collected_subcodes)} < 5")
                return initial_categories or {}
            
            print(f"\n🕵️ GROUNDED THEORY: Generiere Hauptkategorien aus {len(collected_subcodes)} Subcodes")
            
            # Bereite Subcodes fuer LLM-Analyse vor
            subcodes_data = []
            all_keywords = []
            
            for subcode in collected_subcodes:
                subcode_entry = {
                    'name': subcode.get('name', ''),
                    'definition': subcode.get('definition', ''),
                    'keywords': subcode.get('keywords', []),
                    'evidence': subcode.get('evidence', []),
                    'confidence': subcode.get('confidence', 0.7)
                }
                subcodes_data.append(subcode_entry)
                all_keywords.extend(subcode.get('keywords', []))
            
            # Zeige Statistiken
            keyword_counter = Counter(all_keywords)
            top_keywords = keyword_counter.most_common(15)
            avg_confidence = sum(s.get('confidence', 0) for s in collected_subcodes) / len(collected_subcodes)
            
            print(f"\n🧾 Subcode-Analyse vor Hauptkategorien-Generierung:")
            print(f"   - Subcodes: {len(subcodes_data)}")
            print(f"   - Einzigartige Keywords: {len(set(all_keywords))}")
            print(f"   - Durchschnittliche Konfidenz: {avg_confidence:.2f}")
            print(f"   - Top Keywords: {', '.join([f'{kw}({count})' for kw, count in top_keywords[:8]])}")
            
            # Erstelle optimierten Prompt fuer Grounded Theory
            enhanced_prompt = self.prompt_handler.get_main_categories_generation_prompt(
                subcodes_data=subcodes_data,
                top_keywords=top_keywords,
                avg_confidence=avg_confidence
            )
                        
            # LLM-Aufruf
            print("\nℹ️ Generiere Hauptkategorien via Grounded Theory Analyse...")
            
            token_counter.start_request()

            response = await self.llm_provider.create_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte fuer Grounded Theory und qualitative Inhaltsanalyse. Du antwortest auf deutsch."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            llm_response = LLMResponse(response)
            result = json.loads(llm_response.content)
            
            
            token_counter.track_response(response, self.model_name)
            
            # Verarbeite Ergebnisse zu CategoryDefinition-Objekten
            grounded_categories = {}
            subcode_mapping = result.get('subcode_mappings', {})
            
            print(f"\n[OK] Hauptkategorien-Generierung abgeschlossen:")
            
            for i, category_data in enumerate(result.get('main_categories', []), 1):
                name = category_data.get('name', '')
                definition = category_data.get('definition', '')
                
                if name and definition:
                    # Erstelle Subcategories aus zugeordneten Subcodes
                    subcategories = {}
                    assigned_subcodes = []
                    
                    for subcode_data in category_data.get('subcodes', []):
                        subcode_name = subcode_data.get('name', '')
                        subcode_definition = subcode_data.get('definition', '')
                        if subcode_name and subcode_definition:
                            subcategories[subcode_name] = subcode_definition
                            assigned_subcodes.append(subcode_name)
                    
                    # Erstelle CategoryDefinition
                    grounded_categories[name] = CategoryDefinition(
                        name=name,
                        definition=definition,
                        examples=category_data.get('examples', []),
                        rules=category_data.get('rules', []),
                        subcategories=subcategories,
                        added_date=datetime.now().strftime("%Y-%m-%d"),
                        modified_date=datetime.now().strftime("%Y-%m-%d")
                    )
                    
                    # Zeige Details
                    characteristic_keywords = ', '.join(category_data.get('characteristic_keywords', [])[:5])
                    print(f"   {i}. 🔀 '{name}': {len(subcategories)} Subcodes zugeordnet")
                    print(f"      Keywords: {characteristic_keywords}")
                    print(f"      Subcodes: {', '.join(assigned_subcodes[:3])}{'...' if len(assigned_subcodes) > 3 else ''}")
            
            # Meta-Analyse Ergebnisse
            meta = result.get('meta_analysis', {})
            if meta:
                print(f"\nℹ️ Grounded Theory Meta-Analyse:")
                print(f"   - Verarbeitete Subcodes: {meta.get('total_subcodes_processed', len(subcodes_data))}")
                print(f"   - Generierte Hauptkategorien: {len(grounded_categories)}")
                print(f"   - Theoretische SÄttigung: {meta.get('theoretical_saturation', 0):.2f}")
                print(f"   - Subcode-Abdeckung: {meta.get('coverage', 0):.2f}")
            
            # PrÜfe Subcode-Zuordnung
            mapped_subcodes = set(subcode_mapping.values()) if subcode_mapping else set()
            all_subcode_names = set(s['name'] for s in subcodes_data)
            unmapped_subcodes = all_subcode_names - mapped_subcodes
            
            if unmapped_subcodes:
                print(f"\n❌ {len(unmapped_subcodes)} Subcodes wurden nicht zugeordnet:")
                for subcode in list(unmapped_subcodes)[:5]:
                    print(f"   - {subcode}")
                if len(unmapped_subcodes) > 5:
                    print(f"   ... und {len(unmapped_subcodes) - 5} weitere")
            else:
                print(f"\n[OK] Alle {len(all_subcode_names)} Subcodes erfolgreich zugeordnet")
            
            # Kombiniere mit initial categories falls vorhanden
            if initial_categories:
                combined_categories = initial_categories.copy()
                for name, category in grounded_categories.items():
                    combined_categories[name] = category
                print(f"\nðŸ”— Kombiniert mit {len(initial_categories)} initialen Kategorien")
                return combined_categories
            
            return grounded_categories
            
        except Exception as e:
            print(f"⚠️ Fehler bei Grounded Theory Hauptkategorien-Generierung: {str(e)}")
            import traceback
            traceback.print_exc()
            return initial_categories or {}
        
    def _create_category_definition(self, cat_data: dict) -> CategoryDefinition:
        """
        Erstellt CategoryDefinition aus API-Response Dictionary
        GRUND: Wird fuer Kategorienentwicklung benÖtigt
        """
        try:
            return CategoryDefinition(
                name=cat_data.get('name', ''),
                definition=cat_data.get('definition', ''),
                examples=cat_data.get('evidence', []),
                rules=[],  # Wird spaeter entwickelt
                subcategories={
                    sub.get('name', ''): sub.get('definition', '')
                    for sub in cat_data.get('subcategories', [])
                },
                added_date=datetime.now().strftime("%Y-%m-%d"),
                modified_date=datetime.now().strftime("%Y-%m-%d")
            )
        except Exception as e:
            print(f"Fehler bei CategoryDefinition-Erstellung: {str(e)}")
            return None

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Berechnet die Ähnlichkeit zwischen zwei Texten mit Caching."""
        cache_key = f"{hash(text1)}_{hash(text2)}"
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
            
        # Konvertiere Texte zu Sets von WÖrtern
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Berechne Jaccard-Ähnlichkeit
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Cache das Ergebnis
        self.similarity_cache[cache_key] = similarity
        
        return similarity

    def _find_similar_category(self, category: CategoryDefinition, existing_categories: Dict[str, CategoryDefinition]) -> Optional[str]:
        """
        Findet Ähnliche Kategorie basierend auf Ähnlichkeitsschwelle
        """
        for existing_name, existing_cat in existing_categories.items():
            similarity = self._calculate_category_similarity(category, existing_cat)
            
            if similarity > self.similarity_threshold:
                print(f"🕵️ Ähnliche Kategorie gefunden: '{category.name}' â†” '{existing_name}' ({similarity:.2f})")
                return existing_name
        
        return None

    def _extract_base_segment_id(self, coding: Dict) -> str:
        """
        Extrahiert die Basis-Segment-ID fuer ReliabilitÄtsberechnung.
        Behandelt Mehrfachkodierung korrekt.
        
        Args:
            coding: Kodierung mit segment_id
            
        Returns:
            str: Basis-Segment-ID ohne Mehrfachkodierungs-Suffixe
        """
        segment_id = coding.get('segment_id', '')
        
        # Entferne Mehrfachkodierungs-Suffixe
        # Format kann sein: "doc_chunk_5" oder "doc_chunk_5-1" fuer Mehrfachkodierung
        if '-' in segment_id:
            # PrÜfe ob es ein Mehrfachkodierungs-Suffix ist (endet mit -Zahl)
            parts = segment_id.rsplit('-', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_id = parts[0]
            else:
                base_id = segment_id
        else:
            base_id = segment_id
        
        return base_id
    
    def _document_reliability_results(
        self, 
        alpha: float, 
        total_segments: int, 
        total_coders: int, 
        category_frequencies: dict
    ) -> str:
        """
        Generiert einen detaillierten Bericht Über die Intercoder-ReliabilitÄt.

        Args:
            alpha: Krippendorffs Alpha Koeffizient
            total_segments: Gesamtzahl der analysierten Segmente
            total_coders: Gesamtzahl der Kodierer
            category_frequencies: HÄufigkeiten der Kategorien
            
        Returns:
            str: Formatierter Bericht als Markdown-Text
        """
        try:
            # Bestimme das ReliabilitÄtsniveau basierend auf Alpha
            reliability_level = (
                "Excellent" if alpha > 0.8 else
                "Acceptable" if alpha > 0.667 else
                "Poor"
            )
            
            # Erstelle den Bericht
            report = [
                "# Intercoder Reliability Analysis Report",
                f"\n## Overview",
                f"- Number of text segments: {total_segments}",
                f"- Number of coders: {total_coders}",
                f"- Krippendorff's Alpha: {alpha:.3f}",
                f"- Reliability Assessment: {reliability_level}",
                "\n## Category Usage",
                "| Category | Frequency |",
                "|----------|-----------|"
            ]
            
            # FÜge KategorienhÄufigkeiten hinzu
            for category, frequency in sorted(category_frequencies.items(), key=lambda x: x[1], reverse=True):
                report.append(f"| {category} | {frequency} |")
            
            # FÜge Empfehlungen hinzu
            report.extend([
                "\n## Recommendations",
                "Based on the reliability analysis, the following actions are suggested:"
            ])
            
            if alpha < 0.667:
                report.extend([
                    "1. Review and clarify category definitions",
                    "2. Provide additional coder training",
                    "3. Consider merging similar categories",
                    "4. Add more explicit coding rules"
                ])
            elif alpha < 0.8:
                report.extend([
                    "1. Review cases of disagreement",
                    "2. Refine coding guidelines for ambiguous cases",
                    "3. Consider additional coder calibration"
                ])
            else:
                report.extend([
                    "1. Continue with current coding approach",
                    "2. Document successful coding practices",
                    "3. Consider using this category system as a template for future analyses"
                ])
            
            # FÜge detaillierte Analyse hinzu
            report.extend([
                "\n## Detailed Analysis",
                "### Interpretation of Krippendorff's Alpha",
                "- > 0.800: Excellent reliability",
                "- 0.667 - 0.800: Acceptable reliability",
                "- < 0.667: Poor reliability",
                "\n### Category Usage Analysis",
                "- Most frequently used category: " + max(category_frequencies.items(), key=lambda x: x[1])[0],
                "- Least frequently used category: " + min(category_frequencies.items(), key=lambda x: x[1])[0],
                "- Number of categories with single use: " + str(sum(1 for x in category_frequencies.values() if x == 1)),
                "\n### Coder Performance",
                "- Average segments per coder: " + f"{total_segments/total_coders:.1f}" if total_coders > 0 else "N/A"
            ])
            
            # FÜge Zeitstempel hinzu
            report.append(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return '\n'.join(report)
            
        except Exception as e:
            print(f"Error generating reliability report: {str(e)}")
            import traceback
            traceback.print_exc()
            return "# Reliability Report\n\nError generating report"
    
  
    async def _meets_quality_standards(self, category: CategoryDefinition) -> bool:
        """
        PrÜft ob Kategorie strikte QualitÄtsstandards erfÜllt
        VEREINFACHT fuer bessere DurchlÄssigkeit
        """
        # 1. Definition ausreichend lang (weiter reduziert)
        if len(category.definition.split()) < 5:  # reduziert von 10
            print(f"⚠️ '{category.name}': Definition zu kurz ({len(category.definition.split())} WÖrter)")
            return False
        
        # 2. GenÜgend Beispiele (weiter reduziert) 
        if len(category.examples) < 1:  # reduziert von 2
            print(f"⚠️ '{category.name}': Zu wenige Beispiele ({len(category.examples)})")
            return False
        
        # 3. Name nicht zu kurz
        if len(category.name) < 3:
            print(f"⚠️ '{category.name}': Name zu kurz")
            return False
        
        print(f"[OK] '{category.name}': QualitÄtsstandards erfÜllt")
        return True

    async def _auto_merge_categories(self, cat1: CategoryDefinition, cat2: CategoryDefinition, name1: str, name2: str) -> Optional[CategoryDefinition]:
        """
        Automatische intelligente ZusammenfÜhrung Ähnlicher Kategorien
        """
        print(f"ðŸ”— Automatische ZusammenfÜhrung: '{name1}' + '{name2}'")
        
        try:
            # WÄhle besseren Namen
            better_name = self._choose_better_name(name1, name2)
            
            # Kombiniere Definitionen intelligent
            combined_definition = await self._merge_definitions_intelligent(cat1.definition, cat2.definition)
            
            # Kombiniere Beispiele (entferne Duplikate)
            combined_examples = list(set(cat1.examples + cat2.examples))
            
            # Kombiniere Regeln
            combined_rules = list(set(cat1.rules + cat2.rules))
            
            # Kombiniere Subkategorien
            combined_subcats = {**cat1.subcategories, **cat2.subcategories}
            
            # Erstelle zusammengefÜhrte Kategorie
            merged = CategoryDefinition(
                name=better_name,
                definition=combined_definition,
                examples=combined_examples,
                rules=combined_rules,
                subcategories=combined_subcats,
                added_date=min(cat1.added_date, cat2.added_date),
                modified_date=datetime.now().strftime("%Y-%m-%d")
            )
            
            print(f"[OK] ZusammenfÜhrung erfolgreich zu '{better_name}'")
            return merged
            
        except Exception as e:
            print(f"⚠️ Fehler bei automatischer ZusammenfÜhrung: {str(e)}")
            return None

    async def _merge_definitions_intelligent(self, def1: str, def2: str) -> str:
        """
        Intelligente ZusammenfÜhrung von Definitionen via LLM
        """
        prompt = self.prompt_handler.get_definition_enhancement_prompt({
            'definition1': def1,
            'definition2': def2
        })
                
        try:
            response = await self.llm_provider.create_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte fuer qualitative Inhaltsanalyse."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            merged_def = response.choices[0].message.content.strip()
            
            # Fallback falls LLM-Merge fehlschlÄgt
            if len(merged_def.split()) < 15:
                return f"{def1} ZusÄtzlich umfasst dies: {def2}"
            
            return merged_def
            
        except Exception as e:
            print(f"Fehler bei Definition-Merge: {str(e)}")
            return f"{def1} Erweitert um: {def2}"

    def _calculate_category_similarity(self, cat1: CategoryDefinition, cat2: CategoryDefinition) -> float:
        """
        Berechnet Ähnlichkeit zwischen zwei Kategorien basierend auf mehreren Faktoren
        """
        # 1. Name-Ähnlichkeit (30%)
        name_similarity = self._calculate_text_similarity(cat1.name.lower(), cat2.name.lower()) * 0.3
        
        # 2. Definition-Ähnlichkeit (50%)
        def_similarity = self._calculate_text_similarity(cat1.definition, cat2.definition) * 0.5
        
        # 3. Subkategorien-Überlappung (20%)
        subcats1 = set(cat1.subcategories.keys())
        subcats2 = set(cat2.subcategories.keys())
        
        if subcats1 and subcats2:
            subcat_overlap = len(subcats1 & subcats2) / len(subcats1 | subcats2)
        else:
            subcat_overlap = 0
        
        subcat_similarity = subcat_overlap * 0.2
        
        total_similarity = name_similarity + def_similarity + subcat_similarity
        
        return min(total_similarity, 1.0)

    async def _merge_categories_intelligent(self, cat1: CategoryDefinition, cat2: CategoryDefinition, name1: str, name2: str) -> Optional[CategoryDefinition]:
        """
        Intelligente ZusammenfÜhrung mit QualitÄtsprÜfung
        """
        # Verwende die bereits implementierte _auto_merge_categories
        merged = await self._auto_merge_categories(cat1, cat2, name1, name2)
        
        if merged and await self._meets_quality_standards(merged):
            return merged
        
        print(f"⚠️ ZusammengefÜhrte Kategorie erfÜllt QualitÄtsstandards nicht")
        return None

    def _choose_better_name(self, name1: str, name2: str) -> str:
        """
        WÄhlt den besseren Kategorienamen basierend auf Kriterien
        """
        # Kriterien fuer besseren Namen
        score1 = score2 = 0
        
        # 1. LÄnge (nicht zu kurz, nicht zu lang)
        if 5 <= len(name1) <= 25:
            score1 += 1
        if 5 <= len(name2) <= 25:
            score2 += 1
        
        # 2. Keine Sonderzeichen/Zahlen
        if name1.replace('_', '').replace('-', '').isalpha():
            score1 += 1
        if name2.replace('_', '').replace('-', '').isalpha():
            score2 += 1
        
        # 3. Keine englischen WÖrter
        english_words = {'research', 'development', 'management', 'system', 'process', 'analysis'}
        if not any(word.lower() in english_words for word in name1.split('_')):
            score1 += 1
        if not any(word.lower() in english_words for word in name2.split('_')):
            score2 += 1
        
        # 4. KÜrzerer Name bei Gleichstand
        if score1 == score2:
            return name1 if len(name1) <= len(name2) else name2
        
        return name1 if score1 > score2 else name2

    def _update_usage_history(self, category_names: List[str]) -> None:
        """
        Aktualisiert die Nutzungshistorie fuer Kategorien
        """
        for name in category_names:
            if name in self.category_usage_history:
                self.category_usage_history[name] += 1
            else:
                self.category_usage_history[name] = 1
        
        print(f"🧾 Nutzungshistorie aktualisiert fuer: {category_names}")
        print(f"    Aktuelle Nutzung: {dict(list(self.category_usage_history.items())[-3:])}")
    
    def _log_performance(self, 
                        num_segments: int,
                        num_categories: int,
                        processing_time: float) -> None:
        """
        Protokolliert Performance-Metriken.
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'segments_processed': num_segments,
            'categories_developed': num_categories,
            'processing_time': processing_time,
            'segments_per_second': num_segments / processing_time
        }
        
        print("\nPerformance Metrics:")
        print(f"- Segments processed: {num_segments}")
        print(f"- Categories developed: {num_categories}")
        print(f"- Processing time: {processing_time:.2f}s")
        print(f"- Segments/second: {metrics['segments_per_second']:.2f}")
        
        # Speichere Metriken
        self.batch_results.append(metrics)
    def _find_similar_category(self, category: CategoryDefinition, existing_categories: Dict[str, CategoryDefinition]) -> Optional[str]:
        """
        Findet Ähnliche existierende Kategorien basierend auf Namen und Definition.
        """
        try:
            best_match = None
            highest_similarity = 0.0
            
            for existing_name, existing_cat in existing_categories.items():
                # Berechne Ähnlichkeit basierend auf verschiedenen Faktoren
                
                # 1. Name-Ähnlichkeit (gewichtet: 0.3)
                name_similarity = self._calculate_text_similarity(
                    category.name.lower(),
                    existing_name.lower()
                ) * 0.3
                
                # 2. Definitions-Ähnlichkeit (gewichtet: 0.5)
                definition_similarity = self._calculate_text_similarity(
                    category.definition,
                    existing_cat.definition
                ) * 0.5
                
                # 3. Subkategorien-Überlappung (gewichtet: 0.2)
                subcats1 = set(category.subcategories.keys())
                subcats2 = set(existing_cat.subcategories.keys())
                if subcats1 and subcats2:
                    subcat_overlap = len(subcats1 & subcats2) / len(subcats1 | subcats2)
                else:
                    subcat_overlap = 0
                subcat_similarity = subcat_overlap * 0.2
                
                # GesamtÄhnlichkeit
                total_similarity = name_similarity + definition_similarity + subcat_similarity
                
                # Update beste Übereinstimmung
                if total_similarity > highest_similarity:
                    highest_similarity = total_similarity
                    best_match = existing_name
            
            # Nur zurÜckgeben wenn Ähnlichkeit hoch genug
            if highest_similarity > 0.7:  # Schwellenwert fuer Ähnlichkeit
                print(f"\nâš  Hohe Ähnlichkeit ({highest_similarity:.2f}) gefunden:")
                print(f"- Neue Kategorie: {category.name}")
                print(f"- Existierende Kategorie: {best_match}")
                return best_match
                
            return None
            
        except Exception as e:
            print(f"Fehler bei ÄhnlichkeitsprÜfung: {str(e)}")
            return None


# --- Klasse: ManualCoder ---
