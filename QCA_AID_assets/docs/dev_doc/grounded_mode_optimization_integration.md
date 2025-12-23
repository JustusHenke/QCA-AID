# Grounded Mode Integration in OptimizationController

**Erstellungsdatum**: 2025-01-XX  
**Status**: Implementiert

---

## Grounded Mode Workflow (3 Phasen)

### Phase 1: Subcode-Sammlung (wiederholt für alle Batches)

**Workflow pro Batch:**
1. Relevanzprüfung → filtert relevante Segmente
2. Subcode-Extraktion → extrahiert Subcodes und Keywords
3. Zentrale Sammlung → sammelt Subcodes stateful über Batches hinweg
4. Sättigungsprüfung → prüft ob genug Subcodes gesammelt wurden

**API-Calls pro Batch:**
- 1 Call: Relevanzprüfung (Batch)
- 1 Call: Subcode-Extraktion (Batch)
- **Total: 2 Calls pro Batch**

### Phase 2: Hauptkategorien-Generierung (einmalig am Ende)

**Workflow:**
1. Analysiert gesammelte Subcodes
2. Gruppiert verwandte Subcodes zu Hauptkategorien
3. Generiert CategoryDefinitions mit Subkategorien

**API-Calls:**
- 1 Call: Hauptkategorien-Generierung
- **Total: 1 Call**

### Phase 3: Kodierung (einmalig am Ende)

**Workflow:**
1. Kodiert alle Segmente mit generierten Kategorien
2. Pro Kodierer separate API-Calls (wie im Deductive Mode)

**API-Calls:**
- M Calls: Kodierung (M = Anzahl Kodierer, Batch-Verarbeitung)
- **Total: M Calls**

---

## Implementierung im OptimizationController

### Neue Methoden

1. **`_analyze_grounded()`** - Phase 1: Subcode-Sammlung
   - Relevanzprüfung
   - Subcode-Extraktion
   - Zentrale Sammlung (stateful)

2. **`generate_grounded_main_categories()`** - Phase 2: Hauptkategorien-Generierung
   - Analysiert gesammelte Subcodes
   - Generiert Hauptkategorien via LLM
   - Konvertiert zu CategoryDefinitions

3. **`code_with_grounded_categories()`** - Phase 3: Kodierung
   - Kodiert alle Segmente mit generierten Kategorien
   - Pro Kodierer separate Calls

4. **`get_grounded_subcodes()`** - Helper: Gibt gesammelte Subcodes zurück

5. **`reset_grounded_state()`** - Helper: Setzt State zurück

### State-Management

Der OptimizationController speichert jetzt:
- `grounded_subcodes_collection`: Liste aller gesammelten Subcodes
- `grounded_segment_analyses`: Alle Segment-Analysen
- `grounded_keywords_collection`: Alle Keywords

---

## Integration im AnalysisManager

### Beispiel-Integration

```python
# In AnalysisManager._analyze_grounded_mode

# PHASE 1: Subcode-Sammlung (wiederholt für alle Batches)
while True:
    batch = await self._get_next_batch(all_segments, batch_size)
    if not batch:
        break
    
    # Verwende OptimizationController für Subcode-Sammlung
    opt_segments = [{'segment_id': s[0], 'text': s[1]} for s in batch]
    
    subcode_results = await self.optimization_controller.analyze_segments(
        segments=opt_segments,
        analysis_mode=AnalysisMode.GROUNDED,
        research_question=FORSCHUNGSFRAGE
    )
    
    # Sättigungsprüfung
    if await self._assess_grounded_saturation(...):
        break

# PHASE 2: Hauptkategorien-Generierung
grounded_categories = await self.optimization_controller.generate_grounded_main_categories(
    research_question=FORSCHUNGSFRAGE,
    initial_categories=initial_categories
)

# PHASE 3: Kodierung
if grounded_categories:
    all_opt_segments = [{'segment_id': s[0], 'text': s[1]} for s in all_segments]
    
    coding_results = await self.optimization_controller.code_with_grounded_categories(
        all_segments=all_opt_segments,
        grounded_categories=grounded_categories,
        research_question=FORSCHUNGSFRAGE,
        coding_rules=rules,
        coder_settings=config['CODER_SETTINGS']
    )
```

---

## API-Call Vergleich für Grounded Mode

### Alter Workflow (10 Chunks, 2 Batches)

**Phase 1 (pro Batch):**
- Relevanzprüfung: 1 Call
- Subcode-Extraktion: 1 Call
- **Total Phase 1**: 2 Calls × 2 Batches = **4 Calls**

**Phase 2:**
- Hauptkategorien-Generierung: 1 Call
- **Total Phase 2**: **1 Call**

**Phase 3:**
- Kodierung: 6 relevante Segmente × 2 Kodierer = 12 Calls
- **Total Phase 3**: **12 Calls**

**Gesamt**: 4 + 1 + 12 = **17 Calls**

### Neuer OptimizationController (10 Chunks, 2 Batches)

**Phase 1 (pro Batch):**
- Relevanzprüfung: 1 Call (Batch)
- Subcode-Extraktion: 1 Call (Batch)
- **Total Phase 1**: 2 Calls × 2 Batches = **4 Calls**

**Phase 2:**
- Hauptkategorien-Generierung: 1 Call
- **Total Phase 2**: **1 Call**

**Phase 3:**
- Kodierung: 2 Kodierer × 1 Call (Batch) = 2 Calls
- **Total Phase 3**: **2 Calls**

**Gesamt**: 4 + 1 + 2 = **7 Calls**

**Reduktion**: 17 → 7 = **-59%** (10 Calls gespart)

**Calls pro Segment**: 17/10 = 1.7 → 7/10 = **0.7 Calls/Segment**

---

## Vorteile

1. **Batch-Optimierung**: Kodierung erfolgt im Batch statt segmentweise
2. **State-Management**: Subcodes werden zentral gesammelt
3. **Konsistente API**: Gleiche Methoden-Struktur wie andere Modi
4. **Effizienz**: Reduktion um ~59% bei Grounded Mode

---

## Nächste Schritte

1. Integration im AnalysisManager implementieren
2. Sättigungsprüfung integrieren
3. Testing und Validierung



