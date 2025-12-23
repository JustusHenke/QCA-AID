# UnifiedAnalysisManager Implementation für Deductive Mode

**Erstellungsdatum**: 2025-01-XX  
**Status**: Implementiert  
**Betroffene Komponenten**: OptimizationController, UnifiedRelevanceAnalyzer, IntegratedAnalysisManager

---

## Übersicht

Der UnifiedAnalysisManager (OptimizationController) wurde für den Deductive Mode aktiviert und mit allen erforderlichen Features ausgestattet.

## Implementierte Features

### 1. ✅ API-Call für Relevanzprüfung und Kategoriepräferenzen

**Implementierung**: `UnifiedRelevanceAnalyzer.analyze_relevance_with_preferences()`

- Führt eine Batch-Relevanzprüfung für alle Segmente durch
- Bestimmt Kategoriepräferenzen für jedes Segment
- Identifiziert Top-Kategorien pro Segment
- Reduziert API-Calls durch Batch-Verarbeitung

**Verwendung**:
```python
relevance_results = await unified_analyzer.analyze_relevance_with_preferences(
    segments=segments,
    category_definitions=cat_defs,
    research_question=research_question,
    coding_rules=rules
)
```

### 2. ✅ Separate API-Calls für jeden Autocoder mit Temperature

**Implementierung**: `OptimizationController.analyze_segments()` mit `temperature` Parameter

- Jeder konfigurierte Autocoder erhält einen eigenen API-Call
- Temperature-Parameter wird pro Kodierer individuell übergeben
- Unterstützt unterschiedliche Temperature-Einstellungen pro Kodierer

**Verwendung**:
```python
for coder in self.deductive_coders:
    coder_results = await self.optimization_controller.analyze_segments(
        segments=relevant_segments,
        analysis_mode=AnalysisMode.DEDUCTIVE,
        category_definitions=cat_defs,
        research_question=research_question,
        coding_rules=rules,
        batch_size=batch_size,
        temperature=coder.temperature  # Individuelle Temperature pro Kodierer
    )
```

### 3. ✅ Token-Tracker Integration

**Implementierung**: Vollständig integriert in `UnifiedRelevanceAnalyzer`

- Token-Tracking bei jedem API-Call
- Kostenberechnung pro Request
- Session-Statistiken verfügbar
- Automatische Token-Zählung via `get_global_token_counter()`

**Anzeige**:
```python
session_stats = token_counter.get_session_stats()
print(f"   💰 Token-Verbrauch: {session_stats.get('input', 0) + session_stats.get('output', 0)} Tokens")
print(f"   💵 Kosten: ${session_stats.get('cost', 0.0):.4f}")
```

### 4. ✅ Effizienz- und Fortschrittsanzeige

**Implementierung**: In `IntegratedAnalysisManager._analyze_normal_modes()`

- Fortschrittsanzeige während der Kodierung
- Effizienz-Statistiken nach Abschluss:
  - API-Calls gesamt
  - Tokens gesamt
  - Kosten gesamt
  - Calls/Segment
  - Tokens/Segment

**Anzeige**:
```
📊 EFFIZIENZ-STATISTIKEN:
   • API-Calls: 15
   • Tokens: 45,230
   • Kosten: $0.1234
   • Calls/Segment: 0.30
   • Tokens/Segment: 904
```

### 5. ✅ Kompatibilität mit Manual Coder

**Status**: Kompatibel

- Manual Coder arbeitet unabhängig vom OptimizationController
- Manual Codings werden nach der automatischen Analyse hinzugefügt
- Keine Konflikte erwartet, da Manual Coder separate Kodierungen erstellt
- Manual Coder verwendet eigene GUI und eigene Kodierungslogik

**Hinweis**: Manual Coder wird vor der automatischen Analyse ausgeführt (siehe `main.py`), daher keine direkte Interaktion mit dem OptimizationController.

### 6. ✅ Kompatibilität mit anderen Modi

**Status**: Vorbereitet, aber aktuell nur für Deductive Mode aktiviert

**Unterstützte Modi im OptimizationController**:
- ✅ **Deductive**: Vollständig implementiert und aktiviert
- ✅ **Inductive**: Implementiert via `_analyze_inductive()`
- ✅ **Abductive**: Implementiert via `_analyze_abductive()`
- ✅ **Grounded**: Implementiert via `_analyze_grounded()`

**Aktivierung für andere Modi**:
Um andere Modi zu aktivieren, muss in `analysis_manager.py` die Bedingung erweitert werden:

```python
# Aktuell (nur deductive):
if self.optimization_enabled and self.optimization_controller and analysis_mode == 'deductive':

# Für alle Modi:
if self.optimization_enabled and self.optimization_controller:
    mode_mapping = {
        'deductive': AnalysisMode.DEDUCTIVE,
        'inductive': AnalysisMode.INDUCTIVE,
        'abductive': AnalysisMode.ABDUCTIVE,
        'grounded': AnalysisMode.GROUNDED
    }
    if analysis_mode in mode_mapping:
        # Verwende OptimizationController
```

**Hinweis**: Jeder Modus hat spezifische Workflows, die im OptimizationController bereits implementiert sind, aber noch nicht im AnalysisManager integriert wurden.

## Workflow im Deductive Mode

1. **Relevanzprüfung** (einmalig für alle Segmente)
   - API-Call: `analyze_relevance_with_preferences()`
   - Filtert relevante Segmente (Threshold: 0.3)
   - Bestimmt Kategoriepräferenzen

2. **Kodierung** (pro Kodierer)
   - Für jeden konfigurierten Autocoder:
     - API-Call: `analyze_segments()` mit individueller Temperature
     - Batch-Verarbeitung für Effizienz
     - Token-Tracking automatisch

3. **Statistiken**
   - Effizienz-Metriken werden angezeigt
   - Token- und Kosten-Statistiken verfügbar

## Konfiguration

**Aktivierung**:
```python
CONFIG['ENABLE_OPTIMIZATION'] = True  # Standard: True
```

**Deaktivierung**:
```python
CONFIG['ENABLE_OPTIMIZATION'] = False  # Fällt zurück auf Standard-Analyse
```

## Vorteile

1. **Reduzierte API-Calls**: 
   - Vorher: ~2.2 Calls/Segment
   - Nachher: ~0.3-0.5 Calls/Segment (mit Batching)

2. **Bessere Effizienz**:
   - Batch-Verarbeitung reduziert Overhead
   - Caching reduziert redundante Calls

3. **Transparenz**:
   - Detaillierte Statistiken
   - Fortschrittsanzeige
   - Token- und Kosten-Tracking

4. **Flexibilität**:
   - Individuelle Temperature pro Kodierer
   - Unterstützung für alle Modi (vorbereitet)

## Bekannte Einschränkungen

1. **Andere Modi**: Noch nicht aktiviert, aber vorbereitet
2. **Manual Coder**: Keine direkte Integration, aber kompatibel
3. **Caching**: Cache wird zu Beginn geleert für frische Analyse

## Nächste Schritte (Optional)

1. Aktivierung für andere Modi (inductive, abductive, grounded)
2. Erweiterte Caching-Strategien
3. Parallele Verarbeitung mehrerer Kodierer
4. Erweiterte Fortschrittsanzeige mit ETA



