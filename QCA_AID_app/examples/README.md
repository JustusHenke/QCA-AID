# QCA-AID Webapp - Beispielkonfigurationen

Dieses Verzeichnis enthält Beispielkonfigurationen für verschiedene Anwendungsfälle.

## verfügbare Beispiele

### 1. Minimal-Konfiguration
**Datei:** `config-minimal.json`

Eine minimale Konfiguration für schnelle Tests und erste Schritte.

**Verwendung:**
- Kleine Datensätze (< 5 Dokumente)
- Schnelle Prototypen
- Deduktive Analyse

### 2. Standard-Konfiguration
**Datei:** `config-standard.json`

Empfohlene Standardkonfiguration für die meisten Anwendungsfälle.

**Verwendung:**
- Mittlere Datensätze (5-20 Dokumente)
- Ausgewogene Performance und Qualität
- Abduktive Analyse

### 3. Hochpräzisions-Konfiguration
**Datei:** `config-precision.json`

Optimiert für maximale Kodierqualität.

**Verwendung:**
- Wichtige Analysen
- Publikationsreife Ergebnisse
- Mehrere Coder für Intercoder-Reliabilität

### 4. Performance-Konfiguration
**Datei:** `config-performance.json`

Optimiert für schnelle Verarbeitung großer Datenmengen.

**Verwendung:**
- Große Datensätze (> 50 Dokumente)
- Explorative Analysen
- Zeitkritische Projekte

### 5. Grounded Theory Konfiguration
**Datei:** `config-grounded.json`

Speziell für Grounded Theory Ansätze.

**Verwendung:**
- Induktive Kategorienentwicklung
- Theoriebildung aus Daten
- Explorative Forschung

## Verwendung

### Konfiguration laden

1. **In der Webapp:**
   ```
   1. Öffnen Sie den Config-Tab
   2. Klicken Sie auf "Konfiguration laden"
   3. Wählen Sie eine Beispieldatei aus examples/
   4. Passen Sie die Einstellungen nach Bedarf an
   5. Speichern Sie als Ihre eigene Konfiguration
   ```

2. **Kommandozeile:**
   ```bash
   cp examples/config-standard.json QCA-AID-Codebook.json
   python QCA-AID.py
   ```

### Anpassung

Alle Beispielkonfigurationen sind Ausgangspunkte. Passen Sie sie an Ihre Bedürfnisse an:

- **Modell**: Wählen Sie das passende LLM-Modell
- **Chunk-Größe**: Abhängig von Ihrer Dokumentlänge
- **Batch-Größe**: Abhängig von Ihrer Hardware und Geschwindigkeitsanforderungen
- **Analysemodus**: Abhängig von Ihrer Forschungsfrage

## Vergleich der Konfigurationen

| Aspekt | Minimal | Standard | Präzision | Performance | Grounded |
|--------|---------|----------|-----------|-------------|----------|
| **Chunk Size** | 800 | 1000 | 1200 | 1500 | 1000 |
| **Batch Size** | 3 | 5 | 3 | 12 | 5 |
| **Coder** | 1 | 1 | 3 | 1 | 1 |
| **Context** | Nein | Ja | Ja | Nein | Ja |
| **Geschwindigkeit** | Mittel | Gut | Langsam | Sehr schnell | Mittel |
| **Qualität** | Gut | Sehr gut | Exzellent | Akzeptabel | Sehr gut |
| **Kosten** | Niedrig | Mittel | Hoch | Niedrig | Mittel |

## Weitere Informationen

- [Webapp Benutzerhandbuch](../WEBAPP_README.md)
- [Hauptdokumentation](../README.md)
- [Fehlerbehebung](../WEBAPP_TROUBLESHOOTING.md)
