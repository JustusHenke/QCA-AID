# LLM Provider Manager - Beispiele

Dieses Verzeichnis enthält praktische Beispiele für die Verwendung des LLM Provider Managers.

## verfügbare Beispiele

### 1. basic_usage.py
**Basis-Nutzung des Provider Managers**

Zeigt die grundlegenden Funktionen:
- Manager initialisieren
- Alle Modelle abrufen
- Spezifische Modelle suchen
- Provider auflisten
- Modell-Informationen anzeigen
- Statistiken generieren

```bash
python basic_usage.py
```

**Was Sie lernen:**
- Wie man den Manager initialisiert
- Wie man Modelle abruft und durchsucht
- Wie man Modell-Details anzeigt
- Wie man Provider-Statistiken erstellt

---

### 2. filter_usage.py
**Filter-Funktionen**

Demonstriert verschiedene Filter-Möglichkeiten:
- Nach Provider filtern
- Nach Kosten filtern
- Nach Context Window filtern
- Nach Capabilities filtern
- Kombinierte Filter
- Benutzerdefinierte Filter-Logik

```bash
python filter_usage.py
```

**Was Sie lernen:**
- Wie man Modelle nach verschiedenen Kriterien filtert
- Wie man mehrere Filter kombiniert
- Wie man benutzerdefinierte Filter-Logik implementiert
- Wie man Modelle nach Preis-Leistung sortiert

---

### 3. pricing_overrides.py
**Pricing-Overrides**

Zeigt wie man eigene Preisinformationen definiert:
- Standard-Preise anzeigen
- Overrides erstellen und anwenden
- Preise vergleichen (vor/nach Override)
- Ungültige Overrides behandeln
- Template erstellen

```bash
python pricing_overrides.py
```

**Was Sie lernen:**
- Wie man `pricing_overrides.json` erstellt
- Wie man Overrides anwendet
- Wie das System mit ungültigen Overrides umgeht
- Wie man Kostenersparnisse berechnet

**Erstellt:**
- `pricing_overrides_template.json` - Vorlage für eigene Overrides

---

### 4. add_new_provider.py
**Neue Provider hinzuFügen**

Demonstriert das HinzuFügen neuer Provider:
- Via URL (Catwalk-Format)
- Via lokale JSON-Datei
- Mehrere Provider gleichzeitig
- Cache-Integration
- Config-Template erstellen

```bash
python add_new_provider.py
```

**Was Sie lernen:**
- Wie man neue Provider via URL hinzufügt
- Wie man neue Provider via lokale Datei hinzufügt
- Wie man Provider-Configs erstellt
- Wie neue Provider in den Cache integriert werden
- Welche Felder in einer Provider-Config erforderlich sind

**Erstellt:**
- `custom_provider_template.json` - Vorlage für eigene Provider-Configs

---

## Schnellstart

### Voraussetzungen

```bash
# Stelle sicher dass QCA-AID installiert ist
pip install aiohttp
```

### Alle Beispiele ausführen

```bash
# Basis-Nutzung
python QCA_AID_assets/utils/llm/examples/basic_usage.py

# Filter-Nutzung
python QCA_AID_assets/utils/llm/examples/filter_usage.py

# Pricing-Overrides
python QCA_AID_assets/utils/llm/examples/pricing_overrides.py

# Neue Provider hinzuFügen
python QCA_AID_assets/utils/llm/examples/add_new_provider.py
```

### Einzelne Funktionen testen

Sie können die Beispiele auch als Module importieren:

```python
import asyncio
from QCA_AID_assets.utils.llm.examples.basic_usage import basic_usage_example

asyncio.run(basic_usage_example())
```

## Beispiel-Output

### basic_usage.py

```
============================================================
LLM Provider Manager - Basis-Nutzung
============================================================

1. Manager initialisieren...
✓ Manager erfolgreich initialisiert

2. Alle verfügbaren Modelle abrufen...
✓ Gefunden: 45 Modelle

3. Unterstützte Provider:
   - openai: 15 Modelle
   - anthropic: 8 Modelle
   - mistral: 5 Modelle
   - openrouter: 15 Modelle
   - local: 2 Modelle

4. Spezifisches Modell suchen (gpt-4o-mini)...
✓ Modell gefunden:
   Name: GPT-4o Mini
   Provider: openai
   Context Window: 128,000 Tokens
   Kosten Input: $0.15/1M Tokens
   Kosten Output: $0.60/1M Tokens
   Zusätzliche Eigenschaften:
      ✓ Unterstützt Reasoning
      ✓ Unterstützt Attachments
```

### filter_usage.py

```
============================================================
1. Nach Provider filtern
============================================================

OPENAI: 15 Modelle
   - GPT-4o Mini
     ID: gpt-4o-mini
     Kosten: $0.15/$0.60 | Context: 128,000
   - GPT-4o
     ID: gpt-4o
     Kosten: $2.50/$10.00 | Context: 128,000
   ...

============================================================
2. Nach Kosten filtern
============================================================

Günstige Modelle (max $1.00 Input, max $5.00 Output):
Gefunden: 8 Modelle
   - GPT-4o Mini (openai)
     $0.15 in / $0.60 out
   - Claude 3 Haiku (anthropic)
     $0.25 in / $1.25 out
   ...
```

## Tipps und Tricks

### Logging aktivieren

Für detaillierte Ausgaben aktivieren Sie Logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,  # oder INFO, WARNING
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Cache-Verzeichnis anpassen

```python
manager = LLMProviderManager(
    cache_dir="/custom/cache/path"
)
```

### Fehlerbehandlung

```python
try:
    await manager.initialize()
except Exception as e:
    print(f"Fehler: {e}")
    # Fallback-Logik
```

### Performance

- Der erste Aufruf lädt Daten von externen Quellen (langsam)
- Nachfolgende Aufrufe nutzen den Cache (schnell, 24h gültig)
- Verwenden Sie `force_refresh=True` nur wenn nötig

## Häufige Probleme

### Problem: "No module named 'QCA_AID_assets'"

**Lösung:** Führen Sie die Beispiele aus dem Projekt-Root-Verzeichnis aus:

```bash
cd /path/to/qca-aid
python QCA_AID_assets/utils/llm/examples/basic_usage.py
```

### Problem: Keine Modelle werden geladen

**Lösung:**
1. Prüfen Sie Ihre Internetverbindung
2. Stellen Sie sicher dass Fallback-Configs existieren
3. Aktivieren Sie Debug-Logging

### Problem: Lokale Modelle werden nicht erkannt

**Lösung:**
1. Starten Sie LM Studio oder Ollama
2. Prüfen Sie ob die Server laufen:
   - LM Studio: `http://localhost:1234`
   - Ollama: `http://localhost:11434`

## Weitere Ressourcen

- **Haupt-README**: `../README.md` - Vollständige Dokumentation
- **Design-Dokument**: `.kiro/specs/llm-provider-manager/design.md`
- **Requirements**: `.kiro/specs/llm-provider-manager/requirements.md`

## Feedback

Bei Fragen oder Problemen:
1. Prüfen Sie die Logs
2. Lesen Sie die Haupt-Dokumentation
3. Erstellen Sie ein Issue im Repository
