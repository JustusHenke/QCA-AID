# Changelog

## Versionen und Updates

---

## Neu in 0.12.7 (2026-04-22)

### 🔧 Verbesserungen

- **Vereinheitlichtes Laden von Config und Codebook:**
  - Codebook wird jetzt automatisch beim Laden der Konfiguration mitgeladen (Config UI)
  - Separater "Codebook laden"-Dialog im Codebook-Tab entfernt (Redundanz beseitigt)
  - Gespeicherter Dateipfad wird für zukünftige Speicher-Operationen korrekt weitergegeben

- **Robustere Pfadauflösung:**
  - Relative Pfade werden in ConfigManager und CodebookManager korrekt gegen das Projektverzeichnis aufgelöst
  - Verhindert Fehler bei Pfadangaben ohne absoluten Pfad

- **Robustere Codebook-Datenverarbeitung:**
  - `CategoryData.from_dict()` behandelt None-Werte und unerwartete Typen graceful
  - `CodebookData.from_dict()` mit robuster Kodierregeln-Verarbeitung
  - Validierung gibt success=True zurück wenn Kategorien trotz Warnungen nutzbar sind

- **Session-State-basiertes Codebook-Management:**
  - Analyse-Tab und Webapp nutzen Session State statt wiederholtes Laden von Festplatte
  - Verbesserte Statusanzeige im Codebook-Tab (unterscheidet "geladen" vs. "keine Kategorien")

### 🔧 Aktualisierte LLM-Modellkonfigurationen

- **Anthropic:** Claude Opus 4.7 hinzugefügt (mit Reasoning-Levels)
- **OpenRouter:** 
  - Neue Modelle: Claude Opus 4.7, Kimi K2.6, inclusionAI Ling-2.6-flash (free)
  - Entfernte veraltete Modelle: Mercury, Mercury Coder, Llama 4 Maverick, GPT-4o extended, GPT-5 Image/Image Mini, Meituan LongCat Flash
  - Aktualisierte Preise und Parameter für zahlreiche Modelle (DeepSeek, Qwen, MiniMax, Mistral, xAI, Z.ai u.a.)

---

## Neu in 0.12.6.3 (2026-04-18)

### 🐛 Bugfixes / Robustheit

- **Robustere JSON-Reparatur bei abgeschnittenen LLM-Antworten:**
  - Neue `_close_brackets()`-Methode mit korrekter Stack-basierter Klammeranalyse (berücksichtigt Strings)
  - Erkennung und Schließung unterminated Strings
  - Entfernung von trailing Commas vor schließenden Klammern
  - Fallback-Strategie: Zeichenweises Abschneiden bis valides JSON entsteht
  - Ersetzt die bisherige naive Klammerzählung

- **Retry-Logik für Batch-Analyse bei JSON-Fehlern:**
  - Bis zu 2 Wiederholungsversuche bei `JSONDecodeError` in der Batch-Kodierung
  - Graceful Skip: Bei persistentem Fehler wird der Batch übersprungen statt die gesamte Analyse abzubrechen
  - Nicht-JSON-Fehler werden weiterhin sofort geworfen

### 🔧 Verbesserungen

- **Aktualisierte LLM-Modellkonfigurationen:**
  - OpenAI: Neue Modelle (GPT-5.4, GPT-5.4 Pro, GPT-5.4 Nano) hinzugefügt
  - Anthropic: Aktualisierte Modellpreise und -konfigurationen
  - OpenRouter: Erweiterte Modellliste mit aktuellen Preisen

---

## Neu in 0.12.6 (2026-03-05)

### 🌐 Custom Provider Integration

**Custom API Base URL Support für OpenAI-kompatible Endpoints:**
- ✨ **GWDG Academic Cloud Integration**: Vollständige Unterstützung für institutionelle OpenAI-kompatible APIs
  - Neues optionales Konfigurationsfeld `api_base_url` in ConfigData
  - UI-Integration im Konfigurationsreiter mit Validierung und Beispielen
  - Automatische Übergabe der Base URL an alle Analyse-Module
  - Beispiel: `https://chat-ai.academiccloud.de/v1`

- ✨ **Erweiterte Provider-Unterstützung**:
  - Azure OpenAI: `https://your-resource.openai.azure.com/openai/deployments/your-deployment`
  - Lokale OpenAI-kompatible Server (LM Studio, Ollama, Text Generation WebUI)
  - Beliebige OpenAI-kompatible Endpoints

- ✨ **Backend-Implementierung**:
  - `OpenAIProvider` unterstützt jetzt `base_url` Parameter im Constructor
  - `LLMProviderFactory.create_provider()` akzeptiert `base_url` Parameter
  - Alle Analyse-Module übergeben `base_url` an Provider (deductive_coding, inductive_coding, relevance_checker, explorer)
  - CONFIG Dictionary erweitert um `API_BASE_URL` Feld

- ✨ **UI-Features**:
  - Neuer Expander "🔧 Erweiterte Einstellungen: Custom API Base URL"
  - Eingabefeld mit Echtzeit-Validierung (muss mit http:// oder https:// beginnen)
  - Detaillierte Anleitung speziell für GWDG Academic Cloud
  - Beispielkonfigurationen für verschiedene Anwendungsfälle
  - Wird nur bei OpenAI und Local Providern angezeigt

- 📚 **Dokumentation**:
  - `CUSTOM_PROVIDER_GUIDE.md`: Vollständige technische Anleitung
  - `GWDG_INTEGRATION_ANLEITUNG.md`: Kurzanleitung speziell für GWDG
  - README.md aktualisiert mit neuem Abschnitt "Custom OpenAI-kompatible Endpoints"
  - Beispielkonfigurationen für JSON und Excel

**Technische Details:**
- Base URL wird aus der Konfiguration gelesen und an den OpenAI Client übergeben
- Validierung der Base URL (muss mit http:// oder https:// beginnen)
- Backward-kompatibel: Wenn keine Base URL angegeben ist, wird die Standard-OpenAI-URL verwendet
- Funktioniert mit allen Analysemodi (deductive, inductive, abductive, grounded)

**Anwendungsbeispiel GWDG:**
```json
{
  "model_provider": "OpenAI",
  "model_name": "openai-gpt-oss-120b",
  "api_base_url": "https://chat-ai.academiccloud.de/v1"
}
```

---

## Neu in 0.12.5 (2026-01-27)