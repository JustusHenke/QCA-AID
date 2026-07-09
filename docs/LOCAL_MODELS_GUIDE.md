# Anleitung: Lokale Modelle (LM Studio / Ollama) in der Webapp verwenden

## Übersicht

Die QCA-AID Webapp unterstützt jetzt die Verwendung lokaler LLM-Modelle über:
- **LM Studio** (Port 1234)
- **Ollama** (Port 11434)

## Vorteile lokaler Modelle

✅ **Kostenlos** - Keine API-Kosten
✅ **Privat** - Daten bleiben auf Ihrem Computer
✅ **Offline** - Keine Internetverbindung erforderlich
✅ **Kontrolle** - Volle Kontrolle über das Modell

## Schritt-für-Schritt-Anleitung

### 1. LM Studio oder Ollama installieren

#### Option A: LM Studio (Empfohlen für Einsteiger)

1. **Download:** [https://lmstudio.ai/](https://lmstudio.ai/)
2. **Installation:** Installieren Sie LM Studio
3. **Modell herunterladen:**
   - Öffnen Sie LM Studio
   - Gehen Sie zum "Discover" Tab
   - Suchen Sie nach einem Modell (z.B. "Llama 3.1 8B")
   - Klicken Sie auf "Download"
4. **Server starten:**
   - Gehen Sie zum "Local Server" Tab
   - Wählen Sie das heruntergeladene Modell
   - Klicken Sie auf "Start Server"
   - Server läuft auf Port 1234

#### Option B: Ollama (Für fortgeschrittene Nutzer)

1. **Download:** [https://ollama.ai/](https://ollama.ai/)
2. **Installation:** Installieren Sie Ollama
3. **Modell herunterladen:**
   ```bash
   ollama pull llama3.1:8b
   ```
4. **Server läuft automatisch** auf Port 11434

### 2. Modell in der Webapp auswählen

1. **Öffnen Sie die QCA-AID Webapp**
   ```bash
   python start_webapp.py
   ```

2. **Gehen Sie zum Konfiguration-Tab**

3. **Wählen Sie "Local (LM Studio/Ollama)" als Modell-Anbieter**
   ```
   Modell-Anbieter: Local (LM Studio/Ollama)
   ```

4. **Klicken Sie auf "🔄 Lokale Modelle erkennen"**
   - Die Webapp sucht nach laufenden Servern
   - Gefundene Modelle werden angezeigt

5. **Wählen Sie ein erkanntes Modell aus**
   ```
   Modell-Name: [Ihr Modell]
   ```

6. **Speichern Sie die Konfiguration**

### 3. Analyse starten

Jetzt können Sie Ihre Analyse wie gewohnt starten. Das lokale Modell wird verwendet!

## Empfohlene Modelle

### Für QCA-AID geeignete Modelle:

| Modell | Größe | RAM | Geschwindigkeit | Qualität |
|--------|-------|-----|-----------------|----------|
| **Llama 3.1 8B** | 4.7 GB | 8 GB | ⚡⚡⚡ Schnell | ⭐⭐⭐ Gut |
| **Llama 3.1 70B** | 40 GB | 64 GB | ⚡ Langsam | ⭐⭐⭐⭐⭐ Exzellent |
| **Mistral 7B** | 4.1 GB | 8 GB | ⚡⚡⚡ Schnell | ⭐⭐⭐ Gut |
| **Qwen 2.5 14B** | 8.5 GB | 16 GB | ⚡⚡ Mittel | ⭐⭐⭐⭐ Sehr gut |

**Empfehlung für Einsteiger:** Llama 3.1 8B (gute Balance aus Geschwindigkeit und Qualität)

## Fehlerbehebung

### Problem: "Keine lokalen Modelle gefunden"

**Lösung:**
1. Prüfen Sie, ob LM Studio/Ollama läuft
2. Prüfen Sie, ob ein Modell geladen ist
3. Prüfen Sie die Ports:
   - LM Studio: http://localhost:1234
   - Ollama: http://localhost:11434

### Problem: "Fehler bei der Erkennung"

**Lösung:**
1. Starten Sie LM Studio/Ollama neu
2. Starten Sie die Webapp neu
3. Prüfen Sie die Firewall-Einstellungen

### Problem: Modell ist sehr langsam

**Lösung:**
1. Verwenden Sie ein kleineres Modell (z.B. 7B statt 70B)
2. Prüfen Sie, ob Ihr Computer genug RAM hat
3. Schließen Sie andere Programme
4. Verwenden Sie GPU-Beschleunigung (falls verfügbar)

### Problem: Modell gibt schlechte Ergebnisse

**Lösung:**
1. Verwenden Sie ein größeres/besseres Modell
2. Passen Sie die Temperatur in den Coder-Einstellungen an
3. Verbessern Sie Ihre Kategoriendefinitionen im Codebook
4. Erwägen Sie die Verwendung eines kommerziellen Modells (OpenAI, Anthropic)

## Vergleich: Lokal vs. Cloud

| Aspekt | Lokale Modelle | Cloud-Modelle (OpenAI, etc.) |
|--------|----------------|------------------------------|
| **Kosten** | Kostenlos | $0.15 - $30 pro 1M Tokens |
| **Geschwindigkeit** | Abhängig von Hardware | Sehr schnell |
| **Qualität** | Gut bis sehr gut | Exzellent |
| **Privatsphäre** | 100% privat | Daten werden verarbeitet |
| **Offline** | Ja | Nein |
| **Setup** | Komplex | Einfach (nur API-Key) |

## Best Practices

### Für optimale Ergebnisse mit lokalen Modellen:

1. **Verwenden Sie präzise Kategoriendefinitionen**
   - Lokale Modelle benötigen klarere Anweisungen
   - Geben Sie mehr Beispiele im Codebook

2. **Passen Sie die Chunk-Größe an**
   - Kleinere Chunks (500-800 Zeichen) für kleinere Modelle
   - Größere Chunks (1000-1500 Zeichen) für größere Modelle

3. **Nutzen Sie Batch-Verarbeitung**
   - Kleinere Batch-Größen (3-5) für lokale Modelle
   - Verhindert Überlastung des Modells

4. **Testen Sie verschiedene Modelle**
   - Jedes Modell hat Stärken und Schwächen
   - Testen Sie mit einer kleinen Stichprobe

## Technische Details

### LM Studio API

- **Endpoint:** http://localhost:1234/v1/models
- **Format:** OpenAI-kompatibel
- **Dokumentation:** [LM Studio Docs](https://lmstudio.ai/docs)

### Ollama API

- **Endpoint:** http://localhost:11434/api/tags
- **Format:** Ollama-spezifisch
- **Dokumentation:** [Ollama Docs](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Automatische Erkennung

Die Webapp verwendet den `LocalDetector` aus `QCA_AID_assets/utils/llm/local_detector.py`:

1. **Prüft LM Studio** (Port 1234)
2. **Prüft Ollama API** (Port 11434)
3. **Fallback: Ollama CLI** (`ollama list`)
4. **Gibt Liste aller gefundenen Modelle zurück**

## Weitere Ressourcen

- **LM Studio:** [https://lmstudio.ai/](https://lmstudio.ai/)
- **Ollama:** [https://ollama.ai/](https://ollama.ai/)
- **Modell-Übersicht:** [https://huggingface.co/models](https://huggingface.co/models)
- **QCA-AID Dokumentation:** [README.md](README.md)

---

**Viel Erfolg mit lokalen Modellen!** 🚀
