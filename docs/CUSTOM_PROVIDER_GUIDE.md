# Custom Provider Integration Guide

## Übersicht

QCA-AID unterstützt jetzt die Integration von Custom OpenAI-kompatiblen API-Endpoints wie GWDG Academic Cloud, Azure OpenAI und andere Services.

## Konfiguration

### 1. API-Key einrichten

Erstellen Sie eine `.env` Datei im QCA-AID-Projektverzeichnis:

```bash
# Für GWDG Academic Cloud
OPENAI_API_KEY=ihr-gwdg-api-key
```

**Alternative Methoden:**
- Systemumgebungsvariable setzen
- In `~/.environ.env` speichern

### 2. Custom Base URL konfigurieren

#### Option A: Über die Web-App (Empfohlen)

1. Öffnen Sie die QCA-AID Web-App
2. Navigieren Sie zum **Konfiguration**-Tab
3. Wählen Sie **OpenAI** als Provider
4. Klicken Sie auf **"🔧 Erweiterte Einstellungen: Custom API Base URL"**
5. Geben Sie Ihre Base URL ein, z.B.:
   ```
   https://chat-ai.academiccloud.de/v1
   ```
6. Wählen Sie das gewünschte Modell (z.B. `openai-gpt-oss-120b`)
7. Speichern Sie die Konfiguration

#### Option B: Über JSON-Konfigurationsdatei

Bearbeiten Sie Ihre `QCA-AID-Codebook.json`:

```json
{
  "model_provider": "OpenAI",
  "model_name": "openai-gpt-oss-120b",
  "api_base_url": "https://chat-ai.academiccloud.de/v1",
  "data_dir": "input",
  "output_dir": "output",
  ...
}
```

#### Option C: Über Excel-Konfigurationsdatei

In Ihrer `QCA-AID-Codebook.xlsx`, fügen Sie eine neue Zeile hinzu:

| Parameter | Wert |
|-----------|------|
| api_base_url | https://chat-ai.academiccloud.de/v1 |

## Beispiel: GWDG Academic Cloud

### Schritt-für-Schritt Anleitung

1. **API-Key erhalten:**
   - Besuchen Sie die GWDG Academic Cloud
   - Generieren Sie einen API-Key
   - Speichern Sie den Key sicher

2. **Umgebungsvariable setzen:**
   ```bash
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="ihr-gwdg-api-key"
   
   # Linux/Mac
   export OPENAI_API_KEY="ihr-gwdg-api-key"
   ```

3. **Konfiguration in QCA-AID:**
   ```json
   {
     "model_provider": "OpenAI",
     "model_name": "openai-gpt-oss-120b",
     "api_base_url": "https://chat-ai.academiccloud.de/v1"
   }
   ```

4. **Analyse starten:**
   - Die Webapp verwendet automatisch die Custom Base URL
   - Alle API-Calls gehen an den GWDG-Endpoint

## Unterstützte Provider

### OpenAI-kompatible Endpoints

- ✅ **GWDG Academic Cloud**
  - Base URL: `https://chat-ai.academiccloud.de/v1`
  - Modelle: `openai-gpt-oss-120b`, etc.

- ✅ **Azure OpenAI**
  - Base URL: `https://your-resource.openai.azure.com/openai/deployments/your-deployment`
  - Modelle: Ihre Azure-Deployment-Namen

- ✅ **Lokale OpenAI-kompatible Server**
  - LM Studio: `http://localhost:1234/v1`
  - Ollama: `http://localhost:11434/v1`
  - Text Generation WebUI: `http://localhost:5000/v1`

## Technische Details

### Wie es funktioniert

1. **Provider-Initialisierung:**
   ```python
   provider = LLMProviderFactory.create_provider(
       provider_name='openai',
       model_name='openai-gpt-oss-120b',
       base_url='https://chat-ai.academiccloud.de/v1'
   )
   ```

2. **OpenAI Client:**
   ```python
   from openai import AsyncOpenAI
   
   client = AsyncOpenAI(
       api_key=os.getenv('OPENAI_API_KEY'),
       base_url='https://chat-ai.academiccloud.de/v1'
   )
   ```

3. **API-Calls:**
   - Alle Requests gehen an die Custom Base URL
   - OpenAI-kompatible Response-Formate werden erwartet

### Validierung

Die Base URL wird automatisch validiert:
- ✅ Muss mit `http://` oder `https://` beginnen
- ✅ Wird nur bei OpenAI und Local Providern verwendet
- ✅ Optional - Standard-URL wird verwendet wenn leer

## Fehlerbehebung

### Problem: "OPENAI_API_KEY nicht gefunden"

**Lösung:**
```bash
# Prüfen Sie, ob die Umgebungsvariable gesetzt ist
echo $OPENAI_API_KEY  # Linux/Mac
echo $env:OPENAI_API_KEY  # Windows PowerShell

# Setzen Sie die Variable neu
export OPENAI_API_KEY="ihr-key"  # Linux/Mac
$env:OPENAI_API_KEY="ihr-key"  # Windows PowerShell
```

### Problem: "Connection Error"

**Lösung:**
1. Prüfen Sie die Base URL auf Tippfehler
2. Stellen Sie sicher, dass der Endpoint erreichbar ist
3. Prüfen Sie Firewall/Proxy-Einstellungen

### Problem: "Model not found"

**Lösung:**
1. Prüfen Sie, welche Modelle Ihr Provider unterstützt
2. Verwenden Sie den exakten Modellnamen
3. Kontaktieren Sie Ihren Provider für verfügbare Modelle

## Beispiel-Konfigurationen

### GWDG Academic Cloud (Vollständig)

```json
{
  "model_provider": "OpenAI",
  "model_name": "openai-gpt-oss-120b",
  "api_base_url": "https://chat-ai.academiccloud.de/v1",
  "data_dir": "input",
  "output_dir": "output",
  "chunk_size": 1200,
  "chunk_overlap": 50,
  "batch_size": 8,
  "code_with_context": false,
  "multiple_codings": true,
  "multiple_coding_threshold": 0.65,
  "analysis_mode": "deductive",
  "review_mode": "consensus",
  "attribute_labels": {
    "attribut1": "Attribut1",
    "attribut2": "Attribut2",
    "attribut3": "Attribut3"
  },
  "coder_settings": [
    {
      "temperature": 0.3,
      "coder_id": "auto_1"
    }
  ]
}
```

### Azure OpenAI

```json
{
  "model_provider": "OpenAI",
  "model_name": "gpt-4",
  "api_base_url": "https://your-resource.openai.azure.com/openai/deployments/your-deployment",
  ...
}
```

### Lokaler Server (LM Studio)

```json
{
  "model_provider": "OpenAI",
  "model_name": "local-model",
  "api_base_url": "http://localhost:1234/v1",
  ...
}
```

## Programmatische Verwendung

Wenn Sie QCA-AID programmatisch verwenden:

```python
from QCA_AID_assets.core.config import CONFIG
from QCA_AID_assets.utils.llm.factory import LLMProviderFactory

# Setzen Sie die Custom Base URL
CONFIG['API_BASE_URL'] = 'https://chat-ai.academiccloud.de/v1'
CONFIG['MODEL_PROVIDER'] = 'OpenAI'
CONFIG['MODEL_NAME'] = 'openai-gpt-oss-120b'

# Provider wird automatisch mit Custom URL erstellt
provider = LLMProviderFactory.create_provider(
    provider_name=CONFIG['MODEL_PROVIDER'].lower(),
    model_name=CONFIG['MODEL_NAME'],
    base_url=CONFIG.get('API_BASE_URL')
)
```

