# Umgebungsvariablen für LLM Provider Manager

Dieses Dokument beschreibt alle Umgebungsvariablen, die vom LLM Provider Manager und den verschiedenen Provider-Implementierungen verwendet werden.

## Übersicht

Der LLM Provider Manager verwendet Umgebungsvariablen für:
- API-Keys der verschiedenen Provider
- Custom API-Endpoints
- Konfigurationspfade
- Feature-Flags

## Provider API-Keys

### OpenAI

```bash
# Erforderlich für OpenAI-Modelle
export OPENAI_API_KEY="sk-..."
```

**Optional: Custom Endpoint**

```bash
# Für OpenAI-kompatible APIs (z.B. Azure OpenAI)
export OPENAI_API_ENDPOINT="https://custom-openai-endpoint.com/v1"
```

**Verwendung:**

```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'

from QCA_AID_assets.utils.llm import LLMProviderFactory
provider = LLMProviderFactory.create_provider('openai', 'gpt-4o-mini')
```

---

### Anthropic

```bash
# Erforderlich für Anthropic-Modelle (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Optional: Custom Endpoint**

```bash
# Für Custom Anthropic Endpoints
export ANTHROPIC_API_ENDPOINT="https://custom-anthropic-endpoint.com"
```

**Verwendung:**

```python
import os
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'

from QCA_AID_assets.utils.llm import LLMProviderFactory
provider = LLMProviderFactory.create_provider('anthropic', 'claude-3-opus')
```

---

### Mistral

```bash
# Erforderlich für Mistral-Modelle
export MISTRAL_API_KEY="..."
```

**Optional: Custom Endpoint**

```bash
# Für Custom Mistral Endpoints
export MISTRAL_API_ENDPOINT="https://custom-mistral-endpoint.com"
```

**Verwendung:**

```python
import os
os.environ['MISTRAL_API_KEY'] = '...'

from QCA_AID_assets.utils.llm import LLMProviderFactory
provider = LLMProviderFactory.create_provider('mistral', 'mistral-large')
```

---

### OpenRouter

```bash
# Erforderlich für OpenRouter-Modelle
export OPENROUTER_API_KEY="sk-or-..."
```

**Optional: Custom Endpoint**

```bash
# Standard: https://openrouter.ai/api/v1
export OPENROUTER_API_ENDPOINT="https://custom-openrouter-endpoint.com/v1"
```

**Verwendung:**

```python
import os
os.environ['OPENROUTER_API_KEY'] = 'sk-or-...'

from QCA_AID_assets.utils.llm import LLMProviderFactory
provider = LLMProviderFactory.create_provider('openrouter', 'mistral/mistral-large')
```

**Hinweis:** OpenRouter bietet Zugriff auf Modelle verschiedener Provider über eine einzige API.

---

### Lokale Modelle (LM Studio, Ollama)

**Keine API-Keys erforderlich!**

Lokale Modelle benötigen keine Umgebungsvariablen. Die Server müssen lediglich lokal laufen:

- **LM Studio**: `http://localhost:1234`
- **Ollama**: `http://localhost:11434`

**Optional: Custom Endpoints**

```bash
# Falls LM Studio auf anderem Port läuft
export LM_STUDIO_URL="http://localhost:8080"

# Falls Ollama auf anderem Port läuft
export OLLAMA_URL="http://localhost:12345"
```

**Verwendung:**

```python
from QCA_AID_assets.utils.llm import LLMProviderFactory

# LM Studio
provider = LLMProviderFactory.create_provider('local', 'my-local-model')

# Ollama
provider = LLMProviderFactory.create_provider('local', 'llama2')
```

---

## Konfigurationspfade

### Cache-Verzeichnis

```bash
# Standard: ~/.llm_cache
export LLM_CACHE_DIR="/custom/cache/path"
```

**Verwendung:**

```python
import os
cache_dir = os.environ.get('LLM_CACHE_DIR', '~/.llm_cache')

from QCA_AID_assets.utils.llm import LLMProviderManager
manager = LLMProviderManager(cache_dir=cache_dir)
```

### Fallback-Konfigurationen

```bash
# Verzeichnis für lokale Provider-Configs
export LLM_FALLBACK_DIR="/path/to/fallback/configs"
```

**Verwendung:**

```python
import os
fallback_dir = os.environ.get('LLM_FALLBACK_DIR')

from QCA_AID_assets.utils.llm import LLMProviderManager
manager = LLMProviderManager(fallback_dir=fallback_dir)
```

### Pricing-Overrides

```bash
# Verzeichnis für pricing_overrides.json
export LLM_CONFIG_DIR="/path/to/config"
```

**Verwendung:**

```python
import os
config_dir = os.environ.get('LLM_CONFIG_DIR')

from QCA_AID_assets.utils.llm import LLMProviderManager
manager = LLMProviderManager(config_dir=config_dir)
```

---

## Feature-Flags

### Cache-TTL

```bash
# Cache-Gültigkeitsdauer in Stunden (Standard: 24)
export LLM_CACHE_TTL_HOURS="48"
```

### Debug-Modus

```bash
# Aktiviert detailliertes Logging
export LLM_DEBUG="1"
```

**Verwendung:**

```python
import os
import logging

if os.environ.get('LLM_DEBUG') == '1':
    logging.basicConfig(level=logging.DEBUG)
```

---

## Vollständiges Setup-Beispiel

### Linux/macOS

Fügen Sie zu `~/.bashrc` oder `~/.zshrc` hinzu:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Mistral
export MISTRAL_API_KEY="..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."

# Optional: Custom Endpoints
# export OPENAI_API_ENDPOINT="https://custom-endpoint.com/v1"
# export ANTHROPIC_API_ENDPOINT="https://custom-endpoint.com"

# Optional: Konfigurationspfade
# export LLM_CACHE_DIR="/custom/cache"
# export LLM_CONFIG_DIR="/custom/config"
# export LLM_FALLBACK_DIR="/custom/fallback"

# Optional: Feature-Flags
# export LLM_DEBUG="1"
# export LLM_CACHE_TTL_HOURS="48"
```

Dann:

```bash
source ~/.bashrc  # oder ~/.zshrc
```

### Windows (PowerShell)

Fügen Sie zu Ihrem PowerShell-Profil hinzu (`$PROFILE`):

```powershell
# OpenAI
$env:OPENAI_API_KEY = "sk-..."

# Anthropic
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Mistral
$env:MISTRAL_API_KEY = "..."

# OpenRouter
$env:OPENROUTER_API_KEY = "sk-or-..."

# Optional: Custom Endpoints
# $env:OPENAI_API_ENDPOINT = "https://custom-endpoint.com/v1"
# $env:ANTHROPIC_API_ENDPOINT = "https://custom-endpoint.com"

# Optional: Konfigurationspfade
# $env:LLM_CACHE_DIR = "C:\custom\cache"
# $env:LLM_CONFIG_DIR = "C:\custom\config"
# $env:LLM_FALLBACK_DIR = "C:\custom\fallback"

# Optional: Feature-Flags
# $env:LLM_DEBUG = "1"
# $env:LLM_CACHE_TTL_HOURS = "48"
```

### Windows (CMD)

```cmd
REM OpenAI
set OPENAI_API_KEY=sk-...

REM Anthropic
set ANTHROPIC_API_KEY=sk-ant-...

REM Mistral
set MISTRAL_API_KEY=...

REM OpenRouter
set OPENROUTER_API_KEY=sk-or-...
```

**Hinweis:** CMD-Variablen sind nur für die aktuelle Session gültig. Für permanente Variablen verwenden Sie die Windows-Systemeinstellungen.

---

## Python-Konfiguration

### Direkt im Code setzen

```python
import os

# API-Keys
os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'
os.environ['MISTRAL_API_KEY'] = '...'
os.environ['OPENROUTER_API_KEY'] = 'sk-or-...'

# Custom Endpoints
os.environ['OPENAI_API_ENDPOINT'] = 'https://custom-endpoint.com/v1'

# Konfigurationspfade
os.environ['LLM_CACHE_DIR'] = '/custom/cache'
os.environ['LLM_CONFIG_DIR'] = '/custom/config'

# Feature-Flags
os.environ['LLM_DEBUG'] = '1'
```

### Aus .env-Datei laden

Erstellen Sie eine `.env`-Datei:

```env
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
OPENROUTER_API_KEY=sk-or-...

# Optional
OPENAI_API_ENDPOINT=https://custom-endpoint.com/v1
LLM_CACHE_DIR=/custom/cache
LLM_CONFIG_DIR=/custom/config
LLM_DEBUG=1
```

Laden Sie mit `python-dotenv`:

```bash
pip install python-dotenv
```

```python
from dotenv import load_dotenv
import os

# Lade .env-Datei
load_dotenv()

# Variablen sind jetzt verfügbar
api_key = os.environ.get('OPENAI_API_KEY')
```

---

## Sicherheitshinweise

### ⚠️ API-Keys schützen

1. **Niemals in Git committen**
   ```bash
   # .gitignore
   .env
   *.key
   secrets/
   ```

2. **Dateiberechtigungen setzen**
   ```bash
   chmod 600 .env
   ```

3. **Umgebungsvariablen verwenden**
   - Besser als Hardcoding im Code
   - Einfacher zu rotieren
   - Keine Secrets in Logs

4. **Secrets-Management verwenden**
   - Für Produktion: AWS Secrets Manager, Azure Key Vault, etc.
   - Für Entwicklung: `.env`-Dateien (nicht committen!)

### ✅ Best Practices

```python
import os

# ✅ Gut: Aus Umgebungsvariable
api_key = os.environ.get('OPENAI_API_KEY')

# ✅ Gut: Mit Fallback
api_key = os.environ.get('OPENAI_API_KEY', 'default-key')

# ✅ Gut: Validierung
if not os.environ.get('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not set")

# ❌ Schlecht: Hardcoded
api_key = "sk-..."  # NIEMALS!
```

---

## Troubleshooting

### Problem: "API key not found"

**Lösung:**

```python
import os

# Prüfen ob Variable gesetzt ist
print(os.environ.get('OPENAI_API_KEY'))

# Falls None: Variable ist nicht gesetzt
# Setzen Sie die Variable oder laden Sie .env
```

### Problem: "Invalid API key"

**Lösung:**

1. Prüfen Sie ob der Key korrekt ist
2. Prüfen Sie auf Leerzeichen am Anfang/Ende
3. Prüfen Sie ob der Key noch gültig ist

```python
import os

# Entferne Leerzeichen
api_key = os.environ.get('OPENAI_API_KEY', '').strip()
```

### Problem: Custom Endpoint wird nicht verwendet

**Lösung:**

```python
import os

# Prüfen ob Endpoint gesetzt ist
print(os.environ.get('OPENAI_API_ENDPOINT'))

# Setzen Sie explizit
os.environ['OPENAI_API_ENDPOINT'] = 'https://...'
```

---

## Referenz

### Alle Umgebungsvariablen

| Variable | Typ | Standard | Beschreibung |
|----------|-----|----------|--------------|
| `OPENAI_API_KEY` | String | - | OpenAI API-Key |
| `OPENAI_API_ENDPOINT` | URL | - | Custom OpenAI Endpoint |
| `ANTHROPIC_API_KEY` | String | - | Anthropic API-Key |
| `ANTHROPIC_API_ENDPOINT` | URL | - | Custom Anthropic Endpoint |
| `MISTRAL_API_KEY` | String | - | Mistral API-Key |
| `MISTRAL_API_ENDPOINT` | URL | - | Custom Mistral Endpoint |
| `OPENROUTER_API_KEY` | String | - | OpenRouter API-Key |
| `OPENROUTER_API_ENDPOINT` | URL | `https://openrouter.ai/api/v1` | Custom OpenRouter Endpoint |
| `LM_STUDIO_URL` | URL | `http://localhost:1234` | LM Studio Endpoint |
| `OLLAMA_URL` | URL | `http://localhost:11434` | Ollama Endpoint |
| `LLM_CACHE_DIR` | Path | `~/.llm_cache` | Cache-Verzeichnis |
| `LLM_FALLBACK_DIR` | Path | - | Fallback-Config-Verzeichnis |
| `LLM_CONFIG_DIR` | Path | `.` | Config-Verzeichnis |
| `LLM_DEBUG` | Boolean | `0` | Debug-Modus |
| `LLM_CACHE_TTL_HOURS` | Integer | `24` | Cache-TTL in Stunden |

---

## Weitere Ressourcen

- **Haupt-README**: `README.md` - Vollständige Dokumentation
- **Beispiele**: `examples/` - Praktische Beispiele
- **Design-Dokument**: `.kiro/specs/llm-provider-manager/design.md`

## Support

Bei Fragen zu Umgebungsvariablen:
1. Prüfen Sie diese Dokumentation
2. Prüfen Sie die Provider-spezifische Dokumentation
3. Aktivieren Sie Debug-Logging (`LLM_DEBUG=1`)
4. Erstellen Sie ein Issue im Repository
