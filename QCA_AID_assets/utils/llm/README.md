# LLM Provider Manager

Ein erweiterbares System zur Verwaltung mehrerer LLM-Provider (OpenAI, Anthropic, Mistral, OpenRouter, lokale Modelle). Das System lädt Provider-Konfigurationen von externen Quellen, cached diese lokal, normalisiert alle Modelle in ein einheitliches Format und bietet Filter- und Suchfunktionen.

## Features

- ✅ **Multi-Provider-Unterstützung**: OpenAI, Anthropic, Mistral, OpenRouter, LM Studio, Ollama
- ✅ **Automatisches Caching**: 24h TTL für schnellere Initialisierung
- ✅ **Lokale Modell-Erkennung**: Automatische Erkennung von LM Studio und Ollama
- ✅ **Einheitliches Interface**: Alle Modelle im gleichen Format
- ✅ **Flexible Filter-API**: Suche nach Provider, Kosten, Context Window, Capabilities
- ✅ **Pricing-Overrides**: Eigene Preisinformationen definieren
- ✅ **Erweiterbar**: Neue Provider einfach hinzuFügen
- ✅ **Robuste Fehlerbehandlung**: Fallback-Mechanismen bei Netzwerkproblemen

## Installation

Das LLM Provider Manager System ist Teil von QCA-AID und benötigt keine separate Installation.

### Abhängigkeiten

```bash
pip install aiohttp  # Für asynchrone HTTP-Requests
```

## Schnellstart

### Basis-Nutzung

```python
import asyncio
from QCA_AID_assets.utils.llm import LLMProviderManager

async def main():
    # Manager initialisieren
    manager = LLMProviderManager()
    
    # Provider-Daten laden (cached automatisch für 24h)
    await manager.initialize()
    
    # Alle verfügbaren Modelle abrufen
    all_models = manager.get_all_models()
    print(f"verfügbare Modelle: {len(all_models)}")
    
    # Modelle nach Provider filtern
    openai_models = manager.get_models_by_provider('openai')
    print(f"OpenAI Modelle: {len(openai_models)}")
    
    # Spezifisches Modell suchen
    model = manager.get_model_by_id('gpt-4o-mini')
    if model:
        print(f"Modell: {model.model_name}")
        print(f"Context Window: {model.context_window}")
        print(f"Kosten Input: ${model.cost_in}/1M Tokens")
        print(f"Kosten Output: ${model.cost_out}/1M Tokens")
    
    # Unterstützte Provider auflisten
    providers = manager.get_supported_providers()
    print(f"Provider: {', '.join(providers)}")

# Ausführen
asyncio.run(main())
```

### Mit Custom Cache-Verzeichnis

```python
manager = LLMProviderManager(
    cache_dir="/custom/cache/path",
    fallback_dir="/custom/configs/path"
)
await manager.initialize()
```

### Cache-Invalidierung

```python
# Cache löschen und neu laden
manager.invalidate_cache()
await manager.initialize(force_refresh=True)
```

## Filter-Nutzung

### Nach Provider filtern

```python
# Alle OpenAI-Modelle
openai_models = manager.get_models_by_provider('openai')

# Alle lokalen Modelle (LM Studio, Ollama)
local_models = manager.get_models_by_provider('local')

# Alle Anthropic-Modelle
anthropic_models = manager.get_models_by_provider('anthropic')
```

### Nach Kosten filtern

```python
# Günstige Modelle finden (max $1/1M Input-Tokens)
cheap_models = manager.filter_models(
    max_cost_in=1.0,
    max_cost_out=5.0
)

for model in cheap_models:
    print(f"{model.model_name}: ${model.cost_in} in / ${model.cost_out} out")
```

### Nach Context Window filtern

```python
# Modelle mit großem Context Window (min 100k Tokens)
large_context_models = manager.filter_models(
    min_context=100000
)

for model in large_context_models:
    print(f"{model.model_name}: {model.context_window} Tokens")
```

### Nach Capabilities filtern

```python
# Modelle mit Reasoning-Fähigkeit
reasoning_models = manager.filter_models(
    capabilities=['can_reason']
)

# Modelle mit Attachment-Support
attachment_models = manager.filter_models(
    capabilities=['supports_attachments']
)
```

### Kombinierte Filter

```python
# Günstige OpenAI-Modelle mit großem Context Window
filtered_models = manager.filter_models(
    provider='openai',
    max_cost_in=1.0,
    min_context=100000
)

for model in filtered_models:
    print(f"{model.model_name}:")
    print(f"  Context: {model.context_window}")
    print(f"  Kosten: ${model.cost_in} / ${model.cost_out}")
```

### Komplexe Filter-Logik

```python
# Alle Modelle abrufen und manuell filtern
all_models = manager.get_all_models()

# Modelle mit bestimmten Eigenschaften
suitable_models = [
    model for model in all_models
    if model.cost_in and model.cost_in < 2.0
    and model.context_window and model.context_window >= 50000
    and model.options.get('can_reason', False)
]

print(f"Gefunden: {len(suitable_models)} passende Modelle")
```

## Pricing-Overrides

Sie können eigene Preisinformationen für Modelle definieren, um z.B. Unternehmensrabatte oder spezielle Vereinbarungen abzubilden.

### 1. Erstellen Sie `pricing_overrides.json`

Erstellen Sie eine Datei im Konfigurationsverzeichnis (Standard: aktuelles Verzeichnis):

```json
{
    "gpt-4o-mini": {
        "cost_in": 0.10,
        "cost_out": 0.50
    },
    "claude-3-opus": {
        "cost_in": 10.0,
        "cost_out": 30.0
    },
    "gpt-4o": {
        "cost_in": 2.0,
        "cost_out": 8.0
    }
}
```

### 2. Manager mit Config-Verzeichnis initialisieren

```python
manager = LLMProviderManager(
    config_dir="/path/to/config"  # Verzeichnis mit pricing_overrides.json
)
await manager.initialize()
```

### 3. Overrides werden automatisch angewendet

```python
model = manager.get_model_by_id('gpt-4o-mini')
print(f"Überschriebene Kosten: ${model.cost_in} / ${model.cost_out}")
# Output: Überschriebene Kosten: $0.10 / $0.50
```

### Hinweise zu Pricing-Overrides

- ✅ Overrides werden nur auf existierende Modelle angewendet
- ✅ Nicht-existierende Modelle in der Override-Datei werden ignoriert
- ✅ Fehlerhafte Override-Dateien führen zu einer Warnung, nicht zu einem Fehler
- ✅ Ohne `pricing_overrides.json` werden Standard-Preise verwendet

## Neue Provider hinzuFügen

### Via URL (Catwalk-Format)

```python
# Neuen Provider von URL hinzuFügen
await manager.add_provider_source(
    provider_name='custom-provider',
    url='https://example.com/custom-provider.json'
)

# Modelle sind sofort verfügbar
custom_models = manager.get_models_by_provider('custom-provider')
```

### Via lokale JSON-Datei

```python
# Neuen Provider von lokaler Datei hinzuFügen
await manager.add_local_provider_config(
    provider_name='custom-provider',
    config_path='/path/to/custom-provider.json'
)

# Modelle sind sofort verfügbar
custom_models = manager.get_models_by_provider('custom-provider')
```

### Provider-Config-Format

Ihre Provider-Config-Datei sollte dem Catwalk-Format folgen:

```json
{
    "name": "Custom Provider",
    "id": "custom-provider",
    "type": "openai",
    "api_key": "$CUSTOM_API_KEY",
    "api_endpoint": "https://api.custom-provider.com/v1",
    "models": [
        {
            "id": "custom-model-1",
            "name": "Custom Model 1",
            "cost_per_1m_in": 1.5,
            "cost_per_1m_out": 5.0,
            "context_window": 128000,
            "can_reason": true,
            "supports_attachments": false
        },
        {
            "id": "custom-model-2",
            "name": "Custom Model 2",
            "cost_per_1m_in": 0.5,
            "cost_per_1m_out": 2.0,
            "context_window": 64000
        }
    ]
}
```

### Cache-Update deaktivieren

```python
# Provider hinzuFügen ohne Cache-Update
await manager.add_provider_source(
    provider_name='custom-provider',
    url='https://example.com/custom-provider.json',
    update_cache=False
)
```

## Umgebungsvariablen für API-Keys

Das System verwendet Umgebungsvariablen für API-Keys der verschiedenen Provider:

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

Optional: Custom Endpoint

```bash
export OPENAI_API_ENDPOINT="https://custom-openai-endpoint.com/v1"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Optional: Custom Endpoint

```bash
export ANTHROPIC_API_ENDPOINT="https://custom-anthropic-endpoint.com"
```

### Mistral

```bash
export MISTRAL_API_KEY="..."
```

### OpenRouter

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### Lokale Modelle (LM Studio, Ollama)

Keine API-Keys erforderlich. Die Server müssen lokal laufen:

- **LM Studio**: `http://localhost:1234`
- **Ollama**: `http://localhost:11434`

### In Python setzen

```python
import os

# API-Keys programmatisch setzen
os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'
os.environ['OPENROUTER_API_KEY'] = 'sk-or-...'

# Manager initialisieren
manager = LLMProviderManager()
await manager.initialize()
```

## Datenmodell

### NormalizedModel

Alle Modelle werden in ein einheitliches Format normalisiert:

```python
@dataclass
class NormalizedModel:
    provider: str                    # Provider-Name (z.B. 'openai', 'anthropic')
    model_id: str                    # Eindeutige Modell-ID (z.B. 'gpt-4o-mini')
    model_name: str                  # Anzeigename (z.B. 'GPT-4o Mini')
    context_window: Optional[int]    # Max. Tokens (z.B. 128000)
    cost_in: Optional[float]         # Kosten pro 1M Input-Tokens (USD)
    cost_out: Optional[float]        # Kosten pro 1M Output-Tokens (USD)
    options: Dict[str, Any]          # Zusätzliche Eigenschaften
```

### Options-Dictionary

Das `options`-Dictionary enthält provider-spezifische Eigenschaften:

```python
model.options = {
    'can_reason': True,
    'reasoning_levels': ['minimal', 'low', 'medium', 'high'],
    'default_reasoning_effort': 'medium',
    'supports_attachments': True,
    'cost_per_1m_in_cached': 0.13,
    'cost_per_1m_out_cached': 0.13,
    'default_max_tokens': 128000
}
```

### Beispiel-Zugriff

```python
model = manager.get_model_by_id('gpt-4o-mini')

# Standard-Felder
print(f"Provider: {model.provider}")
print(f"ID: {model.model_id}")
print(f"Name: {model.model_name}")
print(f"Context: {model.context_window}")
print(f"Kosten: ${model.cost_in} / ${model.cost_out}")

# Options
if model.options.get('can_reason'):
    print("✓ Unterstützt Reasoning")

if model.options.get('supports_attachments'):
    print("✓ Unterstützt Attachments")

# Cached Pricing
cached_in = model.options.get('cost_per_1m_in_cached')
if cached_in:
    print(f"Cached Input-Kosten: ${cached_in}")
```

## Erweiterte Nutzung

### Asynchrone Initialisierung mit Fehlerbehandlung

```python
async def initialize_manager():
    manager = LLMProviderManager()
    
    try:
        await manager.initialize()
        print("✓ Manager erfolgreich initialisiert")
        return manager
    except Exception as e:
        print(f"✗ Fehler bei Initialisierung: {e}")
        
        # Fallback: Nur lokale Modelle verwenden
        print("Versuche nur lokale Modelle zu laden...")
        manager.invalidate_cache()
        
        try:
            await manager.initialize(force_refresh=True)
            return manager
        except Exception as e2:
            print(f"✗ Auch lokale Modelle nicht verfügbar: {e2}")
            raise
```

### Modell-Statistiken

```python
def print_model_statistics(manager):
    all_models = manager.get_all_models()
    providers = manager.get_supported_providers()
    
    print(f"\n=== Modell-Statistiken ===")
    print(f"Gesamt: {len(all_models)} Modelle")
    print(f"Provider: {len(providers)}")
    
    for provider in providers:
        models = manager.get_models_by_provider(provider)
        print(f"  - {provider}: {len(models)} Modelle")
    
    # Kosten-Statistiken
    models_with_cost = [m for m in all_models if m.cost_in is not None]
    if models_with_cost:
        avg_cost_in = sum(m.cost_in for m in models_with_cost) / len(models_with_cost)
        avg_cost_out = sum(m.cost_out for m in models_with_cost) / len(models_with_cost)
        print(f"\nDurchschnittskosten:")
        print(f"  Input: ${avg_cost_in:.2f}/1M Tokens")
        print(f"  Output: ${avg_cost_out:.2f}/1M Tokens")
```

### Modell-Empfehlung

```python
def recommend_model(manager, task_type: str):
    """Empfiehlt ein Modell basierend auf Task-Typ"""
    
    if task_type == 'coding':
        # Für Coding: Großes Context Window, Reasoning
        models = manager.filter_models(
            capabilities=['can_reason'],
            min_context=100000
        )
        # Sortiere nach Kosten
        models.sort(key=lambda m: m.cost_in or float('inf'))
        return models[0] if models else None
    
    elif task_type == 'chat':
        # Für Chat: Günstig, schnell
        models = manager.filter_models(
            max_cost_in=1.0,
            max_cost_out=5.0
        )
        return models[0] if models else None
    
    elif task_type == 'analysis':
        # Für Analyse: Großes Context Window
        models = manager.filter_models(
            min_context=128000
        )
        # Sortiere nach Context Window (größer zuerst)
        models.sort(key=lambda m: m.context_window or 0, reverse=True)
        return models[0] if models else None
    
    return None

# Nutzung
model = recommend_model(manager, 'coding')
if model:
    print(f"Empfohlen für Coding: {model.model_name}")
```

## Integration mit LLMProviderFactory

Der Provider Manager kann mit der bestehenden Factory kombiniert werden:

```python
from QCA_AID_assets.utils.llm import LLMProviderManager, LLMProviderFactory

async def create_provider_instance(model_id: str):
    # Manager initialisieren
    manager = LLMProviderManager()
    await manager.initialize()
    
    # Modell-Informationen abrufen
    model = manager.get_model_by_id(model_id)
    if not model:
        raise ValueError(f"Modell '{model_id}' nicht gefunden")
    
    # Provider-Instanz erstellen
    provider = LLMProviderFactory.create_provider(
        provider_name=model.provider,
        model_name=model.model_id
    )
    
    return provider, model

# Nutzung
provider, model_info = await create_provider_instance('gpt-4o-mini')
print(f"Provider erstellt: {model_info.model_name}")
print(f"Kosten: ${model_info.cost_in} / ${model_info.cost_out}")
```

## Fehlerbehandlung

### Netzwerkfehler

```python
try:
    await manager.initialize()
except Exception as e:
    print(f"Fehler beim Laden: {e}")
    # System fällt automatisch auf lokale Configs zurück
```

### Keine Modelle verfügbar

```python
from QCA_AID_assets.utils.llm.models import ProviderLoadError

try:
    await manager.initialize()
except ProviderLoadError as e:
    print(f"Keine Modelle geladen: {e}")
    print("Mögliche Ursachen:")
    print("  - Keine Netzwerkverbindung")
    print("  - Fallback-Dateien fehlen")
    print("  - Keine lokalen Server laufen")
```

### Manager nicht initialisiert

```python
manager = LLMProviderManager()

try:
    models = manager.get_all_models()  # Fehler!
except RuntimeError as e:
    print(f"Fehler: {e}")
    # Lösung: Erst initialisieren
    await manager.initialize()
    models = manager.get_all_models()  # OK
```

## Cache-Verwaltung

### Cache-Speicherort

Standard: `~/.llm_cache/providers.json`

Custom:
```python
manager = LLMProviderManager(cache_dir="/custom/cache")
```

### Cache-TTL

Der Cache ist 24 Stunden gültig. Nach Ablauf wird automatisch neu geladen.

### Manuelles Cache-Management

```python
# Cache-Status prüfen
if manager.cache_manager.is_valid():
    print("Cache ist gültig")
else:
    print("Cache ist abgelaufen oder existiert nicht")

# Cache löschen
manager.invalidate_cache()

# Neu laden erzwingen
await manager.initialize(force_refresh=True)
```

## Logging

Das System verwendet Python's `logging`-Modul:

```python
import logging

# Logging aktivieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Manager initialisieren (mit Logs)
manager = LLMProviderManager()
await manager.initialize()
```

### Log-Level

- `DEBUG`: Detaillierte Informationen über interne Vorgänge
- `INFO`: Wichtige Ereignisse (Initialisierung, Cache-Hits, etc.)
- `WARNING`: Nicht-kritische Probleme (fehlende Modelle, Cache-Fehler)
- `ERROR`: Kritische Fehler (Provider-Load-Fehler)

## Performance-Tipps

1. **Cache nutzen**: Lassen Sie den Cache aktiviert für schnellere Initialisierung
2. **Paralleles Laden**: Provider werden automatisch parallel geladen
3. **Filter statt Iteration**: Nutzen Sie `filter_models()` statt manueller Iteration
4. **Lazy Initialization**: Initialisieren Sie den Manager nur wenn benötigt
5. **Wiederverwendung**: Erstellen Sie eine Manager-Instanz und verwenden Sie diese mehrfach

## Troubleshooting

### Problem: Keine Modelle werden geladen

**Lösung:**
1. Prüfen Sie Ihre Netzwerkverbindung
2. Stellen Sie sicher, dass Fallback-Dateien existieren: `QCA_AID_assets/utils/llm/configs/*.json`
3. Aktivieren Sie Debug-Logging: `logging.basicConfig(level=logging.DEBUG)`

### Problem: Lokale Modelle werden nicht erkannt

**Lösung:**
1. Stellen Sie sicher, dass LM Studio oder Ollama läuft
2. Prüfen Sie die URLs:
   - LM Studio: `http://localhost:1234`
   - Ollama: `http://localhost:11434`
3. Testen Sie manuell: `curl http://localhost:1234/v1/models`

### Problem: Pricing-Overrides werden nicht angewendet

**Lösung:**
1. Prüfen Sie den Dateinamen: `pricing_overrides.json`
2. Prüfen Sie das JSON-Format (gültiges JSON)
3. Stellen Sie sicher, dass `config_dir` korrekt gesetzt ist
4. Prüfen Sie, ob die model_ids exakt übereinstimmen

### Problem: Cache wird nicht aktualisiert

**Lösung:**
```python
# Cache manuell invalidieren
manager.invalidate_cache()
await manager.initialize(force_refresh=True)
```

## Lizenz

Teil des QCA-AID Projekts. Siehe Haupt-Lizenz für Details.

## Support

Bei Fragen oder Problemen:
1. Prüfen Sie die Logs mit `logging.basicConfig(level=logging.DEBUG)`
2. Überprüfen Sie die Fallback-Konfigurationen
3. Erstellen Sie ein Issue im QCA-AID Repository
