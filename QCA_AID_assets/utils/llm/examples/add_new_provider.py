"""
Beispiel: Neue Provider hinzuFügen

Dieses Beispiel zeigt wie man neue Provider hinzufügt:
- Via URL (Catwalk-Format)
- Via lokale JSON-Datei
- Custom Provider-Konfiguration
"""

import asyncio
import json
import logging
from pathlib import Path
import sys
import tempfile
import shutil

# Füge Parent-Verzeichnis zum Path hinzu für Import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from QCA_AID_assets.utils.llm.provider_manager import LLMProviderManager


# Logging aktivieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def add_provider_via_url_example():
    """Zeigt wie man einen Provider via URL hinzufügt"""
    
    print("=" * 60)
    print("1. Provider via URL hinzuFügen")
    print("=" * 60)
    
    manager = LLMProviderManager()
    await manager.initialize()
    
    print("\nverfügbare Provider vor dem HinzuFügen:")
    providers_before = manager.get_supported_providers()
    print(f"   {', '.join(providers_before)}")
    
    # Beispiel: Füge einen Provider von einer URL hinzu
    # (In der Praxis würde hier eine echte URL verwendet)
    print("\nHinweis: In diesem Beispiel würde normalerweise eine URL verwendet:")
    print("   await manager.add_provider_source(")
    print("       'custom-provider',")
    print("       'https://example.com/custom-provider.json'")
    print("   )")
    print("\nDa wir keine echte URL haben, siehe nächstes Beispiel für lokale Dateien.")


async def add_provider_via_file_example():
    """Zeigt wie man einen Provider via lokale Datei hinzufügt"""
    
    print("\n" + "=" * 60)
    print("2. Provider via lokale Datei hinzuFügen")
    print("=" * 60)
    
    # Erstelle temporäres Verzeichnis
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Erstelle Custom Provider Config
        custom_config = {
            "name": "Custom Provider",
            "id": "custom-provider",
            "type": "openai",
            "api_key": "$CUSTOM_API_KEY",
            "api_endpoint": "https://api.custom-provider.com/v1",
            "models": [
                {
                    "id": "custom-fast-model",
                    "name": "Custom Fast Model",
                    "cost_per_1m_in": 0.50,
                    "cost_per_1m_out": 1.50,
                    "context_window": 64000,
                    "can_reason": False,
                    "supports_attachments": True
                },
                {
                    "id": "custom-smart-model",
                    "name": "Custom Smart Model",
                    "cost_per_1m_in": 2.00,
                    "cost_per_1m_out": 6.00,
                    "context_window": 128000,
                    "can_reason": True,
                    "supports_attachments": True,
                    "reasoning_levels": ["low", "medium", "high"],
                    "default_reasoning_effort": "medium"
                },
                {
                    "id": "custom-large-model",
                    "name": "Custom Large Model",
                    "cost_per_1m_in": 5.00,
                    "cost_per_1m_out": 15.00,
                    "context_window": 200000,
                    "can_reason": True,
                    "supports_attachments": True
                }
            ]
        }
        
        # Speichere Config
        config_path = Path(temp_dir) / "custom-provider.json"
        with open(config_path, 'w') as f:
            json.dump(custom_config, f, indent=2)
        
        print(f"\n✓ Custom Provider Config erstellt: {config_path}")
        print("\nConfig-Inhalt:")
        print(json.dumps(custom_config, indent=2))
        
        # Manager initialisieren
        print("\nInitialisiere Manager...")
        manager = LLMProviderManager()
        await manager.initialize()
        
        print("\nverfügbare Provider vor dem HinzuFügen:")
        providers_before = manager.get_supported_providers()
        print(f"   {', '.join(providers_before)}")
        print(f"   Gesamt: {len(manager.get_all_models())} Modelle")
        
        # Füge Custom Provider hinzu
        print(f"\nFüge Custom Provider hinzu...")
        await manager.add_local_provider_config(
            'custom-provider',
            str(config_path)
        )
        
        print("\n✓ Custom Provider erfolgreich hinzugefügt!")
        
        # Zeige aktualisierte Provider-Liste
        print("\nverfügbare Provider nach dem HinzuFügen:")
        providers_after = manager.get_supported_providers()
        print(f"   {', '.join(providers_after)}")
        print(f"   Gesamt: {len(manager.get_all_models())} Modelle")
        
        # Zeige Custom Provider Modelle
        print("\nCustom Provider Modelle:")
        custom_models = manager.get_models_by_provider('custom-provider')
        for model in custom_models:
            print(f"\n   {model.model_name} ({model.model_id}):")
            print(f"      Context: {model.context_window:,} Tokens")
            print(f"      Kosten: ${model.cost_in} in / ${model.cost_out} out")
            if model.options.get('can_reason'):
                print(f"      ✓ Reasoning")
            if model.options.get('supports_attachments'):
                print(f"      ✓ Attachments")
    
    finally:
        shutil.rmtree(temp_dir)
        print(f"\n✓ Temporäres Verzeichnis gelöscht")


async def create_provider_config_template():
    """Erstellt eine Vorlage für Provider-Konfiguration"""
    
    print("\n" + "=" * 60)
    print("3. Provider-Config-Vorlage erstellen")
    print("=" * 60)
    
    # Erstelle Template
    template = {
        "name": "My Custom Provider",
        "id": "my-provider",
        "type": "openai",  # oder "anthropic", "openrouter"
        "api_key": "$MY_PROVIDER_API_KEY",
        "api_endpoint": "https://api.my-provider.com/v1",
        "default_large_model_id": "my-large-model",
        "default_small_model_id": "my-small-model",
        "models": [
            {
                "id": "my-small-model",
                "name": "My Small Model",
                "cost_per_1m_in": 0.50,
                "cost_per_1m_out": 1.50,
                "context_window": 64000,
                "default_max_tokens": 16000,
                "can_reason": False,
                "supports_attachments": True
            },
            {
                "id": "my-large-model",
                "name": "My Large Model",
                "cost_per_1m_in": 3.00,
                "cost_per_1m_out": 9.00,
                "context_window": 128000,
                "default_max_tokens": 32000,
                "can_reason": True,
                "reasoning_levels": ["minimal", "low", "medium", "high"],
                "default_reasoning_effort": "medium",
                "supports_attachments": True
            }
        ]
    }
    
    # Speichere Template
    template_path = Path("custom_provider_template.json")
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"\n✓ Vorlage erstellt: {template_path}")
    print("\nVorlage-Inhalt:")
    print(json.dumps(template, indent=2))
    
    print("\n" + "=" * 60)
    print("Anleitung zum Erstellen einer Provider-Config:")
    print("=" * 60)
    
    print("\n1. Pflichtfelder:")
    print("   - name: Anzeigename des Providers")
    print("   - id: Eindeutige Provider-ID (lowercase, keine Leerzeichen)")
    print("   - type: Provider-Typ (openai, anthropic, openrouter)")
    print("   - models: Liste der Modelle")
    
    print("\n2. Modell-Pflichtfelder:")
    print("   - id: Eindeutige Modell-ID")
    print("   - name: Anzeigename des Modells")
    
    print("\n3. Optionale Modell-Felder:")
    print("   - cost_per_1m_in: Input-Kosten pro 1M Tokens (USD)")
    print("   - cost_per_1m_out: Output-Kosten pro 1M Tokens (USD)")
    print("   - context_window: Maximale Token-Anzahl")
    print("   - can_reason: Boolean für Reasoning-Fähigkeit")
    print("   - supports_attachments: Boolean für Attachment-Support")
    print("   - reasoning_levels: Liste von Reasoning-Levels")
    print("   - default_reasoning_effort: Standard Reasoning-Level")
    print("   - Alle anderen Felder werden in 'options' gespeichert")
    
    print("\n4. Verwendung:")
    print("   manager = LLMProviderManager()")
    print("   await manager.initialize()")
    print("   await manager.add_local_provider_config(")
    print("       'my-provider',")
    print("       '/path/to/custom_provider.json'")
    print("   )")


async def multiple_providers_example():
    """Zeigt wie man mehrere Custom Provider hinzufügt"""
    
    print("\n" + "=" * 60)
    print("4. Mehrere Custom Provider hinzuFügen")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Erstelle mehrere Provider Configs
        providers = {
            "provider-a": {
                "name": "Provider A",
                "id": "provider-a",
                "type": "openai",
                "models": [
                    {
                        "id": "provider-a-fast",
                        "name": "Provider A Fast",
                        "cost_per_1m_in": 0.30,
                        "cost_per_1m_out": 0.90,
                        "context_window": 32000
                    },
                    {
                        "id": "provider-a-smart",
                        "name": "Provider A Smart",
                        "cost_per_1m_in": 1.50,
                        "cost_per_1m_out": 4.50,
                        "context_window": 64000,
                        "can_reason": True
                    }
                ]
            },
            "provider-b": {
                "name": "Provider B",
                "id": "provider-b",
                "type": "anthropic",
                "models": [
                    {
                        "id": "provider-b-mini",
                        "name": "Provider B Mini",
                        "cost_per_1m_in": 0.20,
                        "cost_per_1m_out": 0.60,
                        "context_window": 16000
                    },
                    {
                        "id": "provider-b-pro",
                        "name": "Provider B Pro",
                        "cost_per_1m_in": 2.00,
                        "cost_per_1m_out": 6.00,
                        "context_window": 100000,
                        "can_reason": True,
                        "supports_attachments": True
                    }
                ]
            }
        }
        
        # Speichere Configs
        config_paths = {}
        for provider_id, config in providers.items():
            config_path = Path(temp_dir) / f"{provider_id}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            config_paths[provider_id] = config_path
            print(f"✓ Erstellt: {config_path}")
        
        # Manager initialisieren
        print("\nInitialisiere Manager...")
        manager = LLMProviderManager()
        await manager.initialize()
        
        print(f"\nProvider vor dem HinzuFügen: {len(manager.get_supported_providers())}")
        print(f"Modelle vor dem HinzuFügen: {len(manager.get_all_models())}")
        
        # Füge alle Provider hinzu
        print("\nFüge Custom Provider hinzu...")
        for provider_id, config_path in config_paths.items():
            await manager.add_local_provider_config(
                provider_id,
                str(config_path),
                update_cache=False  # Cache erst am Ende aktualisieren
            )
            print(f"   ✓ {provider_id} hinzugefügt")
        
        print(f"\nProvider nach dem HinzuFügen: {len(manager.get_supported_providers())}")
        print(f"Modelle nach dem HinzuFügen: {len(manager.get_all_models())}")
        
        # Zeige alle Provider
        print("\nAlle verfügbaren Provider:")
        for provider in manager.get_supported_providers():
            models = manager.get_models_by_provider(provider)
            print(f"   - {provider}: {len(models)} Modelle")
        
        # Zeige Custom Provider Details
        print("\nCustom Provider Details:")
        for provider_id in providers.keys():
            models = manager.get_models_by_provider(provider_id)
            print(f"\n{provider_id.upper()}:")
            for model in models:
                print(f"   - {model.model_name}")
                print(f"     Kosten: ${model.cost_in} / ${model.cost_out}")
                print(f"     Context: {model.context_window:,} Tokens")
    
    finally:
        shutil.rmtree(temp_dir)


async def cache_integration_example():
    """Zeigt Cache-Integration für neue Provider"""
    
    print("\n" + "=" * 60)
    print("5. Cache-Integration für neue Provider")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    cache_dir = tempfile.mkdtemp()
    
    try:
        # Erstelle Custom Provider
        custom_config = {
            "name": "Cached Provider",
            "id": "cached-provider",
            "type": "openai",
            "models": [
                {
                    "id": "cached-model",
                    "name": "Cached Model",
                    "cost_per_1m_in": 1.00,
                    "cost_per_1m_out": 3.00,
                    "context_window": 50000
                }
            ]
        }
        
        config_path = Path(temp_dir) / "cached-provider.json"
        with open(config_path, 'w') as f:
            json.dump(custom_config, f, indent=2)
        
        # Manager mit Custom Cache-Dir initialisieren
        print(f"Cache-Verzeichnis: {cache_dir}")
        print(f"Config-Verzeichnis: {temp_dir}")
        
        print("\n1. Erste Initialisierung (ohne Custom Provider)...")
        manager = LLMProviderManager(cache_dir=cache_dir)
        await manager.initialize()
        
        providers_before = manager.get_supported_providers()
        print(f"   Provider: {', '.join(providers_before)}")
        
        # Füge Custom Provider hinzu (mit Cache-Update)
        print("\n2. Füge Custom Provider hinzu (update_cache=True)...")
        await manager.add_local_provider_config(
            'cached-provider',
            str(config_path),
            update_cache=True
        )
        
        providers_after = manager.get_supported_providers()
        print(f"   Provider: {', '.join(providers_after)}")
        print("   ✓ Cache wurde aktualisiert")
        
        # Neuer Manager mit gleichem Cache
        print("\n3. Neue Manager-Instanz (lädt aus Cache)...")
        manager2 = LLMProviderManager(cache_dir=cache_dir)
        await manager2.initialize()
        
        providers_cached = manager2.get_supported_providers()
        print(f"   Provider: {', '.join(providers_cached)}")
        
        # Prüfe ob Custom Provider im Cache ist
        if 'cached-provider' in providers_cached:
            print("   ✓ Custom Provider wurde aus Cache geladen!")
            
            model = manager2.get_model_by_id('cached-model')
            if model:
                print(f"   ✓ Custom Model gefunden: {model.model_name}")
        else:
            print("   ✗ Custom Provider nicht im Cache")
    
    finally:
        shutil.rmtree(temp_dir)
        shutil.rmtree(cache_dir)


async def main():
    """Hauptfunktion - führt alle Beispiele aus"""
    
    try:
        # Provider via URL (konzeptionell)
        await add_provider_via_url_example()
        
        # Provider via lokale Datei
        await add_provider_via_file_example()
        
        # Config-Template erstellen
        await create_provider_config_template()
        
        # Mehrere Provider
        await multiple_providers_example()
        
        # Cache-Integration
        await cache_integration_example()
        
        print("\n" + "=" * 60)
        print("Alle Beispiele abgeschlossen!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nBeendet durch Benutzer")
    except Exception as e:
        print(f"\n✗ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
