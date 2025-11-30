"""
Beispiel: Basis-Nutzung des LLM Provider Managers

Dieses Beispiel zeigt die grundlegende Verwendung des LLMProviderManagers:
- Initialisierung
- Modell-Suche
- Provider-Abfrage
- Modell-Informationen abrufen
"""

import asyncio
import logging
from pathlib import Path
import sys

# Füge Parent-Verzeichnis zum Path hinzu für Import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from QCA_AID_assets.utils.llm.provider_manager import LLMProviderManager


# Logging aktivieren für bessere Sichtbarkeit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def basic_usage_example():
    """Zeigt grundlegende Nutzung des Provider Managers"""
    
    print("=" * 60)
    print("LLM Provider Manager - Basis-Nutzung")
    print("=" * 60)
    
    # 1. Manager initialisieren
    print("\n1. Manager initialisieren...")
    manager = LLMProviderManager()
    
    try:
        await manager.initialize()
        print("✓ Manager erfolgreich initialisiert")
    except Exception as e:
        print(f"✗ Fehler bei Initialisierung: {e}")
        return
    
    # 2. Alle verfügbaren Modelle abrufen
    print("\n2. Alle verfügbaren Modelle abrufen...")
    all_models = manager.get_all_models()
    print(f"✓ Gefunden: {len(all_models)} Modelle")
    
    # 3. Unterstützte Provider auflisten
    print("\n3. Unterstützte Provider:")
    providers = manager.get_supported_providers()
    for provider in providers:
        models = manager.get_models_by_provider(provider)
        print(f"   - {provider}: {len(models)} Modelle")
    
    # 4. Spezifisches Modell suchen
    print("\n4. Spezifisches Modell suchen (gpt-4o-mini)...")
    model = manager.get_model_by_id('gpt-4o-mini')
    
    if model:
        print(f"✓ Modell gefunden:")
        print(f"   Name: {model.model_name}")
        print(f"   Provider: {model.provider}")
        print(f"   Context Window: {model.context_window:,} Tokens" if model.context_window else "   Context Window: N/A")
        print(f"   Kosten Input: ${model.cost_in}/1M Tokens" if model.cost_in else "   Kosten Input: N/A")
        print(f"   Kosten Output: ${model.cost_out}/1M Tokens" if model.cost_out else "   Kosten Output: N/A")
        
        # Zusätzliche Eigenschaften
        if model.options:
            print(f"   Zusätzliche Eigenschaften:")
            if model.options.get('can_reason'):
                print(f"      ✓ Unterstützt Reasoning")
            if model.options.get('supports_attachments'):
                print(f"      ✓ Unterstützt Attachments")
    else:
        print("✗ Modell nicht gefunden")
    
    # 5. Modelle nach Provider filtern
    print("\n5. OpenAI-Modelle abrufen...")
    openai_models = manager.get_models_by_provider('openai')
    print(f"✓ Gefunden: {len(openai_models)} OpenAI-Modelle")
    
    # Zeige erste 5 Modelle
    print("   Erste 5 Modelle:")
    for model in openai_models[:5]:
        cost_str = f"${model.cost_in}/{model.cost_out}" if model.cost_in else "N/A"
        print(f"      - {model.model_name} ({model.model_id}) - {cost_str}")
    
    # 6. Lokale Modelle prüfen
    print("\n6. Lokale Modelle prüfen...")
    local_models = manager.get_models_by_provider('local')
    if local_models:
        print(f"✓ Gefunden: {len(local_models)} lokale Modelle")
        for model in local_models:
            print(f"   - {model.model_name} ({model.model_id})")
    else:
        print("   Keine lokalen Modelle gefunden (LM Studio/Ollama nicht aktiv)")
    
    print("\n" + "=" * 60)
    print("Beispiel abgeschlossen!")
    print("=" * 60)


async def model_details_example():
    """Zeigt detaillierte Modell-Informationen"""
    
    print("\n" + "=" * 60)
    print("Detaillierte Modell-Informationen")
    print("=" * 60)
    
    manager = LLMProviderManager()
    await manager.initialize()
    
    # Suche verschiedene Modelle
    model_ids = ['gpt-4o-mini', 'claude-3-opus', 'mistral-large']
    
    for model_id in model_ids:
        model = manager.get_model_by_id(model_id)
        
        if model:
            print(f"\n{model.model_name} ({model.model_id})")
            print("-" * 40)
            print(f"Provider: {model.provider}")
            print(f"Context Window: {model.context_window:,} Tokens" if model.context_window else "Context Window: N/A")
            print(f"Input-Kosten: ${model.cost_in}/1M Tokens" if model.cost_in else "Input-Kosten: N/A")
            print(f"Output-Kosten: ${model.cost_out}/1M Tokens" if model.cost_out else "Output-Kosten: N/A")
            
            # Berechne Beispiel-Kosten
            if model.cost_in and model.cost_out:
                # Beispiel: 10k Input, 2k Output Tokens
                example_cost = (10000 / 1_000_000 * model.cost_in) + (2000 / 1_000_000 * model.cost_out)
                print(f"Beispiel-Kosten (10k in, 2k out): ${example_cost:.4f}")
            
            # Options
            if model.options:
                print("Eigenschaften:")
                for key, value in model.options.items():
                    if isinstance(value, bool) and value:
                        print(f"   ✓ {key}")
                    elif not isinstance(value, bool):
                        print(f"   {key}: {value}")
        else:
            print(f"\n✗ Modell '{model_id}' nicht gefunden")


async def provider_statistics_example():
    """Zeigt Statistiken über alle Provider"""
    
    print("\n" + "=" * 60)
    print("Provider-Statistiken")
    print("=" * 60)
    
    manager = LLMProviderManager()
    await manager.initialize()
    
    all_models = manager.get_all_models()
    providers = manager.get_supported_providers()
    
    print(f"\nGesamt: {len(all_models)} Modelle von {len(providers)} Providern")
    print("\nModelle pro Provider:")
    
    for provider in providers:
        models = manager.get_models_by_provider(provider)
        print(f"\n{provider.upper()}: {len(models)} Modelle")
        
        # Kosten-Statistiken für diesen Provider
        models_with_cost = [m for m in models if m.cost_in is not None]
        if models_with_cost:
            avg_cost_in = sum(m.cost_in for m in models_with_cost) / len(models_with_cost)
            avg_cost_out = sum(m.cost_out for m in models_with_cost) / len(models_with_cost)
            min_cost_in = min(m.cost_in for m in models_with_cost)
            max_cost_in = max(m.cost_in for m in models_with_cost)
            
            print(f"   Durchschnittskosten Input: ${avg_cost_in:.2f}/1M Tokens")
            print(f"   Durchschnittskosten Output: ${avg_cost_out:.2f}/1M Tokens")
            print(f"   Preisspanne Input: ${min_cost_in:.2f} - ${max_cost_in:.2f}")
        
        # Context Window Statistiken
        models_with_context = [m for m in models if m.context_window is not None]
        if models_with_context:
            avg_context = sum(m.context_window for m in models_with_context) / len(models_with_context)
            max_context = max(m.context_window for m in models_with_context)
            print(f"   Durchschnittliches Context Window: {avg_context:,.0f} Tokens")
            print(f"   Größtes Context Window: {max_context:,} Tokens")


async def main():
    """Hauptfunktion - führt alle Beispiele aus"""
    
    try:
        # Basis-Nutzung
        await basic_usage_example()
        
        # Detaillierte Modell-Informationen
        await model_details_example()
        
        # Provider-Statistiken
        await provider_statistics_example()
        
    except KeyboardInterrupt:
        print("\n\nBeendet durch Benutzer")
    except Exception as e:
        print(f"\n✗ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
