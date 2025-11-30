"""
Beispiel: Filter-Nutzung des LLM Provider Managers

Dieses Beispiel zeigt verschiedene Filter-Möglichkeiten:
- Nach Provider filtern
- Nach Kosten filtern
- Nach Context Window filtern
- Nach Capabilities filtern
- Kombinierte Filter
"""

import asyncio
import logging
from pathlib import Path
import sys

# Füge Parent-Verzeichnis zum Path hinzu für Import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from QCA_AID_assets.utils.llm.provider_manager import LLMProviderManager


# Logging aktivieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def filter_by_provider_example(manager):
    """Filtert Modelle nach Provider"""
    
    print("\n" + "=" * 60)
    print("1. Nach Provider filtern")
    print("=" * 60)
    
    providers = ['openai', 'anthropic', 'mistral', 'local']
    
    for provider in providers:
        models = manager.get_models_by_provider(provider)
        print(f"\n{provider.upper()}: {len(models)} Modelle")
        
        if models:
            # Zeige erste 3 Modelle
            for model in models[:3]:
                cost_str = f"${model.cost_in}/{model.cost_out}" if model.cost_in else "N/A"
                context_str = f"{model.context_window:,}" if model.context_window else "N/A"
                print(f"   - {model.model_name}")
                print(f"     ID: {model.model_id}")
                print(f"     Kosten: {cost_str} | Context: {context_str}")


async def filter_by_cost_example(manager):
    """Filtert Modelle nach Kosten"""
    
    print("\n" + "=" * 60)
    print("2. Nach Kosten filtern")
    print("=" * 60)
    
    # Günstige Modelle (max $1/1M Input-Tokens)
    print("\nGünstige Modelle (max $1.00 Input, max $5.00 Output):")
    cheap_models = manager.filter_models(
        max_cost_in=1.0,
        max_cost_out=5.0
    )
    
    if cheap_models:
        print(f"Gefunden: {len(cheap_models)} Modelle")
        for model in cheap_models[:5]:
            print(f"   - {model.model_name} ({model.provider})")
            print(f"     ${model.cost_in:.2f} in / ${model.cost_out:.2f} out")
    else:
        print("   Keine Modelle gefunden")
    
    # Mittelpreisige Modelle
    print("\nMittelpreisige Modelle ($1-$5 Input):")
    mid_models = manager.filter_models(
        max_cost_in=5.0
    )
    # Filtere manuell für min $1
    mid_models = [m for m in mid_models if m.cost_in and m.cost_in >= 1.0]
    
    if mid_models:
        print(f"Gefunden: {len(mid_models)} Modelle")
        for model in mid_models[:5]:
            print(f"   - {model.model_name} ({model.provider})")
            print(f"     ${model.cost_in:.2f} in / ${model.cost_out:.2f} out")
    else:
        print("   Keine Modelle gefunden")


async def filter_by_context_example(manager):
    """Filtert Modelle nach Context Window"""
    
    print("\n" + "=" * 60)
    print("3. Nach Context Window filtern")
    print("=" * 60)
    
    # Modelle mit großem Context (min 100k Tokens)
    print("\nModelle mit großem Context Window (min 100k Tokens):")
    large_context = manager.filter_models(min_context=100000)
    
    if large_context:
        print(f"Gefunden: {len(large_context)} Modelle")
        # Sortiere nach Context Window (größer zuerst)
        large_context.sort(key=lambda m: m.context_window or 0, reverse=True)
        
        for model in large_context[:5]:
            cost_str = f"${model.cost_in}/{model.cost_out}" if model.cost_in else "N/A"
            print(f"   - {model.model_name} ({model.provider})")
            print(f"     Context: {model.context_window:,} Tokens | Kosten: {cost_str}")
    else:
        print("   Keine Modelle gefunden")
    
    # Modelle mit sehr großem Context (min 200k Tokens)
    print("\nModelle mit sehr großem Context Window (min 200k Tokens):")
    xlarge_context = manager.filter_models(min_context=200000)
    
    if xlarge_context:
        print(f"Gefunden: {len(xlarge_context)} Modelle")
        xlarge_context.sort(key=lambda m: m.context_window or 0, reverse=True)
        
        for model in xlarge_context:
            cost_str = f"${model.cost_in}/{model.cost_out}" if model.cost_in else "N/A"
            print(f"   - {model.model_name} ({model.provider})")
            print(f"     Context: {model.context_window:,} Tokens | Kosten: {cost_str}")
    else:
        print("   Keine Modelle gefunden")


async def filter_by_capabilities_example(manager):
    """Filtert Modelle nach Capabilities"""
    
    print("\n" + "=" * 60)
    print("4. Nach Capabilities filtern")
    print("=" * 60)
    
    # Modelle mit Reasoning
    print("\nModelle mit Reasoning-Fähigkeit:")
    reasoning_models = manager.filter_models(capabilities=['can_reason'])
    
    if reasoning_models:
        print(f"Gefunden: {len(reasoning_models)} Modelle")
        for model in reasoning_models[:5]:
            cost_str = f"${model.cost_in}/{model.cost_out}" if model.cost_in else "N/A"
            print(f"   - {model.model_name} ({model.provider})")
            print(f"     Kosten: {cost_str}")
            
            # Zeige Reasoning-Level falls vorhanden
            if 'reasoning_levels' in model.options:
                levels = model.options['reasoning_levels']
                print(f"     Reasoning Levels: {', '.join(levels)}")
    else:
        print("   Keine Modelle gefunden")
    
    # Modelle mit Attachment-Support
    print("\nModelle mit Attachment-Support:")
    attachment_models = manager.filter_models(capabilities=['supports_attachments'])
    
    if attachment_models:
        print(f"Gefunden: {len(attachment_models)} Modelle")
        for model in attachment_models[:5]:
            cost_str = f"${model.cost_in}/{model.cost_out}" if model.cost_in else "N/A"
            print(f"   - {model.model_name} ({model.provider})")
            print(f"     Kosten: {cost_str}")
    else:
        print("   Keine Modelle gefunden")


async def combined_filters_example(manager):
    """Kombiniert mehrere Filter"""
    
    print("\n" + "=" * 60)
    print("5. Kombinierte Filter")
    print("=" * 60)
    
    # Beispiel 1: Günstige OpenAI-Modelle mit großem Context
    print("\nGünstige OpenAI-Modelle mit großem Context:")
    print("(Provider: openai, max $1 Input, min 100k Context)")
    
    filtered = manager.filter_models(
        provider='openai',
        max_cost_in=1.0,
        min_context=100000
    )
    
    if filtered:
        print(f"Gefunden: {len(filtered)} Modelle")
        for model in filtered:
            print(f"   - {model.model_name}")
            print(f"     Kosten: ${model.cost_in} in / ${model.cost_out} out")
            print(f"     Context: {model.context_window:,} Tokens")
    else:
        print("   Keine Modelle gefunden")
    
    # Beispiel 2: Reasoning-Modelle mit moderaten Kosten
    print("\nReasoning-Modelle mit moderaten Kosten:")
    print("(Reasoning: ja, max $5 Input, max $20 Output)")
    
    filtered = manager.filter_models(
        capabilities=['can_reason'],
        max_cost_in=5.0,
        max_cost_out=20.0
    )
    
    if filtered:
        print(f"Gefunden: {len(filtered)} Modelle")
        # Sortiere nach Kosten
        filtered.sort(key=lambda m: m.cost_in or float('inf'))
        
        for model in filtered[:5]:
            print(f"   - {model.model_name} ({model.provider})")
            print(f"     Kosten: ${model.cost_in} in / ${model.cost_out} out")
            print(f"     Context: {model.context_window:,} Tokens" if model.context_window else "     Context: N/A")
    else:
        print("   Keine Modelle gefunden")
    
    # Beispiel 3: Beste Modelle für Coding
    print("\nBeste Modelle für Coding:")
    print("(Reasoning: ja, min 100k Context, sortiert nach Kosten)")
    
    filtered = manager.filter_models(
        capabilities=['can_reason'],
        min_context=100000
    )
    
    if filtered:
        # Sortiere nach Kosten (günstigste zuerst)
        filtered.sort(key=lambda m: m.cost_in or float('inf'))
        
        print(f"Gefunden: {len(filtered)} Modelle")
        print("\nTop 5 (günstigste):")
        for model in filtered[:5]:
            print(f"   - {model.model_name} ({model.provider})")
            print(f"     Kosten: ${model.cost_in} in / ${model.cost_out} out")
            print(f"     Context: {model.context_window:,} Tokens")


async def custom_filter_logic_example(manager):
    """Zeigt benutzerdefinierte Filter-Logik"""
    
    print("\n" + "=" * 60)
    print("6. Benutzerdefinierte Filter-Logik")
    print("=" * 60)
    
    all_models = manager.get_all_models()
    
    # Beispiel 1: Bestes Preis-Leistungs-Verhältnis
    print("\nBestes Preis-Leistungs-Verhältnis:")
    print("(Context/Kosten-Ratio)")
    
    # Berechne Ratio für Modelle mit Kosten und Context
    models_with_data = [
        m for m in all_models 
        if m.cost_in and m.context_window
    ]
    
    if models_with_data:
        # Berechne Tokens pro Dollar
        for model in models_with_data:
            model.value_ratio = model.context_window / model.cost_in
        
        # Sortiere nach Ratio
        models_with_data.sort(key=lambda m: m.value_ratio, reverse=True)
        
        print(f"Top 5 Modelle:")
        for model in models_with_data[:5]:
            print(f"   - {model.model_name} ({model.provider})")
            print(f"     Context: {model.context_window:,} Tokens")
            print(f"     Kosten: ${model.cost_in} in")
            print(f"     Ratio: {model.value_ratio:,.0f} Tokens/$")
    
    # Beispiel 2: Modelle für lange Dokumente
    print("\nModelle für lange Dokumente:")
    print("(min 128k Context, sortiert nach Output-Kosten)")
    
    long_doc_models = [
        m for m in all_models
        if m.context_window and m.context_window >= 128000
        and m.cost_out is not None
    ]
    
    if long_doc_models:
        long_doc_models.sort(key=lambda m: m.cost_out)
        
        print(f"Gefunden: {len(long_doc_models)} Modelle")
        for model in long_doc_models[:5]:
            print(f"   - {model.model_name} ({model.provider})")
            print(f"     Context: {model.context_window:,} Tokens")
            print(f"     Output-Kosten: ${model.cost_out}/1M Tokens")


async def main():
    """Hauptfunktion - führt alle Filter-Beispiele aus"""
    
    print("=" * 60)
    print("LLM Provider Manager - Filter-Beispiele")
    print("=" * 60)
    
    # Manager initialisieren
    print("\nInitialisiere Manager...")
    manager = LLMProviderManager()
    
    try:
        await manager.initialize()
        print("✓ Manager erfolgreich initialisiert")
        
        # Führe alle Beispiele aus
        await filter_by_provider_example(manager)
        await filter_by_cost_example(manager)
        await filter_by_context_example(manager)
        await filter_by_capabilities_example(manager)
        await combined_filters_example(manager)
        await custom_filter_logic_example(manager)
        
        print("\n" + "=" * 60)
        print("Alle Filter-Beispiele abgeschlossen!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nBeendet durch Benutzer")
    except Exception as e:
        print(f"\n✗ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
