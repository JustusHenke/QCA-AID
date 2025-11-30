"""
Beispiel: Pricing-Overrides

Dieses Beispiel zeigt wie man eigene Preisinformationen für Modelle definiert:
- Erstellen einer pricing_overrides.json
- Anwenden von Overrides
- Vergleich vor/nach Override
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


async def without_overrides_example():
    """Zeigt Standard-Preise ohne Overrides"""
    
    print("=" * 60)
    print("1. Standard-Preise (ohne Overrides)")
    print("=" * 60)
    
    manager = LLMProviderManager()
    await manager.initialize()
    
    # Zeige Standard-Preise für einige Modelle
    model_ids = ['gpt-4o-mini', 'gpt-4o', 'claude-3-opus']
    
    print("\nStandard-Preise:")
    for model_id in model_ids:
        model = manager.get_model_by_id(model_id)
        if model:
            print(f"\n{model.model_name} ({model.model_id}):")
            print(f"   Input:  ${model.cost_in}/1M Tokens" if model.cost_in else "   Input:  N/A")
            print(f"   Output: ${model.cost_out}/1M Tokens" if model.cost_out else "   Output: N/A")
        else:
            print(f"\n✗ Modell '{model_id}' nicht gefunden")


async def with_overrides_example():
    """Zeigt Preise mit Overrides"""
    
    print("\n" + "=" * 60)
    print("2. Preise mit Overrides")
    print("=" * 60)
    
    # Erstelle temporäres Verzeichnis für Config
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Erstelle pricing_overrides.json
        overrides = {
            "gpt-4o-mini": {
                "cost_in": 0.10,
                "cost_out": 0.50
            },
            "gpt-4o": {
                "cost_in": 2.0,
                "cost_out": 8.0
            },
            "claude-3-opus": {
                "cost_in": 10.0,
                "cost_out": 30.0
            }
        }
        
        override_path = Path(temp_dir) / "pricing_overrides.json"
        with open(override_path, 'w') as f:
            json.dump(overrides, f, indent=2)
        
        print(f"\n✓ Erstellt: {override_path}")
        print("\nOverride-Inhalt:")
        print(json.dumps(overrides, indent=2))
        
        # Manager mit Config-Verzeichnis initialisieren
        print(f"\nInitialisiere Manager mit config_dir={temp_dir}...")
        manager = LLMProviderManager(config_dir=temp_dir)
        await manager.initialize()
        
        # Zeige überschriebene Preise
        model_ids = ['gpt-4o-mini', 'gpt-4o', 'claude-3-opus']
        
        print("\nÜberschriebene Preise:")
        for model_id in model_ids:
            model = manager.get_model_by_id(model_id)
            if model:
                print(f"\n{model.model_name} ({model.model_id}):")
                print(f"   Input:  ${model.cost_in}/1M Tokens" if model.cost_in else "   Input:  N/A")
                print(f"   Output: ${model.cost_out}/1M Tokens" if model.cost_out else "   Output: N/A")
            else:
                print(f"\n✗ Modell '{model_id}' nicht gefunden")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n✓ Temporäres Verzeichnis gelöscht: {temp_dir}")


async def comparison_example():
    """Vergleicht Preise vor und nach Override"""
    
    print("\n" + "=" * 60)
    print("3. Vergleich: Standard vs. Override")
    print("=" * 60)
    
    # Manager ohne Overrides
    manager_standard = LLMProviderManager()
    await manager_standard.initialize()
    
    # Manager mit Overrides
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Erstelle Overrides mit deutlichen Unterschieden
        overrides = {
            "gpt-4o-mini": {
                "cost_in": 0.05,  # Stark reduziert
                "cost_out": 0.25
            },
            "claude-3-opus": {
                "cost_in": 5.0,   # Halbiert
                "cost_out": 15.0
            }
        }
        
        override_path = Path(temp_dir) / "pricing_overrides.json"
        with open(override_path, 'w') as f:
            json.dump(overrides, f, indent=2)
        
        manager_override = LLMProviderManager(config_dir=temp_dir)
        await manager_override.initialize()
        
        # Vergleiche Preise
        model_ids = ['gpt-4o-mini', 'claude-3-opus']
        
        for model_id in model_ids:
            model_std = manager_standard.get_model_by_id(model_id)
            model_ovr = manager_override.get_model_by_id(model_id)
            
            if model_std and model_ovr:
                print(f"\n{model_std.model_name} ({model_id}):")
                print("-" * 40)
                
                # Input-Kosten
                if model_std.cost_in and model_ovr.cost_in:
                    diff_in = model_ovr.cost_in - model_std.cost_in
                    pct_in = (diff_in / model_std.cost_in) * 100
                    print(f"Input-Kosten:")
                    print(f"   Standard: ${model_std.cost_in:.2f}/1M")
                    print(f"   Override: ${model_ovr.cost_in:.2f}/1M")
                    print(f"   Differenz: ${diff_in:+.2f} ({pct_in:+.1f}%)")
                
                # Output-Kosten
                if model_std.cost_out and model_ovr.cost_out:
                    diff_out = model_ovr.cost_out - model_std.cost_out
                    pct_out = (diff_out / model_std.cost_out) * 100
                    print(f"Output-Kosten:")
                    print(f"   Standard: ${model_std.cost_out:.2f}/1M")
                    print(f"   Override: ${model_ovr.cost_out:.2f}/1M")
                    print(f"   Differenz: ${diff_out:+.2f} ({pct_out:+.1f}%)")
                
                # Beispiel-Berechnung
                if model_std.cost_in and model_std.cost_out:
                    # 100k Input, 20k Output
                    cost_std = (100000/1_000_000 * model_std.cost_in) + (20000/1_000_000 * model_std.cost_out)
                    cost_ovr = (100000/1_000_000 * model_ovr.cost_in) + (20000/1_000_000 * model_ovr.cost_out)
                    savings = cost_std - cost_ovr
                    
                    print(f"Beispiel (100k in, 20k out):")
                    print(f"   Standard: ${cost_std:.4f}")
                    print(f"   Override: ${cost_ovr:.4f}")
                    print(f"   Ersparnis: ${savings:.4f}")
    
    finally:
        shutil.rmtree(temp_dir)


async def invalid_overrides_example():
    """Zeigt Verhalten bei ungültigen Overrides"""
    
    print("\n" + "=" * 60)
    print("4. Ungültige Overrides")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Erstelle Overrides mit nicht-existierenden Modellen
        overrides = {
            "gpt-4o-mini": {
                "cost_in": 0.10,
                "cost_out": 0.50
            },
            "non-existent-model": {
                "cost_in": 1.0,
                "cost_out": 2.0
            },
            "another-fake-model": {
                "cost_in": 5.0,
                "cost_out": 10.0
            }
        }
        
        override_path = Path(temp_dir) / "pricing_overrides.json"
        with open(override_path, 'w') as f:
            json.dump(overrides, f, indent=2)
        
        print(f"\nOverride-Datei enthält:")
        print(f"   ✓ 1 gültiges Modell (gpt-4o-mini)")
        print(f"   ✗ 2 nicht-existierende Modelle")
        
        # Manager initialisieren
        print(f"\nInitialisiere Manager...")
        manager = LLMProviderManager(config_dir=temp_dir)
        await manager.initialize()
        
        print("\n✓ Manager erfolgreich initialisiert")
        print("   (Ungültige Overrides wurden ignoriert)")
        
        # Prüfe gültiges Override
        model = manager.get_model_by_id('gpt-4o-mini')
        if model:
            print(f"\nGültiges Override angewendet:")
            print(f"   {model.model_name}: ${model.cost_in} / ${model.cost_out}")
        
        # Prüfe dass ungültige Overrides ignoriert wurden
        print(f"\nUngültige Overrides wurden ignoriert:")
        print(f"   ✓ System läuft normal weiter")
        print(f"   ✓ Keine Fehler oder Abstürze")
    
    finally:
        shutil.rmtree(temp_dir)


async def create_override_template():
    """Erstellt eine Vorlage für pricing_overrides.json"""
    
    print("\n" + "=" * 60)
    print("5. Vorlage für pricing_overrides.json erstellen")
    print("=" * 60)
    
    # Initialisiere Manager um verfügbare Modelle zu bekommen
    manager = LLMProviderManager()
    await manager.initialize()
    
    # Erstelle Template mit einigen populären Modellen
    template = {
        "gpt-4o-mini": {
            "cost_in": 0.15,
            "cost_out": 0.60
        },
        "gpt-4o": {
            "cost_in": 2.50,
            "cost_out": 10.00
        },
        "claude-3-opus": {
            "cost_in": 15.00,
            "cost_out": 75.00
        },
        "claude-3-sonnet": {
            "cost_in": 3.00,
            "cost_out": 15.00
        },
        "mistral-large": {
            "cost_in": 2.00,
            "cost_out": 6.00
        }
    }
    
    # Speichere Template
    template_path = Path("pricing_overrides_template.json")
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"\n✓ Vorlage erstellt: {template_path}")
    print("\nInhalt:")
    print(json.dumps(template, indent=2))
    
    print("\nAnleitung:")
    print("1. Kopieren Sie pricing_overrides_template.json")
    print("2. Benennen Sie um zu pricing_overrides.json")
    print("3. Passen Sie die Preise an Ihre Bedürfnisse an")
    print("4. Platzieren Sie die Datei im config_dir")
    print("5. Initialisieren Sie den Manager mit config_dir Parameter")


async def main():
    """Hauptfunktion - führt alle Pricing-Override-Beispiele aus"""
    
    try:
        # Standard-Preise
        await without_overrides_example()
        
        # Mit Overrides
        await with_overrides_example()
        
        # Vergleich
        await comparison_example()
        
        # Ungültige Overrides
        await invalid_overrides_example()
        
        # Template erstellen
        await create_override_template()
        
        print("\n" + "=" * 60)
        print("Alle Pricing-Override-Beispiele abgeschlossen!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nBeendet durch Benutzer")
    except Exception as e:
        print(f"\n✗ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
