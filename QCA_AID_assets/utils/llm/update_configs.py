#!/usr/bin/env python3
"""
Manual Config Update Script

Lädt die neuesten Provider-Configs von Catwalk GitHub herunter
und speichert sie lokal.

Usage:
    python update_configs.py
"""

import asyncio
import sys
from pathlib import Path

# Füge Parent-Verzeichnis zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QCA_AID_assets.utils.llm.config_updater import ConfigUpdater


async def main():
    """Hauptfunktion für manuelles Config-Update."""
    print("=" * 80)
    print("PROVIDER CONFIG UPDATE")
    print("=" * 80)
    print()
    print("Lade neueste Configs von Catwalk GitHub...")
    print()
    
    updater = ConfigUpdater()
    
    try:
        # Lade alle Configs
        updated_count = await updater.update_all_configs()
        
        if updated_count > 0:
            # Speichere Metadaten
            updater._save_metadata()
            
            print()
            print("=" * 80)
            print(f"✓ Erfolgreich {updated_count} Config(s) aktualisiert")
            print("=" * 80)
            return 0
        else:
            print()
            print("=" * 80)
            print("⚠️  Keine Configs wurden aktualisiert")
            print("=" * 80)
            return 1
            
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ Fehler beim Update: {e}")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
