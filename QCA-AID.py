"""
QCA-AID: Qualitative Content Analysis with AI Support
======================================================
Version: 0.10.1

Launcher-Skript für QCA-AID.
Alle Funktionalität ist in QCA-AID-assets/ organisiert.

Author: Justus Henke
Contact: justus.henke@hof.uni-halle.de
Repository: https://github.com/JustusHenke/QCA-AID
"""

import os
import sys
import asyncio

# Stelle sicher, dass das Skript-Verzeichnis im Python-Path ist
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Importiere main aus dem refactored Code
from QCA_AID_assets.main import main
from QCA_AID_assets.utils.system import patch_tkinter_for_threaded_exit

# Patch für Tkinter-Threading
patch_tkinter_for_threaded_exit()

if __name__ == "__main__":
    try:
        # Windows-spezifische Event Loop Policy setzen
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Hauptprogramm ausführen
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer beendet")
    except Exception as e:
        print(f"Fehler im Hauptprogramm: {str(e)}")
        raise
