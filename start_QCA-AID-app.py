#!/usr/bin/env python3
"""
Startup script für QCA-AID Webapp (Root-Launcher)
Startet das eigentliche Webapp-Skript im QCA_AID_app Ordner
"""
import sys
import subprocess
from pathlib import Path

# Fix für Unicode-Encoding auf Windows-Konsolen
if sys.platform == 'win32':
    try:
        # Setze stdout und stderr auf UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        # Fallback für ältere Python-Versionen
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def main():
    """Haupteinstiegspunkt - delegiert an QCA_AID_app/start_webapp.py"""
    app_dir = Path(__file__).parent / 'QCA_AID_app'
    start_script = app_dir / 'start_webapp.py'
    
    if not start_script.exists():
        print("Fehler: QCA_AID_app/start_webapp.py nicht gefunden")
        sys.exit(1)
    
    # Wechsle ins App-Verzeichnis und starte das Skript
    try:
        subprocess.run([sys.executable, str(start_script)], cwd=str(app_dir))
    except KeyboardInterrupt:
        print("\nWebapp beendet")
    except Exception as e:
        print(f"\nFehler beim Starten der Webapp: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
