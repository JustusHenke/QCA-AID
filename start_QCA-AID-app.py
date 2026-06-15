#!/usr/bin/env python3
"""
Startup script für QCA-AID Webapp (Root-Launcher)
Startet das eigentliche Webapp-Skript im QCA_AID_app Ordner
"""

import os
import subprocess
import sys
from pathlib import Path

# ============================================================
# .env-Datei laden (VOR allem anderen!)
# ============================================================
# Durchsucht mehrere Orte nach einer .env-Datei:
# 1. Aktuelles Arbeitsverzeichnis (das vom Benutzer gewählte Projekt)
# 2. QCA-AID Repository-Root
# 3. Home-Verzeichnis (~/.environ.env)
_env_loaded = False
for _dotenv_path in [
    Path.cwd() / ".env",
    Path(__file__).resolve().parent / ".env",
    Path.home() / ".environ.env",
]:
    if _dotenv_path.is_file():
        print(f"📄 .env gefunden: {_dotenv_path}")
        try:
            # python-dotenv bevorzugen, falls installiert
            from dotenv import load_dotenv

            load_dotenv(_dotenv_path, override=False)
            _env_loaded = True
            break
        except ImportError:
            # Fallback: manuelles Einlesen
            pass

# Fallback: manuelles Einlesen falls python-dotenv nicht installiert
if not _env_loaded:
    for _dotenv_path in [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path.home() / ".environ.env",
    ]:
        if _dotenv_path.is_file():
            print(f"📄 .env manuell geladen: {_dotenv_path}")
            try:
                with open(_dotenv_path, "r", encoding="utf-8") as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if _line and not _line.startswith("#"):
                            _key, _sep, _val = _line.partition("=")
                            if _sep and _key.strip():
                                os.environ.setdefault(_key.strip(), _val.strip())
                _env_loaded = True
            except Exception:
                pass
            break

# Fix für Unicode-Encoding auf Windows-Konsolen
if sys.platform == "win32":
    try:
        # Setze stdout und stderr auf UTF-8
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        # Fallback für ältere Python-Versionen
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def main():
    """Haupteinstiegspunkt - delegiert an QCA_AID_app/start_webapp.py"""
    app_dir = Path(__file__).parent / "QCA_AID_app"
    start_script = app_dir / "start_webapp.py"

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


if __name__ == "__main__":
    main()
