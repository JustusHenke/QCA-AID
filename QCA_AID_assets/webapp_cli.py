"""
QCA-AID Webapp CLI Entry Point
==============================

Konsolen-Einstiegspunkt für die QCA-AID Streamlit-Webapp.
Wird durch pyproject.toml [project.scripts] als `qcaaid-webapp`-Befehl registriert.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """CLI-Einstiegspunkt für `qcaaid-webapp`."""
    # .env-Datei laden
    from dotenv import load_dotenv

    for dotenv_path in [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path.home() / ".environ.env",
    ]:
        if dotenv_path.is_file():
            load_dotenv(dotenv_path, override=False)
            print(f"📄 .env gefunden: {dotenv_path}")
            break

    # Fix für Unicode-Encoding auf Windows-Konsolen
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            import codecs

            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
            sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

    # Webapp-Skript finden
    webapp_path = Path(__file__).resolve().parent.parent / "QCA_AID_app" / "webapp.py"

    if not webapp_path.exists():
        # Fallback: Suche im installierten Paket
        import QCA_AID_app

        webapp_path = Path(QCA_AID_app.__file__).parent / "webapp.py"

    if not webapp_path.exists():
        print("Fehler: webapp.py nicht gefunden")
        sys.exit(1)

    print("QCA-AID Webapp wird gestartet...")
    print("=" * 50)
    print("Starte Streamlit auf http://127.0.0.1:8501")
    print("Drücke Ctrl+C zum Beenden")
    print("=" * 50)

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(webapp_path)],
            cwd=str(webapp_path.parent),
        )
    except KeyboardInterrupt:
        print("\nWebapp beendet")
    except Exception as e:
        print(f"\nFehler beim Starten: {e}")
        print("Stelle sicher, dass streamlit installiert ist: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
