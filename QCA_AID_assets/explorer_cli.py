"""
QCA-AID Explorer CLI Entry Point
================================

Konsolen-Einstiegspunkt für QCA-AID Explorer.
Wird durch pyproject.toml [project.scripts] als `qcaaid-explorer`-Befehl registriert.
"""

import argparse
import asyncio
import os
import sys


def main():
    """CLI-Einstiegspunkt für `qcaaid-explorer`."""
    from QCA_AID_assets.__version__ import __version__, __version_date__
    from QCA_AID_assets.explorer import main as run_explorer

    parser = argparse.ArgumentParser(
        prog="qcaaid-explorer",
        description="QCA-AID Explorer: Visualisierung und Analyse von Kodierungsdaten",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Beispiele:
  qcaaid-explorer                              Interaktive Exploration
  qcaaid-explorer --non-interactive            Non-interactive Modus
  qcaaid-explorer -n --config config.json     Spezifische Config verwenden
  qcaaid-explorer --version                    Version anzeigen
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"QCA-AID Explorer {__version__} ({__version_date__})",
    )

    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Pfad zur Explorer-Config-JSON-Datei",
    )

    parser.add_argument(
        "-n",
        "--non-interactive",
        action="store_true",
        help="Keine interaktiven Eingaben (für Skripte/Pipelines)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        metavar="PATH",
        help="Output-Verzeichnis (überschreibt Config-Einstellung)",
    )

    args = parser.parse_args()

    cli_args = {
        "config": args.config,
        "non_interactive": args.non_interactive,
        "output_dir": args.output_dir,
    }

    try:
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        asyncio.run(run_explorer(cli_args=cli_args))

    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer beendet")
    except Exception as e:
        print(f"Fehler im Hauptprogramm: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
