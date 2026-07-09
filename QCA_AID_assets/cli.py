"""
QCA-AID CLI Entry Point
=======================

Konsolen-Einstiegspunkt für QCA-AID Hauptanalyse.
Wird durch pyproject.toml [project.scripts] als `qcaaid`-Befehl registriert.
"""

import argparse
import asyncio
import os
import sys


def main():
    """CLI-Einstiegspunkt für `qcaaid`."""
    from QCA_AID_assets.__version__ import __version__, __version_date__
    from QCA_AID_assets.main import main as run_analysis
    from QCA_AID_assets.utils.system import patch_tkinter_for_threaded_exit

    parser = argparse.ArgumentParser(
        prog="qcaaid",
        description="QCA-AID: Qualitative Content Analysis mit KI-Unterstützung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Beispiele:
  qcaaid                                    Interaktive Analyse im aktuellen Projekt
  qcaaid --non-interactive                  Analyse ohne interaktive Eingaben
  qcaaid -n --mode inductive                Non-interactive, induktiver Modus
  qcaaid --project-root /pfad/zum/projekt   Anderes Projektverzeichnis
  qcaaid --config /pfad/codebook.json       Spezifisches Codebook verwenden
  qcaaid --version                          Version anzeigen
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"QCA-AID {__version__} ({__version_date__})",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        metavar="PATH",
        help="Projekt-Verzeichnis (überschreibt .qca-aid-project.json)",
    )

    parser.add_argument(
        "--mode",
        choices=["deductive", "abductive", "inductive", "grounded"],
        help="Analysemodus (überschreibt Codebook-Einstellung)",
    )

    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Pfad zur Codebook-JSON-Datei",
    )

    parser.add_argument(
        "-n",
        "--non-interactive",
        action="store_true",
        help="Keine interaktiven Eingaben (für Skripte/Pipelines)",
    )

    parser.add_argument(
        "--no-manual",
        action="store_true",
        help="Manuelles Kodieren deaktivieren",
    )

    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="PDF-Annotation deaktivieren",
    )

    parser.add_argument(
        "--use-saved-codebook",
        action="store_true",
        help="Gespeichertes induktives Codebook laden (falls vorhanden)",
    )

    args = parser.parse_args()

    # Patch für Tkinter-Threading
    patch_tkinter_for_threaded_exit()

    cli_args = {
        "project_root": args.project_root,
        "mode": args.mode,
        "config": args.config,
        "non_interactive": args.non_interactive,
        "no_manual": args.no_manual,
        "no_pdf": args.no_pdf,
        "use_saved_codebook": args.use_saved_codebook,
    }

    try:
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        asyncio.run(run_analysis(cli_args=cli_args))

    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer beendet")
    except Exception as e:
        print(f"Fehler im Hauptprogramm: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
