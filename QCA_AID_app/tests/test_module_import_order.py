"""
Regressionsschutz gegen zirkuläre Importe
==========================================
`analysis_manager` importierte `ResultsExporter` auf Modulebene, während
`results_exporter` seinerseits `InductiveCoder` aus `analysis` importierte.
Dadurch schlug

    python -c "import QCA_AID_assets.export.results_exporter"

fehl, solange nicht vorher ein analysis-Modul geladen war (ImportError:
partially initialized module). Im App-Flow fiel das nicht auf, weil main.py
analysis zuerst importiert - jedes Skript, das nur den Exporter nutzt, brach.

Die Tests starten für jedes Modul einen frischen Interpreter, damit die
Importreihenfolge wirklich isoliert ist.
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STANDALONE_MODULES = [
    "QCA_AID_assets.export.results_exporter",
    "QCA_AID_assets.analysis.analysis_manager",
    "QCA_AID_assets.export",
    "QCA_AID_assets.analysis",
]


@pytest.mark.parametrize("module", STANDALONE_MODULES)
def test_module_imports_standalone(module):
    """Jedes Modul muss ohne vorherige Importe importierbar sein."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"Import von {module} schlug fehl:\n{result.stderr}"
    )


def test_exporter_is_usable_after_standalone_import():
    """Exporter muss ohne analysis-Import instanziierbar sein (typischer Skript-Fall)."""
    code = (
        "from QCA_AID_assets.export.results_exporter import ResultsExporter\n"
        "exporter = ResultsExporter(output_dir='.', attribute_labels={})\n"
        "assert exporter.analysis_manager is None\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_no_top_level_export_import_in_analysis_manager():
    """analysis_manager darf results_exporter nicht auf Modulebene importieren."""
    source = (
        PROJECT_ROOT / "QCA_AID_assets" / "analysis" / "analysis_manager.py"
    ).read_text(encoding="utf-8")

    for line in source.splitlines():
        # nur Modulebene: lokale Importe in Funktionen sind erwünscht
        if not line.startswith(("import ", "from ")):
            continue
        assert "export.results_exporter" not in line, (
            "Top-Level-Import von export.results_exporter gefunden - "
            "das erzeugt den zirkulären Import wieder"
        )
