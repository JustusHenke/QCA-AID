"""Ablageorte und Namensstempel fuer Export-Artefakte.

Eine Regel fuer alle PDF-Exporte: annotierte PDFs landen in einem eigenen
Unterordner und tragen denselben Namensstempel wie die Excel-Ausgabe
(``QCA-AID_Analysis_<modus>_<stamp>.xlsx``). Zwischen-PDFs, die nur zur
Annotation erzeugt werden, gehoeren nicht in das Ausgabeverzeichnis -
dafuer gibt es :func:`conversion_workspace` (temporaeres Arbeitsverzeichnis).
"""

import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

#: Unterordner im Output-Verzeichnis fuer annotierte PDFs.
ANNOTATED_SUBDIR = "Annotated"

#: Namensstempel-Format (identisch zur Excel-Ausgabe).
STAMP_FORMAT = "%Y%m%d_%H%M%S"

#: Namensbestandteil zwischen Dokumentname und Stempel.
ANNOTATED_SUFFIX = "_QCA_annotiert"


def export_stamp(moment: Optional[datetime] = None) -> str:
    """Liefert den gemeinsamen Namensstempel eines Export-Laufs."""
    return (moment or datetime.now()).strftime(STAMP_FORMAT)


def annotated_dir(output_dir) -> Path:
    """Verzeichnis fuer annotierte PDFs (wird angelegt)."""
    folder = Path(output_dir) / ANNOTATED_SUBDIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def annotated_pdf_path(output_dir, stem: str, stamp: Optional[str] = None) -> Path:
    """Zielpfad eines annotierten PDFs: ``<output>/Annotated/<stem>_QCA_annotiert_<stamp>.pdf``.

    Args:
        output_dir: Ausgabeverzeichnis des Laufs
        stem: Dateiname ohne Endung
        stamp: Stempel des Laufs (Default: jetzt)
    """
    stamp = stamp or export_stamp()
    return annotated_dir(output_dir) / f"{stem}{ANNOTATED_SUFFIX}_{stamp}.pdf"


@contextmanager
def conversion_workspace(output_dir=None) -> Iterator[Path]:
    """Temporaeres Arbeitsverzeichnis fuer PDF-Konvertierungen.

    Liegt im System-Temp (nicht im Ausgabeverzeichnis), damit die nur zur
    Annotation erzeugten PDFs nach dem Lauf verschwinden. ``output_dir`` wird
    ignoriert, bleibt aber im Signatur, damit Aufrufer nichts umbauen muessen.
    """
    del output_dir  # bewusst nicht genutzt: nichts im Output ablegen
    folder = tempfile.mkdtemp(prefix="qca_aid_convert_")
    try:
        yield Path(folder)
    finally:
        for root, _dirs, files in os.walk(folder, topdown=False):
            for name in files:
                try:
                    os.unlink(os.path.join(root, name))
                except OSError:
                    pass
        try:
            os.rmdir(folder)
        except OSError:
            pass


__all__ = [
    "ANNOTATED_SUBDIR",
    "STAMP_FORMAT",
    "ANNOTATED_SUFFIX",
    "export_stamp",
    "annotated_dir",
    "annotated_pdf_path",
    "conversion_workspace",
]
