"""
Document Conversion - Convert documents to PDF format

Duenne Kompatibilitaetsschicht ueber der neuen Ingest-Pipeline
(QCA_AID_assets.preprocessing.document_converter). Die oeffentliche API bleibt
gegenueber dem frueheren Stub unveraendert, damit ResultsExporter und main.py
weiter funktionieren.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from ...preprocessing.document_converter import ConvertError, DocumentConverter


class DocumentToPDFConverter:
    """Konvertiert Dokumente nach PDF und legt sie im Zielverzeichnis ab."""

    def __init__(self, output_dir: str, timeout: float = 120.0):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.converter = DocumentConverter(timeout=timeout)

    def convert(self, input_file: str, target_dir: Optional[str] = None) -> Optional[str]:
        """Konvertiert input_file nach <target_dir>/<stem>.pdf.

        Args:
            input_file: Quelldatei (.docx, .txt, .pdf, ...)
            target_dir: Zielverzeichnis (Default: ``self.output_dir``). Fuer
                Zwischen-PDFs, die nur annotiert und danach verworfen werden,
                gehoert hier ein temporaeres Verzeichnis hin
                (:func:`..paths.conversion_workspace`), damit keine
                unannotierten PDFs im Ausgabeverzeichnis landen.

        Returns:
            Pfad zum PDF oder None bei Fehlschlag (frueheres Verhalten).
        """
        folder = Path(target_dir) if target_dir else Path(self.output_dir)
        folder.mkdir(parents=True, exist_ok=True)

        try:
            result = self.converter.convert(Path(input_file), folder)
        except ConvertError as exc:
            print(f"Konvertierung fehlgeschlagen: {exc}")
            return None
        except Exception as exc:  # nie den Export abbrechen
            print(f"Unerwarteter Konvertierungsfehler: {exc}")
            return None

        target = folder / (Path(input_file).stem + ".pdf")
        if result.pdf_path != target:
            shutil.move(str(result.pdf_path), str(target))
        return str(target)

    def cleanup_temp_pdfs(self) -> None:
        """Entfernt Zwischenartefakte der Pipeline (fruher: temporaere PDFs)."""
        leftover = Path(self.output_dir) / "canonical.pdf"
        try:
            if leftover.exists():
                leftover.unlink()
        except OSError:
            pass


__all__ = [
    "DocumentToPDFConverter",
]