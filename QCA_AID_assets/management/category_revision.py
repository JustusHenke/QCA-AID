"""
Kategorien-Revisions-Manager für QCA-AID
=========================================
Verwaltet Revisionen und Änderungen am Kategoriensystem.
"""

import json
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from openpyxl.styles import Font, PatternFill, Alignment

from ..core.data_models import CategoryChange
from ..core.config import FORSCHUNGSFRAGE, KODIERREGELN, CONFIG
from ..QCA_Prompts import QCAPrompts


class CategoryRevisionManager:
    """
    Manages the iterative refinement of categories during qualitative content analysis.
    
    This class implements Mayring's approach to category revision, ensuring systematic
    documentation of changes, validation of the category system, and maintenance of
    methodological quality standards throughout the analysis process.
    """
    
    def __init__(self, output_dir: str, config: Dict):
        """
        Initializes the CategoryRevisionManager with necessary tracking mechanisms.
        
        Args:
            output_dir (str): Directory for storing revision documentation
        """
        self.config = config
        self.output_dir = output_dir
        self.changes: List[CategoryChange] = []
        self.revision_log_path = os.path.join(output_dir, "category_revisions.json")
        
        # Define quality thresholds for category system
        # self.validator = CategoryValidator(config)

        # Load existing revision history if available
        self._load_revision_history()

        # Prompt-Handler hinzuFügen
        self.prompt_handler = QCAPrompts(
            forschungsfrage=FORSCHUNGSFRAGE,
            kodierregeln=KODIERREGELN,
            deduktive_kategorien=CONFIG.get('DEDUKTIVE_KATEGORIEN', {})
        )

    def _load_revision_history(self) -> None:
        """
        Loads the existing revision history from the JSON file if available.
        This method is called during initialization to restore previous category changes.
        """
        try:
            if os.path.exists(self.revision_log_path):
                with open(self.revision_log_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    # Convert the loaded dictionary data back into CategoryChange objects
                    self.changes = [
                        CategoryChange(
                            category_name=change['category_name'],
                            change_type=change['change_type'],
                            description=change['description'],
                            timestamp=change['timestamp'],
                            old_value=change.get('old_value'),
                            new_value=change.get('new_value'),
                            affected_codings=change.get('affected_codings'),
                            justification=change.get('justification', '')
                        )
                        for change in history
                    ]
                print(f"Loaded revision history: {len(self.changes)} changes")
            else:
                print("No existing revision history found - starting fresh")
                self.changes = []
                
        except Exception as e:
            print(f"Error loading revision history: {str(e)}")
            print("Starting with empty revision history")
            self.changes = []

    def _export_revision_history(self, writer, changes: List['CategoryChange']) -> None:
        """
        Exportiert die Revisionshistorie in ein separates Excel-Sheet.
        
        Args:
            writer: Excel Writer Objekt
            changes: Liste der KategorieÄnderungen
        """
        try:
            # Erstelle DataFrame fuer Revisionshistorie
            revision_data = []
            for change in changes:
                # Erstelle lesbares Änderungsdatum
                change_date = datetime.fromisoformat(change.timestamp)
                
                # Bereite die betroffenen Kodierungen auf
                affected_codings = (
                    ', '.join(change.affected_codings)
                    if change.affected_codings
                    else 'Keine'
                )
                
                # Sammle Änderungsdetails
                if change.old_value and change.new_value:
                    details = []
                    for key in set(change.old_value.keys()) | set(change.new_value.keys()):
                        old = change.old_value.get(key, 'Nicht vorhanden')
                        new = change.new_value.get(key, 'Nicht vorhanden')
                        if old != new:
                            details.append(f"{key}: {old} -> {new}")
                    details_str = '\n'.join(details)
                else:
                    details_str = ''

                revision_data.append({
                    'Datum': change_date.strftime('%Y-%m-%d %H:%M'),
                    'Kategorie': change.category_name,
                    'Art der Änderung': change.change_type,
                    'Beschreibung': change.description,
                    'Details': details_str,
                    'BegrÜndung': change.justification,
                    'Betroffene Kodierungen': affected_codings
                })
            
            if revision_data:
                df_revisions = pd.DataFrame(revision_data)
                sheet_name = 'Revisionshistorie'
                
                # Erstelle Sheet falls nicht vorhanden
                if sheet_name not in writer.sheets:
                    writer.book.create_sheet(sheet_name)
                    
                # Exportiere Daten
                df_revisions.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Formatiere Worksheet
                worksheet = writer.sheets[sheet_name]
                
                # Setze Spaltenbreiten
                column_widths = {
                    'A': 20,  # Datum
                    'B': 25,  # Kategorie
                    'C': 15,  # Art der Änderung
                    'D': 50,  # Beschreibung
                    'E': 50,  # Details
                    'F': 50,  # BegrÜndung
                    'G': 40   # Betroffene Kodierungen
                }
                
                for col, width in column_widths.items():
                    worksheet.column_dimensions[col].width = width
                    
                # Formatiere Überschriften
                for cell in worksheet[1]:
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color='EEEEEE', end_color='EEEEEE', fill_type='solid')
                
                # Aktiviere Zeilenumbruch fuer lange Texte
                for row in worksheet.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical='top')
                
                print(f"Revisionshistorie mit {len(revision_data)} Einträgen exportiert")
                
            else:
                print("Keine Revisionshistorie zum Exportieren vorhanden")
                
        except Exception as e:
            print(f"Warnung: Fehler beim Export der Revisionshistorie: {str(e)}")
            import traceback
            traceback.print_exc()




# --- Hilfsfunktionen ---

