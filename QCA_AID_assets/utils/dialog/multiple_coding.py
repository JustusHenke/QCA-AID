"""
Manual Multiple Coding Dialog

Dialog component for confirming and configuring multiple codings in the GUI.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Optional


class ManualMultipleCodingDialog:
    """
    Dialog für die Bestätigung und Konfiguration von Mehrfachkodierungen.
    
    Displays a modal dialog showing:
    - The text segment being coded
    - The selected categories (main and sub)
    - Information about how the selection will be treated
    - Options to confirm, modify, or cancel the selection
    
    Dialog distinguishes between:
    - Single main category with multiple subcategories (multiple subcodes)
    - Multiple different main categories (multiple codings)
    """
    
    def __init__(self, parent, selected_categories: List[Dict], segment_text: str) -> None:
        """
        Initialize the multiple coding dialog.
        
        Args:
            parent: Parent tkinter widget (usually root window)
            selected_categories: List of selected categories with structure:
                {
                    'name': str,
                    'type': 'main' or 'sub',
                    'main_category': str (category name)
                }
            segment_text: The text segment being coded
        """
        self.parent = parent
        self.selected_categories = selected_categories
        self.segment_text = segment_text
        self.result: Optional[List[Dict]] = None
        self.dialog: Optional[tk.Toplevel] = None
        
    def show_dialog(self) -> Optional[List[Dict]]:
        """
        Zeigt den Bestätigungsdialog für Mehrfachkodierung.
        
        Modal dialog that blocks until user confirms, modifies, or cancels.
        
        Returns:
            List[Dict]: Liste der bestätigten Kodierungen
            "MODIFY": User wants to modify selection
            None: User cancelled the operation
        """
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Mehrfachkodierung bestätigen")
        self.dialog.geometry("600x500")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Zentriere Dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (500 // 2)
        self.dialog.geometry(f"600x500+{x}+{y}")
        
        self._create_widgets()
        
        # Warte auf Schließung des Dialogs
        self.dialog.wait_window()
        
        return self.result
    
    def _create_widgets(self) -> None:
        """Erstellt die Dialog-Widgets."""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Titel
        title_label = ttk.Label(
            main_frame,
            text="Mehrfachkodierung erkannt",
            font=('Arial', 12, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        # Informationstext
        info_text = f"Sie haben {len(self.selected_categories)} Kategorien/Subkategorien ausgewählt.\n"
        
        # Analysiere Auswahltyp
        main_categories = set()
        for cat in self.selected_categories:
            main_categories.add(cat['main_category'])
            
        if len(main_categories) == 1:
            info_text += f"Alle gehören zur Hauptkategorie '{list(main_categories)[0]}'.\n"
            info_text += "Dies wird als eine Kodierung mit mehreren Subkategorien behandelt."
        else:
            info_text += f"Sie umfassen {len(main_categories)} verschiedene Hauptkategorien.\n"
            info_text += "Dies wird als Mehrfachkodierung behandelt (mehrere Zeilen im Export)."
        
        info_label = ttk.Label(main_frame, text=info_text, wraplength=550)
        info_label.pack(pady=(0, 15))
        
        # Textsegment anzeigen
        text_frame = ttk.LabelFrame(main_frame, text="Textsegment")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        text_widget = tk.Text(text_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        text_widget.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        text_widget.config(state=tk.NORMAL)
        text_widget.insert(tk.END, self.segment_text[:1000] + ("..." if len(self.segment_text) > 1000 else ""))
        text_widget.config(state=tk.DISABLED)
        
        # Ausgewählte Kategorien anzeigen
        selection_frame = ttk.LabelFrame(main_frame, text="Ihre Auswahl")
        selection_frame.pack(fill=tk.X, pady=(0, 15))
        
        selection_text = tk.Text(selection_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        selection_text.pack(padx=5, pady=5, fill=tk.X)
        
        selection_text.config(state=tk.NORMAL)
        for i, cat in enumerate(self.selected_categories, 1):
            if cat['type'] == 'main':
                selection_text.insert(tk.END, f"{i}. Hauptkategorie: {cat['name']}\n")
            else:
                selection_text.insert(tk.END, f"{i}. Subkategorie: {cat['name']} (→ {cat['main_category']})\n")
        selection_text.config(state=tk.DISABLED)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(
            button_frame,
            text="Bestätigen",
            command=self._confirm_selection
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="Ändern",
            command=self._modify_selection
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="Abbrechen",
            command=self._cancel_selection
        ).pack(side=tk.RIGHT)
    
    def _confirm_selection(self) -> None:
        """Bestätigt die aktuelle Auswahl."""
        self.result = self.selected_categories
        self.dialog.destroy()
    
    def _modify_selection(self) -> None:
        """Schließt Dialog zum Ändern der Auswahl."""
        self.result = "MODIFY"
        self.dialog.destroy()
    
    def _cancel_selection(self) -> None:
        """Bricht die Mehrfachkodierung ab."""
        self.result = None
        self.dialog.destroy()
