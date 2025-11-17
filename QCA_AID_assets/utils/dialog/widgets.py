"""
Tkinter GUI Widgets

Custom widgets for QCA-AID GUI including multi-select listbox.
"""

import tkinter as tk
from typing import Set


class MultiSelectListbox(tk.Listbox):
    """
    Erweiterte Listbox mit Mehrfachauswahl per Ctrl+Klick.
    
    Supports three selection modes:
    - Single click: Select/deselect single item
    - Ctrl+Click: Toggle individual item in current selection
    - Shift+Click: Select range from last clicked to current click
    
    This is used for manual multiple coding selection in the GUI.
    """
    
    def __init__(self, parent, **kwargs):
        """
        Initialize multi-select listbox.
        
        Args:
            parent: Parent tkinter widget
            **kwargs: Additional arguments passed to tk.Listbox
        """
        # Aktiviere erweiterte Mehrfachauswahl
        kwargs['selectmode'] = tk.EXTENDED
        super().__init__(parent, **kwargs)
        
        # Bindings für Mehrfachauswahl
        self.bind('<Button-1>', self._on_single_click)
        self.bind('<Control-Button-1>', self._on_ctrl_click)
        self.bind('<Shift-Button-1>', self._on_shift_click)
        
        # Speichere ursprüngliche Auswahl für Ctrl-Klick
        self._last_selection: Set[int] = set()
        
    def _on_single_click(self, event) -> None:
        """
        Normale Einzelauswahl mit Aktualisierung der letzten Auswahl.
        
        Args:
            event: Tkinter event object
        """
        # Lasse normale Behandlung durch tkinter zu
        self.after_idle(self._update_last_selection)
        
    def _on_ctrl_click(self, event) -> str:
        """
        Ctrl+Klick für Mehrfachauswahl (Toggle Individual Item).
        
        Args:
            event: Tkinter event object
            
        Returns:
            "break" to prevent default behavior
        """
        index = self.nearest(event.y)
        
        if index in self.curselection():
            # Deselektiere wenn bereits ausgewählt
            self.selection_clear(index)
        else:
            # Füge zur Auswahl hinzu
            self.selection_set(index)
            
        self._update_last_selection()
        return "break"  # Verhindert normale Behandlung
        
    def _on_shift_click(self, event) -> str:
        """
        Shift+Klick für Bereichsauswahl.
        
        Selects a range from the last clicked item to the current click.
        
        Args:
            event: Tkinter event object
            
        Returns:
            "break" to prevent default behavior
        """
        if not self._last_selection:
            return "break"
            
        index = self.nearest(event.y)
        last_indices = list(self._last_selection)
        
        if last_indices:
            start = min(last_indices[0], index)
            end = max(last_indices[0], index)
            
            # Wähle Bereich aus
            for i in range(start, end + 1):
                self.selection_set(i)
                
        self._update_last_selection()
        return "break"
        
    def _update_last_selection(self) -> None:
        """Aktualisiert die gespeicherte Auswahl."""
        self._last_selection = set(self.curselection())
