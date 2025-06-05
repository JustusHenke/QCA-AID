import re
import asyncio
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
import json
import pandas as pd
from openpyxl import load_workbook
import tkinter as tk
from tkinter import ttk, messagebox



CONFIG = { } # hier leer, in Hauptskript integriert 

# --- Klasse: TokenCounter ---
class TokenCounter:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def add_tokens(self, input_tokens: int, output_tokens: int = 0):
        """
        Zählt Input- und Output-Tokens.
        
        Args:
            input_tokens: Anzahl der Input-Tokens
            output_tokens: Anzahl der Output-Tokens (optional, Standard 0)
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def get_report(self):
        return f"Gesamte Token-Nutzung:\n" \
               f"Input Tokens: {self.input_tokens}\n" \
               f"Output Tokens: {self.output_tokens}\n" \
               f"Gesamt Tokens: {self.input_tokens + self.output_tokens}"

token_counter = TokenCounter()



# Hilfsfunktion zur Token-Schätzung
def estimate_tokens(text: str) -> int:
    """
    Schätzt die Anzahl der Tokens in einem Text mit verbesserter Genauigkeit.
    Berücksichtigt verschiedene Zeichentypen, nicht nur Wortgrenzen.
    
    Args:
        text: Zu schätzender Text
        
    Returns:
        int: Geschätzte Tokenanzahl
    """
    if not text:
        return 0
        
    # Grundlegende Schätzung: 1 Token ≈ 4 Zeichen für englischen Text
    # Für deutsche Texte (mit längeren Wörtern) etwas anpassen: 1 Token ≈ 4.5 Zeichen
    char_per_token = 4.5
    
    # Anzahl der Sonderzeichen, die oft eigene Tokens bilden
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    
    # Anzahl der Wörter
    words = len(text.split())
    
    # Gewichtete Berechnung
    estimated_tokens = int(
        (len(text) / char_per_token) * 0.7 +  # Zeichenbasierte Schätzung (70% Gewichtung)
        (words + special_chars) * 0.3          # Wort- und Sonderzeichenbasierte Schätzung (30% Gewichtung)
    )
    
    return max(1, estimated_tokens)  # Mindestens 1 Token

def _safe_speed_calculation(count: int, time_elapsed: float) -> str:
    """
    Sichere Geschwindigkeitsberechnung mit Division-by-Zero Schutz.
    
    Args:
        count: Anzahl der verarbeiteten Elemente
        time_elapsed: Verstrichene Zeit in Sekunden
        
    Returns:
        str: Formatierte Geschwindigkeitsangabe
    """
    if time_elapsed > 0:
        speed = count / time_elapsed
        return f"{speed:.1f} Elemente/Sekunde"
    else:
        return f"{count} Elemente in <0.01s (sehr schnell)"
    
def get_input_with_timeout(prompt: str, timeout: int = 30) -> str:
    """
    Fragt nach Benutzereingabe mit Timeout.
    
    Args:
        prompt: Anzuzeigender Text
        timeout: Timeout in Sekunden
        
    Returns:
        str: Benutzereingabe oder 'n' bei Timeout
    """
    import threading
    import sys
    import time
    from threading import Event
    
    # Plattformspezifische Imports
    if sys.platform == 'win32':
        import msvcrt
    else:
        import select

    answer = {'value': None}
    stop_event = Event()
    
    def input_thread():
        try:
            # Zeige Countdown
            remaining_time = timeout
            while remaining_time > 0 and not stop_event.is_set():
                sys.stdout.write(f'\r{prompt} ({remaining_time}s): ')
                sys.stdout.flush()
                
                # Plattformspezifische Eingabeprüfung
                if sys.platform == 'win32':
                    if msvcrt.kbhit():
                        answer['value'] = msvcrt.getche().decode().strip().lower()
                        sys.stdout.write('\n')
                        stop_event.set()
                        return
                else:
                    if select.select([sys.stdin], [], [], 1)[0]:
                        answer['value'] = sys.stdin.readline().strip().lower()
                        stop_event.set()
                        return
                
                time.sleep(1)
                remaining_time -= 1
            
            # Bei Timeout
            if not stop_event.is_set():
                sys.stdout.write('\n')
                sys.stdout.flush()
                
        except (KeyboardInterrupt, EOFError):
            stop_event.set()
    
    # Starte Input-Thread
    thread = threading.Thread(target=input_thread)
    thread.daemon = True
    thread.start()
    
    # Warte auf Antwort oder Timeout
    thread.join(timeout)
    stop_event.set()
    
    if answer['value'] is None:
        print(f"\nKeine Eingabe innerhalb von {timeout} Sekunden - verwende 'n'")
        return 'n'
        
    return answer['value']

def _calculate_multiple_coding_stats(all_codings: List[Dict]) -> Dict:
    """
    Berechnet Statistiken zur Mehrfachkodierung.
    
    Args:
        all_codings: Liste aller Kodierungen
        
    Returns:
        Dict: Statistiken zur Mehrfachkodierung
    """
    from collections import defaultdict, Counter
    
    segment_counts = defaultdict(int)
    focus_adherence = []
    category_combinations = []
    
    for coding in all_codings:
        segment_id = coding.get('segment_id', '')
        segment_counts[segment_id] += 1
        
        # Focus adherence tracking
        if coding.get('category_focus_used', False):
            focus_adherence.append(coding.get('target_category', '') == coding.get('category', ''))
        
        # Kategorie-Kombinationen sammeln
        if coding.get('total_coding_instances', 1) > 1:
            # Sammle alle Kategorien für dieses Segment
            segment_categories = [c.get('category', '') for c in all_codings 
                                if c.get('segment_id', '') == segment_id]
            if len(segment_categories) > 1:
                category_combinations.append(' + '.join(sorted(set(segment_categories))))
    
    segments_with_multiple = len([count for count in segment_counts.values() if count > 1])
    total_segments = len(segment_counts)
    total_codings = len(all_codings)
    
    combination_counter = Counter(category_combinations)
    
    return {
        'segments_with_multiple': segments_with_multiple,
        'total_segments': total_segments,
        'avg_codings_per_segment': total_codings / total_segments if total_segments > 0 else 0,
        'top_combinations': [combo for combo, _ in combination_counter.most_common(5)],
        'focus_adherence_rate': sum(focus_adherence) / len(focus_adherence) if focus_adherence else 0
    }

def _patch_tkinter_for_threaded_exit():
    """
    Patcht die Tkinter Variable.__del__ Methode, um den RuntimeError beim Beenden zu vermeiden.
    """
    import tkinter
    
    # Originale __del__ Methode speichern
    original_del = tkinter.Variable.__del__
    
    # Neue __del__ Methode definieren, die Ausnahmen abfängt
    def safe_del(self):
        try:
            # Nur aufrufen, wenn _tk existiert und es sich um ein valides Tkinter-Objekt handelt
            if hasattr(self, '_tk') and self._tk:
                original_del(self)
        except (RuntimeError, TypeError, AttributeError):
            # Diese Ausnahmen stillschweigend ignorieren
            pass
    
    # Die ursprüngliche Methode ersetzen
    tkinter.Variable.__del__ = safe_del
    print("Tkinter für sicheres Beenden gepatcht.")


# Neue Ergänzungen für QCA_Utils.py - Mehrfachkodierung im manuellen Modus

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class MultiSelectListbox(tk.Listbox):
    """
    Erweiterte Listbox mit Mehrfachauswahl per Ctrl+Klick
    für manuelle Mehrfachkodierung
    """
    
    def __init__(self, parent, **kwargs):
        # Aktiviere erweiterte Mehrfachauswahl
        kwargs['selectmode'] = tk.EXTENDED
        super().__init__(parent, **kwargs)
        
        # Bindings für Mehrfachauswahl
        self.bind('<Button-1>', self._on_single_click)
        self.bind('<Control-Button-1>', self._on_ctrl_click)
        self.bind('<Shift-Button-1>', self._on_shift_click)
        
        # Speichere ursprüngliche Auswahl für Ctrl-Klick
        self._last_selection = set()
        
    def _on_single_click(self, event):
        """Normale Einzelauswahl"""
        # Lasse normale Behandlung durch tkinter zu
        self.after_idle(self._update_last_selection)
        
    def _on_ctrl_click(self, event):
        """Ctrl+Klick für Mehrfachauswahl"""
        index = self.nearest(event.y)
        
        if index in self.curselection():
            # Deselektiere wenn bereits ausgewählt
            self.selection_clear(index)
        else:
            # Füge zur Auswahl hinzu
            self.selection_set(index)
            
        self._update_last_selection()
        return "break"  # Verhindert normale Behandlung
        
    def _on_shift_click(self, event):
        """Shift+Klick für Bereichsauswahl"""
        if not self._last_selection:
            return
            
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
        
    def _update_last_selection(self):
        """Aktualisiert die gespeicherte Auswahl"""
        self._last_selection = set(self.curselection())

class ManualMultipleCodingDialog:
    """
    Dialog für die Bestätigung und Konfiguration von Mehrfachkodierungen
    """
    
    def __init__(self, parent, selected_categories: List[Dict], segment_text: str):
        self.parent = parent
        self.selected_categories = selected_categories
        self.segment_text = segment_text
        self.result = None
        self.dialog = None
        
    def show_dialog(self) -> Optional[List[Dict]]:
        """
        Zeigt den Bestätigungsdialog für Mehrfachkodierung
        
        Returns:
            Optional[List[Dict]]: Liste der bestätigten Kodierungen oder None bei Abbruch
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
    
    def _create_widgets(self):
        """Erstellt die Dialog-Widgets"""
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
    
    def _confirm_selection(self):
        """Bestätigt die aktuelle Auswahl"""
        self.result = self.selected_categories
        self.dialog.destroy()
    
    def _modify_selection(self):
        """Schließt Dialog zum Ändern der Auswahl"""
        self.result = "MODIFY"
        self.dialog.destroy()
    
    def _cancel_selection(self):
        """Bricht die Mehrfachkodierung ab"""
        self.result = None
        self.dialog.destroy()

def create_multiple_coding_results(selected_categories: List[Dict], 
                                 text: str, 
                                 coder_id: str) -> List:
    """
    Erstellt CodingResult-Objekte für Mehrfachkodierung
    
    Args:
        selected_categories: Liste der ausgewählten Kategorien
        text: Zu kodierender Text
        coder_id: ID des Kodierers
        
    Returns:
        List: Liste von Kodierungs-Dictionaries (da CodingResult nicht importierbar)
    """
    from datetime import datetime
    
    # Gruppiere nach Hauptkategorien
    main_category_groups = {}
    for cat in selected_categories:
        main_cat = cat['main_category']
        if main_cat not in main_category_groups:
            main_category_groups[main_cat] = {
                'main_category': main_cat,
                'subcategories': []
            }
        
        if cat['type'] == 'sub':
            main_category_groups[main_cat]['subcategories'].append(cat['name'])
    
    # Erstelle Kodierungsergebnisse als Dictionaries
    coding_results = []
    total_instances = len(main_category_groups)
    
    for instance_num, (main_cat, group_data) in enumerate(main_category_groups.items(), 1):
        # Erstelle immer Dictionary (einfacher und zuverlässiger)
        coding_result = {
            'category': main_cat,
            'subcategories': tuple(group_data['subcategories']),
            'justification': f"Manuelle Mehrfachkodierung (Instanz {instance_num}/{total_instances})",
            'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
            'text_references': (text[:100],),
            'uncertainties': None,
            'paraphrase': "",
            'keywords': "",
            'multiple_coding_instance': instance_num,
            'total_coding_instances': total_instances,
            'manual_multiple_coding': True,
            'coding_date': datetime.now().isoformat(),
            'coder_id': coder_id,
            'text': text
        }
        
        coding_results.append(coding_result)
    
    return coding_results

def show_multiple_coding_info(parent, num_categories: int, main_categories: List[str]) -> bool:
    """
    Zeigt Informationen über erkannte Mehrfachkodierung
    
    Args:
        parent: Parent-Widget
        num_categories: Anzahl ausgewählter Kategorien
        main_categories: Liste der Hauptkategorien
        
    Returns:
        bool: True wenn fortgefahren werden soll
    """
    if len(main_categories) == 1:
        message = (f"Sie haben {num_categories} Subkategorien der Hauptkategorie "
                  f"'{main_categories[0]}' ausgewählt.\n\n"
                  f"Dies wird als eine Kodierung mit mehreren Subkategorien behandelt.")
        title = "Mehrere Subkategorien"
    else:
        message = (f"Sie haben Kategorien aus {len(main_categories)} verschiedenen "
                  f"Hauptkategorien ausgewählt:\n"
                  f"{', '.join(main_categories)}\n\n"
                  f"Dies wird als Mehrfachkodierung behandelt - das Segment erscheint "
                  f"mehrmals im Export (einmal pro Hauptkategorie).")
        title = "Mehrfachkodierung erkannt"
    
    return messagebox.askyesno(
        title,
        message + "\n\nMöchten Sie fortfahren?",
        parent=parent
    )

# Zusätzliche Hilfsfunktionen für die GUI

def setup_manual_coding_window_enhanced(root, categories, chunk_text, coder_id, is_last_segment=False):
    """
    Erweiterte Einrichtung des manuellen Kodierungsfensters mit Mehrfachauswahl
    
    Args:
        root: Tkinter Root-Fenster
        categories: Verfügbare Kategorien
        chunk_text: Zu kodierender Text
        coder_id: ID des Kodierers
        is_last_segment: Ob es das letzte Segment ist
        
    Returns:
        Dict: GUI-Komponenten für erweiterte Funktionalität
    """
    # Hauptframe
    main_frame = ttk.Frame(root)
    main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    # Instruktionen für Mehrfachauswahl
    instructions = ttk.Label(
        main_frame,
        text="💡 Tipp: Verwenden Sie Strg+Klick für Mehrfachauswahl von Kategorien",
        font=('Arial', 9, 'italic'),
        foreground='blue'
    )
    instructions.pack(pady=(0, 10))
    
    # Fortschrittsinfo
    if is_last_segment:
        progress_label = ttk.Label(
            main_frame,
            text="🏁 LETZTES SEGMENT - Kodierung wird nach diesem Segment abgeschlossen",
            font=('Arial', 10, 'bold'),
            foreground='red'
        )
        progress_label.pack(pady=(0, 10))
    
    return main_frame

def validate_multiple_selection(selected_indices: List[int], 
                              category_map: Dict[int, Dict]) -> Tuple[bool, str, List[Dict]]:
    """
    Validiert eine Mehrfachauswahl von Kategorien
    
    Args:
        selected_indices: Liste der ausgewählten Indizes
        category_map: Mapping von Index zu Kategorie-Info
        
    Returns:
        Tuple[bool, str, List[Dict]]: (Gültig, Fehlermeldung, Kategorien-Liste)
    """
    if not selected_indices:
        return False, "Keine Kategorie ausgewählt", []
    
    if len(selected_indices) == 1:
        # Einzelauswahl - normal verarbeiten
        return True, "", []
    
    # Mehrfachauswahl validieren
    selected_categories = []
    main_categories = set()
    
    for idx in selected_indices:
        if idx not in category_map:
            return False, f"Ungültiger Index: {idx}", []
        
        cat_info = category_map[idx]
        selected_categories.append(cat_info)
        main_categories.add(cat_info['main_category'])
    
    # Prüfe auf gültige Kombinationen
    if len(main_categories) > 3:
        return False, "Maximal 3 verschiedene Hauptkategorien können gleichzeitig ausgewählt werden", []
    
    return True, "", selected_categories


# --- Klasse: ConfigLoader ---
class ConfigLoader:
    def __init__(self, script_dir):
        self.script_dir = script_dir
        self.excel_path = os.path.join(script_dir, "QCA-AID-Codebook.xlsx")
        self.config = {
            'FORSCHUNGSFRAGE': "",
            'KODIERREGELN': {},
            'DEDUKTIVE_KATEGORIEN': {},
            'CONFIG': {}
        }
        
    def load_codebook(self):
        print(f"Versuche Konfiguration zu laden von: {self.excel_path}")
        if not os.path.exists(self.excel_path):
            print(f"Excel-Datei nicht gefunden: {self.excel_path}")
            return False

        try:
            # Öffne die Excel-Datei mit ausführlicher Fehlerbehandlung
            print("\nÖffne Excel-Datei...")
            wb = load_workbook(self.excel_path, read_only=True, data_only=True)
            print(f"Excel-Datei erfolgreich geladen. Verfügbare Sheets: {wb.sheetnames}")
            
            # Prüfe DEDUKTIVE_KATEGORIEN Sheet
            if 'DEDUKTIVE_KATEGORIEN' in wb.sheetnames:
                print("\nLese DEDUKTIVE_KATEGORIEN Sheet...")
                sheet = wb['DEDUKTIVE_KATEGORIEN']
            
            
            # Lade die verschiedenen Komponenten
            self._load_research_question(wb)
            self._load_coding_rules(wb)
            self._load_config(wb)
            
            # Debug-Ausgabe vor dem Laden der Kategorien
            print("\nStarte Laden der deduktiven Kategorien...")
            kategorien = self._load_deduktive_kategorien(wb)
            
            # Prüfe das Ergebnis
            if kategorien:
                print("\nGeladene Kategorien:")
                for name, data in kategorien.items():
                    print(f"\n{name}:")
                    print(f"- Definition: {len(data['definition'])} Zeichen")
                    print(f"- Beispiele: {len(data['examples'])}")
                    print(f"- Regeln: {len(data['rules'])}")
                    print(f"- Subkategorien: {len(data['subcategories'])}")
                
                # Speichere in Config
                self.config['DEDUKTIVE_KATEGORIEN'] = kategorien
                print("\nKategorien erfolgreich in Config gespeichert")
                return True
            else:
                print("\nKeine Kategorien geladen!")
                return False

        except Exception as e:
            print(f"Fehler beim Lesen der Excel-Datei: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()
            return False

    def _load_research_question(self, wb):
        if 'FORSCHUNGSFRAGE' in wb.sheetnames:
            sheet = wb['FORSCHUNGSFRAGE']
            value = sheet['B1'].value
            # print(f"Geladene Forschungsfrage: {value}")  # Debug-Ausgabe
            self.config['FORSCHUNGSFRAGE'] = value


    def _load_coding_rules(self, wb):
        """Lädt Kodierregeln aus dem Excel-Codebook."""
        if 'KODIERREGELN' in wb.sheetnames:
            df = pd.read_excel(self.excel_path, sheet_name='KODIERREGELN', header=0)
            
            # Initialisiere Regelkategorien
            rules = {
                'general': [],       # Allgemeine Kodierregeln
                'format': [],        # Formatregeln
                'exclusion': []      # Neue Kategorie für Ausschlussregeln
            }
            
            # Verarbeite jede Spalte
            for column in df.columns:
                rules_list = df[column].dropna().tolist()
                
                if 'Allgemeine' in column:
                    rules['general'].extend(rules_list)
                elif 'Format' in column:
                    rules['format'].extend(rules_list)
                elif 'Ausschluss' in column:  # Neue Spalte für Ausschlussregeln
                    rules['exclusion'].extend(rules_list)
            
            print("\nKodierregeln geladen:")
            print(f"- Allgemeine Regeln: {len(rules['general'])}")
            print(f"- Formatregeln: {len(rules['format'])}")
            print(f"- Ausschlussregeln: {len(rules['exclusion'])}")
            
            self.config['KODIERREGELN'] = rules

    def _load_deduktive_kategorien(self, wb):
        try:
            if 'DEDUKTIVE_KATEGORIEN' not in wb.sheetnames:
                print("Warnung: Sheet 'DEDUKTIVE_KATEGORIEN' nicht gefunden")
                return {}

            print("\nLade deduktive Kategorien...")
            sheet = wb['DEDUKTIVE_KATEGORIEN']
            
            # Initialisiere Kategorien
            kategorien = {}
            current_category = None
            
            # Hole Header-Zeile
            headers = []
            for cell in sheet[1]:
                headers.append(cell.value)
            # print(f"Gefundene Spalten: {headers}")
            
            # Indizes für Spalten finden
            key_idx = headers.index('Key') if 'Key' in headers else None
            sub_key_idx = headers.index('Sub-Key') if 'Sub-Key' in headers else None
            sub_sub_key_idx = headers.index('Sub-Sub-Key') if 'Sub-Sub-Key' in headers else None
            value_idx = headers.index('Value') if 'Value' in headers else None
            
            if None in [key_idx, sub_key_idx, value_idx]:
                print("Fehler: Erforderliche Spalten fehlen!")
                return {}
                
            # Verarbeite Zeilen
            for row_idx, row in enumerate(sheet.iter_rows(min_row=2), 2):
                try:
                    key = row[key_idx].value
                    sub_key = row[sub_key_idx].value if row[sub_key_idx].value else None
                    sub_sub_key = row[sub_sub_key_idx].value if sub_sub_key_idx is not None and row[sub_sub_key_idx].value else None
                    value = row[value_idx].value if row[value_idx].value else None
                    
                    
                    # Neue Hauptkategorie
                    if key and isinstance(key, str):
                        key = key.strip()
                        if key not in kategorien:
                            print(f"\nNeue Hauptkategorie: {key}")
                            current_category = key
                            kategorien[key] = {
                                'definition': '',
                                'rules': [],
                                'examples': [],
                                'subcategories': {}
                            }
                    
                    # Verarbeite Unterkategorien und Werte
                    if current_category and sub_key:
                        sub_key = sub_key.strip()
                        if isinstance(value, str):
                            value = value.strip()
                            
                        if sub_key == 'definition':
                            kategorien[current_category]['definition'] = value
                            print(f"  Definition hinzugefügt: {len(value)} Zeichen")
                            
                        elif sub_key == 'rules':
                            if value:
                                kategorien[current_category]['rules'].append(value)
                                print(f"  Regel hinzugefügt: {value[:50]}...")
                                
                        elif sub_key == 'examples':
                            if value:
                                kategorien[current_category]['examples'].append(value)
                                print(f"  Beispiel hinzugefügt: {value[:50]}...")
                                
                        elif sub_key == 'subcategories' and sub_sub_key:
                            kategorien[current_category]['subcategories'][sub_sub_key] = value
                            print(f"  Subkategorie hinzugefügt: {sub_sub_key}")
                                
                except Exception as e:
                    print(f"Fehler in Zeile {row_idx}: {str(e)}")
                    continue

            # Validierung der geladenen Daten
            print("\nValidiere geladene Kategorien:")
            # for name, kat in kategorien.items():
            #     print(f"\nKategorie: {name}")
            #     print(f"- Definition: {len(kat['definition'])} Zeichen")
            #     if not kat['definition']:
            #         print("  WARNUNG: Keine Definition!")
            #     print(f"- Regeln: {len(kat['rules'])}")
            #     print(f"- Beispiele: {len(kat['examples'])}")
            #     print(f"- Subkategorien: {len(kat['subcategories'])}")
            #     for sub_name, sub_def in kat['subcategories'].items():
            #         print(f"  • {sub_name}: {sub_def[:50]}...")

            # Ergebnis
            if kategorien:
                print(f"\nErfolgreich {len(kategorien)} Kategorien geladen")
                return kategorien
            else:
                print("\nKeine Kategorien gefunden!")
                return {}

        except Exception as e:
            print(f"Fehler beim Laden der Kategorien: {str(e)}")
            print("Details:")
            import traceback
            traceback.print_exc()
            return {}
        
    def _load_config(self, wb):
        if 'CONFIG' in wb.sheetnames:
            df = pd.read_excel(self.excel_path, sheet_name='CONFIG')
            config = {}
            
            for _, row in df.iterrows():
                key = row['Key']
                sub_key = row['Sub-Key']
                sub_sub_key = row['Sub-Sub-Key']
                value = row['Value']

                if key not in config:
                    config[key] = value if pd.isna(sub_key) else {}

                if not pd.isna(sub_key):
                    if sub_key.startswith('['):  # Für Listen wie CODER_SETTINGS
                        if not isinstance(config[key], list):
                            config[key] = []
                        index = int(sub_key.strip('[]'))
                        while len(config[key]) <= index:
                            config[key].append({})
                        if pd.isna(sub_sub_key):
                            config[key][index] = value
                        else:
                            config[key][index][sub_sub_key] = value
                    else:  # Für verschachtelte Dicts wie ATTRIBUTE_LABELS
                        if not isinstance(config[key], dict):
                            config[key] = {}
                        if pd.isna(sub_sub_key):
                            if key == 'BATCH_SIZE' or sub_key == 'BATCH_SIZE':
                                try:
                                    value = int(value)
                                    print(f"BATCH_SIZE aus Codebook geladen: {value}")
                                except (ValueError, TypeError):
                                    value = 5  # Standardwert
                                    print(f"Warnung: Ungültiger BATCH_SIZE Wert, verwende Standard: {value}")
                            config[key][sub_key] = value
                        else:
                            if sub_key not in config[key]:
                                config[key][sub_key] = {}
                            config[key][sub_key][sub_sub_key] = value

            # Prüfe auf ANALYSIS_MODE in der Konfiguration
            if 'ANALYSIS_MODE' in config:
                valid_modes = {'full', 'abductive', 'deductive'}
                if config['ANALYSIS_MODE'] not in valid_modes:
                    print(f"Warnung: Ungültiger ANALYSIS_MODE '{config['ANALYSIS_MODE']}' im Codebook. Verwende 'deductive'.")
                    config['ANALYSIS_MODE'] = 'deductive'
                else:
                    print(f"ANALYSIS_MODE aus Codebook geladen: {config['ANALYSIS_MODE']}")
            else:
                config['ANALYSIS_MODE'] = 'deductive'  # Standardwert
                print(f"ANALYSIS_MODE nicht im Codebook gefunden, verwende Standard: {config['ANALYSIS_MODE']}")

            # Prüfe auf REVIEW_MODE in der Konfiguration
            if 'REVIEW_MODE' in config:
                valid_modes = {'auto', 'manual', 'consensus', 'majority'}
                if config['REVIEW_MODE'] not in valid_modes:
                    print(f"Warnung: Ungültiger REVIEW_MODE '{config['REVIEW_MODE']}' im Codebook. Verwende 'auto'.")
                    config['REVIEW_MODE'] = 'consensus'
                else:
                    print(f"REVIEW_MODE aus Codebook geladen: {config['REVIEW_MODE']}")
            else:
                config['REVIEW_MODE'] = 'consensus'  # Standardwert
                print(f"REVIEW_MODE nicht im Codebook gefunden, verwende Standard: {config['REVIEW_MODE']}")


            # Stelle sicher, dass ATTRIBUTE_LABELS vorhanden ist
            if 'ATTRIBUTE_LABELS' not in config:
                config['ATTRIBUTE_LABELS'] = {'attribut1': 'Attribut1', 'attribut2': 'Attribut2', 'attribut3': 'Attribut3'}
            elif 'attribut3' not in config['ATTRIBUTE_LABELS']:
                # Füge attribut3 hinzu wenn noch nicht vorhanden
                config['ATTRIBUTE_LABELS']['attribut3'] = 'Attribut3'
            
            # Debug für attribut3
            if config['ATTRIBUTE_LABELS']['attribut3']:
                print(f"Drittes Attribut-Label geladen: {config['ATTRIBUTE_LABELS']['attribut3']}")

            self.config['CONFIG'] = self._sanitize_config(config)
            return True  # Explizite Rückgabe von True
        return False

    def _sanitize_config(self, config):
        """
        Bereinigt und validiert die Konfigurationswerte.
        Überschreibt Standardwerte mit Werten aus dem Codebook.
        
        Args:
            config: Dictionary mit rohen Konfigurationswerten
            
        Returns:
            dict: Bereinigtes Konfigurations-Dictionary
        """
        try:
            sanitized = {}
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Stelle sicher, dass OUTPUT_DIR immer gesetzt wird
            sanitized['OUTPUT_DIR'] = os.path.join(script_dir, 'output')
            os.makedirs(sanitized['OUTPUT_DIR'], exist_ok=True)
            print(f"Standard-Ausgabeverzeichnis gesichert: {sanitized['OUTPUT_DIR']}")
            
            for key, value in config.items():
                # Verzeichnispfade relativ zum aktuellen Arbeitsverzeichnis
                if key in ['DATA_DIR', 'OUTPUT_DIR']:
                    sanitized[key] = os.path.join(script_dir, str(value))
                    # Stelle sicher, dass Verzeichnis existiert
                    os.makedirs(sanitized[key], exist_ok=True)
                    print(f"Verzeichnis gesichert: {sanitized[key]}")
                
                # Numerische Werte für Chunking
                elif key in ['CHUNK_SIZE', 'CHUNK_OVERLAP']:
                    try:
                        # Konvertiere zu Integer und stelle sicher, dass die Werte positiv sind
                        sanitized[key] = max(1, int(value))
                        print(f"Übernehme {key} aus Codebook: {sanitized[key]}")
                    except (ValueError, TypeError):
                        # Wenn Konvertierung fehlschlägt, behalte Standardwert
                        default_value = CONFIG[key]
                        print(f"Warnung: Ungültiger Wert für {key}, verwende Standard: {default_value}")
                        sanitized[key] = default_value
                
                # Coder-Einstellungen mit Typkonvertierung
                elif key == 'CODER_SETTINGS':
                    sanitized[key] = [
                        {
                            'temperature': float(coder['temperature']) 
                                if isinstance(coder.get('temperature'), (int, float, str)) 
                                else 0.3,
                            'coder_id': str(coder.get('coder_id', f'auto_{i}'))
                        }
                        for i, coder in enumerate(value)
                    ]
                
                # Alle anderen Werte unverändert übernehmen
                else:
                    sanitized[key] = value

            # Verarbeitung des CODE_WITH_CONTEXT Parameters
            if 'CODE_WITH_CONTEXT' in config:
                # Konvertiere zu Boolean
                value = config['CODE_WITH_CONTEXT']
                if isinstance(value, str):
                    sanitized['CODE_WITH_CONTEXT'] = value.lower() in ('true', 'ja', 'yes', '1')
                else:
                    sanitized['CODE_WITH_CONTEXT'] = bool(value)
                print(f"Übernehme CODE_WITH_CONTEXT aus Codebook: {sanitized['CODE_WITH_CONTEXT']}")
            else:
                # Standardwert setzen
                sanitized['CODE_WITH_CONTEXT'] = True
                print(f"CODE_WITH_CONTEXT nicht in Codebook gefunden, verwende Standard: {sanitized['CODE_WITH_CONTEXT']}")

            # Verarbeitung des MULTIPLE_CODINGS Parameters
            if 'MULTIPLE_CODINGS' in config:
                # Konvertiere zu Boolean
                value = config['MULTIPLE_CODINGS']
                if isinstance(value, str):
                    sanitized['MULTIPLE_CODINGS'] = value.lower() in ('true', 'ja', 'yes', '1')
                else:
                    sanitized['MULTIPLE_CODINGS'] = bool(value)
                print(f"Übernehme MULTIPLE_CODINGS aus Codebook: {sanitized['MULTIPLE_CODINGS']}")
            else:
                # Standardwert setzen
                sanitized['MULTIPLE_CODINGS'] = True
                print(f"MULTIPLE_CODINGS nicht in Codebook gefunden, verwende Standard: {sanitized['MULTIPLE_CODINGS']}")
            
            # Verarbeitung des MULTIPLE_CODING_THRESHOLD Parameters
            if 'MULTIPLE_CODING_THRESHOLD' in config:
                try:
                    threshold = float(config['MULTIPLE_CODING_THRESHOLD'])
                    if 0.0 <= threshold <= 1.0:
                        sanitized['MULTIPLE_CODING_THRESHOLD'] = threshold
                        print(f"Übernehme MULTIPLE_CODING_THRESHOLD aus Codebook: {sanitized['MULTIPLE_CODING_THRESHOLD']}")
                    else:
                        print(f"Warnung: MULTIPLE_CODING_THRESHOLD muss zwischen 0.0 und 1.0 liegen, verwende Standard: 0.6")
                        sanitized['MULTIPLE_CODING_THRESHOLD'] = 0.6
                except (ValueError, TypeError):
                    print(f"Warnung: Ungültiger MULTIPLE_CODING_THRESHOLD Wert, verwende Standard: 0.6")
                    sanitized['MULTIPLE_CODING_THRESHOLD'] = 0.6
            else:
                sanitized['MULTIPLE_CODING_THRESHOLD'] = 0.6
                print(f"MULTIPLE_CODING_THRESHOLD nicht in Codebook gefunden, verwende Standard: {sanitized['MULTIPLE_CODING_THRESHOLD']}")

            # Verarbeitung des BATCH_SIZE Parameters
            if 'BATCH_SIZE' in config:
                try:
                    batch_size = int(config['BATCH_SIZE'])
                    if batch_size < 1:
                        print("Warnung: BATCH_SIZE muss mindestens 1 sein")
                        batch_size = 5
                    elif batch_size > 20:
                        print("Warnung: BATCH_SIZE > 20 könnte Performance-Probleme verursachen")
                    sanitized['BATCH_SIZE'] = batch_size
                    print(f"Finale BATCH_SIZE: {batch_size}")
                except (ValueError, TypeError):
                    print("Warnung: Ungültiger BATCH_SIZE Wert")
                    sanitized['BATCH_SIZE'] = 5
            else:
                print("BATCH_SIZE nicht in Codebook gefunden, verwende Standard: 5")
                sanitized['BATCH_SIZE'] = 5

            for key, value in config.items():
                if key == 'BATCH_SIZE':
                    continue

            # Stelle sicher, dass CHUNK_OVERLAP kleiner als CHUNK_SIZE ist
            if 'CHUNK_SIZE' in sanitized and 'CHUNK_OVERLAP' in sanitized:
                if sanitized['CHUNK_OVERLAP'] >= sanitized['CHUNK_SIZE']:
                    print(f"Warnung: CHUNK_OVERLAP ({sanitized['CHUNK_OVERLAP']}) muss kleiner sein als CHUNK_SIZE ({sanitized['CHUNK_SIZE']})")
                    sanitized['CHUNK_OVERLAP'] = sanitized['CHUNK_SIZE'] // 10  # 10% als Standardwert
                    
            return sanitized
            
        except Exception as e:
            print(f"Fehler bei der Konfigurationsbereinigung: {str(e)}")
            import traceback
            traceback.print_exc()
            # Verwende im Fehlerfall die Standard-Konfiguration
            return CONFIG
    

    def update_script_globals(self, globals_dict):
        """
        Aktualisiert die globalen Variablen mit den Werten aus der Config.
        """
        try:
            print("\nAktualisiere globale Variablen...")
            
            # Update DEDUKTIVE_KATEGORIEN
            if 'DEDUKTIVE_KATEGORIEN' in self.config:
                deduktive_kat = self.config['DEDUKTIVE_KATEGORIEN']
                if deduktive_kat and isinstance(deduktive_kat, dict):
                    globals_dict['DEDUKTIVE_KATEGORIEN'] = deduktive_kat
                    print(f"\nDEDUKTIVE_KATEGORIEN aktualisiert:")
                    for name in deduktive_kat.keys():
                        print(f"- {name}")
                else:
                    print("Warnung: Keine gültigen DEDUKTIVE_KATEGORIEN in Config")
            
            # WICHTIG: Update das CONFIG Dictionary komplett
            if 'CONFIG' in self.config:
                config_from_codebook = self.config['CONFIG']
                
                # Aktualisiere das globale CONFIG Dictionary
                if 'CONFIG' in globals_dict:
                    globals_dict['CONFIG'].update(config_from_codebook)
                    print(f"\nCONFIG Dictionary aktualisiert mit {len(config_from_codebook)} Einträgen")
                    
                    # Spezielle Debug-Ausgabe für MULTIPLE_CODINGS
                    if 'MULTIPLE_CODINGS' in config_from_codebook:
                        print(f"MULTIPLE_CODINGS aus Codebook: {config_from_codebook['MULTIPLE_CODINGS']}")
                        print(f"MULTIPLE_CODINGS in globalem CONFIG: {globals_dict['CONFIG'].get('MULTIPLE_CODINGS')}")
                    
                    # Spezielle Debug-Ausgabe für andere wichtige Settings
                    important_settings = ['ANALYSIS_MODE', 'REVIEW_MODE', 'CODE_WITH_CONTEXT', 'BATCH_SIZE']
                    for setting in important_settings:
                        if setting in config_from_codebook:
                            print(f"{setting} aus Codebook: {config_from_codebook[setting]}")
                else:
                    print("Warnung: CONFIG nicht im globalen Namespace gefunden")
            
            # Update andere Konfigurationswerte
            for key, value in self.config.items():
                if key not in ['DEDUKTIVE_KATEGORIEN', 'CONFIG'] and key in globals_dict:
                    if isinstance(value, dict) and isinstance(globals_dict[key], dict):
                        globals_dict[key].clear()
                        globals_dict[key].update(value)
                        print(f"Dict {key} aktualisiert")
                    else:
                        globals_dict[key] = value
                        print(f"Variable {key} aktualisiert")

            # Validiere Update
            if 'DEDUKTIVE_KATEGORIEN' in globals_dict:
                kat_count = len(globals_dict['DEDUKTIVE_KATEGORIEN'])
                print(f"\nFinale Validierung: {kat_count} Kategorien in globalem Namespace")
                if kat_count == 0:
                    print("Warnung: Keine Kategorien im globalen Namespace!")
            else:
                print("Fehler: DEDUKTIVE_KATEGORIEN nicht im globalen Namespace!")

            # Finale Validierung der MULTIPLE_CODINGS Einstellung
            if 'CONFIG' in globals_dict and 'MULTIPLE_CODINGS' in globals_dict['CONFIG']:
                final_multiple_codings = globals_dict['CONFIG']['MULTIPLE_CODINGS']
                print(f"\n🔍 FINALE MULTIPLE_CODINGS Einstellung: {final_multiple_codings}")
                if not final_multiple_codings:
                    print("✅ Mehrfachkodierung wurde DEAKTIVIERT")
                else:
                    print("ℹ️ Mehrfachkodierung bleibt AKTIVIERT")

        except Exception as e:
            print(f"Fehler beim Update der globalen Variablen: {str(e)}")
            import traceback
            traceback.print_exc()

    def _load_validation_config(self, wb):
        """Lädt die Validierungskonfiguration aus dem Codebook."""
        if 'CONFIG' not in wb.sheetnames:
            return {}
            
        validation_config = {
            'thresholds': {},
            'english_words': set(),
            'messages': {}
        }
        
        df = pd.read_excel(self.excel_path, sheet_name='CONFIG')
        validation_rows = df[df['Key'] == 'VALIDATION']
        
        for _, row in validation_rows.iterrows():
            sub_key = row['Sub-Key']
            sub_sub_key = row['Sub-Sub-Key']
            value = row['Value']
            
            if sub_key == 'ENGLISH_WORDS':
                if value == 'true':
                    validation_config['english_words'].add(sub_sub_key)
            elif sub_key == 'MESSAGES':
                validation_config['messages'][sub_sub_key] = value
            else:
                # Numerische Schwellenwerte
                try:
                    validation_config['thresholds'][sub_key] = float(value)
                except (ValueError, TypeError):
                    print(f"Warnung: Ungültiger Schwellenwert für {sub_key}: {value}")
        
        return validation_config

    def get_config(self):
        return self.config

    
"""
LLM Provider Module für QCA-AID
Unterstützt OpenAI und Mistral APIs mit einheitlicher Schnittstelle
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Lade Umgebungsvariablen
env_path = os.path.join(os.path.expanduser("~"), '.environ.env')
load_dotenv(env_path)

@dataclass
class LLMResponse:
    """Wrapper für LLM-Antworten um einheitliche Schnittstelle zu gewährleisten"""
    content: str
    model: str = ""
    usage: Dict = None
    
    def __init__(self, response):
        """
        Initialisiert LLMResponse basierend auf dem Provider-Response-Format
        
        Args:
            response: Rohe Antwort vom LLM Provider (OpenAI oder Mistral Format)
        """
        if hasattr(response, 'choices') and response.choices:
            # OpenAI Format
            self.content = response.choices[0].message.content
            self.model = getattr(response, 'model', '')
            self.usage = getattr(response, 'usage', None)
        elif hasattr(response, 'content'):
            # Mistral Format
            self.content = response.content
            self.model = getattr(response, 'model', '')
            self.usage = getattr(response, 'usage', None)
        else:
            # Fallback für unbekannte Formate
            self.content = str(response)
            self.model = "unknown"
            self.usage = None

class LLMProvider(ABC):
    """Abstrakte Basisklasse für LLM Provider"""
    
    def __init__(self):
        self.client = None
        self.model_name = None
        self.initialize_client()
    
    @abstractmethod
    def initialize_client(self):
        """Initialisiert den Client für den jeweiligen Provider"""
        pass
    
    @abstractmethod
    async def create_completion(self, 
                              model: str,
                              messages: List[Dict],
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              response_format: Optional[Dict] = None) -> Any:
        """Erstellt eine Chat Completion"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI Provider Implementation"""
    
    def initialize_client(self):
        """Initialisiert den OpenAI Client"""
        try:
            from openai import AsyncOpenAI
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY nicht in Umgebungsvariablen gefunden")
            
            self.client = AsyncOpenAI(api_key=api_key)
            print("OpenAI Client erfolgreich initialisiert")
            
        except ImportError:
            raise ImportError("OpenAI Bibliothek nicht installiert. Bitte installieren Sie: pip install openai")
        except Exception as e:
            raise Exception(f"Fehler bei OpenAI Client-Initialisierung: {str(e)}")
    
    async def create_completion(self, 
                              model: str,
                              messages: List[Dict],
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              response_format: Optional[Dict] = None) -> Any:
        """
        Erstellt eine OpenAI Chat Completion
        
        Args:
            model: Name des zu verwendenden Modells
            messages: Liste der Chat-Nachrichten
            temperature: Temperatur für die Antwortgenerierung
            max_tokens: Maximale Anzahl von Tokens (optional)
            response_format: Format der Antwort (optional, z.B. {"type": "json_object"})
            
        Returns:
            OpenAI ChatCompletion Response
        """
        try:
            # Erstelle Parameter-Dict
            params = {
                'model': model,
                'messages': messages,
                'temperature': temperature
            }
            
            # Füge optionale Parameter hinzu
            if max_tokens:
                params['max_tokens'] = max_tokens
            if response_format:
                params['response_format'] = response_format
            
            # API Call
            response = await self.client.chat.completions.create(**params)
            return response
            
        except Exception as e:
            print(f"Fehler bei OpenAI API Call: {str(e)}")
            raise

class MistralProvider(LLMProvider):
    """Mistral Provider Implementation"""
    
    def initialize_client(self):
        """Initialisiert den Mistral Client"""
        try:
            from mistralai import Mistral
            
            api_key = os.getenv('MISTRAL_API_KEY')
            if not api_key:
                raise ValueError("MISTRAL_API_KEY nicht in Umgebungsvariablen gefunden")
            
            self.client = Mistral(api_key=api_key)
            print("Mistral Client erfolgreich initialisiert")
            
        except ImportError:
            raise ImportError("Mistral Bibliothek nicht installiert. Bitte installieren Sie: pip install mistralai")
        except Exception as e:
            raise Exception(f"Fehler bei Mistral Client-Initialisierung: {str(e)}")
    
    async def create_completion(self, 
                              model: str,
                              messages: List[Dict],
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              response_format: Optional[Dict] = None) -> Any:
        """
        Erstellt eine Mistral Chat Completion
        
        Args:
            model: Name des zu verwendenden Modells
            messages: Liste der Chat-Nachrichten
            temperature: Temperatur für die Antwortgenerierung
            max_tokens: Maximale Anzahl von Tokens (optional)
            response_format: Format der Antwort (optional, wird bei Mistral ignoriert)
            
        Returns:
            Mistral ChatCompletion Response
        """
        try:
            # Erstelle Parameter-Dict
            params = {
                'model': model,
                'messages': messages,
                'temperature': temperature
            }
            
            # Füge optionale Parameter hinzu
            if max_tokens:
                params['max_tokens'] = max_tokens
            
            # Hinweis: Mistral unterstützt response_format möglicherweise nicht
            if response_format:
                print("Warnung: response_format wird von Mistral möglicherweise nicht unterstützt")
            
            # API Call (synchron, da Mistral möglicherweise kein async unterstützt)
            response = await self._make_async_call(params)
            return response
            
        except Exception as e:
            print(f"Fehler bei Mistral API Call: {str(e)}")
            raise
    
    async def _make_async_call(self, params):
        """Wrapper um synchrone Mistral Calls asynchron zu machen"""
        try:
            # Führe synchronen Call in Thread Pool aus
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.chat.complete(**params)
            )
            return response
        except Exception as e:
            print(f"Fehler bei async Mistral Call: {str(e)}")
            raise

class LLMProviderFactory:
    """Factory Klasse zur Erstellung von LLM Providern"""
    
    @staticmethod
    def create_provider(provider_name: str) -> LLMProvider:
        """
        Erstellt einen LLM Provider basierend auf dem Namen
        
        Args:
            provider_name: Name des Providers ('openai' oder 'mistral')
            
        Returns:
            LLMProvider: Initialisierter Provider
            
        Raises:
            ValueError: Wenn ein unbekannter Provider angefordert wird
        """
        provider_name = provider_name.lower().strip()
        
        print(f"Initialisiere LLM Provider: {provider_name}")
        
        try:
            if provider_name in ['openai', 'gpt']:
                return OpenAIProvider()
            elif provider_name in ['mistral', 'mistralai']:
                return MistralProvider()
            else:
                raise ValueError(f"Unbekannter LLM Provider: {provider_name}. "
                               f"Unterstützte Provider: 'openai', 'mistral'")
                
        except Exception as e:
            print(f"Fehler bei Provider-Erstellung: {str(e)}")
            raise

"""
    Export Utilities 
"""

import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from typing import Dict, List

def _sanitize_text_for_excel(text):
    """
    Bereinigt Text für Excel-Export, entfernt ungültige Zeichen.
    
    Args:
        text: Zu bereinigender Text
        
    Returns:
        str: Bereinigter Text ohne problematische Zeichen
    """
    if text is None:
        return ""
        
    if not isinstance(text, str):
        # Konvertiere zu String falls nötig
        text = str(text)
    
    # Liste von problematischen Zeichen, die in Excel Probleme verursachen können
    # Hier definieren wir Steuerzeichen und einige bekannte Problemzeichen
    problematic_chars = [
        # ASCII-Steuerzeichen 0-31 außer Tab (9), LF (10) und CR (13)
        *[chr(i) for i in range(0, 9)],
        *[chr(i) for i in range(11, 13)],
        *[chr(i) for i in range(14, 32)],
        # Einige bekannte problematische Sonderzeichen
        '\u0000', '\u0001', '\u0002', '\u0003', '\ufffe', '\uffff',
        # Emojis und andere Sonderzeichen, die Probleme verursachen könnten
        '☺', '☻', '♥', '♦', '♣', '♠'
    ]
    
    # Ersetze alle problematischen Zeichen
    for char in problematic_chars:
        text = text.replace(char, '')
    
    # Alternative Methode mit Regex für Steuerzeichen
    import re
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFE\uFFFF]', '', text)
    
    return text

def _generate_pastel_colors(num_colors):
    """
    Generiert eine Palette mit Pastellfarben.
    
    Args:
        num_colors (int): Anzahl der benötigten Farben
    
    Returns:
        List[str]: Liste von Hex-Farbcodes in Pastelltönen
    """
    import colorsys
    
    pastel_colors = []
    for i in range(num_colors):
        # Wähle Hue gleichmäßig über Farbkreis
        hue = i / num_colors
        # Konvertiere HSV zu RGB mit hoher Helligkeit und Sättigung
        rgb = colorsys.hsv_to_rgb(hue, 0.4, 0.95)
        # Konvertiere RGB zu Hex
        hex_color = 'FF{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), 
            int(rgb[1] * 255), 
            int(rgb[2] * 255)
        )
        pastel_colors.append(hex_color)
    
    return pastel_colors

    
def _format_confidence(confidence: dict) -> str:
    """Formatiert die Konfidenz-Werte für den Export"""
    try:
        if isinstance(confidence, dict):
            formatted_values = []
            # Verarbeite jeden Konfidenzwert einzeln
            for key, value in confidence.items():
                if isinstance(value, (int, float)):
                    formatted_values.append(f"{key}: {value:.2f}")
                elif isinstance(value, dict):
                    # Verarbeite verschachtelte Konfidenzwerte
                    nested_values = [f"{k}: {v:.2f}" for k, v in value.items() 
                                if isinstance(v, (int, float))]
                    if nested_values:
                        formatted_values.append(f"{key}: {', '.join(nested_values)}")
                elif isinstance(value, str):
                    formatted_values.append(f"{key}: {value}")
            
            return "\n".join(formatted_values)
        elif isinstance(confidence, (int, float)):
            return f"{float(confidence):.2f}"
        elif isinstance(confidence, str):
            return confidence
        else:
            return "0.00"
            
    except Exception as e:
        print(f"Fehler bei Konfidenz-Formatierung: {str(e)}")
        return "0.00"


# Abbruch der Analyse managen 
    
import keyboard
import threading
import time
import select
import sys
from typing import Optional, Callable, Any, TYPE_CHECKING

# Verwende TYPE_CHECKING um zirkuläre Imports zu vermeiden
if TYPE_CHECKING:
    from typing import TYPE_CHECKING
    # Hier würde normalerweise der Import stehen, aber wir verwenden Any

class EscapeHandler:
    """
    Handler für Escape-Taste um Kodierung sicher zu unterbrechen und Zwischenergebnisse zu speichern.
    """
    
    def __init__(self, analysis_manager: Any):  # Verwende Any statt 'IntegratedAnalysisManager'
        self.analysis_manager = analysis_manager
        self.escape_pressed = False
        self.user_wants_to_abort = False
        self.keyboard_thread = None
        self.monitoring = False
        self._keyboard_available = self._check_keyboard_availability()
        
    def _check_keyboard_availability(self) -> bool:
        """Prüft ob keyboard-Modul verfügbar ist"""
        try:
            import keyboard
            return True
        except ImportError:
            print("⚠️ 'keyboard' Modul nicht installiert. ESC-Handler nicht verfügbar.")
            print("   Installieren Sie mit: pip install keyboard")
            return False
        
    def start_monitoring(self):
        """Startet die Überwachung der Escape-Taste"""
        if not self._keyboard_available:
            return
            
        if self.monitoring:
            return
            
        self.monitoring = True
        self.escape_pressed = False
        self.user_wants_to_abort = False
        
        print("\n💡 Tipp: Drücken Sie ESC um die Kodierung sicher zu unterbrechen und Zwischenergebnisse zu speichern")
        
        # Starte Keyboard-Monitoring in separatem Thread
        self.keyboard_thread = threading.Thread(target=self._monitor_escape, daemon=True)
        self.keyboard_thread.start()
    
    def stop_monitoring(self):
        """Stoppt die Überwachung der Escape-Taste"""
        self.monitoring = False
        if self.keyboard_thread and self.keyboard_thread.is_alive():
            # Warte kurz auf das Ende des Threads
            self.keyboard_thread.join(timeout=1.0)
    
    def _monitor_escape(self):
        """Überwacht die Escape-Taste in separatem Thread"""
        if not self._keyboard_available:
            return
            
        try:
            import keyboard
            while self.monitoring:
                if keyboard.is_pressed('esc'):
                    if not self.escape_pressed:  # Verhindere mehrfache Verarbeitung
                        self.escape_pressed = True
                        self._handle_escape()
                    break
                time.sleep(0.1)  # Kleine Pause um CPU zu schonen
        except Exception as e:
            print(f"Fehler bei Escape-Überwachung: {str(e)}")
    
    def _handle_escape(self):
        """Behandelt das Drücken der Escape-Taste - VERBESSERT"""
        try:
            print("\n\n" + "="*60)
            print("🛑 ESCAPE-TASTE GEDRÜCKT - KODIERUNG UNTERBRECHEN?")
            print("="*60)
            
            # Zeige aktuellen Status
            current_status = self._get_current_status()
            print(f"\nAktueller Status:")
            print(f"- Verarbeitete Segmente: {current_status['processed_segments']}")
            print(f"- Erstellte Kodierungen: {current_status['total_codings']}")
            print(f"- Verstrichene Zeit: {current_status['elapsed_time']:.1f} Sekunden")
            
            if current_status['total_codings'] > 0:
                print(f"\n✅ {current_status['total_codings']} Kodierungen wurden bereits erstellt")
                print("   Diese können als Zwischenergebnisse exportiert werden.")
            else:
                print("\n⚠️  Noch keine Kodierungen vorhanden")
                print("   Ein Export würde leere Ergebnisse erzeugen.")
            
            print("\n" + "="*60)
            print("OPTIONEN:")
            print("j + ENTER = Kodierung beenden und Zwischenergebnisse exportieren")
            print("n + ENTER = Kodierung fortsetzen") 
            print("ESC       = Sofort beenden ohne Export")
            print("="*60)
            
            # Warte auf Benutzereingabe mit Timeout
            choice = self._get_user_choice_with_timeout(timeout=30)
            
            if choice == 'j':
                print("\n✅ Kodierung wird beendet - Zwischenergebnisse werden exportiert...")
                self.user_wants_to_abort = True
                self._trigger_safe_abort()
                
            elif choice == 'n':
                print("\n▶️  Kodierung wird fortgesetzt...")
                self.escape_pressed = False  # Reset für weitere ESC-Presses
                self.start_monitoring()  # Überwachung wieder starten
                
            elif choice == 'abort_immediately':
                print("\n🛑 Sofortiger Abbruch ohne Export...")
                self.user_wants_to_abort = True
                setattr(self.analysis_manager, '_immediate_abort', True)
                self._trigger_safe_abort()
                
            else:  # Timeout oder ungültige Eingabe
                print("\n⏰ Keine gültige Eingabe - Kodierung wird fortgesetzt...")
                self.escape_pressed = False
                self.start_monitoring()
                
        except Exception as e:
            print(f"Fehler bei Escape-Behandlung: {str(e)}")
            print("Kodierung wird fortgesetzt...")
            self.escape_pressed = False
            self.start_monitoring()
    
    def _get_user_choice_with_timeout(self, timeout: int = 30) -> str:
        """Holt Benutzereingabe mit Timeout - KORRIGIERT"""
        print(f"\nIhre Wahl (Timeout in {timeout} Sekunden): ", end="", flush=True)
        
        try:
            if sys.platform == "win32":
                # Windows-spezifische Implementierung mit verbesserter Eingabe-Behandlung
                try:
                    import msvcrt
                    start_time = time.time()
                    input_chars = []
                    
                    while time.time() - start_time < timeout:
                        if msvcrt.kbhit():
                            char = msvcrt.getch()
                            
                            # Handle verschiedene Eingabetypen
                            if char == b'\x1b':  # ESC
                                print("ESC (Sofortiger Abbruch)")
                                return 'abort_immediately'
                            elif char == b'\r':  # Enter
                                user_input = ''.join(input_chars).lower().strip()
                                print()  # Neue Zeile nach Enter
                                if user_input in ['j', 'n']:
                                    return user_input
                                else:
                                    print("Ungültige Eingabe. Bitte 'j' oder 'n' eingeben: ", end="", flush=True)
                                    input_chars = []
                                    continue
                            elif char == b'\x08':  # Backspace
                                if input_chars:
                                    input_chars.pop()
                                    print('\b \b', end="", flush=True)  # Löscht Zeichen visuell
                            else:
                                try:
                                    decoded_char = char.decode('utf-8')
                                    if decoded_char.isprintable():
                                        input_chars.append(decoded_char)
                                        print(decoded_char, end="", flush=True)
                                except UnicodeDecodeError:
                                    pass  # Ignoriere nicht-dekodierbare Zeichen
                        
                        time.sleep(0.05)  # Kürzere Pause für responsivere Eingabe
                    
                    print()  # Neue Zeile nach Timeout
                    return ''  # Timeout erreicht
                    
                except ImportError:
                    print("\nmsvcrt nicht verfügbar, verwende Fallback...")
                    return self._get_input_fallback(timeout)
                    
            else:
                # Unix/Linux-spezifische Implementierung
                return self._get_input_unix_with_timeout(timeout)
                
        except Exception as e:
            print(f"\nFehler bei Eingabe-Behandlung: {str(e)}")
            return self._get_input_fallback(timeout)

    def _get_input_unix_with_timeout(self, timeout: int) -> str:
        """Unix/Linux-spezifische Eingabe mit Timeout"""
        try:
            import select
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                choice = sys.stdin.readline().strip().lower()
                if choice in ['j', 'n']:
                    return choice
                elif choice == 'esc' or choice == '\x1b':
                    return 'abort_immediately'
                else:
                    print("Ungültige Eingabe.")
                    return ''
            else:
                print()  # Neue Zeile nach Timeout
                return ''  # Timeout
        except Exception as e:
            print(f"Fehler bei Unix-Eingabe: {str(e)}")
            return self._get_input_fallback(timeout)

    def _get_input_fallback(self, timeout: int) -> str:
        """Verbesserte Fallback Input-Methode"""
        try:
            print(f"\n(Einfache Eingabe-Methode - kein Timeout verfügbar)")
            print("Geben Sie 'j' für Beenden oder 'n' für Fortsetzen ein:")
            
            while True:
                try:
                    choice = input("Ihre Wahl [j/n]: ").strip().lower()
                    if choice in ['j', 'n']:
                        return choice
                    elif choice in ['esc', 'escape', 'abbruch']:
                        return 'abort_immediately'
                    else:
                        print("Ungültige Eingabe. Bitte 'j' oder 'n' eingeben.")
                except KeyboardInterrupt:
                    print("\nCtrl+C erkannt - sofortiger Abbruch")
                    return 'abort_immediately'
                except EOFError:
                    print("\nEingabe-Ende erkannt - Kodierung wird fortgesetzt")
                    return 'n'
                    
        except Exception as e:
            print(f"Fehler bei Fallback-Eingabe: {str(e)}")
            return 'n'  # Standard: Fortsetzen
    
    def _get_current_status(self) -> dict:
        """Holt den aktuellen Status der Analyse"""
        try:
            # Versuche get_progress_report zu verwenden
            if hasattr(self.analysis_manager, 'get_progress_report'):
                progress_report = self.analysis_manager.get_progress_report()
                return progress_report.get('progress', {
                    'processed_segments': 0,
                    'total_codings': 0,
                    'elapsed_time': 0
                })
            
            # Fallback: Direkte Attribut-Zugriffe
            processed_segments = 0
            total_codings = 0
            elapsed_time = 0
            
            if hasattr(self.analysis_manager, 'processed_segments'):
                processed_segments = len(getattr(self.analysis_manager, 'processed_segments', []))
            
            if hasattr(self.analysis_manager, 'coding_results'):
                total_codings = len(getattr(self.analysis_manager, 'coding_results', []))
            
            if hasattr(self.analysis_manager, 'start_time'):
                from datetime import datetime
                start_time = getattr(self.analysis_manager, 'start_time', None)
                if start_time:
                    elapsed_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'processed_segments': processed_segments,
                'total_codings': total_codings,
                'elapsed_time': elapsed_time
            }
            
        except Exception as e:
            print(f"Fehler beim Ermitteln des Status: {str(e)}")
            return {
                'processed_segments': 0,
                'total_codings': 0,
                'elapsed_time': 0
            }
    
    def _trigger_safe_abort(self):
        """Löst einen sicheren Abbruch aus"""
        # Setze Flag für die Hauptschleife
        if hasattr(self.analysis_manager, '_should_abort'):
            self.analysis_manager._should_abort = True
        
        # Alternative: Setze ein neues Attribut
        setattr(self.analysis_manager, '_escape_abort_requested', True)
        
        print("🔄 Abbruch-Signal gesendet...")
    
    def should_abort(self) -> bool:
        """Prüft ob abgebrochen werden soll"""
        return self.user_wants_to_abort


# Hilfsfunktion für die Integration in bestehende Klassen
def add_escape_handler_to_manager(manager_instance) -> EscapeHandler:
    """
    Fügt einen EscapeHandler zu einem bestehenden Analysis Manager hinzu.
    
    Args:
        manager_instance: Instanz des IntegratedAnalysisManager
        
    Returns:
        EscapeHandler: Konfigurierter EscapeHandler
    """
    escape_handler = EscapeHandler(manager_instance)
    
    # Füge Escape-Handler als Attribut hinzu
    setattr(manager_instance, 'escape_handler', escape_handler)
    
    # Füge Abort-Flag hinzu falls nicht vorhanden
    if not hasattr(manager_instance, '_should_abort'):
        setattr(manager_instance, '_should_abort', False)
    
    # Füge Helper-Methode hinzu
    def check_escape_abort(self):
        """Prüft ob durch Escape abgebrochen werden soll"""
        return (getattr(self, '_should_abort', False) or 
                getattr(self, '_escape_abort_requested', False) or
                (hasattr(self, 'escape_handler') and self.escape_handler.should_abort()))
    
    # Binde die Methode an die Instanz
    import types
    manager_instance.check_escape_abort = types.MethodType(check_escape_abort, manager_instance)
    
    return escape_handler


# Decorator für automatische Escape-Handler Integration
def with_escape_handler(cls):
    """
    Decorator um automatisch einen Escape-Handler zu einer Klasse hinzuzufügen.
    
    Usage:
        @with_escape_handler
        class MyAnalysisManager:
            pass
    """
    original_init = cls.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.escape_handler = EscapeHandler(self)
        self._should_abort = False
        
        # Füge check_escape_abort Methode hinzu
        def check_escape_abort():
            return (getattr(self, '_should_abort', False) or 
                    getattr(self, '_escape_abort_requested', False) or
                    self.escape_handler.should_abort())
        
        self.check_escape_abort = check_escape_abort
    
    cls.__init__ = new_init
    return cls