"""
Manuelle Kodierung für QCA-AID
===============================
Manuelles Kodieren von Textsegmenten mit GUI-Unterstützung.
"""

import json
import asyncio
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Optional, List, Union, Any

from ..core.data_models import CategoryDefinition, CodingResult
from ..QCA_Utils import (
    setup_manual_coding_window_enhanced, 
    validate_multiple_selection,
    create_multiple_coding_results,
    show_multiple_coding_info,
    MultiSelectListbox,
    ManualMultipleCodingDialog
)

class ManualCoder:
    def __init__(self, coder_id: str):
        self.coder_id = coder_id
        self.root = None
        self.text_chunk = None
        self.category_listbox = None
        self.categories = {}
        self.current_coding = None
        self._is_processing = False
        self.current_categories = {}
        self.is_last_segment = False
        
        # NEU: Mehrfachkodierung-Support
        self.multiple_selection_enabled = True
        self.category_map = {}  # Mapping von Listbox-Index zu Kategorie-Info

        
    def _on_selection_change(self, event):
        """
        ERWEITERT: Behandelt Änderungen der Kategorienauswahl mit Nummern-Info
        """
        try:
            selected_indices = self.category_listbox.curselection()
            
            if not selected_indices:
                self.selection_info_label.config(text="Keine Auswahl", foreground='gray')
                return
            
            # Analysiere Auswahl mit Nummern
            selected_categories = []
            main_categories = set()
            selected_numbers = []
            
            for idx in selected_indices:
                if idx in self.category_map:
                    cat_info = self.category_map[idx]
                    selected_categories.append(cat_info)
                    main_categories.add(cat_info['main_category'])
                    selected_numbers.append(cat_info.get('number', '?'))
            
            # Erstelle Infotext mit Nummern
            if len(selected_indices) == 1:
                cat_info = selected_categories[0]
                number = cat_info.get('number', '?')
                if cat_info['type'] == 'main':
                    info_text = f"Nr. {number}: Hauptkategorie '{cat_info['name']}'"
                else:
                    info_text = f"Nr. {number}: Subkategorie '{cat_info['name']}' -> {cat_info['main_category']}"
                self.selection_info_label.config(text=info_text, foreground='black')
            else:
                numbers_text = ", ".join(selected_numbers)
                if len(main_categories) == 1:
                    info_text = f"Nummern {numbers_text}: {len(selected_indices)} Subkategorien von '{list(main_categories)[0]}'"
                    self.selection_info_label.config(text=info_text, foreground='blue')
                else:
                    info_text = f"Nummern {numbers_text}: Mehrfachkodierung ({len(selected_indices)} Kategorien aus {len(main_categories)} Hauptkategorien)"
                    self.selection_info_label.config(text=info_text, foreground='orange')
        except Exception as e:
            print(f"Fehler bei Auswahlaktualisierung: {str(e)}")

    def get_category_by_number(self, number_input: str) -> dict:
        """
        NEU: Gibt Kategorie-Info basierend auf Nummern-Eingabe zurÜck
        
        Args:
            number_input: Eingabe wie "1", "1.2", "3" etc.
            
        Returns:
            dict: Kategorie-Information oder None wenn nicht gefunden
        """
        return self.number_to_category_map.get(number_input.strip())

    def update_category_list_enhanced(self):
        """
        ERWEITERT: Aktualisiert die Kategorienliste mit NUMMERIERUNG fuer bessere Übersicht
        """
        if not self.category_listbox:
            return
            
        self.category_listbox.delete(0, tk.END)
        self.category_map = {}
        self.number_to_category_map = {}  # NEU: Mapping von Nummern zu Kategorien
        
        current_index = 0
        main_category_number = 1
        
        # Sortiere Kategorien alphabetisch fuer bessere Übersicht
        sorted_categories = sorted(self.categories.items())
        
        for cat_name, cat_def in sorted_categories:
            # Hauptkategorie hinzufÜgen mit Nummerierung
            main_number = str(main_category_number)
            display_text = f"{main_number}. 🔀 {cat_name}"
            self.category_listbox.insert(tk.END, display_text)
            
            # Mapping fuer Index und Nummer
            self.category_map[current_index] = {
                'type': 'main',
                'name': cat_name,
                'main_category': cat_name,
                'number': main_number
            }
            self.number_to_category_map[main_number] = {
                'type': 'main',
                'name': cat_name,
                'main_category': cat_name
            }
            
            current_index += 1
            sub_category_number = 1
            
            # Subkategorien hinzufÜgen (eingerÜckt und nummeriert)
            if hasattr(cat_def, 'subcategories') and cat_def.subcategories:
                sorted_subcats = sorted(cat_def.subcategories.items())
                for sub_name, sub_def in sorted_subcats:
                    sub_number = f"{main_category_number}.{sub_category_number}"
                    display_text = f"    {sub_number} 🔀„ {sub_name}"
                    self.category_listbox.insert(tk.END, display_text)
                    
                    # Mapping fuer Index und Nummer
                    self.category_map[current_index] = {
                        'type': 'sub',
                        'name': sub_name,
                        'main_category': cat_name,
                        'definition': sub_def,
                        'number': sub_number
                    }
                    self.number_to_category_map[sub_number] = {
                        'type': 'sub',
                        'name': sub_name,
                        'main_category': cat_name,
                        'definition': sub_def
                    }
                    
                    current_index += 1
                    sub_category_number += 1
            
            main_category_number += 1

        # Scrolle zum Anfang
        if self.category_listbox.size() > 0:
            self.category_listbox.see(0)
        
        print(f"Nummerierte Kategorieliste aktualisiert: {len(self.category_map)} EintrÄge")
        
        # Zeige Nummern-Referenz in einem Label an
        self._update_number_reference()

    def _update_number_reference(self):
        """
        NEU: Zeigt eine kompakte Nummern-Referenz fuer schnelle Eingabe
        """
        if not hasattr(self, 'number_reference_label'):
            return
            
        # Erstelle kompakte Referenz
        reference_lines = []
        main_cats = []
        
        for number, info in self.number_to_category_map.items():
            if info['type'] == 'main':
                main_cats.append(f"{number}={info['name'][:15]}")
            elif len(reference_lines) < 5:  # Zeige nur ersten paar Subkategorien
                reference_lines.append(f"{number}={info['name'][:10]}")
        
        # Kompakte Anzeige
        main_line = " | ".join(main_cats)
        sub_line = " | ".join(reference_lines[:3]) + ("..." if len(reference_lines) > 3 else "")
        
        reference_text = f"Hauptkat.: {main_line}\nSubkat.: {sub_line}"
        self.number_reference_label.config(text=reference_text)

    def _safe_code_selection_enhanced(self):
        """
        ERWEITERT: Thread-sichere Kodierungsauswahl mit Mehrfachkodierung-Support
        """
        if not self._is_processing:
            try:
                self._is_processing = True
                
                selected_indices = list(self.category_listbox.curselection())
                if not selected_indices:
                    messagebox.showwarning("Warnung", "Bitte wÄhlen Sie mindestens eine Kategorie aus.")
                    self._is_processing = False
                    return
                
                # Validiere Auswahl
                from ..QCA_Utils import validate_multiple_selection
                is_valid, error_msg, selected_categories = validate_multiple_selection(
                    selected_indices, self.category_map
                )
                
                if not is_valid:
                    messagebox.showerror("Fehler", error_msg)
                    self._is_processing = False
                    return
                
                # Verarbeite Auswahl
                if len(selected_indices) == 1:
                    # Einzelauswahl - wie bisher
                    self._process_single_selection(selected_indices[0])
                else:
                    # Mehrfachauswahl - neue Logik
                    self._process_multiple_selection(selected_categories)
                
                # Bei letztem Segment Hinweis anzeigen
                if self.is_last_segment:
                    messagebox.showinfo(
                        "Kodierung abgeschlossen",
                        "Die Kodierung des letzten Segments wurde abgeschlossen.\n"
                        "Der manuelle Kodierungsprozess wird beendet."
                    )
                
                self._is_processing = False
                
                # Fenster schlieẞen
                if self.root:
                    try:
                        self.root.destroy()
                        self.root.quit()
                    except Exception as e:
                        print(f"Fehler beim Schlieẞen des Fensters: {str(e)}")
                        
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler bei der Kategorieauswahl: {str(e)}")
                print(f"Fehler bei der Kategorieauswahl: {str(e)}")
                import traceback
                traceback.print_exc()
                self._is_processing = False

    def _process_single_selection(self, index: int):
        """
        Verarbeitet eine Einzelauswahl (als Dictionary statt CodingResult)
        """
        category_info = self.category_map[index]
        
        if category_info['type'] == 'main':
            main_cat = category_info['name']
            sub_cat = None
        else:
            main_cat = category_info['main_category']
            sub_cat = category_info['name']
        
        # Verifiziere Kategorien
        if main_cat not in self.categories:
            messagebox.showerror(
                "Fehler",
                f"Hauptkategorie '{main_cat}' nicht gefunden.\n"
                f"VerfÜgbare Kategorien: {', '.join(self.categories.keys())}"
            )
            return
            
        if sub_cat and sub_cat not in self.categories[main_cat].subcategories:
            messagebox.showerror(
                "Fehler",
                f"Subkategorie '{sub_cat}' nicht in '{main_cat}' gefunden.\n"
                f"VerfÜgbare Subkategorien: {', '.join(self.categories[main_cat].subcategories.keys())}"
            )
            return

        # Erstelle Einzelkodierung als Dictionary
        self.current_coding = {
            'category': main_cat,
            'subcategories': [sub_cat] if sub_cat else [],
            'justification': "Manuelle Kodierung",
            'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
            'text_references': [self.text_chunk.get("1.0", tk.END)[:100]],
            'uncertainties': None,
            'paraphrase': "",
            'keywords': "",
            'manual_coding': True,
            'manual_multiple_coding': False,
            'multiple_coding_instance': 1,
            'total_coding_instances': 1,
            'coding_date': datetime.now().isoformat(),
            'coder_id': self.coder_id
        }
        
        print(f"Einzelkodierung erstellt: {main_cat}" + (f" -> {sub_cat}" if sub_cat else ""))

    def _process_multiple_selection(self, selected_categories: List[Dict]):
        """
        NEUE METHODE: Verarbeitet Mehrfachauswahl von Kategorien
        """
        # Analysiere Auswahltyp
        main_categories = set(cat['main_category'] for cat in selected_categories)
        
        # BestÄtigungsdialog anzeigen
        from ..QCA_Utils import show_multiple_coding_info
        confirmed = show_multiple_coding_info(
            self.root,
            len(selected_categories),
            list(main_categories)
        )
        
        if not confirmed:
            print("Mehrfachkodierung abgebrochen")
            return
        
        # Erstelle Mehrfachkodierung
        if len(main_categories) == 1:
            # Alle Auswahlen gehÖren zu einer Hauptkategorie
            main_cat = list(main_categories)[0]
            subcategories = [
                cat['name'] for cat in selected_categories 
                if cat['type'] == 'sub'
            ]
            
            # FÜge Hauptkategorie hinzu, wenn sie direkt ausgewÄhlt wurde
            main_cat_selected = any(
                cat['type'] == 'main' for cat in selected_categories
            )
            
            self.current_coding = CodingResult(
                category=main_cat,
                subcategories=tuple(subcategories),
                justification=f"Manuelle Kodierung mit {len(subcategories)} Subkategorien",
                confidence={'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                text_references=(self.text_chunk.get("1.0", tk.END)[:100],)
            )
            
            print(f"Einzelkodierung mit mehreren Subkategorien: {main_cat} -> {', '.join(subcategories)}")
        else:
            # Echte Mehrfachkodierung: verschiedene Hauptkategorien
            from ..QCA_Utils import create_multiple_coding_results
            
            coding_results = create_multiple_coding_results(
                selected_categories=selected_categories,
                text=self.text_chunk.get("1.0", tk.END),
                coder_id=self.coder_id
            )
            
            self.current_coding = coding_results
            
            main_cat_names = [result.category if hasattr(result, 'category') else result['category'] 
                            for result in coding_results]
            print(f"Mehrfachkodierung erstellt: {len(coding_results)} Kodierungen fuer {', '.join(main_cat_names)}")

    def _safe_finish_coding_enhanced(self):
        """
        ERWEITERT: Thread-sicherer Abschluss mit Mehrfachkodierung-Support
        """
        if not self._is_processing and self.is_last_segment:
            if messagebox.askyesno(
                "Segment kodieren und abschlieẞen",
                "MÖchten Sie das aktuelle Segment kodieren und den manuellen Kodierungsprozess abschlieẞen?"
            ):
                # Verwende die erweiterte Kodierungslogik
                self._safe_code_selection_enhanced()

    def _safe_skip_chunk(self):
        """Thread-sicheres Überspringen (als Dictionary)"""
        if not self._is_processing:
            self.current_coding = {
                'category': "Nicht kodiert",
                'subcategories': [],
                'justification': "Chunk Übersprungen",
                'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                'text_references': [self.text_chunk.get("1.0", tk.END)[:100]],
                'uncertainties': None,
                'paraphrase': "",
                'keywords': "",
                'manual_coding': True,
                'manual_multiple_coding': False,
                'multiple_coding_instance': 1,
                'total_coding_instances': 1,
                'coding_date': datetime.now().isoformat(),
                'coder_id': self.coder_id
            }
            self._is_processing = False
            
            if self.is_last_segment:
                messagebox.showinfo(
                    "Kodierung abgeschlossen",
                    "Die Kodierung des letzten Segments wurde Übersprungen.\n"
                    "Der manuelle Kodierungsprozess wird beendet."
                )
            
            # KORRIGIERT: FÜge destroy() hinzu um das Fenster komplett zu schlieẞen
            try:
                if self.root and self.root.winfo_exists():
                    self.root.quit()
                    self.root.destroy()  # HINZUGEFÜGT: ZerstÖrt das Fenster komplett
                    self.root = None     # HINZUGEFÜGT: Setze Referenz auf None
            except Exception as e:
                print(f"Info: Fehler beim Schlieẞen des Fensters: {str(e)}")
                # Fallback: Setze root auf None auch bei Fehlern
                self.root = None

    def _safe_abort_coding(self):
        """
        KORRIGIERT: Explizite Abbruch-Funktion (Über Button)
        """
        if not self._is_processing:
            if messagebox.askyesno(
                "Kodierung komplett abbrechen",
                "MÖchten Sie die gesamte manuelle Kodierung beenden?\n\n"
                "Alle bisher kodierten Segmente werden gespeichert."
            ):
                print("Benutzer hat manuelle Kodierung komplett abgebrochen")
                self.current_coding = "ABORT_ALL"
                self._is_processing = False
                
                try:
                    if self.root and self.root.winfo_exists():
                        self.root.quit()
                except Exception as e:
                    print(f"Info: Abbruch-Bereinigung: {str(e)}")

    def _safe_new_main_category(self):
        """Thread-sichere neue Hauptkategorie (unverÄndert)"""
        if not self._is_processing:
            from tkinter import simpledialog
            new_cat = simpledialog.askstring(
                "Neue Hauptkategorie",
                "Geben Sie den Namen der neuen Hauptkategorie ein:"
            )
            if new_cat:
                if new_cat in self.categories:
                    messagebox.showwarning("Warnung", "Diese Kategorie existiert bereits.")
                    return
                self.categories[new_cat] = CategoryDefinition(
                    name=new_cat,
                    definition="",
                    examples=[],
                    rules=[],
                    subcategories={},
                    added_date=datetime.now().strftime("%Y-%m-%d"),
                    modified_date=datetime.now().strftime("%Y-%m-%d")
                )
                self.update_category_list_enhanced()

    def _safe_new_sub_category_enhanced(self):
        """
        ERWEITERT: Neue Subkategorie mit Nummern-Eingabe
        """
        if not self._is_processing:
            from tkinter import simpledialog
            
            # Zeige verfÜgbare Hauptkategorien mit Nummern
            main_cats_info = []
            for number, info in self.number_to_category_map.items():
                if info['type'] == 'main':
                    main_cats_info.append(f"{number} = {info['name']}")
            
            if not main_cats_info:
                messagebox.showwarning("Warnung", "Keine Hauptkategorien verfÜgbar.")
                return
            
            # Erstelle Eingabedialog mit Nummern-Auswahl
            dialog_text = (
                "VerfÜgbare Hauptkategorien:\n" + 
                "\n".join(main_cats_info) + 
                "\n\nGeben Sie die Nummer der Hauptkategorie ein:"
            )
            
            main_cat_input = simpledialog.askstring(
                "Hauptkategorie auswÄhlen (per Nummer)",
                dialog_text
            )
            
            if main_cat_input:
                # PrÜfe ob Eingabe eine gÜltige Nummer ist
                main_cat_info = self.number_to_category_map.get(main_cat_input.strip())
                
                if main_cat_info and main_cat_info['type'] == 'main':
                    main_cat_name = main_cat_info['name']
                    
                    # Dialog fuer neue Subkategorie
                    new_sub = simpledialog.askstring(
                        "Neue Subkategorie",
                        f"Geben Sie den Namen der neuen Subkategorie fuer\n'{main_cat_name}' (Nr. {main_cat_input}) ein:"
                    )
                    
                    if new_sub:
                        if new_sub in self.categories[main_cat_name].subcategories:
                            messagebox.showwarning("Warnung", "Diese Subkategorie existiert bereits.")
                            return
                            
                        # FÜge neue Subkategorie hinzu
                        self.categories[main_cat_name].subcategories[new_sub] = ""
                        
                        # Aktualisiere die Anzeige
                        self.update_category_list_enhanced()
                        
                        # Zeige Erfolg mit neuer Nummer
                        new_number = self._find_number_for_subcategory(main_cat_name, new_sub)
                        messagebox.showinfo(
                            "Subkategorie erstellt", 
                            f"'{new_sub}' wurde als Nr. {new_number} zu '{main_cat_name}' hinzugefÜgt"
                        )
                        
                elif main_cat_input.strip():
                    # Fallback: Versuche Namen-Eingabe
                    if main_cat_input.strip() in self.categories:
                        main_cat_name = main_cat_input.strip()
                        
                        new_sub = simpledialog.askstring(
                            "Neue Subkategorie",
                            f"Geben Sie den Namen der neuen Subkategorie fuer '{main_cat_name}' ein:"
                        )
                        
                        if new_sub and new_sub not in self.categories[main_cat_name].subcategories:
                            self.categories[main_cat_name].subcategories[new_sub] = ""
                            self.update_category_list_enhanced()
                            
                            new_number = self._find_number_for_subcategory(main_cat_name, new_sub)
                            messagebox.showinfo(
                                "Subkategorie erstellt", 
                                f"'{new_sub}' wurde als Nr. {new_number} hinzugefÜgt"
                            )
                    else:
                        messagebox.showwarning(
                            "Warnung", 
                            f"UngÜltige Eingabe: '{main_cat_input}'\n\nBitte verwenden Sie die Nummer (z.B. '1') oder den exakten Namen der Hauptkategorie."
                        )

    def _find_number_for_subcategory(self, main_cat_name: str, sub_name: str) -> str:
        """
        NEU: Findet die Nummer einer Subkategorie
        """
        for number, info in self.number_to_category_map.items():
            if (info['type'] == 'sub' and 
                info['main_category'] == main_cat_name and 
                info['name'] == sub_name):
                return number
        return "?"

    def on_closing(self):
        """Sicheres Schlieẞen des Fensters (unverÄndert)"""
        try:
            if messagebox.askokcancel("Beenden", "MÖchten Sie das Kodieren wirklich beenden?"):
                self.current_coding = None
                self._is_processing = False
                
                if hasattr(self, 'root') and self.root:
                    for attr_name in dir(self):
                        attr = getattr(self, attr_name) 
                        if hasattr(attr, '_tk'):
                            delattr(self, attr_name)
                    
                    try:
                        self.root.quit()
                        self.root.destroy()
                        self.root = None
                    except:
                        pass
        except:
            if hasattr(self, 'root') and self.root:
                try:
                    self.root.quit()
                    self.root.destroy()
                    self.root = None
                except:
                    pass

    def _cleanup_tkinter_resources(self):
        """Bereinigt alle Tkinter-Ressourcen (unverÄndert)"""
        try:
            for attr_name in list(self.__dict__.keys()):
                if attr_name.startswith('_tk_var_'):
                    delattr(self, attr_name)
                    
            self.text_chunk = None
            self.category_listbox = None
            
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Warnung: Fehler bei der Bereinigung von Tkinter-Ressourcen: {str(e)}") 
        
    async def code_chunk(self, chunk: str, categories: Optional[Dict[str, CategoryDefinition]], is_last_segment: bool = False) -> Optional[Union[Dict, List[Dict]]]:
        """
        KORRIGIERT: Minimale Tkinter-Bereinigung nach MainLoop
        """
        try:
            self.categories = self.current_categories or categories
            self.current_coding = None
            self.is_last_segment = is_last_segment
            
            # Erstelle und starte das Tkinter-Fenster im Hauptthread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_enhanced_tk_window, chunk)
            
            # KORRIGIERT: PrÜfe auf ABORT_ALL BEVOR weitere Verarbeitung
            if self.current_coding == "ABORT_ALL":
                return "ABORT_ALL"
            
            # KORRIGIERT: Minimale Bereinigung - root sollte bereits None sein
            if hasattr(self, 'root'):
                self.root = None
            
            # Rest der Verarbeitung bleibt gleich...
            if self.current_coding:
                if isinstance(self.current_coding, list):
                    enhanced_codings = []
                    for coding_dict in self.current_coding:
                        enhanced_coding = coding_dict.copy()
                        enhanced_coding['text'] = chunk
                        enhanced_codings.append(enhanced_coding)
                    self.current_coding = enhanced_codings
                else:
                    if isinstance(self.current_coding, dict):
                        self.current_coding['text'] = chunk
                    else:
                        # CodingResult zu Dict konvertieren
                        self.current_coding = {
                            'category': self.current_coding.category,
                            'subcategories': list(self.current_coding.subcategories),
                            'justification': self.current_coding.justification,
                            'confidence': self.current_coding.confidence,
                            'text_references': list(self.current_coding.text_references),
                            'uncertainties': list(self.current_coding.uncertainties) if self.current_coding.uncertainties else None,
                            'paraphrase': getattr(self.current_coding, 'paraphrase', ''),
                            'keywords': getattr(self.current_coding, 'keywords', ''),
                            'text': chunk,
                            'manual_coding': True,
                            'manual_multiple_coding': False,
                            'multiple_coding_instance': 1,
                            'total_coding_instances': 1,
                            'coding_date': datetime.now().isoformat()
                        }
            
            # Debug-Ausgabe
            if isinstance(self.current_coding, list):
                result_status = f"Mehrfachkodierung mit {len(self.current_coding)} Kodierungen erstellt"
            elif self.current_coding == "ABORT_ALL":
                result_status = "Kodierung abgebrochen"
            elif self.current_coding:
                result_status = "Einzelkodierung erstellt"
            else:
                result_status = "Keine Kodierung"
                
            print(f"ManualCoder Ergebnis: {result_status}")
            
            # Finale Bereinigung (nur Ressourcen, nicht Tkinter)
            self._cleanup_tkinter_resources()
            
            return self.current_coding
            
        except Exception as e:
            print(f"Fehler in code_chunk: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Sichere Bereinigung auch im Fehlerfall
            if hasattr(self, 'root'):
                self.root = None
            self._cleanup_tkinter_resources()
            return None

    def _cleanup_tkinter_safely(self):
        """
        KORRIGIERT: Sichere Bereinigung nur im Hauptthread
        """
        try:
            if hasattr(self, 'root') and self.root:
                # PrÜfe ob wir im Hauptthread sind
                if threading.current_thread() is threading.main_thread():
                    try:
                        # PrÜfe ob das Fenster noch existiert
                        if self.root.winfo_exists():
                            self.root.quit()
                            self.root.destroy()
                            print("Tkinter-Fenster erfolgreich geschlossen")
                        
                    except tk.TclError:
                        # Fenster wurde bereits zerstÖrt - das ist OK
                        print("Tkinter-Fenster war bereits geschlossen")
                        pass
                    except Exception as e:
                        # Andere Fehler - loggen aber nicht abbrechen
                        print(f"Info: Tkinter-Bereinigung: {str(e)}")
                else:
                    # Wir sind nicht im Hauptthread - nur Referenz entfernen
                    print("Tkinter-Bereinigung Übersprungen (nicht im Hauptthread)")
                    
                # Referenz immer entfernen
                self.root = None
                    
        except Exception as e:
            print(f"Info: Tkinter-Bereinigung abgeschlossen: {str(e)}")

    def _run_enhanced_tk_window(self, chunk: str):
        """
        KORRIGIERT: Bessere Thread-Behandlung
        """
        try:
            # Vorherige Fenster sicher schlieẞen (falls vorhanden)
            if hasattr(self, 'root') and self.root:
                try:
                    self.root.quit()
                    self.root.destroy()
                except:
                    pass
                self.root = None
            
            self.root = tk.Tk()
            self.root.title(f"Manueller Coder - {self.coder_id}")
            self.root.geometry("900x700")
            
            # KORRIGIERT: Protokoll fuer sicheres Schlieẞen
            self.root.protocol("WM_DELETE_WINDOW", self._safe_window_close)
        
            # GUI erstellen
            self._create_enhanced_gui(chunk)
            
            # Fenster in den Vordergrund bringen
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.attributes('-topmost', False)
            self.root.focus_force()
            
            # Plattformspezifische Anpassungen
            if platform.system() == "Darwin":  # macOS
                self.root.createcommand('tk::mac::RaiseWindow', self.root.lift)
            
            # KORRIGIERT: MainLoop mit sauberer Beendigung
            try:
                self.root.update()
                self.root.mainloop()
                
                # WICHTIG: Nach mainloop() ist das Fenster bereits geschlossen
                # Setze root auf None ohne weitere quit()/destroy() Aufrufe
                self.root = None
                print("Tkinter MainLoop beendet")
                
            except tk.TclError as tcl_error:
                if "application has been destroyed" in str(tcl_error):
                    # Das ist OK - Fenster wurde ordnungsgemÄẞ geschlossen
                    self.root = None
                    print("Tkinter-Anwendung ordnungsgemÄẞ beendet")
                else:
                    print(f"TclError: {tcl_error}")
                    self.root = None
            
        except Exception as e:
            print(f"Fehler beim Erstellen des Tkinter-Fensters: {str(e)}")
            self.current_coding = None
            self.root = None
            return

    def _create_enhanced_gui(self, chunk: str):
        """
        Erstellt die erweiterte GUI mit Mehrfachauswahl-Support
        """
        # Hauptframe
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Header mit Instruktionen
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Titel
        title_label = ttk.Label(
            header_frame,
            text=f"Manueller Kodierer - {self.coder_id}",
            font=('Arial', 12, 'bold')
        )
        title_label.pack()

        # Instruktionen erweitert um Nummern-Eingabe
        instructions_label = ttk.Label(
            header_frame,
            text="ℹ️ Tipp: Strg+Klick fuer Mehrfachauswahl - Shift+Klick fuer Bereichsauswahl\n" +
                "🌟 Neue Subkategorie: Nur Hauptkategorie-Nummer eingeben (z.B. '1' fuer erste Hauptkategorie)",
            font=('Arial', 9, 'italic'),
            foreground='blue'
        )
        instructions_label.pack(pady=(5, 0))
        
        
        # Fortschrittsinfo bei letztem Segment
        if self.is_last_segment:
            last_segment_label = ttk.Label(
                header_frame,
                text="🏁 LETZTES SEGMENT",
                font=('Arial', 10, 'bold'),
                foreground='red'
            )
            last_segment_label.pack(pady=(5, 0))

        # Textbereich
        text_frame = ttk.LabelFrame(main_frame, text="Textsegment")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.text_chunk = tk.Text(text_frame, height=10, wrap=tk.WORD, font=('Arial', 10))
        text_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_chunk.yview)
        self.text_chunk.config(yscrollcommand=text_scrollbar.set)
        
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_chunk.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text_chunk.insert(tk.END, chunk)

         # Kategorienbereich mit Nummern-Referenz
        category_frame = ttk.LabelFrame(main_frame, text="🌟 Nummerierte Kategorien (Mehrfachauswahl mÖglich)")
        category_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # NEU: Nummern-Referenz Label (kompakt, oben)
        reference_frame = ttk.Frame(category_frame)
        reference_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.number_reference_label = ttk.Label(
            reference_frame,
            text="Lade Nummern-Referenz...",
            font=('Arial', 8),
            foreground='darkblue',
            background='lightgray',
            relief='sunken'
        )
        self.number_reference_label.pack(fill=tk.X, padx=2, pady=2)
        
        
        # Verwende die neue MultiSelectListbox aus QCA_Utils
        from ..QCA_Utils import MultiSelectListbox
        self.category_listbox = MultiSelectListbox(category_frame, font=('Arial', 10))
        
        cat_scrollbar = ttk.Scrollbar(category_frame, orient=tk.VERTICAL, command=self.category_listbox.yview)
        self.category_listbox.config(yscrollcommand=cat_scrollbar.set)
        
        cat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.category_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        
        # Auswahlinfo
        selection_info_frame = ttk.Frame(main_frame)
        selection_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.selection_info_label = ttk.Label(
            selection_info_frame,
            text="Keine Auswahl",
            font=('Arial', 9),
            foreground='gray'
        )
        self.selection_info_label.pack()
        
        # Binding fuer Auswahl-Updates
        self.category_listbox.bind('<<ListboxSelect>>', self._on_selection_change)
        
        # Button-Frame mit klareren Beschriftungen
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        

        # Hauptbuttons mit klareren Texten
        if self.is_last_segment:
            ttk.Button(
                button_frame,
                text="Kodieren & Kodierung beenden",  # Klarerer Text
                command=self._safe_finish_coding_enhanced
            ).pack(side=tk.LEFT, padx=(0, 5))
        else:
            ttk.Button(
                button_frame,
                text="Kodieren & Weiter",  # Klarerer Text
                command=self._safe_code_selection_enhanced
            ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            button_frame,
            text="Neue Hauptkategorie",
            command=self._safe_new_main_category
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="Neue Subkategorie (per Nr.)",  # Aktualisierter Text
            command=self._safe_new_sub_category_enhanced  # Neue Methode
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="Segment ueberspringen",  # Klarerer Text
            command=self._safe_skip_chunk
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="Kodierung komplett beenden",  # Klarerer Text fuer Abbruch
            command=self._safe_abort_coding
        ).pack(side=tk.RIGHT)
        
        # Kategorien laden
        self.update_category_list_enhanced()

    def _safe_window_close(self):
        """
        KORRIGIERT: Sichere Behandlung des Fenster-Schlieẞens ohne Threading-Warnungen
        """
        try:
            # Bei X-Button-Klick -> Kodierung ueberspringen (nicht abbrechen)
            if not self._is_processing:
                if messagebox.askyesno(
                    "Fenster schlieẞen",
                    "MÖchten Sie dieses Segment ueberspringen und zum nÄchsten wechseln?\n\n"
                    "WÄhlen Sie 'Nein' um zum Kodieren zurÜckzukehren."
                ):
                    # Segment ueberspringen
                    self.current_coding = {
                        'category': "Nicht kodiert",
                        'subcategories': [],
                        'justification': "Segment Übersprungen (Fenster geschlossen)",
                        'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                        'text_references': [],
                        'uncertainties': None,
                        'paraphrase': "",
                        'keywords': "",
                        'manual_coding': True,
                        'manual_multiple_coding': False,
                        'multiple_coding_instance': 1,
                        'total_coding_instances': 1,
                        'coding_date': datetime.now().isoformat()
                    }
                    
                    # Fenster sicher schlieẞen
                    if self.root and self.root.winfo_exists():
                        self.root.quit()
                    
        except Exception as e:
            print(f"Info: Fenster-Schlieẞung: {str(e)}")
            # Im Fehlerfall: Kodierung ueberspringen
            self.current_coding = None
            try:
                if self.root:
                    self.root.quit()
            except:
                pass


