"""
Manual Review Components

Interactive GUI for manual review and discrepancy resolution in QCA-AID.

Includes:
- ManualReviewGUI: Simple category-specific review interface
- ManualReviewComponent: Advanced review with discrepancy handling and consensus building
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


class ManualReviewGUI:
    """
    KORRIGIERT: GUI für kategorie-spezifisches manuelles Review
    """
    
    def perform_category_specific_review(self, segments_needing_review: List[Dict]) -> List[Dict]:
        """
        Führt manuelles Review für kategorie-spezifische Segmente durch
        
        Jedes Segment repräsentiert nur eine Hauptkategorie, daher
        sind die Entscheidungen fokussierter und methodisch korrekter
        """
        review_decisions = []
        
        for i, segment in enumerate(segments_needing_review, 1):
            print(f"\n[REVIEW] Review {i}/{len(segments_needing_review)}: {segment['segment_id']}")
            print(f"   Kategorie: {segment['target_category']}")
            
            if segment['is_multiple_coding']:
                instance_info = segment['instance_info']
                print(f"   Teil {instance_info['instance_number']}/{instance_info['total_instances']} von {segment['original_segment_id']}")
                print(f"   Alle Kategorien dieses Segments: {', '.join(instance_info['all_categories'])}")
            
            # Zeige alle Kodierungen für diese spezifische Kategorie
            codings = segment['codings']
            
            print(f"\n   📋 {len(codings)} Kodierungen für Kategorie '{segment['target_category']}':")
            for j, coding in enumerate(codings, 1):
                coder = coding.get('coder_id', 'Unbekannt')
                subcats = coding.get('subcategories', [])
                confidence = coding.get('confidence', {})
                
                print(f"      {j}. {coder}: {segment['target_category']}")
                if subcats:
                    print(f"         Subkategorien: {', '.join(subcats)}")
                if isinstance(confidence, dict):
                    conf_val = confidence.get('total', 0.0)
                    print(f"         Konfidenz: {conf_val:.2f}")
            
            # Lade Textinhalt (von erster Kodierung)
            text_content = codings[0].get('text', 'Kein Text verfügbar')
            
            # Zeige GUI-Dialog für diese eine Kategorie
            decision = self._show_category_review_dialog(segment, text_content)
            
            if decision:
                review_decisions.append(decision)
        
        return review_decisions
    
    def _show_category_review_dialog(self, segment: Dict, text_content: str) -> Optional[Dict]:
        """
        Zeigt Review-Dialog für ein kategorie-spezifisches Segment
        
        Viel fokussierter als vorher, da nur eine Kategorie behandelt wird
        """
        # HIER würde die GUI-Implementierung kommen
        # Für Demo-Zwecke: Automatische Entscheidung basierend auf höchster Konfidenz
        
        codings = segment['codings']
        best_coding = max(codings, key=lambda x: self._extract_confidence_value(x))
        
        decision = best_coding.copy()
        decision.update({
            'segment_id': segment['segment_id'],
            'manual_review': True,
            'review_date': datetime.now().isoformat(),
            'review_justification': f"Automatisch gewählt: Höchste Konfidenz für Kategorie {segment['target_category']}",
            'original_segment_id': segment['original_segment_id'],
            'is_multiple_coding_instance': segment['is_multiple_coding'],
            'instance_info': segment['instance_info']
        })
        
        return decision
    
    def _extract_confidence_value(self, coding: Dict) -> float:
        """Extrahiert Konfidenzwert"""
        confidence = coding.get('confidence', {})
        if isinstance(confidence, dict):
            return confidence.get('total', 0.0)
        elif isinstance(confidence, (int, float)):
            return float(confidence)
        return 0.0


class ManualReviewComponent:
    """
    Komponente für die manuelle Überprüfung und Entscheidung bei Kodierungsunstimmigkeiten.
    Zeigt dem Benutzer Textstellen mit abweichenden Kodierungen und lässt ihn die finale Entscheidung treffen.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialisiert die Manual Review Komponente.
        
        Args:
            output_dir (str): Verzeichnis für Export-Dokumente
        """
        self.output_dir = output_dir
        self.root = None
        self.review_results = []
        self.current_segment = None
        self.current_codings = None
        self.current_index = 0
        self.total_segments = 0
        self._is_processing = False
        
        # Import tkinter innerhalb der Methode, um Abhängigkeiten zu reduzieren
        self.tk = None
        self.ttk = None
        
    async def review_discrepancies(self, segment_codings: dict) -> list:
        """
        Führt einen manuellen Review-Prozess für Segmente mit abweichenden Kodierungen durch.
        
        Args:
            segment_codings: Dictionary mit Segment-ID als Schlüssel und Liste von Kodierungen als Werte
            
        Returns:
            list: Liste der finalen Kodierungsentscheidungen
        """
        try:
            # Importiere tkinter bei Bedarf
            import tkinter as tk
            from tkinter import ttk
            self.tk = tk
            self.ttk = ttk
            
            print("\n=== Manuelle Überprüfung von Kodierungsunstimmigkeiten ===")
            
            # Identifiziere Segmente mit abweichenden Kodierungen
            discrepant_segments = self._identify_discrepancies(segment_codings)
            
            if not discrepant_segments:
                print("Keine Unstimmigkeiten gefunden. Manueller Review nicht erforderlich.")
                return []
                
            self.total_segments = len(discrepant_segments)
            print(f"\nGefunden: {self.total_segments} Segmente mit Kodierungsabweichungen")
            
            # Starte das Tkinter-Fenster für den manuellen Review
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_review_gui, discrepant_segments)
            
            print(f"\nManueller Review abgeschlossen: {len(self.review_results)} Entscheidungen getroffen")
            
            return self.review_results
            
        except Exception as e:
            print(f"Fehler beim manuellen Review: {str(e)}")
            import traceback
            traceback.print_exc()
            if self.root:
                try:
                    self.root.quit()
                    self.root.destroy()
                except:
                    pass
            return []
    
    def _identify_discrepancies(self, segment_codings: dict) -> list:
        """
        Identifiziert Segmente, bei denen verschiedene Kodierer zu unterschiedlichen Ergebnissen kommen.
        
        Args:
            segment_codings: Dictionary mit Segment-ID als Schlüssel und Liste von Kodierungen als Werte
            
        Returns:
            list: Liste von Tuples (segment_id, text, codings) mit Unstimmigkeiten
        """
        discrepancies = []
        
        for segment_id, codings in segment_codings.items():
            # Ignoriere Segmente mit nur einer Kodierung
            if len(codings) <= 1:
                continue
                
            # Prüfe auf Unstimmigkeiten in Hauptkategorien
            categories = set(coding.get('category', '') for coding in codings)
            
            # Prüfe auf menschliche Kodierer
            has_human_coder = any('human' in coding.get('coder_id', '') for coding in codings)
            
            # Wenn mehr als eine Kategorie ODER ein menschlicher Kodierer beteiligt
            if len(categories) > 1 or has_human_coder:
                # Hole den Text des Segments
                text = codings[0].get('text', '')
                if not text:
                    # Alternative Textquelle, falls 'text' nicht direkt verfügbar
                    text = codings[0].get('text_references', [''])[0] if codings[0].get('text_references') else ''
                
                discrepancies.append((segment_id, text, codings))
                
        print(f"Unstimmigkeiten identifiziert: {len(discrepancies)}/{len(segment_codings)} Segmente")
        return discrepancies

    async def review_discrepancies_direct(self, segment_codings: dict, skip_discrepancy_check: bool = False) -> list:
        """
        FIX: Neue Methode für ManualReviewComponent, die optional die Unstimmigkeits-Prüfung überspringt
        
        Args:
            segment_codings: Dictionary mit Segment-ID als Schlüssel und Liste von Kodierungen als Werte
            skip_discrepancy_check: Wenn True, behandle alle übergebenen Segmente als unstimmig
            
        Returns:
            list: Liste der finalen Kodierungsentscheidungen
        """
        try:
            # Importiere tkinter bei Bedarf
            import tkinter as tk
            from tkinter import ttk, messagebox
            self.tk = tk
            self.ttk = ttk
            self.messagebox = messagebox
            
            print("\n=== Manuelle Überprüfung von Kodierungsunstimmigkeiten ===")
            
            if skip_discrepancy_check:
                # FIX: Überspringe eigene Unstimmigkeits-Prüfung und verwende alle übergebenen Segmente
                print(f"🎯 Verwende alle {len(segment_codings)} übergebenen Segmente für Review (Prüfung übersprungen)")
                
                discrepant_segments = []
                for segment_id, codings in segment_codings.items():
                    if len(codings) > 1:  # Nur Segmente mit mehreren Kodierungen
                        # Hole den Text des Segments
                        text = codings[0].get('text', '')
                        if not text:
                            text = codings[0].get('text_references', [''])[0] if codings[0].get('text_references') else ''
                        
                        discrepant_segments.append((segment_id, text, codings))
                        
                print(f"📋 Direkte Übernahme: {len(discrepant_segments)} Segmente für Review")
            else:
                # Normale Unstimmigkeits-Identifikation
                discrepant_segments = self._identify_discrepancies(segment_codings)
            
            if not discrepant_segments:
                print("Keine Unstimmigkeiten gefunden. Manueller Review nicht erforderlich.")
                return []
                
            self.total_segments = len(discrepant_segments)
            print(f"\nGefunden: {self.total_segments} Segmente mit Kodierungsabweichungen")
            
            # Setze Review-Status zurück
            self.review_results = []
            self._review_completed = False
            
            # FIX: Führe interaktive GUI aus
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_interactive_review_gui, discrepant_segments)
            
            print(f"\nManueller Review abgeschlossen: {len(self.review_results)} Entscheidungen getroffen")
            
            return self.review_results
            
        except Exception as e:
            print(f"Fehler beim manuellen Review: {str(e)}")
            import traceback
            traceback.print_exc()
            if self.root:
                try:
                    self.root.quit()
                    self.root.destroy()
                except:
                    pass
            return []

    def _run_interactive_review_gui(self, discrepant_segments: list):
        """
        FIX: Interaktive GUI die echte Benutzereingaben erfasst
        """
        try:
            print(f"🎮 Starte interaktive Review-GUI für {len(discrepant_segments)} Segmente...")
            
            # Cleanup vorheriger Instanzen
            if self.root is not None:
                try:
                    self.root.quit()
                    self.root.destroy()
                except:
                    pass
                    
            self.root = self.tk.Tk()
            self.root.title("QCA-AID Manueller Review")
            self.root.geometry("1000x800")
            
            # FIX: Fenster in den Vordergrund bringen und fokussieren
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.update()
            self.root.attributes('-topmost', False)
            self.root.focus_force()
            
            # Plattformspezifische Anpassungen für macOS
            import platform
            if platform.system() == "Darwin":  # macOS
                try:
                    self.root.createcommand('tk::mac::RaiseWindow', self.root.lift)
                except:
                    pass
            
            # Protokoll für sauberes Schließen
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing_safe)
            
            # FIX: Erstelle erweiterte GUI mit Eingabefeldern
            self._create_interactive_review_gui(discrepant_segments)
            
            # FIX: Starte GUI erst nach dem Aufbau
            self.root.update()  # Stelle sicher, dass GUI vollständig gerendert ist
            print("🖥️ GUI-Fenster erstellt und sichtbar")
            
            # MainLoop mit sauberer Beendigung
            try:
                self.root.mainloop()
                print("📋 GUI MainLoop beendet")
            except Exception as e:
                print(f"Info: MainLoop beendet: {str(e)}")
            finally:
                # WICHTIG: Markiere Review als abgeschlossen
                self._review_completed = True
                    
        except Exception as e:
            print(f"Fehler in Interactive Review-GUI: {str(e)}")
            import traceback
            traceback.print_exc()
            self._review_completed = True
            self.root = None

    def _create_interactive_review_gui(self, discrepant_segments: list):
        """
        FIX: Erstellt interaktive GUI mit Eingabefeldern für manuelle Kodierung
        """
        print("🔧 Erstelle interaktive GUI-Komponenten...")
        
        # Hauptframe
        main_frame = self.ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=self.tk.BOTH, expand=True)
        
        # Aktuelle Segment-Variablen
        self.current_segment_index = 0
        self.current_segment_var = self.tk.StringVar()
        self.current_text_var = self.tk.StringVar()
        
        # Eingabefelder-Variablen
        self.category_var = self.tk.StringVar()
        self.subcategories_var = self.tk.StringVar()
        self.justification_var = self.tk.StringVar()
        
        # Titel
        title_frame = self.ttk.Frame(main_frame)
        title_frame.pack(fill=self.tk.X, pady=(0, 10))
        
        title_label = self.ttk.Label(title_frame, 
                                    text="🎯 QCA-AID Manueller Review", 
                                    font=('Arial', 16, 'bold'))
        title_label.pack()
        
        # Fortschrittsanzeige
        progress_frame = self.ttk.Frame(main_frame)
        progress_frame.pack(fill=self.tk.X, pady=(0, 10))
        
        self.ttk.Label(progress_frame, text="Fortschritt:", font=('Arial', 10, 'bold')).pack(side=self.tk.LEFT)
        self.progress_label = self.ttk.Label(progress_frame, textvariable=self.current_segment_var, font=('Arial', 10))
        self.progress_label.pack(side=self.tk.LEFT, padx=(10, 0))
        
        # Text-Anzeige
        text_frame = self.ttk.LabelFrame(main_frame, text="📄 Textsegment")
        text_frame.pack(fill=self.tk.BOTH, expand=True, pady=(0, 10))
        
        # Text mit Scrollbar
        text_container = self.ttk.Frame(text_frame)
        text_container.pack(fill=self.tk.BOTH, expand=True, padx=5, pady=5)
        
        self.text_display = self.tk.Text(text_container, height=6, wrap=self.tk.WORD, 
                                        state=self.tk.DISABLED, font=('Arial', 11))
        text_scrollbar = self.ttk.Scrollbar(text_container, orient=self.tk.VERTICAL, 
                                        command=self.text_display.yview)
        self.text_display.configure(yscrollcommand=text_scrollbar.set)
        self.text_display.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        text_scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        
        # FIX: Konkurrierende Kodierungen-Bereich
        codings_frame = self.ttk.LabelFrame(main_frame, text="⚡ Konkurrierende Kodierungen")
        codings_frame.pack(fill=self.tk.X, pady=(0, 10))
        
        # Scrollbarer Bereich für Kodierungen
        codings_container = self.ttk.Frame(codings_frame)
        codings_container.pack(fill=self.tk.BOTH, expand=True, padx=5, pady=5)
        
        self.codings_canvas = self.tk.Canvas(codings_container, height=120)
        codings_scrollbar = self.ttk.Scrollbar(codings_container, orient=self.tk.VERTICAL, 
                                            command=self.codings_canvas.yview)
        self.codings_scrollable = self.ttk.Frame(self.codings_canvas)
        
        self.codings_scrollable.bind(
            "<Configure>",
            lambda e: self.codings_canvas.configure(scrollregion=self.codings_canvas.bbox("all"))
        )
        
        self.codings_canvas.create_window((0, 0), window=self.codings_scrollable, anchor="nw")
        self.codings_canvas.configure(yscrollcommand=codings_scrollbar.set)
        
        self.codings_canvas.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        codings_scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        
        # FIX: Variable für ausgewählte Kodierung
        self.selected_coding_var = self.tk.StringVar()
        
        # Eingabebereich
        input_frame = self.ttk.LabelFrame(main_frame, text="✏️ Manuelle Kodierung")
        input_frame.pack(fill=self.tk.X, pady=(0, 10))
        
        # Kategorie
        self.ttk.Label(input_frame, text="Hauptkategorie:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=self.tk.W, padx=5, pady=5)
        self.category_entry = self.ttk.Entry(input_frame, textvariable=self.category_var, width=60, font=('Arial', 10))
        self.category_entry.grid(row=0, column=1, sticky=self.tk.EW, padx=5, pady=5)
        
        # Subkategorien
        self.ttk.Label(input_frame, text="Subkategorien:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=self.tk.W, padx=5, pady=5)
        self.subcats_entry = self.ttk.Entry(input_frame, textvariable=self.subcategories_var, width=60, font=('Arial', 10))
        self.subcats_entry.grid(row=1, column=1, sticky=self.tk.EW, padx=5, pady=5)
        
        # Hilfetext für Subkategorien
        help_label = self.ttk.Label(input_frame, text="(kommagetrennt, z.B.: Politik, Finanzierung, Verwaltung)", 
                                font=('Arial', 9), foreground='gray')
        help_label.grid(row=2, column=1, sticky=self.tk.W, padx=5, pady=(0, 5))
        
        # Begründung
        self.ttk.Label(input_frame, text="Begründung:", font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky=self.tk.W, padx=5, pady=5)
        self.justification_entry = self.ttk.Entry(input_frame, textvariable=self.justification_var, width=60, font=('Arial', 10))
        self.justification_entry.grid(row=3, column=1, sticky=self.tk.EW, padx=5, pady=5)
        
        input_frame.columnconfigure(1, weight=1)
        
        # Button-Bereich
        button_frame = self.ttk.Frame(main_frame)
        button_frame.pack(fill=self.tk.X, pady=(10, 0))
        
        # Navigation Buttons
        nav_frame = self.ttk.Frame(button_frame)
        nav_frame.pack(side=self.tk.LEFT)
        
        self.prev_button = self.ttk.Button(nav_frame, text="← Vorheriges", command=self._previous_segment)
        self.prev_button.pack(side=self.tk.LEFT, padx=(0, 5))
        
        self.next_button = self.ttk.Button(nav_frame, text="Nächstes →", command=self._next_segment)
        self.next_button.pack(side=self.tk.LEFT, padx=5)
        
        # Action Buttons
        action_frame = self.ttk.Frame(button_frame)
        action_frame.pack(side=self.tk.RIGHT)
        
        save_button = self.ttk.Button(action_frame, text="💾 Entscheidung speichern", 
                                    command=self._save_current_decision)
        save_button.pack(side=self.tk.RIGHT, padx=5)
        
        finish_button = self.ttk.Button(action_frame, text="✅ Review abschließen", 
                                    command=self._finish_review)
        finish_button.pack(side=self.tk.RIGHT, padx=5)
        
        # Speichere Segmente und initialisiere
        self.discrepant_segments = discrepant_segments
        self._load_current_segment()
        
        print(f"✅ GUI erfolgreich erstellt mit {len(discrepant_segments)} Segmenten")

    def _load_current_segment(self):
        """
        FIX: Lädt das aktuelle Segment in die GUI und zeigt alle konkurrierenden Kodierungen
        """
        if 0 <= self.current_segment_index < len(self.discrepant_segments):
            segment_id, text, codings = self.discrepant_segments[self.current_segment_index]
            
            # Update Fortschrittsanzeige
            self.current_segment_var.set(f"Segment {self.current_segment_index + 1}/{len(self.discrepant_segments)}: {segment_id}")
            
            # Update Text
            self.text_display.config(state=self.tk.NORMAL)
            self.text_display.delete(1.0, self.tk.END)
            self.text_display.insert(1.0, text)
            self.text_display.config(state=self.tk.DISABLED)
            
            # FIX: Lade konkurrierende Kodierungen
            self._load_competing_codings(codings)
            
            # Vorbefüllung mit der besten existierenden Kodierung
            if codings:
                best_coding = max(codings, key=lambda x: self._extract_confidence_value(x))
                self._apply_coding_to_fields(best_coding)
            
            # Update Button-Status
            self.prev_button.config(state=self.tk.NORMAL if self.current_segment_index > 0 else self.tk.DISABLED)
            self.next_button.config(state=self.tk.NORMAL if self.current_segment_index < len(self.discrepant_segments) - 1 else self.tk.DISABLED)

    def _load_competing_codings(self, codings):
        """
        FIX: Lädt und zeigt alle konkurrierenden Kodierungen als auswählbare Optionen
        """
        # Lösche vorherige Kodierungen
        for widget in self.codings_scrollable.winfo_children():
            widget.destroy()
        
        # FIX: Erstelle Radiobuttons für jede Kodierung + "Nicht kodiert" Option
        for i, coding in enumerate(codings):
            coding_frame = self.ttk.Frame(self.codings_scrollable)
            coding_frame.pack(fill=self.tk.X, padx=5, pady=2)
            
            # Kodierer-Info
            coder_id = coding.get('coder_id', 'Unbekannt')
            category = coding.get('category', 'Keine Kategorie')
            subcats = coding.get('subcategories', [])
            confidence = self._extract_confidence_value(coding)
            
            # Radiobutton für Auswahl
            radio_value = f"coding_{i}"
            radio = self.ttk.Radiobutton(
                coding_frame, 
                text="",
                variable=self.selected_coding_var,
                value=radio_value,
                command=lambda idx=i: self._on_coding_selected(idx)
            )
            radio.pack(side=self.tk.LEFT, padx=(0, 5))
            
            # Kodierungs-Details
            details_text = f"{coder_id}: {category}"
            if subcats:
                details_text += f" → {', '.join(subcats)}"
            details_text += f" (Konfidenz: {confidence:.2f})"
            
            details_label = self.ttk.Label(coding_frame, text=details_text, font=('Arial', 10))
            details_label.pack(side=self.tk.LEFT, padx=5)
            
            # Speichere Kodierung für späteren Zugriff
            setattr(self, f'_coding_{i}', coding)
        
        # FIX: Füge "Nicht kodiert" Option hinzu
        not_coded_frame = self.ttk.Frame(self.codings_scrollable)
        not_coded_frame.pack(fill=self.tk.X, padx=5, pady=2)
        
        # Separator
        separator = self.ttk.Separator(self.codings_scrollable, orient='horizontal')
        separator.pack(fill=self.tk.X, padx=5, pady=5)
        
        not_coded_frame2 = self.ttk.Frame(self.codings_scrollable)
        not_coded_frame2.pack(fill=self.tk.X, padx=5, pady=2)
        
        # "Nicht kodiert" Radiobutton
        not_coded_radio = self.ttk.Radiobutton(
            not_coded_frame2, 
            text="",
            variable=self.selected_coding_var,
            value="not_coded",
            command=lambda: self._on_not_coded_selected()
        )
        not_coded_radio.pack(side=self.tk.LEFT, padx=(0, 5))
        
        # "Nicht kodiert" Label mit Erklärung
        not_coded_label = self.ttk.Label(
            not_coded_frame2, 
            text="🚫 Nicht kodiert (Segment ist nicht relevant für die Forschungsfrage)",
            font=('Arial', 10), 
            foreground='red'
        )
        not_coded_label.pack(side=self.tk.LEFT, padx=5)
        
        # Wähle automatisch die beste Kodierung aus
        if codings:
            best_index = max(range(len(codings)), key=lambda i: self._extract_confidence_value(codings[i]))
            self.selected_coding_var.set(f"coding_{best_index}")
            self._on_coding_selected(best_index)

    def _on_not_coded_selected(self):
        """
        FIX: Wird aufgerufen wenn "Nicht kodiert" ausgewählt wird
        """
        if 0 <= self.current_segment_index < len(self.discrepant_segments):
            segment_id, text, codings = self.discrepant_segments[self.current_segment_index]
            
            # Setze Felder für "Nicht kodiert"
            self.category_var.set("Nicht kodiert")
            self.subcategories_var.set("")
            self.justification_var.set("Segment als nicht relevant für die Forschungsfrage eingestuft")
            
            print(f"🚫 'Nicht kodiert' ausgewählt für {segment_id}")

    def _on_coding_selected(self, coding_index):
        """
        FIX: Wird aufgerufen wenn eine konkurrierende Kodierung ausgewählt wird
        """
        if 0 <= self.current_segment_index < len(self.discrepant_segments):
            segment_id, text, codings = self.discrepant_segments[self.current_segment_index]
            
            if 0 <= coding_index < len(codings):
                selected_coding = codings[coding_index]
                self._apply_coding_to_fields(selected_coding)
                
                coder_id = selected_coding.get('coder_id', 'Unbekannt')
                print(f"📝 Kodierung von {coder_id} ausgewählt für {segment_id}")

    def _apply_coding_to_fields(self, coding):
        """
        FIX: Überträgt eine Kodierung in die Eingabefelder
        """
        self.category_var.set(coding.get('category', ''))
        
        subcats = coding.get('subcategories', [])
        self.subcategories_var.set(', '.join(subcats) if subcats else '')
        
        self.justification_var.set(coding.get('justification', ''))

    def _save_current_decision(self):
        """
        FIX: Speichert die aktuelle manuelle Entscheidung (inkl. "Nicht kodiert")
        """
        if 0 <= self.current_segment_index < len(self.discrepant_segments):
            segment_id, text, codings = self.discrepant_segments[self.current_segment_index]
            
            # Validierung
            category = self.category_var.get().strip()
            if not category:
                self.messagebox.showerror("Fehler", "Bitte geben Sie eine Hauptkategorie ein oder wählen Sie 'Nicht kodiert'!")
                return
            
            # FIX: Spezielle Behandlung für "Nicht kodiert"
            if category == "Nicht kodiert":
                decision = {
                    'segment_id': segment_id,
                    'category': 'Nicht kodiert',
                    'subcategories': [],
                    'justification': self.justification_var.get().strip() or "Segment als nicht relevant eingestuft",
                    'text': text,
                    'manual_review': True,
                    'review_date': datetime.now().isoformat(),
                    'coder_id': 'manual_review',
                    'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                    'is_coded': False,  # FIX: Markiere als nicht kodiert
                    'relevance': False  # FIX: Markiere als nicht relevant
                }
            else:
                # Normale Kodierung
                decision = {
                    'segment_id': segment_id,
                    'category': category,
                    'subcategories': [s.strip() for s in self.subcategories_var.get().split(',') if s.strip()],
                    'justification': self.justification_var.get().strip(),
                    'text': text,
                    'manual_review': True,  # FIX: Wichtig für Review_Typ Export
                    'review_date': datetime.now().isoformat(),
                    'coder_id': 'manual_review',
                    'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0},
                    'is_coded': True,   # FIX: Markiere als kodiert
                    'relevance': True   # FIX: Markiere als relevant
                }
            
            # Entferne vorherige Entscheidung für dieses Segment falls vorhanden
            self.review_results = [r for r in self.review_results if r.get('segment_id') != segment_id]
            
            # Füge neue Entscheidung hinzu
            self.review_results.append(decision)
            
            print(f"✅ Entscheidung gespeichert für {segment_id}: {decision['category']} → {decision.get('subcategories', [])}")
            
            # Bestätigungs-Feedback
            if category == "Nicht kodiert":
                self.messagebox.showinfo("Gespeichert", f"Segment {segment_id} wurde als 'Nicht kodiert' markiert!")
            else:
                self.messagebox.showinfo("Gespeichert", f"Entscheidung für {segment_id} wurde gespeichert!")
            
            # Automatisch zum nächsten Segment
            if self.current_segment_index < len(self.discrepant_segments) - 1:
                self.current_segment_index += 1
                self._load_current_segment()
            else:
                # Letztes Segment erreicht
                self.messagebox.showinfo("Fertig", "Alle Segmente bearbeitet! Sie können das Review jetzt abschließen.")

    def _next_segment(self):
        """Navigiert zum nächsten Segment"""
        if self.current_segment_index < len(self.discrepant_segments) - 1:
            self.current_segment_index += 1
            self._load_current_segment()

    def _previous_segment(self):
        """Navigiert zum vorherigen Segment"""
        if self.current_segment_index > 0:
            self.current_segment_index -= 1
            self._load_current_segment()

    def _finish_review(self):
        """Beendet den Review-Prozess"""
        # Prüfe ob alle Segmente bearbeitet wurden
        unprocessed = len(self.discrepant_segments) - len(self.review_results)
        if unprocessed > 0:
            if not self.messagebox.askyesno("Unvollständig", 
                                        f"Es sind noch {unprocessed} Segmente unbearbeitet. "
                                        f"Trotzdem beenden?"):
                return
        
        print(f"🏁 Review wird beendet mit {len(self.review_results)} von {len(self.discrepant_segments)} Entscheidungen")
        if self.root:
            self.root.quit()

    def _on_closing_safe(self):
        """
        FIX: Fehlende Methode für sauberes Schließen des Review-Fensters
        """
        try:
            # Bestätige Schließen
            if self.messagebox.askokcancel(
                "Review beenden", 
                f"Review beenden?\n{len(self.review_results)} von {len(self.discrepant_segments)} Entscheidungen wurden getroffen."
            ):
                print(f"Review wird beendet mit {len(self.review_results)} Entscheidungen")
                if self.root and self.root.winfo_exists():
                    self.root.quit()
        except Exception as e:
            print(f"Info: Fenster-Schließung: {str(e)}")
            try:
                if self.root:
                    self.root.quit()
            except:
                pass

    def _extract_confidence_value(self, coding: dict) -> float:
        """Extrahiert Konfidenzwert"""
        confidence = coding.get('confidence', {})
        if isinstance(confidence, dict):
            return confidence.get('total', 0.0)
        elif isinstance(confidence, (int, float)):
            return float(confidence)
        return 0.0

    def _update_display(self, text_widget, codings_frame, discrepant_segments, justification_text, progress_var):
        """
        Aktualisiert die Anzeige für das aktuelle Segment.
        """
        # Aktualisiere Fortschrittsanzeige
        progress_var.set(f"Segment {self.current_index + 1}/{self.total_segments}")
        
        # Hole aktuelles Segment und Kodierungen
        segment_id, text, codings = discrepant_segments[self.current_index]
        self.current_segment = segment_id
        self.current_codings = codings
        
        # Setze Text
        text_widget.config(state=self.tk.NORMAL)
        text_widget.delete(1.0, self.tk.END)
        text_widget.insert(self.tk.END, text)
        text_widget.config(state=self.tk.DISABLED)
        
        # Begründungsfeld leeren
        justification_text.delete(1.0, self.tk.END)
        
        # Lösche alte Kodierungsoptionen
        for widget in codings_frame.winfo_children():
            widget.destroy()
            
        # Anzeige-Variable für die ausgewählte Kodierung
        selection_var = self.tk.StringVar()
        
        # Erstelle Radiobuttons für jede Kodierung
        for i, coding in enumerate(codings):
            coder_id = coding.get('coder_id', 'Unbekannt')
            category = coding.get('category', 'Keine Kategorie')
            subcategories = coding.get('subcategories', [])
            if isinstance(subcategories, tuple):
                subcategories = list(subcategories)
            confidence = 0.0
            
            # Extrahiere Konfidenzwert
            if isinstance(coding.get('confidence'), dict):
                confidence = coding['confidence'].get('total', 0.0)
            elif isinstance(coding.get('confidence'), (int, float)):
                confidence = float(coding['confidence'])
                
            # Formatiere die Subkategorien
            subcats_text = ', '.join(subcategories) if subcategories else 'Keine'
            
            # Erstelle Label-Text
            is_human = 'human' in coder_id
            coder_prefix = "[Mensch]" if is_human else "[Auto]"
            radio_text = f"{coder_prefix} {coder_id}: {category} ({confidence:.2f})\nSubkategorien: {subcats_text}"
            
            # Radiobutton mit Rahmen für bessere Sichtbarkeit
            coding_frame = self.ttk.Frame(codings_frame, relief=self.tk.GROOVE, borderwidth=2)
            coding_frame.pack(padx=5, pady=5, fill=self.tk.X)
            
            radio = self.ttk.Radiobutton(
                coding_frame,
                text=radio_text,
                variable=selection_var,
                value=str(i),
                command=lambda idx=i, j_text=justification_text: self._select_coding(idx, j_text)
            )
            radio.pack(padx=5, pady=5, anchor=self.tk.W)
            
            # Begründung anzeigen wenn vorhanden
            justification = coding.get('justification', '')
            if justification:
                just_label = self.ttk.Label(
                    coding_frame, 
                    text=f"Begründung: {justification[:150]}..." if len(justification) > 150 else f"Begründung: {justification}",
                    wraplength=500
                )
                just_label.pack(padx=5, pady=5, anchor=self.tk.W)
        
        # Eigene Kodierung als Option
        custom_frame = self.ttk.Frame(codings_frame, relief=self.tk.GROOVE, borderwidth=2)
        custom_frame.pack(padx=5, pady=5, fill=self.tk.X)
        
        custom_radio = self.ttk.Radiobutton(
            custom_frame,
            text="Eigene Entscheidung eingeben",
            variable=selection_var,
            value="custom",
            command=lambda: self._create_custom_coding(justification_text)
        )
        custom_radio.pack(padx=5, pady=5, anchor=self.tk.W)
        
        # Standardmäßig menschliche Kodierung auswählen, falls vorhanden
        for i, coding in enumerate(codings):
            if 'human' in coding.get('coder_id', ''):
                selection_var.set(str(i))
                self._select_coding(i, justification_text)
                break
    
    def _select_coding(self, coding_index, justification_text):
        """
        Ausgewählte Kodierung für das aktuelle Segment speichern.
        """
        self.selected_coding_index = coding_index
        
        # Hole die ausgewählte Kodierung
        selected_coding = self.current_codings[coding_index]
        
        # Fülle Begründung mit Vorschlag
        existing_just = selected_coding.get('justification', '')
        if existing_just:
            justification_text.delete(1.0, self.tk.END)
            justification_text.insert(self.tk.END, f"Übernommen von {selected_coding.get('coder_id', 'Kodierer')}: {existing_just}")
    
    def _create_custom_coding(self, justification_text):
        """
        Erstellt ein benutzerdefiniertes Kodierungsfenster.
        """
        custom_window = self.tk.Toplevel(self.root)
        custom_window.title("Eigene Kodierung")
        custom_window.geometry("600x500")
        
        input_frame = self.ttk.Frame(custom_window)
        input_frame.pack(padx=10, pady=10, fill=self.tk.BOTH, expand=True)
        
        # Hauptkategorie
        self.ttk.Label(input_frame, text="Hauptkategorie:").grid(row=0, column=0, padx=5, pady=5, sticky=self.tk.W)
        category_entry = self.ttk.Entry(input_frame, width=30)
        category_entry.grid(row=0, column=1, padx=5, pady=5, sticky=self.tk.W+self.tk.E)
        
        # Subkategorien
        self.ttk.Label(input_frame, text="Subkategorien (mit Komma getrennt):").grid(row=1, column=0, padx=5, pady=5, sticky=self.tk.W)
        subcats_entry = self.ttk.Entry(input_frame, width=30)
        subcats_entry.grid(row=1, column=1, padx=5, pady=5, sticky=self.tk.W+self.tk.E)
        
        # Begründung
        self.ttk.Label(input_frame, text="Begründung:").grid(row=2, column=0, padx=5, pady=5, sticky=self.tk.W)
        just_text = self.tk.Text(input_frame, height=5, width=30)
        just_text.grid(row=2, column=1, padx=5, pady=5, sticky=self.tk.W+self.tk.E)
        
        # Buttons
        button_frame = self.ttk.Frame(input_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.ttk.Button(
            button_frame, 
            text="Übernehmen",
            command=lambda: self._apply_custom_coding(
                category_entry.get(),
                subcats_entry.get(),
                just_text.get(1.0, self.tk.END),
                justification_text,
                custom_window
            )
        ).pack(side=self.tk.LEFT, padx=5)
        
        self.ttk.Button(
            button_frame,
            text="Abbrechen",
            command=custom_window.destroy
        ).pack(side=self.tk.LEFT, padx=5)
    
    def _apply_custom_coding(self, category, subcategories, justification, main_just_text, window):
        """
        Übernimmt die benutzerdefinierte Kodierung.
        """
        # Erstelle eine benutzerdefinierte Kodierung
        self.custom_coding = {
            'category': category,
            'subcategories': [s.strip() for s in subcategories.split(',') if s.strip()],
            'justification': justification.strip(),
            'coder_id': 'human_review',
            'confidence': {'total': 1.0, 'category': 1.0, 'subcategories': 1.0}
        }
        
        # Aktualisiere das Begründungsfeld im Hauptfenster
        main_just_text.delete(1.0, self.tk.END)
        main_just_text.insert(self.tk.END, f"Eigene Entscheidung: {justification.strip()}")
        
        # Schließe das Fenster
        window.destroy()
    
    def _navigate(self, direction, text_widget, codings_frame, discrepant_segments, progress_var):
        """
        Navigation zwischen den Segmenten und Speicherung der Entscheidung.
        """
        if self.current_segment is None or self.current_codings is None:
            return
            
        # Speichere aktuelle Entscheidung
        self._save_current_decision(text_widget)
        
        # Berechne neuen Index
        new_index = self.current_index + direction
        
        # Prüfe Grenzen
        if 0 <= new_index < len(discrepant_segments):
            self.current_index = new_index
            self._update_display(text_widget, codings_frame, discrepant_segments, text_widget, progress_var)
        elif new_index >= len(discrepant_segments):
            # Wenn wir am Ende angelangt sind, frage nach Abschluss
            if self.tk.messagebox.askyesno(
                "Review abschließen", 
                "Das war das letzte Segment. Möchten Sie den Review abschließen?"
            ):
                self.root.quit()
    
    def _on_closing(self):
        """Sicheres Schließen des Fensters mit vollständiger Ressourcenfreigabe"""
        try:
            if hasattr(self, 'root') and self.root:
                if self.tk.messagebox.askokcancel(
                    "Review beenden", 
                    "Möchten Sie den Review-Prozess wirklich beenden?\nGetroffene Entscheidungen werden gespeichert."
                ):
                    # Speichere aktuelle Entscheidung falls vorhanden
                    if self.current_segment is not None:
                        justification_text = None
                        for widget in self.root.winfo_children():
                            if isinstance(widget, self.tk.Text):
                                justification_text = widget
                                break
                        
                        if justification_text:
                            self._save_current_decision(justification_text)
                    
                    # Alle Tkinter-Variablen explizit löschen
                    for attr_name in dir(self):
                        attr = getattr(self, attr_name)
                        # Prüfen, ob es sich um eine Tkinter-Variable handelt
                        if hasattr(attr, '_tk'):
                            delattr(self, attr_name)
                    
                    # Fenster schließen
                    self.root.quit()
                    self.root.destroy()
                    self.root = None  # Wichtig: Referenz entfernen
        except:
            # Stelle sicher, dass Fenster auch bei Fehlern geschlossen wird
            if hasattr(self, 'root') and self.root:
                try:
                    self.root.quit()
                    self.root.destroy()
                    self.root = None
                except:
                    pass
