"""
Escape Handler

Manages ESC-key handling during analysis for safe graceful shutdown.
Enables intermediate result export when user aborts analysis.
"""

import signal
import os
import sys
from typing import Any, Callable, Optional
from functools import wraps


class EscapeHandler:
    """
    FIX: Verbesserte ESC-Handler Klasse für QCA-AID
    Behandelt ESC-Taste Abbrüche mit korrekter Thread-Verwaltung
    """
    
    def __init__(self, analysis_manager: Any):  # Verwende Any statt 'IntegratedAnalysisManager'
        # FIX: Imports für die gesamte Klasse
        import threading
        import sys
        import time
        
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
        """Startet die Überwachung der Escape-Taste - FIX: Bessere Thread-Verwaltung"""
        if not self._keyboard_available:
            return
            
        # FIX: Nur starten wenn nicht bereits aktiv
        if self.monitoring:
            return
            
        self.monitoring = True
        self.escape_pressed = False
        self.user_wants_to_abort = False
        
        print("\nℹ️ Tipp: Druecken Sie ESC um die Kodierung sicher zu unterbrechen und Zwischenergebnisse zu speichern")
        
        # FIX: Bereinige alten Thread falls vorhanden
        if self.keyboard_thread and self.keyboard_thread.is_alive():
            self.monitoring = False  # Signalisiere altem Thread zu beenden
            # Kurz warten ohne join() vom gleichen Thread
            import time
            time.sleep(0.2)
            self.monitoring = True  # Wieder aktivieren für neuen Thread
        
        # Starte neuen Keyboard-Monitoring Thread
        import threading
        self.keyboard_thread = threading.Thread(target=self._monitor_escape, daemon=True)
        self.keyboard_thread.start()
    
    def stop_monitoring(self):
        """Stoppt die Überwachung der Escape-Taste - FIX: Thread-Join Problem behoben"""
        import threading
        
        self.monitoring = False
        # FIX: Prüfe ob wir im gleichen Thread sind um join() zu vermeiden
        if (self.keyboard_thread and 
            self.keyboard_thread.is_alive() and 
            self.keyboard_thread != threading.current_thread()):
            # Warte kurz auf das Ende des Threads
            self.keyboard_thread.join(timeout=1.0)
        # FIX: Bei gleichem Thread nur Flag setzen und Thread beendet sich selbst
        elif self.keyboard_thread == threading.current_thread():
            # Monitoring wurde bereits auf False gesetzt, Thread beendet sich
            pass
    
    def _monitor_escape(self):
        """Überwacht die Escape-Taste in separatem Thread - FIX: Verbesserte Behandlung"""
        if not self._keyboard_available:
            return
            
        try:
            import keyboard
            import time
            
            while self.monitoring:
                if keyboard.is_pressed('esc'):
                    if not self.escape_pressed:  # Verhindere mehrfache Verarbeitung
                        self.escape_pressed = True
                        # FIX: Stoppe Monitoring sofort um Interferenzen zu vermeiden
                        self.monitoring = False
                        self._handle_escape()
                    break
                time.sleep(0.1)  # Kleine Pause um CPU zu schonen
        except Exception as e:
            print(f"Fehler bei Escape-Überwachung: {str(e)}")
            # FIX: Bei Fehlern Monitoring sauber beenden
            self.monitoring = False
    
    def _handle_escape(self):
        """Behandelt das Drücken der Escape-Taste - FIX: Thread-sicherer Monitoring-Stop"""
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
            
            # FIX: Sanftes Stoppen des Monitorings (ohne join vom gleichen Thread)
            self.monitoring = False  # Signalisiert dem Thread sich zu beenden
            
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
                self.start_monitoring()  # FIX: Überwachung wieder starten
                
        except Exception as e:
            print(f"Fehler bei Escape-Behandlung: {str(e)}")
            print("Kodierung wird fortgesetzt...")
            self.escape_pressed = False
            # FIX: Nur bei kritischen Fehlern Monitoring neustarten
            if not self.monitoring:
                self.start_monitoring()
    
    def _get_user_choice_with_timeout(self, timeout: int = 30) -> str:
        """Holt Benutzereingabe mit Timeout - FIX: Verbesserte Eingabebehandlung"""
        print(f"\nIhre Wahl (Timeout in {timeout} Sekunden): ", end="", flush=True)
        
        # FIX: Imports direkt am Anfang der Methode
        import sys
        import time
        
        try:
            if sys.platform == "win32":
                # FIX: Windows-spezifische Implementierung mit korrekter Escape-Behandlung
                import msvcrt
                
                start_time = time.time()
                input_chars = []
                
                while time.time() - start_time < timeout:
                    if msvcrt.kbhit():
                        char = msvcrt.getch()
                        
                        # FIX: ESC-Taste erkennen (ASCII 27)
                        if char == b'\x1b':  # ESC-Taste
                            print("\nESC")
                            return 'abort_immediately'
                        
                        # FIX: Enter-Taste erkennen (ASCII 13 oder 10)
                        elif char in [b'\r', b'\n']:
                            print()  # Neue Zeile
                            user_input = ''.join(input_chars).strip().lower()
                            if user_input in ['j', 'n']:
                                return user_input
                            else:
                                print("Ungültige Eingabe. Bitte 'j', 'n' oder ESC drücken.")
                                return ''  # Leere Eingabe für Timeout-Behandlung
                        
                        # FIX: Backspace behandeln (ASCII 8)
                        elif char == b'\x08':  # Backspace
                            if input_chars:
                                input_chars.pop()
                                print('\b \b', end='', flush=True)  # Zeichen löschen
                        
                        # FIX: Nur gültige Zeichen hinzufügen
                        else:
                            try:
                                decoded_char = char.decode('utf-8')
                                if decoded_char.isprintable():
                                    input_chars.append(decoded_char)
                                    print(decoded_char, end='', flush=True)
                            except UnicodeDecodeError:
                                continue  # Ignoriere ungültige Zeichen
                    
                    # FIX: Kurze Pause um CPU zu entlasten
                    time.sleep(0.05)
                
                # FIX: Timeout erreicht
                print("\n⏰ Timeout erreicht")
                return ''
                
            else:
                # FIX: Unix/Linux-Implementierung mit verbesserter Escape-Behandlung
                import select
                import termios
                import tty
                
                # FIX: Terminal-Einstellungen sichern
                old_settings = termios.tcgetattr(sys.stdin)
                
                try:
                    tty.setraw(sys.stdin.fileno())
                    
                    start_time = time.time()
                    input_chars = []
                    
                    while time.time() - start_time < timeout:
                        # FIX: Non-blocking Input mit select
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            char = sys.stdin.read(1)
                            
                            # FIX: ESC-Taste erkennen
                            if ord(char) == 27:  # ESC
                                print("\nESC")
                                return 'abort_immediately'
                            
                            # FIX: Enter-Taste erkennen
                            elif ord(char) in [10, 13]:  # Enter/Return
                                print()  # Neue Zeile
                                user_input = ''.join(input_chars).strip().lower()
                                if user_input in ['j', 'n']:
                                    return user_input
                                else:
                                    print("Ungültige Eingabe. Bitte 'j', 'n' oder ESC drücken.")
                                    return ''
                            
                            # FIX: Backspace behandeln
                            elif ord(char) in [8, 127]:  # Backspace/Delete
                                if input_chars:
                                    input_chars.pop()
                                    print('\b \b', end='', flush=True)
                            
                            # FIX: Normale Zeichen hinzufügen
                            else:
                                if char.isprintable():
                                    input_chars.append(char)
                                    print(char, end='', flush=True)
                
                finally:
                    # FIX: Terminal-Einstellungen wiederherstellen
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                
                # FIX: Timeout erreicht
                print("\n⏰ Timeout erreicht")
                return ''
        
        except Exception as e:
            print(f"\nFehler bei Benutzereingabe: {str(e)}")
            print("Verwende Fallback-Eingabe...")
            
            # FIX: Fallback auf normale input() mit Timeout-Simulation
            try:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Eingabe-Timeout")
                
                # FIX: Nur unter Unix verfügbar
                if hasattr(signal, 'SIGALRM'):
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout)
                    
                    try:
                        user_input = input().strip().lower()
                        signal.alarm(0)  # Timer stoppen
                        
                        if user_input == 'j':
                            return 'j'
                        elif user_input == 'n':
                            return 'n'
                        else:
                            return ''
                    except TimeoutError:
                        print("\n⏰ Timeout erreicht")
                        return ''
                else:
                    # FIX: Einfacher Fallback ohne Timeout (Windows ohne msvcrt)
                    try:
                        user_input = input().strip().lower()
                        if user_input in ['j', 'n']:
                            return user_input
                        else:
                            return ''
                    except (EOFError, KeyboardInterrupt):
                        return 'abort_immediately'
                        
            except Exception as fallback_error:
                print(f"Auch Fallback fehlgeschlagen: {str(fallback_error)}")
                return ''
    
    
    def _get_current_status(self):
        """Ermittelt aktuellen Status der Analyse - FIX: Sichere Attribut-Zugriffe"""
        try:
            # FIX: Sichere Attribut-Zugriffe
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
        """Löst einen sicheren Abbruch aus - FIX: Mehrere Abort-Flags setzen"""
        # FIX: Setze verschiedene Flags für unterschiedliche Implementierungen
        if hasattr(self.analysis_manager, '_should_abort'):
            self.analysis_manager._should_abort = True
        
        # Alternative: Setze ein neues Attribut
        setattr(self.analysis_manager, '_escape_abort_requested', True)
        
        print("[ABORT] Abbruch-Signal gesendet...")
    
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
