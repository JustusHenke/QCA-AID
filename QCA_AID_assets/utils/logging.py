"""
Logging Utilities

Console logging with file tee and timestamp handling for QCA-AID.
"""

import os
import sys
from datetime import datetime
from typing import Optional, TextIO


class TeeWriter:
    """
    Schreibt gleichzeitig in zwei Ausgabe-Streams (Console + Datei).
    Ähnlich dem Unix 'tee' Befehl.
    """
    
    def __init__(self, stream1: TextIO, stream2: TextIO):
        """
        Args:
            stream1: Erster Output-Stream (normalerweise sys.stdout)
            stream2: Zweiter Output-Stream (Log-Datei)
        """
        self.stream1 = stream1
        self.stream2 = stream2
        self.line_buffer = ""  # Buffer for incomplete lines
    
    def write(self, data: str) -> None:
        """Schreibt Daten in beide Streams"""
        try:
            # Write to original console
            # Fixes Windows-Unicode issues by fallback to ASCII replacement
            try:
                self.stream1.write(data)
            except UnicodeEncodeError:
                # On Unicode errors: convert to ASCII with replacement
                self.stream1.write(data.encode('ascii', errors='replace').decode('ascii'))
            self.stream1.flush()
            
            # Write to log file (with timestamp for each line)
            if self.stream2 and not self.stream2.closed:
                # Add data to buffer
                self.line_buffer += data
                
                # Process complete lines (those ending with \n)
                while '\n' in self.line_buffer:
                    line, self.line_buffer = self.line_buffer.split('\n', 1)
                    
                    # Add timestamp to non-empty lines that don't start with ===
                    if line.strip() and not line.startswith('==='):
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        timestamped_line = f"[{timestamp}] {line}\n"
                    else:
                        timestamped_line = f"{line}\n"
                    
                    self.stream2.write(timestamped_line)
                
                self.stream2.flush()
                
        except Exception as e:
            # Fallback to original console on logging errors
            try:
                self.stream1.write(f"[LOG ERROR: {str(e)[:50]}]\n")
            except:
                pass
            self.stream1.flush()
    
    def flush(self) -> None:
        """Flush beide Streams"""
        try:
            # Write any remaining buffered data to log file
            if self.stream2 and not self.stream2.closed and self.line_buffer:
                if self.line_buffer.strip() and not self.line_buffer.startswith('==='):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    timestamped_line = f"[{timestamp}] {self.line_buffer}"
                else:
                    timestamped_line = self.line_buffer
                self.stream2.write(timestamped_line)
                self.line_buffer = ""
            
            self.stream1.flush()
            if self.stream2 and not self.stream2.closed:
                self.stream2.flush()
        except Exception:
            pass
    
    def __getattr__(self, name: str):
        """Delegiert andere Attribute an den ersten Stream"""
        return getattr(self.stream1, name)


class ConsoleLogger:
    """
    Erweitert die Standard-Console-Ausgaben um automatisches Logging in Datei.
    Alle print()-Aufrufe werden sowohl in Console als auch in Log-Datei geschrieben.
    """
    
    def __init__(self, output_dir: str, log_filename: Optional[str] = None):
        """
        Initialisiert den Console Logger.
        
        Args:
            output_dir: Verzeichnis für Log-Dateien
            log_filename: Name der Log-Datei (optional)
        """
        # Use the same path logic as for Excel export
        if not os.path.isabs(output_dir):
            # Relative to script directory, like CONFIG['OUTPUT_DIR']
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(script_dir, output_dir)
        else:
            self.output_dir = output_dir
        
        # Create automatic log file with timestamp
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"console_log_{timestamp}.txt"
        
        self.log_path = os.path.join(self.output_dir, log_filename)
        
        # Store original stdout for restoration
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create output directory and open log file
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[DIR] Output-Verzeichnis: {self.output_dir}")
        except Exception as e:
            print(f"‼️ Fehler beim Erstellen des Output-Verzeichnisses: {e}")
            
        self.log_file: Optional[TextIO] = None
        self.is_active = False
        
        print(f"[LOG] Console Logger initialisiert: {self.log_path}")
    
    def start_logging(self) -> None:
        """
        Startet das parallele Logging aller Console-Ausgaben.
        """
        try:
            # Open log file with UTF-8 encoding
            self.log_file = open(self.log_path, 'w', encoding='utf-8', buffering=1)
            
            # Activate tee writer redirection
            sys.stdout = TeeWriter(self.original_stdout, self.log_file)
            sys.stderr = TeeWriter(self.original_stderr, self.log_file)
            
            self.is_active = True
            
            # Write start message to log
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"=== QCA-AID Console Log gestartet: {current_time} ===")
            print(f"Log-Datei: {self.log_path}")
            print("=" * 60)
            
        except Exception as e:
            print(f"‼️ Fehler beim Starten des Console Loggers: {e}")
            self.stop_logging()
    
    def stop_logging(self) -> None:
        """
        Stoppt das Console Logging und stellt ursprüngliche Ausgabe wieder her.
        """
        if self.is_active:
            try:
                # Write end message to log
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("=" * 60)
                print(f"=== QCA-AID Console Log beendet: {current_time} ===")
                
                # Show full absolute path (before closing log file)
                abs_path = os.path.abspath(self.log_path)
                print(f"[SUCCESS] Console Log gespeichert: {abs_path}")
                print("")  # Leere Zeile für bessere Lesbarkeit
                
                # Restore original stdout/stderr
                sys.stdout = self.original_stdout
                sys.stderr = self.original_stderr
                
                # Close log file
                if self.log_file:
                    self.log_file.close()
                    self.log_file = None
                
                self.is_active = False
                
            except Exception as e:
                print(f"‼️ Fehler beim Stoppen des Console Loggers: {e}")
                # Emergency restoration
                sys.stdout = self.original_stdout
                sys.stderr = self.original_stderr
    
    def __enter__(self):
        """Context Manager Unterstützung - Start"""
        self.start_logging()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Unterstützung - Ende"""
        self.stop_logging()
    
    def get_log_path(self) -> str:
        """Gibt den Pfad zur Log-Datei zurück"""
        return self.log_path


__all__ = [
    'ConsoleLogger',
    'TeeWriter',
]
