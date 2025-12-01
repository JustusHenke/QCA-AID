"""
Analysis Runner
===============
Manages QCA-AID analysis execution as subprocess with progress monitoring and log streaming.
"""

import os
import sys
import subprocess
import threading
import queue
import time
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from datetime import datetime

from webapp_models.file_info import AnalysisStatus


class AnalysisRunner:
    """
    Verwaltet die Ausführung von QCA-AID Analysen als Subprocess.
    
    Responsibilities:
    - Startet QCA-AID Analyse als Subprocess
    - Überwacht Analyseprozess
    - Streamt Logs in Echtzeit
    - Behandelt Fehler und Abbrüche
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialisiert AnalysisRunner.
        
        Args:
            output_dir: Ausgabeverzeichnis für Logs und Ergebnisse
        """
        self.process: Optional[subprocess.Popen] = None
        self.status = AnalysisStatus.create_initial()
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'output')
        self.log_queue: queue.Queue = queue.Queue()
        self.log_thread: Optional[threading.Thread] = None
        self._stop_logging = threading.Event()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start_analysis(self, config: Dict) -> Tuple[bool, List[str]]:
        """
        Startet QCA-AID Analyse mit gegebener Konfiguration.
        
        Requirement 8.1: WHEN der Analyse-Reiter angezeigt wird 
                        THEN das System SHALL eine Schaltfläche "Analyse starten" anzeigen
        Requirement 8.2: WHEN ein Benutzer auf "Analyse starten" klickt 
                        THEN das System SHALL die aktuelle Konfiguration validieren
        Requirement 12.3: Implementiere Fehlerbehandlung für Analyse-Prozess
        
        Args:
            config: Konfigurationsdictionary mit allen Analyse-Parametern
            
        Returns:
            Tuple[bool, List[str]]: (success, messages) - messages include errors and warnings
        """
        messages = []
        
        # Check if analysis is already running
        if self.status.is_running:
            error_msg = "Eine Analyse läuft bereits. Bitte warten Sie, bis sie abgeschlossen ist."
            self.status.set_error(error_msg)
            return False, [error_msg]
        
        # Validate configuration
        is_valid, validation_messages = self._validate_config(config)
        messages.extend(validation_messages)
        
        if not is_valid:
            error_msg = "Konfigurationsvalidierung fehlgeschlagen. Bitte beheben Sie die Fehler."
            self.status.set_error(error_msg)
            return False, messages
        
        try:
            # Reset status
            self.status = AnalysisStatus(
                is_running=True,
                progress=0.0,
                current_step="Analyse wird gestartet"
            )
            self.status.add_log("QCA-AID Analyse wird initialisiert...")
            
            # Prepare command
            script_path = self._get_qca_aid_script_path()
            if not script_path:
                error_msg = (
                    "QCA-AID.py Skript nicht gefunden. "
                    "Stellen Sie sicher, dass Sie sich im richtigen Verzeichnis befinden."
                )
                self.status.set_error(error_msg)
                return False, [error_msg]
            
            self.status.add_log(f"QCA-AID Skript gefunden: {script_path}")
            
            # Build command
            cmd = [sys.executable, str(script_path)]
            
            # Set environment variables from config
            env = os.environ.copy()
            
            # Set flag to skip interactive prompts when running from webapp
            env['QCA_AID_WEBAPP_MODE'] = '1'
            
            # Pass project root to subprocess so CLI respects selected directory
            project_root = config.get('PROJECT_ROOT')
            if project_root:
                env['QCA_AID_PROJECT_ROOT'] = str(project_root)
            
            # Start subprocess
            self.status.add_log(f"Starte Subprocess: {' '.join(cmd)}")
            
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env,
                    cwd=os.path.dirname(script_path)
                )
            except FileNotFoundError as e:
                error_msg = f"Python-Interpreter nicht gefunden: {str(e)}"
                self.status.set_error(error_msg)
                return False, [error_msg]
            except PermissionError as e:
                error_msg = f"Keine Berechtigung zum Ausführen des Skripts: {str(e)}"
                self.status.set_error(error_msg)
                return False, [error_msg]
            
            # Start log streaming thread
            self._stop_logging.clear()
            self.log_thread = threading.Thread(
                target=self._stream_logs,
                daemon=True
            )
            self.log_thread.start()
            
            self.status.add_log("Analyse erfolgreich gestartet")
            self.status.current_step = "Analyse läuft"
            self.status.progress = 0.1
            
            success_msg = "Analyse wurde erfolgreich gestartet"
            messages.append(success_msg)
            
            return True, messages
            
        except OSError as e:
            error_msg = f"Betriebssystemfehler beim Starten der Analyse: {str(e)}"
            self.status.set_error(error_msg)
            return False, [error_msg]
        except Exception as e:
            error_msg = f"Unerwarteter Fehler beim Starten der Analyse: {str(e)}"
            self.status.set_error(error_msg)
            return False, [error_msg]
    
    def get_progress(self) -> Tuple[float, str]:
        """
        Gibt Fortschritt und Status zurück.
        
        Returns:
            Tuple[float, str]: (progress 0.0-1.0, current_step)
        """
        # Update progress based on log analysis
        self._update_progress_from_logs()
        
        # Check if process has finished
        if self.process and self.process.poll() is not None:
            if not self.status.error:
                # Process finished, check for output file
                output_file = self._find_output_file()
                if output_file:
                    self.status.set_complete(output_file)
                else:
                    # Check return code
                    if self.process.returncode == 0:
                        self.status.set_complete("Analysis completed (no output file found)")
                    else:
                        self.status.set_error(f"Analysis failed with return code {self.process.returncode}")
        
        return self.status.progress, self.status.current_step
    
    def get_logs(self) -> List[str]:
        """
        Gibt Log-Zeilen zurück.
        
        Returns:
            List[str]: Liste der Log-Nachrichten
        """
        # Collect any new logs from queue
        while not self.log_queue.empty():
            try:
                log_line = self.log_queue.get_nowait()
                self.status.add_log(log_line)
            except queue.Empty:
                break
        
        return self.status.logs
    
    def stop_analysis(self) -> Tuple[bool, Optional[str]]:
        """
        Stoppt laufende Analyse.
        
        Requirement 12.3: Implementiere Fehlerbehandlung für Analyse-Prozess
        
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        if not self.status.is_running:
            return True, None
        
        try:
            self.status.add_log("Analyse wird gestoppt...")
            
            # Stop log streaming
            self._stop_logging.set()
            
            # Terminate process
            if self.process:
                try:
                    self.process.terminate()
                    
                    # Wait for process to terminate (with timeout)
                    try:
                        self.process.wait(timeout=5)
                        self.status.add_log("Prozess wurde sauber beendet")
                    except subprocess.TimeoutExpired:
                        # Force kill if not terminated
                        self.status.add_log("Prozess reagiert nicht, erzwinge Beendigung...")
                        self.process.kill()
                        self.process.wait()
                        self.status.add_log("Prozess wurde zwangsweise beendet")
                except ProcessLookupError:
                    # Process already terminated
                    self.status.add_log("Prozess war bereits beendet")
                except PermissionError as e:
                    error_msg = f"Keine Berechtigung zum Beenden des Prozesses: {str(e)}"
                    self.status.add_log(error_msg)
                    return False, error_msg
            
            # Wait for log thread to finish
            if self.log_thread and self.log_thread.is_alive():
                self.log_thread.join(timeout=2)
                if self.log_thread.is_alive():
                    self.status.add_log("Warnung: Log-Thread konnte nicht sauber beendet werden")
            
            self.status.is_running = False
            self.status.current_step = "Vom Benutzer gestoppt"
            self.status.add_log("Analyse erfolgreich gestoppt")
            
            return True, None
            
        except OSError as e:
            error_msg = f"Betriebssystemfehler beim Stoppen der Analyse: {str(e)}"
            self.status.set_error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unerwarteter Fehler beim Stoppen der Analyse: {str(e)}"
            self.status.set_error(error_msg)
            return False, error_msg
    
    def _validate_config(self, config: Dict) -> Tuple[bool, List[str]]:
        """
        Validiert Konfiguration vor Analyse-Start.
        
        Requirement 8.2: WHEN ein Benutzer auf "Analyse starten" klickt 
                        THEN das System SHALL die aktuelle Konfiguration validieren
        Requirement 12.4: Implementiere Warnung für fehlende Pflichtfelder
        
        Args:
            config: Konfigurationsdictionary
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = {
            'MODEL_PROVIDER': 'Modell-Anbieter',
            'MODEL_NAME': 'Modell-Name',
            'DATA_DIR': 'Eingabeverzeichnis',
            'OUTPUT_DIR': 'Ausgabeverzeichnis'
        }
        
        for field, display_name in required_fields.items():
            if field not in config or not config[field]:
                errors.append(f"Pflichtfeld fehlt: {display_name}")
        
        # Check and create directories if needed
        if 'DATA_DIR' in config and config['DATA_DIR']:
            data_dir = config['DATA_DIR']
            if not os.path.exists(data_dir):
                try:
                    os.makedirs(data_dir, exist_ok=True)
                    warnings.append(f"Eingabeverzeichnis wurde erstellt: {data_dir}")
                except OSError as e:
                    errors.append(f"Eingabeverzeichnis konnte nicht erstellt werden: {str(e)}")
            elif not os.path.isdir(data_dir):
                errors.append(f"Eingabepfad ist kein Verzeichnis: {data_dir}")
            else:
                # Check if there are any files in input directory
                try:
                    files = [f for f in os.listdir(data_dir) 
                            if os.path.isfile(os.path.join(data_dir, f))]
                    if not files:
                        errors.append(
                            f"Eingabeverzeichnis ist leer: {data_dir}. "
                            "Bitte fügen Sie Dateien hinzu, bevor Sie die Analyse starten."
                        )
                except PermissionError:
                    errors.append(f"Keine Leseberechtigung für Eingabeverzeichnis: {data_dir}")
        
        if 'OUTPUT_DIR' in config and config['OUTPUT_DIR']:
            output_dir = config['OUTPUT_DIR']
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    warnings.append(f"Ausgabeverzeichnis wurde erstellt: {output_dir}")
                except OSError as e:
                    errors.append(f"Ausgabeverzeichnis konnte nicht erstellt werden: {str(e)}")
            elif not os.path.isdir(output_dir):
                errors.append(f"Ausgabepfad ist kein Verzeichnis: {output_dir}")
            else:
                # Check write permissions
                if not os.access(output_dir, os.W_OK):
                    errors.append(f"Keine Schreibberechtigung für Ausgabeverzeichnis: {output_dir}")
        
        # Validate numeric parameters
        numeric_params = {
            'CHUNK_SIZE': (100, 10000, 'Chunk-Größe'),
            'CHUNK_OVERLAP': (0, 1000, 'Chunk-Überlappung'),
            'BATCH_SIZE': (1, 100, 'Batch-Größe')
        }
        
        for param, (min_val, max_val, display_name) in numeric_params.items():
            if param in config:
                try:
                    value = int(config[param])
                    if value < min_val or value > max_val:
                        errors.append(
                            f"{display_name} muss zwischen {min_val} und {max_val} liegen "
                            f"(aktuell: {value})"
                        )
                except (ValueError, TypeError):
                    errors.append(f"{display_name} muss eine gültige Ganzzahl sein")
        
        # Validate chunk overlap vs chunk size
        if 'CHUNK_SIZE' in config and 'CHUNK_OVERLAP' in config:
            try:
                chunk_size = int(config['CHUNK_SIZE'])
                chunk_overlap = int(config['CHUNK_OVERLAP'])
                if chunk_overlap >= chunk_size:
                    errors.append(
                        "Chunk-Überlappung muss kleiner als Chunk-Größe sein "
                        f"(Überlappung: {chunk_overlap}, Größe: {chunk_size})"
                    )
            except (ValueError, TypeError):
                pass  # Already caught above
        
        # Validate analysis mode
        if 'ANALYSIS_MODE' in config:
            valid_modes = ['full', 'deductive', 'abductive', 'inductive', 'grounded']
            if config['ANALYSIS_MODE'] not in valid_modes:
                errors.append(
                    f"Ungültiger Analyse-Modus: {config['ANALYSIS_MODE']}. "
                    f"Muss einer der folgenden sein: {', '.join(valid_modes)}"
                )
        
        # Validate review mode
        if 'REVIEW_MODE' in config:
            valid_modes = ['auto', 'manual', 'consensus', 'majority']
            if config['REVIEW_MODE'] not in valid_modes:
                errors.append(
                    f"Ungültiger Review-Modus: {config['REVIEW_MODE']}. "
                    f"Muss einer der folgenden sein: {', '.join(valid_modes)}"
                )
        
        # Validate coder settings
        if 'CODER_SETTINGS' in config:
            coder_settings = config['CODER_SETTINGS']
            if not isinstance(coder_settings, list) or len(coder_settings) == 0:
                errors.append("Mindestens ein Coder muss konfiguriert sein")
            else:
                for i, coder in enumerate(coder_settings):
                    if not isinstance(coder, dict):
                        errors.append(f"Coder {i+1}: Ungültige Konfiguration")
                        continue
                    
                    if 'temperature' not in coder:
                        errors.append(f"Coder {i+1}: Temperatur fehlt")
                    else:
                        try:
                            temp = float(coder['temperature'])
                            if temp < 0.0 or temp > 2.0:
                                errors.append(
                                    f"Coder {i+1}: Temperatur muss zwischen 0.0 und 2.0 liegen "
                                    f"(aktuell: {temp})"
                                )
                        except (ValueError, TypeError):
                            errors.append(f"Coder {i+1}: Temperatur muss eine Zahl sein")
                    
                    if 'coder_id' not in coder or not coder['coder_id']:
                        errors.append(f"Coder {i+1}: Coder-ID fehlt")
        
        # Add warnings to errors if any (for logging)
        all_messages = errors + warnings
        
        return len(errors) == 0, all_messages
    
    def _get_qca_aid_script_path(self) -> Optional[Path]:
        """
        Findet den Pfad zum QCA-AID.py Skript.
        
        Returns:
            Optional[Path]: Pfad zum Skript oder None wenn nicht gefunden
        """
        # Try current directory
        current_dir = Path.cwd()
        script_path = current_dir / "QCA-AID.py"
        
        if script_path.exists():
            return script_path
        
        # Try parent directory
        parent_dir = current_dir.parent
        script_path = parent_dir / "QCA-AID.py"
        
        if script_path.exists():
            return script_path
        
        # Try script directory
        script_dir = Path(__file__).parent.parent
        script_path = script_dir / "QCA-AID.py"
        
        if script_path.exists():
            return script_path
        
        return None
    
    def _stream_logs(self):
        """
        Streamt Logs vom Subprocess in die Queue.
        Läuft in separatem Thread.
        """
        if not self.process or not self.process.stdout:
            return
        
        try:
            for line in iter(self.process.stdout.readline, ''):
                if self._stop_logging.is_set():
                    break
                
                if line:
                    line = line.rstrip()
                    self.log_queue.put(line)
            
        except Exception as e:
            self.log_queue.put(f"Error streaming logs: {str(e)}")
        
        finally:
            if self.process and self.process.stdout:
                self.process.stdout.close()
    
    def _update_progress_from_logs(self):
        """
        Aktualisiert Fortschritt basierend auf Log-Analyse.
        """
        if not self.status.logs:
            return
        
        # Analyze recent logs for progress indicators
        recent_logs = self.status.logs[-10:]  # Last 10 log entries
        
        # Define progress markers
        progress_markers = {
            'Lese Dokumente': 0.1,
            'Bereite Material': 0.2,
            'Starte manuelle Kodierung': 0.3,
            'Starte integrierte Analyse': 0.4,
            'Kodierung': 0.5,
            'Berechne': 0.7,
            'Review': 0.8,
            'Export': 0.9,
            'erfolgreich': 1.0
        }
        
        # Find highest matching progress
        max_progress = self.status.progress
        current_step = self.status.current_step
        
        for log in recent_logs:
            for marker, progress in progress_markers.items():
                if marker.lower() in log.lower():
                    if progress > max_progress:
                        max_progress = progress
                        current_step = marker
        
        self.status.progress = max_progress
        if current_step != self.status.current_step:
            self.status.current_step = current_step
    
    def _find_output_file(self) -> Optional[str]:
        """
        Sucht nach der generierten Output-Datei.
        
        Returns:
            Optional[str]: Pfad zur Output-Datei oder None
        """
        if not os.path.exists(self.output_dir):
            return None
        
        # Look for XLSX files in output directory
        xlsx_files = []
        for file in os.listdir(self.output_dir):
            if file.endswith('.xlsx') and not file.startswith('~'):
                file_path = os.path.join(self.output_dir, file)
                xlsx_files.append((file_path, os.path.getmtime(file_path)))
        
        if not xlsx_files:
            return None
        
        # Return most recent file
        xlsx_files.sort(key=lambda x: x[1], reverse=True)
        return xlsx_files[0][0]
    
    def get_status(self) -> AnalysisStatus:
        """
        Gibt aktuellen Analysestatus zurück.
        
        Returns:
            AnalysisStatus: Aktueller Status
        """
        # Update progress before returning
        self.get_progress()
        return self.status
    
    def is_running(self) -> bool:
        """
        Prüft ob Analyse läuft.
        
        Returns:
            bool: True wenn Analyse läuft
        """
        return self.status.is_running
