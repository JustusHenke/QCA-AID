"""
File and Analysis Data Models
==============================
Data models for file information, analysis status, and explorer configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import os


@dataclass
class FileInfo:
    """Repräsentiert Dateiinformationen"""
    path: str
    name: str
    size: int
    modified: datetime
    extension: str
    
    def format_size(self) -> str:
        """
        Formatiert Dateigröße lesbar.
        
        Returns:
            str: Formatierte Dateigröße (z.B. "1.5 MB")
        """
        size = self.size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    def format_date(self) -> str:
        """
        Formatiert Datum lesbar.
        
        Returns:
            str: Formatiertes Datum (z.B. "2024-01-15 14:30")
        """
        return self.modified.strftime("%Y-%m-%d %H:%M")
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary"""
        return {
            'path': self.path,
            'name': self.name,
            'size': self.size,
            'modified': self.modified.isoformat(),
            'extension': self.extension,
            'formatted_size': self.format_size(),
            'formatted_date': self.format_date()
        }
    
    @classmethod
    def from_path(cls, file_path: str) -> 'FileInfo':
        """
        Erstellt FileInfo aus Dateipfad.
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            FileInfo: FileInfo-Objekt
        """
        stat = os.stat(file_path)
        return cls(
            path=file_path,
            name=os.path.basename(file_path),
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime),
            extension=os.path.splitext(file_path)[1]
        )


@dataclass
class AnalysisStatus:
    """Repräsentiert Analysestatus"""
    is_running: bool
    progress: float  # 0.0 - 1.0
    current_step: str
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    output_file: Optional[str] = None
    
    def add_log(self, message: str) -> None:
        """
        Fügt Log-Nachricht hinzu.
        
        Args:
            message: Log-Nachricht
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
    
    def set_error(self, error_message: str) -> None:
        """
        Setzt Fehlerstatus.
        
        Args:
            error_message: Fehlermeldung
        """
        self.error = error_message
        self.is_running = False
        self.add_log(f"ERROR: {error_message}")
    
    def set_complete(self, output_file: str) -> None:
        """
        Setzt Analyse als abgeschlossen.
        
        Args:
            output_file: Pfad zur Output-Datei
        """
        self.is_running = False
        self.progress = 1.0
        self.output_file = output_file
        self.current_step = "Abgeschlossen"  # FIX: Verwende deutschen Text
        self.add_log(f"Analyse abgeschlossen: {output_file}")
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary"""
        return {
            'is_running': self.is_running,
            'progress': self.progress,
            'current_step': self.current_step,
            'logs': self.logs,
            'error': self.error,
            'output_file': self.output_file
        }
    
    @classmethod
    def create_initial(cls) -> 'AnalysisStatus':
        """
        Erstellt initialen Analysestatus.
        
        Returns:
            AnalysisStatus: Neuer Analysestatus
        """
        return cls(
            is_running=False,
            progress=0.0,
            current_step="Not started"
        )


@dataclass
class ExplorerConfig:
    """Repräsentiert Explorer-Konfiguration"""
    enabled_charts: List[str] = field(default_factory=list)
    color_scheme: str = "default"
    show_statistics: bool = True
    export_format: str = "xlsx"
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validiert Explorer-Konfiguration.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Validate enabled_charts
        if not isinstance(self.enabled_charts, list):
            errors.append("Enabled charts must be a list")
        
        # Validate color_scheme
        valid_schemes = {'default', 'dark', 'light', 'colorblind'}
        if self.color_scheme not in valid_schemes:
            errors.append(f"Invalid color scheme '{self.color_scheme}'. Must be one of: {', '.join(valid_schemes)}")
        
        # Validate export_format
        valid_formats = {'xlsx', 'csv', 'json', 'html'}
        if self.export_format not in valid_formats:
            errors.append(f"Invalid export format '{self.export_format}'. Must be one of: {', '.join(valid_formats)}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary"""
        return {
            'enabled_charts': self.enabled_charts,
            'color_scheme': self.color_scheme,
            'show_statistics': self.show_statistics,
            'export_format': self.export_format
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExplorerConfig':
        """Erstellt aus Dictionary"""
        return cls(
            enabled_charts=data.get('enabled_charts', []),
            color_scheme=data.get('color_scheme', 'default'),
            show_statistics=data.get('show_statistics', True),
            export_format=data.get('export_format', 'xlsx')
        )
    
    @classmethod
    def create_default(cls) -> 'ExplorerConfig':
        """
        Erstellt Standard-Explorer-Konfiguration.
        
        Returns:
            ExplorerConfig: Standard-Konfiguration
        """
        return cls(
            enabled_charts=[
                'category_distribution',
                'confidence_histogram',
                'coder_agreement',
                'temporal_analysis'
            ],
            color_scheme='default',
            show_statistics=True,
            export_format='xlsx'
        )
