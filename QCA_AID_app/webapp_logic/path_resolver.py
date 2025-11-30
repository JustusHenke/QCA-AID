"""
Path Resolver
=============
Resolves and validates file paths relative to project root.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 8.3, 8.4
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List, Dict


class PathValidationResult:
    """Result of path validation with detailed information"""
    
    def __init__(self, is_valid: bool, error_message: Optional[str] = None,
                 warnings: Optional[List[str]] = None, suggestions: Optional[List[str]] = None):
        """
        Initialize validation result
        
        Args:
            is_valid: Whether the path is valid
            error_message: Error message if invalid
            warnings: List of warning messages
            suggestions: List of suggested actions
        """
        self.is_valid = is_valid
        self.error_message = error_message
        self.warnings = warnings or []
        self.suggestions = suggestions or []
    
    def has_warnings(self) -> bool:
        """Check if there are warnings"""
        return len(self.warnings) > 0
    
    def has_suggestions(self) -> bool:
        """Check if there are suggestions"""
        return len(self.suggestions) > 0


class PathResolver:
    """Resolves and validates file paths"""
    
    def __init__(self, project_root: Path):
        """
        Initialize with project root directory
        
        Args:
            project_root: Project root directory path
        """
        self.project_root = Path(project_root).resolve()
    
    def resolve(self, path: str) -> Path:
        """
        Resolve path (absolute or relative to project root)
        
        Args:
            path: Path string (can be absolute or relative)
        
        Returns:
            Resolved absolute Path object
        """
        path_obj = Path(path)
        
        # If already absolute, return as is
        if path_obj.is_absolute():
            return path_obj.resolve()
        
        # Otherwise resolve relative to project root
        return (self.project_root / path_obj).resolve()
    
    def validate_path(self, path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate path (legacy method for backward compatibility)
        
        Args:
            path: Path to validate
        
        Returns:
            Tuple of (is_valid, error_message)
            error_message is None if valid
        """
        result = self.validate_path_detailed(path)
        return result.is_valid, result.error_message
    
    def validate_path_detailed(self, path: Path, check_writable: bool = False) -> PathValidationResult:
        """
        Validate path with detailed error messages and suggestions
        
        Requirement 6.1: WHEN ein Benutzer einen ungültigen Dateipfad eingibt 
                        THEN das System SHALL eine spezifische Fehlermeldung anzeigen
        
        Args:
            path: Path to validate
            check_writable: Whether to check write permissions
        
        Returns:
            PathValidationResult with detailed information
        """
        try:
            # Check if path string is valid
            path_obj = Path(path)
            path_str = str(path)
            
            # Check for empty path (including "." which is current directory)
            if not path_str or not path_str.strip() or path_str in (".", ""):
                return PathValidationResult(
                    is_valid=False,
                    error_message="Pfad darf nicht leer sein",
                    suggestions=["Geben Sie einen gültigen Dateipfad ein"]
                )
            
            # Check for invalid characters (Windows)
            invalid_chars = '<>"|?*'
            found_invalid = [char for char in invalid_chars if char in path_str]
            if found_invalid:
                return PathValidationResult(
                    is_valid=False,
                    error_message=f"Pfad enthält ungültige Zeichen: {', '.join(found_invalid)}",
                    suggestions=[
                        "Entfernen Sie ungültige Zeichen aus dem Pfad",
                        f"Ungültige Zeichen: {invalid_chars}"
                    ]
                )
            
            # Check if path is too long (Windows MAX_PATH = 260)
            if len(path_str) > 260:
                return PathValidationResult(
                    is_valid=False,
                    error_message=f"Pfad ist zu lang ({len(path_str)} Zeichen, Maximum: 260)",
                    suggestions=[
                        "Verwenden Sie einen kürzeren Pfad",
                        "Verschieben Sie Dateien näher zum Root-Verzeichnis"
                    ]
                )
            
            # Check if parent directory exists (for files)
            if not path_obj.exists():
                parent = path_obj.parent
                if not parent.exists():
                    # Requirement 6.2: Suggest creating directory
                    return PathValidationResult(
                        is_valid=False,
                        error_message=f"Übergeordnetes Verzeichnis existiert nicht: {parent}",
                        suggestions=[
                            f"Verzeichnis '{parent}' erstellen",
                            "Wählen Sie einen anderen Pfad"
                        ]
                    )
                
                # File doesn't exist but parent does - this is OK for new files
                warnings = [f"Datei existiert noch nicht: {path_obj.name}"]
                
                # Check write permissions on parent
                if check_writable and not self.check_write_permissions(path_obj):
                    return PathValidationResult(
                        is_valid=False,
                        error_message=f"Keine Schreibrechte für Verzeichnis: {parent}",
                        suggestions=self._get_permission_suggestions(path_obj)
                    )
                
                return PathValidationResult(
                    is_valid=True,
                    warnings=warnings
                )
            
            # Path exists - check if it's the right type
            if path_obj.is_dir():
                # If we're validating a file path but got a directory
                if path_str.endswith(('.json', '.xlsx', '.pdf', '.txt')):
                    return PathValidationResult(
                        is_valid=False,
                        error_message=f"Pfad ist ein Verzeichnis, keine Datei: {path_obj}",
                        suggestions=["Wählen Sie eine Datei statt eines Verzeichnisses"]
                    )
            
            # Check write permissions if requested
            if check_writable and not self.check_write_permissions(path_obj):
                # Requirement 6.4: Permission warning
                return PathValidationResult(
                    is_valid=False,
                    error_message=f"Keine Schreibrechte für: {path_obj}",
                    suggestions=self._get_permission_suggestions(path_obj)
                )
            
            return PathValidationResult(is_valid=True)
            
        except Exception as e:
            return PathValidationResult(
                is_valid=False,
                error_message=f"Ungültiger Pfad: {str(e)}",
                suggestions=["Überprüfen Sie die Pfad-Syntax"]
            )
    
    def check_write_permissions(self, path: Path) -> bool:
        """
        Check if path has write permissions
        
        Args:
            path: Path to check
        
        Returns:
            True if writable, False otherwise
        """
        try:
            # If path exists, check if writable
            if path.exists():
                return os.access(path, os.W_OK)
            
            # If path doesn't exist, check parent directory
            parent = path.parent
            if parent.exists():
                return os.access(parent, os.W_OK)
            
            return False
            
        except Exception:
            return False
    
    def _get_permission_suggestions(self, path: Path) -> List[str]:
        """
        Get suggestions for permission issues
        
        Requirement 6.4: WHEN keine Schreibrechte vorhanden sind 
                        THEN das System SHALL eine Warnung anzeigen und alternative Speicherorte vorschlagen
        
        Args:
            path: Path with permission issues
        
        Returns:
            List of suggestion strings
        """
        suggestions = []
        alternatives = self.suggest_alternatives(path)
        
        if alternatives:
            suggestions.append("Alternative Speicherorte:")
            for alt in alternatives:
                suggestions.append(f"  • {alt}")
        else:
            suggestions.append("Wählen Sie einen Ordner mit Schreibrechten")
            suggestions.append("Versuchen Sie Ihr Benutzerverzeichnis")
        
        return suggestions
    
    def suggest_alternatives(self, path: Path) -> List[Path]:
        """
        Suggest alternative paths if current path is invalid
        
        Requirement 6.2: WHEN eine Datei nicht existiert 
                        THEN das System SHALL vorschlagen, die Datei zu erstellen oder einen anderen Pfad zu wählen
        
        Args:
            path: Invalid path
        
        Returns:
            List of suggested alternative paths
        """
        suggestions = []
        
        try:
            filename = path.name
            
            # Suggest project root
            project_suggestion = self.project_root / filename
            if project_suggestion.parent.exists() and os.access(project_suggestion.parent, os.W_OK):
                suggestions.append(project_suggestion)
            
            # Suggest user home directory
            home = Path.home()
            home_suggestion = home / filename
            if home_suggestion.parent.exists() and os.access(home_suggestion.parent, os.W_OK):
                suggestions.append(home_suggestion)
            
            # Suggest Documents folder if it exists
            documents = home / "Documents"
            if documents.exists() and os.access(documents, os.W_OK):
                suggestions.append(documents / filename)
            
            # Suggest temp directory as last resort
            temp_dir = Path(tempfile.gettempdir())
            temp_suggestion = temp_dir / filename
            if temp_suggestion.parent.exists() and os.access(temp_suggestion.parent, os.W_OK):
                suggestions.append(temp_suggestion)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_suggestions = []
            for s in suggestions:
                s_resolved = s.resolve()
                if s_resolved not in seen:
                    seen.add(s_resolved)
                    unique_suggestions.append(s)
            
            return unique_suggestions[:3]  # Return top 3
            
        except Exception:
            return []
    
    def validate_directory_path(self, path: Path, auto_create: bool = False) -> PathValidationResult:
        """
        Validate directory path with option to auto-create
        
        Requirement 6.3: WHEN ein Verzeichnis nicht existiert 
                        THEN das System SHALL anbieten, das Verzeichnis zu erstellen
        
        Args:
            path: Directory path to validate
            auto_create: Whether to offer directory creation
        
        Returns:
            PathValidationResult with detailed information
        """
        try:
            path_obj = Path(path)
            
            # First do basic path validation
            basic_result = self.validate_path_detailed(path_obj)
            if not basic_result.is_valid and "existiert nicht" not in basic_result.error_message:
                return basic_result
            
            # Check if path exists
            if not path_obj.exists():
                if auto_create:
                    return PathValidationResult(
                        is_valid=True,
                        warnings=[f"Verzeichnis existiert nicht: {path_obj}"],
                        suggestions=[f"Verzeichnis '{path_obj}' erstellen?"]
                    )
                else:
                    return PathValidationResult(
                        is_valid=False,
                        error_message=f"Verzeichnis existiert nicht: {path_obj}",
                        suggestions=[
                            "Verzeichnis erstellen",
                            "Wählen Sie ein existierendes Verzeichnis"
                        ]
                    )
            
            # Check if it's actually a directory
            if not path_obj.is_dir():
                return PathValidationResult(
                    is_valid=False,
                    error_message=f"Pfad ist keine Verzeichnis: {path_obj}",
                    suggestions=["Wählen Sie ein Verzeichnis statt einer Datei"]
                )
            
            # Check write permissions
            if not os.access(path_obj, os.W_OK):
                return PathValidationResult(
                    is_valid=False,
                    error_message=f"Keine Schreibrechte für Verzeichnis: {path_obj}",
                    suggestions=self._get_permission_suggestions(path_obj)
                )
            
            return PathValidationResult(is_valid=True)
            
        except Exception as e:
            return PathValidationResult(
                is_valid=False,
                error_message=f"Fehler bei Verzeichnis-Validierung: {str(e)}",
                suggestions=["Überprüfen Sie den Pfad"]
            )
    
    def create_directory(self, path: Path) -> Tuple[bool, Optional[str]]:
        """
        Create directory if it doesn't exist
        
        Requirement 6.3: Directory creation functionality
        
        Args:
            path: Directory path to create
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            path_obj = Path(path)
            
            if path_obj.exists():
                if path_obj.is_dir():
                    return True, None
                else:
                    return False, f"Pfad existiert bereits als Datei: {path_obj}"
            
            # Create directory with parents
            path_obj.mkdir(parents=True, exist_ok=True)
            return True, None
            
        except PermissionError:
            return False, f"Keine Berechtigung zum Erstellen von: {path}"
        except Exception as e:
            return False, f"Fehler beim Erstellen des Verzeichnisses: {str(e)}"
    
    def truncate_path(self, path: Path, max_length: int = 50) -> str:
        """
        Truncate long path with ellipsis for display
        
        Args:
            path: Path to truncate
            max_length: Maximum display length
        
        Returns:
            Truncated path string with ellipsis if needed
        """
        path_str = str(path)
        
        if len(path_str) <= max_length:
            return path_str
        
        # Try to keep filename and some parent directories
        parts = path.parts
        filename = parts[-1]
        
        # If filename alone is too long, truncate it
        if len(filename) > max_length - 10:
            return f"...{filename[-(max_length-10):]}"
        
        # Build path from end, adding parts until we hit limit
        result_parts = [filename]
        remaining_length = max_length - len(filename) - 3  # -3 for "..."
        
        for part in reversed(parts[:-1]):
            if len(part) + 1 <= remaining_length:  # +1 for separator
                result_parts.insert(0, part)
                remaining_length -= len(part) + 1
            else:
                break
        
        # Add ellipsis at the beginning
        if len(result_parts) < len(parts):
            result_parts.insert(0, "...")
        
        return str(Path(*result_parts))
    
    def make_relative(self, path: Path) -> Path:
        """
        Make path relative to project root if possible
        
        Args:
            path: Absolute path
        
        Returns:
            Relative path if within project root, otherwise absolute path
        """
        try:
            return path.relative_to(self.project_root)
        except ValueError:
            # Path is not relative to project root
            return path
