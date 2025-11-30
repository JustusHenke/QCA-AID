"""
Validators Module
=================
Centralized validation logic for QCA-AID Webapp.

Provides inline validation for all form fields and comprehensive error handling.

Requirements: 2.5, 3.5, 5.5, 7.4, 7.5, 12.4, 12.5
"""

import os
import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any


class ValidationResult:
    """
    Represents the result of a validation operation.
    """
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        """
        Initialize validation result.
        
        Args:
            is_valid: Whether validation passed
            errors: List of error messages
            warnings: List of warning messages
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def get_all_messages(self) -> List[str]:
        """Get all error and warning messages."""
        return self.errors + self.warnings


class FieldValidator:
    """
    Provides validation methods for individual form fields.
    
    Requirement 3.5: WHEN ein Benutzer einen ungültigen Wert eingibt 
                    THEN das System SHALL eine Inline-Validierungsmeldung anzeigen
    """
    
    @staticmethod
    def validate_required_field(value: Any, field_name: str) -> ValidationResult:
        """
        Validates that a required field is not empty.
        
        Args:
            value: Field value to validate
            field_name: Name of the field for error messages
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        if value is None:
            result.add_error(f"{field_name} ist erforderlich")
        elif isinstance(value, str) and not value.strip():
            result.add_error(f"{field_name} darf nicht leer sein")
        elif isinstance(value, (list, dict)) and len(value) == 0:
            result.add_error(f"{field_name} darf nicht leer sein")
        
        return result
    
    @staticmethod
    def validate_string_length(value: str, field_name: str, 
                               min_length: Optional[int] = None, 
                               max_length: Optional[int] = None) -> ValidationResult:
        """
        Validates string length constraints.
        
        Args:
            value: String value to validate
            field_name: Name of the field for error messages
            min_length: Minimum length (optional)
            max_length: Maximum length (optional)
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            result.add_error(f"{field_name} muss ein Text sein")
            return result
        
        length = len(value.strip())
        
        if min_length is not None and length < min_length:
            result.add_error(f"{field_name} muss mindestens {min_length} Zeichen haben")
        
        if max_length is not None and length > max_length:
            result.add_error(f"{field_name} darf maximal {max_length} Zeichen haben")
        
        return result
    
    @staticmethod
    def validate_word_count(value: str, field_name: str, 
                           min_words: Optional[int] = None) -> ValidationResult:
        """
        Validates minimum word count.
        
        Args:
            value: String value to validate
            field_name: Name of the field for error messages
            min_words: Minimum number of words
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            result.add_error(f"{field_name} muss ein Text sein")
            return result
        
        word_count = len(value.split())
        
        if min_words is not None and word_count < min_words:
            result.add_error(
                f"{field_name} muss mindestens {min_words} Wörter haben (aktuell: {word_count})"
            )
        
        return result
    
    @staticmethod
    def validate_numeric_range(value: Any, field_name: str, 
                               min_value: Optional[float] = None, 
                               max_value: Optional[float] = None) -> ValidationResult:
        """
        Validates numeric value is within range.
        
        Args:
            value: Numeric value to validate
            field_name: Name of the field for error messages
            min_value: Minimum value (optional)
            max_value: Maximum value (optional)
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            result.add_error(f"{field_name} muss eine Zahl sein")
            return result
        
        if min_value is not None and numeric_value < min_value:
            result.add_error(f"{field_name} muss mindestens {min_value} sein")
        
        if max_value is not None and numeric_value > max_value:
            result.add_error(f"{field_name} darf maximal {max_value} sein")
        
        return result
    
    @staticmethod
    def validate_list_length(value: List, field_name: str, 
                            min_items: Optional[int] = None, 
                            max_items: Optional[int] = None) -> ValidationResult:
        """
        Validates list has required number of items.
        
        Args:
            value: List to validate
            field_name: Name of the field for error messages
            min_items: Minimum number of items (optional)
            max_items: Maximum number of items (optional)
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        if not isinstance(value, list):
            result.add_error(f"{field_name} muss eine Liste sein")
            return result
        
        length = len(value)
        
        if min_items is not None and length < min_items:
            result.add_error(
                f"{field_name} muss mindestens {min_items} Einträge haben (aktuell: {length})"
            )
        
        if max_items is not None and length > max_items:
            result.add_error(
                f"{field_name} darf maximal {max_items} Einträge haben (aktuell: {length})"
            )
        
        return result
    
    @staticmethod
    def validate_choice(value: Any, field_name: str, 
                       valid_choices: List[Any]) -> ValidationResult:
        """
        Validates value is one of the valid choices.
        
        Args:
            value: Value to validate
            field_name: Name of the field for error messages
            valid_choices: List of valid choices
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        if value not in valid_choices:
            choices_str = ', '.join(str(c) for c in valid_choices)
            result.add_error(
                f"{field_name} muss einer der folgenden Werte sein: {choices_str}"
            )
        
        return result
    
    @staticmethod
    def validate_identifier(value: str, field_name: str) -> ValidationResult:
        """
        Validates value is a valid identifier (alphanumeric + underscore).
        
        Args:
            value: String value to validate
            field_name: Name of the field for error messages
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            result.add_error(f"{field_name} muss ein Text sein")
            return result
        
        if not re.match(r'^[a-zA-Z0-9_]+$', value):
            result.add_error(
                f"{field_name} darf nur Buchstaben, Zahlen und Unterstriche enthalten"
            )
        
        return result


class FileSystemValidator:
    """
    Provides validation methods for file system operations.
    
    Requirement 7.4: WHEN keine Eingabedateien vorhanden sind 
                    THEN das System SHALL eine Meldung mit Upload-Hinweis anzeigen
    Requirement 7.5: WHEN der INPUT_DIR nicht existiert 
                    THEN das System SHALL den Ordner automatisch erstellen
    Requirement 12.5: Implementiere automatische Verzeichniserstellung
    Requirements 6.1, 6.2, 6.3, 6.4, 8.4: Path validation and error handling
    """
    
    @staticmethod
    def validate_path_realtime(path: str, field_name: str, 
                               path_resolver=None, check_writable: bool = False) -> ValidationResult:
        """
        Real-time path validation with detailed feedback
        
        Requirement 8.4: WHEN ein Benutzer einen Pfad manuell eingibt 
                        THEN das System SHALL diesen in Echtzeit validieren und visuelles Feedback geben
        
        Args:
            path: Path string to validate
            field_name: Name of the field for error messages
            path_resolver: Optional PathResolver instance for detailed validation
            check_writable: Whether to check write permissions
            
        Returns:
            ValidationResult: Validation result with errors and warnings
        """
        result = ValidationResult(True)
        
        if not path or not path.strip():
            result.add_error(f"{field_name} darf nicht leer sein")
            return result
        
        # Use PathResolver for detailed validation if available
        if path_resolver:
            from webapp_logic.path_resolver import PathResolver
            if isinstance(path_resolver, PathResolver):
                path_obj = Path(path)
                detailed_result = path_resolver.validate_path_detailed(path_obj, check_writable)
                
                if not detailed_result.is_valid:
                    result.add_error(f"{field_name}: {detailed_result.error_message}")
                    
                    # Add suggestions as warnings
                    for suggestion in detailed_result.suggestions:
                        result.add_warning(suggestion)
                
                # Add warnings from detailed validation
                for warning in detailed_result.warnings:
                    result.add_warning(warning)
                
                return result
        
        # Fallback to basic validation
        try:
            path_obj = Path(path)
            
            # Check for invalid characters
            invalid_chars = '<>"|?*'
            if any(char in path for char in invalid_chars):
                result.add_error(f"{field_name} enthält ungültige Zeichen")
                return result
            
            # Check if parent exists
            if not path_obj.exists():
                parent = path_obj.parent
                if not parent.exists():
                    result.add_error(f"{field_name}: Übergeordnetes Verzeichnis existiert nicht")
                    result.add_warning("Verzeichnis erstellen oder anderen Pfad wählen")
                else:
                    result.add_warning(f"{field_name}: Datei existiert noch nicht")
            
        except Exception as e:
            result.add_error(f"{field_name}: Ungültiger Pfad - {str(e)}")
        
        return result
    
    @staticmethod
    def validate_directory_exists(path: str, field_name: str, 
                                  auto_create: bool = False) -> ValidationResult:
        """
        Validates that a directory exists, optionally creating it.
        
        Args:
            path: Directory path to validate
            field_name: Name of the field for error messages
            auto_create: Whether to automatically create the directory
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        if not path or not path.strip():
            result.add_error(f"{field_name} darf nicht leer sein")
            return result
        
        dir_path = Path(path)
        
        if not dir_path.exists():
            if auto_create:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    result.add_warning(f"Verzeichnis '{path}' wurde automatisch erstellt")
                except OSError as e:
                    result.add_error(
                        f"Verzeichnis '{path}' konnte nicht erstellt werden: {str(e)}"
                    )
            else:
                result.add_error(f"Verzeichnis '{path}' existiert nicht")
        elif not dir_path.is_dir():
            result.add_error(f"'{path}' ist kein Verzeichnis")
        
        return result
    
    @staticmethod
    def validate_directory_not_empty(path: str, field_name: str, 
                                     extensions: Optional[List[str]] = None) -> ValidationResult:
        """
        Validates that a directory contains files.
        
        Args:
            path: Directory path to validate
            field_name: Name of the field for error messages
            extensions: Optional list of file extensions to check for
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        dir_path = Path(path)
        
        if not dir_path.exists():
            result.add_error(f"Verzeichnis '{path}' existiert nicht")
            return result
        
        if not dir_path.is_dir():
            result.add_error(f"'{path}' ist kein Verzeichnis")
            return result
        
        # Check for files
        files = []
        for item in dir_path.iterdir():
            if item.is_file():
                if extensions:
                    if item.suffix.lower() in [ext.lower() for ext in extensions]:
                        files.append(item)
                else:
                    files.append(item)
        
        if not files:
            if extensions:
                ext_str = ', '.join(extensions)
                result.add_warning(
                    f"Keine Dateien mit Endungen {ext_str} in '{path}' gefunden"
                )
            else:
                result.add_warning(f"Keine Dateien in '{path}' gefunden")
        
        return result
    
    @staticmethod
    def validate_file_exists(path: str, field_name: str) -> ValidationResult:
        """
        Validates that a file exists.
        
        Args:
            path: File path to validate
            field_name: Name of the field for error messages
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        if not path or not path.strip():
            result.add_error(f"{field_name} darf nicht leer sein")
            return result
        
        file_path = Path(path)
        
        if not file_path.exists():
            result.add_error(f"Datei '{path}' existiert nicht")
        elif not file_path.is_file():
            result.add_error(f"'{path}' ist keine Datei")
        
        return result
    
    @staticmethod
    def validate_file_writable(path: str, field_name: str) -> ValidationResult:
        """
        Validates that a file path is writable.
        
        Args:
            path: File path to validate
            field_name: Name of the field for error messages
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        if not path or not path.strip():
            result.add_error(f"{field_name} darf nicht leer sein")
            return result
        
        file_path = Path(path)
        
        # Check if parent directory exists and is writable
        parent_dir = file_path.parent
        
        if not parent_dir.exists():
            result.add_error(f"Verzeichnis '{parent_dir}' existiert nicht")
        elif not os.access(parent_dir, os.W_OK):
            result.add_error(f"Keine Schreibrechte für Verzeichnis '{parent_dir}'")
        
        # If file exists, check if it's writable
        if file_path.exists():
            if not os.access(file_path, os.W_OK):
                result.add_error(f"Keine Schreibrechte für Datei '{path}'")
        
        return result
    
    @staticmethod
    def validate_path_safe(path: str, base_dir: str, field_name: str) -> ValidationResult:
        """
        Validates that a path is safe (no path traversal).
        
        Args:
            path: Path to validate
            base_dir: Base directory that path must be within
            field_name: Name of the field for error messages
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        try:
            abs_path = Path(path).resolve()
            abs_base = Path(base_dir).resolve()
            
            if not abs_path.is_relative_to(abs_base):
                result.add_error(
                    f"{field_name} muss innerhalb von '{base_dir}' liegen"
                )
        except (ValueError, OSError) as e:
            result.add_error(f"Ungültiger Pfad für {field_name}: {str(e)}")
        
        return result


class ConfigValidator:
    """
    Provides validation methods for configuration data.
    
    Requirement 2.5: WHEN eine ungültige Konfigurationsdatei geladen wird 
                    THEN das System SHALL eine Fehlermeldung mit Details anzeigen
    """
    
    @staticmethod
    def validate_config_complete(config_dict: Dict) -> ValidationResult:
        """
        Validates that all required configuration fields are present.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        required_fields = [
            'model_provider',
            'model_name',
            'data_dir',
            'output_dir',
            'chunk_size',
            'chunk_overlap',
            'batch_size',
            'analysis_mode',
            'review_mode'
        ]
        
        for field in required_fields:
            if field not in config_dict or config_dict[field] is None:
                result.add_error(f"Pflichtfeld '{field}' fehlt")
        
        return result
    
    @staticmethod
    def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> ValidationResult:
        """
        Validates chunk size and overlap settings.
        
        Args:
            chunk_size: Chunk size value
            chunk_overlap: Chunk overlap value
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        # Validate chunk size
        size_result = FieldValidator.validate_numeric_range(
            chunk_size, "Chunk-Größe", min_value=100, max_value=10000
        )
        result.errors.extend(size_result.errors)
        
        # Validate chunk overlap
        overlap_result = FieldValidator.validate_numeric_range(
            chunk_overlap, "Chunk-Überlappung", min_value=0, max_value=1000
        )
        result.errors.extend(overlap_result.errors)
        
        # Validate relationship
        if chunk_overlap >= chunk_size:
            result.add_error("Chunk-Überlappung muss kleiner als Chunk-Größe sein")
        
        if result.has_errors():
            result.is_valid = False
        
        return result
    
    @staticmethod
    def validate_coder_settings(coder_settings: List[Dict]) -> ValidationResult:
        """
        Validates coder settings.
        
        Args:
            coder_settings: List of coder setting dictionaries
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        if not coder_settings or len(coder_settings) == 0:
            result.add_error("Mindestens ein Coder muss konfiguriert sein")
            return result
        
        coder_ids = set()
        
        for i, coder in enumerate(coder_settings):
            # Validate temperature
            if 'temperature' not in coder:
                result.add_error(f"Coder {i+1}: Temperatur fehlt")
            else:
                temp_result = FieldValidator.validate_numeric_range(
                    coder['temperature'], f"Coder {i+1} Temperatur", 
                    min_value=0.0, max_value=2.0
                )
                result.errors.extend(temp_result.errors)
            
            # Validate coder_id
            if 'coder_id' not in coder:
                result.add_error(f"Coder {i+1}: Coder-ID fehlt")
            else:
                coder_id = coder['coder_id']
                if not coder_id or not coder_id.strip():
                    result.add_error(f"Coder {i+1}: Coder-ID darf nicht leer sein")
                elif coder_id in coder_ids:
                    result.add_error(f"Coder {i+1}: Coder-ID '{coder_id}' ist nicht eindeutig")
                else:
                    coder_ids.add(coder_id)
        
        if result.has_errors():
            result.is_valid = False
        
        return result


class CodebookValidator:
    """
    Provides validation methods for codebook data.
    
    Requirement 5.5: WHEN die Forschungsfrage leer ist 
                    THEN das System SHALL eine Warnung anzeigen
    """
    
    @staticmethod
    def validate_research_question(question: str) -> ValidationResult:
        """
        Validates research question.
        
        Args:
            question: Research question to validate
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        if not question or not question.strip():
            result.add_warning("Forschungsfrage ist leer")
        elif len(question.split()) < 5:
            result.add_warning(
                "Forschungsfrage sollte mindestens 5 Wörter haben"
            )
        
        return result
    
    @staticmethod
    def validate_category(category_dict: Dict) -> ValidationResult:
        """
        Validates a category.
        
        Args:
            category_dict: Category dictionary to validate
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        # Validate name
        if 'name' not in category_dict or not category_dict['name']:
            result.add_error("Kategoriename fehlt")
        else:
            name_result = FieldValidator.validate_string_length(
                category_dict['name'], "Kategoriename", min_length=3, max_length=50
            )
            result.errors.extend(name_result.errors)
        
        # Validate definition
        if 'definition' not in category_dict or not category_dict['definition']:
            result.add_error("Kategoriedefinition fehlt")
        else:
            def_result = FieldValidator.validate_word_count(
                category_dict['definition'], "Kategoriedefinition", min_words=15
            )
            result.errors.extend(def_result.errors)
        
        # Validate examples
        if 'examples' not in category_dict:
            result.add_error("Beispiele fehlen")
        else:
            examples_result = FieldValidator.validate_list_length(
                category_dict['examples'], "Beispiele", min_items=2
            )
            result.errors.extend(examples_result.errors)
        
        # Validate subcategories
        if 'subcategories' not in category_dict:
            result.add_error("Subkategorien fehlen")
        elif not isinstance(category_dict['subcategories'], dict):
            result.add_error("Subkategorien müssen ein Dictionary sein")
        elif len(category_dict['subcategories']) < 2:
            result.add_error("Mindestens 2 Subkategorien erforderlich")
        
        if result.has_errors():
            result.is_valid = False
        
        return result
    
    @staticmethod
    def validate_codebook_complete(codebook_dict: Dict) -> ValidationResult:
        """
        Validates that codebook has all required components.
        
        Args:
            codebook_dict: Codebook dictionary to validate
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(True)
        
        # Validate research question
        if 'forschungsfrage' not in codebook_dict:
            result.add_warning("Forschungsfrage fehlt")
        else:
            question_result = CodebookValidator.validate_research_question(
                codebook_dict['forschungsfrage']
            )
            result.warnings.extend(question_result.warnings)
        
        # Validate categories
        if 'deduktive_kategorien' not in codebook_dict:
            result.add_error("Kategorien fehlen")
        elif not isinstance(codebook_dict['deduktive_kategorien'], dict):
            result.add_error("Kategorien müssen ein Dictionary sein")
        elif len(codebook_dict['deduktive_kategorien']) == 0:
            result.add_error("Mindestens eine Kategorie erforderlich")
        
        if result.has_errors():
            result.is_valid = False
        
        return result
