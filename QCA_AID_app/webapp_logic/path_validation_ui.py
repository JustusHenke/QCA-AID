"""
Path Validation UI Helpers
===========================
UI helper functions for displaying path validation feedback in Streamlit.

Requirements: 6.1, 6.2, 6.3, 6.4, 8.4, 8.5
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Tuple

from webapp_logic.path_resolver import PathResolver, PathValidationResult


def display_path_validation_feedback(validation_result: PathValidationResult, 
                                     show_success: bool = False) -> None:
    """
    Display validation feedback in Streamlit UI
    
    Requirement 6.1: Display specific error messages for invalid paths
    Requirement 8.4: Visual feedback for path validation
    
    Args:
        validation_result: PathValidationResult from validation
        show_success: Whether to show success message for valid paths
    """
    if not validation_result.is_valid:
        # Show error message
        st.error(f"❌ {validation_result.error_message}")
        
        # Show suggestions if available
        if validation_result.has_suggestions():
            with st.expander("💡 Vorschläge", expanded=True):
                for suggestion in validation_result.suggestions:
                    st.info(suggestion)
    
    elif validation_result.has_warnings():
        # Show warnings for valid but concerning paths
        for warning in validation_result.warnings:
            st.warning(f"⚠️ {warning}")
        
        # Show suggestions if available
        if validation_result.has_suggestions():
            for suggestion in validation_result.suggestions:
                st.info(f"💡 {suggestion}")
    
    elif show_success:
        st.success("✅ Pfad ist gültig")


def validate_and_display_path(path: str, path_resolver: PathResolver, 
                              field_name: str = "Pfad",
                              check_writable: bool = False,
                              show_success: bool = False) -> bool:
    """
    Validate path and display feedback in UI
    
    Requirement 8.4: Real-time path validation with visual feedback
    
    Args:
        path: Path string to validate
        path_resolver: PathResolver instance
        field_name: Name of the field for messages
        check_writable: Whether to check write permissions
        show_success: Whether to show success message
    
    Returns:
        True if valid, False otherwise
    """
    if not path or not path.strip():
        return False
    
    try:
        path_obj = Path(path)
        result = path_resolver.validate_path_detailed(path_obj, check_writable)
        display_path_validation_feedback(result, show_success)
        return result.is_valid
    except Exception as e:
        st.error(f"❌ Fehler bei Validierung: {str(e)}")
        return False


def offer_directory_creation(path: Path, path_resolver: PathResolver) -> bool:
    """
    Offer to create directory if it doesn't exist
    
    Requirement 6.3: WHEN ein Verzeichnis nicht existiert 
                    THEN das System SHALL anbieten, das Verzeichnis zu erstellen
    
    Args:
        path: Directory path
        path_resolver: PathResolver instance
    
    Returns:
        True if directory exists or was created, False otherwise
    """
    path_obj = Path(path)
    
    if path_obj.exists():
        return True
    
    # Validate that we can create the directory
    result = path_resolver.validate_directory_path(path_obj, auto_create=True)
    
    if not result.is_valid:
        display_path_validation_feedback(result)
        return False
    
    # Show creation option
    st.warning(f"⚠️ Verzeichnis existiert nicht: {path_obj}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📁 Verzeichnis erstellen", key=f"create_dir_{hash(str(path_obj))}"):
            success, error = path_resolver.create_directory(path_obj)
            
            if success:
                st.success(f"✅ Verzeichnis erstellt: {path_obj}")
                st.rerun()
                return True
            else:
                st.error(f"❌ {error}")
                return False
    
    with col2:
        if st.button("Abbrechen", key=f"cancel_create_{hash(str(path_obj))}"):
            return False
    
    return False


def display_path_with_tooltip(path: Path, max_length: int = 50) -> None:
    """
    Display path with truncation and full path in tooltip
    
    Requirement 8.3: WHEN ein langer Dateipfad angezeigt wird 
                    THEN das System SHALL diesen mit Ellipsis (...) kürzen und 
                    den vollständigen Pfad als Tooltip anzeigen
    
    Args:
        path: Path to display
        max_length: Maximum display length before truncation
    """
    path_str = str(path)
    
    if len(path_str) <= max_length:
        st.code(path_str, language=None)
    else:
        # Truncate path
        from webapp_logic.path_resolver import PathResolver
        # Create temporary resolver just for truncation
        temp_resolver = PathResolver(Path.cwd())
        truncated = temp_resolver.truncate_path(path, max_length)
        
        # Display with tooltip
        st.markdown(
            f'<div title="{path_str}"><code>{truncated}</code></div>',
            unsafe_allow_html=True
        )


def display_file_operation_success(file_path: Path, operation: str = "geladen") -> None:
    """
    Display success confirmation with file information
    
    Requirement 8.5: WHEN eine Datei erfolgreich geladen wird 
                    THEN das System SHALL eine Bestätigungsmeldung mit Dateiinformationen anzeigen
    
    Args:
        file_path: Path to the file
        operation: Operation performed (e.g., "geladen", "gespeichert")
    """
    path_obj = Path(file_path)
    
    # Get file info
    file_info = []
    
    if path_obj.exists():
        # File size
        size_bytes = path_obj.stat().st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} Bytes"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        
        file_info.append(f"Größe: {size_str}")
        
        # Last modified
        import datetime
        mtime = path_obj.stat().st_mtime
        mod_time = datetime.datetime.fromtimestamp(mtime)
        file_info.append(f"Geändert: {mod_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Display success message
    st.success(f"✅ Datei erfolgreich {operation}: **{path_obj.name}**")
    
    # Display file info
    if file_info:
        st.caption(" | ".join(file_info))


def get_alternative_paths_ui(path: Path, path_resolver: PathResolver) -> Optional[Path]:
    """
    Display alternative path suggestions and let user select one
    
    Requirement 6.2: Suggest alternative paths for invalid paths
    Requirement 6.4: Suggest alternative locations for permission issues
    
    Args:
        path: Original invalid path
        path_resolver: PathResolver instance
    
    Returns:
        Selected alternative path or None
    """
    alternatives = path_resolver.suggest_alternatives(path)
    
    if not alternatives:
        st.info("💡 Keine alternativen Pfade verfügbar")
        return None
    
    st.markdown("**Alternative Speicherorte:**")
    
    selected = None
    for i, alt_path in enumerate(alternatives):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.code(str(alt_path), language=None)
        
        with col2:
            if st.button("Wählen", key=f"select_alt_{i}_{hash(str(alt_path))}"):
                selected = alt_path
    
    return selected


def validate_path_input_realtime(path_input: str, path_resolver: PathResolver,
                                 field_name: str = "Pfad",
                                 check_writable: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Real-time validation for path input fields
    
    Requirement 8.4: Real-time path validation with visual feedback
    
    Args:
        path_input: Path string from input field
        path_resolver: PathResolver instance
        field_name: Name of the field
        check_writable: Whether to check write permissions
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path_input or not path_input.strip():
        return False, f"{field_name} darf nicht leer sein"
    
    try:
        path_obj = Path(path_input)
        result = path_resolver.validate_path_detailed(path_obj, check_writable)
        
        if not result.is_valid:
            return False, result.error_message
        
        return True, None
        
    except Exception as e:
        return False, f"Ungültiger Pfad: {str(e)}"
