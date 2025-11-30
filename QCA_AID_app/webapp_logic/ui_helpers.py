"""
UI Helpers
==========
Helper functions for UI improvements in file operations.

Requirements: 8.1, 8.2, 8.3, 8.5
"""

import streamlit as st
from pathlib import Path
from typing import Optional
from datetime import datetime


def render_path_input_with_browser(
    label: str,
    value: str,
    placeholder: str,
    key: str,
    browser_key: str,
    on_browse_callback,
    help_text: Optional[str] = None,
    max_display_length: int = 50
) -> str:
    """
    Render path input field with file browser button and truncation.
    
    Requirements:
    - 8.1: File browser icon buttons next to path inputs
    - 8.2: Tooltips for browser buttons
    - 8.3: Path truncation with ellipsis for long paths
    
    Args:
        label: Label for the input field
        value: Current value
        placeholder: Placeholder text
        key: Unique key for the input
        browser_key: Unique key for the browser button
        on_browse_callback: Function to call when browse button clicked
        help_text: Optional help text
        max_display_length: Maximum length before truncation
    
    Returns:
        The input value (possibly updated)
    """
    col_path, col_browse = st.columns([4, 1])
    
    with col_path:
        # Truncate display value if too long
        display_value = value
        if len(value) > max_display_length:
            display_value = truncate_path_for_display(value, max_display_length)
        
        # Show full path in help text if truncated
        full_help = help_text or ""
        if len(value) > max_display_length:
            full_help = f"Vollständiger Pfad: {value}\n\n{full_help}" if full_help else f"Vollständiger Pfad: {value}"
        
        file_path = st.text_input(
            label,
            value=value,
            placeholder=placeholder,
            help=full_help,
            key=key
        )
    
    with col_browse:
        st.markdown("<br>", unsafe_allow_html=True)  # Align with input
        # Requirement 8.1, 8.2: Browser button with icon and tooltip
        if st.button("📁", key=browser_key, help="Datei durchsuchen"):
            result = on_browse_callback()
            if result:
                return str(result)
    
    return file_path


def truncate_path_for_display(path: str, max_length: int = 50) -> str:
    """
    Truncate long path with ellipsis for display.
    
    Requirement 8.3: WHEN ein langer Dateipfad angezeigt wird 
                    THEN das System SHALL diesen mit Ellipsis (...) kürzen
    
    Args:
        path: Path to truncate
        max_length: Maximum display length
    
    Returns:
        Truncated path string with ellipsis if needed
    """
    if len(path) <= max_length:
        return path
    
    path_obj = Path(path)
    parts = path_obj.parts
    
    # If it's just a filename that's too long
    if len(parts) == 1:
        filename = parts[0]
        if len(filename) > max_length:
            # Truncate filename in the middle to preserve extension
            if '.' in filename:
                name, ext = filename.rsplit('.', 1)
                available = max_length - len(ext) - 4  # -4 for "..." and "."
                if available > 0:
                    return f"{name[:available]}...{ext}"
            return f"{filename[:max_length-3]}..."
        return filename
    
    # Try to keep filename and some parent directories
    filename = parts[-1]
    
    # If filename alone is too long, truncate it
    if len(filename) > max_length - 10:
        if '.' in filename:
            name, ext = filename.rsplit('.', 1)
            available = max_length - len(ext) - 13  # -13 for ".../" and "..." and "."
            if available > 0:
                return f".../{name[:available]}...{ext}"
        return f".../{filename[:max_length-13]}..."
    
    # Build path from end, adding parts until we hit limit
    result_parts = [filename]
    remaining_length = max_length - len(filename) - 4  # -4 for "..." and separator
    
    for part in reversed(parts[:-1]):
        # +1 for separator (/ or \)
        if len(part) + 1 <= remaining_length:
            result_parts.insert(0, part)
            remaining_length -= len(part) + 1
        else:
            break
    
    # Add ellipsis at the beginning if we didn't include all parts
    if len(result_parts) < len(parts):
        result_parts.insert(0, "...")
    
    # Join with appropriate separator
    separator = '\\' if '\\' in path else '/'
    result = separator.join(result_parts)
    
    # Final check: if still too long, truncate more aggressively
    if len(result) > max_length:
        # Keep just the filename with ellipsis
        if len(filename) <= max_length - 4:
            return f"...{separator}{filename}"
        else:
            # Truncate the filename itself
            if '.' in filename:
                name, ext = filename.rsplit('.', 1)
                available = max_length - len(ext) - 5  # -5 for "..." + "." + separator
                if available > 0:
                    return f"...{separator}{name[:available]}...{ext}"
            return f"...{separator}{filename[:max_length-7]}..."
    
    return result


def show_file_operation_success(
    operation: str,
    file_path: Path,
    additional_info: Optional[dict] = None
):
    """
    Show success confirmation message with file info.
    
    Requirement 8.5: WHEN eine Datei erfolgreich geladen wird 
                    THEN das System SHALL eine Bestätigungsmeldung mit Dateiinformationen anzeigen
    
    Args:
        operation: Operation performed (e.g., "geladen", "gespeichert")
        file_path: Path to the file
        additional_info: Optional dictionary with additional info to display
    """
    file_path = Path(file_path)
    
    # Get file info
    file_info = []
    
    # File name
    file_info.append(f"**Datei:** {file_path.name}")
    
    # File size if exists
    if file_path.exists():
        size_bytes = file_path.stat().st_size
        size_str = format_file_size(size_bytes)
        file_info.append(f"**Größe:** {size_str}")
        
        # Last modified
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        file_info.append(f"**Geändert:** {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # File format
    if file_path.suffix:
        file_info.append(f"**Format:** {file_path.suffix.upper().replace('.', '')}")
    
    # Add additional info if provided
    if additional_info:
        for key, value in additional_info.items():
            file_info.append(f"**{key}:** {value}")
    
    # Show success message with info
    success_msg = f"✅ Datei erfolgreich {operation}!\n\n" + "\n".join(file_info)
    st.success(success_msg)


def show_file_operation_error(
    operation: str,
    file_path: Path,
    error_message: str,
    suggestions: Optional[list] = None
):
    """
    Show error message for file operations with suggestions.
    
    Args:
        operation: Operation attempted (e.g., "laden", "speichern")
        file_path: Path to the file
        error_message: Error message
        suggestions: Optional list of suggestions
    """
    file_path = Path(file_path)
    
    error_msg = f"❌ Fehler beim {operation} der Datei:\n\n"
    error_msg += f"**Datei:** {file_path.name}\n"
    error_msg += f"**Fehler:** {error_message}"
    
    st.error(error_msg)
    
    if suggestions:
        st.info("**Vorschläge:**\n" + "\n".join(f"• {s}" for s in suggestions))


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def render_path_with_tooltip(path: str, max_length: int = 50) -> None:
    """
    Render path with truncation and full path in tooltip.
    
    Requirements:
    - 8.3: Path truncation with ellipsis
    - Show full path in tooltip on hover
    
    Args:
        path: Path to display
        max_length: Maximum display length
    """
    if len(path) <= max_length:
        st.code(path, language=None)
    else:
        truncated = truncate_path_for_display(path, max_length)
        # Use markdown with title attribute for tooltip
        st.markdown(
            f'<code title="{path}">{truncated}</code>',
            unsafe_allow_html=True
        )


def show_real_time_path_validation(
    path: str,
    path_resolver,
    check_writable: bool = False
) -> bool:
    """
    Show real-time path validation feedback.
    
    Requirement 8.4: WHEN ein Benutzer einen Pfad manuell eingibt 
                    THEN das System SHALL diesen in Echtzeit validieren und visuelles Feedback geben
    
    Args:
        path: Path to validate
        path_resolver: PathResolver instance
        check_writable: Whether to check write permissions
    
    Returns:
        True if valid, False otherwise
    """
    if not path or not path.strip():
        return True  # Empty is OK (will use default)
    
    from pathlib import Path
    path_obj = Path(path)
    
    # Validate path
    result = path_resolver.validate_path_detailed(path_obj, check_writable=check_writable)
    
    if not result.is_valid:
        st.error(f"❌ {result.error_message}")
        
        if result.suggestions:
            with st.expander("💡 Vorschläge anzeigen"):
                for suggestion in result.suggestions:
                    st.info(f"• {suggestion}")
        
        return False
    
    if result.has_warnings():
        for warning in result.warnings:
            st.warning(f"⚠️ {warning}")
    
    if result.has_suggestions():
        with st.expander("💡 Hinweise anzeigen"):
            for suggestion in result.suggestions:
                st.info(f"• {suggestion}")
    
    return True
