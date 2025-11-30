"""
File Manager
============
Manages file system operations for the QCA-AID Webapp.
Provides secure file listing, directory management, and file preview functionality.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import streamlit as st
import hashlib

from webapp_models.file_info import FileInfo


class FileManager:
    """
    Manages file system operations with security validations.
    
    Provides methods for:
    - Listing files with metadata extraction
    - Directory creation and validation
    - File preview with character limits
    - Path validation against traversal attacks
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize FileManager.
        
        Args:
            base_dir: Base directory for file operations (defaults to current directory)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.base_dir = self.base_dir.resolve()
    
    def validate_path(self, path: str, base_dir: Optional[str] = None) -> bool:
        """
        Validates that a path is within the allowed base directory.
        Prevents path traversal attacks (e.g., ../../etc/passwd).
        
        Args:
            path: Path to validate
            base_dir: Base directory to validate against (defaults to self.base_dir)
            
        Returns:
            bool: True if path is valid and within base_dir
        """
        if base_dir is None:
            base_dir = self.base_dir
        else:
            base_dir = Path(base_dir).resolve()
        
        try:
            # Resolve the path to its absolute form
            abs_path = Path(path).resolve()
            abs_base = Path(base_dir).resolve()
            
            # Check if the resolved path starts with the base directory
            return abs_path.is_relative_to(abs_base)
        except (ValueError, OSError):
            # Invalid path or permission error
            return False
    
    def ensure_directory(self, path: str) -> Tuple[bool, Optional[str]]:
        """
        Ensures a directory exists, creating it if necessary.
        
        Requirement 7.5: WHEN der INPUT_DIR nicht existiert 
                        THEN das System SHALL den Ordner automatisch erstellen
        Requirement 12.5: Implementiere automatische Verzeichniserstellung
        
        Args:
            path: Directory path to ensure exists
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        # Validate path
        if not self.validate_path(path):
            return False, f"Ungültiger Pfad: {path} liegt außerhalb des erlaubten Verzeichnisses"
        
        dir_path = Path(path)
        
        try:
            # Create directory if it doesn't exist
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                return True, None
            elif dir_path.is_dir():
                return True, None
            else:
                return False, f"Pfad existiert, ist aber kein Verzeichnis: {path}"
        except PermissionError as e:
            return False, f"Keine Berechtigung zum Erstellen des Verzeichnisses {path}: {str(e)}"
        except OSError as e:
            return False, f"Fehler beim Erstellen des Verzeichnisses {path}: {str(e)}"
    
    def _get_directory_hash(self, directory: str) -> str:
        """
        Computes hash of directory state for cache invalidation.
        Uses modification times of all files.
        
        Args:
            directory: Directory path
            
        Returns:
            str: Hash representing directory state
        """
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return "empty"
            
            # Collect all file modification times
            mtimes = []
            for item in dir_path.iterdir():
                if item.is_file():
                    mtimes.append(str(item.stat().st_mtime))
            
            # Hash the concatenated mtimes
            state = '|'.join(sorted(mtimes))
            return hashlib.md5(state.encode()).hexdigest()
        except:
            # Fallback to timestamp
            return str(datetime.now().timestamp())
    
    @st.cache_data(ttl=60, show_spinner=False)
    def _list_files_cached(
        _self,
        directory: str,
        extensions_tuple: Optional[Tuple[str, ...]],
        recursive: bool,
        dir_hash: str
    ) -> List[Dict]:
        """
        Cached file listing helper.
        
        Performance Optimization: Caches file listings for 60 seconds.
        Uses directory hash to invalidate cache when files change.
        
        Args:
            directory: Directory to list
            extensions_tuple: Tuple of extensions (tuple for hashability)
            recursive: Whether to search recursively
            dir_hash: Hash of directory state for cache invalidation
            
        Returns:
            List[Dict]: List of file info dictionaries
        """
        # Convert tuple back to list
        extensions = list(extensions_tuple) if extensions_tuple else None
        
        # Validate path
        if not _self.validate_path(directory):
            raise ValueError(f"Invalid directory: {directory} is outside allowed directory")
        
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise OSError(f"Directory does not exist: {directory}")
        
        if not dir_path.is_dir():
            raise OSError(f"Path is not a directory: {directory}")
        
        files = []
        
        try:
            # Get file pattern
            if recursive:
                pattern = '**/*'
            else:
                pattern = '*'
            
            # Iterate through files
            for file_path in dir_path.glob(pattern):
                # Skip directories
                if not file_path.is_file():
                    continue
                
                # Filter by extension if specified
                if extensions:
                    if file_path.suffix.lower() not in [ext.lower() for ext in extensions]:
                        continue
                
                # Create FileInfo dict (serializable for cache)
                try:
                    file_info = FileInfo.from_path(str(file_path))
                    files.append({
                        'path': file_info.path,
                        'name': file_info.name,
                        'size': file_info.size,
                        'modified': file_info.modified.isoformat(),
                        'extension': file_info.extension
                    })
                except (OSError, PermissionError):
                    # Skip files that can't be accessed
                    continue
            
            # Sort by modification time (newest first)
            files.sort(key=lambda f: f['modified'], reverse=True)
            
        except PermissionError as e:
            raise OSError(f"Permission denied accessing directory {directory}: {str(e)}")
        
        return files
    
    def list_files(
        self, 
        directory: str, 
        extensions: Optional[List[str]] = None,
        recursive: bool = False,
        use_cache: bool = True
    ) -> List[FileInfo]:
        """
        Lists files in a directory with metadata extraction and caching.
        
        Performance Optimization: Uses caching with 60s TTL to avoid repeated filesystem access.
        
        Args:
            directory: Directory to list files from
            extensions: Optional list of file extensions to filter (e.g., ['.xlsx', '.json'])
            recursive: Whether to search subdirectories recursively
            use_cache: Whether to use caching (default: True)
            
        Returns:
            List[FileInfo]: List of FileInfo objects with metadata
            
        Raises:
            ValueError: If path validation fails
            OSError: If directory doesn't exist or can't be read
        """
        if use_cache:
            # Get directory hash for cache invalidation
            dir_hash = self._get_directory_hash(directory)
            
            # Convert extensions to tuple for hashability
            extensions_tuple = tuple(extensions) if extensions else None
            
            # Use cached version
            file_dicts = self._list_files_cached(
                directory,
                extensions_tuple,
                recursive,
                dir_hash
            )
            
            # Convert dicts back to FileInfo objects
            files = []
            for file_dict in file_dicts:
                file_info = FileInfo(
                    path=file_dict['path'],
                    name=file_dict['name'],
                    size=file_dict['size'],
                    modified=datetime.fromisoformat(file_dict['modified']),
                    extension=file_dict['extension']
                )
                files.append(file_info)
            
            return files
        else:
            # Non-cached version (original implementation)
            # Validate path
            if not self.validate_path(directory):
                raise ValueError(f"Invalid directory: {directory} is outside allowed directory")
            
            dir_path = Path(directory)
            
            if not dir_path.exists():
                raise OSError(f"Directory does not exist: {directory}")
            
            if not dir_path.is_dir():
                raise OSError(f"Path is not a directory: {directory}")
            
            files = []
            
            try:
                # Get file pattern
                if recursive:
                    pattern = '**/*'
                else:
                    pattern = '*'
                
                # Iterate through files
                for file_path in dir_path.glob(pattern):
                    # Skip directories
                    if not file_path.is_file():
                        continue
                    
                    # Filter by extension if specified
                    if extensions:
                        if file_path.suffix.lower() not in [ext.lower() for ext in extensions]:
                            continue
                    
                    # Create FileInfo object
                    try:
                        file_info = FileInfo.from_path(str(file_path))
                        files.append(file_info)
                    except (OSError, PermissionError):
                        # Skip files that can't be accessed
                        continue
                
                # Sort by modification time (newest first)
                files.sort(key=lambda f: f.modified, reverse=True)
                
            except PermissionError as e:
                raise OSError(f"Permission denied accessing directory {directory}: {str(e)}")
            
            return files
    
    def get_file_info(self, file_path: str) -> FileInfo:
        """
        Gets detailed information about a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileInfo: File information object
            
        Raises:
            ValueError: If path validation fails
            OSError: If file doesn't exist or can't be accessed
        """
        # Validate path
        if not self.validate_path(file_path):
            raise ValueError(f"Invalid file path: {file_path} is outside allowed directory")
        
        path = Path(file_path)
        
        if not path.exists():
            raise OSError(f"File does not exist: {file_path}")
        
        if not path.is_file():
            raise OSError(f"Path is not a file: {file_path}")
        
        try:
            return FileInfo.from_path(str(path))
        except (OSError, PermissionError) as e:
            raise OSError(f"Failed to get file info for {file_path}: {str(e)}")
    
    @st.cache_data(ttl=120, show_spinner=False)
    def _get_file_preview_cached(
        _self,
        file_path: str,
        max_chars: int,
        encoding: str,
        file_hash: str
    ) -> str:
        """
        Cached file preview helper.
        
        Performance Optimization: Caches file previews for 2 minutes.
        Uses file hash to invalidate cache when file changes.
        
        Args:
            file_path: Path to file
            max_chars: Maximum characters to read
            encoding: File encoding
            file_hash: Hash of file for cache invalidation
            
        Returns:
            str: File preview content
        """
        path = Path(file_path)
        
        try:
            # Lazy loading: Only read the required amount
            with open(path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read(max_chars)
                
                # Add truncation indicator if file is longer
                if len(content) == max_chars:
                    # Check if there's more content
                    next_char = f.read(1)
                    if next_char:
                        content += "\n\n[... truncated ...]"
                
                return content
        except UnicodeDecodeError:
            # Try with different encoding or return binary indicator
            return "[Binary file - preview not available]"
        except (OSError, PermissionError) as e:
            raise OSError(f"Failed to read file {file_path}: {str(e)}")
    
    def _get_file_hash_quick(self, file_path: str) -> str:
        """
        Computes quick hash of file for cache invalidation.
        Uses file size and mtime instead of reading entire file.
        
        Args:
            file_path: Path to file
            
        Returns:
            str: Hash representing file state
        """
        try:
            stat = Path(file_path).stat()
            return f"{stat.st_size}_{stat.st_mtime}"
        except:
            return str(datetime.now().timestamp())
    
    def get_file_preview(
        self, 
        file_path: str, 
        max_chars: int = 500,
        encoding: str = 'utf-8',
        use_cache: bool = True
    ) -> str:
        """
        Gets a preview of a text file's content with lazy loading and caching.
        
        Performance Optimization: 
        - Uses lazy loading to only read required amount
        - Caches previews for 2 minutes to avoid repeated reads
        
        Args:
            file_path: Path to the file
            max_chars: Maximum number of characters to read
            encoding: File encoding (default: utf-8)
            use_cache: Whether to use caching (default: True)
            
        Returns:
            str: Preview of file content (truncated to max_chars)
            
        Raises:
            ValueError: If path validation fails or max_chars is invalid
            OSError: If file doesn't exist or can't be read
        """
        # Validate path
        if not self.validate_path(file_path):
            raise ValueError(f"Invalid file path: {file_path} is outside allowed directory")
        
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        
        path = Path(file_path)
        
        if not path.exists():
            raise OSError(f"File does not exist: {file_path}")
        
        if not path.is_file():
            raise OSError(f"Path is not a file: {file_path}")
        
        if use_cache:
            # Get file hash for cache invalidation
            file_hash = self._get_file_hash_quick(file_path)
            
            # Use cached version
            return self._get_file_preview_cached(
                file_path,
                max_chars,
                encoding,
                file_hash
            )
        else:
            # Non-cached version
            try:
                with open(path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read(max_chars)
                    
                    # Add truncation indicator if file is longer
                    if len(content) == max_chars:
                        # Check if there's more content
                        next_char = f.read(1)
                        if next_char:
                            content += "\n\n[... truncated ...]"
                    
                    return content
            except UnicodeDecodeError:
                # Try with different encoding or return binary indicator
                return "[Binary file - preview not available]"
            except (OSError, PermissionError) as e:
                raise OSError(f"Failed to read file {file_path}: {str(e)}")
    
    def paginate_files(
        self,
        files: List[FileInfo],
        page: int = 1,
        page_size: int = 10
    ) -> Tuple[List[FileInfo], int, int]:
        """
        Paginates a list of files.
        
        Performance Optimization: Enables pagination for long file lists.
        
        Args:
            files: List of FileInfo objects
            page: Current page number (1-indexed)
            page_size: Number of items per page
            
        Returns:
            Tuple[List[FileInfo], int, int]: (paginated_files, total_pages, total_items)
        """
        total_items = len(files)
        total_pages = (total_items + page_size - 1) // page_size  # Ceiling division
        
        # Validate page number
        if page < 1:
            page = 1
        elif page > total_pages and total_pages > 0:
            page = total_pages
        
        # Calculate slice indices
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Return paginated slice
        paginated_files = files[start_idx:end_idx]
        
        return paginated_files, total_pages, total_items
    
    def get_directory_stats(self, directory: str) -> Dict:
        """
        Gets statistics about a directory.
        
        Args:
            directory: Directory to analyze
            
        Returns:
            Dict: Statistics including file count, total size, etc.
            
        Raises:
            ValueError: If path validation fails
            OSError: If directory doesn't exist or can't be accessed
        """
        # Validate path
        if not self.validate_path(directory):
            raise ValueError(f"Invalid directory: {directory} is outside allowed directory")
        
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise OSError(f"Directory does not exist: {directory}")
        
        if not dir_path.is_dir():
            raise OSError(f"Path is not a directory: {directory}")
        
        stats = {
            'total_files': 0,
            'total_size': 0,
            'file_types': {},
            'last_modified': None
        }
        
        try:
            for file_path in dir_path.glob('*'):
                if file_path.is_file():
                    stats['total_files'] += 1
                    
                    # Add to total size
                    try:
                        size = file_path.stat().st_size
                        stats['total_size'] += size
                        
                        # Track file types
                        ext = file_path.suffix.lower() or 'no_extension'
                        stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
                        
                        # Track last modified
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if stats['last_modified'] is None or mtime > stats['last_modified']:
                            stats['last_modified'] = mtime
                    except (OSError, PermissionError):
                        continue
        except PermissionError as e:
            raise OSError(f"Permission denied accessing directory {directory}: {str(e)}")
        
        return stats
