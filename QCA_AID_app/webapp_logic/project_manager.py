"""
Project Manager
===============
Manages project root directory and relative paths.
"""

from pathlib import Path
from typing import Optional, List
import json

from webapp_models.project_data import ProjectSettings
from webapp_logic.path_resolver import PathResolver


class ProjectManager:
    """Manages project root directory and relative paths"""
    
    # Default settings file name
    SETTINGS_FILE = ".qca-aid-project.json"
    
    def __init__(self, root_dir: Optional[Path] = None):
        """
        Initialize with optional root directory
        
        Args:
            root_dir: Project root directory (defaults to current working directory)
        """
        self.root_dir = Path(root_dir).resolve() if root_dir else Path.cwd()
        self.settings = ProjectSettings(project_root=self.root_dir)
        self.path_resolver = PathResolver(self.root_dir)
        
        # Try to load existing settings
        self.load_settings()
    
    def set_root_directory(self, path: Path) -> bool:
        """
        Set new project root directory
        
        Args:
            path: New project root directory
        
        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(path).resolve()
            
            # Validate that path exists and is a directory
            if not path.exists():
                print(f"Path does not exist: {path}")
                return False
            
            if not path.is_dir():
                print(f"Path is not a directory: {path}")
                return False
            
            # Update root directory
            self.root_dir = path
            self.settings.project_root = path
            self.settings.last_modified = self.settings.last_modified.__class__.now()
            
            # Update path resolver
            self.path_resolver = PathResolver(self.root_dir)
            
            # Save settings
            return self.save_settings()
            
        except Exception as e:
            print(f"Error setting root directory: {e}")
            return False
    
    def get_root_directory(self) -> Path:
        """
        Get current project root directory
        
        Returns:
            Project root directory path
        """
        return self.root_dir
    
    def resolve_relative_path(self, relative_path: str) -> Path:
        """
        Resolve relative path from project root
        
        Args:
            relative_path: Relative path string
        
        Returns:
            Absolute path resolved from project root
        """
        return self.path_resolver.resolve(relative_path)
    
    def get_config_path(self, filename: str = "QCA-AID-Codebook.json") -> Path:
        """
        Get path to config file relative to root
        
        Args:
            filename: Config filename (default: QCA-AID-Codebook.json)
        
        Returns:
            Absolute path to config file
        """
        if self.settings.last_config_file and self.settings.last_config_file.exists():
            return self.settings.last_config_file
        
        return self.root_dir / filename
    
    def get_codebook_path(self, filename: str = "QCA-AID-Codebook.json") -> Path:
        """
        Get path to codebook file relative to root
        
        Args:
            filename: Codebook filename (default: QCA-AID-Codebook.json)
        
        Returns:
            Absolute path to codebook file
        """
        if self.settings.last_codebook_file and self.settings.last_codebook_file.exists():
            return self.settings.last_codebook_file
        
        return self.root_dir / filename
    
    def get_input_dir(self) -> Path:
        """
        Get input directory path
        
        Returns:
            Absolute path to input directory
        """
        return self.root_dir / self.settings.input_dir_relative
    
    def get_output_dir(self) -> Path:
        """
        Get output directory path
        
        Returns:
            Absolute path to output directory
        """
        return self.root_dir / self.settings.output_dir_relative
    
    def search_config_files(self) -> List[Path]:
        """
        Search for config files in project root
        
        Returns:
            List of found config file paths
        """
        config_files = []
        
        try:
            # Search for JSON config files
            json_pattern = "QCA-AID-Codebook*.json"
            config_files.extend(self.root_dir.glob(json_pattern))
            
            # Search for XLSX config files
            xlsx_pattern = "QCA-AID-Codebook*.xlsx"
            config_files.extend(self.root_dir.glob(xlsx_pattern))
            
            # Sort by modification time (newest first)
            config_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            return config_files
            
        except Exception as e:
            print(f"Error searching for config files: {e}")
            return []
    
    def save_settings(self) -> bool:
        """
        Save project settings to .qca-aid-project.json
        
        Returns:
            True if successful, False otherwise
        """
        try:
            settings_path = self.root_dir / self.SETTINGS_FILE
            return self.settings.save(settings_path)
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def load_settings(self) -> bool:
        """
        Load project settings from .qca-aid-project.json
        
        Returns:
            True if successful, False otherwise
        """
        try:
            settings_path = self.root_dir / self.SETTINGS_FILE
            
            if not settings_path.exists():
                # No settings file, use defaults
                return False
            
            loaded_settings = ProjectSettings.load(settings_path)
            
            if loaded_settings:
                # Only use loaded settings if they match the current root directory
                # This prevents overriding the explicitly set root directory
                if loaded_settings.project_root == self.root_dir:
                    self.settings = loaded_settings
                    return True
                else:
                    # Settings file is for a different root directory
                    # Keep current settings and update the file
                    self.settings.project_root = self.root_dir
                    self.save_settings()
                    return False
            
            return False
            
        except Exception as e:
            print(f"Error loading settings: {e}")
            return False
    
    def update_last_config_file(self, file_path: Path) -> None:
        """
        Update last used config file
        
        Args:
            file_path: Path to config file
        """
        self.settings.last_config_file = Path(file_path).resolve()
        self.settings.last_modified = self.settings.last_modified.__class__.now()
        self.save_settings()
    
    def update_last_codebook_file(self, file_path: Path) -> None:
        """
        Update last used codebook file
        
        Args:
            file_path: Path to codebook file
        """
        self.settings.last_codebook_file = Path(file_path).resolve()
        self.settings.last_modified = self.settings.last_modified.__class__.now()
        self.save_settings()
    
    def ensure_directories(self) -> bool:
        """
        Ensure input and output directories exist
        
        Returns:
            True if successful, False otherwise
        """
        try:
            input_dir = self.get_input_dir()
            output_dir = self.get_output_dir()
            
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            return True
            
        except Exception as e:
            print(f"Error creating directories: {e}")
            return False
    
    def get_relative_path(self, absolute_path: Path) -> Path:
        """
        Get path relative to project root
        
        Args:
            absolute_path: Absolute path
        
        Returns:
            Path relative to project root, or absolute path if not within project
        """
        return self.path_resolver.make_relative(absolute_path)
    
    def validate_path(self, path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate a path
        
        Args:
            path: Path to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.path_resolver.validate_path(path)
    
    def check_write_permissions(self, path: Path) -> bool:
        """
        Check if path has write permissions
        
        Args:
            path: Path to check
        
        Returns:
            True if writable, False otherwise
        """
        return self.path_resolver.check_write_permissions(path)
