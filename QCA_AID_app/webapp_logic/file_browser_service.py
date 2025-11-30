"""
File Browser Service
====================
Service for opening native file browser dialogs using tkinter.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import tkinter as tk
from tkinter import filedialog


class FileBrowserService:
    """Service for opening native file browser dialogs"""
    
    @staticmethod
    def _create_root() -> tk.Tk:
        """
        Create and configure tkinter root window
        
        Returns:
            Configured tkinter root window
        """
        root = tk.Tk()
        root.withdraw()  # Hide main window
        root.wm_attributes('-topmost', 1)  # Bring to front
        return root
    
    @staticmethod
    def open_file_dialog(
        title: str = "Select File",
        initial_dir: Optional[Path] = None,
        file_types: Optional[List[Tuple[str, str]]] = None
    ) -> Optional[Path]:
        """
        Open native file selection dialog
        
        Args:
            title: Dialog title
            initial_dir: Initial directory to open
            file_types: List of (description, extension) tuples
                       e.g. [("JSON files", "*.json"), ("All files", "*.*")]
        
        Returns:
            Selected file path or None if cancelled
        """
        try:
            root = FileBrowserService._create_root()
            
            # Set default file types if none provided
            if file_types is None:
                file_types = [
                    ("JSON files", "*.json"),
                    ("Excel files", "*.xlsx"),
                    ("All files", "*.*")
                ]
            
            # Convert initial_dir to string if provided
            initial_dir_str = str(initial_dir) if initial_dir else None
            
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title=title,
                initialdir=initial_dir_str,
                filetypes=file_types
            )
            
            root.destroy()
            
            # Return Path object if file was selected, None otherwise
            return Path(file_path) if file_path else None
            
        except Exception as e:
            print(f"Error opening file dialog: {e}")
            return None
    
    @staticmethod
    def open_directory_dialog(
        title: str = "Select Directory",
        initial_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Open native directory selection dialog
        
        Args:
            title: Dialog title
            initial_dir: Initial directory to open
        
        Returns:
            Selected directory path or None if cancelled
        """
        try:
            root = FileBrowserService._create_root()
            
            # Convert initial_dir to string if provided
            initial_dir_str = str(initial_dir) if initial_dir else None
            
            # Open directory dialog
            dir_path = filedialog.askdirectory(
                title=title,
                initialdir=initial_dir_str
            )
            
            root.destroy()
            
            # Return Path object if directory was selected, None otherwise
            return Path(dir_path) if dir_path else None
            
        except Exception as e:
            print(f"Error opening directory dialog: {e}")
            return None
    
    @staticmethod
    def save_file_dialog(
        title: str = "Save File",
        initial_dir: Optional[Path] = None,
        default_filename: str = "",
        file_types: Optional[List[Tuple[str, str]]] = None
    ) -> Optional[Path]:
        """
        Open native save file dialog
        
        Args:
            title: Dialog title
            initial_dir: Initial directory to open
            default_filename: Default filename to suggest
            file_types: List of (description, extension) tuples
                       e.g. [("JSON files", "*.json"), ("All files", "*.*")]
        
        Returns:
            Selected save path or None if cancelled
        """
        try:
            root = FileBrowserService._create_root()
            
            # Set default file types if none provided
            if file_types is None:
                file_types = [
                    ("JSON files", "*.json"),
                    ("Excel files", "*.xlsx"),
                    ("All files", "*.*")
                ]
            
            # Convert initial_dir to string if provided
            initial_dir_str = str(initial_dir) if initial_dir else None
            
            # Determine default extension from file types
            default_extension = ""
            if file_types and len(file_types) > 0:
                # Extract extension from first file type (e.g., "*.json" -> ".json")
                ext_pattern = file_types[0][1]
                if ext_pattern.startswith("*."):
                    default_extension = ext_pattern[1:]  # Remove "*"
            
            # Open save dialog
            file_path = filedialog.asksaveasfilename(
                title=title,
                initialdir=initial_dir_str,
                initialfile=default_filename,
                defaultextension=default_extension,
                filetypes=file_types
            )
            
            root.destroy()
            
            # Return Path object if path was selected, None otherwise
            return Path(file_path) if file_path else None
            
        except Exception as e:
            print(f"Error opening save dialog: {e}")
            return None
    
    @staticmethod
    def open_config_file_dialog(initial_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Open file dialog specifically for config files
        
        Args:
            initial_dir: Initial directory to open
        
        Returns:
            Selected config file path or None if cancelled
        """
        return FileBrowserService.open_file_dialog(
            title="Konfiguration laden",
            initial_dir=initial_dir,
            file_types=[
                ("JSON files", "*.json"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
        )
    
    @staticmethod
    def open_codebook_file_dialog(initial_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Open file dialog specifically for codebook files
        
        Args:
            initial_dir: Initial directory to open
        
        Returns:
            Selected codebook file path or None if cancelled
        """
        return FileBrowserService.open_file_dialog(
            title="Codebook laden",
            initial_dir=initial_dir,
            file_types=[
                ("JSON files", "*.json"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
        )
    
    @staticmethod
    def save_config_file_dialog(
        initial_dir: Optional[Path] = None,
        default_filename: str = "QCA-AID-Codebook.json"
    ) -> Optional[Path]:
        """
        Open save dialog specifically for config files
        
        Args:
            initial_dir: Initial directory to open
            default_filename: Default filename to suggest
        
        Returns:
            Selected save path or None if cancelled
        """
        return FileBrowserService.save_file_dialog(
            title="Konfiguration speichern",
            initial_dir=initial_dir,
            default_filename=default_filename,
            file_types=[
                ("JSON files", "*.json"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
        )
    
    @staticmethod
    def save_codebook_file_dialog(
        initial_dir: Optional[Path] = None,
        default_filename: str = "QCA-AID-Codebook.json"
    ) -> Optional[Path]:
        """
        Open save dialog specifically for codebook files
        
        Args:
            initial_dir: Initial directory to open
            default_filename: Default filename to suggest
        
        Returns:
            Selected save path or None if cancelled
        """
        return FileBrowserService.save_file_dialog(
            title="Codebook speichern",
            initial_dir=initial_dir,
            default_filename=default_filename,
            file_types=[
                ("JSON files", "*.json"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
        )
    
    @staticmethod
    def open_project_directory_dialog(initial_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Open directory dialog specifically for project root selection
        
        Args:
            initial_dir: Initial directory to open
        
        Returns:
            Selected project directory path or None if cancelled
        """
        return FileBrowserService.open_directory_dialog(
            title="Projekt-Verzeichnis auswählen",
            initial_dir=initial_dir
        )
