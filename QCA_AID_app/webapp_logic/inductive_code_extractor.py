"""
Inductive Code Extractor
=========================
Extracts inductive codes from QCA-AID analysis output files.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from webapp_models.codebook_data import CategoryData
from webapp_models.inductive_code_data import InductiveCodeData


class InductiveCodeExtractor:
    """Extracts inductive codes from analysis output files"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize with output directory
        
        Args:
            output_dir: Path to the output directory containing analysis files
        """
        self.output_dir = Path(output_dir)
        self.inductive_codebook_filename = "codebook_inductive.json"
    
    def has_inductive_codes_available(self) -> bool:
        """
        Check if inductive codes are available from any source
        
        Returns:
            True if inductive codes can be loaded, False otherwise
        """
        # Check for JSON codebook first (primary)
        if self.has_inductive_codebook():
            return True
        
        # Check for XLSX files with inductive codes (secondary)
        analysis_files = self.find_analysis_files()
        for file_path in analysis_files:
            if self.has_inductive_codes(file_path):
                return True
        
        return False
    
    def has_inductive_codebook(self) -> bool:
        """
        Check if inductive codebook JSON file exists
        
        Returns:
            True if codebook_inductive.json exists, False otherwise
        """
        codebook_path = self.output_dir / self.inductive_codebook_filename
        return codebook_path.exists()
    
    def get_inductive_codes(self) -> Dict[str, InductiveCodeData]:
        """
        Get inductive codes from the best available source.
        Tries JSON codebook first (primary), then XLSX files (secondary).
        
        Returns:
            Dictionary mapping category names to InductiveCodeData objects
        """
        # Try JSON codebook first (PRIMARY)
        if self.has_inductive_codebook():
            codes = self.load_inductive_codebook()
            if codes:
                return codes
        
        # Fallback to XLSX files (SECONDARY)
        analysis_files = self.find_analysis_files()
        for file_path in analysis_files:
            if self.has_inductive_codes(file_path):
                codes = self.extract_inductive_codes_from_xlsx(file_path)
                if codes:
                    return codes
        
        return {}
    
    def load_inductive_codebook(self) -> Dict[str, InductiveCodeData]:
        """
        Load inductive codes from codebook_inductive.json (PRIMARY METHOD)
        
        Returns:
            Dictionary mapping category names to InductiveCodeData objects
        """
        inductive_codes = {}
        codebook_path = self.output_dir / self.inductive_codebook_filename
        
        if not codebook_path.exists():
            return inductive_codes
        
        try:
            with open(codebook_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract metadata
            metadata = data.get('metadata', {})
            source_file = self.inductive_codebook_filename
            
            # Extract categories
            categories = data.get('categories', {})
            
            for category_name, category_data in categories.items():
                # Check if this is an inductive category
                development_type = category_data.get('development_type', '')
                
                # Only extract inductive categories
                if development_type.lower() in ['induktiv', 'inductive']:
                    # Parse subcategories
                    subcategories = category_data.get('subcategories', {})
                    
                    # Create CategoryData
                    cat_data = CategoryData(
                        name=category_name,
                        definition=category_data.get('definition', ''),
                        rules=category_data.get('rules', []),
                        examples=category_data.get('examples', []),
                        subcategories=subcategories,
                        added_date=category_data.get('added_date', datetime.now().strftime("%Y-%m-%d")),
                        modified_date=category_data.get('last_modified', datetime.now().strftime("%Y-%m-%d"))
                    )
                    
                    # Convert to InductiveCodeData
                    inductive_code = InductiveCodeData.from_category_data(
                        category=cat_data,
                        source_file=source_file,
                        frequency=None  # Not stored in codebook
                    )
                    
                    inductive_codes[category_name] = inductive_code
            
        except Exception as e:
            print(f"Error loading inductive codebook from {codebook_path}: {e}")
        
        return inductive_codes
    
    def find_analysis_files(self) -> List[Path]:
        """
        Find all analysis XLSX files in output directory (SECONDARY METHOD)
        
        Returns:
            List of Path objects for analysis files
        """
        if not self.output_dir.exists():
            return []
        
        analysis_files = []
        
        # Look for XLSX files that match the analysis pattern
        for file_path in self.output_dir.glob("*.xlsx"):
            # Check if it's an analysis file (not a codebook)
            if "Analysis" in file_path.stem or "QCA-AID" in file_path.stem:
                # Exclude files that are clearly codebooks
                if "Codebook" not in file_path.stem:
                    analysis_files.append(file_path)
        
        # Sort by modification time (newest first)
        analysis_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return analysis_files
    
    def has_inductive_codes(self, file_path: Path) -> bool:
        """
        Check if analysis file contains inductive codes
        
        Args:
            file_path: Path to the analysis file
            
        Returns:
            True if file contains inductive codes, False otherwise
        """
        try:
            # Try to read the Kategorien sheet
            df = pd.read_excel(file_path, sheet_name='Kategorien')
            
            # Check if there's a 'Typ' column indicating code type
            if 'Typ' in df.columns:
                # Check for 'induktiv' or 'inductive' type
                has_inductive = df['Typ'].str.lower().str.contains('induktiv|inductive', na=False).any()
                return has_inductive
            
            # If no Typ column, check if there's a development_type in the data
            # This would require checking the actual category definitions
            return False
            
        except Exception as e:
            print(f"Error checking for inductive codes in {file_path}: {e}")
            return False
    
    def extract_inductive_codes_from_xlsx(self, file_path: Path) -> Dict[str, InductiveCodeData]:
        """
        Extract inductive codes from XLSX analysis file (SECONDARY METHOD)
        Use this as a fallback when codebook_inductive.json is not available.
        
        Args:
            file_path: Path to the analysis file
            
        Returns:
            Dictionary mapping category names to InductiveCodeData objects
        """
        inductive_codes = {}
        
        try:
            # Read the Kategorien sheet
            df = pd.read_excel(file_path, sheet_name='Kategorien')
            
            # Filter for inductive categories if Typ column exists
            if 'Typ' in df.columns:
                inductive_df = df[df['Typ'].str.lower().str.contains('induktiv|inductive', na=False)]
            else:
                # If no type column, assume all are potentially inductive
                inductive_df = df
            
            # Extract each category
            for _, row in inductive_df.iterrows():
                category_name = row.get('Hauptkategorie', '')
                
                if not category_name:
                    continue
                
                # Parse subcategories
                subcategories = {}
                subcat_str = row.get('Subkategorien', '')
                if pd.notna(subcat_str) and subcat_str:
                    # Subcategories are comma-separated
                    subcat_list = [s.strip() for s in str(subcat_str).split(',') if s.strip()]
                    # Create dict with empty descriptions (will be filled from detailed data if available)
                    subcategories = {subcat: "" for subcat in subcat_list}
                
                # Create CategoryData first
                category_data = CategoryData(
                    name=category_name,
                    definition=row.get('Definition', ''),
                    rules=[],  # Rules not typically in the Kategorien sheet
                    examples=[],  # Examples not typically in the Kategorien sheet
                    subcategories=subcategories,
                    added_date=datetime.now().strftime("%Y-%m-%d"),
                    modified_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                # Convert to InductiveCodeData
                inductive_code = InductiveCodeData.from_category_data(
                    category=category_data,
                    source_file=file_path.name,
                    frequency=None  # Could be calculated from Kodierungsergebnisse sheet
                )
                
                inductive_codes[category_name] = inductive_code
            
            # Try to enrich with frequency data from Kodierungsergebnisse sheet
            try:
                codings_df = pd.read_excel(file_path, sheet_name='Kodierungsergebnisse')
                if 'Hauptkategorie' in codings_df.columns:
                    category_counts = codings_df['Hauptkategorie'].value_counts()
                    
                    for category_name in inductive_codes:
                        if category_name in category_counts:
                            inductive_codes[category_name].original_frequency = int(category_counts[category_name])
            except Exception as e:
                print(f"Could not extract frequency data: {e}")
            
        except Exception as e:
            print(f"Error extracting inductive codes from {file_path}: {e}")
        
        return inductive_codes
    
    def get_analysis_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Get metadata about analysis file (date, document count, etc.)
        
        Args:
            file_path: Path to the analysis file
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            'document_count': 0,
            'total_codings': 0,
            'category_count': 0,
            'inductive_category_count': 0
        }
        
        try:
            # Get document count and coding count from Kodierungsergebnisse
            df = pd.read_excel(file_path, sheet_name='Kodierungsergebnisse')
            
            if 'Dokument' in df.columns:
                metadata['document_count'] = df['Dokument'].nunique()
            
            metadata['total_codings'] = len(df)
            
            # Get category counts from Kategorien sheet
            cat_df = pd.read_excel(file_path, sheet_name='Kategorien')
            metadata['category_count'] = len(cat_df)
            
            if 'Typ' in cat_df.columns:
                inductive_count = cat_df['Typ'].str.lower().str.contains('induktiv|inductive', na=False).sum()
                metadata['inductive_category_count'] = int(inductive_count)
            
        except Exception as e:
            print(f"Error extracting metadata from {file_path}: {e}")
        
        return metadata
