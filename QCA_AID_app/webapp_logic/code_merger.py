"""
Code Merger
===========
Merges deductive and inductive codes with conflict detection and resolution.
"""

from typing import Dict, List, Tuple, Set
from webapp_models.codebook_data import CategoryData
from webapp_models.inductive_code_data import InductiveCodeData


class CodeMerger:
    """Merges deductive and inductive codes"""
    
    def merge_codes(
        self,
        deductive_codes: Dict[str, CategoryData],
        inductive_codes: Dict[str, CategoryData],
        source_file: str
    ) -> Tuple[Dict[str, CategoryData], List[str]]:
        """
        Merge deductive and inductive codes
        
        Args:
            deductive_codes: Existing deductive categories
            inductive_codes: Imported inductive categories
            source_file: Source file name for metadata
        
        Returns:
            Tuple of (merged_codes, warnings)
            warnings contains list of conflicts/issues
        """
        warnings = []
        merged_codes = deductive_codes.copy()
        
        # Detect conflicts first
        conflicts = self.detect_conflicts(deductive_codes, inductive_codes)
        
        # Process each inductive code
        for code_name, code_data in inductive_codes.items():
            # Convert to InductiveCodeData if not already
            if not isinstance(code_data, InductiveCodeData):
                inductive_code = InductiveCodeData.from_category_data(
                    code_data,
                    source_file=source_file
                )
            else:
                inductive_code = code_data
                inductive_code.source_file = source_file
            
            # Check for conflicts
            conflict_found = False
            for conflict_name, conflict_type in conflicts:
                if conflict_name == code_name:
                    conflict_found = True
                    warnings.append(
                        f"Naming conflict: Category '{code_name}' already exists. "
                        f"Consider renaming the inductive code."
                    )
                    break
            
            # If no conflict, add to merged codes
            if not conflict_found:
                merged_codes[code_name] = inductive_code
            else:
                # Suggest alternative name
                suggested_name = self.suggest_rename(
                    code_name,
                    set(merged_codes.keys())
                )
                warnings.append(
                    f"Suggested alternative name: '{suggested_name}'"
                )
        
        return merged_codes, warnings
    
    def detect_conflicts(
        self,
        deductive_codes: Dict[str, CategoryData],
        inductive_codes: Dict[str, CategoryData]
    ) -> List[Tuple[str, str]]:
        """
        Detect naming conflicts between code sets
        
        Args:
            deductive_codes: Existing deductive categories
            inductive_codes: Imported inductive categories
        
        Returns:
            List of (category_name, conflict_type) tuples
        """
        conflicts = []
        
        # Check for exact name matches
        deductive_names = set(deductive_codes.keys())
        inductive_names = set(inductive_codes.keys())
        
        exact_matches = deductive_names.intersection(inductive_names)
        for name in exact_matches:
            conflicts.append((name, "exact_match"))
        
        # Check for case-insensitive matches
        deductive_lower = {name.lower(): name for name in deductive_names}
        for inductive_name in inductive_names:
            inductive_lower = inductive_name.lower()
            if inductive_lower in deductive_lower:
                deductive_original = deductive_lower[inductive_lower]
                # Only add if not already in exact matches
                if inductive_name != deductive_original:
                    conflicts.append((inductive_name, "case_insensitive_match"))
        
        return conflicts
    
    def suggest_rename(self, original_name: str, existing_names: Set[str]) -> str:
        """
        Suggest alternative name for conflicting category
        
        Args:
            original_name: The original category name
            existing_names: Set of existing category names
        
        Returns:
            Suggested alternative name
        """
        # Strategy 1: Add suffix with incrementing number
        counter = 1
        while True:
            suggested = f"{original_name}_{counter}"
            if suggested not in existing_names:
                return suggested
            counter += 1
            
            # Safety limit to prevent infinite loop
            if counter > 1000:
                # Strategy 2: Add timestamp-based suffix
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                return f"{original_name}_{timestamp}"
