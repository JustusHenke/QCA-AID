"""
Codebook Data Models
====================
Data models for QCA-AID codebook.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime


@dataclass
class CategoryData:
    """Repräsentiert Kategorie im Codebook"""
    name: str
    definition: str
    rules: List[str]
    examples: List[str]
    subcategories: Dict[str, str]
    added_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    modified_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validiert Kategorie.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Validate name
        if not self.name or not self.name.strip():
            errors.append("Category name cannot be empty")
        elif len(self.name) < 3:
            errors.append(f"Category name too short (minimum 3 characters), got {len(self.name)}")
        elif len(self.name) > 50:
            errors.append(f"Category name too long (maximum 50 characters), got {len(self.name)}")
        
        # Validate definition (relaxed - only check if not empty)
        if not self.definition or not self.definition.strip():
            errors.append("Category definition cannot be empty")
        
        # Validate rules (relaxed - just check type)
        if not isinstance(self.rules, list):
            errors.append("Rules must be a list")
        
        # Validate examples (relaxed - allow empty)
        if not isinstance(self.examples, list):
            errors.append("Examples must be a list")
        
        # Validate subcategories (relaxed - allow empty)
        if not isinstance(self.subcategories, dict):
            errors.append("Subcategories must be a dictionary")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary"""
        return {
            'name': self.name,
            'definition': self.definition,
            'rules': self.rules,
            'examples': self.examples,
            'subcategories': self.subcategories,
            'added_date': self.added_date,
            'modified_date': self.modified_date
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CategoryData':
        """Erstellt aus Dictionary"""
        return cls(
            name=data.get('name', ''),
            definition=data.get('definition', ''),
            rules=data.get('rules', []),
            examples=data.get('examples', []),
            subcategories=data.get('subcategories', {}),
            added_date=data.get('added_date', datetime.now().strftime("%Y-%m-%d")),
            modified_date=data.get('modified_date', datetime.now().strftime("%Y-%m-%d"))
        )


@dataclass
class CodebookData:
    """Repräsentiert komplettes Codebook"""
    forschungsfrage: str
    kodierregeln: Dict[str, List[str]]  # general, format, exclusion
    deduktive_kategorien: Dict[str, CategoryData]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validiert Codebook.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Validate research question
        if not self.forschungsfrage or not self.forschungsfrage.strip():
            errors.append("Research question cannot be empty")
        
        # Validate coding rules structure
        if not isinstance(self.kodierregeln, dict):
            errors.append("Coding rules must be a dictionary")
        else:
            required_keys = {'general', 'format', 'exclusion'}
            missing_keys = required_keys - set(self.kodierregeln.keys())
            if missing_keys:
                errors.append(f"Missing coding rule categories: {', '.join(missing_keys)}")
            
            for key, rules in self.kodierregeln.items():
                if not isinstance(rules, list):
                    errors.append(f"Coding rules '{key}' must be a list")
        
        # Validate categories
        if not isinstance(self.deduktive_kategorien, dict):
            errors.append("Deductive categories must be a dictionary")
        elif not self.deduktive_kategorien:
            errors.append("At least one category is required")
        else:
            # Auto-fix: Convert dict categories to CategoryData instances
            for cat_name, category in list(self.deduktive_kategorien.items()):
                # Check by type name instead of isinstance (handles import conflicts)
                cat_type_name = type(category).__name__
                
                if cat_type_name == 'CategoryData':
                    # It's already a CategoryData (even if isinstance fails due to import issues)
                    pass
                elif isinstance(category, dict):
                    # Convert dict to CategoryData
                    if 'name' not in category:
                        category['name'] = cat_name
                    self.deduktive_kategorien[cat_name] = CategoryData.from_dict(category)
                    category = self.deduktive_kategorien[cat_name]
                else:
                    errors.append(f"Category '{cat_name}' has unexpected type: {cat_type_name}")
                    continue
                
                # Validate the category
                is_valid, cat_errors = category.validate()
                if not is_valid:
                    errors.extend([f"Category '{cat_name}': {err}" for err in cat_errors])
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary"""
        return {
            'forschungsfrage': self.forschungsfrage,
            'kodierregeln': self.kodierregeln,
            'deduktive_kategorien': {
                name: cat.to_dict() 
                for name, cat in self.deduktive_kategorien.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CodebookData':
        """Erstellt aus Dictionary"""
        # Parse categories
        kategorien = {}
        if 'deduktive_kategorien' in data:
            for name, cat_data in data['deduktive_kategorien'].items():
                if isinstance(cat_data, CategoryData):
                    kategorien[name] = cat_data
                elif isinstance(cat_data, dict):
                    # Add name to dict if not present
                    if 'name' not in cat_data:
                        cat_data['name'] = name
                    kategorien[name] = CategoryData.from_dict(cat_data)
        
        return cls(
            forschungsfrage=data.get('forschungsfrage', ''),
            kodierregeln=data.get('kodierregeln', {
                'general': [],
                'format': [],
                'exclusion': []
            }),
            deduktive_kategorien=kategorien
        )
