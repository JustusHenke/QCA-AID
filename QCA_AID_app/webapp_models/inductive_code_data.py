"""
Inductive Code Data Models
===========================
Extended data models for inductive codes imported from analysis results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from .codebook_data import CategoryData


@dataclass
class InductiveCodeData(CategoryData):
    """Extended category data for inductive codes"""
    source_file: str = ""  # Origin analysis file
    import_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    is_inductive: bool = True
    original_frequency: Optional[int] = None  # How often it appeared in analysis
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            'source_file': self.source_file,
            'import_date': self.import_date,
            'is_inductive': self.is_inductive,
            'original_frequency': self.original_frequency
        })
        return base_dict
    
    @classmethod
    def from_category_data(
        cls,
        category: CategoryData,
        source_file: str,
        frequency: Optional[int] = None
    ) -> 'InductiveCodeData':
        """Create from regular CategoryData"""
        return cls(
            name=category.name,
            definition=category.definition,
            rules=category.rules,
            examples=category.examples,
            subcategories=category.subcategories,
            added_date=category.added_date,
            modified_date=category.modified_date,
            source_file=source_file,
            import_date=datetime.now().strftime("%Y-%m-%d"),
            is_inductive=True,
            original_frequency=frequency
        )
