"""
Project Data Models
===================
Data models for project management and settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import json


@dataclass
class ProjectSettings:
    """Project settings data model"""
    project_root: Path
    last_config_file: Optional[Path] = None
    last_codebook_file: Optional[Path] = None
    input_dir_relative: str = "input"
    output_dir_relative: str = "output"
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'project_root': str(self.project_root),
            'last_config_file': str(self.last_config_file) if self.last_config_file else None,
            'last_codebook_file': str(self.last_codebook_file) if self.last_codebook_file else None,
            'input_dir_relative': self.input_dir_relative,
            'output_dir_relative': self.output_dir_relative,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectSettings':
        """Create from dictionary"""
        return cls(
            project_root=Path(data['project_root']),
            last_config_file=Path(data['last_config_file']) if data.get('last_config_file') else None,
            last_codebook_file=Path(data['last_codebook_file']) if data.get('last_codebook_file') else None,
            input_dir_relative=data.get('input_dir_relative', 'input'),
            output_dir_relative=data.get('output_dir_relative', 'output'),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
            last_modified=datetime.fromisoformat(data['last_modified']) if 'last_modified' in data else datetime.now()
        )
    
    def save(self, path: Path) -> bool:
        """Save to .qca-aid-project.json"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving project settings: {e}")
            return False
    
    @classmethod
    def load(cls, path: Path) -> Optional['ProjectSettings']:
        """Load from .qca-aid-project.json"""
        try:
            if not path.exists():
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return cls.from_dict(data)
        except Exception as e:
            print(f"Error loading project settings: {e}")
            return None


@dataclass
class InductiveCodeData:
    """Extended category data for inductive codes"""
    name: str
    definition: str
    rules: list
    examples: list
    subcategories: dict
    source_file: str  # Origin analysis file
    import_date: datetime = field(default_factory=datetime.now)
    is_inductive: bool = True
    original_frequency: Optional[int] = None  # How often it appeared in analysis
    added_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    modified_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'definition': self.definition,
            'rules': self.rules,
            'examples': self.examples,
            'subcategories': self.subcategories,
            'source_file': self.source_file,
            'import_date': self.import_date.isoformat(),
            'is_inductive': self.is_inductive,
            'original_frequency': self.original_frequency,
            'added_date': self.added_date,
            'modified_date': self.modified_date
        }
    
    @classmethod
    def from_category_data(cls, category_dict: Dict[str, Any], source_file: str) -> 'InductiveCodeData':
        """Create from regular CategoryData dictionary"""
        return cls(
            name=category_dict.get('name', ''),
            definition=category_dict.get('definition', ''),
            rules=category_dict.get('rules', []),
            examples=category_dict.get('examples', []),
            subcategories=category_dict.get('subcategories', {}),
            source_file=source_file,
            import_date=datetime.now(),
            is_inductive=True,
            original_frequency=category_dict.get('frequency'),
            added_date=category_dict.get('added_date', datetime.now().strftime("%Y-%m-%d")),
            modified_date=category_dict.get('modified_date', datetime.now().strftime("%Y-%m-%d"))
        )
    
    def to_category_data(self):
        """Convert to CategoryData-compatible dictionary"""
        from .codebook_data import CategoryData
        return CategoryData(
            name=self.name,
            definition=self.definition,
            rules=self.rules,
            examples=self.examples,
            subcategories=self.subcategories,
            added_date=self.added_date,
            modified_date=self.modified_date
        )
