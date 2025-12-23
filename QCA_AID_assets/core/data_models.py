"""
Datenmodelle für QCA-AID
========================
Enthält die fundamentalen Datenklassen für Kategorien und Kodierungsergebnisse.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any


@dataclass
class CategoryDefinition:
    """Datenklasse für eine Kategorie im Kodiersystem"""
    name: str
    definition: str
    examples: List[str]
    rules: List[str]
    subcategories: Dict[str, str]
    added_date: str
    modified_date: str

    def replace(self, **changes) -> 'CategoryDefinition':
        """
        Erstellt eine neue Instanz mit aktualisierten Werten.
        Ähnlich wie _replace bei namedtuples.
        """
        new_values = {
            'name': self.name,
            'definition': self.definition,
            'examples': self.examples.copy(),
            'rules': self.rules.copy(),
            'subcategories': self.subcategories.copy(),
            'added_date': self.added_date,
            'modified_date': datetime.now().strftime("%Y-%m-%d")
        }
        new_values.update(changes)
        return CategoryDefinition(**new_values)

    def update_examples(self, new_examples: List[str]) -> None:
        """Fügt neue Beispiele hinzu ohne Duplikate."""
        self.examples = list(set(self.examples + new_examples))
        self.modified_date = datetime.now().strftime("%Y-%m-%d")

    def update_rules(self, new_rules: List[str]) -> None:
        """Fügt neue Kodierregeln hinzu ohne Duplikate."""
        self.rules = list(set(self.rules + new_rules))
        self.modified_date = datetime.now().strftime("%Y-%m-%d")

    def add_subcategories(self, new_subcats: Dict[str, str]) -> None:
        """Fügt neue Subkategorien hinzu."""
        self.subcategories.update(new_subcats)
        self.modified_date = datetime.now().strftime("%Y-%m-%d")

    def to_dict(self) -> Dict:
        """Konvertiert die Kategorie in ein Dictionary."""
        return {
            'name': self.name,
            'definition': self.definition,
            'examples': self.examples,
            'rules': self.rules,
            'subcategories': self.subcategories,
            'added_date': self.added_date,
            'modified_date': self.modified_date
        }


@dataclass(frozen=True)  # Macht die Klasse immutable und hashable
class CodingResult:
    """Datenklasse für ein Kodierungsergebnis"""
    category: str
    subcategories: Tuple[str, ...]  # Änderung von List zu Tuple für Hashability
    justification: str
    confidence: Dict[str, Union[float, Tuple[str, ...]]]  # Ändere List zu Tuple
    text_references: Tuple[str, ...]  # Änderung von List zu Tuple
    uncertainties: Optional[Tuple[str, ...]] = None  # Änderung von List zu Tuple
    paraphrase: str = ""
    keywords: str = "" 

    def __post_init__(self):
        # Konvertiere Listen zu Tupeln, falls nötig
        object.__setattr__(self, 'subcategories', tuple(self.subcategories) if isinstance(self.subcategories, list) else self.subcategories)
        object.__setattr__(self, 'text_references', tuple(self.text_references) if isinstance(self.text_references, list) else self.text_references)
        if self.uncertainties is not None:
            object.__setattr__(self, 'uncertainties', tuple(self.uncertainties) if isinstance(self.uncertainties, list) else self.uncertainties)
        
        # Konvertiere confidence Listen zu Tupeln
        if isinstance(self.confidence, dict):
            new_confidence = {}
            for k, v in self.confidence.items():
                if isinstance(v, list):
                    new_confidence[k] = tuple(v)
                else:
                    new_confidence[k] = v
            object.__setattr__(self, 'confidence', new_confidence)

    def to_dict(self) -> Dict:
        """Konvertiert das CodingResult in ein Dictionary"""
        return {
            'category': self.category,
            'subcategories': list(self.subcategories),  # Zurück zu Liste für JSON-Serialisierung
            'justification': self.justification,
            'confidence': self.confidence,
            'text_references': list(self.text_references),  # Zurück zu Liste
            'uncertainties': list(self.uncertainties) if self.uncertainties else None,
            'paraphrase': self.paraphrase,
            'keywords': self.keywords
        }


@dataclass
class ExtendedCodingResult:
    """Extended coding result for intercoder reliability analysis."""
    segment_id: str
    coder_id: str
    category: str
    subcategories: List[str]
    confidence: float
    justification: str
    analysis_mode: str
    timestamp: datetime
    is_manual: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata dict if None."""
        if self.metadata is None:
            self.metadata = {}
        # Ensure timestamp is datetime object
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'segment_id': self.segment_id,
            'coder_id': self.coder_id,
            'category': self.category,
            'subcategories': self.subcategories,
            'confidence': self.confidence,
            'justification': self.justification,
            'analysis_mode': self.analysis_mode,
            'timestamp': self.timestamp.isoformat(),
            'is_manual': self.is_manual,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtendedCodingResult':
        """Create instance from dictionary."""
        return cls(
            segment_id=data['segment_id'],
            coder_id=data['coder_id'],
            category=data['category'],
            subcategories=data['subcategories'],
            confidence=data['confidence'],
            justification=data['justification'],
            analysis_mode=data['analysis_mode'],
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            is_manual=data.get('is_manual', False),
            metadata=data.get('metadata', {})
        )


@dataclass
class CategoryChange:
    """Dokumentiert eine Änderung an einer Kategorie"""
    category_name: str
    change_type: str  # 'add', 'modify', 'delete', 'merge', 'split'
    description: str
    timestamp: str
    old_value: Optional[dict] = None
    new_value: Optional[dict] = None
    affected_codings: List[str] = None
    justification: str = ""
