"""
Configuration Data Models
=========================
Data models for QCA-AID configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class CoderSetting:
    """Repräsentiert Coder-Konfiguration"""
    temperature: float
    coder_id: str
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validiert Coder-Einstellungen.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Validate temperature range
        if not isinstance(self.temperature, (int, float)):
            errors.append(f"Temperature must be numeric, got {type(self.temperature).__name__}")
        elif not 0.0 <= self.temperature <= 2.0:
            errors.append(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        
        # Validate coder_id
        if not isinstance(self.coder_id, str):
            errors.append(f"Coder ID must be string, got {type(self.coder_id).__name__}")
        elif not self.coder_id.strip():
            errors.append("Coder ID cannot be empty")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary"""
        return {
            'temperature': self.temperature,
            'coder_id': self.coder_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CoderSetting':
        """Erstellt aus Dictionary"""
        return cls(
            temperature=data.get('temperature', 0.3),
            coder_id=data.get('coder_id', 'auto_1')
        )


@dataclass
class ConfigData:
    """Repräsentiert QCA-AID Konfiguration"""
    model_provider: str
    model_name: str
    data_dir: str
    output_dir: str
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    code_with_context: bool
    multiple_codings: bool
    multiple_coding_threshold: float
    analysis_mode: str
    review_mode: str
    attribute_labels: Dict[str, str]
    coder_settings: List[CoderSetting]
    manual_coding_enabled: bool = False
    export_annotated_pdfs: bool = True
    pdf_annotation_fuzzy_threshold: float = 0.85
    relevance_threshold: float = 0.0  # Mindest-Konfidenz für relevante Segmente
    enable_optimization: bool = True  # Neue effiziente Kodiermethode (Batching, Caching)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validiert Konfiguration.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Validate model settings
        if not self.model_provider or not self.model_provider.strip():
            errors.append("Model provider cannot be empty")
        if not self.model_name or not self.model_name.strip():
            errors.append("Model name cannot be empty")
        
        # Validate directories
        if not self.data_dir or not self.data_dir.strip():
            errors.append("Data directory cannot be empty")
        if not self.output_dir or not self.output_dir.strip():
            errors.append("Output directory cannot be empty")
        
        # Validate numeric parameters
        if self.chunk_size <= 0:
            errors.append(f"Chunk size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            errors.append(f"Chunk overlap cannot be negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            errors.append(f"Chunk overlap ({self.chunk_overlap}) must be less than chunk size ({self.chunk_size})")
        if self.batch_size <= 0:
            errors.append(f"Batch size must be positive, got {self.batch_size}")
        
        # Validate thresholds
        if not 0.0 <= self.multiple_coding_threshold <= 1.0:
            errors.append(f"Multiple coding threshold must be between 0.0 and 1.0, got {self.multiple_coding_threshold}")
        if not 0.0 <= self.pdf_annotation_fuzzy_threshold <= 1.0:
            errors.append(f"PDF annotation fuzzy threshold must be between 0.0 and 1.0, got {self.pdf_annotation_fuzzy_threshold}")
        
        # Validate enums
        valid_analysis_modes = {'full', 'abductive', 'deductive', 'inductive', 'grounded'}
        if self.analysis_mode not in valid_analysis_modes:
            errors.append(f"Invalid analysis mode '{self.analysis_mode}'. Must be one of: {', '.join(valid_analysis_modes)}")
        
        valid_review_modes = {'auto', 'manual', 'consensus', 'majority'}
        if self.review_mode not in valid_review_modes:
            errors.append(f"Invalid review mode '{self.review_mode}'. Must be one of: {', '.join(valid_review_modes)}")
        
        # Validate coder settings
        if not self.coder_settings:
            errors.append("At least one coder setting is required")
        else:
            for i, coder in enumerate(self.coder_settings):
                is_valid, coder_errors = coder.validate()
                if not is_valid:
                    errors.extend([f"Coder {i+1}: {err}" for err in coder_errors])
        
        # Validate attribute labels
        if not isinstance(self.attribute_labels, dict):
            errors.append("Attribute labels must be a dictionary")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary"""
        return {
            'model_provider': self.model_provider,
            'model_name': self.model_name,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'batch_size': self.batch_size,
            'code_with_context': self.code_with_context,
            'multiple_codings': self.multiple_codings,
            'multiple_coding_threshold': self.multiple_coding_threshold,
            'analysis_mode': self.analysis_mode,
            'review_mode': self.review_mode,
            'attribute_labels': self.attribute_labels,
            'coder_settings': [coder.to_dict() for coder in self.coder_settings],
            'manual_coding_enabled': self.manual_coding_enabled,
            'export_annotated_pdfs': self.export_annotated_pdfs,
            'pdf_annotation_fuzzy_threshold': self.pdf_annotation_fuzzy_threshold,
            'relevance_threshold': self.relevance_threshold,
            'enable_optimization': self.enable_optimization
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConfigData':
        """Erstellt aus Dictionary"""
        # Parse coder settings
        coder_settings = []
        if 'coder_settings' in data:
            for coder_data in data['coder_settings']:
                if isinstance(coder_data, dict):
                    coder_settings.append(CoderSetting.from_dict(coder_data))
                else:
                    # Handle legacy format
                    coder_settings.append(CoderSetting(
                        temperature=0.3,
                        coder_id='auto_1'
                    ))
        
        # Default coder if none provided
        if not coder_settings:
            coder_settings = [CoderSetting(temperature=0.3, coder_id='auto_1')]
        
        return cls(
            model_provider=data.get('model_provider', 'OpenAI'),
            model_name=data.get('model_name', 'gpt-4o-mini'),
            data_dir=data.get('data_dir', 'input'),
            output_dir=data.get('output_dir', 'output'),
            chunk_size=data.get('chunk_size', 1200),
            chunk_overlap=data.get('chunk_overlap', 50),
            batch_size=data.get('batch_size', 8),
            code_with_context=data.get('code_with_context', False),
            multiple_codings=data.get('multiple_codings', True),
            multiple_coding_threshold=data.get('multiple_coding_threshold', 0.85),
            analysis_mode=data.get('analysis_mode', 'deductive'),
            review_mode=data.get('review_mode', 'consensus'),
            attribute_labels=data.get('attribute_labels', {
                'attribut1': 'Attribut1',
                'attribut2': 'Attribut2',
                'attribut3': 'Attribut3'
            }),
            coder_settings=coder_settings,
            manual_coding_enabled=data.get('manual_coding_enabled', False),
            export_annotated_pdfs=data.get('export_annotated_pdfs', True),
            pdf_annotation_fuzzy_threshold=data.get('pdf_annotation_fuzzy_threshold', 0.85),
            relevance_threshold=data.get('relevance_threshold', 0.0),  # Default: 0.0
            enable_optimization=data.get('enable_optimization', True)  # Default: True
        )
