"""
Explorer Configuration Data Models
===================================
Data models for QCA-AID Explorer analysis configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class AnalysisConfig:
    """Represents a single analysis configuration"""
    name: str
    analysis_type: str  # 'netzwerk', 'heatmap', 'summary_paraphrase', etc.
    active: bool
    filters: Dict[str, Optional[str]]  # filter_Dokument, filter_Hauptkategorie, etc.
    params: Dict[str, Any]  # Analysis-specific parameters
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validates this analysis configuration.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Validate name
        if not isinstance(self.name, str):
            errors.append(f"Name must be string, got {type(self.name).__name__}")
        elif not self.name.strip():
            errors.append("Name cannot be empty")
        
        # Validate analysis_type
        valid_types = {
            'netzwerk', 'heatmap', 'summary_paraphrase', 
            'summary_reasoning', 'custom_summary', 'sentiment_analysis',
            'sunburst', 'treemap'
        }
        if not isinstance(self.analysis_type, str):
            errors.append(f"Analysis type must be string, got {type(self.analysis_type).__name__}")
        elif self.analysis_type not in valid_types:
            errors.append(
                f"Invalid analysis type '{self.analysis_type}'. "
                f"Must be one of: {', '.join(sorted(valid_types))}"
            )
        
        # Validate active
        if not isinstance(self.active, bool):
            errors.append(f"Active must be boolean, got {type(self.active).__name__}")
        
        # Validate filters
        if not isinstance(self.filters, dict):
            errors.append(f"Filters must be dictionary, got {type(self.filters).__name__}")
        else:
            valid_filter_keys = {
                'Dokument', 'Hauptkategorie', 'Subkategorien', 
                'Attribut_1', 'Attribut_2'
            }
            for key in self.filters.keys():
                if key not in valid_filter_keys:
                    errors.append(
                        f"Invalid filter key '{key}'. "
                        f"Must be one of: {', '.join(sorted(valid_filter_keys))}"
                    )
        
        # Validate params
        if not isinstance(self.params, dict):
            errors.append(f"Params must be dictionary, got {type(self.params).__name__}")
        else:
            # Validate analysis-specific parameters
            if self.analysis_type == 'netzwerk':
                errors.extend(self._validate_network_params())
            elif self.analysis_type == 'heatmap':
                errors.extend(self._validate_heatmap_params())
            elif self.analysis_type in ('summary_paraphrase', 'summary_reasoning', 'custom_summary'):
                errors.extend(self._validate_summary_params())
            elif self.analysis_type == 'sentiment_analysis':
                errors.extend(self._validate_sentiment_params())
        
        return len(errors) == 0, errors
    
    def _validate_network_params(self) -> List[str]:
        """Validates network analysis parameters"""
        errors = []
        
        if 'node_size_factor' in self.params:
            val = self.params['node_size_factor']
            if not isinstance(val, (int, float)):
                errors.append(f"node_size_factor must be numeric, got {type(val).__name__}")
            elif val <= 0:
                errors.append(f"node_size_factor must be positive, got {val}")
        
        if 'layout_iterations' in self.params:
            val = self.params['layout_iterations']
            if not isinstance(val, int):
                errors.append(f"layout_iterations must be integer, got {type(val).__name__}")
            elif val <= 0:
                errors.append(f"layout_iterations must be positive, got {val}")
        
        if 'gravity' in self.params:
            val = self.params['gravity']
            if not isinstance(val, (int, float)):
                errors.append(f"gravity must be numeric, got {type(val).__name__}")
        
        if 'scaling' in self.params:
            val = self.params['scaling']
            if not isinstance(val, (int, float)):
                errors.append(f"scaling must be numeric, got {type(val).__name__}")
            elif val <= 0:
                errors.append(f"scaling must be positive, got {val}")
        
        return errors
    
    def _validate_heatmap_params(self) -> List[str]:
        """Validates heatmap analysis parameters"""
        errors = []
        
        # Required parameters
        required = ['x_attribute', 'y_attribute', 'z_attribute']
        for param in required:
            if param not in self.params:
                errors.append(f"Heatmap analysis requires '{param}' parameter")
            elif not isinstance(self.params[param], str):
                errors.append(f"{param} must be string, got {type(self.params[param]).__name__}")
        
        if 'annot' in self.params and not isinstance(self.params['annot'], bool):
            errors.append(f"annot must be boolean, got {type(self.params['annot']).__name__}")
        
        if 'figsize' in self.params:
            val = self.params['figsize']
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                errors.append(f"figsize must be tuple/list of 2 numbers, got {val}")
            elif not all(isinstance(x, (int, float)) and x > 0 for x in val):
                errors.append(f"figsize values must be positive numbers, got {val}")
        
        return errors
    
    def _validate_summary_params(self) -> List[str]:
        """Validates summary analysis parameters"""
        errors = []
        
        if 'text_column' not in self.params:
            errors.append("Summary analysis requires 'text_column' parameter")
        elif not isinstance(self.params['text_column'], str):
            errors.append(f"text_column must be string, got {type(self.params['text_column']).__name__}")
        
        if self.analysis_type == 'custom_summary':
            if 'prompt_template' not in self.params:
                errors.append("Custom summary requires 'prompt_template' parameter")
            elif not isinstance(self.params['prompt_template'], str):
                errors.append(f"prompt_template must be string, got {type(self.params['prompt_template']).__name__}")
            elif not self.params['prompt_template'].strip():
                errors.append("prompt_template cannot be empty for custom_summary")
        
        return errors
    
    def _validate_sentiment_params(self) -> List[str]:
        """Validates sentiment analysis parameters"""
        errors = []
        
        if 'text_column' not in self.params:
            errors.append("Sentiment analysis requires 'text_column' parameter")
        elif not isinstance(self.params['text_column'], str):
            errors.append(f"text_column must be string, got {type(self.params['text_column']).__name__}")
        
        if 'temperature' in self.params:
            val = self.params['temperature']
            if not isinstance(val, (int, float)):
                errors.append(f"temperature must be numeric, got {type(val).__name__}")
            elif not 0.0 <= val <= 2.0:
                errors.append(f"temperature must be between 0.0 and 2.0, got {val}")
        
        if 'figsize' in self.params:
            val = self.params['figsize']
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                errors.append(f"figsize must be tuple/list of 2 numbers, got {val}")
            elif not all(isinstance(x, (int, float)) and x > 0 for x in val):
                errors.append(f"figsize values must be positive numbers, got {val}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts to dictionary for serialization.
        
        Backward Compatibility (Requirement 9.3, 9.5):
        - Preserves all parameters including unknown ones
        - Maintains format structure compatible with original Python version
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'name': self.name,
            'filters': self.filters.copy(),
            'params': {
                'active': self.active,
                'analysis_type': self.analysis_type,
                **self.params  # All params preserved, including unknown ones
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisConfig':
        """
        Creates from dictionary.
        
        Backward Compatibility (Requirement 9.4):
        - Defaults 'active' to True if not present in old config files
        - Preserves unknown parameters in params dictionary
        
        Args:
            data: Dictionary with analysis configuration
            
        Returns:
            AnalysisConfig: Analysis configuration object
        """
        # Extract params
        params = data.get('params', {}).copy()
        
        # Extract active and analysis_type from params
        # Requirement 9.4: Default to True if 'active' parameter is missing (backward compatibility)
        active = params.pop('active', True)
        
        # Handle 'enabled' as alias for 'active' (backward compatibility)
        if 'enabled' in params:
            active = params.pop('enabled')
        
        # Convert string values to boolean if needed
        if isinstance(active, str):
            active = active.lower() in ('true', 'ja', 'yes', '1')
        else:
            active = bool(active)
        
        analysis_type = params.pop('analysis_type', 'netzwerk')
        
        # Extract filters
        filters = data.get('filters', {})
        
        # Normalize filter keys (remove 'filter_' prefix if present for backward compatibility)
        normalized_filters = {}
        for key, value in filters.items():
            # Remove 'filter_' prefix if present
            if key.startswith('filter_'):
                key = key[7:]  # Remove 'filter_' prefix
            normalized_filters[key] = value
        
        # Ensure all expected filter keys exist
        expected_filters = ['Dokument', 'Hauptkategorie', 'Subkategorien', 'Attribut_1', 'Attribut_2']
        for filter_key in expected_filters:
            if filter_key not in normalized_filters:
                normalized_filters[filter_key] = None
        
        # Note: Unknown parameters in params are preserved automatically
        # This ensures backward compatibility with config files that have additional parameters
        
        return cls(
            name=data.get('name', 'Unnamed Analysis'),
            analysis_type=analysis_type,
            active=active,
            filters=normalized_filters,
            params=params  # All remaining params are preserved, including unknown ones
        )
    
    @classmethod
    def create_default(cls, analysis_type: str, name: Optional[str] = None) -> 'AnalysisConfig':
        """
        Creates with default values for analysis type.
        
        Args:
            analysis_type: Type of analysis
            name: Optional name for the analysis
            
        Returns:
            AnalysisConfig: Analysis configuration with defaults
        """
        # Default name based on type
        if name is None:
            type_names = {
                'netzwerk': 'Netzwerkanalyse',
                'heatmap': 'Heatmap-Analyse',
                'summary_paraphrase': 'Zusammenfassung (Paraphrase)',
                'summary_reasoning': 'Zusammenfassung (Reasoning)',
                'custom_summary': 'Benutzerdefinierte Zusammenfassung',
                'sentiment_analysis': 'Sentiment-Analyse'
            }
            name = type_names.get(analysis_type, 'Neue Analyse')
        
        # Default filters (all None)
        filters = {
            'Dokument': None,
            'Hauptkategorie': None,
            'Subkategorien': None,
            'Attribut_1': None,
            'Attribut_2': None
        }
        
        # Default parameters based on type
        if analysis_type == 'netzwerk':
            params = {
                'node_size_factor': 1.0,
                'layout_iterations': 50,
                'gravity': 0.1,
                'scaling': 1.0
            }
        elif analysis_type == 'heatmap':
            params = {
                'x_attribute': 'Hauptkategorie',
                'y_attribute': 'Subkategorien',
                'z_attribute': 'count',
                'cmap': 'viridis',
                'figsize': [10, 8],
                'annot': True,
                'fmt': '.2f'
            }
        elif analysis_type == 'summary_paraphrase':
            params = {
                'text_column': 'Paraphrase',
                'prompt_template': ''  # Empty means use default
            }
        elif analysis_type == 'summary_reasoning':
            params = {
                'text_column': 'Reasoning',
                'prompt_template': ''  # Empty means use default
            }
        elif analysis_type == 'custom_summary':
            params = {
                'text_column': 'Paraphrase',
                'prompt_template': 'Summarize the following text:'
            }
        elif analysis_type == 'sentiment_analysis':
            params = {
                'text_column': 'Paraphrase',
                'sentiment_categories': ['positive', 'neutral', 'negative'],
                'color_mapping': {
                    'positive': 'green',
                    'neutral': 'gray',
                    'negative': 'red'
                },
                'chart_title': 'Sentiment Analysis',
                'temperature': 0.3,
                'crosstab_dimensions': [],
                'figsize': [10, 6]
            }
        else:
            params = {}
        
        return cls(
            name=name,
            analysis_type=analysis_type,
            active=True,
            filters=filters,
            params=params
        )


@dataclass
class ExplorerConfigData:
    """Represents complete explorer configuration"""
    base_config: Dict[str, Any]  # provider, model, temperature, etc.
    analysis_configs: List[AnalysisConfig]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validates entire configuration.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Validate base_config
        if not isinstance(self.base_config, dict):
            errors.append(f"base_config must be dictionary, got {type(self.base_config).__name__}")
        else:
            # Check for required base config parameters
            required_params = ['provider', 'model', 'output_dir']
            for param in required_params:
                if param not in self.base_config:
                    errors.append(f"base_config missing required parameter: {param}")
            
            # Validate temperature if present
            if 'temperature' in self.base_config:
                temp = self.base_config['temperature']
                if not isinstance(temp, (int, float)):
                    errors.append(f"temperature must be numeric, got {type(temp).__name__}")
                elif not 0.0 <= temp <= 2.0:
                    errors.append(f"temperature must be between 0.0 and 2.0, got {temp}")
        
        # Validate analysis_configs
        if not isinstance(self.analysis_configs, list):
            errors.append(f"analysis_configs must be list, got {type(self.analysis_configs).__name__}")
        else:
            for i, config in enumerate(self.analysis_configs):
                if not isinstance(config, AnalysisConfig):
                    errors.append(f"analysis_configs[{i}] must be AnalysisConfig instance")
                else:
                    is_valid, config_errors = config.validate()
                    if not is_valid:
                        errors.extend([f"Analysis '{config.name}': {err}" for err in config_errors])
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'base_config': self.base_config.copy(),
            'analysis_configs': [config.to_dict() for config in self.analysis_configs]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExplorerConfigData':
        """
        Creates from dictionary.
        
        Args:
            data: Dictionary with explorer configuration
            
        Returns:
            ExplorerConfigData: Explorer configuration object
        """
        base_config = data.get('base_config', {})
        
        # Parse analysis configs
        analysis_configs = []
        for config_data in data.get('analysis_configs', []):
            analysis_configs.append(AnalysisConfig.from_dict(config_data))
        
        return cls(
            base_config=base_config,
            analysis_configs=analysis_configs
        )
    
    @classmethod
    def create_default(cls) -> 'ExplorerConfigData':
        """
        Creates with default values.
        
        Returns:
            ExplorerConfigData: Default explorer configuration
        """
        base_config = {
            'provider': 'openai',
            'model': 'gpt-4o-mini',
            'temperature': 0.7,
            'script_dir': '',
            'output_dir': 'output',
            'explore_file': 'QCA-AID_Analysis.xlsx',
            'clean_keywords': True,
            'similarity_threshold': 0.7
        }
        
        # Create one default network analysis
        analysis_configs = [
            AnalysisConfig.create_default('netzwerk')
        ]
        
        return cls(
            base_config=base_config,
            analysis_configs=analysis_configs
        )
