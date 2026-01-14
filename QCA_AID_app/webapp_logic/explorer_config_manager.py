"""
Explorer Configuration Manager
===============================
Manages loading, saving, and manipulation of QCA-AID Explorer analysis configurations.
Integrates with existing ExplorerConfigLoader and provides CRUD operations for analyses.
"""

import sys
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path

# Add parent directory to path to access QCA_AID_assets
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import webapp models
from webapp_models.explorer_config_data import ExplorerConfigData, AnalysisConfig
from webapp_logic.category_loader import CategoryLoader


class ExplorerConfigManager:
    """
    Manages QCA-AID Explorer configuration for the webapp.
    
    Responsibilities:
    - Loads and saves explorer configurations (XLSX and JSON)
    - Provides CRUD operations for analysis configurations
    - Validates configurations
    - Integrates with ExplorerConfigLoader
    
    Requirements: 1.1, 1.5, 7.1, 7.2, 7.3, 7.4, 7.5
    """
    
    def __init__(self, project_dir: Optional[str] = None):
        """
        Initializes ExplorerConfigManager with project directory.
        
        Args:
            project_dir: Project directory (default: current directory)
        """
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.xlsx_path = self.project_dir / "QCA-AID-Explorer-Config.xlsx"
        self.json_path = self.project_dir / "QCA-AID-Explorer-Config.json"
        self.category_loader = None
        
        # Try to load categories from analysis results
        self._load_categories()

    def _load_config_file(self, file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load configuration file directly without interactive synchronization.
        
        Args:
            file_path: Path to configuration file (.xlsx or .json)
            
        Returns:
            Tuple of (base_config, analysis_configs_raw)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        import pandas as pd
        from QCA_AID_assets.utils.config.converter import ConfigConverter
        from QCA_AID_assets.core.validators import ConfigValidator
        
        config_path = Path(file_path)
        
        if config_path.suffix.lower() == '.json':
            # Load from JSON
            config_data = ConfigConverter.load_json(str(config_path))
            
            # Validate JSON structure (but allow enabled_charts and color_scheme - they are part of the new format!)
            is_valid_structure, structure_errors = ConfigValidator.validate_json_config(config_data)
            if not is_valid_structure:
                error_msg = "JSON configuration is invalid:\n" + "\n".join(f"  - {e}" for e in structure_errors)
                raise ValueError(error_msg)
            
            base_config = config_data.get('base_config', {})
            analysis_configs_raw = config_data.get('analysis_configs', [])
            
        else:
            # Load from XLSX
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            # Read base sheet
            base_df = pd.read_excel(str(config_path), sheet_name='Basis')
            base_config = {}
            for _, row in base_df.iterrows():
                param_name = str(row['Parameter'])
                param_value = row['Wert']
                if pd.isna(param_value):
                    param_value = None
                base_config[param_name] = param_value
            
            # Read analysis sheets
            excel = pd.ExcelFile(str(config_path))
            analysis_configs_raw = []
            
            for sheet_name in excel.sheet_names:
                if sheet_name.lower() != 'basis':
                    analysis_df = pd.read_excel(str(config_path), sheet_name=sheet_name)
                    
                    analysis_config = {'name': sheet_name}
                    filter_params = {}
                    other_params = {}
                    
                    for _, row in analysis_df.iterrows():
                        param_name = str(row['Parameter'])
                        param_value = row['Wert']
                        
                        # Handle active/enabled parameter
                        if param_name.lower() in ('active', 'enabled'):
                            if pd.isna(param_value):
                                param_value = True
                            elif isinstance(param_value, str):
                                param_value = param_value.lower() in ('true', 'ja', 'yes', '1')
                            else:
                                param_value = bool(param_value)
                            other_params['active'] = param_value
                            continue
                        
                        if pd.isna(param_value):
                            param_value = None
                        
                        if param_name.startswith('filter_'):
                            filter_name = param_name[7:]
                            filter_params[filter_name] = param_value
                        else:
                            other_params[param_name] = param_value
                    
                    if 'active' not in other_params:
                        other_params['active'] = True
                    
                    analysis_config['filters'] = filter_params
                    analysis_config['params'] = other_params
                    analysis_configs_raw.append(analysis_config)
        
        return base_config, analysis_configs_raw
    
    def load_config(self, file_path: Optional[str] = None) -> Tuple[bool, Optional[ExplorerConfigData], List[str]]:
        """
        Loads explorer configuration from XLSX or JSON using ExplorerConfigLoader.
        
        Requirement 1.1: WHEN the webapp loads an Explorer config file 
                        THEN the system SHALL parse all analysis configurations from the file
        Requirement 9.1: WHEN loading an XLSX config file with multiple analysis sheets 
                        THEN the system SHALL parse each sheet as a separate analysis configuration
        Requirement 9.2: WHEN loading a JSON config file with analysis_configs array 
                        THEN the system SHALL parse each array element as a separate analysis configuration
        Requirement 9.4: WHEN an old config file lacks an 'active' parameter 
                        THEN the system SHALL default that analysis to active
        
        Args:
            file_path: Path to configuration file (optional, uses default paths if None)
            
        Returns:
            Tuple[bool, Optional[ExplorerConfigData], List[str]]: 
                (success, config_data, error_messages)
        """
        errors = []
        
        try:
            # Determine file path
            if file_path:
                file_path = Path(file_path)
                if not file_path.exists():
                    return False, None, [f"File not found: {file_path}"]
            else:
                # Use default paths - prefer JSON if it exists
                if self.json_path.exists():
                    file_path = self.json_path
                elif self.xlsx_path.exists():
                    file_path = self.xlsx_path
                else:
                    return False, None, ["No configuration file found"]
            
            # Load configuration directly without interactive synchronization
            # The ExplorerConfigLoader triggers interactive prompts which block the webapp
            try:
                base_config, analysis_configs_raw = self._load_config_file(str(file_path))
                
                # Convert raw analysis configs to AnalysisConfig objects
                # The from_dict method handles backward compatibility:
                # - Defaults 'active' to True if missing (Requirement 9.4)
                # - Preserves unknown parameters
                analysis_configs = []
                for config_data in analysis_configs_raw:
                    try:
                        analysis_config = AnalysisConfig.from_dict(config_data)
                        analysis_configs.append(analysis_config)
                    except Exception as e:
                        errors.append(f"Error parsing analysis '{config_data.get('name', 'unknown')}': {str(e)}")
                
                # Create ExplorerConfigData
                config_data = ExplorerConfigData(
                    base_config=base_config,
                    analysis_configs=analysis_configs
                )
                
                # Validate configuration
                is_valid, validation_errors = config_data.validate()
                if not is_valid:
                    errors.extend(validation_errors)
                    # Return config_data anyway so UI can display values
                    return False, config_data, errors
                
                return True, config_data, []
                
            except Exception as e:
                return False, None, [f"Error loading configuration: {str(e)}"]
                
        except Exception as e:
            return False, None, [f"Unexpected error: {str(e)}"]
    
    def save_config(self, config: ExplorerConfigData, file_path: Optional[str] = None, 
                   format: str = 'json') -> Tuple[bool, List[str]]:
        """
        Saves explorer configuration to XLSX or JSON.
        
        Requirement 1.5: WHEN the user saves the configuration 
                        THEN the system SHALL persist all analysis configurations to the config file
        Requirement 7.5: WHEN adding or removing analyses 
                        THEN the system SHALL update the config file to reflect the changes
        Requirement 9.3: WHEN saving configurations 
                        THEN the system SHALL maintain the same file format structure as the original Python version
        Requirement 9.5: WHEN synchronizing XLSX and JSON files 
                        THEN the system SHALL preserve all analysis configurations in both formats
        
        Args:
            config: ExplorerConfigData object to save
            file_path: Path to target file (optional, uses default path)
            format: Format ('xlsx' or 'json')
            
        Returns:
            Tuple[bool, List[str]]: (success, error_messages)
        """
        errors = []
        
        try:
            # Validate configuration before saving
            is_valid, validation_errors = config.validate()
            if not is_valid:
                return False, validation_errors
            
            # Determine file path
            if file_path:
                file_path = Path(file_path)
            else:
                if format == 'json':
                    file_path = self.json_path
                elif format == 'xlsx':
                    file_path = self.xlsx_path
                else:
                    return False, [f"Invalid format: {format}"]
            
            # Convert to dictionary format expected by ExplorerConfigLoader
            # The to_dict method preserves all parameters including unknown ones
            config_dict = config.to_dict()
            
            # Save based on format
            if format == 'json':
                success, save_errors = self._save_to_json(config_dict, str(file_path))
            elif format == 'xlsx':
                success, save_errors = self._save_to_xlsx(config_dict, str(file_path))
            else:
                return False, [f"Invalid format: {format}"]
            
            if not success:
                return False, save_errors
            
            return True, []
            
        except Exception as e:
            return False, [f"Error saving configuration: {str(e)}"]
    
    def add_analysis(self, analysis_type: str, name: Optional[str] = None) -> AnalysisConfig:
        """
        Creates new analysis configuration with default values.
        
        Requirement 7.1: WHEN the user clicks an "Add Analysis" button 
                        THEN the system SHALL create a new analysis configuration with default values
        Requirement 7.2: WHEN the user selects an analysis type for a new configuration 
                        THEN the system SHALL initialize appropriate default parameters for that type
        
        Args:
            analysis_type: Type of analysis to create
            name: Optional name for the analysis
            
        Returns:
            AnalysisConfig: New analysis configuration with defaults
        """
        return AnalysisConfig.create_default(analysis_type, name)
    
    def remove_analysis(self, config: ExplorerConfigData, analysis_index: int) -> Tuple[bool, Optional[ExplorerConfigData], List[str]]:
        """
        Removes analysis configuration by index.
        
        Requirement 7.3: WHEN the user clicks a "Remove Analysis" button 
                        THEN the system SHALL delete that analysis configuration
        
        Args:
            config: Current ExplorerConfigData
            analysis_index: Index of analysis to remove
            
        Returns:
            Tuple[bool, Optional[ExplorerConfigData], List[str]]: 
                (success, updated_config, error_messages)
        """
        errors = []
        
        try:
            # Validate index
            if analysis_index < 0 or analysis_index >= len(config.analysis_configs):
                return False, None, [f"Invalid analysis index: {analysis_index}"]
            
            # Create new list without the specified analysis
            new_analysis_configs = [
                analysis for i, analysis in enumerate(config.analysis_configs)
                if i != analysis_index
            ]
            
            # Create new config with updated analysis list
            new_config = ExplorerConfigData(
                base_config=config.base_config.copy(),
                analysis_configs=new_analysis_configs
            )
            
            return True, new_config, []
            
        except Exception as e:
            return False, None, [f"Error removing analysis: {str(e)}"]
    
    def update_analysis(self, config: ExplorerConfigData, analysis_index: int, 
                       updates: Dict[str, Any]) -> Tuple[bool, Optional[ExplorerConfigData], List[str]]:
        """
        Updates analysis configuration.
        
        Args:
            config: Current ExplorerConfigData
            analysis_index: Index of analysis to update
            updates: Dictionary of updates to apply
            
        Returns:
            Tuple[bool, Optional[ExplorerConfigData], List[str]]: 
                (success, updated_config, error_messages)
        """
        errors = []
        
        try:
            # Validate index
            if analysis_index < 0 or analysis_index >= len(config.analysis_configs):
                return False, None, [f"Invalid analysis index: {analysis_index}"]
            
            # Get the analysis to update
            analysis = config.analysis_configs[analysis_index]
            
            # Create updated analysis
            updated_analysis = AnalysisConfig(
                name=updates.get('name', analysis.name),
                analysis_type=updates.get('analysis_type', analysis.analysis_type),
                active=updates.get('active', analysis.active),
                filters=updates.get('filters', analysis.filters.copy()),
                params=updates.get('params', analysis.params.copy())
            )
            
            # Validate updated analysis
            is_valid, validation_errors = updated_analysis.validate()
            if not is_valid:
                return False, None, validation_errors
            
            # Create new analysis list with updated analysis
            new_analysis_configs = [
                updated_analysis if i == analysis_index else analysis
                for i, analysis in enumerate(config.analysis_configs)
            ]
            
            # Create new config
            new_config = ExplorerConfigData(
                base_config=config.base_config.copy(),
                analysis_configs=new_analysis_configs
            )
            
            return True, new_config, []
            
        except Exception as e:
            return False, None, [f"Error updating analysis: {str(e)}"]
    
    def reorder_analyses(self, config: ExplorerConfigData, new_order: List[int]) -> Tuple[bool, Optional[ExplorerConfigData], List[str]]:
        """
        Reorders analysis configurations.
        
        Requirement 7.4: WHEN the user reorders analysis tabs 
                        THEN the system SHALL update the configuration order accordingly
        
        Args:
            config: Current ExplorerConfigData
            new_order: List of indices representing new order
            
        Returns:
            Tuple[bool, Optional[ExplorerConfigData], List[str]]: 
                (success, updated_config, error_messages)
        """
        errors = []
        
        try:
            # Validate new_order
            if len(new_order) != len(config.analysis_configs):
                return False, None, [f"new_order length ({len(new_order)}) must match number of analyses ({len(config.analysis_configs)})"]
            
            # Check all indices are valid and unique
            if set(new_order) != set(range(len(config.analysis_configs))):
                return False, None, ["new_order must contain all indices exactly once"]
            
            # Reorder analyses
            new_analysis_configs = [config.analysis_configs[i] for i in new_order]
            
            # Create new config
            new_config = ExplorerConfigData(
                base_config=config.base_config.copy(),
                analysis_configs=new_analysis_configs
            )
            
            return True, new_config, []
            
        except Exception as e:
            return False, None, [f"Error reordering analyses: {str(e)}"]
    
    def validate_analysis(self, analysis: AnalysisConfig) -> Tuple[bool, List[str]]:
        """
        Validates single analysis configuration.
        
        Args:
            analysis: AnalysisConfig to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        return analysis.validate()
    
    def get_default_parameters(self, analysis_type: str) -> Dict[str, Any]:
        """
        Gets default parameters for analysis type.
        
        Args:
            analysis_type: Type of analysis
            
        Returns:
            Dict[str, Any]: Default parameters for the analysis type
        """
        # Create a default analysis and extract its params
        default_analysis = AnalysisConfig.create_default(analysis_type)
        return default_analysis.params.copy()
    
    # Private helper methods
    
    def _save_to_json(self, config_dict: Dict, json_path: str) -> Tuple[bool, List[str]]:
        """
        Saves configuration to JSON file.
        
        Args:
            config_dict: Configuration as dictionary
            json_path: Path to target JSON file
            
        Returns:
            Tuple[bool, List[str]]: (success, errors)
        """
        try:
            from QCA_AID_assets.utils.config.converter import ConfigConverter
            
            # Save using ConfigConverter
            ConfigConverter.save_json(config_dict, json_path)
            
            return True, []
            
        except IOError as e:
            return False, [f"Error writing JSON file: {str(e)}"]
        except Exception as e:
            return False, [f"Error saving JSON file: {str(e)}"]
    
    def _save_to_xlsx(self, config_dict: Dict, xlsx_path: str) -> Tuple[bool, List[str]]:
        """
        Saves configuration to XLSX file.
        
        Backward Compatibility (Requirement 9.3, 9.5):
        - Maintains same XLSX structure as original Python version
        - Preserves all parameters including unknown ones
        
        Args:
            config_dict: Configuration as dictionary
            xlsx_path: Path to target XLSX file
            
        Returns:
            Tuple[bool, List[str]]: (success, errors)
        """
        try:
            import pandas as pd
            
            # Create Excel writer
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                # Write base config sheet
                base_config = config_dict['base_config']
                base_df = pd.DataFrame([
                    {'Parameter': key, 'Wert': value}
                    for key, value in base_config.items()
                ])
                base_df.to_excel(writer, sheet_name='Basis', index=False)
                
                # Write each analysis config as a separate sheet
                for analysis_config in config_dict['analysis_configs']:
                    sheet_name = analysis_config['name']
                    
                    # Combine filters and params
                    rows = []
                    
                    # Add filter parameters with 'filter_' prefix (original format)
                    for filter_key, filter_value in analysis_config['filters'].items():
                        rows.append({
                            'Parameter': f'filter_{filter_key}',
                            'Wert': filter_value if filter_value is not None else ''
                        })
                    
                    # Add all params (including unknown ones - backward compatibility)
                    for param_key, param_value in analysis_config['params'].items():
                        rows.append({
                            'Parameter': param_key,
                            'Wert': param_value
                        })
                    
                    # Create DataFrame and write
                    analysis_df = pd.DataFrame(rows)
                    analysis_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return True, []
            
        except ValueError as e:
            return False, [f"Error converting to XLSX: {str(e)}"]
        except Exception as e:
            return False, [f"Error saving XLSX file: {str(e)}"]
    
    def _load_categories(self) -> None:
        """
        Lädt Kategorien aus verfügbaren Analyseergebnissen.
        
        Sucht nach QCA-AID_Analysis_*.xlsx Dateien im konfigurierten output-Verzeichnis
        und versucht, Kategorien aus dem "Kategorien"-Sheet zu laden.
        """
        try:
            # Versuche zuerst, output_dir aus der Konfiguration zu lesen
            output_dir_name = "output"  # Default
            
            # Prüfe ob eine Konfigurationsdatei existiert
            if self.xlsx_path.exists():
                try:
                    import pandas as pd
                    base_df = pd.read_excel(str(self.xlsx_path), sheet_name='Basis')
                    for _, row in base_df.iterrows():
                        if str(row['Parameter']) == 'output_dir':
                            output_dir_name = str(row['Wert'])
                            break
                except Exception:
                    pass  # Verwende Default
            elif self.json_path.exists():
                try:
                    import json
                    with open(self.json_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                        output_dir_name = config_data.get('base_config', {}).get('output_dir', 'output')
                except Exception:
                    pass  # Verwende Default
            
            # Suche nach Analyseergebnissen im konfigurierten output-Verzeichnis
            output_dir = self.project_dir / output_dir_name
            if not output_dir.exists():
                return
            
            # Finde die neueste QCA-AID_Analysis_*.xlsx Datei
            analysis_files = list(output_dir.glob("QCA-AID_Analysis_*.xlsx"))
            if not analysis_files:
                return
            
            # Sortiere nach Änderungsdatum (neueste zuerst)
            analysis_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_file = analysis_files[0]
            
            # Versuche Kategorien zu laden
            self.category_loader = CategoryLoader(str(latest_file))
            
        except Exception:
            # Fehler beim Laden - das ist OK, wir verwenden dann keine intelligenten Dropdowns
            self.category_loader = None
    
    def get_category_loader(self) -> Optional[CategoryLoader]:
        """
        Gibt den CategoryLoader zurück falls verfügbar.
        
        Returns:
            CategoryLoader-Instanz oder None falls nicht verfügbar
        """
        return self.category_loader
    
    def get_filter_options(self) -> Dict[str, Any]:
        """
        Gibt verfügbare Filter-Optionen zurück, einschließlich Kategorien.
        
        Returns:
            Dictionary mit verfügbaren Filter-Optionen
        """
        options = {
            'main_categories': [],
            'subcategories': [],
            'category_mapping': {},
            'has_categories': False
        }
        
        if self.category_loader and self.category_loader.is_loaded:
            options.update({
                'main_categories': self.category_loader.get_main_categories(),
                'subcategories': self.category_loader.get_all_subcategories(),
                'category_mapping': self.category_loader.get_category_mapping(),
                'has_categories': True
            })
        
        return options
    
    def validate_analysis_filters(self, analysis: AnalysisConfig) -> List[str]:
        """
        Validiert die Filter einer Analysekonfiguration gegen verfügbare Kategorien.
        
        Args:
            analysis: Analysekonfiguration mit Filtern
            
        Returns:
            Liste von Validierungsfehlern (leer wenn alles gültig ist)
        """
        if not self.category_loader or not self.category_loader.is_loaded:
            return []  # Keine Validierung möglich ohne Kategorien
        
        errors = []
        filters = analysis.filters
        
        main_category = filters.get('Hauptkategorie')
        subcategories_str = filters.get('Subkategorien')
        
        # Parse Subkategorien falls als String angegeben
        subcategories = None
        if subcategories_str:
            if isinstance(subcategories_str, str):
                subcategories = [sub.strip() for sub in subcategories_str.split(',') if sub.strip()]
            elif isinstance(subcategories_str, list):
                subcategories = subcategories_str
        
        # Validiere gegen Kategorien
        is_valid, validation_errors = self.category_loader.validate_filter_values(
            main_category, subcategories
        )
        
        if not is_valid:
            errors.extend(validation_errors)
        
        return errors
