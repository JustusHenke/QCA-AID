"""
Property-Based Tests for JSON Validation

Tests JSON format validation and structure validation.
"""

import os
import tempfile
import json
from pathlib import Path
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from QCA_AID_assets.core.validators import ConfigValidator
from QCA_AID_assets.utils.config.converter import ConfigConverter


# Generators for property-based testing (reuse from test_config_converter.py)

@composite
def base_config_strategy(draw):
    """Generiert zufällige aber valide Basis-Konfigurationen"""
    providers = ['openai', 'mistral', 'anthropic']
    models = ['gpt-4o-mini', 'gpt-4', 'mistral-large', 'claude-3']
    
    return {
        'provider': draw(st.sampled_from(providers)),
        'model': draw(st.sampled_from(models)),
        'temperature': draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)),
        'script_dir': draw(st.text(min_size=0, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-')),
        'output_dir': draw(st.text(min_size=1, max_size=15, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-')),
        'explore_file': draw(st.text(min_size=1, max_size=15, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-')) + '.xlsx',
        'clean_keywords': draw(st.booleans()),
        'similarity_threshold': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    }


@composite
def analysis_config_strategy(draw):
    """Generiert zufällige aber valide Analyse-Konfigurationen"""
    analysis_types = ['netzwerk', 'matrix', 'timeline', 'comparison']
    layouts = ['forceatlas2', 'circular', 'hierarchical', 'random']
    
    # Generiere Filter
    num_filters = draw(st.integers(min_value=0, max_value=3))
    filters = {}
    for i in range(num_filters):
        filter_name = f'Attribut_{i+1}'
        filter_value = draw(st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_ '))
        filters[filter_name] = filter_value
    
    # Generiere Parameter
    params = {
        'active': draw(st.booleans()),
        'analysis_type': draw(st.sampled_from(analysis_types)),
        'layout': draw(st.sampled_from(layouts))
    }
    
    # Generiere Namen
    name = draw(st.text(min_size=1, max_size=15, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_ '))
    
    return {
        'name': name,
        'filters': filters,
        'params': params
    }


@composite
def config_dict_strategy(draw):
    """Generiert vollständige Konfigurationsstrukturen"""
    base_config = draw(base_config_strategy())
    
    # Generiere 1-3 Analyse-Konfigurationen
    num_analyses = draw(st.integers(min_value=1, max_value=3))
    analysis_configs = []
    
    for i in range(num_analyses):
        analysis = draw(analysis_config_strategy())
        # Stelle sicher, dass Namen eindeutig sind und nicht leer
        name_base = analysis['name'].strip() or 'analysis'
        analysis['name'] = f"{name_base}_{i}"
        analysis_configs.append(analysis)
    
    return {
        'base_config': base_config,
        'analysis_configs': analysis_configs
    }


# Property-Based Tests

# Feature: config-json-migration, Property 5: JSON-Formatierung ist korrekt
@given(config_data=config_dict_strategy())
@settings(max_examples=100, deadline=None)
def test_json_formatting_is_correct(config_data):
    """
    **Feature: config-json-migration, Property 5: JSON-Formatierung ist korrekt**
    **Validates: Requirements 4.1, 4.2, 4.3**
    
    For any configuration data, the saved JSON file should:
    - Be indented (formatted)
    - Use UTF-8 encoding
    - Contain required top-level keys 'base_config' and 'analysis_configs'
    """
    tmpdir = tempfile.mkdtemp()
    try:
        json_path = os.path.join(tmpdir, 'test_config.json')
        
        # Speichere Konfiguration als JSON
        ConfigConverter.save_json(config_data, json_path)
        
        # Prüfe dass Datei existiert
        assert os.path.exists(json_path), "JSON-Datei wurde nicht erstellt"
        
        # Prüfe UTF-8 Encoding
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            loaded_data = json.loads(content)
        
        # Prüfe dass Daten korrekt geladen werden können
        assert loaded_data is not None, "JSON konnte nicht geladen werden"
        
        # Prüfe erforderliche Top-Level Keys
        assert 'base_config' in loaded_data, "JSON fehlt 'base_config' Key"
        assert 'analysis_configs' in loaded_data, "JSON fehlt 'analysis_configs' Key"
        
        # Prüfe Einrückung (Formatierung)
        # Eine formatierte JSON-Datei sollte Newlines enthalten
        assert '\n' in content, "JSON ist nicht formatiert (keine Zeilenumbrüche)"
        
        # Prüfe dass Einrückung vorhanden ist (Spaces oder Tabs)
        # ConfigConverter.save_json verwendet indent=2, also sollten Spaces vorhanden sein
        assert '  ' in content or '\t' in content, "JSON ist nicht eingerückt"
        
        # Verwende ConfigValidator für vollständige Validierung
        is_valid, errors = ConfigValidator.validate_json_format(json_path)
        assert is_valid, f"JSON-Format-Validierung fehlgeschlagen: {errors}"
        
        # Validiere Struktur und Datentypen
        is_valid_structure, structure_errors = ConfigValidator.validate_json_config(loaded_data)
        assert is_valid_structure, f"JSON-Struktur-Validierung fehlgeschlagen: {structure_errors}"
        
    finally:
        # Cleanup mit Retry-Logik für Windows
        import time
        import shutil
        for attempt in range(3):
            try:
                shutil.rmtree(tmpdir)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1)
                else:
                    pass  # Ignoriere Fehler beim letzten Versuch


if __name__ == '__main__':
    # Führe Tests aus
    import pytest
    pytest.main([__file__, '-v'])
