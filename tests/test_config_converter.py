"""
Property-Based Tests for ConfigConverter

Tests the round-trip conversion between XLSX and JSON formats.
"""

import os
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from QCA_AID_assets.utils.config.converter import ConfigConverter


# Generators for property-based testing

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

# Feature: config-json-migration, Property 1: Konvertierung erhält Datenstruktur (Round-Trip)
@given(config_data=config_dict_strategy())
@settings(max_examples=100, deadline=None)
def test_round_trip_conversion(config_data):
    """
    **Feature: config-json-migration, Property 1: Konvertierung erhält Datenstruktur (Round-Trip)**
    **Validates: Requirements 1.3**
    
    For any configuration with base_config and analysis_configs,
    converting to XLSX and back to JSON should preserve the structure.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        xlsx_path = os.path.join(tmpdir, 'test_config.xlsx')
        json_path = os.path.join(tmpdir, 'test_config.json')
        
        # Speichere ursprüngliche Daten als JSON
        ConfigConverter.save_json(config_data, json_path)
        
        # Konvertiere JSON -> XLSX
        ConfigConverter.json_to_xlsx(config_data, xlsx_path)
        
        # Konvertiere XLSX -> JSON
        result_data = ConfigConverter.xlsx_to_json(xlsx_path)
        
        # Vergleiche Strukturen
        assert 'base_config' in result_data, "base_config fehlt nach Round-Trip"
        assert 'analysis_configs' in result_data, "analysis_configs fehlt nach Round-Trip"
        
        # Vergleiche base_config
        assert set(result_data['base_config'].keys()) == set(config_data['base_config'].keys()), \
            "base_config Keys unterscheiden sich nach Round-Trip"
        
        for key in config_data['base_config'].keys():
            original_value = config_data['base_config'][key]
            result_value = result_data['base_config'][key]
            
            # Spezielle Behandlung für Floats (Rundungsfehler)
            if isinstance(original_value, float):
                assert abs(original_value - result_value) < 0.0001, \
                    f"base_config[{key}] unterscheidet sich: {original_value} != {result_value}"
            # Behandle leere Strings und None als äquivalent (Excel-Konvention)
            elif (original_value == '' or original_value is None) and (result_value == '' or result_value is None):
                pass  # Beide sind "leer", das ist OK
            else:
                assert original_value == result_value, \
                    f"base_config[{key}] unterscheidet sich: {original_value} != {result_value}"
        
        # Vergleiche analysis_configs
        assert len(result_data['analysis_configs']) == len(config_data['analysis_configs']), \
            "Anzahl der analysis_configs unterscheidet sich nach Round-Trip"
        
        # Sortiere nach Namen für Vergleich
        original_analyses = sorted(config_data['analysis_configs'], key=lambda x: x['name'])
        result_analyses = sorted(result_data['analysis_configs'], key=lambda x: x['name'])
        
        for orig_analysis, result_analysis in zip(original_analyses, result_analyses):
            assert orig_analysis['name'] == result_analysis['name'], \
                "Analyse-Namen unterscheiden sich nach Round-Trip"
            
            # Vergleiche Filter
            assert set(orig_analysis['filters'].keys()) == set(result_analysis['filters'].keys()), \
                f"Filter-Keys für {orig_analysis['name']} unterscheiden sich"
            
            for filter_key in orig_analysis['filters'].keys():
                orig_val = orig_analysis['filters'][filter_key]
                result_val = result_analysis['filters'][filter_key]
                # Behandle leere Strings und None als äquivalent
                if (orig_val == '' or orig_val is None) and (result_val == '' or result_val is None):
                    continue
                assert orig_val == result_val, \
                    f"Filter {filter_key} unterscheidet sich: {orig_val} != {result_val}"
            
            # Vergleiche Parameter
            assert set(orig_analysis['params'].keys()) == set(result_analysis['params'].keys()), \
                f"Param-Keys für {orig_analysis['name']} unterscheiden sich"
            
            for param_key in orig_analysis['params'].keys():
                orig_val = orig_analysis['params'][param_key]
                result_val = result_analysis['params'][param_key]
                # Behandle leere Strings und None als äquivalent
                if (orig_val == '' or orig_val is None) and (result_val == '' or result_val is None):
                    continue
                assert orig_val == result_val, \
                    f"Parameter {param_key} unterscheidet sich: {orig_val} != {result_val}"
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
