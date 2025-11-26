"""
Property-Based Tests for ConfigSynchronizer

Tests the difference detection functionality between XLSX and JSON configurations.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite
import copy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from QCA_AID_assets.utils.config.synchronizer import ConfigSynchronizer
from QCA_AID_assets.utils.config.converter import ConfigConverter


# Generators for property-based testing (reuse from converter tests)

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


@composite
def modified_config_pair_strategy(draw):
    """
    Generiert ein Paar von Konfigurationen mit garantierten Unterschieden.
    
    Returns:
        Tuple[Dict, Dict, List[str]]: (xlsx_data, json_data, expected_difference_paths)
    """
    # Erstelle Basis-Konfiguration
    original_config = draw(config_dict_strategy())
    
    # Erstelle Kopie für Modifikation
    modified_config = copy.deepcopy(original_config)
    
    # Liste der erwarteten Differenz-Pfade
    expected_diffs = []
    
    # Wähle zufällig welche Art von Änderung vorgenommen wird
    modification_type = draw(st.sampled_from([
        'base_config_value',
        'analysis_filter_value',
        'analysis_param_value',
        'analysis_only_in_one'
    ]))
    
    if modification_type == 'base_config_value':
        # Ändere einen Wert in base_config
        key = draw(st.sampled_from(list(original_config['base_config'].keys())))
        
        if key == 'provider':
            new_value = draw(st.sampled_from(['openai', 'mistral', 'anthropic']))
            # Stelle sicher, dass der Wert unterschiedlich ist
            while new_value == original_config['base_config'][key]:
                new_value = draw(st.sampled_from(['openai', 'mistral', 'anthropic']))
            modified_config['base_config'][key] = new_value
        elif key == 'temperature' or key == 'similarity_threshold':
            # Ändere Float-Wert signifikant
            original_val = original_config['base_config'][key]
            modified_config['base_config'][key] = original_val + 0.5 if original_val < 1.5 else original_val - 0.5
        elif key == 'clean_keywords':
            modified_config['base_config'][key] = not original_config['base_config'][key]
        else:
            # String-Wert
            modified_config['base_config'][key] = original_config['base_config'][key] + '_modified'
        
        expected_diffs.append(f"base_config.{key}")
    
    elif modification_type == 'analysis_filter_value':
        # Ändere einen Filter-Wert in einer Analyse
        if original_config['analysis_configs']:
            analysis_idx = draw(st.integers(min_value=0, max_value=len(original_config['analysis_configs'])-1))
            analysis = modified_config['analysis_configs'][analysis_idx]
            
            if analysis['filters']:
                filter_key = draw(st.sampled_from(list(analysis['filters'].keys())))
                analysis['filters'][filter_key] = analysis['filters'][filter_key] + '_modified'
                expected_diffs.append(f"analysis_configs[{analysis['name']}].filters.{filter_key}")
    
    elif modification_type == 'analysis_param_value':
        # Ändere einen Parameter-Wert in einer Analyse
        if original_config['analysis_configs']:
            analysis_idx = draw(st.integers(min_value=0, max_value=len(original_config['analysis_configs'])-1))
            analysis = modified_config['analysis_configs'][analysis_idx]
            
            param_key = draw(st.sampled_from(list(analysis['params'].keys())))
            
            if param_key == 'active':
                analysis['params'][param_key] = not analysis['params'][param_key]
            else:
                analysis['params'][param_key] = analysis['params'][param_key] + '_modified'
            
            expected_diffs.append(f"analysis_configs[{analysis['name']}].params.{param_key}")
    
    elif modification_type == 'analysis_only_in_one':
        # Füge eine Analyse nur in einer Konfiguration hinzu
        new_analysis = draw(analysis_config_strategy())
        new_analysis['name'] = 'unique_analysis_' + draw(st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz0123456789'))
        modified_config['analysis_configs'].append(new_analysis)
        expected_diffs.append(f"analysis_configs[{new_analysis['name']}]")
    
    # Stelle sicher, dass mindestens eine Differenz existiert
    assume(len(expected_diffs) > 0)
    
    return original_config, modified_config, expected_diffs


# Property-Based Tests

# Feature: config-json-migration, Property 2: Differenzerkennung ist vollständig
@given(config_pair=modified_config_pair_strategy())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
def test_difference_detection_completeness(config_pair):
    """
    **Feature: config-json-migration, Property 2: Differenzerkennung ist vollständig**
    **Validates: Requirements 2.1, 2.2**
    
    For any pair of XLSX and JSON configurations with different values,
    the system should correctly identify and list all differences.
    """
    xlsx_data, json_data, expected_diff_paths = config_pair
    
    # Erstelle Synchronizer (Pfade sind nicht relevant für diesen Test)
    synchronizer = ConfigSynchronizer('dummy.xlsx', 'dummy.json')
    
    # Erkenne Differenzen
    differences = synchronizer._detect_differences(xlsx_data, json_data)
    
    # Es sollten Differenzen gefunden werden
    assert len(differences) > 0, \
        f"Keine Differenzen gefunden, aber Änderungen wurden vorgenommen: {expected_diff_paths}"
    
    # Prüfe, dass alle erwarteten Differenz-Pfade in den gefundenen Differenzen vorkommen
    for expected_path in expected_diff_paths:
        # Suche nach dem Pfad in den Differenz-Strings
        found = any(expected_path in diff for diff in differences)
        assert found, \
            f"Erwartete Differenz '{expected_path}' wurde nicht gefunden. Gefundene Differenzen: {differences}"
    
    # Prüfe, dass jede Differenz entweder beide Werte enthält oder anzeigt dass etwas nur in einer Datei vorhanden ist
    for diff in differences:
        has_both_values = 'XLSX=' in diff and 'JSON=' in diff
        has_only_in_one = 'Nur in XLSX vorhanden' in diff or 'Nur in JSON vorhanden' in diff
        assert has_both_values or has_only_in_one, \
            f"Differenz sollte entweder beide Werte oder 'Nur in X vorhanden' enthalten: {diff}"


@given(config_data=config_dict_strategy())
@settings(max_examples=100, deadline=None)
def test_no_differences_for_identical_configs(config_data):
    """
    Test dass identische Konfigurationen keine Differenzen erzeugen.
    
    For any configuration, comparing it with itself should yield no differences.
    """
    # Erstelle Synchronizer
    synchronizer = ConfigSynchronizer('dummy.xlsx', 'dummy.json')
    
    # Erkenne Differenzen zwischen identischen Configs
    differences = synchronizer._detect_differences(config_data, config_data)
    
    # Es sollten keine Differenzen gefunden werden
    assert len(differences) == 0, \
        f"Identische Konfigurationen sollten keine Differenzen haben, aber gefunden: {differences}"


def _configs_equal_with_tolerance(config1: Dict[str, Any], config2: Dict[str, Any], rel_tol=1e-14) -> bool:
    """
    Vergleicht zwei Konfigurationen mit Toleranz für Float-Werte.
    
    Excel hat eine Präzisionsbeschränkung von ~15 signifikanten Stellen für Floats.
    Diese Funktion verwendet eine relative Toleranz für Float-Vergleiche.
    """
    import math
    
    def compare_values(v1, v2):
        """Vergleicht zwei Werte mit Toleranz für Floats"""
        if type(v1) != type(v2):
            # Spezialfall: int und float mit gleichem Wert
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                return math.isclose(float(v1), float(v2), rel_tol=rel_tol)
            return False
        
        if isinstance(v1, float):
            return math.isclose(v1, v2, rel_tol=rel_tol)
        elif isinstance(v1, dict):
            return compare_dicts(v1, v2)
        elif isinstance(v1, list):
            return compare_lists(v1, v2)
        else:
            return v1 == v2
    
    def compare_dicts(d1, d2):
        """Vergleicht zwei Dictionaries rekursiv"""
        if set(d1.keys()) != set(d2.keys()):
            return False
        return all(compare_values(d1[k], d2[k]) for k in d1.keys())
    
    def compare_lists(l1, l2):
        """Vergleicht zwei Listen rekursiv"""
        if len(l1) != len(l2):
            return False
        return all(compare_values(v1, v2) for v1, v2 in zip(l1, l2))
    
    return compare_dicts(config1, config2)


# Feature: config-json-migration, Property 3: Synchronisation stellt Konsistenz her
@given(config_pair=modified_config_pair_strategy())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
def test_synchronization_establishes_consistency(config_pair):
    """
    **Feature: config-json-migration, Property 3: Synchronisation stellt Konsistenz her**
    **Validates: Requirements 2.4, 2.5**
    
    For any pair of XLSX and JSON configurations, after synchronization
    (regardless of which source is chosen), both files should have identical contents
    (within Excel's precision limits for floating point numbers).
    
    Note: Excel can only store ~15 significant digits for floats, so we use
    tolerance-based comparison for float values.
    """
    import tempfile
    import shutil
    
    xlsx_data, json_data, _ = config_pair
    
    # Erstelle temporäres Verzeichnis
    tmp_dir = tempfile.mkdtemp()
    
    try:
        # Erstelle temporäre Dateien
        xlsx_path = Path(tmp_dir) / 'test_config.xlsx'
        json_path = Path(tmp_dir) / 'test_config.json'
        
        # Schreibe initiale Daten
        ConfigConverter.json_to_xlsx(xlsx_data, str(xlsx_path))
        ConfigConverter.save_json(json_data, str(json_path))
        
        # Erstelle Synchronizer
        synchronizer = ConfigSynchronizer(str(xlsx_path), str(json_path))
        
        # Test 1: Synchronisation mit XLSX als Quelle
        # Simuliere Benutzer wählt XLSX
        synchronizer._update_from_xlsx()
        
        # Lade beide Dateien
        xlsx_after_sync = ConfigConverter.xlsx_to_json(str(xlsx_path))
        json_after_sync = ConfigConverter.load_json(str(json_path))
        
        # Beide sollten jetzt identisch sein
        differences_after_xlsx_sync = synchronizer._detect_differences(xlsx_after_sync, json_after_sync)
        assert len(differences_after_xlsx_sync) == 0, \
            f"Nach Synchronisation von XLSX sollten keine Differenzen existieren, aber gefunden: {differences_after_xlsx_sync}"
        
        # Prüfe dass JSON jetzt die XLSX-Daten enthält (mit Toleranz für Floats)
        assert _configs_equal_with_tolerance(json_after_sync, xlsx_data), \
            f"JSON sollte nach Synchronisation die XLSX-Daten enthalten.\nErwartet: {xlsx_data}\nErhalten: {json_after_sync}"
        
        # Test 2: Synchronisation mit JSON als Quelle
        # Schreibe wieder unterschiedliche Daten
        ConfigConverter.json_to_xlsx(xlsx_data, str(xlsx_path))
        ConfigConverter.save_json(json_data, str(json_path))
        
        # Simuliere Benutzer wählt JSON
        synchronizer._update_from_json()
        
        # Lade beide Dateien
        xlsx_after_json_sync = ConfigConverter.xlsx_to_json(str(xlsx_path))
        json_after_json_sync = ConfigConverter.load_json(str(json_path))
        
        # Beide sollten jetzt identisch sein
        differences_after_json_sync = synchronizer._detect_differences(xlsx_after_json_sync, json_after_json_sync)
        assert len(differences_after_json_sync) == 0, \
            f"Nach Synchronisation von JSON sollten keine Differenzen existieren, aber gefunden: {differences_after_json_sync}"
        
        # Prüfe dass XLSX jetzt die JSON-Daten enthält (mit Toleranz für Floats)
        assert _configs_equal_with_tolerance(xlsx_after_json_sync, json_data), \
            f"XLSX sollte nach Synchronisation die JSON-Daten enthalten.\nErwartet: {json_data}\nErhalten: {xlsx_after_json_sync}"
        
    finally:
        # Räume temporäres Verzeichnis auf
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_automatic_synchronization_without_differences():
    """
    Test dass Synchronisation ohne Differenzen automatisch abläuft.
    
    Validates: Requirements 5.3
    
    When both XLSX and JSON exist with identical content, synchronization
    should complete automatically without user interaction and return 'json'.
    """
    import tempfile
    import shutil
    
    # Erstelle temporäres Verzeichnis
    tmp_dir = tempfile.mkdtemp()
    
    try:
        # Erstelle temporäre Dateien
        xlsx_path = Path(tmp_dir) / 'test_config.xlsx'
        json_path = Path(tmp_dir) / 'test_config.json'
        
        # Erstelle identische Konfigurationsdaten
        config_data = {
            'base_config': {
                'provider': 'openai',
                'model': 'gpt-4o-mini',
                'temperature': 0.7,
                'script_dir': '',
                'output_dir': 'output',
                'explore_file': 'test.xlsx',
                'clean_keywords': True,
                'similarity_threshold': 0.7
            },
            'analysis_configs': [
                {
                    'name': 'Test_Analysis',
                    'filters': {'Attribut_1': 'value1'},
                    'params': {'active': True, 'analysis_type': 'netzwerk'}
                }
            ]
        }
        
        # Schreibe identische Daten in beide Dateien
        ConfigConverter.json_to_xlsx(config_data, str(xlsx_path))
        ConfigConverter.save_json(config_data, str(json_path))
        
        # Erstelle Synchronizer
        synchronizer = ConfigSynchronizer(str(xlsx_path), str(json_path))
        
        # Führe Synchronisation durch
        # Dies sollte automatisch ablaufen ohne Benutzerinteraktion
        result = synchronizer.sync()
        
        # Ergebnis sollte 'json' sein (JSON wird bevorzugt)
        assert result == 'json', \
            f"Synchronisation ohne Differenzen sollte 'json' zurückgeben, aber erhielt: {result}"
        
        # Beide Dateien sollten noch existieren
        assert xlsx_path.exists(), "XLSX-Datei sollte nach Synchronisation existieren"
        assert json_path.exists(), "JSON-Datei sollte nach Synchronisation existieren"
        
        # Beide Dateien sollten noch identisch sein
        xlsx_after = ConfigConverter.xlsx_to_json(str(xlsx_path))
        json_after = ConfigConverter.load_json(str(json_path))
        
        differences = synchronizer._detect_differences(xlsx_after, json_after)
        assert len(differences) == 0, \
            f"Nach automatischer Synchronisation sollten keine Differenzen existieren: {differences}"
        
    finally:
        # Räume temporäres Verzeichnis auf
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    # Führe Tests aus
    import pytest
    pytest.main([__file__, '-v'])
