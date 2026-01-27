#!/usr/bin/env python3
"""
Property-Based Test for Test Mode Determinism
============================================
Property 18: Test Mode Determinism
Validates: Requirements 6.2

**Feature: dynamic-cache-system, Property 18: Test Mode Determinism**

This test validates that when test mode is activated, the cache system produces 
deterministic results for identical inputs across multiple test runs.
"""

import sys
import os
import time
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
import unittest
from unittest.mock import patch

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager
from QCA_AID_assets.optimization.cache import ModeAwareCache
from QCA_AID_assets.optimization.cache_debug_tools import TestMode, LogLevel


class TestProperty18TestModeDeterminism(unittest.TestCase):
    """
    Property-Based Test for Test Mode Determinism.
    
    **Feature: dynamic-cache-system, Property 18: Test Mode Determinism**
    **Validates: Requirements 6.2**
    
    Property: For any cache system with test mode activated, identical inputs 
    should produce identical results across multiple test runs.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = ModeAwareCache(max_size=100)
        self.manager = DynamicCacheManager(
            self.cache, 
            test_mode=True, 
            debug_level='DEBUG'
        )
        
        # Test data generators
        self.test_operations = [
            'relevance_check', 'category_development', 'coding', 'confidence_scoring'
        ]
        self.analysis_modes = ['deductive', 'inductive', 'abductive', 'grounded']
        self.coder_configs = [
            [{'coder_id': 'coder_1', 'temperature': 0.1}],
            [
                {'coder_id': 'coder_1', 'temperature': 0.1},
                {'coder_id': 'coder_2', 'temperature': 0.3}
            ],
            [
                {'coder_id': 'coder_1', 'temperature': 0.1},
                {'coder_id': 'coder_2', 'temperature': 0.3},
                {'coder_id': 'coder_3', 'temperature': 0.5}
            ]
        ]
    
    def generate_test_scenario(self, seed: int) -> Dict[str, Any]:
        """
        Generate a deterministic test scenario based on seed.
        
        Args:
            seed: Random seed for deterministic generation
            
        Returns:
            Test scenario dictionary
        """
        # Use seed for deterministic generation
        random.seed(seed)
        
        scenario = {
            'seed': seed,
            'coder_config': random.choice(self.coder_configs),
            'analysis_mode': random.choice(self.analysis_modes),
            'operations': []
        }
        
        # Generate 5-15 operations
        num_operations = random.randint(5, 15)
        
        for i in range(num_operations):
            operation = {
                'type': random.choice(self.test_operations),
                'segment_text': f'Test segment {i} with seed {seed}',
                'research_question': f'Research question {random.randint(1, 5)}',
                'category_definitions': {
                    f'cat_{j}': f'Definition {j} for seed {seed}'
                    for j in range(random.randint(1, 4))
                }
            }
            scenario['operations'].append(operation)
        
        return scenario
    
    def execute_test_scenario(self, scenario: Dict[str, Any], test_mode: str = "deterministic") -> Dict[str, Any]:
        """
        Execute a test scenario and collect results.
        
        Args:
            scenario: Test scenario to execute
            test_mode: Test mode to use
            
        Returns:
            Execution results
        """
        # Enable test mode with scenario seed
        test_result = self.manager.enable_test_mode(test_mode, seed=scenario['seed'])
        self.assertEqual(test_result['status'], 'success')
        
        # Configure analysis mode and coders
        self.manager.configure_analysis_mode(scenario['analysis_mode'])
        self.manager.configure_coders(scenario['coder_config'])
        
        # Execute operations and collect results
        results = {
            'cache_keys': [],
            'cache_operations': [],
            'strategy_info': {},
            'statistics': {},
            'test_mode_status': {}
        }
        
        # Record initial state
        results['strategy_info'] = self.manager.get_mode_specific_cache_rules()
        
        # Execute each operation
        for operation in scenario['operations']:
            # Generate cache keys based on operation type and strategy
            if self.manager.should_cache_shared(operation['type']):
                cache_key = self.manager.get_shared_cache_key(
                    operation['type'],
                    segment_text=operation['segment_text'],
                    research_question=operation['research_question'],
                    analysis_mode=scenario['analysis_mode']
                )
                results['cache_keys'].append(('shared', cache_key))
            
            # For multi-coder scenarios, also generate coder-specific keys
            if len(scenario['coder_config']) > 1 and self.manager.should_cache_per_coder(operation['type']):
                for coder_config in scenario['coder_config']:
                    coder_key = self.manager.get_coder_specific_key(
                        coder_config['coder_id'],
                        operation['type'],
                        segment_text=operation['segment_text'],
                        research_question=operation['research_question'],
                        analysis_mode=scenario['analysis_mode']
                    )
                    results['cache_keys'].append(('coder_specific', coder_key, coder_config['coder_id']))
            
            # Record operation details
            results['cache_operations'].append({
                'type': operation['type'],
                'shared': self.manager.should_cache_shared(operation['type']),
                'per_coder': self.manager.should_cache_per_coder(operation['type'])
            })
        
        # Collect final state
        results['statistics'] = self.manager.get_statistics().__dict__
        results['test_mode_status'] = self.manager.get_test_mode_status()
        
        # Disable test mode
        self.manager.disable_test_mode()
        
        return results
    
    def test_deterministic_mode_produces_identical_results(self):
        """
        Test that deterministic mode produces identical results for the same inputs.
        
        Property: For any test scenario with a fixed seed, running the scenario
        multiple times in deterministic mode should produce identical results.
        """
        # Generate test scenarios with different seeds
        test_seeds = [42, 123, 456, 789, 999]
        
        for seed in test_seeds:
            with self.subTest(seed=seed):
                # Generate scenario
                scenario = self.generate_test_scenario(seed)
                
                # Execute scenario multiple times
                results_run1 = self.execute_test_scenario(scenario, "deterministic")
                results_run2 = self.execute_test_scenario(scenario, "deterministic")
                results_run3 = self.execute_test_scenario(scenario, "deterministic")
                
                # Verify identical results
                self.assertEqual(
                    results_run1['cache_keys'], 
                    results_run2['cache_keys'],
                    f"Cache keys differ between runs for seed {seed}"
                )
                
                self.assertEqual(
                    results_run1['cache_keys'], 
                    results_run3['cache_keys'],
                    f"Cache keys differ between runs for seed {seed}"
                )
                
                self.assertEqual(
                    results_run1['cache_operations'], 
                    results_run2['cache_operations'],
                    f"Cache operations differ between runs for seed {seed}"
                )
                
                self.assertEqual(
                    results_run1['strategy_info'], 
                    results_run2['strategy_info'],
                    f"Strategy info differs between runs for seed {seed}"
                )
                
                # Verify test mode was active
                self.assertTrue(results_run1['test_mode_status']['active'])
                self.assertEqual(results_run1['test_mode_status']['info']['mode'], 'deterministic')
                self.assertEqual(results_run1['test_mode_status']['info']['seed'], seed)
    
    def test_different_seeds_produce_different_results(self):
        """
        Test that different seeds produce different results (to verify randomness works).
        
        Property: For any two different seeds, the generated test scenarios should
        be different, demonstrating that the deterministic mode is actually using
        the seed for randomization.
        """
        seed1, seed2 = 42, 123
        
        # Generate scenarios with different seeds
        scenario1 = self.generate_test_scenario(seed1)
        scenario2 = self.generate_test_scenario(seed2)
        
        # Scenarios should be different (at least in some aspects)
        scenarios_different = (
            scenario1['coder_config'] != scenario2['coder_config'] or
            scenario1['analysis_mode'] != scenario2['analysis_mode'] or
            len(scenario1['operations']) != len(scenario2['operations']) or
            scenario1['operations'] != scenario2['operations']
        )
        
        self.assertTrue(
            scenarios_different,
            "Different seeds should produce different test scenarios"
        )
        
        # Execute both scenarios
        results1 = self.execute_test_scenario(scenario1, "deterministic")
        results2 = self.execute_test_scenario(scenario2, "deterministic")
        
        # Results should be different (unless by extreme coincidence)
        results_different = (
            results1['cache_keys'] != results2['cache_keys'] or
            results1['cache_operations'] != results2['cache_operations'] or
            results1['strategy_info'] != results2['strategy_info']
        )
        
        self.assertTrue(
            results_different,
            "Different seeds should produce different execution results"
        )
    
    def test_deterministic_mode_with_cache_operations(self):
        """
        Test deterministic behavior with actual cache operations.
        
        Property: For any sequence of cache operations with deterministic mode,
        the cache state and hit/miss patterns should be identical across runs.
        """
        seed = 555
        scenario = self.generate_test_scenario(seed)
        
        def execute_with_cache_operations():
            """Execute scenario with actual cache operations."""
            # Enable deterministic mode
            self.manager.enable_test_mode("deterministic", seed=seed)
            
            # Configure system
            self.manager.configure_analysis_mode(scenario['analysis_mode'])
            self.manager.configure_coders(scenario['coder_config'])
            
            # Clear cache to start fresh
            self.cache.clear()
            
            cache_results = []
            
            # Execute operations with cache interactions
            for i, operation in enumerate(scenario['operations']):
                # Simulate cache lookup and store
                if self.manager.should_cache_shared(operation['type']):
                    cache_key = self.manager.get_shared_cache_key(
                        operation['type'],
                        segment_text=operation['segment_text'],
                        analysis_mode=scenario['analysis_mode']
                    )
                    
                    # Check if in cache (hit/miss)
                    is_hit = cache_key in self.cache.cache
                    
                    # Store mock result if not in cache
                    if not is_hit:
                        self.cache.set(
                            analysis_mode=scenario['analysis_mode'],
                            segment_text=operation['segment_text'],
                            value={'mock_result': f'result_{i}', 'seed': seed},
                            operation_type=operation['type']
                        )
                    
                    cache_results.append({
                        'operation_index': i,
                        'cache_key': cache_key,
                        'hit': is_hit,
                        'cache_size_after': len(self.cache.cache)
                    })
            
            # Get final statistics
            final_stats = self.manager.get_statistics()
            
            self.manager.disable_test_mode()
            
            return {
                'cache_results': cache_results,
                'final_cache_size': len(self.cache.cache),
                'final_stats': final_stats.__dict__
            }
        
        # Execute multiple times
        run1_results = execute_with_cache_operations()
        run2_results = execute_with_cache_operations()
        run3_results = execute_with_cache_operations()
        
        # Verify identical cache behavior
        self.assertEqual(
            run1_results['cache_results'],
            run2_results['cache_results'],
            "Cache hit/miss patterns should be identical in deterministic mode"
        )
        
        self.assertEqual(
            run1_results['cache_results'],
            run3_results['cache_results'],
            "Cache hit/miss patterns should be identical in deterministic mode"
        )
        
        self.assertEqual(
            run1_results['final_cache_size'],
            run2_results['final_cache_size'],
            "Final cache size should be identical in deterministic mode"
        )
    
    def test_record_and_replay_mode_determinism(self):
        """
        Test that record and replay modes produce deterministic behavior.
        
        Property: For any recorded session, replaying it should produce
        identical cache operations and results.
        """
        seed = 777
        scenario = self.generate_test_scenario(seed)
        
        # Create temporary file for recording
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            recording_file = temp_file.name
        
        try:
            # Record session
            self.manager.enable_test_mode("record", recording_file=recording_file)
            
            # Configure and execute operations
            self.manager.configure_analysis_mode(scenario['analysis_mode'])
            self.manager.configure_coders(scenario['coder_config'])
            
            recorded_keys = []
            for operation in scenario['operations'][:3]:  # Use first 3 operations
                if self.manager.should_cache_shared(operation['type']):
                    cache_key = self.manager.get_shared_cache_key(
                        operation['type'],
                        segment_text=operation['segment_text']
                    )
                    recorded_keys.append(cache_key)
            
            # Stop recording
            self.manager.disable_test_mode()
            
            # Verify recording file exists and has content
            self.assertTrue(os.path.exists(recording_file))
            
            # Replay session multiple times
            replay_results = []
            for replay_run in range(2):
                self.manager.enable_test_mode("replay", recording_file=recording_file)
                
                # Configure system (should match recorded configuration)
                self.manager.configure_analysis_mode(scenario['analysis_mode'])
                self.manager.configure_coders(scenario['coder_config'])
                
                replayed_keys = []
                for operation in scenario['operations'][:3]:
                    if self.manager.should_cache_shared(operation['type']):
                        cache_key = self.manager.get_shared_cache_key(
                            operation['type'],
                            segment_text=operation['segment_text']
                        )
                        replayed_keys.append(cache_key)
                
                replay_results.append(replayed_keys)
                self.manager.disable_test_mode()
            
            # Verify replay produces identical results
            self.assertEqual(
                recorded_keys,
                replay_results[0],
                "First replay should match recorded session"
            )
            
            self.assertEqual(
                replay_results[0],
                replay_results[1],
                "Multiple replays should produce identical results"
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(recording_file):
                os.unlink(recording_file)
    
    def test_test_mode_isolation(self):
        """
        Test that test mode doesn't affect normal operation.
        
        Property: For any cache operations, enabling and disabling test mode
        should not affect the normal cache behavior when test mode is disabled.
        """
        seed = 888
        scenario = self.generate_test_scenario(seed)
        
        # Execute operations in normal mode
        self.manager.configure_analysis_mode(scenario['analysis_mode'])
        self.manager.configure_coders(scenario['coder_config'])
        
        normal_keys = []
        for operation in scenario['operations'][:3]:
            if self.manager.should_cache_shared(operation['type']):
                cache_key = self.manager.get_shared_cache_key(
                    operation['type'],
                    segment_text=operation['segment_text']
                )
                normal_keys.append(cache_key)
        
        # Enable and disable test mode
        self.manager.enable_test_mode("deterministic", seed=seed)
        test_mode_status = self.manager.get_test_mode_status()
        self.assertTrue(test_mode_status['active'])
        
        self.manager.disable_test_mode()
        test_mode_status = self.manager.get_test_mode_status()
        self.assertFalse(test_mode_status['active'])
        
        # Execute same operations again in normal mode
        after_test_keys = []
        for operation in scenario['operations'][:3]:
            if self.manager.should_cache_shared(operation['type']):
                cache_key = self.manager.get_shared_cache_key(
                    operation['type'],
                    segment_text=operation['segment_text']
                )
                after_test_keys.append(cache_key)
        
        # Results should be identical (test mode shouldn't affect normal operation)
        self.assertEqual(
            normal_keys,
            after_test_keys,
            "Test mode should not affect normal cache operation"
        )
    
    def test_comprehensive_determinism_property(self):
        """
        Comprehensive property test for deterministic behavior.
        
        Property: For any cache system configuration and any sequence of operations,
        when test mode is enabled with the same seed, all aspects of cache behavior
        should be deterministic and reproducible.
        """
        # Test multiple scenarios to increase confidence
        test_cases = [
            {'seed': 100, 'iterations': 3},
            {'seed': 200, 'iterations': 3},
            {'seed': 300, 'iterations': 3}
        ]
        
        for test_case in test_cases:
            with self.subTest(seed=test_case['seed']):
                seed = test_case['seed']
                scenario = self.generate_test_scenario(seed)
                
                # Collect results from multiple iterations
                all_results = []
                
                for iteration in range(test_case['iterations']):
                    # Enable deterministic mode
                    self.manager.enable_test_mode("deterministic", seed=seed)
                    
                    # Configure system
                    self.manager.configure_analysis_mode(scenario['analysis_mode'])
                    self.manager.configure_coders(scenario['coder_config'])
                    
                    # Clear cache for consistent starting state
                    self.cache.clear()
                    
                    # Execute operations and collect comprehensive results
                    iteration_results = {
                        'cache_keys': [],
                        'strategy_decisions': [],
                        'cache_states': []
                    }
                    
                    for op_index, operation in enumerate(scenario['operations']):
                        # Record strategy decisions
                        shared_decision = self.manager.should_cache_shared(operation['type'])
                        per_coder_decision = self.manager.should_cache_per_coder(operation['type'])
                        
                        iteration_results['strategy_decisions'].append({
                            'operation': operation['type'],
                            'shared': shared_decision,
                            'per_coder': per_coder_decision
                        })
                        
                        # Generate and record cache keys
                        if shared_decision:
                            shared_key = self.manager.get_shared_cache_key(
                                operation['type'],
                                segment_text=operation['segment_text']
                            )
                            iteration_results['cache_keys'].append(('shared', shared_key))
                        
                        if per_coder_decision and len(scenario['coder_config']) > 1:
                            for coder_config in scenario['coder_config']:
                                coder_key = self.manager.get_coder_specific_key(
                                    coder_config['coder_id'],
                                    operation['type'],
                                    segment_text=operation['segment_text']
                                )
                                iteration_results['cache_keys'].append(
                                    ('coder', coder_key, coder_config['coder_id'])
                                )
                        
                        # Record cache state
                        iteration_results['cache_states'].append({
                            'operation_index': op_index,
                            'cache_size': len(self.cache.cache),
                            'strategy_name': self.manager.current_strategy.name if self.manager.current_strategy else None
                        })
                    
                    all_results.append(iteration_results)
                    self.manager.disable_test_mode()
                
                # Verify all iterations produced identical results
                first_result = all_results[0]
                for i, result in enumerate(all_results[1:], 1):
                    self.assertEqual(
                        first_result['cache_keys'],
                        result['cache_keys'],
                        f"Cache keys differ between iteration 0 and {i} for seed {seed}"
                    )
                    
                    self.assertEqual(
                        first_result['strategy_decisions'],
                        result['strategy_decisions'],
                        f"Strategy decisions differ between iteration 0 and {i} for seed {seed}"
                    )
                    
                    self.assertEqual(
                        first_result['cache_states'],
                        result['cache_states'],
                        f"Cache states differ between iteration 0 and {i} for seed {seed}"
                    )


def run_property_18_tests():
    """Run Property 18: Test Mode Determinism tests."""
    print("=" * 80)
    print("PROPERTY 18: TEST MODE DETERMINISM")
    print("Feature: dynamic-cache-system")
    print("Validates: Requirements 6.2")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestProperty18TestModeDeterminism))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("PROPERTY TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nPROPERTY 18 RESULT: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_property_18_tests()
    sys.exit(0 if success else 1)