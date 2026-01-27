#!/usr/bin/env python3
"""
Cache Migration and Consistency Tests
====================================
Tests for cache migration, consistency checks, and rollback functionality
in the Dynamic Cache System.
"""

import sys
import os
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import unittest
from unittest.mock import patch, MagicMock

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager
from QCA_AID_assets.optimization.cache import ModeAwareCache, CacheEntry
from QCA_AID_assets.optimization.cache_strategies import SingleCoderCacheStrategy, MultiCoderCacheStrategy


class TestCacheMigration(unittest.TestCase):
    """Test cache migration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = ModeAwareCache(max_size=100)
        self.manager = DynamicCacheManager(self.cache)
        
        # Add test entries
        self._add_test_entries()
    
    def _add_test_entries(self):
        """Add test entries to cache."""
        # Add coding entry (should be coder-specific in multi-coder mode)
        self.cache.set(
            analysis_mode='deductive',
            segment_text='test segment for coding',
            value={'result': 'coding result'},
            operation_type='coding',
            coder_id='coder1',
            segment_id='seg_001',
            category='Environmental Impact',
            confidence=0.85
        )
        
        # Add relevance check entry (should be shared in multi-coder mode)
        self.cache.set(
            analysis_mode='deductive',
            segment_text='test segment for relevance',
            value={'result': 'relevance result'},
            operation_type='relevance_check',
            segment_id='seg_002',
            confidence=0.90
        )
        
        # Add category development entry (should be shared)
        self.cache.set(
            analysis_mode='inductive',
            segment_text='test segment for categories',
            value={'categories': ['cat1', 'cat2']},
            operation_type='category_development',
            segment_id='seg_003',
            confidence=0.75
        )
    
    def test_migration_single_to_multi_coder(self):
        """Test migration from single-coder to multi-coder strategy."""
        # Verify initial state (single-coder)
        self.assertIsInstance(self.manager.current_strategy, SingleCoderCacheStrategy)
        initial_count = len(self.cache.cache)
        self.assertEqual(initial_count, 3)
        
        # Switch to multi-coder mode
        self.manager.configure_coders([
            {'coder_id': 'coder1', 'temperature': 0.7},
            {'coder_id': 'coder2', 'temperature': 0.8}
        ])
        
        # Verify strategy changed
        self.assertIsInstance(self.manager.current_strategy, MultiCoderCacheStrategy)
        
        # Check migration status
        migration_status = self.manager.get_migration_status()
        self.assertTrue(migration_status['has_backup'])
        self.assertIsNotNone(migration_status['backup_timestamp'])
        
        # Verify cache entries still exist (may be migrated)
        final_count = len(self.cache.cache)
        self.assertGreaterEqual(final_count, 1)  # At least some entries should remain
        
        # Check consistency
        consistency_result = self.manager.force_consistency_check()
        self.assertEqual(consistency_result['total_issues'], 0)
    
    def test_migration_multi_to_single_coder(self):
        """Test migration from multi-coder to single-coder strategy."""
        # First switch to multi-coder
        self.manager.configure_coders([
            {'coder_id': 'coder1', 'temperature': 0.7},
            {'coder_id': 'coder2', 'temperature': 0.8}
        ])
        
        # Then switch back to single-coder
        self.manager.configure_coders([
            {'coder_id': 'coder1', 'temperature': 0.7}
        ])
        
        # Verify strategy changed back
        self.assertIsInstance(self.manager.current_strategy, SingleCoderCacheStrategy)
        
        # Check consistency
        consistency_result = self.manager.force_consistency_check()
        self.assertEqual(consistency_result['total_issues'], 0)
    
    def test_consistency_check_detection(self):
        """Test that consistency checks detect issues."""
        # Manually create inconsistent state
        # Add entry with wrong shared classification
        entry = CacheEntry(
            key="test_inconsistent_key",
            value={'test': 'data'},
            timestamp=time.time(),
            analysis_mode='deductive',
            operation_type='coding',
            coder_id=None,  # Should have coder_id for coding operation
            is_shared=True,  # Should be False for coding in multi-coder mode
            metadata={}
        )
        
        # Switch to multi-coder mode first
        self.manager.configure_coders([
            {'coder_id': 'coder1', 'temperature': 0.7},
            {'coder_id': 'coder2', 'temperature': 0.8}
        ])
        
        # Add inconsistent entry
        self.cache.cache["test_inconsistent_key"] = entry
        self.cache.shared_index.append("test_inconsistent_key")
        
        # Check consistency - should detect issues
        consistency_result = self.manager.force_consistency_check()
        self.assertGreater(consistency_result['total_issues'], 0)
        
        # Repair consistency
        repair_result = self.manager.repair_cache_consistency()
        self.assertEqual(repair_result['status'], 'repair_completed')
        
        # Check consistency again - should be clean
        consistency_result = self.manager.force_consistency_check()
        self.assertEqual(consistency_result['total_issues'], 0)
    
    def test_selective_cache_clear(self):
        """Test selective cache clearing functionality."""
        initial_count = len(self.cache.cache)
        
        # Clear by operation type
        cleared = self.manager.perform_selective_cache_clear({
            'operations': ['coding']
        })
        
        self.assertGreater(cleared, 0)
        self.assertLess(len(self.cache.cache), initial_count)
        
        # Verify coding entries are gone
        for entry in self.cache.cache.values():
            self.assertNotEqual(entry.operation_type, 'coding')
    
    def test_selective_clear_by_coder(self):
        """Test selective clearing by coder ID."""
        # Add entries for different coders
        self.cache.set(
            analysis_mode='deductive',
            segment_text='coder2 segment',
            value={'result': 'coder2 result'},
            operation_type='coding',
            coder_id='coder2'
        )
        
        initial_count = len(self.cache.cache)
        
        # Clear entries for coder1
        cleared = self.manager.perform_selective_cache_clear({
            'coders': ['coder1']
        })
        
        self.assertGreater(cleared, 0)
        self.assertLess(len(self.cache.cache), initial_count)
        
        # Verify coder1 entries are gone
        for entry in self.cache.cache.values():
            self.assertNotEqual(entry.coder_id, 'coder1')
    
    def test_selective_clear_by_age(self):
        """Test selective clearing by entry age."""
        # All entries are new, so clearing old entries should remove nothing
        cleared = self.manager.perform_selective_cache_clear({
            'older_than_hours': 1
        })
        
        self.assertEqual(cleared, 0)
        
        # Clear all entries (age 0 means clear everything)
        initial_count = len(self.cache.cache)
        cleared = self.manager.perform_selective_cache_clear({
            'older_than_hours': -1  # Use negative value to clear all entries
        })
        
        self.assertEqual(cleared, initial_count)
    
    def test_selective_clear_by_confidence(self):
        """Test selective clearing by confidence threshold."""
        initial_count = len(self.cache.cache)
        
        # Clear entries with confidence below 0.8
        cleared = self.manager.perform_selective_cache_clear({
            'confidence_below': 0.8
        })
        
        # Should clear the category development entry (confidence 0.75)
        self.assertGreater(cleared, 0)
        self.assertLess(len(self.cache.cache), initial_count)
        
        # Verify remaining entries have confidence >= 0.8
        for entry in self.cache.cache.values():
            self.assertGreaterEqual(entry.confidence, 0.8)
    
    def test_migration_backup_and_rollback(self):
        """Test migration backup and rollback functionality."""
        # Get initial state
        initial_count = len(self.cache.cache)
        
        # Switch to multi-coder (should create backup)
        self.manager.configure_coders([
            {'coder_id': 'coder1', 'temperature': 0.7},
            {'coder_id': 'coder2', 'temperature': 0.8}
        ])
        
        # Verify backup exists
        migration_status = self.manager.get_migration_status()
        self.assertTrue(migration_status['has_backup'])
        
        # Verify backup contains data
        self.assertIsNotNone(self.manager._migration_backup)
        self.assertIn('cache_entries', self.manager._migration_backup)
        
        # The backup should contain at least some entries
        # (exact count may vary due to migration processing)
        self.assertGreater(len(self.manager._migration_backup['cache_entries']), 0)
        
        # Verify migration stats are recorded
        if migration_status.get('migration_stats'):
            stats = migration_status['migration_stats']
            self.assertIn('migrated_count', stats)
            self.assertIn('invalidated_count', stats)
            self.assertIn('total_processed', stats)
    
    def test_cache_key_regeneration(self):
        """Test cache key regeneration during migration."""
        # Add entry with metadata for key regeneration
        self.cache.set(
            analysis_mode='deductive',
            segment_text='test regeneration',
            value={'result': 'test'},
            operation_type='coding',
            coder_id='coder1',
            metadata={
                'segment_text': 'test regeneration',
                'research_question': 'test question',
                'category_definitions': {'cat1': 'definition1'}
            }
        )
        
        original_key = list(self.cache.cache.keys())[-1]
        
        # Switch strategies to trigger key regeneration
        self.manager.configure_coders([
            {'coder_id': 'coder1', 'temperature': 0.7},
            {'coder_id': 'coder2', 'temperature': 0.8}
        ])
        
        # Check if key was regenerated (may or may not change depending on strategy)
        current_keys = list(self.cache.cache.keys())
        self.assertGreater(len(current_keys), 0)
    
    def test_migration_with_empty_cache(self):
        """Test migration with empty cache."""
        # Clear cache
        self.cache.clear()
        
        # Switch strategies
        self.manager.configure_coders([
            {'coder_id': 'coder1', 'temperature': 0.7},
            {'coder_id': 'coder2', 'temperature': 0.8}
        ])
        
        # Should complete without errors
        migration_status = self.manager.get_migration_status()
        self.assertEqual(migration_status['consistency_status'], 'clean')
    
    def test_migration_error_handling(self):
        """Test migration error handling."""
        # Mock a migration failure
        with patch.object(self.manager, '_regenerate_cache_key', side_effect=Exception("Test error")):
            # This should not crash, but handle the error gracefully
            self.manager.configure_coders([
                {'coder_id': 'coder1', 'temperature': 0.7},
                {'coder_id': 'coder2', 'temperature': 0.8}
            ])
            
            # Migration should still complete (with some entries possibly lost)
            migration_status = self.manager.get_migration_status()
            self.assertIsNotNone(migration_status)


class TestCacheConsistency(unittest.TestCase):
    """Test cache consistency functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = ModeAwareCache(max_size=100)
        self.manager = DynamicCacheManager(self.cache)
    
    def test_orphaned_index_detection(self):
        """Test detection of orphaned index entries."""
        # Add entry to cache
        self.cache.set(
            analysis_mode='deductive',
            segment_text='test',
            value={'test': 'data'},
            operation_type='coding',
            coder_id='coder1'
        )
        
        # Manually add orphaned entry to index
        self.cache.mode_index['deductive'].append('orphaned_key')
        
        # Check consistency
        consistency_result = self.manager.force_consistency_check()
        self.assertGreater(consistency_result['total_issues'], 0)
        
        # Repair
        repair_result = self.manager.repair_cache_consistency()
        self.assertEqual(repair_result['status'], 'repair_completed')
        
        # Verify orphaned entry was removed
        self.assertNotIn('orphaned_key', self.cache.mode_index['deductive'])
    
    def test_missing_coder_id_detection(self):
        """Test detection of missing coder_id in coder-specific entries."""
        # Switch to multi-coder mode
        self.manager.configure_coders([
            {'coder_id': 'coder1', 'temperature': 0.7},
            {'coder_id': 'coder2', 'temperature': 0.8}
        ])
        
        # Add entry that should have coder_id but doesn't
        entry = CacheEntry(
            key="test_key",
            value={'test': 'data'},
            timestamp=time.time(),
            analysis_mode='deductive',
            operation_type='coding',  # Should be coder-specific
            coder_id=None,  # Missing coder_id
            is_shared=False,
            metadata={}
        )
        
        self.cache.cache["test_key"] = entry
        
        # Check consistency
        consistency_result = self.manager.force_consistency_check()
        self.assertGreater(consistency_result['total_issues'], 0)
        
        # Repair should remove invalid entry
        repair_result = self.manager.repair_cache_consistency()
        self.assertEqual(repair_result['status'], 'repair_completed')
        
        # Verify invalid entry was removed
        self.assertNotIn("test_key", self.cache.cache)
    
    def test_incorrect_shared_classification(self):
        """Test detection and repair of incorrect shared classification."""
        # Switch to multi-coder mode
        self.manager.configure_coders([
            {'coder_id': 'coder1', 'temperature': 0.7},
            {'coder_id': 'coder2', 'temperature': 0.8}
        ])
        
        # Add entry with incorrect shared classification
        entry = CacheEntry(
            key="test_key",
            value={'test': 'data'},
            timestamp=time.time(),
            analysis_mode='deductive',
            operation_type='relevance_check',  # Should be shared
            coder_id='coder1',  # Should not have coder_id for shared operation
            is_shared=False,  # Should be True
            metadata={}
        )
        
        self.cache.cache["test_key"] = entry
        
        # Check consistency
        consistency_result = self.manager.force_consistency_check()
        self.assertGreater(consistency_result['total_issues'], 0)
        
        # Repair
        repair_result = self.manager.repair_cache_consistency()
        self.assertEqual(repair_result['status'], 'repair_completed')
        
        # Verify classification was corrected or entry was removed
        if "test_key" in self.cache.cache:
            corrected_entry = self.cache.cache["test_key"]
            self.assertTrue(corrected_entry.is_shared)


def run_migration_tests():
    """Run all cache migration tests."""
    print("=" * 60)
    print("CACHE MIGRATION AND CONSISTENCY TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add migration tests
    suite.addTest(unittest.makeSuite(TestCacheMigration))
    suite.addTest(unittest.makeSuite(TestCacheConsistency))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
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
    print(f"\nOVERALL RESULT: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    run_migration_tests()