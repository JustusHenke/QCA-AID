"""
Test Manual Coder Integration
============================
Tests for manual coder integration with the dynamic cache system.
Validates cache isolation, reliability data storage, and monitoring.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import pytest
import tempfile
from datetime import datetime
from typing import List, Dict, Any

from QCA_AID_assets.optimization.dynamic_cache_manager import DynamicCacheManager
from QCA_AID_assets.optimization.cache import ModeAwareCache
from QCA_AID_assets.core.data_models import ExtendedCodingResult


class TestManualCoderIntegration:
    """Test suite for manual coder integration functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Create test cache and manager
        self.base_cache = ModeAwareCache(max_size=100)
        self.cache_manager = DynamicCacheManager(
            base_cache=self.base_cache,
            reliability_db_path=self.temp_db.name,
            test_mode=True
        )
        
        # Configure with mixed coders
        self.cache_manager.configure_coders([
            {'coder_id': 'auto_1', 'temperature': 0.7},
            {'coder_id': 'auto_2', 'temperature': 0.9},
            {'coder_id': 'manual', 'temperature': 0.0}  # Manual coder
        ])
    
    def teardown_method(self):
        """Clean up test environment."""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_manual_coding_storage(self):
        """Test that manual codings are stored correctly in reliability database."""
        # Store manual coding
        self.cache_manager.store_manual_coding(
            segment_id="test_segment_1",
            category="Test Category",
            subcategories=["Sub1", "Sub2"],
            justification="Manual test coding",
            confidence=0.9,
            analysis_mode="inductive"
        )
        
        # Retrieve and verify
        manual_codings = self.cache_manager.get_manual_codings(["test_segment_1"])
        
        assert len(manual_codings) == 1
        assert manual_codings[0].coder_id == "manual"
        assert manual_codings[0].is_manual == True
        assert manual_codings[0].category == "Test Category"
        assert manual_codings[0].subcategories == ["Sub1", "Sub2"]
    
    def test_manual_cache_isolation(self):
        """Test that manual coder cache entries are properly isolated."""
        # Generate cache keys for different operations
        manual_coding_key = self.cache_manager.get_coder_specific_key(
            "manual", "coding", segment_text="test"
        )
        auto_coding_key = self.cache_manager.get_coder_specific_key(
            "auto_1", "coding", segment_text="test"
        )
        
        # Keys should be different for coder-specific operations
        assert manual_coding_key != auto_coding_key
        
        # Shared operations should generate same keys
        manual_shared_key = self.cache_manager.get_shared_cache_key(
            "relevance_check", segment_text="test"
        )
        auto_shared_key = self.cache_manager.get_shared_cache_key(
            "relevance_check", segment_text="test"
        )
        
        assert manual_shared_key == auto_shared_key
    
    def test_manual_coder_statistics(self):
        """Test that manual coder statistics are tracked correctly."""
        # Add some manual codings
        for i in range(3):
            self.cache_manager.store_manual_coding(
                segment_id=f"test_segment_{i}",
                category=f"Category_{i}",
                subcategories=[],
                justification="Test",
                confidence=0.8,
                analysis_mode="inductive"
            )
        
        # Add some automatic codings
        for i in range(2):
            auto_result = ExtendedCodingResult(
                segment_id=f"test_segment_{i}",
                coder_id="auto_1",
                category=f"Auto_Category_{i}",
                subcategories=[],
                confidence=0.7,
                justification="Auto test",
                analysis_mode="inductive",
                timestamp=datetime.now(),
                is_manual=False
            )
            self.cache_manager.store_for_reliability(auto_result)
        
        # Get reliability summary
        summary = self.cache_manager.get_reliability_summary()
        
        assert summary['manual_coder_stats']['total_manual_codings'] == 3
        assert summary['manual_coder_stats']['total_automatic_codings'] == 2
        assert summary['manual_coder_stats']['manual_percentage'] == 60.0  # 3/5 * 100
        assert 'manual' in summary['manual_coder_stats']['manual_coders']
    
    def test_mixed_coder_reliability_analysis(self):
        """Test reliability analysis with mixed manual and automatic coders."""
        # Create test segments with both manual and automatic codings
        test_segments = ["seg_1", "seg_2", "seg_3"]
        
        # Add manual codings
        for seg_id in test_segments:
            self.cache_manager.store_manual_coding(
                segment_id=seg_id,
                category="Manual_Category",
                subcategories=[],
                justification="Manual coding",
                confidence=0.9,
                analysis_mode="inductive"
            )
        
        # Add automatic codings for first two segments
        for seg_id in test_segments[:2]:
            auto_result = ExtendedCodingResult(
                segment_id=seg_id,
                coder_id="auto_1",
                category="Manual_Category",  # Same category for agreement
                subcategories=[],
                confidence=0.8,
                justification="Auto coding",
                analysis_mode="inductive",
                timestamp=datetime.now(),
                is_manual=False
            )
            self.cache_manager.store_for_reliability(auto_result)
        
        # Test manual + auto coder combination
        test_results = self.cache_manager.test_manual_auto_coder_combination(test_segments)
        
        assert test_results['manual_codings'] == 3
        assert test_results['automatic_codings'] == 2
        assert test_results['segments_with_both'] == 2
        assert test_results['cache_isolation_verified'] == True
        
        # Check reliability metrics
        reliability_metrics = test_results['reliability_metrics']
        assert reliability_metrics['agreement_rate'] == 1.0  # Perfect agreement
        assert reliability_metrics['total_comparisons'] == 2
    
    def test_manual_coder_monitoring_dashboard(self):
        """Test that manual coder information appears in monitoring dashboard."""
        # Add some manual codings
        self.cache_manager.store_manual_coding(
            segment_id="dashboard_test",
            category="Dashboard Category",
            subcategories=[],
            justification="Dashboard test",
            confidence=0.85,
            analysis_mode="inductive"
        )
        
        # Get dashboard data
        dashboard_data = self.cache_manager.get_monitoring_dashboard_data()
        
        assert 'manual_coder_info' in dashboard_data
        assert dashboard_data['key_metrics']['manual_coders'] >= 1
        assert dashboard_data['key_metrics']['manual_codings'] >= 1
        assert 'manual_coder_integration' in dashboard_data['performance_indicators']
        
        # Check manual coder specific info
        manual_info = dashboard_data['manual_coder_info']
        assert 'cache_isolation_verified' in manual_info
        assert 'manual_percentage' in manual_info
    
    def test_cache_behavior_with_mixed_coders(self):
        """Test cache behavior with mixed manual and automatic coders."""
        # Test cache behavior
        cache_test = self.cache_manager._test_mixed_coder_cache_behavior()
        
        assert cache_test['manual_entries_isolated'] == True
        assert cache_test['shared_operations_work'] == True
        assert cache_test['coder_specific_operations_separated'] == True
        assert cache_test['cache_key_generation_correct'] == True
        assert len(cache_test['errors']) == 0
    
    def test_manual_coder_cache_statistics(self):
        """Test manual coder specific cache statistics."""
        # Get cache statistics
        stats = self.cache_manager.get_statistics()
        
        # Should have manual_coder_stats field
        assert hasattr(stats, 'manual_coder_stats')
        assert isinstance(stats.manual_coder_stats, dict)
        
        # Check expected fields
        manual_stats = stats.manual_coder_stats
        expected_fields = [
            'manual_cache_entries',
            'manual_hit_rate', 
            'manual_operations',
            'manual_memory_usage_mb',
            'isolated_from_automatic'
        ]
        
        for field in expected_fields:
            assert field in manual_stats


if __name__ == "__main__":
    # Run basic tests
    test_suite = TestManualCoderIntegration()
    
    print("Running Manual Coder Integration Tests...")
    
    try:
        test_suite.setup_method()
        
        # Run individual tests
        test_methods = [
            'test_manual_coding_storage',
            'test_manual_cache_isolation', 
            'test_manual_coder_statistics',
            'test_mixed_coder_reliability_analysis',
            'test_manual_coder_monitoring_dashboard',
            'test_cache_behavior_with_mixed_coders',
            'test_manual_coder_cache_statistics'
        ]
        
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            try:
                print(f"  Running {test_method}...")
                getattr(test_suite, test_method)()
                print(f"  ✓ {test_method} PASSED")
                passed += 1
            except Exception as e:
                print(f"  ✗ {test_method} FAILED: {e}")
                failed += 1
            finally:
                # Reset for next test
                test_suite.teardown_method()
                test_suite.setup_method()
        
        test_suite.teardown_method()
        
        print(f"\nTest Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("✅ All manual coder integration tests passed!")
        else:
            print(f"❌ {failed} tests failed")
            
    except Exception as e:
        print(f"Test setup failed: {e}")
        test_suite.teardown_method()