"""
Metrics Collector for API Call Optimization
Tracks API calls, processing time, and quality metrics for optimization validation.
"""

import time
import json
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class APICallMetrics:
    """Metrics for a single API call."""
    timestamp: str
    analysis_mode: str
    call_type: str  # 'relevance', 'coding', 'category_development', etc.
    segment_ids: List[str]
    tokens_used: int
    processing_time_ms: float
    batch_size: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class SegmentMetrics:
    """Metrics for a single segment processing."""
    segment_id: str
    analysis_mode: str
    total_api_calls: int
    total_tokens: int
    total_processing_time_ms: float
    quality_score: Optional[float] = None
    confidence_score: Optional[float] = None


@dataclass
class BatchMetrics:
    """Metrics for a batch processing."""
    batch_id: str
    analysis_mode: str
    segment_count: int
    total_api_calls: int
    total_tokens: int
    total_processing_time_ms: float
    avg_processing_time_per_segment_ms: float
    api_calls_per_segment: float
    tokens_per_segment: float


class MetricsCollector:
    """Collects and analyzes metrics for API call optimization."""
    
    def __init__(self, output_dir: str = "optimization_metrics"):
        """Initialize metrics collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.api_call_metrics: List[APICallMetrics] = []
        self.segment_metrics: Dict[str, SegmentMetrics] = {}
        self.batch_metrics: List[BatchMetrics] = []
        
        # Current processing context
        self.current_batch_id: Optional[str] = None
        self.current_analysis_mode: Optional[str] = None
        self.current_segments: List[str] = []
        
        # Performance tracking
        self.start_time: Optional[float] = None
        self.batch_start_time: Optional[float] = None
        
    def start_analysis(self, analysis_mode: str, batch_id: str):
        """Start tracking a new analysis."""
        self.current_analysis_mode = analysis_mode
        self.current_batch_id = batch_id
        self.current_segments = []
        self.batch_start_time = time.time()
        
    def add_segment(self, segment_id: str):
        """Add a segment to current batch."""
        self.current_segments.append(segment_id)
        
    def record_api_call(self, call_type: str, tokens_used: int, 
                       processing_time_ms: float, success: bool = True,
                       error_message: Optional[str] = None):
        """Record an API call."""
        if not self.current_analysis_mode or not self.current_batch_id:
            raise ValueError("No analysis in progress. Call start_analysis() first.")
            
        metrics = APICallMetrics(
            timestamp=datetime.now().isoformat(),
            analysis_mode=self.current_analysis_mode,
            call_type=call_type,
            segment_ids=self.current_segments.copy(),
            tokens_used=tokens_used,
            processing_time_ms=processing_time_ms,
            batch_size=len(self.current_segments),
            success=success,
            error_message=error_message
        )
        
        self.api_call_metrics.append(metrics)
        
        # Update segment metrics
        for segment_id in self.current_segments:
            if segment_id not in self.segment_metrics:
                self.segment_metrics[segment_id] = SegmentMetrics(
                    segment_id=segment_id,
                    analysis_mode=self.current_analysis_mode,
                    total_api_calls=0,
                    total_tokens=0,
                    total_processing_time_ms=0.0
                )
            
            seg_metrics = self.segment_metrics[segment_id]
            seg_metrics.total_api_calls += 1
            seg_metrics.total_tokens += tokens_used // len(self.current_segments) if self.current_segments else tokens_used
            seg_metrics.total_processing_time_ms += processing_time_ms / len(self.current_segments) if self.current_segments else processing_time_ms
        
        return metrics
    
    def end_batch(self):
        """End current batch processing and calculate batch metrics."""
        if not self.current_batch_id or not self.current_analysis_mode:
            return
            
        # FIX: Check if batch_start_time exists to prevent double-invocation crashes
        if self.batch_start_time is None:
            return

        batch_processing_time = (time.time() - self.batch_start_time) * 1000  # ms
        
        # Get API calls for this batch
        batch_calls = [
            m for m in self.api_call_metrics 
            if m.analysis_mode == self.current_analysis_mode 
            and set(m.segment_ids) == set(self.current_segments)
        ]
        
        total_api_calls = len(batch_calls)
        total_tokens = sum(m.tokens_used for m in batch_calls)
        segment_count = len(self.current_segments)
        
        if segment_count > 0:
            metrics = BatchMetrics(
                batch_id=self.current_batch_id,
                analysis_mode=self.current_analysis_mode,
                segment_count=segment_count,
                total_api_calls=total_api_calls,
                total_tokens=total_tokens,
                total_processing_time_ms=batch_processing_time,
                avg_processing_time_per_segment_ms=batch_processing_time / segment_count,
                api_calls_per_segment=total_api_calls / segment_count,
                tokens_per_segment=total_tokens / segment_count
            )
            
            self.batch_metrics.append(metrics)
        
        # Reset current batch
        self.current_segments = []
        self.batch_start_time = None
        
    def end_analysis(self):
        """End current analysis."""
        self.end_batch()
        self.current_analysis_mode = None
        self.current_batch_id = None
        
    def add_segment_quality_score(self, segment_id: str, quality_score: float, confidence_score: float):
        """Add quality score for a segment."""
        if segment_id in self.segment_metrics:
            self.segment_metrics[segment_id].quality_score = quality_score
            self.segment_metrics[segment_id].confidence_score = confidence_score
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        if not self.api_call_metrics:
            return {}
        
        # Group by analysis mode
        mode_summaries = {}
        all_modes = set(m.analysis_mode for m in self.api_call_metrics)
        
        for mode in all_modes:
            mode_calls = [m for m in self.api_call_metrics if m.analysis_mode == mode]
            mode_batches = [b for b in self.batch_metrics if b.analysis_mode == mode]
            mode_segments = [s for s in self.segment_metrics.values() if s.analysis_mode == mode]
            
            if not mode_calls:
                continue
                
            # Calculate averages
            api_calls_per_segment_list = [b.api_calls_per_segment for b in mode_batches]
            tokens_per_segment_list = [b.tokens_per_segment for b in mode_batches]
            processing_time_list = [b.avg_processing_time_per_segment_ms for b in mode_batches]
            
            mode_summaries[mode] = {
                "total_api_calls": len(mode_calls),
                "total_segments": len(mode_segments),
                "total_batches": len(mode_batches),
                "avg_api_calls_per_segment": statistics.mean(api_calls_per_segment_list) if api_calls_per_segment_list else 0,
                "avg_tokens_per_segment": statistics.mean(tokens_per_segment_list) if tokens_per_segment_list else 0,
                "avg_processing_time_per_segment_ms": statistics.mean(processing_time_list) if processing_time_list else 0,
                "success_rate": len([c for c in mode_calls if c.success]) / len(mode_calls) if mode_calls else 0,
                "quality_scores": [s.quality_score for s in mode_segments if s.quality_score is not None],
                "confidence_scores": [s.confidence_score for s in mode_segments if s.confidence_score is not None]
            }
        
        # Overall summary
        overall_summary = {
            "total_api_calls": len(self.api_call_metrics),
            "total_segments": len(self.segment_metrics),
            "total_batches": len(self.batch_metrics),
            "avg_api_calls_per_segment": statistics.mean([b.api_calls_per_segment for b in self.batch_metrics]) if self.batch_metrics else 0,
            "avg_tokens_per_segment": statistics.mean([b.tokens_per_segment for b in self.batch_metrics]) if self.batch_metrics else 0,
            "analysis_modes": mode_summaries,
            "timestamp": datetime.now().isoformat()
        }
        
        return overall_summary
    
    def save_metrics(self, filename: str = "metrics_summary.json"):
        """Save all metrics to JSON file."""
        summary = self.get_summary()
        
        # Add detailed metrics
        output = {
            "summary": summary,
            "api_call_metrics": [asdict(m) for m in self.api_call_metrics],
            "segment_metrics": {sid: asdict(m) for sid, m in self.segment_metrics.items()},
            "batch_metrics": [asdict(m) for m in self.batch_metrics]
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print(f"Metrics saved to {output_path}")
        return output_path
    
    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """Compare current metrics with baseline."""
        # Load baseline
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
        
        current_summary = self.get_summary()
        comparison = {}
        
        for mode, baseline_data in baseline.items():
            if mode not in current_summary.get('analysis_modes', {}):
                continue
                
            current_data = current_summary['analysis_modes'][mode]
            
            comparison[mode] = {
                "api_calls_per_segment": {
                    "baseline": baseline_data.get('api_calls_per_segment', 0),
                    "current": current_data.get('avg_api_calls_per_segment', 0),
                    "change": current_data.get('avg_api_calls_per_segment', 0) - baseline_data.get('api_calls_per_segment', 0),
                    "change_percent": ((current_data.get('avg_api_calls_per_segment', 0) - baseline_data.get('api_calls_per_segment', 0)) 
                                      / baseline_data.get('api_calls_per_segment', 1)) * 100 if baseline_data.get('api_calls_per_segment', 0) > 0 else 0
                },
                "processing_time": {
                    "baseline": baseline_data.get('processing_time_seconds', 0) * 1000,  # Convert to ms
                    "current": current_data.get('avg_processing_time_per_segment_ms', 0),
                    "change": current_data.get('avg_processing_time_per_segment_ms', 0) - (baseline_data.get('processing_time_seconds', 0) * 1000),
                    "change_percent": ((current_data.get('avg_processing_time_per_segment_ms', 0) - (baseline_data.get('processing_time_seconds', 0) * 1000))
                                      / (baseline_data.get('processing_time_seconds', 0) * 1000)) * 100 if baseline_data.get('processing_time_seconds', 0) > 0 else 0
                }
            }
        
        return comparison
    
    def print_summary(self):
        """Print summary of collected metrics."""
        summary = self.get_summary()
        
        if not summary:
            print("No metrics collected yet.")
            return
            
        print("\n" + "=" * 60)
        print("METRICS COLLECTOR SUMMARY")
        print("=" * 60)
        print(f"Total API Calls: {summary['total_api_calls']}")
        print(f"Total Segments: {summary['total_segments']}")
        print(f"Total Batches: {summary['total_batches']}")
        print(f"Average API Calls per Segment: {summary['avg_api_calls_per_segment']:.2f}")
        print(f"Average Tokens per Segment: {summary['avg_tokens_per_segment']:.1f}")
        
        print("\n" + "-" * 60)
        print("BY ANALYSIS MODE:")
        print("-" * 60)
        
        for mode, mode_data in summary.get('analysis_modes', {}).items():
            print(f"\n{mode.upper()}:")
            print(f"  Segments: {mode_data['total_segments']}")
            print(f"  API Calls: {mode_data['total_api_calls']}")
            print(f"  Batches: {mode_data['total_batches']}")
            print(f"  API Calls/Segment: {mode_data['avg_api_calls_per_segment']:.2f}")
            print(f"  Tokens/Segment: {mode_data['avg_tokens_per_segment']:.1f}")
            print(f"  Time/Segment: {mode_data['avg_processing_time_per_segment_ms']:.1f} ms")
            print(f"  Success Rate: {mode_data['success_rate']:.1%}")
            
            if mode_data['quality_scores']:
                print(f"  Avg Quality Score: {statistics.mean(mode_data['quality_scores']):.3f}")
            if mode_data['confidence_scores']:
                print(f"  Avg Confidence Score: {statistics.mean(mode_data['confidence_scores']):.3f}")


# Singleton instance for easy access
_global_collector = None

def get_global_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector

def start_collection(analysis_mode: str, batch_id: str):
    """Start metrics collection for an analysis."""
    collector = get_global_metrics_collector()
    collector.start_analysis(analysis_mode, batch_id)

def record_api_call(call_type: str, tokens_used: int, processing_time_ms: float, 
                   success: bool = True, error_message: Optional[str] = None):
    """Record an API call."""
    collector = get_global_metrics_collector()
    collector.record_api_call(call_type, tokens_used, processing_time_ms, success, error_message)

def add_segment_to_batch(segment_id: str):
    """Add segment to current batch."""
    collector = get_global_metrics_collector()
    collector.add_segment(segment_id)

def end_batch_collection():
    """End current batch collection."""
    collector = get_global_metrics_collector()
    collector.end_batch()

def end_analysis_collection():
    """End current analysis collection."""
    collector = get_global_metrics_collector()
    collector.end_analysis()

def add_segment_quality(segment_id: str, quality_score: float, confidence_score: float):
    """Add quality score for a segment."""
    collector = get_global_metrics_collector()
    collector.add_segment_quality_score(segment_id, quality_score, confidence_score)

def save_metrics(filename: str = "metrics_summary.json"):
    """Save collected metrics."""
    collector = get_global_metrics_collector()
    return collector.save_metrics(filename)

def print_metrics_summary():
    """Print metrics summary."""
    collector = get_global_metrics_collector()
    collector.print_summary()

def compare_with_baseline(baseline_file: str):
    """Compare with baseline metrics."""
    collector = get_global_metrics_collector()
    return collector.compare_with_baseline(baseline_file)