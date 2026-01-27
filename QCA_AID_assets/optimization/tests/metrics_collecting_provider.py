"""
Metrics Collecting LLM Provider Wrapper
Wraps an LLM provider to collect API call metrics for optimization analysis.
"""

import time
from typing import Optional, Dict, Any, List
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

# Try to import, but for simulation we don't need actual imports
try:
    from QCA_AID_assets.utils.llm.base import LLMProvider
    from QCA_AID_assets.utils.llm.response import LLMResponse
except ImportError:
    # Mock classes for simulation
    class LLMProvider:
        pass
    
    class LLMResponse:
        pass

from QCA_AID_assets.optimization.tests.metrics_collector import record_api_call, add_segment_to_batch


class MetricsCollectingProvider(LLMProvider):
    """LLM provider wrapper that collects API call metrics."""
    
    def __init__(self, wrapped_provider: LLMProvider, call_type: str = "llm_call"):
        """
        Initialize metrics collecting wrapper.
        
        Args:
            wrapped_provider: The actual LLM provider to wrap
            call_type: Type of API call for metrics (e.g., 'relevance', 'coding')
        """
        self.wrapped_provider = wrapped_provider
        self.call_type = call_type
        
    async def create_completion(self, model: str, messages: List[Dict[str, str]], 
                               temperature: float = 0.3, **kwargs) -> Dict[str, Any]:
        """
        Create completion with metrics collection.
        """
        start_time = time.time()
        
        try:
            # Track this API call
            response = await self.wrapped_provider.create_completion(
                model=model, 
                messages=messages, 
                temperature=temperature, 
                **kwargs
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Estimate tokens (simplified - in practice should count actual tokens)
            # For now, use rough estimate based on content length
            total_chars = sum(len(m.get('content', '')) for m in messages)
            if isinstance(response, dict) and 'choices' in response:
                response_text = response['choices'][0].get('message', {}).get('content', '')
                total_chars += len(response_text)
            
            # Rough estimate: ~4 tokens per 100 characters
            estimated_tokens = max(1, int(total_chars * 0.04))
            
            # Record the API call
            record_api_call(
                call_type=self.call_type,
                tokens_used=estimated_tokens,
                processing_time_ms=processing_time_ms,
                success=True
            )
            
            return response
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Record failed API call
            record_api_call(
                call_type=self.call_type,
                tokens_used=0,
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )
            
            raise
    
    async def create_chat_completion(self, model: str, messages: List[Dict[str, str]], 
                                    temperature: float = 0.3, **kwargs) -> LLMResponse:
        """
        Create chat completion with metrics collection.
        """
        start_time = time.time()
        
        try:
            # Track this API call
            response = await self.wrapped_provider.create_chat_completion(
                model=model, 
                messages=messages, 
                temperature=temperature, 
                **kwargs
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Estimate tokens
            total_chars = sum(len(m.get('content', '')) for m in messages)
            if hasattr(response, 'content'):
                total_chars += len(response.content)
            
            estimated_tokens = max(1, int(total_chars * 0.04))
            
            # Record the API call
            record_api_call(
                call_type=self.call_type,
                tokens_used=estimated_tokens,
                processing_time_ms=processing_time_ms,
                success=True
            )
            
            return response
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Record failed API call
            record_api_call(
                call_type=self.call_type,
                tokens_used=0,
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )
            
            raise
    
    async def check_model_capabilities(self, model: str) -> Dict[str, Any]:
        """
        Check model capabilities with metrics collection.
        """
        start_time = time.time()
        
        try:
            result = await self.wrapped_provider.check_model_capabilities(model)
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Record the API call
            record_api_call(
                call_type=f"{self.call_type}_capabilities",
                tokens_used=100,  # Rough estimate
                processing_time_ms=processing_time_ms,
                success=True
            )
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Record failed API call
            record_api_call(
                call_type=f"{self.call_type}_capabilities",
                tokens_used=0,
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )
            
            raise
    
    def supports_json_mode(self, model: str) -> bool:
        """Check if model supports JSON mode."""
        return self.wrapped_provider.supports_json_mode(model)
    
    def supports_temperature(self, model: str) -> bool:
        """Check if model supports temperature parameter."""
        return self.wrapped_provider.supports_temperature(model)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self.wrapped_provider.get_available_models()
    
    def get_model_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing information for a model."""
        return self.wrapped_provider.get_model_pricing(model)


def create_metrics_collecting_provider(base_provider: LLMProvider, call_type: str = "llm_call") -> LLMProvider:
    """
    Create a metrics-collecting wrapper around a provider.
    
    Args:
        base_provider: The LLM provider to wrap
        call_type: Type of API calls for metrics categorization
        
    Returns:
        LLMProvider: Wrapped provider that collects metrics
    """
    return MetricsCollectingProvider(base_provider, call_type)