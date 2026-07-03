"""
Unit tests for performance monitoring system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio

from garmin_mcp.monitoring import (
    PerformanceMetric,
    RequestMetrics,
    PerformanceMonitor,
    CircuitBreaker,
    CacheMetrics,
    PerformanceDecorator
)


class TestPerformanceMetric:
    """Test cases for PerformanceMetric."""
    
    def test_performance_metric_creation(self):
        """Test creating a performance metric."""
        metric = PerformanceMetric(
            name="test_metric",
            value=100.5,
            unit="ms",
            metadata={"source": "test"}
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 100.5
        assert metric.unit == "ms"
        assert metric.metadata["source"] == "test"
        assert isinstance(metric.timestamp, datetime)
    
    def test_performance_metric_defaults(self):
        """Test performance metric with defaults."""
        metric = PerformanceMetric(name="test", value=50.0)
        
        assert metric.unit == "ms"
        assert isinstance(metric.metadata, dict)
        assert len(metric.metadata) == 0


class TestRequestMetrics:
    """Test cases for RequestMetrics."""
    
    def test_request_metrics_creation(self):
        """Test creating request metrics."""
        start_time = datetime.now()
        
        request = RequestMetrics(
            request_id="test_123",
            tool_name="test_tool",
            start_time=start_time
        )
        
        assert request.request_id == "test_123"
        assert request.tool_name == "test_tool"
        assert request.start_time == start_time
        assert request.end_time is None
        assert request.success is True
        assert request.cache_hit is False
    
    def test_request_completion_success(self):
        """Test completing request successfully."""
        request = RequestMetrics(
            request_id="test_123",
            tool_name="test_tool", 
            start_time=datetime.now()
        )
        
        request.complete(success=True)
        
        assert request.success is True
        assert request.end_time is not None
        assert request.duration_ms is not None
        assert request.duration_ms >= 0
    
    def test_request_completion_failure(self):
        """Test completing request with failure."""
        request = RequestMetrics(
            request_id="test_123",
            tool_name="test_tool",
            start_time=datetime.now()
        )
        
        error_message = "Test error"
        request.complete(success=False, error_message=error_message)
        
        assert request.success is False
        assert request.error_message == error_message
        assert request.end_time is not None


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
    
    def test_record_metric(self):
        """Test recording a metric."""
        initial_count = len(self.monitor.metrics)
        
        self.monitor.record_metric("test_metric", 100.5, "ms", source="test")
        
        assert len(self.monitor.metrics) == initial_count + 1
        
        metric = self.monitor.metrics[-1]
        assert metric.name == "test_metric"
        assert metric.value == 100.5
        assert metric.unit == "ms"
        assert metric.metadata["source"] == "test"
    
    def test_record_metric_limit(self):
        """Test metric recording with limit."""
        # Add many metrics to test limit
        for i in range(1050):
            self.monitor.record_metric(f"metric_{i}", i)
        
        # Should keep only last 1000
        assert len(self.monitor.metrics) == 1000
        assert self.monitor.metrics[0].name == "metric_50"  # First 50 should be removed
    
    def test_start_request(self):
        """Test starting a request."""
        initial_count = len(self.monitor.request_metrics)
        
        request = self.monitor.start_request("test_123", "test_tool")
        
        assert len(self.monitor.request_metrics) == initial_count + 1
        assert request.request_id == "test_123"
        assert request.tool_name == "test_tool"
    
    def test_request_limit(self):
        """Test request metrics limit."""
        # Add many requests to test limit
        for i in range(550):
            self.monitor.start_request(f"req_{i}", "test_tool")
        
        # Should keep only last 500
        assert len(self.monitor.request_metrics) == 500
        assert self.monitor.request_metrics[0].request_id == "req_50"
    
    def test_record_cache_operations(self):
        """Test recording cache operations."""
        initial_hits = self.monitor.cache_stats["hits"]
        initial_misses = self.monitor.cache_stats["misses"]
        
        self.monitor.record_cache_hit("test_tool")
        self.monitor.record_cache_miss("test_tool")
        
        assert self.monitor.cache_stats["hits"] == initial_hits + 1
        assert self.monitor.cache_stats["misses"] == initial_misses + 1
    
    def test_record_api_call(self):
        """Test recording API call."""
        initial_count = self.monitor.api_rate_limit_stats["requests_made"]
        
        self.monitor.record_api_call(150.0)
        
        assert self.monitor.api_rate_limit_stats["requests_made"] == initial_count + 1
    
    def test_record_rate_limit_delay(self):
        """Test recording rate limit delay."""
        initial_delays = self.monitor.api_rate_limit_stats["rate_limit_delays"]
        initial_total = self.monitor.api_rate_limit_stats["total_delay_ms"]
        
        self.monitor.record_rate_limit_delay(500.0)
        
        assert self.monitor.api_rate_limit_stats["rate_limit_delays"] == initial_delays + 1
        assert self.monitor.api_rate_limit_stats["total_delay_ms"] == initial_total + 500.0
    
    def test_record_error(self):
        """Test recording errors."""
        initial_count = self.monitor.error_counts["test_tool_ValueError"]
        
        self.monitor.record_error("ValueError", "test_tool")
        
        assert self.monitor.error_counts["test_tool_ValueError"] == initial_count + 1
    
    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        # No operations - should be 0
        assert self.monitor.get_cache_hit_rate() == 0.0
        
        # Add some operations
        self.monitor.record_cache_hit("tool1")
        self.monitor.record_cache_hit("tool2")
        self.monitor.record_cache_miss("tool3")
        
        # 2 hits out of 3 total = 66.67%
        hit_rate = self.monitor.get_cache_hit_rate()
        assert abs(hit_rate - 66.67) < 0.01
    
    def test_average_response_time(self):
        """Test average response time calculation."""
        # No completed requests - should be 0
        assert self.monitor.get_average_response_time() == 0.0
        
        # Add completed requests
        req1 = self.monitor.start_request("req1", "tool1")
        req1.complete()
        req1.duration_ms = 100.0
        
        req2 = self.monitor.start_request("req2", "tool1")
        req2.complete()
        req2.duration_ms = 200.0
        
        # Average should be 150.0
        avg_time = self.monitor.get_average_response_time()
        assert avg_time == 150.0
        
        # Test tool-specific average
        avg_tool_time = self.monitor.get_average_response_time("tool1")
        assert avg_tool_time == 150.0
    
    def test_success_rate(self):
        """Test success rate calculation."""
        # No requests - should be 100%
        assert self.monitor.get_success_rate() == 100.0
        
        # Add requests
        req1 = self.monitor.start_request("req1", "tool1")
        req1.complete(success=True)
        
        req2 = self.monitor.start_request("req2", "tool1")
        req2.complete(success=False)
        
        # 1 success out of 2 = 50%
        success_rate = self.monitor.get_success_rate()
        assert success_rate == 50.0
    
    def test_performance_summary(self):
        """Test getting performance summary."""
        # Add some test data
        self.monitor.record_cache_hit("tool1")
        self.monitor.record_cache_miss("tool1")
        self.monitor.record_api_call(100.0)
        
        req = self.monitor.start_request("req1", "tool1")
        req.complete()
        req.duration_ms = 150.0
        
        summary = self.monitor.get_performance_summary()
        
        # Check required fields
        required_fields = ["uptime_seconds", "cache", "api_calls", "requests", "tools", "errors", "timestamp"]
        for field in required_fields:
            assert field in summary
        
        # Check cache info
        assert "hit_rate_percent" in summary["cache"]
        assert "hits" in summary["cache"]
        assert "misses" in summary["cache"]
        
        # Check requests info
        assert "total" in summary["requests"]
        assert "avg_response_time_ms" in summary["requests"]
        assert "success_rate_percent" in summary["requests"]
    
    def test_health_status(self):
        """Test getting health status."""
        # Start with healthy status (no issues)
        health = self.monitor.get_health_status()
        assert health["status"] == "healthy"
        assert len(health["issues"]) == 0
        
        # Add some issues
        # Low success rate
        for i in range(10):
            req = self.monitor.start_request(f"req_{i}", "tool1")
            req.complete(success=False)
        
        health = self.monitor.get_health_status()
        assert health["status"] in ["degraded", "unhealthy"]
        assert len(health["issues"]) > 0


class TestCircuitBreaker:
    """Test cases for CircuitBreaker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=1)
    
    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker initial state."""
        assert self.circuit_breaker.state == "closed"
        assert self.circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        async def successful_func():
            return "success"
        
        result = await self.circuit_breaker.call(successful_func)
        assert result == "success"
        assert self.circuit_breaker.state == "closed"
        assert self.circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failures(self):
        """Test circuit breaker with failures."""
        async def failing_func():
            raise ValueError("Test error")
        
        # First few failures should go through
        for i in range(2):
            with pytest.raises(ValueError):
                await self.circuit_breaker.call(failing_func)
            assert self.circuit_breaker.state == "closed"
        
        # Third failure should open circuit
        with pytest.raises(ValueError):
            await self.circuit_breaker.call(failing_func)
        assert self.circuit_breaker.state == "open"
        
        # Fourth call should be blocked
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await self.circuit_breaker.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        async def failing_func():
            raise ValueError("Test error")
        
        async def successful_func():
            return "success"
        
        # Trigger circuit breaker to open
        for i in range(3):
            with pytest.raises(ValueError):
                await self.circuit_breaker.call(failing_func)
        
        assert self.circuit_breaker.state == "open"
        
        # Wait for timeout
        await asyncio.sleep(1.1)
        
        # Next call should try half-open
        result = await self.circuit_breaker.call(successful_func)
        assert result == "success"
        assert self.circuit_breaker.state == "closed"
        assert self.circuit_breaker.failure_count == 0
    
    def test_circuit_breaker_sync_function(self):
        """Test circuit breaker with synchronous function."""
        def sync_func():
            return "sync_success"
        
        # This should work with asyncio.run in the actual call
        # For testing, we'll create a simple wrapper
        async def test_sync():
            return await self.circuit_breaker.call(sync_func)
        
        result = asyncio.run(test_sync())
        assert result == "sync_success"


class TestCacheMetrics:
    """Test cases for CacheMetrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = {}
        self.monitor = PerformanceMonitor()
        self.cache_metrics = CacheMetrics(self.cache, self.monitor)
    
    def test_cache_get_hit(self):
        """Test cache get with hit."""
        self.cache["test_key"] = "test_value"
        
        initial_hits = self.monitor.cache_stats["hits"]
        
        result = self.cache_metrics.get("test_key", "test_tool")
        
        assert result == "test_value"
        assert self.monitor.cache_stats["hits"] == initial_hits + 1
    
    def test_cache_get_miss(self):
        """Test cache get with miss."""
        initial_misses = self.monitor.cache_stats["misses"]
        
        result = self.cache_metrics.get("nonexistent_key", "test_tool")
        
        assert result is None
        assert self.monitor.cache_stats["misses"] == initial_misses + 1
    
    def test_cache_set(self):
        """Test cache set."""
        self.cache_metrics.set("test_key", "test_value", "test_tool")
        
        assert self.cache["test_key"] == "test_value"
        # Should record a metric
        assert len(self.monitor.metrics) > 0
    
    def test_cache_contains(self):
        """Test cache contains."""
        self.cache["test_key"] = "test_value"
        
        assert "test_key" in self.cache_metrics
        assert "nonexistent_key" not in self.cache_metrics
    
    def test_cache_getitem_setitem(self):
        """Test cache getitem and setitem."""
        self.cache_metrics["test_key"] = "test_value"
        assert self.cache_metrics["test_key"] == "test_value"


class TestPerformanceDecorator:
    """Test cases for PerformanceDecorator."""
    
    @pytest.mark.asyncio
    async def test_async_function_monitoring(self):
        """Test monitoring async function."""
        monitor = PerformanceMonitor()
        
        @PerformanceDecorator.monitor_performance("test_async_tool")
        async def test_async_func(value):
            await asyncio.sleep(0.01)  # Small delay
            return value * 2
        
        initial_requests = len(monitor.request_metrics)
        
        # Patch get_performance_monitor to return our test monitor
        with patch('garmin_mcp.monitoring.get_performance_monitor', return_value=monitor):
            result = await test_async_func(5)
        
        assert result == 10
        assert len(monitor.request_metrics) == initial_requests + 1
        
        # Check request was recorded
        request = monitor.request_metrics[-1]
        assert request.tool_name == "test_async_tool"
        assert request.success is True
    
    def test_sync_function_monitoring(self):
        """Test monitoring sync function."""
        monitor = PerformanceMonitor()
        
        @PerformanceDecorator.monitor_performance("test_sync_tool")
        def test_sync_func(value):
            return value * 3
        
        initial_requests = len(monitor.request_metrics)
        
        # Patch get_performance_monitor to return our test monitor
        with patch('garmin_mcp.monitoring.get_performance_monitor', return_value=monitor):
            result = test_sync_func(7)
        
        assert result == 21
        assert len(monitor.request_metrics) == initial_requests + 1
        
        # Check request was recorded
        request = monitor.request_metrics[-1]
        assert request.tool_name == "test_sync_tool"
        assert request.success is True
    
    @pytest.mark.asyncio
    async def test_async_function_error_monitoring(self):
        """Test monitoring async function with error."""
        monitor = PerformanceMonitor()
        
        @PerformanceDecorator.monitor_performance("test_error_tool")
        async def test_error_func():
            raise ValueError("Test error")
        
        with patch('garmin_mcp.monitoring.get_performance_monitor', return_value=monitor):
            with pytest.raises(ValueError):
                await test_error_func()
        
        # Check error was recorded
        request = monitor.request_metrics[-1]
        assert request.tool_name == "test_error_tool"
        assert request.success is False
        assert request.error_message == "Test error"
        
        # Check error count
        assert monitor.error_counts["test_error_tool_ValueError"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])