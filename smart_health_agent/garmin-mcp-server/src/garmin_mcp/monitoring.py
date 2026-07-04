"""
Performance monitoring and observability for Garmin MCP Server.
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str = "ms"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    tool_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    cache_hit: bool = False
    api_calls_made: int = 0
    data_points_returned: int = 0
    
    def complete(self, success: bool = True, error_message: Optional[str] = None):
        """Mark request as completed."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success
        self.error_message = error_message


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.request_metrics: List[RequestMetrics] = []
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
        self.api_rate_limit_stats = {
            "requests_made": 0,
            "rate_limit_delays": 0,
            "total_delay_ms": 0
        }
        self.error_counts = defaultdict(int)
        self._start_time = datetime.now()
    
    def record_metric(self, name: str, value: float, unit: str = "ms", **metadata):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            metadata=metadata
        )
        self.metrics.append(metric)
        
        # Keep only last 1000 metrics to prevent memory growth
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
        
        logger.debug("Performance metric recorded", 
                    metric_name=name, value=value, unit=unit)
    
    def start_request(self, request_id: str, tool_name: str) -> RequestMetrics:
        """Start tracking a request."""
        request = RequestMetrics(
            request_id=request_id,
            tool_name=tool_name,
            start_time=datetime.now()
        )
        self.request_metrics.append(request)
        
        # Keep only last 500 requests to prevent memory growth
        if len(self.request_metrics) > 500:
            self.request_metrics = self.request_metrics[-500:]
        
        return request
    
    def record_cache_hit(self, tool_name: str):
        """Record a cache hit."""
        self.cache_stats["hits"] += 1
        self.record_metric(f"{tool_name}_cache_hit", 1, "count")
    
    def record_cache_miss(self, tool_name: str):
        """Record a cache miss."""
        self.cache_stats["misses"] += 1
        self.record_metric(f"{tool_name}_cache_miss", 1, "count")
    
    def record_api_call(self, duration_ms: float):
        """Record an API call to Garmin."""
        self.api_rate_limit_stats["requests_made"] += 1
        self.record_metric("garmin_api_call", duration_ms, "ms")
    
    def record_rate_limit_delay(self, delay_ms: float):
        """Record a rate limit delay."""
        self.api_rate_limit_stats["rate_limit_delays"] += 1
        self.api_rate_limit_stats["total_delay_ms"] += delay_ms
        self.record_metric("rate_limit_delay", delay_ms, "ms")
    
    def record_error(self, error_type: str, tool_name: str = "unknown"):
        """Record an error occurrence."""
        self.error_counts[f"{tool_name}_{error_type}"] += 1
        self.record_metric(f"error_{error_type}", 1, "count", tool=tool_name)
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total == 0:
            return 0.0
        return (self.cache_stats["hits"] / total) * 100
    
    def get_average_response_time(self, tool_name: Optional[str] = None) -> float:
        """Get average response time in milliseconds."""
        completed_requests = [
            r for r in self.request_metrics 
            if r.duration_ms is not None and (tool_name is None or r.tool_name == tool_name)
        ]
        
        if not completed_requests:
            return 0.0
        
        return sum(r.duration_ms for r in completed_requests) / len(completed_requests)
    
    def get_success_rate(self, tool_name: Optional[str] = None) -> float:
        """Get success rate percentage."""
        relevant_requests = [
            r for r in self.request_metrics 
            if r.end_time is not None and (tool_name is None or r.tool_name == tool_name)
        ]
        
        if not relevant_requests:
            return 100.0
        
        successful = sum(1 for r in relevant_requests if r.success)
        return (successful / len(relevant_requests)) * 100
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        uptime = datetime.now() - self._start_time
        
        # Calculate tool-specific metrics
        tool_stats = {}
        for tool in set(r.tool_name for r in self.request_metrics):
            tool_stats[tool] = {
                "avg_response_time_ms": self.get_average_response_time(tool),
                "success_rate_percent": self.get_success_rate(tool),
                "request_count": sum(1 for r in self.request_metrics if r.tool_name == tool)
            }
        
        return {
            "uptime_seconds": uptime.total_seconds(),
            "cache": {
                "hit_rate_percent": self.get_cache_hit_rate(),
                "hits": self.cache_stats["hits"],
                "misses": self.cache_stats["misses"],
                "total_requests": self.cache_stats["hits"] + self.cache_stats["misses"]
            },
            "api_calls": {
                "total_requests": self.api_rate_limit_stats["requests_made"],
                "rate_limit_delays": self.api_rate_limit_stats["rate_limit_delays"],
                "avg_delay_ms": (
                    self.api_rate_limit_stats["total_delay_ms"] / 
                    max(1, self.api_rate_limit_stats["rate_limit_delays"])
                )
            },
            "requests": {
                "total": len(self.request_metrics),
                "avg_response_time_ms": self.get_average_response_time(),
                "success_rate_percent": self.get_success_rate(),
                "completed": sum(1 for r in self.request_metrics if r.end_time is not None)
            },
            "tools": tool_stats,
            "errors": dict(self.error_counts),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for health checks."""
        summary = self.get_performance_summary()
        
        # Determine health based on metrics
        health_issues = []
        
        if summary["requests"]["success_rate_percent"] < 90:
            health_issues.append("Low success rate")
        
        if summary["requests"]["avg_response_time_ms"] > 5000:
            health_issues.append("High response times")
        
        # Only meaningful once the cache has seen traffic; a fresh monitor
        # would otherwise report degraded at 0%.
        if summary["cache"]["total_requests"] > 0 and summary["cache"]["hit_rate_percent"] < 30:
            health_issues.append("Low cache hit rate")
        
        if summary["api_calls"]["rate_limit_delays"] > 10:
            health_issues.append("Frequent rate limiting")
        
        status = "healthy" if not health_issues else "degraded" if len(health_issues) <= 2 else "unhealthy"
        
        return {
            "status": status,
            "issues": health_issues,
            "uptime_seconds": summary["uptime_seconds"],
            "last_check": datetime.now().isoformat()
        }


class CircuitBreaker:
    """Circuit breaker for external API calls."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout_seconds)
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class CacheMetrics:
    """Enhanced cache with performance monitoring."""
    
    def __init__(self, cache, monitor: PerformanceMonitor):
        self.cache = cache
        self.monitor = monitor
    
    def get(self, key: str, tool_name: str = "unknown"):
        """Get item from cache with monitoring."""
        if key in self.cache:
            self.monitor.record_cache_hit(tool_name)
            return self.cache[key]
        else:
            self.monitor.record_cache_miss(tool_name)
            return None
    
    def set(self, key: str, value: Any, tool_name: str = "unknown"):
        """Set item in cache with monitoring."""
        self.cache[key] = value
        self.monitor.record_metric(f"{tool_name}_cache_set", 1, "count")
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self.cache
    
    def __getitem__(self, key: str):
        """Get item from cache."""
        return self.cache[key]
    
    def __setitem__(self, key: str, value: Any):
        """Set item in cache."""
        self.cache[key] = value


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor


class PerformanceDecorator:
    """Decorator for monitoring function performance."""
    
    @staticmethod
    def monitor_performance(tool_name: str):
        """Decorator to monitor function performance."""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                request_id = f"{tool_name}_{int(time.time() * 1000)}"
                request = monitor.start_request(request_id, tool_name)
                
                try:
                    result = await func(*args, **kwargs)
                    request.complete(success=True)
                    return result
                except Exception as e:
                    request.complete(success=False, error_message=str(e))
                    monitor.record_error(type(e).__name__, tool_name)
                    raise
            
            def sync_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                request_id = f"{tool_name}_{int(time.time() * 1000)}"
                request = monitor.start_request(request_id, tool_name)
                
                try:
                    result = func(*args, **kwargs)
                    request.complete(success=True)
                    return result
                except Exception as e:
                    request.complete(success=False, error_message=str(e))
                    monitor.record_error(type(e).__name__, tool_name)
                    raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator