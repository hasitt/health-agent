"""
Utility functions for Garmin MCP Server.
"""

import logging
import sys
from datetime import datetime, date, timedelta
from typing import Any, Dict, Optional, Union
from pathlib import Path

import structlog
from structlog.types import Processor


def setup_logging(
    level: str = "INFO",
    format_type: str = "text",
    development: bool = False
) -> None:
    """
    Set up structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_type: Log format (json, text)
        development: Whether to enable development mode features
    """
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,  # Important for MCP STDIO transport
        level=getattr(logging, level.upper(), logging.INFO),
    )
    
    # Configure structlog processors
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="ISO"),
    ]
    
    if development:
        processors.extend([
            structlog.dev.set_exc_info,
        ])
    
    # Add appropriate renderer based on format
    if format_type == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(colors=development)
        )
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    
    # Set external library log levels
    logging.getLogger("garminconnect").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    

def parse_date(date_str: Optional[str], default: Optional[date] = None) -> date:
    """
    Parse a date string into a date object.
    
    Args:
        date_str: Date string in various formats (YYYY-MM-DD, etc.)
        default: Default date to return if date_str is None
        
    Returns:
        Parsed date object
        
    Raises:
        ValueError: If date string is invalid
    """
    if date_str is None:
        if default is not None:
            return default
        return date.today()
    
    # Try different date formats
    formats = [
        "%Y-%m-%d",      # 2025-08-06
        "%m/%d/%Y",      # 08/06/2025
        "%d/%m/%Y",      # 06/08/2025
        "%Y%m%d",        # 20250806
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    # Try relative date parsing
    date_str_lower = date_str.lower().strip()
    today = date.today()
    
    if date_str_lower in ["today", "now"]:
        return today
    elif date_str_lower == "yesterday":
        return today - timedelta(days=1)
    elif date_str_lower == "tomorrow":
        return today + timedelta(days=1)
    elif date_str_lower.endswith("days ago"):
        try:
            days = int(date_str_lower.split()[0])
            return today - timedelta(days=days)
        except (ValueError, IndexError):
            pass
    
    raise ValueError(f"Invalid date format: {date_str}")


def parse_date_range(
    start_date: Optional[str],
    end_date: Optional[str],
    default_days: int = 7,
    max_days: int = 90
) -> tuple[date, date]:
    """
    Parse and validate a date range.
    
    Args:
        start_date: Start date string
        end_date: End date string  
        default_days: Default range in days if not specified
        max_days: Maximum allowed range in days
        
    Returns:
        Tuple of (start_date, end_date)
        
    Raises:
        ValueError: If date range is invalid
    """
    today = date.today()
    
    if start_date is None and end_date is None:
        # Default range
        end = today
        start = end - timedelta(days=default_days - 1)
    elif start_date is None:
        # Only end date specified
        end = parse_date(end_date)
        start = end - timedelta(days=default_days - 1)
    elif end_date is None:
        # Only start date specified
        start = parse_date(start_date)
        end = min(start + timedelta(days=default_days - 1), today)
    else:
        # Both dates specified
        start = parse_date(start_date)
        end = parse_date(end_date)
    
    # Validate range
    if start > end:
        raise ValueError("Start date must be before or equal to end date")
    
    if end > today:
        raise ValueError("End date cannot be in the future")
    
    range_days = (end - start).days + 1
    if range_days > max_days:
        raise ValueError(f"Date range too large: {range_days} days (max: {max_days})")
    
    return start, end


def safe_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.
    
    Args:
        data: Dictionary to search
        *keys: Nested keys to traverse
        default: Default value if key not found
        
    Returns:
        Value if found, default otherwise
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def format_duration(seconds: Union[int, float]) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 0:
        return "0:00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def format_distance(meters: Union[int, float], unit: str = "km") -> str:
    """
    Format distance with appropriate units.
    
    Args:
        meters: Distance in meters
        unit: Target unit (km, mi)
        
    Returns:
        Formatted distance string
    """
    if unit.lower() == "mi":
        miles = meters / 1609.34
        return f"{miles:.2f} mi"
    else:
        km = meters / 1000.0
        return f"{km:.2f} km"


def calculate_trends(
    current_values: list[Union[int, float]], 
    previous_values: list[Union[int, float]]
) -> Dict[str, Any]:
    """
    Calculate trend analysis between two periods.
    
    Args:
        current_values: Values for current period
        previous_values: Values for previous period
        
    Returns:
        Dictionary with trend analysis
    """
    if not current_values or not previous_values:
        return {"trend": "unknown", "change": 0, "change_percent": 0}
    
    current_avg = sum(current_values) / len(current_values)
    previous_avg = sum(previous_values) / len(previous_values)
    
    if previous_avg == 0:
        change_percent = 0
    else:
        change_percent = ((current_avg - previous_avg) / previous_avg) * 100
    
    change = current_avg - previous_avg
    
    # Determine trend direction
    if abs(change_percent) < 5:
        trend = "stable"
    elif change_percent > 0:
        trend = "increasing"
    else:
        trend = "decreasing"
    
    return {
        "trend": trend,
        "change": round(change, 2),
        "change_percent": round(change_percent, 1),
        "current_average": round(current_avg, 2),
        "previous_average": round(previous_avg, 2),
    }


def sanitize_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize sensitive data for logging.
    
    Args:
        data: Dictionary that may contain sensitive data
        
    Returns:
        Dictionary with sensitive values masked
    """
    sensitive_keys = {
        "password", "token", "secret", "key", "auth", "credential",
        "garmin_password", "oauth_client_secret", "jwt_secret_key"
    }
    
    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            sanitized[key] = "***MASKED***"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_for_logging(value)
        elif isinstance(value, str) and len(value) > 100:
            # Truncate very long strings
            sanitized[key] = value[:100] + "..."
        else:
            sanitized[key] = value
    
    return sanitized


def validate_metrics_list(metrics: Optional[list[str]]) -> list[str]:
    """
    Validate and normalize metrics list.
    
    Args:
        metrics: List of requested metrics
        
    Returns:
        Validated list of metrics
        
    Raises:
        ValueError: If invalid metric specified
    """
    valid_metrics = {
        "steps", "sleep", "heart_rate", "stress", "activities",
        "calories", "distance", "body_battery"
    }
    
    if metrics is None:
        return list(valid_metrics)
    
    # Normalize and validate
    normalized = []
    for metric in metrics:
        metric_lower = metric.lower().strip()
        if metric_lower in valid_metrics:
            normalized.append(metric_lower)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    if not normalized:
        raise ValueError("At least one metric must be specified")
    
    return normalized


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls: list[datetime] = []
    
    async def acquire(self) -> None:
        """Acquire permission to make an API call."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Remove old calls
        self.calls = [call_time for call_time in self.calls if call_time > cutoff]
        
        # Check if we can make a call
        if len(self.calls) >= self.calls_per_minute:
            # Calculate wait time
            oldest_call = min(self.calls)
            wait_time = 60 - (now - oldest_call).total_seconds()
            
            if wait_time > 0:
                import asyncio
                await asyncio.sleep(wait_time)
        
        # Record this call
        self.calls.append(now)