"""
Advanced parameter validation and error handling for Garmin MCP Server.
Provides AI-friendly error messages, comprehensive date parsing, and parameter suggestions.
"""

import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import calendar

from pydantic import BaseModel, validator, Field
import structlog

logger = structlog.get_logger(__name__)


class ValidationSeverity(str, Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationError(Exception):
    """Custom validation error with AI-friendly messages."""
    
    def __init__(
        self,
        message: str,
        field: str = None,
        suggestions: List[str] = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        examples: List[str] = None
    ):
        self.message = message
        self.field = field
        self.suggestions = suggestions or []
        self.severity = severity
        self.examples = examples or []
        super().__init__(message)
    
    def to_mcp_error(self) -> Dict[str, Any]:
        """Convert to MCP-compliant error format."""
        error_data = {
            "message": self.message,
            "severity": self.severity.value
        }
        
        if self.field:
            error_data["field"] = self.field
            
        if self.suggestions:
            error_data["suggestions"] = self.suggestions
            
        if self.examples:
            error_data["examples"] = self.examples
            
        return error_data


class DateParser:
    """Advanced date parsing with multiple format support."""
    
    # Common date formats
    DATE_FORMATS = [
        "%Y-%m-%d",        # 2024-01-15
        "%m/%d/%Y",        # 01/15/2024
        "%d/%m/%Y",        # 15/01/2024
        "%m-%d-%Y",        # 01-15-2024
        "%d-%m-%Y",        # 15-01-2024
        "%Y%m%d",          # 20240115
        "%B %d, %Y",       # January 15, 2024
        "%b %d, %Y",       # Jan 15, 2024
        "%d %B %Y",        # 15 January 2024
        "%d %b %Y",        # 15 Jan 2024
    ]
    
    # Relative date patterns
    RELATIVE_PATTERNS = {
        r"today": lambda: date.today(),
        r"yesterday": lambda: date.today() - timedelta(days=1),
        r"tomorrow": lambda: date.today() + timedelta(days=1),
        r"(\d+)\s*days?\s*ago": lambda m: date.today() - timedelta(days=int(m.group(1))),
        r"(\d+)\s*weeks?\s*ago": lambda m: date.today() - timedelta(weeks=int(m.group(1))),
        r"(\d+)\s*months?\s*ago": lambda m: DateParser._subtract_months(date.today(), int(m.group(1))),
        r"last\s+week": lambda: date.today() - timedelta(weeks=1),
        r"last\s+month": lambda: DateParser._subtract_months(date.today(), 1),
        r"this\s+week": lambda: date.today() - timedelta(days=date.today().weekday()),
        r"this\s+month": lambda: date.today().replace(day=1),
    }
    
    @staticmethod
    def _subtract_months(start_date: date, months: int) -> date:
        """Subtract months from a date."""
        month = start_date.month - months
        year = start_date.year + month // 12
        month = month % 12
        if month <= 0:
            month += 12
            year -= 1
        
        # Handle day overflow (e.g., Jan 31 -> Feb 28)
        try:
            return start_date.replace(year=year, month=month)
        except ValueError:
            # Day doesn't exist in target month, use last day of month
            last_day = calendar.monthrange(year, month)[1]
            return start_date.replace(year=year, month=month, day=min(start_date.day, last_day))
    
    @classmethod
    def parse_date(cls, date_input: Union[str, date, datetime]) -> date:
        """
        Parse various date formats into a date object.
        
        Args:
            date_input: Date string, date, or datetime object
            
        Returns:
            Parsed date object
            
        Raises:
            ValidationError: If date cannot be parsed
        """
        # datetime subclasses date, so it must be checked first
        if isinstance(date_input, datetime):
            return date_input.date()
        elif isinstance(date_input, date):
            return date_input
        
        if not isinstance(date_input, str):
            raise ValidationError(
                f"Date must be a string, date, or datetime object, not {type(date_input).__name__}",
                field="date",
                suggestions=["Use ISO format (YYYY-MM-DD)", "Use relative terms like 'today', 'yesterday'"],
                examples=["2024-01-15", "today", "3 days ago", "last week"]
            )
        
        date_str = date_input.strip().lower()
        
        # Try relative date patterns first
        for pattern, handler in cls.RELATIVE_PATTERNS.items():
            match = re.match(pattern, date_str, re.IGNORECASE)
            if match:
                try:
                    if match.groups():
                        # Pattern has capture groups, pass match to handler
                        return handler(match)
                    else:
                        # No capture groups, just call handler
                        return handler()
                except Exception as e:
                    logger.warning("Relative date parsing failed", pattern=pattern, input=date_str, error=str(e))
        
        # Try standard date formats
        for date_format in cls.DATE_FORMATS:
            try:
                parsed_date = datetime.strptime(date_input, date_format).date()
                
                # Validate date is reasonable (not too far in future/past)
                today = date.today()
                if parsed_date > today + timedelta(days=1):
                    raise ValidationError(
                        f"Date '{date_input}' is in the future. Health data is only available for past dates.",
                        field="date",
                        suggestions=["Use today's date or a past date", "Check if you meant a different year"],
                        examples=["today", "yesterday", str(today)]
                    )
                
                # Warn if date is very old (likely data won't be available)
                if parsed_date < today - timedelta(days=365 * 2):
                    logger.warning("Very old date requested", date=str(parsed_date), input=date_input)
                
                return parsed_date
                
            except ValueError:
                continue
        
        # If no format worked, provide helpful error
        raise ValidationError(
            f"Unable to parse date '{date_input}'. Please use a supported format.",
            field="date",
            suggestions=[
                "Use ISO format (YYYY-MM-DD)",
                "Try relative terms like 'today', 'yesterday', '3 days ago'",
                "Use common formats like MM/DD/YYYY or DD/MM/YYYY"
            ],
            examples=["2024-01-15", "01/15/2024", "today", "yesterday", "1 week ago"]
        )
    
    @classmethod
    def validate_date_range(cls, start_date: date, end_date: date, max_days: int = 365) -> Tuple[date, date]:
        """
        Validate a date range.
        
        Args:
            start_date: Start date
            end_date: End date  
            max_days: Maximum allowed range in days
            
        Returns:
            Validated date range tuple
            
        Raises:
            ValidationError: If date range is invalid
        """
        if start_date > end_date:
            raise ValidationError(
                f"Start date ({start_date}) cannot be after end date ({end_date})",
                field="date_range",
                suggestions=["Swap the start and end dates", "Check if you meant different dates"],
                examples=[f"{end_date} to {start_date}", "today to yesterday"]
            )
        
        range_days = (end_date - start_date).days
        if range_days > max_days:
            raise ValidationError(
                f"Date range too large: {range_days} days (maximum: {max_days} days)",
                field="date_range",
                suggestions=[
                    f"Reduce range to {max_days} days or less",
                    "Use monthly or weekly summary tools for larger ranges",
                    "Break request into smaller date ranges"
                ],
                examples=[
                    f"{end_date - timedelta(days=max_days)} to {end_date}",
                    f"{start_date} to {start_date + timedelta(days=max_days)}"
                ]
            )
        
        return start_date, end_date


class ParameterValidator:
    """Comprehensive parameter validation for all tools."""
    
    @staticmethod
    def validate_days_parameter(days: Any, max_days: int = 365, min_days: int = 1) -> int:
        """Validate days parameter."""
        if days is None:
            return 7  # Default
        
        try:
            days_int = int(days)
        except (TypeError, ValueError):
            raise ValidationError(
                f"Days parameter must be a number, got '{days}' ({type(days).__name__})",
                field="days",
                suggestions=["Use a positive integer", "Try common values like 7, 14, 30"],
                examples=["7", "14", "30", "90"]
            )
        
        if days_int < min_days:
            raise ValidationError(
                f"Days parameter must be at least {min_days}, got {days_int}",
                field="days",
                suggestions=[f"Use {min_days} or higher", "Check if you meant a different value"],
                examples=[str(min_days), "7", "14", "30"]
            )
        
        if days_int > max_days:
            raise ValidationError(
                f"Days parameter cannot exceed {max_days}, got {days_int}",
                field="days",
                suggestions=[
                    f"Use {max_days} or less",
                    "Use monthly summaries for longer periods",
                    "Break into smaller time periods"
                ],
                examples=[str(max_days), "30", "90", "180"]
            )
        
        return days_int
    
    @staticmethod
    def validate_metrics_parameter(metrics: Any, available_metrics: List[str]) -> List[str]:
        """Validate metrics parameter."""
        if metrics is None:
            return available_metrics  # Default to all
        
        if isinstance(metrics, str):
            metrics = [m.strip().lower() for m in metrics.split(',')]
        elif not isinstance(metrics, list):
            raise ValidationError(
                f"Metrics must be a string or list, got {type(metrics).__name__}",
                field="metrics",
                suggestions=["Use comma-separated string", "Use list of metric names"],
                examples=["steps,sleep,heart_rate", '["steps", "sleep", "heart_rate"]']
            )
        
        # Normalize and validate each metric
        normalized_metrics = []
        invalid_metrics = []
        
        for metric in metrics:
            if not isinstance(metric, str):
                invalid_metrics.append(str(metric))
                continue
            
            metric_normalized = metric.strip().lower().replace(' ', '_').replace('-', '_')
            
            # Find best match from available metrics
            best_match = None
            for available in available_metrics:
                if metric_normalized == available.lower().replace(' ', '_').replace('-', '_'):
                    best_match = available
                    break
                elif metric_normalized in available.lower().replace(' ', '_').replace('-', '_'):
                    best_match = available
                    break
            
            if best_match:
                normalized_metrics.append(best_match)
            else:
                invalid_metrics.append(metric)
        
        if invalid_metrics:
            suggestions = [
                f"Available metrics: {', '.join(available_metrics)}",
                "Check spelling and try again",
                "Use underscore format (e.g., 'heart_rate' not 'heart rate')"
            ]
            
            # Try to suggest close matches
            close_matches = []
            for invalid in invalid_metrics:
                for available in available_metrics:
                    if invalid.lower() in available.lower() or available.lower() in invalid.lower():
                        close_matches.append(available)
            
            if close_matches:
                suggestions.insert(0, f"Did you mean: {', '.join(set(close_matches))}")
            
            raise ValidationError(
                f"Invalid metrics: {', '.join(invalid_metrics)}",
                field="metrics",
                suggestions=suggestions,
                examples=available_metrics[:5]  # Show first 5 as examples
            )
        
        return normalized_metrics
    
    @staticmethod
    def validate_limit_parameter(limit: Any, max_limit: int = 1000, default_limit: int = 100) -> int:
        """Validate limit parameter."""
        if limit is None:
            return default_limit
        
        try:
            limit_int = int(limit)
        except (TypeError, ValueError):
            raise ValidationError(
                f"Limit must be a number, got '{limit}' ({type(limit).__name__})",
                field="limit",
                suggestions=["Use a positive integer", f"Default is {default_limit}"],
                examples=[str(default_limit), "50", "200", "500"]
            )
        
        if limit_int < 1:
            raise ValidationError(
                f"Limit must be at least 1, got {limit_int}",
                field="limit",
                suggestions=["Use a positive number", f"Default is {default_limit}"],
                examples=["1", str(default_limit), "50", "100"]
            )
        
        if limit_int > max_limit:
            raise ValidationError(
                f"Limit cannot exceed {max_limit}, got {limit_int}",
                field="limit",
                suggestions=[
                    f"Use {max_limit} or less",
                    "Use pagination for larger results",
                    f"Default limit is {default_limit}"
                ],
                examples=[str(max_limit), str(default_limit), "500"]
            )
        
        return limit_int


class ToolParameterSchemas:
    """Pydantic schemas for all tool parameters."""
    
    class DateParameter(BaseModel):
        """Date parameter schema."""
        date: Optional[str] = Field(
            default=None,
            description="Date in YYYY-MM-DD format, or relative terms like 'today', 'yesterday', '3 days ago'",
            example="2024-01-15"
        )
        
        @validator('date', pre=True, always=True)
        def validate_date(cls, v):
            if v is None:
                return str(date.today())
            try:
                parsed_date = DateParser.parse_date(v)
                return str(parsed_date)
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(
                    f"Invalid date format: {e}",
                    field="date",
                    suggestions=["Use YYYY-MM-DD format", "Try 'today', 'yesterday'"],
                    examples=["2024-01-15", "today", "yesterday"]
                )
    
    class DateRangeParameter(BaseModel):
        """Date range parameter schema."""
        start_date: Optional[str] = Field(
            default=None,
            description="Start date in YYYY-MM-DD format or relative terms",
            example="2024-01-01"
        )
        end_date: Optional[str] = Field(
            default=None,
            description="End date in YYYY-MM-DD format or relative terms", 
            example="2024-01-31"
        )
        days: Optional[int] = Field(
            default=7,
            ge=1,
            le=365,
            description="Number of days to include (if dates not specified)",
            example=30
        )
        
        @validator('start_date', 'end_date', pre=True)
        def validate_dates(cls, v):
            if v is None:
                return v
            try:
                parsed_date = DateParser.parse_date(v)
                return str(parsed_date)
            except ValidationError:
                raise
    
    class MetricsParameter(BaseModel):
        """Metrics selection parameter schema."""
        metrics: Optional[Union[str, List[str]]] = Field(
            default=None,
            description="Comma-separated metrics or list of metrics to include",
            example="steps,sleep,heart_rate"
        )
        
        @validator('metrics', pre=True)
        def validate_metrics(cls, v):
            if v is None:
                return v
            if isinstance(v, str):
                return [m.strip() for m in v.split(',')]
            return v
    
    class LimitParameter(BaseModel):
        """Limit parameter schema."""
        limit: Optional[int] = Field(
            default=100,
            ge=1,
            le=1000,
            description="Maximum number of items to return",
            example=50
        )


def validate_tool_parameters(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate parameters for a specific tool.
    
    Args:
        tool_name: Name of the tool being called
        parameters: Raw parameters from tool call
        
    Returns:
        Validated and normalized parameters
        
    Raises:
        ValidationError: If parameters are invalid
    """
    logger.info("Validating tool parameters", tool=tool_name, params=list(parameters.keys()))
    
    try:
        # Tool-specific validation
        if tool_name in ['get_daily_summary', 'get_sleep_data', 'get_heart_rate_data', 'get_stress_data', 'get_activities']:
            # Single date tools
            schema = ToolParameterSchemas.DateParameter(**parameters)
            return {"date": schema.date}
        
        elif tool_name in ['get_date_range_data']:
            # Date range tools
            schema = ToolParameterSchemas.DateRangeParameter(**parameters)
            validated = {"days": schema.days}
            if schema.start_date:
                validated["start_date"] = schema.start_date
            if schema.end_date:
                validated["end_date"] = schema.end_date
            return validated
        
        elif tool_name in ['get_weekly_summary', 'get_monthly_summary']:
            # Period-based tools
            days = ParameterValidator.validate_days_parameter(
                parameters.get('days'),
                max_days=365 if 'monthly' in tool_name else 90
            )
            return {"days": days}
        
        elif tool_name in ['get_trends_analysis', 'get_goals_progress', 'get_health_insights']:
            # Analytics tools
            days = ParameterValidator.validate_days_parameter(
                parameters.get('days', 30),
                max_days=365,
                min_days=7
            )
            validated = {"days": days}
            
            # Optional metrics parameter
            if 'metrics' in parameters:
                available_metrics = ['steps', 'sleep', 'heart_rate', 'stress', 'activities', 'calories']
                metrics = ParameterValidator.validate_metrics_parameter(
                    parameters['metrics'], 
                    available_metrics
                )
                validated["metrics"] = metrics
            
            return validated
        
        elif tool_name in ['get_steps_detail', 'get_body_battery']:
            # Detail tools
            schema = ToolParameterSchemas.DateParameter(**parameters)
            return {"date": schema.date}
        
        else:
            # Unknown tool - return parameters as-is but log warning
            logger.warning("No validation schema for tool", tool=tool_name)
            return parameters
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error("Parameter validation failed", tool=tool_name, error=str(e))
        raise ValidationError(
            f"Parameter validation failed: {e}",
            field="parameters",
            suggestions=["Check parameter format and try again", "Refer to tool documentation"],
            severity=ValidationSeverity.ERROR
        )


def create_ai_friendly_error(error: Exception, tool_name: str = None) -> Dict[str, Any]:
    """
    Create AI-friendly error message from any exception.
    
    Args:
        error: The exception that occurred
        tool_name: Name of the tool where error occurred
        
    Returns:
        MCP-compliant error response
    """
    if isinstance(error, ValidationError):
        return {
            "error": error.to_mcp_error(),
            "isError": True
        }
    
    # Convert other exceptions to ValidationError format
    error_message = str(error)
    error_type = type(error).__name__
    
    # Provide context-specific suggestions
    suggestions = []
    examples = []
    
    if "authentication" in error_message.lower() or "login" in error_message.lower():
        suggestions = [
            "Check your Garmin Connect credentials",
            "Try logging in to Garmin Connect website first",
            "Verify your account is not locked or suspended"
        ]
        examples = ["Update credentials in configuration", "Clear cached tokens"]
    elif "network" in error_message.lower() or "connection" in error_message.lower():
        suggestions = [
            "Check your internet connection",
            "Verify Garmin Connect services are available",
            "Try again in a few moments"
        ]
    elif "rate limit" in error_message.lower():
        suggestions = [
            "Wait a few minutes before trying again",
            "Reduce the frequency of requests",
            "Use summary tools instead of detailed queries"
        ]
    elif "not found" in error_message.lower() or "404" in error_message:
        suggestions = [
            "Check if the requested date has available data",
            "Try a more recent date",
            "Verify your device was connected and syncing"
        ]
    
    return {
        "error": {
            "message": f"{error_type}: {error_message}",
            "field": tool_name,
            "suggestions": suggestions,
            "examples": examples,
            "severity": ValidationSeverity.ERROR.value
        },
        "isError": True
    }


# Export validation functions for use in server
__all__ = [
    'ValidationError',
    'ValidationSeverity', 
    'DateParser',
    'ParameterValidator',
    'ToolParameterSchemas',
    'validate_tool_parameters',
    'create_ai_friendly_error'
]