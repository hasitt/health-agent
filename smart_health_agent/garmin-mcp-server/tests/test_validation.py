"""
Unit tests for parameter validation system.
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import patch

from garmin_mcp.validation import (
    ValidationError,
    DateParser,
    ParameterValidator,
    validate_tool_parameters,
    create_ai_friendly_error
)


class TestDateParser:
    """Test cases for DateParser."""
    
    def test_parse_iso_date(self):
        """Test parsing ISO date format."""
        result = DateParser.parse_date("2024-08-06")
        assert result == date(2024, 8, 6)
    
    def test_parse_today(self):
        """Test parsing 'today'."""
        result = DateParser.parse_date("today")
        assert result == date.today()
    
    def test_parse_yesterday(self):
        """Test parsing 'yesterday'."""
        result = DateParser.parse_date("yesterday")
        expected = date.today() - timedelta(days=1)
        assert result == expected
    
    def test_parse_days_ago(self):
        """Test parsing 'X days ago'."""
        result = DateParser.parse_date("5 days ago")
        expected = date.today() - timedelta(days=5)
        assert result == expected
    
    def test_parse_weeks_ago(self):
        """Test parsing 'X weeks ago'."""
        result = DateParser.parse_date("2 weeks ago")
        expected = date.today() - timedelta(weeks=2)
        assert result == expected
    
    def test_parse_various_formats(self):
        """Test various date formats."""
        test_cases = [
            ("01/15/2024", date(2024, 1, 15)),
            ("15/01/2024", date(2024, 1, 15)),  # DD/MM/YYYY
            ("2024-01-15", date(2024, 1, 15)),
            ("20240115", date(2024, 1, 15)),
        ]
        
        for date_str, expected in test_cases:
            result = DateParser.parse_date(date_str)
            assert result == expected, f"Failed to parse {date_str}"
    
    def test_parse_invalid_date(self):
        """Test parsing invalid date raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DateParser.parse_date("invalid_date_format")
        
        assert "Unable to parse date" in str(exc_info.value)
        assert exc_info.value.suggestions is not None
        assert exc_info.value.examples is not None
    
    def test_parse_future_date_error(self):
        """Test parsing future date raises ValidationError."""
        future_date = date.today() + timedelta(days=10)
        future_str = future_date.strftime("%Y-%m-%d")
        
        with pytest.raises(ValidationError) as exc_info:
            DateParser.parse_date(future_str)
        
        assert "is in the future" in str(exc_info.value)
    
    def test_parse_date_object(self):
        """Test parsing date object returns same object."""
        test_date = date(2024, 8, 6)
        result = DateParser.parse_date(test_date)
        assert result == test_date
    
    def test_parse_datetime_object(self):
        """Test parsing datetime object returns date part."""
        test_datetime = datetime(2024, 8, 6, 15, 30)
        result = DateParser.parse_date(test_datetime)
        assert result == date(2024, 8, 6)


class TestParameterValidator:
    """Test cases for ParameterValidator."""
    
    def test_validate_days_default(self):
        """Test days parameter with default."""
        result = ParameterValidator.validate_days_parameter(None)
        assert result == 7
    
    def test_validate_days_valid(self):
        """Test days parameter with valid values."""
        test_cases = [1, 7, 30, 90, 365]
        
        for days in test_cases:
            result = ParameterValidator.validate_days_parameter(days)
            assert result == days
    
    def test_validate_days_invalid_type(self):
        """Test days parameter with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            ParameterValidator.validate_days_parameter("not_a_number")
        
        assert "must be a number" in str(exc_info.value)
        assert exc_info.value.suggestions is not None
    
    def test_validate_days_too_small(self):
        """Test days parameter too small."""
        with pytest.raises(ValidationError):
            ParameterValidator.validate_days_parameter(0)
    
    def test_validate_days_too_large(self):
        """Test days parameter too large."""
        with pytest.raises(ValidationError):
            ParameterValidator.validate_days_parameter(400)
    
    def test_validate_metrics_default(self):
        """Test metrics parameter with default."""
        available = ["steps", "sleep", "heart_rate"]
        result = ParameterValidator.validate_metrics_parameter(None, available)
        assert result == available
    
    def test_validate_metrics_string(self):
        """Test metrics parameter as comma-separated string."""
        available = ["steps", "sleep", "heart_rate", "stress"]
        result = ParameterValidator.validate_metrics_parameter("steps,sleep", available)
        assert result == ["steps", "sleep"]
    
    def test_validate_metrics_list(self):
        """Test metrics parameter as list."""
        available = ["steps", "sleep", "heart_rate"]
        result = ParameterValidator.validate_metrics_parameter(["steps", "sleep"], available)
        assert result == ["steps", "sleep"]
    
    def test_validate_metrics_invalid(self):
        """Test metrics parameter with invalid metrics."""
        available = ["steps", "sleep"]
        
        with pytest.raises(ValidationError) as exc_info:
            ParameterValidator.validate_metrics_parameter(["invalid_metric"], available)
        
        assert "Invalid metrics" in str(exc_info.value)
    
    def test_validate_limit_default(self):
        """Test limit parameter with default."""
        result = ParameterValidator.validate_limit_parameter(None)
        assert result == 100
    
    def test_validate_limit_valid(self):
        """Test limit parameter with valid values."""
        test_cases = [1, 50, 100, 500, 1000]
        
        for limit in test_cases:
            result = ParameterValidator.validate_limit_parameter(limit)
            assert result == limit
    
    def test_validate_limit_invalid(self):
        """Test limit parameter with invalid values."""
        invalid_cases = [0, -1, 1001, "not_a_number"]
        
        for invalid_limit in invalid_cases:
            with pytest.raises(ValidationError):
                ParameterValidator.validate_limit_parameter(invalid_limit)


class TestToolParameterValidation:
    """Test cases for tool parameter validation."""
    
    def test_validate_daily_summary_valid(self):
        """Test validating daily summary tool parameters."""
        result = validate_tool_parameters("get_daily_summary", {"date": "today"})
        assert "date" in result
        assert result["date"] == str(date.today())
    
    def test_validate_daily_summary_no_params(self):
        """Test validating daily summary with no parameters."""
        result = validate_tool_parameters("get_daily_summary", {})
        assert "date" in result
        assert result["date"] == str(date.today())
    
    def test_validate_trends_analysis_valid(self):
        """Test validating trends analysis parameters."""
        result = validate_tool_parameters("get_trends_analysis", {"days": 30})
        assert result["days"] == 30
    
    def test_validate_trends_analysis_with_metrics(self):
        """Test validating trends analysis with metrics."""
        result = validate_tool_parameters("get_trends_analysis", {
            "days": 30,
            "metrics": "steps,sleep"
        })
        assert result["days"] == 30
        assert "metrics" in result
        assert "steps" in result["metrics"]
        assert "sleep" in result["metrics"]
    
    def test_validate_date_range_data(self):
        """Test validating date range data parameters."""
        result = validate_tool_parameters("get_date_range_data", {
            "start_date": "2024-08-01",
            "end_date": "2024-08-06",
            "days": 7
        })
        assert "start_date" in result
        assert "end_date" in result
        assert "days" in result
    
    def test_validate_unknown_tool(self):
        """Test validating unknown tool returns parameters as-is."""
        params = {"some_param": "some_value"}
        result = validate_tool_parameters("unknown_tool", params)
        assert result == params
    
    def test_validate_invalid_parameters(self):
        """Test validating invalid parameters raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_tool_parameters("get_daily_summary", {"date": "invalid_date"})


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_create_ai_friendly_error_validation(self):
        """Test creating AI-friendly error from ValidationError."""
        validation_error = ValidationError(
            "Test error message",
            field="test_field",
            suggestions=["suggestion1", "suggestion2"],
            examples=["example1", "example2"]
        )
        
        result = create_ai_friendly_error(validation_error, "test_tool")
        
        assert result["isError"] is True
        assert "error" in result
        assert result["error"]["message"] == "Test error message"
        assert result["error"]["field"] == "test_field"
        assert result["error"]["suggestions"] == ["suggestion1", "suggestion2"]
        assert result["error"]["examples"] == ["example1", "example2"]
    
    def test_create_ai_friendly_error_generic(self):
        """Test creating AI-friendly error from generic exception."""
        generic_error = ValueError("Generic test error")
        
        result = create_ai_friendly_error(generic_error, "test_tool")
        
        assert result["isError"] is True
        assert "error" in result
        assert "ValueError: Generic test error" in result["error"]["message"]
        assert result["error"]["field"] == "test_tool"
    
    def test_create_ai_friendly_error_with_suggestions(self):
        """Test error handling includes contextual suggestions."""
        auth_error = Exception("authentication failed")
        
        result = create_ai_friendly_error(auth_error, "get_profile")
        
        assert result["isError"] is True
        assert len(result["error"]["suggestions"]) > 0
        
    def test_validation_error_to_mcp_error(self):
        """Test ValidationError to MCP error conversion."""
        error = ValidationError(
            "Test message",
            field="test_field",
            suggestions=["test_suggestion"],
            examples=["test_example"]
        )
        
        result = error.to_mcp_error()
        
        assert result["message"] == "Test message"
        assert result["field"] == "test_field"
        assert result["suggestions"] == ["test_suggestion"]
        assert result["examples"] == ["test_example"]
        assert result["severity"] == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])