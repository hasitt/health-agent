"""
Unit tests for AI optimization system.
"""

import pytest
from datetime import datetime

from garmin_mcp.ai_optimization import (
    HealthDataFormatter,
    ConversationOptimizer,
    PromptTemplates,
    enhance_data_for_ai
)


class TestHealthDataFormatter:
    """Test cases for HealthDataFormatter."""
    
    def test_format_general_success(self):
        """Test general formatting with successful data."""
        data = {"status": "success", "some_data": "value"}
        result = HealthDataFormatter.format_for_conversation(data, "general")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_format_general_no_data(self):
        """Test general formatting with no data."""
        result = HealthDataFormatter.format_for_conversation(None, "general")
        assert result == "No health data available."
        
        result = HealthDataFormatter.format_for_conversation({"status": "error"}, "general")
        assert result == "No health data available."
    
    def test_format_daily_summary(self):
        """Test daily summary formatting."""
        data = {
            "status": "success",
            "summary": {
                "date": "2024-08-06",
                "steps": 12500,
                "distance_km": 8.2,
                "active_calories": 456,
                "goal_steps": 10000
            }
        }
        
        result = HealthDataFormatter.format_for_conversation(data, "daily_summary")
        
        assert "12,500 steps" in result
        assert "8.20 km" in result
        assert "456 active calories" in result
        assert "achieved your goal" in result.lower()
    
    def test_format_daily_summary_goal_not_reached(self):
        """Test daily summary formatting when goal not reached."""
        data = {
            "status": "success",
            "summary": {
                "date": "2024-08-06",
                "steps": 7500,
                "distance_km": 5.1,
                "active_calories": 300,
                "goal_steps": 10000
            }
        }
        
        result = HealthDataFormatter.format_for_conversation(data, "daily_summary")
        
        assert "7,500 steps" in result
        assert "75%" in result or "2,500 steps remaining" in result
    
    def test_format_sleep_data(self):
        """Test sleep data formatting."""
        data = {
            "status": "success",
            "sleep_data": {
                "date": "2024-08-05",
                "sleep_duration_hours": 7.5,
                "sleep_score": 82
            }
        }
        
        result = HealthDataFormatter.format_for_conversation(data, "sleep_analysis")
        
        assert "7.5 hours" in result
        assert "82/100" in result
        assert "good quality" in result.lower()
    
    def test_format_sleep_data_poor_quality(self):
        """Test sleep data formatting with poor quality."""
        data = {
            "status": "success",
            "sleep_data": {
                "date": "2024-08-05",
                "sleep_duration_hours": 5.2,
                "sleep_score": 45
            }
        }
        
        result = HealthDataFormatter.format_for_conversation(data, "sleep_analysis")
        
        assert "5.2 hours" in result
        assert "45/100" in result
        assert "poor quality" in result.lower()
        assert "Consider getting more sleep" in result
    
    def test_format_activity_data(self):
        """Test activity data formatting."""
        data = {
            "status": "success",
            "activities_data": {
                "total_activities": 2,
                "total_duration_minutes": 75,
                "total_distance_km": 6.5,
                "total_calories": 420,
                "activities": [
                    {"activity_name": "Running"},
                    {"activity_name": "Cycling"}
                ]
            }
        }
        
        result = HealthDataFormatter.format_for_conversation(data, "activity_summary")
        
        assert "2 activity(ies)" in result
        assert "75 minutes" in result
        assert "6.50 km" in result
        assert "420 calories" in result
        assert "Running" in result
        assert "Cycling" in result
    
    def test_format_trends_data(self):
        """Test trends data formatting."""
        data = {
            "status": "success",
            "trends_analysis": {
                "trends": [
                    {
                        "metric": "daily_steps",
                        "trend_direction": "increasing",
                        "change_percent": 12.5,
                        "significance": "moderate"
                    },
                    {
                        "metric": "sleep_duration",
                        "trend_direction": "decreasing", 
                        "change_percent": -8.3,
                        "significance": "significant"
                    }
                ]
            }
        }
        
        result = HealthDataFormatter.format_for_conversation(data, "trends")
        
        assert "📈" in result
        assert "Daily Steps: up 12.5%" in result
        assert "📉" in result
        assert "Sleep Duration: down 8.3%" in result
    
    def test_format_insights_data(self):
        """Test insights data formatting."""
        data = {
            "status": "success",
            "health_insights": {
                "insights": [
                    {
                        "title": "Improved Sleep Pattern",
                        "description": "Your sleep consistency has improved over the past week",
                        "recommendation": "Continue maintaining this sleep schedule",
                        "confidence": 0.85,
                        "trend": "positive"
                    },
                    {
                        "title": "Activity Decline",
                        "description": "Your daily activity has decreased by 15%",
                        "recommendation": "Try to increase daily walks",
                        "confidence": 0.72,
                        "trend": "negative"
                    }
                ]
            }
        }
        
        result = HealthDataFormatter.format_for_conversation(data, "insights")
        
        assert "✅ Improved Sleep Pattern" in result
        assert "⚠️ Activity Decline" in result
        assert "Continue maintaining this sleep schedule" in result
        assert "Try to increase daily walks" in result


class TestConversationOptimizer:
    """Test cases for ConversationOptimizer."""
    
    def test_create_contextual_summary_empty(self):
        """Test creating summary with empty data."""
        result = ConversationOptimizer.create_contextual_summary([], "general")
        
        assert result["conversation_context"] == "general"
        assert result["formatted_response"] == "No health data available for analysis."
        assert result["data_completeness"] == "missing"
    
    def test_create_contextual_summary_performance(self):
        """Test creating summary with performance context."""
        data_points = [
            {"steps": 10000, "heart_rate": 140, "activity_duration": 45},
            {"steps": 12000, "heart_rate": 135, "activity_duration": 60}
        ]
        
        result = ConversationOptimizer.create_contextual_summary(data_points, "performance")
        
        assert result["conversation_context"] == "performance"
        assert len(result["key_insights"]) > 0
        assert len(result["actionable_items"]) > 0
        assert "performance" in result["key_insights"][0].lower()
    
    def test_create_contextual_summary_health_check(self):
        """Test creating summary with health check context."""
        data_points = [{"sleep": 7.5, "stress": 25, "rhr": 55}]
        
        result = ConversationOptimizer.create_contextual_summary(data_points, "health_check")
        
        assert result["conversation_context"] == "health_check"
        assert "health" in result["key_insights"][0].lower()
        assert len(result["actionable_items"]) > 0
    
    def test_create_contextual_summary_goals(self):
        """Test creating summary with goal context."""
        data_points = [{"goal_progress": 85, "streak": 7}]
        
        result = ConversationOptimizer.create_contextual_summary(data_points, "goal_progress")
        
        assert result["conversation_context"] == "goal_progress"
        assert "goal" in result["key_insights"][0].lower()
    
    def test_create_contextual_summary_comparison(self):
        """Test creating summary with comparison context."""
        data_points = [{"week1": 8000, "week2": 9500}]
        
        result = ConversationOptimizer.create_contextual_summary(data_points, "comparison")
        
        assert result["conversation_context"] == "comparison"
        assert "trends" in result["key_insights"][0].lower()
    
    def test_data_completeness_assessment(self):
        """Test data completeness assessment."""
        # Minimal data - should be partial
        minimal_data = [{"steps": 5000}]
        result = ConversationOptimizer.create_contextual_summary(minimal_data, "general")
        assert result["data_completeness"] == "partial"
        
        # Rich data - should be complete
        rich_data = [{"steps": 5000, "sleep": 7.5, "heart_rate": 65, "activities": 2}]
        result = ConversationOptimizer.create_contextual_summary(rich_data, "general")
        assert result["data_completeness"] == "complete"


class TestPromptTemplates:
    """Test cases for PromptTemplates."""
    
    def test_get_template_daily_check_in(self):
        """Test getting daily check-in template."""
        result = PromptTemplates.get_template("daily_check_in")
        
        assert isinstance(result, str)
        assert "daily check-in" in result.lower()
        assert "achievements" in result.lower()
        assert "recommendation" in result.lower()
    
    def test_get_template_weekly_review(self):
        """Test getting weekly review template."""
        result = PromptTemplates.get_template("weekly_review")
        
        assert isinstance(result, str)
        assert "weekly" in result.lower()
        assert "performance" in result.lower()
    
    def test_get_template_performance_analysis(self):
        """Test getting performance analysis template."""
        result = PromptTemplates.get_template("performance_analysis")
        
        assert isinstance(result, str)
        assert "performance" in result.lower()
        assert "optimization" in result.lower()
    
    def test_get_template_sleep_optimization(self):
        """Test getting sleep optimization template."""
        result = PromptTemplates.get_template("sleep_optimization")
        
        assert isinstance(result, str)
        assert "sleep" in result.lower()
        assert "optimization" in result.lower()
    
    def test_get_template_goal_coaching(self):
        """Test getting goal coaching template."""
        result = PromptTemplates.get_template("goal_coaching")
        
        assert isinstance(result, str)
        assert "coach" in result.lower()
        assert "goal" in result.lower()
    
    def test_get_template_unknown(self):
        """Test getting unknown template returns default."""
        result = PromptTemplates.get_template("unknown_template")
        
        assert isinstance(result, str)
        assert "Analyze the provided health data" in result
    
    def test_create_custom_template(self):
        """Test creating custom template."""
        focus_areas = ["sleep quality", "activity levels", "stress management"]
        
        result = PromptTemplates.create_custom_template(focus_areas, "friendly")
        
        assert isinstance(result, str)
        assert "conversational and encouraging" in result
        assert "sleep quality" in result
        assert "activity levels" in result
        assert "stress management" in result
    
    def test_create_custom_template_clinical_tone(self):
        """Test creating custom template with clinical tone."""
        focus_areas = ["heart rate variability"]
        
        result = PromptTemplates.create_custom_template(focus_areas, "clinical")
        
        assert isinstance(result, str)
        assert "professional and analytical" in result
        assert "heart rate variability" in result
    
    def test_create_custom_template_unknown_tone(self):
        """Test creating custom template with unknown tone."""
        focus_areas = ["general health"]
        
        result = PromptTemplates.create_custom_template(focus_areas, "unknown_tone")
        
        assert isinstance(result, str)
        assert "balanced" in result


class TestDataEnhancement:
    """Test cases for data enhancement functions."""
    
    def test_enhance_data_for_ai_basic(self):
        """Test basic data enhancement."""
        data = {"status": "success", "steps": 10000}
        
        result = enhance_data_for_ai(data, "daily_summary")
        
        assert "ai_formatted" in result
        assert "ai_context" in result
        assert result["ai_context"]["conversation_ready"] is True
        assert result["ai_context"]["format_context"] == "daily_summary"
        assert isinstance(result["ai_context"]["enhancement_timestamp"], str)
    
    def test_enhance_data_for_ai_trends_context(self):
        """Test data enhancement with trends context."""
        data = {"trends": [{"metric": "steps", "direction": "up"}]}
        
        result = enhance_data_for_ai(data, "trends")
        
        assert "ai_context" in result
        assert "conversation_hints" in result["ai_context"]
        hints = result["ai_context"]["conversation_hints"]
        
        assert any("trend" in hint.lower() for hint in hints)
        assert any("benchmark" in hint.lower() for hint in hints)
    
    def test_enhance_data_for_ai_insights_context(self):
        """Test data enhancement with insights context."""
        data = {"insights": [{"title": "Test", "recommendation": "Do something"}]}
        
        result = enhance_data_for_ai(data, "insights")
        
        assert "ai_context" in result
        assert "conversation_hints" in result["ai_context"]
        hints = result["ai_context"]["conversation_hints"]
        
        assert any("actionable" in hint.lower() for hint in hints)
        assert any("reasoning" in hint.lower() for hint in hints)
    
    def test_enhance_data_for_ai_invalid_data(self):
        """Test data enhancement with invalid data."""
        result = enhance_data_for_ai(None, "general")
        assert result is None
        
        result = enhance_data_for_ai("not a dict", "general")
        assert result == "not a dict"
    
    def test_enhance_data_key_metrics_tracking(self):
        """Test that enhancement tracks available metrics."""
        data = {
            "steps": 10000,
            "sleep": 7.5,
            "heart_rate": 65,
            "activities": 2
        }
        
        result = enhance_data_for_ai(data, "general")
        
        expected_metrics = ["steps", "sleep", "heart_rate", "activities"]
        actual_metrics = result["ai_context"]["key_metrics_available"]
        
        for metric in expected_metrics:
            assert metric in actual_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])