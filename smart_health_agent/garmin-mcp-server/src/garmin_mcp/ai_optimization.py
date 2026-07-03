"""
AI agent optimization features for Garmin MCP Server.
Provides context-aware formatting, prompt templates, and conversation flow optimization.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import json

import structlog

logger = structlog.get_logger(__name__)


class HealthDataFormatter:
    """Formats health data for optimal AI agent consumption."""
    
    @staticmethod
    def format_for_conversation(data: Dict[str, Any], context: str = "general") -> str:
        """
        Format health data into natural language suitable for AI conversations.
        
        Args:
            data: Health data dictionary
            context: Context for formatting (general, summary, comparison, etc.)
        
        Returns:
            Natural language formatted string
        """
        if not data or data.get('status') != 'success':
            return "No health data available."
        
        if context == "daily_summary":
            return HealthDataFormatter._format_daily_summary(data)
        elif context == "sleep_analysis":
            return HealthDataFormatter._format_sleep_data(data)
        elif context == "activity_summary":
            return HealthDataFormatter._format_activity_data(data)
        elif context == "trends":
            return HealthDataFormatter._format_trends_data(data)
        elif context == "insights":
            return HealthDataFormatter._format_insights_data(data)
        else:
            return HealthDataFormatter._format_general(data)
    
    @staticmethod
    def _format_daily_summary(data: Dict[str, Any]) -> str:
        """Format daily summary for conversation."""
        if 'summary' not in data:
            return "Daily summary data not available."
        
        summary = data['summary']
        date_str = summary.get('date', 'today')
        steps = summary.get('steps', 0)
        distance = summary.get('distance_km', 0)
        calories = summary.get('active_calories', 0)
        goal_steps = summary.get('goal_steps')
        
        # Build natural language description
        parts = [f"On {date_str}, you walked {steps:,} steps"]
        
        if distance > 0:
            parts.append(f"covering {distance:.2f} km")
        
        if calories > 0:
            parts.append(f"and burned {calories} active calories")
        
        # Goal progress
        if goal_steps:
            if steps >= goal_steps:
                parts.append(f"✅ You achieved your step goal of {goal_steps:,} steps!")
            else:
                remaining = goal_steps - steps
                percentage = (steps / goal_steps) * 100
                parts.append(f"You're {percentage:.0f}% toward your goal of {goal_steps:,} steps ({remaining:,} steps remaining)")
        
        return ". ".join(parts) + "."
    
    @staticmethod
    def _format_sleep_data(data: Dict[str, Any]) -> str:
        """Format sleep data for conversation."""
        if 'sleep_data' not in data:
            return "Sleep data not available."
        
        sleep = data['sleep_data']
        duration = sleep.get('sleep_duration_hours', 0)
        score = sleep.get('sleep_score')
        date_str = sleep.get('date', 'last night')
        
        # Build sleep description
        parts = [f"You slept for {duration:.1f} hours {date_str}"]
        
        if score is not None:
            if score >= 85:
                quality = "excellent"
            elif score >= 75:
                quality = "good"
            elif score >= 60:
                quality = "fair"
            else:
                quality = "poor"
            parts.append(f"with a sleep score of {score}/100 ({quality} quality)")
        
        # Add recommendations
        if duration < 6:
            parts.append("Consider getting more sleep for better health and performance")
        elif duration >= 8:
            parts.append("Great job getting adequate rest!")
        
        return ". ".join(parts) + "."
    
    @staticmethod
    def _format_activity_data(data: Dict[str, Any]) -> str:
        """Format activity data for conversation."""
        if 'activities_data' not in data:
            return "Activity data not available."
        
        activities = data['activities_data']
        total_activities = activities.get('total_activities', 0)
        total_duration = activities.get('total_duration_minutes', 0)
        total_distance = activities.get('total_distance_km', 0)
        total_calories = activities.get('total_calories', 0)
        
        if total_activities == 0:
            return "No activities recorded for this day."
        
        parts = [f"You completed {total_activities} activity(ies)"]
        
        if total_duration > 0:
            parts.append(f"totaling {total_duration} minutes")
        
        if total_distance > 0:
            parts.append(f"covering {total_distance:.2f} km")
        
        if total_calories > 0:
            parts.append(f"and burning {total_calories} calories")
        
        # Add activity details if available
        activity_list = activities.get('activities', [])
        if activity_list:
            activity_types = [act.get('activity_name', 'Unknown') for act in activity_list]
            if len(activity_types) <= 3:
                parts.append(f"Activities: {', '.join(activity_types)}")
        
        return ". ".join(parts) + "."
    
    @staticmethod
    def _format_trends_data(data: Dict[str, Any]) -> str:
        """Format trends analysis for conversation."""
        if 'trends_analysis' not in data:
            return "Trends analysis not available."
        
        trends = data['trends_analysis']
        trend_list = trends.get('trends', [])
        
        if not trend_list:
            return "No trends detected in your health data."
        
        parts = []
        for trend in trend_list:
            metric = trend.get('metric', 'unknown')
            direction = trend.get('trend_direction', 'stable')
            change_percent = trend.get('change_percent', 0)
            significance = trend.get('significance', 'minimal')
            
            if significance == 'minimal':
                continue  # Skip minimal changes
            
            if direction == 'increasing':
                emoji = "📈"
                desc = f"up {abs(change_percent):.1f}%"
            elif direction == 'decreasing':
                emoji = "📉"
                desc = f"down {abs(change_percent):.1f}%"
            else:
                emoji = "📊"
                desc = "stable"
            
            parts.append(f"{emoji} {metric.replace('_', ' ').title()}: {desc} ({significance} change)")
        
        if not parts:
            return "Your health metrics are stable with no significant changes."
        
        return "Trends in your health data: " + "; ".join(parts) + "."
    
    @staticmethod
    def _format_insights_data(data: Dict[str, Any]) -> str:
        """Format health insights for conversation."""
        if 'health_insights' not in data:
            return "Health insights not available."
        
        insights = data['health_insights'].get('insights', [])
        
        if not insights:
            return "No significant health insights detected."
        
        formatted_insights = []
        for insight in insights:
            title = insight.get('title', 'Insight')
            description = insight.get('description', '')
            recommendation = insight.get('recommendation')
            confidence = insight.get('confidence', 0)
            trend = insight.get('trend', 'neutral')
            
            # Add emoji based on trend
            if trend == 'positive':
                emoji = "✅"
            elif trend == 'negative':
                emoji = "⚠️"
            else:
                emoji = "ℹ️"
            
            insight_text = f"{emoji} {title}: {description}"
            
            if recommendation:
                insight_text += f" Recommendation: {recommendation}"
            
            formatted_insights.append(insight_text)
        
        return "\n\n".join(formatted_insights)
    
    @staticmethod
    def _format_general(data: Dict[str, Any]) -> str:
        """General formatting for unknown data types."""
        try:
            # Extract key metrics if available
            if isinstance(data, dict) and 'status' in data:
                if data['status'] == 'success':
                    return "Health data retrieved successfully."
                else:
                    return f"Health data status: {data.get('message', 'Unknown status')}"
            return "Health data processed successfully."
        except Exception:
            return "Health data available but format unclear."


class ConversationOptimizer:
    """Optimizes data presentation for conversational AI interactions."""
    
    @staticmethod
    def create_contextual_summary(
        data_points: List[Dict[str, Any]], 
        query_intent: str = "general"
    ) -> Dict[str, Any]:
        """
        Create a contextual summary optimized for AI conversations.
        
        Args:
            data_points: List of health data points
            query_intent: Detected intent of the user query
            
        Returns:
            Optimized summary for AI consumption
        """
        summary = {
            "conversation_context": query_intent,
            "key_insights": [],
            "actionable_items": [],
            "formatted_response": "",
            "data_completeness": "complete"
        }
        
        if not data_points:
            summary["formatted_response"] = "No health data available for analysis."
            summary["data_completeness"] = "missing"
            return summary
        
        # Analyze data completeness
        available_metrics = set()
        for data_point in data_points:
            if isinstance(data_point, dict):
                available_metrics.update(data_point.keys())
        
        if len(available_metrics) < 3:
            summary["data_completeness"] = "partial"
        
        # Generate context-specific insights
        if query_intent == "performance":
            summary = ConversationOptimizer._optimize_for_performance(data_points, summary)
        elif query_intent == "health_check":
            summary = ConversationOptimizer._optimize_for_health_check(data_points, summary)
        elif query_intent == "goal_progress":
            summary = ConversationOptimizer._optimize_for_goals(data_points, summary)
        elif query_intent == "comparison":
            summary = ConversationOptimizer._optimize_for_comparison(data_points, summary)
        else:
            summary = ConversationOptimizer._optimize_general(data_points, summary)
        
        return summary
    
    @staticmethod
    def _optimize_for_performance(data_points: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize for performance-related queries."""
        summary["key_insights"] = [
            "Focus on activity levels, recovery metrics, and performance trends",
            "Look for correlations between sleep, heart rate, and activity performance"
        ]
        summary["actionable_items"] = [
            "Monitor resting heart rate for recovery status",
            "Track sleep quality impact on next-day performance",
            "Identify optimal activity patterns"
        ]
        return summary
    
    @staticmethod
    def _optimize_for_health_check(data_points: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize for health check queries."""
        summary["key_insights"] = [
            "Overall health status assessment",
            "Identify any concerning patterns or trends",
            "Highlight positive health indicators"
        ]
        summary["actionable_items"] = [
            "Address any health concerns identified",
            "Maintain positive health behaviors",
            "Consider consulting healthcare providers for unusual patterns"
        ]
        return summary
    
    @staticmethod
    def _optimize_for_goals(data_points: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize for goal progress queries."""
        summary["key_insights"] = [
            "Current progress toward fitness goals",
            "Identify successful strategies and patterns",
            "Highlight areas needing attention"
        ]
        summary["actionable_items"] = [
            "Celebrate achieved goals and milestones",
            "Adjust strategies for underperforming goals",
            "Set realistic and progressive targets"
        ]
        return summary
    
    @staticmethod
    def _optimize_for_comparison(data_points: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize for comparison queries."""
        summary["key_insights"] = [
            "Trends and changes over time",
            "Comparative analysis of different periods",
            "Identify improvement areas and successes"
        ]
        summary["actionable_items"] = [
            "Continue successful patterns",
            "Address declining trends",
            "Set targets based on historical performance"
        ]
        return summary
    
    @staticmethod
    def _optimize_general(data_points: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
        """General optimization for unclear intent."""
        summary["key_insights"] = [
            "General health and fitness overview",
            "Recent activity and recovery patterns",
            "Overall wellness indicators"
        ]
        summary["actionable_items"] = [
            "Maintain consistent healthy habits",
            "Monitor key health metrics",
            "Focus on areas showing room for improvement"
        ]
        return summary


class PromptTemplates:
    """Pre-defined prompt templates for optimal AI interactions."""
    
    DAILY_CHECK_IN = """
    Based on the health data provided, give me a brief daily check-in summary including:
    - Today's key achievements and metrics
    - How I'm progressing toward my goals  
    - Any notable patterns or concerns
    - One specific recommendation for tomorrow
    
    Keep the response conversational and motivating.
    """
    
    WEEKLY_REVIEW = """
    Provide a comprehensive weekly health review including:
    - Overall weekly performance summary
    - Goal achievement analysis
    - Key trends and patterns observed
    - Areas of improvement and success
    - Specific recommendations for the coming week
    
    Format as a structured but friendly weekly report.
    """
    
    PERFORMANCE_ANALYSIS = """
    Analyze my health and fitness performance focusing on:
    - Current performance levels vs. historical data
    - Recovery and readiness indicators
    - Performance-limiting factors identified
    - Optimization recommendations
    
    Provide actionable insights for performance improvement.
    """
    
    SLEEP_OPTIMIZATION = """
    Focus on sleep patterns and optimization:
    - Recent sleep quality and duration analysis
    - Factors affecting sleep performance
    - Sleep's impact on other health metrics
    - Specific sleep improvement strategies
    
    Provide personalized sleep optimization advice.
    """
    
    GOAL_COACHING = """
    Act as a health coach reviewing goal progress:
    - Current goal achievement status
    - Successful strategies and patterns
    - Challenges and obstacles identified
    - Motivational feedback and encouragement
    - Adjusted recommendations and next steps
    
    Provide supportive and actionable coaching guidance.
    """
    
    @staticmethod
    def get_template(template_name: str) -> str:
        """Get a specific prompt template."""
        templates = {
            "daily_check_in": PromptTemplates.DAILY_CHECK_IN,
            "weekly_review": PromptTemplates.WEEKLY_REVIEW,
            "performance_analysis": PromptTemplates.PERFORMANCE_ANALYSIS,
            "sleep_optimization": PromptTemplates.SLEEP_OPTIMIZATION,
            "goal_coaching": PromptTemplates.GOAL_COACHING,
        }
        return templates.get(template_name, "Analyze the provided health data and provide insights.")
    
    @staticmethod
    def create_custom_template(focus_areas: List[str], tone: str = "friendly") -> str:
        """Create a custom prompt template based on focus areas."""
        tone_adjectives = {
            "friendly": "conversational and encouraging",
            "clinical": "professional and analytical", 
            "coaching": "motivational and action-oriented",
            "technical": "detailed and data-focused"
        }
        
        template = f"""
        Analyze the health data with a {tone_adjectives.get(tone, 'balanced')} approach, focusing on:
        """
        
        for area in focus_areas:
            template += f"\n        - {area}"
        
        template += f"""
        
        Provide insights and recommendations in a {tone} tone that helps the user understand
        their health patterns and take actionable steps for improvement.
        """
        
        return template.strip()


def enhance_data_for_ai(data: Dict[str, Any], context: str = "general") -> Dict[str, Any]:
    """
    Enhance raw health data with AI-optimized formatting and context.
    
    Args:
        data: Raw health data
        context: Context for enhancement
        
    Returns:
        Enhanced data with AI-friendly formatting
    """
    if not data or not isinstance(data, dict):
        return data
    
    enhanced = data.copy()
    
    # Add natural language formatting
    enhanced["ai_formatted"] = HealthDataFormatter.format_for_conversation(data, context)
    
    # Add contextual metadata
    enhanced["ai_context"] = {
        "format_context": context,
        "conversation_ready": True,
        "key_metrics_available": list(data.keys()) if isinstance(data, dict) else [],
        "enhancement_timestamp": datetime.now().isoformat()
    }
    
    # Add conversation hints for AI
    if context == "daily_summary":
        enhanced["ai_context"]["conversation_hints"] = [
            "Focus on goal achievement and progress",
            "Highlight notable changes from typical patterns",
            "Provide encouraging or motivational feedback"
        ]
    elif context == "trends":
        enhanced["ai_context"]["conversation_hints"] = [
            "Explain trend significance and implications",
            "Compare to healthy benchmarks",
            "Suggest actions based on trends"
        ]
    elif context == "insights":
        enhanced["ai_context"]["conversation_hints"] = [
            "Prioritize actionable insights",
            "Explain the reasoning behind recommendations", 
            "Connect insights to user goals and preferences"
        ]
    
    return enhanced