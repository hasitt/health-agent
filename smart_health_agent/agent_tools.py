"""
LangChain tools for the Health Detective agent.
These tools wrap existing health analysis and visualization functions 
to make them available to the LangChain agent.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field, validator
from typing import List, Any, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
import ast

# Import existing modules
import health_visualizations
import trend_analyzer
from database import db

logger = logging.getLogger(__name__)

# Get available metrics from the existing configuration
try:
    from smart_health_ollama import ALL_AVAILABLE_METRICS
except ImportError:
    # Fallback list if import fails
    ALL_AVAILABLE_METRICS = [
        'total_steps', 'active_calories', 'sleep_duration_hours', 'sleep_score',
        'total_calories', 'protein_g', 'mood', 'energy', 'stress'
    ]

class TimeSeriesPlotInput(BaseModel):
    """Input schema for time series plot generation."""
    metrics: List[str] = Field(
        description=f"List of health metrics to plot. Valid options: {', '.join(ALL_AVAILABLE_METRICS)}"
    )
    days: int = Field(
        description="Number of past days to include in the plot (default: 30, max: 365)",
        default=30,
        ge=1,
        le=365
    )
    
    @validator('metrics', pre=True)
    def parse_metrics(cls, v):
        """Parse metrics input to handle string representations of lists."""
        if isinstance(v, str):
            # Handle string representation of a list like "['total_steps']"
            if v.startswith('[') and v.endswith(']'):
                try:
                    # Use ast.literal_eval for safe evaluation of list literals
                    parsed = ast.literal_eval(v)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]  # Ensure all items are strings
                except (ValueError, SyntaxError):
                    pass
            # Handle comma-separated string like "total_steps,active_calories"
            if ',' in v:
                return [metric.strip() for metric in v.split(',')]
            # Handle single metric as string
            return [v.strip()]
        elif isinstance(v, list):
            # Already a list, ensure all items are strings
            return [str(item) for item in v]
        else:
            # Convert other types to single-item list
            return [str(v)]

class TrendAnalysisInput(BaseModel):
    """Input schema for trend analysis."""
    metric_x: str = Field(
        description=f"Independent variable metric. Valid options: {', '.join(ALL_AVAILABLE_METRICS)}"
    )
    metric_y: str = Field(
        description=f"Dependent variable metric. Valid options: {', '.join(ALL_AVAILABLE_METRICS)}" 
    )
    analysis_type: str = Field(
        description="Type of analysis to perform",
        default="correlation"
    )
    days: int = Field(
        description="Number of past days to include in analysis (default: 30)",
        default=30,
        ge=7,
        le=365
    )

class HealthDataQueryInput(BaseModel):
    """Input schema for querying basic health data."""
    data_type: str = Field(
        description="Type of health data to query: sleep, activity, nutrition, mood, or all"
    )
    days: int = Field(
        description="Number of past days to include (default: 7)",
        default=7,
        ge=1,
        le=90
    )

@tool("generate_time_series_plots", args_schema=TimeSeriesPlotInput)
def generate_time_series_plots_tool(metrics: List[str], days: int = 30) -> Dict[str, Any]:
    """
    Generates time-series Plotly figures for health metrics over a specified period.
    
    This tool creates interactive charts showing how health metrics change over time.
    Perfect for visualizing trends in sleep, activity, nutrition, or mood data.
    """
    try:
        logger.info(f"Generating time series plots for metrics: {metrics}, days: {days}")
        
        # Validate metrics
        invalid_metrics = [m for m in metrics if m not in ALL_AVAILABLE_METRICS]
        if invalid_metrics:
            return {
                "error": f"Invalid metrics: {invalid_metrics}. Valid options: {ALL_AVAILABLE_METRICS}",
                "plot": None
            }
        
        # Generate the plot using existing visualization function (it generates plots for all metrics)
        # Import current_user_id from the main module
        try:
            from smart_health_ollama import current_user_id
            user_id = current_user_id
        except ImportError:
            user_id = 1  # Fallback to default user
            
        plots = health_visualizations.generate_time_series_plots(
            user_id=user_id, 
            days=days
        )
        
        if plots and len(plots) > 0:
            # Return simple success message - plot will be handled separately by the UI
            return {
                "plot": plots[0],  # Plot for UI display
                "message": f"✅ Successfully generated interactive visualization for {', '.join(metrics)} over {days} days."
            }
        else:
            return {
                "error": "No data available for the requested time period",
                "plot": None
            }
            
    except Exception as e:
        logger.error(f"Error generating time series plots: {e}")
        return {
            "error": f"Failed to generate plots: {str(e)}",
            "plot": None
        }

@tool("perform_custom_analysis", args_schema=TrendAnalysisInput)
def perform_custom_analysis_tool(
    metric_x: str, 
    metric_y: str, 
    analysis_type: str = "correlation", 
    days: int = 30
) -> Dict[str, Any]:
    """
    Analyzes relationships between two health metrics using statistical methods.
    
    This tool performs correlation analysis, regression, or other statistical analyses
    to understand how different health metrics relate to each other.
    """
    try:
        logger.info(f"Performing {analysis_type} analysis: {metric_x} vs {metric_y} over {days} days")
        
        # Validate metrics
        if metric_x not in ALL_AVAILABLE_METRICS:
            return {"error": f"Invalid metric_x: {metric_x}. Valid options: {ALL_AVAILABLE_METRICS}"}
        
        if metric_y not in ALL_AVAILABLE_METRICS:
            return {"error": f"Invalid metric_y: {metric_y}. Valid options: {ALL_AVAILABLE_METRICS}"}
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Get current user ID
        try:
            from smart_health_ollama import current_user_id
            user_id = current_user_id
        except ImportError:
            user_id = 1  # Fallback to default user
        
        # Perform analysis using existing trend analyzer
        result = trend_analyzer.analyze_custom_correlation(
            user_id=user_id,
            metric_x_name=metric_x,
            metric_y_name=metric_y,
            start_date=start_date,
            end_date=end_date,
            correlation_type=analysis_type if analysis_type in ['pearson', 'average_comparison', 'time_based_impact'] else 'pearson',
            filters=[]
        )
        
        # The result contains the analysis, but no plot is generated by this function
        plot = None  # This function doesn't generate plots
        
        return {
            "analysis_result": result,
            "plot": plot,
            "metric_x": metric_x,
            "metric_y": metric_y,
            "analysis_type": analysis_type,
            "days": days
        }
        
    except Exception as e:
        logger.error(f"Error performing custom analysis: {e}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "analysis_result": None,
            "plot": None
        }

@tool("get_health_data_summary", args_schema=HealthDataQueryInput)
def get_health_data_summary_tool(data_type: str = "all", days: int = 7) -> Dict[str, Any]:
    """
    Retrieves and summarizes recent health data for analysis.
    
    This tool provides current statistics and trends for sleep, activity, 
    nutrition, mood, or all health metrics over a specified time period.
    """
    try:
        logger.info(f"Getting health data summary for {data_type} over {days} days")
        
        db.connect()
        conn = db.get_connection()
        cursor = conn.cursor()
        
        summary = {}
        
        # Get sleep data
        if data_type in ["sleep", "all"]:
            cursor.execute('''
                SELECT AVG(sleep_duration_hours), AVG(sleep_score), COUNT(*),
                       MIN(sleep_duration_hours), MAX(sleep_duration_hours)
                FROM garmin_sleep 
                WHERE date >= date('now', '-{} days') AND sleep_duration_hours > 0
            '''.format(days))
            
            sleep_data = cursor.fetchone()
            if sleep_data and sleep_data[2] > 0:
                summary['sleep'] = {
                    'avg_duration': round(sleep_data[0] or 0, 1),
                    'avg_score': round(sleep_data[1] or 0, 0),
                    'min_duration': round(sleep_data[3] or 0, 1),
                    'max_duration': round(sleep_data[4] or 0, 1),
                    'nights_tracked': sleep_data[2],
                    'period_days': days
                }
        
        # Get activity data
        if data_type in ["activity", "all"]:
            cursor.execute('''
                SELECT AVG(total_steps), AVG(active_calories), COUNT(*),
                       MIN(total_steps), MAX(total_steps)
                FROM garmin_daily_summary 
                WHERE date >= date('now', '-{} days') AND total_steps > 0
            '''.format(days))
            
            activity_data = cursor.fetchone()
            if activity_data and activity_data[2] > 0:
                summary['activity'] = {
                    'avg_steps': round(activity_data[0] or 0, 0),
                    'avg_active_calories': round(activity_data[1] or 0, 0),
                    'min_steps': round(activity_data[3] or 0, 0),
                    'max_steps': round(activity_data[4] or 0, 0),
                    'days_tracked': activity_data[2],
                    'period_days': days
                }
        
        # Get nutrition data
        if data_type in ["nutrition", "all"]:
            cursor.execute('''
                SELECT AVG(total_calories), AVG(protein_g), COUNT(*),
                       MIN(total_calories), MAX(total_calories)
                FROM food_log_daily 
                WHERE date >= date('now', '-{} days') AND total_calories > 0
            '''.format(days))
            
            nutrition_data = cursor.fetchone()
            if nutrition_data and nutrition_data[2] > 0:
                summary['nutrition'] = {
                    'avg_calories': round(nutrition_data[0] or 0, 0),
                    'avg_protein': round(nutrition_data[1] or 0, 1),
                    'min_calories': round(nutrition_data[3] or 0, 0),
                    'max_calories': round(nutrition_data[4] or 0, 0),
                    'days_tracked': nutrition_data[2],
                    'period_days': days
                }
        
        # Get mood data
        if data_type in ["mood", "all"]:
            cursor.execute('''
                SELECT AVG(mood), AVG(energy), AVG(stress), COUNT(*),
                       MIN(mood), MAX(mood)
                FROM subjective_wellbeing 
                WHERE date >= date('now', '-{} days') AND mood > 0
            '''.format(days))
            
            mood_data = cursor.fetchone()
            if mood_data and mood_data[3] > 0:
                summary['mood'] = {
                    'avg_mood': round(mood_data[0] or 0, 1),
                    'avg_energy': round(mood_data[1] or 0, 1),
                    'avg_stress': round(mood_data[2] or 0, 1),
                    'min_mood': round(mood_data[4] or 0, 1),
                    'max_mood': round(mood_data[5] or 0, 1),
                    'days_tracked': mood_data[3],
                    'period_days': days
                }
        
        return {
            "summary": summary,
            "data_type": data_type,
            "days": days,
            "available_categories": list(summary.keys())
        }
        
    except Exception as e:
        logger.error(f"Error getting health data summary: {e}")
        return {
            "error": f"Failed to get health data: {str(e)}",
            "summary": {}
        }

# Export the tools list for easy import
HEALTH_AGENT_TOOLS = [
    generate_time_series_plots_tool,
    perform_custom_analysis_tool,
    get_health_data_summary_tool
]