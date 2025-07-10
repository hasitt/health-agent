"""
Health Trends Analyzer
Provides specific correlation analysis for health data trends.
"""

import database
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger('trend_analyzer')

def get_hourly_stress_consistency(user_id: int, date_obj: Optional[date] = None, 
                                stress_threshold: int = 25, 
                                consistency_threshold_minutes: int = 30) -> List[str]:
    """
    Identify periods within a day where stress was consistently above a threshold.
    
    Args:
        user_id: Database user ID
        date_obj: Specific date to analyze (defaults to yesterday)
        stress_threshold: Stress level threshold (default 25)
        consistency_threshold_minutes: Minimum duration for consistent period (default 30 minutes)
    
    Returns:
        List of factual observations about consistent high-stress periods
    """
    try:
        if date_obj is None:
            date_obj = (datetime.now() - timedelta(days=1)).date()
        
        with database.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get granular stress data for the specified date
            cursor.execute("""
                SELECT timestamp, stress_level 
                FROM garmin_stress_details 
                WHERE user_id = ? AND date = ? AND stress_level > ?
                ORDER BY timestamp
            """, (user_id, date_obj, stress_threshold))
            
            stress_points = cursor.fetchall()
            
            if not stress_points:
                return [f"On {date_obj}, no stress readings above {stress_threshold} were recorded."]
            
            # Group continuous high-stress periods
            consistent_periods = []
            current_period_start = None
            current_period_end = None
            
            for i, (timestamp_str, stress_level) in enumerate(stress_points):
                timestamp = datetime.fromisoformat(timestamp_str)
                
                if current_period_start is None:
                    current_period_start = timestamp
                    current_period_end = timestamp
                else:
                    # Check if this point is within reasonable time gap (e.g., 10 minutes)
                    time_gap = (timestamp - current_period_end).total_seconds() / 60
                    
                    if time_gap <= 10:  # Continue current period
                        current_period_end = timestamp
                    else:  # Start new period
                        # Check if previous period meets minimum duration
                        duration = (current_period_end - current_period_start).total_seconds() / 60
                        if duration >= consistency_threshold_minutes:
                            consistent_periods.append((current_period_start, current_period_end, duration))
                        
                        current_period_start = timestamp
                        current_period_end = timestamp
            
            # Check final period
            if current_period_start and current_period_end:
                duration = (current_period_end - current_period_start).total_seconds() / 60
                if duration >= consistency_threshold_minutes:
                    consistent_periods.append((current_period_start, current_period_end, duration))
            
            # Format results
            observations = []
            if not consistent_periods:
                observations.append(f"On {date_obj}, no periods of sustained high stress (>{stress_threshold}) lasting {consistency_threshold_minutes}+ minutes were detected.")
            else:
                for start_time, end_time, duration in consistent_periods:
                    observations.append(
                        f"On {date_obj}, stress was consistently above {stress_threshold} from "
                        f"{start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} ({int(duration)} minutes)."
                    )
            
            return observations
            
    except Exception as e:
        logger.error(f"Failed to analyze stress consistency: {e}")
        return [f"Error analyzing stress consistency for {date_obj}: {str(e)}"]

def get_steps_vs_sleep_effect(user_id: int, steps_threshold: int = 10000, 
                            days_back: int = 60) -> List[str]:
    """
    Analyze the impact of high step counts on subsequent night's sleep quality.
    
    Args:
        user_id: Database user ID
        steps_threshold: Step count threshold for "high step days" (default 10,000)
        days_back: Number of days to analyze (default 60)
    
    Returns:
        List of factual observations about steps vs sleep correlation
    """
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        with database.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get daily steps and corresponding next-day sleep scores
            cursor.execute("""
                SELECT 
                    ds.date,
                    ds.total_steps,
                    ds.avg_daily_stress,
                    COALESCE(sleep_next.sleep_score, 0) as next_sleep_score
                FROM garmin_daily_summary ds
                LEFT JOIN garmin_sleep sleep_next ON 
                    sleep_next.user_id = ds.user_id AND 
                    sleep_next.date = DATE(ds.date, '+1 day')
                WHERE ds.user_id = ? 
                    AND ds.date BETWEEN ? AND ?
                    AND ds.total_steps > 0
                ORDER BY ds.date
            """, (user_id, start_date, end_date))
            
            data = cursor.fetchall()
            
            if not data:
                return [f"No step/sleep data available for the last {days_back} days."]
            
            # Categorize days
            high_step_days = []
            low_step_days = []
            
            for date_str, steps, avg_stress, next_sleep_score in data:
                if next_sleep_score > 0:  # Only include days with valid sleep data
                    if steps >= steps_threshold:
                        high_step_days.append((steps, avg_stress, next_sleep_score))
                    elif steps < 5000:  # Low step threshold
                        low_step_days.append((steps, avg_stress, next_sleep_score))
            
            observations = []
            
            if not high_step_days and not low_step_days:
                observations.append(f"Insufficient data for steps vs sleep analysis over the last {days_back} days.")
                return observations
            
            # Calculate averages for high step days
            if high_step_days:
                avg_sleep_after_high_steps = sum(sleep for _, _, sleep in high_step_days) / len(high_step_days)
                avg_stress_on_high_step_days = sum(stress for _, stress, _ in high_step_days) / len(high_step_days)
                high_step_count = len(high_step_days)
            else:
                avg_sleep_after_high_steps = 0
                avg_stress_on_high_step_days = 0
                high_step_count = 0
            
            # Calculate averages for low step days
            if low_step_days:
                avg_sleep_after_low_steps = sum(sleep for _, _, sleep in low_step_days) / len(low_step_days)
                low_step_count = len(low_step_days)
            else:
                avg_sleep_after_low_steps = 0
                low_step_count = 0
            
            # Generate factual observations
            observations.append(
                f"Over the last {days_back} days: {high_step_count} days had {steps_threshold}+ steps, "
                f"{low_step_count} days had <5,000 steps."
            )
            
            if high_step_count > 0:
                observations.append(
                    f"Average sleep score on nights following {steps_threshold}+ step days: {avg_sleep_after_high_steps:.1f}. "
                    f"Average stress on {steps_threshold}+ step days: {avg_stress_on_high_step_days:.1f}."
                )
            
            if low_step_count > 0:
                observations.append(
                    f"Average sleep score on nights following <5,000 step days: {avg_sleep_after_low_steps:.1f}."
                )
            
            if high_step_count > 0 and low_step_count > 0:
                sleep_difference = avg_sleep_after_high_steps - avg_sleep_after_low_steps
                observations.append(
                    f"Sleep score difference: {sleep_difference:+.1f} points (high step days vs low step days)."
                )
            
            return observations
            
    except Exception as e:
        logger.error(f"Failed to analyze steps vs sleep effect: {e}")
        return [f"Error analyzing steps vs sleep correlation: {str(e)}"]

def get_activity_type_rhr_impact(user_id: int, days_back: int = 90) -> List[str]:
    """
    Compare the RHR impact of weight training versus cardio activities.
    
    Args:
        user_id: Database user ID
        days_back: Number of days to analyze (default 90)
    
    Returns:
        List of factual observations about activity type vs RHR correlation
    """
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        # Define activity type categories
        cardio_types = ['running', 'cycling', 'swimming', 'cardio', 'treadmill', 'elliptical']
        weight_training_types = ['strength_training', 'weight_lifting', 'resistance', 'weightlifting', 'gym']
        
        with database.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get activities and corresponding daily RHR
            cursor.execute("""
                SELECT 
                    DATE(a.start_time) as activity_date,
                    a.activity_type,
                    a.duration_minutes,
                    ds.avg_daily_rhr
                FROM garmin_activities a
                JOIN garmin_daily_summary ds ON 
                    ds.user_id = a.user_id AND 
                    ds.date = DATE(a.start_time)
                WHERE a.user_id = ? 
                    AND DATE(a.start_time) BETWEEN ? AND ?
                    AND ds.avg_daily_rhr > 0
                    AND a.duration_minutes > 20
                ORDER BY a.start_time
            """, (user_id, start_date, end_date))
            
            activity_data = cursor.fetchall()
            
            if not activity_data:
                return [f"No activity/RHR data available for the last {days_back} days."]
            
            # Categorize days by primary activity type
            weight_training_days = {}
            cardio_days = {}
            
            for activity_date, activity_type, duration, rhr in activity_data:
                activity_type_lower = activity_type.lower()
                
                # Check if it's weight training
                is_weight_training = any(wt_type in activity_type_lower for wt_type in weight_training_types)
                # Check if it's cardio
                is_cardio = any(cardio_type in activity_type_lower for cardio_type in cardio_types)
                
                if is_weight_training:
                    if activity_date not in weight_training_days:
                        weight_training_days[activity_date] = []
                    weight_training_days[activity_date].append((duration, rhr))
                elif is_cardio:
                    if activity_date not in cardio_days:
                        cardio_days[activity_date] = []
                    cardio_days[activity_date].append((duration, rhr))
            
            # Calculate average RHR for each activity type (taking the RHR from the day, not averaging multiple activities)
            weight_training_rhrs = []
            cardio_rhrs = []
            
            for date, activities in weight_training_days.items():
                # Take RHR from the day (all activities on same day have same RHR)
                rhr = activities[0][1]
                weight_training_rhrs.append(rhr)
            
            for date, activities in cardio_days.items():
                # Take RHR from the day
                rhr = activities[0][1]
                cardio_rhrs.append(rhr)
            
            observations = []
            
            # Generate factual observations
            wt_days = len(weight_training_rhrs)
            cardio_days_count = len(cardio_rhrs)
            
            observations.append(
                f"Over the last {days_back} days: {wt_days} days with weight training activities, "
                f"{cardio_days_count} days with cardio activities (20+ minutes duration)."
            )
            
            if wt_days > 0:
                avg_rhr_wt = sum(weight_training_rhrs) / len(weight_training_rhrs)
                observations.append(f"Average RHR on weight training days: {avg_rhr_wt:.1f} bpm.")
            
            if cardio_days_count > 0:
                avg_rhr_cardio = sum(cardio_rhrs) / len(cardio_rhrs)
                observations.append(f"Average RHR on cardio days: {avg_rhr_cardio:.1f} bpm.")
            
            if wt_days > 0 and cardio_days_count > 0:
                rhr_difference = avg_rhr_wt - avg_rhr_cardio
                observations.append(
                    f"RHR difference: {rhr_difference:+.1f} bpm (weight training days vs cardio days)."
                )
            
            if wt_days == 0 and cardio_days_count == 0:
                observations.append("No weight training or cardio activities (20+ minutes) found in the specified period.")
            
            return observations
            
    except Exception as e:
        logger.error(f"Failed to analyze activity type RHR impact: {e}")
        return [f"Error analyzing activity type vs RHR correlation: {str(e)}"]

def get_comprehensive_trends_summary(user_id: int) -> Dict[str, List[str]]:
    """
    Get a comprehensive summary of all trend analyses.
    
    Args:
        user_id: Database user ID
    
    Returns:
        Dictionary containing all trend analysis results
    """
    return {
        'stress_consistency': get_hourly_stress_consistency(user_id),
        'steps_vs_sleep': get_steps_vs_sleep_effect(user_id),
        'activity_rhr_impact': get_activity_type_rhr_impact(user_id)
    }

def get_recent_stress_patterns(user_id: int, days_back: int = 7) -> List[str]:
    """
    Analyze recent stress patterns for the last week.
    
    Args:
        user_id: Database user ID
        days_back: Number of days to analyze (default 7)
    
    Returns:
        List of factual observations about recent stress patterns
    """
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        with database.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get daily stress averages
            cursor.execute("""
                SELECT date, avg_daily_stress, max_daily_stress
                FROM garmin_daily_summary
                WHERE user_id = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """, (user_id, start_date, end_date))
            
            stress_data = cursor.fetchall()
            
            if not stress_data:
                return [f"No stress data available for the last {days_back} days."]
            
            observations = []
            stress_values = [stress for _, stress, _ in stress_data if stress > 0]
            
            if stress_values:
                avg_stress = sum(stress_values) / len(stress_values)
                max_stress = max(max_stress for _, _, max_stress in stress_data if max_stress > 0)
                high_stress_days = len([s for s in stress_values if s > 30])
                
                observations.append(
                    f"Last {days_back} days average stress: {avg_stress:.1f}/100. "
                    f"Peak stress: {max_stress}/100. High stress days (>30): {high_stress_days}."
                )
            
            return observations
            
    except Exception as e:
        logger.error(f"Failed to analyze recent stress patterns: {e}")
        return [f"Error analyzing recent stress patterns: {str(e)}"] 