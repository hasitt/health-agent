"""
Pydantic models for Garmin MCP Server data structures.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


# Base response models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    timestamp: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="success")
    message: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response model."""
    status: str = Field(default="error")
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Health data models
class DailySummary(BaseModel):
    """Daily activity summary data."""
    date: date
    steps: int = Field(ge=0)
    distance_km: float = Field(ge=0, description="Distance in kilometers")
    active_calories: int = Field(ge=0, description="Active calories burned")
    total_calories: Optional[int] = Field(default=None, ge=0, description="Total calories burned")
    floors_climbed: Optional[int] = Field(default=None, ge=0)
    active_minutes: Optional[int] = Field(default=None, ge=0)
    goal_steps: Optional[int] = Field(default=None, ge=0, description="Daily step goal")
    goal_achievement_percent: Optional[float] = Field(default=None, ge=0, le=200)
    
    @property
    def distance_miles(self) -> float:
        """Distance in miles."""
        return self.distance_km * 0.621371
    
    @property
    def step_goal_met(self) -> bool:
        """Whether step goal was achieved."""
        return self.goal_steps is not None and self.steps >= self.goal_steps


class SleepData(BaseModel):
    """Sleep data and metrics."""
    date: date
    sleep_duration_hours: float = Field(ge=0, le=24)
    sleep_score: Optional[int] = Field(default=None, ge=0, le=100, description="Sleep quality score")
    deep_sleep_minutes: Optional[int] = Field(default=None, ge=0)
    light_sleep_minutes: Optional[int] = Field(default=None, ge=0)
    rem_sleep_minutes: Optional[int] = Field(default=None, ge=0)
    awake_minutes: Optional[int] = Field(default=None, ge=0)
    sleep_start_time: Optional[datetime] = None
    sleep_end_time: Optional[datetime] = None
    restfulness: Optional[str] = Field(default=None, description="poor, fair, good, excellent")
    
    @property
    def sleep_duration_minutes(self) -> int:
        """Sleep duration in minutes."""
        return int(self.sleep_duration_hours * 60)
    
    @property
    def sleep_efficiency(self) -> Optional[float]:
        """Sleep efficiency percentage."""
        if self.awake_minutes is None:
            return None
        total_time = self.sleep_duration_minutes + self.awake_minutes
        if total_time == 0:
            return None
        return (self.sleep_duration_minutes / total_time) * 100


class HeartRateData(BaseModel):
    """Heart rate data and statistics."""
    date: date
    resting_hr: Optional[int] = Field(default=None, ge=30, le=200, description="Resting heart rate")
    max_hr: Optional[int] = Field(default=None, ge=30, le=220, description="Maximum heart rate")
    avg_hr: Optional[int] = Field(default=None, ge=30, le=200, description="Average heart rate")
    hr_zones: Optional[Dict[str, int]] = Field(
        default=None, description="Time in each HR zone (minutes)"
    )
    hrv_score: Optional[float] = Field(default=None, ge=0, description="Heart rate variability")
    
    @validator("max_hr")
    def validate_max_hr_vs_resting(cls, v, values):
        """Ensure max HR is greater than resting HR."""
        resting = values.get("resting_hr")
        if v is not None and resting is not None and v <= resting:
            raise ValueError("Maximum HR must be greater than resting HR")
        return v


class StressData(BaseModel):
    """Stress level data and patterns."""
    date: date
    avg_stress: int = Field(ge=0, le=100, description="Average stress level")
    max_stress: int = Field(ge=0, le=100, description="Maximum stress level")
    min_stress: int = Field(ge=0, le=100, description="Minimum stress level")
    stress_duration_minutes: Optional[int] = Field(default=None, ge=0)
    rest_duration_minutes: Optional[int] = Field(default=None, ge=0)
    low_stress_minutes: Optional[int] = Field(default=None, ge=0)
    medium_stress_minutes: Optional[int] = Field(default=None, ge=0)
    high_stress_minutes: Optional[int] = Field(default=None, ge=0)
    stress_category: Optional[str] = Field(default=None, description="low, medium, high")
    
    @property
    def stress_percentage(self) -> Optional[float]:
        """Percentage of day under stress."""
        if self.stress_duration_minutes is None:
            return None
        return (self.stress_duration_minutes / (24 * 60)) * 100


class ActivitySummary(BaseModel):
    """Individual activity/workout summary."""
    activity_id: str
    activity_name: str
    activity_type: str
    start_time: datetime
    duration_minutes: int = Field(ge=0)
    distance_km: float = Field(ge=0)
    calories: int = Field(ge=0)
    avg_hr: Optional[int] = Field(default=None, ge=30, le=200)
    max_hr: Optional[int] = Field(default=None, ge=30, le=220)
    elevation_gain_m: Optional[float] = Field(default=None, ge=0)
    avg_pace: Optional[str] = Field(default=None, description="Average pace (min/km or min/mi)")
    
    @property
    def distance_miles(self) -> float:
        """Distance in miles."""
        return self.distance_km * 0.621371


class ActivitiesData(BaseModel):
    """Daily activities data."""
    date: date
    activities: List[ActivitySummary] = Field(default_factory=list)
    total_activities: int = Field(ge=0)
    total_duration_minutes: int = Field(ge=0)
    total_distance_km: float = Field(ge=0)
    total_calories: int = Field(ge=0)
    
    @property
    def total_distance_miles(self) -> float:
        """Total distance in miles."""
        return self.total_distance_km * 0.621371


# Aggregate data models
class WeeklySummary(BaseModel):
    """Weekly summary with averages and trends."""
    start_date: date
    end_date: date
    avg_steps: float = Field(ge=0)
    avg_sleep_hours: float = Field(ge=0, le=24)
    avg_sleep_score: Optional[float] = Field(default=None, ge=0, le=100)
    avg_resting_hr: Optional[float] = Field(default=None, ge=30, le=200)
    avg_stress: Optional[float] = Field(default=None, ge=0, le=100)
    total_distance_km: float = Field(ge=0)
    total_calories: int = Field(ge=0)
    total_activities: int = Field(ge=0)
    active_days: int = Field(ge=0, le=7, description="Days with significant activity")
    step_goal_days: int = Field(ge=0, le=7, description="Days meeting step goal")
    
    @property
    def total_distance_miles(self) -> float:
        """Total distance in miles."""
        return self.total_distance_km * 0.621371
    
    @property
    def step_goal_percentage(self) -> float:
        """Percentage of days meeting step goal."""
        return (self.step_goal_days / 7) * 100


class MonthlySummary(BaseModel):
    """Monthly summary with patterns and insights."""
    start_date: date
    end_date: date
    total_days: int = Field(ge=1, le=31)
    avg_daily_steps: float = Field(ge=0)
    avg_sleep_hours: float = Field(ge=0, le=24)
    avg_sleep_score: Optional[float] = Field(default=None, ge=0, le=100)
    avg_resting_hr: Optional[float] = Field(default=None, ge=30, le=200)
    avg_stress: Optional[float] = Field(default=None, ge=0, le=100)
    total_distance_km: float = Field(ge=0)
    total_calories: int = Field(ge=0)
    total_activities: int = Field(ge=0)
    best_step_day: Optional[date] = None
    best_sleep_night: Optional[date] = None
    consistency_score: Optional[float] = Field(default=None, ge=0, le=100)
    
    @property
    def avg_daily_distance_km(self) -> float:
        """Average daily distance in kilometers."""
        return self.total_distance_km / self.total_days if self.total_days > 0 else 0


# Analysis models
class TrendAnalysis(BaseModel):
    """Trend analysis between periods."""
    metric: str
    current_period: str
    previous_period: str
    current_average: float
    previous_average: float
    change: float
    change_percent: float
    trend_direction: str = Field(description="increasing, decreasing, stable")
    significance: str = Field(description="significant, moderate, minimal")


class GoalProgress(BaseModel):
    """Progress toward fitness goals."""
    goal_type: str = Field(description="steps, sleep, activities, etc.")
    target_value: Union[int, float]
    current_value: Union[int, float]
    achievement_percent: float = Field(ge=0, le=200)
    days_achieved: int = Field(ge=0)
    total_days: int = Field(ge=1)
    streak: int = Field(ge=0, description="Current achievement streak")
    best_streak: int = Field(ge=0, description="Best achievement streak")
    
    @property
    def goal_met(self) -> bool:
        """Whether the goal is currently met."""
        return self.current_value >= self.target_value
    
    @property
    def achievement_rate(self) -> float:
        """Rate of goal achievement (days achieved / total days)."""
        return (self.days_achieved / self.total_days) * 100 if self.total_days > 0 else 0


class HealthInsight(BaseModel):
    """AI-generated health insight."""
    category: str = Field(description="sleep, activity, recovery, etc.")
    title: str
    description: str
    confidence: float = Field(ge=0, le=1, description="Confidence in insight")
    recommendation: Optional[str] = None
    data_points: List[str] = Field(default_factory=list)
    trend: Optional[str] = Field(default=None, description="positive, negative, neutral")


# Profile and device models
class UserProfile(BaseModel):
    """User profile information."""
    full_name: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    unit_system: str = Field(default="metric", description="metric or imperial")
    time_zone: Optional[str] = None
    birth_date: Optional[date] = None
    gender: Optional[str] = None
    height_cm: Optional[float] = Field(default=None, ge=0)
    weight_kg: Optional[float] = Field(default=None, ge=0)
    
    @property
    def height_inches(self) -> Optional[float]:
        """Height in inches."""
        return self.height_cm / 2.54 if self.height_cm else None
    
    @property
    def weight_lbs(self) -> Optional[float]:
        """Weight in pounds."""
        return self.weight_kg * 2.20462 if self.weight_kg else None


class DeviceInfo(BaseModel):
    """Connected device information."""
    device_id: str
    device_name: str
    device_type: str
    firmware_version: Optional[str] = None
    battery_level: Optional[int] = Field(default=None, ge=0, le=100)
    last_sync: Optional[datetime] = None
    is_primary: bool = Field(default=False)
    capabilities: List[str] = Field(default_factory=list)


# Composite response models
class DailyDataResponse(BaseResponse):
    """Complete daily data response."""
    date: date
    summary: DailySummary
    sleep: Optional[SleepData] = None
    heart_rate: Optional[HeartRateData] = None
    stress: Optional[StressData] = None
    activities: Optional[ActivitiesData] = None


class WeeklyDataResponse(BaseResponse):
    """Weekly data response."""
    summary: WeeklySummary
    daily_data: List[DailyDataResponse] = Field(default_factory=list)
    trends: List[TrendAnalysis] = Field(default_factory=list)


class MonthlyDataResponse(BaseResponse):
    """Monthly data response."""
    summary: MonthlySummary
    weekly_summaries: List[WeeklySummary] = Field(default_factory=list)
    insights: List[HealthInsight] = Field(default_factory=list)


class ProfileResponse(BaseResponse):
    """User profile response."""
    profile: UserProfile
    devices: List[DeviceInfo] = Field(default_factory=list)
    goals: List[GoalProgress] = Field(default_factory=list)