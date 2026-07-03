# Garmin MCP Server API Documentation

## Overview

The Garmin MCP Server provides access to Garmin Connect health and fitness data through the Model Context Protocol (MCP). It implements JSON-RPC 2.0 for communication and provides 15 tools plus 4 resource endpoints.

## MCP Protocol Support

- **Protocol Version**: MCP 1.2+
- **Transport**: STDIO (Standard Input/Output)
- **Format**: JSON-RPC 2.0
- **Features**: Tools, Resources, Parameter validation, Error handling

## Authentication

The server authenticates with Garmin Connect using email/password credentials passed as environment variables:

```bash
GARMIN_EMAIL=your_email@example.com
GARMIN_PASSWORD=your_password_here
```

## Available Tools

### Daily Health Data

#### `get_daily_summary`
Get daily activity summary including steps, distance, and calories.

**Parameters:**
- `date` (optional, string): Target date in YYYY-MM-DD format, or relative terms like "today", "yesterday", "3 days ago"
  - Default: today
  - Examples: "2024-08-06", "today", "yesterday", "1 week ago"

**Response:**
```json
{
  "status": "success",
  "summary": {
    "date": "2024-08-06",
    "steps": 12500,
    "distance_km": 8.2,
    "active_calories": 456,
    "total_calories": 2150,
    "goal_steps": 10000,
    "goal_progress_percent": 125,
    "floors_climbed": 15,
    "active_minutes": 85
  }
}
```

#### `get_sleep_data`
Get sleep information including duration, score, and sleep stages.

**Parameters:**
- `date` (optional, string): Sleep date (night of sleep)
  - Default: last night

**Response:**
```json
{
  "status": "success", 
  "sleep_data": {
    "date": "2024-08-05",
    "sleep_duration_hours": 7.5,
    "sleep_score": 82,
    "deep_sleep_minutes": 95,
    "light_sleep_minutes": 280,
    "rem_sleep_minutes": 75,
    "awake_minutes": 15,
    "sleep_start_time": "23:15",
    "sleep_end_time": "06:45"
  }
}
```

#### `get_heart_rate_data`
Get heart rate data and analysis for a specific date.

**Parameters:**
- `date` (optional, string): Target date
  - Default: today

**Response:**
```json
{
  "status": "success",
  "heart_rate_data": {
    "date": "2024-08-06",
    "resting_hr": 58,
    "max_hr": 185,
    "avg_hr": 78,
    "hr_zones": {
      "zone_1": 45,
      "zone_2": 120, 
      "zone_3": 30,
      "zone_4": 15,
      "zone_5": 5
    },
    "stress_score": 25
  }
}
```

#### `get_stress_data`
Get stress level measurements and analysis.

**Parameters:**
- `date` (optional, string): Target date
  - Default: today

**Response:**
```json
{
  "status": "success",
  "stress_data": {
    "date": "2024-08-06",
    "avg_stress": 28,
    "max_stress": 65,
    "stress_time_low": 480,
    "stress_time_medium": 120,
    "stress_time_high": 15,
    "rest_periods": 6
  }
}
```

#### `get_body_battery`
Get body battery energy levels throughout the day.

**Parameters:**
- `date` (optional, string): Target date
  - Default: today

**Response:**
```json
{
  "status": "success",
  "body_battery": {
    "date": "2024-08-06", 
    "start_level": 95,
    "end_level": 45,
    "lowest_level": 25,
    "highest_level": 95,
    "charged_amount": 50,
    "drained_amount": 70
  }
}
```

### Activity Data

#### `get_activities`
Get list of recorded activities within a date range.

**Parameters:**
- `days` (optional, number): Number of days to look back
  - Default: 7
  - Range: 1-365
- `limit` (optional, number): Maximum activities to return  
  - Default: 100
  - Range: 1-1000

**Response:**
```json
{
  "status": "success",
  "activities_data": {
    "total_activities": 5,
    "date_range": "2024-07-30 to 2024-08-06",
    "total_duration_minutes": 320,
    "total_distance_km": 25.8,
    "total_calories": 1450,
    "activities": [
      {
        "activity_id": "12345",
        "activity_name": "Running",
        "start_time": "2024-08-06T07:30:00",
        "duration_minutes": 35,
        "distance_km": 5.2,
        "calories": 280,
        "avg_heart_rate": 158
      }
    ]
  }
}
```

#### `get_activity_detail`
Get detailed information about a specific activity.

**Parameters:**
- `activity_id` (required, string): Activity ID from get_activities

**Response:**
```json
{
  "status": "success",
  "activity_detail": {
    "activity_id": "12345",
    "activity_name": "Running",
    "sport": "running",
    "start_time": "2024-08-06T07:30:00",
    "duration": "00:35:24",
    "distance_km": 5.2,
    "avg_pace": "06:48 min/km",
    "calories": 280,
    "avg_heart_rate": 158,
    "max_heart_rate": 175,
    "elevation_gain": 45,
    "training_effect": {
      "aerobic": 3.2,
      "anaerobic": 1.5
    }
  }
}
```

#### `get_steps_detail`
Get detailed step counts throughout the day.

**Parameters:**
- `date` (optional, string): Target date
  - Default: today

**Response:**
```json
{
  "status": "success",
  "steps_detail": {
    "date": "2024-08-06",
    "total_steps": 12500,
    "goal_steps": 10000,
    "hourly_steps": [
      {"hour": "00:00", "steps": 0},
      {"hour": "07:00", "steps": 2500},
      {"hour": "08:00", "steps": 1200}
    ]
  }
}
```

### Analysis & Insights

#### `get_trends_analysis`
Get trends and patterns in your health data over time.

**Parameters:**
- `days` (optional, number): Analysis period in days
  - Default: 30
  - Range: 7-365
- `metrics` (optional, string or array): Specific metrics to analyze
  - Default: all available metrics
  - Options: "steps", "sleep", "heart_rate", "stress", "calories", "distance", "active_minutes"
  - Format: "steps,sleep,heart_rate" or ["steps", "sleep", "heart_rate"]

**Response:**
```json
{
  "status": "success",
  "trends_analysis": {
    "period": "30 days",
    "end_date": "2024-08-06",
    "trends": [
      {
        "metric": "daily_steps",
        "trend_direction": "increasing",
        "change_percent": 12.5,
        "significance": "moderate",
        "current_avg": 11250,
        "previous_avg": 10000,
        "recommendation": "Great progress! Keep up the increased activity level."
      }
    ]
  }
}
```

#### `get_health_insights`
Get AI-powered health pattern insights and recommendations.

**Parameters:**
- `days` (optional, number): Analysis period
  - Default: 30
  - Range: 7-90

**Response:**
```json
{
  "status": "success",
  "health_insights": {
    "analysis_period": "30 days",
    "insights": [
      {
        "title": "Improved Sleep Consistency",
        "description": "Your sleep timing has become more regular over the past 2 weeks",
        "recommendation": "Continue maintaining this sleep schedule for optimal recovery",
        "confidence": 0.85,
        "trend": "positive",
        "metrics_involved": ["sleep_duration", "sleep_score"]
      }
    ]
  }
}
```

#### `get_date_range_data`
Get comprehensive health data across a date range.

**Parameters:**
- `start_date` (required, string): Start date
- `end_date` (required, string): End date  
- `days` (optional, number): Alternative to end_date, number of days from start_date

**Response:**
```json
{
  "status": "success", 
  "date_range_data": {
    "start_date": "2024-08-01",
    "end_date": "2024-08-06", 
    "total_days": 6,
    "summary": {
      "avg_steps": 10500,
      "total_distance_km": 42.5,
      "avg_sleep_hours": 7.2,
      "avg_stress": 32
    },
    "daily_data": [
      {
        "date": "2024-08-01",
        "steps": 9800,
        "sleep_hours": 7.5,
        "stress_avg": 28
      }
    ]
  }
}
```

### Profile & System

#### `get_profile`
Get user profile and device information.

**Response:**
```json
{
  "status": "success",
  "profile": {
    "user_id": "user123",
    "display_name": "John Doe",
    "email": "john@example.com",
    "devices": [
      {
        "device_id": "device123",
        "device_name": "Garmin Forerunner 945",
        "last_sync": "2024-08-06T09:30:00"
      }
    ]
  }
}
```

#### `get_goals`
Get daily goals and progress tracking.

**Response:**
```json
{
  "status": "success",
  "goals": {
    "daily_steps": {
      "goal": 10000,
      "current": 8750,
      "progress_percent": 87.5
    },
    "weekly_active_minutes": {
      "goal": 150,
      "current": 125,
      "progress_percent": 83.3
    }
  }
}
```

#### `get_performance_summary`
Get MCP server performance metrics and health status.

**Response:**
```json
{
  "status": "success",
  "performance": {
    "uptime_seconds": 3600,
    "cache": {
      "hit_rate_percent": 75.5,
      "hits": 302,
      "misses": 98
    },
    "requests": {
      "total": 150,
      "success_rate_percent": 98.7,
      "avg_response_time_ms": 285
    },
    "health_status": "healthy"
  }
}
```

#### `get_health_snapshot`
Get comprehensive health overview combining multiple data sources.

**Parameters:**
- `date` (optional, string): Target date
  - Default: today

**Response:**
```json
{
  "status": "success",
  "health_snapshot": {
    "date": "2024-08-06",
    "activity": {
      "steps": 12500,
      "active_calories": 456,
      "goal_progress": 125
    },
    "sleep": {
      "duration_hours": 7.5,
      "score": 82,
      "quality": "good"
    },
    "wellness": {
      "stress_avg": 28,
      "body_battery": 65,
      "resting_hr": 58
    },
    "summary_text": "Great day! You exceeded your step goal by 25%, got quality sleep (7.5h, score 82/100), and maintained low stress levels."
  }
}
```

## MCP Resources

The server provides 4 resource endpoints for direct data access:

### `garmin://profile`
Returns user profile data in JSON format.

### `garmin://daily/{date}`
Returns daily summary for the specified date.
- `{date}`: Date in YYYY-MM-DD format

### `garmin://goals/current` 
Returns current goal progress and status.

### `garmin://performance/summary`
Returns server performance metrics and health status.

## Error Handling

All API responses follow a consistent error format:

```json
{
  "status": "error",
  "error": {
    "message": "Invalid date format",
    "field": "date",
    "suggestions": [
      "Use YYYY-MM-DD format like '2024-08-06'",
      "Try relative dates like 'today', 'yesterday'"
    ],
    "examples": ["2024-08-06", "today", "3 days ago"],
    "severity": "error"
  }
}
```

### Common Error Types

- **ValidationError**: Invalid parameters or date formats
- **AuthenticationError**: Invalid Garmin Connect credentials  
- **RateLimitError**: Too many API requests
- **ServiceError**: Garmin Connect service unavailable
- **CircuitBreakerError**: Service temporarily disabled due to failures

## Parameter Validation

### Date Parameters
- **Formats**: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY, YYYYMMDD
- **Relative**: "today", "yesterday", "N days ago", "N weeks ago"
- **Validation**: Must not be in the future, must be within last 2 years

### Numeric Parameters  
- **days**: 1-365 (depending on context)
- **limit**: 1-1000
- **Validation**: Must be positive integers within specified ranges

### Metrics Parameter
- **Format**: Comma-separated string or array
- **Options**: steps, sleep, heart_rate, stress, calories, distance, active_minutes
- **Example**: "steps,sleep" or ["steps", "sleep"]

## Rate Limiting & Caching

- **Rate Limit**: 120 requests per minute to Garmin Connect
- **Cache TTL**: 30 minutes for most data, 5 minutes for real-time data
- **Circuit Breaker**: Activates after 3 consecutive failures, recovers after 60 seconds

## Performance Features

- **Intelligent Caching**: Reduces API calls and improves response times
- **AI Optimization**: Formats data for better Claude conversations
- **Performance Monitoring**: Tracks metrics and health status
- **Error Recovery**: Circuit breakers and retry mechanisms
- **Structured Logging**: Comprehensive logging for debugging