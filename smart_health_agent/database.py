"""
SQLite Database Management for Smart Health Agent
Handles all database operations for Garmin health data and manual inputs.
"""

import sqlite3
import os
import logging
from datetime import datetime, date
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
from pathlib import Path

# Setup logging
logger = logging.getLogger('database')

# Database configuration
DB_PATH = Path(__file__).parent / "data" / "health_data.db"

def ensure_data_directory():
    """Ensure the data directory exists."""
    DB_PATH.parent.mkdir(exist_ok=True)

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    ensure_data_directory()
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        yield conn
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def create_tables():
    """Create all necessary tables if they don't exist."""
    logger.info("Creating database tables...")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                garmin_user_id TEXT UNIQUE NOT NULL,
                name TEXT,
                garmin_access_token TEXT NOT NULL,
                garmin_refresh_token TEXT NOT NULL,
                garmin_token_expiry_date DATETIME NOT NULL,
                last_garmin_sync_date DATETIME
            )
        """)
        
        # Garmin daily summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS garmin_daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date DATE NOT NULL,
                total_steps INTEGER,
                active_calories INTEGER,
                resting_calories INTEGER,
                distance_km REAL,
                avg_daily_rhr INTEGER,
                avg_daily_stress INTEGER,
                max_daily_stress INTEGER,
                min_daily_stress INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, date)
            )
        """)
        
        # Garmin sleep table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS garmin_sleep (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date DATE NOT NULL,
                sleep_start_time DATETIME NOT NULL,
                sleep_end_time DATETIME NOT NULL,
                total_sleep_minutes INTEGER,
                sleep_score INTEGER,
                deep_sleep_minutes INTEGER,
                rem_sleep_minutes INTEGER,
                light_sleep_minutes INTEGER,
                awake_minutes INTEGER,
                avg_rhr_during_sleep INTEGER,
                avg_stress_during_sleep INTEGER,
                max_stress_during_sleep INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Garmin activities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS garmin_activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                garmin_activity_id TEXT UNIQUE NOT NULL,
                activity_type TEXT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME NOT NULL,
                duration_minutes INTEGER,
                calories_burned INTEGER,
                distance_km REAL,
                avg_hr INTEGER,
                max_hr INTEGER,
                training_effect REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Garmin granular stress data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS garmin_stress_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date DATE NOT NULL,
                timestamp DATETIME NOT NULL,
                stress_level INTEGER NOT NULL,
                body_battery_level INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_stress_user_date 
            ON garmin_stress_details(user_id, date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_stress_user_timestamp 
            ON garmin_stress_details(user_id, timestamp)
        """)
        
        # Food log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS food_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                meal_type TEXT,
                food_item_name TEXT NOT NULL,
                quantity REAL,
                unit TEXT,
                calories INTEGER,
                protein_g REAL,
                carbs_g REAL,
                fats_g REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Supplements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS supplements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                supplement_name TEXT NOT NULL,
                dosage TEXT,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Subjective wellbeing table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subjective_wellbeing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                mood_rating INTEGER,
                energy_rating INTEGER,
                stress_rating INTEGER,
                notes_text TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Genetics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS genetics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                genetic_marker TEXT NOT NULL,
                genotype TEXT NOT NULL,
                implication TEXT,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        conn.commit()
        logger.info("Database tables created successfully")

def insert_or_update_user(garmin_user_id: str, name: str, access_token: str, 
                         refresh_token: str, token_expiry: datetime) -> int:
    """Insert or update user data and return user_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE garmin_user_id = ?", (garmin_user_id,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            # User exists, update it
            user_id = existing_user[0]
            cursor.execute("""
                UPDATE users 
                SET name = ?, garmin_access_token = ?, garmin_refresh_token = ?, garmin_token_expiry_date = ?
                WHERE garmin_user_id = ?
            """, (name, access_token, refresh_token, token_expiry, garmin_user_id))
        else:
            # User doesn't exist, insert new
            cursor.execute("""
                INSERT INTO users 
                (garmin_user_id, name, garmin_access_token, garmin_refresh_token, garmin_token_expiry_date)
                VALUES (?, ?, ?, ?, ?)
            """, (garmin_user_id, name, access_token, refresh_token, token_expiry))
            
            # Get the new user_id
            cursor.execute("SELECT id FROM users WHERE garmin_user_id = ?", (garmin_user_id,))
            user_id = cursor.fetchone()[0]
        
        conn.commit()
        return user_id

def get_user_by_garmin_id(garmin_user_id: str) -> Optional[Dict]:
    """Get user data by Garmin user ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE garmin_user_id = ?", (garmin_user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def update_last_sync_date(user_id: int, sync_date: datetime):
    """Update the last Garmin sync date for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET last_garmin_sync_date = ? WHERE id = ?
        """, (sync_date, user_id))
        conn.commit()

def insert_daily_summary(user_id: int, date: date, summary_data: Dict[str, Any]):
    """Insert or replace daily summary data."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO garmin_daily_summary 
            (user_id, date, total_steps, active_calories, resting_calories, distance_km, 
             avg_daily_rhr, avg_daily_stress, max_daily_stress, min_daily_stress)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, date, 
            summary_data.get('total_steps'),
            summary_data.get('active_calories'),
            summary_data.get('resting_calories'),
            summary_data.get('distance_km'),
            summary_data.get('avg_daily_rhr'),
            summary_data.get('avg_daily_stress'),
            summary_data.get('max_daily_stress'),
            summary_data.get('min_daily_stress')
        ))
        conn.commit()

def insert_sleep_data(user_id: int, sleep_data: Dict[str, Any]):
    """Insert or replace sleep data."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO garmin_sleep 
            (user_id, date, sleep_start_time, sleep_end_time, total_sleep_minutes, 
             sleep_score, deep_sleep_minutes, rem_sleep_minutes, light_sleep_minutes, 
             awake_minutes, avg_rhr_during_sleep, avg_stress_during_sleep, max_stress_during_sleep)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            sleep_data.get('date'),
            sleep_data.get('sleep_start_time'),
            sleep_data.get('sleep_end_time'),
            sleep_data.get('total_sleep_minutes'),
            sleep_data.get('sleep_score'),
            sleep_data.get('deep_sleep_minutes'),
            sleep_data.get('rem_sleep_minutes'),
            sleep_data.get('light_sleep_minutes'),
            sleep_data.get('awake_minutes'),
            sleep_data.get('avg_rhr_during_sleep'),
            sleep_data.get('avg_stress_during_sleep'),
            sleep_data.get('max_stress_during_sleep')
        ))
        conn.commit()

def insert_activity_data(user_id: int, activity_data: Dict[str, Any]):
    """Insert or replace activity data."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO garmin_activities 
            (user_id, garmin_activity_id, activity_type, start_time, end_time, 
             duration_minutes, calories_burned, distance_km, avg_hr, max_hr, training_effect)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            activity_data.get('garmin_activity_id'),
            activity_data.get('activity_type'),
            activity_data.get('start_time'),
            activity_data.get('end_time'),
            activity_data.get('duration_minutes'),
            activity_data.get('calories_burned'),
            activity_data.get('distance_km'),
            activity_data.get('avg_hr'),
            activity_data.get('max_hr'),
            activity_data.get('training_effect')
        ))
        conn.commit()

def insert_stress_details(user_id: int, date_obj, stress_timeline: List[Dict[str, Any]]):
    """Insert granular stress data for a specific date."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # First clear existing data for this user and date
        cursor.execute("""
            DELETE FROM garmin_stress_details 
            WHERE user_id = ? AND date = ?
        """, (user_id, date_obj))
        
        # Insert new stress data points
        for stress_point in stress_timeline:
            cursor.execute("""
                INSERT INTO garmin_stress_details 
                (user_id, date, timestamp, stress_level, body_battery_level)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                date_obj,
                stress_point.get('timestamp'),
                stress_point.get('stress_level'),
                stress_point.get('body_battery_level', 0)  # Default to 0 if not available
            ))
        
        conn.commit()

def get_latest_data(user_id: int, days: int = 1) -> Dict[str, Any]:
    """Get the latest data for a user from all tables."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get latest daily summary
        cursor.execute("""
            SELECT * FROM garmin_daily_summary 
            WHERE user_id = ? 
            ORDER BY date DESC 
            LIMIT ?
        """, (user_id, days))
        daily_summaries = [dict(row) for row in cursor.fetchall()]
        
        # Get latest sleep data
        cursor.execute("""
            SELECT * FROM garmin_sleep 
            WHERE user_id = ? 
            ORDER BY date DESC 
            LIMIT ?
        """, (user_id, days))
        sleep_data = [dict(row) for row in cursor.fetchall()]
        
        # Get latest activities (last 7 days to get recent activities)
        cursor.execute("""
            SELECT * FROM garmin_activities 
            WHERE user_id = ? 
            ORDER BY start_time DESC 
            LIMIT 10
        """, (user_id,))
        activities = [dict(row) for row in cursor.fetchall()]
        
        return {
            'daily_summaries': daily_summaries,
            'sleep_data': sleep_data,
            'activities': activities
        }

def get_sync_status(user_id: int) -> Optional[datetime]:
    """Get the last sync date for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT last_garmin_sync_date FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row and row[0]:
            return datetime.fromisoformat(row[0])
        return None 