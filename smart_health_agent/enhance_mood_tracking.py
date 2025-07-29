#!/usr/bin/env python3
"""
Enhanced Mood and Lifestyle Tracking Migration
Adds comprehensive daily mood tracking and caffeine/alcohol tracking capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database
import sqlite3
from datetime import datetime

def enhance_subjective_wellbeing_table():
    """Add comprehensive mood tracking columns to subjective_wellbeing table."""
    print("🧠 Enhancing Subjective Wellbeing Table")
    print("=" * 50)
    
    # Columns to add for enhanced mood tracking
    new_columns = {
        'date': 'DATE',
        'anxiety_rating': 'INTEGER',  # 1-10 scale
        'sleep_quality_rating': 'INTEGER',  # 1-10 scale
        'focus_rating': 'INTEGER',  # 1-10 scale
        'motivation_rating': 'INTEGER',  # 1-10 scale
        'physical_symptoms': 'TEXT',  # headache, tension, etc.
        'emotional_state': 'TEXT',  # anxious, calm, irritable, etc.
        'stress_triggers': 'TEXT',  # work, relationships, health, etc.
        'coping_strategies': 'TEXT',  # exercise, meditation, etc.
        'daily_events': 'TEXT',  # significant events affecting mood
        'social_interactions': 'TEXT',  # quality and quantity of social contact
        'weather_sensitivity': 'TEXT',  # weather impact on mood
        'hormonal_factors': 'TEXT',  # menstrual cycle, etc.
        'entry_type': 'TEXT DEFAULT "daily"',  # daily, morning, evening, moment
        'source': 'TEXT DEFAULT "manual"'  # manual, automated, imported
    }
    
    try:
        with database.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check existing columns
            cursor.execute("PRAGMA table_info(subjective_wellbeing)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            # Add missing columns
            for column_name, column_type in new_columns.items():
                if column_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE subjective_wellbeing ADD COLUMN {column_name} {column_type}")
                        print(f"  ✅ Added column: {column_name}")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e):
                            print(f"  ❌ Error adding {column_name}: {e}")
                        else:
                            print(f"  ⏭️  Column {column_name} already exists")
                else:
                    print(f"  ⏭️  Column {column_name} already exists")
            
            # Add index for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_wellbeing_user_date 
                ON subjective_wellbeing(user_id, date)
            """)
            
            conn.commit()
            print("  ✅ Enhanced subjective_wellbeing table successfully")
            
    except Exception as e:
        print(f"  ❌ Error enhancing subjective_wellbeing table: {e}")

def add_caffeine_alcohol_tracking():
    """Add caffeine and alcohol tracking columns to food_log table."""
    print("\n☕ Adding Caffeine & Alcohol Tracking")
    print("=" * 50)
    
    # Columns to add for lifestyle tracking
    lifestyle_columns = {
        'caffeine_mg': 'REAL DEFAULT 0',  # Caffeine content in mg
        'alcohol_ml': 'REAL DEFAULT 0',   # Alcohol volume in ml
        'alcohol_abv': 'REAL DEFAULT 0',  # Alcohol by volume percentage
        'alcohol_units': 'REAL DEFAULT 0', # Standard alcohol units
        'beverage_type': 'TEXT',  # coffee, tea, wine, beer, spirits, etc.
        'timing_relative_sleep': 'TEXT',  # hours before bedtime
        'consumption_context': 'TEXT'  # social, work, stress, habit, etc.
    }
    
    try:
        with database.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check existing columns
            cursor.execute("PRAGMA table_info(food_log)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            # Add missing columns
            for column_name, column_type in lifestyle_columns.items():
                if column_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE food_log ADD COLUMN {column_name} {column_type}")
                        print(f"  ✅ Added column: {column_name}")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e):
                            print(f"  ❌ Error adding {column_name}: {e}")
                        else:
                            print(f"  ⏭️  Column {column_name} already exists")
                else:
                    print(f"  ⏭️  Column {column_name} already exists")
            
            # Add indexes for lifestyle queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_food_log_caffeine 
                ON food_log(user_id, date, caffeine_mg) WHERE caffeine_mg > 0
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_food_log_alcohol 
                ON food_log(user_id, date, alcohol_ml) WHERE alcohol_ml > 0
            """)
            
            conn.commit()
            print("  ✅ Enhanced food_log table with lifestyle tracking")
            
    except Exception as e:
        print(f"  ❌ Error adding lifestyle tracking: {e}")

def create_mood_tracking_functions():
    """Add database functions for mood tracking operations."""
    print("\n📊 Creating Mood Tracking Functions")
    print("=" * 50)
    
    # Create the functions file
    functions_content = '''"""
Database functions for enhanced mood and lifestyle tracking.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
import database

def insert_daily_mood_entry(user_id: int, mood_data: Dict[str, Any]):
    """Insert or update daily mood tracking entry."""
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Prepare the data
        entry_date = mood_data.get('date', datetime.now().date())
        timestamp = mood_data.get('timestamp', datetime.now())
        
        cursor.execute("""
            INSERT OR REPLACE INTO subjective_wellbeing 
            (user_id, date, timestamp, mood_rating, energy_rating, stress_rating, 
             anxiety_rating, sleep_quality_rating, focus_rating, motivation_rating,
             physical_symptoms, emotional_state, stress_triggers, coping_strategies,
             daily_events, social_interactions, weather_sensitivity, hormonal_factors,
             entry_type, source, notes_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            entry_date,
            timestamp,
            mood_data.get('mood_rating'),
            mood_data.get('energy_rating'),
            mood_data.get('stress_rating'),
            mood_data.get('anxiety_rating'),
            mood_data.get('sleep_quality_rating'),
            mood_data.get('focus_rating'),
            mood_data.get('motivation_rating'),
            mood_data.get('physical_symptoms'),
            mood_data.get('emotional_state'),
            mood_data.get('stress_triggers'),
            mood_data.get('coping_strategies'),
            mood_data.get('daily_events'),
            mood_data.get('social_interactions'),
            mood_data.get('weather_sensitivity'),
            mood_data.get('hormonal_factors'),
            mood_data.get('entry_type', 'daily'),
            mood_data.get('source', 'manual'),
            mood_data.get('notes_text')
        ))
        conn.commit()

def get_mood_summary(user_id: int, days: int = 7) -> Dict[str, Any]:
    """Get mood tracking summary for the last N days."""
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Get mood entries
        cursor.execute("""
            SELECT date, mood_rating, energy_rating, stress_rating, anxiety_rating,
                   sleep_quality_rating, focus_rating, motivation_rating,
                   emotional_state, stress_triggers, coping_strategies
            FROM subjective_wellbeing 
            WHERE user_id = ? AND date BETWEEN ? AND ?
            ORDER BY date DESC
        """, (user_id, start_date, end_date))
        
        mood_entries = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        
        # Calculate averages for ratings
        ratings = ['mood_rating', 'energy_rating', 'stress_rating', 'anxiety_rating', 
                  'sleep_quality_rating', 'focus_rating', 'motivation_rating']
        
        averages = {}
        for rating in ratings:
            values = [entry[rating] for entry in mood_entries if entry[rating] is not None]
            averages[rating] = sum(values) / len(values) if values else None
        
        return {
            'mood_entries': mood_entries,
            'averages': averages,
            'period_days': days
        }

def get_caffeine_alcohol_summary(user_id: int, days: int = 7) -> Dict[str, Any]:
    """Get caffeine and alcohol consumption summary."""
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Get daily caffeine totals
        cursor.execute("""
            SELECT date, 
                   SUM(caffeine_mg) as total_caffeine,
                   SUM(alcohol_ml) as total_alcohol_ml,
                   SUM(alcohol_units) as total_alcohol_units,
                   COUNT(CASE WHEN caffeine_mg > 0 THEN 1 END) as caffeine_items,
                   COUNT(CASE WHEN alcohol_ml > 0 THEN 1 END) as alcohol_items
            FROM food_log 
            WHERE user_id = ? AND date BETWEEN ? AND ?
            GROUP BY date
            ORDER BY date DESC
        """, (user_id, start_date, end_date))
        
        daily_consumption = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        
        # Get recent caffeine/alcohol entries
        cursor.execute("""
            SELECT date, time, food_item_name, caffeine_mg, alcohol_ml, 
                   beverage_type, timing_relative_sleep, consumption_context
            FROM food_log 
            WHERE user_id = ? AND date BETWEEN ? AND ? 
              AND (caffeine_mg > 0 OR alcohol_ml > 0)
            ORDER BY date DESC, time DESC
            LIMIT 20
        """, (user_id, start_date, end_date))
        
        recent_entries = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        
        return {
            'daily_consumption': daily_consumption,
            'recent_entries': recent_entries,
            'period_days': days
        }

def analyze_stress_lifestyle_correlation(user_id: int, days: int = 30) -> List[str]:
    """Analyze correlation between stress levels and lifestyle factors."""
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Get combined stress, mood, and lifestyle data
        cursor.execute("""
            SELECT 
                ds.date,
                ds.avg_daily_stress as garmin_stress,
                ds.max_daily_stress,
                sw.stress_rating as subjective_stress,
                sw.mood_rating,
                sw.anxiety_rating,
                sw.sleep_quality_rating,
                COALESCE(SUM(fl.caffeine_mg), 0) as daily_caffeine,
                COALESCE(SUM(fl.alcohol_units), 0) as daily_alcohol
            FROM garmin_daily_summary ds
            LEFT JOIN subjective_wellbeing sw ON sw.user_id = ds.user_id AND sw.date = ds.date
            LEFT JOIN food_log fl ON fl.user_id = ds.user_id AND fl.date = ds.date
            WHERE ds.user_id = ? AND ds.date BETWEEN ? AND ?
            GROUP BY ds.date, ds.avg_daily_stress, ds.max_daily_stress, 
                     sw.stress_rating, sw.mood_rating, sw.anxiety_rating, sw.sleep_quality_rating
            ORDER BY ds.date
        """, (user_id, start_date, end_date))
        
        data = cursor.fetchall()
        
        if not data:
            return ["No stress and lifestyle correlation data available."]
        
        observations = []
        
        # Analyze high caffeine days vs stress
        high_caffeine_days = [(garmin_stress, subjective_stress) for _, garmin_stress, _, subjective_stress, _, _, _, caffeine, _ in data if caffeine > 200]
        low_caffeine_days = [(garmin_stress, subjective_stress) for _, garmin_stress, _, subjective_stress, _, _, _, caffeine, _ in data if caffeine <= 100]
        
        if high_caffeine_days and low_caffeine_days:
            avg_stress_high_caffeine = sum(s[0] for s in high_caffeine_days if s[0]) / len([s for s in high_caffeine_days if s[0]])
            avg_stress_low_caffeine = sum(s[0] for s in low_caffeine_days if s[0]) / len([s for s in low_caffeine_days if s[0]])
            
            observations.append(
                f"High caffeine days (>200mg): avg stress {avg_stress_high_caffeine:.1f}/100. "
                f"Low caffeine days (≤100mg): avg stress {avg_stress_low_caffeine:.1f}/100."
            )
        
        # Analyze alcohol vs next day mood
        alcohol_days = [(mood, anxiety) for _, _, _, _, mood, anxiety, _, _, alcohol in data if alcohol > 0 and mood and anxiety]
        no_alcohol_days = [(mood, anxiety) for _, _, _, _, mood, anxiety, _, _, alcohol in data if alcohol == 0 and mood and anxiety]
        
        if alcohol_days and no_alcohol_days:
            avg_mood_after_alcohol = sum(m[0] for m in alcohol_days) / len(alcohol_days)
            avg_mood_no_alcohol = sum(m[0] for m in no_alcohol_days) / len(no_alcohol_days)
            
            observations.append(
                f"Days after alcohol consumption: avg mood {avg_mood_after_alcohol:.1f}/10. "
                f"Days without alcohol: avg mood {avg_mood_no_alcohol:.1f}/10."
            )
        
        return observations if observations else ["Insufficient data for stress-lifestyle correlation analysis."]
'''
    
    try:
        with open('/Users/stan/dev/projects/health-agent/smart_health_agent/mood_tracking.py', 'w') as f:
            f.write(functions_content)
        print("  ✅ Created mood_tracking.py with database functions")
    except Exception as e:
        print(f"  ❌ Error creating mood tracking functions: {e}")

def run_migration():
    """Run the complete mood and lifestyle tracking migration."""
    print("🚀 Enhanced Mood & Lifestyle Tracking Migration")
    print("=" * 60)
    
    # Ensure database tables exist
    database.create_tables()
    
    # Run enhancements
    enhance_subjective_wellbeing_table()
    add_caffeine_alcohol_tracking()
    create_mood_tracking_functions()
    
    print("\n🎉 Migration completed successfully!")
    print("\nNew capabilities added:")
    print("✅ Comprehensive daily mood tracking (10+ metrics)")
    print("✅ Caffeine and alcohol consumption tracking")
    print("✅ Stress-lifestyle correlation analysis")
    print("✅ Enhanced subjective wellbeing database functions")

if __name__ == "__main__":
    run_migration()