"""
Complete Database Management for Smart Health Agent
Handles all database operations with a global Database instance.
"""

import sqlite3
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple
import os
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_PATH = 'health_data.db'

class Database:
    """Database management class with all health data operations."""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        """Initialize database connection."""
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def get_connection(self):
        """Get database connection, creating one if needed."""
        if not self.conn:
            self.connect()
        return self.conn
    
    def migrate_database_schema(self):
        """Migrate existing database schema to support enhanced subjective wellbeing fields."""
        try:
            logger.info("Checking for database schema migrations...")
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check if subjective_wellbeing table exists and get its schema
            cursor.execute("PRAGMA table_info(subjective_wellbeing)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            logger.info(f"Current subjective_wellbeing columns: {existing_columns}")
            
            # Add new enhanced subjective wellbeing columns if they don't exist
            enhanced_columns = [
                ('sleep_quality', 'INTEGER CHECK(sleep_quality >= 1 AND sleep_quality <= 10)'),
                ('focus', 'INTEGER CHECK(focus >= 1 AND focus <= 10)'),
                ('motivation', 'INTEGER CHECK(motivation >= 1 AND motivation <= 10)'),
                ('emotional_state', 'TEXT'),
                ('stress_triggers', 'TEXT'),
                ('coping_strategies', 'TEXT'),
                ('physical_symptoms', 'TEXT'),
                ('daily_events', 'TEXT')
            ]
            
            migrations_applied = 0
            for column_name, column_type in enhanced_columns:
                if column_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE subjective_wellbeing ADD COLUMN {column_name} {column_type}")
                        logger.info(f"Added column {column_name} to subjective_wellbeing")
                        migrations_applied += 1
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e).lower():
                            logger.error(f"Error adding column {column_name}: {e}")
                else:
                    logger.debug(f"Column {column_name} already exists in subjective_wellbeing")
            
            # Check if we need to update mood/energy/stress constraints from 1-5 to 1-10 scale
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='subjective_wellbeing'")
            table_schema = cursor.fetchone()[0]
            
            if "mood <= 5" in table_schema or "energy <= 5" in table_schema or "stress <= 5" in table_schema:
                logger.info("Updating mood/energy/stress constraints from 1-5 to 1-10 scale...")
                
                # Create a new table with updated constraints
                cursor.execute("""
                    CREATE TABLE subjective_wellbeing_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        date DATE NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        mood INTEGER CHECK(mood >= 1 AND mood <= 10),
                        energy INTEGER CHECK(energy >= 1 AND energy <= 10),
                        stress INTEGER CHECK(stress >= 1 AND stress <= 10),
                        sleep_quality INTEGER CHECK(sleep_quality >= 1 AND sleep_quality <= 10),
                        focus INTEGER CHECK(focus >= 1 AND focus <= 10),
                        motivation INTEGER CHECK(motivation >= 1 AND motivation <= 10),
                        emotional_state TEXT,
                        stress_triggers TEXT,
                        coping_strategies TEXT,
                        physical_symptoms TEXT,
                        daily_events TEXT,
                        notes TEXT,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id),
                        UNIQUE(user_id, date)
                    )
                """)
                
                # Copy existing data to new table, handling missing columns
                cursor.execute("PRAGMA table_info(subjective_wellbeing)")
                old_columns = [row[1] for row in cursor.fetchall()]
                
                # Build INSERT statement with only existing columns
                existing_cols = ['id', 'user_id', 'date', 'timestamp', 'mood', 'energy', 'stress', 'notes', 'last_updated']
                available_cols = [col for col in existing_cols if col in old_columns]
                
                cursor.execute(f"""
                    INSERT INTO subjective_wellbeing_new ({', '.join(available_cols)})
                    SELECT {', '.join(available_cols)} FROM subjective_wellbeing
                """)
                
                # Drop old table and rename new table
                cursor.execute("DROP TABLE subjective_wellbeing")
                cursor.execute("ALTER TABLE subjective_wellbeing_new RENAME TO subjective_wellbeing")
                
                logger.info("✅ Updated subjective wellbeing constraints to 1-10 scale")
                migrations_applied += 1
            else:
                logger.info("Mood/energy/stress constraints already support 1-10 scale")
            
            # Check if food_log table needs entry_id column for duplicate prevention
            cursor.execute("PRAGMA table_info(food_log)")
            food_log_columns = [row[1] for row in cursor.fetchall()]
            logger.info(f"Current food_log columns: {food_log_columns}")
            
            if 'entry_id' not in food_log_columns:
                logger.info("Adding entry_id column to food_log table for duplicate prevention...")
                try:
                    cursor.execute("ALTER TABLE food_log ADD COLUMN entry_id TEXT")
                    # Create a unique index on entry_id to prevent duplicates
                    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_food_log_entry_id ON food_log(entry_id)")
                    logger.info("✅ Added entry_id column and unique index to food_log table")
                    migrations_applied += 1
                    
                    # Generate entry_ids for existing records that don't have them
                    cursor.execute("SELECT id, user_id, date, food_item_name, quantity, unit, calories, protein_g, carbs_g, fats_g FROM food_log WHERE entry_id IS NULL")
                    existing_records = cursor.fetchall()
                    
                    import hashlib
                    updated_records = 0
                    for record in existing_records:
                        record_id, user_id, date, item_name, quantity, unit, calories, protein_g, carbs_g, fats_g = record
                        # Generate entry_id using same logic as cronometer_parser
                        natural_key_components = (
                            str(date),
                            str(item_name or ''),
                            str(quantity or 0),
                            str(unit or ''),
                            str(calories or 0),
                            str(protein_g or 0),
                            str(carbs_g or 0),
                            str(fats_g or 0)
                        )
                        entry_id = hashlib.sha256("".join(natural_key_components).encode('utf-8')).hexdigest()
                        
                        try:
                            cursor.execute("UPDATE food_log SET entry_id = ? WHERE id = ?", (entry_id, record_id))
                            updated_records += 1
                        except sqlite3.IntegrityError:
                            # If there's a duplicate entry_id, delete this record as it's a duplicate
                            cursor.execute("DELETE FROM food_log WHERE id = ?", (record_id,))
                            logger.debug(f"Removed duplicate food_log record with id {record_id}")
                    
                    logger.info(f"✅ Updated {updated_records} existing food_log records with entry_ids")
                    
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        logger.error(f"Error adding entry_id column: {e}")
            else:
                logger.info("entry_id column already exists in food_log table")
            
            conn.commit()
            if migrations_applied > 0:
                logger.info(f"✅ Applied {migrations_applied} database schema migrations successfully!")
            else:
                logger.info("✅ Database schema is up to date")
                
        except Exception as e:
            logger.error(f"Database migration error: {e}")
            raise

    def create_tables(self):
        """Create all database tables with proper schema."""
        try:
            logger.info("Creating database tables...")
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_active DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Garmin daily summary table - COMPLETE SCHEMA
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS garmin_daily_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    total_steps INTEGER DEFAULT 0,
                    distance_km REAL DEFAULT 0,
                    calories_burned INTEGER DEFAULT 0,
                    active_calories INTEGER DEFAULT 0,
                    resting_calories INTEGER DEFAULT 0,
                    active_minutes INTEGER DEFAULT 0,
                    floors_climbed INTEGER DEFAULT 0,
                    avg_daily_stress INTEGER DEFAULT 0,
                    max_daily_stress INTEGER DEFAULT 0,
                    min_daily_stress INTEGER DEFAULT 0,
                    resting_heart_rate INTEGER DEFAULT 0,
                    avg_daily_rhr INTEGER DEFAULT 0,
                    avg_heart_rate INTEGER DEFAULT 0,
                    max_heart_rate INTEGER DEFAULT 0,
                    vo2_max REAL DEFAULT 0,
                    body_battery_charged INTEGER DEFAULT 0,
                    body_battery_drained INTEGER DEFAULT 0,
                    body_battery_highest INTEGER DEFAULT 0,
                    body_battery_lowest INTEGER DEFAULT 0,
                    intensity_minutes_moderate INTEGER DEFAULT 0,
                    intensity_minutes_vigorous INTEGER DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(user_id, date)
                )
            """)
            
            # Garmin sleep table - COMPLETE SCHEMA
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS garmin_sleep (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    sleep_start_time DATETIME,
                    sleep_end_time DATETIME,
                    total_sleep_minutes INTEGER DEFAULT 0,
                    sleep_duration_hours REAL DEFAULT 0,
                    deep_sleep_minutes INTEGER DEFAULT 0,
                    light_sleep_minutes INTEGER DEFAULT 0,
                    rem_sleep_minutes INTEGER DEFAULT 0,
                    awake_minutes INTEGER DEFAULT 0,
                    sleep_score INTEGER DEFAULT 0,
                    sleep_quality TEXT,
                    avg_sleep_stress REAL DEFAULT 0,
                    restlessness REAL DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(user_id, date)
                )
            """)
            
            # Garmin activities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS garmin_activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    activity_id TEXT UNIQUE NOT NULL,
                    activity_type TEXT NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    duration_minutes INTEGER DEFAULT 0,
                    distance_km REAL DEFAULT 0,
                    calories_burned INTEGER DEFAULT 0,
                    avg_heart_rate INTEGER DEFAULT 0,
                    max_heart_rate INTEGER DEFAULT 0,
                    elevation_gain_m REAL DEFAULT 0,
                    avg_speed_kmh REAL DEFAULT 0,
                    max_speed_kmh REAL DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Garmin stress details table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS garmin_stress_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    timestamp DATETIME NOT NULL,
                    stress_level INTEGER NOT NULL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(user_id, timestamp)
                )
            """)
            
            # Food log table (individual entries)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS food_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT NOT NULL UNIQUE,
                    user_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    time TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    meal_type TEXT,
                    food_item_name TEXT NOT NULL,
                    category TEXT DEFAULT 'Food',
                    quantity REAL,
                    unit TEXT,
                    calories REAL DEFAULT 0,
                    protein_g REAL DEFAULT 0,
                    carbs_g REAL DEFAULT 0,
                    fats_g REAL DEFAULT 0,
                    fiber_g REAL DEFAULT 0,
                    sugar_g REAL DEFAULT 0,
                    sodium_mg REAL DEFAULT 0,
                    cholesterol_mg REAL DEFAULT 0,
                    saturated_fat_g REAL DEFAULT 0,
                    trans_fat_g REAL DEFAULT 0,
                    caffeine_mg REAL DEFAULT 0,
                    alcohol_g REAL DEFAULT 0,
                    vitamin_c_mg REAL DEFAULT 0,
                    iron_mg REAL DEFAULT 0,
                    calcium_mg REAL DEFAULT 0,
                    potassium_mg REAL DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Food log daily summary table - COMPLETE SCHEMA
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS food_log_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    total_calories REAL DEFAULT 0,
                    protein_g REAL DEFAULT 0,
                    carbohydrates_g REAL DEFAULT 0,
                    fat_g REAL DEFAULT 0,
                    fiber_g REAL DEFAULT 0,
                    sugar_g REAL DEFAULT 0,
                    sodium_mg REAL DEFAULT 0,
                    caffeine_mg REAL DEFAULT 0,
                    alcohol_units REAL DEFAULT 0,
                    vitamin_c_mg REAL DEFAULT 0,
                    iron_mg REAL DEFAULT 0,
                    calcium_mg REAL DEFAULT 0,
                    potassium_mg REAL DEFAULT 0,
                    cholesterol_mg REAL DEFAULT 0,
                    saturated_fat_g REAL DEFAULT 0,
                    trans_fat_g REAL DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(user_id, date)
                )
            """)
            
            # Subjective wellbeing table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subjective_wellbeing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    mood INTEGER CHECK(mood >= 1 AND mood <= 10),
                    energy INTEGER CHECK(energy >= 1 AND energy <= 10),
                    stress INTEGER CHECK(stress >= 1 AND stress <= 10),
                    sleep_quality INTEGER CHECK(sleep_quality >= 1 AND sleep_quality <= 10),
                    focus INTEGER CHECK(focus >= 1 AND focus <= 10),
                    motivation INTEGER CHECK(motivation >= 1 AND motivation <= 10),
                    emotional_state TEXT,
                    stress_triggers TEXT,
                    coping_strategies TEXT,
                    physical_symptoms TEXT,
                    daily_events TEXT,
                    notes TEXT,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(user_id, date)
                )
            """)
            
            # Sync status table - CRITICAL FOR SYNC FUNCTIONS
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sync_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    sync_type TEXT NOT NULL,
                    last_sync_time DATETIME,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    records_synced INTEGER DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(user_id, sync_type)
                )
            """)
            
            # Create default user if none exists
            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO users (username, email) 
                    VALUES ('default_user', 'user@example.com')
                """)
            
            conn.commit()
            logger.info("Database tables created successfully")
            
            # Run database migrations after table creation
            self.migrate_database_schema()
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    # Data existence check functions
    def data_exists_for_date(self, user_id: int, target_date: date) -> bool:
        """Check if Garmin data already exists for a specific date."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM garmin_daily_summary 
                WHERE user_id = ? AND date = ?
            """, (user_id, target_date))
            
            result = cursor.fetchone()
            return result[0] > 0 if result else False
            
        except Exception as e:
            logger.error(f"Error checking if data exists for {target_date}: {e}")
            return False
    
    # Sync status functions
    def get_sync_status(self, user_id: int, sync_type: str) -> Optional[datetime]:
        """Get sync status for a specific sync type."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT last_sync_time FROM sync_status 
                WHERE user_id = ? AND sync_type = ?
            """, (user_id, sync_type))
            
            result = cursor.fetchone()
            if result and result[0]:
                return datetime.fromisoformat(result[0])
            return None
            
        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return None
    
    def update_sync_status(self, user_id: int, sync_type: str, last_sync_time: datetime, 
                          status: str = 'completed', error_message: Optional[str] = None,
                          records_synced: int = 0):
        """Update sync status for a specific sync type."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sync_status 
                (user_id, sync_type, last_sync_time, status, error_message, records_synced, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, sync_type, last_sync_time, status, error_message, records_synced, datetime.now()))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating sync status: {e}")
            raise
    
    def get_latest_garmin_sync_date(self, user_id: int) -> Optional[date]:
        """Get the latest Garmin sync date."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(date) FROM garmin_daily_summary WHERE user_id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            if result and result[0]:
                return datetime.strptime(result[0], '%Y-%m-%d').date()
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest Garmin sync date: {e}")
            return None
    
    def update_last_sync_date(self, user_id: int, sync_date: date):
        """Update Garmin sync status."""
        self.update_sync_status(user_id, 'garmin_daily', datetime.combine(sync_date, datetime.min.time()))
    
    # Garmin data insertion functions
    def insert_daily_summary(self, user_id: int, date: date, data: Dict[str, Any]):
        """Insert or update daily summary data."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Build dynamic INSERT OR REPLACE query
            columns = ['user_id', 'date'] + list(data.keys())
            values = [user_id, date] + list(data.values())
            placeholders = ','.join(['?'] * len(columns))
            
            cursor.execute(f"""
                INSERT OR REPLACE INTO garmin_daily_summary ({','.join(columns)})
                VALUES ({placeholders})
            """, values)
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error inserting daily summary: {e}")
            raise
    
    def insert_sleep_data(self, user_id: int, data: Dict[str, Any]):
        """Insert or update sleep data."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            columns = ['user_id'] + list(data.keys())
            values = [user_id] + list(data.values())
            placeholders = ','.join(['?'] * len(columns))
            
            cursor.execute(f"""
                INSERT OR REPLACE INTO garmin_sleep ({','.join(columns)})
                VALUES ({placeholders})
            """, values)
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error inserting sleep data: {e}")
            raise
    
    def insert_activity_data(self, user_id: int, data: Dict[str, Any]):
        """Insert activity data."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            columns = ['user_id'] + list(data.keys())
            values = [user_id] + list(data.values())
            placeholders = ','.join(['?'] * len(columns))
            
            cursor.execute(f"""
                INSERT OR REPLACE INTO garmin_activities ({','.join(columns)})
                VALUES ({placeholders})
            """, values)
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error inserting activity data: {e}")
            raise
    
    # Data retrieval functions
    def get_garmin_daily_summary(self, user_id: int, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get Garmin daily summary data for date range."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM garmin_daily_summary 
                WHERE user_id = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """, (user_id, start_date, end_date))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting Garmin daily summary: {e}")
            return []
    
    def get_garmin_sleep(self, user_id: int, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get Garmin sleep data for date range."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM garmin_sleep 
                WHERE user_id = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """, (user_id, start_date, end_date))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting Garmin sleep data: {e}")
            return []
    
    def get_garmin_activities(self, user_id: int, start_datetime: str, end_datetime: str) -> List[Dict[str, Any]]:
        """Get Garmin activities for datetime range."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM garmin_activities 
                WHERE user_id = ? AND start_time BETWEEN ? AND ?
                ORDER BY start_time
            """, (user_id, start_datetime, end_datetime))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting Garmin activities: {e}")
            return []
    
    def get_garmin_stress_details(self, user_id: int, start_datetime: str, end_datetime: str) -> List[Dict[str, Any]]:
        """Get detailed stress data for datetime range."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM garmin_stress_details 
                WHERE user_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, (user_id, start_datetime, end_datetime))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting stress details: {e}")
            return []
    
    def get_food_log_daily_summary(self, user_id: int, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get daily food log summary for date range."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM food_log_daily 
                WHERE user_id = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """, (user_id, start_date, end_date))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting food log daily summary: {e}")
            return []
    
    def get_food_log_entries(self, user_id: int, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get individual food log entries for date range."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, food_item_name as item_name, quantity as amount, unit as units, 
                       calories, protein_g, carbs_g as carbohydrates_g, fats_g as fat_g, 
                       caffeine_mg, alcohol_g, timestamp
                FROM food_log 
                WHERE user_id = ? AND date BETWEEN ? AND ?
                ORDER BY date DESC, timestamp DESC
            """, (user_id, start_date, end_date))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting food log entries: {e}")
            return []
    
    def get_subjective_wellbeing(self, user_id: int, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get subjective wellbeing data for date range."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM subjective_wellbeing 
                WHERE user_id = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """, (user_id, start_date, end_date))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting subjective wellbeing data: {e}")
            return []
    
    def upsert_subjective_wellbeing(self, user_id: int, date: str, mood: int, energy: int, stress: int, 
                                   sleep_quality: int = None, focus: int = None, motivation: int = None,
                                   emotional_state: str = None, stress_triggers: str = None, 
                                   coping_strategies: str = None, physical_symptoms: str = None,
                                   daily_events: str = None, notes: str = None):
        """Insert or update subjective wellbeing entry with enhanced fields."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO subjective_wellbeing 
                (user_id, date, mood, energy, stress, sleep_quality, focus, motivation,
                 emotional_state, stress_triggers, coping_strategies, physical_symptoms,
                 daily_events, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, date, mood, energy, stress, sleep_quality, focus, motivation,
                  emotional_state, stress_triggers, coping_strategies, physical_symptoms,
                  daily_events, notes))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error upserting subjective wellbeing: {e}")
            raise
    
    def upsert_food_log_entry(self, user_id: int, entry_id: str, date: str, item_name: str, 
                             amount: float, units: str, calories: float, protein_g: float, 
                             carbohydrates_g: float, fat_g: float, caffeine_mg: float, alcohol_units: float):
        """Insert or update individual food log entry using entry_id as natural key."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Parse date to get time component (set to noon for simplicity)
            timestamp = f"{date} 12:00:00"
            
            cursor.execute("""
                INSERT OR REPLACE INTO food_log 
                (entry_id, user_id, date, time, timestamp, food_item_name, quantity, unit, 
                 calories, protein_g, carbs_g, fats_g, caffeine_mg, alcohol_g)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (entry_id, user_id, date, "12:00", timestamp, item_name, amount, units,
                  calories, protein_g, carbohydrates_g, fat_g, caffeine_mg, alcohol_units))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error upserting food log entry: {e}")
            raise

    def upsert_food_log_daily_entry(self, user_id: int, date: str, nutrition_data: Dict[str, Any]):
        """Insert or update daily food log summary entry."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Build dynamic INSERT OR REPLACE query
            columns = ['user_id', 'date'] + list(nutrition_data.keys())
            values = [user_id, date] + list(nutrition_data.values())
            placeholders = ','.join(['?'] * len(columns))
            
            cursor.execute(f"""
                INSERT OR REPLACE INTO food_log_daily ({','.join(columns)})
                VALUES ({placeholders})
            """, values)
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error upserting food log daily entry: {e}")
            raise
    
    # Garmin-specific upsert methods (for compatibility with sync function)
    def upsert_garmin_daily_summary(self, user_id: int, date: str, steps: int, avg_rhr: int, 
                                   avg_stress: int = 0, max_stress: int = 0, min_stress: int = 0,
                                   active_calories: int = 0, distance_km: float = 0):
        """Insert or update Garmin daily summary data."""
        data = {
            'total_steps': steps,
            'avg_daily_rhr': avg_rhr,
            'avg_daily_stress': avg_stress,
            'max_daily_stress': max_stress,
            'min_daily_stress': min_stress,
            'active_calories': active_calories,
            'distance_km': distance_km
        }
        self.insert_daily_summary(user_id, date, data)
    
    def upsert_garmin_sleep(self, user_id: int, date: str, duration_hours: float, sleep_score: int):
        """Insert or update Garmin sleep data."""
        data = {
            'date': date,
            'sleep_duration_hours': duration_hours,
            'sleep_score': sleep_score
        }
        self.insert_sleep_data(user_id, data)
    
    def upsert_garmin_stress_detail(self, user_id: int, timestamp: str, stress_level: int):
        """Insert or update Garmin stress detail data."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Extract date from timestamp
            date_part = timestamp.split(' ')[0]
            
            cursor.execute("""
                INSERT OR REPLACE INTO garmin_stress_details 
                (user_id, date, timestamp, stress_level)
                VALUES (?, ?, ?, ?)
            """, (user_id, date_part, timestamp, stress_level))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error upserting Garmin stress detail: {e}")
            raise
    
    def upsert_garmin_activity(self, user_id: int, activity_data: Dict[str, Any]):
        """Insert or update Garmin activity data."""
        self.insert_activity_data(user_id, activity_data)

# CRITICAL: Create global database instance for import
db = Database()