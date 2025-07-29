#!/usr/bin/env python3
"""
Database Migration Script
Adds missing columns to existing tables for Cronometer CSV support.
"""

import sqlite3
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('migration')

# Database configuration
DB_PATH = Path(__file__).parent / "data" / "health_data.db"

def migrate_database():
    """Migrate database to add missing columns for Cronometer support."""
    logger.info("Starting database migration...")
    
    if not DB_PATH.exists():
        logger.info("Database doesn't exist yet - no migration needed")
        return True
    
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Check if food_log table exists and get its schema
        cursor.execute("PRAGMA table_info(food_log)")
        food_log_columns = [row[1] for row in cursor.fetchall()]
        logger.info(f"Current food_log columns: {food_log_columns}")
        
        # Check if supplements table exists and get its schema
        cursor.execute("PRAGMA table_info(supplements)")
        supplements_columns = [row[1] for row in cursor.fetchall()]
        logger.info(f"Current supplements columns: {supplements_columns}")
        
        # Migrate food_log table
        if 'food_log' in [table[0] for table in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
            logger.info("Migrating food_log table...")
            
            # Add missing columns to food_log
            missing_food_columns = [
                ('date', 'DATE'),
                ('time', 'TIME'),
                ('category', 'TEXT DEFAULT "Food"'),
                ('fiber_g', 'REAL DEFAULT 0'),
                ('sugar_g', 'REAL DEFAULT 0'),
                ('sodium_mg', 'REAL DEFAULT 0'),
                ('vitamin_c_mg', 'REAL DEFAULT 0'),
                ('iron_mg', 'REAL DEFAULT 0'),
                ('calcium_mg', 'REAL DEFAULT 0'),
                ('source', 'TEXT DEFAULT "cronometer"'),
                ('notes', 'TEXT')
            ]
            
            for column_name, column_type in missing_food_columns:
                if column_name not in food_log_columns:
                    try:
                        cursor.execute(f"ALTER TABLE food_log ADD COLUMN {column_name} {column_type}")
                        logger.info(f"Added column {column_name} to food_log")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e).lower():
                            logger.error(f"Error adding column {column_name}: {e}")
                else:
                    logger.info(f"Column {column_name} already exists in food_log")
            
            # Update existing calories column to REAL if it's INTEGER
            try:
                cursor.execute("PRAGMA table_info(food_log)")
                for row in cursor.fetchall():
                    if row[1] == 'calories' and 'INTEGER' in row[2]:
                        logger.info("Calories column is INTEGER, will work with REAL values")
            except Exception as e:
                logger.warning(f"Could not check calories column type: {e}")
        
        # Migrate supplements table
        if 'supplements' in [table[0] for table in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
            logger.info("Migrating supplements table...")
            
            # Add missing columns to supplements
            missing_supplement_columns = [
                ('date', 'DATE'),
                ('time', 'TIME'),
                ('quantity', 'REAL'),
                ('unit', 'TEXT'),
                ('calories', 'REAL DEFAULT 0'),
                ('source', 'TEXT DEFAULT "cronometer"')
            ]
            
            for column_name, column_type in missing_supplement_columns:
                if column_name not in supplements_columns:
                    try:
                        cursor.execute(f"ALTER TABLE supplements ADD COLUMN {column_name} {column_type}")
                        logger.info(f"Added column {column_name} to supplements")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e).lower():
                            logger.error(f"Error adding column {column_name}: {e}")
                else:
                    logger.info(f"Column {column_name} already exists in supplements")
        
        # Remove old unique constraints and create new ones (this requires recreating tables)
        # For now, we'll just add the columns and let the unique constraints work on inserts
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Database migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = migrate_database()
    if success:
        print("✅ Database migration completed successfully!")
        print("You can now import Cronometer CSV files.")
    else:
        print("❌ Migration failed. Check the logs for details.")
        print("You may need to delete the database file and restart the application.")