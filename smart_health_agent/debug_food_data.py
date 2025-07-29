#!/usr/bin/env python3
"""
Debug Food Data - Check what's actually in the database
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database
import sqlite3
from datetime import datetime, timedelta

def debug_food_data():
    """Debug what food data is actually in the database."""
    print("🔍 Debugging Food Data in Database")
    print("=" * 50)
    
    try:
        with database.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check all users
            print("👥 Users in database:")
            cursor.execute("SELECT id, garmin_user_id, name FROM users")
            users = cursor.fetchall()
            for user in users:
                print(f"   User {user[0]}: {user[1]} ({user[2]})")
            
            if not users:
                print("   No users found!")
                return
            
            # Check food_log table structure
            print(f"\n📋 Food Log Table Structure:")
            cursor.execute("PRAGMA table_info(food_log)")
            columns = cursor.fetchall()
            for col in columns:
                print(f"   {col[1]} ({col[2]})")
            
            # Check all food entries
            print(f"\n🍽️ All Food Entries:")
            cursor.execute("SELECT COUNT(*) FROM food_log")
            total_food = cursor.fetchone()[0]
            print(f"   Total food entries: {total_food}")
            
            if total_food > 0:
                cursor.execute("""
                    SELECT user_id, date, time, food_item_name, calories, source 
                    FROM food_log 
                    ORDER BY date DESC, time DESC 
                    LIMIT 10
                """)
                entries = cursor.fetchall()
                for entry in entries:
                    print(f"   User {entry[0]}: {entry[1]} {entry[2]} - {entry[3]} ({entry[4]} cal) [{entry[5]}]")
            
            # Check supplements
            print(f"\n💊 All Supplement Entries:")
            cursor.execute("SELECT COUNT(*) FROM supplements")
            total_supplements = cursor.fetchone()[0]
            print(f"   Total supplement entries: {total_supplements}")
            
            if total_supplements > 0:
                cursor.execute("""
                    SELECT user_id, date, time, supplement_name, source 
                    FROM supplements 
                    ORDER BY date DESC, time DESC 
                    LIMIT 10
                """)
                entries = cursor.fetchall()
                for entry in entries:
                    print(f"   User {entry[0]}: {entry[1]} {entry[2]} - {entry[3]} [{entry[4]}]")
            
            # Test the food log summary function for each user
            print(f"\n📊 Testing Food Log Summary Function:")
            for user in users:
                user_id = user[0]
                print(f"\n   User {user_id} ({user[1]}):")
                
                try:
                    summary = database.get_food_log_summary(user_id, days=30)  # Check last 30 days
                    print(f"     Daily summaries: {len(summary['daily_summaries'])}")
                    print(f"     Recent food entries: {len(summary['recent_food_entries'])}")
                    print(f"     Recent supplements: {len(summary['recent_supplements'])}")
                    
                    if summary['daily_summaries']:
                        print(f"     Sample daily summary: {summary['daily_summaries'][0]}")
                    
                    if summary['recent_food_entries']:
                        print(f"     Sample food entry: {summary['recent_food_entries'][0]}")
                        
                except Exception as e:
                    print(f"     ERROR getting summary: {e}")
            
            # Check date ranges
            print(f"\n📅 Date Range Analysis:")
            cursor.execute("SELECT MIN(date), MAX(date) FROM food_log WHERE date IS NOT NULL")
            food_date_range = cursor.fetchone()
            if food_date_range[0]:
                print(f"   Food entries date range: {food_date_range[0]} to {food_date_range[1]}")
            else:
                print("   No food entries with dates found")
            
            cursor.execute("SELECT MIN(date), MAX(date) FROM supplements WHERE date IS NOT NULL")
            supp_date_range = cursor.fetchone()
            if supp_date_range[0]:
                print(f"   Supplement entries date range: {supp_date_range[0]} to {supp_date_range[1]}")
            else:
                print("   No supplement entries with dates found")
            
            # Check current date for comparison
            today = datetime.now().date()
            week_ago = today - timedelta(days=7)
            print(f"   Today: {today}")
            print(f"   7 days ago: {week_ago}")
            
    except Exception as e:
        print(f"❌ Error debugging food data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_food_data()