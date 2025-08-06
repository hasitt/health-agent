#!/usr/bin/env python3
"""
Safe script to reset food log data for clean re-import with improved parsing.
"""

import sqlite3
import json
from datetime import datetime
from database import db

def backup_food_data():
    """Create a backup of current food data before deletion."""
    try:
        db.connect()
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Backup individual food entries
        cursor.execute('SELECT * FROM food_log')
        food_entries = [dict(row) for row in cursor.fetchall()]
        
        # Backup daily summaries
        cursor.execute('SELECT * FROM food_log_daily')
        daily_summaries = [dict(row) for row in cursor.fetchall()]
        
        backup_data = {
            'backup_timestamp': datetime.now().isoformat(),
            'food_entries_count': len(food_entries),
            'daily_summaries_count': len(daily_summaries),
            'food_entries': food_entries,
            'daily_summaries': daily_summaries
        }
        
        # Save backup to file
        backup_filename = f"food_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_filename, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        print(f"✅ Backup created: {backup_filename}")
        print(f"   - {len(food_entries)} individual food entries")
        print(f"   - {len(daily_summaries)} daily summaries")
        
        return backup_filename
        
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return None

def delete_food_data():
    """Safely delete only food-related data, preserving all other health data."""
    try:
        db.connect()
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Count records before deletion
        cursor.execute('SELECT COUNT(*) FROM food_log')
        food_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM food_log_daily')
        daily_count = cursor.fetchone()[0]
        
        print(f"Preparing to delete:")
        print(f"   - {food_count} individual food entries")
        print(f"   - {daily_count} daily food summaries")
        
        # Delete food data (but preserve all other health data)
        cursor.execute('DELETE FROM food_log')
        cursor.execute('DELETE FROM food_log_daily')
        
        # Update sync status to reflect the reset
        cursor.execute('''
            UPDATE sync_status 
            SET last_sync_time = NULL, status = 'reset', records_synced = 0 
            WHERE sync_type = 'cronometer'
        ''')
        
        conn.commit()
        
        # Verify deletion
        cursor.execute('SELECT COUNT(*) FROM food_log')
        remaining_food = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM food_log_daily')
        remaining_daily = cursor.fetchone()[0]
        
        # Verify other data is preserved
        cursor.execute('SELECT COUNT(*) FROM garmin_daily_summary')
        garmin_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM subjective_wellbeing')
        mood_count = cursor.fetchone()[0]
        
        print(f"✅ Deletion completed:")
        print(f"   - Food entries remaining: {remaining_food} (should be 0)")
        print(f"   - Daily summaries remaining: {remaining_daily} (should be 0)")
        print(f"   - Garmin data preserved: {garmin_count} entries")
        print(f"   - Mood data preserved: {mood_count} entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Error deleting food data: {e}")
        import traceback
        traceback.print_exc()
        return False

def reset_food_data():
    """Complete process: backup, then delete food data."""
    print("🔄 Starting food data reset process...")
    print("=" * 50)
    
    # Step 1: Create backup
    print("Step 1: Creating backup...")
    backup_file = backup_food_data()
    if not backup_file:
        print("❌ Backup failed - aborting reset process!")
        return False
    
    # Step 2: Confirm deletion
    print("\nStep 2: Ready to delete food data...")
    print("⚠️  This will delete ALL food log entries and daily summaries")
    print("⚠️  Garmin data, mood data, and other health data will be preserved")
    print(f"⚠️  Backup saved as: {backup_file}")
    
    confirm = input("\nType 'DELETE' to confirm deletion: ")
    if confirm != 'DELETE':
        print("❌ Deletion cancelled - no changes made")
        return False
    
    # Step 3: Delete data
    print("\nStep 3: Deleting food data...")
    success = delete_food_data()
    
    if success:
        print("\n🎉 Food data reset completed successfully!")
        print("\nNext steps:")
        print("1. Re-upload your Cronometer CSV file")
        print("2. The improved parser will process it with:")
        print("   - Proper units extraction")
        print("   - Realistic caffeine calculations")
        print("   - Natural key deduplication")
        print("3. Your backup is safely stored if needed")
        return True
    else:
        print("\n💥 Reset process failed!")
        return False

if __name__ == "__main__":
    success = reset_food_data()
    if not success:
        print("\n🚨 IMPORTANT: No changes were made to your database")