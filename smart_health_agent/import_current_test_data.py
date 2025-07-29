#!/usr/bin/env python3
"""
Import current test data for the main user session
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database
import cronometer_parser
import tempfile
import csv
from datetime import datetime, timedelta

def create_current_test_csv():
    """Create a test CSV with current dates."""
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    two_days_ago = today - timedelta(days=2)
    
    sample_data = [
        ["Date", "Time", "Food Name", "Amount", "Unit", "Calories", "Protein (g)", "Carbohydrates (g)", "Fat (g)"],
        [today.strftime('%Y-%m-%d'), "08:30:00", "Oatmeal Steel Cut", "1", "cup", "150", "4", "27", "3"],
        [today.strftime('%Y-%m-%d'), "08:35:00", "Banana", "1", "medium", "105", "1.3", "27", "0.4"],
        [today.strftime('%Y-%m-%d'), "12:30:00", "Chicken Breast", "150", "g", "231", "43.5", "0", "5"],
        [today.strftime('%Y-%m-%d'), "09:00:00", "Vitamin D3 Supplement", "1", "capsule", "0", "0", "0", "0"],
        [yesterday.strftime('%Y-%m-%d'), "19:00:00", "Salmon Fillet", "120", "g", "206", "22", "0", "12"],
        [yesterday.strftime('%Y-%m-%d'), "19:05:00", "Quinoa", "0.5", "cup", "111", "4", "20", "1.8"],
        [two_days_ago.strftime('%Y-%m-%d'), "08:00:00", "Greek Yogurt", "1", "cup", "130", "20", "9", "0"],
        [two_days_ago.strftime('%Y-%m-%d'), "13:00:00", "Tuna Salad", "1", "serving", "200", "25", "5", "8"]
    ]
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
    writer = csv.writer(temp_file)
    writer.writerows(sample_data)
    temp_file.close()
    
    return temp_file.name

def import_test_data():
    """Import test data for the main user."""
    print("🧪 Importing Current Test Data")
    print("=" * 40)
    
    try:
        # Get the main user (garmin_user_1)
        with database.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE garmin_user_id = 'garmin_user_1'")
            result = cursor.fetchone()
            
            if result:
                user_id = result[0]
                print(f"✅ Found main user: {user_id}")
            else:
                print("❌ Main user not found, creating one...")
                user_id = database.insert_or_update_user(
                    garmin_user_id="garmin_user_1",
                    name="Main User",
                    access_token="token",
                    refresh_token="refresh",
                    token_expiry=datetime.now() + timedelta(days=365)
                )
                print(f"✅ Created main user: {user_id}")
        
        # Create and import test CSV
        csv_path = create_current_test_csv()
        print(f"📄 Created test CSV: {csv_path}")
        
        # Import the data
        result = cronometer_parser.parse_cronometer_food_entries_csv(csv_path, user_id)
        print(f"📊 Import Results:")
        print(f"   Food entries: {result['food_entries']}")
        print(f"   Supplement entries: {result['supplement_entries']}")
        print(f"   Errors: {result['errors']}")
        
        # Test the food log display
        from smart_health_ollama import format_food_log_display
        display_output = format_food_log_display(user_id)
        print(f"\n📋 Food Log Display Output:")
        print(display_output)
        
        # Clean up
        os.unlink(csv_path)
        print(f"\n✅ Import completed successfully!")
        
    except Exception as e:
        print(f"❌ Error importing test data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import_test_data()