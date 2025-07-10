"""
Sample test script for Cronometer CSV parsing functionality.
This script creates a sample CSV file and tests the parsing logic.
"""

import csv
import tempfile
import os
from datetime import datetime, date, time
import cronometer_parser
import database

def create_sample_cronometer_csv():
    """Create a sample Cronometer CSV file for testing."""
    
    # Sample data that mimics Cronometer export format
    sample_data = [
        ["Date", "Time", "Food Name", "Amount", "Unit", "Calories", "Protein (g)", "Carbohydrates (g)", "Fat (g)", "Fiber (g)", "Sugar (g)", "Sodium (mg)", "Vitamin C (mg)", "Iron (mg)", "Calcium (mg)"],
        ["2024-01-15", "08:30:00", "Oatmeal, Steel Cut", "1", "cup", "150", "4", "27", "3", "4", "1", "2", "0", "1.5", "20"],
        ["2024-01-15", "08:35:00", "Banana", "1", "medium", "105", "1.3", "27", "0.4", "3.1", "14.4", "1", "10.3", "0.3", "6"],
        ["2024-01-15", "12:30:00", "Chicken Breast", "150", "g", "231", "43.5", "0", "5", "0", "0", "104", "0", "0.9", "15"],
        ["2024-01-15", "12:35:00", "Brown Rice", "0.5", "cup", "108", "2.5", "22", "0.9", "1.8", "0.4", "5", "0", "0.4", "10"],
        ["2024-01-15", "09:00:00", "Vitamin D3 Supplement", "1", "capsule", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        ["2024-01-15", "19:00:00", "Salmon Fillet", "120", "g", "206", "22", "0", "12", "0", "0", "59", "0", "0.3", "9"],
        ["2024-01-15", "19:05:00", "Quinoa", "0.5", "cup", "111", "4", "20", "1.8", "2.5", "0.9", "7", "0", "1.5", "17"]
    ]
    
    # Create temporary CSV file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
    writer = csv.writer(temp_file)
    writer.writerows(sample_data)
    temp_file.close()
    
    return temp_file.name

def test_cronometer_parsing():
    """Test the Cronometer CSV parsing functionality."""
    
    print("🧪 Testing Cronometer CSV Parsing Functionality")
    print("=" * 50)
    
    # Initialize database
    database.create_tables()
    print("✅ Database tables created")
    
    # Create a test user
    user_id = database.insert_or_update_user(
        garmin_user_id="test_user_cronometer",
        name="Test User for Cronometer",
        access_token="test_token",
        refresh_token="test_refresh",
        token_expiry=datetime.now()
    )
    print(f"✅ Test user created with ID: {user_id}")
    
    # Create sample CSV
    csv_file_path = create_sample_cronometer_csv()
    print(f"✅ Sample CSV created: {csv_file_path}")
    
    try:
        # Validate CSV
        validation_result = cronometer_parser.validate_cronometer_csv(csv_file_path)
        print(f"✅ CSV validation result: {validation_result['is_valid']}")
        if not validation_result['is_valid']:
            print(f"❌ Validation issues: {validation_result['issues']}")
            return
        
        # Parse CSV
        import_summary = cronometer_parser.parse_cronometer_food_entries_csv(csv_file_path, user_id)
        print(f"\n📊 Import Summary:")
        print(f"   Total rows: {import_summary['total_rows']}")
        print(f"   Food entries: {import_summary['food_entries']}")
        print(f"   Supplement entries: {import_summary['supplement_entries']}")
        print(f"   Errors: {import_summary['errors']}")
        
        if import_summary['error_details']:
            print(f"   Error details: {import_summary['error_details']}")
        
        # Get food log summary
        food_summary = database.get_food_log_summary(user_id, days=7)
        print(f"\n📈 Food Log Summary:")
        print(f"   Daily summaries: {len(food_summary['daily_summaries'])}")
        print(f"   Recent food entries: {len(food_summary['recent_food_entries'])}")
        print(f"   Recent supplements: {len(food_summary['recent_supplements'])}")
        
        # Display some sample data
        if food_summary['daily_summaries']:
            day = food_summary['daily_summaries'][0]
            print(f"\n📅 Sample day ({day['date']}):")
            print(f"   Entries: {day['food_entries']}")
            print(f"   Calories: {day['total_calories']:.0f}")
            print(f"   Protein: {day['total_protein']:.1f}g")
            print(f"   Carbs: {day['total_carbs']:.1f}g")
            print(f"   Fat: {day['total_fats']:.1f}g")
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if os.path.exists(csv_file_path):
            os.unlink(csv_file_path)
        print(f"🧹 Cleaned up temporary file")

if __name__ == "__main__":
    test_cronometer_parsing()