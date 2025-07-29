#!/usr/bin/env python3
"""
Quick CSV Test Tool
Run this to quickly test CSV parsing without the full UI
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cronometer_parser
import database

def test_csv_import(csv_path):
    """Test CSV import with detailed error reporting."""
    print(f"🧪 Testing CSV Import: {csv_path}")
    print("=" * 50)
    
    if not os.path.exists(csv_path):
        print(f"❌ File does not exist: {csv_path}")
        return False
    
    try:
        # Initialize database
        print("📀 Initializing database...")
        database.create_tables()
        
        # Create test user
        print("👤 Creating test user...")
        user_id = database.insert_or_update_user(
            garmin_user_id="csv_test_user",
            name="CSV Test User",
            access_token="test_token",
            refresh_token="test_refresh",
            token_expiry=database.datetime.now()
        )
        print(f"   User ID: {user_id}")
        
        # Validate CSV
        print("🔍 Validating CSV format...")
        validation = cronometer_parser.validate_cronometer_csv(csv_path)
        print(f"   Valid: {validation['is_valid']}")
        
        if not validation['is_valid']:
            print("❌ Validation Issues:")
            for issue in validation['issues']:
                print(f"   - {issue}")
            return False
        
        print("✅ CSV validation passed")
        
        # Parse CSV
        print("📊 Parsing CSV...")
        result = cronometer_parser.parse_cronometer_food_entries_csv(csv_path, user_id)
        
        print(f"📈 Import Results:")
        print(f"   File size: {result.get('file_info', {}).get('size_kb', 'unknown')} KB")
        print(f"   Total rows: {result['total_rows']}")
        print(f"   Food entries: {result['food_entries']}")
        print(f"   Supplement entries: {result['supplement_entries']}")
        print(f"   Errors: {result['errors']}")
        
        if result['error_details']:
            print("⚠️  Error Details:")
            for error in result['error_details']:
                print(f"   - {error}")
        
        if result['food_entries'] > 0 or result['supplement_entries'] > 0:
            print("✅ Import successful!")
            
            # Get food log summary
            food_summary = database.get_food_log_summary(user_id, days=30)
            print(f"\n📋 Food Log Summary:")
            print(f"   Daily summaries: {len(food_summary['daily_summaries'])}")
            print(f"   Recent entries: {len(food_summary['recent_food_entries'])}")
            print(f"   Recent supplements: {len(food_summary['recent_supplements'])}")
            
            return True
        else:
            print("❌ No data was imported")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_csv():
    """Create a sample CSV for testing."""
    sample_data = """Date,Time,Food Name,Amount,Unit,Calories,Protein (g),Carbohydrates (g),Fat (g)
2024-01-15,08:30:00,Oatmeal Steel Cut,1,cup,150,4,27,3
2024-01-15,08:35:00,Banana,1,medium,105,1.3,27,0.4
2024-01-15,12:30:00,Chicken Breast,150,g,231,43.5,0,5
2024-01-15,09:00:00,Vitamin D3 Supplement,1,capsule,0,0,0,0"""
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write(sample_data)
    temp_file.close()
    
    return temp_file.name

if __name__ == "__main__":
    print("🩺 Quick CSV Test Tool")
    print("=" * 30)
    
    if len(sys.argv) == 2:
        csv_file = sys.argv[1]
        print(f"Testing provided CSV: {csv_file}")
    else:
        print("No CSV provided, creating sample CSV for testing...")
        csv_file = create_sample_csv()
        print(f"Created sample CSV: {csv_file}")
    
    success = test_csv_import(csv_file)
    
    if len(sys.argv) != 2:  # Clean up sample file
        try:
            os.unlink(csv_file)
            print(f"🧹 Cleaned up sample file")
        except:
            pass
    
    print(f"\n{'✅ Test completed successfully!' if success else '❌ Test failed!'}")
    
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("1. Check that your CSV has 'Date' and 'Food Name' columns")
        print("2. Ensure the file is a valid CSV format")
        print("3. Try exporting a fresh CSV from Cronometer")
        print("4. Run: python csv_debug_tool.py <your_csv_file> for detailed analysis")