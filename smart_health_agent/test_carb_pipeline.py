#!/usr/bin/env python3
"""
Test the complete carbohydrate parsing pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cronometer_parser
import database
import logging

# Configure logging to see debug output
logging.basicConfig(level=logging.INFO)

def test_complete_pipeline():
    """Test the complete carbohydrate parsing pipeline"""
    print("🧪 Testing Complete Carbohydrate Parsing Pipeline")
    print("=" * 50)
    
    # Ensure we have a user for testing
    print("📊 Setting up test user...")
    
    # Create a test user if not exists
    from datetime import datetime
    test_user_id = database.insert_or_update_user(
        garmin_user_id="test_carb_user",
        name="Test Carb User", 
        access_token="test",
        refresh_token="test",
        token_expiry=datetime.now()
    )
    print(f"Using test user ID: {test_user_id}")
    
    # Parse our test CSV
    print("\n📁 Parsing test CSV...")
    test_csv = "test_carbs.csv"
    
    if os.path.exists(test_csv):
        result = cronometer_parser.parse_cronometer_food_entries_csv(test_csv, test_user_id)
        print(f"Parse result: {result}")
        
        # Check what was stored in the database
        print("\n🗄️ Checking database results...")
        with database.get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT food_item_name, carbs_g, calories, protein_g, fats_g, date
                FROM food_log 
                WHERE user_id = ? AND source = 'cronometer'
                ORDER BY timestamp DESC
            ''', (test_user_id,))
            
            rows = cursor.fetchall()
            if rows:
                print("✅ Found stored entries:")
                for row in rows:
                    print(f"  {row[0]}: {row[1]}g carbs, {row[3]}g protein, {row[4]}g fat, {row[2]} cal, {row[5]}")
            else:
                print("❌ No entries found in database!")
                
        print("\n✅ Pipeline test complete!")
        
    else:
        print(f"❌ Test CSV file '{test_csv}' not found!")
        print("Creating test CSV...")
        
        with open(test_csv, 'w') as f:
            f.write("Date,Time,Food Name,Amount,Unit,Calories,Protein (g),Carbohydrates (g),Fat (g)\n")
            f.write("2024-01-15,08:00:00,Debug Test Banana,1,medium,105,1.3,27.0,0.4\n")
            f.write("2024-01-15,12:00:00,Debug Test Rice,100,g,130,2.7,28.0,0.3\n")
        
        print(f"✅ Created test CSV, please re-run the test")

if __name__ == "__main__":
    test_complete_pipeline()