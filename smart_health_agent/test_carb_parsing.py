#!/usr/bin/env python3
"""
Test carbohydrate parsing specifically
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cronometer_parser
import database
import csv

def test_carb_parsing():
    """Test carbohydrate parsing with a known CSV"""
    print("🧪 Testing Carbohydrate Parsing")
    print("=" * 40)
    
    # First, test with our test CSV
    test_csv = "test_carbs.csv"
    
    print(f"📁 Reading test CSV: {test_csv}")
    with open(test_csv, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        print(f"Headers: {headers}")
        
        # Find carbs column
        carbs_mapping = cronometer_parser.CRONOMETER_COLUMN_MAPPING['carbs']
        carbs_index = cronometer_parser.find_column_index(headers, carbs_mapping)
        print(f"Carbs mapping: {carbs_mapping}")
        print(f"Found carbs at index: {carbs_index}")
        if carbs_index is not None:
            print(f"Column name: '{headers[carbs_index]}'")
        
        print("\n📊 Testing rows:")
        for i, row in enumerate(reader):
            if carbs_index is not None:
                carb_raw = row[carbs_index]
                carb_converted = cronometer_parser.safe_float_convert(carb_raw)
                print(f"  Row {i+1}: '{carb_raw}' -> {carb_converted}g")
            else:
                print(f"  Row {i+1}: No carbs column found")
    
    print("\n🗄️ Checking current database:")
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get users
        cursor.execute("SELECT id FROM users LIMIT 1")
        user_row = cursor.fetchone()
        if user_row:
            user_id = user_row[0]
            print(f"Using user ID: {user_id}")
            
            # Check recent entries
            cursor.execute('''
                SELECT food_item_name, carbs_g, calories, protein_g, fats_g
                FROM food_log 
                WHERE user_id = ?
                ORDER BY timestamp DESC 
                LIMIT 10
            ''', (user_id,))
            
            rows = cursor.fetchall()
            if rows:
                print("Recent food entries:")
                for row in rows:
                    print(f"  {row[0]}: {row[1]}g carbs, {row[3]}g protein, {row[4]}g fat, {row[2]} cal")
            else:
                print("No food entries found")
        else:
            print("No users found in database")

if __name__ == "__main__":
    test_carb_parsing()