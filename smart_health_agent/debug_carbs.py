#!/usr/bin/env python3
"""
Debug script to check carbohydrate parsing issues
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database
import cronometer_parser

def check_carbs_in_database():
    """Check current carbohydrate values in database"""
    print("Checking carbohydrate values in database...")
    
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check a few recent food entries
        cursor.execute('''
            SELECT food_item_name, carbs_g, calories, protein_g, fats_g, date, source 
            FROM food_log 
            ORDER BY timestamp DESC 
            LIMIT 15
        ''')
        
        rows = cursor.fetchall()
        if rows:
            print('Recent food entries with macronutrients:')
            for row in rows:
                print(f'  {row[0]}: {row[1]}g carbs, {row[3]}g protein, {row[4]}g fat, {row[2]} cal, {row[5]}, source: {row[6]}')
        else:
            print('No food entries found in database')
            
        # Check database schema
        cursor.execute('PRAGMA table_info(food_log)')
        columns = cursor.fetchall()
        print('\nFood log table columns:')
        for col in columns:
            if 'carb' in col[1].lower():
                print(f'  {col[1]} ({col[2]})')

def test_carb_column_mapping():
    """Test the carbohydrate column mapping"""
    print("\nTesting carbohydrate column mapping...")
    
    # Test headers
    test_headers = ['Date', 'Time', 'Food Name', 'Amount', 'Unit', 'Calories', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)']
    
    # Find carbs column
    carbs_mapping = cronometer_parser.CRONOMETER_COLUMN_MAPPING['carbs']
    carbs_index = cronometer_parser.find_column_index(test_headers, carbs_mapping)
    
    print(f"Test headers: {test_headers}")
    print(f"Carbs mapping options: {carbs_mapping}")
    print(f"Found carbs at index: {carbs_index}")
    if carbs_index is not None:
        print(f"Column name: '{test_headers[carbs_index]}'")
    
    # Test value conversion
    test_values = ['5.2', '10.0', '0', '', '15.7g', 'N/A']
    print(f"\nTesting value conversion:")
    for val in test_values:
        converted = cronometer_parser.safe_float_convert(val)
        print(f"  '{val}' -> {converted}")

if __name__ == "__main__":
    check_carbs_in_database()
    test_carb_column_mapping()