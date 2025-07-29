#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Working directory:", os.getcwd())
print("Python path:", sys.path)

try:
    import cronometer_parser
    print("✅ cronometer_parser imported successfully")
    
    # Test the column mapping
    test_headers = ['Date', 'Time', 'Food Name', 'Amount', 'Unit', 'Calories', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)']
    carbs_mapping = cronometer_parser.CRONOMETER_COLUMN_MAPPING['carbs']
    carbs_index = cronometer_parser.find_column_index(test_headers, carbs_mapping)
    
    print(f"Headers: {test_headers}")
    print(f"Carbs mapping: {carbs_mapping}")
    print(f"Found carbs at index: {carbs_index}")
    if carbs_index is not None:
        print(f"Column name: '{test_headers[carbs_index]}'")
    
    # Test value conversion
    test_values = ['27.0', '28.0', '0.0', '', '15.7']
    print("Testing value conversion:")
    for val in test_values:
        converted = cronometer_parser.safe_float_convert(val)
        print(f"  '{val}' -> {converted}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()