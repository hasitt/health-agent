#!/usr/bin/env python3
"""
CSV Debug Tool for Cronometer Import Issues
This tool helps diagnose common problems with CSV imports.
"""

import csv
import sys
import os
from pathlib import Path

def analyze_csv_file(csv_path):
    """Analyze a CSV file to identify potential import issues."""
    print(f"🔍 Analyzing CSV file: {csv_path}")
    print("=" * 60)
    
    if not os.path.exists(csv_path):
        print(f"❌ File does not exist: {csv_path}")
        return False
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            # Read first few lines
            first_lines = []
            for i, line in enumerate(file):
                first_lines.append(line.strip())
                if i >= 5:  # Read first 6 lines
                    break
        
        print("📄 First few lines of file:")
        for i, line in enumerate(first_lines):
            print(f"  Line {i+1}: {line[:100]}{'...' if len(line) > 100 else ''}")
        
        # Detect delimiter
        with open(csv_path, 'r', encoding='utf-8') as file:
            sample = file.read(1024)
            file.seek(0)
            
            sniffer = csv.Sniffer()
            try:
                delimiter = sniffer.sniff(sample).delimiter
                print(f"📊 Detected delimiter: '{delimiter}'")
            except:
                delimiter = ','
                print(f"📊 Using default delimiter: ','")
        
        # Read headers and sample data
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimiter)
            headers = next(reader)
            
            print(f"\n📋 Headers ({len(headers)} columns):")
            for i, header in enumerate(headers):
                print(f"  {i}: '{header}'")
            
            # Check for required columns
            required_patterns = {
                'date': ['date', 'day', 'timestamp'],
                'food_name': ['food', 'name', 'item', 'product'],
                'amount': ['amount', 'quantity', 'serving'],
                'calories': ['calorie', 'energy', 'kcal']
            }
            
            print(f"\n🔍 Column Analysis:")
            for req_type, patterns in required_patterns.items():
                found_columns = []
                for i, header in enumerate(headers):
                    header_lower = header.lower()
                    if any(pattern in header_lower for pattern in patterns):
                        found_columns.append(f"{i}:'{header}'")
                
                if found_columns:
                    print(f"  {req_type.upper()}: Found {found_columns}")
                else:
                    print(f"  {req_type.upper()}: ❌ NOT FOUND")
            
            # Read sample data rows
            print(f"\n📊 Sample Data (first 3 rows):")
            for i, row in enumerate(reader):
                if i >= 3:
                    break
                print(f"  Row {i+1}: {len(row)} columns")
                for j, cell in enumerate(row[:5]):  # Show first 5 columns
                    print(f"    Col {j}: '{cell}'")
                if len(row) > 5:
                    print(f"    ... and {len(row)-5} more columns")
        
        print(f"\n✅ CSV analysis complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")
        return False

def check_cronometer_format(csv_path):
    """Check if CSV matches expected Cronometer format."""
    print(f"\n🏥 Checking Cronometer format compatibility...")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)
            
            # Known Cronometer export patterns
            cronometer_indicators = [
                'date', 'time', 'food name', 'amount', 'unit', 'calories',
                'protein', 'carbohydrate', 'fat', 'fiber'
            ]
            
            headers_lower = [h.lower() for h in headers]
            matches = 0
            
            for indicator in cronometer_indicators:
                if any(indicator in header for header in headers_lower):
                    matches += 1
            
            compatibility = (matches / len(cronometer_indicators)) * 100
            
            print(f"📈 Cronometer compatibility: {compatibility:.1f}%")
            print(f"   Matched {matches}/{len(cronometer_indicators)} expected patterns")
            
            if compatibility >= 60:
                print("✅ Likely a valid Cronometer export")
            elif compatibility >= 30:
                print("⚠️  Possibly a Cronometer export, but may need manual review")
            else:
                print("❌ Does not appear to be a standard Cronometer export")
                print("   Expected columns like: Date, Time, Food Name, Amount, Calories, etc.")
            
            return compatibility >= 30
            
    except Exception as e:
        print(f"❌ Error checking format: {e}")
        return False

def provide_solutions():
    """Provide common solutions for CSV import issues."""
    print(f"\n💡 Common Solutions:")
    print("=" * 40)
    print("1. ✅ Export Format:")
    print("   - In Cronometer, go to Settings > Data Export")
    print("   - Select 'Food & Recipe Entries'")
    print("   - Choose date range")
    print("   - Download as CSV")
    print()
    print("2. 📁 File Issues:")
    print("   - Ensure file has .csv extension")
    print("   - Check file isn't corrupted")
    print("   - Try re-downloading from Cronometer")
    print()
    print("3. 🔧 Format Issues:")
    print("   - Ensure CSV has headers in first row")
    print("   - Verify columns include: Date, Food Name, Amount")
    print("   - Check for special characters in food names")
    print()
    print("4. 🗄️ Database Issues:")
    print("   - Ensure Garmin data is synced first")
    print("   - Check database permissions")
    print("   - Try restarting the application")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python csv_debug_tool.py <path_to_csv_file>")
        print("Example: python csv_debug_tool.py ~/Downloads/cronometer_export.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    print("🩺 Cronometer CSV Debug Tool")
    print("=" * 60)
    
    # Analyze the file
    if analyze_csv_file(csv_file):
        check_cronometer_format(csv_file)
    
    provide_solutions()
    
    print("\n📝 Next Steps:")
    print("1. If the CSV looks correct, try the import again")
    print("2. If there are format issues, re-export from Cronometer")
    print("3. If problems persist, check the application logs")
    print("4. Contact support with this debug output")