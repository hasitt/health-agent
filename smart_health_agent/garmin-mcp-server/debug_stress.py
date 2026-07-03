#!/usr/bin/env python3
"""
Debug script to examine Garmin stress data API responses
"""

import os
import json
from datetime import date, timedelta
from garminconnect import Garmin

def debug_stress_data():
    """Debug what stress data APIs return."""
    
    email = os.getenv("GARMIN_EMAIL") 
    password = os.getenv("GARMIN_PASSWORD")
    
    if not email or not password:
        print("ERROR: GARMIN_EMAIL and GARMIN_PASSWORD environment variables required")
        return
    
    print(f"Connecting to Garmin with {email[:3]}***")
    
    # Initialize Garmin client
    garmin = Garmin(email, password)
    
    # Try to load saved tokens
    tokens_file = ".garmin_tokens"
    if os.path.exists(tokens_file):
        try:
            with open(tokens_file, "r") as f:
                tokens_string = f.read().strip()
                garmin.garth.loads(tokens_string)
                print("✅ Loaded saved tokens")
        except Exception as e:
            print(f"Token load failed: {e}")
            garmin.login()
            print("✅ Fresh login successful")
    else:
        garmin.login() 
        print("✅ Fresh login successful")
    
    # Test yesterday's date
    test_date = date.today() - timedelta(days=1)
    date_str = test_date.isoformat()
    
    print(f"\n=== TESTING STRESS DATA FOR {date_str} ===")
    
    # Method 1: Direct stress data API
    try:
        print("\n1. Testing garmin.get_stress_data():")
        stress_data = garmin.get_stress_data(date_str)
        print(f"Type: {type(stress_data)}")
        if stress_data:
            print("Keys:", list(stress_data.keys()) if isinstance(stress_data, dict) else "Not a dict")
            print("Data:", json.dumps(stress_data, indent=2, default=str))
        else:
            print("❌ No stress data returned")
    except Exception as e:
        print(f"❌ get_stress_data() failed: {e}")
    
    # Method 2: Daily summary approach
    try:
        print("\n2. Testing garmin.get_daily_summary():")
        daily_stats = garmin.get_daily_summary(date_str)
        print(f"Type: {type(daily_stats)}")
        if daily_stats:
            print("Keys:", list(daily_stats.keys()) if isinstance(daily_stats, dict) else "Not a dict")
            # Look for stress-related keys
            stress_keys = [k for k in daily_stats.keys() if 'stress' in k.lower()]
            print("Stress-related keys:", stress_keys)
            if stress_keys:
                for key in stress_keys:
                    print(f"{key}: {daily_stats[key]}")
        else:
            print("❌ No daily summary returned")
    except Exception as e:
        print(f"❌ get_daily_summary() failed: {e}")
    
    # Method 3: User summary approach
    try:
        print("\n3. Testing garmin.get_user_summary():")
        user_stats = garmin.get_user_summary(date_str)
        print(f"Type: {type(user_stats)}")
        if user_stats:
            print("Keys:", list(user_stats.keys()) if isinstance(user_stats, dict) else "Not a dict")
            # Look for stress-related keys
            stress_keys = [k for k in user_stats.keys() if 'stress' in k.lower()]
            print("Stress-related keys:", stress_keys)
            if stress_keys:
                for key in stress_keys:
                    print(f"{key}: {user_stats[key]}")
        else:
            print("❌ No user summary returned")
    except Exception as e:
        print(f"❌ get_user_summary() failed: {e}")
    
    # Method 4: Try to find all available methods
    print("\n4. Available Garmin client methods with 'stress' in name:")
    stress_methods = [method for method in dir(garmin) if 'stress' in method.lower()]
    print("Stress methods:", stress_methods)

if __name__ == "__main__":
    debug_stress_data()