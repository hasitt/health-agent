#!/usr/bin/env python3
"""
Debug sleep timestamp parsing to fix timezone issues
"""

import os
import json
from datetime import datetime, date, timedelta
from garminconnect import Garmin

def debug_sleep_timestamps():
    """Debug sleep timestamp parsing and timezone conversion."""
    
    email = os.getenv("GARMIN_EMAIL") or os.getenv("GARMIN_USERNAME")
    password = os.getenv("GARMIN_PASSWORD")
    
    print(f"🔍 Debugging Sleep Timestamps")
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
    
    # Test yesterday's date for sleep data
    test_date = date.today() - timedelta(days=1)
    date_str = test_date.isoformat()
    
    print(f"\n=== TESTING SLEEP TIMESTAMPS FOR {date_str} ===")
    
    try:
        sleep_data = garmin.get_sleep_data(date_str)
        print(f"Sleep data type: {type(sleep_data)}")
        print(f"Sleep data keys: {list(sleep_data.keys()) if isinstance(sleep_data, dict) else 'Not a dict'}")
        
        if sleep_data and 'dailySleepDTO' in sleep_data:
            daily_sleep = sleep_data['dailySleepDTO']
            print(f"\nDaily sleep keys: {list(daily_sleep.keys())}")
            
            # Extract timestamp fields
            sleep_start = daily_sleep.get('sleepStartTimestampLocal')
            sleep_end = daily_sleep.get('sleepEndTimestampLocal')
            sleep_start_gmt = daily_sleep.get('sleepStartTimestampGMT')
            sleep_end_gmt = daily_sleep.get('sleepEndTimestampGMT')
            
            print(f"\n📅 RAW TIMESTAMP DATA:")
            print(f"Sleep Start Local: {sleep_start}")
            print(f"Sleep End Local: {sleep_end}")  
            print(f"Sleep Start GMT: {sleep_start_gmt}")
            print(f"Sleep End GMT: {sleep_end_gmt}")
            
            # Convert timestamps to readable format
            if sleep_start:
                # Test different conversions
                print(f"\n🕒 TIMESTAMP CONVERSIONS:")
                print(f"Sleep Start Raw: {sleep_start}")
                
                # Method 1: Direct millisecond conversion
                start_dt1 = datetime.fromtimestamp(sleep_start / 1000)
                print(f"Method 1 (ms/1000): {start_dt1} ({start_dt1.strftime('%I:%M %p')})")
                
                # Method 2: Direct second conversion (this will likely fail)
                try:
                    start_dt2 = datetime.fromtimestamp(sleep_start)
                    print(f"Method 2 (direct): {start_dt2} ({start_dt2.strftime('%I:%M %p')})")
                except (ValueError, OSError) as e:
                    print(f"Method 2 failed: {e}")
                
                # Method 3: UTC conversion
                start_dt3 = datetime.utcfromtimestamp(sleep_start / 1000)
                print(f"Method 3 (UTC): {start_dt3} ({start_dt3.strftime('%I:%M %p')})")
                
            if sleep_end:
                print(f"\nSleep End Raw: {sleep_end}")
                
                end_dt1 = datetime.fromtimestamp(sleep_end / 1000)
                print(f"Method 1 (ms/1000): {end_dt1} ({end_dt1.strftime('%I:%M %p')})")
                
                try:
                    end_dt2 = datetime.fromtimestamp(sleep_end)
                    print(f"Method 2 (direct): {end_dt2} ({end_dt2.strftime('%I:%M %p')})")
                except (ValueError, OSError) as e:
                    print(f"Method 2 failed: {e}")
                
                end_dt3 = datetime.utcfromtimestamp(sleep_end / 1000)
                print(f"Method 3 (UTC): {end_dt3} ({end_dt3.strftime('%I:%M %p')})")
            
            # Check timezone information
            print(f"\n🌍 SYSTEM TIMEZONE INFO:")
            import time
            print(f"System timezone: {time.tzname}")
            print(f"Current time: {datetime.now()}")
            print(f"UTC time: {datetime.utcnow()}")
            
            # Compare with what our current server returns
            print(f"\n🔧 CURRENT SERVER OUTPUT:")
            start_time = datetime.fromtimestamp(sleep_start/1000).strftime("%I:%M %p") if sleep_start else "Unknown"
            end_time = datetime.fromtimestamp(sleep_end/1000).strftime("%I:%M %p") if sleep_end else "Unknown"
            print(f"Current server shows: {start_time} - {end_time}")
            
        else:
            print("❌ No dailySleepDTO found in sleep data")
            
    except Exception as e:
        print(f"❌ Error getting sleep data: {e}")

if __name__ == "__main__":
    debug_sleep_timestamps()