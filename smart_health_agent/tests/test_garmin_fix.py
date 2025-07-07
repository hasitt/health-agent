#!/usr/bin/env python3
"""
Test script to verify Garmin integration fixes.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_garmin_integration():
    """Test the Garmin integration with error handling."""
    
    print("🧪 Testing Garmin Integration Fixes")
    print("=" * 40)
    
    try:
        from garmin_utils import get_garmin_health_data, GarminConnectError
        
        print("✅ Successfully imported garmin_utils")
        
        # Test the integration
        print("\n📊 Attempting to fetch Garmin data...")
        health_data = get_garmin_health_data()
        
        print("✅ Successfully retrieved Garmin data!")
        print(f"📈 Steps: {health_data.get('steps', {}).get('steps', 0)}")
        print(f"💓 Heart Rate: {health_data.get('heart_rate', {}).get('resting_hr', 0)} bpm")
        print(f"😴 Sleep: {health_data.get('sleep', {}).get('sleep_hours', 0)} hours")
        print(f"📊 HRV: {health_data.get('hrv', {}).get('hrv_weekly_average', 0)}")
        
        print("\n🎉 Garmin integration is working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages: pip install garminconnect python-dotenv")
        return False
        
    except GarminConnectError as e:
        print(f"❌ Garmin connection error: {e}")
        print("Please check your credentials in .env file")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_data_processing():
    """Test the data processing function."""
    
    print("\n🔧 Testing Data Processing")
    print("=" * 30)
    
    try:
        from smart_health_ollama import process_garmin_data
        
        # Test with sample data
        sample_garmin_data = {
            'steps': {
                'steps': 8500,
                'distance_meters': 6500,
                'calories': 2100,
                'active_calories': 1800
            },
            'heart_rate': {
                'resting_hr': 65,
                'max_hr': 180,
                'avg_hr': 75
            },
            'sleep': {
                'sleep_hours': 7.5,
                'sleep_score': 85,
                'deep_sleep_seconds': 14400,
                'rem_sleep_seconds': 7200
            },
            'hrv': {
                'hrv_weekly_average': 45
            }
        }
        
        processed_data = process_garmin_data(sample_garmin_data)
        
        print("✅ Data processing successful!")
        print(f"📈 Processed steps: {processed_data['steps']}")
        print(f"💓 Processed heart rate: {processed_data['heart_rate']}")
        print(f"😴 Processed sleep: {processed_data['sleep_hours']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        return False

if __name__ == "__main__":
    print("🏃‍♂️ Smart Health Agent - Garmin Integration Test")
    print("=" * 50)
    
    # Test 1: Garmin integration
    garmin_ok = test_garmin_integration()
    
    # Test 2: Data processing
    processing_ok = test_data_processing()
    
    print("\n📋 Test Results")
    print("=" * 20)
    print(f"Garmin Integration: {'✅ PASS' if garmin_ok else '❌ FAIL'}")
    print(f"Data Processing: {'✅ PASS' if processing_ok else '❌ FAIL'}")
    
    if garmin_ok and processing_ok:
        print("\n🎉 All tests passed! Garmin integration is ready to use.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.") 