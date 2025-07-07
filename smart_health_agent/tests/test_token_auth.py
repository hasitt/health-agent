#!/usr/bin/env python3
"""
Test script to verify token-based authentication for Garmin Connect.
This should avoid rate limits by persisting authentication tokens.
"""

import sys
import os
from datetime import datetime

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_token_authentication():
    """Test token-based authentication and data retrieval."""
    
    print("🔐 Testing Token-Based Authentication")
    print("=" * 40)
    
    try:
        from garmin_utils import GarminHealthData
        
        # Test 1: First login (should save tokens)
        print("\n1️⃣ First Login (should save tokens):")
        garmin1 = GarminHealthData()
        garmin1.login()
        
        # Get some data to verify it works
        steps = garmin1.get_daily_steps()
        print(f"   Steps today: {steps.get('steps', 0)}")
        
        # Test 2: Second login (should resume from tokens)
        print("\n2️⃣ Second Login (should resume from tokens):")
        garmin2 = GarminHealthData()
        garmin2.login()
        
        # Get data again to verify it still works
        steps2 = garmin2.get_daily_steps()
        print(f"   Steps today: {steps2.get('steps', 0)}")
        
        # Test 3: Verify tokens were saved
        print("\n3️⃣ Checking saved tokens:")
        token_dir = ".garmin_tokens"
        if os.path.exists(token_dir):
            print(f"   ✅ Token directory exists: {token_dir}")
            token_files = os.listdir(token_dir)
            print(f"   📁 Token files: {token_files}")
        else:
            print("   ❌ Token directory not found")
        
        # Test 4: Test context manager
        print("\n4️⃣ Testing context manager:")
        with GarminHealthData() as garmin3:
            steps3 = garmin3.get_daily_steps()
            print(f"   Steps today: {steps3.get('steps', 0)}")
        
        print("\n✅ Token-based authentication test completed successfully!")
        print("\n💡 Benefits:")
        print("   - Avoids rate limits by reusing saved tokens")
        print("   - Faster subsequent logins")
        print("   - More reliable for automated scripts")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_token_authentication() 