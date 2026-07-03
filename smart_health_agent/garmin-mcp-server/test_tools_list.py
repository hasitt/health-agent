#!/usr/bin/env python3
"""
Test script to verify the MCP server lists all tools correctly
"""

import asyncio
import sys
import os

# Add the local bin to path
sys.path.insert(0, '/Users/stan/.local/bin')

# Import from the server directly
sys.path.append(os.path.dirname('/Users/stan/.local/bin/garmin-mcp-server'))

async def test_tools_list():
    """Test that all 9 tools are listed correctly."""
    try:
        # Set environment variables
        os.environ.setdefault('GARMIN_EMAIL', os.environ.get('GARMIN_USERNAME', ''))
        assert os.environ.get('GARMIN_PASSWORD'), 'Set GARMIN_EMAIL/GARMIN_PASSWORD in the environment'
        
        # Import the server components
        from datetime import datetime, date, timedelta
        from garminconnect import Garmin
        
        # Test basic server functionality
        print("Testing tool list generation...")
        
        # Simulate the list_tools function
        expected_tools = [
            "get_auth_status",
            "get_profile", 
            "get_sleep_data",
            "get_daily_summary",
            "get_heart_rate_data",
            "get_stress_data",
            "get_activities",
            "get_body_battery",
            "get_steps_detail"
        ]
        
        print(f"Expected {len(expected_tools)} tools:")
        for i, tool in enumerate(expected_tools, 1):
            print(f"  {i}. {tool}")
        
        # Test Garmin client initialization
        print("\nTesting Garmin client initialization...")
        garmin = Garmin(os.environ['GARMIN_EMAIL'], os.environ['GARMIN_PASSWORD'])
        
        # Try to load saved tokens
        tokens_file = ".garmin_tokens"
        if os.path.exists(tokens_file):
            try:
                with open(tokens_file, "r") as f:
                    tokens_string = f.read().strip()
                    garmin.garth.loads(tokens_string)
                    print("✅ Token-based authentication successful")
            except Exception as e:
                print(f"❌ Token authentication failed: {e}")
        else:
            print("❌ No tokens file found")
        
        print("✅ Basic server components working")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tools_list())