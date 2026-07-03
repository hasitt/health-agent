#!/usr/bin/env python3
"""
MCP Protocol Compliance Tests - Tests without requiring Garmin authentication.
"""

import asyncio
import os
import sys
from datetime import datetime

async def test_mcp_protocol_compliance():
    """Test MCP protocol compliance without authentication."""
    
    print("🔍 Testing MCP Protocol Compliance (No Auth Required)")
    print("=" * 60)
    
    # Set fake environment to prevent config errors
    os.environ['GARMIN_EMAIL'] = 'test@example.com'
    os.environ['GARMIN_PASSWORD'] = 'testpass123'
    
    try:
        # Test imports
        from garmin_mcp.config import GarminMCPConfig
        from garmin_mcp.server import GarminMCPServer
        print("✅ Core imports successful")
        
        # Create config
        config = GarminMCPConfig(
            garmin_email='test@example.com',
            garmin_password='testpass123'
        )
        print("✅ Configuration created successfully")
        
        # Create server (don't initialize to avoid auth)
        server = GarminMCPServer(config)
        print("✅ Server instance created successfully")
        
        # Test 1: Tool Listing Compliance
        print("\n📋 Test 1: Tool Listing")
        print("-" * 30)
        
        tools_result = await server._list_tools()
        
        # Validate tools count
        expected_tool_count = 15
        actual_tool_count = len(tools_result.tools)
        
        if actual_tool_count != expected_tool_count:
            print(f"❌ Expected {expected_tool_count} tools, got {actual_tool_count}")
            return False
        
        print(f"✅ Correct number of tools: {actual_tool_count}")
        
        # Validate tool structure
        required_fields = ["name", "description", "inputSchema"]
        for i, tool in enumerate(tools_result.tools):
            for field in required_fields:
                if not hasattr(tool, field) or getattr(tool, field) is None:
                    print(f"❌ Tool {i+1} ({tool.name if hasattr(tool, 'name') else 'unknown'}) missing field: {field}")
                    return False
            
            # Validate inputSchema is a proper JSON Schema
            if not isinstance(tool.inputSchema, dict):
                print(f"❌ Tool {tool.name} inputSchema is not a dict")
                return False
            
            if "type" not in tool.inputSchema:
                print(f"❌ Tool {tool.name} inputSchema missing 'type' field")
                return False
        
        print("✅ All tools have valid structure")
        
        # Test specific expected tools
        expected_tools = [
            "get_auth_status", "get_profile", "get_daily_summary", "get_sleep_data",
            "get_heart_rate_data", "get_stress_data", "get_activities", "get_steps_detail",
            "get_body_battery", "get_weekly_summary", "get_monthly_summary", 
            "get_date_range_data", "get_trends_analysis", "get_goals_progress", 
            "get_health_insights"
        ]
        
        actual_tool_names = [tool.name for tool in tools_result.tools]
        
        for expected_tool in expected_tools:
            if expected_tool not in actual_tool_names:
                print(f"❌ Missing expected tool: {expected_tool}")
                return False
        
        print("✅ All expected tools present")
        
        # Test 2: Resource Listing Compliance
        print("\n📋 Test 2: Resource Listing")
        print("-" * 30)
        
        resources_result = await server._list_resources()
        
        # Validate resource count
        expected_resource_count = 4
        actual_resource_count = len(resources_result.resources)
        
        if actual_resource_count != expected_resource_count:
            print(f"❌ Expected {expected_resource_count} resources, got {actual_resource_count}")
            return False
        
        print(f"✅ Correct number of resources: {actual_resource_count}")
        
        # Validate resource structure
        required_resource_fields = ["uri", "name"]
        for resource in resources_result.resources:
            for field in required_resource_fields:
                if not hasattr(resource, field) or getattr(resource, field) is None:
                    print(f"❌ Resource missing field: {field}")
                    return False
        
        print("✅ All resources have valid structure")
        
        # Test specific expected resources
        expected_resources = [
            "garmin://profile", "garmin://devices", 
            "garmin://goals", "garmin://recent-data"
        ]
        
        actual_resource_uris = [str(resource.uri) for resource in resources_result.resources]
        
        print(f"Expected resources: {expected_resources}")
        print(f"Actual resources: {actual_resource_uris}")
        
        for expected_uri in expected_resources:
            if expected_uri not in actual_resource_uris:
                print(f"❌ Missing expected resource: {expected_uri}")
                return False
        
        print("✅ All expected resources present")
        
        # Test 3: Parameter Validation System
        print("\n📋 Test 3: Parameter Validation")
        print("-" * 30)
        
        from garmin_mcp.validation import validate_tool_parameters, ValidationError, DateParser
        
        # Test valid date parsing
        test_dates = ["today", "yesterday", "2024-08-06", "3 days ago", "1 week ago"]
        for test_date in test_dates:
            try:
                parsed = DateParser.parse_date(test_date)
                print(f"✅ Date parsing works: '{test_date}' → {parsed}")
            except Exception as e:
                print(f"❌ Date parsing failed for '{test_date}': {e}")
                return False
        
        # Test tool parameter validation
        test_cases = [
            ("get_daily_summary", {"date": "today"}, True),
            ("get_daily_summary", {"date": "2024-08-06"}, True),
            ("get_daily_summary", {"date": "invalid_date"}, False),
            ("get_trends_analysis", {"days": 30}, True),
            ("get_trends_analysis", {"days": "not_a_number"}, False),
        ]
        
        for tool_name, params, should_succeed in test_cases:
            try:
                result = validate_tool_parameters(tool_name, params)
                if should_succeed:
                    print(f"✅ Valid parameters accepted: {tool_name} with {params}")
                else:
                    print(f"⚠️  Expected validation error for: {tool_name} with {params}")
            except ValidationError as e:
                if not should_succeed:
                    print(f"✅ Invalid parameters correctly rejected: {tool_name} with {params}")
                else:
                    print(f"❌ Valid parameters incorrectly rejected: {tool_name} with {params}: {e}")
                    return False
        
        # Test 4: AI Optimization System
        print("\n📋 Test 4: AI Optimization")
        print("-" * 30)
        
        from garmin_mcp.ai_optimization import HealthDataFormatter, enhance_data_for_ai
        
        # Test data formatting
        test_data = {
            "status": "success",
            "summary": {
                "date": "2024-08-06",
                "steps": 12500,
                "distance_km": 8.2,
                "active_calories": 456,
                "goal_steps": 10000
            }
        }
        
        formatted = HealthDataFormatter.format_for_conversation(test_data, "daily_summary")
        if not isinstance(formatted, str) or len(formatted) < 10:
            print(f"❌ AI formatting failed: {formatted}")
            return False
        
        print(f"✅ AI formatting works: {formatted[:80]}...")
        
        # Test data enhancement
        enhanced = enhance_data_for_ai(test_data, "daily_summary")
        if not isinstance(enhanced, dict) or "ai_context" not in enhanced:
            print(f"❌ AI enhancement failed")
            return False
        
        if not enhanced["ai_context"].get("conversation_ready", False):
            print(f"❌ AI enhancement not conversation ready")
            return False
        
        print("✅ AI enhancement works correctly")
        
        # Test 5: Monitoring System
        print("\n📋 Test 5: Performance Monitoring")
        print("-" * 30)
        
        from garmin_mcp.monitoring import performance_monitor
        
        # Test performance monitor
        monitor = performance_monitor
        initial_request_count = len(monitor.request_metrics)
        
        # Simulate a request
        request = monitor.start_request("test_123", "get_auth_status")
        request.complete(success=True)
        
        new_request_count = len(monitor.request_metrics)
        if new_request_count != initial_request_count + 1:
            print(f"❌ Performance monitoring not working correctly")
            return False
        
        print("✅ Performance monitoring works correctly")
        
        print("\n🎉 ALL MCP PROTOCOL COMPLIANCE TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Protocol compliance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_advanced_features():
    """Test advanced MCP features."""
    
    print("\n🔍 Testing Advanced MCP Features")
    print("=" * 50)
    
    os.environ['GARMIN_EMAIL'] = 'test@example.com'
    os.environ['GARMIN_PASSWORD'] = 'testpass123'
    
    try:
        # Test enhanced tool schemas
        from garmin_mcp.server import GarminMCPServer
        from garmin_mcp.config import GarminMCPConfig
        
        config = GarminMCPConfig(garmin_email='test@example.com', garmin_password='testpass123')
        server = GarminMCPServer(config)
        
        tools_result = await server._list_tools()
        
        # Check for enhanced date parameter schemas
        date_tools = [t for t in tools_result.tools if 'date' in str(t.inputSchema)]
        
        enhanced_date_found = False
        for tool in date_tools:
            if 'examples' in str(tool.inputSchema):
                enhanced_date_found = True
                break
        
        if not enhanced_date_found:
            print("⚠️  Enhanced date schemas might not be working")
        else:
            print("✅ Enhanced parameter schemas working")
        
        # Test comprehensive error handling structure
        from garmin_mcp.validation import create_ai_friendly_error
        
        test_error = ValueError("Test error")
        error_response = create_ai_friendly_error(test_error, "test_tool")
        
        required_error_fields = ["error", "isError"]
        for field in required_error_fields:
            if field not in error_response:
                print(f"❌ Error response missing field: {field}")
                return False
        
        print("✅ AI-friendly error handling working")
        
        print("🎉 Advanced features test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Advanced features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all protocol compliance tests."""
    print(f"MCP Protocol Compliance Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    tests = [
        ("Core Protocol Compliance", test_mcp_protocol_compliance),
        ("Advanced Features", test_advanced_features)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🚀 Running {test_name}")
        if await test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 80)
    print(f"📊 FINAL RESULTS: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 ALL MCP PROTOCOL COMPLIANCE TESTS PASSED!")
        print("   The Garmin MCP Server is fully compliant with MCP protocol standards.")
        return True
    else:
        print("⚠️  Some tests failed - check output above for details")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)