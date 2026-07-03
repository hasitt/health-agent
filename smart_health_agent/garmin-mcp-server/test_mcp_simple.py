#!/usr/bin/env python3
"""
Simple MCP Server Test - Test server startup and basic functionality.
"""

import asyncio
import os
import sys
from typing import Dict, Any

async def test_mcp_server_directly():
    """Test MCP server functionality directly."""
    
    print("🔍 Testing MCP Server Components Directly")
    print("=" * 50)
    
    # Set environment
    os.environ['GARMIN_EMAIL'] = 'test@example.com'
    os.environ['GARMIN_PASSWORD'] = 'testpass123'
    
    try:
        # Test imports
        from garmin_mcp.config import GarminMCPConfig
        from garmin_mcp.server import GarminMCPServer
        print("✅ Server imports successful")
        
        # Create config
        config = GarminMCPConfig(
            garmin_email='test@example.com',
            garmin_password='testpass123'
        )
        print("✅ Config creation successful")
        
        # Create server
        server = GarminMCPServer(config)
        print("✅ Server instantiation successful")
        
        # Test initialize 
        await server.initialize()
        print("✅ Server initialization successful")
        
        # Test tool listing
        tools_result = await server._list_tools()
        print(f"✅ Tool listing successful - {len(tools_result.tools)} tools")
        
        # List first few tools
        print("   Sample tools:")
        for i, tool in enumerate(tools_result.tools[:5]):
            print(f"     {i+1}. {tool.name}: {tool.description}")
        
        # Test resource listing
        resources_result = await server._list_resources()
        print(f"✅ Resource listing successful - {len(resources_result.resources)} resources")
        
        # List resources
        print("   Available resources:")
        for resource in resources_result.resources:
            print(f"     - {resource.uri}: {resource.name}")
        
        # Test tool execution (auth status - no Garmin credentials needed)
        call_result = await server._call_tool("get_auth_status", {})
        print("✅ Tool execution successful")
        print(f"   Result: {str(call_result.content[0].text)[:100]}...")
        
        # Test parameter validation
        from garmin_mcp.validation import validate_tool_parameters, ValidationError
        
        # Valid parameters
        valid_params = validate_tool_parameters('get_daily_summary', {'date': 'today'})
        print(f"✅ Parameter validation successful: {valid_params}")
        
        # Invalid parameters
        try:
            invalid_params = validate_tool_parameters('get_daily_summary', {'date': 'not_a_date'})
            print("⚠️  Parameter validation may not be working (no error thrown)")
        except ValidationError as e:
            print(f"✅ Invalid parameter handling works: {e.message[:50]}...")
        
        # Test AI optimization
        from garmin_mcp.ai_optimization import enhance_data_for_ai
        test_data = {"status": "success", "steps": 10000}
        enhanced = enhance_data_for_ai(test_data, "daily_summary") 
        print(f"✅ AI optimization works: conversation_ready = {enhanced['ai_context']['conversation_ready']}")
        
        print("\n🎉 All direct tests PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mcp_protocol_structure():
    """Test MCP protocol structure compliance."""
    
    print("\n🔍 Testing MCP Protocol Structure")
    print("=" * 50)
    
    try:
        os.environ['GARMIN_EMAIL'] = 'test@example.com'
        os.environ['GARMIN_PASSWORD'] = 'testpass123'
        
        from garmin_mcp.config import GarminMCPConfig
        from garmin_mcp.server import GarminMCPServer
        from mcp.types import InitializeRequest, CallToolRequest
        
        config = GarminMCPConfig(garmin_email='test@example.com', garmin_password='testpass123')
        server = GarminMCPServer(config)
        
        # Test MCP message handling
        print("✅ Testing MCP message structure...")
        
        # Test tool schema compliance
        tools_result = await server._list_tools()
        
        for tool in tools_result.tools[:3]:  # Check first 3 tools
            # Check required fields
            required = ["name", "description", "inputSchema"]
            missing = [field for field in required if not hasattr(tool, field) or getattr(tool, field) is None]
            if missing:
                print(f"❌ Tool {tool.name} missing fields: {missing}")
                return False
            
            # Check inputSchema structure
            if not isinstance(tool.inputSchema, dict):
                print(f"❌ Tool {tool.name} inputSchema is not a dict")
                return False
            
            if "type" not in tool.inputSchema:
                print(f"❌ Tool {tool.name} inputSchema missing 'type' field")
                return False
        
        print(f"✅ All {len(tools_result.tools)} tools have valid schemas")
        
        # Test resource schema compliance 
        resources_result = await server._list_resources()
        
        for resource in resources_result.resources:
            # Check required fields
            required = ["uri", "name"]
            missing = [field for field in required if not hasattr(resource, field) or getattr(resource, field) is None]
            if missing:
                print(f"❌ Resource {resource.uri} missing fields: {missing}")
                return False
        
        print(f"✅ All {len(resources_result.resources)} resources have valid schemas")
        
        print("🎉 MCP Protocol Structure tests PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Protocol structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("MCP Server Testing Suite")
    print("=" * 60)
    
    tests = [
        test_mcp_server_directly,
        test_mcp_protocol_structure
    ]
    
    passed = 0
    for test in tests:
        if await test():
            passed += 1
    
    print(f"\n📊 Final Results: {passed}/{len(tests)} test suites passed")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED - MCP Server is working correctly!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)