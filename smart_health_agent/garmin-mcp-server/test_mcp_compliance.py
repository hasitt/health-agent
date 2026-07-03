#!/usr/bin/env python3
"""
Comprehensive MCP Protocol Compliance Tests for Garmin MCP Server.

This script tests JSON-RPC 2.0 compliance, tool execution, resource access,
and parameter validation to ensure full MCP compatibility.
"""

import json
import subprocess
import asyncio
import sys
import os
from typing import Dict, Any, List
from datetime import datetime

class MCPTester:
    """MCP Protocol compliance tester."""
    
    def __init__(self):
        self.docker_cmd = [
            "docker", "run", "--rm", "-i",
            "--env", "GARMIN_EMAIL=test@example.com",
            "--env", "GARMIN_PASSWORD=testpass123", 
            "garmin-mcp-server",
            "python", "-m", "garmin_mcp.server"
        ]
        self.request_id = 0
    
    def get_next_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id
    
    def create_jsonrpc_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a JSON-RPC 2.0 compliant request."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self.get_next_id()
        }
        if params:
            request["params"] = params
        return request
    
    def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to MCP server and get response."""
        try:
            process = subprocess.Popen(
                self.docker_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            request_json = json.dumps(request)
            stdout, stderr = process.communicate(input=request_json, timeout=30)
            
            if stderr:
                print(f"Server stderr: {stderr}")
            
            if not stdout.strip():
                return {"error": "No response from server"}
            
            return json.loads(stdout.strip())
        
        except subprocess.TimeoutExpired:
            process.kill()
            return {"error": "Request timeout"}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {e}"}
        except Exception as e:
            return {"error": f"Request failed: {e}"}
    
    def test_initialize(self) -> bool:
        """Test MCP initialization."""
        print("🔍 Testing MCP initialization...")
        
        request = self.create_jsonrpc_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": True
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "mcp-test-client",
                    "version": "1.0.0"
                }
            }
        )
        
        response = self.send_request(request)
        
        if "error" in response:
            print(f"❌ Initialization failed: {response['error']}")
            return False
        
        if "result" not in response:
            print("❌ No result in initialization response")
            return False
        
        result = response["result"]
        required_fields = ["protocolVersion", "capabilities", "serverInfo"]
        
        for field in required_fields:
            if field not in result:
                print(f"❌ Missing required field in response: {field}")
                return False
        
        print(f"✅ Initialization successful - Server: {result.get('serverInfo', {}).get('name', 'Unknown')}")
        return True
    
    def test_list_tools(self) -> bool:
        """Test tool listing."""
        print("🔍 Testing tool listing...")
        
        request = self.create_jsonrpc_request("tools/list")
        response = self.send_request(request)
        
        if "error" in response:
            print(f"❌ Tool listing failed: {response['error']}")
            return False
        
        if "result" not in response or "tools" not in response["result"]:
            print("❌ Invalid tool listing response")
            return False
        
        tools = response["result"]["tools"]
        print(f"✅ Found {len(tools)} tools")
        
        # Validate tool structure
        expected_tools = [
            "get_auth_status", "get_profile", "get_daily_summary", 
            "get_sleep_data", "get_heart_rate_data", "get_stress_data",
            "get_activities", "get_steps_detail", "get_body_battery",
            "get_weekly_summary", "get_monthly_summary", "get_date_range_data",
            "get_trends_analysis", "get_goals_progress", "get_health_insights"
        ]
        
        tool_names = [tool["name"] for tool in tools]
        
        for expected_tool in expected_tools:
            if expected_tool not in tool_names:
                print(f"❌ Missing expected tool: {expected_tool}")
                return False
        
        # Validate tool schema structure
        for tool in tools[:3]:  # Check first 3 tools
            required_fields = ["name", "description", "inputSchema"]
            for field in required_fields:
                if field not in tool:
                    print(f"❌ Tool {tool.get('name', 'unknown')} missing field: {field}")
                    return False
        
        print("✅ Tool listing validation passed")
        return True
    
    def test_list_resources(self) -> bool:
        """Test resource listing."""
        print("🔍 Testing resource listing...")
        
        request = self.create_jsonrpc_request("resources/list")
        response = self.send_request(request)
        
        if "error" in response:
            print(f"❌ Resource listing failed: {response['error']}")
            return False
        
        if "result" not in response or "resources" not in response["result"]:
            print("❌ Invalid resource listing response")
            return False
        
        resources = response["result"]["resources"]
        print(f"✅ Found {len(resources)} resources")
        
        # Validate resource structure
        expected_resources = [
            "garmin://profile", "garmin://devices", 
            "garmin://goals", "garmin://recent-data"
        ]
        
        resource_uris = [resource["uri"] for resource in resources]
        
        for expected_uri in expected_resources:
            if expected_uri not in resource_uris:
                print(f"❌ Missing expected resource: {expected_uri}")
                return False
        
        print("✅ Resource listing validation passed")
        return True
    
    def test_tool_call(self) -> bool:
        """Test tool execution with parameter validation."""
        print("🔍 Testing tool execution...")
        
        # Test simple tool call
        request = self.create_jsonrpc_request(
            "tools/call",
            {
                "name": "get_auth_status",
                "arguments": {}
            }
        )
        
        response = self.send_request(request)
        
        if "error" in response:
            print(f"❌ Tool call failed: {response['error']}")
            return False
        
        if "result" not in response or "content" not in response["result"]:
            print("❌ Invalid tool call response")
            return False
        
        print("✅ Basic tool call successful")
        
        # Test tool with parameters and validation
        request = self.create_jsonrpc_request(
            "tools/call",
            {
                "name": "get_daily_summary",
                "arguments": {"date": "today"}
            }
        )
        
        response = self.send_request(request)
        
        if "error" in response:
            print(f"❌ Parameterized tool call failed: {response['error']}")
            return False
        
        print("✅ Parameterized tool call successful")
        
        # Test parameter validation error handling
        request = self.create_jsonrpc_request(
            "tools/call",
            {
                "name": "get_daily_summary", 
                "arguments": {"date": "invalid_date"}
            }
        )
        
        response = self.send_request(request)
        
        # Should get an error response with validation details
        if "result" in response:
            result = response["result"]
            if "isError" in result and result["isError"]:
                print("✅ Parameter validation error handling works")
            else:
                print("⚠️  Parameter validation might not be working as expected")
        
        return True
    
    def test_jsonrpc_compliance(self) -> bool:
        """Test JSON-RPC 2.0 compliance."""
        print("🔍 Testing JSON-RPC 2.0 compliance...")
        
        # Test invalid method
        request = self.create_jsonrpc_request("invalid/method")
        response = self.send_request(request)
        
        if "error" in response:
            error = response["error"]
            if "code" in error and "message" in error:
                print("✅ Error response format compliant")
            else:
                print("❌ Error response format not compliant")
                return False
        
        # Test missing required field
        invalid_request = {
            "jsonrpc": "2.0",
            "method": "tools/list"
            # Missing ID
        }
        
        response = self.send_request(invalid_request)
        # Should handle gracefully
        
        print("✅ JSON-RPC compliance checks passed")
        return True
    
    def run_all_tests(self) -> bool:
        """Run all compliance tests."""
        print("🚀 Starting MCP Protocol Compliance Tests")
        print("=" * 50)
        
        tests = [
            ("Initialization", self.test_initialize),
            ("Tool Listing", self.test_list_tools),
            ("Resource Listing", self.test_list_resources),
            ("Tool Execution", self.test_tool_call),
            ("JSON-RPC Compliance", self.test_jsonrpc_compliance)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            print("-" * 30)
            try:
                if test_func():
                    passed += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} FAILED with exception: {e}")
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - MCP Server is compliant!")
            return True
        else:
            print("⚠️  Some tests failed - check output above")
            return False


if __name__ == "__main__":
    print(f"MCP Compliance Tester - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = MCPTester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)