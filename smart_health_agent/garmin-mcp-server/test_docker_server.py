#!/usr/bin/env python3
"""
Test the Docker MCP server to verify all 15 tools are available
"""

import asyncio
import json
import subprocess
import sys

async def test_docker_mcp_server():
    """Test Docker MCP server tools listing."""
    print("🐳 Testing Docker MCP Server...")
    
    try:
        # Start the Docker container with MCP server
        cmd = [
            'docker', 'run', '--rm', '-i', 
            '--env-file', '.env',
            'garmin-mcp-server'
        ]
        
        print("Starting Docker container...")
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/Users/stan/dev/projects/health-agent-MCP/smart_health_agent/garmin-mcp-server'
        )
        
        # Send MCP initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-docker-client",
                    "version": "1.0.0"
                }
            }
        }
        
        print("Sending initialize request...")
        process.stdin.write(json.dumps(init_request) + '\n')
        process.stdin.flush()
        
        # Read initialize response
        init_response = process.stdout.readline().strip()
        print(f"Initialize response: {init_response[:100]}...")
        
        # Send tools/list request
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        print("Sending tools/list request...")
        process.stdin.write(json.dumps(tools_request) + '\n')
        process.stdin.flush()
        
        # Read tools response
        tools_response = process.stdout.readline().strip()
        
        try:
            tools_data = json.loads(tools_response)
            if 'result' in tools_data and 'tools' in tools_data['result']:
                tools = tools_data['result']['tools']
                print(f"\n✅ Found {len(tools)} tools in Docker server:")
                
                expected_tools = [
                    "get_auth_status", "get_profile", "get_daily_summary", 
                    "get_sleep_data", "get_heart_rate_data", "get_stress_data",
                    "get_activities", "get_weekly_summary", "get_monthly_summary",
                    "get_date_range_data", "get_trends_analysis", "get_goals_progress",
                    "get_health_insights", "get_steps_detail", "get_body_battery"
                ]
                
                found_tools = [tool['name'] for tool in tools]
                
                print("\n📋 Tool Comparison:")
                for i, expected in enumerate(expected_tools, 1):
                    status = "✅" if expected in found_tools else "❌"
                    print(f"  {i:2d}. {status} {expected}")
                
                missing = set(expected_tools) - set(found_tools)
                extra = set(found_tools) - set(expected_tools)
                
                if missing:
                    print(f"\n❌ Missing tools: {missing}")
                if extra:
                    print(f"\n➕ Extra tools: {extra}")
                
                if len(found_tools) >= 15 and not missing:
                    print(f"\n🎉 SUCCESS: Docker server has all {len(found_tools)} expected tools!")
                    return True
                else:
                    print(f"\n⚠️  PARTIAL: Found {len(found_tools)} tools, expected 15")
                    return False
                
            else:
                print("❌ Invalid tools response format")
                print(f"Response: {tools_response}")
                return False
                
        except json.JSONDecodeError as e:
            print(f"❌ Could not parse tools response: {e}")
            print(f"Raw response: {tools_response}")
            return False
        
        # Terminate the process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            
    except Exception as e:
        print(f"❌ Error testing Docker server: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_docker_mcp_server())
    sys.exit(0 if result else 1)