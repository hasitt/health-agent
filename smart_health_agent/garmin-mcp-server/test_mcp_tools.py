#!/usr/bin/env python3
"""
Direct test of the MCP server to list tools
"""

import asyncio
import json
import subprocess
import sys
import os

async def test_mcp_tools():
    """Test MCP server tools listing directly."""
    try:
        # Set environment variables
        env = os.environ.copy()
        env['GARMIN_EMAIL'] = os.environ.get('GARMIN_EMAIL', os.environ.get('GARMIN_USERNAME', ''))
        env['GARMIN_PASSWORD'] = os.environ.get('GARMIN_PASSWORD', '')
        env['LOG_LEVEL'] = 'INFO'
        
        print("Starting MCP server and testing tools list...")
        
        # Start the MCP server process
        process = subprocess.Popen(
            ['/Users/stan/.pyenv/versions/smart-health-agent/bin/python', 
             '/Users/stan/.local/bin/garmin-mcp-server'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
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
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        print("Sending initialize request...")
        process.stdin.write(json.dumps(init_request) + '\n')
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline()
        print(f"Initialize response: {response_line.strip()}")
        
        # Send tools/list request
        tools_request = {
            "jsonrpc": "2.0", 
            "id": 2,
            "method": "tools/list"
        }
        
        print("Sending tools/list request...")
        process.stdin.write(json.dumps(tools_request) + '\n')
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline()
        print(f"Tools list response: {response_line.strip()}")
        
        try:
            tools_response = json.loads(response_line)
            if 'result' in tools_response and 'tools' in tools_response['result']:
                tools = tools_response['result']['tools']
                print(f"\n✅ Found {len(tools)} tools:")
                for i, tool in enumerate(tools, 1):
                    print(f"  {i}. {tool['name']} - {tool['description']}")
            else:
                print("❌ Invalid tools response format")
        except json.JSONDecodeError:
            print("❌ Could not parse tools response")
        
        # Terminate the process
        process.terminate()
        process.wait(timeout=5)
        
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp_tools())