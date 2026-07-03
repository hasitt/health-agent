#!/usr/bin/env python3
"""
Minimal MCP client to test communication with the Garmin MCP server directly.
This bypasses Claude Desktop to isolate protocol issues.
"""

import asyncio
import json
import os
import sys
import subprocess
from typing import Dict, Any

class MCPTestClient:
    """Simple MCP client for testing protocol communication."""
    
    def __init__(self):
        self.server_process = None
        self.reader = None
        self.writer = None
        self.request_id = 1
    
    async def start_server(self):
        """Start the Garmin MCP server process."""
        env = {
            "GARMIN_EMAIL": os.environ.get("GARMIN_EMAIL", os.environ.get("GARMIN_USERNAME", "")),
            "GARMIN_PASSWORD": os.environ.get("GARMIN_PASSWORD", ""),
            "LOG_LEVEL": "DEBUG"
        }
        
        print("🚀 Starting Garmin MCP server...")
        self.server_process = await asyncio.create_subprocess_exec(
            "/Users/stan/.pyenv/versions/smart-health-agent/bin/python",
            "simple_server.py",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**env}
        )
        
        self.reader = self.server_process.stdout
        self.writer = self.server_process.stdin
        
        # Give server time to start
        await asyncio.sleep(2)
        print("✅ Server started")
    
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC message and get response."""
        if not self.writer:
            raise RuntimeError("Server not started")
        
        message_json = json.dumps(message)
        print(f"📤 Sending: {message_json}")
        
        self.writer.write(f"{message_json}\n".encode())
        await self.writer.drain()
        
        # Read response
        try:
            response_line = await asyncio.wait_for(self.reader.readline(), timeout=10.0)
            response_text = response_line.decode().strip()
            print(f"📥 Received: {response_text}")
            
            if response_text:
                return json.loads(response_text)
            else:
                return {"error": "No response received"}
        except asyncio.TimeoutError:
            return {"error": "Timeout waiting for response"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {e}", "raw_response": response_text}
    
    async def test_initialize(self):
        """Test MCP initialization."""
        print("\n🔧 Testing MCP initialization...")
        
        init_message = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0"
                }
            }
        }
        
        self.request_id += 1
        response = await self.send_message(init_message)
        
        if "result" in response:
            print("✅ Initialization successful")
            
            # Send initialized notification  
            initialized_message = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            
            print("📤 Sending initialized notification...")
            self.writer.write(f"{json.dumps(initialized_message)}\n".encode())
            await self.writer.drain()
            
            return True
        else:
            print(f"❌ Initialization failed: {response}")
            return False
    
    async def test_tools_list(self):
        """Test listing available tools."""
        print("\n🛠️  Testing tools/list...")
        
        tools_message = {
            "jsonrpc": "2.0", 
            "id": self.request_id,
            "method": "tools/list",
            "params": {}
        }
        
        self.request_id += 1
        response = await self.send_message(tools_message)
        
        if "result" in response and "tools" in response["result"]:
            tools = response["result"]["tools"]
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools[:3]:  # Show first 3
                print(f"  - {tool.get('name', 'unknown')}")
            if len(tools) > 3:
                print(f"  ... and {len(tools) - 3} more")
            return True
        else:
            print(f"❌ Tools list failed: {response}")
            return False
    
    async def test_sleep_data(self):
        """Test calling get_sleep_data tool."""
        print("\n😴 Testing get_sleep_data tool...")
        
        sleep_message = {
            "jsonrpc": "2.0",
            "id": self.request_id, 
            "method": "tools/call",
            "params": {
                "name": "get_sleep_data",
                "arguments": {"date": "last night"}
            }
        }
        
        self.request_id += 1
        response = await self.send_message(sleep_message)
        
        if "result" in response:
            print("✅ Sleep data call successful")
            content = response["result"].get("content", [])
            if content and len(content) > 0:
                text = content[0].get("text", "")[:200]  # First 200 chars
                print(f"📊 Sleep data preview: {text}...")
            return True
        else:
            print(f"❌ Sleep data call failed: {response}")
            return False
    
    async def cleanup(self):
        """Clean up server process."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        
        if self.server_process:
            print("🛑 Stopping server...")
            self.server_process.terminate()
            try:
                await asyncio.wait_for(self.server_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.server_process.kill()
                await self.server_process.wait()
        
        print("✅ Cleanup complete")

async def main():
    """Run the MCP test client."""
    client = MCPTestClient()
    
    try:
        await client.start_server()
        
        # Run tests
        tests = [
            client.test_initialize,
            client.test_tools_list, 
            client.test_sleep_data
        ]
        
        results = []
        for test in tests:
            try:
                result = await test()
                results.append(result)
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                results.append(False)
        
        # Summary
        print(f"\n📋 Test Results: {sum(results)}/{len(results)} passed")
        if all(results):
            print("🎉 All MCP protocol tests PASSED!")
        else:
            print("⚠️  Some tests failed - MCP protocol issues detected")
            
    except Exception as e:
        print(f"💥 Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())