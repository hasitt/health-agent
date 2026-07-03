#!/usr/bin/env python3
"""
Simple Garmin MCP Server - Working implementation with correct MCP patterns.
"""

import asyncio
import os
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent
import structlog

# Configure logging to go to STDERR only
import logging
import sys
logging.basicConfig(
    format="%(message)s",
    stream=sys.stderr,
    level=logging.INFO,
)

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.dev.ConsoleRenderer(colors=False)
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Create the server
server = Server("garmin-mcp-server")

@server.list_tools()
async def list_tools():
    """List available Garmin tools."""
    from mcp.types import Tool
    logger.info("Tools list requested")
    
    return [
        Tool(
            name="get_sleep_data",
            description="Get sleep data from Garmin Connect",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date for sleep data (e.g., 'last night', 'yesterday', '2024-08-06')"
                    }
                }
            }
        ),
        Tool(
            name="get_daily_summary", 
            description="Get daily activity summary from Garmin",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date for summary (e.g., 'today', 'yesterday', '2024-08-06')"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    logger.info("Tool called", tool_name=name, arguments=arguments)
    
    if name == "get_sleep_data":
        # Mock sleep data for now
        return [
            TextContent(
                type="text",
                text="""Sleep Data for last night:
                
🛌 **Sleep Duration**: 7 hours 23 minutes
💤 **Sleep Score**: 84/100 (Good)

**Sleep Stages**:
- Deep Sleep: 1h 45m (24%)
- Light Sleep: 4h 12m (57%) 
- REM Sleep: 1h 26m (19%)

**Sleep Quality**: Your sleep was good last night with adequate deep sleep and REM cycles. You went to bed around 10:30 PM and woke up naturally around 6:00 AM.

*Note: This is mock data. Real Garmin integration will be added once authentication is working.*"""
            )
        ]
    
    elif name == "get_daily_summary":
        # Mock daily summary
        return [
            TextContent(
                type="text", 
                text="""Daily Activity Summary for today:

📱 **Steps**: 8,247 / 10,000 (82% of goal)
🔥 **Active Calories**: 387 calories
📏 **Distance**: 5.8 km
🏃 **Active Minutes**: 45 minutes

**Heart Rate**:
- Resting: 58 bpm
- Average: 78 bpm  
- Max: 142 bpm

**Goals Progress**:
- Steps: 82% complete (1,753 to go)
- Active Minutes: 150% complete (goal achieved!)

*Note: This is mock data. Real Garmin integration will be added once authentication is working.*"""
            )
        ]
    
    else:
        return [
            TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )
        ]

async def main():
    """Run the MCP server."""
    logger.info("Starting Simple Garmin MCP Server")
    
    # Check for credentials 
    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD") 
    
    if not email or not password:
        logger.error("GARMIN_EMAIL and GARMIN_PASSWORD environment variables required")
        return
    
    logger.info("Environment variables found", email=email[:3] + "***")
    
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Server started with STDIO transport")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())