#!/usr/bin/env python3
"""
Garmin MCP Server - Main server implementation.

A Model Context Protocol server that provides access to Garmin Connect health 
and fitness data for AI assistants like Claude.
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import structlog
from mcp import ClientSession, StdioServerParameters
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    GetPromptRequest,
    GetPromptResult,
    ListPromptsRequest,
    ListPromptsResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListToolsRequest,
    ListToolsResult,
    ReadResourceRequest,
    ReadResourceResult,
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
import mcp.server.stdio
from pydantic import BaseModel, Field
from typing_extensions import Literal

from . import __version__
from .config import GarminMCPConfig
from .auth import GarminAuthenticator
from .data_fetcher import GarminDataFetcher
from .models import *
from .utils import setup_logging
from .validation import validate_tool_parameters, create_ai_friendly_error, ValidationError

# Configure structured logging
logger = structlog.get_logger(__name__)

class GarminMCPServer:
    """Main Garmin MCP Server class."""
    
    def __init__(self, config: GarminMCPConfig):
        self.config = config
        self.server = Server("garmin-mcp-server")
        self.authenticator: Optional[GarminAuthenticator] = None
        self.data_fetcher: Optional[GarminDataFetcher] = None
        
        # Register handlers
        self._register_handlers()
        
    def _register_handlers(self) -> None:
        """Register all MCP message handlers."""
        # The handlers are registered using decorators below
        pass
        
    async def initialize(self) -> None:
        """Initialize the server components."""
        logger.info("Initializing Garmin MCP Server", version=__version__)
        
        try:
            # Initialize authenticator
            self.authenticator = GarminAuthenticator(self.config)
            await self.authenticator.initialize()
            
            # Initialize data fetcher
            self.data_fetcher = GarminDataFetcher(
                authenticator=self.authenticator,
                config=self.config
            )
            
            logger.info("Garmin MCP Server initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Garmin MCP Server", error=str(e))
            raise
    
    async def _list_tools(self) -> ListToolsResult:
        """List available tools."""
        tools = [
            # Authentication tools
            Tool(
                name="get_auth_status",
                description="Check the authentication status with Garmin Connect",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name="get_profile", 
                description="Get user profile information from Garmin Connect",
                inputSchema={
                    "type": "object", 
                    "properties": {},
                    "required": []
                }
            ),
            
            # Daily data tools
            Tool(
                name="get_daily_summary",
                description="Get daily activity summary including steps, calories, and distance",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format, or relative terms like 'today', 'yesterday', '3 days ago' (default: today)",
                            "examples": ["2024-01-15", "today", "yesterday", "1 week ago"]
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="get_sleep_data",
                description="Get sleep data including duration, score, and quality metrics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format, or relative terms like 'today', 'yesterday', '3 days ago' (default: today)",
                            "examples": ["2024-01-15", "today", "yesterday", "1 week ago"]
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="get_heart_rate_data",
                description="Get heart rate data including resting HR and daily statistics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format, or relative terms like 'today', 'yesterday', '3 days ago' (default: today)",
                            "examples": ["2024-01-15", "today", "yesterday", "1 week ago"]
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="get_stress_data",
                description="Get stress level data and patterns",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format, or relative terms like 'today', 'yesterday', '3 days ago' (default: today)",
                            "examples": ["2024-01-15", "today", "yesterday", "1 week ago"]
                        },
                        "include_details": {
                            "type": "boolean",
                            "description": "Include hourly stress detail data",
                            "default": False
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="get_activities",
                description="Get activities and workouts for a specific date",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format, or relative terms like 'today', 'yesterday', '3 days ago' (default: today)",
                            "examples": ["2024-01-15", "today", "yesterday", "1 week ago"]
                        }
                    },
                    "required": []
                }
            ),
            
            # Historical data tools
            Tool(
                name="get_weekly_summary",
                description="Get 7-day summary with averages and trends",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "end_date": {
                            "type": "string",
                            "format": "date", 
                            "description": "End date in YYYY-MM-DD format (defaults to today)"
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="get_monthly_summary",
                description="Get 30-day summary with patterns and insights",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "end_date": {
                            "type": "string",
                            "format": "date",
                            "description": "End date in YYYY-MM-DD format (defaults to today)"
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="get_date_range_data", 
                description="Get data for a custom date range",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format, or relative terms like '1 week ago'",
                            "examples": ["2024-01-01", "1 week ago", "last month"]
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format, or relative terms like 'today'",
                            "examples": ["2024-01-31", "today", "yesterday"]
                        },
                        "metrics": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["steps", "sleep", "heart_rate", "stress", "activities"]
                            },
                            "description": "Specific metrics to retrieve (default: all)"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            ),
            
            # Analytics tools  
            Tool(
                name="get_trends_analysis",
                description="Get week-over-week trend analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "weeks": {
                            "type": "integer",
                            "minimum": 2,
                            "maximum": 12,
                            "description": "Number of weeks to analyze (default: 4)",
                            "default": 4
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="get_goals_progress",
                description="Get progress toward fitness goals",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "period": {
                            "type": "string",
                            "enum": ["daily", "weekly", "monthly"],
                            "description": "Time period for goal analysis",
                            "default": "weekly"
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="get_health_insights", 
                description="Get AI-powered health pattern insights",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "minimum": 7,
                            "maximum": 90, 
                            "description": "Number of days to analyze (default: 30)",
                            "default": 30
                        }
                    },
                    "required": []
                }
            ),
            
            # Detailed data tools
            Tool(
                name="get_steps_detail",
                description="Get detailed step data with hourly breakdown for a specific date",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format, or relative terms like 'today', 'yesterday' (default: today)",
                            "examples": ["2024-01-15", "today", "yesterday", "3 days ago"]
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="get_body_battery",
                description="Get body battery (energy level) data and patterns for a specific date",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format, or relative terms like 'today', 'yesterday' (default: today)",
                            "examples": ["2024-01-15", "today", "yesterday", "3 days ago"]
                        }
                    },
                    "required": []
                }
            ),
        ]
        
        logger.debug("Listed tools", tool_count=len(tools))
        return ListToolsResult(tools=tools)
    
    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Call a specific tool with comprehensive parameter validation."""
        logger.info("Calling tool", tool_name=name, arguments=arguments)
        
        try:
            if not self.data_fetcher:
                raise RuntimeError("Data fetcher not initialized")
            
            # Validate and normalize parameters
            try:
                validated_args = validate_tool_parameters(name, arguments)
                logger.debug("Parameters validated", tool_name=name, validated_args=validated_args)
            except ValidationError as ve:
                logger.warning("Parameter validation failed", tool_name=name, error=str(ve))
                error_response = create_ai_friendly_error(ve, name)
                return CallToolResult(
                    content=[TextContent(type="text", text=str(error_response))],
                    isError=True
                )
                
            # Route to appropriate handler with validated parameters
            if name == "get_auth_status":
                result = await self.data_fetcher.get_auth_status()
            elif name == "get_profile":
                result = await self.data_fetcher.get_profile()
            elif name == "get_daily_summary":
                result = await self.data_fetcher.get_daily_summary(**validated_args)
            elif name == "get_sleep_data":
                result = await self.data_fetcher.get_sleep_data(**validated_args)
            elif name == "get_heart_rate_data":
                result = await self.data_fetcher.get_heart_rate_data(**validated_args)
            elif name == "get_stress_data":
                result = await self.data_fetcher.get_stress_data(**validated_args)
            elif name == "get_activities":
                result = await self.data_fetcher.get_activities(**validated_args)
            elif name == "get_steps_detail":
                result = await self.data_fetcher.get_steps_detail(**validated_args)
            elif name == "get_body_battery":
                result = await self.data_fetcher.get_body_battery(**validated_args)
            elif name == "get_weekly_summary":
                result = await self.data_fetcher.get_weekly_summary(**validated_args)
            elif name == "get_monthly_summary":
                result = await self.data_fetcher.get_monthly_summary(**validated_args)
            elif name == "get_date_range_data":
                result = await self.data_fetcher.get_date_range_data(**validated_args)
            elif name == "get_trends_analysis":
                result = await self.data_fetcher.get_trends_analysis(**validated_args)
            elif name == "get_goals_progress":
                result = await self.data_fetcher.get_goals_progress(**validated_args)
            elif name == "get_health_insights":
                result = await self.data_fetcher.get_health_insights(**validated_args)
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            logger.debug("Tool executed successfully", tool_name=name)
            return CallToolResult(content=[TextContent(type="text", text=str(result))])
            
        except Exception as e:
            logger.error("Tool execution failed", tool_name=name, error=str(e))
            error_response = create_ai_friendly_error(e, name)
            return CallToolResult(
                content=[TextContent(type="text", text=str(error_response))],
                isError=True
            )
    
    async def _list_resources(self) -> ListResourcesResult:
        """List available resources."""
        resources = [
            Resource(
                uri="garmin://profile",
                name="User Profile",
                description="Current user profile and account information",
                mimeType="application/json"
            ),
            Resource(
                uri="garmin://devices", 
                name="Connected Devices",
                description="List of connected Garmin devices and their status",
                mimeType="application/json"
            ),
            Resource(
                uri="garmin://goals",
                name="Fitness Goals", 
                description="Current fitness goals and targets",
                mimeType="application/json"
            ),
            Resource(
                uri="garmin://recent-data",
                name="Recent Health Data",
                description="Summary of recent health and fitness metrics",
                mimeType="application/json"
            ),
        ]
        
        logger.debug("Listed resources", resource_count=len(resources))
        return ListResourcesResult(resources=resources)
    
    async def _read_resource(self, uri: str) -> ReadResourceResult:
        """Read a specific resource."""
        logger.info("Reading resource", uri=uri)
        
        try:
            if not self.data_fetcher:
                raise RuntimeError("Data fetcher not initialized")
            
            if uri == "garmin://profile":
                content = await self.data_fetcher.get_profile()
            elif uri == "garmin://devices":
                content = await self.data_fetcher.get_devices()
            elif uri == "garmin://goals":
                content = await self.data_fetcher.get_goals()
            elif uri == "garmin://recent-data":
                content = await self.data_fetcher.get_recent_summary()
            else:
                raise ValueError(f"Unknown resource URI: {uri}")
            
            logger.debug("Resource read successfully", uri=uri)
            return ReadResourceResult(
                contents=[
                    TextContent(type="text", text=str(content))
                ]
            )
            
        except Exception as e:
            logger.error("Resource read failed", uri=uri, error=str(e))
            return ReadResourceResult(
                contents=[
                    TextContent(type="text", text=f"Error reading resource: {str(e)}")
                ]
            )
    
    async def _list_prompts(self) -> ListPromptsResult:
        """List available prompts (if any)."""
        # For now, we don't provide prompts, but this could be extended
        return ListPromptsResult(prompts=[])
    
    async def _get_prompt(self, name: str, arguments: Dict[str, Any]) -> GetPromptResult:
        """Get a specific prompt (if any)."""
        raise ValueError(f"Unknown prompt: {name}")

async def main() -> None:
    """Main server entry point."""
    # Set up logging
    setup_logging()
    
    logger.info("Starting Garmin MCP Server", version=__version__)
    
    try:
        # Load configuration
        config = GarminMCPConfig.from_env()
        
        # Create and initialize server
        garmin_server = GarminMCPServer(config)
        await garmin_server.initialize()
        
        # Determine transport method
        if config.mcp_transport == "stdio":
            # STDIO transport (default for Claude Desktop)
            logger.info("Starting server with STDIO transport")
            
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await garmin_server.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="garmin-mcp-server",
                        server_version=__version__,
                        capabilities=garmin_server.server.get_capabilities(
                            notification_options=None,
                            experimental_capabilities=None
                        )
                    )
                )
                
        elif config.mcp_transport == "http":
            # HTTP transport (for remote access)
            logger.info("Starting server with HTTP transport", 
                       host=config.mcp_host, port=config.mcp_port)
            
            # This would require additional HTTP server setup
            # For now, we'll use STDIO as the primary transport
            raise NotImplementedError("HTTP transport not yet implemented")
            
        else:
            raise ValueError(f"Unknown transport: {config.mcp_transport}")
            
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error("Server failed to start", error=str(e), exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Garmin MCP Server stopped")

if __name__ == "__main__":
    asyncio.run(main())