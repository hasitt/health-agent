# Claude Desktop Integration Guide

## Quick Start

1. **Build the Docker image** (if not already built):
   ```bash
   docker build -t garmin-mcp-server .
   ```

2. **Set up Claude Desktop configuration**:
   
   **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
   **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

   Add this configuration to your Claude Desktop config:

   ```json
   {
     "mcpServers": {
       "garmin-mcp": {
         "command": "docker",
         "args": [
           "run",
           "--rm",
           "-i",
           "-e", "GARMIN_EMAIL=your_email@example.com",
           "-e", "GARMIN_PASSWORD=your_password_here",
           "-e", "LOG_LEVEL=INFO",
           "garmin-mcp-server"
         ]
       }
     }
   }
   ```

3. **Replace credentials**:
   - `your_email@example.com` → Your Garmin Connect email
   - `your_password_here` → Your Garmin Connect password

4. **Restart Claude Desktop** to load the new MCP server

## Available Tools

The Garmin MCP server provides 15 tools for accessing your health data:

### Daily Data
- `get_daily_summary` - Get daily activity summary (steps, distance, calories)
- `get_sleep_data` - Get sleep information (duration, score, stages)  
- `get_heart_rate_data` - Get heart rate data and analysis
- `get_stress_data` - Get stress level measurements
- `get_body_battery` - Get body battery energy levels

### Activity Data
- `get_activities` - Get list of recorded activities
- `get_activity_detail` - Get detailed information about specific activity
- `get_steps_detail` - Get detailed step counts throughout the day

### Health Analysis
- `get_trends_analysis` - Get trends and patterns in your health data
- `get_health_insights` - Get AI-powered health pattern insights
- `get_date_range_data` - Get comprehensive data across date ranges

### Profile & Goals
- `get_profile` - Get user profile and device information
- `get_goals` - Get daily goals and progress
- `get_performance_summary` - Get MCP server performance metrics

### Advanced Features
- `get_health_snapshot` - Get comprehensive health overview

## Example Usage

Once configured, you can ask Claude questions like:

- "Show me my activity summary for today"
- "How has my sleep been trending over the past month?"
- "What were my heart rate zones during yesterday's workout?"
- "Give me insights about my health patterns this week"
- "How close am I to reaching my daily goals?"

## Resource Endpoints

The server also provides 4 MCP resource endpoints for direct data access:

- `garmin://profile` - User profile data
- `garmin://daily/{date}` - Daily summary for specific date
- `garmin://goals/current` - Current goal progress
- `garmin://performance/summary` - Server performance metrics

## Troubleshooting

### Authentication Issues
- Verify your Garmin Connect credentials are correct
- Check that you can log in to Garmin Connect website with the same credentials
- Ensure two-factor authentication is disabled or properly handled

### Connection Problems
- Restart Claude Desktop after configuration changes
- Check Docker is running and the image was built successfully
- Review logs by setting `LOG_LEVEL=DEBUG` in the configuration

### Performance Optimization
- The server includes intelligent caching to minimize API calls
- Rate limiting prevents hitting Garmin Connect API limits
- Circuit breakers protect against service failures

## Security Notes

- Credentials are passed as environment variables to the Docker container
- No credentials are stored persistently by the MCP server
- All communication uses HTTPS when connecting to Garmin Connect
- Consider using Docker secrets for production deployments

## Advanced Configuration

### Custom Log Level
```json
"env": {
  "LOG_LEVEL": "DEBUG"
}
```

### Custom Cache Settings
```json
"env": {
  "CACHE_TTL_SECONDS": "1800",
  "MAX_RETRIES": "3"
}
```

### Resource Limits
```json
"args": [
  "run", "--rm", "-i",
  "--memory=256m",
  "--cpus=0.5",
  "-e", "GARMIN_EMAIL=your_email@example.com",
  "-e", "GARMIN_PASSWORD=your_password_here",
  "garmin-mcp-server"
]
```

For more advanced configuration options, see the [Configuration Guide](docs/configuration.md).