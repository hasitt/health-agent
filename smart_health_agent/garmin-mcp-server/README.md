# Garmin MCP Server

A Model Context Protocol (MCP) server that provides access to Garmin Connect health and fitness data for AI assistants like Claude.

## Features

- 🏃‍♂️ **Comprehensive Health Data**: 9 tools covering steps, sleep, heart rate, stress, body battery, and activities
- 🔐 **Secure Authentication**: Token-based authentication with persistent storage
- ⚡ **Real-time Access**: Direct integration with Garmin Connect API
- 🧠 **AI Optimized**: Formatted responses perfect for AI assistant interactions
- 🔧 **Reliable**: Robust error handling and data validation
- 🧪 **MCP Compliant**: Full MCP protocol support for Claude Desktop

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Garmin Connect account
- Claude Desktop (for integration)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/garmin-mcp-server.git
cd garmin-mcp-server

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your Garmin credentials

# Run the server
python -m garmin_mcp.server
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t garmin-mcp-server .
docker run -p 8000:8000 --env-file .env garmin-mcp-server
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required: Garmin Connect credentials
GARMIN_EMAIL=your_email@example.com
GARMIN_PASSWORD=your_password

# Optional: Server configuration
MCP_TRANSPORT=stdio  # or http
LOG_LEVEL=INFO
CACHE_TTL=300
```

### Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "garmin": {
      "command": "python",
      "args": ["-m", "garmin_mcp.server"],
      "cwd": "/path/to/garmin-mcp-server",
      "env": {
        "GARMIN_EMAIL": "your_email@example.com",
        "GARMIN_PASSWORD": "your_password"
      }
    }
  }
}
```

## Available Tools (9 Total)

### Authentication & Profile
- `get_auth_status` - Check authentication status with Garmin Connect
- `get_profile` - Get user profile information and account details

### Daily Health Data
- `get_daily_summary` - Steps, calories, distance, and activity overview
- `get_sleep_data` - Sleep duration, score, stages, and quality metrics
- `get_heart_rate_data` - Resting HR, daily statistics, and heart rate zones
- `get_stress_data` - Stress levels, patterns, and daily analysis
- `get_body_battery` - Energy levels, charging/draining patterns (NEW!)
- `get_steps_detail` - Detailed step data with hourly breakdown (NEW!)

### Activities & Workouts
- `get_activities` - Activities and workouts for any specific date

### Historical Analytics (Available in comprehensive server)
- `get_weekly_summary` - 7-day averages and trends
- `get_monthly_summary` - 30-day patterns and insights
- `get_date_range_data` - Custom date range queries
- `get_trends_analysis` - Week-over-week trend analysis
- `get_health_insights` - AI-powered health pattern analysis

### Analytics
- `get_trends_analysis` - Week-over-week comparisons
- `get_goals_progress` - Progress toward fitness goals
- `get_health_insights` - AI-powered health pattern analysis

### Authentication
- `get_auth_status` - Check connection status
- `get_profile` - User profile and device information

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with MCP Inspector
mcp-inspector garmin-mcp-server
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/garmin_mcp --cov-report=html

# Run specific test types
pytest -m unit      # Unit tests only
pytest -m integration  # Integration tests only
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Security scanning
bandit -r src/
```

## Monitoring and Observability

### Health Checks

```bash
# Check server health (HTTP mode)
curl http://localhost:8000/health

# Check metrics
curl http://localhost:9000/metrics
```

### Logging

The server uses structured logging with configurable levels:

```bash
# Debug logging
LOG_LEVEL=DEBUG python -m garmin_mcp.server

# JSON formatted logs  
LOG_FORMAT=json python -m garmin_mcp.server
```

## Security

### Authentication
- OAuth 2.1 for HTTP transport
- Encrypted token storage
- Rate limiting and DDoS protection

### Privacy
- PII masking in logs
- Secure credential handling
- HIPAA-aware health data processing

### Best Practices
- Regular token rotation
- Input validation and sanitization
- Security audit logging

## Troubleshooting

### Common Issues

**Authentication Failed**
```bash
# Check credentials
python -c "from garmin_mcp.auth import test_connection; test_connection()"

# Clear cached tokens
rm -f .garmin_tokens
```

**Rate Limited**
```bash
# Wait and retry, or adjust rate limits in .env
GARMIN_API_RATE_LIMIT=5
```

**MCP Connection Issues**
```bash
# Test with MCP Inspector
mcp-inspector garmin-mcp-server

# Check Claude Desktop logs
tail -f ~/Library/Logs/Claude/claude-desktop.log
```

### Debug Mode

```bash
DEBUG_GARMIN_API=true LOG_LEVEL=DEBUG python -m garmin_mcp.server
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- 📖 [Documentation](https://github.com/yourusername/garmin-mcp-server#readme)
- 🐛 [Issues](https://github.com/yourusername/garmin-mcp-server/issues)
- 💬 [Discussions](https://github.com/yourusername/garmin-mcp-server/discussions)