# Garmin Connect Integration

This document describes how to integrate Garmin Connect data into the Smart Health Agent system.

## Overview

The Garmin integration allows you to fetch real-time health data from your Garmin device, including:
- Daily step counts
- Heart rate data (resting and throughout the day)
- Sleep data (duration, quality, sleep score)
- HRV (Heart Rate Variability) data
- Weekly averages and trends

## Authentication

### Token-Based Authentication (Recommended)

The system now supports **token-based authentication** to avoid Garmin's rate limits. This feature:

- **Saves authentication tokens** after the first successful login
- **Reuses saved tokens** for subsequent requests
- **Avoids rate limits** by not requiring fresh login each time
- **Automatically handles token expiration** and falls back to fresh login

#### How It Works

1. **First Login**: Performs fresh authentication and saves tokens to `.garmin_tokens/` directory
2. **Subsequent Logins**: Attempts to resume session using saved tokens
3. **Token Validation**: Tests saved tokens with a simple API call
4. **Fallback**: If tokens are invalid/expired, performs fresh login and saves new tokens

#### Benefits

- ✅ **Avoids Rate Limits**: No repeated login attempts
- ✅ **Faster Access**: Instant session resumption
- ✅ **More Reliable**: Better for automated scripts
- ✅ **Secure**: Tokens stored locally and added to `.gitignore`

### Setup

#### 1. Install Dependencies

```bash
pip install garminconnect python-dotenv
```

#### 2. Set Up Credentials

Create a `.env` file in the project root:

```bash
# Garmin Connect credentials
GARMIN_USERNAME=your_email@example.com
GARMIN_PASSWORD=your_password
```

**Note**: For security, never commit your `.env` file to version control.

#### 3. Run Setup Script (Optional)

```bash
python setup_garmin.py
```

This interactive script will help you set up your credentials securely.

## Usage

### Basic Usage

```python
from garmin_utils import GarminHealthData

# Initialize with token persistence
garmin = GarminHealthData()

# Login (will use saved tokens if available)
garmin.login()

# Fetch data
steps = garmin.get_daily_steps()
sleep = garmin.get_sleep_data()
heart_rate = garmin.get_heart_rate_data()

# Logout and cleanup
garmin.logout()
```

### Context Manager Usage (Recommended)

```python
from garmin_utils import GarminHealthData

# Automatically handles login/logout
with GarminHealthData() as garmin:
    steps = garmin.get_daily_steps()
    sleep = garmin.get_sleep_data()
    # ... other operations
```

### Custom Token Directory

```python
# Use a custom directory for tokens
garmin = GarminHealthData(token_dir="/path/to/tokens")
```

## Available Data Methods

### Daily Data

- `get_daily_steps(date)` - Get step count and statistics
- `get_heart_rate_data(date)` - Get heart rate data (resting and throughout day)
- `get_sleep_data(date)` - Get sleep duration, quality, and sleep score
- `get_hrv_data(date)` - Get Heart Rate Variability data

### Comprehensive Data

- `get_comprehensive_health_data(date)` - Get all health metrics for a date
- `get_weekly_averages()` - Get 7-day averages and trends

### Data Format

All methods return structured dictionaries with relevant health metrics:

```python
{
    "total_steps": 1166,
    "sleep_hours": 7.05,
    "sleep_score": 49,
    "resting_hr": 49,
    "hrv_avg": 45.2,
    # ... additional metrics
}
```

## Rate Limiting

### Understanding Rate Limits

Garmin Connect has rate limits to prevent abuse:
- **Login Rate Limit**: 1-hour block after multiple failed login attempts
- **API Rate Limit**: Limits on API calls per time period

### Avoiding Rate Limits

1. **Use Token Persistence**: The system automatically saves and reuses authentication tokens
2. **Limit API Calls**: Don't make excessive requests in short time periods
3. **Handle Errors Gracefully**: The system includes error handling for rate limit responses
4. **Use Caching**: Consider caching data locally for frequently accessed metrics

### Rate Limit Recovery

If you hit a rate limit:
1. **Wait 1 hour** for the block to reset
2. **Delete saved tokens** if they're causing issues: `rm -rf .garmin_tokens/`
3. **Retry** with fresh authentication

## Troubleshooting

### Common Issues

#### "Too Many Requests" Error
- **Cause**: Hit Garmin's rate limit
- **Solution**: Wait 1 hour, delete `.garmin_tokens/` directory, retry

#### "Invalid Credentials" Error
- **Cause**: Wrong username/password or account locked
- **Solution**: Verify credentials, check Garmin account status

#### "No Data Available" Error
- **Cause**: No data for the requested date or device not synced
- **Solution**: Check device sync status, try different dates

#### HRV Data Not Available
- **Cause**: Device doesn't support HRV or data not available
- **Solution**: Check device compatibility, verify HRV is enabled

### Debug Mode

Enable detailed logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Scripts

Use the provided test scripts to verify functionality:

```bash
# Test basic functionality
python test_garmin_fix.py

# Test token authentication
python test_token_auth.py

# Debug specific issues
python debug_garmin.py
python debug_steps.py
python debug_hrv.py
```

## Security Considerations

### Token Storage

- **Local Storage**: Tokens are stored locally in `.garmin_tokens/` directory
- **Git Ignored**: The directory is automatically added to `.gitignore`
- **Secure**: Tokens are stored using garth's secure token format

### Credential Management

- **Environment Variables**: Use `.env` file for credentials (never commit to git)
- **No Hardcoding**: Never hardcode credentials in source code
- **Regular Rotation**: Consider changing passwords periodically

### API Usage

- **Respectful Usage**: Don't make excessive API calls
- **Caching**: Cache data locally when possible
- **Error Handling**: Implement proper error handling for API failures

## Integration with Smart Health Agent

The Garmin integration is designed to work seamlessly with the Smart Health Agent:

1. **Data Source**: Select "Garmin" as your health data source
2. **Automatic Fetching**: Data is automatically fetched when needed
3. **Context Integration**: Health data is used for personalized recommendations
4. **Real-time Updates**: Data is refreshed when the app is used

## Future Enhancements

Potential improvements for the Garmin integration:

- [ ] **Webhook Support**: Real-time data updates via webhooks
- [ ] **Historical Data**: Bulk download of historical data
- [ ] **Activity Data**: Integration with workout and activity data
- [ ] **Device Management**: Support for multiple Garmin devices
- [ ] **Advanced Metrics**: Additional health metrics and analytics 