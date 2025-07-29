import os
import json
import asyncio
import logging
from datetime import datetime, timedelta, date
from garminconnect import Garmin
import garminconnect
from dotenv import load_dotenv
from database import db # Import the global db instance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Debug: Check garminconnect library version
try:
    logger.info(f"DEBUG: garminconnect library version: {garminconnect.__version__}")
except AttributeError:
    logger.info("DEBUG: garminconnect library version: Unknown (no __version__ attribute)")

# Load environment variables from .env file
load_dotenv()

# Garmin tokens file
GARMIN_TOKENS_FILE = ".garmin_tokens"

# Global Garmin client instance
garmin_client = None

# Global variable to store the current user ID (for multi-user support later)
current_user_id = 1 # Defaulting to 1 for now

def _save_garmin_tokens(client):
    """Saves Garmin authentication tokens to a file."""
    try:
        # Use dumps() method to get token string
        tokens_string = client.garth.dumps()
        logger.debug(f"Token string length: {len(tokens_string) if tokens_string else 0}")
        
        with open(GARMIN_TOKENS_FILE, "w") as f:
            f.write(tokens_string)
        logger.info("Authentication tokens saved to %s", GARMIN_TOKENS_FILE)
        
        # Verify the file was written correctly
        if os.path.exists(GARMIN_TOKENS_FILE):
            file_size = os.path.getsize(GARMIN_TOKENS_FILE)
            logger.debug(f"Token file created successfully, size: {file_size} bytes")
        else:
            logger.error("Token file was not created successfully")
    except Exception as e:
        logger.error("Failed to save Garmin tokens: %s", e)

def _load_garmin_tokens():
    """Loads Garmin authentication tokens from a file."""
    logger.debug(f"Checking for token file: {GARMIN_TOKENS_FILE}")
    if os.path.exists(GARMIN_TOKENS_FILE):
        logger.debug(f"Token file exists: {GARMIN_TOKENS_FILE}")
        try:
            with open(GARMIN_TOKENS_FILE, "r") as f:
                tokens_string = f.read().strip()
                logger.debug(f"Successfully loaded token string from file. Length: {len(tokens_string)}")
                return tokens_string if tokens_string else None
        except Exception as e:
            logger.error("Failed to load Garmin tokens: %s", e)
            # If loading fails, delete the corrupted token file
            os.remove(GARMIN_TOKENS_FILE)
            logger.info("Cleaned up invalid authentication tokens")
    else:
        logger.debug(f"Token file does not exist: {GARMIN_TOKENS_FILE}")
    return None

def initialize_garmin_client():
    """Initializes and logs into the Garmin Connect client."""
    global garmin_client
    if garmin_client:
        logger.info("Garmin client already initialized.")
        return garmin_client

    # Load and validate credentials at runtime - support both GARMIN_EMAIL and GARMIN_USERNAME
    GARMIN_EMAIL = os.getenv("GARMIN_EMAIL") or os.getenv("GARMIN_USERNAME")
    GARMIN_PASSWORD = os.getenv("GARMIN_PASSWORD")
    
    logger.debug(f"Loaded credentials from environment: Email/Username={'Set' if GARMIN_EMAIL else 'Not Set'}, Password={'Set' if GARMIN_PASSWORD else 'Not Set'}")
    
    if not GARMIN_EMAIL or not GARMIN_PASSWORD:
        error_msg = (
            "Garmin credentials not found. Please ensure your .env file contains:\n"
            "GARMIN_EMAIL=your_email@example.com (or GARMIN_USERNAME=your_email@example.com)\n"
            "GARMIN_PASSWORD=your_password\n"
            f"Current working directory: {os.getcwd()}\n"
            f".env file exists: {os.path.exists('.env')}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.debug(f"Creating Garmin client with email: {GARMIN_EMAIL[:3]}...{GARMIN_EMAIL[-10:] if len(GARMIN_EMAIL) > 13 else '***'}")
    garmin_client = Garmin(GARMIN_EMAIL, GARMIN_PASSWORD)
    logger.debug("Garmin client created successfully")

    # Step 1: Try to load and use saved tokens
    tokens_string = _load_garmin_tokens()
    if tokens_string:
        logger.info("Found saved tokens, attempting token-based login...")
        try:
            # Load tokens into the Garmin client
            garmin_client.garth.loads(tokens_string)
            logger.info("✅ Token-based login successful - tokens loaded into Garmin client.")
            return garmin_client
            
        except Exception as token_error:
            logger.warning(f"Token-based login failed: {token_error}")
            logger.info("Cleaning up invalid tokens and attempting fresh credential login...")
            # Remove invalid token file
            if os.path.exists(GARMIN_TOKENS_FILE):
                os.remove(GARMIN_TOKENS_FILE)
                logger.debug(f"Deleted invalid token file: {GARMIN_TOKENS_FILE}")
    else:
        logger.info("No saved tokens found, proceeding with credential-based login...")
    
    # Step 2: Perform fresh credential-based login with retry logic
    logger.info("🔑 Attempting fresh credential-based login to Garmin Connect...")
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Login attempt {attempt}/{max_attempts}")
            garmin_client.login()
            logger.info("✅ Fresh credential login successful!")
            
            # Save tokens after successful login
            logger.info("Saving authentication tokens for future use...")
            _save_garmin_tokens(garmin_client)
            logger.info("✅ Authentication tokens saved successfully")
            
            return garmin_client
            
        except Exception as login_error:
            logger.error(f"❌ Login attempt {attempt}/{max_attempts} failed: {login_error}")
            
            if attempt == max_attempts:
                error_msg = (
                    f"❌ Failed to login to Garmin Connect after {max_attempts} attempts.\n"
                    "Please verify your credentials in the .env file:\n"
                    "1. Check GARMIN_EMAIL is correct\n"
                    "2. Check GARMIN_PASSWORD is correct\n"
                    "3. Try logging into connect.garmin.com with these credentials\n"
                    f"Last error: {login_error}"
                )
                logger.error(error_msg)
                raise Exception(error_msg)
            else:
                logger.info(f"Retrying in 2 seconds... ({max_attempts - attempt} attempts remaining)")
                import time
                time.sleep(2)  # Brief delay between attempts
    
    return None

def _get_date_range_for_sync(days_to_sync):
    """Calculates the date range for syncing data."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days_to_sync - 1)
    return start_date, end_date

async def sync_garmin_data(user_id, days_to_sync=30, force_refresh=False):
    """
    Syncs recent Garmin data for a specified number of days.
    If force_refresh is True, all data for the period is re-downloaded.
    Otherwise, only missing data since last sync is fetched.
    """
    global garmin_client
    if not garmin_client:
        try:
            garmin_client = initialize_garmin_client()
        except Exception as e:
            logger.error("Failed to login to Garmin Connect: %s", e)
            return False # Indicate sync failure

    logger.debug("Set current_user_id to %d", user_id)
    logger.info("Syncing recent %d days of Garmin data for user %d", days_to_sync, user_id)

    today = datetime.now().date()
    start_date_sync, end_date_sync = _get_date_range_for_sync(days_to_sync)

    last_sync_time = db.get_sync_status(user_id, 'garmin')
    last_sync_date = None
    if last_sync_time and not force_refresh:
        try:
            if isinstance(last_sync_time, datetime):
                last_sync_date = last_sync_time.date()
            else:
                last_sync_date = datetime.strptime(last_sync_time, '%Y-%m-%d').date()
            # Start sync from the day after last sync, up to today
            start_date_sync = max(start_date_sync, last_sync_date + timedelta(days=1))
            logger.info("Incremental sync: Fetching data from %s to %s", start_date_sync, end_date_sync)
        except ValueError:
            logger.warning("Invalid last sync time format in DB: %s. Performing full refresh for period.", last_sync_time_str)
            force_refresh = True # Fallback to full refresh if sync status is bad

    if force_refresh:
        logger.info("Force refresh: Fetching all data from %s to %s", start_date_sync, end_date_sync)

    synced_days = 0
    current_date = start_date_sync
    while current_date <= end_date_sync:
        date_str = current_date.strftime('%Y-%m-%d')
        logger.info("Storing Garmin data for user %d on %s", user_id, date_str)

        try:
            # Fetch comprehensive health data
            logger.info("Fetching comprehensive health data for %s", date_str)
            
            # Debug: Inspect garmin_client object
            logger.debug(f"DEBUG: Type of garmin_client: {type(garmin_client)}")
            logger.debug(f"DEBUG: garmin_client is None: {garmin_client is None}")
            if garmin_client:
                available_methods = [method for method in dir(garmin_client) if not method.startswith('_')]
                logger.debug(f"DEBUG: Available methods in garmin_client: {available_methods}")
                # Look for methods that might be the correct one
                summary_methods = [method for method in available_methods if 'summary' in method.lower()]
                daily_methods = [method for method in available_methods if 'daily' in method.lower()]
                logger.debug(f"DEBUG: Methods containing 'summary': {summary_methods}")
                logger.debug(f"DEBUG: Methods containing 'daily': {daily_methods}")
            
            # Get daily steps data
            steps_data = garmin_client.get_daily_steps(current_date, current_date)
            steps = 0
            steps_distance_km = 0
            if steps_data and len(steps_data) > 0:
                steps = steps_data[0].get('totalSteps', 0)
                steps_distance_km = steps_data[0].get('totalDistance', 0) / 1000.0  # Convert meters to km
                logger.info("Retrieved %d steps and %.2f km from steps data for %s", steps, steps_distance_km, date_str)
            else:
                logger.info("No steps data for %s", date_str)

            # Get active calories from activities
            activities = garmin_client.get_activities_by_date(date_str, date_str)
            active_calories = 0
            activities_distance_km = 0
            if activities:
                for activity in activities:
                    if 'activeCalories' in activity:
                        active_calories += activity.get('activeCalories', 0)
                    elif 'calories' in activity:
                        active_calories += activity.get('calories', 0)
                    if 'distance' in activity:
                        activities_distance_km += activity.get('distance', 0) / 1000.0  # Convert meters to km
                logger.info("Retrieved %d active calories and %.2f km from %d activities for %s", 
                           active_calories, activities_distance_km, len(activities), date_str)
            else:
                logger.info("No activities data for %s", date_str)
            
            # Use the maximum distance from either steps or activities
            distance_km = max(steps_distance_km, activities_distance_km)

            # Get comprehensive stress data using get_all_day_stress
            avg_stress = 0
            max_stress = 0  
            min_stress = 100  # Start high, will be lowered
            try:
                stress_data = garmin_client.get_all_day_stress(current_date)
                if stress_data:
                    avg_stress = stress_data.get('avgStressLevel', 0)
                    max_stress = stress_data.get('maxStressLevel', 0)
                    
                    # Calculate min stress from stress values array
                    stress_values = stress_data.get('stressValuesArray', [])
                    if stress_values:
                        valid_stress_values = [val[1] for val in stress_values if val[1] > 0]  # Filter out -1, -2 values
                        if valid_stress_values:
                            min_stress = min(valid_stress_values)
                        else:
                            min_stress = 0
                    else:
                        min_stress = 0
                    
                    logger.info("Retrieved stress data for %s: avg=%d, max=%d, min=%d, %d data points", 
                               date_str, avg_stress, max_stress, min_stress, len(stress_values))
                else:
                    logger.info("No stress data for %s", date_str)
            except Exception as e:
                logger.warning("Could not retrieve stress data for %s: %s", date_str, e)

            # Sleep - add comprehensive debug logging as per user's instructions
            try:
                logger.debug(f"Attempting to retrieve sleep data for {date_str}")
                sleep_data = garmin_client.get_sleep_data(date_str)
                logger.debug(f"DEBUG: Raw Sleep Data from API for {date_str}: {sleep_data}")
                
                # Initialize default values
                sleep_duration_hours = 0
                sleep_score = 0
                
                if sleep_data:
                    logger.debug(f"DEBUG: Sleep data keys: {list(sleep_data.keys()) if isinstance(sleep_data, dict) else 'Not a dict'}")
                    
                    # Extract sleep duration - based on actual API structure from debug output
                    if sleep_data.get('dailySleepDTO', {}).get('sleepTimeSeconds'):
                        sleep_time_seconds = sleep_data['dailySleepDTO']['sleepTimeSeconds']
                        sleep_duration_hours = sleep_time_seconds / 3600
                        logger.debug(f"DEBUG: Found dailySleepDTO.sleepTimeSeconds: {sleep_time_seconds} seconds = {sleep_duration_hours} hours")
                    elif sleep_data.get('sleepTimeSeconds'):
                        sleep_duration_hours = sleep_data['sleepTimeSeconds'] / 3600
                        logger.debug(f"DEBUG: Found direct sleepTimeSeconds: {sleep_data['sleepTimeSeconds']} seconds = {sleep_duration_hours} hours")
                    elif sleep_data.get('durationInSeconds'):
                        sleep_duration_hours = sleep_data['durationInSeconds'] / 3600
                        logger.debug(f"DEBUG: Found durationInSeconds: {sleep_data['durationInSeconds']} seconds = {sleep_duration_hours} hours")
                    elif sleep_data.get('totalSleepTimeSeconds'):
                        sleep_duration_hours = sleep_data['totalSleepTimeSeconds'] / 3600
                        logger.debug(f"DEBUG: Found totalSleepTimeSeconds: {sleep_data['totalSleepTimeSeconds']} seconds = {sleep_duration_hours} hours")
                    else:
                        logger.debug("DEBUG: No duration field found in sleep data")
                    
                    # Extract sleep score - based on actual API structure from debug output
                    if sleep_data.get('dailySleepDTO', {}).get('sleepScores', {}).get('overall', {}).get('value'):
                        sleep_score = sleep_data['dailySleepDTO']['sleepScores']['overall']['value']
                        logger.debug(f"DEBUG: Found dailySleepDTO.sleepScores.overall.value: {sleep_score}")
                    elif sleep_data.get('dailySleepDTO', {}).get('sleepScores', {}).get('overallScore'):
                        sleep_score = sleep_data['dailySleepDTO']['sleepScores']['overallScore']
                        logger.debug(f"DEBUG: Found dailySleepDTO.sleepScores.overallScore: {sleep_score}")
                    elif sleep_data.get('sleepScores', {}).get('overall', {}).get('value'):
                        sleep_score = sleep_data['sleepScores']['overall']['value']
                        logger.debug(f"DEBUG: Found direct sleepScores.overall.value: {sleep_score}")
                    elif sleep_data.get('sleepScores', {}).get('overallScore'):
                        sleep_score = sleep_data['sleepScores']['overallScore']
                        logger.debug(f"DEBUG: Found direct sleepScores.overallScore: {sleep_score}")
                    elif sleep_data.get('overallScore'):
                        sleep_score = sleep_data['overallScore']
                        logger.debug(f"DEBUG: Direct overallScore: {sleep_score}")
                    else:
                        logger.debug("DEBUG: No sleep score field found in sleep data")
                
                else:
                    logger.info("No sleep data returned from API for %s", date_str)
                
                logger.info("Retrieved sleep data for %s: %.2f hours, score: %d", date_str, sleep_duration_hours, sleep_score)
                
            except Exception as e:
                logger.warning("Error retrieving sleep data for %s: %s", date_str, e)
                sleep_duration_hours = 0
                sleep_score = 0

            # Heart Rate - extract from sleep data (workaround for garminconnect library bug)
            avg_rhr = 0
            
            # FIXED: Extract RHR from sleep data instead of broken heart rate APIs
            # The get_heart_rates and get_rhr_day methods have a garminconnect library bug 
            # where they pass None as user ID, but sleep data contains restingHeartRate  
            logger.debug(f"Extracting RHR from sleep data for {date_str}")
            
            if sleep_data and 'restingHeartRate' in sleep_data:
                avg_rhr = sleep_data['restingHeartRate']
                if avg_rhr and avg_rhr > 0:
                    logger.info("Retrieved RHR data from sleep API for %s: %d bpm", date_str, avg_rhr)
                else:
                    logger.info("Sleep data contains restingHeartRate but value is invalid for %s", date_str)
                    avg_rhr = 0
            else:
                logger.info("No restingHeartRate found in sleep data for %s", date_str)
                avg_rhr = 0

            # Store comprehensive data with all collected metrics
            db.upsert_garmin_daily_summary(
                user_id, date_str, steps, avg_rhr,
                avg_stress=avg_stress,
                max_stress=max_stress,
                min_stress=min_stress,
                active_calories=active_calories,
                distance_km=distance_km
            )
            logger.debug(f"DEBUG: About to store sleep data - user_id: {user_id}, date: {date_str}, hours: {sleep_duration_hours}, score: {sleep_score}")
            db.upsert_garmin_sleep(user_id, date_str, sleep_duration_hours, sleep_score)
            logger.info("Successfully retrieved comprehensive health data for %s", date_str)

            # Store granular stress data from stressValuesArray
            try:
                if stress_data and 'stressValuesArray' in stress_data:
                    stress_values = stress_data['stressValuesArray']
                    stored_stress_points = 0
                    for stress_entry in stress_values:
                        if len(stress_entry) >= 2 and stress_entry[1] > 0:  # Valid stress reading
                            timestamp_ms = stress_entry[0]
                            stress_level = stress_entry[1]
                            
                            # Convert timestamp from milliseconds to datetime
                            timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000)
                            timestamp_str = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Store individual stress reading
                            db.upsert_garmin_stress_detail(user_id, timestamp_str, stress_level)
                            stored_stress_points += 1
                    
                    logger.info("Stored %d granular stress data points for %s", stored_stress_points, date_str)
                else:
                    logger.info("No granular stress data available for %s", date_str)
            except Exception as e:
                logger.warning("Could not store granular stress data for %s: %s", date_str, e)

            # Activities - simplified for now
            try:
                activities = garmin_client.get_activities_by_date(date_str, date_str)
                logger.info("Retrieved %d activities for %s", len(activities) if activities else 0, date_str)
                # TODO: Process activities data structure properly
            except Exception as e:
                logger.warning("Could not retrieve activities for %s: %s", date_str, e)
                activities = []
            
            logger.info("Successfully stored data for %s", date_str)

            synced_days += 1

        except Exception as e:
            logger.error("Error syncing Garmin data for %s: %s", date_str, e)
            # Continue to next day even if one day fails
        current_date += timedelta(days=1)

    db.update_sync_status(user_id, 'garmin', today.strftime('%Y-%m-%d'))
    logger.info("Successfully synced %d/%d days of recent data", synced_days, days_to_sync)
    return True # Indicate sync success

