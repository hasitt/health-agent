import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import garminconnect

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without it
    pass

# Load environment variables
load_dotenv()

from garminconnect import Garmin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GarminConnectError(Exception):
    """Custom exception for Garmin Connect related errors."""
    pass

class GarminHealthData:
    """
    A class to handle Garmin Connect authentication and data retrieval.
    Supports token persistence to avoid rate limits.
    """
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None, token_dir: str = ".garmin_tokens"):
        """
        Initialize Garmin Health Data client.
        
        Args:
            username: Garmin Connect username/email (optional, uses env vars if not provided)
            password: Garmin Connect password (optional, uses env vars if not provided)
            token_dir: Directory to store authentication tokens (default: .garmin_tokens)
        """
        self.username = username or os.getenv('GARMIN_USERNAME')
        self.password = password or os.getenv('GARMIN_PASSWORD')
        self.token_dir = token_dir
        self.client = None
        
        if not self.username or not self.password:
            raise GarminConnectError("Garmin credentials not provided. Set GARMIN_USERNAME and GARMIN_PASSWORD environment variables or pass them as parameters.")
    
    def login(self) -> bool:
        """
        Login to Garmin Connect with token persistence.
        
        Returns:
            bool: True if login successful, False otherwise
        """
        try:
            logger.info("Attempting to login to Garmin Connect...")
            
            # Create token directory if it doesn't exist
            os.makedirs(self.token_dir, exist_ok=True)
            
            # Try to resume from saved tokens first
            if self._try_resume_session():
                logger.info("Successfully resumed session from saved tokens")
                return True
            
            # If resume fails, perform fresh login
            logger.info("No valid saved tokens found, performing fresh login...")
            self.client = garminconnect.Garmin(self.username, self.password)
            self.client.login()
            
            # Save tokens for future use
            self._save_session()
            
            logger.info("Successfully logged in to Garmin Connect")
            return True
            
        except Exception as e:
            error_msg = f"Failed to login to Garmin Connect: {str(e)}"
            logger.error(error_msg)
            raise GarminConnectError(error_msg) from e
    
    def _try_resume_session(self) -> bool:
        """
        Try to resume session from saved tokens.
        
        Returns:
            bool: True if session resumed successfully, False otherwise
        """
        try:
            if not os.path.exists(self.token_dir):
                return False
            
            # Create client and try to load saved tokens
            self.client = garminconnect.Garmin(self.username, self.password)
            self.client.garth.load(self.token_dir)
            
            # Test if the session is still valid by making a simple API call
            # We'll try to get user profile as a test
            test_data = self.client.get_user_summary(datetime.now().strftime('%Y-%m-%d'))
            if test_data:
                logger.info("Successfully resumed session from saved tokens")
                return True
                
        except Exception as e:
            logger.debug(f"Failed to resume session: {e}")
            # Only clean up tokens if the session is actually invalid
            self._cleanup_invalid_tokens()
        
        return False
    
    def _save_session(self) -> None:
        """
        Save authentication tokens for future use.
        """
        try:
            if self.client and self.client.garth:
                self.client.garth.dump(self.token_dir)
                logger.info(f"Authentication tokens saved to {self.token_dir}")
        except Exception as e:
            logger.warning(f"Failed to save authentication tokens: {e}")
    
    def _cleanup_invalid_tokens(self) -> None:
        """
        Clean up invalid authentication tokens.
        """
        try:
            import shutil
            if os.path.exists(self.token_dir):
                shutil.rmtree(self.token_dir)
                logger.info("Cleaned up invalid authentication tokens")
        except Exception as e:
            logger.warning(f"Failed to cleanup invalid tokens: {e}")
    
    def logout(self) -> None:
        """
        Logout and cleanup saved tokens.
        """
        try:
            if self.client:
                # Don't automatically clean up tokens - let them persist
                self.client = None
                logger.info("Successfully logged out (tokens preserved for future use)")
        except Exception as e:
            logger.warning(f"Error during logout: {e}")
    
    def force_cleanup(self) -> None:
        """
        Force cleanup of saved tokens (use when tokens are invalid).
        """
        try:
            self._cleanup_invalid_tokens()
            logger.info("Forced cleanup of authentication tokens")
        except Exception as e:
            logger.warning(f"Error during forced cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.login()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.logout()
    
    def get_daily_steps(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Fetch daily step data for a specific date.
        
        Args:
            date: Date to fetch data for (defaults to today)
            
        Returns:
            Dict containing step statistics
        """
        if not self.client:
            raise GarminConnectError("Not logged in. Call login() first.")
        
        try:
            target_date = date or datetime.now()
            date_str = target_date.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching step data for {date_str}")
            
            # Get detailed steps data by time intervals
            steps_data = self.client.get_steps_data(date_str)
            
            # Calculate total steps from all intervals
            total_steps = sum(entry.get('steps', 0) for entry in steps_data) if steps_data else 0
            
            # Also get user summary for additional metrics
            try:
                daily_summary = self.client.get_user_summary(date_str)
                distance_meters = daily_summary.get('totalDistanceMeters', 0)
                calories = daily_summary.get('totalKilocalories', 0)
                active_calories = daily_summary.get('activeKilocalories', 0)
                bmr_calories = daily_summary.get('bmrKilocalories', 0)
            except Exception as e:
                logger.warning(f"Failed to get daily summary for {date_str}: {e}")
                distance_meters = 0
                calories = 0
                active_calories = 0
                bmr_calories = 0
            
            # Extract step data
            steps_data_result = {
                'date': date_str,
                'steps': total_steps,
                'distance_meters': distance_meters,
                'calories': calories,
                'active_calories': active_calories,
                'bmr_calories': bmr_calories,
                'steps_by_interval': steps_data if steps_data else []
            }
            
            logger.info(f"Retrieved {steps_data_result['steps']} steps for {date_str}")
            return steps_data_result
            
        except Exception as e:
            error_msg = f"Failed to fetch step data for {date_str}: {str(e)}"
            logger.error(error_msg)
            raise GarminConnectError(error_msg) from e
    
    def get_heart_rate_data(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Fetch heart rate data for a specific date.
        
        Args:
            date: Date to fetch data for (defaults to today)
            
        Returns:
            Dict containing heart rate statistics
        """
        if not self.client:
            raise GarminConnectError("Not logged in. Call login() first.")
        
        try:
            target_date = date or datetime.now()
            date_str = target_date.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching heart rate data for {date_str}")
            
            # Get heart rate data
            hr_data = self.client.get_heart_rates(date_str)
            
            # Handle case where heart rate data is None or empty
            if hr_data is None:
                logger.warning(f"No heart rate data available for {date_str}, using defaults")
                hr_data = {}
            
            # Extract heart rate statistics with safe defaults
            hr_stats = {
                'date': date_str,
                'resting_hr': hr_data.get('restingHeartRate', 0) if hr_data else 0,
                'max_hr': hr_data.get('maxHeartRate', 0) if hr_data else 0,
                'min_hr': hr_data.get('minHeartRate', 0) if hr_data else 0,
                'avg_hr': hr_data.get('averageHeartRate', 0) if hr_data else 0,
                'hr_zones': hr_data.get('heartRateZones', []) if hr_data else []
            }
            
            logger.info(f"Retrieved heart rate data for {date_str}")
            return hr_stats
            
        except Exception as e:
            logger.warning(f"Failed to fetch heart rate data for {date_str}: {str(e)}")
            # Return default heart rate data instead of raising an exception
            return {
                'date': date_str,
                'resting_hr': 0,
                'max_hr': 0,
                'min_hr': 0,
                'avg_hr': 0,
                'hr_zones': []
            }
    
    def get_hrv_data(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Fetch Heart Rate Variability (HRV) data for a specific date.
        
        Args:
            date: Date to fetch data for (defaults to today)
            
        Returns:
            Dict containing HRV data
        """
        if not self.client:
            raise GarminConnectError("Not logged in. Call login() first.")
        
        try:
            target_date = date or datetime.now()
            date_str = target_date.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching HRV data for {date_str}")
            
            # Get HRV data
            hrv_data = self.client.get_hrv_data(date_str)
            
            # Handle case where HRV data is None or empty
            if hrv_data is None:
                logger.warning(f"No HRV data available for {date_str}, using defaults")
                hrv_data = {}
            
            # Extract HRV statistics with safe defaults
            hrv_stats = {
                'date': date_str,
                'hrv_values': hrv_data.get('hrvValues', []) if hrv_data else [],
                'hrv_summary': hrv_data.get('hrvSummary', {}) if hrv_data else {},
                'hrv_weekly_average': hrv_data.get('weeklyAverage', 0) if hrv_data else 0
            }
            
            logger.info(f"Retrieved HRV data for {date_str}")
            return hrv_stats
            
        except Exception as e:
            logger.warning(f"Failed to fetch HRV data for {date_str}: {str(e)}")
            # Return default HRV data instead of raising an exception
            return {
                'date': date_str,
                'hrv_values': [],
                'hrv_summary': {},
                'hrv_weekly_average': 0
            }
    
    def get_sleep_data(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Fetch sleep data for a specific date.
        
        Args:
            date: Date to fetch data for (defaults to today)
            
        Returns:
            Dict containing sleep data
        """
        if not self.client:
            raise GarminConnectError("Not logged in. Call login() first.")
        
        try:
            target_date = date or datetime.now()
            date_str = target_date.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching sleep data for {date_str}")
            
            # Get sleep data
            sleep_data = self.client.get_sleep_data(date_str)
            
            # Handle case where sleep data is None or empty
            if sleep_data is None:
                logger.warning(f"No sleep data available for {date_str}, using defaults")
                sleep_data = {}
            
            # Extract daily sleep data from nested structure
            daily_sleep = sleep_data.get('dailySleepDTO', {}) if sleep_data else {}
            sleep_scores = daily_sleep.get('sleepScores', {}) if daily_sleep else {}
            
            # Extract sleep statistics with safe defaults
            sleep_stats = {
                'date': date_str,
                'sleep_time_seconds': daily_sleep.get('sleepTimeSeconds', 0),
                'sleep_hours': round(daily_sleep.get('sleepTimeSeconds', 0) / 3600, 2),
                'deep_sleep_seconds': daily_sleep.get('deepSleepSeconds', 0),
                'light_sleep_seconds': daily_sleep.get('lightSleepSeconds', 0),
                'rem_sleep_seconds': daily_sleep.get('remSleepSeconds', 0),
                'awake_sleep_seconds': daily_sleep.get('awakeSleepSeconds', 0),
                'sleep_score': sleep_scores.get('overall', {}).get('value', 0) if sleep_scores else 0,
                'sleep_start': daily_sleep.get('sleepStartTimestampGMT', None),
                'sleep_end': daily_sleep.get('sleepEndTimestampGMT', None),
                'sleep_quality': sleep_scores.get('overall', {}).get('qualifierKey', 'UNKNOWN') if sleep_scores else 'UNKNOWN'
            }
            
            logger.info(f"Retrieved sleep data for {date_str}: {sleep_stats['sleep_hours']} hours, score: {sleep_stats['sleep_score']}")
            return sleep_stats
            
        except Exception as e:
            logger.warning(f"Failed to fetch sleep data for {date_str}: {str(e)}")
            # Return default sleep data instead of raising an exception
            return {
                'date': date_str,
                'sleep_time_seconds': 0,
                'sleep_hours': 0,
                'deep_sleep_seconds': 0,
                'light_sleep_seconds': 0,
                'rem_sleep_seconds': 0,
                'awake_sleep_seconds': 0,
                'sleep_score': 0,
                'sleep_start': None,
                'sleep_end': None,
                'sleep_quality': 'UNKNOWN'
            }
    
    def get_comprehensive_health_data(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Fetch all health data (steps, heart rate, HRV, sleep, stress) for a specific date.
        
        Args:
            date: Date to fetch data for (defaults to today)
            
        Returns:
            Dict containing all health metrics
        """
        if not self.client:
            raise GarminConnectError("Not logged in. Call login() first.")
        
        try:
            target_date = date or datetime.now()
            date_str = target_date.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching comprehensive health data for {date_str}")
            
            # Fetch all data types with individual error handling
            health_data: Dict[str, Any] = {
                'date': date_str,
                'timestamp': datetime.now().isoformat(),
            }
            
            # Try to get steps data
            try:
                health_data['steps'] = self.get_daily_steps(target_date)
            except Exception as e:
                logger.warning(f"Failed to get steps data: {e}")
                health_data['steps'] = {'date': date_str, 'steps': 0, 'distance_meters': 0, 'calories': 0, 'active_calories': 0, 'bmr_calories': 0}
            
            # Try to get heart rate data
            try:
                health_data['heart_rate'] = self.get_heart_rate_data(target_date)
            except Exception as e:
                logger.warning(f"Failed to get heart rate data: {e}")
                health_data['heart_rate'] = {'date': date_str, 'resting_hr': 0, 'max_hr': 0, 'min_hr': 0, 'avg_hr': 0, 'hr_zones': []}
            
            # Try to get HRV data
            try:
                health_data['hrv'] = self.get_hrv_data(target_date)
            except Exception as e:
                logger.warning(f"Failed to get HRV data: {e}")
                health_data['hrv'] = {'date': date_str, 'hrv_values': [], 'hrv_summary': {}, 'hrv_weekly_average': 0}
            
            # Try to get sleep data
            try:
                health_data['sleep'] = self.get_sleep_data(target_date)
            except Exception as e:
                logger.warning(f"Failed to get sleep data: {e}")
                health_data['sleep'] = {'date': date_str, 'sleep_time_seconds': 0, 'sleep_hours': 0, 'deep_sleep_seconds': 0, 'light_sleep_seconds': 0, 'rem_sleep_seconds': 0, 'awake_sleep_seconds': 0, 'sleep_score': 0, 'sleep_start': None, 'sleep_end': None, 'sleep_quality': 'UNKNOWN'}
            
            # Try to get stress data from user summary
            try:
                summary = self.client.get_user_summary(date_str)
                stress_fields = [
                    'averageStressLevel', 'maxStressLevel', 'stressDuration', 'restStressDuration',
                    'activityStressDuration', 'uncategorizedStressDuration', 'totalStressDuration',
                    'lowStressDuration', 'mediumStressDuration', 'highStressDuration',
                    'stressPercentage', 'restStressPercentage', 'activityStressPercentage',
                    'uncategorizedStressPercentage', 'lowStressPercentage', 'mediumStressPercentage',
                    'highStressPercentage', 'stressQualifier'
                ]
                health_data['stress'] = {k: summary.get(k) for k in stress_fields if k in summary}
            except Exception as e:
                logger.warning(f"Failed to get stress data: {e}")
                health_data['stress'] = {}
            
            logger.info(f"Successfully retrieved comprehensive health data for {date_str}")
            return health_data
            
        except Exception as e:
            error_msg = f"Failed to fetch comprehensive health data for {date_str}: {str(e)}"
            logger.error(error_msg)
            raise GarminConnectError(error_msg) from e
    
    def get_weekly_averages(self, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate weekly averages for the past 7 days.
        
        Args:
            end_date: End date for the week (defaults to today)
            
        Returns:
            Dict containing weekly averages
        """
        if not self.client:
            raise GarminConnectError("Not logged in. Call login() first.")
        
        try:
            end_date = end_date or datetime.now()
            start_date = end_date - timedelta(days=6)
            
            logger.info(f"Calculating weekly averages from {start_date.date()} to {end_date.date()}")
            
            # Collect data for each day
            daily_data = []
            current_date = start_date
            
            while current_date <= end_date:
                try:
                    day_data = self.get_comprehensive_health_data(current_date)
                    daily_data.append(day_data)
                except GarminConnectError as e:
                    logger.warning(f"Could not fetch data for {current_date.date()}: {e}")
                
                current_date += timedelta(days=1)
            
            if not daily_data:
                raise GarminConnectError("No data available for the specified week")
            
            # Calculate averages
            total_steps = sum(day['steps']['steps'] for day in daily_data)
            total_sleep_hours = sum(day['sleep']['sleep_hours'] for day in daily_data)
            resting_hr_values = [day['heart_rate']['resting_hr'] for day in daily_data if day['heart_rate']['resting_hr'] > 0]
            avg_resting_hr = sum(resting_hr_values) / len(resting_hr_values) if resting_hr_values else 0
            
            weekly_averages = {
                'period': f"{start_date.date()} to {end_date.date()}",
                'steps_avg_7d': round(total_steps / len(daily_data), 2),
                'sleep_hours_avg_7d': round(total_sleep_hours / len(daily_data), 2),
                'resting_hr_avg_7d': round(avg_resting_hr, 2),
                'days_with_data': len(daily_data),
                'total_days': 7
            }
            
            logger.info(f"Calculated weekly averages: {weekly_averages}")
            return weekly_averages
            
        except Exception as e:
            error_msg = f"Failed to calculate weekly averages: {str(e)}"
            logger.error(error_msg)
            raise GarminConnectError(error_msg) from e
    
    def get_detailed_stress_data(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Fetch detailed time-series stress data for TCM analysis.
        
        Args:
            date: Date to fetch data for (defaults to today)
            
        Returns:
            Dict containing detailed stress data with timestamps for TCM mapping
        """
        if not self.client:
            raise GarminConnectError("Not logged in. Call login() first.")
        
        try:
            target_date = date or datetime.now()
            date_str = target_date.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching detailed stress data for TCM analysis on {date_str}")
            
            # Get detailed stress data with timestamps
            stress_data = self.client.get_stress_data(date_str)
            
            if not stress_data:
                logger.warning(f"No stress data available for {date_str}")
                return {
                    'date': date_str,
                    'stress_timeline': [],
                    'body_battery_timeline': [],
                    'summary': {
                        'avg_stress': 0,
                        'max_stress': 0,
                        'high_stress_periods': 0,
                        'data_points': 0
                    }
                }
            
            # Extract stress timeline data
            stress_timeline = []
            if 'stressValuesArray' in stress_data and stress_data['stressValuesArray']:
                for entry in stress_data['stressValuesArray']:
                    if len(entry) >= 2:
                        timestamp_ms = entry[0]
                        stress_level = entry[1]
                        
                        # Convert timestamp to datetime
                        try:
                            timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000)
                            stress_timeline.append({
                                'timestamp': timestamp_dt,
                                'timestamp_ms': timestamp_ms,
                                'stress_level': stress_level,
                                'datetime_str': timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
                            })
                        except (ValueError, OSError) as e:
                            logger.warning(f"Invalid timestamp in stress data: {timestamp_ms}, error: {e}")
                            continue
            
            # Extract body battery timeline data
            body_battery_timeline = []
            if 'bodyBatteryValuesArray' in stress_data and stress_data['bodyBatteryValuesArray']:
                for entry in stress_data['bodyBatteryValuesArray']:
                    if len(entry) >= 4:
                        timestamp_ms = entry[0]
                        status = entry[1]
                        level = entry[2]
                        version = entry[3]
                        
                        # Convert timestamp to datetime
                        try:
                            timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000)
                            body_battery_timeline.append({
                                'timestamp': timestamp_dt,
                                'timestamp_ms': timestamp_ms,
                                'status': status,
                                'level': level,
                                'version': version,
                                'datetime_str': timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
                            })
                        except (ValueError, OSError) as e:
                            logger.warning(f"Invalid timestamp in body battery data: {timestamp_ms}, error: {e}")
                            continue
            
            # Calculate summary statistics
            stress_levels = [point['stress_level'] for point in stress_timeline if point['stress_level'] >= 0]
            avg_stress = sum(stress_levels) / len(stress_levels) if stress_levels else 0
            max_stress = max(stress_levels) if stress_levels else 0
            high_stress_periods = len([level for level in stress_levels if level > 50])  # Threshold for high stress
            
            summary = {
                'avg_stress': round(avg_stress, 2),
                'max_stress': max_stress,
                'high_stress_periods': high_stress_periods,
                'data_points': len(stress_timeline),
                'time_range': {
                    'start': stress_data.get('startTimestampGMT'),
                    'end': stress_data.get('endTimestampGMT')
                }
            }
            
            result = {
                'date': date_str,
                'stress_timeline': stress_timeline,
                'body_battery_timeline': body_battery_timeline,
                'summary': summary,
                'raw_data_keys': list(stress_data.keys())  # For debugging
            }
            
            logger.info(f"Retrieved {len(stress_timeline)} stress data points for {date_str}")
            logger.info(f"Stress summary: avg={summary['avg_stress']}, max={summary['max_stress']}, high_periods={summary['high_stress_periods']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to fetch detailed stress data for {date_str}: {str(e)}"
            logger.error(error_msg)
            raise GarminConnectError(error_msg) from e
    
    def get_stress_data_for_tcm(self, date: Optional[datetime] = None) -> List[tuple]:
        """
        Fetch stress data in the format required for TCM analysis.
        
        Args:
            date: Date to fetch data for (defaults to today)
            
        Returns:
            List of (timestamp, stress_level) tuples for TCM mapping
        """
        try:
            detailed_data = self.get_detailed_stress_data(date)
            
            # Convert to the format expected by TCM analysis
            stress_tuples = []
            for point in detailed_data.get('stress_timeline', []):
                if point['stress_level'] >= 0:  # Filter out invalid readings
                    stress_tuples.append((point['timestamp'], point['stress_level']))
            
            logger.info(f"Prepared {len(stress_tuples)} stress data points for TCM analysis")
            return stress_tuples
            
        except Exception as e:
            logger.error(f"Failed to prepare stress data for TCM: {e}")
            return []
    
    def get_multi_day_stress_data(self, start_date: Optional[datetime] = None, days: int = 7) -> Dict[str, Any]:
        """
        Fetch stress data for multiple days for trend analysis.
        
        Args:
            start_date: Start date (defaults to 7 days ago)
            days: Number of days to fetch (default 7)
            
        Returns:
            Dict containing stress data for multiple days
        """
        if not self.client:
            raise GarminConnectError("Not logged in. Call login() first.")
        
        try:
            end_date = start_date or datetime.now()
            start_date = end_date - timedelta(days=days-1)
            
            logger.info(f"Fetching stress data from {start_date.date()} to {end_date.date()}")
            
            multi_day_data = {
                'date_range': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d'),
                    'days': days
                },
                'daily_data': {},
                'trends': {}
            }
            
            current_date = start_date
            all_stress_levels = []
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                
                try:
                    daily_stress = self.get_detailed_stress_data(current_date)
                    multi_day_data['daily_data'][date_str] = daily_stress
                    
                    # Collect stress levels for trend analysis
                    stress_levels = [point['stress_level'] for point in daily_stress.get('stress_timeline', []) if point['stress_level'] >= 0]
                    if stress_levels:
                        all_stress_levels.extend(stress_levels)
                    
                    logger.info(f"Retrieved stress data for {date_str}: {daily_stress['summary']['data_points']} points")
                    
                except Exception as e:
                    logger.warning(f"Failed to get stress data for {date_str}: {e}")
                    multi_day_data['daily_data'][date_str] = {
                        'date': date_str,
                        'stress_timeline': [],
                        'body_battery_timeline': [],
                        'summary': {'avg_stress': 0, 'max_stress': 0, 'high_stress_periods': 0, 'data_points': 0}
                    }
                
                current_date += timedelta(days=1)
            
            # Calculate trends
            if all_stress_levels:
                multi_day_data['trends'] = {
                    'overall_avg_stress': round(sum(all_stress_levels) / len(all_stress_levels), 2),
                    'overall_max_stress': max(all_stress_levels),
                    'total_data_points': len(all_stress_levels),
                    'high_stress_frequency': round(len([s for s in all_stress_levels if s > 50]) / len(all_stress_levels) * 100, 2)
                }
            
            logger.info(f"Completed multi-day stress data collection: {len(multi_day_data['daily_data'])} days")
            return multi_day_data
            
        except Exception as e:
            error_msg = f"Failed to fetch multi-day stress data: {str(e)}"
            logger.error(error_msg)
            raise GarminConnectError(error_msg) from e

def get_garmin_health_data(username: Optional[str] = None, password: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to get Garmin health data with automatic login.
    
    Args:
        username: Garmin Connect username/email (optional, uses env vars if not provided)
        password: Garmin Connect password (optional, uses env vars if not provided)
        
    Returns:
        Dict containing comprehensive health data for today
        
    Raises:
        GarminConnectError: If authentication or data fetching fails
    """
    try:
        # Initialize and login
        garmin = GarminHealthData(username, password)
        garmin.login()
        
        # Get comprehensive health data for today
        health_data = garmin.get_comprehensive_health_data()
        
        return health_data
        
    except GarminConnectError:
        # Re-raise our custom exception
        raise
    except Exception as e:
        # Convert other exceptions to our custom type
        raise GarminConnectError(f"Unexpected error: {str(e)}") from e

# Example usage and testing
if __name__ == "__main__":
    try:
        # Test the functionality
        print("Testing Garmin Connect integration...")
        
        # Get today's health data
        health_data = get_garmin_health_data()
        
        print("Successfully retrieved health data:")
        print(f"Steps: {health_data['steps']['steps']}")
        print(f"Sleep: {health_data['sleep']['sleep_hours']} hours")
        print(f"Resting HR: {health_data['heart_rate']['resting_hr']} bpm")
        
    except GarminConnectError as e:
        print(f"Garmin Connect error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}") 