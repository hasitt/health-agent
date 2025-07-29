import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import scipy.stats as stats
import logging
from collections import defaultdict

from database import db # Import the global db instance

logger = logging.getLogger(__name__)

# Define all available metrics and their source tables/columns
# This dictionary is crucial for _load_unified_dataset
AVAILABLE_METRICS = {
    # Garmin Daily Summary
    'total_steps': {'table': 'garmin_daily_summary', 'column': 'total_steps'},
    'avg_daily_rhr': {'table': 'garmin_daily_summary', 'column': 'avg_daily_rhr'},
    'avg_daily_stress': {'table': 'garmin_daily_summary', 'column': 'avg_daily_stress'},
    'max_daily_stress': {'table': 'garmin_daily_summary', 'column': 'max_daily_stress'},
    'min_daily_stress': {'table': 'garmin_daily_summary', 'column': 'min_daily_stress'},
    'active_calories': {'table': 'garmin_daily_summary', 'column': 'active_calories'},
    'distance_km': {'table': 'garmin_daily_summary', 'column': 'distance_km'},

    # Garmin Sleep
    'sleep_duration_hours': {'table': 'garmin_sleep', 'column': 'sleep_duration_hours'},
    'sleep_score': {'table': 'garmin_sleep', 'column': 'sleep_score'},

    # Food Log Daily
    'total_calories': {'table': 'food_log_daily', 'column': 'total_calories'},
    'protein_g': {'table': 'food_log_daily', 'column': 'protein_g'},
    'carbohydrates_g': {'table': 'food_log_daily', 'column': 'carbohydrates_g'},
    'fat_g': {'table': 'food_log_daily', 'column': 'fat_g'},
    'caffeine_mg': {'table': 'food_log_daily', 'column': 'caffeine_mg'},
    'alcohol_units': {'table': 'food_log_daily', 'column': 'alcohol_units'},

    # Subjective Wellbeing
    'mood': {'table': 'subjective_wellbeing', 'column': 'mood'},
    'energy': {'table': 'subjective_wellbeing', 'column': 'energy'},
    'subjective_stress': {'table': 'subjective_wellbeing', 'column': 'stress'}, # Renamed to avoid conflict with garmin_stress
    'sleep_quality': {'table': 'subjective_wellbeing', 'column': 'sleep_quality'},
    'focus': {'table': 'subjective_wellbeing', 'column': 'focus'},
    'motivation': {'table': 'subjective_wellbeing', 'column': 'motivation'},
    'emotional_state': {'table': 'subjective_wellbeing', 'column': 'emotional_state'},
    'stress_triggers': {'table': 'subjective_wellbeing', 'column': 'stress_triggers'},
    'coping_strategies': {'table': 'subjective_wellbeing', 'column': 'coping_strategies'},
    'physical_symptoms': {'table': 'subjective_wellbeing', 'column': 'physical_symptoms'},
    'daily_events': {'table': 'subjective_wellbeing', 'column': 'daily_events'}
}

def _load_unified_dataset(user_id, start_date, end_date, metric_x_name, metric_y_name, filters, available_metrics_map):
    """
    Loads all required metrics (for X, Y, and all filters) into a single, unified Pandas DataFrame.
    Performs outer joins by date to keep all relevant days.
    """
    logger.debug(f"DEBUG: _load_unified_dataset called with start={start_date}, end={end_date}, metrics=X:{metric_x_name}, Y:{metric_y_name}, filters={filters}")
    
    required_metrics = {metric_x_name, metric_y_name}
    if filters:
        for filter_metric, _ in filters.items():
            required_metrics.add(filter_metric)

    logger.debug(f"Required metrics for unified dataset: {required_metrics}")

    # Group metrics by their source table
    tables_to_load = defaultdict(list)
    for metric in required_metrics:
        if metric in available_metrics_map:
            table_info = available_metrics_map[metric]
            tables_to_load[table_info['table']].append((metric, table_info['column']))
        else:
            logger.warning(f"Metric '{metric}' not found in AVAILABLE_METRICS map. It will not be loaded.")

    logger.debug(f"Tables needed: {dict(tables_to_load)}")

    unified_df = pd.DataFrame()
    
    # Load data from each required table and merge
    for table_name, metrics_in_table in tables_to_load.items():
        columns_to_fetch = [col_info[1] for col_info in metrics_in_table] # Get actual DB column names
        
        if table_name == 'garmin_daily_summary':
            data = db.get_garmin_daily_summary(user_id, start_date, end_date)
        elif table_name == 'garmin_sleep':
            data = db.get_garmin_sleep(user_id, start_date, end_date)
        elif table_name == 'food_log_daily':
            data = db.get_food_log_daily_summary(user_id, start_date, end_date)
        elif table_name == 'subjective_wellbeing':
            data = db.get_subjective_wellbeing(user_id, start_date, end_date)
        else:
            logger.warning(f"Unsupported table for unified dataset: {table_name}. Skipping.")
            continue
        
        # Debug: Print raw data from database 
        logger.debug(f"DEBUG: Raw data from {table_name}: {data[:2] if data else 'EMPTY'}")
        
        if not data:
            logger.debug(f"No data loaded from {table_name}")
            continue

        df_table = pd.DataFrame(data)
        df_table['date'] = pd.to_datetime(df_table['date']).dt.date # Ensure date format consistency
        logger.debug(f"Loaded {len(df_table)} rows from {table_name}")
        
        # Debug: Print DataFrame columns before rename
        logger.debug(f"DEBUG: df_table from {table_name} columns BEFORE rename: {df_table.columns.tolist()}")

        # Rename columns to their metric names if different from DB column names
        # This ensures the DataFrame uses the 'metric_x_name' as column names
        col_rename_map = {col_db: metric_name for metric_name, col_db in metrics_in_table}
        logger.debug(f"DEBUG: col_rename_map for {table_name}: {col_rename_map}")
        df_table.rename(columns=col_rename_map, inplace=True)
        
        # Debug: Print DataFrame columns after rename
        logger.debug(f"DEBUG: df_table from {table_name} columns AFTER rename: {df_table.columns.tolist()}")
        
        # Select only relevant columns before merging to avoid conflicts
        available_metrics = [metric_name for metric_name, _ in metrics_in_table if metric_name in df_table.columns]
        logger.debug(f"DEBUG: Available metrics for {table_name}: {available_metrics}")
        df_table = df_table[['date'] + available_metrics]
        logger.debug(f"DEBUG: df_table final columns for {table_name}: {df_table.columns.tolist()}")

        if unified_df.empty:
            unified_df = df_table
            logger.debug(f"DEBUG: unified_df initialized with {table_name} data, columns: {unified_df.columns.tolist()}")
        else:
            # Use outer merge to keep all dates from both sides
            logger.debug(f"DEBUG: unified_df columns BEFORE merging {table_name}: {unified_df.columns.tolist()}")
            unified_df = pd.merge(unified_df, df_table, on='date', how='outer')
            logger.debug(f"DEBUG: unified_df columns AFTER merging {table_name}: {unified_df.columns.tolist()}")

    # Ensure date column is set as index or kept consistent
    if not unified_df.empty:
        unified_df = unified_df.sort_values(by='date').reset_index(drop=True)
        # Fill NaN values with 0 for numerical columns where appropriate, or handle them during correlation
        # For now, correlation will dropna, but for display, 0 might be better.
        # unified_df = unified_df.fillna(0) # Decide if this is appropriate for all metrics

    logger.debug(f"Unified dataset created with {len(unified_df)} rows")
    logger.debug(f"Unified DataFrame columns after loading: {unified_df.columns.tolist()}")
    logger.debug(f"Unified DataFrame head:\n{unified_df.head()}")
    logger.debug(f"DEBUG: _load_unified_dataset returning DataFrame with columns: {unified_df.columns.tolist()} and shape {unified_df.shape}")
    return unified_df

def _apply_filters_to_dataframe(df, filters):
    """
    Applies a list of filters to a Pandas DataFrame.
    """
    if df.empty:
        return df

    filtered_df = df.copy() # Work on a copy

    logger.debug(f"DataFrame shape BEFORE filtering: {filtered_df.shape}")
    logger.debug(f"Unified DataFrame columns: {filtered_df.columns.tolist()}")

    initial_rows = len(filtered_df)

    for filter_metric, (operator, filter_value) in filters.items():
        logger.debug(f"Applying filter: {filter_metric} {operator} {filter_value}")
        
        # Ensure the filter_metric exists in the DataFrame
        if filter_metric not in filtered_df.columns:
            logger.warning(f"Filter metric '{filter_metric}' not found in DataFrame. Available columns: {filtered_df.columns.tolist()}. Skipping filter.")
            continue # Skip this filter, but continue with others

        # Convert filter_value to appropriate type if necessary (e.g., float, int)
        # Attempt to convert the DataFrame column to numeric, coercing errors to NaN
        # This helps in filtering mixed-type columns or columns with non-numeric strings
        col_data = pd.to_numeric(filtered_df[filter_metric], errors='coerce')

        # Handle different operators
        if operator == '>':
            filtered_df = filtered_df[col_data > filter_value]
        elif operator == '>=':
            filtered_df = filtered_df[col_data >= filter_value]
        elif operator == '<':
            filtered_df = filtered_df[col_data < filter_value]
        elif operator == '<=':
            filtered_df = filtered_df[col_data <= filter_value]
        elif operator == '==':
            filtered_df = filtered_df[col_data == filter_value]
        elif operator == '!=':
            filtered_df = filtered_df[col_data != filter_value]
        elif operator == 'between':
            if isinstance(filter_value, list) and len(filter_value) == 2:
                filtered_df = filtered_df[
                    (col_data >= filter_value[0]) &
                    (col_data <= filter_value[1])
                ]
            else:
                logger.warning(f"'between' operator for '{filter_metric}' requires a list of two values. Skipping filter.")
        elif operator == 'in':
            # Ensure filter_value is iterable (e.g., list of strings/numbers)
            if isinstance(filter_value, list):
                filtered_df = filtered_df[col_data.isin(filter_value)]
            else:
                logger.warning(f"'in' operator for '{filter_metric}' requires a list of values. Skipping filter.")
        elif operator == 'like':
            # This requires string columns and regex or string methods
            filtered_df = filtered_df[filtered_df[filter_metric].astype(str).str.contains(str(filter_value), na=False, regex=False)]
        else:
            logger.warning(f"Unknown operator '{operator}' for '{filter_metric}'. Skipping filter.")

        logger.debug(f"Filter '{filter_metric} {operator} {filter_value}' reduced dataset from {initial_rows} to {len(filtered_df)} rows")
        initial_rows = len(filtered_df) # Update for next filter in chain

    logger.debug(f"DataFrame shape AFTER filtering: {filtered_df.shape}")
    return filtered_df

def analyze_custom_correlation(user_id, metric_x_name, metric_y_name, start_date, end_date, correlation_type='pearson', filters=None):
    """
    Analyzes the correlation between two custom metrics over a date range, with optional filters.

    Args:
        user_id (int): The ID of the user.
        metric_x_name (str): The name of the first metric (e.g., 'total_steps').
        metric_y_name (str): The name of the second metric (e.g., 'sleep_score').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        correlation_type (str): Type of analysis ('pearson', 'average_comparison', 'time_based_impact').
        filters (dict, optional): A dictionary of filters to apply.
            Example: {'metric_name': ('operator', value)}
            Operators: '>', '>=', '<', '<=', '==', '!=', 'between', 'in', 'like'
            'between' value: [min, max]
            'in' value: [val1, val2, ...]
            'like' value: '%pattern%' or 'pattern%' etc.
            Defaults to None (no filters).

    Returns:
        dict: A dictionary containing analysis results (e.g., correlation coefficient, summary).
    """
    if not db.conn:
        db.connect()

    logger.info(f"Performing correlation analysis for {metric_x_name} vs {metric_y_name} from {start_date} to {end_date}")
    if filters:
        logger.info(f"Applying {len(filters)} filters to correlation analysis: {list(filters.keys())}")
        logger.info(f"Validating {len(filters)} filters for correlation analysis")
    else:
        logger.info("No filters applied to correlation analysis")

    try:
        # Load unified dataset including X, Y, and all filter metrics
        unified_df = _load_unified_dataset(user_id, start_date, end_date,
                                           metric_x_name, metric_y_name, filters, AVAILABLE_METRICS)
        
        # Apply filters if provided
        if filters:
            filtered_df = _apply_filters_to_dataframe(unified_df, filters)
        else:
            filtered_df = unified_df.copy()

        # Drop rows with NaN in either X or Y metric for correlation calculation
        # This ensures that only complete data points are used for correlation
        initial_filtered_rows = len(filtered_df)
        filtered_df = filtered_df.dropna(subset=[metric_x_name, metric_y_name])
        logger.debug(f"DataFrame shape AFTER removing nulls: {filtered_df.shape}")
        if initial_filtered_rows != len(filtered_df):
            logger.debug(f"Removed {initial_filtered_rows - len(filtered_df)} rows due to nulls in {metric_x_name} or {metric_y_name}.")


        sample_size = len(filtered_df)
        if sample_size < 2: # Pearson requires at least 2 points
            return {
                'status': 'error',
                'message': f"Insufficient data points ({sample_size}) for correlation after filtering. Need at least 2.",
                'sample_size': sample_size,
                'filters_applied': filters
            }

        x_values = filtered_df[metric_x_name]
        y_values = filtered_df[metric_y_name]

        logger.debug(f"X values shape: {x_values.shape}, Y values shape: {y_values.shape}")
        logger.debug(f"X values head:\n{x_values.head()}")
        logger.debug(f"Y values head:\n{y_values.head()}")

        results = {
            'metrics': f"{metric_x_name} vs {metric_y_name}",
            'analysis_type': correlation_type,
            'date_range': f"{start_date} to {end_date}",
            'sample_size': sample_size,
            'filters_applied': filters if filters else "No filters applied",
            'status': 'success'
        }

        if correlation_type == 'pearson':
            # Ensure data is numeric
            x_values_numeric = pd.to_numeric(x_values, errors='coerce').dropna()
            y_values_numeric = pd.to_numeric(y_values, errors='coerce').dropna()

            if len(x_values_numeric) < 2 or len(y_values_numeric) < 2:
                 return {
                    'status': 'error',
                    'message': f"Insufficient numeric data points ({len(x_values_numeric)} for X, {len(y_values_numeric)} for Y) for Pearson correlation after cleaning. Need at least 2.",
                    'sample_size': len(x_values_numeric),
                    'filters_applied': filters
                }
            
            # Align data after dropping NaNs
            common_index = x_values_numeric.index.intersection(y_values_numeric.index)
            if len(common_index) < 2:
                 return {
                    'status': 'error',
                    'message': f"Insufficient common numeric data points ({len(common_index)}) for Pearson correlation after cleaning. Need at least 2.",
                    'sample_size': len(common_index),
                    'filters_applied': filters
                }
            
            r, p_value = stats.pearsonr(x_values_numeric.loc[common_index], y_values_numeric.loc[common_index])
            
            correlation_strength = "no"
            if abs(r) >= 0.7:
                correlation_strength = "strong"
            elif abs(r) >= 0.5:
                correlation_strength = "moderate"
            elif abs(r) >= 0.3:
                correlation_strength = "weak"

            correlation_direction = "positive" if r > 0 else "negative" if r < 0 else "no"
            
            results['correlation_coefficient'] = round(r, 3)
            results['p_value'] = round(p_value, 3)
            results['description'] = (
                f"Found a {correlation_strength} {correlation_direction} correlation (r={round(r, 3)}) "
                f"between {metric_x_name} and {metric_y_name} over {sample_size} days. "
                f"{'Higher' if r > 0 else 'Lower'} {metric_x_name} tends to be associated with "
                f"{'higher' if r > 0 else 'lower'} {metric_y_name}."
            )
            results['filtered_data'] = filtered_df.to_dict(orient='records') # Return filtered data for plotting

        elif correlation_type == 'average_comparison':
            # Example: Compare Y when X is above/below median
            median_x = x_values.median()
            above_median_y = y_values[x_values > median_x].mean()
            below_median_y = y_values[x_values <= median_x].mean()
            
            results['comparison_summary'] = {
                f'Average {metric_y_name} when {metric_x_name} > {median_x}': round(above_median_y, 2),
                f'Average {metric_y_name} when {metric_x_name} <= {median_x}': round(below_median_y, 2)
            }
            results['description'] = (
                f"When {metric_x_name} was above its median ({round(median_x, 2)}), "
                f"the average {metric_y_name} was {round(above_median_y, 2)}. "
                f"When {metric_x_name} was at or below its median, "
                f"the average {metric_y_name} was {round(below_median_y, 2)}."
            )
            results['filtered_data'] = filtered_df.to_dict(orient='records') # Return filtered data for plotting

        elif correlation_type == 'time_based_impact':
            # Example: Analyze Y's average over different times of day/week based on X
            # This would require more granular data or specific time-based metrics
            # For daily data, we can look at day of week trends
            filtered_df['day_of_week'] = filtered_df['date'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d').strftime('%A'))
            avg_y_by_day = filtered_df.groupby('day_of_week')[metric_y_name].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            results['time_based_summary'] = avg_y_by_day.to_dict()
            results['description'] = (
                f"Average {metric_y_name} by day of week based on {sample_size} days: "
                f"{', '.join([f'{day}: {round(val, 2)}' for day, val in avg_y_by_day.items() if pd.notna(val)])}."
            )
            results['filtered_data'] = filtered_df.to_dict(orient='records') # Return filtered data for plotting
        else:
            results['status'] = 'error'
            results['message'] = "Unsupported correlation type."

        return results

    except Exception as e:
        logger.error(f"Failed to analyze custom correlation: {e}", exc_info=True)
        return {'status': 'error', 'message': f"Error analyzing correlation: {e}"}


def get_daily_summary_trends(user_id, days=30):
    """Retrieves recent daily summary trends for the dashboard."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days - 1)
    
    daily_data = db.get_garmin_daily_summary(user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    sleep_data = db.get_garmin_sleep(user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    food_data = db.get_food_log_daily_summary(user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    logger.debug(f"DEBUG: Raw Food Data from DB: {food_data}")

    df_daily = pd.DataFrame(daily_data)
    df_sleep = pd.DataFrame(sleep_data)
    df_food = pd.DataFrame(food_data)

    logger.debug(f"DEBUG: df_food columns: {df_food.columns.tolist()}")
    logger.debug(f"DEBUG: df_food shape: {df_food.shape}")

    # Convert 'date' columns to datetime objects for merging
    for df in [df_daily, df_sleep, df_food]:
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.date

    # Merge dataframes on 'date'
    merged_df = pd.DataFrame({'date': [start_date + timedelta(days=i) for i in range(days)]})
    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date

    logger.debug(f"DEBUG: merged_df columns before merging: {merged_df.columns.tolist()}")

    if not df_daily.empty:
        merged_df = pd.merge(merged_df, df_daily, on='date', how='left', suffixes=('', '_garmin'))
    if not df_sleep.empty:
        merged_df = pd.merge(merged_df, df_sleep, on='date', how='left', suffixes=('', '_sleep'))
    if not df_food.empty:
        merged_df = pd.merge(merged_df, df_food, on='date', how='left', suffixes=('', '_food'))

    logger.debug(f"DEBUG: merged_df columns after food merge: {merged_df.columns.tolist()}")

    # Ensure all expected columns exist, even if no data is present
    # This prevents KeyError when accessing nutrition columns
    expected_cols = ['total_steps', 'avg_daily_rhr', 'avg_daily_stress', 'sleep_duration_hours', 'sleep_score',
                     'total_calories', 'protein_g', 'carbohydrates_g', 'fat_g', 'caffeine_mg', 'alcohol_units',
                     'active_calories', 'distance_km']
    
    for col in expected_cols:
        if col not in merged_df.columns:
            merged_df[col] = 0
            logger.debug(f"DEBUG: Added missing column '{col}' with default value 0")

    # Fill NaN for numerical columns with 0 for aggregation
    for col in expected_cols:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)
    
    # Calculate averages for the period
    avg_summary = {
        'total_days': len(merged_df),
        'avg_steps': round(merged_df['total_steps'].mean(), 0),
        'avg_sleep_hours': round(merged_df['sleep_duration_hours'].mean(), 2),
        'avg_sleep_score': round(merged_df['sleep_score'].mean(), 0),
        'avg_stress': round(merged_df['avg_daily_stress'].mean(), 0),
        'avg_rhr': round(merged_df['avg_daily_rhr'].mean(), 0),
        'avg_calories': round(merged_df['total_calories'].mean(), 0),
        'avg_protein': round(merged_df['protein_g'].mean(), 1),
        'avg_carbs': round(merged_df['carbohydrates_g'].mean(), 1),
        'avg_fat': round(merged_df['fat_g'].mean(), 1),
        'avg_caffeine': round(merged_df['caffeine_mg'].mean(), 1),
        'avg_alcohol': round(merged_df['alcohol_units'].mean(), 1),
        'avg_active_calories': round(merged_df['active_calories'].mean(), 0),
        'avg_distance_km': round(merged_df['distance_km'].mean(), 1)
    }

    return avg_summary, merged_df # Return both summary and the dataframe for plotting if needed

def get_morning_report_data(user_id, report_date):
    """
    Generates a concise morning report for a specific date.
    """
    report_date_str = report_date.strftime('%Y-%m-%d')
    
    daily_summary = db.get_garmin_daily_summary(user_id, report_date_str, report_date_str)
    sleep_data = db.get_garmin_sleep(user_id, report_date_str, report_date_str)
    food_data = db.get_food_log_daily_summary(user_id, report_date_str, report_date_str)
    subjective_data = db.get_subjective_wellbeing(user_id, report_date_str, report_date_str)

    report = {
        'date': report_date_str,
        'has_data': False,
        'sleep': {},
        'activity': {},
        'stress': {},
        'nutrition': {},
        'subjective': {}
    }

    if daily_summary:
        s = daily_summary[0]
        report['activity'] = {
            'total_steps': s.get('total_steps', 0),
            'active_calories': s.get('active_calories', 0),
            'distance_km': s.get('distance_km', 0),
        }
        report['stress'] = {
            'avg_daily_stress': s.get('avg_daily_stress', 0),
            'max_daily_stress': s.get('max_daily_stress', 0),
            'min_daily_stress': s.get('min_daily_stress', 0),
            'avg_daily_rhr': s.get('avg_daily_rhr', 0)
        }
        report['has_data'] = True

    if sleep_data:
        sl = sleep_data[0]
        report['sleep'] = {
            'duration_hours': round(sl.get('sleep_duration_hours', 0), 2),
            'score': sl.get('sleep_score', 0)
        }
        report['has_data'] = True

    if food_data:
        f = food_data[0]
        report['nutrition'] = {
            'total_calories': round(f.get('total_calories', 0), 0),
            'protein_g': round(f.get('protein_g', 0), 1),
            'carbohydrates_g': round(f.get('carbohydrates_g', 0), 1),
            'fat_g': round(f.get('fat_g', 0), 1),
            'caffeine_mg': round(f.get('caffeine_mg', 0), 1),
            'alcohol_units': round(f.get('alcohol_units', 0), 1)
        }
        report['has_data'] = True
    
    if subjective_data:
        sub = subjective_data[0]
        report['subjective'] = {
            'mood': sub.get('mood'),
            'energy': sub.get('energy'),
            'stress': sub.get('stress'),
            'sleep_quality': sub.get('sleep_quality'),
            'focus': sub.get('focus'),
            'motivation': sub.get('motivation'),
            'emotional_state': sub.get('emotional_state'),
            'stress_triggers': sub.get('stress_triggers'),
            'coping_strategies': sub.get('coping_strategies'),
            'physical_symptoms': sub.get('physical_symptoms'),
            'daily_events': sub.get('daily_events'),
            'notes': sub.get('notes')
        }
        report['has_data'] = True

    return report

def analyze_consistent_high_stress_hours(user_id, date_range_days=30, stress_threshold=25):
    """
    Analyzes consistent hourly stress patterns over a date range.
    Identifies hours of the day that consistently show average stress above a threshold.
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=date_range_days - 1)
    
    stress_details_data = db.get_garmin_stress_details(user_id, 
                                                       start_date.strftime('%Y-%m-%d 00:00:00'), 
                                                       end_date.strftime('%Y-%m-%d 23:59:59'))
    
    if not stress_details_data:
        return "No granular stress data available for analysis."

    df_stress = pd.DataFrame(stress_details_data)
    df_stress['timestamp'] = pd.to_datetime(df_stress['timestamp'])
    df_stress['hour'] = df_stress['timestamp'].dt.hour
    df_stress['date'] = df_stress['timestamp'].dt.date

    # Calculate average stress per hour of day across all days
    hourly_avg_stress = df_stress.groupby('hour')['stress_level'].mean()

    # Identify hours consistently above threshold
    consistent_high_stress_hours = {}
    for hour in range(24):
        if hour in hourly_avg_stress and hourly_avg_stress[hour] > stress_threshold:
            consistent_high_stress_hours[hour] = round(hourly_avg_stress[hour], 2)

    if not consistent_high_stress_hours:
        return f"No consistent high-stress hours (avg stress > {stress_threshold}) found over the last {date_range_days} days."
    
    # Format output string
    summary = f"Over the last {date_range_days} days, stress was consistently elevated (> {stress_threshold}) during the following hours:\n"
    for hour, avg_stress in sorted(consistent_high_stress_hours.items()):
        summary += f"- {hour:02d}:00 - {hour:02d}:59 (Average Stress: {avg_stress})\n"
    
    return summary

def analyze_stress_recovery_after_activity(user_id, activity_type=None, date_range_days=30, baseline_threshold=25, min_baseline_duration_minutes=30):
    """
    Analyzes how long it takes for stress to return to a baseline after activities.
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=date_range_days - 1)

    activities = db.get_garmin_activities(user_id, 
                                          start_date.strftime('%Y-%m-%d %H:%M:%S'), 
                                          (end_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')) # Fetch slightly beyond end date

    if activity_type:
        activities = [a for a in activities if a['activity_type'].lower() == activity_type.lower()]

    if not activities:
        return f"No {'matching ' if activity_type else ''}activities found for analysis in the last {date_range_days} days."

    recovery_times = []

    for activity in activities:
        activity_end_time_dt = datetime.strptime(activity['end_time'], '%Y-%m-%d %H:%M:%S')
        
        # Fetch stress data starting from activity end time for a few hours/day
        stress_data_window = db.get_garmin_stress_details(user_id, 
                                                          activity_end_time_dt.strftime('%Y-%m-%d %H:%M:%S'), 
                                                          (activity_end_time_dt + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S')) # Look up to 6 hours after
        
        if not stress_data_window:
            logger.debug(f"No stress data found after activity {activity['activity_id']} at {activity_end_time_dt}")
            continue

        df_stress = pd.DataFrame(stress_data_window)
        df_stress['timestamp'] = pd.to_datetime(df_stress['timestamp'])
        df_stress = df_stress.sort_values('timestamp').reset_index(drop=True)

        recovery_start_index = -1
        baseline_duration_counter = 0

        for i in range(len(df_stress)):
            if df_stress.loc[i, 'timestamp'] < activity_end_time_dt:
                continue # Skip stress data before activity end

            stress_level = df_stress.loc[i, 'stress_level']
            
            if stress_level <= baseline_threshold:
                if recovery_start_index == -1:
                    recovery_start_index = i # Mark potential start of recovery
                
                # Check if we have enough consecutive baseline readings
                if i + 1 < len(df_stress):
                    time_diff_to_next = (df_stress.loc[i+1, 'timestamp'] - df_stress.loc[i, 'timestamp']).total_seconds() / 60
                    if time_diff_to_next < 2: # Assuming stress data is roughly per minute
                        baseline_duration_counter += time_diff_to_next
                    else: # Gap in data, reset counter
                        baseline_duration_counter = 0
                        recovery_start_index = -1
                else: # End of data, count current segment
                    baseline_duration_counter += 1 # Assume 1 minute for last point
            else:
                baseline_duration_counter = 0
                recovery_start_index = -1 # Reset if stress goes above threshold

            if baseline_duration_counter >= min_baseline_duration_minutes:
                time_to_recover_minutes = (df_stress.loc[recovery_start_index, 'timestamp'] - activity_end_time_dt).total_seconds() / 60
                recovery_times.append(time_to_recover_minutes)
                break # Found recovery for this activity, move to next activity

    if not recovery_times:
        return f"Could not determine stress recovery time for any {'matching ' if activity_type else ''}activities in the last {date_range_days} days (baseline < {baseline_threshold} for {min_baseline_duration_minutes} min not met)."

    avg_recovery = np.mean(recovery_times)
    min_recovery = np.min(recovery_times)
    max_recovery = np.max(recovery_times)

    summary = (
        f"Based on {len(recovery_times)} {'matching ' if activity_type else ''}activities over the last {date_range_days} days:\n"
        f"- Average stress recovery time to baseline (<{baseline_threshold}) was {round(avg_recovery, 1)} minutes.\n"
        f"- Recovery times ranged from {round(min_recovery, 1)} to {round(max_recovery, 1)} minutes."
    )
    return summary
