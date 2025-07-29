import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import io
import base64
import logging
import plotly.express as px
import plotly.graph_objects as go

from database import db # Import the global db instance

logger = logging.getLogger(__name__)

def generate_time_series_plots(user_id, days=30):
    """
    Generates time-series plots for key health metrics over a specified number of days.
    Returns a list of Plotly figures (JSON serializable).
    """
    logger.info(f"Generating time-series plots for user {user_id} over {days} days")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days - 1)
    
    try:
        # Fetch data from database
        daily_summary_data = db.get_garmin_daily_summary(user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        sleep_data = db.get_garmin_sleep(user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        food_data = db.get_food_log_daily_summary(user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        subjective_data = db.get_subjective_wellbeing(user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Debug: Print raw data from database 
        logger.debug(f"DEBUG: Raw data from db.get_garmin_daily_summary: {daily_summary_data[:2] if daily_summary_data else 'EMPTY'}")
        logger.debug(f"DEBUG: Raw data from db.get_garmin_sleep: {sleep_data[:2] if sleep_data else 'EMPTY'}")
        logger.debug(f"DEBUG: Raw data from db.get_food_log_daily_summary: {food_data[:2] if food_data else 'EMPTY'}")
        logger.debug(f"DEBUG: Raw data from db.get_subjective_wellbeing: {subjective_data[:2] if subjective_data else 'EMPTY'}")

        # Create DataFrames
        df_daily = pd.DataFrame(daily_summary_data)
        df_sleep = pd.DataFrame(sleep_data)
        df_food = pd.DataFrame(food_data)
        df_subjective = pd.DataFrame(subjective_data)

        # Convert 'date' columns to datetime objects
        for df in [df_daily, df_sleep, df_food, df_subjective]:
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

        # Create a full date range to ensure all days are present, even if no data
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Merge all dataframes onto the full date range
        # Use outer merge initially to ensure all dates are kept, then fillna for plotting
        merged_df = pd.DataFrame(index=full_date_range)
        logger.debug(f"DEBUG: health_visualizations: Initial merged_df columns: {merged_df.columns.tolist()}")
        
        # Track which columns are available after merging
        available_columns = set()
        
        if not df_daily.empty:
            daily_columns = ['total_steps', 'avg_daily_rhr', 'avg_daily_stress', 'active_calories', 'distance_km']
            # Only merge columns that actually exist in df_daily
            existing_daily_columns = [col for col in daily_columns if col in df_daily.columns]
            if existing_daily_columns:
                merged_df = merged_df.merge(df_daily[existing_daily_columns], left_index=True, right_index=True, how='left')
                available_columns.update(existing_daily_columns)
                logger.debug(f"DEBUG: health_visualizations: merged_df columns after {existing_daily_columns} merge: {merged_df.columns.tolist()}")
        
        if not df_sleep.empty:
            sleep_columns = ['sleep_duration_hours', 'sleep_score']
            existing_sleep_columns = [col for col in sleep_columns if col in df_sleep.columns]
            if existing_sleep_columns:
                merged_df = merged_df.merge(df_sleep[existing_sleep_columns], left_index=True, right_index=True, how='left')
                available_columns.update(existing_sleep_columns)
                logger.debug(f"DEBUG: health_visualizations: merged_df columns after {existing_sleep_columns} merge: {merged_df.columns.tolist()}")
        
        if not df_food.empty:
            food_columns = ['total_calories', 'protein_g', 'carbohydrates_g', 'fat_g', 'caffeine_mg', 'alcohol_units']
            existing_food_columns = [col for col in food_columns if col in df_food.columns]
            if existing_food_columns:
                merged_df = merged_df.merge(df_food[existing_food_columns], left_index=True, right_index=True, how='left')
                available_columns.update(existing_food_columns)
                logger.debug(f"DEBUG: health_visualizations: merged_df columns after {existing_food_columns} merge: {merged_df.columns.tolist()}")
        
        if not df_subjective.empty:
            subjective_columns = ['mood', 'energy', 'stress']
            existing_subjective_columns = [col for col in subjective_columns if col in df_subjective.columns]
            if existing_subjective_columns:
                # Rename 'stress' to 'subjective_stress' to avoid conflict with Garmin stress
                df_subjective_renamed = df_subjective[existing_subjective_columns].copy()
                if 'stress' in df_subjective_renamed.columns:
                    df_subjective_renamed = df_subjective_renamed.rename(columns={'stress': 'subjective_stress'})
                    available_columns.add('subjective_stress')
                    available_columns.update([col for col in existing_subjective_columns if col != 'stress'])
                else:
                    available_columns.update(existing_subjective_columns)
                merged_df = merged_df.merge(df_subjective_renamed, left_index=True, right_index=True, how='left')
                logger.debug(f"DEBUG: health_visualizations: merged_df columns after {existing_subjective_columns} merge: {merged_df.columns.tolist()}")

        logger.info(f"Available columns after merging: {sorted(available_columns)}")
        logger.debug(f"DEBUG: health_visualizations: Final merged_df shape: {merged_df.shape}, columns: {merged_df.columns.tolist()}")
        
        plots = []

        # List of metrics to plot
        metrics_to_plot = {
            'Activity': ['total_steps', 'active_calories', 'distance_km'],
            'Sleep': ['sleep_duration_hours', 'sleep_score'],
            'Stress & RHR': ['avg_daily_stress', 'avg_daily_rhr'],
            'Nutrition (Calories)': ['total_calories'],
            'Nutrition (Macros)': ['protein_g', 'carbohydrates_g', 'fat_g'],
            'Stimulants': ['caffeine_mg', 'alcohol_units'],
            'Subjective Wellbeing': ['mood', 'energy', 'subjective_stress']
        }

        for category, metrics in metrics_to_plot.items():
            # Filter metrics to only those actually present in merged_df
            existing_metrics = [m for m in metrics if m in available_columns]
            
            if not existing_metrics:
                logger.info(f"Skipping {category} plot: No relevant data columns found. Needed: {metrics}, Available: {sorted(available_columns)}")
                continue
            
            df_plot = merged_df[existing_metrics].copy()
            df_plot.index.name = 'Date'
            df_plot = df_plot.reset_index()
            
            # Melt for easier plotting with Plotly Express
            df_melted = df_plot.melt(id_vars=['Date'], var_name='Metric', value_name='Value')
            
            # Filter out rows where Value is NaN for cleaner plots
            df_melted = df_melted.dropna(subset=['Value'])

            if not df_melted.empty:
                fig = px.line(df_melted, x='Date', y='Value', color='Metric',
                              title=f'{category} Trends (Last {days} Days)',
                              labels={'Value': 'Value', 'Date': 'Date'},
                              hover_data={'Value': True}) # Show value on hover
                fig.update_layout(hovermode="x unified") # Nice hover effect
                plots.append(fig) # Return Plotly figure object directly for Gradio
            else:
                logger.info(f"No data for {category} trends in the last {days} days.")

        logger.info(f"Generated {len(plots)} time-series plots")
        return plots
    except Exception as e:
        logger.error(f"Error generating time-series plots: {e}", exc_info=True)
        return [] # Return empty list on error

def generate_correlation_plot(df, metric_x_name, metric_y_name, correlation_type):
    """
    Generates a Plotly figure for correlation visualization.
    Returns a Plotly figure (JSON serializable).
    """
    if df.empty or metric_x_name not in df.columns or metric_y_name not in df.columns:
        logger.warning(f"Cannot generate plot: DataFrame is empty or missing columns ({metric_x_name}, {metric_y_name}).")
        return None

    df_plot = df.copy()
    # Ensure columns are numeric for plotting
    df_plot[metric_x_name] = pd.to_numeric(df_plot[metric_x_name], errors='coerce')
    df_plot[metric_y_name] = pd.to_numeric(df_plot[metric_y_name], errors='coerce')
    df_plot.dropna(subset=[metric_x_name, metric_y_name], inplace=True)

    if df_plot.empty:
        logger.warning(f"No valid numeric data points for plotting after cleaning for {metric_x_name} vs {metric_y_name}.")
        return None

    if correlation_type == 'pearson':
        fig = px.scatter(df_plot, x=metric_x_name, y=metric_y_name,
                         title=f'Scatter Plot: {metric_x_name} vs {metric_y_name}',
                         labels={metric_x_name: metric_x_name.replace('_', ' ').title(),
                                 metric_y_name: metric_y_name.replace('_', ' ').title()},
                         hover_data=['date'])
        fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(template="plotly_dark", hovermode="closest")
        logger.info(f"Creating Pearson scatter plot for {len(df_plot)} data points")
        return fig

    elif correlation_type == 'average_comparison':
        # This assumes the 'comparison_summary' from trend_analyzer is structured for plotting
        # For simplicity, let's assume X is categorical or binned for comparison
        # Here we'll just plot the averages if df_plot has 'category' and 'value'
        # This needs to be aligned with how trend_analyzer returns data for this type
        # For now, let's just create a simple bar chart based on a binning of X
        
        # Example: Bin X metric into 'Low', 'Medium', 'High'
        bins = pd.qcut(df_plot[metric_x_name], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
        df_plot['X_Category'] = bins
        
        if 'X_Category' not in df_plot.columns:
            logger.warning("Cannot create average comparison plot: X_Category not generated.")
            return None

        avg_y_by_x_category = df_plot.groupby('X_Category')[metric_y_name].mean().reset_index()
        
        fig = px.bar(avg_y_by_x_category, x='X_Category', y=metric_y_name,
                     title=f'Average {metric_y_name} by {metric_x_name} Category',
                     labels={'X_Category': f'{metric_x_name} Category', metric_y_name: f'Average {metric_y_name}'},
                     color=metric_y_name, color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(template="plotly_dark")
        logger.info(f"Creating average_comparison plot with {len(df_plot)} data points")
        return fig

    elif correlation_type == 'time_based_impact':
        # Assuming df_plot has 'date' and 'day_of_week' from trend_analyzer
        if 'day_of_week' not in df_plot.columns:
            df_plot['day_of_week'] = pd.to_datetime(df_plot['date']).dt.day_name()
        
        avg_y_by_day = df_plot.groupby('day_of_week')[metric_y_name].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]).reset_index()
        
        fig = px.bar(avg_y_by_day, x='day_of_week', y=metric_y_name,
                     title=f'Average {metric_y_name} by Day of Week',
                     labels={'day_of_week': 'Day of Week', metric_y_name: f'Average {metric_y_name}'},
                     color=metric_y_name, color_continuous_scale=px.colors.sequential.Plasma)
        fig.update_layout(template="plotly_dark")
        logger.info(f"Creating time_based_impact plot with {len(df_plot)} data points")
        return fig

    return None

def generate_hourly_stress_plot(user_id, days=30):
    """
    Generates a bar chart of average stress level by hour of day.
    Returns a Plotly figure (JSON serializable).
    """
    logger.info(f"Generating hourly stress plot for user {user_id} over {days} days")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days - 1)

    try:
        stress_details_data = db.get_garmin_stress_details(user_id, 
                                                           start_date.strftime('%Y-%m-%d 00:00:00'), 
                                                           end_date.strftime('%Y-%m-%d 23:59:59'))
        
        if not stress_details_data:
            logger.info("No granular stress data available for hourly plot.")
            return None

        df_stress = pd.DataFrame(stress_details_data)
        df_stress['timestamp'] = pd.to_datetime(df_stress['timestamp'])
        df_stress['hour'] = df_stress['timestamp'].dt.hour

        hourly_avg_stress = df_stress.groupby('hour')['stress_level'].mean().reset_index()
        hourly_avg_stress['hour_label'] = hourly_avg_stress['hour'].apply(lambda x: f"{x:02d}:00")

        fig = px.bar(hourly_avg_stress, x='hour_label', y='stress_level',
                     title=f'Average Stress Level by Hour of Day (Last {days} Days)',
                     labels={'hour_label': 'Hour of Day', 'stress_level': 'Average Stress Level'},
                     color='stress_level', color_continuous_scale=px.colors.sequential.Plasma)
        
        fig.update_layout(xaxis={'categoryorder':'category ascending'}, template="plotly_dark")
        logger.info(f"Generated hourly stress plot with {len(hourly_avg_stress)} hours.")
        return fig

    except Exception as e:
        logger.error(f"Error generating hourly stress plot: {e}", exc_info=True)
        return None

