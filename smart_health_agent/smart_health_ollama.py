import gradio as gr
import pandas as pd
import json
from datetime import datetime, timedelta, date
import os
import asyncio
import logging
from dotenv import load_dotenv

# Import your modules
from database import db
from garmin_utils import sync_garmin_data, initialize_garmin_client # Import initialize_garmin_client
from cronometer_parser import parse_cronometer_food_entries_csv
import trend_analyzer
import health_visualizations # For graphs
# import llm_interface # Future LLM integration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Global State & Initialization ---
current_user_id = 1 # Default user ID
db_initialized = False
garmin_sync_status = "Not synced"
cronometer_upload_status = "No data uploaded"

# Define all available metrics for UI dropdowns and backend mapping
# These names must match the column names in the unified DataFrame produced by trend_analyzer._load_unified_dataset
ALL_AVAILABLE_METRICS = [
    # Garmin Daily Summary
    'total_steps', 'avg_daily_rhr', 'avg_daily_stress', 'max_daily_stress',
    'min_daily_stress', 'active_calories', 'distance_km',
    # Garmin Sleep
    'sleep_duration_hours', 'sleep_score',
    # Food Log Daily
    'total_calories', 'protein_g', 'carbohydrates_g', 'fat_g',
    'caffeine_mg', 'alcohol_units',
    # Subjective Wellbeing - Enhanced with all new fields
    'mood', 'energy', 'subjective_stress', 'sleep_quality', 'focus', 'motivation',
    'emotional_state', 'stress_triggers', 'coping_strategies', 'physical_symptoms', 'daily_events'
]

# --- Backend Functions (called by Gradio UI) ---

async def initialize_app_state():
    """Initializes database and sets up background sync."""
    global db_initialized, garmin_sync_status
    if not db_initialized:
        try:
            db.connect()
            db.create_tables()
            db_initialized = True
            logger.info("Database initialized successfully")
            # Attempt to initialize Garmin client on startup, but don't block
            await initialize_garmin_client_and_sync_status()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return "❌ Database initialization failed"
    return "✅ Database initialized"

async def initialize_garmin_client_and_sync_status():
    """Initializes Garmin client and updates sync status in background."""
    global garmin_sync_status
    logger.debug("Starting background Garmin sync...")
    try:
        # initialize_garmin_client() # This is now called inside sync_garmin_data
        # We don't need to call initialize_garmin_client directly here, sync_garmin_data handles it.
        # Just update initial status from DB
        last_sync = db.get_sync_status(current_user_id, 'garmin')
        if last_sync:
            garmin_sync_status = f"Last synced: {last_sync}"
        else:
            garmin_sync_status = "Garmin configured - ready to sync"
        logger.debug("Final startup_sync_status: %s", garmin_sync_status)
    except Exception as e:
        logger.error(f"Error during startup Garmin client init or status fetch: {e}")
        garmin_sync_status = "Garmin configured - error"

def get_app_status():
    """Returns the current status of the database and Garmin sync."""
    status_text = f"Status: {'✅ Database initialized' if db_initialized else '❌ Database not initialized'} | "
    status_text += f"Garmin sync: {garmin_sync_status}"
    return status_text

async def handle_garmin_sync(force_refresh_checkbox):
    """Handles Garmin data synchronization."""
    global garmin_sync_status
    logger.info("🔄 Garmin sync started in background...")
    garmin_sync_status = "🔄 Syncing..."
    yield get_app_status(), garmin_sync_status # Update UI immediately

    try:
        success = await sync_garmin_data(current_user_id, days_to_sync=30, force_refresh=force_refresh_checkbox)
        if success:
            garmin_sync_status = "✅ Garmin sync completed successfully"
        else:
            garmin_sync_status = "❌ Garmin sync failed"
    except Exception as e:
        logger.error(f"Error during Garmin sync: {e}")
        garmin_sync_status = f"❌ Garmin sync failed: {e}"
    finally:
        logger.info("Garmin sync completed")
        yield get_app_status(), garmin_sync_status # Final update

def handle_cronometer_upload(file_obj):
    """Handles Cronometer CSV file upload and parsing."""
    if file_obj is None:
        # Only update the import status, do not touch the food log summary
        return "❌ No file selected.", gr.update(), None  # gr.update() prevents update

    file_path = file_obj.name
    logger.info(f"Attempting to parse Cronometer CSV: {file_path}")
    
    try:
        # Delete existing food log data before re-uploading to prevent duplicates
        # This is a simple approach for now, a more sophisticated merge could be done
        # db.delete_all_food_log_data(current_user_id) # Need to implement this in database.py if desired
        
        parsed_food, parsed_supplements = parse_cronometer_food_entries_csv(file_path, current_user_id)
        
        if parsed_food + parsed_supplements > 0:
            status_message = f"✅ Successfully parsed {parsed_food + parsed_supplements} entries. Imported {parsed_food} food entries, {parsed_supplements} supplements."
            # Update Cronometer last upload status in DB (handled by cronometer_parser)
        else:
            status_message = "❌ Failed to parse any entries. Check console for errors."
        
        # Refresh food log summary
        food_summary_text, _ = get_food_log_summary()

        # Return the status, food summary, AND clear the file upload component
        return status_message, food_summary_text, None

    except Exception as e:
        logger.error(f"Error processing Cronometer upload: {e}", exc_info=True)
        return f"❌ Error processing file: {e}", "", None

def get_food_log_summary():
    """Retrieves and formats the food log summary for the dashboard."""
    today = date.today()
    last_7_days = today - timedelta(days=6)
    last_30_days = today - timedelta(days=29)

    # Fetch daily aggregated food data
    df_food = pd.DataFrame(db.get_food_log_daily_summary(current_user_id, last_30_days.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')))
    
    if df_food.empty:
        return "Food Log Summary (Last 30 days): No data available.", ""

    # Ensure numerical columns are numeric, coercing errors
    for col in ['total_calories', 'protein_g', 'carbohydrates_g', 'fat_g', 'caffeine_mg', 'alcohol_units']:
        if col in df_food.columns:
            df_food[col] = pd.to_numeric(df_food[col], errors='coerce').fillna(0)

    total_entries = len(df_food)
    avg_daily_calories = round(df_food['total_calories'].mean(), 0)
    total_caffeine = round(df_food['caffeine_mg'].sum(), 0)
    total_alcohol_units = round(df_food['alcohol_units'].sum(), 1)
    
    # Count days with data
    days_with_data = df_food['date'].nunique()
    latest_entry_date = df_food['date'].max() if not df_food.empty else "N/A"

    summary_text = (
        f"Food Log Summary (Last {total_entries} days): "
        f"Total Entries: {total_entries} 🥗 | "
        f"Avg Daily Calories: {avg_daily_calories} kcal 🔥 | "
        f"Total Caffeine: {total_caffeine} mg ☕ | "
        f"Total Alcohol: {total_alcohol_units} Units 🍺 | "
        f"Days with Data: {days_with_data}/{total_entries} 📅 | "
        f"Latest Entry: {latest_entry_date}"
    )
    return summary_text, df_food.to_dict(orient='records') # Return dict for potential display of raw data

def get_recent_food_entries_display():
    """Retrieves and formats recent individual food log entries for display."""
    today = date.today()
    # Fetch entries for the last 30 days, or adjust as needed
    start_date = today - timedelta(days=29) 

    entries = db.get_food_log_entries(current_user_id, start_date.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))

    if not entries:
        return "No recent food log entries found."

    # Create a DataFrame for easier formatting and sorting
    df_entries = pd.DataFrame(entries)
    df_entries['date'] = pd.to_datetime(df_entries['date'])
    df_entries = df_entries.sort_values(by=['date', 'item_name'], ascending=[False, True])

    # Select and format relevant columns for display
    display_cols = ['date', 'item_name', 'amount', 'units', 'calories', 'protein_g', 'carbohydrates_g', 'fat_g']
    df_display = df_entries[display_cols].copy()

    # Format numerical columns
    df_display['calories'] = df_display['calories'].round(0).astype(int)
    df_display['protein_g'] = df_display['protein_g'].round(1)
    df_display['carbohydrates_g'] = df_display['carbohydrates_g'].round(1)
    df_display['fat_g'] = df_display['fat_g'].round(1)
    df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d') # Format date back to string

    # Convert DataFrame to Markdown table
    markdown_table = "### Recent Food Entries\n\n"
    markdown_table += df_display.to_markdown(index=False)

    return markdown_table

def get_morning_report_display(report_date=None):
    """Generates the morning report for display."""
    if report_date is None:
        report_date = date.today() - timedelta(days=1) # Default to yesterday
    elif isinstance(report_date, str):
        try:
            report_date = datetime.strptime(report_date, '%Y-%m-%d').date()
        except ValueError:
            report_date = date.today() - timedelta(days=1) # Fallback to yesterday

    report = trend_analyzer.get_morning_report_data(current_user_id, report_date)
    
    if not report['has_data']:
        return f"Morning Report for {report_date.strftime('%Y-%m-%d')}:\nNo data available for yesterday.", ""

    report_str = f"**Morning Report for {report['date']}**\n\n"
    
    if report['sleep']:
        report_str += f"😴 **Sleep:** {report['sleep']['duration_hours']} hours, Score: {report['sleep']['score']}\n"
    
    if report['activity']:
        report_str += f"🏃‍♂️ **Activity:** {report['activity']['total_steps']} steps, {report['activity']['active_calories']} active kcal, {report['activity']['distance_km']} km\n"
    
    if report['stress']:
        report_str += f"🧘‍♀️ **Stress:** Avg: {report['stress']['avg_daily_stress']}, Max: {report['stress']['max_daily_stress']}, Min: {report['stress']['min_daily_stress']}, RHR: {report['stress']['avg_daily_rhr']}\n"
    
    if report['nutrition']:
        report_str += f"🍏 **Nutrition:** {report['nutrition']['total_calories']} kcal, P:{report['nutrition']['protein_g']}g, C:{report['nutrition']['carbohydrates_g']}g, F:{report['nutrition']['fat_g']}g\n"
        report_str += f"☕ **Stimulants:** Caffeine: {report['nutrition']['caffeine_mg']}mg, Alcohol: {report['nutrition']['alcohol_units']} units\n"
    
    if report['subjective']:
        sub = report['subjective']
        # Core wellbeing metrics (1-10 scale)
        core_metrics = []
        if sub.get('mood'): core_metrics.append(f"Mood: {sub['mood']}")
        if sub.get('energy'): core_metrics.append(f"Energy: {sub['energy']}")
        if sub.get('stress'): core_metrics.append(f"Stress: {sub['stress']}")
        if sub.get('sleep_quality'): core_metrics.append(f"Sleep Quality: {sub['sleep_quality']}")
        if sub.get('focus'): core_metrics.append(f"Focus: {sub['focus']}")
        if sub.get('motivation'): core_metrics.append(f"Motivation: {sub['motivation']}")
        
        if core_metrics:
            report_str += f"😊 **Wellbeing (1-10):** {', '.join(core_metrics)}\n"
        
        # Detailed text fields
        if sub.get('emotional_state'):
            report_str += f"💭 **Emotional State:** {sub['emotional_state']}\n"
        if sub.get('stress_triggers'):
            report_str += f"⚡ **Stress Triggers:** {sub['stress_triggers']}\n"
        if sub.get('coping_strategies'):
            report_str += f"🛡️ **Coping Strategies:** {sub['coping_strategies']}\n"
        if sub.get('physical_symptoms'):
            report_str += f"🏥 **Physical Symptoms:** {sub['physical_symptoms']}\n"
        if sub.get('daily_events'):
            report_str += f"📅 **Daily Events:** {sub['daily_events']}\n"
        if sub.get('notes'):
            report_str += f"📝 **Additional Notes:** {sub['notes']}\n"

    return report_str, json.dumps(report, indent=2) # Return JSON for debugging/details

def get_recent_trends_display(days=30):
    """Generates a summary of recent trends."""
    avg_summary, _ = trend_analyzer.get_daily_summary_trends(current_user_id, days)
    
    if avg_summary['total_days'] == 0:
        return f"No recent trend data available for the last {days} days."

    trend_str = f"**Recent Trends (Last {avg_summary['total_days']} Days)**\n\n"
    trend_str += f"- Avg Steps: {avg_summary['avg_steps']}\n"
    trend_str += f"- Avg Sleep: {avg_summary['avg_sleep_hours']} hours (Score: {avg_summary['avg_sleep_score']})\n"
    trend_str += f"- Avg Stress: {avg_summary['avg_stress']} (RHR: {avg_summary['avg_rhr']})\n"
    trend_str += f"- Avg Calories: {avg_summary['avg_calories']} kcal\n"
    trend_str += f"- Avg Protein: {avg_summary['avg_protein']}g, Carbs: {avg_summary['avg_carbs']}g, Fat: {avg_summary['avg_fat']}g\n"
    trend_str += f"- Avg Caffeine: {avg_summary['avg_caffeine']}mg, Alcohol: {avg_summary['avg_alcohol']} units\n"
    trend_str += f"- Avg Active Calories: {avg_summary['avg_active_calories']} kcal, Distance: {avg_summary['avg_distance_km']} km\n"
    
    return trend_str

def submit_mood_entry(mood, energy, stress, sleep_quality, focus, motivation, 
                     emotional_state, stress_triggers, coping_strategies, 
                     physical_symptoms, daily_events, notes, entry_date):
    """Submits an enhanced subjective wellbeing entry with all new fields."""
    try:
        if isinstance(entry_date, str):
            entry_date_str = entry_date
        else:
            entry_date_str = entry_date.strftime('%Y-%m-%d')
        
        db.upsert_subjective_wellbeing(
            current_user_id, entry_date_str, mood, energy, stress, 
            sleep_quality, focus, motivation, emotional_state, stress_triggers, 
            coping_strategies, physical_symptoms, daily_events, notes
        )
        return f"✅ Enhanced wellbeing entry for {entry_date_str} saved successfully!"
    except Exception as e:
        return f"❌ Error saving wellbeing entry: {e}"

# --- Custom Analysis Functions ---

# This gr.State will hold the list of active filters
active_filters_state = gr.State([])

def render_active_filters_ui(filters_list):
    """
    Renders the UI for the active filters list.
    """
    if not filters_list:
        return gr.Markdown("No active filters.")

    # Create a list of Gradio components for each filter row
    filter_rows = []
    # Header Row
    filter_rows.append(gr.Row(
        gr.Markdown("**Metric**"),
        gr.Markdown("**Operator**"),
        gr.Markdown("**Value**"),
    ))

    for i, filter_dict in enumerate(filters_list):
        filter_rows.append(gr.Row(
            gr.Markdown(f"{filter_dict['metric']}"),
            gr.Markdown(f"{filter_dict['operator']}"),
            gr.Markdown(f"{filter_dict['value']}"),
        ))
    
    return gr.Column(filter_rows)


def add_filter_logic(metric, operator, value, current_filters_list):
    """
    Adds a new filter to the active filters list.
    """
    if not metric or not operator or value is None or value == '':
        return current_filters_list, render_active_filters_ui(current_filters_list), "❌ Please fill all filter fields."

    try:
        # Convert value to appropriate type based on operator
        if operator in ['>', '>=', '<', '<=', '==', '!=']:
            processed_value = float(value)
        elif operator == 'between':
            values = [float(v.strip()) for v in str(value).split(',') if v.strip()]
            if len(values) != 2:
                return current_filters_list, render_active_filters_ui(current_filters_list), "❌ 'between' requires two comma-separated numeric values (e.g., '10,20')."
            processed_value = sorted(values) # Ensure min, max order
        elif operator == 'in':
            processed_value = [v.strip() for v in str(value).split(',') if v.strip()]
            if not processed_value:
                return current_filters_list, render_active_filters_ui(current_filters_list), "❌ 'in' requires at least one comma-separated value."
        elif operator == 'like':
            processed_value = str(value).strip()
            if not processed_value:
                return current_filters_list, render_active_filters_ui(current_filters_list), "❌ 'like' requires a non-empty value."
        else:
            processed_value = value # For other types, keep as is

    except ValueError:
        return current_filters_list, render_active_filters_ui(current_filters_list), "❌ Invalid value type for selected operator."

    new_filter = {'metric': metric, 'operator': operator, 'value': processed_value}

    # Deduplication: Check if an identical filter already exists
    if new_filter in current_filters_list:
        return current_filters_list, render_active_filters_ui(current_filters_list), "⚠️ Filter already exists!"
    
    new_filters_list = current_filters_list + [new_filter]
    return new_filters_list, render_active_filters_ui(new_filters_list), "✅ Filter added!"

def clear_all_filters(current_filters_list):
    """Clears all active filters."""
    new_filters_list = []
    # Return the updated state and the re-rendered UI for the filters
    return new_filters_list, render_active_filters_ui(new_filters_list), "✅ All filters cleared!"

def perform_custom_analysis(metric_x, metric_y, analysis_type, start_date_str, end_date_str, current_filters_list):
    """
    Performs custom correlation analysis and generates results and a plot.
    """
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        return "❌ Invalid date format. Please use YYYY-MM-DD.", None

    # Convert filters_list (from Gradio state) into the dictionary format expected by trend_analyzer
    filters_dict = {f['metric']: (f['operator'], f['value']) for f in current_filters_list} if current_filters_list else None

    analysis_results = trend_analyzer.analyze_custom_correlation(
        current_user_id, metric_x, metric_y,
        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'),
        correlation_type=analysis_type, filters=filters_dict
    )

    if analysis_results['status'] == 'error':
        return f"❌ Analysis error: {analysis_results['message']}", None

    # Format results for display
    result_str = f"### Correlation Analysis Results\n\n"
    result_str += f"**Metrics:** {analysis_results['metrics']}\n"
    result_str += f"**Analysis Type:** {analysis_results['analysis_type'].replace('_', ' ').title()}\n"
    result_str += f"**Date Range:** {analysis_results['date_range']}\n"
    result_str += f"**Sample Size:** {analysis_results['sample_size']} data points\n"
    
    if analysis_results['filters_applied'] and analysis_results['filters_applied'] != "No filters applied":
        result_str += f"**✅ FILTERS APPLIED:** {len(analysis_results['filters_applied'])} filter(s) - Data filtered before analysis\n"
        # Add a check here: Ensure it's a dict before iterating
        if isinstance(analysis_results['filters_applied'], dict):
            for metric, (op, val) in analysis_results['filters_applied'].items():
                result_str += f"  • **{metric}** {op} **{val}**\n"
            result_str += f"  _Applied {len(analysis_results['filters_applied'])} filter(s) to data before correlation analysis_\n"
        else:
            # This case should ideally not happen if logic is perfect, but for robustness
            result_str += f"  _Filters applied: {analysis_results['filters_applied']}_\n"  # Display the string directly
    else:
        result_str += "**No filters applied** - Analysis includes all available data\n"

    result_str += "\n**Results:**\n"
    result_str += analysis_results['description'] + "\n"
    if 'correlation_coefficient' in analysis_results:
        result_str += f"**Correlation Coefficient:** {analysis_results['correlation_coefficient']} ({analysis_results['description'].split('(')[0].strip()})\n"
    
    # Generate plot
    plot_figure = None
    if 'filtered_data' in analysis_results and analysis_results['sample_size'] >= 2:
        df_plot = pd.DataFrame(analysis_results['filtered_data'])
        plot_figure = health_visualizations.generate_correlation_plot(df_plot, metric_x, metric_y, analysis_type)
        if plot_figure:
            logger.info(f"Plot created successfully, returning figure")
        else:
            logger.warning("Failed to generate plot for correlation analysis.")

    return result_str, plot_figure

# --- Gradio UI Definition ---

with gr.Blocks(theme=gr.themes.Soft(), title="Smart Health Agent") as demo:
    gr.Markdown("# Smart Health Agent")
    gr.Markdown("### Enhanced Version: Complete health analysis with tabbed interface")

    # App Status Display
    app_status_text = gr.Markdown(get_app_status())

    # Initialize database and Garmin client on app startup
    demo.load(initialize_app_state, outputs=app_status_text)

    with gr.Tabs():
        with gr.TabItem("Dashboard"):
            with gr.Column():
                gr.Markdown("## Morning Report & Health Overview")
                
                with gr.Row():
                    morning_report_date_picker = gr.Textbox(label="Morning Report for (YYYY-MM-DD)", value=(date.today() - timedelta(days=1)).strftime('%Y-%m-%d'))
                    refresh_morning_report_btn = gr.Button("🔄 Refresh Morning Report")
                
                morning_report_output = gr.Markdown("No data available for yesterday.")
                morning_report_json_output = gr.Textbox(label="Morning Report (JSON for Debug)", interactive=False, visible=False)

                refresh_morning_report_btn.click(
                    get_morning_report_display,
                    inputs=[morning_report_date_picker],
                    outputs=[morning_report_output, morning_report_json_output]
                )
                morning_report_date_picker.change(
                    get_morning_report_display,
                    inputs=[morning_report_date_picker],
                    outputs=[morning_report_output, morning_report_json_output]
                )

                gr.Markdown("---")
                gr.Markdown("## Recent Trends")
                refresh_trends_btn = gr.Button("🔄 Refresh Trends")
                recent_trends_output = gr.Markdown("No recent data available for trends.")
                refresh_trends_btn.click(
                    get_recent_trends_display,
                    inputs=[],
                    outputs=[recent_trends_output]
                )

                gr.Markdown("---")
                gr.Markdown("## Data Sync")
                with gr.Row():
                    sync_garmin_btn = gr.Button("🔄 Sync Garmin Data")
                    force_refresh_checkbox = gr.Checkbox(label="Force refresh existing data", value=False)
                    garmin_sync_status_text = gr.Textbox(label="Sync Status", interactive=False, value=garmin_sync_status)
                
                sync_garmin_btn.click(
                    handle_garmin_sync,
                    inputs=[force_refresh_checkbox],
                    outputs=[app_status_text, garmin_sync_status_text] # Update overall app status and specific sync status
                )
                
        with gr.TabItem("Nutrition Data"):
            gr.Markdown("## Cronometer CSV Upload & Food Log Management")
            gr.Markdown("""
            **Instructions:**
            1. Export your data from Cronometer as 'Food & Recipe Entries' CSV
            2. Upload the CSV file below
            3. Click 'Import Data' to process and store in database
            
            **Supported data:** Food entries, nutrition facts, caffeine estimation, alcohol tracking
            """)
            with gr.Row():
                cronometer_file_upload = gr.File(label="Select Cronometer CSV File", file_types=[".csv"])
                import_status_output = gr.Textbox(label="Import Status", interactive=False)
            
            import_cronometer_btn = gr.Button("Import Data")
            food_log_summary_output = gr.Markdown("Food Log Summary: No data uploaded yet.")

            import_cronometer_btn.click(
                handle_cronometer_upload,
                inputs=[cronometer_file_upload],
                outputs=[import_status_output, food_log_summary_output, cronometer_file_upload]
            )
            
            gr.Markdown("## Food Log Analysis")
            with gr.Row():
                view_recent_entries_btn = gr.Button("View Recent Entries")
                export_food_data_btn = gr.Button("Export Food Data")
            
            # New component to display recent entries
            recent_food_entries_output = gr.Markdown("Click 'View Recent Entries' to see your detailed food log.")

            view_recent_entries_btn.click(
                get_recent_food_entries_display,
                inputs=[],
                outputs=[recent_food_entries_output]
            )
            # Placeholder for future functionality:
            # export_food_data_btn.click(...)


        with gr.TabItem("Graphs"):
            gr.Markdown("## Health Visualizations")
            with gr.Row():
                graph_days_slider = gr.Slider(minimum=7, maximum=180, value=30, step=1, label="Days to Display")
                refresh_graphs_btn = gr.Button("🔄 Refresh Graphs")
            
            # Create a fixed number of gr.Plot components (max expected: 7 time-series + 1 hourly = 8 plots)
            plot_outputs = []
            with gr.Column():
                for i in range(8):
                    plot_outputs.append(gr.Plot(label=f"Health Metric {i+1}"))

            def refresh_all_graphs(days):
                plots_fig_objects = health_visualizations.generate_time_series_plots(current_user_id, days)
                hourly_stress_fig_object = health_visualizations.generate_hourly_stress_plot(current_user_id, days)
                
                all_plots_to_return = []
                if plots_fig_objects:
                    all_plots_to_return.extend(plots_fig_objects)
                if hourly_stress_fig_object:
                    all_plots_to_return.append(hourly_stress_fig_object)
                
                # Pad with None to match the number of plot_outputs (8 total)
                while len(all_plots_to_return) < 8:
                    all_plots_to_return.append(None)
                
                return all_plots_to_return[:8]  # Return exactly 8 items

            refresh_graphs_btn.click(
                refresh_all_graphs,
                inputs=[graph_days_slider],
                outputs=plot_outputs
            )
            graph_days_slider.change(
                refresh_all_graphs,
                inputs=[graph_days_slider],
                outputs=plot_outputs
            )
            # Initial load of graphs
            demo.load(lambda: refresh_all_graphs(30), outputs=plot_outputs)


        with gr.TabItem("Daily Mood Tracker"):
            gr.Markdown("## Daily Mood & Wellbeing Tracker")
            with gr.Column():
                mood_date_picker = gr.Textbox(label="Date (YYYY-MM-DD)", value=date.today().strftime('%Y-%m-%d'))
                
                gr.Markdown("### Core Wellbeing Metrics (1-10 Scale)")
                with gr.Row():
                    mood_slider = gr.Slider(minimum=1, maximum=10, step=1, label="Mood (1=Terrible, 10=Excellent)", value=5)
                    energy_slider = gr.Slider(minimum=1, maximum=10, step=1, label="Energy (1=Exhausted, 10=Energized)", value=5)
                    stress_slider = gr.Slider(minimum=1, maximum=10, step=1, label="Stress (1=Calm, 10=Overwhelmed)", value=5)
                
                with gr.Row():
                    sleep_quality_slider = gr.Slider(minimum=1, maximum=10, step=1, label="Sleep Quality (1=Poor, 10=Excellent)", value=5)
                    focus_slider = gr.Slider(minimum=1, maximum=10, step=1, label="Focus (1=Scattered, 10=Sharp)", value=5)
                    motivation_slider = gr.Slider(minimum=1, maximum=10, step=1, label="Motivation (1=None, 10=High)", value=5)
                
                gr.Markdown("### Detailed Wellbeing Notes")
                with gr.Row():
                    with gr.Column():
                        emotional_state_textbox = gr.Textbox(label="Emotional State", placeholder="Describe your overall emotional state...")
                        stress_triggers_textbox = gr.Textbox(label="Stress Triggers", placeholder="What caused stress today?...")
                        coping_strategies_textbox = gr.Textbox(label="Coping Strategies", placeholder="How did you manage stress/challenges?...")
                    with gr.Column():
                        physical_symptoms_textbox = gr.Textbox(label="Physical Symptoms", placeholder="Headaches, tension, fatigue, etc...")
                        daily_events_textbox = gr.Textbox(label="Daily Events", placeholder="Significant events or activities...")
                        notes_textbox = gr.Textbox(label="Additional Notes", placeholder="Any other observations about your day...")
                
                submit_mood_btn = gr.Button("Save Daily Entry")
                mood_status_output = gr.Textbox(label="Status", interactive=False)
            
            submit_mood_btn.click(
                submit_mood_entry,
                inputs=[mood_slider, energy_slider, stress_slider, sleep_quality_slider, focus_slider, 
                       motivation_slider, emotional_state_textbox, stress_triggers_textbox, 
                       coping_strategies_textbox, physical_symptoms_textbox, daily_events_textbox, 
                       notes_textbox, mood_date_picker],
                outputs=[mood_status_output]
            )

        with gr.TabItem("Custom Analysis"):
            gr.Markdown("## Custom Correlation Analysis")
            
            with gr.Row():
                metric_x_dropdown = gr.Dropdown(ALL_AVAILABLE_METRICS, label="X Metric")
                metric_y_dropdown = gr.Dropdown(ALL_AVAILABLE_METRICS, label="Y Metric")
                analysis_type_dropdown = gr.Dropdown(["pearson", "average_comparison", "time_based_impact"], label="Analysis Type")
            
            with gr.Row():
                start_date_picker = gr.Textbox(label="Start Date (YYYY-MM-DD)", value=(date.today() - timedelta(days=30)).strftime('%Y-%m-%d'))
                end_date_picker = gr.Textbox(label="End Date (YYYY-MM-DD)", value=date.today().strftime('%Y-%m-%d'))
            
            gr.Markdown("### Filters")
            gr.Markdown("**Add New Filter**")
            with gr.Column():
                with gr.Row():
                    filter_metric_dropdown = gr.Dropdown(ALL_AVAILABLE_METRICS, label="Filter Metric")
                    filter_operator_dropdown = gr.Dropdown(['>', '>=', '<', '<=', '==', '!=', 'between', 'in', 'like'], label="Operator")
                    filter_value_input = gr.Textbox(label="Value (e.g., '25', '10,20', 'running,walking')")
                add_filter_btn = gr.Button("Add Filter")
                filter_add_status = gr.Textbox(label="Filter Status", interactive=False)
            
            # This gr.State will hold the list of active filters
            active_filters_state = gr.State([])

            # This is the dynamic area that will display active filters
            gr.Markdown("**Active Filters**")
            active_filters_display_area = gr.Column()
            
            # Initial render of filters (empty)
            demo.load(lambda: render_active_filters_ui([]), outputs=active_filters_display_area)

            add_filter_btn.click(
                add_filter_logic,
                inputs=[filter_metric_dropdown, filter_operator_dropdown, filter_value_input, active_filters_state],
                outputs=[active_filters_state, active_filters_display_area, filter_add_status]
            )

            clear_filters_btn = gr.Button("Clear All Filters")
            clear_filters_btn.click(
                clear_all_filters,
                inputs=[active_filters_state], # Pass the state to clear
                outputs=[active_filters_state, active_filters_display_area, filter_add_status] # Update state, re-render UI, clear status
            )

            gr.Markdown("---")
            gr.Markdown("## Analysis")
            analyze_correlation_btn = gr.Button("✨ Analyze Correlation")
            
            with gr.Row():
                analysis_results_output = gr.Textbox(label="Analysis Results", interactive=False, lines=10)
                correlation_plot_output = gr.Plot(label="Correlation Visualization")
            
            analyze_correlation_btn.click(
                perform_custom_analysis,
                inputs=[
                    metric_x_dropdown, metric_y_dropdown, analysis_type_dropdown,
                    start_date_picker, end_date_picker, active_filters_state
                ],
                outputs=[analysis_results_output, correlation_plot_output]
            )

# Launch the Gradio app
if __name__ == "__main__":
    logger.info("Starting Smart Health Agent with full tabbed interface...")
    # Initialize database and run migrations on startup
    db.connect()
    db.create_tables() # This also calls migrate_database_schema
    logger.info("Database initialized successfully")

    # Start Garmin sync in background, but don't block app launch
    # This is now handled by the handle_garmin_sync button click
    # For initial status, ensure get_app_status is called on demo.load
    
    demo.launch(server_name="0.0.0.0", server_port=7861)
