import gradio as gr
import pandas as pd
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Tuple, Optional
import os
import asyncio
import logging
import re
from dotenv import load_dotenv

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import ollama, fall back to mock if not available
try:
    import ollama
    OLLAMA_AVAILABLE = True
    logger.info("Ollama client available for real LLM integration")
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available - using mock LLM interface")

# Import LangChain components
try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain integration available")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    logger.warning(f"LangChain not available - falling back to basic chat: {e}")

# Import your modules
from database import db
from garmin_utils import sync_garmin_data, initialize_garmin_client # Import initialize_garmin_client
from cronometer_parser import parse_cronometer_food_entries_csv
import trend_analyzer
import health_visualizations # For graphs
import llm_interface # LLM integration for Health Detective


# Load environment variables
load_dotenv()

# --- Global State & Initialization ---
current_user_id = 1 # Default user ID
db_initialized = False
garmin_sync_status = "Not synced"
cronometer_upload_status = "No data uploaded"

# --- LLM Configuration ---
LLM_MODEL = "llama3.2:3b"  # Now supports tool calling with downloaded model
OLLAMA_HOST = "http://localhost:11434"
LLM_CLIENT = None

# --- LangChain Agent Configuration ---
LANGCHAIN_AGENT = None
LANGCHAIN_LLM = None

def initialize_llm_client():
    """Initialize the Ollama client for real LLM integration."""
    global LLM_CLIENT, LLM_MODEL
    if OLLAMA_AVAILABLE:
        try:
            LLM_CLIENT = ollama.Client(host=OLLAMA_HOST)
            # Test if the model is available
            models = LLM_CLIENT.list()
            available_models = [model.model for model in models.models] if hasattr(models, 'models') else []
            
            if LLM_MODEL not in available_models:
                logger.warning(f"Model {LLM_MODEL} not found. Available models: {available_models}")
                # Try common alternatives
                for alt_model in ["llama3.2", "llama3", "mistral", "phi3"]:
                    if any(alt_model in model for model in available_models):
                        LLM_MODEL = [model for model in available_models if alt_model in model][0]
                        logger.info(f"Using alternative model: {LLM_MODEL}")
                        break
                else:
                    logger.error(f"No suitable LLM model found. Available: {available_models}")
                    return False
            
            logger.info(f"LLM Client initialized successfully with model: {LLM_MODEL}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            return False
    else:
        logger.info("Using mock LLM interface - Ollama not available")
        return False

def initialize_langchain_agent():
    """Initialize the LangChain agent with tools for health analysis."""
    global LANGCHAIN_AGENT, LANGCHAIN_LLM
    
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain not available - cannot initialize agent")
        return False
    
    try:
        # Import tools here to avoid circular import
        from agent_tools import HEALTH_AGENT_TOOLS
        
        # Initialize the ChatOllama LLM
        LANGCHAIN_LLM = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.7,
            num_predict=400,  # Reduce from 800 to 400 for faster responses
            keep_alive='10m'  # Keep model in memory longer to avoid reload time
        )
        
        # Create the agent prompt template
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Personal Health Coach and Data Analyst named "Health Detective." 
            You analyze user health data to provide personalized, actionable insights and recommendations.

            PERSONALITY:
            - Empathetic and supportive, never judgmental
            - Data-driven but speaks in plain language  
            - Encouraging and motivational
            - Focuses on patterns and actionable advice
            - Uses emojis appropriately to make responses engaging
            - ALWAYS maintain conversation context and respond to follow-up questions

            CAPABILITIES:
            You have access to powerful tools for health data analysis:
            1. get_health_data_summary: Get recent health statistics and trends
            2. generate_time_series_plots: Create visualizations showing health metrics over time
            3. perform_custom_analysis: Analyze relationships between different health metrics

            WORKFLOW:
            1. When users ask about their health, first use get_health_data_summary to understand their recent data
            2. Provide insights based on the data, including specific numbers and trends
            3. If users ask for plots/graphs/visualizations, ALWAYS use generate_time_series_plots tool
            4. For relationship questions (correlation, impact), use perform_custom_analysis
            5. For follow-up questions like "yes" or "show me", refer to conversation context and take appropriate action
            6. Always provide actionable recommendations based on the data

            CONVERSATION HANDLING:
            - Pay attention to chat history and maintain context
            - If user says "yes" to your offer to create a plot, immediately use generate_time_series_plots
            - If user asks follow-up questions, build on previous responses
            - Remember what you discussed earlier in the conversation

            COMMUNICATION STYLE:
            - Start with a warm greeting and show you understand their question
            - Present data clearly with specific numbers and context
            - Explain what the data means for their health
            - Provide 2-3 specific, actionable recommendations
            - When offering to create visualizations, actually create them if user agrees
            - Use markdown formatting for clarity
            
            VISUALIZATION HANDLING:
            - When you successfully call generate_time_series_plots, do NOT describe the plot data in detail
            - Do NOT create text tables or markdown representations of the data
            - Simply acknowledge that the visualization was created and provide brief insights
            - The actual interactive chart will be displayed separately by the UI
            - Focus on interpreting what the trends might mean for the user's health
            
            Remember: You're not just reporting data - you're helping people understand their health story and take positive action."""),
            
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        
        # Check if model supports tool calling based on model name
        supports_tools = "llama" in LLM_MODEL.lower() and ("3.1" in LLM_MODEL or "3.2" in LLM_MODEL)
        
        if supports_tools:
            try:
                # Try to create tool-calling agent
                agent = create_tool_calling_agent(LANGCHAIN_LLM, HEALTH_AGENT_TOOLS, agent_prompt)
                LANGCHAIN_AGENT = AgentExecutor(
                    agent=agent, 
                    tools=HEALTH_AGENT_TOOLS,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=5,
                    return_intermediate_steps=True  # Enable returning tool outputs
                )
                logger.info("Created tool-calling agent with health analysis tools")
            except Exception as tool_error:
                logger.warning(f"Tool-calling agent creation failed: {tool_error}")
                supports_tools = False
        
        if not supports_tools:
            logger.info(f"Model {LLM_MODEL} doesn't support tool calling - creating simple conversational chain")
            
            # Create a simple conversational chain for models without tool support
            from langchain_core.runnables import RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser
            
            # Simplified prompt for non-tool-calling models
            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Personal Health Coach and Data Analyst named "Health Detective."
                You analyze user health data to provide personalized, actionable insights and recommendations.
                
                You are empathetic, supportive, data-driven but speak in plain language.
                Focus on patterns and actionable advice. Use emojis appropriately.
                
                When users ask about their health data, provide insights and recommendations based on 
                the context provided. Be encouraging and motivational while being factual."""),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            # Create a simple chain
            LANGCHAIN_AGENT = simple_prompt | LANGCHAIN_LLM | StrOutputParser()
            logger.info("Created simple conversational chain")
        
        logger.info("LangChain agent initialized successfully with health tools")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize LangChain agent: {e}")
        return False

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
    """Initializes database, LLM client, and sets up background sync."""
    global db_initialized, garmin_sync_status
    if not db_initialized:
        try:
            db.connect()
            db.create_tables()
            db_initialized = True
            logger.info("Database initialized successfully")
            
            # LLM and LangChain agent are now initialized in main startup
            # Skip duplicate initialization here to avoid conflicts
            
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

# --- Health Detective LLM Functions ---

def summarize_health_data_for_llm(data_context: Dict[str, Any]) -> str:
    """
    Convert raw health data into a concise, LLM-friendly summary.
    
    Args:
        data_context: Dictionary containing raw health data
        
    Returns:
        Formatted string summary for LLM context
    """
    try:
        summary_parts = []
        date_range = data_context.get('date_range', {})
        
        if date_range:
            summary_parts.append(f"DATA PERIOD: {date_range.get('start_date')} to {date_range.get('end_date')} ({date_range.get('days')} days)")
        
        # Sleep Data Summary
        sleep_data = data_context.get('sleep_data', [])
        if sleep_data:
            durations = [entry.get('sleep_duration_hours') for entry in sleep_data if entry.get('sleep_duration_hours')]
            scores = [entry.get('sleep_score') for entry in sleep_data if entry.get('sleep_score')]
            
            if durations:
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)
                summary_parts.append(f"SLEEP ({len(durations)} nights): Avg {avg_duration:.1f}h (range: {min_duration:.1f}-{max_duration:.1f}h)")
            
            if scores:
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                good_nights = sum(1 for s in scores if s >= 80)
                summary_parts.append(f"SLEEP QUALITY: Avg {avg_score:.0f}/100 (range: {min_score}-{max_score}), {good_nights}/{len(scores)} nights ≥80")
        
        # Activity Data Summary
        activity_data = data_context.get('activity_data', [])
        if activity_data:
            steps = [entry.get('total_steps') for entry in activity_data if entry.get('total_steps')]
            calories = [entry.get('active_calories') for entry in activity_data if entry.get('active_calories')]
            distance = [entry.get('distance_km') for entry in activity_data if entry.get('distance_km')]
            
            if steps:
                avg_steps = sum(steps) / len(steps)
                active_days = sum(1 for s in steps if s >= 7000)
                summary_parts.append(f"ACTIVITY ({len(steps)} days): Avg {avg_steps:.0f} steps/day, {active_days}/{len(steps)} days ≥7k steps")
            
            if calories:
                avg_calories = sum(calories) / len(calories)
                summary_parts.append(f"ACTIVE CALORIES: Avg {avg_calories:.0f} kcal/day")
        
        # Nutrition Data Summary  
        nutrition_data = data_context.get('nutrition_data', [])
        if nutrition_data:
            total_cals = [entry.get('total_calories') for entry in nutrition_data if entry.get('total_calories')]
            protein = [entry.get('protein_g') for entry in nutrition_data if entry.get('protein_g')]
            
            if total_cals:
                avg_cals = sum(total_cals) / len(total_cals)
                summary_parts.append(f"NUTRITION ({len(total_cals)} days): Avg {avg_cals:.0f} kcal/day")
            
            if protein:
                avg_protein = sum(protein) / len(protein)
                summary_parts.append(f"PROTEIN: Avg {avg_protein:.0f}g/day")
        
        # Stress Data Summary
        stress_data = data_context.get('stress_data', [])
        if stress_data:
            stress_levels = [entry.get('avg_daily_stress') for entry in stress_data if entry.get('avg_daily_stress')]
            rhr_values = [entry.get('avg_daily_rhr') for entry in stress_data if entry.get('avg_daily_rhr') and entry.get('avg_daily_rhr') > 0]
            
            if stress_levels:
                avg_stress = sum(stress_levels) / len(stress_levels)
                high_stress_days = sum(1 for s in stress_levels if s >= 60)
                summary_parts.append(f"STRESS ({len(stress_levels)} days): Avg {avg_stress:.0f}/100, {high_stress_days}/{len(stress_levels)} high stress days (≥60)")
            
            if rhr_values:
                avg_rhr = sum(rhr_values) / len(rhr_values)
                summary_parts.append(f"RESTING HR: Avg {avg_rhr:.0f} bpm")
        
        # Mood Data Summary
        mood_data = data_context.get('mood_data', [])
        if mood_data:
            moods = [entry.get('mood') for entry in mood_data if entry.get('mood')]
            energy = [entry.get('energy') for entry in mood_data if entry.get('energy')]
            
            if moods:
                avg_mood = sum(moods) / len(moods)
                good_days = sum(1 for m in moods if m >= 7)
                summary_parts.append(f"MOOD ({len(moods)} entries): Avg {avg_mood:.1f}/10, {good_days}/{len(moods)} good days (≥7)")
            
            if energy:
                avg_energy = sum(energy) / len(energy)
                summary_parts.append(f"ENERGY: Avg {avg_energy:.1f}/10")
        
        # Recent trends (last 7 days vs previous period)
        if len(summary_parts) > 1:
            summary_parts.append("\\nRECENT PATTERNS: Focus on last 7 days for trend analysis")
        
        return "\\n".join(summary_parts) if summary_parts else "No health data available for analysis."
        
    except Exception as e:
        logger.error(f"Error summarizing health data: {e}")
        return "Error processing health data for analysis."

def get_relevant_health_data(user_message: str, days_back: int = 30) -> Dict[str, Any]:
    """
    Intelligently fetch relevant health data based on user's message/query.
    
    Args:
        user_message: The user's question or request
        days_back: How many days back to fetch data (default 30)
        
    Returns:
        Dictionary containing relevant health data for LLM context
    """
    try:
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Analyze user message to determine what data to fetch
        message_lower = user_message.lower()
        data_context = {}
        
        # Always include basic info
        data_context['date_range'] = {
            'start_date': start_date_str,
            'end_date': end_date_str,
            'days': days_back
        }
        
        # Sleep data (if sleep-related query or general)
        if any(word in message_lower for word in ['sleep', 'rest', 'tired', 'fatigue']) or 'general' in message_lower:
            sleep_data = db.get_garmin_sleep(current_user_id, start_date_str, end_date_str)
            if sleep_data:
                data_context['sleep_data'] = sleep_data
                logger.debug(f"Fetched {len(sleep_data)} sleep records")
        
        # Activity data (if activity-related query or general)
        if any(word in message_lower for word in ['steps', 'activity', 'exercise', 'walk', 'active']) or 'general' in message_lower:
            activity_data = db.get_garmin_daily_summary(current_user_id, start_date_str, end_date_str)
            if activity_data:
                data_context['activity_data'] = activity_data
                logger.debug(f"Fetched {len(activity_data)} activity records")
        
        # Nutrition data (if nutrition-related query or general)
        if any(word in message_lower for word in ['food', 'nutrition', 'eat', 'calories', 'diet']) or 'general' in message_lower:
            nutrition_data = db.get_food_log_daily_summary(current_user_id, start_date_str, end_date_str)
            if nutrition_data:
                data_context['nutrition_data'] = nutrition_data
                logger.debug(f"Fetched {len(nutrition_data)} nutrition records")
        
        # Stress data (if stress-related query or general)
        if any(word in message_lower for word in ['stress', 'heart rate', 'rhr']) or 'general' in message_lower:
            # Use activity data for stress info since it includes daily stress
            if 'activity_data' not in data_context:
                stress_data = db.get_garmin_daily_summary(current_user_id, start_date_str, end_date_str)
                if stress_data:
                    data_context['stress_data'] = stress_data
            else:
                data_context['stress_data'] = data_context['activity_data']
        
        # Mood/subjective data (if mood-related query or general)
        if any(word in message_lower for word in ['mood', 'feel', 'energy', 'mental', 'wellbeing']) or 'general' in message_lower:
            mood_data = db.get_subjective_wellbeing(current_user_id, start_date_str, end_date_str)
            if mood_data:
                data_context['mood_data'] = mood_data
                logger.debug(f"Fetched {len(mood_data)} mood records")
        
        # If it's a correlation query, get everything
        if any(word in message_lower for word in ['correlat', 'relationship', 'connect', 'impact', 'affect']):
            # Fetch all available data types
            for data_type in ['sleep_data', 'activity_data', 'nutrition_data', 'mood_data']:
                if data_type not in data_context:
                    if data_type == 'sleep_data':
                        data = db.get_garmin_sleep(current_user_id, start_date_str, end_date_str)
                    elif data_type == 'activity_data':
                        data = db.get_garmin_daily_summary(current_user_id, start_date_str, end_date_str)
                    elif data_type == 'nutrition_data':
                        data = db.get_food_log_daily_summary(current_user_id, start_date_str, end_date_str)
                    elif data_type == 'mood_data':
                        data = db.get_subjective_wellbeing(current_user_id, start_date_str, end_date_str)
                    
                    if data:
                        data_context[data_type] = data
        
        logger.info(f"Health Detective: Fetched data context with {len(data_context)} data types for query: '{user_message[:50]}...'")
        return data_context
        
    except Exception as e:
        logger.error(f"Error fetching health data for LLM: {e}")
        return {'error': f"Could not fetch health data: {str(e)}"}

def create_system_prompt() -> str:
    """Create the system prompt for the Health Detective LLM."""
    return """You are a Personal Health Coach and Data Analyst named "Health Detective." You analyze user health data to provide personalized, actionable insights.

PERSONALITY:
- Empathetic and supportive, never judgmental
- Data-driven but speaks in plain language
- Encouraging and motivational
- Focuses on patterns and actionable advice

CAPABILITIES:
- Analyze sleep, activity, nutrition, stress, and mood data
- Identify health patterns and correlations
- Provide evidence-based recommendations
- Generate visualizations when helpful

TOOL CALLING:
When a graph would help illustrate your insights, use this exact format:
TOOL_CALL:{"function": "generate_visualization", "type": "sleep_trends|activity_trends|nutrition_trends|stress_trends|mood_trends|correlation_analysis"}

RESPONSE STYLE:
- Start with a warm, personal greeting
- Present data insights clearly with specific numbers
- Use emojis sparingly but effectively (📊 📈 💤 🏃 🍎 😌)
- End with actionable suggestions or follow-up questions
- Keep responses conversational and engaging

EXAMPLE RESPONSES:
User: "How has my sleep been?"
You: "Looking at your recent sleep data, I can see some interesting patterns! 😴

Over the past 28 nights, you've averaged 7.8 hours of sleep, which is great - right in that healthy 7-9 hour range. Your sleep quality score has been averaging around 82/100, with 18 out of 28 nights hitting that 'good sleep' threshold of 80+.

I notice your sleep duration has been pretty consistent, ranging from 6.8 to 8.5 hours. That consistency is excellent for your circadian rhythm!

TOOL_CALL:{"function": "generate_visualization", "type": "sleep_trends"}

What I'd love to explore with you: Have you noticed any patterns in what helps you get those higher quality sleep scores? Things like exercise timing, evening routine, or room temperature can make a big difference!"

Always provide specific, data-backed insights and be genuinely helpful."""

def create_user_prompt(user_message: str, health_data_summary: str, conversation_history: List[Dict]) -> str:
    """Create the complete prompt for the LLM including context and conversation history."""
    
    # Format conversation history
    history_str = ""
    if conversation_history:
        recent_history = conversation_history[-6:]  # Last 3 exchanges
        for entry in recent_history:
            role = entry.get('role', 'user')
            content = entry.get('content', '')
            history_str += f"{role.upper()}: {content}\\n"
    
    prompt = f"""HEALTH DATA CONTEXT:
{health_data_summary}

CONVERSATION HISTORY:
{history_str}

CURRENT USER MESSAGE: {user_message}

Provide a helpful, data-driven response as the Health Detective. Use specific numbers from the health data when relevant. If a visualization would be helpful, include a TOOL_CALL."""
    
    return prompt

def parse_llm_response(llm_output: str) -> Tuple[str, Optional[str]]:
    """
    Parse LLM response to extract tool calls and clean text.
    
    Args:
        llm_output: Raw LLM response
        
    Returns:
        Tuple of (cleaned_text_response, tool_call_info)
    """
    try:
        # Look for tool call pattern
        tool_call_match = re.search(r'TOOL_CALL:\s*(\{[^}]+\})', llm_output)
        
        if tool_call_match:
            tool_call_json = tool_call_match.group(1)
            # Remove the tool call from the text response
            cleaned_text = re.sub(r'TOOL_CALL:\s*\{[^}]+\}', '', llm_output).strip()
            return cleaned_text, tool_call_json
        else:
            return llm_output.strip(), None
            
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        return llm_output.strip(), None

def execute_tool_call(tool_call_json: str, data_context: Dict[str, Any]) -> Optional[Any]:
    """
    Execute a tool call to generate visualizations.
    
    Args:
        tool_call_json: JSON string with tool call information
        data_context: Health data context
        
    Returns:
        Plotly figure object or None
    """
    try:
        tool_call = json.loads(tool_call_json)
        viz_type = tool_call.get('type', 'general')
        
        logger.info(f"Executing tool call for visualization type: {viz_type}")
        
        # Generate appropriate visualization based on type
        if viz_type in ['sleep_trends', 'sleep']:
            plots = health_visualizations.generate_time_series_plots(current_user_id, 30)
            # Return the sleep-specific plot (usually index 1)
            return plots[1] if len(plots) > 1 else plots[0] if plots else None
            
        elif viz_type in ['activity_trends', 'activity']:
            plots = health_visualizations.generate_time_series_plots(current_user_id, 30)
            # Return the activity-specific plot (usually index 0)
            return plots[0] if plots else None
            
        elif viz_type in ['nutrition_trends', 'nutrition']:
            plots = health_visualizations.generate_time_series_plots(current_user_id, 30)
            # Return nutrition plots (calories and macros)
            nutrition_plots = [p for i, p in enumerate(plots) if i >= 3 and i <= 4]
            return nutrition_plots[0] if nutrition_plots else None
            
        elif viz_type in ['stress_trends', 'stress']:
            plots = health_visualizations.generate_time_series_plots(current_user_id, 30)
            # Return stress plot (usually index 2)
            return plots[2] if len(plots) > 2 else None
            
        elif viz_type in ['mood_trends', 'mood']:
            plots = health_visualizations.generate_time_series_plots(current_user_id, 30)
            # Return mood plot (usually last)
            return plots[-1] if plots else None
            
        elif viz_type == 'hourly_stress':
            return health_visualizations.generate_hourly_stress_plot(current_user_id, 30)
            
        else:
            # Default to general health overview
            plots = health_visualizations.generate_time_series_plots(current_user_id, 30)
            return plots[0] if plots else None
            
    except Exception as e:
        logger.error(f"Error executing tool call: {e}")
        return None

def call_ollama_llm(prompt: str) -> str:
    """
    Call the Ollama LLM with the given prompt.
    
    Args:
        prompt: Complete prompt for the LLM
        
    Returns:
        LLM response text
    """
    try:
        if LLM_CLIENT and OLLAMA_AVAILABLE:
            logger.info(f"Calling Ollama LLM with model: {LLM_MODEL}")
            
            response = LLM_CLIENT.chat(
                model=LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': create_system_prompt()},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.7,  # Slightly creative but not too random
                    'top_p': 0.9,       # Focus on likely tokens
                    'num_predict': 800,  # Reasonable response length
                }
            )
            
            llm_response = response['message']['content']
            logger.info(f"LLM responded with {len(llm_response)} characters")
            return llm_response
            
        else:
            # Fallback to mock LLM if Ollama not available
            logger.info("Using fallback mock LLM response")
            return llm_interface.get_llm_response(prompt, {})[0]
            
    except Exception as e:
        logger.error(f"Error calling Ollama LLM: {e}")
        return f"I apologize, but I'm having trouble connecting to my analysis engine right now. Please try again in a moment, or let me know if you'd like me to try a different approach to help you with your health data."

def get_basic_health_stats() -> Dict[str, Any]:
    """Get basic health statistics from the last 7 days."""
    try:
        db.connect()
        conn = db.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Get recent sleep data (last 7 days)
        cursor.execute('''
            SELECT AVG(sleep_duration_hours), AVG(sleep_score), COUNT(*) 
            FROM garmin_sleep 
            WHERE date >= date('now', '-7 days') AND sleep_duration_hours > 0
        ''')
        sleep_data = cursor.fetchone()
        if sleep_data and sleep_data[2] > 0:
            stats['sleep'] = {
                'avg_duration': round(sleep_data[0] or 0, 1),
                'avg_score': round(sleep_data[1] or 0, 0),
                'nights_tracked': sleep_data[2]
            }
        
        # Get recent activity data (last 7 days)
        cursor.execute('''
            SELECT AVG(total_steps), AVG(active_calories), COUNT(*) 
            FROM garmin_daily_summary 
            WHERE date >= date('now', '-7 days') AND total_steps > 0
        ''')
        activity_data = cursor.fetchone()
        if activity_data and activity_data[2] > 0:
            stats['activity'] = {
                'avg_steps': round(activity_data[0] or 0, 0),
                'avg_active_calories': round(activity_data[1] or 0, 0),
                'days_tracked': activity_data[2]
            }
        
        # Get recent nutrition data (last 7 days)
        cursor.execute('''
            SELECT AVG(total_calories), AVG(protein_g), COUNT(*) 
            FROM food_log_daily 
            WHERE date >= date('now', '-7 days') AND total_calories > 0
        ''')
        nutrition_data = cursor.fetchone()
        if nutrition_data and nutrition_data[2] > 0:
            stats['nutrition'] = {
                'avg_calories': round(nutrition_data[0] or 0, 0),
                'avg_protein': round(nutrition_data[1] or 0, 1),
                'days_tracked': nutrition_data[2]
            }
        
        # Get recent mood data (last 7 days)
        cursor.execute('''
            SELECT AVG(mood), AVG(energy), AVG(stress), COUNT(*) 
            FROM subjective_wellbeing 
            WHERE date >= date('now', '-7 days') AND mood > 0
        ''')
        mood_data = cursor.fetchone()
        if mood_data and mood_data[3] > 0:
            stats['mood'] = {
                'avg_mood': round(mood_data[0] or 0, 1),
                'avg_energy': round(mood_data[1] or 0, 1),
                'avg_stress': round(mood_data[2] or 0, 1),
                'days_tracked': mood_data[3]
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting basic health stats: {e}")
        return {}

def respond_to_chat(message: str, chat_history: List[Dict]) -> Tuple[str, List[Dict], Optional[Any]]:
    """
    Handle user chat messages using LangChain agent with health analysis tools.
    
    Args:
        message: User's message/question
        chat_history: Current chat history in message format
        
    Returns:
        Tuple of (empty_message, updated_chat_history, optional_graph)
    """
    try:
        # Ensure chat_history is always a list (defensive programming)
        if chat_history is None:
            chat_history = []
        
        if not message.strip():
            return "", chat_history, None
        
        logger.info(f"Health Detective Agent: Processing message: '{message[:50]}...'")
        
        # Use LangChain agent if available, otherwise fall back to basic chat
        if LANGCHAIN_AGENT and LANGCHAIN_AVAILABLE:
            return _handle_with_langchain_agent(message, chat_history)
        else:
            logger.warning("LangChain agent not available - using fallback basic chat")
            return _handle_with_basic_chat(message, chat_history)
            
    except Exception as e:
        logger.error(f"Error in respond_to_chat: {e}", exc_info=True)
        error_response = "I apologize, I encountered an error while processing your request. Let me try a simple response."
        updated_history = (chat_history or []) + [
            {"role": "user", "content": message}, 
            {"role": "assistant", "content": error_response}
        ]
        return "", updated_history, None

def _handle_with_langchain_agent(message: str, chat_history: List[Dict]) -> Tuple[str, List[Dict], Optional[Any]]:
    """Handle chat using the LangChain agent with tools."""
    try:
        # Convert chat history to LangChain format
        langchain_history = []
        for msg in chat_history:
            if msg.get("role") == "user":
                langchain_history.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                langchain_history.append(AIMessage(content=msg["content"]))
        
        logger.info(f"Invoking LangChain agent with {len(langchain_history)} history messages")
        
        # Check if we have a tool-calling agent or simple chain
        if hasattr(LANGCHAIN_AGENT, 'invoke') and hasattr(LANGCHAIN_AGENT, 'tools'):
            # Tool-calling agent
            response = LANGCHAIN_AGENT.invoke({
                "input": message,
                "chat_history": langchain_history
            })
            
            # Extract the response text
            response_text = response.get("output", "I apologize, but I couldn't generate a proper response.")
            
            # Check if the agent used tools and generated plots
            plot_output = None
            intermediate_steps = response.get("intermediate_steps", [])
            
            # Look for plot outputs in the intermediate steps
            for step in intermediate_steps:
                if len(step) >= 2:
                    tool_output = step[1]  # The tool result
                    if isinstance(tool_output, dict):
                        # Check for errors first
                        if "error" in tool_output:
                            logger.warning(f"Tool returned error: {tool_output['error']}")
                        # Check for successful plot generation
                        elif "plot" in tool_output and tool_output["plot"] is not None:
                            plot_output = tool_output["plot"]
                            logger.info("✅ Agent generated a plot visualization")
                            break
        else:
            # Simple conversational chain
            # Get health data to provide context
            health_stats = get_basic_health_stats()
            
            # Add health context to the message
            context_message = f"User's recent health data: {health_stats}\n\nUser question: {message}"
            
            response_text = LANGCHAIN_AGENT.invoke({
                "input": context_message,
                "chat_history": langchain_history
            })
            
            plot_output = None
            logger.info("Used simple conversational chain (no tools)")
        
        # Update chat history
        updated_history = chat_history + [
            {"role": "user", "content": message}, 
            {"role": "assistant", "content": response_text}
        ]
        
        logger.info(f"LangChain Agent: Generated response ({len(response_text)} chars), plot: {bool(plot_output)}")
        return "", updated_history, plot_output
        
    except Exception as e:
        logger.error(f"Error with LangChain agent: {e}", exc_info=True)
        # Fall back to basic chat on error
        return _handle_with_basic_chat(message, chat_history)

def _handle_with_basic_chat(message: str, chat_history: List[Dict]) -> Tuple[str, List[Dict], Optional[Any]]:
    """Fallback basic chat handler when LangChain agent is not available."""
    try:
        logger.info(f"Basic Chat Fallback: Processing message: '{message[:50]}...'")
        
        # Get basic health stats for context
        health_stats = get_basic_health_stats()
        logger.info(f"Retrieved health stats: {list(health_stats.keys())}")
        
        # Simple response based on keywords
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['hello', 'hi', 'hey']):
            if health_stats:
                overview = "Hello! I'm your Health Detective. Recent data summary:\n"
                if 'sleep' in health_stats:
                    sleep = health_stats['sleep']
                    overview += f"🛏️ Sleep: {sleep['avg_duration']}h avg\n"
                if 'activity' in health_stats:
                    activity = health_stats['activity']
                    overview += f"🚶 Activity: {activity['avg_steps']:,} steps avg\n"
                if 'nutrition' in health_stats:
                    nutrition = health_stats['nutrition']
                    overview += f"🍽️ Nutrition: {nutrition['avg_calories']} cal avg\n"
                if 'mood' in health_stats:
                    mood = health_stats['mood']
                    overview += f"😊 Mood: {mood['avg_mood']}/10 avg\n"
                overview += "\nWhat would you like to explore in detail?"
                response_text = overview
            else:
                response_text = "Hello! I'm your Health Detective. I don't see recent health data - make sure your devices are synced!"
        
        elif 'test' in message_lower:
            response_text = f"✅ Basic chat is working! Available data: {', '.join(health_stats.keys()) if health_stats else 'None'}"
        
        else:
            response_text = f"I'm your Health Detective! I'm currently in basic mode (LangChain agent unavailable). I can see data for: {', '.join(health_stats.keys()) if health_stats else 'None'}. Try asking about your health!"
        
        # Update chat history
        updated_history = chat_history + [
            {"role": "user", "content": message}, 
            {"role": "assistant", "content": response_text}
        ]
        
        logger.info(f"Basic Chat: Generated response ({len(response_text)} chars)")
        return "", updated_history, None
        
    except Exception as e:
        logger.error(f"Error in basic chat fallback: {e}", exc_info=True)
        error_response = "I apologize, I'm having technical difficulties. Please try again."
        updated_history = (chat_history or []) + [
            {"role": "user", "content": message}, 
            {"role": "assistant", "content": error_response}
        ]
        return "", updated_history, None

def respond_to_chat_complex(message: str, chat_history: List[Dict]) -> Tuple[str, List[Dict], Optional[Any]]:
    """
    Handle user chat messages and generate real LLM responses with health insights.
    
    Args:
        message: User's message/question
        chat_history: Current chat history in message format
        
    Returns:
        Tuple of (empty_message, updated_chat_history, optional_graph)
    """
    try:
        # Ensure chat_history is always a list (defensive programming)
        if chat_history is None:
            chat_history = []
        
        if not message.strip():
            return "", chat_history, None
        
        logger.info(f"Health Detective: Processing user message: '{message[:100]}...'")
        
        # Step 1: Get relevant health data based on the user's message
        data_context = get_relevant_health_data(message, days_back=90)  # Get more data for better context
        
        # Step 2: Summarize health data for LLM context
        health_data_summary = summarize_health_data_for_llm(data_context)
        logger.info(f"Health data summary: {len(health_data_summary)} characters")
        
        # Step 3: Create comprehensive prompt for LLM
        user_prompt = create_user_prompt(message, health_data_summary, chat_history)
        
        # Step 4: Call the real LLM (Ollama) or fallback to mock
        llm_raw_response = call_ollama_llm(user_prompt)
        
        # Step 5: Parse LLM response for tool calls
        llm_text_response, tool_call_json = parse_llm_response(llm_raw_response)
        
        # Step 6: Execute tool calls if present
        graph_output = None
        if tool_call_json:
            logger.info(f"Health Detective: Tool call detected: {tool_call_json}")
            graph_output = execute_tool_call(tool_call_json, data_context)
            
            if graph_output:
                llm_text_response += "\\n\\n📊 **I've generated a visualization to help illustrate these patterns:**"
            else:
                logger.warning("Tool call failed to generate graph")
        
        # Step 7: Update chat history with new message format
        updated_history = chat_history + [
            {"role": "user", "content": message}, 
            {"role": "assistant", "content": llm_text_response}
        ]
        
        logger.info(f"Health Detective: Generated response with {len(llm_text_response)} characters, graph: {bool(graph_output)}")
        return "", updated_history, graph_output
        
    except Exception as e:
        logger.error(f"Error in Health Detective chat response: {e}", exc_info=True)
        # Add more specific debugging information
        logger.error(f"Message: '{message}', Chat history type: {type(chat_history)}, Length: {len(chat_history) if chat_history else 'None'}")
        error_response = "I apologize, but I encountered an error while analyzing your health data. This might be due to a connection issue with my analysis engine. Please try again in a moment."
        updated_history = (chat_history or []) + [
            {"role": "user", "content": message}, 
            {"role": "assistant", "content": error_response}
        ]
        return "", updated_history, None

def generate_suggested_graph(graph_suggestion: str, data_context: Dict[str, Any]) -> Optional[Any]:
    """
    Generate appropriate graphs based on LLM suggestions.
    
    Args:
        graph_suggestion: The graph suggestion from LLM (e.g., "GRAPH_SUGGESTION: sleep_trends")
        data_context: Health data context
        
    Returns:
        Plotly figure object or None
    """
    try:
        # Extract the graph type from the suggestion
        if ":" in graph_suggestion:
            graph_type = graph_suggestion.split(":", 1)[1].strip()
        else:
            graph_type = graph_suggestion.replace("GRAPH_SUGGESTION", "").strip()
        
        logger.info(f"Health Detective: Generating graph of type: {graph_type}")
        
        # Generate appropriate visualization based on type
        if graph_type in ['sleep_trends', 'sleep']:
            return health_visualizations.generate_time_series_plots(current_user_id, 30)
        elif graph_type in ['activity_trends', 'activity']:
            return health_visualizations.generate_time_series_plots(current_user_id, 30)
        elif graph_type in ['nutrition_trends', 'nutrition']:
            return health_visualizations.generate_time_series_plots(current_user_id, 30)
        elif graph_type in ['stress_trends', 'stress']:
            return health_visualizations.generate_time_series_plots(current_user_id, 30)
        elif graph_type in ['mood_trends', 'mood']:
            return health_visualizations.generate_time_series_plots(current_user_id, 30)
        elif graph_type in ['hourly_stress', 'stress_hourly']:
            return health_visualizations.generate_hourly_stress_plot(current_user_id, 30)
        elif graph_type in ['correlation_analysis', 'correlation']:
            # For now, return general plots - could be enhanced for specific correlations
            return health_visualizations.generate_time_series_plots(current_user_id, 30)
        else:
            # Default to general health overview
            plots = health_visualizations.generate_time_series_plots(current_user_id, 30)
            return plots[0] if plots else None
        
    except Exception as e:
        logger.error(f"Error generating suggested graph: {e}")
        return None

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

        with gr.TabItem("AI Health Coach"):
            gr.Markdown("# 🤖 AI Health Detective")
            gr.Markdown("**Your personal health data analyst and coach. Ask me anything about your health patterns, trends, and insights!**")
            
            with gr.Row():
                with gr.Column(scale=2):
                    health_chatbot = gr.Chatbot(
                        label="Health Detective Chat",
                        height=500,
                        show_label=True,
                        container=True,
                        show_copy_button=True,
                        type="messages"
                    )
                    
                    with gr.Row():
                        health_chat_input = gr.Textbox(
                            label="Ask me about your health data...",
                            placeholder="Try: 'How has my sleep been lately?' or 'Show me my activity trends' or 'What's the relationship between my sleep and mood?'",
                            lines=2,
                            max_lines=5,
                            show_label=False,
                            container=False
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=0)
                    
                    with gr.Row():
                        clear_chat_btn = gr.Button("Clear Chat", variant="secondary")
                        example_questions_btn = gr.Button("Example Questions", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 **Example Questions:**")
                    gr.Markdown("""
                    **Sleep Analysis:**
                    - "How has my sleep quality been this month?"
                    - "What's my average sleep duration?"
                    - "Show me my sleep trends"
                    
                    **Activity & Fitness:**
                    - "How active have I been lately?"
                    - "What's my daily step average?"
                    - "Show me my activity patterns"
                    
                    **Nutrition Insights:**
                    - "How's my nutrition looking?"
                    - "What are my calorie trends?"
                    - "Show me my protein intake"
                    
                    **Stress & Recovery:**
                    - "How are my stress levels?"
                    - "What's my heart rate trend?"
                    - "Show me stress patterns by hour"
                    
                    **Mood & Wellbeing:**
                    - "How has my mood been?"
                    - "What affects my energy levels?"
                    - "Show me my wellbeing trends"
                    
                    **Correlations:**
                    - "How does sleep affect my mood?"
                    - "What's the relationship between activity and sleep?"
                    - "Does nutrition impact my energy?"
                    """)
            
            # Graph output for AI-suggested visualizations
            ai_suggested_graph = gr.Plot(label="AI-Generated Insights Visualization", visible=False)
            
            # Chat interaction handlers
            def handle_chat_submit(message, chat_history):
                """Handle chat message submission"""
                empty_msg, updated_history, graph = respond_to_chat(message, chat_history)
                
                # Update graph visibility and content
                if graph:
                    if isinstance(graph, list) and len(graph) > 0:
                        # If multiple graphs returned, show the first one
                        return empty_msg, updated_history, gr.Plot(value=graph[0], visible=True)
                    else:
                        # Single graph
                        return empty_msg, updated_history, gr.Plot(value=graph, visible=True)
                else:
                    # No graph suggested
                    return empty_msg, updated_history, gr.Plot(visible=False)
            
            def clear_chat():
                """Clear the chat history"""
                return [], gr.Plot(visible=False)
            
            def show_example_questions():
                """Show example questions in the chat"""
                examples = [
                    {"role": "user", "content": "How has my sleep been lately?"},
                    {"role": "assistant", "content": "I'd be happy to analyze your recent sleep patterns! Let me look at your sleep data..."},
                    {"role": "user", "content": "What's my average daily steps?"},
                    {"role": "assistant", "content": "Let me check your activity data to calculate your daily step average..."},
                    {"role": "user", "content": "Show me the relationship between my sleep and mood"},
                    {"role": "assistant", "content": "Great question! I'll analyze how your sleep quality correlates with your mood ratings..."}
                ]
                return examples, gr.Plot(visible=False)
            
            # Wire up the chat interactions
            health_chat_input.submit(
                handle_chat_submit,
                inputs=[health_chat_input, health_chatbot],
                outputs=[health_chat_input, health_chatbot, ai_suggested_graph]
            )
            
            send_btn.click(
                handle_chat_submit,
                inputs=[health_chat_input, health_chatbot],
                outputs=[health_chat_input, health_chatbot, ai_suggested_graph]
            )
            
            clear_chat_btn.click(
                clear_chat,
                outputs=[health_chatbot, ai_suggested_graph]
            )
            
            example_questions_btn.click(
                show_example_questions,
                outputs=[health_chatbot, ai_suggested_graph]
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
    
    # Initialize LLM client and LangChain agent
    llm_initialized = initialize_llm_client()
    if llm_initialized:
        logger.info(f"LLM client ready with model: {LLM_MODEL}")
    else:
        logger.info("Using mock LLM interface")
    
    # Initialize LangChain agent
    langchain_success = initialize_langchain_agent()
    if langchain_success:
        logger.info("LangChain agent ready for intelligent conversations")
    else:
        logger.warning("LangChain agent initialization failed - using fallback chat")

    # Start Garmin sync in background, but don't block app launch
    # This is now handled by the handle_garmin_sync button click
    # For initial status, ensure get_app_status is called on demo.load
    
    demo.launch(server_name="0.0.0.0", server_port=7861)
