"""
Smart Health Agent - Streamlined MVP
Core Garmin data integration with simplified morning reports.
"""

import os
import json
import requests
import gradio as gr
import tempfile
import shutil
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Import configuration and logging
from config import Config, get_logger

# LangGraph imports
from langgraph.graph import StateGraph, END, START

# Core LLM / embedding imports
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# Garmin Connect integration
try:
    from garmin_utils import get_garmin_health_data, GarminConnectError, GarminHealthData, start_hybrid_sync
    GARMIN_AVAILABLE = True
except ImportError:
    GARMIN_AVAILABLE = False
    get_logger('health_agent').warning("Garmin integration not available")

# Database integration
import database

# Cronometer CSV parsing
import cronometer_parser

# RAG / Milvus imports
# document_processor removed - not part of MVP

# Initialize components
logger = get_logger('health_agent')
chat_logger = get_logger('chat')
rag_logger = get_logger('rag_setup')
ui_logger = get_logger('ui')

# Global LLM instance
llm = OllamaLLM(
    model=Config.OLLAMA_MODEL,
    temperature=Config.OLLAMA_TEMPERATURE,
    base_url=Config.OLLAMA_HOST
)

# Global vectorstore (initialized once)
global_vectorstore = None

###############################################################################
# AGENT STATE
###############################################################################

class HealthAgentState(BaseModel):
    """State object for the health agent workflow."""
    messages: List[BaseMessage] = Field(default_factory=list)
    health_data: Dict[str, Any] = Field(default_factory=dict)
    weather_data: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[BaseMessage] = Field(default_factory=list)
    rag_context: Dict[str, Any] = Field(default_factory=dict)
    streaming_response: str = Field(default="")
    morning_report: str = Field(default="")
    
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

###############################################################################
# WEATHER AGENT
###############################################################################

def get_weather_data(latitude: float, longitude: float) -> dict:
    """Retrieve weather conditions for health recommendations."""
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m", "relative_humidity_2m", "weather_code"],
        "timezone": "America/Los_Angeles"
    }
    
    # Default values
    weather_data = {"temperature": 20, "humidity": 50, "condition": "Unknown"}
    
    try:
        resp = requests.get(base_url, params=params)
        data = resp.json()
        if resp.status_code == 200 and "current" in data:
            current = data["current"]
            weather_descriptions = {
                0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
                3: "Overcast", 45: "Foggy", 51: "Light drizzle",
                53: "Moderate drizzle", 61: "Light rain",
                63: "Moderate rain", 65: "Heavy rain"
            }
            weather_data = {
                "temperature": current.get("temperature_2m", 20),
                "humidity": current.get("relative_humidity_2m", 50),
                "condition": weather_descriptions.get(current.get("weather_code", 0), "Unknown")
            }
    except Exception as e:
        logger.error(f"Error retrieving weather data: {e}")
    
    return weather_data

###############################################################################
# HEALTH METRICS AGENT (Simplified)
###############################################################################

def analyze_health_metrics(state: HealthAgentState) -> HealthAgentState:
    """Analyze fitness data and evaluate vitals."""
    logger.info("Processing health data")
    
    # Extract core metrics
    hr = state.health_data.get('heart_rate', 0)
    sleep_hrs = state.health_data.get('sleep_hours', 0)
    steps = state.health_data.get('steps', 0)
    
    logger.info(f"Metrics - HR: {hr}, Sleep: {sleep_hrs}h, Steps: {steps}")

    # Evaluate vitals
    vitals_status = {
        'heart_rate': 'Normal' if 60 <= hr <= 100 else 'Abnormal',
        'sleep': 'Optimal' if 7 <= sleep_hrs <= 9 else 'Suboptimal',
        'activity': 'Active' if steps >= 10000 else 'Sedentary'
    }
    
    # Update state
    state.health_data.update({
        'vitals_status': vitals_status,
        'last_processed': datetime.now()
    })
    
    logger.info(f"Processed vitals status: {vitals_status}")
    return state

###############################################################################
# MEDICAL KNOWLEDGE AGENT (Simplified)
###############################################################################

def search_medical_knowledge(state: HealthAgentState) -> HealthAgentState:
    """Search medical documents for relevant health insights."""
    logger.info("Searching medical knowledge")
    global global_vectorstore
    
    # Extract the last user message for the query
    user_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_query = str(msg.content)  # Ensure it's a string
            break
    
    if not user_query:
        logger.info("No user query found for RAG search.")
        return state
    
    # Search knowledge base if available
    if global_vectorstore is not None:
        try:
            docs = global_vectorstore.similarity_search(user_query, k=2)
            rag_context = "\n".join([doc.page_content for doc in docs])
            state.rag_context = {"search_results": rag_context}
            logger.info("RAG search completed successfully.")
        except Exception as e:
            logger.error(f"Error during RAG search: {e}")
            state.rag_context = {"error": f"RAG search failed: {e}"}
    else:
        logger.info("No vector store available - skipping RAG search for MVP")
        state.rag_context = {"info": "Medical knowledge search not available in MVP mode"}
    
    return state

###############################################################################
# RECOMMENDATION AGENT (Simplified)
###############################################################################

def generate_recommendations(state: HealthAgentState) -> HealthAgentState:
    """Generate personalized recommendations based on health data and trend analysis."""
    logger.info("Generating personalized recommendations with trend analysis")
    
    # Get trend analysis results
    trend_results = get_trend_analysis_results(state)
    
    # Build enhanced context with trend data
    context = build_recommendation_context(state, trend_results)
    prompt = create_recommendation_prompt(context)
    
    recommendation_text = ""
    try:
        for chunk in llm.stream(prompt):
            recommendation_text += chunk
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        recommendation_text = "I'm sorry, I couldn't generate detailed recommendations at this moment."
    
    state.recommendations.append(AIMessage(content=recommendation_text))
    return state

def get_trend_analysis_results(state: HealthAgentState) -> Dict[str, List[str]]:
    """Get trend analysis results for the current user."""
    global current_user_id
    
    trend_results = {
        'stress_consistency': [],
        'recent_stress': [],
        'steps_vs_sleep': [],
        'activity_rhr_impact': []
    }
    
    if current_user_id is None:
        logger.warning("No current user ID available for trend analysis")
        return trend_results
    
    try:
        import trend_analyzer
        
        # Get stress consistency analysis
        trend_results['stress_consistency'] = trend_analyzer.get_hourly_stress_consistency(current_user_id)
        
        # Get recent stress patterns
        trend_results['recent_stress'] = trend_analyzer.get_recent_stress_patterns(current_user_id)
        
        # Get steps vs sleep analysis
        trend_results['steps_vs_sleep'] = trend_analyzer.get_steps_vs_sleep_effect(current_user_id)
        
        # Get activity type vs RHR analysis
        trend_results['activity_rhr_impact'] = trend_analyzer.get_activity_type_rhr_impact(current_user_id)
        
        logger.info("Successfully retrieved trend analysis results")
        
    except Exception as e:
        logger.error(f"Error getting trend analysis results: {e}")
    
    return trend_results

def build_recommendation_context(state: HealthAgentState, trend_results: Dict[str, List[str]] = None) -> str:
    """Build enhanced context for personalized recommendations including trend analysis."""
    health_data = state.health_data
    sleep_hours = health_data.get('sleep_hours', 0)
    sleep_score = health_data.get('garmin_data', {}).get('sleep_score', 0)
    resting_hr = health_data.get('heart_rate', 0)
    steps = health_data.get('steps', 0)
    avg_stress = health_data.get('stress_metrics', {}).get('avg_stress', 0)

    # Build basic health data context
    context_parts = [
        "=== CURRENT HEALTH DATA ===",
        f"Sleep: {sleep_hours:.1f} hours (Score: {sleep_score}/100)",
        f"Resting Heart Rate: {resting_hr} bpm",
        f"Yesterday's Steps: {steps:,}",
        f"Overall Stress Level: {avg_stress}/100",
        ""
    ]
    
    # Add trend analysis results if available
    if trend_results:
        context_parts.append("=== IDENTIFIED HEALTH TRENDS ===")
        
        # Stress consistency patterns
        if trend_results.get('stress_consistency'):
            context_parts.append("Stress Consistency Analysis:")
            for observation in trend_results['stress_consistency']:
                context_parts.append(f"  • {observation}")
            context_parts.append("")
        
        # Recent stress patterns
        if trend_results.get('recent_stress'):
            context_parts.append("Recent Stress Patterns (7 days):")
            for observation in trend_results['recent_stress']:
                context_parts.append(f"  • {observation}")
            context_parts.append("")
        
        # Steps vs sleep correlation
        if trend_results.get('steps_vs_sleep'):
            context_parts.append("Steps vs Sleep Quality Analysis:")
            for observation in trend_results['steps_vs_sleep']:
                context_parts.append(f"  • {observation}")
            context_parts.append("")
        
        # Activity type vs RHR impact
        if trend_results.get('activity_rhr_impact'):
            context_parts.append("Activity Type vs Resting Heart Rate Analysis:")
            for observation in trend_results['activity_rhr_impact']:
                context_parts.append(f"  • {observation}")
            context_parts.append("")
    
    # Add medical knowledge context if available
    if state.rag_context and "search_results" in state.rag_context:
        context_parts.extend([
            "=== MEDICAL KNOWLEDGE CONTEXT ===",
            state.rag_context['search_results'],
            ""
        ])

    return "\n".join(context_parts)

def create_recommendation_prompt(context: str) -> str:
    """Create an enhanced prompt for personalized health recommendations based on trend analysis."""
    return f"""You are a highly knowledgeable and empathetic AI health coach. Your goal is to provide personalized, actionable insights and recommendations based on the user's health data and identified trends.

ROLE & RESPONSIBILITIES:
- Analyze provided factual health data and identified trends
- Generate personalized, actionable insights and practical recommendations  
- Be supportive, encouraging, and clear in your communication
- Focus on evidence-based advice that can improve health outcomes

CRITICAL CONSTRAINTS:
- Do NOT invent or assume any data not provided in the context
- Only interpret and analyze the data explicitly given to you
- If a trend is neutral, unclear, or insufficient data exists, state this factually
- Base all recommendations strictly on the provided information
- Avoid medical diagnosis - focus on lifestyle and wellness recommendations

Here is the user's health data and identified trends:

{context}

Based on this information, please provide:

1. **Key Insights**: What do the trends and current data tell us about the user's health patterns?

2. **Actionable Recommendations**: 3-5 specific, practical steps the user can take to improve their health or maintain positive trends.

3. **Areas for Attention**: Any patterns that might warrant closer monitoring or additional data collection (e.g., "Consider tracking X to better understand Y").

4. **Positive Reinforcement**: Acknowledge any positive trends or healthy behaviors evident in the data.

Keep your response comprehensive but concise (aim for 200-300 words). Be encouraging and focus on achievable, realistic advice."""

###############################################################################
# MORNING REPORT AGENT (Simplified)
###############################################################################

def generate_morning_report(state: HealthAgentState) -> HealthAgentState:
    """Generate an enhanced morning report with trend insights for actionable daily guidance."""
    logger.info("Generating enhanced morning report with trend insights")
    
    # Extract core metrics
    health_data = state.health_data
    garmin_raw = health_data.get('garmin_data', {})
    
    sleep_hours = health_data.get('sleep_hours', 0)
    sleep_score = garmin_raw.get('sleep_score', 0)
    resting_hr = health_data.get('heart_rate', 0)
    steps = health_data.get('steps', 0)
    avg_stress = health_data.get('stress_metrics', {}).get('avg_stress', 0)
    
    # Get trend analysis results for context
    trend_results = get_trend_analysis_results(state)
    
    # Build context with basic metrics and key trends
    context_parts = [
        "=== YESTERDAY'S HEALTH METRICS ===",
        f"Sleep: {sleep_hours:.1f} hours (Score: {sleep_score}/100)",
        f"Resting Heart Rate: {resting_hr} bpm", 
        f"Steps: {steps:,}",
        f"Average Stress: {avg_stress}/100",
        ""
    ]
    
    # Add most relevant trend insights for morning context
    if trend_results:
        context_parts.append("=== KEY RECENT TRENDS ===")
        
        # Add recent stress patterns (most relevant for daily planning)
        if trend_results.get('recent_stress'):
            context_parts.append("Recent Stress Pattern:")
            context_parts.append(f"  • {trend_results['recent_stress'][0]}" if trend_results['recent_stress'] else "  • No recent stress data")
        
        # Add most recent steps vs sleep finding  
        if trend_results.get('steps_vs_sleep'):
            context_parts.append("Activity-Sleep Correlation:")
            context_parts.append(f"  • {trend_results['steps_vs_sleep'][0]}" if trend_results['steps_vs_sleep'] else "  • No activity correlation data")
        
        context_parts.append("")
    
    context = "\n".join(context_parts)
    
    # Create morning report prompt focused on actionable daily insights
    prompt = f"""You are an AI health coach providing a concise morning briefing. Based on yesterday's health metrics and recent trends, provide a focused morning report for actionable daily planning.

{context}

Generate a morning report with this structure:

**Yesterday's Summary:**
- Brief assessment of yesterday's key metrics (2-3 sentences)

**Today's Focus:**
- ONE primary actionable recommendation for today based on the data and trends
- ONE secondary consideration if warranted by the trends

**Quick Insight:**
- One brief insight from the recent trends that's relevant for today's planning

Keep the entire response under 150 words. Be concise, actionable, and encouraging while staying strictly factual about the provided data."""
    
    # Generate the morning report
    morning_report = ""
    try:
        for chunk in llm.stream(prompt):
            morning_report += chunk
    except Exception as e:
        logger.error(f"Error generating morning report: {e}")
        # Fallback to a concise report
        morning_report = f"""**Yesterday's Summary:**
Sleep: {sleep_hours:.1f} hrs (Score: {sleep_score}/100), RHR: {resting_hr} bpm, Steps: {steps:,}, Stress: {avg_stress}/100

**Today's Focus:**
Focus on maintaining consistent activity levels and stress management.

**Quick Insight:**
Continue monitoring daily patterns for optimal health outcomes."""
    
    # Store the morning report in state
    state.morning_report = morning_report
    state.recommendations.append(AIMessage(content=morning_report))
    
    logger.info("Enhanced morning report generated successfully")
    return state

###############################################################################
# RAG SETUP (Optimized for single initialization)
###############################################################################

def setup_knowledge_base(docs_folder: str):
    """Initialize RAG system with medical documents (simplified for MVP)."""
    global global_vectorstore
    
    setup_start = datetime.now()
    
    if global_vectorstore is not None:
        rag_logger.info("Knowledge base already initialized - skipping")
        return global_vectorstore
    
    rag_logger.info(f"Initializing knowledge base: {docs_folder}")

    # For MVP, skip RAG entirely to focus on core Garmin functionality
    rag_logger.info("Skipping RAG setup for MVP - focusing on core Garmin functionality")
    
    total_setup_time = (datetime.now() - setup_start).total_seconds()
    rag_logger.info(f"Knowledge base setup skipped in {total_setup_time:.2f} seconds")
    return None

###############################################################################
# WORKFLOW (Simplified)
###############################################################################

def build_health_workflow():
    """Build the streamlined health agent workflow."""
    graph = StateGraph(HealthAgentState)
    graph.add_node("health_metrics", analyze_health_metrics)
    graph.add_node("medical_knowledge", search_medical_knowledge) 
    graph.add_node("generate_recommendations", generate_recommendations)
    graph.add_node("generate_morning_report", generate_morning_report)
    graph.add_edge(START, "health_metrics")
    graph.add_edge("health_metrics", "medical_knowledge")
    graph.add_edge("medical_knowledge", "generate_recommendations")
    graph.add_edge("generate_recommendations", "generate_morning_report")
    graph.add_edge("generate_morning_report", END)
    
    return graph.compile()

###############################################################################
# DATA PROCESSING (Garmin Only)
###############################################################################

def process_garmin_data(raw_data: dict) -> dict:
    """Process Garmin data into expected format."""
    try:
        steps_data = raw_data.get('steps', {})
        heart_rate_data = raw_data.get('heart_rate', {})
        sleep_data = raw_data.get('sleep', {})
        stress_data = raw_data.get('stress', {})
        
        processed_data = {
            'heart_rate': heart_rate_data.get('resting_hr', 0),
            'steps': steps_data.get('steps', 0),
            'sleep_hours': sleep_data.get('sleep_hours', 0),
            'calories': steps_data.get('calories', 0),
            'last_updated': raw_data.get('timestamp', datetime.now().isoformat()),
            'garmin_data': {
                'distance_meters': steps_data.get('distance_meters', 0),
                'sleep_score': sleep_data.get('sleep_score', 0),
                'stress': stress_data
            },
            # Add detailed sleep data for morning report
            'sleep': sleep_data,
            'stress_metrics': { # General stress metrics
                'avg_stress': stress_data.get('average_stress_level', 0),
                'stress_level': stress_data.get('overall_stress_level', 'Unknown')
            }
        }
        
        logger.info(f"Processed Garmin data: {processed_data['steps']} steps, {processed_data['sleep_hours']:.1f}h sleep, {processed_data['heart_rate']} bpm")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error processing Garmin data: {e}")
        return {}

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

def get_coordinates(city_name: str) -> tuple[float, float]:
    """Get coordinates for a city name."""
    try:
        geolocator = Nominatim(user_agent="smart_health_app")
        location = geolocator.geocode(city_name)
        if location:
            # Extract coordinates safely
            lat = getattr(location, 'latitude', None)
            lon = getattr(location, 'longitude', None)
            if lat is not None and lon is not None:
                return (float(lat), float(lon))
        
        # Default to San Francisco if city not found
        return (37.7749, -122.4194)
    except (GeocoderTimedOut, Exception) as e:
        ui_logger.error(f"Error getting coordinates for {city_name}: {e}")
        return (37.7749, -122.4194)

###############################################################################
# CHAT INTERFACE
###############################################################################

def chat_interact(user_message: str, chat_history: list, agent_state: HealthAgentState) -> tuple[str, list, HealthAgentState]:
    """Handle chat interactions and update the Gradio chat history."""
    ui_logger.info(f"User message: {user_message}")
    new_message = HumanMessage(content=user_message)
    agent_state.messages.append(new_message)

    try:
        # Build and run workflow
        app = build_health_workflow()
        result_state = app.invoke(agent_state)
        
        # Ensure result_state is a HealthAgentState object
        if isinstance(result_state, dict):
            # Convert dict back to HealthAgentState if needed
            result_state = HealthAgentState(**result_state)
        
        # Get response content
        if result_state.recommendations:
            response_content = result_state.recommendations[-1].content
        else:
            response_content = "I'm sorry, I couldn't process that request fully. Can you try again?"

        agent_response = AIMessage(content=response_content)
        result_state.messages.append(agent_response)
        chat_history.append((user_message, response_content))
        ui_logger.info("Agent response generated successfully")

        return "", chat_history, result_state
        
    except Exception as e:
        ui_logger.error(f"Error in chat interaction: {e}")
        error_message = f"Error processing your request: {e}"
        chat_history.append((user_message, error_message))
        return "", chat_history, agent_state

###############################################################################
# UI FUNCTIONS (Streamlined)
###############################################################################

# Global variables for user session
current_user_id = None
current_garmin_client = None

def initialize_database():
    """Initialize the SQLite database."""
    try:
        database.create_tables()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def format_latest_data_display(user_id: int) -> str:
    """Format the latest data from the database for display."""
    try:
        data = database.get_latest_data(user_id, days=1)
        
        output_lines = ["--- Latest Garmin Data ---"]
        
        # Daily Summary
        if data['daily_summaries']:
            summary = data['daily_summaries'][0]
            output_lines.extend([
                f"Daily Summary ({summary['date']}):",
                f"  Steps: {summary.get('total_steps', 0):,}",
                f"  Active Calories: {summary.get('active_calories', 0)}",
                f"  Avg RHR: {summary.get('avg_daily_rhr', 0)} bpm",
                f"  Avg Stress: {summary.get('avg_daily_stress', 0)}/100",
                ""
            ])
        else:
            output_lines.extend(["Daily Summary: No data available", ""])
        
        # Sleep Data
        if data['sleep_data']:
            sleep = data['sleep_data'][0]
            
            # Format sleep start/end times properly
            sleep_start_raw = sleep.get('sleep_start_time', 'Unknown')
            sleep_end_raw = sleep.get('sleep_end_time', 'Unknown')
            
            # Convert to readable format - handle both timestamp and string formats
            def format_sleep_time(time_raw):
                if time_raw == 'Unknown' or time_raw is None:
                    return 'Unknown'
                
                # Handle raw timestamp (milliseconds since epoch)
                if isinstance(time_raw, (int, float)):
                    try:
                        # Convert milliseconds to seconds
                        timestamp_seconds = time_raw / 1000
                        dt = datetime.fromtimestamp(timestamp_seconds)
                        return dt.strftime('%H:%M')
                    except:
                        return str(time_raw)
                
                # Handle datetime string
                if isinstance(time_raw, str):
                    try:
                        dt = datetime.fromisoformat(time_raw.replace('Z', '+00:00'))
                        return dt.strftime('%H:%M')
                    except:
                        return str(time_raw)
                
                return str(time_raw)
            
            sleep_start = format_sleep_time(sleep_start_raw)
            sleep_end = format_sleep_time(sleep_end_raw)
            
            total_sleep = sleep.get('total_sleep_minutes', 0)
            sleep_score = sleep.get('sleep_score', 0)
            deep = sleep.get('deep_sleep_minutes', 0)
            rem = sleep.get('rem_sleep_minutes', 0)
            light = sleep.get('light_sleep_minutes', 0)
            awake = sleep.get('awake_minutes', 0)
            
            output_lines.extend([
                f"Sleep ({sleep['date']}):",
                f"  Start: {sleep_start}",
                f"  End: {sleep_end}",
                f"  Total Sleep: {total_sleep} min (Score: {sleep_score}/100)",
                f"  Deep/REM/Light/Awake: {deep}/{rem}/{light}/{awake} min",
                ""
            ])
        else:
            output_lines.extend(["Sleep: No data available", ""])
        
        # Activities
        if data['activities']:
            output_lines.append("Recent Activities:")
            for activity in data['activities'][:5]:  # Show last 5 activities
                activity_type = activity.get('activity_type', 'Unknown')
                start_time_raw = activity.get('start_time', 'Unknown')
                duration = activity.get('duration_minutes', 0)
                distance = activity.get('distance_km', 0)
                calories = activity.get('calories_burned', 0)
                
                # Format start time
                try:
                    if isinstance(start_time_raw, str) and start_time_raw != 'Unknown':
                        start_time = datetime.fromisoformat(start_time_raw.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
                    else:
                        start_time = str(start_time_raw)
                except:
                    start_time = str(start_time_raw)
                
                output_lines.append(
                    f"  - {activity_type} at {start_time} for {duration} min "
                    f"({distance} km, {calories} cal)"
                )
        else:
            output_lines.extend(["Activities: No recent activities"])
        
        return "\n".join(output_lines)
        
    except Exception as e:
        logger.error(f"Failed to format latest data: {e}")
        return f"Error retrieving data: {e}"

def sync_garmin_data() -> tuple[str, str]:
    """Sync Garmin data using hybrid strategy."""
    global current_user_id, current_garmin_client
    
    if not GARMIN_AVAILABLE:
        return "❌ Garmin integration not available", ""
    
    try:
        # Initialize Garmin client if not already done
        if current_garmin_client is None:
            current_garmin_client = GarminHealthData()
            current_garmin_client.login()
            
            # Get user info and store in database
            # For MVP, we'll use a simple user ID approach
            garmin_user_id = "garmin_user_1"  # In real app, get from Garmin API
            current_user_id = database.insert_or_update_user(
                garmin_user_id=garmin_user_id,
                name="Garmin User",
                access_token="token",  # In real app, get actual tokens
                refresh_token="refresh_token",
                token_expiry=datetime.now() + timedelta(days=365)
            )
        
        # Start hybrid sync (enable background for comprehensive historical data)
        if current_user_id is not None:
            sync_success = start_hybrid_sync(current_user_id, current_garmin_client, enable_background=True)
            
            if sync_success:
                # Get and display latest data
                latest_data = format_latest_data_display(current_user_id)
                return "✅ Garmin sync completed (recent 30 days)", latest_data
            else:
                return "❌ Failed to sync Garmin data", ""
        else:
            return "❌ Failed to initialize user", ""
            
    except Exception as e:
        logger.error(f"Failed to sync Garmin data: {e}")
        return f"❌ Sync failed: {e}", ""

def show_latest_data() -> str:
    """Show the latest data from database."""
    global current_user_id
    
    if current_user_id is None:
        return "Please sync Garmin data first"
    
    return format_latest_data_display(current_user_id)

def format_food_log_display(user_id: int) -> str:
    """Format the food log data from the database for display."""
    try:
        food_data = database.get_food_log_summary(user_id, days=7)
        
        output_lines = ["--- Food Log Summary (Last 7 Days) ---"]
        
        # Daily summaries
        if food_data['daily_summaries']:
            output_lines.append("\n📊 Daily Nutrition Totals:")
            for day in food_data['daily_summaries']:
                output_lines.append(
                    f"  {day['date']}: {day['food_entries']} entries, "
                    f"{day['total_calories']:.0f} cal, "
                    f"P:{day['total_protein']:.1f}g C:{day['total_carbs']:.1f}g F:{day['total_fats']:.1f}g"
                )
        else:
            output_lines.append("\n📊 Daily Nutrition: No data available")
        
        # Recent food entries
        if food_data['recent_food_entries']:
            output_lines.extend(["\n🍽️ Recent Food Entries:", ""])
            for entry in food_data['recent_food_entries'][:10]:  # Show last 10
                timestamp = entry.get('timestamp', '')
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = dt.strftime('%m/%d %H:%M')
                    except:
                        time_str = timestamp
                else:
                    time_str = str(timestamp)
                
                quantity = entry.get('quantity', 0)
                unit = entry.get('unit', '')
                quantity_str = f"{quantity:.1f} {unit}" if unit else f"{quantity:.1f}"
                
                output_lines.append(
                    f"  {time_str}: {entry.get('food_item_name', 'Unknown')} "
                    f"({quantity_str}, {entry.get('calories', 0):.0f} cal)"
                )
        else:
            output_lines.extend(["\n🍽️ Recent Food Entries: No entries found"])
        
        # Recent supplements
        if food_data['recent_supplements']:
            output_lines.extend(["\n💊 Recent Supplements:", ""])
            for supplement in food_data['recent_supplements'][:5]:  # Show last 5
                timestamp = supplement.get('timestamp', '')
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = dt.strftime('%m/%d %H:%M')
                    except:
                        time_str = timestamp
                else:
                    time_str = str(timestamp)
                
                quantity = supplement.get('quantity', 0)
                unit = supplement.get('unit', '')
                quantity_str = f"{quantity:.1f} {unit}" if unit else f"{quantity:.1f}"
                
                output_lines.append(
                    f"  {time_str}: {supplement.get('supplement_name', 'Unknown')} ({quantity_str})"
                )
        else:
            output_lines.extend(["\n💊 Recent Supplements: No supplements found"])
        
        return "\n".join(output_lines)
        
    except Exception as e:
        logger.error(f"Failed to format food log display: {e}")
        return f"Error retrieving food log data: {e}"

def upload_cronometer_data(file) -> tuple[str, str]:
    """Handle Cronometer CSV file upload and processing."""
    global current_user_id
    
    if current_user_id is None:
        return "❌ Please sync Garmin data first to establish user session", ""
    
    # More robust file validation for Gradio
    if file is None:
        return "❌ No file selected. Please select a CSV file first.", ""
    
    # Check if file is a proper file object with read capability
    if not hasattr(file, 'read') and not hasattr(file, 'name'):
        return "❌ Invalid file object. Please select a valid CSV file.", ""
    
    # Handle case where file might be a string path or other object
    if isinstance(file, str):
        return "❌ Invalid file format. Please upload a CSV file directly.", ""
    
    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
            temp_file_path = temp_file.name
            
            # Handle different file object types
            if hasattr(file, 'read'):
                # File-like object
                file.seek(0)  # Reset file pointer
                shutil.copyfileobj(file, temp_file)
            elif hasattr(file, 'name') and os.path.exists(file.name):
                # File path object
                with open(file.name, 'rb') as source_file:
                    shutil.copyfileobj(source_file, temp_file)
            else:
                return "❌ Cannot read the uploaded file. Please try again.", ""
        
        logger.info(f"Processing Cronometer CSV: {temp_file_path}")
        
        # Validate CSV before processing
        validation_result = cronometer_parser.validate_cronometer_csv(temp_file_path)
        
        if not validation_result['is_valid']:
            issues = "\n".join(validation_result['issues'])
            return f"❌ Invalid CSV file:\n{issues}", ""
        
        # Parse and import the CSV
        import_summary = cronometer_parser.parse_cronometer_food_entries_csv(temp_file_path, current_user_id)
        
        # Format success message
        status_msg = f"""✅ Cronometer data imported successfully!
        
📊 Import Summary:
• Total rows processed: {import_summary['total_rows']}
• Food entries imported: {import_summary['food_entries']}
• Supplement entries imported: {import_summary['supplement_entries']}
• Errors: {import_summary['errors']}"""
        
        if import_summary['error_details']:
            status_msg += f"\n\n⚠️ Issues encountered:\n" + "\n".join(import_summary['error_details'][:5])
        
        # Get updated food log display
        food_log_display = format_food_log_display(current_user_id)
        
        logger.info(f"Cronometer import completed: {import_summary}")
        return status_msg, food_log_display
        
    except Exception as e:
        error_msg = f"❌ Failed to process Cronometer CSV: {str(e)}"
        logger.error(error_msg)
        return error_msg, ""
        
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")

def show_food_log() -> str:
    """Show the current food log data."""
    global current_user_id
    
    if current_user_id is None:
        return "Please sync Garmin data first to establish user session"
    
    return format_food_log_display(current_user_id)

def initialize_system(data_source: str, folder_path: str, city_name: str):
    """Initialize the health agent system - Garmin only."""
    start_time = datetime.now()
    ui_logger.info(f"Starting system initialization at {start_time}")

    try:
        # Initialize RAG (only once)
        rag_start = datetime.now()
        ui_logger.info("Step 1: Initializing knowledge base...")
        setup_knowledge_base(folder_path)
        rag_end = datetime.now()
        ui_logger.info(f"Knowledge base setup took: {(rag_end - rag_start).total_seconds():.2f} seconds")

        # Get Garmin health data
        garmin_start = datetime.now()
        ui_logger.info("Step 2: Getting Garmin data...")
        health_data = {}
        if data_source == "Garmin":
            if GARMIN_AVAILABLE:
                raw_data = get_garmin_health_data()
                health_data = process_garmin_data(raw_data)
                ui_logger.info("Successfully retrieved Garmin data")
            else:
                ui_logger.error("Garmin integration not available")
                raise ValueError("Garmin integration not available")
        else:
            # Should not be reached with streamlined UI
            ui_logger.error(f"Unsupported data source: {data_source}")
            raise ValueError(f"Unsupported data source: {data_source}")
        garmin_end = datetime.now()
        ui_logger.info(f"Garmin data retrieval took: {(garmin_end - garmin_start).total_seconds():.2f} seconds")

        # Get weather data
        weather_start = datetime.now()
        ui_logger.info("Step 3: Getting weather data...")
        latitude, longitude = get_coordinates(city_name)
        weather_data = get_weather_data(latitude, longitude)
        weather_end = datetime.now()
        ui_logger.info(f"Weather data retrieval took: {(weather_end - weather_start).total_seconds():.2f} seconds")

        # Initialize agent state with data
        state_start = datetime.now()
        ui_logger.info("Step 4: Initializing agent state...")
        state = HealthAgentState(health_data=health_data, weather_data=weather_data)
        state_end = datetime.now()
        ui_logger.info(f"Agent state initialization took: {(state_end - state_start).total_seconds():.2f} seconds")

        # Generate initial morning report
        report_start = datetime.now()
        ui_logger.info("Step 5: Generating initial morning report...")
        state = generate_morning_report(state)
        report_end = datetime.now()
        ui_logger.info(f"Morning report generation took: {(report_end - report_start).total_seconds():.2f} seconds")
        
        # Convert morning report to proper chat format
        if state.morning_report:
            chat_messages = [("System", state.morning_report)]
        else:
            chat_messages = [("System", "Morning report generated. System ready!")]
        
        total_time = (datetime.now() - start_time).total_seconds()
        ui_logger.info(f"Total initialization time: {total_time:.2f} seconds")
        
        return "System Initialized!", chat_messages, state

    except Exception as e:
        ui_logger.error(f"Initialization error: {e}")
        error_chat = [("System", f"Error: Could not initialize system: {e}")]
        return f"Initialization Failed: {e}", error_chat, HealthAgentState()

def create_ui():
    """Create the new database-focused Gradio user interface."""
    # Initialize database on startup
    initialize_database()
    
    with gr.Blocks() as demo:
        gr.Markdown("# Smart Health Agent 🩺")
        gr.Markdown("**MVP Version**: Garmin data storage, analysis, and trend correlations")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Garmin Data Sync")
                sync_button = gr.Button("Sync Garmin Data", variant="primary")
                sync_status = gr.Textbox(label="Sync Status", interactive=False)
                
            with gr.Column():
                gr.Markdown("### Data Display")
                show_data_button = gr.Button("Show Latest Data")
                
            with gr.Column():
                gr.Markdown("### Cronometer Data Upload")
                cronometer_file = gr.File(
                    label="Upload Cronometer CSV",
                    file_types=[".csv"],
                    file_count="single"
                )
                upload_cronometer_button = gr.Button("Import Cronometer Data", variant="secondary")
                cronometer_status = gr.Textbox(label="Import Status", interactive=False)
        
        # Data display areas
        with gr.Row():
            with gr.Column():
                data_display = gr.Textbox(
                    label="Latest Garmin Data",
                    value="Click 'Sync Garmin Data' to start",
                    lines=12,
                    interactive=False
                )
            
            with gr.Column():
                food_display = gr.Textbox(
                    label="Food Log Summary (Last 7 Days)",
                    value="Import Cronometer data to view food log",
                    lines=12,
                    interactive=False
                )
                show_food_button = gr.Button("Refresh Food Log", variant="secondary", size="sm")
        
        # Health Trends Analysis Section
        gr.Markdown("## Health Trends Analysis 📈")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Factual Trend Data")
                stress_consistency = gr.Textbox(
                    label="Stress Consistency Analysis",
                    value="Click 'Analyze Trends' to view stress patterns",
                    lines=4,
                    interactive=False
                )
                
                recent_stress = gr.Textbox(
                    label="Recent Stress Patterns (7 days)",
                    value="Click 'Analyze Trends' to view recent patterns",
                    lines=3,
                    interactive=False
                )
                
            with gr.Column():
                gr.Markdown("### Activity & Sleep Correlations")
                steps_sleep = gr.Textbox(
                    label="Steps vs Sleep Quality",
                    value="Click 'Analyze Trends' to view correlations",
                    lines=4,
                    interactive=False
                )
                
                activity_rhr = gr.Textbox(
                    label="Activity Type vs RHR Impact",
                    value="Click 'Analyze Trends' to view activity analysis",
                    lines=4,
                    interactive=False
                )
        
        # Trends analysis button
        analyze_trends_button = gr.Button("Analyze Trends", variant="secondary")
        
        # AI Insights Section - Separate from factual data
        gr.Markdown("## AI Health Coach Insights 🤖")
        gr.Markdown("*AI-generated interpretations and recommendations based on your trend data*")
        
        with gr.Row():
            with gr.Column():
                ai_insights = gr.Textbox(
                    label="Personalized Health Insights",
                    value="Click 'Generate AI Insights' after analyzing trends to receive personalized recommendations",
                    lines=8,
                    interactive=False
                )
                
            with gr.Column():
                morning_report_display = gr.Textbox(
                    label="Enhanced Morning Report",
                    value="AI morning report will appear here after generating insights",
                    lines=8,
                    interactive=False
                )
        
        # AI insights generation button
        generate_insights_button = gr.Button("Generate AI Insights", variant="primary")

        # Event handlers
        def handle_sync():
            status, data = sync_garmin_data()
            return status, data if data else "No data to display yet"

        def handle_show_data():
            return show_latest_data()
        
        def handle_cronometer_upload(file):
            status, food_data = upload_cronometer_data(file)
            return status, food_data
        
        def handle_analyze_trends():
            """Analyze health trends using the trend analyzer."""
            global current_user_id
            
            if current_user_id is None:
                error_msg = "Please sync Garmin data first"
                return error_msg, error_msg, error_msg, error_msg
            
            try:
                import trend_analyzer
                
                # Get stress consistency analysis
                stress_consistency_results = trend_analyzer.get_hourly_stress_consistency(current_user_id)
                stress_consistency_text = "\n".join(stress_consistency_results)
                
                # Get recent stress patterns
                recent_stress_results = trend_analyzer.get_recent_stress_patterns(current_user_id)
                recent_stress_text = "\n".join(recent_stress_results)
                
                # Get steps vs sleep analysis
                steps_sleep_results = trend_analyzer.get_steps_vs_sleep_effect(current_user_id)
                steps_sleep_text = "\n".join(steps_sleep_results)
                
                # Get activity type vs RHR analysis
                activity_rhr_results = trend_analyzer.get_activity_type_rhr_impact(current_user_id)
                activity_rhr_text = "\n".join(activity_rhr_results)
                
                return stress_consistency_text, recent_stress_text, steps_sleep_text, activity_rhr_text
                
            except Exception as e:
                error_msg = f"Error analyzing trends: {e}"
                logger.error(error_msg)
                return error_msg, error_msg, error_msg, error_msg
        
        def handle_generate_ai_insights():
            """Generate AI insights and morning report using trend analysis and LLM."""
            global current_user_id
            
            if current_user_id is None:
                error_msg = "Please sync Garmin data first"
                return error_msg, error_msg
            
            try:
                # Get the latest health data from database
                latest_data = database.get_latest_data(current_user_id, days=1)
                
                if not latest_data['daily_summaries']:
                    return "No recent health data available for AI analysis", "No data for morning report"
                
                # Process the health data into the format expected by the agent
                daily_summary = latest_data['daily_summaries'][0]
                sleep_data = latest_data['sleep_data'][0] if latest_data['sleep_data'] else {}
                
                # Create health data in the format expected by the agent
                processed_health_data = {
                    'heart_rate': daily_summary.get('avg_daily_rhr', 0),
                    'steps': daily_summary.get('total_steps', 0),
                    'sleep_hours': sleep_data.get('total_sleep_minutes', 0) / 60 if sleep_data.get('total_sleep_minutes') else 0,
                    'calories': daily_summary.get('active_calories', 0),
                    'garmin_data': {
                        'sleep_score': sleep_data.get('sleep_score', 0),
                    },
                    'stress_metrics': {
                        'avg_stress': daily_summary.get('avg_daily_stress', 0),
                        'stress_level': 'Moderate' if daily_summary.get('avg_daily_stress', 0) > 30 else 'Low'
                    }
                }
                
                # Create health agent state
                state = HealthAgentState(health_data=processed_health_data)
                
                # Generate AI recommendations
                state = generate_recommendations(state)
                ai_recommendations = state.recommendations[-1].content if state.recommendations else "No recommendations generated"
                
                # Generate enhanced morning report
                state = generate_morning_report(state)
                morning_report = state.morning_report if state.morning_report else "No morning report generated"
                
                logger.info("AI insights generated successfully")
                return ai_recommendations, morning_report
                
            except Exception as e:
                error_msg = f"Error generating AI insights: {e}"
                logger.error(error_msg)
                return error_msg, error_msg

        sync_button.click(
            handle_sync,
            inputs=[],
            outputs=[sync_status, data_display]
        )

        show_data_button.click(
            handle_show_data,
            inputs=[],
            outputs=[data_display]
        )
        
        analyze_trends_button.click(
            handle_analyze_trends,
            inputs=[],
            outputs=[stress_consistency, recent_stress, steps_sleep, activity_rhr]
        )
        
        generate_insights_button.click(
            handle_generate_ai_insights,
            inputs=[],
            outputs=[ai_insights, morning_report_display]
        )
        
        upload_cronometer_button.click(
            handle_cronometer_upload,
            inputs=[cronometer_file],
            outputs=[cronometer_status, food_display]
        )
        
        show_food_button.click(
            show_food_log,
            inputs=[],
            outputs=[food_display]
        )
    
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    create_ui() 