"""
Smart Health Agent - Streamlined MVP
Core Garmin data integration with simplified morning reports.
"""

import os
import json
import requests
import gradio as gr
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
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# Garmin Connect integration
try:
    from garmin_utils import get_garmin_health_data, GarminConnectError, GarminHealthData
    GARMIN_AVAILABLE = True
except ImportError:
    GARMIN_AVAILABLE = False
    get_logger('health_agent').warning("Garmin integration not available")

# RAG / Milvus imports
import document_processor as dp

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
    
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

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
    """Generate simple recommendations based on health data."""
    logger.info("Generating general recommendations")
    
    context = build_recommendation_context(state)
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

def build_recommendation_context(state: HealthAgentState) -> str:
    """Build context for general recommendations."""
    health_data = state.health_data
    sleep_hours = health_data.get('sleep_hours', 0)
    sleep_score = health_data.get('garmin_data', {}).get('sleep_score', 0)
    resting_hr = health_data.get('heart_rate', 0)
    steps = health_data.get('steps', 0)
    avg_stress = health_data.get('stress_metrics', {}).get('avg_stress', 0)

    context_parts = [
        f"Sleep: {sleep_hours:.1f} hours (Score: {sleep_score}/100)",
        f"Resting Heart Rate: {resting_hr} bpm",
        f"Yesterday's Steps: {steps:,}",
        f"Overall Stress Level: {avg_stress}/100"
    ]
    
    if state.rag_context and "search_results" in state.rag_context:
        context_parts.append(f"Medical Knowledge Context: {state.rag_context['search_results']}")

    return "\\n".join(context_parts)

def create_recommendation_prompt(context: str) -> str:
    """Create a simplified prompt for general recommendations."""
    return f"""You are a helpful health assistant. Based on the following health data:

{context}

Provide a concise and encouraging health recommendation. Focus on general well-being and basic actionable advice related to sleep, activity, heart rate, or stress. Limit to 2-3 sentences."""

###############################################################################
# MORNING REPORT AGENT (Simplified)
###############################################################################

def generate_morning_report(state: HealthAgentState) -> HealthAgentState:
    """Generate a simplified automated morning report based on Garmin health data."""
    logger.info("Generating simplified morning report")
    
    # Extract core metrics
    health_data = state.health_data
    garmin_raw = health_data.get('garmin_data', {})
    
    sleep_hours = health_data.get('sleep_hours', 0)
    sleep_score = garmin_raw.get('sleep_score', 0)
    resting_hr = health_data.get('heart_rate', 0)
    steps = health_data.get('steps', 0)
    
    # Get simplified sleep breakdown
    sleep_breakdown_text = ""
    if 'sleep' in health_data and isinstance(health_data['sleep'], dict):
        sleep_data = health_data['sleep']
        sleep_quality = sleep_data.get('sleep_quality', 'Unknown')
        
        sleep_breakdown_text = f"""
Sleep Details:
- Duration: {sleep_hours:.1f} hours
- Quality: {sleep_quality}
- Score: {sleep_score}/100"""
    
    # Get simplified stress information
    stress_info = ""
    if 'stress_metrics' in health_data:
        stress_data = health_data['stress_metrics']
        avg_stress = stress_data.get('avg_stress', 0)
        stress_level = stress_data.get('stress_level', 'Unknown')
        
        stress_info = f"""
Stress Overview:
- Average Level: {avg_stress}/100
- Overall Status: {stress_level}"""
    
    # Build concise context for LLM
    context = f"""Yesterday's Health Metrics:
- Sleep: {sleep_hours:.1f} hours (Score: {sleep_score}/100)
- Resting Heart Rate: {resting_hr} bpm
- Yesterday's Steps: {steps:,}
- Avg Stress: {health_data.get('stress_metrics', {}).get('avg_stress', 0)}/100"""
    
    # Create concise, data-driven morning report prompt
    prompt = f"""Generate a morning report. Be extremely concise and data-driven. Present yesterday's key metrics directly. Then, provide ONE clear, actionable recommendation for today based on these metrics. Omit all pleasantries, lengthy explanations, and verbose encouragement.

{context}

FORMAT REQUIREMENTS:
- Start immediately with data (no greetings)
- Present metrics in this exact format:
  Sleep: [X hrs] (Score: [Y/100])
  RHR: [Z bpm] 
  Steps Yesterday: [A]
  Avg Stress: [B/100]
- Follow with: "Recommendation: [Single concise action for today based on data]"
- Maximum 6 lines total
- Be objective and factual, not conversational

Generate the concise morning report now:"""
    
    # Generate the morning report
    morning_report = ""
    try:
        for chunk in llm.stream(prompt):
            morning_report += chunk
    except Exception as e:
        logger.error(f"Error generating morning report: {e}")
        # Fallback to a concise report
        morning_report = f"""Sleep: {sleep_hours:.1f} hrs (Score: {sleep_score}/100)
RHR: {resting_hr} bpm
Steps Yesterday: {steps:,}
Avg Stress: {health_data.get('stress_metrics', {}).get('avg_stress', 0)}/100

Recommendation: Focus on today's goals."""
    
    # Store the morning report in state
    state.morning_report = morning_report
    state.recommendations.append(AIMessage(content=morning_report))
    
    logger.info("Morning report generated successfully")
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
    """Create the streamlined Gradio user interface."""
    with gr.Blocks() as demo:  # Removed theme to avoid the themes import error
        gr.Markdown("# Smart Health Agent 🩺")
        gr.Markdown("Your personalized AI health assistant with Garmin integration.")

        with gr.Row():
            data_source_input = gr.Radio(
                ["Garmin"], # Only Garmin option
                label="Select Data Source",
                value="Garmin",
                info="Garmin: Real data from Garmin Connect (requires setup)."
            )
            city_input = gr.Textbox(label="Enter Your City for Weather", value="San Francisco")
            init_button = gr.Button("Initialize System")
        
        system_status = gr.Textbox(label="System Status")
        chat_history = gr.Chatbot(label="Chat History")
        user_message = gr.Textbox(label="Your Health Query")
        send_button = gr.Button("Send")

        # Store state
        agent_state = gr.State(HealthAgentState())

        init_button.click(
            initialize_system,
            inputs=[data_source_input, gr.State("documents"), city_input],  # Use actual documents folder
            outputs=[system_status, chat_history, agent_state]
        )

        send_button.click(
            chat_interact,
            inputs=[user_message, chat_history, agent_state],
            outputs=[user_message, chat_history, agent_state]
        )

        user_message.submit(
            chat_interact,
            inputs=[user_message, chat_history, agent_state],
            outputs=[user_message, chat_history, agent_state]
        )
    
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    create_ui() 