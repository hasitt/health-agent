# Smart Health Agent

A GPU-accelerated personalized health trend analysis and recommendation system focused on Garmin data, with future expansion to food, labs, and genetics.

## Overview

This repository demonstrates a minimal yet powerful workflow to analyze Garmin wearable data and deliver personalized, proactive health insights:

- **Garmin data integration:** Analyze HRV, HR, steps, sleep, and activity type.
- **Trend analysis engine:** Identify hourly and daily trends, correlations, and rolling patterns.
- **Insight generation:** Produce proactive text-based summaries (and future visual plots).
- **LLM integration:** Generate natural language explanations on-demand.

The system uses a modular agent design with LangGraph for orchestration and Ollama for LLM capabilities.

## Features

- **Hourly HRV and HR analysis:** Identify rest, low, medium, and high stress trends.
- **Daily activity and sleep correlation:** Understand how activity impacts recovery.
- **Proactive insight push:** Automatic summaries to surprise and delight users.
- **On-demand Q&A:** User asks, system explains trends naturally.
- **Future-ready:** Easy expansion to food tracking, labs, and genetics.

## System Architecture

```
MVP Flow
Garmin API -> Health Metrics Agent -> Insight Engine + Recommendation Agent -> User
```

Future modules may include weather and medical knowledge agents with vector database retrieval.

## Requirements

- Python 3.10+
- Ollama installed and running
- Garmin API credentials (using garminconnect package)
- Required Python packages (see `requirements.txt`)

## Repository Contents

- `health_metrics_agent.py`: Ingests and preprocesses Garmin data
- `insight_engine.py`: Core statistical and ML analysis logic
- `recommendation_agent.py`: Summarizes insights and interacts with user queries
- `garmin_utils.py`: Helper functions for Garmin API connection

## Installation

```bash
git clone https://github.com/yourname/smart_health_agent.git
cd smart_health_agent
pip install -r requirements.txt
```

Set up Garmin credentials using `.env`:

```
GARMIN_USERNAME=your_email@example.com
GARMIN_PASSWORD=your_password
```

Run initial data ingestion:

```bash
python health_metrics_agent.py
```

Start the main application:

```bash
python recommendation_agent.py
```

## Using the Application

After setup, the app analyzes historical Garmin data immediately. Users receive periodic insight summaries and can ask questions like "Why was my HRV low yesterday?".

## Future Roadmap

- Add food tracking
- Incorporate lab and genetics data for advanced personalization
- Introduce graphical trend visualizations (Plotly)
- Weather and environmental context integration
- Fully autonomous recommendation agent with continuous learning

## Technical Notes

Data is structured in a single master timeline DataFrame with hourly granularity. HRV values are hourly means derived from Garmin RMSSD 5‑min segments. Sleep scores and stages are joined daily for correlation analysis. The modular design makes it easy to add new agents.

## Acknowledgments

- Garmin Connect (unofficial Python integration)
- LangGraph for multi-agent orchestration
- Ollama for LLM-powered summaries

