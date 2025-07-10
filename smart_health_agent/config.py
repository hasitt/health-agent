"""
Configuration and logging setup for Smart Health Agent.
"""

import logging
import os
from pathlib import Path

# Application configuration
class Config:
    """Application configuration settings."""
    
    # Ollama configuration
    OLLAMA_HOST = "http://localhost:11434"
    OLLAMA_MODEL = 'gemma3:4b-it-q4_K_M'
    OLLAMA_TEMPERATURE = 0.2
    
    # Database configuration
    MILVUS_URI = "http://localhost:19530"
    COLLECTION_NAME = "health_docs_rag"
    
    # File paths
    BASE_DIR = Path(__file__).parent
    # TCM_KNOWLEDGE_FILE removed - not part of MVP
    
    # Logging configuration
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Agent settings
    MAX_SEARCH_RESULTS = 5
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

def setup_logging(level=None):
    """Setup application logging."""
    if level is None:
        level = Config.LOG_LEVEL
    
    logging.basicConfig(
        level=level,
        format=Config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('smart_health_agent.log')
        ]
    )
    
    # Create loggers for different components
    loggers = {}
    for component in ['health_agent', 'knowledge_agent', 'recommendation_agent', 
                     'weather_agent', 'rag_setup', 'garmin', 'ui', 'chat']:
        logger = logging.getLogger(component)
        loggers[component] = logger
    
    return loggers

# Global logger instances
loggers = setup_logging()

# Convenience function to get logger
def get_logger(name):
    """Get a logger instance for the specified component."""
    return loggers.get(name, logging.getLogger(name)) 