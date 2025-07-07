# TCM Knowledge Ingestion System

This document describes the Traditional Chinese Medicine (TCM) knowledge ingestion system that enables Retrieval Augmented Generation (RAG) for TCM insights in the Smart Health Agent.

## Overview

The TCM Knowledge Ingestion System consists of three main components:

1. **TCM Knowledge Base** (`tcm_knowledge_base.md`) - Comprehensive markdown file containing TCM knowledge
2. **Ingestion Script** (`tcm_knowledge_ingestion.py`) - Main ingestion engine
3. **Initialization Script** (`initialize_tcm_knowledge.py`) - Simple interface for integration

## Files

### `tcm_knowledge_base.md`
A comprehensive markdown file containing detailed TCM knowledge including:
- Five Elements Theory (Wood, Fire, Earth, Metal, Water)
- 24-Hour Organ Clock with detailed organ functions
- Stress patterns and TCM interpretations
- Dietary therapy and food properties
- Lifestyle recommendations
- Herbal medicine information
- Acupressure points
- Prevention and wellness practices
- Integration with modern health

### `tcm_knowledge_ingestion.py`
The main ingestion engine that:
- Loads TCM knowledge from the markdown file
- Chunks the content for optimal RAG retrieval
- Embeds documents using HuggingFace embeddings
- Stores embeddings in Milvus vector database
- Provides verification and management functions

### `initialize_tcm_knowledge.py`
A simple interface for:
- Initializing TCM knowledge in Milvus
- Checking if TCM knowledge exists
- Getting information about ingested TCM knowledge
- Integration with the main Smart Health Agent system

## Installation and Setup

### Prerequisites
1. Milvus server running (default: `http://localhost:19530`)
2. Required Python packages (already in `requirements.txt`):
   - `langchain` and related packages
   - `pymilvus`
   - `sentence-transformers`
   - `transformers`

### Quick Start

1. **Check if TCM knowledge exists:**
   ```bash
   python initialize_tcm_knowledge.py --action check
   ```

2. **Initialize TCM knowledge:**
   ```bash
   python initialize_tcm_knowledge.py --action init
   ```

3. **Get TCM knowledge information:**
   ```bash
   python initialize_tcm_knowledge.py --action info
   ```

4. **Force re-ingestion:**
   ```bash
   python initialize_tcm_knowledge.py --action init --force
   ```

## Usage

### Command Line Usage

#### Basic Initialization
```bash
# Initialize TCM knowledge with default settings
python initialize_tcm_knowledge.py

# Initialize with custom settings
python initialize_tcm_knowledge.py --chunk-size 800 --overlap 150

# Force re-ingestion
python initialize_tcm_knowledge.py --force
```

#### Advanced Options
```bash
# Custom Milvus URI and collection
python initialize_tcm_knowledge.py --milvus-uri http://your-milvus:19530 --collection my_health_knowledge

# Check status
python initialize_tcm_knowledge.py --action check

# Get detailed information
python initialize_tcm_knowledge.py --action info
```

### Programmatic Usage

#### From Python Code
```python
from initialize_tcm_knowledge import initialize_tcm_knowledge, check_tcm_knowledge_status

# Initialize TCM knowledge
success = initialize_tcm_knowledge(
    milvus_uri="http://localhost:19530",
    collection_name="health_knowledge",
    force=False,
    chunk_size=1000,
    chunk_overlap=200
)

# Check if TCM knowledge exists
exists = check_tcm_knowledge_status()
```

#### Integration with Smart Health Agent
```python
# In your main application
from initialize_tcm_knowledge import initialize_tcm_knowledge

def setup_health_agent():
    # Initialize TCM knowledge if not already present
    if not check_tcm_knowledge_status():
        logger.info("Initializing TCM knowledge...")
        initialize_tcm_knowledge()
    
    # Continue with agent setup
    # ...
```

## Configuration

### Default Settings
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Milvus URI**: `http://localhost:19530`
- **Collection Name**: `health_knowledge`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

### Customization
You can customize the ingestion process by modifying the parameters:

```python
# Custom chunking settings
initialize_tcm_knowledge(
    chunk_size=800,      # Smaller chunks for more granular retrieval
    chunk_overlap=150    # Less overlap to reduce redundancy
)

# Custom Milvus settings
initialize_tcm_knowledge(
    milvus_uri="http://your-milvus-server:19530",
    collection_name="custom_health_collection"
)
```

## Integration with Smart Health Agent

### Automatic Initialization
The TCM knowledge can be automatically initialized when the Smart Health Agent starts:

```python
# In smart_health_ollama.py or similar
from initialize_tcm_knowledge import initialize_tcm_knowledge, check_tcm_knowledge_status

def initialize_health_system():
    # Ensure TCM knowledge is available
    if not check_tcm_knowledge_status():
        logger.info("Initializing TCM knowledge for enhanced health insights...")
        initialize_tcm_knowledge()
    
    # Initialize other components
    # ...
```

### Enhanced RAG Queries
With TCM knowledge in Milvus, the `MedicalKnowledgeAgent` can now retrieve TCM-specific information:

```python
# Example RAG query that includes TCM context
query = f"""
User health data: {health_metrics}
TCM stress patterns: {tcm_insights}
Weather conditions: {weather_data}

Please provide holistic health recommendations combining modern medicine and TCM principles.
"""
```

## Testing

### Run Tests
```bash
# Run comprehensive tests
python test_tcm_ingestion.py

# Test specific components
python -c "
from test_tcm_ingestion import test_tcm_content_loading
test_tcm_content_loading()
"
```

### Verification
After ingestion, verify that TCM knowledge is properly stored:

```bash
# Check if TCM knowledge exists
python initialize_tcm_knowledge.py --action check

# Get detailed information
python initialize_tcm_knowledge.py --action info
```

## Troubleshooting

### Common Issues

1. **Milvus Connection Error**
   ```
   Error: Could not connect to Milvus server
   Solution: Ensure Milvus is running and accessible
   ```

2. **Import Errors**
   ```
   Error: Module not found
   Solution: Install required packages: pip install -r requirements.txt
   ```

3. **TCM Knowledge File Not Found**
   ```
   Error: TCM knowledge file not found
   Solution: Ensure tcm_knowledge_base.md exists in the same directory
   ```

4. **Embedding Model Download**
   ```
   Error: Could not download embedding model
   Solution: Check internet connection and try again
   ```

### Logs
The system generates detailed logs in `tcm_ingestion.log` for debugging:

```bash
# View logs
tail -f tcm_ingestion.log

# Check for errors
grep ERROR tcm_ingestion.log
```

## Performance Considerations

### Chunking Strategy
- **Smaller chunks (500-800 chars)**: More granular retrieval, better for specific questions
- **Larger chunks (1000-1500 chars)**: Better context preservation, good for comprehensive answers
- **Overlap (100-300 chars)**: Balances context preservation with storage efficiency

### Embedding Model
- **Current**: `sentence-transformers/all-MiniLM-L6-v2` (fast, good quality)
- **Alternative**: `sentence-transformers/all-mpnet-base-v2` (higher quality, slower)

### Storage
- TCM knowledge typically creates 50-100 chunks
- Each chunk requires ~384-dimensional embedding vector
- Total storage: ~50-100KB for embeddings

## Maintenance

### Updating TCM Knowledge
1. Edit `tcm_knowledge_base.md`
2. Re-ingest with force flag:
   ```bash
   python initialize_tcm_knowledge.py --action init --force
   ```

### Backup and Restore
```bash
# Backup TCM knowledge
python initialize_tcm_knowledge.py --action info > tcm_backup.json

# Restore (if needed)
# The system will automatically re-ingest if knowledge is missing
```

### Monitoring
```bash
# Check TCM knowledge status regularly
python initialize_tcm_knowledge.py --action check

# Monitor collection size
python initialize_tcm_knowledge.py --action info
```

## Future Enhancements

### Planned Features
1. **Multi-language Support**: Chinese TCM knowledge base
2. **Versioning**: Track different versions of TCM knowledge
3. **Incremental Updates**: Update only changed sections
4. **Custom Embeddings**: Domain-specific TCM embeddings
5. **Validation**: Automated quality checks for ingested knowledge

### Integration Opportunities
1. **External TCM APIs**: Integrate with TCM databases
2. **User Feedback**: Learn from user interactions
3. **Personalization**: Adapt recommendations based on user constitution
4. **Seasonal Updates**: Automatically adjust recommendations based on seasons

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `tcm_ingestion.log`
3. Run tests with `python test_tcm_ingestion.py`
4. Verify Milvus connection and collection status

## Conclusion

The TCM Knowledge Ingestion System provides a robust foundation for integrating Traditional Chinese Medicine insights into the Smart Health Agent. By following this guide, you can successfully deploy and maintain TCM knowledge for enhanced health recommendations. 