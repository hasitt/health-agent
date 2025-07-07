# Smart Health Agent - Tests and Debug Scripts

This directory contains all test and debug scripts for the Smart Health Agent application.

## Test Scripts

### Core Integration Tests
- **`test_garmin_fix.py`** - Tests Garmin integration with error handling
- **`test_tcm_integration.py`** - Tests complete TCM-Garmin integration workflow
- **`test_data_flow.py`** - Traces data flow from Garmin to recommendations
- **`test_system_after_init.py`** - Verifies system status after initialization

### Component-Specific Tests
- **`test_tcm_basic.py`** - Basic TCM functionality tests (doesn't require Milvus)
- **`test_tcm_stress.py`** - TCM stress data functionality tests
- **`test_tcm_ingestion.py`** - TCM knowledge ingestion tests

## Debug Scripts

### Data Retrieval Debug
- **`debug_garmin.py`** - Comprehensive Garmin data retrieval debugging
- **`debug_steps.py`** - Specific debugging for steps data retrieval
- **`debug_hrv.py`** - Heart Rate Variability data debugging

## Usage

### Running Tests
```bash
# Run from the main application directory
cd /path/to/health-agent/smart_health_agent

# Basic TCM tests (no external dependencies)
python tests/test_tcm_basic.py

# Garmin integration tests (requires Garmin credentials)
python tests/test_garmin_fix.py

# Complete data flow test
python tests/test_data_flow.py

# System status verification
python tests/test_system_after_init.py
```

### Debug Scripts
```bash
# Debug Garmin data retrieval
python tests/debug_garmin.py

# Debug specific date
python tests/debug_garmin.py --date 2024-01-15

# Debug steps data specifically
python tests/debug_steps.py
```

## Test Categories

### ✅ Passing Tests
- Basic TCM functionality
- Garmin data retrieval
- Data processing pipeline
- Agent workflow execution

### ⚠️ Environment-Dependent Tests
- Garmin integration (requires valid credentials)
- Milvus vector database (requires running instance)
- Ollama LLM (requires running Ollama server)

### 🧪 Debugging Tools
- Comprehensive logging and error analysis
- Step-by-step data flow tracing
- Component isolation testing

## Notes

- All test scripts are designed to be run independently
- Debug scripts provide detailed logging for troubleshooting
- Tests that require external services will gracefully handle missing dependencies
- Use these scripts to verify system functionality after changes or deployments 