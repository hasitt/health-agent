**Development TODO for Smart Health Agent**

1. **Verify Garmin integration**
   - Run `python smart_health_agent/setup_garmin.py` to create a `.env` with credentials.
   - Use the "Test Garmin connection" option in the setup script.
   - Execute `python smart_health_agent/tests/test_garmin_fix.py` to confirm data can be retrieved.
   - Ensure token persistence in `.garmin_tokens` and handle login errors.

2. **Ingest Garmin data and store it in a unified structure**
   - Use `garmin_utils.get_comprehensive_health_data()` (and related helpers) to fetch daily health metrics.
   - Design a master DataFrame (e.g., with Pandas) that merges steps, HR, HRV, sleep, and stress data by timestamp.
   - Persist this DataFrame to disk (CSV or lightweight database) for repeatable analysis.

3. **Implement initial trend analysis**
   - Create functions to compute rolling averages and correlations for HRV, resting HR, steps, and sleep.
   - Leverage `garmin_utils.get_multi_day_stress_data()` and `get_weekly_averages()` for multi-day views.
   - Add simple console or file outputs summarizing detected trends.

4. **Write tests for the trend analysis**
   - Add unit tests in `smart_health_agent/tests/` that verify the trend functions work with sample data.
   - Ensure tests run without requiring external services when possible.

5. **Proceed with additional features only after trends are reliable**
   - Once Garmin data retrieval and trend analysis are stable, expand to:
     - Weather integration
     - TCM stress mapping
     - LLM-based recommendations
     - Visualization tools (e.g., Plotly graphs)
     - Future modules such as food, labs, or genetics
