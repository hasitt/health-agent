# Smart Health Agent - Code Review TODO

## High Priority Issues

### Database Improvements

- [ ] **Add unique constraints to sleep table** (database.py:78,100)
  - Add `UNIQUE(user_id, date)` constraint to `garmin_sleep` table
  - Prevents duplicate sleep records for same user/date

- [ ] **Implement connection pooling** (database.py:24-42)
  - Add SQLite connection pooling for better concurrent access
  - Consider using `sqlite3.connect()` with threading support

### Garmin Integration Enhancements

- [ ] **Make rate limiting configurable** (garmin_utils.py:963)
  - Move hardcoded delays (0.5s, 2.0s) to config.py
  - Allow user customization based on API usage patterns

- [ ] **Implement exponential backoff** (garmin_utils.py:972)
  - Replace simple retry with exponential backoff strategy
  - Improve resilience to temporary API failures

- [ ] **Enhance token cleanup logic** (garmin_utils.py:126-136)
  - Add more robust error handling for token directory cleanup
  - Handle edge cases like permission errors

### Configuration Issues

- [ ] **Fix empty OLLAMA_HOST** (config.py:14)
  - Set proper default value (e.g., "http://localhost:11434")
  - Add validation for required config values

## Medium Priority Issues

### Trend Analysis Improvements

- [ ] **Add timezone handling** (trend_analyzer.py throughout)
  - Implement proper timezone conversions for timestamp operations
  - Consider user's local timezone vs UTC storage

- [ ] **Make analysis thresholds configurable** (trend_analyzer.py:14,97,203)
  - Move hardcoded values to config (stress: 25, steps: 10000, duration: 30min)
  - Allow user customization of analysis parameters

- [ ] **Add statistical significance checks** (trend_analyzer.py:158,295)
  - Check sample sizes before calculating correlations
  - Add confidence intervals or p-values for meaningful analysis

- [ ] **Improve activity type matching** (trend_analyzer.py:220)
  - Use fuzzy string matching for activity type categorization
  - Add user-defined activity type mappings

### UI/UX Enhancements

- [ ] **Add input validation** (smart_health_ollama.py:429)
  - Validate user inputs in chat interface
  - Sanitize inputs before processing

- [ ] **Improve accessibility** (smart_health_ollama.py:714)
  - Add ARIA labels to UI components
  - Implement keyboard navigation support
  - Add screen reader compatibility

### Code Quality Improvements

- [ ] **Timestamp consistency** (database.py, garmin_utils.py)
  - Standardize timestamp format across all modules
  - Use consistent datetime handling patterns

- [ ] **Add data validation** (database.py:256,278,305)
  - Validate data types before database insertion
  - Add range checks for health metrics

## Low Priority Enhancements

### Performance Optimizations

- [ ] **Optimize large dataset queries** (trend_analyzer.py)
  - Add pagination for large date ranges
  - Implement query result caching

- [ ] **Database indexing review** (database.py:136-143)
  - Analyze query patterns and add composite indexes
  - Monitor query performance with EXPLAIN QUERY PLAN

### Documentation and Testing

- [ ] **Add comprehensive unit tests**
  - Test database operations with mock data
  - Test trend analysis calculations
  - Test error handling scenarios

- [ ] **Add API documentation**
  - Document all public functions and classes
  - Add usage examples for each module

- [ ] **Add logging configuration**
  - Make log levels configurable
  - Add structured logging for better monitoring

## Code Review Summary

### ✅ Strengths
- Excellent separation of concerns
- Proper parameterized queries (no SQL injection)
- Good error handling patterns
- Factual display principle compliance
- Comprehensive logging

### ⚠️ Areas for Improvement
- Missing database constraints
- Hardcoded configuration values
- Limited timezone handling
- Basic retry mechanisms

### 🎯 Overall Assessment
**Ready for next phase** with recommended improvements. The codebase demonstrates solid architectural foundation and defensive programming practices.

---

## Previous Development TODO (Completed)

### ✅ Completed Items

1. **Verify Garmin integration**
   - ✅ Run `python smart_health_agent/setup_garmin.py` to create a `.env` with credentials.
   - ✅ Use the "Test Garmin connection" option in the setup script.
   - ✅ Execute `python smart_health_agent/tests/test_garmin_fix.py` to confirm data can be retrieved.
   - ✅ Ensure token persistence in `.garmin_tokens` and handle login errors.

2. **Ingest Garmin data and store it in a unified structure**
   - ✅ Use `garmin_utils.get_comprehensive_health_data()` (and related helpers) to fetch daily health metrics.
   - ✅ Design a master DataFrame (e.g., with Pandas) that merges steps, HR, HRV, sleep, and stress data by timestamp.
   - ✅ Persist this DataFrame to disk (CSV or lightweight database) for repeatable analysis.

3. **Implement initial trend analysis**
   - ✅ Create functions to compute rolling averages and correlations for HRV, resting HR, steps, and sleep.
   - ✅ Leverage `garmin_utils.get_multi_day_stress_data()` and `get_weekly_averages()` for multi-day views.
   - ✅ Add simple console or file outputs summarizing detected trends.

4. **Write tests for the trend analysis**
   - ✅ Add unit tests in `smart_health_agent/tests/` that verify the trend functions work with sample data.
   - ✅ Ensure tests run without requiring external services when possible.

5. **Proceed with additional features only after trends are reliable**
   - ✅ Once Garmin data retrieval and trend analysis are stable, expand to:
     - ✅ Weather integration
     - ✅ TCM stress mapping
     - ✅ LLM-based recommendations
     - ✅ Visualization tools (e.g., Plotly graphs)
     - 🔄 Future modules such as food, labs, or genetics

---

*Code review TODO generated on 2025-07-10*
