#!/usr/bin/env python3
"""
Structured evaluation script for the LangChain Health Detective agent.

This script generates realistic health data, creates test cases based on known
functionalities and bugs, and provides automated pass/fail evaluation.
"""

import logging
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
import random
import numpy as np
from dataclasses import dataclass
import io
from contextlib import redirect_stderr, redirect_stdout

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

@dataclass
class TestCase:
    """Structure for individual test cases"""
    name: str
    description: str
    query: str
    expected_behavior: str
    expected_tool_calls: List[str] = None
    follow_up_query: Optional[str] = None
    expected_follow_up: Optional[str] = None
    data_requirements: List[str] = None
    bug_category: str = "functionality"

@dataclass
class TestResult:
    """Structure for test results"""
    test_case: TestCase
    passed: bool
    agent_response: str
    tool_calls_made: List[str]
    errors: List[str]
    execution_time: float

class HealthDataGenerator:
    """Generates realistic health data with patterns and correlations"""
    
    def __init__(self, days=60):
        self.days = days
        self.start_date = date.today() - timedelta(days=days)
        
    def generate_sample_data(self) -> Dict[str, Any]:
        """Generate comprehensive sample health data with realistic patterns"""
        
        logger.info(f"Generating {self.days} days of sample health data...")
        
        # Initialize data structure
        sample_data = {
            'sleep_data': [],
            'activity_data': [],
            'nutrition_data': [],
            'mood_data': [],
            'stress_data': []
        }
        
        # Generate correlated data with realistic patterns
        for i in range(self.days):
            current_date = self.start_date + timedelta(days=i)
            
            # Base patterns with some randomness
            day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
            is_weekend = day_of_week >= 5
            
            # Generate sleep data (with clear patterns)
            sleep_duration = self._generate_sleep_duration(day_of_week, i)
            sleep_score = self._generate_sleep_score(sleep_duration, i)
            deep_sleep_pct = self._generate_deep_sleep(sleep_duration, sleep_score)
            
            # Generate stress data (inversely correlated with sleep)
            avg_stress = self._generate_stress(sleep_duration, deep_sleep_pct, is_weekend, i)
            rhr = self._generate_rhr(avg_stress, sleep_duration, i)
            
            # Generate activity data
            steps = self._generate_steps(is_weekend, avg_stress, i)
            active_calories = self._generate_active_calories(steps)
            
            # Generate nutrition data (with some missing days)
            nutrition = self._generate_nutrition(i, avg_stress)
            
            # Generate mood data (correlated with sleep and stress)
            mood_data = self._generate_mood(sleep_score, avg_stress, i)
            
            # Add data with some missing values to test "data not available" scenarios
            if i % 15 != 0:  # Skip every 15th day for some metrics
                sample_data['sleep_data'].append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'sleep_duration_hours': sleep_duration,
                    'sleep_score': sleep_score,
                    'deep_sleep_percentage': deep_sleep_pct
                })
                
            if i % 20 != 0:  # Skip every 20th day for stress data
                sample_data['stress_data'].append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'avg_daily_stress': avg_stress,
                    'avg_daily_rhr': rhr
                })
            
            # Activity data - always present
            sample_data['activity_data'].append({
                'date': current_date.strftime('%Y-%m-%d'),
                'total_steps': steps,
                'active_calories': active_calories,
                'distance_km': round(steps * 0.0008, 2)  # Rough conversion
            })
            
            # Nutrition data - missing for some days
            if nutrition:
                sample_data['nutrition_data'].append(nutrition)
                
            # Mood data - present most days
            if i % 10 != 0:  # Skip every 10th day
                sample_data['mood_data'].append(mood_data)
        
        # Add metadata
        sample_data['metadata'] = {
            'generated_on': datetime.now().isoformat(),
            'total_days': self.days,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': (self.start_date + timedelta(days=self.days-1)).strftime('%Y-%m-%d'),
            'patterns': {
                'sleep_stress_correlation': 'negative (better sleep = lower stress)',
                'weekend_activity': 'reduced on weekends',
                'missing_data': 'nutrition: 20%, mood: 10%, sleep: 7%, stress: 5%'
            }
        }
        
        logger.info(f"Generated data: {len(sample_data['sleep_data'])} sleep records, "
                   f"{len(sample_data['activity_data'])} activity records, "
                   f"{len(sample_data['nutrition_data'])} nutrition records, "
                   f"{len(sample_data['mood_data'])} mood records")
        
        return sample_data
    
    def _generate_sleep_duration(self, day_of_week: int, day_index: int) -> float:
        """Generate realistic sleep duration with weekly patterns"""
        base_sleep = 7.5  # Base 7.5 hours
        weekend_bonus = 0.8 if day_of_week >= 5 else 0
        random_variation = random.normalvariate(0, 0.8)
        trend = -0.002 * day_index  # Slight decline over time
        return max(4.0, min(10.0, base_sleep + weekend_bonus + random_variation + trend))
    
    def _generate_sleep_score(self, duration: float, day_index: int) -> int:
        """Generate sleep score based on duration"""
        base_score = min(100, max(0, (duration - 4) * 15 + 50))
        random_variation = random.normalvariate(0, 10)
        return int(max(0, min(100, base_score + random_variation)))
    
    def _generate_deep_sleep(self, duration: float, score: int) -> float:
        """Generate deep sleep percentage"""
        base_pct = (score / 100) * 25 + random.normalvariate(0, 3)
        return max(5.0, min(35.0, base_pct))
    
    def _generate_stress(self, sleep_duration: float, deep_sleep_pct: float, is_weekend: bool, day_index: int) -> int:
        """Generate stress inversely correlated with sleep quality"""
        base_stress = 40
        sleep_factor = -(sleep_duration - 7) * 8  # Better sleep = lower stress
        deep_sleep_factor = -(deep_sleep_pct - 15) * 1.2
        weekend_reduction = -10 if is_weekend else 0
        random_variation = random.normalvariate(0, 8)
        
        stress = base_stress + sleep_factor + deep_sleep_factor + weekend_reduction + random_variation
        return int(max(10, min(80, stress)))
    
    def _generate_rhr(self, stress: int, sleep_duration: float, day_index: int) -> int:
        """Generate resting heart rate correlated with stress"""
        base_rhr = 65
        stress_factor = (stress - 40) * 0.3
        sleep_factor = -(sleep_duration - 7.5) * 2
        random_variation = random.normalvariate(0, 3)
        
        rhr = base_rhr + stress_factor + sleep_factor + random_variation
        return int(max(45, min(90, rhr)))
    
    def _generate_steps(self, is_weekend: bool, stress: int, day_index: int) -> int:
        """Generate daily steps with weekend and stress patterns"""
        base_steps = 7000
        weekend_factor = -1500 if is_weekend else 1000
        stress_factor = -(stress - 40) * 20  # High stress = less activity
        random_variation = random.normalvariate(0, 1500)
        
        steps = base_steps + weekend_factor + stress_factor + random_variation
        return max(500, int(steps))
    
    def _generate_active_calories(self, steps: int) -> int:
        """Generate active calories based on steps"""
        base_calories = steps * 0.04  # Rough conversion
        variation = random.normalvariate(0, 15)
        return max(20, int(base_calories + variation))
    
    def _generate_nutrition(self, day_index: int, stress: int) -> Optional[Dict]:
        """Generate nutrition data (with some missing days)"""
        if day_index % 5 == 0:  # Skip every 5th day
            return None
            
        current_date = self.start_date + timedelta(days=day_index)
        
        base_calories = 2200
        stress_eating = (stress - 40) * 10  # Higher stress = more calories
        random_variation = random.normalvariate(0, 200)
        
        total_calories = max(1200, base_calories + stress_eating + random_variation)
        
        return {
            'date': current_date.strftime('%Y-%m-%d'),
            'total_calories': int(total_calories),
            'protein_g': int(total_calories * 0.05),  # ~20% protein calories
            'carbohydrates_g': int(total_calories * 0.125),  # ~50% carb calories
            'fat_g': int(total_calories * 0.033),  # ~30% fat calories
            'caffeine_mg': random.randint(50, 300)
        }
    
    def _generate_mood(self, sleep_score: int, stress: int, day_index: int) -> Dict:
        """Generate mood data correlated with sleep and stress"""
        current_date = self.start_date + timedelta(days=day_index)
        
        # Mood (1-10 scale, correlated with sleep and inverse stress)
        base_mood = 6
        sleep_factor = (sleep_score - 60) * 0.03
        stress_factor = -(stress - 40) * 0.05
        random_variation = random.normalvariate(0, 1)
        
        mood = max(1, min(10, base_mood + sleep_factor + stress_factor + random_variation))
        
        # Energy similar pattern
        energy = max(1, min(10, mood + random.normalvariate(0, 0.8)))
        
        return {
            'date': current_date.strftime('%Y-%m-%d'),
            'mood': round(mood, 1),
            'energy': round(energy, 1),
            'stress': round(stress / 10, 1)  # Convert to 1-10 scale
        }

class AgentEvaluator:
    """Evaluates the LangChain agent against generated test cases"""
    
    def __init__(self):
        self.test_results = []
        
    def generate_test_cases(self, sample_data: Dict[str, Any]) -> List[TestCase]:
        """Generate comprehensive test cases based on sample data and known issues"""
        
        test_cases = []
        
        # Test Case 1: Data Availability Issues
        test_cases.append(TestCase(
            name="missing_data_handling",
            description="Test agent handles missing data correctly",
            query="What was my sleep duration on the days when data is missing?",
            expected_behavior="Should report specific days with no data available",
            expected_tool_calls=["get_health_data_summary"],
            bug_category="data_availability"
        ))
        
        # Test Case 2: Data Accuracy
        recent_sleep = [d for d in sample_data['sleep_data'][-7:]]
        if recent_sleep:
            avg_sleep = sum(d['sleep_duration_hours'] for d in recent_sleep) / len(recent_sleep)
            test_cases.append(TestCase(
                name="data_accuracy_sleep",
                description="Test agent provides accurate numerical data",
                query="What's my average sleep duration over the past 7 days?",
                expected_behavior=f"Should report approximately {avg_sleep:.1f} hours",
                expected_tool_calls=["get_health_data_summary"],
                data_requirements=["sleep_data"],
                bug_category="data_accuracy"
            ))
        
        # Test Case 3: Visualization Tool Usage
        test_cases.append(TestCase(
            name="visualization_generation",
            description="Test agent generates visualizations when requested",
            query="Show me a graph of my steps for the past 7 days",
            expected_behavior="Should generate and display a plot",
            expected_tool_calls=["generate_time_series_plots"],
            bug_category="visualization_failure"
        ))
        
        # Test Case 4: Follow-up Context Handling
        test_cases.append(TestCase(
            name="context_preservation",
            description="Test agent maintains context in follow-up questions",
            query="How was my sleep last night?",
            expected_behavior="Should provide specific sleep data",
            follow_up_query="What date was that from?",
            expected_follow_up="Should mention the specific date discussed",
            expected_tool_calls=["get_health_data_summary"],
            bug_category="context_handling"
        ))
        
        # Test Case 5: Date Range Calculations
        test_cases.append(TestCase(
            name="date_range_4_weeks",
            description="Test agent correctly calculates 4-week periods",
            query="What's my average stress over the last 4 weeks?",
            expected_behavior="Should use 28-day range and provide numerical average",
            expected_tool_calls=["get_health_data_summary"],
            bug_category="date_calculations"
        ))
        
        # Test Case 6: Correlation Analysis
        test_cases.append(TestCase(
            name="correlation_analysis",
            description="Test agent performs correlation analysis correctly",
            query="Is my deep sleep correlated with my stress over the last 30 days?",
            expected_behavior="Should mention correlation coefficient and p-value",
            expected_tool_calls=["perform_custom_analysis"],
            bug_category="statistical_analysis"
        ))
        
        # Test Case 7: Multi-metric Summary
        test_cases.append(TestCase(
            name="comprehensive_summary",
            description="Test agent provides comprehensive health summary",
            query="Give me a complete overview of my health trends",
            expected_behavior="Should include sleep, activity, nutrition, and mood data",
            expected_tool_calls=["get_health_data_summary"],
            bug_category="comprehensive_analysis"
        ))
        
        # Test Case 8: Specific Metric Trends
        test_cases.append(TestCase(
            name="rhr_trend_analysis",
            description="Test agent analyzes specific metric trends",
            query="How has my resting heart rate been trending?",
            expected_behavior="Should provide RHR trend analysis with specific numbers",
            expected_tool_calls=["get_health_data_summary"],
            bug_category="trend_analysis"
        ))
        
        # Test Case 9: Plot Generation Follow-up
        test_cases.append(TestCase(
            name="plot_generation_followup",
            description="Test agent generates plots when user confirms",
            query="Can you analyze my activity patterns?",
            expected_behavior="Should offer to create visualizations",
            follow_up_query="yes",
            expected_follow_up="Should generate and display activity plots",
            expected_tool_calls=["get_health_data_summary", "generate_time_series_plots"],
            bug_category="plot_generation"
        ))
        
        # Test Case 10: Edge Case - Very Recent Data
        test_cases.append(TestCase(
            name="recent_data_query",
            description="Test agent handles very recent data queries",
            query="How did I sleep yesterday compared to my average?",
            expected_behavior="Should compare yesterday's sleep to historical average",
            expected_tool_calls=["get_health_data_summary"],
            bug_category="recent_data_handling"
        ))
        
        logger.info(f"Generated {len(test_cases)} test cases covering various scenarios")
        return test_cases
    
    def setup_test_environment(self, sample_data: Dict[str, Any]) -> bool:
        """Set up test environment by inserting sample data into database"""
        try:
            # Import database and health app modules
            from database import db
            
            # Connect to database
            db.connect()
            db.create_tables()
            
            # Clear existing test data for user_id=999 (test user)
            conn = db.get_connection()
            cursor = conn.cursor()
            
            test_user_id = 999
            cursor.execute("DELETE FROM garmin_sleep WHERE user_id = ?", (test_user_id,))
            cursor.execute("DELETE FROM garmin_daily_summary WHERE user_id = ?", (test_user_id,))
            cursor.execute("DELETE FROM food_log_daily WHERE user_id = ?", (test_user_id,))
            cursor.execute("DELETE FROM subjective_wellbeing WHERE user_id = ?", (test_user_id,))
            
            # Insert sample data
            for sleep_record in sample_data['sleep_data']:
                sleep_duration_hours = sleep_record['sleep_duration_hours']
                total_sleep_minutes = int(sleep_duration_hours * 60)
                deep_sleep_pct = sleep_record.get('deep_sleep_percentage', 20)
                
                cursor.execute("""
                    INSERT INTO garmin_sleep 
                    (user_id, date, sleep_duration_hours, sleep_score, total_sleep_minutes, 
                     deep_sleep_minutes, light_sleep_minutes, rem_sleep_minutes, awake_minutes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    test_user_id, sleep_record['date'], sleep_duration_hours,
                    sleep_record['sleep_score'], total_sleep_minutes,
                    int(total_sleep_minutes * deep_sleep_pct / 100),  # Deep sleep minutes
                    int(total_sleep_minutes * 0.6),  # Light sleep minutes
                    int(total_sleep_minutes * 0.2),  # REM sleep minutes
                    int(total_sleep_minutes * 0.05)  # Awake minutes
                ))
            
            for activity_record in sample_data['activity_data']:
                cursor.execute("""
                    INSERT INTO garmin_daily_summary
                    (user_id, date, total_steps, active_calories, distance_km, avg_daily_rhr, avg_daily_stress)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    test_user_id, activity_record['date'], activity_record['total_steps'],
                    activity_record['active_calories'], activity_record['distance_km'],
                    70, 40  # Default values for RHR and stress (will be updated below)
                ))
            
            # Add stress data to existing records
            for stress_record in sample_data['stress_data']:
                cursor.execute("""
                    UPDATE garmin_daily_summary 
                    SET avg_daily_rhr = ?, avg_daily_stress = ?
                    WHERE user_id = ? AND date = ?
                """, (
                    stress_record['avg_daily_rhr'], stress_record['avg_daily_stress'],
                    test_user_id, stress_record['date']
                ))
            
            for nutrition_record in sample_data['nutrition_data']:
                cursor.execute("""
                    INSERT INTO food_log_daily
                    (user_id, date, total_calories, protein_g, carbohydrates_g, fat_g, caffeine_mg)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    test_user_id, nutrition_record['date'], nutrition_record['total_calories'],
                    nutrition_record['protein_g'], nutrition_record['carbohydrates_g'],
                    nutrition_record['fat_g'], nutrition_record['caffeine_mg']
                ))
            
            for mood_record in sample_data['mood_data']:
                cursor.execute("""
                    INSERT INTO subjective_wellbeing
                    (user_id, date, mood, energy, stress, sleep_quality, focus, motivation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    test_user_id, mood_record['date'], mood_record['mood'],
                    mood_record['energy'], mood_record['stress'], 
                    7, 7, 7  # Default values for other fields
                ))
            
            conn.commit()
            logger.info(f"Successfully inserted test data for user {test_user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set up test environment: {e}")
            return False
    
    def run_evaluation(self, test_cases: List[TestCase]) -> List[TestResult]:
        """Run all test cases and evaluate agent responses"""
        
        try:
            # Import agent modules
            from smart_health_ollama import (
                respond_to_chat, LANGCHAIN_AGENT, LANGCHAIN_AVAILABLE,
                current_user_id, initialize_llm_client, initialize_langchain_agent
            )
            from database import db
            
            # Initialize database connection
            db.connect()
            
            # Initialize LLM and agent if not already done
            if not LANGCHAIN_AGENT or not LANGCHAIN_AVAILABLE:
                logger.info("Initializing LangChain agent for testing...")
                llm_success = initialize_llm_client()
                agent_success = initialize_langchain_agent()
                
                if not agent_success:
                    logger.error("Failed to initialize LangChain agent for testing")
                    return []
                else:
                    logger.info("LangChain agent initialized successfully for testing")
            
            # Temporarily change user_id to test user
            original_user_id = current_user_id
            import smart_health_ollama
            smart_health_ollama.current_user_id = 999  # Test user
            
            # Re-check agent availability after initialization
            from smart_health_ollama import LANGCHAIN_AGENT, LANGCHAIN_AVAILABLE
            if not LANGCHAIN_AGENT or not LANGCHAIN_AVAILABLE:
                logger.error("LangChain agent is still not available after initialization")
                return []
            
            results = []
            
            for i, test_case in enumerate(test_cases, 1):
                logger.info(f"Running test {i}/{len(test_cases)}: {test_case.name}")
                
                start_time = datetime.now()
                result = self._run_single_test(test_case)
                end_time = datetime.now()
                
                result.execution_time = (end_time - start_time).total_seconds()
                results.append(result)
                
                # Log result
                status = "✅ PASS" if result.passed else "❌ FAIL"
                logger.info(f"  {status} - {test_case.name} ({result.execution_time:.2f}s)")
                
                if not result.passed:
                    for error in result.errors:
                        logger.warning(f"    Error: {error}")
                        
            # Restore original user_id
            smart_health_ollama.current_user_id = original_user_id
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to run evaluation: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case and evaluate the result"""
        
        from smart_health_ollama import respond_to_chat, LANGCHAIN_AGENT
        
        errors = []
        tool_calls_made = []
        
        try:
            # Capture logging output to detect tool calls
            log_stream = io.StringIO()
            log_handler = logging.StreamHandler(log_stream)
            log_handler.setLevel(logging.INFO)
            
            # Add handler to capture logs
            logging.getLogger('agent_tools').addHandler(log_handler)
            logging.getLogger('smart_health_ollama').addHandler(log_handler)
            
            # Skip monkey-patching for now, rely on log-based detection
            
            # Run initial query
            chat_history = []
            empty_msg, updated_history, plot = respond_to_chat(test_case.query, chat_history)
            
            # Extract tool calls from logs
            log_output = log_stream.getvalue()
            if "Getting health data summary" in log_output:
                if "get_health_data_summary" not in tool_calls_made:
                    tool_calls_made.append("get_health_data_summary")
            if "Generating time series plots" in log_output:
                if "generate_time_series_plots" not in tool_calls_made:
                    tool_calls_made.append("generate_time_series_plots")
            if "Performing" in log_output and "analysis" in log_output:
                if "perform_custom_analysis" not in tool_calls_made:
                    tool_calls_made.append("perform_custom_analysis")
            
            # Remove the log handler
            logging.getLogger('agent_tools').removeHandler(log_handler)
            logging.getLogger('smart_health_ollama').removeHandler(log_handler)
            
            if not updated_history or len(updated_history) < 2:
                errors.append("Agent did not provide a response")
                return TestResult(test_case, False, "", [], errors, 0)
            
            agent_response = updated_history[-1]['content']
            
            # Also check for tool calls in the response text and logs
            response_text = str(updated_history).lower()
            if "getting health data summary" in response_text or "get_health_data_summary" in response_text:
                if "get_health_data_summary" not in tool_calls_made:
                    tool_calls_made.append("get_health_data_summary")
            if "generating time series plots" in response_text or "generate_time_series_plots" in response_text:
                if "generate_time_series_plots" not in tool_calls_made:
                    tool_calls_made.append("generate_time_series_plots")
            if "performing" in response_text and "analysis" in response_text:
                if "perform_custom_analysis" not in tool_calls_made:
                    tool_calls_made.append("perform_custom_analysis")
                
            # Run follow-up query if specified
            follow_up_response = ""
            if test_case.follow_up_query:
                empty_msg, updated_history, plot = respond_to_chat(
                    test_case.follow_up_query, updated_history
                )
                if updated_history and len(updated_history) >= 2:
                    follow_up_response = updated_history[-1]['content']
            
            # Evaluate the response
            passed = self._evaluate_response(
                test_case, agent_response, follow_up_response, 
                tool_calls_made, plot, errors
            )
            
            full_response = agent_response
            if follow_up_response:
                full_response += "\n[FOLLOW-UP]\n" + follow_up_response
                
            return TestResult(test_case, passed, full_response, tool_calls_made, errors, 0)
            
        except Exception as e:
            errors.append(f"Exception during test execution: {str(e)}")
            return TestResult(test_case, False, "", [], errors, 0)
    
    def _evaluate_response(self, test_case: TestCase, response: str, follow_up: str,
                          tool_calls: List[str], plot: Any, errors: List[str]) -> bool:
        """Evaluate agent response against expected behavior"""
        
        passed = True
        response_lower = response.lower()
        follow_up_lower = follow_up.lower() if follow_up else ""
        
        # Check expected tool calls
        if test_case.expected_tool_calls:
            for expected_tool in test_case.expected_tool_calls:
                if expected_tool not in tool_calls:
                    errors.append(f"Expected tool '{expected_tool}' was not called")
                    passed = False
        
        # Bug-specific checks
        if test_case.bug_category == "data_availability":
            # Look for mentions of data availability or provide reasonable data summary
            if ("data not available" not in response_lower and "no data" not in response_lower and 
                "missing" not in response_lower and "unavailable" not in response_lower and
                len(response.strip()) < 100):  # Allow for data summaries as valid responses
                errors.append("Did not properly address data availability question")
                passed = False
                    
        elif test_case.bug_category == "data_accuracy":
            # Look for numerical values in response
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            if not numbers:
                errors.append("No numerical data provided in response")
                passed = False
                
        elif test_case.bug_category == "visualization_failure":
            # Check if plot was generated OR agent attempted to generate plot
            plot_generated = (plot is not None or 
                            "generate_time_series_plots" in tool_calls or
                            "generating" in response_lower)
            if not plot_generated:
                errors.append("No plot generation attempt detected")
                passed = False
            if ("graph" not in response_lower and "plot" not in response_lower and 
                "chart" not in response_lower and "visualization" not in response_lower):
                errors.append("Response does not acknowledge visualization request")
                # Don't fail for this alone if plot was generated
                
        elif test_case.bug_category == "context_handling":
            if test_case.follow_up_query and follow_up:
                if "date" not in follow_up_lower and len(follow_up.strip()) < 10:
                    errors.append("Follow-up response lacks context awareness")
                    passed = False
                    
        elif test_case.bug_category == "statistical_analysis":
            if "correlation" not in response_lower:
                errors.append("Statistical analysis response missing correlation mention")
                passed = False
            # Look for correlation coefficient or p-value
            if not any(term in response_lower for term in ["coefficient", "p-value", "significant"]):
                errors.append("Statistical analysis missing key statistical terms")
                passed = False
                
        elif test_case.bug_category == "plot_generation":
            if test_case.follow_up_query == "yes" and follow_up:
                if plot is None:
                    errors.append("Plot not generated after user confirmation")
                    passed = False
                    
        # General checks
        if len(response.strip()) < 50:
            errors.append("Response too short, likely incomplete")
            passed = False
            
        if "error" in response_lower or "failed" in response_lower:
            errors.append("Response indicates error or failure")
            passed = False
            
        return passed
    
    def generate_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        if not results:
            return {"error": "No test results to report"}
            
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Group by bug category
        category_results = {}
        for result in results:
            category = result.test_case.bug_category
            if category not in category_results:
                category_results[category] = {"passed": 0, "failed": 0, "total": 0}
            
            category_results[category]["total"] += 1
            if result.passed:
                category_results[category]["passed"] += 1
            else:
                category_results[category]["failed"] += 1
        
        # Calculate average execution time
        avg_execution_time = sum(r.execution_time for r in results) / total_tests
        
        # Identify most common errors
        all_errors = []
        for result in results:
            all_errors.extend(result.errors)
        
        error_frequency = {}
        for error in all_errors:
            error_frequency[error] = error_frequency.get(error, 0) + 1
        
        most_common_errors = sorted(error_frequency.items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.1f}%",
                "avg_execution_time": f"{avg_execution_time:.2f}s"
            },
            "category_breakdown": category_results,
            "most_common_errors": most_common_errors,
            "failed_tests": [
                {
                    "name": r.test_case.name,
                    "description": r.test_case.description,
                    "errors": r.errors,
                    "execution_time": f"{r.execution_time:.2f}s"
                }
                for r in results if not r.passed
            ],
            "detailed_results": [
                {
                    "test_name": r.test_case.name,
                    "passed": r.passed,
                    "query": r.test_case.query,
                    "response_length": len(r.agent_response),
                    "tools_used": r.tool_calls_made,
                    "execution_time": f"{r.execution_time:.2f}s",
                    "errors": r.errors
                }
                for r in results
            ]
        }
        
        return report

def main():
    """Main evaluation function"""
    
    logger.info("Starting LangChain Health Detective Agent Evaluation")
    logger.info("=" * 60)
    
    # Initialize components
    data_generator = HealthDataGenerator(days=60)
    evaluator = AgentEvaluator()
    
    try:
        # Step 1: Generate sample data
        logger.info("Step 1: Generating sample health data...")
        sample_data = data_generator.generate_sample_data()
        
        # Step 2: Set up test environment
        logger.info("Step 2: Setting up test environment...")
        if not evaluator.setup_test_environment(sample_data):
            logger.error("Failed to set up test environment")
            return
        
        # Step 3: Generate test cases
        logger.info("Step 3: Generating test cases...")
        test_cases = evaluator.generate_test_cases(sample_data)
        
        # Step 4: Run evaluation
        logger.info("Step 4: Running evaluation...")
        results = evaluator.run_evaluation(test_cases)
        
        if not results:
            logger.error("No test results obtained")
            return
        
        # Step 5: Generate and display report
        logger.info("Step 5: Generating evaluation report...")
        report = evaluator.generate_report(results)
        
        # Display report
        print("\n" + "=" * 60)
        print("LANGCHAIN HEALTH DETECTIVE AGENT - EVALUATION REPORT")
        print("=" * 60)
        
        summary = report["summary"]
        print(f"📊 SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed']} ✅")
        print(f"   Failed: {summary['failed']} ❌")
        print(f"   Success Rate: {summary['success_rate']}")
        print(f"   Avg Execution Time: {summary['avg_execution_time']}")
        
        print(f"\n📂 RESULTS BY CATEGORY:")
        for category, stats in report["category_breakdown"].items():
            success_rate = (stats['passed'] / stats['total']) * 100
            print(f"   {category.replace('_', ' ').title()}: "
                  f"{stats['passed']}/{stats['total']} ({success_rate:.0f}%)")
        
        if report["most_common_errors"]:
            print(f"\n⚠️  MOST COMMON ERRORS:")
            for error, count in report["most_common_errors"]:
                print(f"   • {error} ({count}x)")
        
        if report["failed_tests"]:
            print(f"\n❌ FAILED TESTS DETAILS:")
            for failed_test in report["failed_tests"]:
                print(f"   • {failed_test['name']}: {failed_test['description']}")
                for error in failed_test['errors']:
                    print(f"     - {error}")
        
        # Save detailed report to file
        report_file = Path(current_dir) / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        print("=" * 60)
        
        return report
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()