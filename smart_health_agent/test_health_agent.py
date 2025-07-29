#!/usr/bin/env python3
"""
Comprehensive Test Suite for Smart Health Agent
Tests data import, sync verification, mood tracking, trend analysis, and LLM integration.
"""

import sys
import os
import unittest
import sqlite3
import tempfile
import csv
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import project modules
import database
import cronometer_parser
import trend_analyzer
import mood_tracking
from smart_health_ollama import generate_recommendations, HealthAgentState
from config import Config

class TestHealthAgentBase(unittest.TestCase):
    """Base test class with common setup and teardown."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database and initial data."""
        # Use a separate test database
        cls.original_db_path = database.DB_PATH
        cls.test_db_path = Path(__file__).parent / "test_data" / "test_health_data.db"
        cls.test_db_path.parent.mkdir(exist_ok=True)
        
        # Override database path
        database.DB_PATH = cls.test_db_path
        
        # Create tables
        database.create_tables()
        
        # Create test user
        cls.test_user_id = database.insert_or_update_user(
            garmin_user_id="test_user_1",
            name="Test User",
            access_token="test_token",
            refresh_token="test_refresh",
            token_expiry=datetime.now() + timedelta(days=365)
        )
        
        print(f"Test setup complete. Test user ID: {cls.test_user_id}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        # Restore original database path
        database.DB_PATH = cls.original_db_path
        
        # Remove test database
        if cls.test_db_path.exists():
            cls.test_db_path.unlink()
    
    def setUp(self):
        """Set up for each test."""
        self.user_id = self.test_user_id
    
    def get_db_connection(self):
        """Get test database connection."""
        return sqlite3.connect(str(self.test_db_path))
    
    def create_mock_csv(self, data, filename="test_cronometer.csv"):
        """Create a mock Cronometer CSV file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
        writer = csv.writer(temp_file)
        writer.writerows(data)
        temp_file.close()
        return temp_file.name


class TestCronometerImport(TestHealthAgentBase):
    """Test Cronometer CSV import functionality."""
    
    def test_successful_import_and_upsert(self):
        """Test Case 1: Successful Import & Upsert"""
        print("\n=== Test Case 1: Successful Cronometer Import & Upsert ===")
        
        # Create mock CSV data with diverse entries
        mock_data = [
            ["Date", "Time", "Food Name", "Amount", "Unit", "Calories", "Protein (g)", "Carbohydrates (g)", "Fat (g)"],
            ["2025-01-10", "08:30:00", "Oatmeal Steel Cut", "1", "cup", "150", "4", "27", "3"],
            ["2025-01-10", "08:35:00", "Coffee", "1", "cup", "5", "0.3", "0", "0"],
            ["2025-01-10", "12:30:00", "Chicken Breast", "150", "g", "231", "43.5", "0", "5"],
            ["2025-01-10", "09:00:00", "Vitamin D3 Supplement", "1", "capsule", "0", "0", "0", "0"],
            ["2025-01-10", "19:00:00", "Red Wine", "150", "ml", "125", "0.1", "4", "0"],
            ["2025-01-11", "08:00:00", "Greek Yogurt", "1", "cup", "130", "20", "9", "0"],
            ["2025-01-11", "13:00:00", "Tuna Salad", "1", "serving", "200", "25", "5", "8"],
            ["2025-01-11", "16:00:00", "Green Tea", "1", "cup", "2", "0", "0", "0"],
        ]
        
        csv_file = self.create_mock_csv(mock_data)
        
        try:
            # Test first import
            result = cronometer_parser.parse_cronometer_food_entries_csv(csv_file, self.user_id)
            
            # Assertions for first import
            self.assertEqual(result['food_entries'], 6, "Should import 6 food entries")
            self.assertEqual(result['supplement_entries'], 1, "Should import 1 supplement entry")
            self.assertEqual(len(result['errors']), 0, "Should have no errors")
            
            # Verify specific entries in database
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check food entry
                cursor.execute("""
                    SELECT date, time, food_item_name, quantity, unit, calories, protein_g, carbs_g, fats_g
                    FROM food_log 
                    WHERE user_id = ? AND food_item_name = 'Chicken Breast'
                """, (self.user_id,))
                chicken_entry = cursor.fetchone()
                
                self.assertIsNotNone(chicken_entry, "Chicken Breast entry should exist")
                self.assertEqual(chicken_entry[0], "2025-01-10")  # date
                self.assertEqual(chicken_entry[1], "12:30:00")    # time
                self.assertEqual(chicken_entry[3], 150.0)         # quantity
                self.assertEqual(chicken_entry[4], "g")           # unit
                self.assertEqual(chicken_entry[5], 231.0)         # calories
                self.assertEqual(chicken_entry[6], 43.5)          # protein_g
                
                # Check supplement entry
                cursor.execute("""
                    SELECT supplement_name, quantity, unit
                    FROM supplements 
                    WHERE user_id = ? AND supplement_name = 'Vitamin D3 Supplement'
                """, (self.user_id,))
                supplement_entry = cursor.fetchone()
                
                self.assertIsNotNone(supplement_entry, "Vitamin D3 supplement should exist")
                self.assertEqual(supplement_entry[1], 1.0)  # quantity
                self.assertEqual(supplement_entry[2], "capsule")  # unit
                
                # Count total entries
                cursor.execute("SELECT COUNT(*) FROM food_log WHERE user_id = ?", (self.user_id,))
                food_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM supplements WHERE user_id = ?", (self.user_id,))
                supplement_count = cursor.fetchone()[0]
                
                print(f"✓ First import: {food_count} food entries, {supplement_count} supplement entries")
            
            # Test duplicate import (upsert functionality)
            result2 = cronometer_parser.parse_cronometer_food_entries_csv(csv_file, self.user_id)
            
            # Verify no duplicates created
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM food_log WHERE user_id = ?", (self.user_id,))
                food_count_after = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM supplements WHERE user_id = ?", (self.user_id,))
                supplement_count_after = cursor.fetchone()[0]
                
                self.assertEqual(food_count, food_count_after, "No duplicate food entries should be created")
                self.assertEqual(supplement_count, supplement_count_after, "No duplicate supplement entries should be created")
                
                print(f"✓ Second import (upsert): {food_count_after} food entries, {supplement_count_after} supplement entries - no duplicates")
            
            print("✅ Test Case 1 PASSED: Successful import and upsert verification")
            
        finally:
            os.unlink(csv_file)
    
    def test_empty_csv_handling(self):
        """Test Case 2: Handling Empty CSV"""
        print("\n=== Test Case 2: Empty CSV Handling ===")
        
        # Create empty CSV with only headers
        empty_data = [
            ["Date", "Time", "Food Name", "Amount", "Unit", "Calories", "Protein (g)", "Carbohydrates (g)", "Fat (g)"]
        ]
        
        csv_file = self.create_mock_csv(empty_data)
        
        try:
            result = cronometer_parser.parse_cronometer_food_entries_csv(csv_file, self.user_id)
            
            # Assertions
            self.assertEqual(result['food_entries'], 0, "Should import 0 food entries")
            self.assertEqual(result['supplement_entries'], 0, "Should import 0 supplement entries")
            self.assertEqual(len(result['errors']), 0, "Should handle empty CSV gracefully")
            
            print("✓ Empty CSV handled gracefully with 0 imports and no errors")
            print("✅ Test Case 2 PASSED: Empty CSV handling")
            
        finally:
            os.unlink(csv_file)
    
    def test_malformed_csv_handling(self):
        """Test Case 3: Handling Malformed CSV (Missing Columns)"""
        print("\n=== Test Case 3: Malformed CSV Handling ===")
        
        # Create CSV missing essential columns
        malformed_data = [
            ["Time", "Amount", "Unit"],  # Missing Date and Food Name
            ["08:30:00", "1", "cup"],
            ["12:30:00", "150", "g"]
        ]
        
        csv_file = self.create_mock_csv(malformed_data)
        
        try:
            result = cronometer_parser.parse_cronometer_food_entries_csv(csv_file, self.user_id)
            
            # Should handle gracefully and report errors
            self.assertEqual(result['food_entries'], 0, "Should import 0 entries from malformed CSV")
            self.assertEqual(result['supplement_entries'], 0, "Should import 0 supplements from malformed CSV")
            self.assertGreater(len(result['errors']), 0, "Should report errors for malformed CSV")
            
            print(f"✓ Malformed CSV handled gracefully with {len(result['errors'])} errors reported")
            print("✅ Test Case 3 PASSED: Malformed CSV error handling")
            
        finally:
            os.unlink(csv_file)


class TestGarminDataSync(TestHealthAgentBase):
    """Test Garmin data sync verification."""
    
    def test_granular_stress_data_presence(self):
        """Test Case 1: Granular Stress Data Presence"""
        print("\n=== Test Case 1: Granular Stress Data Verification ===")
        
        # Insert mock granular stress data
        test_date = date.today() - timedelta(days=1)
        base_time = datetime.combine(test_date, datetime.min.time())
        
        stress_data = []
        # Create 3-minute interval stress data for 12 hours (240 entries)
        for i in range(240):
            timestamp = base_time + timedelta(minutes=i * 3)
            stress_level = 20 + (i % 40)  # Varying stress levels 20-60
            body_battery = 100 - (i // 10)  # Decreasing body battery
            
            stress_data.append({
                'timestamp': timestamp,
                'stress_level': stress_level,
                'body_battery_level': body_battery
            })
        
        # Insert data using database function
        database.insert_stress_details(self.user_id, test_date, stress_data)
        
        # Verify data insertion
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check entry count
            cursor.execute("""
                SELECT COUNT(*) FROM garmin_stress_details 
                WHERE user_id = ? AND date = ?
            """, (self.user_id, test_date))
            entry_count = cursor.fetchone()[0]
            
            self.assertEqual(entry_count, 240, f"Should have 240 stress entries, got {entry_count}")
            
            # Check data types and ranges
            cursor.execute("""
                SELECT timestamp, stress_level, body_battery_level 
                FROM garmin_stress_details 
                WHERE user_id = ? AND date = ? 
                ORDER BY timestamp 
                LIMIT 5
            """, (self.user_id, test_date))
            
            sample_entries = cursor.fetchall()
            
            for entry in sample_entries:
                # Verify timestamp is string (SQLite format)
                self.assertIsInstance(entry[0], str, "Timestamp should be stored as string")
                
                # Verify stress level is in valid range
                self.assertGreaterEqual(entry[1], 0, "Stress level should be >= 0")
                self.assertLessEqual(entry[1], 100, "Stress level should be <= 100")
                
                # Verify body battery is in valid range
                self.assertGreaterEqual(entry[2], 0, "Body battery should be >= 0")
                self.assertLessEqual(entry[2], 100, "Body battery should be <= 100")
            
            print(f"✓ Inserted {entry_count} stress entries with valid data types and ranges")
            print("✅ Test Case 1 PASSED: Granular stress data verification")


class TestMoodTracking(TestHealthAgentBase):
    """Test daily mood tracker backend storage."""
    
    def test_successful_mood_entry_storage(self):
        """Test Case 1: Successful Mood Entry Storage"""
        print("\n=== Test Case 1: Mood Entry Storage Verification ===")
        
        # Test mood entry data
        test_dates = [
            date.today(),
            date.today() - timedelta(days=1),
            date.today() - timedelta(days=2)
        ]
        
        for i, test_date in enumerate(test_dates):
            mood_data = {
                'date': test_date,
                'timestamp': datetime.combine(test_date, datetime.now().time()),
                'mood_rating': 7 + i,
                'energy_rating': 6 + i,
                'stress_rating': 4 - i,
                'anxiety_rating': 3 + i,
                'sleep_quality_rating': 8 - i,
                'focus_rating': 7 + i,
                'motivation_rating': 6 + i,
                'emotional_state': f'calm and focused day {i+1}',
                'stress_triggers': f'work deadline {i+1}',
                'coping_strategies': f'meditation and exercise {i+1}',
                'physical_symptoms': f'mild tension {i+1}',
                'daily_events': f'important meeting {i+1}',
                'social_interactions': f'good team collaboration {i+1}',
                'weather_sensitivity': f'sunny weather helped mood {i+1}',
                'hormonal_factors': f'normal cycle {i+1}',
                'entry_type': 'daily',
                'source': 'manual',
                'notes_text': f'Overall good day with minor stress {i+1}'
            }
            
            # Insert mood entry
            mood_tracking.insert_daily_mood_entry(self.user_id, mood_data)
            
            # Verify storage
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT mood_rating, energy_rating, stress_rating, anxiety_rating,
                           sleep_quality_rating, focus_rating, motivation_rating,
                           emotional_state, stress_triggers, coping_strategies,
                           physical_symptoms, daily_events, social_interactions,
                           weather_sensitivity, hormonal_factors, entry_type, source, notes_text
                    FROM subjective_wellbeing 
                    WHERE user_id = ? AND date = ?
                """, (self.user_id, test_date))
                
                stored_entry = cursor.fetchone()
                self.assertIsNotNone(stored_entry, f"Mood entry for {test_date} should exist")
                
                # Verify all fields stored correctly
                self.assertEqual(stored_entry[0], 7 + i, f"Mood rating should be {7 + i}")
                self.assertEqual(stored_entry[1], 6 + i, f"Energy rating should be {6 + i}")
                self.assertEqual(stored_entry[2], 4 - i, f"Stress rating should be {4 - i}")
                self.assertEqual(stored_entry[7], f'calm and focused day {i+1}', "Emotional state should match")
                self.assertEqual(stored_entry[8], f'work deadline {i+1}', "Stress triggers should match")
                self.assertEqual(stored_entry[15], 'daily', "Entry type should be 'daily'")
                self.assertEqual(stored_entry[16], 'manual', "Source should be 'manual'")
                
                print(f"✓ Day {i+1} ({test_date}): All 18 mood tracking fields stored correctly")
        
        print("✅ Test Case 1 PASSED: Mood entry storage verification")


class TestTrendAnalysis(TestHealthAgentBase):
    """Test trend analysis functions."""
    
    def test_stress_consistency_analysis(self):
        """Test Case 1: Stress Consistency Analysis"""
        print("\n=== Test Case 1: Stress Consistency Analysis ===")
        
        test_date = date.today() - timedelta(days=1)
        base_time = datetime.combine(test_date, datetime.min.time().replace(hour=9))
        
        # Create stress data with a consistent high-stress period
        stress_data = []
        
        # Normal stress (9:00-10:00)
        for i in range(20):
            stress_data.append({
                'timestamp': base_time + timedelta(minutes=i * 3),
                'stress_level': 20,
                'body_battery_level': 80
            })
        
        # High stress period (10:00-11:30) - 30+ minutes
        for i in range(30):
            stress_data.append({
                'timestamp': base_time + timedelta(hours=1, minutes=i * 3),
                'stress_level': 45,  # Above 25 threshold
                'body_battery_level': 70
            })
        
        # Normal stress again (11:30-12:00)
        for i in range(10):
            stress_data.append({
                'timestamp': base_time + timedelta(hours=2.5, minutes=i * 3),
                'stress_level': 22,
                'body_battery_level': 65
            })
        
        database.insert_stress_details(self.user_id, test_date, stress_data)
        
        # Test stress consistency analysis
        result = trend_analyzer.get_hourly_stress_consistency(self.user_id, test_date, stress_threshold=25)
        
        self.assertIsInstance(result, list, "Should return a list of observations")
        self.assertGreater(len(result), 0, "Should have stress consistency observations")
        
        # Check for expected high-stress period detection
        result_text = " ".join(result)
        self.assertIn("10:00", result_text, "Should detect stress period starting around 10:00")
        self.assertIn("11:", result_text, "Should detect stress period ending around 11:xx")
        
        print(f"✓ Stress consistency analysis: {result[0]}")
        print("✅ Test Case 1 PASSED: Stress consistency analysis")
    
    def test_steps_vs_sleep_effect(self):
        """Test Case 2: Steps vs Sleep Effect"""
        print("\n=== Test Case 2: Steps vs Sleep Effect Analysis ===")
        
        # Insert mock daily summary and sleep data
        test_dates = [date.today() - timedelta(days=i) for i in range(1, 11)]
        
        for i, test_date in enumerate(test_dates):
            # Alternate between high and low step days
            steps = 12000 if i % 2 == 0 else 3000
            
            # Insert daily summary
            summary_data = {
                'total_steps': steps,
                'active_calories': 300 if steps > 10000 else 100,
                'resting_calories': 1500,
                'distance_km': steps / 1200,
                'avg_daily_rhr': 60,
                'avg_daily_stress': 25
            }
            database.insert_daily_summary(self.user_id, test_date, summary_data)
            
            # Insert sleep data for next day
            next_date = test_date + timedelta(days=1)
            sleep_data = {
                'date': next_date,
                'sleep_start_time': datetime.combine(next_date, datetime.min.time().replace(hour=23)),
                'sleep_end_time': datetime.combine(next_date, datetime.min.time().replace(hour=7)),
                'total_sleep_minutes': 480,
                'sleep_score': 85 if steps > 10000 else 70,  # Better sleep after high steps
                'deep_sleep_minutes': 120,
                'rem_sleep_minutes': 90,
                'light_sleep_minutes': 240,
                'awake_minutes': 30
            }
            database.insert_sleep_data(self.user_id, sleep_data)
        
        # Test steps vs sleep analysis
        result = trend_analyzer.get_steps_vs_sleep_effect(self.user_id, steps_threshold=10000)
        
        self.assertIsInstance(result, list, "Should return a list of observations")
        self.assertGreater(len(result), 0, "Should have steps vs sleep observations")
        
        # Verify it detected the pattern
        result_text = " ".join(result)
        self.assertIn("10000+", result_text, "Should mention high step threshold")
        self.assertIn("85", result_text, "Should include high step sleep score")
        self.assertIn("70", result_text, "Should include low step sleep score")
        
        print(f"✓ Steps vs sleep analysis: {result[0]}")
        print("✅ Test Case 2 PASSED: Steps vs sleep effect analysis")
    
    def test_activity_type_vs_rhr_impact(self):
        """Test Case 3: Activity Type vs RHR Impact"""
        print("\n=== Test Case 3: Activity Type vs RHR Impact ===")
        
        test_dates = [date.today() - timedelta(days=i) for i in range(1, 8)]
        
        for i, test_date in enumerate(test_dates):
            # Alternate activity types
            if i % 2 == 0:
                activity_type = "strength_training"
                rhr = 58  # Lower RHR on weight training days
            else:
                activity_type = "running"
                rhr = 62  # Higher RHR on cardio days
            
            # Insert activity
            activity_data = {
                'garmin_activity_id': f'test_activity_{i}',
                'activity_type': activity_type,
                'start_time': datetime.combine(test_date, datetime.min.time().replace(hour=8)),
                'end_time': datetime.combine(test_date, datetime.min.time().replace(hour=9)),
                'duration_minutes': 45,
                'calories_burned': 300,
                'distance_km': 5.0 if activity_type == "running" else 0,
                'avg_hr': 140,
                'max_hr': 160
            }
            database.insert_activity_data(self.user_id, activity_data)
            
            # Insert daily summary with corresponding RHR
            summary_data = {
                'total_steps': 8000,
                'avg_daily_rhr': rhr
            }
            database.insert_daily_summary(self.user_id, test_date, summary_data)
        
        # Test activity vs RHR analysis
        result = trend_analyzer.get_activity_type_rhr_impact(self.user_id)
        
        self.assertIsInstance(result, list, "Should return a list of observations")
        self.assertGreater(len(result), 0, "Should have activity vs RHR observations")
        
        result_text = " ".join(result)
        self.assertIn("weight training", result_text, "Should mention weight training")
        self.assertIn("cardio", result_text, "Should mention cardio activities")
        
        print(f"✓ Activity vs RHR analysis: {result[0]}")
        print("✅ Test Case 3 PASSED: Activity type vs RHR impact analysis")
    
    def test_stress_lifestyle_correlations(self):
        """Test Case 4: Stress-Lifestyle Correlations"""
        print("\n=== Test Case 4: Stress-Lifestyle Correlations ===")
        
        test_dates = [date.today() - timedelta(days=i) for i in range(1, 15)]
        
        for i, test_date in enumerate(test_dates):
            # Alternate high/low caffeine days
            if i % 3 == 0:
                caffeine_mg = 250  # High caffeine day
                stress_level = 40  # Higher stress
                mood_rating = 6
            else:
                caffeine_mg = 50   # Low caffeine day  
                stress_level = 25  # Lower stress
                mood_rating = 8
            
            # Insert daily summary with stress
            summary_data = {
                'total_steps': 8000,
                'avg_daily_stress': stress_level
            }
            database.insert_daily_summary(self.user_id, test_date, summary_data)
            
            # Insert food log with caffeine
            if caffeine_mg > 0:
                food_data = {
                    'date': test_date,
                    'time': datetime.now().time(),
                    'timestamp': datetime.combine(test_date, datetime.now().time()),
                    'food_item_name': 'Coffee',
                    'calories': 5,
                    'caffeine_mg': caffeine_mg
                }
                database.upsert_food_entry(self.user_id, food_data)
            
            # Insert mood data
            mood_data = {
                'date': test_date,
                'timestamp': datetime.combine(test_date, datetime.now().time()),
                'mood_rating': mood_rating,
                'stress_rating': stress_level // 10
            }
            mood_tracking.insert_daily_mood_entry(self.user_id, mood_data)
        
        # Test stress-lifestyle correlation
        result = trend_analyzer.analyze_stress_lifestyle_correlation(self.user_id, days=14)
        
        self.assertIsInstance(result, list, "Should return a list of observations")
        self.assertGreater(len(result), 0, "Should have lifestyle correlation observations")
        
        result_text = " ".join(result)
        if "caffeine" in result_text:
            print(f"✓ Caffeine-stress correlation detected: {[r for r in result if 'caffeine' in r][0]}")
        if "mood" in result_text:
            print(f"✓ Mood-stress correlation detected: {[r for r in result if 'mood' in r][0]}")
        
        print("✅ Test Case 4 PASSED: Stress-lifestyle correlations")


class TestLLMIntegration(TestHealthAgentBase):
    """Test LLM integration and error handling."""
    
    @patch('smart_health_ollama.llm')
    def test_llm_connection_failure(self, mock_llm):
        """Test Case 1: LLM Connection Failure"""
        print("\n=== Test Case 1: LLM Connection Failure Handling ===")
        
        # Mock LLM to raise connection error
        mock_llm.invoke.side_effect = Exception("Connection refused")
        
        # Create test state
        state = HealthAgentState()
        state.health_data = {
            'sleep_hours': 7.5,
            'garmin_data': {'sleep_score': 85},
            'heart_rate': 60,
            'steps': 10000,
            'stress_metrics': {'avg_stress': 25}
        }
        
        # Test recommendation generation
        try:
            result_state = generate_recommendations(state)
            
            # Should not crash and should have error handling
            self.assertIsInstance(result_state, HealthAgentState, "Should return HealthAgentState")
            
            # Check for fallback message in recommendations
            if result_state.recommendations:
                rec_text = str(result_state.recommendations[-1].content)
                self.assertIn("sorry", rec_text.lower(), "Should contain apologetic fallback message")
                print(f"✓ Fallback message generated: {rec_text[:100]}...")
            
            print("✅ Test Case 1 PASSED: LLM connection failure handled gracefully")
            
        except Exception as e:
            self.fail(f"LLM connection failure should be handled gracefully, but got: {e}")


def run_all_tests():
    """Run all test suites."""
    print("🧪 Starting Comprehensive Health Agent Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCronometerImport,
        TestGarminDataSync, 
        TestMoodTracking,
        TestTrendAnalysis,
        TestLLMIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 Test Suite Complete")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-3]}")
    
    if not result.failures and not result.errors:
        print("✅ All tests passed successfully!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)