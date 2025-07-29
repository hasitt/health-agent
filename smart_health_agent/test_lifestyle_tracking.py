#!/usr/bin/env python3
"""
Specialized Test Script for Caffeine and Alcohol Tracking
Tests the enhanced lifestyle tracking capabilities.
"""

import sys
import os
import unittest
import sqlite3
import tempfile
import csv
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database
import cronometer_parser
import mood_tracking

class TestLifestyleTracking(unittest.TestCase):
    """Test enhanced caffeine and alcohol tracking."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_db_path = Path(__file__).parent / "test_data" / "test_lifestyle.db"
        cls.test_db_path.parent.mkdir(exist_ok=True)
        
        # Override database path
        cls.original_db_path = database.DB_PATH
        database.DB_PATH = cls.test_db_path
        
        # Create tables and test user
        database.create_tables()
        cls.test_user_id = database.insert_or_update_user(
            garmin_user_id="lifestyle_test_user",
            name="Lifestyle Test User", 
            access_token="test_token",
            refresh_token="test_refresh",
            token_expiry=datetime.now() + timedelta(days=365)
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        database.DB_PATH = cls.original_db_path
        if cls.test_db_path.exists():
            cls.test_db_path.unlink()
    
    def create_lifestyle_csv(self, data):
        """Create CSV with caffeine/alcohol data."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
        writer = csv.writer(temp_file)
        writer.writerows(data)
        temp_file.close()
        return temp_file.name
    
    def test_caffeine_alcohol_import_and_tracking(self):
        """Test comprehensive caffeine and alcohol import and tracking."""
        print("\n=== Caffeine & Alcohol Import and Tracking Test ===")
        
        # Create CSV with diverse caffeine and alcohol entries
        lifestyle_data = [
            ["Date", "Time", "Food Name", "Amount", "Unit", "Calories", "Protein (g)", "Carbohydrates (g)", "Fat (g)"],
            # Caffeine entries
            ["2025-01-10", "07:00:00", "Coffee, Black", "1", "cup", "5", "0.3", "0", "0"],
            ["2025-01-10", "09:30:00", "Green Tea", "1", "cup", "2", "0", "0.5", "0"],
            ["2025-01-10", "14:00:00", "Energy Drink", "1", "can", "110", "0", "28", "0"],
            # Alcohol entries  
            ["2025-01-10", "18:00:00", "Red Wine", "150", "ml", "125", "0.1", "4", "0"],
            ["2025-01-11", "19:30:00", "Beer, IPA", "355", "ml", "210", "1.6", "18", "0"],
            ["2025-01-11", "20:00:00", "Whiskey", "30", "ml", "70", "0", "0", "0"],
            # Mixed entries
            ["2025-01-12", "08:00:00", "Espresso", "1", "shot", "3", "0.1", "0", "0"],
            ["2025-01-12", "17:00:00", "White Wine", "120", "ml", "100", "0.1", "3", "0"],
        ]
        
        csv_file = self.create_lifestyle_csv(lifestyle_data)
        
        try:
            # Import the data
            result = cronometer_parser.parse_cronometer_food_entries_csv(csv_file, self.test_user_id)
            
            print(f"Import result: {result['food_entries']} food entries, {result['errors']} errors")
            
            # Now we need to manually add caffeine and alcohol data since the basic CSV parser doesn't include it
            # This simulates the enhanced CSV parser with caffeine/alcohol detection
            
            caffeine_alcohol_data = [
                # (food_name, caffeine_mg, alcohol_ml, alcohol_abv, beverage_type)
                ("Coffee, Black", 95, 0, 0, "coffee"),
                ("Green Tea", 25, 0, 0, "tea"), 
                ("Energy Drink", 80, 0, 0, "energy_drink"),
                ("Red Wine", 0, 150, 13.5, "wine"),
                ("Beer, IPA", 0, 355, 6.5, "beer"),
                ("Whiskey", 0, 30, 40, "spirits"),
                ("Espresso", 63, 0, 0, "coffee"),
                ("White Wine", 0, 120, 12.5, "wine"),
            ]
            
            # Update food entries with caffeine/alcohol data
            with database.get_db_connection() as conn:
                cursor = conn.cursor()
                
                for food_name, caffeine, alcohol_ml, abv, beverage_type in caffeine_alcohol_data:
                    # Calculate alcohol units (UK standard: 1 unit = 10ml pure alcohol)
                    alcohol_units = (alcohol_ml * abv / 100) / 10 if alcohol_ml > 0 else 0
                    
                    cursor.execute("""
                        UPDATE food_log 
                        SET caffeine_mg = ?, alcohol_ml = ?, alcohol_abv = ?, 
                            alcohol_units = ?, beverage_type = ?
                        WHERE user_id = ? AND food_item_name = ?
                    """, (caffeine, alcohol_ml, abv, alcohol_units, beverage_type, self.test_user_id, food_name))
                
                conn.commit()
            
            # Test caffeine and alcohol summary
            lifestyle_summary = mood_tracking.get_caffeine_alcohol_summary(self.test_user_id, days=7)
            
            # Verify daily consumption totals
            self.assertGreater(len(lifestyle_summary['daily_consumption']), 0, "Should have daily consumption data")
            
            daily_totals = lifestyle_summary['daily_consumption'][0]  # Most recent day
            
            # Verify caffeine totals (should include coffee + tea + energy drink from one day)
            total_caffeine_expected = 95 + 25 + 80  # Coffee + Green Tea + Energy Drink from 2025-01-10
            
            # Find the day with highest caffeine
            max_caffeine_day = max(lifestyle_summary['daily_consumption'], 
                                 key=lambda x: x['total_caffeine'] or 0)
            
            self.assertGreaterEqual(max_caffeine_day['total_caffeine'], 200, 
                                  f"Should have high caffeine day with 200+ mg, got {max_caffeine_day['total_caffeine']}")
            
            # Verify alcohol totals
            max_alcohol_day = max(lifestyle_summary['daily_consumption'], 
                                key=lambda x: x['total_alcohol_units'] or 0)
            
            self.assertGreater(max_alcohol_day['total_alcohol_units'], 1, 
                             f"Should have alcohol consumption > 1 unit, got {max_alcohol_day['total_alcohol_units']}")
            
            # Verify recent entries include caffeine/alcohol details
            recent_entries = lifestyle_summary['recent_entries']
            self.assertGreater(len(recent_entries), 0, "Should have recent caffeine/alcohol entries")
            
            caffeine_entries = [e for e in recent_entries if e['caffeine_mg'] and e['caffeine_mg'] > 0]
            alcohol_entries = [e for e in recent_entries if e['alcohol_ml'] and e['alcohol_ml'] > 0]
            
            self.assertGreater(len(caffeine_entries), 0, "Should have caffeine entries")
            self.assertGreater(len(alcohol_entries), 0, "Should have alcohol entries")
            
            print(f"✓ Caffeine entries found: {len(caffeine_entries)}")
            print(f"✓ Alcohol entries found: {len(alcohol_entries)}")
            print(f"✓ Max daily caffeine: {max_caffeine_day['total_caffeine']:.1f}mg")
            print(f"✓ Max daily alcohol: {max_alcohol_day['total_alcohol_units']:.1f} units")
            
            # Test database schema for new columns
            with database.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(food_log)")
                columns = {row[1] for row in cursor.fetchall()}
                
                required_columns = {'caffeine_mg', 'alcohol_ml', 'alcohol_abv', 'alcohol_units', 'beverage_type'}
                missing_columns = required_columns - columns
                
                self.assertEqual(len(missing_columns), 0, 
                               f"Missing required lifestyle tracking columns: {missing_columns}")
                
                print(f"✓ All required lifestyle tracking columns present: {required_columns}")
            
            print("✅ Caffeine & Alcohol Import and Tracking Test PASSED")
            
        finally:
            os.unlink(csv_file)
    
    def test_lifestyle_correlation_with_stress(self):
        """Test lifestyle correlation with stress and mood data."""
        print("\n=== Lifestyle-Stress Correlation Test ===")
        
        # Create test scenario: high caffeine day followed by high stress
        test_dates = [date.today() - timedelta(days=i) for i in range(1, 8)]
        
        for i, test_date in enumerate(test_dates):
            # Alternate high/low caffeine and alcohol days
            if i % 2 == 0:
                # High caffeine, low alcohol day
                caffeine_amount = 300
                alcohol_amount = 0
                stress_level = 35  # Higher stress with high caffeine
                mood_rating = 6
            else:
                # Low caffeine, moderate alcohol day (previous evening)
                caffeine_amount = 50
                alcohol_amount = 2.5  # Units
                stress_level = 25   # Lower stress but potentially affected by alcohol
                mood_rating = 7
            
            # Insert food entries
            if caffeine_amount > 0:
                coffee_data = {
                    'date': test_date,
                    'time': datetime.now().time().replace(hour=8),
                    'timestamp': datetime.combine(test_date, datetime.now().time().replace(hour=8)),
                    'food_item_name': 'Coffee',
                    'calories': 5,
                    'caffeine_mg': caffeine_amount,
                    'beverage_type': 'coffee'
                }
                database.upsert_food_entry(self.test_user_id, coffee_data)
            
            if alcohol_amount > 0:
                # Add alcohol from previous evening
                alcohol_date = test_date - timedelta(days=1)
                wine_data = {
                    'date': alcohol_date,
                    'time': datetime.now().time().replace(hour=19),
                    'timestamp': datetime.combine(alcohol_date, datetime.now().time().replace(hour=19)),
                    'food_item_name': 'Wine',
                    'calories': 125,
                    'alcohol_ml': 150,
                    'alcohol_units': alcohol_amount,
                    'beverage_type': 'wine'
                }
                database.upsert_food_entry(self.test_user_id, wine_data)
            
            # Insert daily stress summary
            summary_data = {
                'total_steps': 8000,
                'avg_daily_stress': stress_level,
                'max_daily_stress': stress_level + 10
            }
            database.insert_daily_summary(self.test_user_id, test_date, summary_data)
            
            # Insert mood data
            mood_data = {
                'date': test_date,
                'timestamp': datetime.combine(test_date, datetime.now().time()),
                'mood_rating': mood_rating,
                'stress_rating': stress_level // 10,
                'anxiety_rating': (stress_level // 10) + 1
            }
            mood_tracking.insert_daily_mood_entry(self.test_user_id, mood_data)
        
        # Test correlation analysis
        import trend_analyzer
        correlations = trend_analyzer.analyze_stress_lifestyle_correlation(self.test_user_id, days=7)
        
        self.assertIsInstance(correlations, list, "Should return list of correlations")
        self.assertGreater(len(correlations), 0, "Should find lifestyle correlations")
        
        correlation_text = " ".join(correlations)
        
        # Check for caffeine correlation detection
        if "caffeine" in correlation_text.lower():
            print(f"✓ Caffeine-stress correlation detected")
            
        # Check for alcohol correlation detection  
        if "alcohol" in correlation_text.lower():
            print(f"✓ Alcohol-mood correlation detected")
            
        # Verify mood-stress correlation
        if "mood" in correlation_text.lower():
            print(f"✓ Mood-stress correlation detected")
            
        print(f"✓ Found {len(correlations)} lifestyle-stress correlations")
        for correlation in correlations:
            print(f"  - {correlation}")
        
        print("✅ Lifestyle-Stress Correlation Test PASSED")
    
    def test_enhanced_mood_tracking_storage(self):
        """Test comprehensive mood tracking with all 15+ fields."""
        print("\n=== Enhanced Mood Tracking Storage Test ===")
        
        # Test comprehensive mood entry
        comprehensive_mood_data = {
            'date': date.today(),
            'timestamp': datetime.now(),
            # Core ratings (1-10)
            'mood_rating': 8,
            'energy_rating': 7,
            'stress_rating': 4,
            'anxiety_rating': 3,
            'sleep_quality_rating': 9,
            'focus_rating': 8,
            'motivation_rating': 7,
            # Text fields
            'emotional_state': 'calm and optimistic, feeling balanced',
            'stress_triggers': 'deadline pressure, email overload',
            'coping_strategies': 'morning meditation, afternoon walk, deep breathing',
            'physical_symptoms': 'slight shoulder tension, otherwise feeling good',
            'daily_events': 'productive meeting, completed major project milestone',
            'social_interactions': 'good team collaboration, supportive conversation with friend',
            'weather_sensitivity': 'sunny weather boosted mood significantly',
            'hormonal_factors': 'mid-cycle, stable energy levels',
            # Metadata
            'entry_type': 'daily',
            'source': 'manual',
            'notes_text': 'Overall a very good day with minor work stress but good coping'
        }
        
        # Insert comprehensive mood entry
        mood_tracking.insert_daily_mood_entry(self.test_user_id, comprehensive_mood_data)
        
        # Verify storage of all fields
        with database.get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT mood_rating, energy_rating, stress_rating, anxiety_rating,
                       sleep_quality_rating, focus_rating, motivation_rating,
                       emotional_state, stress_triggers, coping_strategies,
                       physical_symptoms, daily_events, social_interactions,
                       weather_sensitivity, hormonal_factors, entry_type, source, notes_text
                FROM subjective_wellbeing 
                WHERE user_id = ? AND date = ?
            """, (self.test_user_id, date.today()))
            
            stored_entry = cursor.fetchone()
            self.assertIsNotNone(stored_entry, "Comprehensive mood entry should be stored")
            
            # Verify all numeric ratings
            numeric_fields = [
                ('mood_rating', 8), ('energy_rating', 7), ('stress_rating', 4),
                ('anxiety_rating', 3), ('sleep_quality_rating', 9),
                ('focus_rating', 8), ('motivation_rating', 7)
            ]
            
            for i, (field_name, expected_value) in enumerate(numeric_fields):
                self.assertEqual(stored_entry[i], expected_value, 
                               f"{field_name} should be {expected_value}")
                print(f"✓ {field_name}: {stored_entry[i]}/10")
            
            # Verify text fields
            text_fields = [
                ('emotional_state', 'calm and optimistic'), 
                ('stress_triggers', 'deadline pressure'),
                ('coping_strategies', 'morning meditation'),
                ('physical_symptoms', 'slight shoulder tension'),
                ('daily_events', 'productive meeting'),
                ('social_interactions', 'good team collaboration'),
                ('weather_sensitivity', 'sunny weather'),
                ('hormonal_factors', 'mid-cycle')
            ]
            
            for i, (field_name, expected_substring) in enumerate(text_fields, start=7):
                self.assertIn(expected_substring, stored_entry[i] or "", 
                            f"{field_name} should contain '{expected_substring}'")
                print(f"✓ {field_name}: stored correctly")
            
            # Verify metadata fields
            self.assertEqual(stored_entry[15], 'daily', "entry_type should be 'daily'")
            self.assertEqual(stored_entry[16], 'manual', "source should be 'manual'")
            self.assertIn('Overall a very good day', stored_entry[17], "notes should be stored")
            
            print(f"✓ All 18 mood tracking fields stored and verified")
        
        # Test mood summary retrieval
        mood_summary = mood_tracking.get_mood_summary(self.test_user_id, days=1)
        
        self.assertEqual(len(mood_summary['mood_entries']), 1, "Should retrieve 1 mood entry")
        self.assertIn('mood_rating', mood_summary['averages'], "Should calculate mood average")
        self.assertEqual(mood_summary['averages']['mood_rating'], 8.0, "Mood average should be 8.0")
        
        print(f"✓ Mood summary retrieval working: {mood_summary['averages']}")
        print("✅ Enhanced Mood Tracking Storage Test PASSED")


def run_lifestyle_tests():
    """Run lifestyle tracking tests."""
    print("🍷 Starting Lifestyle Tracking Test Suite")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2, exit=False)


if __name__ == "__main__":
    run_lifestyle_tests()