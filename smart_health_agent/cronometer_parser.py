"""
Cronometer CSV Parser
Handles parsing and processing of Cronometer "Food & Recipe Entries" CSV exports.
"""

import csv
import logging
import re
from datetime import datetime, date, time
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import database

# Setup logging
logger = logging.getLogger('cronometer_parser')

# Common supplement keywords for categorization
SUPPLEMENT_KEYWORDS = [
    'vitamin', 'mineral', 'supplement', 'capsule', 'tablet', 'powder',
    'probiotic', 'protein powder', 'creatine', 'omega', 'fish oil',
    'magnesium', 'calcium', 'iron', 'zinc', 'b12', 'b-12', 'vitamin d',
    'vitamin c', 'multivitamin', 'multi-vitamin', 'bcaa', 'amino acid',
    'whey', 'casein', 'glutamine', 'collagen', 'fiber supplement',
    'melatonin', 'ashwagandha', 'turmeric', 'curcumin'
]

# Common Cronometer CSV column mappings
CRONOMETER_COLUMN_MAPPING = {
    'date': ['Date', 'date'],
    'time': ['Time', 'time'],
    'food_name': ['Food Name', 'food_name', 'Food', 'Item Name', 'Name'],
    'amount': ['Amount', 'amount', 'Quantity', 'quantity'],
    'unit': ['Unit', 'unit', 'Units', 'units'],
    'calories': ['Calories', 'calories', 'Energy (kcal)', 'Energy'],
    'protein': ['Protein (g)', 'protein', 'Protein', 'protein_g'],
    'carbs': ['Carbohydrates (g)', 'carbohydrates', 'Carbs', 'carbs', 'carbs_g'],
    'fat': ['Fat (g)', 'fat', 'Fat', 'fats', 'fats_g'],
    'fiber': ['Fiber (g)', 'fiber', 'Fiber', 'fiber_g'],
    'sugar': ['Sugars (g)', 'sugar', 'Sugar', 'sugars', 'sugar_g'],
    'sodium': ['Sodium (mg)', 'sodium', 'Sodium', 'sodium_mg'],
    'vitamin_c': ['Vitamin C (mg)', 'vitamin_c', 'Vitamin C', 'vitamin_c_mg'],
    'iron': ['Iron (mg)', 'iron', 'Iron', 'iron_mg'],
    'calcium': ['Calcium (mg)', 'calcium', 'Calcium', 'calcium_mg']
}

def identify_supplement(food_name: str) -> bool:
    """
    Determine if a food item is likely a supplement based on its name.
    
    Args:
        food_name: Name of the food/supplement item
        
    Returns:
        bool: True if likely a supplement, False otherwise
    """
    food_name_lower = food_name.lower()
    return any(keyword in food_name_lower for keyword in SUPPLEMENT_KEYWORDS)

def find_column_index(headers: List[str], possible_names: List[str]) -> Optional[int]:
    """
    Find the index of a column by checking against possible names.
    
    Args:
        headers: List of CSV header names
        possible_names: List of possible column names to match
        
    Returns:
        int or None: Index of the column if found, None otherwise
    """
    headers_lower = [h.lower().strip() for h in headers]
    for possible_name in possible_names:
        possible_lower = possible_name.lower().strip()
        if possible_lower in headers_lower:
            return headers_lower.index(possible_lower)
    return None

def safe_float_convert(value: str, default: float = 0.0) -> float:
    """
    Safely convert a string value to float with fallback.
    
    Args:
        value: String value to convert
        default: Default value if conversion fails
        
    Returns:
        float: Converted value or default
    """
    if not value or value.strip() == '':
        return default
    
    try:
        # Remove any non-numeric characters except decimal point and minus
        cleaned_value = re.sub(r'[^\d.-]', '', str(value).strip())
        if cleaned_value == '' or cleaned_value == '-':
            return default
        return float(cleaned_value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{value}' to float, using default {default}")
        return default

def parse_date_time(date_str: str, time_str: str) -> Tuple[date, time, datetime]:
    """
    Parse date and time strings into proper date, time, and datetime objects.
    
    Args:
        date_str: Date string from CSV
        time_str: Time string from CSV
        
    Returns:
        Tuple of (date, time, datetime) objects
    """
    try:
        # Parse date - try common formats
        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
        parsed_date = None
        
        for date_format in date_formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), date_format).date()
                break
            except ValueError:
                continue
        
        if parsed_date is None:
            raise ValueError(f"Could not parse date: {date_str}")
        
        # Parse time - try common formats
        time_formats = ['%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p']
        parsed_time = None
        
        time_str_clean = time_str.strip()
        for time_format in time_formats:
            try:
                parsed_time = datetime.strptime(time_str_clean, time_format).time()
                break
            except ValueError:
                continue
        
        if parsed_time is None:
            # Default to midnight if time parsing fails
            logger.warning(f"Could not parse time '{time_str}', defaulting to 00:00:00")
            parsed_time = time(0, 0, 0)
        
        # Combine date and time
        parsed_datetime = datetime.combine(parsed_date, parsed_time)
        
        return parsed_date, parsed_time, parsed_datetime
        
    except Exception as e:
        logger.error(f"Error parsing date/time '{date_str}' / '{time_str}': {e}")
        # Fallback to current date/time
        now = datetime.now()
        return now.date(), now.time(), now

def detect_meal_type(time_obj: time) -> str:
    """
    Detect likely meal type based on time of day.
    
    Args:
        time_obj: Time object
        
    Returns:
        str: Likely meal type
    """
    hour = time_obj.hour
    
    if 5 <= hour < 11:
        return 'Breakfast'
    elif 11 <= hour < 15:
        return 'Lunch'
    elif 15 <= hour < 18:
        return 'Snack'
    elif 18 <= hour <= 23:
        return 'Dinner'
    else:
        return 'Late Night'

def parse_cronometer_food_entries_csv(file_path: str, user_id: int) -> Dict[str, Any]:
    """
    Parse Cronometer "Food & Recipe Entries" CSV export and store in database.
    
    Args:
        file_path: Path to the uploaded CSV file
        user_id: Database user ID
        
    Returns:
        Dict containing import summary
    """
    logger.info(f"Starting to parse Cronometer CSV: {file_path}")
    
    summary = {
        'total_rows': 0,
        'food_entries': 0,
        'supplement_entries': 0,
        'errors': 0,
        'error_details': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            # Try to detect delimiter
            sample = csvfile.read(1024)
            csvfile.seek(0)
            
            # Detect CSV dialect
            sniffer = csv.Sniffer()
            try:
                delimiter = sniffer.sniff(sample).delimiter
            except:
                delimiter = ','  # Default to comma
            
            reader = csv.reader(csvfile, delimiter=delimiter)
            
            # Read headers
            headers = next(reader)
            logger.info(f"CSV headers: {headers}")
            
            # Map column indices
            column_indices = {}
            for key, possible_names in CRONOMETER_COLUMN_MAPPING.items():
                index = find_column_index(headers, possible_names)
                column_indices[key] = index
                if index is not None:
                    logger.info(f"Mapped {key} to column {index}: '{headers[index]}'")
                else:
                    logger.warning(f"Could not find column for {key}")
            
            # Check required columns
            required_columns = ['date', 'food_name']
            missing_required = [col for col in required_columns if column_indices[col] is None]
            
            if missing_required:
                error_msg = f"Missing required columns: {missing_required}"
                logger.error(error_msg)
                summary['error_details'].append(error_msg)
                return summary
            
            # Process each row
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                summary['total_rows'] += 1
                
                try:
                    if len(row) < len(headers):
                        # Pad row with empty strings if shorter than headers
                        row.extend([''] * (len(headers) - len(row)))
                    
                    # Extract required fields
                    date_str = row[column_indices['date']].strip() if column_indices['date'] is not None else ''
                    time_str = row[column_indices['time']].strip() if column_indices['time'] is not None else '00:00:00'
                    food_name = row[column_indices['food_name']].strip() if column_indices['food_name'] is not None else ''
                    
                    if not date_str or not food_name:
                        logger.warning(f"Row {row_num}: Missing date or food name, skipping")
                        summary['errors'] += 1
                        continue
                    
                    # Parse date and time
                    parsed_date, parsed_time, parsed_datetime = parse_date_time(date_str, time_str)
                    
                    # Extract nutritional data
                    amount = safe_float_convert(row[column_indices['amount']] if column_indices['amount'] is not None else '', 1.0)
                    unit = row[column_indices['unit']].strip() if column_indices['unit'] is not None else ''
                    calories = safe_float_convert(row[column_indices['calories']] if column_indices['calories'] is not None else '')
                    protein = safe_float_convert(row[column_indices['protein']] if column_indices['protein'] is not None else '')
                    carbs = safe_float_convert(row[column_indices['carbs']] if column_indices['carbs'] is not None else '')
                    fat = safe_float_convert(row[column_indices['fat']] if column_indices['fat'] is not None else '')
                    fiber = safe_float_convert(row[column_indices['fiber']] if column_indices['fiber'] is not None else '')
                    sugar = safe_float_convert(row[column_indices['sugar']] if column_indices['sugar'] is not None else '')
                    sodium = safe_float_convert(row[column_indices['sodium']] if column_indices['sodium'] is not None else '')
                    vitamin_c = safe_float_convert(row[column_indices['vitamin_c']] if column_indices['vitamin_c'] is not None else '')
                    iron = safe_float_convert(row[column_indices['iron']] if column_indices['iron'] is not None else '')
                    calcium = safe_float_convert(row[column_indices['calcium']] if column_indices['calcium'] is not None else '')
                    
                    # Determine if it's a supplement
                    is_supplement = identify_supplement(food_name)
                    
                    if is_supplement:
                        # Store as supplement
                        supplement_data = {
                            'date': parsed_date,
                            'time': parsed_time,
                            'timestamp': parsed_datetime,
                            'supplement_name': food_name,
                            'quantity': amount,
                            'unit': unit,
                            'dosage': f"{amount} {unit}" if unit else str(amount),
                            'calories': calories,
                            'source': 'cronometer',
                            'notes': f"Imported from Cronometer CSV on {datetime.now().strftime('%Y-%m-%d')}"
                        }
                        
                        database.upsert_supplement_entry(user_id, supplement_data)
                        summary['supplement_entries'] += 1
                        
                    else:
                        # Store as food
                        food_data = {
                            'date': parsed_date,
                            'time': parsed_time,
                            'timestamp': parsed_datetime,
                            'meal_type': detect_meal_type(parsed_time),
                            'food_item_name': food_name,
                            'category': 'Food',
                            'quantity': amount,
                            'unit': unit,
                            'calories': calories,
                            'protein_g': protein,
                            'carbs_g': carbs,
                            'fats_g': fat,
                            'fiber_g': fiber,
                            'sugar_g': sugar,
                            'sodium_mg': sodium,
                            'vitamin_c_mg': vitamin_c,
                            'iron_mg': iron,
                            'calcium_mg': calcium,
                            'source': 'cronometer',
                            'notes': f"Imported from Cronometer CSV on {datetime.now().strftime('%Y-%m-%d')}"
                        }
                        
                        database.upsert_food_entry(user_id, food_data)
                        summary['food_entries'] += 1
                    
                except Exception as e:
                    error_msg = f"Row {row_num}: Error processing row - {str(e)}"
                    logger.error(error_msg)
                    summary['errors'] += 1
                    summary['error_details'].append(error_msg)
                    continue
            
            logger.info(f"CSV parsing complete. Summary: {summary}")
            
    except Exception as e:
        error_msg = f"Error reading CSV file: {str(e)}"
        logger.error(error_msg)
        summary['error_details'].append(error_msg)
        summary['errors'] += 1
    
    return summary

def validate_cronometer_csv(file_path: str) -> Dict[str, Any]:
    """
    Validate a Cronometer CSV file before processing.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dict containing validation results
    """
    validation_result = {
        'is_valid': False,
        'issues': [],
        'column_mapping': {},
        'sample_data': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            # Read first few lines for validation
            sample_lines = [csvfile.readline() for _ in range(5)]
            csvfile.seek(0)
            
            # Try to parse as CSV
            reader = csv.reader(csvfile)
            headers = next(reader)
            
            # Check for required columns
            column_indices = {}
            for key, possible_names in CRONOMETER_COLUMN_MAPPING.items():
                index = find_column_index(headers, possible_names)
                column_indices[key] = index
            
            validation_result['column_mapping'] = column_indices
            
            # Check for minimum required columns
            if column_indices['date'] is None:
                validation_result['issues'].append("Missing Date column")
            if column_indices['food_name'] is None:
                validation_result['issues'].append("Missing Food Name column")
            
            # Read a few sample rows
            sample_count = 0
            for row in reader:
                if sample_count >= 3:  # Limit sample size
                    break
                validation_result['sample_data'].append(row)
                sample_count += 1
            
            # Determine if valid
            validation_result['is_valid'] = len(validation_result['issues']) == 0
            
            if validation_result['is_valid']:
                validation_result['issues'].append("File appears to be a valid Cronometer export")
            
    except Exception as e:
        validation_result['issues'].append(f"Error reading file: {str(e)}")
    
    return validation_result