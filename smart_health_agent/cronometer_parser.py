import csv
import uuid
import hashlib
from datetime import datetime
from collections import defaultdict
import logging
import re

from database import db # Import the global db instance

logger = logging.getLogger(__name__)

# Lookup tables for caffeine and alcohol estimation based on common beverages
# Values are per common serving size (e.g., mg per 8oz coffee, units per standard beer)
CAFFEINE_LOOKUP = {
    'coffee': 95,      # mg per 8oz cup
    'espresso': 64,    # mg per shot
    'black tea': 47,   # mg per 8oz cup
    'green tea': 28,   # mg per 8oz cup
    'energy drink': 160, # mg per typical 16oz can
    'coca-cola': 34,   # mg per 12oz can
    'pepsi': 38,       # mg per 12oz can
    'dark chocolate': 12, # mg per oz
    'milk chocolate': 6 # mg per oz
}

# Alcohol units (standard drink) estimation
# UK unit is 8g ethanol. US standard drink is 14g ethanol.
# Let's use UK units for simplicity: 1 unit = 8g ethanol
# Alcohol density is ~0.789 g/ml
# So, 1 unit = 8g / 0.789 g/ml = ~10.14 ml pure ethanol
ALCOHOL_LOOKUP = {
    'beer': {'abv': 0.05, 'serving_ml': 500}, # 5% ABV, 500ml pint -> 2 units
    'wine': {'abv': 0.12, 'serving_ml': 175}, # 12% ABV, 175ml glass -> 2.1 units
    'spirit': {'abv': 0.40, 'serving_ml': 25}, # 40% ABV, 25ml shot -> 1 unit
    'cider': {'abv': 0.045, 'serving_ml': 500} # 4.5% ABV, 500ml pint -> 2.25 units
}

def estimate_caffeine(item_name, amount, units):
    """Estimates caffeine in mg based on item name, amount, and units."""
    item_name_lower = item_name.lower()
    for key, caffeine_per_serving in CAFFEINE_LOOKUP.items():
        if key in item_name_lower:
            # Simple estimation: assume 1 unit of 'cup', 'can', 'oz' corresponds to lookup
            # This is a simplification and would need more robust unit conversion for accuracy
            if 'cup' in units.lower() or 'cups' in units.lower():
                return caffeine_per_serving * amount
            elif 'ml' in units.lower():
                # Rough conversion: assume 8oz = 240ml for coffee/tea
                return (caffeine_per_serving / 240) * amount
            elif 'oz' in units.lower():
                # Rough conversion: assume 8oz = 226g for coffee/tea
                return (caffeine_per_serving / 8) * amount
            elif 'can' in units.lower() or 'cans' in units.lower():
                return caffeine_per_serving * amount
            elif 'shot' in units.lower() or 'shots' in units.lower():
                return caffeine_per_serving * amount
            # If no specific unit match, return based on a single serving
            return caffeine_per_serving * amount # Assume 'amount' is number of servings
    return 0.0

def estimate_alcohol_units(item_name, amount, units):
    """Estimates alcohol units based on item name, amount, and units."""
    item_name_lower = item_name.lower()
    for key, data in ALCOHOL_LOOKUP.items():
        if key in item_name_lower:
            abv = data['abv']
            serving_ml = data['serving_ml']
            
            # Convert input units to ml for calculation
            if 'ml' in units.lower():
                total_ml = amount
            elif 'oz' in units.lower():
                total_ml = amount * 29.5735 # 1 oz = 29.5735 ml
            elif 'cup' in units.lower() or 'cups' in units.lower():
                total_ml = amount * 240 # 1 cup = 240 ml (approx)
            elif 'pint' in units.lower() or 'pints' in units.lower():
                total_ml = amount * 568 # 1 pint = 568 ml (UK)
            elif 'can' in units.lower() or 'cans' in units.lower():
                # Assume a standard can size if not specified, e.g., 330ml or 500ml
                total_ml = amount * 330 # Or 500, depends on common can size
            else:
                # Fallback: assume 'amount' is number of standard servings
                total_ml = amount * serving_ml

            # Alcohol (g) = Volume (ml) * (ABV/100) * Density (0.789 g/ml)
            # Alcohol units (UK) = Alcohol (g) / 8g per unit
            alcohol_grams = total_ml * abv * 0.789
            alcohol_units = alcohol_grams / 8.0
            return alcohol_units
    return 0.0

def parse_cronometer_food_entries_csv(file_path, user_id):
    """
    Parses a Cronometer 'Food & Recipe Entries' CSV file and stores data in the database.
    Aggregates daily nutrition data into food_log_daily.
    """
    parsed_food_entries = 0
    parsed_supplements = 0
    daily_nutrition_totals = defaultdict(lambda: defaultdict(float))

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read headers first to ensure consistency
            reader = csv.reader(f)
            headers = [h.strip() for h in next(reader)] # Strip whitespace from headers
            logger.debug(f"CSV Headers detected: {headers}")

            # Map expected headers to actual headers, handling variations
            header_map = {
                'Date': ['Date', 'Day'], # Add 'Day' as a possible header for 'Date'
                'Item Name': ['Item Name', 'Food Name', 'Name'],
                'Amount': ['Amount'],
                'Units': ['Units'], # Keep this mapping
                'Calories': ['Calories', 'Energy (kcal)'],
                'Protein (g)': ['Protein (g)', 'Protein'],
                'Carbohydrates (g)': ['Carbohydrates (g)', 'Carbs (g)', 'Carbs'],
                'Fat (g)': ['Fat (g)', 'Fat'],
                'Category': ['Category'] # To distinguish food vs supplement
            }

            # Define truly critical headers that MUST be present
            CRITICAL_HEADERS = ['Date', 'Item Name', 'Amount', 'Calories', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)']

            # Create a reverse map for quick lookup of actual column names
            actual_header_lookup = {}
            for expected, possible_names in header_map.items():
                found = False
                for name in possible_names:
                    if name in headers:
                        actual_header_lookup[expected] = name
                        found = True
                        break
                # Only raise error for truly critical headers if not found
                if not found and expected in CRITICAL_HEADERS:
                    logger.error(f"Missing critical header: {expected}. Found headers: {headers}")
                    raise ValueError(f"Missing critical CSV header: {expected}")

            # Re-initialize DictReader with the file object (or reopen it)
            f.seek(0)
            dict_reader = csv.DictReader(f)

            # Skip header row if DictReader doesn't do it automatically (it should)
            # next(dict_reader) # No, DictReader handles this.

            for i, row in enumerate(dict_reader):
                if not row.get(actual_header_lookup['Date']) or not row.get(actual_header_lookup['Item Name']):
                    logger.debug(f"Skipping row {i+1} due to missing Date or Item Name: {row}")
                    continue

                try:
                    entry_date_str = row[actual_header_lookup['Date']].strip()
                    item_name = row[actual_header_lookup['Item Name']].strip()
                    amount_str_raw = row.get(actual_header_lookup['Amount'], '0').strip()
                    units = row.get(actual_header_lookup.get('Units'), '').strip() # Use .get for optional header
                    category = row.get(actual_header_lookup.get('Category', 'Category'), '').strip() # Use .get for optional header

                    # Extract numeric amount from strings that may contain units (e.g., "300.00 g", "2.00 each")
                    # Use regex to find the first sequence of numbers (integers or floats)
                    amount_match = re.search(r'(\d+\.?\d*)', amount_str_raw)
                    if amount_match:
                        amount = float(amount_match.group(1))
                    else:
                        amount = 0.0 # Default to 0 if no number found
                        logger.warning(f"Could not extract numeric amount from '{amount_str_raw}'. Defaulting to 0.0.")

                    # If no Units column exists, try to extract unit info from Amount column for estimation
                    if not units and amount_str_raw:
                        # Use the original amount string for unit detection in estimation functions
                        units_for_estimation = amount_str_raw
                    else:
                        units_for_estimation = units

                    # Convert numerical fields, handling empty strings or non-numeric values
                    calories = float(row.get(actual_header_lookup['Calories'], '0').strip() or '0')
                    protein_g = float(row.get(actual_header_lookup['Protein (g)'], '0').strip() or '0')
                    carbohydrates_g = float(row.get(actual_header_lookup['Carbohydrates (g)'], '0').strip() or '0')
                    fat_g = float(row.get(actual_header_lookup['Fat (g)'], '0').strip() or '0')

                    # Estimate caffeine and alcohol if not explicitly provided
                    # Cronometer CSVs don't always have direct columns for these, so infer
                    caffeine_mg = estimate_caffeine(item_name, amount, units_for_estimation)
                    alcohol_units = estimate_alcohol_units(item_name, amount, units_for_estimation)

                    # Create a "natural key" to identify unique entries for upserting
                    # Combine key fields and hash them to create a stable, unique ID
                    natural_key_components = (
                        entry_date_str,
                        item_name,
                        str(amount), # Convert to string to include in hash
                        units,
                        str(calories), # Convert to string to include in hash
                        str(protein_g),
                        str(carbohydrates_g),
                        str(fat_g)
                    )
                    # Use a cryptographic hash for robustness and stable IDs
                    entry_id = hashlib.sha256("".join(natural_key_components).encode('utf-8')).hexdigest()

                    db.upsert_food_log_entry(
                        user_id, entry_id, entry_date_str, item_name, amount, units,
                        calories, protein_g, carbohydrates_g, fat_g, caffeine_mg, alcohol_units
                    )

                    # Aggregate for daily summary
                    daily_nutrition_totals[entry_date_str]['total_calories'] += calories
                    daily_nutrition_totals[entry_date_str]['protein_g'] += protein_g
                    daily_nutrition_totals[entry_date_str]['carbohydrates_g'] += carbohydrates_g
                    daily_nutrition_totals[entry_date_str]['fat_g'] += fat_g
                    daily_nutrition_totals[entry_date_str]['caffeine_mg'] += caffeine_mg
                    daily_nutrition_totals[entry_date_str]['alcohol_units'] += alcohol_units

                    if "supplement" in category.lower():
                        parsed_supplements += 1
                    else:
                        parsed_food_entries += 1

                except ValueError as ve:
                    logger.warning(f"Skipping row {i+1} due to data conversion error: {ve} - Row: {row}")
                except KeyError as ke:
                    logger.warning(f"Skipping row {i+1} due to missing expected column in row: {ke} - Row: {row}")
                except Exception as e:
                    logger.error(f"Unexpected error processing row {i+1}: {e} - Row: {row}")

        # After parsing all entries, update daily summaries
        for date_str, totals in daily_nutrition_totals.items():
            db.upsert_food_log_daily_entry(user_id, date_str, totals)
        
        # Update sync status
        db.update_sync_status(user_id, 'cronometer', datetime.now(), 'completed', records_synced=parsed_food_entries + parsed_supplements)
        
        logger.info(f"Successfully parsed {parsed_food_entries} food entries and {parsed_supplements} supplements")
        return parsed_food_entries, parsed_supplements
        
    except Exception as e:
        logger.error(f"Error parsing Cronometer CSV: {e}", exc_info=True)
        db.update_sync_status(user_id, 'cronometer', datetime.now(), 'failed', error_message=str(e))
        return 0, 0