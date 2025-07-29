#!/usr/bin/env python3
"""
Test script for the new robust filter management system.
This validates the core logic before integration.
"""

def test_filter_logic():
    """Test the new filter management functions independently."""
    
    # Mock logger
    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    
    logger = MockLogger()
    
    # Test parse_filter_value function
    def _parse_filter_value(value_str: str, operator: str):
        """Parse filter value string based on operator type with comprehensive validation."""
        if not value_str or not isinstance(value_str, str):
            raise ValueError("Value cannot be empty or null")
            
        value_str = value_str.strip()
        
        if not value_str:
            raise ValueError("Value cannot be empty after trimming whitespace")
        
        if operator == 'between':
            parts = [v.strip() for v in value_str.split(',')]
            if len(parts) != 2:
                raise ValueError("Between operator requires exactly 2 comma-separated values (e.g., '10,20')")
            
            try:
                return [float(parts[0]), float(parts[1])]
            except ValueError:
                raise ValueError("Between operator requires numeric values (e.g., '10,20')")
        
        elif operator == 'in':
            parts = [v.strip() for v in value_str.split(',') if v.strip()]
            if not parts:
                raise ValueError("In operator requires at least one value")
            
            parsed_parts = []
            for part in parts:
                try:
                    parsed_parts.append(float(part))
                except ValueError:
                    if part:
                        parsed_parts.append(part)
            
            if not parsed_parts:
                raise ValueError("In operator requires at least one valid value")
            return parsed_parts
        
        elif operator == 'like':
            if len(value_str) < 1:
                raise ValueError("Like operator requires at least 1 character")
            if not value_str.startswith('%') and not value_str.endswith('%'):
                return f"%{value_str}%"
            return value_str
        
        else:
            if operator in ['>', '>=', '<', '<=']:
                try:
                    return float(value_str)
                except ValueError:
                    raise ValueError(f"Operator '{operator}' requires a numeric value")
            
            try:
                return float(value_str)
            except ValueError:
                return value_str
    
    # Test add filter function
    def handle_add_filter_robust(current_filters: list, metric: str, operator: str, value: str) -> tuple:
        """Add a new filter with comprehensive validation and deduplication."""
        print(f"[ADD FILTER] metric={metric}, operator={operator}, value='{value}', current_count={len(current_filters)}")
        
        if not metric or metric.strip() == "":
            print("Filter add rejected: No metric selected")
            return current_filters, "❌ Please select a metric before adding filter.", ""
        
        if not operator or operator.strip() == "":
            print("Filter add rejected: No operator selected")
            return current_filters, "❌ Please select an operator before adding filter.", ""
        
        if not value or value.strip() == "":
            print("Filter add rejected: Empty value")
            return current_filters, "❌ Please enter a value before adding filter.", ""
        
        try:
            parsed_value = _parse_filter_value(value.strip(), operator)
            
            new_filter = {
                'metric': metric,
                'operator': operator,
                'value': parsed_value,
                'raw_value': value.strip()
            }
            
            # Check for exact duplicates
            for existing_filter in current_filters:
                if (existing_filter['metric'] == metric and 
                    existing_filter['operator'] == operator and 
                    existing_filter['raw_value'] == value.strip()):
                    print("Duplicate filter detected")
                    return current_filters, f"⚠️ Filter '{metric} {operator} {value}' already exists!", ""
            
            new_filters_list = current_filters + [new_filter]
            print(f"Filter added successfully. Total filters: {len(new_filters_list)}")
            
            return new_filters_list, f"✅ Filter for '{metric.replace('_', ' ')}' added successfully!", ""
            
        except ValueError as ve:
            print(f"Filter validation error: {ve}")
            return current_filters, f"❌ Invalid value: {str(ve)}", value
        except Exception as e:
            print(f"Filter add error: {e}")
            return current_filters, f"❌ Error adding filter: {str(e)}", value
    
    # Test remove filter function
    def handle_remove_filter_robust(current_filters: list, remove_index: int) -> tuple:
        """Remove a filter by index with comprehensive validation."""
        print(f"[REMOVE FILTER] index={remove_index}, current_count={len(current_filters)}")
        
        if remove_index < 0 or remove_index >= len(current_filters):
            print(f"Invalid remove index: {remove_index}")
            return current_filters, f"❌ Invalid filter index: {remove_index}"
        
        try:
            removed_filter = current_filters[remove_index]
            new_filters_list = [f for i, f in enumerate(current_filters) if i != remove_index]
            
            print(f"Filter removed successfully: {removed_filter['metric']}. Remaining: {len(new_filters_list)}")
            
            if len(new_filters_list) == 0:
                return new_filters_list, "✅ Filter removed. No active filters remaining."
            else:
                return new_filters_list, f"✅ Filter '{removed_filter['metric'].replace('_', ' ')}' removed successfully."
                
        except Exception as e:
            print(f"Error removing filter at index {remove_index}: {e}")
            return current_filters, f"❌ Error removing filter: {str(e)}"
    
    # Run tests
    print("=== TESTING FILTER MANAGEMENT SYSTEM ===\n")
    
    # Test 1: Add valid filters
    print("Test 1: Adding valid filters")
    filters = []
    
    filters, msg, clear = handle_add_filter_robust(filters, "avg_daily_stress", ">", "30")
    print(f"Result: {msg}")
    print(f"Filters: {len(filters)}")
    
    filters, msg, clear = handle_add_filter_robust(filters, "daily_alcohol_units", "==", "0")
    print(f"Result: {msg}")
    print(f"Filters: {len(filters)}")
    
    # Test 2: Try to add duplicate
    print("\nTest 2: Adding duplicate filter")
    filters, msg, clear = handle_add_filter_robust(filters, "avg_daily_stress", ">", "30")
    print(f"Result: {msg}")
    print(f"Filters: {len(filters)}")
    
    # Test 3: Add invalid filters
    print("\nTest 3: Adding invalid filters")
    
    filters, msg, clear = handle_add_filter_robust(filters, "", ">", "30")
    print(f"Result: {msg}")
    
    filters, msg, clear = handle_add_filter_robust(filters, "stress", "", "30")
    print(f"Result: {msg}")
    
    filters, msg, clear = handle_add_filter_robust(filters, "stress", ">", "")
    print(f"Result: {msg}")
    
    # Test 4: Remove filters
    print("\nTest 4: Removing filters")
    
    filters, msg = handle_remove_filter_robust(filters, 0)
    print(f"Result: {msg}")
    print(f"Filters: {len(filters)}")
    
    filters, msg = handle_remove_filter_robust(filters, 0)
    print(f"Result: {msg}")
    print(f"Filters: {len(filters)}")
    
    # Test 5: Remove invalid index
    print("\nTest 5: Remove invalid index")
    filters, msg = handle_remove_filter_robust(filters, 5)
    print(f"Result: {msg}")
    
    print("\n=== ALL TESTS COMPLETED ===")
    return True

if __name__ == "__main__":
    test_filter_logic()