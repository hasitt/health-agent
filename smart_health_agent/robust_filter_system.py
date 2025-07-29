#!/usr/bin/env python3
"""
Robust Dynamic Filter Management System for Gradio
This implements the proper approach with dynamic component creation and event binding.
"""

import gradio as gr
from typing import List, Dict, Tuple, Any

class RobustFilterManager:
    """Manages dynamic filter creation and removal with proper Gradio integration."""
    
    def __init__(self):
        self.remove_buttons = []  # Track dynamically created buttons
    
    def create_filter_display(self, filters_list: List[Dict], active_filters_state: gr.State) -> gr.Column:
        """Create the complete filter display with dynamic buttons and proper event binding."""
        
        # Clear previous button references
        self.remove_buttons.clear()
        
        with gr.Column(elem_id="active-filters-container") as filters_container:
            if not filters_list:
                gr.Markdown("### 🔍 Active Filters")
                gr.Markdown("*No filters applied. Add filters above to refine your analysis.*", 
                           elem_classes=["text-muted"])
            else:
                gr.Markdown(f"### 🔍 Active Filters ({len(filters_list)})")
                
                # Header row
                with gr.Row():
                    gr.Markdown("**Metric**", scale=4)
                    gr.Markdown("**Condition**", scale=3)
                    gr.Markdown("**Value**", scale=4)
                    gr.Markdown("**Action**", scale=1)
                
                # Create each filter row with dynamic remove button
                for i, filter_dict in enumerate(filters_list):
                    with gr.Row(elem_id=f"filter_row_{i}"):
                        # Display filter details
                        gr.Markdown(f"**{filter_dict['metric'].replace('_', ' ').title()}**", scale=4)
                        
                        # Format operator display
                        operator_display = {
                            '>': 'greater than', '>=': 'greater than or equal to',
                            '<': 'less than', '<=': 'less than or equal to',
                            '==': 'equal to', '!=': 'not equal to',
                            'between': 'between', 'in': 'in list', 'like': 'contains'
                        }.get(filter_dict['operator'], filter_dict['operator'])
                        
                        gr.Markdown(f"*{operator_display}*", scale=3)
                        gr.Markdown(f"`{filter_dict['raw_value']}`", scale=4)
                        
                        # Create dynamic remove button
                        remove_btn = gr.Button("✕", scale=1, size="sm", variant="secondary",
                                             elem_id=f"remove_btn_{i}")
                        
                        # Store button reference for event binding
                        self.remove_buttons.append((remove_btn, i))
        
        return filters_container
    
    def bind_remove_events(self, active_filters_state: gr.State, 
                          active_filters_display_area: gr.Column,
                          remove_logic_func):
        """Bind click events to all dynamically created remove buttons."""
        
        for remove_btn, filter_index in self.remove_buttons:
            # Create closure to capture the correct index
            def make_remove_handler(idx):
                def remove_handler():
                    current_filters = active_filters_state.value
                    return remove_logic_func(idx, current_filters)
                return remove_handler
            
            # Bind the click event
            remove_btn.click(
                make_remove_handler(filter_index),
                inputs=[],
                outputs=[active_filters_state, active_filters_display_area]
            )
    
    def add_filter_logic(self, current_filters: List[Dict], metric: str, 
                        operator: str, value: str) -> Tuple[List[Dict], str, str]:
        """Add a new filter with validation and deduplication."""
        
        # Comprehensive validation
        if not metric or metric.strip() == "":
            return current_filters, "❌ Please select a metric before adding filter.", ""
        
        if not operator or operator.strip() == "":
            return current_filters, "❌ Please select an operator before adding filter.", ""
        
        if not value or value.strip() == "":
            return current_filters, "❌ Please enter a value before adding filter.", ""
        
        try:
            # Parse value (simplified for demo)
            parsed_value = self._parse_filter_value(value.strip(), operator)
            
            # Create new filter
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
                    return current_filters, f"⚠️ Filter '{metric} {operator} {value}' already exists!", ""
            
            # Add new filter
            new_filters_list = current_filters + [new_filter]
            return new_filters_list, f"✅ Filter for '{metric.replace('_', ' ')}' added successfully!", ""
            
        except ValueError as ve:
            return current_filters, f"❌ Invalid value: {str(ve)}", value
        except Exception as e:
            return current_filters, f"❌ Error adding filter: {str(e)}", value
    
    def remove_filter_logic(self, remove_index: int, 
                           current_filters: List[Dict]) -> Tuple[List[Dict], str]:
        """Remove a filter by index."""
        
        if remove_index < 0 or remove_index >= len(current_filters):
            return current_filters, f"❌ Invalid filter index: {remove_index}"
        
        try:
            removed_filter = current_filters[remove_index]
            new_filters_list = [f for i, f in enumerate(current_filters) if i != remove_index]
            
            if len(new_filters_list) == 0:
                return new_filters_list, "✅ Filter removed. No active filters remaining."
            else:
                return new_filters_list, f"✅ Filter '{removed_filter['metric'].replace('_', ' ')}' removed successfully."
                
        except Exception as e:
            return current_filters, f"❌ Error removing filter: {str(e)}"
    
    def _parse_filter_value(self, value_str: str, operator: str):
        """Parse filter value based on operator type."""
        if operator in ['>', '>=', '<', '<=']:
            try:
                return float(value_str)
            except ValueError:
                raise ValueError(f"Operator '{operator}' requires a numeric value")
        
        if operator == 'between':
            parts = [v.strip() for v in value_str.split(',')]
            if len(parts) != 2:
                raise ValueError("Between operator requires exactly 2 comma-separated values")
            try:
                return [float(parts[0]), float(parts[1])]
            except ValueError:
                raise ValueError("Between operator requires numeric values")
        
        if operator == 'in':
            parts = [v.strip() for v in value_str.split(',') if v.strip()]
            if not parts:
                raise ValueError("In operator requires at least one value")
            return parts
        
        # For equality and like operators
        return value_str


# Demo implementation
def create_demo_interface():
    """Create a demo interface showing the robust filter system."""
    
    filter_manager = RobustFilterManager()
    
    with gr.Blocks(title="Robust Filter System Demo") as demo:
        gr.Markdown("# Robust Dynamic Filter Management System")
        gr.Markdown("This demonstrates proper Gradio dynamic component creation with working X buttons.")
        
        # State management
        active_filters_state = gr.State([])
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Add New Filter")
                
                filter_metric = gr.Dropdown(
                    choices=["avg_daily_stress", "daily_alcohol_units", "total_calories", "sleep_hours"],
                    label="Metric",
                    value="avg_daily_stress"
                )
                
                filter_operator = gr.Dropdown(
                    choices=[">", ">=", "<", "<=", "==", "!=", "between", "in"],
                    label="Operator", 
                    value=">"
                )
                
                filter_value = gr.Textbox(
                    label="Value",
                    placeholder="e.g., 30 or 10,20 for between",
                    value=""
                )
                
                add_filter_btn = gr.Button("Add Filter", variant="primary")
                
                filter_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### Active Filters")
                
                # This is the dynamic display area
                active_filters_display_area = gr.Column(elem_id="filters-display")
        
        # Event handlers
        def handle_add_filter(current_filters, metric, operator, value):
            new_filters, status_msg, clear_value = filter_manager.add_filter_logic(
                current_filters, metric, operator, value
            )
            
            # Create new display
            new_display = filter_manager.create_filter_display(new_filters, active_filters_state)
            
            # Bind remove events (this is where the magic happens)
            filter_manager.bind_remove_events(
                active_filters_state, 
                active_filters_display_area,
                filter_manager.remove_filter_logic
            )
            
            return new_filters, new_display, status_msg, clear_value
        
        def handle_remove_filter(remove_index, current_filters):
            new_filters, status_msg = filter_manager.remove_filter_logic(remove_index, current_filters)
            
            # Create new display
            new_display = filter_manager.create_filter_display(new_filters, active_filters_state)
            
            # Re-bind remove events for remaining filters
            filter_manager.bind_remove_events(
                active_filters_state,
                active_filters_display_area, 
                filter_manager.remove_filter_logic
            )
            
            return new_filters, new_display
        
        # Wire up events
        add_filter_btn.click(
            handle_add_filter,
            inputs=[active_filters_state, filter_metric, filter_operator, filter_value],
            outputs=[active_filters_state, active_filters_display_area, filter_status, filter_value]
        )
        
        # Initialize empty display
        demo.load(
            lambda: filter_manager.create_filter_display([], active_filters_state),
            outputs=[active_filters_display_area]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo_interface()
    demo.launch(share=True)