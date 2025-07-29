"""
Simple but robust filter management for Gradio.
This uses a single HTML display with JavaScript that properly communicates with Gradio state.
"""

import gradio as gr
from typing import List, Dict, Tuple

def create_filter_html(filters_list: List[Dict]) -> str:
    """Create HTML display for filters with working remove buttons."""
    
    if not filters_list:
        return """
        <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef; margin: 10px 0;'>
            <h4 style='margin: 0 0 10px 0; color: #495057;'>🔍 Active Filters</h4>
            <p style='margin: 0; color: #6c757d; font-style: italic;'>
                <em>No filters applied. Add filters above to refine your analysis.</em>
            </p>
        </div>
        """
    
    html_parts = [f"""
        <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef; margin: 10px 0;'>
            <h4 style='margin: 0 0 15px 0; color: #495057;'>
                🔍 Active Filters ({len(filters_list)})
            </h4>
            <table style='width: 100%; border-collapse: collapse; background: white; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                <thead>
                    <tr style='background: #e9ecef;'>
                        <th style='padding: 12px; text-align: left; font-weight: 600; color: #212529; border-bottom: 1px solid #dee2e6;'>Metric</th>
                        <th style='padding: 12px; text-align: left; font-weight: 600; color: #212529; border-bottom: 1px solid #dee2e6;'>Condition</th>
                        <th style='padding: 12px; text-align: left; font-weight: 600; color: #212529; border-bottom: 1px solid #dee2e6;'>Value</th>
                        <th style='padding: 12px; text-align: center; font-weight: 600; color: #212529; border-bottom: 1px solid #dee2e6; width: 80px;'>Action</th>
                    </tr>
                </thead>
                <tbody>
    """]
    
    for i, filter_dict in enumerate(filters_list):
        operator_display = {
            '>': 'greater than', '>=': 'greater than or equal to',
            '<': 'less than', '<=': 'less than or equal to',
            '==': 'equal to', '!=': 'not equal to',
            'between': 'between', 'in': 'in list', 'like': 'contains'
        }.get(filter_dict['operator'], filter_dict['operator'])
        
        row_bg = '#ffffff' if i % 2 == 0 else '#f8f9fa'
        
        html_parts.append(f"""
            <tr style='background: {row_bg};'>
                <td style='padding: 12px; border-bottom: 1px solid #e9ecef; font-weight: 500; color: #212529;'>
                    <code style='background: #d1ecf1; color: #0c5460; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; font-weight: 600;'>{filter_dict['metric']}</code>
                </td>
                <td style='padding: 12px; border-bottom: 1px solid #e9ecef; color: #212529; font-weight: 500;'>
                    {operator_display}
                </td>
                <td style='padding: 12px; border-bottom: 1px solid #e9ecef; color: #212529;'>
                    <strong style='color: #0d6efd;'>{filter_dict['raw_value']}</strong>
                </td>
                <td style='padding: 12px; border-bottom: 1px solid #e9ecef; text-align: center;'>
                    <button onclick='triggerRemoveFilter({i})' 
                            style='background: #dc3545; color: white; border: none; border-radius: 4px; width: 30px; height: 30px; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; font-size: 14px; transition: background 0.2s;'
                            onmouseover='this.style.background="#c82333"'
                            onmouseout='this.style.background="#dc3545"'
                            title='Remove this filter' type='button'>
                        ✕
                    </button>
                </td>
            </tr>
        """)
    
    html_parts.append(f"""
                </tbody>
            </table>
            <p style='margin: 15px 0 0 0; color: #6c757d; font-size: 0.9em;'>
                <em>💡 Click the <strong>✕</strong> button to remove individual filters.</em>
            </p>
        </div>
        
        <script>
        function triggerRemoveFilter(index) {{
            console.log('🗑️ Triggering remove for filter index:', index);
            
            // Find the remove index input - try multiple selectors
            const indexInput = document.querySelector('#remove-index-input input') ||
                             document.querySelector('#remove-index-input textarea') ||
                             document.querySelector('input[data-testid*="remove-index"]') ||
                             document.querySelector('textarea[data-testid*="remove-index"]');
            
            if (indexInput) {{
                indexInput.value = index.toString();
                
                // Dispatch events to notify Gradio
                ['input', 'change'].forEach(eventType => {{
                    indexInput.dispatchEvent(new Event(eventType, {{ bubbles: true, composed: true }}));
                }});
                
                console.log('✅ Set remove index to:', index);
                
                // Trigger the remove button
                setTimeout(() => {{
                    const removeBtn = document.querySelector('#remove-trigger-btn') ||
                                    document.querySelector('button[data-testid*="remove-trigger"]');
                    
                    if (removeBtn) {{
                        removeBtn.click();
                        console.log('✅ Triggered remove button');
                    }} else {{
                        console.error('❌ Remove trigger button not found');
                    }}
                }}, 50);
                
            }} else {{
                console.error('❌ Remove index input not found');
            }}
        }}
        </script>
    """)
    
    return "".join(html_parts)


def validate_and_parse_filter(metric: str, operator: str, value: str) -> Dict:
    """Validate and parse a filter with comprehensive checks."""
    
    if not metric or metric.strip() == "":
        raise ValueError("Please select a metric")
    
    if not operator or operator.strip() == "":
        raise ValueError("Please select an operator")
    
    if not value or value.strip() == "":
        raise ValueError("Please enter a value")
    
    value = value.strip()
    
    # Parse value based on operator
    if operator in ['>', '>=', '<', '<=']:
        try:
            parsed_value = float(value)
        except ValueError:
            raise ValueError(f"Operator '{operator}' requires a numeric value")
    elif operator == 'between':
        parts = [v.strip() for v in value.split(',')]
        if len(parts) != 2:
            raise ValueError("Between requires exactly 2 comma-separated values (e.g., '10,20')")
        try:
            parsed_value = [float(parts[0]), float(parts[1])]
        except ValueError:
            raise ValueError("Between requires numeric values")
    elif operator == 'in':
        parts = [v.strip() for v in value.split(',') if v.strip()]
        if not parts:
            raise ValueError("In requires at least one value")
        parsed_value = parts
    else:
        # For == != and like
        parsed_value = value
    
    return {
        'metric': metric,
        'operator': operator,
        'value': parsed_value,
        'raw_value': value
    }


def create_demo():
    """Create demo interface with working filter management."""
    
    with gr.Blocks(title="Robust Filter System") as demo:
        gr.Markdown("# Simple Robust Filter Management")
        
        # State
        filters_state = gr.State([])
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Add Filter")
                
                metric_dropdown = gr.Dropdown(
                    choices=["avg_daily_stress", "daily_alcohol_units", "total_calories", "sleep_hours"],
                    label="Metric",
                    value="avg_daily_stress"
                )
                
                operator_dropdown = gr.Dropdown(
                    choices=[">", ">=", "<", "<=", "==", "!=", "between", "in"],
                    label="Operator",
                    value=">"
                )
                
                value_input = gr.Textbox(
                    label="Value",
                    placeholder="e.g., 30 or 10,20 for between"
                )
                
                add_btn = gr.Button("Add Filter", variant="primary")
                status_box = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=2):
                # Filter display
                filters_html = gr.HTML(
                    value=create_filter_html([]),
                    label="Active Filters"
                )
        
        # Hidden components for remove functionality
        remove_index_input = gr.Textbox(
            value="",
            visible=False,
            elem_id="remove-index-input"
        )
        
        remove_trigger_btn = gr.Button(
            "Remove",
            visible=False,
            elem_id="remove-trigger-btn"
        )
        
        # Add filter logic
        def add_filter(current_filters, metric, operator, value):
            try:
                new_filter = validate_and_parse_filter(metric, operator, value)
                
                # Check duplicates
                for existing in current_filters:
                    if (existing['metric'] == metric and 
                        existing['operator'] == operator and 
                        existing['raw_value'] == value):
                        return (
                            current_filters,
                            create_filter_html(current_filters),
                            f"⚠️ Filter '{metric} {operator} {value}' already exists!",
                            ""
                        )
                
                new_filters = current_filters + [new_filter]
                return (
                    new_filters,
                    create_filter_html(new_filters),
                    f"✅ Filter added successfully!",
                    ""  # Clear input
                )
                
            except ValueError as e:
                return (
                    current_filters,
                    create_filter_html(current_filters),
                    f"❌ {str(e)}",
                    value
                )
        
        # Remove filter logic
        def remove_filter(remove_index_str, current_filters):
            try:
                remove_index = int(remove_index_str)
                if 0 <= remove_index < len(current_filters):
                    removed_filter = current_filters[remove_index]
                    new_filters = [f for i, f in enumerate(current_filters) if i != remove_index]
                    
                    status_msg = f"✅ Filter '{removed_filter['metric']}' removed!" if new_filters else "✅ All filters cleared!"
                    
                    return (
                        new_filters,
                        create_filter_html(new_filters),
                        status_msg,
                        ""  # Clear remove index
                    )
                else:
                    return (
                        current_filters,
                        create_filter_html(current_filters),
                        f"❌ Invalid filter index: {remove_index}",
                        ""
                    )
            except (ValueError, TypeError):
                return (
                    current_filters,
                    create_filter_html(current_filters),
                    "❌ Invalid remove request",
                    ""
                )
        
        # Wire up events
        add_btn.click(
            add_filter,
            inputs=[filters_state, metric_dropdown, operator_dropdown, value_input],
            outputs=[filters_state, filters_html, status_box, value_input]
        )
        
        remove_trigger_btn.click(
            remove_filter,
            inputs=[remove_index_input, filters_state],
            outputs=[filters_state, filters_html, status_box, remove_index_input]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()