"""
Daily Mood Tracker UI Components for Gradio Integration
Comprehensive mood and lifestyle tracking interface.
"""

import gradio as gr
import mood_tracking
from datetime import datetime, date
from typing import Dict, Any, Optional

def create_mood_tracker_interface(current_user_id: Optional[int] = None):
    """Create comprehensive mood tracking UI components."""
    
    def submit_mood_entry(
        mood_rating: int,
        energy_rating: int,
        stress_rating: int,
        anxiety_rating: int,
        sleep_quality_rating: int,
        focus_rating: int,
        motivation_rating: int,
        emotional_state: str,
        stress_triggers: str,
        coping_strategies: str,
        physical_symptoms: str,
        daily_events: str,
        social_interactions: str,
        notes: str
    ) -> str:
        """Handle mood tracking form submission."""
        
        if current_user_id is None:
            return "❌ Please sync Garmin data first to establish user session"
        
        try:
            mood_data = {
                'date': datetime.now().date(),
                'timestamp': datetime.now(),
                'mood_rating': mood_rating if mood_rating > 0 else None,
                'energy_rating': energy_rating if energy_rating > 0 else None,
                'stress_rating': stress_rating if stress_rating > 0 else None,
                'anxiety_rating': anxiety_rating if anxiety_rating > 0 else None,
                'sleep_quality_rating': sleep_quality_rating if sleep_quality_rating > 0 else None,
                'focus_rating': focus_rating if focus_rating > 0 else None,
                'motivation_rating': motivation_rating if motivation_rating > 0 else None,
                'emotional_state': emotional_state.strip() if emotional_state.strip() else None,
                'stress_triggers': stress_triggers.strip() if stress_triggers.strip() else None,
                'coping_strategies': coping_strategies.strip() if coping_strategies.strip() else None,
                'physical_symptoms': physical_symptoms.strip() if physical_symptoms.strip() else None,
                'daily_events': daily_events.strip() if daily_events.strip() else None,
                'social_interactions': social_interactions.strip() if social_interactions.strip() else None,
                'notes_text': notes.strip() if notes.strip() else None,
                'entry_type': 'daily',
                'source': 'manual'
            }
            
            mood_tracking.insert_daily_mood_entry(current_user_id, mood_data)
            
            return f"✅ Mood entry saved successfully for {datetime.now().strftime('%Y-%m-%d')}"
            
        except Exception as e:
            return f"❌ Error saving mood entry: {e}"
    
    def get_mood_summary_display() -> str:
        """Generate mood tracking summary display."""
        
        if current_user_id is None:
            return "Please sync Garmin data first to view mood summary"
        
        try:
            summary = mood_tracking.get_mood_summary(current_user_id, days=7)
            
            if not summary['mood_entries']:
                return "No mood tracking data available. Submit your first mood entry above!"
            
            output_lines = ["--- Mood Tracking Summary (Last 7 Days) ---", ""]
            
            # Show averages
            averages = summary['averages']
            if any(avg is not None for avg in averages.values()):
                output_lines.append("📊 Average Ratings (1-10 scale):")
                for rating_type, avg_value in averages.items():
                    if avg_value is not None:
                        rating_name = rating_type.replace('_rating', '').replace('_', ' ').title()
                        output_lines.append(f"  {rating_name}: {avg_value:.1f}/10")
                output_lines.append("")
            
            # Show recent entries
            output_lines.append("📝 Recent Mood Entries:")
            for entry in summary['mood_entries'][:5]:  # Show last 5 entries
                date_str = entry['date']
                mood = entry['mood_rating']
                energy = entry['energy_rating']
                stress = entry['stress_rating']
                anxiety = entry['anxiety_rating']
                
                ratings = []
                if mood: ratings.append(f"Mood: {mood}/10")
                if energy: ratings.append(f"Energy: {energy}/10")
                if stress: ratings.append(f"Stress: {stress}/10")
                if anxiety: ratings.append(f"Anxiety: {anxiety}/10")
                
                ratings_str = ", ".join(ratings) if ratings else "No ratings"
                output_lines.append(f"  {date_str}: {ratings_str}")
                
                if entry['emotional_state']:
                    output_lines.append(f"    Emotional state: {entry['emotional_state']}")
                if entry['stress_triggers']:
                    output_lines.append(f"    Stress triggers: {entry['stress_triggers']}")
                if entry['coping_strategies']:
                    output_lines.append(f"    Coping strategies: {entry['coping_strategies']}")
                output_lines.append("")
            
            return "\\n".join(output_lines)
            
        except Exception as e:
            return f"Error retrieving mood summary: {e}"
    
    def get_lifestyle_summary_display() -> str:
        """Generate caffeine/alcohol consumption summary."""
        
        if current_user_id is None:
            return "Please sync Garmin data first to view lifestyle summary"
        
        try:
            summary = mood_tracking.get_caffeine_alcohol_summary(current_user_id, days=7)
            
            if not summary['daily_consumption']:
                return "No caffeine/alcohol consumption data available. Track via food log imports!"
            
            output_lines = ["--- Lifestyle Summary (Last 7 Days) ---", ""]
            
            # Daily consumption overview
            output_lines.append("☕ Daily Consumption Overview:")
            for day in summary['daily_consumption'][:7]:
                date_str = day['date']
                caffeine = day['total_caffeine'] or 0
                alcohol = day['total_alcohol_units'] or 0
                
                consumption_parts = []
                if caffeine > 0:
                    consumption_parts.append(f"{caffeine:.0f}mg caffeine")
                if alcohol > 0:
                    consumption_parts.append(f"{alcohol:.1f} alcohol units")
                
                consumption_str = ", ".join(consumption_parts) if consumption_parts else "None"
                output_lines.append(f"  {date_str}: {consumption_str}")
            
            output_lines.append("")
            
            # Recent entries
            if summary['recent_entries']:
                output_lines.append("🍷 Recent Caffeine/Alcohol Entries:")
                for entry in summary['recent_entries'][:8]:
                    time_str = f"{entry['date']} {entry['time']}"
                    item_name = entry['food_item_name']
                    
                    details = []
                    if entry['caffeine_mg'] and entry['caffeine_mg'] > 0:
                        details.append(f"{entry['caffeine_mg']:.0f}mg caffeine")
                    if entry['alcohol_ml'] and entry['alcohol_ml'] > 0:
                        details.append(f"{entry['alcohol_ml']:.0f}ml alcohol")
                    
                    details_str = f" ({', '.join(details)})" if details else ""
                    output_lines.append(f"  {time_str}: {item_name}{details_str}")
                    
                    if entry['consumption_context']:
                        output_lines.append(f"    Context: {entry['consumption_context']}")
            
            return "\\n".join(output_lines)
            
        except Exception as e:
            return f"Error retrieving lifestyle summary: {e}"
    
    # Create UI components
    with gr.Tab("Daily Mood Tracker"):
        gr.Markdown("### Daily Mood & Wellbeing Tracker")
        gr.Markdown("Track your daily mood, stress levels, and lifestyle factors for comprehensive health insights.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Core Mood Metrics (1-10 scale)")
                
                mood_rating = gr.Slider(
                    minimum=0, maximum=10, step=1, value=5,
                    label="Overall Mood (1=Very Low, 10=Excellent)"
                )
                
                energy_rating = gr.Slider(
                    minimum=0, maximum=10, step=1, value=5,
                    label="Energy Level (1=Exhausted, 10=Very Energetic)"
                )
                
                stress_rating = gr.Slider(
                    minimum=0, maximum=10, step=1, value=5,
                    label="Stress Level (1=Very Calm, 10=Very Stressed)"
                )
                
                anxiety_rating = gr.Slider(
                    minimum=0, maximum=10, step=1, value=5,
                    label="Anxiety Level (1=Very Calm, 10=Very Anxious)"
                )
            
            with gr.Column():
                gr.Markdown("#### Additional Metrics")
                
                sleep_quality_rating = gr.Slider(
                    minimum=0, maximum=10, step=1, value=5,
                    label="Sleep Quality (1=Very Poor, 10=Excellent)"
                )
                
                focus_rating = gr.Slider(
                    minimum=0, maximum=10, step=1, value=5,
                    label="Focus/Concentration (1=Very Poor, 10=Excellent)"
                )
                
                motivation_rating = gr.Slider(
                    minimum=0, maximum=10, step=1, value=5,
                    label="Motivation Level (1=None, 10=Very High)"
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Emotional & Social Factors")
                
                emotional_state = gr.Textbox(
                    label="Emotional State",
                    placeholder="e.g., calm, anxious, excited, frustrated...",
                    lines=2
                )
                
                social_interactions = gr.Textbox(
                    label="Social Interactions",
                    placeholder="Quality and quantity of social contact today...",
                    lines=2
                )
                
                daily_events = gr.Textbox(
                    label="Significant Daily Events",
                    placeholder="Important events affecting your mood today...",
                    lines=2
                )
            
            with gr.Column():
                gr.Markdown("#### Stress & Coping")
                
                stress_triggers = gr.Textbox(
                    label="Stress Triggers",
                    placeholder="e.g., work deadline, relationship issue, health concern...",
                    lines=2
                )
                
                coping_strategies = gr.Textbox(
                    label="Coping Strategies Used",
                    placeholder="e.g., exercise, meditation, talking to friend...",
                    lines=2
                )
                
                physical_symptoms = gr.Textbox(
                    label="Physical Symptoms",
                    placeholder="e.g., headache, tension, fatigue, restlessness...",
                    lines=2
                )
        
        notes = gr.Textbox(
            label="Additional Notes",
            placeholder="Any other relevant observations about your mood, stress, or wellbeing today...",
            lines=3
        )
        
        submit_mood_button = gr.Button("Save Daily Mood Entry", variant="primary")
        mood_status = gr.Textbox(label="Status", interactive=False)
        
        # Summary displays
        with gr.Row():
            with gr.Column():
                mood_summary_display = gr.Textbox(
                    label="Mood Tracking Summary",
                    value="Submit your first mood entry to view summary",
                    lines=15,
                    interactive=False
                )
                refresh_mood_button = gr.Button("Refresh Mood Summary", variant="secondary", size="sm")
            
            with gr.Column():
                lifestyle_summary_display = gr.Textbox(
                    label="Lifestyle Tracking Summary",
                    value="Import food log data with caffeine/alcohol to view summary",
                    lines=15,
                    interactive=False
                )
                refresh_lifestyle_button = gr.Button("Refresh Lifestyle Summary", variant="secondary", size="sm")
        
        # Event handlers
        submit_mood_button.click(
            fn=submit_mood_entry,
            inputs=[
                mood_rating, energy_rating, stress_rating, anxiety_rating,
                sleep_quality_rating, focus_rating, motivation_rating,
                emotional_state, stress_triggers, coping_strategies,
                physical_symptoms, daily_events, social_interactions, notes
            ],
            outputs=mood_status
        )
        
        refresh_mood_button.click(
            fn=get_mood_summary_display,
            outputs=mood_summary_display
        )
        
        refresh_lifestyle_button.click(
            fn=get_lifestyle_summary_display,
            outputs=lifestyle_summary_display
        )
    
    return {
        'submit_mood_entry': submit_mood_entry,
        'get_mood_summary_display': get_mood_summary_display,
        'get_lifestyle_summary_display': get_lifestyle_summary_display,
        'components': {
            'mood_summary_display': mood_summary_display,
            'lifestyle_summary_display': lifestyle_summary_display
        }
    }