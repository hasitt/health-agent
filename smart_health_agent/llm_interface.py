"""
Mock LLM Interface for Health Detective functionality.
This provides intelligent health coaching responses based on user queries and health data context.
"""

import re
import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Tuple, Optional
import random

logger = logging.getLogger(__name__)

class HealthDetectiveLLM:
    """
    Mock LLM interface that simulates intelligent health coaching responses.
    In production, this would connect to actual LLM APIs (Ollama, GPT, Claude, etc.)
    """
    
    def __init__(self):
        self.conversation_history = []
        self.health_coaching_prompts = {
            'sleep': [
                "Your sleep patterns show some interesting trends. Let me analyze your recent sleep data.",
                "Sleep is crucial for recovery and performance. Based on your data, here's what I see:",
                "Looking at your sleep metrics, I notice some patterns worth discussing."
            ],
            'activity': [
                "Your activity levels tell an interesting story. Let me break down what I see:",
                "Physical activity is key to overall health. Here's my analysis of your recent patterns:",
                "I've been tracking your movement patterns - here are some insights:"
            ],
            'nutrition': [
                "Your nutritional data reveals some important patterns. Let me share what I found:",
                "Nutrition plays a huge role in how you feel and perform. Here's my analysis:",
                "Looking at your food intake, I can see some trends worth exploring:"
            ],
            'stress': [
                "Stress management is crucial for long-term health. Your data shows:",
                "I've been monitoring your stress patterns, and here's what stands out:",
                "Stress levels can impact everything from sleep to recovery. Here's what I see:"
            ],
            'mood': [
                "Your subjective wellbeing data provides valuable insights into your mental health:",
                "Mental health is just as important as physical health. Your mood patterns show:",
                "I've been tracking your emotional wellbeing, and here are some observations:"
            ]
        }
    
    def get_response(self, user_message: str, data_context: Dict[str, Any] = None) -> Tuple[str, Optional[str]]:
        """
        Generate an intelligent response based on user message and health data context.
        
        Args:
            user_message: The user's question or request
            data_context: Dictionary containing relevant health data
            
        Returns:
            Tuple of (response_text, graph_suggestion)
        """
        try:
            # Analyze the user message to determine intent
            intent = self._analyze_user_intent(user_message)
            
            # Generate contextual response based on intent and data
            response = self._generate_contextual_response(intent, user_message, data_context)
            
            # Check if a graph should be suggested
            graph_suggestion = self._suggest_graph(intent, data_context)
            
            # Store conversation for context
            self.conversation_history.append({
                'user': user_message,
                'assistant': response,
                'timestamp': datetime.now().isoformat(),
                'intent': intent
            })
            
            return response, graph_suggestion
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "I apologize, but I encountered an error while analyzing your health data. Please try rephrasing your question.", None
    
    def _analyze_user_intent(self, message: str) -> str:
        """Analyze user message to determine health-related intent."""
        message_lower = message.lower()
        
        # Sleep-related queries
        if any(word in message_lower for word in ['sleep', 'sleeping', 'rest', 'tired', 'fatigue']):
            return 'sleep'
        
        # Activity-related queries
        if any(word in message_lower for word in ['steps', 'exercise', 'activity', 'workout', 'walk', 'run', 'active']):
            return 'activity'
        
        # Nutrition-related queries
        if any(word in message_lower for word in ['food', 'eat', 'nutrition', 'diet', 'calories', 'protein', 'carbs']):
            return 'nutrition'
        
        # Stress-related queries
        if any(word in message_lower for word in ['stress', 'pressure', 'anxiety', 'heart rate', 'rhr']):
            return 'stress'
        
        # Mood/wellbeing queries
        if any(word in message_lower for word in ['mood', 'feel', 'energy', 'motivation', 'wellbeing', 'mental']):
            return 'mood'
        
        # General health queries
        if any(word in message_lower for word in ['health', 'wellness', 'overview', 'summary', 'trends']):
            return 'general'
        
        # Correlation/analysis queries
        if any(word in message_lower for word in ['correlat', 'relationship', 'impact', 'affect', 'connect']):
            return 'correlation'
        
        return 'general'
    
    def _generate_contextual_response(self, intent: str, user_message: str, data_context: Dict[str, Any]) -> str:
        """Generate a contextual response based on intent and available data."""
        
        if not data_context:
            return self._generate_no_data_response(intent)
        
        # Start with an appropriate opening based on intent
        opening = random.choice(self.health_coaching_prompts.get(intent, [
            "That's a great question about your health data. Let me take a look:",
            "I've analyzed your recent health patterns, and here's what I found:",
            "Based on your health data, here are some insights:"
        ]))
        
        response_parts = [opening]
        
        if intent == 'sleep':
            response_parts.extend(self._analyze_sleep_data(data_context))
        elif intent == 'activity':
            response_parts.extend(self._analyze_activity_data(data_context))
        elif intent == 'nutrition':
            response_parts.extend(self._analyze_nutrition_data(data_context))
        elif intent == 'stress':
            response_parts.extend(self._analyze_stress_data(data_context))
        elif intent == 'mood':
            response_parts.extend(self._analyze_mood_data(data_context))
        elif intent == 'correlation':
            response_parts.extend(self._analyze_correlations(data_context))
        else:
            response_parts.extend(self._generate_general_analysis(data_context))
        
        # Add a proactive suggestion or question
        response_parts.append(self._generate_proactive_nudge(intent, data_context))
        
        return "\n\n".join(response_parts)
    
    def _analyze_sleep_data(self, data_context: Dict[str, Any]) -> List[str]:
        """Analyze sleep-related data and generate insights."""
        insights = []
        
        sleep_data = data_context.get('sleep_data', [])
        if sleep_data:
            durations = [entry.get('sleep_duration_hours', 0) for entry in sleep_data if entry.get('sleep_duration_hours')]
            scores = [entry.get('sleep_score', 0) for entry in sleep_data if entry.get('sleep_score')]
            
            if durations:
                avg_duration = sum(durations) / len(durations)
                insights.append(f"📊 **Sleep Duration**: Over the last {len(durations)} nights, you've averaged {avg_duration:.1f} hours of sleep.")
                
                if avg_duration < 7:
                    insights.append("⚠️ **Insight**: You're getting less than the recommended 7-9 hours. Consider adjusting your bedtime routine.")
                elif avg_duration > 9:
                    insights.append("💤 **Insight**: You're sleeping more than average. This could indicate recovery needs or check if sleep quality is optimal.")
                else:
                    insights.append("✅ **Insight**: Your sleep duration is within the healthy range!")
            
            if scores:
                avg_score = sum(scores) / len(scores)
                insights.append(f"🎯 **Sleep Quality**: Your average sleep score is {avg_score:.0f}/100.")
                
                if avg_score >= 80:
                    insights.append("🌟 **Great job!** Your sleep quality is excellent. Keep up your current sleep habits.")
                elif avg_score >= 60:
                    insights.append("👍 **Good progress!** Your sleep quality is decent, but there's room for improvement.")
                else:
                    insights.append("🔄 **Opportunity**: Your sleep quality could benefit from some adjustments to your routine.")
        
        return insights
    
    def _analyze_activity_data(self, data_context: Dict[str, Any]) -> List[str]:
        """Analyze activity-related data and generate insights."""
        insights = []
        
        activity_data = data_context.get('activity_data', [])
        if activity_data:
            steps = [entry.get('total_steps', 0) for entry in activity_data if entry.get('total_steps')]
            calories = [entry.get('active_calories', 0) for entry in activity_data if entry.get('active_calories')]
            
            if steps:
                avg_steps = sum(steps) / len(steps)
                insights.append(f"👣 **Daily Steps**: You're averaging {avg_steps:.0f} steps per day over {len(steps)} days.")
                
                if avg_steps >= 10000:
                    insights.append("🎉 **Excellent!** You're consistently hitting the 10k steps goal. That's fantastic for cardiovascular health.")
                elif avg_steps >= 7000:
                    insights.append("👍 **Good work!** You're quite active, though there's room to push toward 10k steps daily.")
                else:
                    insights.append("💪 **Opportunity**: Consider gradually increasing your daily movement. Even small increases help!")
            
            if calories:
                avg_calories = sum(calories) / len(calories)
                insights.append(f"🔥 **Active Calories**: You're burning an average of {avg_calories:.0f} active calories daily.")
        
        return insights
    
    def _analyze_nutrition_data(self, data_context: Dict[str, Any]) -> List[str]:
        """Analyze nutrition-related data and generate insights."""
        insights = []
        
        nutrition_data = data_context.get('nutrition_data', [])
        if nutrition_data:
            calories = [entry.get('total_calories', 0) for entry in nutrition_data if entry.get('total_calories')]
            protein = [entry.get('protein_g', 0) for entry in nutrition_data if entry.get('protein_g')]
            
            if calories:
                avg_calories = sum(calories) / len(calories)
                insights.append(f"🍽️ **Daily Intake**: You're consuming an average of {avg_calories:.0f} calories per day.")
                
                if avg_calories < 1500:
                    insights.append("⚠️ **Consideration**: Your calorie intake seems low. Ensure you're meeting your energy needs.")
                elif avg_calories > 2500:
                    insights.append("📊 **Observation**: Higher calorie intake detected. Consider if this aligns with your activity level and goals.")
            
            if protein:
                avg_protein = sum(protein) / len(protein)
                insights.append(f"🥩 **Protein Intake**: You're getting {avg_protein:.0f}g of protein daily on average.")
                
                if avg_protein < 50:
                    insights.append("💪 **Suggestion**: Consider increasing protein intake for better muscle maintenance and satiety.")
        
        return insights
    
    def _analyze_stress_data(self, data_context: Dict[str, Any]) -> List[str]:
        """Analyze stress-related data and generate insights."""
        insights = []
        
        stress_data = data_context.get('stress_data', [])
        if stress_data:
            avg_stress = [entry.get('avg_daily_stress', 0) for entry in stress_data if entry.get('avg_daily_stress')]
            rhr = [entry.get('avg_daily_rhr', 0) for entry in stress_data if entry.get('avg_daily_rhr')]
            
            if avg_stress:
                mean_stress = sum(avg_stress) / len(avg_stress)
                insights.append(f"😤 **Stress Levels**: Your average daily stress is {mean_stress:.0f}/100.")
                
                if mean_stress >= 60:
                    insights.append("🧘 **High stress detected**: Consider stress management techniques like meditation, deep breathing, or gentle exercise.")
                elif mean_stress <= 30:
                    insights.append("😌 **Great stress management!** Your stress levels are well-controlled.")
            
            if rhr:
                avg_rhr = sum(rhr) / len(rhr)
                insights.append(f"❤️ **Resting Heart Rate**: Your average RHR is {avg_rhr:.0f} bpm, which can indicate fitness and recovery status.")
        
        return insights
    
    def _analyze_mood_data(self, data_context: Dict[str, Any]) -> List[str]:
        """Analyze mood and subjective wellbeing data."""
        insights = []
        
        mood_data = data_context.get('mood_data', [])
        if mood_data:
            moods = [entry.get('mood', 0) for entry in mood_data if entry.get('mood')]
            energy = [entry.get('energy', 0) for entry in mood_data if entry.get('energy')]
            
            if moods:
                avg_mood = sum(moods) / len(moods)
                insights.append(f"😊 **Mood Tracking**: Your average mood rating is {avg_mood:.1f}/10 over {len(moods)} entries.")
                
                if avg_mood >= 7:
                    insights.append("🌟 **Positive outlook!** Your mood has been consistently good. Keep up whatever you're doing!")
                elif avg_mood <= 4:
                    insights.append("💙 **Support available**: Your mood has been lower recently. Consider talking to someone or exploring mood-boosting activities.")
            
            if energy:
                avg_energy = sum(energy) / len(energy)
                insights.append(f"⚡ **Energy Levels**: Your average energy rating is {avg_energy:.1f}/10.")
        
        return insights
    
    def _analyze_correlations(self, data_context: Dict[str, Any]) -> List[str]:
        """Analyze correlations between different health metrics."""
        insights = []
        
        # This is a simplified correlation analysis
        insights.append("🔗 **Pattern Recognition**: Looking for connections in your health data...")
        
        sleep_data = data_context.get('sleep_data', [])
        mood_data = data_context.get('mood_data', [])
        activity_data = data_context.get('activity_data', [])
        
        if sleep_data and mood_data:
            insights.append("💡 **Sleep-Mood Connection**: Better sleep often correlates with improved mood and energy levels.")
        
        if activity_data and sleep_data:
            insights.append("🔄 **Activity-Sleep Link**: Regular physical activity typically improves sleep quality and duration.")
        
        return insights
    
    def _generate_general_analysis(self, data_context: Dict[str, Any]) -> List[str]:
        """Generate a general health overview."""
        insights = []
        
        insights.append("📈 **Health Overview**: Here's a snapshot of your recent health patterns:")
        
        # Quick summary of each data type available
        if data_context.get('sleep_data'):
            insights.append("✅ Sleep data is being tracked")
        if data_context.get('activity_data'):
            insights.append("✅ Activity data is available")
        if data_context.get('nutrition_data'):
            insights.append("✅ Nutrition data is logged")
        if data_context.get('mood_data'):
            insights.append("✅ Mood tracking is active")
        
        return insights
    
    def _generate_proactive_nudge(self, intent: str, data_context: Dict[str, Any]) -> str:
        """Generate a proactive suggestion or follow-up question."""
        nudges = {
            'sleep': [
                "Would you like me to analyze how your sleep affects your next-day energy levels?",
                "I can show you a graph of your sleep trends if that would be helpful.",
                "Have you noticed any patterns in what helps you sleep better?"
            ],
            'activity': [
                "Would you like to see how your activity levels correlate with your sleep quality?",
                "I can create a visualization of your step trends over time.",
                "Are there specific days when you're more or less active?"
            ],
            'nutrition': [
                "Would you like me to analyze how your nutrition affects your energy levels?",
                "I can show you a breakdown of your macronutrient intake patterns.",
                "Have you noticed how certain foods affect how you feel?"
            ],
            'general': [
                "What specific aspect of your health would you like to explore further?",
                "I can create custom visualizations to help you understand your patterns better.",
                "Is there a particular health goal you're working toward?"
            ]
        }
        
        return "🤔 " + random.choice(nudges.get(intent, nudges['general']))
    
    def _generate_no_data_response(self, intent: str) -> str:
        """Generate response when no data context is available."""
        return f"""I'd love to help you analyze your {intent} data, but I don't see any recent data to work with. 

🔧 **Quick Check:**
- Make sure your Garmin data is synced
- Ensure Cronometer data has been uploaded
- Check that mood tracking entries have been submitted

Once your data is available, I can provide personalized insights and recommendations! What specific health metrics would you like to start tracking?"""
    
    def _suggest_graph(self, intent: str, data_context: Dict[str, Any]) -> Optional[str]:
        """Suggest appropriate graph types based on intent and available data."""
        if not data_context:
            return None
        
        # Graph suggestions based on intent and available data
        if intent == 'sleep' and data_context.get('sleep_data'):
            return "GRAPH_SUGGESTION: sleep_trends"
        elif intent == 'activity' and data_context.get('activity_data'):
            return "GRAPH_SUGGESTION: activity_trends"
        elif intent == 'nutrition' and data_context.get('nutrition_data'):
            return "GRAPH_SUGGESTION: nutrition_trends"
        elif intent == 'stress' and data_context.get('stress_data'):
            return "GRAPH_SUGGESTION: stress_trends"
        elif intent == 'mood' and data_context.get('mood_data'):
            return "GRAPH_SUGGESTION: mood_trends"
        elif intent == 'correlation':
            return "GRAPH_SUGGESTION: correlation_analysis"
        
        return None

# Global instance for the application
health_detective = HealthDetectiveLLM()

def get_llm_response(prompt: str, data_context: Dict[str, Any] = None) -> Tuple[str, Optional[str]]:
    """
    Main interface function for getting LLM responses.
    This function will be called from smart_health_ollama.py
    
    Args:
        prompt: User's message/question
        data_context: Dictionary containing relevant health data
        
    Returns:
        Tuple of (response_text, graph_suggestion)
    """
    return health_detective.get_response(prompt, data_context)