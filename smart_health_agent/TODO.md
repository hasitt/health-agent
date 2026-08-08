Smart Health Agent Project Roadmap
This document outlines the current status, immediate next steps, and future phases for the development of your personalized Smart Health Agent. It serves as a living roadmap to guide our progress and ensure clarity on features and priorities.

🎯 Project Vision & Overarching Goal
To build a comprehensive, AI-powered personal health agent that integrates diverse health data (wearables, dietary, subjective wellbeing), identifies meaningful correlations and trends, and provides personalized, actionable, and empathetic insights and recommendations through an intuitive and interactive interface.

✅ Current State: Achieved Milestones
We have established a robust foundation, successfully implementing the following core functionalities:

Garmin Data Infrastructure:

Seamless collection and storage of historical and granular data (daily summaries, sleep, activities, minute-by-minute stress levels) in health_data.db.

Efficient sync strategy with performance optimizations (indexing, upsert logic).

Cronometer Data Infrastructure:

Manual CSV import functionality for "Food & Recipe Entries" (including detailed nutrients, caffeine, alcohol) into food_log and supplements tables.

Robust parsing and upsert logic for dietary data.

Subjective Wellbeing & Mood Tracking:

Enhanced subjective_wellbeing table with 15+ new mood and lifestyle tracking fields.

Comprehensive "Daily Mood Tracker" UI for data entry.

Initial Trend Analysis Functions:

Implemented specific correlation analyses: Stress Consistency, Steps vs. Sleep Effect, Activity Type vs. RHR Impact.

Advanced analytics for stress-lifestyle correlations (caffeine/alcohol vs. stress/mood, mood ratings vs. Garmin stress).

Enhanced LLM Integration:

OllamaLLM is integrated and receives comprehensive health context (Garmin, subjective, lifestyle, trends).

LLM provides personalized, interpretive, and actionable insights with an enhanced "Expert AI Health Analyst and Coach" role, demonstrating deeper inferential reasoning.

Adherence to the "factual display" principle for raw data, with LLM handling interpretation.

Gradio UI:

Organized tabbed interface (Main Dashboard, Daily Mood Tracker).

Corrected and accurate display of daily summary data (steps, stress, peak stress).

Improved display formatting with emoji indicators and better organization.

📥 Backlog — Requested Features
Quick Subjective Check-In (emoji tap-through) — requested 2026-08-08
Goal: Make logging subjective wellbeing effortless so it actually gets done daily. Subjective tracking has been the weak link (no entries since Aug 2025), which also starves the mood/energy correlation analyses of recent data.

Concept: On app open, present a lightweight guided check-in — one question at a time, each answered with a single tap on three large smileys (🙁 sad / 😐 neutral / 🙂 happy). Tapping auto-advances to the next question: mood → energy → stress → sleep quality → focus → motivation. A few taps and you're done; offer a "skip" and a "done for today" so it's never a chore.

Implementation notes:
- Reuse the existing write path: submit_mood_entry(mood, energy, stress, sleep_quality, focus, motivation, ...) -> db.upsert_subjective_wellbeing (smart_health_ollama.py:507). No new table needed — subjective_wellbeing already stores these on a 1–10 scale (CHECK 1..10).
- Map the 3 smileys onto the 1–10 scale (e.g. 🙁=2, 😐=5, 🙂=8). Consider a 5-face variant later for finer granularity; start with 3 for speed.
- Note per-metric valence: for stress, 🙂 should mean "low stress" (invert the mapping) so a happy face is always the "good" answer.
- Show it on launch only if today has no entry yet; make it dismissable.
- Once flowing, this revives the mood/energy correlations that currently return "no data recorded" for recent windows.

🚀 Next Phases: Roadmap for Future Development
Phase 1: Enhanced Visualization (Immediate Next Step)
Goal: Provide clear, interactive graphs and trending data visualizations within the Gradio UI to enhance user understanding and provide richer context for LLM interpretation.

Implementation:

Integrate a plotting library (e.g., matplotlib or plotly) into the application.

Create a dedicated "Graphs" or "Trends Visuals" tab in the Gradio UI.

Develop functions to generate the following initial visualizations:

Time-Series Plots:

Daily Average Stress (Garmin & Subjective) over time

Daily Sleep Score & Deep Sleep Percentage over time

Daily Resting Heart Rate over time

Daily Total Steps over time

Daily Body Battery (if granularly available from Garmin) over time

Weekly Averages for Mood, Energy, Anxiety, Focus ratings

Daily Caffeine and Alcohol intake over time

Correlation-Specific Visualizations (based on existing data):

Carb timing vs. deep-sleep minutes: Scatter plot of dinner carb intake vs. deep sleep % for that night.

Step count vs. Sleep score: Scatter plot of daily steps vs. next-day sleep score.

Evening resistance session vs. next-day RHR: Bar chart comparing average RHR on days with evening strength vs. other days.

Red-meat dinner vs. lower RHR: Bar chart comparing average RHR after red meat dinner vs. other dinners.

Caffeine after 15:00 vs. REM%: Scatter plot of late caffeine intake vs. REM sleep percentage.

High magnesium intake vs. reduced bedtime latency: Scatter plot of daily Mg intake vs. sleep latency.

Meditation minutes vs. HRV trend: Line graph showing 7-day HRV rolling average alongside meditation minutes.

Omega-3 vs. HRV: Scatter plot of daily EPA+DHA vs. 24-hour HRV average.

Na:K ratio vs. SpO2: Scatter plot of Na:K ratio vs. overnight SpO2.

Fiber vs. Body Battery recharge: Scatter plot of daily fiber vs. next-day Body Battery recharge.

Last calorie ≥3h pre-bed vs. Deep-sleep: Bar chart comparing deep sleep % for early vs. late dinners.

Alcohol vs. Respiration Rate: Scatter plot of alcohol intake vs. night-time respiration rate.

Evening yoga vs. Morning stress: Bar chart comparing morning stress score after evening yoga vs. other evenings.

Lunch sat-fat vs. Body Battery slump: Scatter plot of lunch saturated fat vs. afternoon Body Battery slump.

Vitamin B6 vs. Wake episodes/Vivid REM: Scatter plot of B6 intake vs. wake episodes/vivid REM.

Phase 2: Iterative Correlation Implementation & LLM Refinement
Goal: Implement more of the advanced correlation logic and further refine the LLM's ability to interpret and explain these complex patterns, preparing for a conversational interface.

Implementation:

Develop backend logic for remaining desired correlations (e.g., those requiring more advanced Garmin metrics like Anaerobic Load, VO₂max, ATL/CTL ratio, specific training zones).

Update the LLM's prompt to explicitly reference and interpret these newly calculated and visualized correlations, encouraging even deeper "health detective" reasoning.

Focus on generating more nuanced "if-then" scenarios and predictive insights.

Phase 3: Interactive Chat-Based System & Comprehensive Data Integration
Goal: Transform the application into a fully interactive, conversational AI health coach with dynamic visualizations and integrate a truly comprehensive range of external data sources.

Interactive Chat Interface:

Implement a persistent chat history within the UI.

Integrate voice-to-text and text-to-speech for natural language interaction.

Enable dynamic display of relevant graphs and data snippets directly within the chat conversation based on user queries.

Allow quick UI selections/checkboxes for common data entry (e.g., mood, quick food logs).

Comprehensive Data Source Expansion:

All Wearables: Explore and integrate data from a wider array of wearable devices beyond Garmin and Cronometer, ensuring a holistic view of activity, sleep, and physiological metrics.

All Labs: Develop robust mechanisms for importing, parsing, and analyzing various lab results (e.g., blood work, hormone panels, nutrient deficiencies), securely integrating them into the LLM context.

All Genetic Reports: Investigate secure and privacy-preserving methods to integrate and interpret genetic predisposition data from various providers, enabling highly personalized and preventative insights.

Advanced Data Integrations (Specific Examples):

Weather Data: Integrate with a weather API to pull local temperature/humidity data for correlations like "Hydration on hot days vs. stress score."

Other Wearables/Apps: Expand data source options as needed (e.g., specific Garmin Connect IQ app data for screen time, dedicated light sensors).

Proactive Insights & Nudges:

Develop a system for the LLM to proactively offer insights or "nudges" based on real-time data patterns (e.g., "Your HRV is lower today, consider a recovery day").

Phase 4: Standalone Cloud-Based Phone Application
Goal: Evolve the application from a local Gradio-based interface to a cloud-hosted, standalone mobile application for broader accessibility and enhanced user experience.

Platform & Deployment:

Migrate the backend logic to a scalable cloud infrastructure.

Develop native or cross-platform mobile applications (iOS/Android) that connect to the cloud backend.

Implement robust user authentication and data synchronization for a mobile environment.

Optimize UI/UX for mobile devices, including push notifications for proactive insights.

✨ Key Principles Maintained
Factual Display: Raw data and initial trend summaries will remain factual and uninterpreted in the UI.

LLM Interpretation: The LLM is solely responsible for synthesizing, interpreting, and providing actionable recommendations.

Personalization: All insights and recommendations will be highly tailored to the individual user's data.

Privacy & Security: Continued adherence to secure data handling practices.

This roadmap will guide our future development. Let's start with Phase 1: Enhanced Visualization as our immediate next step.